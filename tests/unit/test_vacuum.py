"""Vacuum metadata scan tests."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import duckdb
import polars as pl
import pytest

from valuestream.config import model
from valuestream.engine.ledger import insert_chunk, insert_run, start_run
from valuestream.store import vacuum


@pytest.mark.unit
def test_file_metadata_uses_one_parquet_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "aggregate.parquet"
    created_at = dt.datetime(2026, 7, 13, tzinfo=dt.UTC)
    pl.DataFrame(
        {
            "config_hash": ["hash", "hash"],
            "chunk_id": ["chunk", "chunk"],
            "pipeline_run_id": ["run", "run"],
            "created_at": [created_at, created_at],
        }
    ).write_parquet(path)
    original = pl.scan_parquet
    calls = 0

    def counted_scan(*args: Any, **kwargs: Any) -> pl.LazyFrame:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(vacuum.pl, "scan_parquet", counted_scan)

    metadata = vacuum._file_metadata(path)

    assert calls == 1
    assert metadata.config_hashes == {"hash"}
    assert metadata.chunk_ids == {"chunk"}
    assert metadata.run_ids == {"run"}
    assert metadata.created_order_ns == int(created_at.timestamp() * 1_000_000_000)


@pytest.mark.unit
def test_file_metadata_uses_mtime_when_created_at_is_absent(tmp_path: Path) -> None:
    path = tmp_path / "aggregate.parquet"
    pl.DataFrame(
        {
            "config_hash": ["hash"],
            "chunk_id": ["chunk"],
            "pipeline_run_id": ["run"],
        }
    ).write_parquet(path)

    metadata = vacuum._file_metadata(path)

    assert metadata.created_order_ns == path.stat().st_mtime_ns


@pytest.mark.unit
def test_checkpoint_vacuum_keeps_current_rolling_database_and_removes_stale_state(
    tmp_path: Path,
) -> None:
    processor = model.FrequencyResponseProcessor.model_validate(
        {
            "id": "frequency",
            "source": "ih",
            "kind": "frequency_response",
            "group_by": ["ExposureBucket"],
            "time": {"property": "DecisionTime", "grain": "daily"},
            "states": {"Contacts": {"type": "count"}},
            "columns": {
                "customer": "CustomerID",
                "interaction": "InteractionID",
                "action": "ActionID",
                "placement": "Placement",
                "rank": "Rank",
                "outcome": "Outcome",
                "propensity": "Propensity",
            },
            "alternative_group_by": ["Placement"],
            "positive_values": ["Clicked"],
            "exposure_values": ["Impression"],
            "candidate_values": ["Pending"],
            "checkpoint": {"mode": "persistent_sharded", "shards": 4},
        }
    )
    source_scan = processor.model_copy(
        update={
            "id": "source-scan",
            "checkpoint": model.FrequencyResponseCheckpoint(mode="source_scan"),
        }
    )
    catalog = SimpleNamespace(processors=SimpleNamespace(processors=[processor, source_scan]))
    state_root = tmp_path / ".valuestream/state/frequency_response"
    source_root = state_root / "source=ih"
    processor_root = source_root / "processor=frequency"
    processor_root.mkdir(parents=True)
    current_database = processor_root / "rolling.duckdb"
    current_database.write_bytes(b"current rolling state")
    current_wal = processor_root / "rolling.duckdb.wal"
    current_wal.write_bytes(b"current rolling WAL")

    legacy_nested = processor_root / "config=old/layout=old"
    legacy_nested.mkdir(parents=True)
    temporary = processor_root / ".rewrite.tmp"
    temporary.mkdir(parents=True)
    removed_processor = source_root / "processor=removed"
    removed_processor.mkdir(parents=True)
    source_scan_processor = source_root / "processor=source-scan"
    source_scan_processor.mkdir(parents=True)
    removed_source = state_root / "source=removed/processor=frequency"
    removed_source.mkdir(parents=True)
    legacy_runtime = state_root / "schema=6/hash=1/polars=1.43.2"
    legacy_source = legacy_runtime / "source=ih"
    legacy_layout = legacy_source / "processor=frequency/config=old/layout=old/chunk=2024-01-01"
    legacy_layout.mkdir(parents=True)

    while_running = vacuum._stale_processor_state_dirs(
        tmp_path,
        cast(Any, catalog),
        source_ids={"ih"},
        running_run_ids={"ih": frozenset({"run"})},
        include_tmp=True,
    )

    stale = vacuum._stale_processor_state_dirs(
        tmp_path,
        cast(Any, catalog),
        source_ids=None,
        running_run_ids={},
        include_tmp=True,
    )
    scoped_stale = vacuum._stale_processor_state_dirs(
        tmp_path,
        cast(Any, catalog),
        source_ids={"ih"},
        running_run_ids={},
        include_tmp=True,
    )

    assert while_running == []
    assert legacy_runtime.parents[1] in scoped_stale
    assert removed_source.parent not in scoped_stale
    assert stale == sorted(
        [
            legacy_nested.parent,
            temporary,
            removed_processor,
            source_scan_processor,
            removed_source.parent,
            legacy_runtime.parents[1],
        ]
    )
    result = vacuum.vacuum_processor_state(
        tmp_path,
        cast(Any, catalog),
        include_tmp=True,
    )

    assert result.dirs_deleted == len(stale)
    assert current_database.exists()
    assert current_wal.exists()
    assert not legacy_nested.exists()
    assert not source_scan_processor.exists()


@pytest.mark.unit
def test_checkpoint_vacuum_does_not_prune_history_inside_current_rolling_database(
    tmp_path: Path,
) -> None:
    processor = model.FrequencyResponseProcessor.model_validate(
        {
            "id": "frequency",
            "source": "ih",
            "kind": "frequency_response",
            "group_by": ["ExposureBucket"],
            "time": {"property": "DecisionTime", "grain": "daily"},
            "states": {"Contacts": {"type": "count"}},
            "columns": {
                "customer": "CustomerID",
                "interaction": "InteractionID",
                "action": "ActionID",
                "placement": "Placement",
                "rank": "Rank",
                "outcome": "Outcome",
                "propensity": "Propensity",
            },
            "alternative_group_by": ["Placement"],
            "positive_values": ["Clicked"],
            "exposure_values": ["Impression"],
            "candidate_values": ["Pending"],
            "checkpoint": {
                "mode": "persistent_sharded",
                "shards": 4,
                "retention_days": 8,
            },
        }
    )
    catalog = SimpleNamespace(processors=SimpleNamespace(processors=[processor]))
    processor_dir = tmp_path / ".valuestream/state/frequency_response/source=ih/processor=frequency"
    processor_dir.mkdir(parents=True)
    temporary = processor_dir / ".load.tmp"
    temporary.mkdir()
    database = processor_dir / "rolling.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("CREATE TABLE history (chunk_id VARCHAR)")
        connection.executemany(
            "INSERT INTO history VALUES (?)",
            [(f"2024-01-{day:02d}",) for day in range(1, 11)],
        )

    stale = vacuum._stale_processor_state_dirs(
        tmp_path,
        cast(Any, catalog),
        source_ids=None,
        running_run_ids={},
        include_tmp=False,
    )
    result = vacuum.vacuum_processor_state(
        tmp_path,
        cast(Any, catalog),
        include_tmp=False,
    )

    assert stale == []
    assert result.dirs_deleted == 0
    assert temporary.exists()
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM history").fetchone() == (10,)


@pytest.mark.unit
def test_vacuum_retains_only_new_run_inside_selected_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_a = tmp_path / "aggregates" / "a" / "processor" / "daily" / "period=2026-07"
    source_b = tmp_path / "aggregates" / "b" / "processor" / "daily" / "period=2026-07"
    source_a.mkdir(parents=True)
    source_b.mkdir(parents=True)
    old_a = source_a / "part-old-chunk.parquet"
    new_a = source_a / "part-new-chunk.parquet"
    old_b = source_b / "part-old-chunk.parquet"
    for path in (old_a, new_a, old_b):
        path.write_bytes(b"aggregate")

    monkeypatch.setattr(
        vacuum,
        "_current_processor_hashes",
        lambda catalog, source_ids=None: {
            (source_id, "processor"): "hash" for source_id in (source_ids or {"a", "b"})
        },
    )
    monkeypatch.setattr(
        vacuum,
        "_successful_chunk_keys",
        lambda workspace, source_id: {("new", "chunk"), ("old", "chunk")},
    )

    def metadata(path: Path) -> vacuum._FileMetadata:
        run_id = "new" if path == new_a else "old"
        return vacuum._FileMetadata(
            config_hashes=frozenset({"hash"}),
            chunk_ids=frozenset({"chunk"}),
            run_ids=frozenset({run_id}),
            created_order_ns=2 if run_id == "new" else 1,
        )

    monkeypatch.setattr(vacuum, "_file_metadata", metadata)

    result = vacuum.vacuum_workspace(
        tmp_path,
        cast(Any, object()),
        include_tmp=False,
        source_ids={"a"},
        retained_run_ids={"a": "new"},
    )

    assert result.paths == (old_a,)
    assert not old_a.exists()
    assert new_a.exists()
    assert old_b.exists()


@pytest.mark.unit
def test_vacuum_removes_old_partial_after_newer_successful_empty_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_run_id = "11111111-1111-4111-8111-111111111111"
    new_run_id = "22222222-2222-4222-8222-222222222222"
    chunk_id = "2026-07-01"
    base = tmp_path / "aggregates" / "ih" / "processor" / "daily" / "period=2026-07"
    base.mkdir(parents=True)
    old_partial = base / f"part-{old_run_id}-{chunk_id}.parquet"
    pl.DataFrame(
        {
            "config_hash": ["processor-hash"],
            "chunk_id": [chunk_id],
            "pipeline_run_id": [old_run_id],
            "created_at": [dt.datetime(2026, 7, 1, 1, tzinfo=dt.UTC)],
        }
    ).write_parquet(old_partial)

    for run_id, finished_at, rows_kept in (
        (old_run_id, dt.datetime(2026, 7, 1, 1, tzinfo=dt.UTC), 1),
        (new_run_id, dt.datetime(2026, 7, 2, 1, tzinfo=dt.UTC), 0),
    ):
        started_at = finished_at - dt.timedelta(minutes=1)
        insert_run(
            tmp_path,
            run_id=run_id,
            workspace="test",
            source_id="ih",
            config_hash="source-hash",
            started_at=started_at,
            finished_at=finished_at,
            status="ok",
            rows_in=rows_kept,
            rows_kept=rows_kept,
            chunks_total=1,
            chunks_ok=1,
            chunks_failed=0,
        )
        insert_chunk(
            tmp_path,
            source_id="ih",
            chunk_id=chunk_id,
            files=[],
            rows_in=rows_kept,
            rows_kept=rows_kept,
            started_at=started_at,
            finished_at=finished_at,
            status="ok",
            error=None,
            pipeline_run_id=run_id,
        )

    monkeypatch.setattr(
        vacuum,
        "_current_processor_hashes",
        lambda catalog, source_ids=None: {("ih", "processor"): "processor-hash"},
    )

    result = vacuum.vacuum_workspace(
        tmp_path,
        cast(Any, object()),
        include_tmp=False,
    )

    assert result.paths == (old_partial,)
    assert not old_partial.exists()


@pytest.mark.unit
def test_vacuum_does_not_leak_old_partial_across_source_contract_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_a_run_id = "33333333-3333-4333-8333-333333333333"
    contract_b_run_id = "44444444-4444-4444-8444-444444444444"
    chunk_id = "2026-07-01"
    base = tmp_path / "aggregates" / "ih" / "processor" / "daily" / "period=2026-07"
    base.mkdir(parents=True)
    paths: dict[str, Path] = {}
    for run_id, finished_at, source_hash, rows_kept in (
        (
            contract_a_run_id,
            dt.datetime(2026, 7, 1, 1, tzinfo=dt.UTC),
            "source-contract-a",
            1,
        ),
        (
            contract_b_run_id,
            dt.datetime(2026, 7, 2, 1, tzinfo=dt.UTC),
            "source-contract-b",
            0,
        ),
    ):
        if rows_kept:
            path = base / f"part-{run_id}-{chunk_id}.parquet"
            paths[source_hash] = path
            pl.DataFrame(
                {
                    "config_hash": ["processor-contract"],
                    "chunk_id": [chunk_id],
                    "pipeline_run_id": [run_id],
                    "created_at": [finished_at],
                }
            ).write_parquet(path)
        started_at = finished_at - dt.timedelta(minutes=1)
        insert_run(
            tmp_path,
            run_id=run_id,
            workspace="test",
            source_id="ih",
            config_hash=source_hash,
            started_at=started_at,
            finished_at=finished_at,
            status="ok",
            rows_in=rows_kept,
            rows_kept=rows_kept,
            chunks_total=1,
            chunks_ok=1,
            chunks_failed=0,
        )
        insert_chunk(
            tmp_path,
            source_id="ih",
            chunk_id=chunk_id,
            files=[],
            rows_in=rows_kept,
            rows_kept=rows_kept,
            started_at=started_at,
            finished_at=finished_at,
            status="ok",
            error=None,
            pipeline_run_id=run_id,
        )

    monkeypatch.setattr(
        vacuum,
        "_current_processor_hashes",
        lambda catalog, source_ids=None: {("ih", "processor"): "processor-contract"},
    )

    result = vacuum.vacuum_workspace(
        tmp_path,
        cast(Any, object()),
        include_tmp=False,
    )

    assert result.paths == (paths["source-contract-a"],)
    assert not paths["source-contract-a"].exists()


@pytest.mark.unit
def test_vacuum_rejects_incomplete_retained_run_scope(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cover exactly"):
        vacuum.vacuum_workspace(
            tmp_path,
            cast(Any, object()),
            include_tmp=False,
            source_ids={"a", "b"},
            retained_run_ids={"a": "new"},
        )


@pytest.mark.unit
def test_vacuum_preserves_final_and_temporary_files_for_running_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "11111111-1111-4111-8111-111111111111"
    base = tmp_path / "aggregates" / "a" / "processor" / "daily" / "period=2026-07"
    base.mkdir(parents=True)
    final = base / f"part-{run_id}-chunk.parquet"
    temporary = base / f".part-{run_id}-chunk.parquet.tmp"
    final.write_bytes(b"aggregate")
    temporary.write_bytes(b"temporary")
    start_run(
        tmp_path,
        run_id=run_id,
        workspace="test",
        source_id="a",
        config_hash="source-hash",
        started_at=dt.datetime(2026, 7, 16, tzinfo=dt.UTC),
        chunks_total=1,
    )
    monkeypatch.setattr(
        vacuum,
        "_current_processor_hashes",
        lambda catalog, source_ids=None: {("a", "processor"): "processor-hash"},
    )
    monkeypatch.setattr(
        vacuum,
        "_file_metadata",
        lambda path: vacuum._FileMetadata(
            config_hashes=frozenset({"processor-hash"}),
            chunk_ids=frozenset({"chunk"}),
            run_ids=frozenset({run_id}),
            created_order_ns=1,
        ),
    )

    result = vacuum.vacuum_workspace(
        tmp_path,
        cast(Any, object()),
        include_tmp=False,
    )

    assert result.files_deleted == 0
    assert final.exists()
    assert temporary.exists()
