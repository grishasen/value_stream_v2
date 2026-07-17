"""Aggregate parquet store tests."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import polars as pl
import pytest

from valuestream.engine import ledger
from valuestream.store.parquet import write_aggregate, write_aggregate_with_receipts


@pytest.mark.unit
def test_write_aggregate_rejects_null_period(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "period": [None],
            "Count": [1],
        }
    )

    with pytest.raises(ValueError, match="null period"):
        write_aggregate(
            frame,
            tmp_path,
            source_id="ih",
            processor_id="engagement",
            grain="daily",
            run_id="run",
            chunk_id="chunk",
        )


@pytest.mark.unit
def test_write_aggregate_returns_in_memory_lineage_receipts(tmp_path: Path) -> None:
    created_at = dt.datetime(2026, 7, 17, tzinfo=dt.UTC)
    frame = pl.DataFrame(
        {
            "period": ["2026-06", "2026-07", "2026-07"],
            "Count": [1, 2, 3],
            "pipeline_run_id": ["run"] * 3,
            "chunk_id": ["chunk"] * 3,
            "created_at": [created_at] * 3,
            "config_hash": ["hash"] * 3,
        }
    )

    receipts = write_aggregate_with_receipts(
        frame,
        tmp_path,
        source_id="ih",
        processor_id="engagement",
        grain="daily",
        run_id="run",
        chunk_id="chunk",
    )

    assert [receipt.period for receipt in receipts] == ["2026-06", "2026-07"]
    assert [receipt.rows for receipt in receipts] == [1, 2]
    assert all(receipt.config_hash == "hash" for receipt in receipts)
    assert all(receipt.created_at == created_at for receipt in receipts)
    assert all(receipt.path.is_file() for receipt in receipts)
    assert all(receipt.size_bytes == receipt.path.stat().st_size for receipt in receipts)


@pytest.mark.unit
def test_write_aggregate_rejects_mixed_partition_provenance(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "period": ["2026-07", "2026-07"],
            "pipeline_run_id": ["run", "other-run"],
            "chunk_id": ["chunk", "chunk"],
            "created_at": [dt.datetime(2026, 7, 17, tzinfo=dt.UTC)] * 2,
            "config_hash": ["hash", "hash"],
        }
    )

    with pytest.raises(ValueError, match="pipeline_run_id"):
        write_aggregate_with_receipts(
            frame,
            tmp_path,
            source_id="ih",
            processor_id="engagement",
            grain="daily",
            run_id="run",
            chunk_id="chunk",
        )


@pytest.mark.unit
def test_write_receipts_commit_lineage_without_rescanning_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_at = dt.datetime(2026, 7, 17, tzinfo=dt.UTC)
    run_id = "00000000-0000-0000-0000-000000000001"
    receipts = write_aggregate_with_receipts(
        pl.DataFrame(
            {
                "period": ["2026-07"],
                "Count": [3],
                "pipeline_run_id": [run_id],
                "chunk_id": ["chunk"],
                "created_at": [created_at],
                "config_hash": ["hash"],
            }
        ),
        tmp_path,
        source_id="ih",
        processor_id="engagement",
        grain="daily",
        run_id=run_id,
        chunk_id="chunk",
    )

    def unexpected_scan(*args: object, **kwargs: object) -> None:
        raise AssertionError("write-time lineage must not rescan aggregate Parquet")

    monkeypatch.setattr(ledger.pl, "scan_parquet", unexpected_scan)
    assert ledger.insert_lineage_records(tmp_path, records=receipts) == 1

    with duckdb.connect(str(tmp_path / "meta" / "lineage.duckdb"), read_only=True) as conn:
        row = conn.execute(
            """
            SELECT CAST(pipeline_run_id AS VARCHAR), chunk_id, source_id, processor_id,
                   grain, period, config_hash, rows, created_at
            FROM lineage
            """
        ).fetchone()
    assert row is not None
    assert row[:8] == (
        run_id,
        "chunk",
        "ih",
        "engagement",
        "daily",
        "2026-07",
        "hash",
        1,
    )
    assert isinstance(row[8], dt.datetime)


@pytest.mark.unit
def test_write_receipts_reject_missing_file_before_lineage_commit(tmp_path: Path) -> None:
    run_id = "00000000-0000-0000-0000-000000000001"
    receipts = write_aggregate_with_receipts(
        pl.DataFrame(
            {
                "period": ["2026-07"],
                "pipeline_run_id": [run_id],
                "chunk_id": ["chunk"],
                "created_at": [dt.datetime(2026, 7, 17, tzinfo=dt.UTC)],
                "config_hash": ["hash"],
            }
        ),
        tmp_path,
        source_id="ih",
        processor_id="engagement",
        grain="daily",
        run_id=run_id,
        chunk_id="chunk",
    )
    receipts[0].path.unlink()

    with pytest.raises(ValueError, match="path is missing"):
        ledger.insert_lineage_records(tmp_path, records=receipts)


@pytest.mark.unit
def test_write_receipts_store_workspace_absolute_lineage_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    monkeypatch.chdir(tmp_path)
    run_id = "00000000-0000-0000-0000-000000000001"
    receipts = write_aggregate_with_receipts(
        pl.DataFrame(
            {
                "period": ["2026-07"],
                "pipeline_run_id": [run_id],
                "chunk_id": ["chunk"],
                "created_at": [dt.datetime(2026, 7, 17, tzinfo=dt.UTC)],
                "config_hash": ["hash"],
            }
        ),
        Path("workspace"),
        source_id="ih",
        processor_id="engagement",
        grain="daily",
        run_id=run_id,
        chunk_id="chunk",
    )
    assert not receipts[0].path.is_absolute()
    assert ledger.insert_lineage_records(Path("workspace"), records=receipts) == 1

    monkeypatch.chdir(tmp_path.parent)
    paths = ledger.aggregate_lineage_paths(
        workspace,
        source_id="ih",
        processor_id="engagement",
        grain="daily",
    )["hash"]
    assert len(paths) == 1
    assert paths[0].is_absolute()
    assert paths[0].is_file()
