"""Vacuum metadata scan tests."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

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
def test_vacuum_rejects_incomplete_retained_run_scope(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cover exactly"):
        vacuum.vacuum_workspace(
            tmp_path,
            cast(Any, object()),
            include_tmp=False,
            source_ids={"a", "b"},
            retained_run_ids={"a": "new"},
        )
