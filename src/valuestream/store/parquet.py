"""Aggregate parquet store helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from valuestream.utils.timer import timed


def aggregate_dir(
    workspace_path: str | Path,
    *,
    source_id: str,
    processor_id: str,
    grain: str,
) -> Path:
    """Return ``aggregates/<source>/<processor>/<grain>``."""
    return Path(workspace_path) / "aggregates" / source_id / processor_id / grain


@timed
def write_aggregate(
    frame: pl.DataFrame,
    workspace_path: str | Path,
    *,
    source_id: str,
    processor_id: str,
    grain: str,
    run_id: str,
    chunk_id: str,
) -> list[Path]:
    """Write ``frame`` into hive-style ``period=...`` parquet partitions."""
    if frame.is_empty():
        return []
    if "period" not in frame.columns:
        raise ValueError("aggregate frame must contain a period column")
    null_periods = frame["period"].null_count()
    if null_periods:
        raise ValueError(
            "aggregate frame contains "
            f"{null_periods} row(s) with null period; check timestamp parsing and "
            "derive_calendar transforms for this source"
        )

    base = aggregate_dir(
        workspace_path,
        source_id=source_id,
        processor_id=processor_id,
        grain=grain,
    )
    written: list[Path] = []
    safe_chunk = _safe_name(chunk_id)
    safe_run = _safe_name(run_id)
    # Split once instead of re-filtering the frame per period; each partition
    # still goes through a write-then-rename so readers never see partials.
    partitions = frame.partition_by("period", as_dict=True, include_key=True)
    for key in sorted(partitions, key=lambda key: str(key[0])):
        period = str(key[0])
        partition_dir = base / f"period={period}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        path = partition_dir / f"part-{safe_run}-{safe_chunk}.parquet"
        tmp = partition_dir / f".{path.name}.tmp"
        partitions[key].write_parquet(tmp)
        tmp.replace(path)
        written.append(path)
    return written


def scan_aggregate(
    workspace_path: str | Path,
    *,
    source_id: str,
    processor_id: str,
    grain: str,
    paths: Iterable[str | Path] | None = None,
) -> pl.LazyFrame:
    """Scan selected or all Parquet files for one physical aggregate."""
    base = aggregate_dir(
        workspace_path,
        source_id=source_id,
        processor_id=processor_id,
        grain=grain,
    )
    if paths is not None:
        selected = [str(Path(path)) for path in paths]
        if not selected:
            raise ValueError("aggregate scan paths cannot be empty")
        return pl.scan_parquet(selected, hive_partitioning=True, glob=False)
    pattern = str(base / "**" / "*.parquet")
    return pl.scan_parquet(pattern, hive_partitioning=True, glob=True)


def aggregate_exists(
    workspace_path: str | Path,
    *,
    source_id: str,
    processor_id: str,
    grain: str,
) -> bool:
    """Return true when at least one parquet file exists for the aggregate."""
    base = aggregate_dir(
        workspace_path,
        source_id=source_id,
        processor_id=processor_id,
        grain=grain,
    )
    return base.exists() and any(base.glob("**/*.parquet"))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


__all__ = ["aggregate_dir", "aggregate_exists", "scan_aggregate", "write_aggregate"]
