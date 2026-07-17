"""Aggregate parquet store helpers."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from valuestream.utils.timer import timed


@dataclass(frozen=True, slots=True)
class AggregateWriteReceipt:
    """Lineage metadata captured while an in-memory partition is written."""

    path: Path
    pipeline_run_id: str
    chunk_id: str
    source_id: str
    processor_id: str
    grain: str
    period: str
    config_hash: str
    rows: int
    size_bytes: int
    created_at: dt.datetime


def aggregate_dir(
    workspace_path: str | Path,
    *,
    source_id: str,
    processor_id: str,
    grain: str,
) -> Path:
    """Return ``aggregates/<source>/<processor>/<grain>``."""
    return Path(workspace_path) / "aggregates" / source_id / processor_id / grain


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
    return [
        receipt.path
        for receipt in write_aggregate_with_receipts(
            frame,
            workspace_path,
            source_id=source_id,
            processor_id=processor_id,
            grain=grain,
            run_id=run_id,
            chunk_id=chunk_id,
        )
    ]


@timed
def write_aggregate_with_receipts(
    frame: pl.DataFrame,
    workspace_path: str | Path,
    *,
    source_id: str,
    processor_id: str,
    grain: str,
    run_id: str,
    chunk_id: str,
) -> list[AggregateWriteReceipt]:
    """Write aggregate partitions and return metadata without rereading Parquet."""
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
    written: list[AggregateWriteReceipt] = []
    safe_chunk = _safe_name(chunk_id)
    safe_run = _safe_name(run_id)
    # Split once instead of re-filtering the frame per period; each partition
    # still goes through a write-then-rename so readers never see partials.
    partitions = frame.partition_by("period", as_dict=True, include_key=True)
    for key in sorted(partitions, key=lambda key: str(key[0])):
        period = str(key[0])
        partition = partitions[key]
        config_hash = _single_string_value(partition, "config_hash")
        if _single_string_value(partition, "pipeline_run_id") != run_id:
            raise ValueError("aggregate pipeline_run_id does not match the write request")
        if _single_string_value(partition, "chunk_id") != chunk_id:
            raise ValueError("aggregate chunk_id does not match the write request")
        created_at = _single_value(partition, "created_at")
        if not isinstance(created_at, dt.datetime):
            raise ValueError("aggregate created_at must contain a timestamp")
        partition_dir = base / f"period={period}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        path = partition_dir / f"part-{safe_run}-{safe_chunk}.parquet"
        tmp = partition_dir / f".{path.name}.tmp"
        partition.write_parquet(tmp)
        tmp.replace(path)
        size_bytes = path.stat().st_size
        written.append(
            AggregateWriteReceipt(
                path=path,
                pipeline_run_id=run_id,
                chunk_id=chunk_id,
                source_id=source_id,
                processor_id=processor_id,
                grain=grain,
                period=period,
                config_hash=config_hash,
                rows=partition.height,
                size_bytes=size_bytes,
                created_at=created_at,
            )
        )
    return written


def _single_string_value(frame: pl.DataFrame, column: str) -> str:
    return str(_single_value(frame, column))


def _single_value(frame: pl.DataFrame, column: str) -> object:
    if column not in frame.columns:
        raise ValueError(f"aggregate frame must contain a {column!r} column")
    series = frame.get_column(column)
    values = series.unique().to_list()
    if series.null_count() or len(values) != 1:
        raise ValueError(f"aggregate {column} must contain exactly one non-null value")
    return values[0]


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


__all__ = [
    "AggregateWriteReceipt",
    "aggregate_dir",
    "aggregate_exists",
    "scan_aggregate",
    "write_aggregate",
    "write_aggregate_with_receipts",
]
