"""Freshness metadata for dashboard tiles and ops pages."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from valuestream.config import model
from valuestream.config.canonical import processor_computation_hash
from valuestream.engine.ledger import successful_chunk_keys
from valuestream.processors import grain_levels
from valuestream.store.meta import meta_dir
from valuestream.store.parquet import aggregate_exists, scan_aggregate


@dataclass(frozen=True)
class Freshness:
    """Freshness summary for a metric tile."""

    latest_period: str | None
    last_created_at: dt.datetime | None
    last_run_finished_at: dt.datetime | None
    status: str


def metric_freshness(
    workspace_path: str | Path,
    catalog: model.Catalog,
    metric_name: str,
    *,
    grain: str,
) -> Freshness:
    """Return aggregate and run freshness for ``metric_name``."""
    grain = model.normalize_grain_name(grain)
    metric = catalog.metrics.metrics.get(metric_name)
    if metric is None:
        return Freshness(None, None, None, "unknown metric")
    processor = next((p for p in catalog.processors.processors if p.id == metric.source), None)
    if processor is None:
        return Freshness(None, None, None, "unknown processor")
    latest_period: str | None = None
    last_created_at: dt.datetime | None = None
    run = latest_run(workspace_path, source_id=processor.source)
    config_hash = processor_computation_hash(catalog, processor)
    for stored_grain in grain_levels.aggregate_grain_candidates(processor, grain):
        if not aggregate_exists(
            workspace_path,
            source_id=processor.source,
            processor_id=processor.id,
            grain=stored_grain,
        ):
            continue
        scanned = scan_aggregate(
            workspace_path,
            source_id=processor.source,
            processor_id=processor.id,
            grain=stored_grain,
        )
        current = scanned.filter(pl.col("config_hash") == config_hash)
        filtered = _filter_successful_chunks_lazy(current, workspace_path, processor.source)
        count_plan = current.select(pl.len().alias("current_rows"))
        if filtered is None:
            if count_plan.collect().item() == 0:
                continue
            break
        # Keep the existence check and freshness aggregation in one optimized
        # graph so their shared Parquet scan can be eliminated.
        count_frame, frame = pl.collect_all(
            [
                count_plan,
                filtered.select(
                    pl.col("period").cast(pl.String).max().alias("latest_period"),
                    pl.col("created_at").max().alias("last_created_at"),
                ),
            ]
        )
        if count_frame.item() == 0:
            continue
        if not frame.is_empty():
            row = frame.row(0, named=True)
            latest_period = row["latest_period"]
            last_created_at = row["last_created_at"]
        break
    return Freshness(
        latest_period=latest_period,
        last_created_at=last_created_at,
        last_run_finished_at=run.get("finished_at"),
        status=str(run.get("status", "not run")),
    )


def _filter_successful_chunks(
    frame: pl.DataFrame,
    workspace_path: str | Path,
    source_id: str,
) -> pl.DataFrame:
    if frame.is_empty() or not {"pipeline_run_id", "chunk_id"} <= set(frame.columns):
        return frame.head(0)
    keys = successful_chunk_keys(workspace_path, source_id=source_id)
    if not keys:
        return frame.head(0)
    key_frame = pl.DataFrame(
        {
            "pipeline_run_id": [run_id for run_id, _ in keys],
            "chunk_id": [chunk_id for _, chunk_id in keys],
        }
    )
    joined = frame.with_columns(pl.col("pipeline_run_id").cast(pl.String)).join(
        key_frame,
        on=["pipeline_run_id", "chunk_id"],
        how="inner",
    )
    if "created_at" not in joined.columns:
        return joined
    return joined.filter(pl.col("created_at") == pl.col("created_at").max().over("chunk_id"))


def _filter_successful_chunks_lazy(
    frame: pl.LazyFrame,
    workspace_path: str | Path,
    source_id: str,
) -> pl.LazyFrame | None:
    names = set(frame.collect_schema().names())
    if not {"pipeline_run_id", "chunk_id"} <= names:
        return None
    keys = successful_chunk_keys(workspace_path, source_id=source_id)
    if not keys:
        return None
    ordered_keys = sorted(keys)
    key_frame = pl.DataFrame(
        {
            "pipeline_run_id": [run_id for run_id, _ in ordered_keys],
            "chunk_id": [chunk_id for _, chunk_id in ordered_keys],
        }
    ).lazy()
    joined = frame.with_columns(pl.col("pipeline_run_id").cast(pl.String)).join(
        key_frame,
        on=["pipeline_run_id", "chunk_id"],
        how="inner",
    )
    if "created_at" not in names:
        return joined
    return joined.filter(pl.col("created_at") == pl.col("created_at").max().over("chunk_id"))


def recent_runs(workspace_path: str | Path, *, limit: int = 20) -> pl.DataFrame:
    """Return recent pipeline run metadata."""
    db_path = meta_dir(workspace_path) / "pipeline_runs.duckdb"
    if not db_path.exists():
        return pl.DataFrame()
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return conn.execute(
            """
            SELECT id, source_id, status, started_at, finished_at, rows_in, rows_kept,
                   chunks_total, chunks_ok, chunks_failed
            FROM pipeline_runs
            ORDER BY finished_at DESC
            LIMIT ?
            """,
            (limit,),
        ).pl()


def latest_run(workspace_path: str | Path, *, source_id: str | None = None) -> dict[str, Any]:
    """Return the latest run row as a dict, or an empty dict."""
    db_path = meta_dir(workspace_path) / "pipeline_runs.duckdb"
    if not db_path.exists():
        return {}
    where = "WHERE source_id = ?" if source_id is not None else ""
    params = (source_id,) if source_id is not None else ()
    with duckdb.connect(str(db_path), read_only=True) as conn:
        row = conn.execute(
            f"""
            SELECT id, source_id, status, started_at, finished_at, rows_in, rows_kept,
                   chunks_total, chunks_ok, chunks_failed
            FROM pipeline_runs
            {where}
            ORDER BY finished_at DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
        columns = [desc[0] for desc in conn.description] if conn.description else []
    return dict(zip(columns, row, strict=False)) if row else {}


def freshness_label(freshness: Freshness) -> str:
    """Human-readable freshness label."""
    period = freshness.latest_period or "no aggregate"
    run = (
        freshness.last_run_finished_at.isoformat(sep=" ", timespec="minutes")
        if freshness.last_run_finished_at
        else "not run"
    )
    return f"{freshness.status} · latest {period} · run {run}"


__all__ = ["Freshness", "freshness_label", "latest_run", "metric_freshness", "recent_runs"]
