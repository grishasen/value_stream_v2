"""Legacy DuckDB aggregate backfill into Value Stream Parquet layout."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import polars as pl

from valuestream.config.canonical import processor_computation_hash
from valuestream.config.loader import load
from valuestream.engine import ledger
from valuestream.store.duckdb_views import refresh_aggregate_views
from valuestream.store.parquet import write_aggregate
from valuestream.utils.ids import new_pipeline_run_id
from valuestream.utils.timer import timed


@dataclass(frozen=True)
class BackfilledTable:
    """One legacy table copied into the aggregate store."""

    table: str
    source_id: str
    processor_id: str
    grain: str
    rows: int
    written: tuple[Path, ...]


@dataclass
class BackfillResult:
    """Summary for one legacy backfill run."""

    run_id: str
    tables: list[BackfilledTable] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return sum(table.rows for table in self.tables)


@timed
def backfill_from_legacy_db(
    workspace_path: str | Path,
    legacy_db: str | Path,
) -> BackfillResult:
    """Copy legacy aggregate-shaped DuckDB tables into Value Stream aggregates."""
    workspace = Path(workspace_path)
    catalog = load(workspace)
    legacy_path = Path(legacy_db)
    run_id = new_pipeline_run_id()
    result = BackfillResult(run_id=run_id)
    ledger.ensure(workspace)

    with duckdb.connect(str(legacy_path), read_only=True) as conn:
        tables = _legacy_tables(conn)
        for processor in catalog.processors.processors:
            config_hash = processor_computation_hash(catalog, processor)
            for grain in processor.grains:
                table_name = _find_table(
                    tables,
                    source_id=processor.source,
                    processor_id=processor.id,
                    grain=grain,
                )
                if table_name is None:
                    result.skipped.append(f"{processor.source}/{processor.id}/{grain}")
                    continue
                chunk_id = f"legacy-{table_name}"
                frame = conn.execute(f"SELECT * FROM {_quote_identifier(table_name)}").pl()
                frame = _with_backfill_columns(frame, config_hash, run_id, grain, chunk_id)
                written = write_aggregate(
                    frame,
                    workspace,
                    source_id=processor.source,
                    processor_id=processor.id,
                    grain=grain,
                    run_id=run_id,
                    chunk_id=chunk_id,
                )
                result.tables.append(
                    BackfilledTable(
                        table=table_name,
                        source_id=processor.source,
                        processor_id=processor.id,
                        grain=grain,
                        rows=frame.height,
                        written=tuple(written),
                    )
                )
                _insert_synthetic_chunk(
                    workspace,
                    source_id=processor.source,
                    chunk_id=chunk_id,
                    rows=frame.height,
                    run_id=run_id,
                    legacy_path=legacy_path,
                )
    _insert_synthetic_run(workspace, catalog.pipelines.workspace, result, run_id)
    refresh_aggregate_views(workspace, catalog)
    return result


def _with_backfill_columns(
    frame: pl.DataFrame,
    config_hash: str,
    run_id: str,
    grain: str,
    chunk_id: str,
) -> pl.DataFrame:
    now = dt.datetime.now(dt.UTC)
    out = frame
    if "period" not in out.columns:
        out = out.with_columns(_infer_period_expr(out, grain).alias("period"))
    return out.with_columns(
        pl.lit(run_id).alias("pipeline_run_id"),
        pl.lit(chunk_id).alias("chunk_id"),
        pl.lit(now).alias("created_at"),
        pl.lit(config_hash).alias("config_hash"),
    )


def _infer_period_expr(frame: pl.DataFrame, grain: str) -> pl.Expr:
    for column in ("Day", "day", "as_of_date"):
        if column in frame.columns:
            return pl.col(column).cast(pl.String).str.slice(0, 7)
    if grain == "summary":
        return pl.lit("ALL")
    if "Month" in frame.columns:
        return pl.col("Month").cast(pl.String)
    if "month" in frame.columns:
        return pl.col("month").cast(pl.String)
    return pl.lit("ALL")


def _legacy_tables(conn: duckdb.DuckDBPyConnection) -> set[str]:
    rows = conn.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        """
    ).fetchall()
    return {str(row[0]) for row in rows}


def _find_table(
    tables: set[str],
    *,
    source_id: str,
    processor_id: str,
    grain: str,
) -> str | None:
    candidates = [
        f"aggregate_{source_id}_{processor_id}_{grain}",
        f"{source_id}_{processor_id}_{grain}",
        f"{processor_id}_{grain}",
        processor_id if grain == "summary" else "",
    ]
    normalized = {_normalize_table(name): name for name in tables}
    for candidate in candidates:
        if not candidate:
            continue
        found = normalized.get(_normalize_table(candidate))
        if found is not None:
            return found
    return None


def _insert_synthetic_chunk(
    workspace: Path,
    *,
    source_id: str,
    chunk_id: str,
    rows: int,
    run_id: str,
    legacy_path: Path,
) -> None:
    now = dt.datetime.now(dt.UTC)
    ledger.insert_chunk(
        workspace,
        source_id=source_id,
        chunk_id=chunk_id,
        files=(legacy_path,),
        rows_in=rows,
        rows_kept=rows,
        started_at=now,
        finished_at=now,
        status="ok",
        error=None,
        pipeline_run_id=run_id,
    )


def _insert_synthetic_run(
    workspace: Path,
    workspace_name: str,
    result: BackfillResult,
    run_id: str,
) -> None:
    now = dt.datetime.now(dt.UTC)
    chunks = len(result.tables)
    ledger.insert_run(
        workspace,
        run_id=run_id,
        workspace=workspace_name,
        source_id="legacy",
        config_hash="legacy-backfill",
        started_at=now,
        finished_at=now,
        status="ok",
        rows_in=result.rows,
        rows_kept=result.rows,
        chunks_total=chunks,
        chunks_ok=chunks,
        chunks_failed=0,
    )


def _normalize_table(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


__all__ = ["BackfillResult", "BackfilledTable", "backfill_from_legacy_db"]
