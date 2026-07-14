"""DuckDB SQL views over aggregate parquet files."""

from __future__ import annotations

from pathlib import Path

import duckdb

from valuestream.config import model
from valuestream.config.canonical import processor_computation_hash
from valuestream.store.meta import meta_dir
from valuestream.store.parquet import aggregate_dir, aggregate_exists
from valuestream.utils.timer import timed


def views_db_path(workspace_path: str | Path) -> Path:
    """Return the DuckDB file that holds aggregate SQL views."""
    return meta_dir(workspace_path) / "aggregate_views.duckdb"


def aggregate_view_name(source_id: str, processor_id: str, grain: str) -> str:
    """Return the stable SQL view name for an aggregate."""
    return "_".join(
        [
            "aggregate",
            _safe_identifier(source_id),
            _safe_identifier(processor_id),
            _safe_identifier(grain),
        ]
    )


@timed
def refresh_aggregate_views(
    workspace_path: str | Path,
    catalog: model.Catalog,
) -> list[str]:
    """Create or refresh DuckDB views for every existing aggregate."""
    db_path = views_db_path(workspace_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    refreshed: list[str] = []
    with duckdb.connect(str(db_path)) as conn:
        _refresh_successful_chunks_table(conn, workspace_path)
        for processor in catalog.processors.processors:
            for grain in processor.grains:
                name = aggregate_view_name(processor.source, processor.id, grain)
                conn.execute(f'DROP VIEW IF EXISTS "{name}"')
                if not aggregate_exists(
                    workspace_path,
                    source_id=processor.source,
                    processor_id=processor.id,
                    grain=grain,
                ):
                    continue
                pattern = str(
                    aggregate_dir(
                        workspace_path,
                        source_id=processor.source,
                        processor_id=processor.id,
                        grain=grain,
                    )
                    / "**"
                    / "*.parquet"
                )
                conn.execute(
                    f"""
                    CREATE VIEW "{name}" AS
                    SELECT aggregate_rows.*
                    FROM read_parquet(
                        '{_sql_string(pattern)}',
                        hive_partitioning = true,
                        union_by_name = true
                    ) AS aggregate_rows
                    JOIN successful_chunks AS chunks
                      ON CAST(aggregate_rows.pipeline_run_id AS VARCHAR) = chunks.pipeline_run_id
                     AND aggregate_rows.chunk_id = chunks.chunk_id
                     AND chunks.source_id = '{_sql_string(processor.source)}'
                    WHERE aggregate_rows.config_hash = '{processor_computation_hash(catalog, processor)}'
                    QUALIFY aggregate_rows.created_at = MAX(aggregate_rows.created_at)
                        OVER (PARTITION BY aggregate_rows.chunk_id, aggregate_rows.config_hash)
                    """
                )
                refreshed.append(name)
    return refreshed


def _refresh_successful_chunks_table(
    conn: duckdb.DuckDBPyConnection,
    workspace_path: str | Path,
) -> None:
    chunks_db = meta_dir(workspace_path) / "chunks.duckdb"
    runs_db = meta_dir(workspace_path) / "pipeline_runs.duckdb"
    conn.execute("DROP TABLE IF EXISTS successful_chunks")
    if not chunks_db.exists() or not runs_db.exists():
        conn.execute(
            """
            CREATE TABLE successful_chunks (
                pipeline_run_id VARCHAR,
                chunk_id VARCHAR,
                source_id VARCHAR
            )
            """
        )
        return
    conn.execute(f"ATTACH '{_sql_string(str(chunks_db))}' AS chunks_meta (READ_ONLY)")
    conn.execute(f"ATTACH '{_sql_string(str(runs_db))}' AS runs_meta (READ_ONLY)")
    conn.execute(
        """
        CREATE TABLE successful_chunks AS
        SELECT CAST(c.pipeline_run_id AS VARCHAR) AS pipeline_run_id,
               c.chunk_id,
               c.source_id
        FROM chunks_meta.chunks c
        JOIN runs_meta.pipeline_runs r ON c.pipeline_run_id = r.id
        WHERE c.status = 'ok'
          AND r.status IN ('ok', 'partial')
        """
    )
    conn.execute("DETACH chunks_meta")
    conn.execute("DETACH runs_meta")


def _safe_identifier(value: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in value]
    out = "".join(chars).strip("_") or "x"
    return out if out[0].isalpha() else f"x_{out}"


def _sql_string(value: str) -> str:
    return value.replace("'", "''")


__all__ = ["aggregate_view_name", "refresh_aggregate_views", "views_db_path"]
