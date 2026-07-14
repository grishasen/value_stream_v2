"""DuckDB metadata ledger helpers."""

from __future__ import annotations

import fcntl
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import duckdb
import polars as pl

from valuestream.store.meta import META_DB_FILENAMES, init_meta_dbs, meta_dir
from valuestream.utils.hashing import sha256_chained


def ensure(workspace_path: str | Path) -> None:
    """Create metadata databases if needed.

    Fast path: when all metadata files already exist this is four ``stat``
    calls, so callers on hot paths (per-chunk inserts) pay no DuckDB
    connection or DDL cost.
    """
    metadata_dir = meta_dir(workspace_path)
    if all((metadata_dir / name).exists() for name in META_DB_FILENAMES):
        return
    init_meta_dbs(workspace_path)


def file_fingerprint(files: list[Path] | tuple[Path, ...]) -> str:
    """Hash file paths, mtimes, and sizes for chunk idempotency metadata."""
    parts: list[str] = []
    for path in sorted(files):
        stat = path.stat()
        parts.append(f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}")
    return sha256_chained(parts)


def chunk_done(
    workspace_path: str | Path,
    *,
    source_id: str,
    chunk_id: str,
    files: list[Path] | tuple[Path, ...],
    config_hash: str,
) -> bool:
    """Return true if this source/chunk/config has a successful prior run."""
    _ensure_ledger_read_dbs(workspace_path)
    chunks_db = meta_dir(workspace_path) / "chunks.duckdb"
    runs_db = meta_dir(workspace_path) / "pipeline_runs.duckdb"
    with duckdb.connect(str(chunks_db), read_only=True) as conn:
        _attach_runs_db(conn, runs_db)
        fingerprint = file_fingerprint(files)
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM chunks c
            JOIN runs_db.pipeline_runs r ON c.pipeline_run_id = r.id
            WHERE c.source_id = ?
              AND c.chunk_id = ?
              AND c.status = 'ok'
              AND c.file_hash = ?
              AND r.config_hash = ?
              AND r.status IN ('ok', 'partial')
            """,
            (source_id, chunk_id, fingerprint, config_hash),
        ).fetchone()
    return bool(row and row[0] > 0)


def done_chunk_ids(
    workspace_path: str | Path,
    *,
    source_id: str,
    config_hash: str,
    file_hashes: Mapping[str, str] | None = None,
) -> set[str]:
    """Return chunk ids with a successful prior run under ``config_hash``.

    One query replaces a per-chunk :func:`chunk_done` loop when planning a
    source run.
    """
    _ensure_ledger_read_dbs(workspace_path)
    chunks_db = meta_dir(workspace_path) / "chunks.duckdb"
    runs_db = meta_dir(workspace_path) / "pipeline_runs.duckdb"
    with duckdb.connect(str(chunks_db), read_only=True) as conn:
        _attach_runs_db(conn, runs_db)
        rows = conn.execute(
            """
            SELECT c.chunk_id, c.file_hash
            FROM chunks c
            JOIN runs_db.pipeline_runs r ON c.pipeline_run_id = r.id
            WHERE c.source_id = ?
              AND c.status = 'ok'
              AND r.config_hash = ?
              AND r.status IN ('ok', 'partial')
            """,
            (source_id, config_hash),
        ).fetchall()
    if file_hashes is None:
        return {str(chunk_id) for chunk_id, _ in rows}
    return {
        str(chunk_id)
        for chunk_id, stored_hash in rows
        if file_hashes.get(str(chunk_id)) == str(stored_hash)
    }


@contextmanager
def source_run_lock(workspace_path: str | Path, source_id: str) -> Iterator[None]:
    """Prevent concurrent ingestion runs for one workspace source."""

    ensure(workspace_path)
    safe_source = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in source_id
    )
    path = meta_dir(workspace_path) / f"source_{safe_source}.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"source {source_id!r} already has an ingestion run in progress"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def successful_chunk_keys(
    workspace_path: str | Path,
    *,
    source_id: str,
) -> set[tuple[str, str]]:
    """Return ``(pipeline_run_id, chunk_id)`` pairs that are safe to query."""
    _ensure_ledger_read_dbs(workspace_path)
    chunks_db = meta_dir(workspace_path) / "chunks.duckdb"
    runs_db = meta_dir(workspace_path) / "pipeline_runs.duckdb"
    with duckdb.connect(str(chunks_db), read_only=True) as conn:
        _attach_runs_db(conn, runs_db)
        rows = conn.execute(
            """
            SELECT CAST(c.pipeline_run_id AS VARCHAR), c.chunk_id
            FROM chunks c
            JOIN runs_db.pipeline_runs r ON c.pipeline_run_id = r.id
            WHERE c.source_id = ?
              AND c.status = 'ok'
              AND r.status IN ('ok', 'partial')
            """,
            (source_id,),
        ).fetchall()
    return {(str(run_id), str(chunk_id)) for run_id, chunk_id in rows}


def aggregate_lineage_paths(
    workspace_path: str | Path,
    *,
    source_id: str,
    processor_id: str,
    grain: str,
) -> dict[str, tuple[Path, ...]]:
    """Return aggregate partial paths grouped by processor computation hash."""
    _ensure_ledger_read_dbs(workspace_path)
    lineage_db = meta_dir(workspace_path) / "lineage.duckdb"
    if not lineage_db.exists():
        ensure(workspace_path)
    with duckdb.connect(str(lineage_db), read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT config_hash, partial_path
            FROM lineage
            WHERE source_id = ?
              AND processor_id = ?
              AND grain = ?
            ORDER BY config_hash, partial_path
            """,
            (source_id, processor_id, grain),
        ).fetchall()
    grouped: dict[str, list[Path]] = {}
    for config_hash, partial_path in rows:
        grouped.setdefault(str(config_hash), []).append(Path(str(partial_path)))
    return {config_hash: tuple(paths) for config_hash, paths in grouped.items()}


def _attach_runs_db(conn: duckdb.DuckDBPyConnection, runs_db: str | Path) -> None:
    try:
        conn.execute(f"ATTACH '{_sql_string(str(runs_db))}' AS runs_db (READ_ONLY)")
    except duckdb.BinderException as exc:
        if "already exists" not in str(exc):
            raise


def _ensure_ledger_read_dbs(workspace_path: str | Path) -> None:
    metadata_dir = meta_dir(workspace_path)
    if (metadata_dir / "chunks.duckdb").exists() and (
        metadata_dir / "pipeline_runs.duckdb"
    ).exists():
        return
    ensure(workspace_path)


def _sql_string(value: str) -> str:
    return value.replace("'", "''")


def insert_run(
    workspace_path: str | Path,
    *,
    run_id: str,
    workspace: str,
    source_id: str,
    config_hash: str,
    started_at: object,
    finished_at: object,
    status: str,
    rows_in: int,
    rows_kept: int,
    chunks_total: int,
    chunks_ok: int,
    chunks_failed: int,
) -> None:
    """Insert one pipeline run row."""
    ensure(workspace_path)
    with duckdb.connect(str(meta_dir(workspace_path) / "pipeline_runs.duckdb")) as conn:
        conn.execute(
            """
            INSERT INTO pipeline_runs
            (id, workspace, source_id, config_hash, started_at, finished_at, status,
             rows_in, rows_kept, chunks_total, chunks_ok, chunks_failed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                workspace,
                source_id,
                config_hash,
                started_at,
                finished_at,
                status,
                rows_in,
                rows_kept,
                chunks_total,
                chunks_ok,
                chunks_failed,
            ),
        )


def insert_chunk(
    workspace_path: str | Path,
    *,
    source_id: str,
    chunk_id: str,
    files: list[Path] | tuple[Path, ...],
    rows_in: int,
    rows_kept: int,
    started_at: object,
    finished_at: object,
    status: str,
    error: str | None,
    pipeline_run_id: str,
) -> None:
    """Insert one chunk ledger row."""
    ensure(workspace_path)
    with duckdb.connect(str(meta_dir(workspace_path) / "chunks.duckdb")) as conn:
        conn.execute(
            """
            INSERT INTO chunks
            (source_id, chunk_id, files, file_hash, rows_in, rows_kept,
             started_at, finished_at, status, error, pipeline_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                chunk_id,
                json.dumps([str(path) for path in files]),
                file_fingerprint(files),
                rows_in,
                rows_kept,
                started_at,
                finished_at,
                status,
                error,
                pipeline_run_id,
            ),
        )


def insert_config_version(
    workspace_path: str | Path,
    *,
    config_hash: str,
    yaml: str,
    introduced_at: object,
) -> None:
    """Persist one canonical catalog or computation contract once."""

    ensure(workspace_path)
    with duckdb.connect(str(meta_dir(workspace_path) / "config_versions.duckdb")) as conn:
        conn.execute(
            """
            INSERT INTO config_versions (config_hash, yaml, introduced_at)
            VALUES (?, ?, ?)
            ON CONFLICT (config_hash) DO NOTHING
            """,
            (config_hash, yaml, introduced_at),
        )


def insert_lineage_files(
    workspace_path: str | Path,
    *,
    pipeline_run_id: str,
    chunk_id: str,
    paths: tuple[Path, ...] | list[Path],
) -> None:
    """Record file-level lineage for successfully produced aggregate partials."""

    ensure(workspace_path)
    aggregate_root = Path(workspace_path) / "aggregates"
    records: list[tuple[object, ...]] = []
    for path in paths:
        try:
            relative = path.relative_to(aggregate_root)
        except ValueError:
            continue
        if len(relative.parts) < 5 or not relative.parts[3].startswith("period="):
            continue
        source_id, processor_id, grain = relative.parts[:3]
        period = relative.parts[3].removeprefix("period=")
        summary = (
            pl.scan_parquet(path)
            .select(
                pl.col("config_hash").first().alias("config_hash"),
                pl.col("created_at").max().alias("created_at"),
                pl.len().alias("rows"),
            )
            .collect()
            .row(0, named=True)
        )
        records.append(
            (
                pipeline_run_id,
                chunk_id,
                source_id,
                processor_id,
                grain,
                period,
                str(path),
                str(summary["config_hash"]),
                int(summary["rows"]),
                summary["created_at"],
            )
        )
    if not records:
        return
    with duckdb.connect(str(meta_dir(workspace_path) / "lineage.duckdb")) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO lineage
            (pipeline_run_id, chunk_id, source_id, processor_id, grain, period,
             partial_path, config_hash, rows, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )


__all__ = [
    "aggregate_lineage_paths",
    "chunk_done",
    "done_chunk_ids",
    "ensure",
    "file_fingerprint",
    "insert_chunk",
    "insert_config_version",
    "insert_lineage_files",
    "insert_run",
    "source_run_lock",
    "successful_chunk_keys",
]
