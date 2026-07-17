"""DuckDB metadata ledger helpers."""

from __future__ import annotations

import fcntl
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl

from valuestream.store.meta import META_DB_FILENAMES, init_meta_dbs, meta_dir
from valuestream.utils.hashing import sha256_chained


@dataclass(frozen=True)
class RecoveredRun:
    """Summary of one interrupted run finalized under the source lock."""

    run_id: str
    status: str
    chunks_ok: int
    chunks_failed: int


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


def start_run(
    workspace_path: str | Path,
    *,
    run_id: str,
    workspace: str,
    source_id: str,
    config_hash: str,
    started_at: object,
    chunks_total: int,
) -> None:
    """Create the durable ``running`` publication barrier for a source run."""

    insert_run(
        workspace_path,
        run_id=run_id,
        workspace=workspace,
        source_id=source_id,
        config_hash=config_hash,
        started_at=started_at,
        finished_at=None,
        status="running",
        rows_in=0,
        rows_kept=0,
        chunks_total=chunks_total,
        chunks_ok=0,
        chunks_failed=0,
    )


def finalize_run(
    workspace_path: str | Path,
    *,
    run_id: str,
    finished_at: object,
    status: str,
    rows_in: int,
    rows_kept: int,
    chunks_total: int,
    chunks_ok: int,
    chunks_failed: int,
) -> None:
    """Finalize an existing run row without changing its immutable identity."""

    if status not in {"ok", "partial", "failed"}:
        raise ValueError(f"invalid final pipeline run status: {status!r}")
    ensure(workspace_path)
    with duckdb.connect(str(meta_dir(workspace_path) / "pipeline_runs.duckdb")) as conn:
        exists = conn.execute(
            "SELECT COUNT(*) FROM pipeline_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if not exists or exists[0] != 1:
            raise ValueError(f"pipeline run {run_id!r} does not exist")
        conn.execute(
            """
            UPDATE pipeline_runs
            SET finished_at = ?, status = ?, rows_in = ?, rows_kept = ?,
                chunks_total = ?, chunks_ok = ?, chunks_failed = ?
            WHERE id = ?
            """,
            (
                finished_at,
                status,
                rows_in,
                rows_kept,
                chunks_total,
                chunks_ok,
                chunks_failed,
                run_id,
            ),
        )


def finalize_incomplete_run(
    workspace_path: str | Path,
    *,
    run_id: str,
    finished_at: object,
) -> RecoveredRun:
    """Finalize an interrupted in-process run from its committed chunk rows."""

    summary = _run_chunk_summary(workspace_path, run_id)
    status = "partial" if summary["chunks_ok"] else "failed"
    finalize_run(
        workspace_path,
        run_id=run_id,
        finished_at=finished_at,
        status=status,
        rows_in=summary["rows_in"],
        rows_kept=summary["rows_kept"],
        chunks_total=summary["chunks_total"],
        chunks_ok=summary["chunks_ok"],
        chunks_failed=summary["chunks_failed"],
    )
    return RecoveredRun(
        run_id=run_id,
        status=status,
        chunks_ok=summary["chunks_ok"],
        chunks_failed=summary["chunks_failed"],
    )


def recover_stale_runs(
    workspace_path: str | Path,
    *,
    source_id: str,
    config_hash: str,
    file_hashes: Mapping[str, str],
    expected_outputs: Mapping[tuple[str, str], str],
    finished_at: object,
) -> tuple[RecoveredRun, ...]:
    """Verify and finalize prior ``running`` rows while holding the source lock.

    A successful chunk row is the durable commit marker, but recovery still
    validates its current input fingerprint, lineage, physical file set, and
    embedded provenance before publishing the stale run as ``partial``.
    """

    ensure(workspace_path)
    runs_db = meta_dir(workspace_path) / "pipeline_runs.duckdb"
    with duckdb.connect(str(runs_db), read_only=True) as conn:
        stale_rows = conn.execute(
            """
            SELECT CAST(id AS VARCHAR), config_hash
            FROM pipeline_runs
            WHERE source_id = ? AND status = 'running'
            ORDER BY started_at, id
            """,
            (source_id,),
        ).fetchall()

    recovered: list[RecoveredRun] = []
    for raw_run_id, raw_config_hash in stale_rows:
        run_id = str(raw_run_id)
        run_hash_matches = str(raw_config_hash) == config_hash
        for chunk_id, stored_hash in _ok_chunks_for_run(workspace_path, source_id, run_id):
            reason: str | None = None
            if not run_hash_matches:
                reason = "source computation hash changed before interrupted-run recovery"
            elif file_hashes.get(chunk_id) != stored_hash:
                reason = "input fingerprint changed before interrupted-run recovery"
            else:
                reason = _chunk_recovery_error(
                    workspace_path,
                    source_id=source_id,
                    run_id=run_id,
                    chunk_id=chunk_id,
                    expected_outputs=expected_outputs,
                )
            if reason is not None:
                _mark_chunk_recovery_failed(
                    workspace_path,
                    source_id=source_id,
                    run_id=run_id,
                    chunk_id=chunk_id,
                    reason=reason,
                )
        recovered.append(
            finalize_incomplete_run(
                workspace_path,
                run_id=run_id,
                finished_at=finished_at,
            )
        )
    return tuple(recovered)


def _run_chunk_summary(workspace_path: str | Path, run_id: str) -> dict[str, int]:
    chunks_db = meta_dir(workspace_path) / "chunks.duckdb"
    runs_db = meta_dir(workspace_path) / "pipeline_runs.duckdb"
    with duckdb.connect(str(chunks_db), read_only=True) as conn:
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(rows_in), 0),
                COALESCE(SUM(rows_kept), 0),
                COUNT(*) FILTER (WHERE status = 'ok'),
                COUNT(*) FILTER (WHERE status = 'failed'),
                COUNT(*)
            FROM chunks
            WHERE pipeline_run_id = ?
            """,
            (run_id,),
        ).fetchone()
    with duckdb.connect(str(runs_db), read_only=True) as conn:
        total_row = conn.execute(
            "SELECT chunks_total FROM pipeline_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    if row is None or total_row is None:
        raise ValueError(f"pipeline run {run_id!r} does not exist")
    recorded_total = int(total_row[0] or 0)
    return {
        "rows_in": int(row[0] or 0),
        "rows_kept": int(row[1] or 0),
        "chunks_ok": int(row[2] or 0),
        "chunks_failed": int(row[3] or 0),
        "chunks_total": max(recorded_total, int(row[4] or 0)),
    }


def _ok_chunks_for_run(
    workspace_path: str | Path,
    source_id: str,
    run_id: str,
) -> list[tuple[str, str]]:
    with duckdb.connect(str(meta_dir(workspace_path) / "chunks.duckdb"), read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT chunk_id, file_hash
            FROM chunks
            WHERE source_id = ? AND pipeline_run_id = ? AND status = 'ok'
            ORDER BY chunk_id
            """,
            (source_id, run_id),
        ).fetchall()
    return [(str(chunk_id), str(file_hash)) for chunk_id, file_hash in rows]


def _chunk_recovery_error(  # noqa: PLR0911
    workspace_path: str | Path,
    *,
    source_id: str,
    run_id: str,
    chunk_id: str,
    expected_outputs: Mapping[tuple[str, str], str],
) -> str | None:
    workspace = Path(workspace_path)
    aggregate_root = (workspace / "aggregates" / source_id).resolve()
    lineage_db = meta_dir(workspace) / "lineage.duckdb"
    with duckdb.connect(str(lineage_db), read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT processor_id, grain, period, partial_path, config_hash, rows
            FROM lineage
            WHERE source_id = ? AND pipeline_run_id = ? AND chunk_id = ?
            ORDER BY processor_id, grain, period
            """,
            (source_id, run_id, chunk_id),
        ).fetchall()

    lineage_paths = {Path(str(row[3])).resolve() for row in rows}
    safe_run = _safe_name(run_id)
    safe_chunk = _safe_name(chunk_id)
    physical_paths = {
        path.resolve()
        for path in aggregate_root.glob(f"*/*/period=*/part-{safe_run}-{safe_chunk}.parquet")
        if path.is_file()
    }
    if physical_paths != lineage_paths:
        return "physical aggregate files do not match committed lineage"

    for processor_id, grain, period, raw_path, lineage_hash, lineage_rows in rows:
        path = Path(str(raw_path)).resolve()
        if not path.is_relative_to(aggregate_root) or not path.is_file():
            return f"lineage path is missing or outside the source aggregate root: {path}"
        expected_hash = expected_outputs.get((str(processor_id), str(grain)))
        if expected_hash is None or str(lineage_hash) != expected_hash:
            return "lineage processor/grain or computation hash is not current"
        relative = path.relative_to(aggregate_root)
        if len(relative.parts) < 4:
            return f"lineage path has an invalid aggregate layout: {path}"
        path_processor, path_grain, path_period = relative.parts[:3]
        if (
            path_processor != str(processor_id)
            or path_grain != str(grain)
            or path_period != f"period={period}"
        ):
            return f"lineage metadata does not match aggregate path: {path}"
        try:
            summary = (
                pl.scan_parquet(path)
                .select(
                    pl.col("pipeline_run_id").cast(pl.String).drop_nulls().unique().implode(),
                    pl.col("chunk_id").cast(pl.String).drop_nulls().unique().implode(),
                    pl.col("config_hash").cast(pl.String).drop_nulls().unique().implode(),
                    pl.col("period").cast(pl.String).drop_nulls().unique().implode(),
                    pl.len().alias("rows"),
                )
                .collect()
                .row(0)
            )
        except Exception as exc:
            return f"aggregate provenance cannot be read: {path}: {exc}"
        run_ids, chunk_ids, config_hashes, periods, physical_rows = summary
        if run_ids != [run_id] or chunk_ids != [chunk_id]:
            return f"aggregate run/chunk provenance does not match lineage: {path}"
        if config_hashes != [expected_hash] or periods != [str(period)]:
            return f"aggregate config/period provenance does not match lineage: {path}"
        if lineage_rows is not None and int(physical_rows) != int(lineage_rows):
            return f"aggregate row count does not match lineage: {path}"
    return None


def _mark_chunk_recovery_failed(
    workspace_path: str | Path,
    *,
    source_id: str,
    run_id: str,
    chunk_id: str,
    reason: str,
) -> None:
    with duckdb.connect(str(meta_dir(workspace_path) / "chunks.duckdb")) as conn:
        conn.execute(
            """
            UPDATE chunks
            SET status = 'failed', error = ?
            WHERE source_id = ? AND pipeline_run_id = ? AND chunk_id = ?
            """,
            (f"recovery verification failed: {reason}", source_id, run_id, chunk_id),
        )


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_" for character in value
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
) -> int:
    """Record complete file lineage and return the committed record count."""

    ensure(workspace_path)
    aggregate_root = Path(workspace_path) / "aggregates"
    records: list[tuple[object, ...]] = []
    for path in paths:
        try:
            relative = path.relative_to(aggregate_root)
        except ValueError as exc:
            raise ValueError(f"aggregate lineage path is outside the workspace: {path}") from exc
        if len(relative.parts) < 5 or not relative.parts[3].startswith("period="):
            raise ValueError(f"aggregate lineage path has an invalid layout: {path}")
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
    if len(records) != len(paths):
        raise RuntimeError("not every aggregate path produced a lineage record")
    if not records:
        return 0
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
    return len(records)


__all__ = [
    "RecoveredRun",
    "aggregate_lineage_paths",
    "chunk_done",
    "done_chunk_ids",
    "ensure",
    "file_fingerprint",
    "finalize_incomplete_run",
    "finalize_run",
    "insert_chunk",
    "insert_config_version",
    "insert_lineage_files",
    "insert_run",
    "recover_stale_runs",
    "source_run_lock",
    "start_run",
    "successful_chunk_keys",
]
