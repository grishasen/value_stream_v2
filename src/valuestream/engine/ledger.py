"""DuckDB metadata ledger helpers."""

from __future__ import annotations

import fcntl
import json
import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl

from valuestream.store.meta import META_DB_FILENAMES, init_meta_dbs, meta_dir
from valuestream.store.parquet import AggregateWriteReceipt
from valuestream.utils.hashing import sha256_chained

logger = logging.getLogger(__name__)


_RECOVERY_PATH_COLUMN = "__recovery_path"


@dataclass(frozen=True)
class RecoveredRun:
    """Summary of one interrupted run finalized under the source lock."""

    run_id: str
    status: str
    chunks_ok: int
    chunks_failed: int


@dataclass(frozen=True)
class RecoveryProgress:
    """Live progress for one batched interrupted-run provenance scan."""

    source_id: str
    run_id: str
    processor_id: str
    grain: str
    group_order: int
    groups_total: int
    files: tuple[Path, ...]


RecoveryProgressCallback = Callable[[RecoveryProgress], None]


@dataclass(frozen=True)
class _RecoveryLineage:
    """One persisted lineage record needed to verify an interrupted run."""

    processor_id: str
    grain: str
    period: str
    path: Path
    config_hash: str
    rows: int | None


@dataclass(frozen=True)
class _RecoveryFileSummary:
    """Embedded provenance summarized for one aggregate part file."""

    run_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    config_hashes: tuple[str, ...]
    periods: tuple[str, ...]
    rows: int
    has_null_provenance: bool


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
    progress_callback: RecoveryProgressCallback | None = None,
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
        committed_chunks = _ok_chunks_for_run(workspace_path, source_id, run_id)
        logger.info(
            "Verifying interrupted source run: source=%s run_id=%s committed_chunks=%s",
            source_id,
            run_id,
            len(committed_chunks),
        )
        errors: dict[str, str] = {}
        deep_candidates: list[str] = []
        for chunk_id, stored_hash in committed_chunks:
            if not run_hash_matches:
                errors[chunk_id] = "source computation hash changed before interrupted-run recovery"
            elif file_hashes.get(chunk_id) != stored_hash:
                errors[chunk_id] = "input fingerprint changed before interrupted-run recovery"
            else:
                deep_candidates.append(chunk_id)
        if deep_candidates:
            errors.update(
                _run_recovery_errors(
                    workspace_path,
                    source_id=source_id,
                    run_id=run_id,
                    chunk_ids=tuple(deep_candidates),
                    expected_outputs=expected_outputs,
                    progress_callback=progress_callback,
                )
            )
        if errors:
            _mark_chunk_recovery_failures(
                workspace_path,
                source_id=source_id,
                run_id=run_id,
                errors=errors,
            )
        logger.info(
            "Interrupted source run verification finished: source=%s run_id=%s "
            "retained_chunks=%s rejected_chunks=%s",
            source_id,
            run_id,
            len(committed_chunks) - len(errors),
            len(errors),
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
                COALESCE(SUM(rows_in) FILTER (WHERE status = 'ok'), 0),
                COALESCE(SUM(rows_kept) FILTER (WHERE status = 'ok'), 0),
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


def _run_recovery_errors(
    workspace_path: str | Path,
    *,
    source_id: str,
    run_id: str,
    chunk_ids: tuple[str, ...],
    expected_outputs: Mapping[tuple[str, str], str],
    progress_callback: RecoveryProgressCallback | None,
) -> dict[str, str]:
    """Deep-verify retained chunks with one metadata query and batched file scans."""

    workspace = Path(workspace_path)
    aggregate_root = (workspace / "aggregates" / source_id).resolve()
    records_by_chunk = _lineage_records_for_run(
        workspace,
        source_id=source_id,
        run_id=run_id,
    )
    physical_path_index = _physical_recovery_path_index(aggregate_root, run_id)
    errors: dict[str, str] = {}
    scan_groups: dict[tuple[str, str], set[Path]] = {}

    for chunk_id in chunk_ids:
        records = records_by_chunk.get(chunk_id, ())
        reason = _chunk_recovery_metadata_error(
            records,
            aggregate_root=aggregate_root,
            run_id=run_id,
            chunk_id=chunk_id,
            physical_path_index=physical_path_index,
            expected_outputs=expected_outputs,
        )
        if reason is not None:
            errors[chunk_id] = reason
            continue
        for record in records:
            scan_groups.setdefault((record.processor_id, record.grain), set()).add(record.path)

    summaries, scan_errors = _scan_recovery_groups(
        source_id,
        run_id,
        scan_groups,
        progress_callback=progress_callback,
    )
    for chunk_id in chunk_ids:
        if chunk_id in errors:
            continue
        for record in records_by_chunk.get(chunk_id, ()):
            reason = _recovery_provenance_error(
                record,
                run_id=run_id,
                chunk_id=chunk_id,
                expected_hash=expected_outputs[(record.processor_id, record.grain)],
                summaries=summaries,
                scan_errors=scan_errors,
            )
            if reason is not None:
                errors[chunk_id] = reason
                break
    return errors


def _lineage_records_for_run(
    workspace_path: str | Path,
    *,
    source_id: str,
    run_id: str,
) -> dict[str, tuple[_RecoveryLineage, ...]]:
    """Fetch all lineage for one stale run in a single metadata query."""

    lineage_db = meta_dir(workspace_path) / "lineage.duckdb"
    with duckdb.connect(str(lineage_db), read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT chunk_id, processor_id, grain, period, partial_path, config_hash, rows
            FROM lineage
            WHERE source_id = ? AND pipeline_run_id = ?
            ORDER BY chunk_id, processor_id, grain, period
            """,
            (source_id, run_id),
        ).fetchall()
    grouped: dict[str, list[_RecoveryLineage]] = {}
    for chunk_id, processor_id, grain, period, raw_path, config_hash, row_count in rows:
        grouped.setdefault(str(chunk_id), []).append(
            _RecoveryLineage(
                processor_id=str(processor_id),
                grain=str(grain),
                period=str(period),
                path=Path(str(raw_path)).resolve(),
                config_hash=str(config_hash),
                rows=int(row_count) if row_count is not None else None,
            )
        )
    return {chunk_id: tuple(records) for chunk_id, records in grouped.items()}


def _physical_recovery_path_index(
    aggregate_root: Path,
    run_id: str,
) -> dict[str, frozenset[Path]]:
    """Index physical part paths for one stale run with a single tree traversal."""

    safe_run = _safe_name(run_id)
    grouped: dict[str, set[Path]] = {}
    for path in aggregate_root.glob(f"*/*/period=*/part-{safe_run}-*.parquet"):
        if path.is_file():
            grouped.setdefault(path.name, set()).add(path.resolve())
    return {name: frozenset(paths) for name, paths in grouped.items()}


def _chunk_recovery_metadata_error(
    records: tuple[_RecoveryLineage, ...],
    *,
    aggregate_root: Path,
    run_id: str,
    chunk_id: str,
    physical_path_index: Mapping[str, frozenset[Path]],
    expected_outputs: Mapping[tuple[str, str], str],
) -> str | None:
    """Validate lineage completeness, layout, and current computation hashes."""

    expected_name = f"part-{_safe_name(run_id)}-{_safe_name(chunk_id)}.parquet"
    physical_paths = physical_path_index.get(expected_name, frozenset())
    lineage_paths = {record.path for record in records}
    if physical_paths != lineage_paths:
        return "physical aggregate files do not match committed lineage"

    for record in records:
        path = record.path
        if not path.is_relative_to(aggregate_root) or not path.is_file():
            return f"lineage path is missing or outside the source aggregate root: {path}"
        expected_hash = expected_outputs.get((record.processor_id, record.grain))
        if expected_hash is None or record.config_hash != expected_hash:
            return "lineage processor/grain or computation hash is not current"
        relative = path.relative_to(aggregate_root)
        if len(relative.parts) < 4:
            return f"lineage path has an invalid aggregate layout: {path}"
        path_processor, path_grain, path_period = relative.parts[:3]
        if (
            path_processor != record.processor_id
            or path_grain != record.grain
            or path_period != f"period={record.period}"
        ):
            return f"lineage metadata does not match aggregate path: {path}"
    return None


def _scan_recovery_groups(
    source_id: str,
    run_id: str,
    groups: Mapping[tuple[str, str], set[Path]],
    *,
    progress_callback: RecoveryProgressCallback | None,
) -> tuple[dict[Path, _RecoveryFileSummary], dict[Path, str]]:
    """Scan schema-compatible processor/grain files together, isolating bad files on error."""

    summaries: dict[Path, _RecoveryFileSummary] = {}
    errors: dict[Path, str] = {}
    ordered_groups = sorted(groups.items())
    for index, ((processor_id, grain), raw_paths) in enumerate(ordered_groups, start=1):
        paths = tuple(sorted(raw_paths))
        if progress_callback is not None:
            progress_callback(
                RecoveryProgress(
                    source_id=source_id,
                    run_id=run_id,
                    processor_id=processor_id,
                    grain=grain,
                    group_order=index,
                    groups_total=len(ordered_groups),
                    files=paths,
                )
            )
        logger.info(
            "Verifying interrupted run aggregate group: run_id=%s group=%s/%s "
            "processor=%s grain=%s files=%s",
            run_id,
            index,
            len(ordered_groups),
            processor_id,
            grain,
            len(paths),
        )
        try:
            summaries.update(_scan_recovery_group(paths))
        except Exception:
            logger.warning(
                "Batched recovery provenance scan failed; retrying files separately: "
                "run_id=%s processor=%s grain=%s files=%s",
                run_id,
                processor_id,
                grain,
                len(paths),
                exc_info=True,
            )
            for path in paths:
                try:
                    summaries.update(_scan_recovery_group((path,)))
                except Exception as exc:
                    errors[path] = str(exc)
    return summaries, errors


def _scan_recovery_group(paths: tuple[Path, ...]) -> dict[Path, _RecoveryFileSummary]:
    """Read only embedded provenance columns and summarize each supplied Parquet file."""

    if not paths:
        return {}
    frame = (
        pl.scan_parquet(
            [str(path) for path in paths],
            glob=False,
            hive_partitioning=False,
            include_file_paths=_RECOVERY_PATH_COLUMN,
        )
        .group_by(_RECOVERY_PATH_COLUMN)
        .agg(
            pl.col("pipeline_run_id").cast(pl.String).drop_nulls().unique().sort().alias("run_ids"),
            pl.col("chunk_id").cast(pl.String).drop_nulls().unique().sort().alias("chunk_ids"),
            pl.col("config_hash")
            .cast(pl.String)
            .drop_nulls()
            .unique()
            .sort()
            .alias("config_hashes"),
            pl.col("period").cast(pl.String).drop_nulls().unique().sort().alias("periods"),
            pl.len().alias("rows"),
            (
                pl.col("pipeline_run_id").is_null().any()
                | pl.col("chunk_id").is_null().any()
                | pl.col("config_hash").is_null().any()
                | pl.col("period").is_null().any()
            ).alias("has_null_provenance"),
        )
        .collect()
    )
    summaries: dict[Path, _RecoveryFileSummary] = {}
    for row in frame.iter_rows(named=True):
        path = Path(str(row[_RECOVERY_PATH_COLUMN])).resolve()
        summaries[path] = _RecoveryFileSummary(
            run_ids=tuple(str(value) for value in row["run_ids"]),
            chunk_ids=tuple(str(value) for value in row["chunk_ids"]),
            config_hashes=tuple(str(value) for value in row["config_hashes"]),
            periods=tuple(str(value) for value in row["periods"]),
            rows=int(row["rows"]),
            has_null_provenance=bool(row["has_null_provenance"]),
        )
    return summaries


def _recovery_provenance_error(  # noqa: PLR0911 - explicit invariant diagnostics
    record: _RecoveryLineage,
    *,
    run_id: str,
    chunk_id: str,
    expected_hash: str,
    summaries: Mapping[Path, _RecoveryFileSummary],
    scan_errors: Mapping[Path, str],
) -> str | None:
    """Compare one lineage record with its batched physical-file summary."""

    path = record.path
    if path in scan_errors:
        return f"aggregate provenance cannot be read: {path}: {scan_errors[path]}"
    summary = summaries.get(path)
    if summary is None:
        return f"aggregate provenance cannot be read: {path}: file contains no rows"
    if summary.has_null_provenance:
        return f"aggregate provenance contains null run/chunk/config/period values: {path}"
    if summary.run_ids != (run_id,) or summary.chunk_ids != (chunk_id,):
        return f"aggregate run/chunk provenance does not match lineage: {path}"
    if summary.config_hashes != (expected_hash,) or summary.periods != (record.period,):
        return f"aggregate config/period provenance does not match lineage: {path}"
    if record.rows is not None and summary.rows != record.rows:
        return f"aggregate row count does not match lineage: {path}"
    return None


def _mark_chunk_recovery_failures(
    workspace_path: str | Path,
    *,
    source_id: str,
    run_id: str,
    errors: Mapping[str, str],
) -> None:
    with duckdb.connect(str(meta_dir(workspace_path) / "chunks.duckdb")) as conn:
        conn.executemany(
            """
            UPDATE chunks
            SET status = 'failed', error = ?
            WHERE source_id = ? AND pipeline_run_id = ? AND chunk_id = ?
            """,
            [
                (f"recovery verification failed: {reason}", source_id, run_id, chunk_id)
                for chunk_id, reason in errors.items()
            ],
        )


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_" for character in value
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
    aggregate_root = (Path(workspace_path) / "aggregates").resolve()
    records: list[tuple[object, ...]] = []
    for raw_path in paths:
        path = raw_path.resolve()
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


def insert_lineage_records(
    workspace_path: str | Path,
    *,
    records: tuple[AggregateWriteReceipt, ...] | list[AggregateWriteReceipt],
) -> int:
    """Persist write-time lineage receipts without reopening aggregate files."""

    ensure(workspace_path)
    if not records:
        return 0
    aggregate_root = (Path(workspace_path) / "aggregates").resolve()
    values: list[tuple[object, ...]] = []
    for record in records:
        path = record.path.resolve()
        try:
            relative = path.relative_to(aggregate_root)
        except ValueError as exc:
            raise ValueError(
                f"aggregate lineage path is outside the workspace: {record.path}"
            ) from exc
        expected_prefix = (
            record.source_id,
            record.processor_id,
            record.grain,
            f"period={record.period}",
        )
        if len(relative.parts) < 5 or relative.parts[:4] != expected_prefix:
            raise ValueError(
                f"aggregate write receipt does not match its physical path: {record.path}"
            )
        try:
            size_bytes = path.stat().st_size
        except FileNotFoundError as exc:
            raise ValueError(f"aggregate write receipt path is missing: {record.path}") from exc
        if size_bytes != record.size_bytes:
            raise ValueError(
                f"aggregate write receipt size does not match its physical file: {record.path}"
            )
        values.append(
            (
                record.pipeline_run_id,
                record.chunk_id,
                record.source_id,
                record.processor_id,
                record.grain,
                record.period,
                str(path),
                record.config_hash,
                record.rows,
                record.created_at,
            )
        )
    with duckdb.connect(str(meta_dir(workspace_path) / "lineage.duckdb")) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO lineage
            (pipeline_run_id, chunk_id, source_id, processor_id, grain, period,
             partial_path, config_hash, rows, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
    return len(values)


__all__ = [
    "RecoveredRun",
    "RecoveryProgress",
    "RecoveryProgressCallback",
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
    "insert_lineage_records",
    "insert_run",
    "recover_stale_runs",
    "source_run_lock",
    "start_run",
    "successful_chunk_keys",
]
