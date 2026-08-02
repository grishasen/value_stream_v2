"""Retention cleanup for Phase 1 aggregate storage."""

from __future__ import annotations

import datetime as dt
import shutil
import tempfile
from collections.abc import Mapping, Set
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import duckdb
import polars as pl
import polars.selectors as cs

from valuestream.config import model
from valuestream.config.canonical import processor_computation_hash
from valuestream.store.meta import meta_dir
from valuestream.utils.timer import timed


@dataclass(frozen=True)
class VacuumResult:
    """Summary of a vacuum run."""

    files_deleted: int
    dirs_deleted: int
    bytes_deleted: int
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class _Candidate:
    path: Path
    is_successful: bool
    created_order_ns: int


@dataclass(frozen=True)
class _FileMetadata:
    config_hashes: frozenset[str]
    chunk_ids: frozenset[str]
    run_ids: frozenset[str]
    created_order_ns: int


@timed
def vacuum_processor_state(
    workspace_path: str | Path,
    catalog: model.Catalog,
    *,
    include_tmp: bool = True,
    dry_run: bool = False,
    source_ids: Set[str] | None = None,
) -> VacuumResult:
    """Apply bounded retention to rebuildable processor checkpoints only."""

    workspace = Path(workspace_path)
    scoped_source_ids = None if source_ids is None else frozenset(source_ids)
    running_run_ids = _running_run_ids(workspace, scoped_source_ids)
    deleted_dirs: list[Path] = []
    checkpoint_dirs = _stale_processor_state_dirs(
        workspace,
        catalog,
        source_ids=scoped_source_ids,
        running_run_ids=running_run_ids,
        include_tmp=include_tmp,
    )
    bytes_deleted = _remove_dirs(checkpoint_dirs, deleted_dirs, dry_run)
    return VacuumResult(
        files_deleted=0,
        dirs_deleted=len(deleted_dirs),
        bytes_deleted=bytes_deleted,
        paths=tuple(deleted_dirs),
    )


@timed
def vacuum_workspace(
    workspace_path: str | Path,
    catalog: model.Catalog,
    *,
    include_tmp: bool = True,
    dry_run: bool = False,
    source_ids: Set[str] | None = None,
    retained_run_ids: Mapping[str, str] | None = None,
) -> VacuumResult:
    """Delete superseded aggregates, processor checkpoints, and reader temp dirs.

    ``source_ids`` limits aggregate cleanup to those sources. When
    ``retained_run_ids`` is supplied, it must cover exactly that source scope;
    aggregate files from any other run in the scope are removed. This mode is
    used only after a successful, exclusive clean rebuild.
    """
    workspace = Path(workspace_path)
    scoped_source_ids = None if source_ids is None else frozenset(source_ids)
    retained = None if retained_run_ids is None else dict(retained_run_ids)
    if retained is not None:
        retained_sources = frozenset(retained)
        if scoped_source_ids is None:
            scoped_source_ids = retained_sources
        elif scoped_source_ids != retained_sources:
            raise ValueError("retained_run_ids must cover exactly the vacuum source scope")
    deleted_files: list[Path] = []
    deleted_dirs: list[Path] = []
    running_run_ids = _running_run_ids(workspace, scoped_source_ids)

    bytes_deleted = _remove_files(
        _aggregate_temp_files(workspace, scoped_source_ids, running_run_ids),
        deleted_files,
        dry_run,
    )
    current_hashes = _current_processor_hashes(catalog, scoped_source_ids)
    candidates, stale = _partition_aggregate_files(
        workspace,
        current_hashes,
        source_ids=scoped_source_ids,
        retained_run_ids=retained,
        running_run_ids=running_run_ids,
    )
    bytes_deleted += _remove_files(stale, deleted_files, dry_run)
    bytes_deleted += _remove_files(
        _superseded_candidates(candidates),
        deleted_files,
        dry_run,
    )
    checkpoint_dirs = _stale_processor_state_dirs(
        workspace,
        catalog,
        source_ids=scoped_source_ids,
        running_run_ids=running_run_ids,
        include_tmp=include_tmp,
    )
    bytes_deleted += _remove_dirs(checkpoint_dirs, deleted_dirs, dry_run)
    if include_tmp:
        bytes_deleted += _remove_dirs(_pega_temp_dirs(catalog), deleted_dirs, dry_run)

    return VacuumResult(
        files_deleted=len(deleted_files),
        dirs_deleted=len(deleted_dirs),
        bytes_deleted=bytes_deleted,
        paths=(*deleted_files, *deleted_dirs),
    )


def _partition_aggregate_files(
    workspace: Path,
    current_hashes: dict[tuple[str, str], str],
    *,
    source_ids: Set[str] | None = None,
    retained_run_ids: Mapping[str, str] | None = None,
    running_run_ids: Mapping[str, frozenset[str]] | None = None,
) -> tuple[dict[tuple[str, str, str, str, str, str], list[_Candidate]], list[Path]]:
    candidates: dict[tuple[str, str, str, str, str, str], list[_Candidate]] = {}
    stale: list[Path] = []
    successful_by_source: dict[str, set[tuple[str, str]]] = {}
    for path in _aggregate_files(workspace, source_ids):
        identity = _aggregate_identity(workspace, path)
        if identity is None:
            continue
        source_id, processor_id, grain, period = identity
        current_hash = current_hashes.get((source_id, processor_id))
        metadata = _file_metadata(path)
        if metadata.run_ids.intersection((running_run_ids or {}).get(source_id, frozenset())):
            continue
        if (
            current_hash is None
            or not metadata.config_hashes
            or current_hash not in metadata.config_hashes
        ):
            stale.append(path)
            continue
        if len(metadata.chunk_ids) != 1:
            stale.append(path)
            continue
        chunk_id = next(iter(metadata.chunk_ids))
        if len(metadata.run_ids) != 1:
            stale.append(path)
            continue
        run_id = next(iter(metadata.run_ids))
        if retained_run_ids is not None and run_id != retained_run_ids[source_id]:
            stale.append(path)
            continue
        if source_id not in successful_by_source:
            successful_by_source[source_id] = _successful_chunk_keys(workspace, source_id)
        successful = successful_by_source[source_id]
        candidates.setdefault(
            (source_id, processor_id, grain, period, chunk_id, current_hash), []
        ).append(
            _Candidate(
                path=path,
                is_successful=(run_id, chunk_id) in successful,
                created_order_ns=metadata.created_order_ns,
            )
        )
    return candidates, stale


def _superseded_candidates(
    candidates: dict[tuple[str, str, str, str, str, str], list[_Candidate]],
) -> list[Path]:
    superseded: list[Path] = []
    for group in candidates.values():
        successful = [item for item in group if item.is_successful]
        if not successful:
            superseded.extend(item.path for item in group)
            continue
        keep = max(successful, key=lambda item: item.created_order_ns)
        superseded.extend(item.path for item in group if item.path != keep.path)
    return superseded


def _aggregate_files(workspace: Path, source_ids: Set[str] | None = None) -> list[Path]:
    root = workspace / "aggregates"
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.glob("*/*/*/period=*/*.parquet")
        if path.is_file() and _path_is_in_source_scope(root, path, source_ids)
    )


def _stale_processor_state_dirs(  # noqa: PLR0912
    workspace: Path,
    catalog: model.Catalog,
    *,
    source_ids: Set[str] | None,
    running_run_ids: Mapping[str, frozenset[str]],
    include_tmp: bool,
) -> list[Path]:
    """Return stale rolling frequency-checkpoint directories.

    Checkpoint publication is outside the business ledger. Vacuum therefore
    scopes it independently, never removes state for a running source, keeps
    the flat directory for each configured persistent processor, and drops
    obsolete or legacy directory layouts. Bounded history inside
    ``rolling.duckdb`` is owned by :class:`RollingCheckpoint`; vacuum must not
    inspect or prune that database (or its WAL).
    """

    root = workspace / ".valuestream" / "state" / "frequency_response"
    if not root.exists():
        return []
    current_processors = {
        (processor.source, processor.id)
        for processor in catalog.processors.processors
        if isinstance(processor, model.FrequencyResponseProcessor)
        and processor.checkpoint.mode == "persistent_sharded"
        and (source_ids is None or processor.source in source_ids)
    }
    stale: set[Path] = set()

    # The supported layout is deliberately flat:
    # source=<source>/processor=<processor>/rolling.duckdb. Compatibility
    # revisions and hashes live in DuckDB metadata and are validated when the
    # store is opened, not represented by directories vacuum must understand.
    source_dirs = sorted(path for path in root.glob("source=*") if path.is_dir())
    for source_dir in source_dirs:
        source_id = unquote(source_dir.name.removeprefix("source="))
        if source_ids is not None and source_id not in source_ids:
            continue
        if running_run_ids.get(source_id):
            continue

        configured_for_source = {
            processor_id
            for processor_source, processor_id in current_processors
            if processor_source == source_id
        }
        if not configured_for_source:
            stale.add(source_dir)
            continue

        for child in sorted(source_dir.iterdir()):
            if not child.is_dir():
                continue
            if not child.name.startswith("processor="):
                if include_tmp or not _is_processor_state_temp_dir(child):
                    stale.add(child)
                continue
            processor_id = unquote(child.name.removeprefix("processor="))
            if processor_id not in configured_for_source:
                stale.add(child)
                continue

            # A current flat processor directory contains files only. Any
            # child directory is either a previous nested layout or temporary
            # work. Preserve temporary directories only when requested.
            for nested in sorted(path for path in child.iterdir() if path.is_dir()):
                if include_tmp or not _is_processor_state_temp_dir(nested):
                    stale.add(nested)

    # Everything below a non-source top-level directory belongs to a legacy
    # schema/hash/runtime hierarchy. Delete the whole hierarchy when it is safe
    # to do so; source-scoped or active runs retain their exact source subtree.
    for legacy_root in sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith("source=")
    ):
        if _is_processor_state_temp_dir(legacy_root) and not include_tmp:
            continue
        legacy_sources = _legacy_processor_state_source_dirs(legacy_root)
        if not legacy_sources:
            if source_ids is None:
                stale.add(legacy_root)
            continue
        if all(
            (source_ids is None or source_id in source_ids) and not running_run_ids.get(source_id)
            for source_id, _ in legacy_sources
        ):
            stale.add(legacy_root)
            continue
        for source_id, legacy_source_dir in legacy_sources:
            if source_ids is not None and source_id not in source_ids:
                continue
            if not running_run_ids.get(source_id):
                stale.add(legacy_source_dir)

    return _outermost_directories(stale)


def _legacy_processor_state_source_dirs(root: Path) -> list[tuple[str, Path]]:
    """Return outermost source directories embedded in one legacy hierarchy."""

    source_dirs = _outermost_directories({path for path in root.rglob("source=*") if path.is_dir()})
    return sorted((unquote(path.name.removeprefix("source=")), path) for path in source_dirs)


def _is_processor_state_temp_dir(path: Path) -> bool:
    name = path.name
    return name.endswith(".tmp") or name.startswith(".tmp-")


def _outermost_directories(paths: Set[Path]) -> list[Path]:
    """Return a stable list with descendants of selected directories removed."""

    kept: list[Path] = []
    for path in sorted(paths, key=lambda item: (len(item.parts), str(item))):
        if any(parent == path or parent in path.parents for parent in kept):
            continue
        kept.append(path)
    return sorted(kept)


def _aggregate_temp_files(
    workspace: Path,
    source_ids: Set[str] | None = None,
    running_run_ids: Mapping[str, frozenset[str]] | None = None,
) -> list[Path]:
    root = workspace / "aggregates"
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.glob("**/*.tmp")
        if path.is_file()
        and _path_is_in_source_scope(root, path, source_ids)
        and not _temp_belongs_to_running_run(root, path, running_run_ids or {})
    )


def _temp_belongs_to_running_run(
    root: Path,
    path: Path,
    running_run_ids: Mapping[str, frozenset[str]],
) -> bool:
    try:
        source_id = path.relative_to(root).parts[0]
    except (IndexError, ValueError):
        return False
    return any(run_id in path.name for run_id in running_run_ids.get(source_id, frozenset()))


def _path_is_in_source_scope(root: Path, path: Path, source_ids: Set[str] | None) -> bool:
    if source_ids is None:
        return True
    try:
        source_id = path.relative_to(root).parts[0]
    except (IndexError, ValueError):
        return False
    return source_id in source_ids


def _aggregate_identity(workspace: Path, path: Path) -> tuple[str, str, str, str] | None:
    try:
        rel = path.relative_to(workspace / "aggregates")
    except ValueError:
        return None
    if len(rel.parts) < 5 or not rel.parts[3].startswith("period="):
        return None
    return rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3].removeprefix("period=")


def _current_processor_hashes(
    catalog: model.Catalog,
    source_ids: Set[str] | None = None,
) -> dict[tuple[str, str], str]:
    return {
        (processor.source, processor.id): processor_computation_hash(catalog, processor)
        for processor in catalog.processors.processors
        if source_ids is None or processor.source in source_ids
    }


def _file_metadata(path: Path) -> _FileMetadata:
    """Read all aggregate provenance fields with one projected Parquet scan."""
    try:
        fallback_order_ns = path.stat().st_mtime_ns
    except FileNotFoundError:
        fallback_order_ns = 0
    try:
        frame = (
            pl.scan_parquet(path)
            .select(
                pl.col("config_hash").drop_nulls().unique().implode().alias("config_hashes"),
                pl.col("chunk_id").drop_nulls().unique().implode().alias("chunk_ids"),
                pl.col("pipeline_run_id").drop_nulls().unique().implode().alias("run_ids"),
                cs.by_name("created_at", require_all=False).max(),
            )
            .collect()
        )
    except Exception:
        return _FileMetadata(frozenset(), frozenset(), frozenset(), fallback_order_ns)
    value = frame["created_at"].max() if "created_at" in frame.columns else None
    if isinstance(value, dt.datetime):
        normalized = value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)
        created_order_ns = int(normalized.timestamp() * 1_000_000_000)
    else:
        created_order_ns = fallback_order_ns
    return _FileMetadata(
        config_hashes=frozenset(str(value) for value in frame["config_hashes"].item()),
        chunk_ids=frozenset(str(value) for value in frame["chunk_ids"].item()),
        run_ids=frozenset(str(value) for value in frame["run_ids"].item()),
        created_order_ns=created_order_ns,
    )


def _successful_chunk_keys(workspace: Path, source_id: str) -> set[tuple[str, str]]:
    chunks_db = meta_dir(workspace) / "chunks.duckdb"
    runs_db = meta_dir(workspace) / "pipeline_runs.duckdb"
    if not chunks_db.exists() or not runs_db.exists():
        return set()
    escaped_runs = str(runs_db).replace("'", "''")
    with duckdb.connect(str(chunks_db), read_only=True) as conn:
        conn.execute(f"ATTACH '{escaped_runs}' AS runs_meta (READ_ONLY)")
        rows = conn.execute(
            """
            SELECT CAST(c.pipeline_run_id AS VARCHAR), c.chunk_id
            FROM chunks c
            JOIN runs_meta.pipeline_runs r ON c.pipeline_run_id = r.id
            WHERE c.source_id = ?
              AND c.status = 'ok'
              AND r.status IN ('ok', 'partial', 'failed')
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY c.source_id, c.chunk_id
                ORDER BY c.finished_at DESC NULLS LAST,
                         r.finished_at DESC NULLS LAST,
                         c.started_at DESC NULLS LAST,
                         r.started_at DESC NULLS LAST,
                         CAST(c.pipeline_run_id AS VARCHAR) DESC
            ) = 1
            """,
            (source_id,),
        ).fetchall()
    return {(str(run_id), str(chunk_id)) for run_id, chunk_id in rows}


def _running_run_ids(
    workspace: Path,
    source_ids: Set[str] | None,
) -> dict[str, frozenset[str]]:
    runs_db = meta_dir(workspace) / "pipeline_runs.duckdb"
    if not runs_db.exists():
        return {}
    with duckdb.connect(str(runs_db), read_only=True) as conn:
        if source_ids is None:
            rows = conn.execute(
                "SELECT source_id, CAST(id AS VARCHAR) FROM pipeline_runs WHERE status = 'running'"
            ).fetchall()
        elif not source_ids:
            rows = []
        else:
            placeholders = ", ".join("?" for _ in source_ids)
            rows = conn.execute(
                f"SELECT source_id, CAST(id AS VARCHAR) FROM pipeline_runs "
                f"WHERE status = 'running' AND source_id IN ({placeholders})",
                tuple(sorted(source_ids)),
            ).fetchall()
    grouped: dict[str, set[str]] = {}
    for source_id, run_id in rows:
        grouped.setdefault(str(source_id), set()).add(str(run_id))
    return {source_id: frozenset(run_ids) for source_id, run_ids in grouped.items()}


def _pega_temp_dirs(catalog: model.Catalog) -> list[Path]:
    roots = {Path(tempfile.gettempdir())}
    for source in catalog.pipelines.sources:
        archive_temp_dir = source.reader.archive_temp_dir
        if isinstance(archive_temp_dir, str | Path):
            roots.add(Path(archive_temp_dir))
    out: list[Path] = []
    for root in roots:
        if root.exists():
            out.extend(path for path in root.glob("dataset_export_*") if path.is_dir())
    return sorted(set(out))


def _remove_files(paths: list[Path], deleted: list[Path], dry_run: bool) -> int:
    bytes_deleted = 0
    for path in paths:
        bytes_deleted += _file_size(path)
        deleted.append(path)
        if not dry_run:
            path.unlink(missing_ok=True)
    return bytes_deleted


def _remove_dirs(paths: list[Path], deleted: list[Path], dry_run: bool) -> int:
    bytes_deleted = 0
    for path in paths:
        size = _dir_size(path)
        if dry_run:
            bytes_deleted += size
            deleted.append(path)
            continue
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            continue
        bytes_deleted += size
        deleted.append(path)
    return bytes_deleted


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _dir_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += _file_size(child)
    return total


__all__ = ["VacuumResult", "vacuum_processor_state", "vacuum_workspace"]
