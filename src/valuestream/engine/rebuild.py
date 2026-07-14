"""Exclusive clean-rebuild orchestration for aggregate storage."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from valuestream.config import model
from valuestream.config.canonical import catalog_config_hash
from valuestream.config.loader import load
from valuestream.config.validate import validate_catalog
from valuestream.engine import ledger
from valuestream.engine.runner import (
    ChunkProgressCallback,
    PipelineRunResult,
    _run_source_locked,
)
from valuestream.store.duckdb_views import refresh_aggregate_views
from valuestream.store.vacuum import VacuumResult, vacuum_workspace
from valuestream.utils.logger import get_logger
from valuestream.utils.timer import timed

logger = get_logger(__name__)


class CleanRebuildError(RuntimeError):
    """A clean rebuild did not complete its guarded rebuild-and-cleanup workflow."""

    def __init__(
        self,
        message: str,
        *,
        runs: tuple[PipelineRunResult, ...] = (),
    ) -> None:
        super().__init__(message)
        self.runs = runs


@dataclass(frozen=True)
class CleanRebuildResult:
    """Successful clean rebuild and the physical cleanup it performed."""

    source_ids: tuple[str, ...]
    runs: tuple[PipelineRunResult, ...]
    vacuum: VacuumResult

    @property
    def chunks_rebuilt(self) -> int:
        """Return the number of source chunks rebuilt successfully."""

        return sum(run.chunks_ok for run in self.runs)


@timed
def clean_rebuild(
    workspace_path: str | Path,
    *,
    source_ids: Iterable[str] | None = None,
    parallel: int = 1,
    progress_callback: ChunkProgressCallback | None = None,
) -> CleanRebuildResult:
    """Rebuild selected sources, then remove every superseded aggregate file.

    The operation holds every selected source lock from the first rebuild
    through cleanup. Old files are removed only when every discovered chunk
    completed successfully under an unchanged catalog. Metadata ledgers and
    configuration history are retained for audit.
    """

    workspace = Path(workspace_path)
    initial_catalog = load(workspace)
    _require_valid_catalog(initial_catalog)
    selected = _select_source_ids(initial_catalog, source_ids)
    initial_hash = catalog_config_hash(initial_catalog)
    runs: list[PipelineRunResult] = []
    cleanup_started = False

    logger.info(
        "Starting clean rebuild: workspace=%s sources=%s",
        workspace,
        ",".join(selected),
    )
    try:
        with ExitStack() as locks:
            for source_id in sorted(selected):
                locks.enter_context(ledger.source_run_lock(workspace, source_id))

            locked_catalog = load(workspace)
            _require_valid_catalog(locked_catalog)
            if catalog_config_hash(locked_catalog) != initial_hash:
                raise CleanRebuildError(
                    "Catalog changed while the rebuild was acquiring source locks; "
                    "no old aggregate files were removed. Retry with the current catalog."
                )

            for source_id in selected:
                result = _run_source_locked(
                    workspace,
                    source_id,
                    force=True,
                    parallel=parallel,
                    progress_callback=progress_callback,
                )
                runs.append(result)
                _require_complete_run(workspace, result, runs=tuple(runs))

            current_catalog = load(workspace)
            _require_valid_catalog(current_catalog)
            if catalog_config_hash(current_catalog) != initial_hash:
                raise CleanRebuildError(
                    "Catalog changed during the rebuild; new aggregates were kept, but no "
                    "old aggregate files were removed. Retry with the current catalog.",
                    runs=tuple(runs),
                )

            retained_run_ids = {run.source_id: run.run_id for run in runs}
            cleanup_started = True
            vacuum_result = vacuum_workspace(
                workspace,
                current_catalog,
                include_tmp=False,
                source_ids=frozenset(selected),
                retained_run_ids=retained_run_ids,
            )
            refresh_aggregate_views(workspace, current_catalog)
    except CleanRebuildError:
        logger.warning(
            "Clean rebuild stopped before cleanup: workspace=%s sources=%s",
            workspace,
            ",".join(selected),
        )
        raise
    except Exception as exc:
        logger.exception(
            "Clean rebuild failed before completion: workspace=%s sources=%s",
            workspace,
            ",".join(selected),
        )
        if cleanup_started:
            message = (
                "Clean rebuild failed after aggregate cleanup started. The completed new "
                f"runs remain available; inspect the store before retrying: {exc}"
            )
        else:
            message = f"Clean rebuild failed; no cleanup step was started: {exc}"
        raise CleanRebuildError(message, runs=tuple(runs)) from exc

    result = CleanRebuildResult(
        source_ids=selected,
        runs=tuple(runs),
        vacuum=vacuum_result,
    )
    logger.info(
        "Clean rebuild finished: workspace=%s sources=%s chunks=%s files_deleted=%s",
        workspace,
        ",".join(selected),
        result.chunks_rebuilt,
        result.vacuum.files_deleted,
    )
    return result


def _select_source_ids(
    catalog: model.Catalog,
    requested: Iterable[str] | None,
) -> tuple[str, ...]:
    available = tuple(source.id for source in catalog.pipelines.sources)
    if requested is None:
        selected = available
    else:
        requested_set = frozenset(requested)
        unknown = sorted(requested_set.difference(available))
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"unknown clean-rebuild source(s): {names}")
        selected = tuple(source_id for source_id in available if source_id in requested_set)
    if not selected:
        raise ValueError("clean rebuild requires at least one source")
    return selected


def _require_valid_catalog(catalog: model.Catalog) -> None:
    validation = validate_catalog(catalog)
    if validation.ok:
        return
    messages = "; ".join(f"{issue.location}: {issue.message}" for issue in validation.issues)
    raise ValueError(f"catalog does not validate: {messages}")


def _require_complete_run(
    workspace: Path,
    result: PipelineRunResult,
    *,
    runs: tuple[PipelineRunResult, ...],
) -> None:
    if result.chunks_total == 0:
        raise CleanRebuildError(
            f"Source {result.source_id!r} discovered no chunks; old aggregates were "
            "preserved to avoid deleting data when an input location may be unavailable.",
            runs=runs,
        )
    complete = (
        result.status == "ok"
        and result.chunks_failed == 0
        and result.chunks_skipped == 0
        and result.chunks_ok == result.chunks_total
        and all(chunk.status == "ok" for chunk in result.chunks)
    )
    if not complete:
        raise CleanRebuildError(
            f"Source {result.source_id!r} did not rebuild every discovered chunk "
            f"({result.chunks_ok} ok, {result.chunks_failed} failed, "
            f"{result.chunks_skipped} skipped); old aggregates were preserved.",
            runs=runs,
        )

    aggregate_root = (workspace / "aggregates" / result.source_id).resolve()
    missing_or_external = [
        path
        for chunk in result.chunks
        for path in chunk.written
        if not path.exists() or not path.resolve().is_relative_to(aggregate_root)
    ]
    if missing_or_external:
        raise CleanRebuildError(
            f"Source {result.source_id!r} reported aggregate files that could not be "
            "verified; old aggregates were preserved.",
            runs=runs,
        )


__all__ = ["CleanRebuildError", "CleanRebuildResult", "clean_rebuild"]
