"""Phase 1 source runner."""

from __future__ import annotations

import datetime as dt
import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import polars as pl

from valuestream.config import model
from valuestream.config.canonical import (
    catalog_config_hash,
    frequency_checkpoint_layout_hash,
    processor_computation_config,
    processor_computation_hash,
    serialize,
    source_computation_config,
    source_computation_hash,
)
from valuestream.config.loader import load
from valuestream.config.validate import validate_catalog
from valuestream.engine import ledger
from valuestream.processors import grain_levels
from valuestream.processors.binary_outcome import ChunkContext
from valuestream.processors.context import SOURCE_ORDER_COLUMN
from valuestream.processors.frequency_response import (
    TARGET_CHUNK_COLUMN,
)
from valuestream.processors.frequency_response import (
    FrequencyResponseProcessor as FrequencyResponseRuntime,
)
from valuestream.processors.frequency_response import (
    required_history_input_columns as frequency_response_required_history_input_columns,
)
from valuestream.processors.frequency_response import (
    required_input_columns as frequency_response_required_input_columns,
)
from valuestream.processors.frequency_response import (
    validate_current_input_schema as validate_frequency_response_current_input_schema,
)
from valuestream.processors.registry import ProcessorRuntime, create_processor
from valuestream.readers import cleanup_temporaries, discover, read
from valuestream.readers.discovery import Chunk
from valuestream.store.duckdb_views import refresh_aggregate_views
from valuestream.store.parquet import AggregateWriteReceipt, write_aggregate_with_receipts
from valuestream.store.processor_state import (
    CheckpointManifest,
)
from valuestream.store.processor_state import (
    load_manifest as load_processor_state_manifest,
)
from valuestream.store.processor_state import (
    scan_shard as scan_processor_state_shard,
)
from valuestream.store.processor_state import (
    write_checkpoint as write_processor_state_checkpoint,
)
from valuestream.store.vacuum import vacuum_processor_state
from valuestream.transforms import apply_transforms
from valuestream.utils.ids import new_pipeline_run_id
from valuestream.utils.logger import get_logger
from valuestream.utils.time import utc_now
from valuestream.utils.timer import timed

_Processor: TypeAlias = ProcessorRuntime
_CollectEngine: TypeAlias = Literal["auto", "in-memory", "streaming"]
_ChunkProgressStatus: TypeAlias = Literal["processing", "recovering", "skipped"]
logger = get_logger(__name__)
_SUPPORTED_TARGET_GRAINS = grain_levels.SUPPORTED_TARGET_GRAINS


@dataclass(frozen=True)
class ChunkProgress:
    """Live progress details for one discovered source chunk."""

    source_id: str
    chunk_id: str
    chunk_name: str
    chunk_order: int
    chunks_total: int
    status: _ChunkProgressStatus
    files: tuple[Path, ...] = ()


ChunkProgressCallback: TypeAlias = Callable[[ChunkProgress], None]


@dataclass(frozen=True)
class ChunkRunResult:
    """Outcome for one discovered chunk."""

    chunk_id: str
    status: str
    rows_in: int = 0
    rows_kept: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None
    written: tuple[Path, ...] = ()


@dataclass(frozen=True)
class _ChunkOutcome:
    """Processing result plus the metadata the parent records to the ledger.

    Chunk processing is side-effect free with respect to the metadata ledger
    so it can run in worker processes; the parent process is the single
    DuckDB writer.
    """

    result: ChunkRunResult
    files: tuple[Path, ...]
    started_at: dt.datetime
    finished_at: dt.datetime
    lineage: tuple[AggregateWriteReceipt, ...] = ()


@dataclass(frozen=True)
class _ChunkPlan:
    """One target chunk plus its bounded raw-input dependency set."""

    chunk: Chunk
    history_chunks: tuple[Chunk, ...] = ()

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def history_files(self) -> tuple[Path, ...]:
        return tuple(
            sorted(
                {
                    file_path
                    for history_chunk in self.history_chunks
                    for file_path in history_chunk.files
                }
            )
        )

    @property
    def dependency_files(self) -> tuple[Path, ...]:
        return tuple(sorted({*self.chunk.files, *self.history_files}))


@dataclass(frozen=True)
class PipelineRunResult:
    """Source run summary."""

    run_id: str
    source_id: str
    status: str
    chunks_total: int
    chunks_ok: int
    chunks_failed: int
    chunks_skipped: int
    rows_in: int
    rows_kept: int
    chunks: tuple[ChunkRunResult, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class WorkspaceRunResult:
    """Workspace run summary for one or more sources."""

    status: str
    sources_total: int
    sources_ok: int
    sources_failed: int
    sources_partial: int
    results: tuple[PipelineRunResult, ...] = field(default_factory=tuple)


@timed
def run_workspace(
    workspace_path: str | Path,
    *,
    force: bool = False,
    parallel: int = 1,
    progress_callback: ChunkProgressCallback | None = None,
) -> WorkspaceRunResult:
    """Run every source in a workspace."""
    started = time.perf_counter()
    workspace = Path(workspace_path)
    logger.info(
        f"Starting workspace run: workspace={workspace}, force={force}, parallel={parallel}"
    )
    catalog = load(workspace)
    validation = validate_catalog(catalog, defer_unobserved_source_expressions=True)
    if not validation.ok:
        messages = "; ".join(f"{i.location}: {i.message}" for i in validation.issues)
        raise ValueError(f"catalog does not validate: {messages}")

    results = tuple(
        run_source(
            workspace,
            source.id,
            force=force,
            parallel=parallel,
            progress_callback=progress_callback,
        )
        for source in catalog.pipelines.sources
    )
    sources_failed = sum(1 for result in results if result.status == "failed")
    sources_partial = sum(1 for result in results if result.status == "partial")
    sources_ok = sum(1 for result in results if result.status == "ok")
    status = "failed" if sources_failed else "partial" if sources_partial else "ok"
    elapsed_ms = _elapsed_ms(started)
    logger.info(
        f"Workspace run finished: workspace={workspace}, status={status}, "
        f"sources_total={len(results)}, sources_ok={sources_ok}, "
        f"sources_partial={sources_partial}, sources_failed={sources_failed}, "
        f"time={elapsed_ms:.03f}ms"
    )
    return WorkspaceRunResult(
        status=status,
        sources_total=len(results),
        sources_ok=sources_ok,
        sources_failed=sources_failed,
        sources_partial=sources_partial,
        results=results,
    )


@timed
def run_source(
    workspace_path: str | Path,
    source_id: str,
    *,
    force: bool = False,
    parallel: int = 1,
    progress_callback: ChunkProgressCallback | None = None,
) -> PipelineRunResult:
    """Run one source while holding its workspace-scoped advisory lock."""

    workspace = Path(workspace_path)
    with ledger.source_run_lock(workspace, source_id):
        return _run_source_locked(
            workspace,
            source_id,
            force=force,
            parallel=parallel,
            progress_callback=progress_callback,
        )


def _run_source_locked(  # noqa: PLR0912, PLR0915
    workspace_path: str | Path,
    source_id: str,
    *,
    force: bool = False,
    parallel: int = 1,
    progress_callback: ChunkProgressCallback | None = None,
) -> PipelineRunResult:
    """Run one source through readers, transforms, binary processors, and parquet writes.

    ``parallel`` > 1 processes chunks in a process pool: sketch building is
    Python-level work that holds the GIL, so worker processes are what make
    the initial load scale with cores. Ledger writes stay in this (parent)
    process — DuckDB metadata files are single-writer.
    """
    started = time.perf_counter()
    workspace = Path(workspace_path)
    catalog = load(workspace)
    validation = validate_catalog(catalog, defer_unobserved_source_expressions=True)
    if not validation.ok:
        messages = "; ".join(f"{i.location}: {i.message}" for i in validation.issues)
        raise ValueError(f"catalog does not validate: {messages}")

    source = next(
        (candidate for candidate in catalog.pipelines.sources if candidate.id == source_id), None
    )
    if source is None:
        raise ValueError(f"unknown source {source_id!r}")

    processors = _processors_for_source(catalog, source_id)
    chunks = discover(workspace, source)
    chunk_plans = _plan_source_chunks(source, chunks, processors)

    run_id = new_pipeline_run_id()
    config_hash = source_computation_hash(catalog, source_id)
    started_at = utc_now()
    ledger.ensure(workspace)
    _record_config_versions(
        workspace,
        catalog,
        source_id=source_id,
        introduced_at=started_at,
    )
    logger.info(
        f"Starting source run: source={source_id}, run_id={run_id}, "
        f"workspace={workspace}, force={force}, parallel={parallel}"
    )

    chunks_total = len(chunk_plans)
    fingerprints = {
        plan.chunk_id: ledger.file_fingerprint(plan.dependency_files) for plan in chunk_plans
    }
    expected_outputs = {
        (processor.id, grain): processor.config_hash
        for processor in processors
        for grain in processor.config.grains
        if grain in _SUPPORTED_TARGET_GRAINS
    }
    recovered = ledger.recover_stale_runs(
        workspace,
        source_id=source_id,
        config_hash=config_hash,
        file_hashes=fingerprints,
        expected_outputs=expected_outputs,
        finished_at=utc_now(),
        progress_callback=_recovery_progress_adapter(progress_callback),
    )
    if recovered:
        logger.warning(
            "Recovered %s interrupted source run(s): source=%s runs=%s",
            len(recovered),
            source_id,
            ",".join(item.run_id for item in recovered),
        )
        refresh_aggregate_views(workspace, catalog)

    done_chunks: set[str] = set()
    if not force:
        done_chunks = ledger.done_chunk_ids(
            workspace,
            source_id=source_id,
            config_hash=config_hash,
            file_hashes=fingerprints,
        )
    ledger.start_run(
        workspace,
        run_id=run_id,
        workspace=catalog.pipelines.workspace,
        source_id=source_id,
        config_hash=config_hash,
        started_at=started_at,
        chunks_total=chunks_total,
    )
    run_finalized = False
    try:
        results_by_order: dict[int, ChunkRunResult] = {}
        to_process: list[tuple[int, _ChunkPlan]] = []
        for chunk_order, plan in enumerate(chunk_plans, start=1):
            chunk = plan.chunk
            if not force and plan.chunk_id in done_chunks:
                _notify_chunk_progress(
                    progress_callback,
                    source=source,
                    chunk=chunk,
                    chunk_order=chunk_order,
                    chunks_total=chunks_total,
                    status="skipped",
                )
                logger.debug(f"Skipping already processed chunk: {chunk.chunk_id}")
                results_by_order[chunk_order] = ChunkRunResult(
                    chunk_id=chunk.chunk_id, status="skipped"
                )
                continue
            to_process.append((chunk_order, plan))

        checkpoint_failures = _ensure_persistent_frequency_checkpoints(
            workspace,
            source,
            processors,
            [plan for _, plan in to_process],
            force=force,
        )
        ready_to_process: list[tuple[int, _ChunkPlan]] = []
        for chunk_order, plan in to_process:
            checkpoint_error = checkpoint_failures.get(plan.chunk_id)
            if checkpoint_error is None:
                ready_to_process.append((chunk_order, plan))
                continue
            _notify_chunk_progress(
                progress_callback,
                source=source,
                chunk=plan.chunk,
                chunk_order=chunk_order,
                chunks_total=chunks_total,
                status="processing",
            )
            outcome = _checkpoint_failure_outcome(plan, checkpoint_error)
            _record_chunk_outcome(workspace, source_id, run_id, outcome)
            results_by_order[chunk_order] = outcome.result
        to_process = ready_to_process

        if parallel > 1 and len(to_process) > 1:
            _run_chunks_parallel(
                workspace,
                source,
                processors,
                to_process,
                run_id,
                parallel=parallel,
                chunks_total=chunks_total,
                progress_callback=progress_callback,
                results_by_order=results_by_order,
            )
        else:
            for chunk_order, plan in to_process:
                chunk = plan.chunk
                _notify_chunk_progress(
                    progress_callback,
                    source=source,
                    chunk=chunk,
                    chunk_order=chunk_order,
                    chunks_total=chunks_total,
                    status="processing",
                )
                outcome = _process_chunk(workspace, source, processors, plan, run_id)
                _record_chunk_outcome(workspace, source_id, run_id, outcome)
                results_by_order[chunk_order] = outcome.result

        chunk_results = [results_by_order[order] for order in sorted(results_by_order)]
        finished_at = utc_now()
        chunks_ok = sum(1 for chunk in chunk_results if chunk.status == "ok")
        chunks_failed = sum(1 for chunk in chunk_results if chunk.status == "failed")
        chunks_skipped = sum(1 for chunk in chunk_results if chunk.status == "skipped")
        rows_in = sum(chunk.rows_in for chunk in chunk_results if chunk.status == "ok")
        rows_kept = sum(chunk.rows_kept for chunk in chunk_results if chunk.status == "ok")
        status = "failed" if chunks_failed else "ok"

        ledger.finalize_run(
            workspace,
            run_id=run_id,
            finished_at=finished_at,
            status=status,
            rows_in=rows_in,
            rows_kept=rows_kept,
            chunks_total=len(chunk_results),
            chunks_ok=chunks_ok,
            chunks_failed=chunks_failed,
        )
        run_finalized = True
        refresh_aggregate_views(workspace, catalog)
        try:
            checkpoint_vacuum = vacuum_processor_state(
                workspace,
                catalog,
                source_ids={source_id},
            )
            if checkpoint_vacuum.dirs_deleted:
                logger.info(
                    "Pruned %s stale processor checkpoint generation(s): source=%s bytes=%s",
                    checkpoint_vacuum.dirs_deleted,
                    source_id,
                    checkpoint_vacuum.bytes_deleted,
                )
        except Exception:
            # Processor state is a rebuildable acceleration cache. A cleanup
            # failure must be visible, but cannot roll back published aggregates.
            logger.exception(
                "Could not apply processor checkpoint retention: source=%s",
                source_id,
            )
        elapsed_ms = _elapsed_ms(started)
        logger.info(
            f"Source run finished: source={source_id}, run_id={run_id}, status={status}, "
            f"chunks_ok={chunks_ok}, chunks_skipped={chunks_skipped}, "
            f"chunks_failed={chunks_failed}, rows_in={rows_in}, rows_kept={rows_kept}, "
            f"time={elapsed_ms:.03f}ms"
        )

        return PipelineRunResult(
            run_id=run_id,
            source_id=source_id,
            status=status,
            chunks_total=len(chunk_results),
            chunks_ok=chunks_ok,
            chunks_failed=chunks_failed,
            chunks_skipped=chunks_skipped,
            rows_in=rows_in,
            rows_kept=rows_kept,
            chunks=tuple(chunk_results),
        )
    finally:
        if not run_finalized:
            try:
                interrupted = ledger.finalize_incomplete_run(
                    workspace,
                    run_id=run_id,
                    finished_at=utc_now(),
                )
                refresh_aggregate_views(workspace, catalog)
                logger.warning(
                    "Finalized interrupted source run: source=%s run_id=%s status=%s "
                    "chunks_ok=%s chunks_failed=%s",
                    source_id,
                    run_id,
                    interrupted.status,
                    interrupted.chunks_ok,
                    interrupted.chunks_failed,
                )
            except Exception:
                logger.exception(
                    "Could not finalize interrupted source run: source=%s run_id=%s",
                    source_id,
                    run_id,
                )


@timed
def _process_chunk(
    workspace: Path,
    source: model.Source,
    processors: list[_Processor],
    plan: _ChunkPlan,
    run_id: str,
) -> _ChunkOutcome:
    """Read, transform, aggregate, and write one chunk (no ledger writes)."""
    chunk = plan.chunk
    perf_started = time.perf_counter()
    started_at = utc_now()
    rows_in = 0
    rows_kept = 0
    debugging = _debugging_enabled(source)
    logger.debug(f"Processing chunk: {chunk.chunk_id}")
    try:
        raw, transformed, frequency_transformed = _prepare_chunk_frames(
            source,
            processors,
            plan,
        )
        if debugging:
            _log_chunk_schema(source, chunk, "raw", raw)
        ctx = ChunkContext(
            pipeline_run_id=run_id,
            chunk_id=chunk.chunk_id,
            created_at=dt.datetime.now(dt.UTC),
        )
        source_engine: _CollectEngine = "streaming" if source.reader.streaming else "auto"
        normal_processors = [
            processor for processor in processors if not _is_frequency_processor(processor)
        ]
        frequency_processors = [
            processor for processor in processors if _is_frequency_processor(processor)
        ]
        source_scan_frequency_processors = [
            processor
            for processor in frequency_processors
            if not _is_persistent_frequency_processor(processor)
        ]
        directly_collected_processors = [
            processor
            for processor in processors
            if not _is_persistent_frequency_processor(processor)
        ]
        current_schema = transformed.collect_schema()
        _validate_processor_input_columns(normal_processors, current_schema)
        if frequency_processors:
            _validate_frequency_current_input_columns(
                frequency_processors,
                current_schema,
            )
        if source_scan_frequency_processors:
            if frequency_transformed is None:  # pragma: no cover - construction invariant
                raise RuntimeError("frequency-response processor input was not prepared")
            _validate_processor_input_columns(
                source_scan_frequency_processors,
                frequency_transformed.collect_schema(),
            )
        if debugging:
            _log_chunk_schema(source, chunk, "transformed", transformed)
            if frequency_transformed is not None:
                _log_chunk_schema(source, chunk, "frequency_transformed", frequency_transformed)
        if source.materialize_transforms:
            rows_in, rows_kept, processor_frames, written = _collect_chunk_materialized(
                workspace,
                source,
                directly_collected_processors,
                chunk,
                run_id,
                ctx,
                source_engine,
                raw,
                transformed,
                frequency_transformed,
            )
        else:
            rows_in, rows_kept, processor_frames, written = _collect_chunk_lazy(
                workspace,
                source,
                directly_collected_processors,
                chunk,
                run_id,
                ctx,
                source_engine,
                raw,
                transformed,
                frequency_transformed,
            )

        processor_frames.extend(
            _collect_persistent_frequency_frames(
                workspace,
                source,
                frequency_processors,
                plan,
                ctx,
            )
        )

        # Transfer ownership before writing so the writer can drop each frame
        # without mutating the list returned by the collection stage.  Keeping
        # the original list alive would retain every processor DataFrame until
        # all outputs had been written.
        owned_processor_frames = deque(processor_frames)
        del processor_frames
        written.extend(
            _write_collected_processor_outputs(
                workspace,
                source,
                processors=owned_processor_frames,
                chunk=chunk,
                ctx=ctx,
                run_id=run_id,
                debugging=debugging,
            )
        )
        return _finish_successful_chunk(
            source,
            plan,
            run_id,
            started_at,
            perf_started,
            rows_in,
            rows_kept,
            written,
            debugging,
        )
    except Exception as exc:
        finished_at = utc_now()
        elapsed_ms = _elapsed_ms(perf_started)
        logger.exception(
            f"Chunk failed: source={source.id}, chunk={chunk.chunk_id}, "
            f"run_id={run_id}, time={elapsed_ms:.03f}ms"
        )
        return _ChunkOutcome(
            result=ChunkRunResult(
                chunk_id=chunk.chunk_id,
                status="failed",
                rows_in=rows_in,
                rows_kept=rows_kept,
                elapsed_ms=elapsed_ms,
                error=str(exc),
            ),
            files=plan.dependency_files,
            started_at=started_at,
            finished_at=finished_at,
        )
    finally:
        cleanup_temporaries()


def _prepare_chunk_frames(
    source: model.Source,
    processors: list[_Processor],
    plan: _ChunkPlan,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame | None]:
    """Build current-only and optional bounded-history transform graphs."""

    current_raw = read(source.reader, plan.chunk.files)
    normal_raw = current_raw
    if _requires_stable_source_order(processors):
        # Polars may schedule group inputs differently between the regular and
        # streaming engines. Preserve scan order explicitly for the score
        # processor's bounded, order-sensitive sampling helpers.
        normal_raw = normal_raw.with_row_index(SOURCE_ORDER_COLUMN)
    current_transformed = apply_transforms(normal_raw, source)

    source_scan_frequency_processors = [
        processor
        for processor in processors
        if _is_frequency_processor(processor) and not _is_persistent_frequency_processor(processor)
    ]
    if not source_scan_frequency_processors:
        return normal_raw, current_transformed, None

    marked_current = current_transformed.with_columns(pl.lit(True).alias(TARGET_CHUNK_COLUMN))
    if not plan.history_chunks:
        return normal_raw, current_transformed, marked_current

    # Source transforms are chunk-scoped semantics. In particular, configured
    # deduplication must not remove a target row merely because the same key
    # appeared in a history chunk. Transform each discovered chunk independently
    # before building the ephemeral dependency frame.
    history_columns = _frequency_history_input_columns(source_scan_frequency_processors)
    marked_history: list[pl.LazyFrame] = []
    for history_chunk in plan.history_chunks:
        history = apply_transforms(read(source.reader, history_chunk.files), source)
        history_schema = history.collect_schema()
        _validate_frequency_history_input_columns(
            source_scan_frequency_processors,
            history_schema,
            history_chunk.chunk_id,
        )
        # Only frequency keys, outcome/time, and raw processor-filter fields
        # can affect a later target. Dropping every other history field avoids
        # both unnecessary I/O through the lazy plan and irrelevant dtype drift
        # when history is combined with the full current-day schema.
        history = history.select(
            *(name for name in history_schema.names() if name in history_columns)
        )
        marked_history.append(history.with_columns(pl.lit(False).alias(TARGET_CHUNK_COLUMN)))
    frequency_transformed = pl.concat(
        [*marked_history, marked_current],
        how="diagonal_relaxed",
    )
    return normal_raw, current_transformed, frequency_transformed


def _ensure_persistent_frequency_checkpoints(  # noqa: PLR0912, PLR0915
    workspace: Path,
    source: model.Source,
    processors: list[_Processor],
    plans: list[_ChunkPlan],
    *,
    force: bool = False,
) -> dict[str, str]:
    """Prepare and validate checkpoint dependencies, isolating failures by target."""

    persistent = [
        processor for processor in processors if _is_persistent_frequency_processor(processor)
    ]
    if not persistent or not plans:
        return {}

    required_chunks: dict[str, Chunk] = {}
    for plan in plans:
        for chunk in (*plan.history_chunks, plan.chunk):
            required_chunks[chunk.chunk_id] = chunk

    engine: Literal["auto", "streaming"] = "streaming" if source.reader.streaming else "auto"
    manifests: dict[tuple[str, str], CheckpointManifest] = {}
    failures: dict[tuple[str, str], str] = {}
    for chunk_id in sorted(required_chunks):
        chunk = required_chunks[chunk_id]
        raw_fingerprint = ledger.file_fingerprint(chunk.files)
        transformed: pl.LazyFrame | None = None
        transformed_schema: pl.Schema | None = None
        try:
            for candidate in persistent:
                processor = _as_persistent_frequency_processor(candidate)
                key = (processor.id, chunk.chunk_id)
                layout_hash = frequency_checkpoint_layout_hash(processor.config)
                try:
                    manifest = None
                    if not force:
                        # Validate every byte once in the parent before any worker
                        # receives an unchecked manifest for shard-at-a-time reads.
                        manifest = load_processor_state_manifest(
                            workspace,
                            source_id=source.id,
                            processor_id=processor.id,
                            config_hash=processor.config_hash,
                            layout_hash=layout_hash,
                            chunk_id=chunk.chunk_id,
                            raw_fingerprint=raw_fingerprint,
                            validate=True,
                        )
                    if manifest is None:
                        logger.info(
                            "Building persistent frequency checkpoint: "
                            "source=%s processor=%s chunk=%s force=%s",
                            source.id,
                            processor.id,
                            chunk.chunk_id,
                            force,
                        )
                        if transformed is None:
                            transformed = apply_transforms(read(source.reader, chunk.files), source)
                            transformed_schema = transformed.collect_schema()
                        if transformed_schema is None:  # pragma: no cover - paired assignment
                            raise RuntimeError("transformed checkpoint schema was not prepared")
                        _validate_frequency_current_input_columns([processor], transformed_schema)
                        manifest = write_processor_state_checkpoint(
                            processor.checkpoint_contacts_lazy(transformed),
                            workspace,
                            source_id=source.id,
                            processor_id=processor.id,
                            config_hash=processor.config_hash,
                            layout_hash=layout_hash,
                            chunk_id=chunk.chunk_id,
                            raw_fingerprint=raw_fingerprint,
                            customer_column=processor.config.columns.customer,
                            shard_count=processor.config.checkpoint.shards,
                            engine=engine,
                            replace_existing=force,
                        )
                    _validate_checkpoint_manifest(processor, manifest)
                    manifests[key] = manifest
                except Exception as exc:
                    failures[key] = str(exc)
                    logger.exception(
                        "Persistent frequency checkpoint failed: source=%s processor=%s chunk=%s",
                        source.id,
                        processor.id,
                        chunk.chunk_id,
                    )
        finally:
            cleanup_temporaries()

    plan_failures: dict[str, str] = {}
    for plan in plans:
        errors: list[str] = []
        closure = (*plan.history_chunks, plan.chunk)
        for candidate in persistent:
            processor = _as_persistent_frequency_processor(candidate)
            closure_manifests: list[CheckpointManifest] = []
            for chunk in closure:
                key = (processor.id, chunk.chunk_id)
                if key in failures:
                    errors.append(
                        f"processor {processor.id!r} checkpoint {chunk.chunk_id!r}: {failures[key]}"
                    )
                    continue
                manifest = manifests.get(key)
                if manifest is None:  # pragma: no cover - preparation invariant
                    errors.append(
                        f"processor {processor.id!r} checkpoint {chunk.chunk_id!r} was not prepared"
                    )
                    continue
                closure_manifests.append(manifest)
            try:
                _validate_checkpoint_customer_dtypes(processor, closure_manifests)
            except ValueError as exc:
                errors.append(str(exc))
        if errors:
            plan_failures[plan.chunk_id] = "; ".join(dict.fromkeys(errors))
    return plan_failures


def _collect_persistent_frequency_frames(
    workspace: Path,
    source: model.Source,
    processors: list[_Processor],
    plan: _ChunkPlan,
    ctx: ChunkContext,
) -> list[tuple[_Processor, pl.DataFrame]]:
    """Aggregate persistent checkpoint closures one bounded customer shard at a time."""

    collected: list[tuple[_Processor, pl.DataFrame]] = []
    for candidate in processors:
        if not _is_persistent_frequency_processor(candidate):
            continue
        processor = _as_persistent_frequency_processor(candidate)
        current = _load_checkpoint_manifest(workspace, source, processor, plan.chunk)
        history = [
            _load_checkpoint_manifest(workspace, source, processor, chunk)
            for chunk in plan.history_chunks
        ]
        _validate_checkpoint_customer_dtypes(processor, [*history, current])
        shard_partials: list[pl.DataFrame] = []
        history_shards = [set(manifest.nonempty_shard_ids) for manifest in history]
        for shard_id in current.nonempty_shard_ids:
            current_scan = scan_processor_state_shard(current, shard_id)
            historical_scans = [
                scan_processor_state_shard(manifest, shard_id)
                for manifest, available in zip(history, history_shards, strict=True)
                if shard_id in available
            ]
            partial = processor.checkpoint_aggregate_lazy(
                current_scan,
                historical_scans,
                ctx,
            ).collect()
            if not partial.is_empty():
                shard_partials.append(partial)
        daily = (
            pl.concat(shard_partials, how="diagonal_relaxed") if shard_partials else pl.DataFrame()
        )
        collected.append((processor, daily))
    return collected


def _load_checkpoint_manifest(
    workspace: Path,
    source: model.Source,
    processor: FrequencyResponseRuntime,
    chunk: Chunk,
) -> CheckpointManifest:
    raw_fingerprint = ledger.file_fingerprint(chunk.files)
    layout_hash = frequency_checkpoint_layout_hash(processor.config)
    manifest = load_processor_state_manifest(
        workspace,
        source_id=source.id,
        processor_id=processor.id,
        config_hash=processor.config_hash,
        layout_hash=layout_hash,
        chunk_id=chunk.chunk_id,
        raw_fingerprint=raw_fingerprint,
        validate=False,
    )
    if manifest is None:
        raise RuntimeError(
            f"persistent frequency checkpoint is missing for processor {processor.id!r}, "
            f"chunk {chunk.chunk_id!r}"
        )
    _validate_checkpoint_manifest(processor, manifest)
    return manifest


def _validate_checkpoint_manifest(
    processor: FrequencyResponseRuntime,
    manifest: CheckpointManifest,
) -> None:
    expected_customer = processor.config.columns.customer
    expected_shards = processor.config.checkpoint.shards
    expected_layout = frequency_checkpoint_layout_hash(processor.config)
    if manifest.layout_hash != expected_layout:
        raise ValueError(
            f"frequency checkpoint layout hash {manifest.layout_hash!r} does not match "
            f"{expected_layout!r}"
        )
    if manifest.customer_column != expected_customer:
        raise ValueError(
            f"frequency checkpoint customer column {manifest.customer_column!r} does not "
            f"match {expected_customer!r}"
        )
    if manifest.shard_count != expected_shards:
        raise ValueError(
            f"frequency checkpoint shard count {manifest.shard_count} does not match "
            f"{expected_shards}"
        )


def _validate_checkpoint_customer_dtypes(
    processor: FrequencyResponseRuntime,
    manifests: Sequence[CheckpointManifest],
) -> None:
    # Empty partitions route no identity and therefore cannot disagree with a
    # non-empty partition's hash domain. In particular, an all-null source day
    # can legitimately retain a zero-row checkpoint whose inferred dtype is
    # ``Null`` after the processor drops unusable identities.
    dtypes = sorted({manifest.customer_dtype for manifest in manifests if manifest.rows > 0})
    if len(dtypes) > 1:
        raise ValueError(
            f"frequency checkpoint customer dtype drift for processor {processor.id!r}: "
            f"{', '.join(dtypes)}; normalize the authoritative source identity type "
            "before persistent sharding"
        )


def _checkpoint_failure_outcome(plan: _ChunkPlan, error: str) -> _ChunkOutcome:
    timestamp = utc_now()
    return _ChunkOutcome(
        result=ChunkRunResult(
            chunk_id=plan.chunk_id,
            status="failed",
            error=f"persistent frequency checkpoint preparation failed: {error}",
        ),
        files=plan.dependency_files,
        started_at=timestamp,
        finished_at=timestamp,
    )


def _processor_input_frames(
    processors: list[_Processor],
    current: pl.LazyFrame,
    frequency: pl.LazyFrame | None,
) -> list[tuple[_Processor, pl.LazyFrame]]:
    """Bind each processor to its current-only or bounded-history frame."""

    inputs: list[tuple[_Processor, pl.LazyFrame]] = []
    for processor in processors:
        if _is_frequency_processor(processor):
            if frequency is None:  # pragma: no cover - construction invariant
                raise RuntimeError("frequency-response processor input was not prepared")
            inputs.append((processor, frequency))
        else:
            inputs.append((processor, current))
    return inputs


_ChunkFrames = tuple[
    int,
    int,
    list[tuple["_Processor", pl.DataFrame]],
    list[AggregateWriteReceipt],
]


def _collect_chunk_materialized(  # noqa: PLR0917
    workspace: Path,
    source: model.Source,
    processors: list[_Processor],
    chunk: Chunk,
    run_id: str,
    ctx: ChunkContext,
    source_engine: _CollectEngine,
    raw: pl.LazyFrame,
    transformed: pl.LazyFrame,
    frequency_transformed: pl.LazyFrame | None,
) -> _ChunkFrames:
    """Collect transforms once, then fan processors out over the materialized frame."""
    transform_plans = [
        raw.select(pl.len().alias("rows_in")),
        transformed,
    ]
    if frequency_transformed is not None:
        transform_plans.append(frequency_transformed)
    counts_and_transformed = pl.collect_all(transform_plans, engine=source_engine)
    rows_in = int(counts_and_transformed[0]["rows_in"][0])
    transformed_frame = counts_and_transformed[1]
    frequency_frame = counts_and_transformed[2] if frequency_transformed is not None else None
    del counts_and_transformed
    rows_kept = transformed_frame.height
    processor_inputs = _processor_input_frames(
        processors,
        transformed_frame.lazy(),
        frequency_frame.lazy() if frequency_frame is not None else None,
    )
    processor_frames: list[tuple[_Processor, pl.DataFrame]] = []
    written: list[AggregateWriteReceipt] = []
    try:
        # Python sketch/map-groups nodes are not streaming-native.  The source
        # scan and transforms above use the configured engine; processor plans
        # run on the one in-memory transformed frame with the regular engine.
        processor_frames = _collect_processor_frames(processor_inputs, ctx, "in-memory")
    except Exception:
        logger.warning(
            f"Batched processor collect failed for chunk {chunk.chunk_id}; "
            "falling back to sequential per-processor execution",
            exc_info=True,
        )
        written = _run_processors_sequential(
            workspace,
            source,
            processor_inputs,
            ctx,
            run_id,
            chunk.chunk_id,
        )
    finally:
        del processor_inputs
        if frequency_frame is not None:
            del frequency_frame
        del transformed_frame
    return rows_in, rows_kept, processor_frames, written


def _collect_chunk_lazy(  # noqa: PLR0917
    workspace: Path,
    source: model.Source,
    processors: list[_Processor],
    chunk: Chunk,
    run_id: str,
    ctx: ChunkContext,
    engine: _CollectEngine,
    raw: pl.LazyFrame,
    transformed: pl.LazyFrame,
    frequency_transformed: pl.LazyFrame | None,
) -> _ChunkFrames:
    """Collect counts and all processor plans in one batched pass."""
    processor_frames: list[tuple[_Processor, pl.DataFrame]] = []
    written: list[AggregateWriteReceipt] = []
    try:
        processor_inputs = _processor_input_frames(
            processors,
            transformed,
            frequency_transformed,
        )
        processor_plans = [
            (processor, processor.chunk_aggregate_lazy(processor_input, ctx))
            for processor, processor_input in processor_inputs
        ]
        lazy_frames = [
            raw.select(pl.len().alias("rows_in")),
            transformed.select(pl.len().alias("rows_kept")),
            *(plan for _, plan in processor_plans),
        ]
        # Keep the source-selected engine for this unmaterialized graph. Polars
        # places in-memory barriers around Python UDF nodes while retaining a
        # streaming source scan; forcing the whole graph to ``in-memory`` here
        # would silently disable the source's streaming setting.
        collected = pl.collect_all(lazy_frames, engine=engine)
        rows_in = int(collected[0]["rows_in"][0])
        rows_kept = int(collected[1]["rows_kept"][0])
        processor_frames = list(
            zip(
                (processor for processor, _ in processor_plans),
                collected[2:],
                strict=True,
            )
        )
    except Exception:
        logger.warning(
            f"Batched lazy collect failed for chunk {chunk.chunk_id}; "
            "falling back to sequential per-processor execution",
            exc_info=True,
        )
        rows_in = int(raw.select(pl.len().alias("rows_in")).collect()["rows_in"][0])
        transform_plans = [transformed]
        if frequency_transformed is not None:
            transform_plans.append(frequency_transformed)
        transformed_frames = pl.collect_all(transform_plans, engine=engine)
        transformed_frame = transformed_frames[0]
        frequency_frame = transformed_frames[1] if frequency_transformed is not None else None
        rows_kept = transformed_frame.height
        processor_inputs = _processor_input_frames(
            processors,
            transformed_frame.lazy(),
            frequency_frame.lazy() if frequency_frame is not None else None,
        )
        written = _run_processors_sequential(
            workspace,
            source,
            processor_inputs,
            ctx,
            run_id,
            chunk.chunk_id,
        )
    return rows_in, rows_kept, processor_frames, written


def _finish_successful_chunk(  # noqa: PLR0917
    source: model.Source,
    plan: _ChunkPlan,
    run_id: str,
    started_at: dt.datetime,
    perf_started: float,
    rows_in: int,
    rows_kept: int,
    written: list[AggregateWriteReceipt],
    debugging: bool,
) -> _ChunkOutcome:
    chunk = plan.chunk
    finished_at = utc_now()
    elapsed_ms = _elapsed_ms(perf_started)
    logger.debug(f"Chunk processing time: {elapsed_ms:.03f}ms")
    if debugging:
        _log_chunk_rows(source, chunk, rows_in=rows_in, rows_kept=rows_kept)
    logger.info(
        f"Chunk processed: source={source.id}, chunk={chunk.chunk_id}, "
        f"run_id={run_id}, rows_in={rows_in}, rows_kept={rows_kept}, "
        f"written={len(written)}, time={elapsed_ms:.03f}ms"
    )
    return _ChunkOutcome(
        result=ChunkRunResult(
            chunk_id=chunk.chunk_id,
            status="ok",
            rows_in=rows_in,
            rows_kept=rows_kept,
            elapsed_ms=elapsed_ms,
            written=tuple(receipt.path for receipt in written),
        ),
        files=plan.dependency_files,
        started_at=started_at,
        finished_at=finished_at,
        lineage=tuple(written),
    )


def _record_chunk_outcome(
    workspace: Path,
    source_id: str,
    run_id: str,
    outcome: _ChunkOutcome,
) -> None:
    """Commit lineage, then the chunk row, in the parent process."""
    if outcome.result.status == "ok":
        lineage_count = ledger.insert_lineage_records(workspace, records=outcome.lineage)
        if lineage_count != len(outcome.lineage):
            raise RuntimeError(
                f"chunk {outcome.result.chunk_id!r} produced incomplete aggregate lineage"
            )
    ledger.insert_chunk(
        workspace,
        source_id=source_id,
        chunk_id=outcome.result.chunk_id,
        files=outcome.files,
        rows_in=outcome.result.rows_in,
        rows_kept=outcome.result.rows_kept,
        started_at=outcome.started_at,
        finished_at=outcome.finished_at,
        status=outcome.result.status,
        error=outcome.result.error,
        pipeline_run_id=run_id,
    )


def _record_config_versions(
    workspace: Path,
    catalog: model.Catalog,
    *,
    source_id: str,
    introduced_at: dt.datetime,
) -> None:
    versions: list[tuple[str, object]] = [
        (catalog_config_hash(catalog), catalog),
        (
            source_computation_hash(catalog, source_id),
            source_computation_config(catalog, source_id),
        ),
    ]
    versions.extend(
        (
            processor_computation_hash(catalog, processor),
            processor_computation_config(catalog, processor),
        )
        for processor in catalog.processors.processors
        if processor.source == source_id
    )
    for config_hash, payload in versions:
        ledger.insert_config_version(
            workspace,
            config_hash=config_hash,
            yaml=serialize(payload).decode("utf-8"),
            introduced_at=introduced_at,
        )


def _run_chunks_parallel(
    workspace: Path,
    source: model.Source,
    processors: list[_Processor],
    to_process: list[tuple[int, _ChunkPlan]],
    run_id: str,
    *,
    parallel: int,
    chunks_total: int,
    progress_callback: ChunkProgressCallback | None,
    results_by_order: dict[int, ChunkRunResult],
) -> None:
    """Process chunks in a process pool, recording ledger rows as they finish.

    Worker processes sidestep the GIL held by Python sketch building, so the
    initial load scales with cores. Parquet part files are per-chunk, so
    worker writes never collide; the ledger stays parent-only.
    """
    max_workers = min(parallel, len(to_process))
    logger.info(
        f"Processing {len(to_process)} chunk(s) with {max_workers} worker process(es): "
        f"source={source.id}, run_id={run_id}"
    )
    try:
        pool = ProcessPoolExecutor(max_workers=max_workers)
    except (NotImplementedError, PermissionError):
        logger.warning(
            "Process-pool execution is unavailable; falling back to sequential chunks: "
            f"source={source.id}, run_id={run_id}",
            exc_info=True,
        )
        for chunk_order, plan in to_process:
            chunk = plan.chunk
            _notify_chunk_progress(
                progress_callback,
                source=source,
                chunk=chunk,
                chunk_order=chunk_order,
                chunks_total=chunks_total,
                status="processing",
            )
            outcome = _process_chunk(workspace, source, processors, plan, run_id)
            _record_chunk_outcome(workspace, source.id, run_id, outcome)
            results_by_order[chunk_order] = outcome.result
        return
    with pool:
        futures: dict[Future[_ChunkOutcome], tuple[int, _ChunkPlan]] = {}
        for chunk_order, plan in to_process:
            chunk = plan.chunk
            _notify_chunk_progress(
                progress_callback,
                source=source,
                chunk=chunk,
                chunk_order=chunk_order,
                chunks_total=chunks_total,
                status="processing",
            )
            future = pool.submit(_process_chunk, workspace, source, processors, plan, run_id)
            futures[future] = (chunk_order, plan)
        pending = set(futures)
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                chunk_order, _ = futures[future]
                outcome = future.result()
                _record_chunk_outcome(workspace, source.id, run_id, outcome)
                results_by_order[chunk_order] = outcome.result


@timed
def _collect_processor_frames(
    processor_inputs: list[tuple[_Processor, pl.LazyFrame]],
    ctx: ChunkContext,
    engine: _CollectEngine,
) -> list[tuple[_Processor, pl.DataFrame]]:
    plans = [
        (processor, processor.chunk_aggregate_lazy(frame, ctx))
        for processor, frame in processor_inputs
    ]
    if not plans:
        return []
    frames = pl.collect_all([plan for _, plan in plans], engine=engine)
    return list(zip((processor for processor, _ in plans), frames, strict=True))


def _write_collected_processor_outputs(
    workspace: Path,
    source: model.Source,
    *,
    processors: deque[tuple[_Processor, pl.DataFrame]],
    chunk: Chunk,
    ctx: ChunkContext,
    run_id: str,
    debugging: bool,
) -> list[AggregateWriteReceipt]:
    """Write and consume an owned queue of processor frames in source order."""

    written: list[AggregateWriteReceipt] = []
    while processors:
        processor, daily = processors.popleft()
        if debugging:
            _log_processor_frame(source, chunk, processor, "base", daily)
        written.extend(
            _write_processor_outputs(
                workspace,
                source,
                processor,
                daily,
                ctx,
                run_id,
                chunk.chunk_id,
            )
        )
        del daily
    return written


@timed
def _run_processors_sequential(  # noqa: PLR0917
    workspace: Path,
    source: model.Source,
    processor_inputs: list[tuple[_Processor, pl.LazyFrame]],
    ctx: ChunkContext,
    run_id: str,
    chunk_id: str,
) -> list[AggregateWriteReceipt]:
    written: list[AggregateWriteReceipt] = []
    for processor, frame in processor_inputs:
        daily = processor.chunk_aggregate(frame, ctx)
        if _debugging_enabled(source):
            _log_processor_frame(source, chunk_id, processor, "base", daily)
        written.extend(
            _write_processor_outputs(
                workspace,
                source,
                processor,
                daily,
                ctx,
                run_id,
                chunk_id,
            )
        )
        del daily
    return written


@timed
def _write_processor_outputs(  # noqa: PLR0917
    workspace: Path,
    source: model.Source,
    processor: _Processor,
    daily: pl.DataFrame,
    ctx: ChunkContext,
    run_id: str,
    chunk_id: str,
) -> list[AggregateWriteReceipt]:
    written: list[AggregateWriteReceipt] = []
    for grain in processor.config.grains:
        if grain not in _SUPPORTED_TARGET_GRAINS:
            continue
        aggregate = processor.compact(daily, grain, ctx)
        if _debugging_enabled(source):
            _log_processor_frame(source, chunk_id, processor, grain, aggregate)
        written.extend(
            write_aggregate_with_receipts(
                aggregate,
                workspace,
                source_id=source.id,
                processor_id=processor.id,
                grain=grain,
                run_id=run_id,
                chunk_id=chunk_id,
            )
        )
        del aggregate
    return written


def _processors_for_source(catalog: model.Catalog, source_id: str) -> list[_Processor]:
    processors: list[_Processor] = []
    for processor in catalog.processors.processors:
        if processor.source != source_id:
            continue
        computation_hash = processor_computation_hash(catalog, processor)
        processors.append(create_processor(processor, computation_hash=computation_hash))
    return processors


def _plan_source_chunks(
    source: model.Source,
    chunks: list[Chunk],
    processors: list[_Processor],
) -> list[_ChunkPlan]:
    """Attach bounded calendar-day history to frequency-response targets."""

    frequency_processors = [
        processor for processor in processors if _is_frequency_processor(processor)
    ]
    if not frequency_processors:
        return [_ChunkPlan(chunk) for chunk in chunks]

    dated_chunks: list[tuple[dt.date, Chunk]] = []
    for chunk in chunks:
        try:
            chunk_date = dt.date.fromisoformat(chunk.chunk_id)
        except ValueError as exc:
            raise ValueError(
                f"source {source.id!r} uses frequency_response processors, so chunk IDs "
                f"must use ISO YYYY-MM-DD calendar dates; got {chunk.chunk_id!r}"
            ) from exc
        if chunk_date.isoformat() != chunk.chunk_id:
            raise ValueError(
                f"source {source.id!r} uses frequency_response processors, so chunk IDs "
                f"must use ISO YYYY-MM-DD calendar dates; got {chunk.chunk_id!r}"
            )
        dated_chunks.append((chunk_date, chunk))

    lookback_days = _frequency_lookback_days(frequency_processors)
    plans: list[_ChunkPlan] = []
    for target_date, target in dated_chunks:
        earliest = target_date - dt.timedelta(days=lookback_days)
        history = tuple(
            chunk
            for chunk_date, chunk in sorted(dated_chunks, key=lambda item: item[0])
            if earliest <= chunk_date < target_date
        )
        plans.append(_ChunkPlan(target, history))
    return plans


def _frequency_lookback_days(processors: list[_Processor]) -> int:
    dependency_hours = max(
        processor.config.window_hours + processor.config.partition_lag_hours
        for processor in processors
        if isinstance(processor.config, model.FrequencyResponseProcessor)
    )
    return math.ceil(dependency_hours / 24)


def _is_frequency_processor(processor: _Processor) -> bool:
    return isinstance(processor.config, model.FrequencyResponseProcessor)


def _is_persistent_frequency_processor(processor: _Processor) -> bool:
    return (
        isinstance(processor.config, model.FrequencyResponseProcessor)
        and processor.config.checkpoint.mode == "persistent_sharded"
    )


def _as_persistent_frequency_processor(
    processor: _Processor,
) -> FrequencyResponseRuntime:
    if not isinstance(processor, FrequencyResponseRuntime):
        raise TypeError(f"processor {processor.id!r} is not a frequency-response runtime")
    if processor.config.checkpoint.mode != "persistent_sharded":
        raise TypeError(f"processor {processor.id!r} does not use persistent checkpoints")
    return processor


def _requires_stable_source_order(processors: list[_Processor]) -> bool:
    return any(
        isinstance(processor.config, model.ScoreDistributionProcessor)
        and {"personalization", "novelty"}.intersection(processor.state_specs)
        for processor in processors
    )


def _validate_processor_input_columns(  # noqa: PLR0912
    processors: list[_Processor],
    schema: pl.Schema,
) -> None:
    """Fail before aggregation when configured inputs are absent.

    Processor implementations deliberately tolerate some optional inputs, but
    authored dimensions, properties, keys, and state sources are contractual.
    Silently omitting those fields changes aggregate semantics and is unsafe.
    """

    existing = set(schema.names())
    failures: list[str] = []
    for processor in processors:
        config = processor.config
        if isinstance(config, model.FrequencyResponseProcessor):
            required = set(frequency_response_required_input_columns(config))
        else:
            required = set(config.group_by)
            if config.time.column:
                required.add(config.time.column)
            required.update(_configured_state_source_columns(config))
            required.update(str(value) for value in getattr(config, "dedup_keys", []))

            if isinstance(config, model.BinaryOutcomeProcessor | model.ScoreDistributionProcessor):
                required.add(config.outcome.column)
            if isinstance(config, model.NumericDistributionProcessor):
                required.update(config.properties)
            elif isinstance(config, model.ScoreDistributionProcessor):
                for spec in model.effective_processor_states(config).values():
                    if spec.type == "tdigest":
                        required.add(_score_state_source_column(spec))
                if "personalization" in model.effective_processor_states(config):
                    required.update({"CustomerID", "Name"})
                if "novelty" in model.effective_processor_states(config):
                    required.update({"CustomerID", "InteractionID", "Name"})
            elif isinstance(config, model.EntityLifecycleProcessor):
                required.update(config.keys.model_dump().values())
            elif isinstance(config, model.EntitySetProcessor):
                required.add(config.entity)
            elif isinstance(config, model.FunnelProcessor):
                if config.entity:
                    required.add(config.entity)
            elif isinstance(config, model.SnapshotProcessor):
                if config.snapshot_kind == "accumulating":
                    required.add(config.entity)
                if config.as_of_property:
                    required.add(config.as_of_property)
                required.update(milestone.property for milestone in config.milestones)

        missing = sorted(column for column in required if column and column not in existing)
        if missing:
            failures.append(f"{config.id}: {', '.join(missing)}")
    if failures:
        raise ValueError("processor input columns are missing: " + "; ".join(failures))


def _validate_frequency_current_input_columns(
    processors: Sequence[_Processor],
    schema: pl.Schema,
) -> None:
    """Reject target-day schema gaps that bounded history could otherwise mask."""

    failures: list[str] = []
    for processor in processors:
        config = processor.config
        if not isinstance(config, model.FrequencyResponseProcessor):
            continue
        try:
            validate_frequency_response_current_input_schema(config, schema)
        except (TypeError, ValueError) as exc:
            failures.append(str(exc))
    if failures:
        raise ValueError(
            "frequency-response target chunk input schema is invalid: " + "; ".join(failures)
        )


def _frequency_history_input_columns(processors: list[_Processor]) -> frozenset[str]:
    """Return the union of narrow history fields needed by frequency processors."""

    columns: set[str] = set()
    for processor in processors:
        config = processor.config
        if isinstance(config, model.FrequencyResponseProcessor):
            columns.update(frequency_response_required_history_input_columns(config))
    return frozenset(columns)


def _validate_frequency_history_input_columns(
    processors: list[_Processor],
    schema: pl.Schema,
    chunk_id: str,
) -> None:
    """Reject dependency-day schema gaps before a relaxed history union masks them."""

    existing = set(schema.names())
    failures: list[str] = []
    for processor in processors:
        config = processor.config
        if not isinstance(config, model.FrequencyResponseProcessor):
            continue
        required = frequency_response_required_history_input_columns(config)
        missing = sorted(required - existing)
        if missing:
            failures.append(f"{config.id}: missing {', '.join(missing)}")
            continue
        decision_dtype = schema[config.time.property]
        if decision_dtype.base_type() != pl.Datetime:
            failures.append(
                f"{config.id}: {config.time.property} must be datetime, got {decision_dtype}"
            )
        rank_dtype = schema[config.columns.rank]
        if not rank_dtype.is_integer():
            failures.append(f"{config.id}: {config.columns.rank} must be integer, got {rank_dtype}")
    if failures:
        raise ValueError(
            f"frequency-response history chunk {chunk_id!r} has invalid input schema: "
            + "; ".join(failures)
        )


def _configured_state_source_columns(config: model.Processor) -> set[str]:
    columns: set[str] = set()
    for spec in model.effective_processor_states(config).values():
        source_column = getattr(spec, "source_column", None)
        if source_column:
            columns.add(source_column)
    return columns


def _score_state_source_column(spec: model.StateSpec) -> str:
    source_column = getattr(spec, "source_column", None)
    if not source_column:
        raise ValueError("score digest state requires source_column")
    return str(source_column)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _notify_chunk_progress(
    callback: ChunkProgressCallback | None,
    *,
    source: model.Source,
    chunk: Chunk,
    chunk_order: int,
    chunks_total: int,
    status: _ChunkProgressStatus,
) -> None:
    if callback is None:
        return
    callback(
        ChunkProgress(
            source_id=source.id,
            chunk_id=chunk.chunk_id,
            chunk_name=chunk.chunk_id,
            chunk_order=chunk_order,
            chunks_total=chunks_total,
            status=status,
            files=chunk.files,
        )
    )


def _recovery_progress_adapter(
    callback: ChunkProgressCallback | None,
) -> ledger.RecoveryProgressCallback | None:
    if callback is None:
        return None

    def update(progress: ledger.RecoveryProgress) -> None:
        callback(
            ChunkProgress(
                source_id=progress.source_id,
                chunk_id=progress.run_id,
                chunk_name=(
                    f"recovery {progress.run_id[:8]} · {progress.processor_id}/{progress.grain}"
                ),
                chunk_order=progress.group_order,
                chunks_total=progress.groups_total,
                status="recovering",
                files=progress.files,
            )
        )

    return update


def _debugging_enabled(source: model.Source) -> bool:
    return source.debugging


def _log_chunk_schema(
    source: model.Source,
    chunk: Chunk,
    stage: str,
    frame: pl.LazyFrame,
) -> None:
    schema = frame.collect_schema()
    formatted = ", ".join(f"{name}:{dtype}" for name, dtype in schema.items())
    logger.debug(
        f"Chunk schema: source={source.id}, chunk={chunk.chunk_id}, "
        f"stage={stage}, schema=[{formatted}]"
    )


def _log_chunk_rows(
    source: model.Source,
    chunk: Chunk,
    *,
    rows_in: int,
    rows_kept: int,
) -> None:
    logger.debug(
        f"Chunk rows: source={source.id}, chunk={chunk.chunk_id}, "
        f"rows_in={rows_in}, rows_kept={rows_kept}"
    )


def _log_processor_frame(
    source: model.Source,
    chunk: Chunk | str,
    processor: _Processor,
    stage: str,
    frame: pl.DataFrame,
) -> None:
    chunk_id = chunk.chunk_id if isinstance(chunk, Chunk) else chunk
    period_nulls = frame["period"].null_count() if "period" in frame.columns else "n/a"
    formatted = ", ".join(f"{name}:{dtype}" for name, dtype in frame.schema.items())
    logger.debug(
        f"Processor frame: source={source.id}, chunk={chunk_id}, "
        f"processor={processor.id}, stage={stage}, rows={frame.height}, "
        f"period_nulls={period_nulls}, schema=[{formatted}]"
    )


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "on"}
    return bool(value)


__all__ = [
    "ChunkProgress",
    "ChunkProgressCallback",
    "ChunkRunResult",
    "PipelineRunResult",
    "WorkspaceRunResult",
    "run_source",
    "run_workspace",
]
