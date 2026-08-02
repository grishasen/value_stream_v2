"""Phase 1 source runner."""

from __future__ import annotations

import datetime as dt
import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Literal, TypeAlias, cast

import polars as pl
import pyarrow as pa

from valuestream.config import model
from valuestream.config.canonical import (
    catalog_config_hash,
    processor_computation_config,
    processor_computation_hash,
    serialize,
    source_computation_config,
    source_computation_hash,
)
from valuestream.config.loader import load
from valuestream.config.validate import validate_catalog
from valuestream.engine import ledger
from valuestream.engine.frequency import DuckDBFrequencySession
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
    RollingCheckpoint,
    rolling_checkpoint_path,
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
    if parallel > 1 and any(_is_persistent_frequency_processor(item) for item in processors):
        logger.warning(
            "Persistent frequency state requires chronological single-process chunks; "
            "using parallel=1 instead of parallel=%s: source=%s",
            parallel,
            source_id,
        )
        parallel = 1

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
            with _PersistentFrequencyCoordinator(
                workspace,
                source,
                processors,
                force=force,
            ) as frequency_coordinator:
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
                    outcome = _process_chunk(
                        workspace,
                        source,
                        processors,
                        plan,
                        run_id,
                        frequency_coordinator=frequency_coordinator,
                    )
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
    *,
    frequency_coordinator: _PersistentFrequencyCoordinator | None = None,
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
            # The materialized transform frame doubles as the rolling staging
            # input, so persistent frequency state is fed without a second
            # source scan; `_collect_chunk_materialized` runs the rolling
            # collection itself once the frame exists.
            (
                rows_in,
                rows_kept,
                processor_frames,
                written,
                persistent_processor_frames,
            ) = _collect_chunk_materialized(
                workspace,
                source,
                directly_collected_processors,
                plan,
                run_id,
                ctx,
                source_engine,
                raw,
                transformed,
                frequency_transformed,
                frequency_processors=frequency_processors,
                frequency_coordinator=frequency_coordinator,
            )
        else:
            persistent_processor_frames = _collect_rolling_frequency_frames(
                frequency_processors,
                frequency_coordinator,
                transformed,
                plan,
                ctx,
                engine=source_engine,
            )
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
        processor_frames.extend(persistent_processor_frames)
        frames_by_processor = {
            processor.id: (processor, frame) for processor, frame in processor_frames
        }
        processor_frames = [
            frames_by_processor[processor.id]
            for processor in processors
            if processor.id in frames_by_processor
        ]

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


def _collect_rolling_frequency_frames(
    processors: Sequence[_Processor],
    coordinator: _PersistentFrequencyCoordinator | None,
    current: pl.LazyFrame,
    plan: _ChunkPlan,
    ctx: ChunkContext,
    *,
    engine: _CollectEngine,
) -> list[tuple[_Processor, pl.DataFrame]]:
    if not any(_is_persistent_frequency_processor(processor) for processor in processors):
        return []
    if coordinator is None:  # pragma: no cover - scheduling invariant
        raise RuntimeError("persistent frequency processing requires a rolling-state coordinator")
    return coordinator.collect(current, plan, ctx, engine=engine)


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


class _PersistentFrequencyCoordinator:
    """Own rolling frequency stores for one chronological source run."""

    def __init__(
        self,
        workspace: Path,
        source: model.Source,
        processors: Sequence[_Processor],
        *,
        force: bool,
    ) -> None:
        self.workspace = workspace
        self.source = source
        self.processors = tuple(
            _as_persistent_frequency_processor(processor)
            for processor in processors
            if _is_persistent_frequency_processor(processor)
        )
        self.force = force
        self._stack = ExitStack()
        self._checkpoints: dict[str, RollingCheckpoint] = {}
        self._open_errors: dict[str, Exception] = {}
        self._entered = False

    def __enter__(self) -> _PersistentFrequencyCoordinator:
        if self._entered:
            raise RuntimeError("persistent frequency coordinator is already open")
        self._stack.__enter__()
        self._entered = True
        for processor in self.processors:
            path = rolling_checkpoint_path(
                self.workspace,
                source_id=self.source.id,
                processor_id=processor.id,
            )
            wal_path = path.with_name(f"{path.name}.wal")
            if not self.force and not path.exists() and not wal_path.exists():
                continue
            try:
                checkpoint = self._open_checkpoint(processor, customer_dtype="Null")
                checkpoint.prune_retention()
            except Exception as exc:
                if self.force:
                    self._entered = False
                    self._stack.__exit__(type(exc), exc, exc.__traceback__)
                    raise
                self._open_errors[processor.id] = exc
                logger.exception(
                    "Could not maintain rolling frequency state; a pending target will fail "
                    "closed: source=%s processor=%s path=%s",
                    self.source.id,
                    processor.id,
                    path,
                )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self._entered = False
        self._checkpoints.clear()
        self._open_errors.clear()
        return self._stack.__exit__(exc_type, exc_value, traceback)

    def collect(
        self,
        current: pl.LazyFrame,
        plan: _ChunkPlan,
        ctx: ChunkContext,
        *,
        engine: _CollectEngine,
    ) -> list[tuple[_Processor, pl.DataFrame]]:
        """Reconcile, calculate, and advance each processor's bounded state."""

        if not self._entered:  # pragma: no cover - lifecycle invariant
            raise RuntimeError("persistent frequency coordinator is not open")
        collected: list[tuple[_Processor, pl.DataFrame]] = []
        for processor in self.processors:
            open_error = self._open_errors.get(processor.id)
            if open_error is not None:
                raise open_error
            prepared = processor.checkpoint_contacts_lazy(current)
            checkpoint = self._checkpoint_for(processor, prepared.collect_schema())
            self._reconcile_history(processor, checkpoint, plan, engine=engine)
            shards = self._stage(
                checkpoint,
                prepared,
                plan.chunk,
                engine=engine,
            )
            try:
                with DuckDBFrequencySession(
                    config=processor.config,
                    connection=checkpoint.connection,
                    current_chunk_id=plan.chunk_id,
                ) as session:
                    shard_partials = _collect_shard_partials(session, processor, shards, ctx)
                daily = (
                    pl.concat(shard_partials, how="diagonal_relaxed")
                    if shard_partials
                    else pl.DataFrame()
                )
                checkpoint.commit_staged()
            except Exception:
                checkpoint.abort_staged()
                raise
            collected.append((processor, daily))
        return collected

    def _checkpoint_for(
        self,
        processor: FrequencyResponseRuntime,
        prepared_schema: pl.Schema,
    ) -> RollingCheckpoint:
        existing = self._checkpoints.get(processor.id)
        if existing is not None:
            return existing
        customer_column = processor.config.columns.customer
        if customer_column not in prepared_schema:
            raise ValueError(
                f"frequency_response processor {processor.id!r} checkpoint is missing "
                f"customer column {customer_column!r}"
            )
        return self._open_checkpoint(
            processor,
            customer_dtype=str(prepared_schema[customer_column]),
        )

    def _open_checkpoint(
        self,
        processor: FrequencyResponseRuntime,
        *,
        customer_dtype: str,
    ) -> RollingCheckpoint:
        existing = self._checkpoints.get(processor.id)
        if existing is not None:
            return existing
        checkpoint = self._stack.enter_context(
            RollingCheckpoint(
                self.workspace,
                source_id=self.source.id,
                processor_id=processor.id,
                config_hash=processor.config_hash,
                customer_column=processor.config.columns.customer,
                customer_dtype=customer_dtype,
                shard_count=processor.config.checkpoint.shards,
                retention_days=processor.config.checkpoint_retention_days,
                history_projection=processor.checkpoint_history_projection(),
                force=self.force,
                duckdb_threads=processor.config.checkpoint.threads,
                duckdb_memory_limit=processor.config.checkpoint.memory_limit,
            )
        )
        self._checkpoints[processor.id] = checkpoint
        return checkpoint

    def _reconcile_history(
        self,
        processor: FrequencyResponseRuntime,
        checkpoint: RollingCheckpoint,
        plan: _ChunkPlan,
        *,
        engine: _CollectEngine,
    ) -> None:
        history_chunks = _processor_history_chunks(processor, plan)
        expected = tuple(
            (chunk.chunk_id, ledger.file_fingerprint(chunk.files)) for chunk in history_chunks
        )
        missing = checkpoint.reconcile_history(expected)
        if not missing:
            return
        history_by_id = {chunk.chunk_id: chunk for chunk in history_chunks}
        for chunk_id in missing:
            chunk = history_by_id.get(chunk_id)
            if chunk is None:  # pragma: no cover - store contract invariant
                raise RuntimeError(
                    f"rolling frequency state requested unknown history chunk {chunk_id!r}"
                )
            logger.info(
                "Filling rolling frequency history: source=%s processor=%s chunk=%s",
                self.source.id,
                processor.id,
                chunk.chunk_id,
            )
            try:
                transformed = apply_transforms(read(self.source.reader, chunk.files), self.source)
                _validate_frequency_current_input_columns(
                    [processor],
                    transformed.collect_schema(),
                )
                self._stage(
                    checkpoint,
                    processor.checkpoint_contacts_lazy(transformed),
                    chunk,
                    engine=engine,
                )
                checkpoint.commit_staged()
            except Exception:
                checkpoint.abort_staged()
                raise

    @staticmethod
    def _stage(
        checkpoint: RollingCheckpoint,
        prepared: pl.LazyFrame,
        chunk: Chunk,
        *,
        engine: _CollectEngine,
    ) -> tuple[int, ...]:
        checkpoint_engine: Literal["auto", "streaming"] = (
            "streaming" if engine == "streaming" else "auto"
        )
        return checkpoint.stage_current(
            prepared,
            chunk_id=chunk.chunk_id,
            raw_fingerprint=ledger.file_fingerprint(chunk.files),
            engine=checkpoint_engine,
        )


def _collect_shard_partials(
    session: DuckDBFrequencySession,
    processor: FrequencyResponseRuntime,
    shards: Sequence[int],
    ctx: ChunkContext,
) -> list[pl.DataFrame]:
    """Aggregate focal shards without moving the DuckDB connection across threads.

    A single shard keeps DuckDB's batched Polars stream. With multiple shards,
    the owning thread fetches each detached Arrow table while one worker runs
    only its Polars tail over the preceding table. This overlaps the two engines
    while bounding detached focal data to at most two whole shards.
    """

    partials: list[pl.DataFrame] = []
    if not shards:
        return partials

    if len(shards) == 1:
        partial = processor.aggregate_focal_lazy(session.focal_lazy(shards[0]), ctx).collect()
        if not partial.is_empty():
            partials.append(partial)
        return partials

    def aggregate_detached(focal_table: pa.Table) -> pl.DataFrame:
        focal = cast(pl.DataFrame, pl.from_arrow(focal_table)).lazy()
        return processor.aggregate_focal_lazy(focal, ctx).collect()

    with ThreadPoolExecutor(max_workers=1) as pool:
        first_table = session.focal_table(shards[0])
        pending: Future[pl.DataFrame] = pool.submit(aggregate_detached, first_table)
        del first_table
        for shard_id in shards[1:]:
            next_table = session.focal_table(shard_id)
            partial = pending.result()
            if not partial.is_empty():
                partials.append(partial)
            pending = pool.submit(aggregate_detached, next_table)
            del next_table
        partial = pending.result()
        if not partial.is_empty():
            partials.append(partial)
    return partials


def _processor_history_chunks(
    processor: FrequencyResponseRuntime,
    plan: _ChunkPlan,
) -> tuple[Chunk, ...]:
    """Return this processor's closure from the source-wide maximum plan."""

    target_date = dt.date.fromisoformat(plan.chunk_id)
    lookback_days = math.ceil(
        (processor.config.window_hours + processor.config.partition_lag_hours) / 24
    )
    earliest = target_date - dt.timedelta(days=lookback_days)
    return tuple(
        chunk for chunk in plan.history_chunks if dt.date.fromisoformat(chunk.chunk_id) >= earliest
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

_MaterializedChunkFrames = tuple[
    int,
    int,
    list[tuple["_Processor", pl.DataFrame]],
    list[AggregateWriteReceipt],
    list[tuple["_Processor", pl.DataFrame]],
]


def _collect_chunk_materialized(  # noqa: PLR0917
    workspace: Path,
    source: model.Source,
    processors: list[_Processor],
    plan: _ChunkPlan,
    run_id: str,
    ctx: ChunkContext,
    source_engine: _CollectEngine,
    raw: pl.LazyFrame,
    transformed: pl.LazyFrame,
    frequency_transformed: pl.LazyFrame | None,
    *,
    frequency_processors: list[_Processor],
    frequency_coordinator: _PersistentFrequencyCoordinator | None,
) -> _MaterializedChunkFrames:
    """Collect transforms once, then fan processors out over the materialized frame.

    Rolling frequency staging consumes the materialized frame directly, so
    persistent state is advanced without re-reading and re-transforming the
    source files for the same chunk.
    """
    chunk = plan.chunk
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
    persistent_processor_frames = _collect_rolling_frequency_frames(
        frequency_processors,
        frequency_coordinator,
        transformed_frame.lazy(),
        plan,
        ctx,
        engine="auto",
    )
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
    return rows_in, rows_kept, processor_frames, written, persistent_processor_frames


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
    target_chunks = (
        sorted(dated_chunks, key=lambda item: item[0])
        if any(_is_persistent_frequency_processor(item) for item in frequency_processors)
        else dated_chunks
    )
    for target_date, target in target_chunks:
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
