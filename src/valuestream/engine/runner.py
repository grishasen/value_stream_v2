"""Phase 1 source runner."""

from __future__ import annotations

import datetime as dt
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import polars as pl

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
from valuestream.processors import grain_levels
from valuestream.processors.binary_outcome import ChunkContext
from valuestream.processors.context import SOURCE_ORDER_COLUMN
from valuestream.processors.registry import ProcessorRuntime, create_processor
from valuestream.readers import cleanup_temporaries, discover, read
from valuestream.readers.discovery import Chunk
from valuestream.store.duckdb_views import refresh_aggregate_views
from valuestream.store.parquet import AggregateWriteReceipt, write_aggregate_with_receipts
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
    validation = validate_catalog(catalog)
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
    status = (
        "failed"
        if sources_failed and not sources_ok and not sources_partial
        else "partial"
        if sources_failed or sources_partial
        else "ok"
    )
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


def _run_source_locked(  # noqa: PLR0915
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
    validation = validate_catalog(catalog)
    if not validation.ok:
        messages = "; ".join(f"{i.location}: {i.message}" for i in validation.issues)
        raise ValueError(f"catalog does not validate: {messages}")

    source = next(
        (candidate for candidate in catalog.pipelines.sources if candidate.id == source_id), None
    )
    if source is None:
        raise ValueError(f"unknown source {source_id!r}")

    processors = _processors_for_source(catalog, source_id)

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

    chunks = discover(workspace, source)
    chunks_total = len(chunks)
    fingerprints = {chunk.chunk_id: ledger.file_fingerprint(chunk.files) for chunk in chunks}
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
        to_process: list[tuple[int, Chunk]] = []
        for chunk_order, chunk in enumerate(chunks, start=1):
            if not force and chunk.chunk_id in done_chunks:
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
            to_process.append((chunk_order, chunk))

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
            for chunk_order, chunk in to_process:
                _notify_chunk_progress(
                    progress_callback,
                    source=source,
                    chunk=chunk,
                    chunk_order=chunk_order,
                    chunks_total=chunks_total,
                    status="processing",
                )
                outcome = _process_chunk(workspace, source, processors, chunk, run_id)
                _record_chunk_outcome(workspace, source_id, run_id, outcome)
                results_by_order[chunk_order] = outcome.result

        chunk_results = [results_by_order[order] for order in sorted(results_by_order)]
        finished_at = utc_now()
        chunks_ok = sum(1 for chunk in chunk_results if chunk.status == "ok")
        chunks_failed = sum(1 for chunk in chunk_results if chunk.status == "failed")
        chunks_skipped = sum(1 for chunk in chunk_results if chunk.status == "skipped")
        rows_in = sum(chunk.rows_in for chunk in chunk_results if chunk.status == "ok")
        rows_kept = sum(chunk.rows_kept for chunk in chunk_results if chunk.status == "ok")
        status = (
            "failed" if chunks_failed and not chunks_ok else "partial" if chunks_failed else "ok"
        )

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
    chunk: Chunk,
    run_id: str,
) -> _ChunkOutcome:
    """Read, transform, aggregate, and write one chunk (no ledger writes)."""
    perf_started = time.perf_counter()
    started_at = utc_now()
    rows_in = 0
    rows_kept = 0
    debugging = _debugging_enabled(source)
    logger.debug(f"Processing chunk: {chunk.chunk_id}")
    try:
        raw = read(source.reader, chunk.files)
        if _requires_stable_source_order(processors):
            # Polars may schedule group inputs differently between the regular
            # and streaming engines. Preserve scan order explicitly for the
            # score processor's bounded, order-sensitive sampling helpers.
            raw = raw.with_row_index(SOURCE_ORDER_COLUMN)
        if debugging:
            _log_chunk_schema(source, chunk, "raw", raw)
        ctx = ChunkContext(
            pipeline_run_id=run_id,
            chunk_id=chunk.chunk_id,
            created_at=dt.datetime.now(dt.UTC),
        )
        source_engine: _CollectEngine = "streaming" if source.reader.streaming else "auto"
        transformed = apply_transforms(raw, source)
        _validate_processor_input_columns(processors, transformed.collect_schema())
        if debugging:
            _log_chunk_schema(source, chunk, "transformed", transformed)
        if source.materialize_transforms:
            rows_in, rows_kept, processor_frames, written = _collect_chunk_materialized(
                workspace,
                source,
                processors,
                chunk,
                run_id,
                ctx,
                source_engine,
                raw,
                transformed,
            )
        else:
            rows_in, rows_kept, processor_frames, written = _collect_chunk_lazy(
                workspace,
                source,
                processors,
                chunk,
                run_id,
                ctx,
                source_engine,
                raw,
                transformed,
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
            chunk,
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
            files=chunk.files,
            started_at=started_at,
            finished_at=finished_at,
        )
    finally:
        cleanup_temporaries()


_ChunkFrames = tuple[
    int,
    int,
    list[tuple["_Processor", pl.DataFrame]],
    list[AggregateWriteReceipt],
]


def _collect_chunk_materialized(
    workspace: Path,
    source: model.Source,
    processors: list[_Processor],
    chunk: Chunk,
    run_id: str,
    ctx: ChunkContext,
    source_engine: _CollectEngine,
    raw: pl.LazyFrame,
    transformed: pl.LazyFrame,
) -> _ChunkFrames:
    """Collect transforms once, then fan processors out over the materialized frame."""
    counts_and_transformed = pl.collect_all(
        [
            raw.select(pl.len().alias("rows_in")),
            transformed,
        ],
        engine=source_engine,
    )
    rows_in = int(counts_and_transformed[0]["rows_in"][0])
    transformed_frame = counts_and_transformed[1]
    del counts_and_transformed
    rows_kept = transformed_frame.height
    processor_input = transformed_frame.lazy()
    processor_frames: list[tuple[_Processor, pl.DataFrame]] = []
    written: list[AggregateWriteReceipt] = []
    try:
        # Python sketch/map-groups nodes are not streaming-native.  The source
        # scan and transforms above use the configured engine; processor plans
        # run on the one in-memory transformed frame with the regular engine.
        processor_frames = _collect_processor_frames(processors, processor_input, ctx, "in-memory")
    except Exception:
        logger.warning(
            f"Batched processor collect failed for chunk {chunk.chunk_id}; "
            "falling back to sequential per-processor execution",
            exc_info=True,
        )
        written = _run_processors_sequential(
            workspace, source, processors, processor_input, ctx, run_id, chunk.chunk_id
        )
    finally:
        del processor_input
        del transformed_frame
    return rows_in, rows_kept, processor_frames, written


def _collect_chunk_lazy(
    workspace: Path,
    source: model.Source,
    processors: list[_Processor],
    chunk: Chunk,
    run_id: str,
    ctx: ChunkContext,
    engine: _CollectEngine,
    raw: pl.LazyFrame,
    transformed: pl.LazyFrame,
) -> _ChunkFrames:
    """Collect counts and all processor plans in one batched pass."""
    processor_frames: list[tuple[_Processor, pl.DataFrame]] = []
    written: list[AggregateWriteReceipt] = []
    try:
        processor_plans = [
            (processor, processor.chunk_aggregate_lazy(transformed, ctx))
            for processor in processors
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
        transformed_frame = transformed.collect()
        rows_kept = transformed_frame.height
        written = _run_processors_sequential(
            workspace, source, processors, transformed_frame.lazy(), ctx, run_id, chunk.chunk_id
        )
    return rows_in, rows_kept, processor_frames, written


def _finish_successful_chunk(
    source: model.Source,
    chunk: Chunk,
    run_id: str,
    started_at: dt.datetime,
    perf_started: float,
    rows_in: int,
    rows_kept: int,
    written: list[AggregateWriteReceipt],
    debugging: bool,
) -> _ChunkOutcome:
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
        files=chunk.files,
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
    to_process: list[tuple[int, Chunk]],
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
        for chunk_order, chunk in to_process:
            _notify_chunk_progress(
                progress_callback,
                source=source,
                chunk=chunk,
                chunk_order=chunk_order,
                chunks_total=chunks_total,
                status="processing",
            )
            outcome = _process_chunk(workspace, source, processors, chunk, run_id)
            _record_chunk_outcome(workspace, source.id, run_id, outcome)
            results_by_order[chunk_order] = outcome.result
        return
    with pool:
        futures: dict[Future[_ChunkOutcome], tuple[int, Chunk]] = {}
        for chunk_order, chunk in to_process:
            _notify_chunk_progress(
                progress_callback,
                source=source,
                chunk=chunk,
                chunk_order=chunk_order,
                chunks_total=chunks_total,
                status="processing",
            )
            future = pool.submit(_process_chunk, workspace, source, processors, chunk, run_id)
            futures[future] = (chunk_order, chunk)
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
    processors: list[_Processor],
    frame: pl.LazyFrame,
    ctx: ChunkContext,
    engine: _CollectEngine,
) -> list[tuple[_Processor, pl.DataFrame]]:
    plans = [(processor, processor.chunk_aggregate_lazy(frame, ctx)) for processor in processors]
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
def _run_processors_sequential(
    workspace: Path,
    source: model.Source,
    processors: list[_Processor],
    frame: pl.LazyFrame,
    ctx: ChunkContext,
    run_id: str,
    chunk_id: str,
) -> list[AggregateWriteReceipt]:
    written: list[AggregateWriteReceipt] = []
    for processor in processors:
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
def _write_processor_outputs(
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
        extra = dict(config.model_extra or {})
        required = set(config.group_by)
        if config.time is not None and config.time.column:
            required.add(config.time.column)
        required.update(_configured_state_source_columns(config))
        required.update(str(value) for value in _extra_string_list(extra, "dedup_keys"))

        if isinstance(config, model.BinaryOutcomeProcessor | model.ScoreDistributionProcessor):
            outcome = extra.get("outcome")
            required.add(
                str(outcome.get("column", "Outcome"))
                if isinstance(outcome, dict)
                else str(extra.get("outcome_column", "Outcome"))
            )
        if isinstance(config, model.NumericDistributionProcessor):
            required.update(_extra_string_list(extra, "properties"))
        elif isinstance(config, model.ScoreDistributionProcessor):
            for name, spec in model.effective_processor_states(config).items():
                if spec.type == "tdigest":
                    required.add(_score_state_source_column(config, name, spec))
            if "personalization" in model.effective_processor_states(config):
                required.update({"CustomerID", "Name"})
            if "novelty" in model.effective_processor_states(config):
                required.update({"CustomerID", "InteractionID", "Name"})
        elif isinstance(config, model.EntityLifecycleProcessor):
            raw_keys = extra.get("keys")
            keys: dict[str, object] = raw_keys if isinstance(raw_keys, dict) else {}
            required.update(
                {
                    str(keys.get("customer_id", "CustomerID")),
                    str(keys.get("order_id", "OrderID")),
                    str(keys.get("monetary", "Monetary")),
                    str(keys.get("purchase_date", "PurchaseDate")),
                }
            )
        elif isinstance(config, model.EntitySetProcessor):
            required.add(str(extra.get("entity", "CustomerID")))
        elif isinstance(config, model.FunnelProcessor):
            if extra.get("entity"):
                required.add(str(extra["entity"]))
        elif isinstance(config, model.SnapshotProcessor):
            if config.snapshot_kind == "accumulating":
                entity = extra.get("entity")
                if entity:
                    required.add(str(entity))
            as_of_column = extra.get("as_of_column")
            if as_of_column:
                required.add(str(as_of_column))
            milestones = extra.get("milestones", [])
            if isinstance(milestones, list):
                required.update(
                    str(item["column"])
                    for item in milestones
                    if isinstance(item, dict) and item.get("column")
                )

        missing = sorted(column for column in required if column and column not in existing)
        if missing:
            failures.append(f"{config.id}: {', '.join(missing)}")
    if failures:
        raise ValueError("processor input columns are missing: " + "; ".join(failures))


def _configured_state_source_columns(config: model.Processor) -> set[str]:
    columns: set[str] = set()
    for name, spec in model.effective_processor_states(config).items():
        extra = dict(spec.model_extra or {})
        if isinstance(config, model.NumericDistributionProcessor) and extra.get("per_property"):
            continue
        source_column = extra.get("source_column")
        if source_column:
            columns.add(str(source_column))
        elif spec.type in {"value_sum", "min", "max", "cpc", "hll", "theta", "topk"}:
            if isinstance(config, model.EntitySetProcessor) and spec.type in {
                "cpc",
                "hll",
                "theta",
                "topk",
            }:
                columns.add(str((config.model_extra or {}).get("entity", "CustomerID")))
            elif isinstance(config, model.ScoreDistributionProcessor) and spec.type in {
                "cpc",
                "hll",
                "theta",
            }:
                columns.add("CustomerID")
            elif isinstance(config, model.SnapshotProcessor) and spec.type in {
                "cpc",
                "hll",
                "theta",
                "topk",
            }:
                columns.add(str((config.model_extra or {}).get("entity", "CustomerID")))
            elif not isinstance(config, model.EntityLifecycleProcessor):
                columns.add(name)
    return columns


def _score_state_source_column(
    config: model.ScoreDistributionProcessor,
    state_name: str,
    spec: model.StateSpec,
) -> str:
    state_extra = dict(spec.model_extra or {})
    if state_extra.get("source_column"):
        return str(state_extra["source_column"])
    if state_name.endswith("_tdigest"):
        return state_name.removesuffix("_tdigest")
    score_role = str(state_extra.get("score", "primary"))
    properties = _extra_string_list(dict(config.model_extra or {}), "score_properties")
    if score_role == "primary":
        return properties[0] if properties else "Propensity"
    if score_role == "calibrated":
        if len(properties) > 1:
            return properties[1]
        return properties[0] if properties else "FinalPropensity"
    return score_role


def _extra_string_list(extra: dict[str, object], key: str) -> list[str]:
    raw = extra.get(key, [])
    return [str(value) for value in raw] if isinstance(raw, list) else []


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
    return source.debugging or _truthy(dict(source.reader.model_extra or {}).get("debugging"))


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
