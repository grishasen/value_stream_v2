"""Data loading workflow page."""

from __future__ import annotations

import threading
import time
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st

from valuestream.engine import clean_rebuild, run_source, run_workspace
from valuestream.readers.discovery import discover
from valuestream.ui import components
from valuestream.ui.context import (
    ValueStreamContext,
    processors_for_source,
    source_last_run,
    source_root,
)
from valuestream.ui.instrumentation import (
    AuthoringEvent,
    AuthoringOutcome,
    AuthoringStage,
    record_event,
    workflow_from_handoff,
)
from valuestream.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _BackgroundRun:
    """One detached source/workspace run and its poll-friendly progress."""

    label: str
    started_at: float
    thread: threading.Thread | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None


# Runs execute on daemon threads keyed by workspace+scope so a browser reload
# or websocket reconnect only detaches the *view*; the load itself continues
# and any session polling this registry picks it back up.
_BACKGROUND_RUNS: dict[str, _BackgroundRun] = {}
_BACKGROUND_RUNS_LOCK = threading.Lock()


def _background_run_key(workspace: Path, scope: str) -> str:
    return f"{Path(workspace).resolve()}::{scope}"


def _progress_recorder(run: _BackgroundRun) -> Callable[[Any], None]:
    """Return a chunk-progress callback that never touches Streamlit state."""

    def update(progress: Any) -> None:
        run.progress = {
            "source_id": getattr(progress, "source_id", ""),
            "chunk_name": getattr(progress, "chunk_name", ""),
            "chunk_order": int(getattr(progress, "chunk_order", 0) or 0),
            "chunks_total": int(getattr(progress, "chunks_total", 0) or 0),
            "status": str(getattr(progress, "status", "processing")),
        }

    return update


def _start_background_run(
    key: str,
    label: str,
    target: Callable[[Callable[[Any], None]], Any],
) -> _BackgroundRun:
    with _BACKGROUND_RUNS_LOCK:
        existing = _BACKGROUND_RUNS.get(key)
        if existing is not None and existing.thread is not None and existing.thread.is_alive():
            return existing
        run = _BackgroundRun(label=label, started_at=time.perf_counter())

        def work() -> None:
            try:
                run.result = target(_progress_recorder(run))
            except Exception as exc:
                logger.exception("Background data load failed: %s", label)
                run.error = str(exc)

        thread = threading.Thread(target=work, name=f"vs-data-load-{label}", daemon=True)
        run.thread = thread
        _BACKGROUND_RUNS[key] = run
        thread.start()
        return run


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3_600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _run_summary(result: Any) -> str:
    if hasattr(result, "sources_ok"):
        return (
            f"{result.status}: {result.sources_ok} source(s) ok, "
            f"{result.sources_partial} partial, {result.sources_failed} failed"
        )
    if hasattr(result, "chunks_rebuilt"):
        return (
            f"rebuilt {result.chunks_rebuilt} chunk(s) across {len(result.source_ids)} "
            f"source(s); removed {result.vacuum.files_deleted} superseded aggregate file(s) "
            f"({_format_bytes(result.vacuum.bytes_deleted)})"
        )
    if hasattr(result, "chunks_ok"):
        return (
            f"{result.status}: {result.chunks_ok} chunk(s) ok, "
            f"{result.chunks_failed} failed, {result.rows_kept:,} rows kept"
        )
    return str(result)


def _render_background_runs(workspace: Path) -> bool:
    """Show every active or just-finished run for this workspace.

    Returns whether any run is still alive so the caller can schedule a poll
    rerun after the rest of the page has rendered.
    """

    prefix = f"{Path(workspace).resolve()}::"
    with _BACKGROUND_RUNS_LOCK:
        entries = [(key, run) for key, run in _BACKGROUND_RUNS.items() if key.startswith(prefix)]
    any_alive = False
    for key, run in entries:
        alive = run.thread is not None and run.thread.is_alive()
        if alive:
            any_alive = True
            progress = dict(run.progress)
            elapsed = _format_elapsed(time.perf_counter() - run.started_at)
            total = progress.get("chunks_total") or 0
            if total:
                verb = {
                    "processing": "Processing",
                    "recovering": "Verifying",
                    "skipped": "Skipping",
                }.get(str(progress.get("status")), "Processing")
                order = progress.get("chunk_order") or 0
                st.progress(
                    min(order / total, 1.0),
                    text=(
                        f"{run.label} — {verb} chunk {order}/{total}: "
                        f"{progress.get('chunk_name', '')} · Elapsed {elapsed}"
                    ),
                )
            else:
                st.progress(0.0, text=f"{run.label} — discovering files... · Elapsed {elapsed}")
            st.caption(
                "The run continues on the server — reloading or leaving this page "
                "does not interrupt it."
            )
        else:
            with _BACKGROUND_RUNS_LOCK:
                _BACKGROUND_RUNS.pop(key, None)
            if run.error is not None:
                st.toast("Data load failed.", icon=":material/error:")
                st.error(f"{run.label} failed: {run.error}")
            elif run.result is not None:
                st.toast("Data load finished.", icon=":material/database_upload:")
                st.success(f"{run.label} — {_run_summary(run.result)}")
    return any_alive


def render(ctx: ValueStreamContext) -> None:
    """Render guided data loading controls."""
    components.render_page_header(
        "Data Load",
        "Load, refresh, and validate source data before opening reports.",
        status="ok" if ctx.validation.ok else "blocked",
        status_label="Catalog OK" if ctx.validation.ok else "Catalog blocked",
    )

    if not ctx.validation.ok:
        components.render_validation_summary(ctx.validation.issues, ok=False)
        st.stop()

    sources = _ordered_sources(ctx.catalog.pipelines.sources)
    if not sources:
        st.info(
            "No data sources are configured for this workspace. Add a source in "
            "AI Configuration Studio before loading or rebuilding data."
        )
        st.link_button(
            "Add source in AI Configuration Studio",
            "/ai_configuration_studio",
            icon=":material/add_circle:",
            type="primary",
        )
        return

    force_all = st.toggle(
        "Force rebuild",
        value=False,
        help="Re-run chunks even when the current files and catalog hash were already processed.",
    )
    run_col, rebuild_col = st.columns([0.5, 0.5], vertical_alignment="center")
    if run_col.button("Run All Sources", type="primary", icon=":material/play_arrow:"):
        _run_all(ctx, force=force_all)
    if rebuild_col.button(
        "Rebuild from scratch",
        icon=":material/delete_sweep:",
        help=(
            "Force-rebuild one or all sources, verify complete success, then remove "
            "superseded aggregate files. Run and config audit metadata is retained."
        ),
    ):
        st.session_state["data_load_clean_rebuild_confirm"] = False
        _clean_rebuild_dialog(ctx)

    runs_active = _render_background_runs(ctx.workspace)

    tabs = st.tabs([_source_tab_label(source.id) for source in sources])
    for tab, source in zip(tabs, sources, strict=True):
        with tab:
            _render_source_tab(ctx, source, force_default=force_all)

    if runs_active:
        time.sleep(1.5)
        st.rerun()


def _ordered_sources(sources: Iterable[Any]) -> list[Any]:
    """Show newer/default-adjacent sources first without mutating the catalog."""
    return list(sources)


def _source_tab_label(source_id: str) -> str:
    lower = source_id.casefold()
    if "holding" in lower or "clv" in lower:
        return "Product Holdings / CLV"
    if "ih" in lower or "interaction" in lower:
        return "Interaction History"
    return source_id


def _render_source_tab(ctx: ValueStreamContext, source: Any, *, force_default: bool) -> None:
    root = source_root(ctx, source)
    latest = source_last_run(ctx, source.id)
    chunks = _safe_discover(ctx, source)
    processors = processors_for_source(ctx, source.id)

    components.metric_cards(
        [
            {"label": "Source", "value": source.id},
            {"label": "Reader", "value": source.reader.kind},
            {
                "label": "Files",
                "value": len({file for chunk in chunks for file in chunk.files}),
            },
            {"label": "Chunks", "value": len(chunks)},
            {"label": "Processors", "value": len(processors)},
            {"label": "Last Status", "value": str(latest.get("status", "not run"))},
        ],
        columns=6,
    )
    st.caption(f"Root: `{root}` · Pattern: `{source.reader.file_pattern}`")

    mode = st.segmented_control(
        "Data Source",
        ["Workspace folder", "Upload files"],
        default="Workspace folder",
        key=f"data_load_mode_{source.id}",
        help="Use files already in the configured workspace folder, or upload files into that folder before running.",
    )

    if mode == "Upload files":
        _render_upload(ctx, source, root)
        chunks = _safe_discover(ctx, source)

    with components.card():
        st.write("### Run Source")
        run_cols = st.columns([0.25, 0.25, 0.5], vertical_alignment="center")
        force_source = run_cols[0].toggle(
            "Force source",
            value=force_default,
            key=f"data_load_force_{source.id}",
        )
        if run_cols[1].button(
            "Run Source",
            key=f"data_load_run_{source.id}",
            type="primary",
            icon=":material/play_arrow:",
            disabled=not chunks,
        ):
            _run_one(ctx, source.id, force=force_source)
        if not chunks:
            run_cols[2].warning("No files discovered for this source.")
        else:
            run_cols[2].success(f"{len(chunks)} chunk(s) ready.")

    with components.bordered_panel(
        "Discovered Chunks", "The idempotent units Value Stream will process."
    ):
        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "files": len(chunk.files),
                "first_file": str(chunk.files[0]) if chunk.files else "",
            }
            for chunk in chunks[:100]
        ]
        if rows:
            components.dataframe_with_search(rows, key=f"data_load_chunks_{source.id}")
        else:
            st.info("No chunks discovered.")


def _render_upload(ctx: ValueStreamContext, source: Any, root: Path) -> None:
    accepted = _accepted_types(source.reader.kind)
    uploaded = st.file_uploader(
        "Upload source files",
        type=accepted,
        accept_multiple_files=True,
        key=f"data_load_upload_{source.id}",
        help=f"Files are saved into `{root}` before the source run.",
    )
    if not uploaded:
        return
    if st.button("Save Uploaded Files", key=f"data_load_save_uploads_{source.id}"):
        root.mkdir(parents=True, exist_ok=True)
        saved = 0
        extracted = 0
        for upload in uploaded:
            path = root / upload.name
            path.write_bytes(upload.getbuffer())
            saved += 1
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path, "r") as zip_ref:
                    zip_ref.extractall(root)
                    extracted += len(zip_ref.namelist())
        st.toast("Uploaded files saved.", icon=":material/database_upload:")
        st.success(f"Saved {saved} file(s) into `{root}`. Extracted {extracted} zipped member(s).")


def _run_all(ctx: ValueStreamContext, *, force: bool) -> None:
    _record_authoring_run_started()
    workspace = ctx.workspace

    def target(progress_callback: Callable[[Any], None]) -> Any:
        return run_workspace(workspace, force=force, progress_callback=progress_callback)

    _start_background_run(
        _background_run_key(workspace, "workspace"),
        "Workspace run",
        target,
    )
    st.rerun()


def _run_one(ctx: ValueStreamContext, source_id: str, *, force: bool) -> None:
    _record_authoring_run_started()
    workspace = ctx.workspace

    def target(progress_callback: Callable[[Any], None]) -> Any:
        return run_source(workspace, source_id, force=force, progress_callback=progress_callback)

    _start_background_run(
        _background_run_key(workspace, f"source:{source_id}"),
        f"Source run · {source_id}",
        target,
    )
    st.rerun()


def _record_authoring_run_started() -> None:
    """Record an explicit run only when Data Load was reached from authoring."""

    handoff_workflow = workflow_from_handoff(st.query_params.get("from"))
    if handoff_workflow is None:
        return
    record_event(
        st.session_state,
        event=AuthoringEvent.RUN_STARTED,
        workflow=handoff_workflow,
        stage=AuthoringStage.RUN,
        outcome=AuthoringOutcome.STARTED,
        once=True,
    )


@st.dialog(
    "Rebuild from scratch",
    width="medium",
    icon=":material/delete_sweep:",
    on_dismiss="rerun",
)
def _clean_rebuild_dialog(ctx: ValueStreamContext) -> None:
    st.warning(
        "This operation force-rebuilds the selected source data and permanently removes "
        "older aggregate Parquet files only after every discovered chunk succeeds. "
        "Run history and configuration audit records are kept."
    )
    scope = st.segmented_control(
        "Scope",
        ["Current source", "All sources"],
        default="Current source",
        required=True,
        width="stretch",
        key="data_load_clean_rebuild_scope",
    )
    sources = _ordered_sources(ctx.catalog.pipelines.sources)
    if scope == "All sources":
        selected_source_ids = tuple(source.id for source in ctx.catalog.pipelines.sources)
        st.caption(f"{len(selected_source_ids)} configured source(s) will be rebuilt.")
    else:
        selected_source = st.selectbox(
            "Source",
            sources,
            format_func=lambda source: f"{_source_tab_label(source.id)} · {source.id}",
            key="data_load_clean_rebuild_source",
        )
        selected_source_ids = (selected_source.id,)

    files, bytes_used = _aggregate_inventory(ctx.workspace, selected_source_ids)
    components.metric_cards(
        [
            {"label": "Sources", "value": len(selected_source_ids)},
            {"label": "Aggregate files", "value": files},
            {"label": "Current storage", "value": _format_bytes(bytes_used)},
        ],
        columns=3,
    )
    st.caption(
        "Deletion starts after the forced runs and coverage checks complete. If any source "
        "fails, discovers no chunks, or the catalog changes, existing files are preserved."
    )
    confirmed = st.checkbox(
        "I understand that superseded aggregate files in this scope will be permanently deleted.",
        key="data_load_clean_rebuild_confirm",
    )
    if st.button(
        "Rebuild and remove old aggregates",
        type="primary",
        icon=":material/restart_alt:",
        disabled=not confirmed,
        width="stretch",
    ):
        _run_clean_rebuild(ctx, selected_source_ids)


def _run_clean_rebuild(ctx: ValueStreamContext, source_ids: tuple[str, ...]) -> None:
    workspace = ctx.workspace

    def target(progress_callback: Callable[[Any], None]) -> Any:
        return clean_rebuild(
            workspace,
            source_ids=source_ids,
            progress_callback=progress_callback,
        )

    _start_background_run(
        _background_run_key(workspace, "rebuild"),
        "Clean rebuild · " + ", ".join(source_ids),
        target,
    )
    st.rerun()


def _aggregate_inventory(workspace: Path, source_ids: Iterable[str]) -> tuple[int, int]:
    files = 0
    bytes_used = 0
    for source_id in frozenset(source_ids):
        source_root = workspace / "aggregates" / source_id
        if source_root.exists():
            for path in source_root.glob("**/*.parquet"):
                try:
                    if path.is_file():
                        files += 1
                        bytes_used += path.stat().st_size
                except OSError:
                    continue
    return files, bytes_used


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _safe_discover(ctx: ValueStreamContext, source: Any):
    try:
        return discover(ctx.workspace, source)
    except Exception:
        logger.exception(
            "Source discovery failed: workspace=%s source=%s",
            ctx.workspace,
            source.id,
        )
        return []


def _accepted_types(kind: str) -> list[str]:
    if kind == "pega_ds_export":
        return ["zip", "json"]
    if kind == "parquet":
        return ["parquet", "zip"]
    if kind == "csv":
        return ["csv", "zip"]
    if kind == "xlsx":
        return ["xlsx", "xls", "zip"]
    return ["zip", "json", "parquet", "csv", "xlsx", "xls"]
