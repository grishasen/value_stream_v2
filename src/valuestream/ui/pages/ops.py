"""Pipeline and operations page."""

from __future__ import annotations

import streamlit as st

from valuestream.engine import run_workspace
from valuestream.ui import components
from valuestream.ui.context import (
    ValueStreamContext,
    chunks_for_run,
    processors_for_source,
    recent_runs_frame,
    source_last_run,
)
from valuestream.utils.logger import get_logger

logger = get_logger(__name__)


def render(ctx: ValueStreamContext) -> None:
    """Render operational state for the workspace."""
    components.render_page_header(
        "Pipelines / Ops",
        "Sources, recent runs, chunk detail, and freshness indicators.",
        status="ok" if ctx.validation.ok else "blocked",
        status_label="Catalog OK" if ctx.validation.ok else "Catalog blocked",
    )

    run_cols = st.columns([0.18, 0.18, 0.64], vertical_alignment="center")
    force = run_cols[0].toggle("Force rebuild", value=False, key="ops_force")
    if run_cols[1].button("Run Workspace", type="primary", icon=":material/play_arrow:"):
        try:
            with st.status("Running workspace", expanded=True) as status:
                chunk_progress = components.chunk_progress_indicator(include_source=True)
                result = run_workspace(
                    ctx.workspace,
                    force=force,
                    progress_callback=chunk_progress,
                )
                status.write(
                    f"{result.sources_ok} ok, {result.sources_partial} partial, {result.sources_failed} failed."
                )
                status.update(label=f"Workspace run {result.status}", state="complete")
            st.success(f"Workspace run finished: {result.status}")
        except Exception as exc:  # pragma: no cover - Streamlit display path
            logger.exception("Ops workspace run failed: workspace=%s", ctx.workspace)
            st.error(str(exc))
    run_cols[2].caption("Use Data Load for source-specific upload and run controls.")

    _source_cards(ctx)
    _runs_and_chunks(ctx)


def _source_cards(ctx: ValueStreamContext) -> None:
    st.write("### Sources")
    sources = ctx.catalog.pipelines.sources
    if not sources:
        st.info("No sources configured.")
        return
    columns = st.columns(min(3, len(sources)))
    for idx, source in enumerate(sources):
        latest = source_last_run(ctx, source.id)
        with columns[idx % len(columns)], components.card():
            top = st.columns([0.72, 0.28], vertical_alignment="center")
            top[0].write(f"#### {source.id}")
            with top[1]:
                components.status_badge(
                    str(latest.get("status", "not run")),
                    str(latest.get("status", "pending")),
                )
            st.caption(
                f"{source.reader.kind} · {len(processors_for_source(ctx, source.id))} processor(s)"
            )
            st.write(
                f"Rows kept: **{components.format_count(latest.get('rows_kept'))}** · "
                f"Chunks: **{latest.get('chunks_ok', 0)} / {latest.get('chunks_total', 0)}**"
            )
            st.progress(_progress(latest), text="chunk completion")


def _runs_and_chunks(ctx: ValueStreamContext) -> None:
    runs = recent_runs_frame(ctx, limit=50)
    left, right = st.columns([1.2, 1.0], gap="large")
    with left, components.bordered_panel("Recent Runs", "Select a run to inspect chunk details."):
        if runs.is_empty():
            st.info("No pipeline runs recorded.")
            return
        selected = st.selectbox(
            "Run",
            range(runs.height),
            format_func=lambda idx: (
                f"{runs[idx, 'source_id']} · {runs[idx, 'status']} · "
                f"{components.format_timestamp(runs[idx, 'finished_at'])}"
            ),
            label_visibility="collapsed",
        )
        st.dataframe(runs, hide_index=True, width="stretch", height=360)

    with right, components.bordered_panel("Chunks", "Chunk ledger for the selected run."):
        if runs.is_empty():
            st.info("No run selected.")
            return
        run_id = str(runs[selected, "id"] or "") if "id" in runs.columns else ""
        if not run_id or run_id == "nan":
            st.info("Run id is not included in the current recent-runs projection.")
            return
        chunks = chunks_for_run(ctx, run_id)
        if chunks.is_empty():
            st.info("No chunk detail found for this run.")
        else:
            components.dataframe_with_search(chunks, key="ops_chunks", height=360)


def _progress(latest: dict) -> float:
    total = latest.get("chunks_total") or 0
    ok = latest.get("chunks_ok") or 0
    try:
        return float(ok) / float(total) if total else 0.0
    except ZeroDivisionError:
        return 0.0
