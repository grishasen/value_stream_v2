"""Configuration builder page."""

from __future__ import annotations

import datetime as dt
import secrets
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import plotly.graph_objects as go  # type: ignore[import-untyped]
import polars as pl
import streamlit as st
import yaml

from valuestream.ai.settings import (
    DEFAULT_CHAT_AGENT_PROMPT,
    load_chat_with_data_config,
    load_llm_settings_config,
    write_chat_with_data_config,
)
from valuestream.charts import render_chart
from valuestream.config import model
from valuestream.readers.discovery import discover
from valuestream.readers.io import cleanup_temporaries, read
from valuestream.ui import (
    builder,
    components,
    config_help,
    dimension_profile,
    field_remap,
    forms,
    recipe_library,
)
from valuestream.ui.context import ValueStreamContext, catalog_counts, processors_for_source
from valuestream.ui.data import query_tile
from valuestream.ui.instrumentation import (
    AuthoringEvent,
    AuthoringOutcome,
    AuthoringStage,
    AuthoringWorkflow,
    record_event,
    start_journey,
)
from valuestream.ui.presentation import humanize_identifier
from valuestream.ui.theme import (
    PLOTLY_DARK_COLORWAY,
    PLOTLY_LIGHT_COLORWAY,
    dashboard_theme,
)
from valuestream.utils.logger import get_logger
from valuestream.utils.names import capitalize_fields

logger = get_logger(__name__)

PROCESSOR_KINDS = list(forms.PROCESSOR_KIND_OPTIONS)
METRIC_ACTION_CREATE = "Create Metric"
METRIC_ACTION_EDIT = "Edit Existing Metric"
METRIC_CREATE_LIBRARY = "From Recipe Library"
METRIC_CREATE_SCRATCH = "From Scratch"
NEW_TILE_KEY = "__new_tile_draft__"
NEW_TILE_LABEL = "New tile draft"
NEW_DASHBOARD_KEY = "__new_dashboard__"
NEW_PAGE_KEY = "__new_page__"


@dataclass(frozen=True)
class BuilderStepDefinition:
    """One compact Builder navigation step and its user-facing task."""

    label: str
    phase: str
    task: str


BUILDER_STEP_DEFINITIONS = (
    BuilderStepDefinition(
        "Workspace Health", "Define", "Confirm that the current workspace is valid."
    ),
    BuilderStepDefinition("Sources", "Define", "Review how source files become working rows."),
    BuilderStepDefinition(
        "Processors", "Define", "Define the aggregate states computed for each source."
    ),
    BuilderStepDefinition(
        "Dimensions", "Define", "Choose safe dimensions for filtering and breakdowns."
    ),
    BuilderStepDefinition("Metrics", "Model", "Create or maintain decision-ready metrics."),
    BuilderStepDefinition("Reports / Tiles", "Report", "Place metrics into useful report views."),
    BuilderStepDefinition(
        "Chat Review", "Review", "Review the aggregate context available to Chat With Data."
    ),
    BuilderStepDefinition(
        "Settings", "Review", "Review shared workspace defaults and report appearance."
    ),
    BuilderStepDefinition(
        "Export current workspace",
        "Finish",
        "Choose the next useful outcome, then download catalog files if needed.",
    ),
)
BUILDER_STEPS = tuple(step.label for step in BUILDER_STEP_DEFINITIONS)
BUILDER_STEP_BY_LABEL = {step.label: step for step in BUILDER_STEP_DEFINITIONS}
BUILDER_LAST_OUTCOME_KEY = "builder_last_apply_outcome"
BUILDER_DIMENSION_PROPOSALS_KEY = "builder_dimension_pending_proposals"
BUILDER_PENDING_TILE_DELETE_KEY = "builder_tile_delete_draft"


@dataclass(frozen=True)
class ReportLibraryGroup:
    """Purpose-led group of chart types shown in the report library."""

    label: str
    icon: str
    description: str
    chart_types: tuple[str, ...]


REPORT_LIBRARY_GROUPS = {
    "summary": ReportLibraryGroup(
        label="Summary & detail",
        icon=":material/dashboard:",
        description="Headline values, progress indicators, and exact supporting detail.",
        chart_types=("kpi_card", "gauge", "table"),
    ),
    "trend": ReportLibraryGroup(
        label="Trends over time",
        icon=":material/show_chart:",
        description="Change, seasonality, and two-measure movement across time.",
        chart_types=(
            "line",
            "stacked_area",
            "combo",
            "calendar_heatmap",
            "descriptive_line",
        ),
    ),
    "compare": ReportLibraryGroup(
        label="Compare & rank",
        icon=":material/leaderboard:",
        description="Category comparisons, contribution, ranking, and uncertainty.",
        chart_types=(
            "bar",
            "waterfall",
            "pareto",
            "interval",
            "bar_polar",
            "experiment_z_score",
            "experiment_odds_ratio",
        ),
    ),
    "explore": ReportLibraryGroup(
        label="Distributions & models",
        icon=":material/query_stats:",
        description="Relationships, distributions, diagnostic curves, and model quality.",
        chart_types=(
            "scatter",
            "heatmap",
            "cohort_heatmap",
            "boxplot",
            "histogram",
            "corr",
            "rfm_density",
            "descriptive_boxplot",
            "descriptive_histogram",
            "descriptive_heatmap",
            "calibration_curve",
            "roc_curve",
            "precision_recall_curve",
            "gain_curve",
            "lift_curve",
            "model",
        ),
    ),
    "flow": ReportLibraryGroup(
        label="Flow & composition",
        icon=":material/account_tree:",
        description="Mix, hierarchy, geography, journeys, and lifecycle progression.",
        chart_types=(
            "treemap",
            "clv_treemap",
            "donut",
            "geo_map",
            "sankey",
            "funnel",
            "descriptive_funnel",
            "exposure",
        ),
    ),
}

REPORT_LIBRARY_GROUP_BY_CHART = {
    chart_type: group_id
    for group_id, group in REPORT_LIBRARY_GROUPS.items()
    for chart_type in group.chart_types
}

REPORT_LIBRARY_CHART_DESCRIPTIONS = {
    "bar": "Compare magnitudes across discrete categories.",
    "bar_polar": "Compare cyclical or directional categories around a circle.",
    "boxplot": "Compare medians, spread, and outliers between groups.",
    "calendar_heatmap": "Reveal daily activity and seasonality on a calendar grid.",
    "calibration_curve": "Compare predicted probabilities with observed outcomes.",
    "clv_treemap": "Show customer-value hierarchy through nested area.",
    "cohort_heatmap": "Compare retention or behavior across cohort periods.",
    "combo": "Place two measures on coordinated bar and line axes.",
    "corr": "Scan the strength and direction of pairwise relationships.",
    "descriptive_boxplot": "Compare aggregate distribution summaries by group.",
    "descriptive_funnel": "Follow aggregate descriptive measures through ordered stages.",
    "descriptive_heatmap": "Compare an aggregate statistic across two dimensions.",
    "descriptive_histogram": "Show the distribution of an aggregate numeric property.",
    "descriptive_line": "Track an aggregate statistic over time or ordered categories.",
    "donut": "Show a small set of parts as shares of a whole.",
    "experiment_odds_ratio": "Compare experiment effects with confidence intervals.",
    "experiment_z_score": "Compare standardized experiment effects around zero.",
    "exposure": "Show how populations progress through exposure levels.",
    "funnel": "Show volume retained through ordered journey stages.",
    "gain_curve": "Show cumulative positives captured as coverage increases.",
    "gauge": "Show a current value against a reference or operating range.",
    "geo_map": "Compare a measure across geographic locations.",
    "heatmap": "Compare intensity across two categorical dimensions.",
    "histogram": "Show how numeric observations are distributed across bins.",
    "interval": "Compare estimates together with their uncertainty bounds.",
    "kpi_card": "Present one decision-ready value with optional comparison context.",
    "lift_curve": "Show improvement over random selection as coverage increases.",
    "line": "Track one or more measures across time or an ordered axis.",
    "model": "Summarize model performance across thresholds or score bands.",
    "pareto": "Rank contributors and show their cumulative share.",
    "precision_recall_curve": "Show the precision-recall tradeoff across thresholds.",
    "rfm_density": "Map customer density across recency and frequency/value space.",
    "roc_curve": "Show true-positive versus false-positive performance by threshold.",
    "sankey": "Show volume flowing between stages or categories.",
    "scatter": "Explore the relationship between two numeric measures.",
    "stacked_area": "Show total change over time together with category composition.",
    "table": "Inspect exact ranked or operational values in native tabular form.",
    "treemap": "Show hierarchical composition through nested area.",
    "waterfall": "Explain how positive and negative contributions build a total.",
}

REPORT_LIBRARY_PILLS_MAX = 12
REPORT_LIBRARY_CHART_LABELS = {
    "clv_treemap": "CLV treemap",
    "kpi_card": "KPI card",
    "precision_recall_curve": "Precision-recall curve",
    "rfm_density": "RFM density",
    "roc_curve": "ROC curve",
}


def render(ctx: ValueStreamContext) -> None:
    """Render validation-first YAML catalog builders."""
    start_journey(st.session_state, workflow=AuthoringWorkflow.BUILDER)
    components.render_page_header(
        "Configuration Builder",
        "Make one reviewable workspace change at a time, apply it safely, then open a report or refresh data.",
        status="ok" if ctx.validation.ok else "warning",
        status_label="Catalog OK" if ctx.validation.ok else "Needs review",
    )
    with st.sidebar:
        if st.button("Reload catalog", icon=":material/refresh:"):
            st.rerun()

    _render_apply_notice()
    _render_immediate_config_warnings(ctx)
    with st.expander("How this workflow works", expanded=False, icon=":material/help:"):
        st.markdown(
            "1. Edit one object in the current step.\n"
            "2. Review the draft revision and apply it to the workspace.\n"
            "3. Open a report immediately, or use Data Load when aggregate data must be refreshed."
        )
        st.caption(
            "YAML remains the source of behavior. Exact generated definitions stay available "
            "under Technical details."
        )
    _builder_steps(ctx)


def _builder_steps(ctx: ValueStreamContext) -> None:
    steps = list(BUILDER_STEPS)
    next_step = st.session_state.pop("builder_next_step", None)
    if next_step in steps:
        st.session_state["builder_step"] = next_step
        current_step = next_step
    else:
        current_step = st.session_state.get("builder_step", steps[0])
    if current_step not in steps:
        current_step = steps[0]
        st.session_state["builder_step"] = current_step
    st.session_state["builder_step"] = current_step
    if st.session_state.get("builder_step_jump") != current_step:
        st.session_state["builder_step_jump"] = current_step

    index = steps.index(current_step)
    definition = BUILDER_STEP_BY_LABEL[current_step]
    with components.card():
        task_col, controls_col = st.columns([0.58, 0.42], vertical_alignment="center")
        with task_col:
            st.caption(f"Step {index + 1} of {len(steps)} · {definition.phase}")
            st.write(f"### {definition.label}")
            st.caption(definition.task)
        with controls_col:
            # Bottom alignment levels the primary action with the Jump input
            # itself rather than with the label + input block's midpoint.
            jump_col, action_col = st.columns([0.52, 0.48], vertical_alignment="bottom")
            with jump_col:
                st.selectbox(
                    "Jump to step",
                    steps,
                    key="builder_step_jump",
                    on_change=_apply_builder_jump,
                )
            with action_col:
                save_slot = st.empty()
                # The active editor fills this placeholder after it computes
                # its canonical dirty state. The initial write supports
                # fragments.
                save_slot.caption("")
        st.progress((index + 1) / len(steps), text=f"{definition.phase} · {definition.label}")
        registry = st.session_state.get(builder.BUILDER_DRAFTS_KEY, {})
        pending_count = len(registry) if isinstance(registry, dict) else 0
        if pending_count:
            st.caption(
                f"{pending_count} unapplied draft{'s' if pending_count != 1 else ''} preserved "
                "while you move between steps."
            )
        back_col, task_nav_col = st.columns([0.2, 0.8], vertical_alignment="center")
        back_col.button(
            "Back",
            icon=":material/arrow_back:",
            disabled=index == 0,
            width="stretch",
            key="builder_back",
            on_click=_set_builder_step,
            args=(steps[max(index - 1, 0)],),
        )
        task_nav_col.caption(
            "Jump when you know where to go; the primary Continue action keeps the guided order."
        )
    draft_slot = st.empty()
    # Claim the cross-fragment draft region on the full app run. Some editor
    # modes return before rendering a draft, then populate it on a fragment
    # rerun after the user changes mode.
    draft_slot.caption("")
    handlers = {
        "Workspace Health": lambda: _health(ctx, save_slot),
        "Sources": lambda: _source_builder(ctx, save_slot, draft_slot),
        "Processors": lambda: _processor_builder(ctx, save_slot, draft_slot),
        "Dimensions": lambda: _dimensions_builder(ctx, save_slot, draft_slot),
        "Metrics": lambda: _metric_builder(ctx.workspace, ctx.catalog, save_slot, draft_slot),
        "Reports / Tiles": lambda: _tile_builder(ctx.workspace, ctx.catalog, save_slot, draft_slot),
        "Chat Review": lambda: _chat_review(ctx, save_slot, draft_slot),
        "Settings": lambda: _settings_builder(ctx, save_slot, draft_slot),
        "Export current workspace": lambda: _export_current_workspace(ctx, save_slot),
    }
    handlers[current_step]()


def _set_builder_step(step: str) -> None:
    """Synchronize contextual Builder navigation before Streamlit reruns."""
    st.session_state["builder_step"] = step
    st.session_state["builder_step_jump"] = step


def _apply_builder_jump() -> None:
    """Move to the selected outline step on the widget-triggered rerun."""
    selected = st.session_state.get("builder_step_jump")
    if selected in BUILDER_STEPS:
        st.session_state["builder_step"] = selected


def _next_builder_step() -> str | None:
    current = st.session_state.get("builder_step", BUILDER_STEPS[0])
    if current not in BUILDER_STEPS:
        return BUILDER_STEPS[0]
    index = BUILDER_STEPS.index(current)
    return BUILDER_STEPS[index + 1] if index + 1 < len(BUILDER_STEPS) else None


def _render_continue_primary(save_slot: Any) -> None:
    """Render the current step's primary Continue action."""
    next_step = _next_builder_step()
    save_slot.empty()
    clicked = save_slot.button(
        "Continue",
        type="primary",
        icon=":material/arrow_forward:",
        disabled=next_step is None,
        width="stretch",
        key=f"builder_primary_continue_{st.session_state.get('builder_step', 'step')}",
    )
    if clicked and next_step is not None:
        # Most editors render this action from inside a fragment. An on_click
        # callback would advance session state but only rerun the fragment, so
        # the page would never move; escalate to a full-app rerun instead.
        # Only the plain step key may change here: the Jump widget key is
        # already instantiated for this run, and _builder_steps re-syncs it
        # before the widget is created on the next run.
        st.session_state["builder_step"] = next_step
        st.rerun(scope="app")


class _ApplyActionSlot:
    """Placeholder adapter that gives shared review flows Builder vocabulary."""

    def __init__(self, slot: Any) -> None:
        self._slot = slot

    def empty(self) -> Any:
        return self._slot.empty()

    def button(self, label: str, *args: Any, **kwargs: Any) -> Any:
        del label
        kwargs["icon"] = ":material/publish:"
        kwargs["width"] = "stretch"
        return self._slot.button("Apply to workspace", *args, **kwargs)


def _claim_fragment_action_slots(save_slot: Any, draft_slot: Any) -> None:
    """Reserve external action regions during every fragment's initial run."""

    save_slot.empty()
    draft_slot.empty()


def _render_editor_primary_action(
    *,
    save_slot: Any,
    draft_slot: Any,
    status: builder.BuilderDraftStatus,
    valid: bool,
    widget_prefixes: tuple[str, ...],
    help_text: str,
) -> bool:
    """Render one canonical draft status and the single active apply action."""
    registered = builder.registered_builder_draft(st.session_state, status.key)
    draft_slot.empty()
    if not status.dirty:
        _render_continue_primary(save_slot)
        if registered is not None:
            _render_registered_draft(
                draft_slot=draft_slot,
                status=status,
                registered=registered,
                widget_prefixes=widget_prefixes,
            )
        return False

    builder.update_builder_draft_registry(
        st.session_state,
        status,
        widget_prefixes=widget_prefixes,
    )
    if valid:
        _record_builder_valid_proposal()
    restored = st.session_state.pop("builder_restored_draft_key", None) == status.key

    with draft_slot.container(border=True):
        message_col, discard_col = st.columns([0.8, 0.2], vertical_alignment="center")
        with message_col:
            st.write(f"**Editing draft · revision `{status.revision}`**")
            st.caption(
                ("Restored from this session. " if restored else "Stored for this session. ")
                + ("Ready to apply." if valid else "Resolve the highlighted issue before applying.")
            )
        discard_col.button(
            "Discard draft",
            icon=":material/delete_sweep:",
            width="stretch",
            key=f"builder_discard_{builder.widget_key_fragment(status.key)}",
            on_click=_discard_registered_builder_draft,
            args=(status.key, widget_prefixes),
        )

    save_slot.empty()
    apply_requested = save_slot.button(
        "Apply to workspace",
        type="primary",
        icon=":material/publish:",
        disabled=not valid,
        help=help_text,
        width="stretch",
        key=f"builder_apply_{builder.widget_key_fragment(status.key)}",
    )
    if apply_requested:
        _record_builder_reviewed()
    return apply_requested


def _render_registered_draft(
    *,
    draft_slot: Any,
    status: builder.BuilderDraftStatus,
    registered: Mapping[str, Any],
    widget_prefixes: tuple[str, ...],
) -> None:
    """Offer an explicit restore when Streamlit cleaned step-local widget state."""
    revision = str(registered.get("revision", "saved") or "saved")
    same_baseline = registered.get("baseline_hash") == status.baseline_hash
    widget_state = registered.get("widget_state")
    can_restore = same_baseline and isinstance(widget_state, Mapping) and bool(widget_state)
    with draft_slot.container(border=True):
        message_col, restore_col, discard_col = st.columns(
            [0.62, 0.19, 0.19], vertical_alignment="center"
        )
        with message_col:
            st.write(f"**Unapplied draft available · revision `{revision}`**")
            st.caption(
                "Restore the exact session draft before applying it."
                if same_baseline
                else "The persisted object changed after this draft was stored. Discard it to continue."
            )
        restore_col.button(
            "Restore draft",
            icon=":material/restore:",
            disabled=not can_restore,
            width="stretch",
            key=f"builder_restore_{builder.widget_key_fragment(status.key)}",
            on_click=_restore_registered_builder_draft,
            args=(status.key,),
        )
        discard_col.button(
            "Discard draft",
            icon=":material/delete_sweep:",
            width="stretch",
            key=f"builder_discard_available_{builder.widget_key_fragment(status.key)}",
            on_click=_discard_registered_builder_draft,
            args=(status.key, widget_prefixes),
        )


def _restore_registered_builder_draft(key: str) -> None:
    """Restore shadow widget values in the callback prefix before rerendering."""
    if builder.restore_builder_draft(st.session_state, key):
        st.session_state["builder_restored_draft_key"] = key


def _discard_registered_builder_draft(
    key: str,
    widget_prefixes: tuple[str, ...],
) -> None:
    """Discard a registered proposal before the editor widgets rerender."""
    builder.discard_builder_draft(
        st.session_state,
        key,
        widget_prefixes=widget_prefixes,
    )


def _complete_builder_apply(
    *,
    status: builder.BuilderDraftStatus,
    scope: str,
    label: str,
    source_ids: list[str] | tuple[str, ...] = (),
    message: str,
) -> None:
    """Record a safe outcome and refresh the catalog after a successful apply."""
    requires_data_run = builder.builder_requires_data_run(
        scope,
        status.baseline_payload,
        status.draft_payload,
    )
    outcome = builder.builder_apply_outcome(
        label,
        source_ids=source_ids,
        requires_data_run=requires_data_run,
    )
    _store_builder_apply_outcome(outcome)
    _record_builder_applied(outcome)
    raw = st.session_state.get(builder.BUILDER_DRAFTS_KEY)
    registry = dict(raw) if isinstance(raw, dict) else {}
    registry.pop(status.key, None)
    st.session_state[builder.BUILDER_DRAFTS_KEY] = registry
    st.session_state["builder_apply_notice"] = message
    st.rerun(scope="app")


def _record_builder_valid_proposal() -> None:
    """Emit the first structurally ready Builder proposal without draft details."""
    record_event(
        st.session_state,
        event=AuthoringEvent.VALID_PROPOSAL,
        workflow=AuthoringWorkflow.BUILDER,
        stage=AuthoringStage.DRAFT,
        outcome=AuthoringOutcome.SUCCESS,
        once=True,
    )


def _record_builder_reviewed() -> None:
    """Record the explicit review decision represented by the apply click."""
    record_event(
        st.session_state,
        event=AuthoringEvent.REVIEWED,
        workflow=AuthoringWorkflow.BUILDER,
        stage=AuthoringStage.REVIEW,
        outcome=AuthoringOutcome.SUCCESS,
        once=True,
    )


def _record_builder_apply_failed(exc: Exception) -> None:
    """Emit an allowlisted apply failure without exception or catalog payloads."""
    if isinstance(exc, TimeoutError):
        outcome = AuthoringOutcome.TIMEOUT
    elif isinstance(exc, (PermissionError, ValueError)):
        outcome = AuthoringOutcome.BLOCKED
    else:
        outcome = AuthoringOutcome.ERROR
    record_event(
        st.session_state,
        event=AuthoringEvent.FAILED,
        workflow=AuthoringWorkflow.BUILDER,
        stage=AuthoringStage.APPLY,
        outcome=outcome,
    )


def _record_builder_applied(outcome: builder.BuilderApplyOutcome) -> None:
    """Emit the bounded Builder apply event without catalog or sample details."""
    record_event(
        st.session_state,
        event=AuthoringEvent.APPLIED,
        workflow=AuthoringWorkflow.BUILDER,
        stage=AuthoringStage.APPLY,
        outcome=AuthoringOutcome.SUCCESS,
        requires_data_run=outcome.requires_data_run,
    )


def _store_builder_apply_outcome(outcome: builder.BuilderApplyOutcome) -> None:
    """Keep an unresolved data-refresh requirement ahead of later report-only work."""
    previous = st.session_state.get(BUILDER_LAST_OUTCOME_KEY)
    if isinstance(previous, Mapping) and previous.get("action") == "run_data":
        if not outcome.requires_data_run:
            return
        previous_sources = previous.get("source_ids")
        previous_source_ids = previous_sources if isinstance(previous_sources, list | tuple) else ()
        source_ids = {
            *outcome.source_ids,
            *(str(source_id) for source_id in previous_source_ids),
        }
        outcome = builder.builder_apply_outcome(
            "Workspace changes",
            source_ids=source_ids,
            requires_data_run=True,
        )
    st.session_state[BUILDER_LAST_OUTCOME_KEY] = asdict(outcome)


def _render_apply_notice() -> None:
    notice = st.session_state.pop("builder_apply_notice", None)
    if notice:
        st.success(str(notice), icon=":material/check_circle:")


def _technical_yaml(label: str, text: str) -> None:
    """Keep exact YAML and expression AST available without leading with it."""
    with st.expander(f"Technical details · {label}", expanded=False):
        st.code(text or "{}", language="yaml")


def _render_outcome_handoff() -> None:
    """Render the next valuable action after the most recent apply."""
    raw = st.session_state.get(BUILDER_LAST_OUTCOME_KEY)
    if not isinstance(raw, Mapping):
        st.info("Apply a configuration change to receive a report or data-refresh recommendation.")
        return
    label = str(raw.get("label", "Configuration") or "Configuration")
    action = str(raw.get("action", "open_report") or "open_report")
    message = str(raw.get("message", "") or "")
    if action == "run_data":
        st.warning(f"**Data refresh required · {label}**\n\n{message}")
        st.link_button(
            "Run data",
            "/data_load?from=builder",
            icon=":material/database_upload:",
            type="primary",
            width="stretch",
        )
        return
    st.success(f"**Report ready · {label}**\n\n{message}")
    st.link_button(
        "Open report",
        "/reports?from=builder",
        icon=":material/area_chart:",
        type="primary",
        width="stretch",
    )


def _dimension_pending_proposals() -> dict[str, dict[str, Any]]:
    raw = st.session_state.get(BUILDER_DIMENSION_PROPOSALS_KEY)
    return dict(raw) if isinstance(raw, dict) else {}


def _stage_dimension_proposal(
    key: str,
    processor_def: dict[str, Any],
    *,
    metric_defs: Mapping[str, dict[str, Any]] | None = None,
) -> None:
    proposals = _dimension_pending_proposals()
    proposals[key] = {
        "processor": processor_def,
        "metrics": dict(metric_defs or {}),
    }
    st.session_state[BUILDER_DIMENSION_PROPOSALS_KEY] = proposals


def _report_inventory_tab(catalog: model.Catalog) -> None:
    rows = _report_inventory_rows(catalog)
    if not rows:
        st.info("No report tiles configured yet.")
        return
    components.dataframe_with_search(
        rows,
        key="builder_report_inventory",
        search_columns=["Dashboard", "Page", "Report", "Metric", "Chart"],
        height=520,
        column_order=["Dashboard", "Page", "Report", "Metric", "Chart"],
    )
    if st.toggle(
        "Show technical IDs",
        value=False,
        key="builder_report_inventory_technical_ids",
    ):
        st.dataframe(
            _report_inventory_rows(catalog, technical=True),
            hide_index=True,
            width="stretch",
            height=360,
        )


def _report_inventory_rows(
    catalog: model.Catalog,
    *,
    technical: bool = False,
) -> list[dict[str, str]]:
    """Return searchable report inventory rows with human labels first."""
    rows: list[dict[str, str]] = []
    for dashboard in catalog.dashboards.dashboards:
        for page in dashboard.pages:
            for tile in page.tiles:
                if technical:
                    rows.append(
                        {
                            "Dashboard ID": dashboard.id,
                            "Page ID": page.id,
                            "Tile ID": tile.id,
                            "Metric ID": tile.metric,
                            "Chart kind": tile.chart,
                        }
                    )
                    continue
                metric = catalog.metrics.metrics.get(tile.metric)
                display_label = (
                    str(metric.display.label or "")
                    if metric is not None and metric.display is not None
                    else ""
                )
                rows.append(
                    {
                        "Dashboard": dashboard.title or humanize_identifier(dashboard.id),
                        "Page": page.title or humanize_identifier(page.id),
                        "Report": tile.title or humanize_identifier(tile.id),
                        "Metric": display_label or humanize_identifier(tile.metric),
                        "Chart": humanize_identifier(tile.chart),
                    }
                )
    return rows


def _render_immediate_config_warnings(ctx: ValueStreamContext) -> None:
    warnings = _funnel_stage_warnings(ctx.catalog)
    if not warnings:
        return
    with components.bordered_panel(
        "Processor Configuration Warnings",
        "These issues block valid aggregates and should be fixed before running sources.",
    ):
        for warning in warnings:
            st.warning(warning)


def _funnel_stage_warnings(catalog: model.Catalog) -> list[str]:
    warnings: list[str] = []
    for processor in catalog.processors.processors:
        if processor.kind != "funnel":
            continue
        stages = dict(processor.model_extra or {}).get("stages")
        if not isinstance(stages, list) or not stages:
            warnings.append(
                f"`{processor.id}` is a funnel processor but has no stages. "
                "Add at least one `stages` entry with a name and Boolean `when` expression."
            )
            continue
        missing = builder.stage_names_missing_when(stages)
        if missing:
            warnings.append(
                f"`{processor.id}` has funnel stage(s) without a `when` expression: "
                f"{', '.join(missing)}. Add a Boolean `when` expression to each stage."
            )
    return warnings


def _health(ctx: ValueStreamContext, save_slot: Any) -> None:
    _render_continue_primary(save_slot)
    components.key_value_strip(
        [{"label": key, "value": value} for key, value in catalog_counts(ctx).items()],
    )
    components.render_validation_summary(ctx.validation.issues, ok=ctx.validation.ok)

    with components.bordered_panel(
        "Review Progress", "Configuration areas to check before export."
    ):
        areas = [
            ("Sources", "ready" if ctx.catalog.pipelines.sources else "warning"),
            ("Processors", "ready" if ctx.catalog.processors.processors else "warning"),
            ("Metrics", "ready" if ctx.catalog.metrics.metrics else "warning"),
            ("Reports", "ready" if ctx.catalog.dashboards.dashboards else "warning"),
        ]
        cols = st.columns(len(areas))
        for col, (label, status) in zip(cols, areas, strict=True):
            with col:
                components.status_badge(label, status)


@st.fragment()
def _render_default_values_editor(
    rows_key: str,
    editor_key: str,
    field_options: list[str],
) -> None:
    picker_key = f"{editor_key}_field_picker"
    picker_col, action_col = st.columns([0.78, 0.22], vertical_alignment="bottom")
    selected_fields = picker_col.multiselect(
        "Add Field",
        field_options,
        accept_new_options=True,
        key=picker_key,
        placeholder="Select existing or type new",
        help=config_help.field_help("default.field"),
    )
    action_col.button(
        "Add",
        icon=":material/add:",
        disabled=not selected_fields,
        key=f"{picker_key}_add",
        on_click=components.add_default_fields_from_picker,
        args=(rows_key, picker_key, editor_key),
    )
    default_frame = builder.editor_frame(
        st.session_state.get(rows_key, []),
        ["Field", "Default Value", "Enabled"],
        builder.blank_default_row,
    )
    if default_frame.is_empty():
        st.info("No default values yet — choose a field above or add a row below.")
    edited_defaults = st.data_editor(
        default_frame,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=editor_key,
        column_config={
            "Field": st.column_config.TextColumn(
                "Field", width="medium", help=config_help.field_help("default.field")
            ),
            "Default Value": st.column_config.TextColumn(
                "Default Value",
                width="medium",
                help=config_help.field_help("default.value"),
            ),
            "Enabled": st.column_config.CheckboxColumn(
                "Enabled", width="small", help=config_help.field_help("row.enabled")
            ),
        },
    )
    st.session_state[rows_key] = builder.normalize_editor_rows(edited_defaults)


@st.fragment()
def _render_filter_rows_editor(
    rows_key: str,
    editor_key: str,
    filter_frame: Any,
    field_options: list[str],
    *,
    value_width: str = "large",
) -> None:
    if getattr(filter_frame, "is_empty", lambda: False)():
        st.info("No filters yet — add a row below when the source needs one.")
    edited_filters = st.data_editor(
        filter_frame,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=editor_key,
        column_config={
            "Field": st.column_config.SelectboxColumn(
                "Field",
                options=field_options,
                required=False,
                width="medium",
                help=config_help.field_help("filter.field"),
            ),
            "Operator": st.column_config.SelectboxColumn(
                "Operator",
                options=builder.FILTER_OPERATORS,
                required=False,
                width="small",
                help=config_help.field_help("filter.operator"),
            ),
            "Value": st.column_config.TextColumn(
                "Value", width=value_width, help=config_help.field_help("filter.value")
            ),
            "Enabled": st.column_config.CheckboxColumn(
                "Enabled", width="small", help=config_help.field_help("row.enabled")
            ),
        },
    )
    st.session_state[rows_key] = builder.normalize_editor_rows(edited_filters)
    try:
        compiled_filter = builder.compile_filter_rows(st.session_state[rows_key])
        _technical_yaml("Compiled filter AST", builder.expression_yaml(compiled_filter) or "{}")
    except Exception as exc:
        logger.exception("Failed to compile filter rows: editor_key=%s", editor_key)
        st.error(str(exc))


@st.fragment()
def _render_calculated_rows_editor(
    calc_key: str,
    editor_key: str,
    calculation_frame: Any,
) -> None:
    if getattr(calculation_frame, "is_empty", lambda: False)():
        st.info("No calculated fields yet — add a row below when you need one.")
    edited_calcs = st.data_editor(
        calculation_frame,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=editor_key,
        column_config={
            "Name": st.column_config.TextColumn(
                "Name", width="small", help=config_help.field_help("calculation.name")
            ),
            "Mode": st.column_config.SelectboxColumn(
                "Mode",
                options=builder.CALCULATION_MODES,
                width="medium",
                help=config_help.field_help("calculation.mode"),
            ),
            "Left": st.column_config.TextColumn(
                "Left", width="medium", help=config_help.field_help("calculation.left")
            ),
            "Right Kind": st.column_config.SelectboxColumn(
                "Right Kind",
                options=["Field", "Literal"],
                width="small",
                help=config_help.field_help("calculation.right_kind"),
            ),
            "Right": st.column_config.TextColumn(
                "Right", width="medium", help=config_help.field_help("calculation.right")
            ),
            "Expression": st.column_config.TextColumn(
                "Expression",
                width="large",
                help=config_help.field_help("calculation.expression"),
            ),
            "Enabled": st.column_config.CheckboxColumn(
                "Enabled", width="small", help=config_help.field_help("row.enabled")
            ),
        },
    )
    st.session_state[calc_key] = builder.normalize_editor_rows(edited_calcs)
    try:
        _technical_yaml(
            "Generated calculated transforms",
            yaml.safe_dump(
                {"transforms": builder.build_derive_column_transforms(st.session_state[calc_key])},
                sort_keys=False,
            ),
        )
    except Exception as exc:
        logger.exception("Failed to build calculated field transforms: editor_key=%s", editor_key)
        st.error(str(exc))


@st.fragment()
def _render_state_rows_editor(
    state_key: str,
    editor_key: str,
    state_frame: Any,
) -> None:
    edited_states = st.data_editor(
        state_frame,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=editor_key,
        column_config={
            "State": st.column_config.TextColumn(
                "State", width="medium", help=config_help.field_help("state.name")
            ),
            "Type": st.column_config.SelectboxColumn(
                "Type",
                options=builder.STATE_TYPES,
                width="medium",
                help=config_help.field_help("state.type"),
            ),
            "Source Column": st.column_config.TextColumn(
                "Source Column",
                width="medium",
                help=config_help.field_help("state.source_column"),
            ),
            "Derived From": st.column_config.TextColumn(
                "Derived From",
                width="large",
                disabled=True,
                help=config_help.field_help("state.derived_from"),
            ),
            "Enabled": st.column_config.CheckboxColumn(
                "Enabled", width="small", help=config_help.field_help("row.enabled")
            ),
        },
    )
    st.session_state[state_key] = builder.normalize_editor_rows(edited_states)


@st.dialog(
    "Delete source and dependencies",
    width="medium",
    icon=":material/delete_forever:",
    on_dismiss="rerun",
)
def _delete_source_dialog(ctx: ValueStreamContext, source_id: str) -> None:
    try:
        plan = builder.source_cascade_plan(ctx.catalog, source_id)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.warning(
        f"Source `{plan.source_id}` and every catalog definition that depends on it "
        "will be removed from the active workspace."
    )
    components.metric_cards(
        [
            {"label": "Processors", "value": len(plan.processor_ids)},
            {"label": "Metrics", "value": len(plan.metric_ids)},
            {"label": "Report tiles", "value": len(plan.tile_locations)},
            {"label": "Page filters", "value": len(plan.page_filter_locations)},
        ],
        columns=4,
    )
    with st.expander("Definitions to remove", expanded=True):
        st.markdown("**Processors**")
        st.code("\n".join(plan.processor_ids) or "None")
        st.markdown("**Metrics**")
        st.code("\n".join(plan.metric_ids) or "None")
        st.markdown("**Report tiles** (`dashboard/page/tile`)")
        st.code("\n".join(plan.tile_locations) or "None")
        if plan.page_filter_locations:
            st.markdown("**Unsupported page filters** (`dashboard/page/field`)")
            st.code("\n".join(plan.page_filter_locations))
    st.caption(
        "Dashboard and page containers are kept. Related Chat descriptions are removed. "
        "Aggregate Parquet files and run history are not deleted by this catalog action."
    )
    confirmed = st.checkbox(
        f"I understand that `{plan.source_id}` and the definitions listed above will be deleted.",
        key=f"builder_delete_source_confirm_{plan.source_id}",
        help="Confirm the catalog cascade before enabling the permanent delete action.",
    )
    if st.button(
        "Delete source and dependencies",
        type="primary",
        icon=":material/delete_forever:",
        disabled=not confirmed,
        width="stretch",
        key=f"builder_delete_source_confirm_action_{plan.source_id}",
    ):
        try:
            deleted = builder.delete_source_cascade(ctx.workspace, plan.source_id)
        except Exception as exc:  # pragma: no cover - Streamlit display path
            _record_builder_apply_failed(exc)
            logger.exception("Failed to delete source cascade: source=%s", plan.source_id)
            st.error(str(exc))
            return
        st.session_state["builder_source_delete_notice"] = (
            f"Deleted source `{deleted.source_id}`, {len(deleted.processor_ids)} processor(s), "
            f"{len(deleted.metric_ids)} metric(s), and {len(deleted.tile_locations)} tile(s)."
        )
        st.rerun()


@st.fragment()
def _source_builder(  # noqa: PLR0912, PLR0915
    ctx: ValueStreamContext,
    save_slot: Any,
    draft_slot: Any,
) -> None:
    _claim_fragment_action_slots(save_slot, draft_slot)
    delete_notice = st.session_state.pop("builder_source_delete_notice", None)
    if delete_notice:
        st.session_state.pop("builder_source_select", None)
        st.toast(str(delete_notice), icon=":material/delete_sweep:")
    if not ctx.catalog.pipelines.sources:
        st.info("No sources configured.")
        _render_continue_primary(save_slot)
        return

    source_col, delete_col = st.columns([8, 2], vertical_alignment="bottom")
    with source_col:
        source = st.selectbox(
            "Source",
            ctx.catalog.pipelines.sources,
            format_func=_source_choice_label,
            key="builder_source_select",
            help=config_help.field_help("source.selector"),
        )
    if delete_col.button(
        "Delete source",
        icon=":material/delete_forever:",
        width="stretch",
        key="builder_delete_source",
        help="Preview and remove this Source together with its dependent catalog definitions.",
    ):
        st.session_state[f"builder_delete_source_confirm_{source.id}"] = False
        _delete_source_dialog(ctx, source.id)
    rename_key = f"builder_source_rename_capitalize_{source.id}"
    if rename_key not in st.session_state:
        st.session_state[rename_key] = _source_has_transform(
            source,
            "rename_capitalize",
        )
    use_rename_capitalize = bool(st.session_state[rename_key])
    _sync_source_rename_capitalize_state(ctx, source, use_rename_capitalize)
    field_mapping = _source_rename_mapping(ctx, source, True) if use_rename_capitalize else {}
    field_options = _source_field_options(ctx, source, rename_capitalize=use_rename_capitalize)
    source_dict = builder.source_to_dict(source)
    reader_dict = dict(source_dict.get("reader", {}))

    with components.bordered_panel(
        "Runtime Settings", "Edit file loading, grouping, and source schema settings."
    ):
        source_id = st.text_input(
            "Source ID",
            value=source.id,
            key=f"builder_source_id_{source.id}",
            help=config_help.field_help("source.id"),
        )
        description = st.text_area(
            "Description",
            value=source.description,
            key=f"builder_source_desc_{source.id}",
            height=80,
            help=config_help.field_help("source.description"),
        )
        c1, c2, c3 = st.columns(3)
        reader_kind = c1.selectbox(
            "Reader",
            ["pega_ds_export", "parquet", "csv", "xlsx"],
            index=["pega_ds_export", "parquet", "csv", "xlsx"].index(source.reader.kind),
            key=f"builder_source_reader_{source.id}",
            help=config_help.field_help("source.reader"),
        )
        file_pattern = c2.text_input(
            "File Pattern",
            value=source.reader.file_pattern,
            key=f"builder_source_pattern_{source.id}",
            help=config_help.field_help("source.file_pattern"),
        )
        group_by_filename = c3.text_input(
            "Group Pattern",
            value=source.reader.group_by_filename or "",
            key=f"builder_source_group_{source.id}",
            help=config_help.field_help("source.group_pattern"),
        )
        c4, c5, c6 = st.columns(3)
        root = c4.text_input(
            "Root",
            value=str(reader_dict.get("root", reader_dict.get("base_dir", ""))),
            key=f"builder_source_root_{source.id}",
            help=config_help.field_help("source.root"),
        )
        streaming = c5.checkbox(
            "Streaming",
            value=bool(source.reader.streaming),
            key=f"builder_source_streaming_{source.id}",
            help=config_help.field_help("source.streaming"),
        )
        hive_partitioning = c6.checkbox(
            "Hive Partitioning",
            value=bool(reader_dict.get("hive_partitioning", False)),
            key=f"builder_source_hive_{source.id}",
            help=config_help.field_help("source.hive_partitioning"),
        )
        use_rename_capitalize = st.toggle(
            "Use Rename / Capitalize Transform",
            key=rename_key,
            help=config_help.field_help("source.rename_capitalize"),
        )
        if use_rename_capitalize:
            st.caption(
                "`rename_capitalize` converts source columns to the legacy Pega-aware "
                "capitalized schema, for example `pyName` to `Name`."
            )
        ts_key = f"builder_source_ts_{source.id}"
        ts_kwargs: dict[str, Any] = {"key": ts_key}
        if ts_key not in st.session_state:
            ts_kwargs["value"] = field_remap.remap_field_name(
                source.schema_.timestamp_column or "",
                field_mapping,
            )
        timestamp_column = st.text_input(
            "Timestamp Column",
            help=config_help.field_help("source.timestamp_column"),
            **ts_kwargs,
        )
        natural_key_key = f"builder_source_natural_{source.id}"
        natural_key_kwargs: dict[str, Any] = {
            "accept_new_options": True,
            "key": natural_key_key,
        }
        if natural_key_key not in st.session_state:
            natural_key_kwargs["default"] = [
                field
                for field in field_remap.remap_field_list(source.schema_.natural_key, field_mapping)
                if field in field_options
            ]
        natural_key = st.multiselect(
            "Natural Key",
            field_options,
            help=config_help.field_help("source.natural_key"),
            **natural_key_kwargs,
        )
        drop_columns_key = f"builder_source_drop_{source.id}"
        drop_columns_kwargs: dict[str, Any] = {
            "accept_new_options": True,
            "key": drop_columns_key,
        }
        if drop_columns_key not in st.session_state:
            drop_columns_kwargs["default"] = [
                field
                for field in field_remap.remap_field_list(
                    source.schema_.drop_columns, field_mapping
                )
                if field in field_options
            ]
        drop_columns = st.multiselect(
            "Drop Columns",
            field_options,
            help=config_help.field_help("source.drop_columns"),
            **drop_columns_kwargs,
        )

    defaults_key = f"builder_source_defaults_{source.id}"
    if defaults_key not in st.session_state:
        default_values = field_remap.remap_default_values(
            builder.source_defaults(source), field_mapping
        )
        st.session_state[defaults_key] = builder.default_rows_from_values(default_values)
    with components.bordered_panel(
        "Default Values", "Defaults run before source filters and derived fields."
    ):
        _render_default_values_editor(
            defaults_key,
            f"builder_source_defaults_editor_{source.id}",
            field_options,
        )

    filter_expression = builder.first_filter_expression(source)
    if filter_expression and field_mapping:
        filter_expression = field_remap.remap_expression_fields(filter_expression, field_mapping)
    filter_rows = builder.filter_rows_from_expression(filter_expression)
    filter_mode_key = f"builder_source_filter_mode_{source.id}"
    if filter_mode_key not in st.session_state:
        st.session_state[filter_mode_key] = "Rules" if filter_rows is not None else "Raw AST"
    with components.bordered_panel(
        "Source Filter", "Define dataset-level filters with rule rows or raw AST YAML."
    ):
        mode = st.segmented_control(
            "Filter Mode",
            ["Rules", "Raw AST"],
            default=st.session_state[filter_mode_key],
            key=f"{filter_mode_key}_control",
            help=config_help.field_help("source.filter_mode"),
        )
        st.session_state[filter_mode_key] = mode
        if mode == "Rules":
            rows_key = f"builder_source_filter_rows_{source.id}"
            if rows_key not in st.session_state:
                st.session_state[rows_key] = filter_rows or []
            filter_frame = builder.editor_frame(
                st.session_state[rows_key],
                ["Field", "Operator", "Value", "Enabled"],
                builder.blank_filter_row,
            )
            _render_filter_rows_editor(
                rows_key,
                f"builder_source_filter_editor_{source.id}",
                filter_frame,
                field_options,
            )
            try:
                compiled_filter = builder.compile_filter_rows(st.session_state[rows_key])
            except Exception:
                logger.exception("Failed to compile source filter rows: source=%s", source.id)
                compiled_filter = None
        else:
            raw_key = f"builder_source_raw_filter_{source.id}"
            raw_default = builder.expression_yaml(filter_expression)
            raw_filter = st.text_area(
                "Filter AST YAML",
                value=st.session_state.setdefault(raw_key, raw_default),
                height=220,
                key=f"{raw_key}_editor",
                help=config_help.field_help("source.filter_ast"),
            )
            st.session_state[raw_key] = raw_filter
            compiled_filter = (
                builder.parse_expression_yaml(raw_filter) if raw_filter.strip() else None
            )

    with components.bordered_panel(
        "Calculated Fields",
        "Create `derive_column` transforms with builder rows, AST YAML, or Polars.",
    ):
        calc_key = f"builder_source_calcs_{source.id}"
        if calc_key not in st.session_state:
            st.session_state[calc_key] = field_remap.remap_calculation_row_values(
                builder.calculated_rows_from_source(source),
                field_mapping,
            )
        calc_rows = st.session_state[calc_key]
        calc_rows = builder.calculated_rows_for_editor(calc_rows)
        st.session_state[calc_key] = calc_rows
        calculation_frame = builder.editor_frame(
            calc_rows,
            ["Name", "Mode", "Left", "Right Kind", "Right", "Expression", "Enabled"],
            builder.blank_calculated_row,
        )
        with st.popover("Examples", icon=":material/flare:"):
            st.code(
                "Name: Margin\nMode: Subtract\nLeft: Revenue\nRight Kind: Field\nRight: Cost",
                language="yaml",
            )
            st.code(
                "op: date_diff\nunit: seconds\nend: {col: OutcomeTime}\nstart: {col: DecisionTime}",
                language="yaml",
            )
            st.code(
                'pl.col("Revenue") - pl.col("Cost")',
                language="python",
            )
        _render_calculated_rows_editor(
            calc_key,
            f"builder_source_calcs_editor_{source.id}",
            calculation_frame,
        )
        calculated_rows_valid = True
        try:
            builder.build_derive_column_transforms(st.session_state[calc_key])
        except Exception:
            logger.exception("Failed to validate source calculated rows: source=%s", source.id)
            calculated_rows_valid = False

    source_def = _build_source_definition(
        source=source,
        source_id=source_id.strip(),
        description=description.strip(),
        reader_kind=reader_kind,
        file_pattern=file_pattern.strip(),
        group_by_filename=group_by_filename.strip() or None,
        root=root.strip(),
        streaming=streaming,
        hive_partitioning=hive_partitioning,
        timestamp_column=timestamp_column.strip() or None,
        natural_key=natural_key,
        drop_columns=drop_columns,
        default_rows=st.session_state[defaults_key],
        use_rename_capitalize=use_rename_capitalize,
        filter_expression=compiled_filter,
        calculated_rows=st.session_state[calc_key] if calculated_rows_valid else [],
    )
    with components.bordered_panel(
        "Generated configuration",
    ):
        _technical_yaml(
            "Generated source transforms",
            yaml.safe_dump(
                {"transforms": source_def.get("transforms", [])},
                sort_keys=False,
            ),
        )
        _technical_yaml(
            "Generated source YAML",
            yaml.safe_dump({"sources": [source_def]}, sort_keys=False),
        )
    draft_status = builder.builder_draft_status(
        f"source:{source.id}",
        builder.source_to_dict(source),
        source_def,
    )
    if _render_editor_primary_action(
        save_slot=save_slot,
        draft_slot=draft_slot,
        status=draft_status,
        valid=bool(source_id.strip()) and calculated_rows_valid,
        widget_prefixes=("builder_source_",),
        help_text=f"Validate and apply source {source.id!r} to the active workspace.",
    ):
        try:
            with builder.validated_catalog_transaction(ctx.workspace):
                builder.write_source_definition(ctx.workspace, source_def)
            _complete_builder_apply(
                status=draft_status,
                scope="source",
                label=description.strip() or humanize_identifier(source_id.strip()),
                source_ids=[source_id.strip()],
                message="Source applied to the workspace.",
            )
        except Exception as exc:  # pragma: no cover - Streamlit display path
            _record_builder_apply_failed(exc)
            logger.exception("Failed to write source definition: source=%s", source_id.strip())
            st.error(str(exc))


@st.fragment()
def _dimensions_builder(  # noqa: PLR0912, PLR0915
    ctx: ValueStreamContext, save_slot: Any, draft_slot: Any
) -> None:
    _claim_fragment_action_slots(save_slot, draft_slot)
    with components.bordered_panel(
        "Dimension Coverage", "Review and update processor group-by fields."
    ):
        rows = []
        for processor in ctx.catalog.processors.processors:
            rows.append(
                {
                    "Processor": processor.id,
                    "Source": processor.source,
                    "Kind": processor.kind,
                    "Group By": ", ".join(processor.group_by),
                    "Grains": ", ".join(processor.grains),
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch", height=280)

    if not ctx.catalog.processors.processors:
        _render_continue_primary(save_slot)
        return
    processor = st.selectbox(
        "Processor To Edit",
        ctx.catalog.processors.processors,
        format_func=_processor_choice_label_human,
        key="builder_dimension_processor",
        help=config_help.field_help("dimension.processor"),
    )
    source = next(
        (source for source in ctx.catalog.pipelines.sources if source.id == processor.source), None
    )
    options = _source_field_options(ctx, source) if source else list(processor.group_by)
    field_mapping = _source_rename_mapping(ctx, source, True) if source else {}
    group_key = f"builder_dimension_group_by_{processor.id}"
    if group_key not in st.session_state:
        st.session_state[group_key] = [
            field
            for field in field_remap.remap_field_list(processor.group_by, field_mapping)
            if field in options
        ]
    profile_sample: pl.DataFrame | None = None
    profile_rows: list[dimension_profile.DimensionProfileRow] = []
    if source is not None:
        profile_sample, profile_rows = _dimension_profile_panel(ctx, source, processor)
        _dimension_pack_panel(processor, options, group_key)
        _dimension_promotion_panel(
            ctx,
            source,
            processor,
            options,
            group_key,
            profile_sample,
            profile_rows,
        )
    selected = st.multiselect(
        "Group-By Dimensions",
        options,
        accept_new_options=True,
        key=group_key,
        help=config_help.field_help("dimension.group_by"),
    )
    processor_def = builder.processor_to_dict(processor)
    processor_def["dimensions"] = selected
    processor_def.pop("group_by", None)
    pending_proposals = _dimension_pending_proposals()
    if pending_proposals:
        st.info(
            f"{len(pending_proposals)} additional exploration proposal"
            f"{'s are' if len(pending_proposals) != 1 else ' is'} included in this draft."
        )
    _technical_yaml(
        "Generated dimension YAML",
        yaml.safe_dump({"processors": [processor_def]}, sort_keys=False),
    )
    draft_status = builder.builder_draft_status(
        f"dimensions:{processor.id}",
        {"processor": builder.processor_to_dict(processor), "proposals": {}},
        {"processor": processor_def, "proposals": pending_proposals},
    )
    if _render_editor_primary_action(
        save_slot=save_slot,
        draft_slot=draft_slot,
        status=draft_status,
        valid=bool(processor.id),
        widget_prefixes=("builder_dimension_", "builder_exploration_", "builder_sketch_"),
        help_text=(
            "Validate and apply the selected dimensions and staged exploration definitions. "
            "Data is not run from this action."
        ),
    ):
        try:
            with builder.validated_catalog_transaction(ctx.workspace):
                builder.write_processor_definition(ctx.workspace, processor_def)
                for proposal in pending_proposals.values():
                    staged_processor = proposal.get("processor")
                    if isinstance(staged_processor, dict):
                        builder.write_processor_definition(ctx.workspace, staged_processor)
                    staged_metrics = proposal.get("metrics")
                    if isinstance(staged_metrics, Mapping):
                        for metric_name, metric_def in staged_metrics.items():
                            if isinstance(metric_def, dict):
                                builder.write_metric_definition(
                                    ctx.workspace, str(metric_name), metric_def
                                )
            source_ids = [processor.source]
            source_ids.extend(
                str(proposal.get("processor", {}).get("source", ""))
                for proposal in pending_proposals.values()
                if isinstance(proposal.get("processor"), Mapping)
            )
            st.session_state.pop(BUILDER_DIMENSION_PROPOSALS_KEY, None)
            _complete_builder_apply(
                status=draft_status,
                scope="dimensions",
                label=processor.description or humanize_identifier(processor.id),
                source_ids=source_ids,
                message="Dimension draft applied. Use Data Load to refresh affected aggregates.",
            )
        except Exception as exc:  # pragma: no cover - Streamlit display path
            _record_builder_apply_failed(exc)
            logger.exception("Failed to write processor dimensions: processor=%s", processor.id)
            st.error(str(exc))
    if source is not None:
        _temporary_exploration_panel(
            ctx,
            source,
            processor,
            options,
            selected,
            profile_sample,
            profile_rows,
        )
        _sketch_exploration_panel(
            ctx,
            source,
            processor,
            options,
            selected,
            profile_rows,
        )


def _dimension_profile_panel(
    ctx: ValueStreamContext,
    source: model.Source,
    processor: model.Processor,
) -> tuple[pl.DataFrame | None, list[dimension_profile.DimensionProfileRow]]:
    with components.bordered_panel(
        "Dimension Profiler",
        "Profile source fields before promoting them into aggregate dimensions.",
    ):
        sample = dimension_profile.source_profile_sample(ctx, source)
        if sample is None or sample.is_empty():
            st.info(
                "No source sample is available. Add source files or use Data Load before profiling."
            )
            return None, []
        rows = dimension_profile.source_dimension_profile_rows(ctx, source, sample)
        if not rows:
            st.info("No fields found in the sampled source.")
            return sample, []
        profile_frame = dimension_profile.profile_frame(rows).rename(
            {"Current Usage": "Current Processors"}
        )
        components.key_value_strip(
            [
                {"label": "Profiled fields", "value": len(rows)},
                {
                    "label": "Recommended",
                    "value": sum(row.recommendation == "Recommended" for row in rows),
                },
                {
                    "label": "Needs review",
                    "value": sum(row.recommendation == "Review" for row in rows),
                },
                {
                    "label": "Already active",
                    "value": sum(row.recommendation == "Active" for row in rows),
                },
            ]
        )
        filter_choice = st.segmented_control(
            "Profile Filter",
            ["All", "Recommended", "Review", "Avoid", "Active"],
            default="All",
            key=f"builder_dimension_profile_filter_{source.id}",
            help=config_help.field_help("dimension.profile_filter"),
        )
        filtered = profile_frame
        if filter_choice and filter_choice != "All":
            filtered = filtered.filter(pl.col("Recommendation") == filter_choice)
        st.dataframe(filtered, hide_index=True, width="stretch", height=360)
        recommended = dimension_profile.recommended_fields(
            rows,
            existing_fields=processor.group_by,
        )
        if recommended:
            st.caption(
                "Recommended fields are low-cardinality, non-identity fields in the transformed sample. "
                "Select them in Group-By Dimensions below, then apply and re-run the source."
            )
            if st.button(
                "Add recommended to selection",
                icon=":material/add:",
                key=f"builder_dimension_add_recommended_{processor.id}",
            ):
                key = f"builder_dimension_group_by_{processor.id}"
                current = [
                    str(value)
                    for value in st.session_state.get(key, list(processor.group_by))
                    if str(value)
                ]
                st.session_state[key] = builder.dedupe([*current, *recommended])
                st.rerun()
    return sample, rows


def _dimension_pack_panel(
    processor: model.Processor,
    options: list[str],
    group_key: str,
) -> None:
    with components.bordered_panel(
        "Default Dimension Packs",
        "Apply common industry dimensions without editing YAML.",
    ):
        pack_names = dimension_profile.dimension_pack_names()
        if not pack_names:
            st.info("No dimension packs are available.")
            return
        pack_name = st.selectbox(
            "Dimension Pack",
            pack_names,
            key=f"builder_dimension_pack_{processor.id}",
            help=config_help.field_help("dimension.pack"),
        )
        pack_fields = dimension_profile.dimension_pack_fields(options, pack_name)
        current = builder.dedupe(
            [str(value) for value in st.session_state.get(group_key, processor.group_by)]
        )
        missing = [
            field
            for field in dimension_profile.DIMENSION_PACKS.get(pack_name, ())
            if field.casefold() not in {option.casefold() for option in options}
        ]
        st.write(
            {
                "Available pack fields": ", ".join(pack_fields) or "None",
                "Already selected": ", ".join([field for field in pack_fields if field in current])
                or "None",
                "Missing from source": ", ".join(missing) or "None",
            }
        )
        if st.button(
            f"Add {pack_name} dimensions",
            icon=":material/library_add:",
            disabled=not pack_fields,
            key=f"builder_dimension_apply_pack_{processor.id}",
        ):
            st.session_state[group_key] = builder.dedupe([*current, *pack_fields])
            st.rerun()


def _dimension_promotion_panel(
    ctx: ValueStreamContext,
    source: model.Source,
    processor: model.Processor,
    options: list[str],
    group_key: str,
    sample: pl.DataFrame | None,
    rows: list[dimension_profile.DimensionProfileRow],
) -> None:
    with components.bordered_panel(
        "One-Click Dimension Promotion",
        "Add a source field as a filterable dimension, preview aggregate growth, then backfill.",
    ):
        current = builder.dedupe(
            [str(value) for value in st.session_state.get(group_key, processor.group_by)]
        )
        profile_by_field = {row.field: row for row in rows}
        candidates = _dimension_promotion_candidates(options, current, rows)
        if not candidates:
            st.info("Every profiled field is already selected for this processor.")
            return
        default_index = (
            0
            if profile_by_field[candidates[0]].recommendation in {"Recommended", "Review"}
            else None
        )
        field = st.selectbox(
            "Field To Promote",
            candidates,
            index=default_index,
            placeholder="Choose a reviewed field",
            format_func=lambda value: f"Add {value} as a filterable dimension",
            key=f"builder_dimension_promote_field_{processor.id}",
            help=config_help.field_help("dimension.promote_field"),
        )
        if field is None:
            st.info(
                "Only fields marked Avoid remain. Choose one explicitly after reviewing its risk."
            )
            return
        row = profile_by_field[field]
        st.write(
            {
                "Recommendation": row.recommendation,
                "Safe For Group-By": row.safe_for_group_by,
                "Cardinality": row.cardinality,
                "Null %": round(row.null_rate * 100, 1),
                "Reason": row.reason,
            }
        )
        if row.safe_for_group_by == "No":
            st.warning(
                "This field is flagged before promotion because it can explode aggregates or expose sensitive detail."
            )
        elif row.safe_for_group_by == "Review":
            st.info(
                "Review this field before promotion; aggregate growth or privacy risk may be material."
            )
        if sample is not None and not sample.is_empty():
            preview = dimension_profile.aggregate_size_preview(sample, current, [field])
            components.key_value_strip(
                [
                    {"label": "Current groups", "value": preview.current_rows},
                    {"label": "Projected groups", "value": preview.projected_rows},
                    {"label": "Added groups", "value": preview.added_rows},
                    {"label": "Expansion", "value": f"{preview.expansion_factor:.1f}x"},
                ]
            )
        next_dimensions = builder.dedupe([*current, field])
        if st.button(
            f"Add {field} as filterable dimension",
            icon=":material/add:",
            key=f"builder_dimension_promote_add_{processor.id}",
        ):
            st.session_state[group_key] = next_dimensions
            st.rerun()


def _dimension_promotion_candidates(
    options: list[str],
    current: list[str],
    rows: list[dimension_profile.DimensionProfileRow],
) -> list[str]:
    """Rank safe dimension candidates first without preselecting Avoid fields."""
    profile_by_field = {row.field: row for row in rows}
    rank = {"Recommended": 0, "Review": 1, "Avoid": 2, "Active": 3}
    candidates = [
        field
        for field in options
        if field not in current and profile_by_field.get(field) is not None
    ]
    return sorted(
        candidates,
        key=lambda field: (
            rank.get(profile_by_field[field].recommendation, 4),
            field.casefold(),
            field,
        ),
    )


def _temporary_exploration_panel(
    ctx: ValueStreamContext,
    source: model.Source,
    processor: model.Processor,
    options: list[str],
    selected_dimensions: list[str],
    sample: pl.DataFrame | None,
    rows: list[dimension_profile.DimensionProfileRow],
) -> None:
    with components.bordered_panel(
        "Temporary Exploration Aggregates",
        "Create time-limited processors for exploratory breakdowns before promoting them.",
    ):
        dim_key = f"builder_exploration_dims_{processor.id}"
        if dim_key not in st.session_state:
            defaults = (
                dimension_profile.dimension_pack_fields(options)[:3] or selected_dimensions[:3]
            )
            st.session_state[dim_key] = [field for field in defaults if field in options]
        explore_dims = st.multiselect(
            "Exploration Dimensions",
            options,
            accept_new_options=True,
            key=dim_key,
            help=config_help.field_help("dimension.exploration_dimensions"),
        )
        window_days = st.number_input(
            "Source Window Days",
            min_value=1,
            max_value=730,
            value=30,
            step=1,
            key=f"builder_exploration_window_{processor.id}",
            help=config_help.field_help("dimension.window_days"),
        )
        ttl_days = st.number_input(
            "Expire After Days",
            min_value=1,
            max_value=90,
            value=14,
            step=1,
            key=f"builder_exploration_ttl_{processor.id}",
            help=config_help.field_help("dimension.ttl_days"),
        )
        if sample is not None and not sample.is_empty() and explore_dims:
            preview = dimension_profile.aggregate_size_preview(sample, [], explore_dims)
            components.key_value_strip(
                [
                    {"label": "Exploration groups", "value": preview.projected_rows},
                    {"label": "Sample rows", "value": preview.sample_rows},
                    {"label": "Dimensions", "value": len(explore_dims)},
                ]
            )
        if rows:
            unsafe = [
                row.field
                for row in rows
                if row.field in explore_dims and row.safe_for_group_by == "No"
            ]
            if unsafe:
                st.warning("High-risk exploration dimensions: " + ", ".join(unsafe))
        processor_def = _temporary_processor_def(
            processor,
            source,
            explore_dims,
            ttl_days=int(ttl_days),
            window_days=int(window_days),
            sample=sample,
        )
        _technical_yaml(
            "Exploration processor YAML",
            yaml.safe_dump({"processors": [processor_def]}, sort_keys=False),
        )
        if st.button(
            "Add exploration to draft",
            icon=":material/add:",
            disabled=not explore_dims,
            key=f"builder_exploration_create_{processor.id}",
        ):
            _stage_dimension_proposal(str(processor_def["id"]), processor_def)
            st.toast("Exploration added to this step's draft.", icon=":material/draft:")
            st.rerun()
        _exploration_lifecycle_controls(ctx, source.id)


def _sketch_exploration_panel(
    ctx: ValueStreamContext,
    source: model.Source,
    processor: model.Processor,
    options: list[str],
    selected_dimensions: list[str],
    rows: list[dimension_profile.DimensionProfileRow],
) -> None:
    with components.bordered_panel(
        "Top-K And Sketch Exploration",
        "Answer high-cardinality questions with sketches instead of exploding dimensions.",
    ):
        if not options:
            st.info("No source fields are available for sketch exploration.")
            return
        recommendations = dimension_profile.sketch_recommendations(rows)
        if recommendations:
            st.dataframe(recommendations, hide_index=True, width="stretch", height=180)
        profile_by_field = {row.field: row for row in rows}
        topk_options = sorted(
            builder.dedupe(options),
            key=lambda field: (
                profile_by_field.get(field) is None
                or profile_by_field[field].recommendation == "Avoid",
                field.casefold(),
                field,
            ),
        )
        safe_topk_options = [
            field
            for field in topk_options
            if field in profile_by_field
            and profile_by_field[field].recommendation in {"Recommended", "Review"}
        ]
        default_topk = _default_topk_field(safe_topk_options) if safe_topk_options else None
        entity_default = _default_entity_field(options, source)
        include_topk = st.checkbox(
            "Top frequent values",
            value=False,
            key=f"builder_sketch_include_topk_{processor.id}",
            help=config_help.field_help("dimension.topk_enabled"),
        )
        topk_field = st.selectbox(
            "Top-K Field",
            topk_options,
            index=builder.option_index(topk_options, default_topk) if default_topk else None,
            placeholder="Select a field explicitly",
            disabled=not include_topk,
            key=f"builder_sketch_topk_field_{processor.id}",
            help=config_help.field_help("dimension.topk_field"),
        )
        include_cpc = st.checkbox(
            "Unique entity count",
            value=True,
            key=f"builder_sketch_include_cpc_{processor.id}",
            help=config_help.field_help("dimension.cpc_enabled"),
        )
        include_theta = st.checkbox(
            "Theta set state for overlap",
            value=True,
            key=f"builder_sketch_include_theta_{processor.id}",
            help=config_help.field_help("dimension.theta_enabled"),
        )
        entity_field = st.selectbox(
            "Entity Field",
            options,
            index=builder.option_index(options, entity_default),
            key=f"builder_sketch_entity_field_{processor.id}",
            help=config_help.field_help("dimension.entity_field"),
        )
        dim_key = f"builder_sketch_dims_{processor.id}"
        if dim_key not in st.session_state:
            st.session_state[dim_key] = [
                field for field in selected_dimensions[:2] if field in options
            ]
        sketch_dims = st.multiselect(
            "Sketch Grouping Dimensions",
            options,
            accept_new_options=True,
            key=dim_key,
            help=config_help.field_help("dimension.sketch_group_by"),
        )
        processor_def, metric_defs = _sketch_processor_and_metrics(
            source,
            base_processor=processor,
            dimensions=sketch_dims,
            topk_field=str(topk_field or "") if include_topk else "",
            entity_field=entity_field,
            include_cpc=include_cpc,
            include_theta=include_theta,
        )
        _technical_yaml(
            "Sketch exploration YAML",
            yaml.safe_dump(
                {
                    "processors": [processor_def],
                    "metrics": metric_defs,
                },
                sort_keys=False,
            ),
        )
        if st.button(
            "Add sketch exploration to draft",
            icon=":material/add:",
            disabled=not processor_def.get("states"),
            key=f"builder_sketch_create_{processor.id}",
        ):
            _stage_dimension_proposal(
                str(processor_def["id"]),
                processor_def,
                metric_defs=metric_defs,
            )
            st.toast("Sketch exploration added to this step's draft.", icon=":material/draft:")
            st.rerun()


def _processor_def_with_dimensions(
    processor: model.Processor,
    dimensions: list[str],
) -> dict[str, Any]:
    processor_def = builder.processor_to_dict(processor)
    processor_def["dimensions"] = builder.dedupe(dimensions)
    processor_def.pop("group_by", None)
    return processor_def


def _temporary_processor_def(
    processor: model.Processor,
    source: model.Source,
    dimensions: list[str],
    *,
    ttl_days: int,
    window_days: int,
    sample: pl.DataFrame | None,
) -> dict[str, Any]:
    now = _utc_now()
    processor_def = _processor_def_with_dimensions(processor, dimensions)
    processor_def["id"] = _exploration_id("explore", processor.id, dimensions)
    processor_def["description"] = (
        f"Temporary exploration aggregate for {processor.id}: {', '.join(dimensions)}"
    )
    processor_def["exploration"] = {
        "temporary": True,
        "base_processor": processor.id,
        "created_at": now.isoformat(),
        "expires_at": (now + dt.timedelta(days=ttl_days)).isoformat(),
        "ttl_days": ttl_days,
        "time_window_days": window_days,
        "promoted": False,
    }
    time_filter = _time_window_filter(source, sample, window_days)
    if time_filter is not None:
        processor_def["filter"] = _combine_filters(processor_def.get("filter"), time_filter)
    return processor_def


def _sketch_processor_and_metrics(
    source: model.Source,
    *,
    base_processor: model.Processor,
    dimensions: list[str],
    topk_field: str,
    entity_field: str,
    include_cpc: bool,
    include_theta: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    now = _utc_now()
    fields = [field for field in [topk_field, entity_field] if field]
    processor_id = _exploration_id("explore_sketch", base_processor.id, fields)
    states: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    if topk_field:
        state_name = _state_name("Top", topk_field, "topk")
        states[state_name] = {
            "type": "topk",
            "source_column": topk_field,
            "lg_max_map_size": 12,
        }
        metrics[f"{processor_id}_topk"] = builder.build_topk_items_metric(
            processor_id,
            state_name,
            limit=10,
        )
    if include_cpc and entity_field:
        state_name = _state_name("Unique", entity_field, "cpc")
        states[state_name] = {"type": "cpc", "source_column": entity_field, "lg_k": 11}
        metrics[f"{processor_id}_unique"] = builder.build_approx_distinct_metric(
            processor_id,
            state_name,
        )
    if include_theta and entity_field:
        state_name = _state_name("Audience", entity_field, "theta")
        states[state_name] = {"type": "theta", "source_column": entity_field, "lg_k": 12}
    return (
        {
            "id": processor_id,
            "source": source.id,
            "kind": "entity_set",
            "description": f"Sketch exploration states for {base_processor.id}.",
            "dimensions": builder.dedupe(dimensions),
            "entity": entity_field,
            "states": states,
            "exploration": {
                "temporary": True,
                "base_processor": base_processor.id,
                "created_at": now.isoformat(),
                "expires_at": (now + dt.timedelta(days=14)).isoformat(),
                "ttl_days": 14,
                "sketch": True,
                "promoted": False,
            },
        },
        metrics,
    )


def _exploration_lifecycle_controls(ctx: ValueStreamContext, source_id: str) -> None:
    exploration_processors = [
        processor
        for processor in ctx.catalog.processors.processors
        if processor.source == source_id and _exploration_meta(processor)
    ]
    if not exploration_processors:
        return
    st.write("#### Existing Explorations")
    st.dataframe(
        [
            {
                "Processor": processor.id,
                "Base": _exploration_meta(processor).get("base_processor", ""),
                "Dimensions": ", ".join(processor.group_by),
                "Status": _exploration_status(processor),
                "Expires": _exploration_meta(processor).get("expires_at", ""),
            }
            for processor in exploration_processors
        ],
        hide_index=True,
        width="stretch",
        height=180,
    )
    selected = st.selectbox(
        "Exploration To Promote",
        exploration_processors,
        format_func=lambda item: f"{item.id} ({_exploration_status(item)})",
        key=f"builder_exploration_promote_select_{source_id}",
        help=config_help.field_help("dimension.exploration_selector"),
    )
    if st.button(
        "Add promotion to draft",
        icon=":material/publish:",
        key=f"builder_exploration_promote_{source_id}",
    ):
        processor_def = builder.processor_to_dict(selected)
        metadata = dict(processor_def.get("exploration", {}))
        metadata["temporary"] = False
        metadata["promoted"] = True
        metadata["promoted_at"] = _utc_now().isoformat()
        metadata.pop("expires_at", None)
        processor_def["exploration"] = metadata
        _stage_dimension_proposal(f"promote:{selected.id}", processor_def)
        st.toast("Promotion added to this step's draft.", icon=":material/draft:")
        st.rerun()


def _exploration_meta(processor: model.Processor) -> dict[str, Any]:
    extra = dict(processor.model_extra or {})
    metadata = extra.get("exploration")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _exploration_status(processor: model.Processor) -> str:
    metadata = _exploration_meta(processor)
    if metadata.get("promoted"):
        return "Promoted"
    expires_at = _parse_datetime(str(metadata.get("expires_at", "") or ""))
    if expires_at is not None and expires_at < _utc_now():
        return "Expired"
    return "Temporary"


def _time_window_filter(
    source: model.Source,
    sample: pl.DataFrame | None,
    window_days: int,
) -> dict[str, Any] | None:
    column = _time_window_column(source, sample)
    if not column:
        return None
    cutoff = (_utc_now().date() - dt.timedelta(days=window_days)).isoformat()
    return {
        "polars": f"pl.col({column!r}).cast(pl.String) >= pl.lit({cutoff!r})",
    }


def _time_window_column(source: model.Source, sample: pl.DataFrame | None) -> str:
    columns = list(sample.columns) if sample is not None else []
    candidates = [
        source.schema_.timestamp_column,
        "Day",
        "Date",
        "DecisionTime",
        "OutcomeTime",
        "Timestamp",
    ]
    for column in candidates:
        if column and column in columns:
            return str(column)
    for column in columns:
        if dimension_profile.looks_like_measure_or_time_field(column):
            return column
    return ""


def _combine_filters(
    existing: Any,
    added: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(existing, dict) and existing:
        return {"op": "and", "args": [existing, added]}
    return added


def _default_entity_field(options: list[str], source: model.Source) -> str:
    for field in source.schema_.natural_key:
        if field in options:
            return str(field)
    for field in options:
        compact = field.casefold().replace("_", "").replace("-", "").replace(" ", "")
        if compact in {"customerid", "subjectid", "accountid", "entityid"}:
            return field
    for field in options:
        if dimension_profile.looks_like_identity_field(field):
            return field
    return options[0] if options else ""


def _default_topk_field(options: list[str]) -> str:
    preferred = ("Campaign", "CampaignName", "Issue", "Channel", "Treatment")
    for field in preferred:
        for option in options:
            if option.casefold() == field.casefold():
                return option
    return options[0] if options else ""


def _exploration_id(prefix: str, base: str, fields: list[str]) -> str:
    parts = [_slug_token(prefix), _slug_token(base), *[_slug_token(field) for field in fields]]
    stem = "_".join(part for part in parts if part)
    timestamp = _utc_now().strftime("%Y%m%d%H%M%S")
    return f"{stem[:72]}_{timestamp}"


def _state_name(prefix: str, field: str, suffix: str) -> str:
    token = _slug_token(field).title().replace("_", "")
    return f"{prefix}{token}_{suffix}"


def _slug_token(value: str) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    return "_".join("".join(chars).split("_")).strip("_")


def _parse_datetime(value: str) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@st.fragment()
def _processor_builder(  # noqa: PLR0912, PLR0915
    ctx: ValueStreamContext, save_slot: Any, draft_slot: Any
) -> None:
    _claim_fragment_action_slots(save_slot, draft_slot)
    processors = ctx.catalog.processors.processors
    if not ctx.catalog.pipelines.sources:
        st.info("No sources configured.")
        _render_continue_primary(save_slot)
        return

    mode_options = ["Create New Processor"]
    if processors:
        mode_options.append("Edit Existing Processor")
    mode = (
        st.segmented_control(
            "Processor Mode",
            mode_options,
            default=mode_options[-1],
            selection_mode="single",
            key="builder_processor_mode",
            help=config_help.field_help("processor.mode"),
        )
        or mode_options[-1]
    )
    creating = mode == "Create New Processor"
    if creating:
        processor = _new_processor_template(ctx)
    else:
        processor = st.selectbox(
            "Processor",
            processors,
            format_func=_processor_choice_label_human,
            key="builder_processor_select",
            help=config_help.field_help("processor.selector"),
        )
    source = next(
        (
            candidate
            for candidate in ctx.catalog.pipelines.sources
            if candidate.id == processor.source
        ),
        None,
    )
    field_options = _source_field_options(ctx, source) if source else list(processor.group_by)
    field_mapping = _source_rename_mapping(ctx, source, True) if source else {}
    processor_def = builder.processor_to_dict(processor)

    with components.bordered_panel(
        "Processor Editor", "Configure the processor identity, source binding, and grain shape."
    ):
        identity_col, source_col, kind_col, description_col = st.columns(
            [1, 1, 1, 1],
            gap="xsmall",
            vertical_alignment="bottom",
        )
        processor_id = identity_col.text_input(
            "Processor ID",
            value=processor.id,
            key=f"builder_proc_id_{processor.id}",
            help=config_help.field_help("processor.id"),
        )
        source_ids = [source.id for source in ctx.catalog.pipelines.sources]
        source_id = source_col.selectbox(
            "Source",
            source_ids,
            index=source_ids.index(processor.source),
            key=f"builder_proc_source_{processor.id}",
            help=config_help.field_help("processor.source"),
        )
        kind = kind_col.selectbox(
            "Kind",
            PROCESSOR_KINDS,
            index=PROCESSOR_KINDS.index(processor.kind),
            key=f"builder_proc_kind_{processor.id}",
            help=config_help.field_help("processor.kind"),
        )
        description = description_col.text_input(
            "Description",
            value=processor.description,
            key=f"builder_proc_desc_{processor.id}",
            help=config_help.field_help("processor.description"),
        )

        group_by = st.multiselect(
            "Group By",
            field_options,
            default=[
                field
                for field in field_remap.remap_field_list(processor.group_by, field_mapping)
                if field in field_options
            ],
            accept_new_options=True,
            key=f"builder_proc_group_{processor.id}",
            help=config_help.field_help("processor.group_by"),
        )
        time_col, grain_col = st.columns(2)
        time_column = time_col.text_input(
            "Time Column",
            value=field_remap.remap_field_name(
                processor.time.column if processor.time and processor.time.column else "",
                field_mapping,
            ),
            key=f"builder_proc_time_{processor.id}",
            help=config_help.field_help("processor.time_column"),
        )
        grains = grain_col.multiselect(
            "Grains",
            list(forms.PROCESSOR_GRAIN_OPTIONS),
            default=[
                builder.display_grain(grain)
                for grain in processor.grains
                if builder.display_grain(grain) in forms.PROCESSOR_GRAIN_OPTIONS
            ],
            key=f"builder_proc_grains_{processor.id}",
            help=config_help.field_help("processor.grains"),
        )
        kind_settings = _processor_kind_settings(
            processor,
            kind,
            field_options,
            field_mapping,
        )

    state_key = f"builder_proc_states_{processor.id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = _state_rows(processor, field_mapping)
    state_frame = builder.editor_frame(
        st.session_state[state_key],
        ["State", "Type", "Source Column", "Derived From", "Enabled"],
        _blank_state_row,
    )
    with components.bordered_panel(
        "Derived Outputs", "Edit aggregate states written by this processor."
    ):
        _render_state_rows_editor(
            state_key,
            f"builder_proc_state_editor_{processor.id}",
            state_frame,
        )

    filter_expression = builder.first_filter_expression(processor)
    filter_rows = builder.filter_rows_from_expression(filter_expression)
    filter_mode_key = f"builder_proc_filter_mode_{processor.id}"
    if filter_mode_key not in st.session_state:
        st.session_state[filter_mode_key] = "Rules" if filter_rows is not None else "Raw AST"
    with components.bordered_panel(
        "Processor Filter", "Optional pre-aggregation filter for this processor."
    ):
        mode = st.segmented_control(
            "Filter Mode",
            ["Rules", "Raw AST"],
            default=st.session_state[filter_mode_key],
            key=f"{filter_mode_key}_control",
            help=config_help.field_help("processor.filter_mode"),
        )
        st.session_state[filter_mode_key] = mode
        if mode == "Rules":
            rows_key = f"builder_proc_filter_rows_{processor.id}"
            if rows_key not in st.session_state:
                st.session_state[rows_key] = filter_rows or []
            filter_frame = builder.editor_frame(
                st.session_state[rows_key],
                ["Field", "Operator", "Value", "Enabled"],
                builder.blank_filter_row,
            )
            _render_filter_rows_editor(
                rows_key,
                f"builder_proc_filter_editor_{processor.id}",
                filter_frame,
                field_options,
                value_width="medium",
            )
            try:
                compiled_filter = builder.compile_filter_rows(st.session_state[rows_key])
            except Exception:
                logger.exception(
                    "Failed to compile processor filter rows: processor=%s", processor.id
                )
                compiled_filter = None
        else:
            raw_key = f"builder_proc_raw_filter_{processor.id}"
            raw_filter = st.text_area(
                "Filter AST YAML",
                value=st.session_state.setdefault(
                    raw_key, builder.expression_yaml(filter_expression)
                ),
                height=220,
                key=f"{raw_key}_editor",
                help=config_help.field_help("processor.filter_ast"),
            )
            compiled_filter = (
                builder.parse_expression_yaml(raw_filter) if raw_filter.strip() else None
            )

    time_def = dict(processor_def.get("time", {}))
    time_def.update(
        {
            "column": time_column.strip() or None,
            "grains": grains or ["Summary"],
        }
    )
    processor_def.update(
        {
            "id": processor_id.strip(),
            "source": source_id,
            "kind": kind,
            "description": description.strip(),
            "dimensions": group_by,
            "time": time_def,
        }
    )
    for managed_key in forms.PROCESSOR_KIND_MANAGED_FIELDS:
        processor_def.pop(managed_key, None)
    processor_def.update(kind_settings)
    processor_def.pop("group_by", None)
    processor_def["states"] = _build_state_defs(
        processor,
        st.session_state.get(state_key, _state_rows(processor, field_mapping)),
    )
    if compiled_filter:
        processor_def["filter"] = compiled_filter
    else:
        processor_def.pop("filter", None)
    if not creating:
        processor_def = _preserve_untouched_processor_definition(processor, processor_def)
    _technical_yaml(
        "Generated processor YAML",
        yaml.safe_dump({"processors": [processor_def]}, sort_keys=False),
    )
    draft_status = builder.builder_draft_status(
        f"processor:{'new' if creating else processor.id}",
        None if creating else builder.processor_to_dict(processor),
        processor_def,
    )
    if _render_editor_primary_action(
        save_slot=save_slot,
        draft_slot=draft_slot,
        status=draft_status,
        valid=bool(processor_id.strip()),
        widget_prefixes=("builder_processor_", "builder_proc_"),
        help_text=(
            "Validate and apply this processor to the active workspace. "
            "Use Data Load separately to materialize aggregates."
        ),
    ):
        try:
            with builder.validated_catalog_transaction(ctx.workspace):
                builder.write_processor_definition(ctx.workspace, processor_def)
            message = "Processor created." if creating else "Processor written."
            _complete_builder_apply(
                status=draft_status,
                scope="processor",
                label=description.strip() or humanize_identifier(processor_id.strip()),
                source_ids=[source_id],
                message=message + " Use Data Load to refresh matching aggregates.",
            )
        except Exception as exc:  # pragma: no cover - Streamlit display path
            _record_builder_apply_failed(exc)
            logger.exception(
                "Failed to write processor definition: processor=%s", processor_id.strip()
            )
            st.error(str(exc))


def _processor_kind_settings(
    processor: model.Processor,
    kind: str,
    field_options: list[str],
    field_mapping: dict[str, str],
) -> dict[str, Any]:
    processor_def = _remap_processor_def_fields(builder.processor_to_dict(processor), field_mapping)
    return forms.processor_kind_fields(
        processor_def,
        kind,
        field_options=field_options,
        key_prefix=f"builder_proc_{processor.id}",
    )


def _remap_processor_def_fields(
    processor_def: dict[str, Any],
    field_mapping: dict[str, str],
) -> dict[str, Any]:
    """Remap source-field references in a processor definition copy."""
    if not field_mapping:
        return processor_def
    out = yaml.safe_load(yaml.safe_dump(processor_def, sort_keys=False))
    entities = out.get("entities")
    if isinstance(entities, dict) and entities.get("subject"):
        entities["subject"] = field_remap.remap_field_name(str(entities["subject"]), field_mapping)
    outcome = out.get("outcome")
    if isinstance(outcome, dict) and outcome.get("column"):
        outcome["column"] = field_remap.remap_field_name(str(outcome["column"]), field_mapping)
    for key in ("outcome_column", "variant_column"):
        if out.get(key):
            out[key] = field_remap.remap_field_name(str(out[key]), field_mapping)
    for key in ("properties", "score_properties"):
        values = out.get(key)
        if isinstance(values, list):
            out[key] = field_remap.remap_field_list([str(item) for item in values], field_mapping)
    return out


@st.fragment()
def _metric_builder(  # noqa: PLR0911, PLR0912, PLR0915
    workspace: Path,
    catalog: model.Catalog,
    save_slot: Any,
    draft_slot: Any,
) -> None:
    _claim_fragment_action_slots(save_slot, draft_slot)
    processors = catalog.processors.processors
    if not processors:
        st.info("No processors configured.")
        _render_continue_primary(save_slot)
        return

    metric_names = sorted(catalog.metrics.metrics)
    metric_defs_by_name = {
        name: builder.metric_to_dict(catalog.metrics.metrics[name]) for name in metric_names
    }
    feedback = _consume_pending_metric_refresh(st.session_state, metric_defs_by_name)
    if feedback:
        _render_metric_write_feedback(feedback)

    mode_options = _metric_mode_options(metric_names)
    if st.session_state.get("builder_metric_mode") not in mode_options:
        st.session_state.pop("builder_metric_mode", None)
    with st.container(border=True):
        st.write("### Metric Workflow")
        st.caption(
            "Create a metric from a reviewed recipe or from scratch, then use the same "
            "editor to inspect and maintain catalog metrics."
        )
        mode = (
            st.segmented_control(
                "Metric action",
                mode_options,
                default=st.session_state.get("builder_metric_mode", mode_options[0]),
                selection_mode="single",
                key="builder_metric_mode",
                help=config_help.field_help("metric.action"),
            )
            or mode_options[0]
        )
        creation_method = METRIC_CREATE_LIBRARY
        if mode == METRIC_ACTION_CREATE:
            creation_method = (
                st.segmented_control(
                    "Create from",
                    [METRIC_CREATE_LIBRARY, METRIC_CREATE_SCRATCH],
                    default=st.session_state.get(
                        "builder_metric_creation_method", METRIC_CREATE_LIBRARY
                    ),
                    selection_mode="single",
                    key="builder_metric_creation_method",
                    help=config_help.field_help("metric.create_from"),
                )
                or METRIC_CREATE_LIBRARY
            )
            if creation_method == METRIC_CREATE_LIBRARY:
                st.caption(
                    "Choose a documented business KPI, bind it to processor fields and "
                    "algorithms, and add the generated metric to the catalog."
                )
            else:
                st.caption(
                    "Choose a processor and metric kind, then define the calculation directly."
                )
        else:
            st.caption("Choose an existing metric by processor and kind to inspect or edit it.")

    if mode == METRIC_ACTION_CREATE and creation_method == METRIC_CREATE_LIBRARY:
        recipe_request = recipe_library.render_recipe_library(
            catalog=catalog,
            key_prefix="builder_kpi_recipes",
            submit_label="Add recipe to catalog",
            expanded=True,
            submit_placeholder=_ApplyActionSlot(save_slot),
        )
        if recipe_request is None:
            return
        _record_builder_valid_proposal()
        _record_builder_reviewed()
        try:
            with builder.validated_catalog_transaction(workspace):
                if recipe_request.processor_def:
                    builder.write_processor_definition(
                        workspace,
                        recipe_request.processor_def,
                    )
                builder.write_metric_definition(
                    workspace,
                    recipe_request.metric_id,
                    recipe_request.metric_def,
                )
                if recipe_request.report_target and recipe_request.tile_def:
                    target = recipe_request.report_target
                    builder.write_tile_definition(
                        workspace,
                        dashboard_id=target.dashboard_id,
                        dashboard_title=target.dashboard_title,
                        page_id=target.page_id,
                        page_title=target.page_title,
                        tile=recipe_request.tile_def,
                    )
            outcome_scope = (
                "recipe_with_state" if recipe_request.materialization is not None else "metric"
            )
            outcome_sources = (
                [recipe_request.materialization.source_id]
                if recipe_request.materialization is not None
                else []
            )
            apply_outcome = builder.builder_apply_outcome(
                humanize_identifier(recipe_request.metric_id),
                source_ids=outcome_sources,
                requires_data_run=outcome_scope == "recipe_with_state",
            )
            _store_builder_apply_outcome(apply_outcome)
            _record_builder_applied(apply_outcome)
            st.session_state["builder_apply_notice"] = (
                f"Metric {humanize_identifier(recipe_request.metric_id)!r} applied."
            )
            _queue_metric_refresh(
                st.session_state,
                metric_id=recipe_request.metric_id,
                metric_def=recipe_request.metric_def,
                message=(
                    f"Metric `{recipe_request.metric_id}` was added to "
                    "`catalog/metrics.yaml` and opened for editing."
                ),
                issues=[],
                materialization=recipe_request.materialization,
            )
            st.rerun(scope="app")
        except Exception as exc:  # pragma: no cover - Streamlit display path
            _record_builder_apply_failed(exc)
            logger.exception("Failed to add KPI recipe to the catalog")
            st.error(str(exc))
            return

    left, right = st.columns([1.05, 1.2], gap="large")
    with left, st.container(border=True):
        editing = mode == METRIC_ACTION_EDIT
        st.write("### Edit Metric" if editing else "### Create Metric From Scratch")
        last_mode = st.session_state.get("builder_metric_last_mode")
        if last_mode != mode:
            st.session_state["builder_metric_last_mode"] = mode
            if mode == METRIC_ACTION_CREATE:
                st.session_state["builder_metric_create_counter"] = (
                    int(st.session_state.get("builder_metric_create_counter", 0)) + 1
                )
        selected_metric_name = ""
        seed_metric_def: dict[str, Any] = {}
        seed_kind = ""
        metric_label = ""
        metric_name = ""
        if editing:
            editable_processors = _metric_processors_for_definitions(
                processors, metric_defs_by_name
            )
            if not editable_processors:
                st.info("No editable metrics are available.")
                return
            current_metric = st.session_state.get(
                "builder_metric_selected_id",
                st.session_state.get("builder_metric_select"),
            )
            current_metric_def = (
                metric_defs_by_name.get(str(current_metric), {})
                if isinstance(current_metric, str)
                else {}
            )
            processor = st.selectbox(
                "Processor",
                editable_processors,
                index=_processor_index_by_id(
                    editable_processors,
                    str(current_metric_def.get("source", "") or ""),
                ),
                format_func=_processor_choice_label_human,
                key="builder_metric_processor_edit",
                help=config_help.field_help("metric.processor"),
            )
            metric_kinds = _metric_kinds_for_source(metric_defs_by_name, processor.id)
            if not metric_kinds:
                st.info("Selected processor has no editable metric kinds.")
                return
            current_kind = (
                str(current_metric_def.get("kind", "") or "")
                if current_metric_def.get("source") == processor.id
                else ""
            )
            metric_kind = st.selectbox(
                "Metric Kind",
                metric_kinds,
                index=builder.option_index(metric_kinds, current_kind),
                format_func=builder.metric_kind_label,
                key=f"builder_metric_kind_edit_{processor.id}",
                help=config_help.field_help("metric.kind"),
            )
            st.caption(builder.metric_kind_help(metric_kind))
            metric_choices = _metric_names_for_source_kind(
                metric_defs_by_name, processor.id, metric_kind
            )
            if not metric_choices:
                st.info("Selected processor and kind have no editable metrics.")
                return
            metric_key = f"builder_metric_select_{processor.id}_{metric_kind}"
            if st.session_state.get(metric_key) not in metric_choices:
                st.session_state.pop(metric_key, None)
                current_metric = None
            selected_metric_name = st.selectbox(
                "Metric",
                metric_choices,
                index=builder.option_index(metric_choices, current_metric),
                format_func=lambda name: _metric_choice_label(catalog, name),
                key=metric_key,
                help=config_help.field_help("metric.selector"),
            )
            st.session_state["builder_metric_selected_id"] = selected_metric_name
            seed_metric_def = metric_defs_by_name[selected_metric_name]
            seed_kind = metric_kind
            metric_label = selected_metric_name
            metric_name = selected_metric_name
            metric_token = f"edit_{selected_metric_name}"
            st.caption(f"Metric ID: `{metric_name}`")
        else:
            create_counter = int(st.session_state.get("builder_metric_create_counter", 0))
            create_token = f"create_{create_counter}"
            processors_by_id = {processor.id: processor for processor in processors}
            processor_ids = list(processors_by_id)
            processor_choice = st.selectbox(
                "Processor",
                ["", *processor_ids],
                index=0,
                format_func=lambda value: _processor_choice_label(value, processors_by_id),
                key=f"builder_metric_processor_{create_token}",
                help=config_help.field_help("metric.processor"),
            )
            if not processor_choice:
                st.info("Select a processor to create a metric.")
                return
            processor = processors_by_id[str(processor_choice)]
            metric_kind_options = builder.metric_kind_options(processor)
            if not metric_kind_options:
                st.warning("Selected processor has no executable metric kinds yet.")
                return
            metric_kind = st.selectbox(
                "Metric Kind",
                ["", *metric_kind_options],
                index=0,
                format_func=lambda kind: (
                    builder.metric_kind_label(kind) if kind else "Select metric kind"
                ),
                key=f"builder_metric_kind_{create_token}_{processor.id}",
                help=config_help.field_help("metric.kind"),
            )
            if not metric_kind:
                st.info("Select a metric kind to continue.")
                return
            st.caption(builder.metric_kind_help(metric_kind))
            metric_token = f"{create_token}_{processor.id}_{metric_kind}"
            metric_label = st.text_input(
                "Metric Name",
                value="",
                placeholder=builder.title_from_identifier(
                    builder.default_metric_name(processor, metric_kind)
                ),
                key=f"builder_metric_name_{metric_token}_{processor.id}_{metric_kind}",
                help=config_help.field_help("metric.id"),
            )
            if metric_label.strip():
                metric_name = builder.generated_catalog_id(
                    metric_label,
                    _stable_random_suffix(
                        st.session_state, f"builder_metric_id_suffix_{metric_token}"
                    ),
                    fallback="metric",
                )
                st.caption(f"Metric ID: `{metric_name}`")
        description = st.text_area(
            "Description",
            value=str(seed_metric_def.get("description", "") or ""),
            height=80,
            key=f"builder_metric_desc_{metric_token}_{processor.id}_{metric_kind}",
            help=config_help.field_help("metric.description"),
        )
        depends_on = st.text_input(
            "Depends On",
            value=", ".join(map(str, seed_metric_def.get("depends_on", []) or [])),
            key=f"builder_metric_depends_{metric_token}_{processor.id}_{metric_kind}",
            placeholder="metric_a, metric_b",
            help=config_help.field_help("metric.depends_on"),
        )
        display = _metric_display_controls(
            seed_metric_def.get("display"),
            key_suffix=f"{metric_token}_{processor.id}_{metric_kind}",
        )
        metric_def = _metric_definition_form(
            processor,
            metric_kind,
            seed=seed_metric_def if seed_kind == metric_kind else {},
            key_suffix=f"{metric_token}_{processor.id}_{metric_kind}",
        )
        if metric_def is None:
            return
        if description.strip():
            metric_def["description"] = description.strip()
        dependencies = [item.strip() for item in depends_on.split(",") if item.strip()]
        if dependencies:
            metric_def["depends_on"] = dependencies
        if display:
            metric_def["display"] = display
        if editing:
            metric_def = _preserve_untouched_metric_definition(seed_metric_def, metric_def)
        metric_ready = bool(metric_name.strip()) and (editing or bool(metric_label.strip()))
        draft_status = builder.builder_draft_status(
            f"metric:{selected_metric_name or metric_name or 'new'}",
            seed_metric_def if editing else None,
            metric_def if metric_ready else {"metric": metric_def, "name": metric_name},
        )
        if _render_editor_primary_action(
            save_slot=save_slot,
            draft_slot=draft_slot,
            status=draft_status,
            valid=metric_ready,
            widget_prefixes=("builder_metric_",),
            help_text=(
                "Validate and apply this metric to the active workspace. Existing aggregate "
                "states can be used without running data."
            ),
        ):
            try:
                if not editing and metric_name.strip() in catalog.metrics.metrics:
                    st.error(f"Metric {metric_name.strip()!r} already exists.")
                    return
                with builder.validated_catalog_transaction(workspace):
                    builder.write_metric_definition(workspace, metric_name.strip(), metric_def)
                apply_outcome = builder.builder_apply_outcome(
                    humanize_identifier(metric_name.strip())
                )
                _store_builder_apply_outcome(apply_outcome)
                _record_builder_applied(apply_outcome)
                raw_registry = st.session_state.get(builder.BUILDER_DRAFTS_KEY)
                registry = dict(raw_registry) if isinstance(raw_registry, dict) else {}
                registry.pop(draft_status.key, None)
                st.session_state[builder.BUILDER_DRAFTS_KEY] = registry
                st.session_state["builder_apply_notice"] = (
                    f"Metric {humanize_identifier(metric_name.strip())!r} applied."
                )
                _queue_metric_refresh(
                    st.session_state,
                    metric_id=metric_name.strip(),
                    metric_def=metric_def,
                    message=(
                        f"Metric `{metric_name.strip()}` was written to "
                        "`catalog/metrics.yaml` and opened for editing."
                    ),
                    issues=[],
                )
                st.rerun(scope="app")
            except Exception as exc:  # pragma: no cover - Streamlit display path
                _record_builder_apply_failed(exc)
                logger.exception(
                    "Failed to write metric definition: metric=%s", metric_name.strip()
                )
                st.error(str(exc))

    with right, st.container(border=True):
        st.write("### Review")
        st.caption(
            "The draft revision above is compared with the applied metric. Exact YAML remains "
            "available for expert review."
        )
        _technical_yaml(
            "Generated metric YAML",
            builder.metric_yaml(metric_name or "metric_id", metric_def),
        )


def _metric_definition_form(
    processor: model.Processor,
    metric_kind: str,
    *,
    seed: dict[str, Any] | None = None,
    key_suffix: str | None = None,
) -> dict[str, Any] | None:
    seed = seed or {}
    suffix = key_suffix or f"{processor.id}_{metric_kind}"
    extra = dict(processor.model_extra or {})
    roles = extra.get("variant_role_map", {})
    ctx = forms.MetricFormContext(
        state_options=lambda types: builder.state_columns_by_type(processor, *sorted(types)),
        digest_pairs=builder.digest_state_pair_options(processor),
        funnel_stages=builder.funnel_stage_names(processor),
        default_variant_column=str(extra.get("variant_column", "") or ""),
        variant_roles=dict(roles) if isinstance(roles, dict) else {},
        state_label=lambda state: _digest_state_label(processor, state),
        default_digest_pair=lambda final: builder.default_curve_digest_states(
            processor, final=final
        ),
    )
    fields = forms.metric_kind_fields(
        metric_kind,
        seed,
        ctx,
        key_prefix=f"builder_metric_{suffix}",
    )
    if fields is None:
        return None
    return {"source": processor.id, "kind": metric_kind, **fields}


def _metric_display_controls(raw: Any, *, key_suffix: str) -> dict[str, Any]:
    seed = dict(raw) if isinstance(raw, Mapping) else {}
    with st.expander("Report presentation", expanded=False):
        label = st.text_input(
            "Display label",
            value=str(seed.get("label", "") or ""),
            key=f"builder_metric_display_label_{key_suffix}",
            help=config_help.field_help("metric.display_label"),
        )
        unit = st.text_input(
            "Unit",
            value=str(seed.get("unit", "") or ""),
            key=f"builder_metric_display_unit_{key_suffix}",
            placeholder="orders, EUR, seconds",
            help=config_help.field_help("metric.unit"),
        )
        formats = ["", "percent", "integer", "number", "currency"]
        value_format = st.selectbox(
            "Default value format",
            formats,
            index=builder.option_index(formats, seed.get("value_format")),
            format_func=lambda value: "Unspecified" if not value else value.title(),
            key=f"builder_metric_display_format_{key_suffix}",
            help=config_help.field_help("metric.value_format"),
        )
        directions = ["neutral", "higher_is_better", "lower_is_better"]
        direction = st.selectbox(
            "Direction",
            directions,
            index=builder.option_index(directions, seed.get("direction") or "neutral"),
            format_func=lambda value: value.replace("_", " ").title(),
            key=f"builder_metric_display_direction_{key_suffix}",
            help=config_help.field_help("metric.direction"),
        )
    display = {
        "label": label.strip(),
        "unit": unit.strip(),
        "value_format": value_format or None,
        "direction": direction,
    }
    return {
        key: value
        for key, value in display.items()
        if value not in (None, "") and (key != "direction" or value != "neutral")
    }


def _preserve_untouched_metric_definition(
    seed_metric_def: dict[str, Any],
    metric_def: dict[str, Any],
) -> dict[str, Any]:
    """Keep the exact authored metric when the visual form is semantically unchanged."""

    try:
        baseline = model.Metrics.model_validate({"metrics": {"metric": seed_metric_def}}).metrics[
            "metric"
        ]
        candidate = model.Metrics.model_validate({"metrics": {"metric": metric_def}}).metrics[
            "metric"
        ]
    except ValueError:
        return metric_def
    status = builder.builder_draft_status(
        "metric-editor:metric",
        baseline.model_dump(mode="json", by_alias=True, exclude_none=True),
        candidate.model_dump(mode="json", by_alias=True, exclude_none=True),
    )
    return seed_metric_def if not status.dirty else metric_def


def _processor_index_by_id(processors: list[model.Processor], processor_id: str) -> int:
    for index, processor in enumerate(processors):
        if processor.id == processor_id:
            return index
    return 0


def _tile_editor_token(
    dashboard_id: str | None,
    page_id: str | None,
    tile: dict[str, Any],
) -> str:
    tile_id = str(tile.get("id", "active") or "active")
    return f"{dashboard_id or 'new'}__{page_id or 'new'}__{tile_id}"


def _start_new_tile_editor(session_state: MutableMapping[str, Any]) -> None:
    counter = int(session_state.get("builder_tile_new_counter", 0)) + 1
    session_state["builder_tile_new_counter"] = counter
    session_state["builder_tile_seed"] = (None, None, {})
    session_state["builder_tile_editor_token"] = f"new_{counter}"


TileOption = tuple[str, str, str, dict[str, Any]]


def _report_library_options(
    catalog: model.Catalog,
    *,
    search: str,
    metric_filter: str,
    chart_filter: str,
) -> list[TileOption]:
    """Return configured tiles matching the compact library controls."""

    options: list[TileOption] = []
    normalized_search = search.strip().casefold()
    for dashboard in catalog.dashboards.dashboards:
        for page in dashboard.pages:
            for tile in page.tiles:
                if metric_filter not in ("All", tile.metric):
                    continue
                if chart_filter not in ("All", tile.chart):
                    continue
                searchable = " ".join(
                    (
                        dashboard.id,
                        dashboard.title,
                        page.id,
                        page.title,
                        tile.id,
                        tile.title,
                        tile.metric,
                        tile.chart,
                    )
                ).casefold()
                if normalized_search and normalized_search not in searchable:
                    continue
                options.append(
                    (
                        dashboard.id,
                        page.id,
                        tile.id,
                        tile.model_dump(
                            mode="json",
                            by_alias=True,
                            exclude_none=True,
                            exclude_unset=True,
                        ),
                    )
                )
    return options


def _tile_option_key(option: TileOption) -> str:
    return f"{option[0]}/{option[1]}/{option[2]}"


def _report_library_tile_label(option: TileOption) -> str:
    tile = option[3]
    title = str(tile.get("title") or option[2])
    return f"{title} · {humanize_identifier(option[1])}"


def _report_library_chart_label(chart_type: str) -> str:
    return REPORT_LIBRARY_CHART_LABELS.get(chart_type, humanize_identifier(chart_type))


def _select_report_library_tile(widget_key: str) -> None:
    selected = st.session_state.get(widget_key)
    if selected:
        st.session_state["builder_tile_selection_override"] = selected


def _render_report_library_browser(
    catalog: model.Catalog,
    metric_names: list[str],
) -> tuple[list[TileOption], str]:
    """Render a purpose-led visual tile picker and return its visible selection."""

    st.write("### Report library")
    configured_chart_types = sorted(
        {
            tile.chart
            for dashboard in catalog.dashboards.dashboards
            for page in dashboard.pages
            for tile in page.tiles
        }
    )
    filter_columns = st.columns([2, 1, 1], vertical_alignment="bottom")
    with filter_columns[0]:
        search = st.text_input(
            "Search",
            key="builder_tile_search",
            placeholder="Title, dashboard, page, metric, or chart type",
            help=config_help.field_help("report.library_search"),
        )
    with filter_columns[1]:
        metric_filter = st.selectbox(
            "Metric",
            ["All", *metric_names],
            key="builder_metric_filter",
            help=config_help.field_help("report.metric_filter"),
        )
    with filter_columns[2]:
        chart_filter = st.selectbox(
            "Chart type",
            ["All", *configured_chart_types],
            key="builder_chart_filter",
            help=config_help.field_help("report.chart_filter"),
        )

    filtered_options = _report_library_options(
        catalog,
        search=search,
        metric_filter=metric_filter,
        chart_filter=chart_filter,
    )
    if not filtered_options:
        st.caption("No existing report tiles match these filters.")
        return [], NEW_TILE_KEY

    available_groups = [
        group_id
        for group_id, group in REPORT_LIBRARY_GROUPS.items()
        if any(option[3].get("chart") in group.chart_types for option in filtered_options)
    ]
    selected_before_grouping = _selected_tile(
        filtered_options,
        st.session_state.get("builder_selected_tile_key"),
    )
    selected_group = (
        REPORT_LIBRARY_GROUP_BY_CHART.get(str(selected_before_grouping[3].get("chart")))
        if selected_before_grouping is not None
        else None
    )
    purpose_key = "builder_report_library_purpose"
    if st.session_state.get(purpose_key) not in available_groups:
        st.session_state[purpose_key] = (
            selected_group if selected_group in available_groups else available_groups[0]
        )
    purpose = st.segmented_control(
        "Purpose",
        available_groups,
        key=purpose_key,
        required=True,
        width="stretch",
        format_func=lambda group_id: (
            f"{REPORT_LIBRARY_GROUPS[group_id].icon} {REPORT_LIBRARY_GROUPS[group_id].label}"
        ),
        help=config_help.field_help("report.library_purpose"),
    )
    purpose = str(purpose or available_groups[0])
    group = REPORT_LIBRARY_GROUPS[purpose]
    st.caption(group.description)

    visible_options = [
        option for option in filtered_options if option[3].get("chart") in group.chart_types
    ]
    visible_keys = [_tile_option_key(option) for option in visible_options]
    selected_tile_key = str(st.session_state.get("builder_selected_tile_key", ""))
    if selected_tile_key not in visible_keys:
        selected_tile_key = visible_keys[0]
        st.session_state["builder_selected_tile_key"] = selected_tile_key

    visible_chart_types = [
        chart_type
        for chart_type in group.chart_types
        if any(option[3].get("chart") == chart_type for option in visible_options)
    ]
    theme_base = str(dashboard_theme().get("base", "light"))
    for chart_type in visible_chart_types:
        chart_options = [
            option for option in visible_options if option[3].get("chart") == chart_type
        ]
        _render_report_library_chart_group(
            chart_type,
            chart_options,
            selected_tile_key=selected_tile_key,
            theme_base=theme_base,
        )
    return visible_options, selected_tile_key


def _render_report_library_chart_group(
    chart_type: str,
    tile_options: list[TileOption],
    *,
    selected_tile_key: str,
    theme_base: str,
) -> None:
    """Render one compact chart-type preview with selectable report tiles."""

    with st.container(border=True):
        preview_column, report_column = st.columns([1, 4], vertical_alignment="top")
        with preview_column:
            st.plotly_chart(
                _chart_library_preview(chart_type, theme_base=theme_base),
                width="stretch",
                height=140,
                theme=None,
                key=f"builder_report_library_preview_{chart_type}",
                config={"displayModeBar": False, "staticPlot": True, "responsive": True},
            )
        with report_column:
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f"**{_report_library_chart_label(chart_type)}**")
                st.badge(
                    f"{len(tile_options)} report{'s' if len(tile_options) != 1 else ''}",
                    color="gray",
                )
            st.caption(
                REPORT_LIBRARY_CHART_DESCRIPTIONS.get(
                    chart_type,
                    "Use this chart type to present the selected metric.",
                )
            )
            tile_keys = [_tile_option_key(option) for option in tile_options]
            tile_labels = {
                _tile_option_key(option): _report_library_tile_label(option)
                for option in tile_options
            }
            default = selected_tile_key if selected_tile_key in tile_keys else None
            if len(tile_keys) > REPORT_LIBRARY_PILLS_MAX:
                widget_key = f"builder_report_library_select_{chart_type}"
                if widget_key in st.session_state and st.session_state[widget_key] != default:
                    del st.session_state[widget_key]
                widget_has_state = widget_key in st.session_state
                st.selectbox(
                    f"Open {_report_library_chart_label(chart_type)} report",
                    tile_keys,
                    index=(
                        None if widget_has_state or default is None else tile_keys.index(default)
                    ),
                    key=widget_key,
                    format_func=lambda tile_key: tile_labels[tile_key],
                    placeholder=f"Choose from {len(tile_keys)} reports",
                    help="Choose a configured report tile to open in the editor below.",
                    on_change=_select_report_library_tile,
                    args=(widget_key,),
                )
            else:
                widget_key = f"builder_report_library_pills_{chart_type}"
                if widget_key in st.session_state and st.session_state[widget_key] != default:
                    del st.session_state[widget_key]
                widget_has_state = widget_key in st.session_state
                st.pills(
                    f"Open {_report_library_chart_label(chart_type)} report",
                    tile_keys,
                    key=widget_key,
                    default=None if widget_has_state else default,
                    format_func=lambda tile_key: tile_labels[tile_key],
                    label_visibility="collapsed",
                    width="stretch",
                    on_change=_select_report_library_tile,
                    args=(widget_key,),
                )


def _chart_library_preview(  # noqa: PLR0912, PLR0915
    chart_type: str, *, theme_base: str = "light"
) -> go.Figure:
    """Return a small representative Plotly figure for one chart kind."""

    dark = theme_base == "dark"
    colors = PLOTLY_DARK_COLORWAY if dark else PLOTLY_LIGHT_COLORWAY
    ink = "#f0f3f0" if dark else "#1a1a1a"
    soft = "#171c18" if dark else "#e8ebe6"
    transparent = "rgba(0,0,0,0)"
    x = [0, 1, 2, 3, 4]
    y = [1.0, 2.8, 2.1, 4.2, 3.6]
    figure = go.Figure()

    if chart_type == "kpi_card":
        figure.add_trace(
            go.Indicator(
                mode="number+delta",
                value=72,
                number={"suffix": "%", "font": {"size": 28}},
                delta={"reference": 64, "position": "bottom"},
            )
        )
    elif chart_type == "gauge":
        figure.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=72,
                number={"font": {"size": 22}},
                gauge={
                    "axis": {"range": [0, 100], "visible": False},
                    "bar": {"color": colors[0]},
                    "bgcolor": soft,
                },
            )
        )
    elif chart_type == "table":
        figure.add_trace(
            go.Table(
                header={"values": ["Rank", "Value"], "fill_color": soft},
                cells={"values": [[1, 2, 3], [42, 31, 19]], "fill_color": transparent},
            )
        )
    elif chart_type == "stacked_area":
        figure.add_trace(go.Scatter(x=x, y=y, stackgroup="one", line={"color": colors[0]}))
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[1.8, 1.2, 2.0, 1.4, 2.1],
                stackgroup="one",
                line={"color": colors[1]},
            )
        )
    elif chart_type == "combo":
        figure.add_trace(go.Bar(x=x, y=[2, 4, 3, 5, 4], marker_color=colors[0]))
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[35, 48, 43, 62, 58],
                mode="lines+markers",
                line={"color": colors[1], "width": 3},
                yaxis="y2",
            )
        )
        figure.update_layout(yaxis2={"overlaying": "y", "side": "right", "visible": False})
    elif chart_type in {
        "calendar_heatmap",
        "heatmap",
        "cohort_heatmap",
        "corr",
        "rfm_density",
        "descriptive_heatmap",
    }:
        figure.add_trace(
            go.Heatmap(
                z=[[1, 2, 4, 3], [2, 5, 3, 1], [4, 3, 2, 5]],
                colorscale=[[0, soft], [1, colors[0]]],
                showscale=False,
            )
        )
    elif chart_type in {"bar", "experiment_z_score"}:
        figure.add_trace(go.Bar(x=["A", "B", "C", "D"], y=[4, 7, 5, 8], marker_color=colors[0]))
    elif chart_type == "waterfall":
        figure.add_trace(
            go.Waterfall(
                x=["Start", "+", "-", "End"],
                y=[5, 3, -2, 0],
                measure=["absolute", "relative", "relative", "total"],
                increasing={"marker": {"color": colors[2]}},
                decreasing={"marker": {"color": colors[1]}},
            )
        )
    elif chart_type == "pareto":
        figure.add_trace(go.Bar(x=["A", "B", "C", "D"], y=[8, 5, 3, 2], marker_color=colors[0]))
        figure.add_trace(
            go.Scatter(
                x=["A", "B", "C", "D"],
                y=[0.44, 0.72, 0.89, 1.0],
                mode="lines+markers",
                line={"color": colors[1], "width": 3},
                yaxis="y2",
            )
        )
        figure.update_layout(yaxis2={"overlaying": "y", "side": "right", "visible": False})
    elif chart_type in {"interval", "experiment_odds_ratio"}:
        figure.add_trace(
            go.Scatter(
                x=[1.2, 2.1, 2.8, 3.7],
                y=["A", "B", "C", "D"],
                mode="markers",
                marker={"color": colors[0], "size": 9},
                error_x={"type": "data", "array": [0.3, 0.5, 0.4, 0.6]},
            )
        )
    elif chart_type == "bar_polar":
        figure.add_trace(
            go.Barpolar(theta=[0, 72, 144, 216, 288], r=[4, 7, 5, 8, 6], marker_color=colors[0])
        )
    elif chart_type == "scatter":
        figure.add_trace(
            go.Scatter(
                x=[1, 2, 2.6, 3.5, 4.2, 4.8],
                y=[1.2, 2.8, 2.1, 4.0, 3.5, 5.0],
                mode="markers",
                marker={"size": [7, 11, 8, 14, 10, 13], "color": colors[0]},
            )
        )
    elif chart_type in {"boxplot", "descriptive_boxplot"}:
        figure.add_trace(
            go.Box(y=[1.0, 1.5, 1.8, 2.1, 2.3, 3.8], marker_color=colors[0], boxpoints="outliers")
        )
    elif chart_type in {"histogram", "descriptive_histogram"}:
        figure.add_trace(
            go.Histogram(x=[1, 1, 2, 2, 2, 3, 3, 4, 5], marker_color=colors[0], nbinsx=5)
        )
    elif chart_type in {"treemap", "clv_treemap"}:
        figure.add_trace(
            go.Treemap(
                labels=["All", "A", "B", "C"],
                parents=["", "All", "All", "All"],
                values=[10, 5, 3, 2],
                marker={"colors": [soft, colors[0], colors[1], colors[2]]},
            )
        )
    elif chart_type == "donut":
        figure.add_trace(
            go.Pie(values=[52, 29, 19], labels=["A", "B", "C"], hole=0.55, marker_colors=colors[:3])
        )
    elif chart_type == "geo_map":
        figure.add_trace(
            go.Scattergeo(
                lon=[-100, 10, 138],
                lat=[38, 51, 36],
                mode="markers",
                marker={"size": [13, 9, 11], "color": colors[:3]},
            )
        )
        figure.update_geos(
            showcoastlines=False,
            showframe=False,
            showland=True,
            landcolor=soft,
            bgcolor=transparent,
            projection_type="natural earth",
        )
    elif chart_type == "sankey":
        figure.add_trace(
            go.Sankey(
                node={"label": ["A", "B", "C"], "color": colors[:3], "pad": 8},
                link={"source": [0, 0, 1], "target": [1, 2, 2], "value": [5, 2, 3]},
            )
        )
    elif chart_type in {"funnel", "descriptive_funnel"}:
        figure.add_trace(
            go.Funnel(y=["Visit", "Engage", "Convert"], x=[100, 62, 27], marker_color=colors[:3])
        )
    elif chart_type == "exposure":
        figure.add_trace(
            go.Bar(x=["1", "2", "3", "4+"], y=[100, 72, 43, 21], marker_color=colors[0])
        )
    else:
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line={"color": colors[0], "width": 3, "shape": "spline"},
                marker={"size": 6},
            )
        )
        if chart_type in {
            "calibration_curve",
            "roc_curve",
            "precision_recall_curve",
            "gain_curve",
            "lift_curve",
        }:
            figure.add_trace(
                go.Scatter(
                    x=[0, 4],
                    y=[0, 4],
                    mode="lines",
                    line={"color": colors[1], "dash": "dot", "width": 1},
                )
            )

    figure.update_layout(
        height=132,
        margin={"l": 5, "r": 5, "t": 5, "b": 5},
        showlegend=False,
        hovermode=False,
        paper_bgcolor=transparent,
        plot_bgcolor=transparent,
        font={"color": ink, "size": 10},
    )
    figure.update_xaxes(visible=False, fixedrange=True)
    figure.update_yaxes(visible=False, fixedrange=True)
    return figure


@st.fragment()
def _tile_builder(  # noqa: PLR0912, PLR0915
    workspace: Path,
    catalog: model.Catalog,
    save_slot: Any,
    draft_slot: Any,
) -> None:
    _claim_fragment_action_slots(save_slot, draft_slot)
    metric_names = sorted(catalog.metrics.metrics)
    if not metric_names:
        st.info("No metrics configured.")
        _render_continue_primary(save_slot)
        return

    selected_tile_override = st.session_state.pop("builder_tile_selection_override", None)
    if selected_tile_override is not None:
        st.session_state["builder_selected_tile_key"] = selected_tile_override

    with st.container(border=True):
        tile_options, selected_tile_key = _render_report_library_browser(catalog, metric_names)
        st.session_state["builder_selected_tile_key"] = selected_tile_key
        if selected_tile_key != st.session_state.get("builder_tile_last_selected_key"):
            st.session_state["builder_tile_last_selected_key"] = selected_tile_key
            if selected_tile_key == NEW_TILE_KEY:
                if "builder_tile_seed" not in st.session_state:
                    _start_new_tile_editor(st.session_state)
            else:
                st.session_state.pop("builder_tile_seed", None)
                st.session_state.pop("builder_tile_editor_token", None)
        selected_seed = _selected_tile(tile_options, selected_tile_key)
        with st.container(horizontal=True, horizontal_alignment="distribute"):
            new_tile = st.button("New", icon=":material/add_2:", key="builder_new_tile")
            duplicate_tile = st.button(
                "Duplicate",
                icon=":material/content_copy:",
                key="builder_duplicate_tile",
                disabled=selected_seed is None,
            )
            delete_tile = st.button(
                "Delete",
                icon=":material/delete:",
                key="builder_delete_tile",
                disabled=selected_seed is None,
            )
        if new_tile:
            _start_new_tile_editor(st.session_state)
            st.session_state["builder_tile_selection_override"] = NEW_TILE_KEY
            st.rerun()
        if duplicate_tile and selected_seed is not None:
            counter = int(st.session_state.get("builder_tile_new_counter", 0)) + 1
            st.session_state["builder_tile_new_counter"] = counter
            seed = dict(selected_seed[3])
            seed.pop("id", None)
            seed["title"] = f"{seed.get('title', 'Tile')} Copy"
            st.session_state["builder_tile_seed"] = (selected_seed[0], selected_seed[1], seed)
            st.session_state["builder_tile_editor_token"] = f"new_{counter}"
        if delete_tile and selected_seed is not None:
            st.session_state[BUILDER_PENDING_TILE_DELETE_KEY] = {
                "dashboard_id": selected_seed[0],
                "page_id": selected_seed[1],
                "tile_id": selected_seed[2],
                "title": str(selected_seed[3].get("title", selected_seed[2])),
            }
            st.rerun()

        pending_delete = st.session_state.get(BUILDER_PENDING_TILE_DELETE_KEY)
        if (
            isinstance(pending_delete, Mapping)
            and selected_seed is not None
            and (
                pending_delete.get("dashboard_id"),
                pending_delete.get("page_id"),
                pending_delete.get("tile_id"),
            )
            == selected_seed[:3]
        ):
            tile_label = str(pending_delete.get("title", selected_seed[2]))
            st.warning(
                f"Tile {tile_label!r} is staged for deletion. Apply to workspace to remove it, "
                "or discard this draft to keep it."
            )
            deletion_status = builder.builder_draft_status(
                f"report-delete:{selected_seed[0]}/{selected_seed[1]}/{selected_seed[2]}",
                {"deleted": False, "tile": selected_seed[3]},
                {"deleted": True, "tile_id": selected_seed[2]},
            )
            if _render_editor_primary_action(
                save_slot=save_slot,
                draft_slot=draft_slot,
                status=deletion_status,
                valid=True,
                widget_prefixes=(BUILDER_PENDING_TILE_DELETE_KEY,),
                help_text=f"Validate and remove tile {tile_label!r} from the active workspace.",
            ):
                try:
                    with builder.validated_catalog_transaction(workspace):
                        deleted = builder.delete_tile_definition(
                            workspace,
                            dashboard_id=selected_seed[0],
                            page_id=selected_seed[1],
                            tile_id=selected_seed[2],
                        )
                    if not deleted:
                        st.warning("Tile was not found; the workspace was not changed.")
                        return
                    st.session_state.pop(BUILDER_PENDING_TILE_DELETE_KEY, None)
                    _complete_builder_apply(
                        status=deletion_status,
                        scope="report",
                        label=tile_label,
                        message="Tile removed from the workspace.",
                    )
                except Exception as exc:  # pragma: no cover - Streamlit display path
                    _record_builder_apply_failed(exc)
                    logger.exception(
                        "Failed to delete tile definition: dashboard=%s page=%s tile=%s",
                        selected_seed[0],
                        selected_seed[1],
                        selected_seed[2],
                    )
                    st.error(str(exc))
            return

    with st.container(border=True):
        st.write("### Tile Editor")
        seed_context = st.session_state.get("builder_tile_seed")
        if seed_context:
            seed_dashboard, seed_page, seed_tile = seed_context
        elif selected_seed is not None:
            seed_dashboard, seed_page, seed_tile = (
                selected_seed[0],
                selected_seed[1],
                selected_seed[3],
            )
        else:
            seed_dashboard, seed_page, seed_tile = None, None, {}
        editor_token = st.session_state.get(
            "builder_tile_editor_token",
            _tile_editor_token(seed_dashboard, seed_page, seed_tile),
        )
        is_new_tile = selected_tile_key == NEW_TILE_KEY or str(editor_token).startswith("new_")
        if is_new_tile:
            st.caption(NEW_TILE_LABEL)
        mode = st.segmented_control(
            "Editing Mode",
            ["Visual", "Raw YAML"],
            default="Visual",
            key=f"builder_tile_mode_{editor_token}",
            help=config_help.field_help("report.editing_mode"),
        )
        seed_metric = seed_tile.get("metric") if seed_tile.get("metric") in metric_names else None
        if seed_metric is None and not is_new_tile:
            seed_metric = metric_names[0]
        metric_name = st.selectbox(
            "Metric",
            metric_names,
            index=metric_names.index(seed_metric) if seed_metric is not None else None,
            placeholder="Select metric",
            key=f"builder_tile_metric_{editor_token}",
            help=config_help.field_help("report.metric"),
        )
        chart_choices = (
            builder.chart_choices_for_metric(catalog, metric_name) if metric_name else []
        )
        seed_chart = seed_tile.get("chart") if seed_tile.get("chart") in chart_choices else None
        if seed_chart is None and chart_choices and not is_new_tile:
            seed_chart = chart_choices[0]
        chart_kind = st.selectbox(
            "Chart",
            chart_choices,
            index=chart_choices.index(seed_chart) if seed_chart is not None else None,
            placeholder="Select chart" if metric_name else "Select metric first",
            disabled=not metric_name,
            key=f"builder_tile_chart_{editor_token}",
            help=config_help.field_help("report.chart"),
        )
        if metric_name and chart_kind:
            components.key_value_strip(
                [
                    {"label": key, "value": value}
                    for key, value in builder.chart_recipe_summary(
                        catalog, metric_name, chart_kind
                    ).items()
                ]
            )
        compatible_field_keys = _compatible_tile_field_keys(str(chart_kind or ""))
        selection_matches_seed = bool(
            not is_new_tile
            and metric_name == seed_tile.get("metric")
            and chart_kind == seed_tile.get("chart")
        )
        if selection_matches_seed:
            defaults = {
                key: value
                for key, value in seed_tile.items()
                if key in compatible_field_keys or key in builder.CHART_SETTING_FIELDS
            }
        else:
            defaults = (
                builder.default_tile_fields(catalog, metric_name, chart_kind)
                if metric_name and chart_kind
                else {}
            )
            defaults.update(
                {
                    key: value
                    for key, value in seed_tile.items()
                    if key in builder.CHART_SETTING_FIELDS
                }
            )
        field_options = (
            ["", *builder.chart_field_options(catalog, metric_name)] if metric_name else [""]
        )

        dashboards_by_id = {dashboard.id: dashboard for dashboard in catalog.dashboards.dashboards}
        dashboard_choices = [*dashboards_by_id, NEW_DASHBOARD_KEY]
        first_dashboard = (
            seed_dashboard
            if seed_dashboard in dashboards_by_id
            else next(iter(dashboards_by_id), NEW_DASHBOARD_KEY)
        )
        dashboard_choice = st.selectbox(
            "Dashboard",
            dashboard_choices,
            index=builder.option_index(dashboard_choices, first_dashboard),
            format_func=lambda value: _dashboard_choice_label(value, dashboards_by_id),
            key=f"builder_dashboard_choice_{editor_token}",
            help=config_help.field_help("report.dashboard"),
        )
        if dashboard_choice == NEW_DASHBOARD_KEY:
            dashboard_title = st.text_input(
                "Dashboard Name",
                value=builder.title_from_identifier(seed_dashboard) if seed_dashboard else "",
                key=f"builder_dashboard_name_{editor_token}",
                help=config_help.field_help("report.dashboard_id"),
            )
            dashboard_id = _generated_catalog_id(
                dashboard_title,
                _stable_random_suffix(
                    st.session_state, f"builder_dashboard_id_suffix_{editor_token}"
                ),
                fallback="dashboard",
            )
            existing_dashboard = None
        else:
            existing_dashboard = dashboards_by_id[str(dashboard_choice)]
            dashboard_id = existing_dashboard.id
            dashboard_title = existing_dashboard.title

        pages_by_id = (
            {page.id: page for page in existing_dashboard.pages} if existing_dashboard else {}
        )
        page_choices = [*pages_by_id, NEW_PAGE_KEY]
        first_page = (
            seed_page if seed_page in pages_by_id else next(iter(pages_by_id), NEW_PAGE_KEY)
        )
        page_choice = st.selectbox(
            "Page",
            page_choices,
            index=builder.option_index(page_choices, first_page),
            format_func=lambda value: _page_choice_label(value, pages_by_id),
            key=f"builder_page_choice_{editor_token}_{dashboard_id}",
            help=config_help.field_help("report.page"),
        )
        if page_choice == NEW_PAGE_KEY:
            page_title = st.text_input(
                "Page Name",
                value=builder.title_from_identifier(seed_page) if seed_page else "",
                key=f"builder_page_name_{editor_token}_{dashboard_id}",
                help=config_help.field_help("report.page_id"),
            )
            page_id = _generated_catalog_id(
                page_title,
                _stable_random_suffix(st.session_state, f"builder_page_id_suffix_{editor_token}"),
                fallback="page",
            )
        else:
            page = pages_by_id[str(page_choice)]
            page_id = page.id
            page_title = page.title

        page_filters, page_time_filter, page_settings_ready = _page_settings_editor(
            dashboard_id=dashboard_id,
            dashboard_title=dashboard_title,
            page_id=page_id,
            page_title=page_title,
            page=pages_by_id.get(str(page_choice)),
            key_suffix=f"{editor_token}_{dashboard_id}_{page_id}",
        )
        existing_page = pages_by_id.get(str(page_choice))

        default_tile_title = str(
            seed_tile.get(
                "title",
                "" if is_new_tile or not metric_name else metric_name.replace("_", " "),
            )
        )
        title = st.text_input(
            "Tile Title",
            value=default_tile_title,
            key=f"builder_tile_title_{editor_token}",
            help=config_help.field_help("report.tile_title"),
        )
        tile_id = (
            str(seed_tile["id"])
            if seed_tile.get("id") and not is_new_tile
            else _generated_catalog_id(
                title.strip() or str(metric_name or chart_kind or "tile"),
                _stable_random_suffix(st.session_state, f"builder_tile_id_suffix_{editor_token}"),
                fallback="tile",
            )
        )

        if mode == "Raw YAML":
            default_raw_tile = seed_tile
            if not default_raw_tile and metric_name and chart_kind and tile_id.strip():
                default_raw_tile = builder.build_tile(
                    tile_id=tile_id.strip(),
                    title=title.strip() or metric_name,
                    metric_name=metric_name,
                    chart_kind=chart_kind,
                    fields=defaults,
                )
            raw_tile = st.text_area(
                "Tile YAML",
                value=yaml.safe_dump(default_raw_tile, sort_keys=False) if default_raw_tile else "",
                height=360,
                key=f"builder_raw_tile_{editor_token}",
                help=config_help.field_help("report.tile_yaml"),
            )
            try:
                loaded_tile = yaml.safe_load(raw_tile) or {}
                built_tile = loaded_tile.get("tile", loaded_tile)
                if not isinstance(built_tile, dict):
                    raise ValueError("Tile YAML must be a mapping")
            except Exception as exc:
                built_tile = {}
                logger.exception("Failed to parse tile YAML: tile=%s", tile_id.strip())
                st.error(str(exc))
        elif metric_name and chart_kind:
            fields = _tile_field_controls(
                chart_kind,
                defaults,
                field_options,
                key_suffix=editor_token,
                catalog=catalog,
                metric_name=metric_name,
            )
            built_tile = (
                builder.build_tile(
                    tile_id=tile_id.strip(),
                    title=title.strip() or metric_name,
                    metric_name=metric_name,
                    chart_kind=chart_kind,
                    fields=fields,
                )
                if tile_id.strip()
                else {}
            )
        else:
            built_tile = {}
        if selected_seed is not None and not is_new_tile and built_tile:
            built_tile = _preserve_untouched_tile_definition(seed_tile, built_tile)

        if st.button("Preview Tile", icon=":material/preview:", disabled=not built_tile):
            _preview_tile(workspace, catalog, built_tile)
        baseline_page = None
        if existing_page is not None:
            baseline_page = {
                "id": existing_page.id,
                "title": existing_page.title,
                "filters": [
                    item.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for item in existing_page.filters
                ],
                "time_filter": existing_page.time_filter.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                ),
            }
        proposed_page = {
            "id": page_id.strip(),
            "title": page_title.strip() or builder.title_from_identifier(page_id),
            "filters": page_filters,
            "time_filter": page_time_filter,
        }
        draft_status = builder.builder_draft_status(
            f"report:{selected_tile_key}",
            {
                "dashboard_id": seed_dashboard,
                "page": baseline_page,
                "tile": seed_tile if selected_seed is not None else None,
            },
            {
                "dashboard_id": dashboard_id.strip(),
                "page": proposed_page,
                "tile": built_tile,
            },
        )
        report_ready = bool(
            built_tile and dashboard_id.strip() and page_id.strip() and page_settings_ready
        )
        if _render_editor_primary_action(
            save_slot=save_slot,
            draft_slot=draft_slot,
            status=draft_status,
            valid=report_ready,
            widget_prefixes=(
                "builder_tile_",
                "builder_dashboard_",
                "builder_page_",
                "builder_report_",
            ),
            help_text=(
                "Validate and apply this report tile and its page settings to the workspace."
            ),
        ):
            try:
                with builder.validated_catalog_transaction(workspace):
                    builder.write_tile_definition(
                        workspace,
                        dashboard_id=dashboard_id.strip(),
                        dashboard_title=dashboard_title.strip()
                        or builder.title_from_identifier(dashboard_id),
                        page_id=page_id.strip(),
                        page_title=page_title.strip() or builder.title_from_identifier(page_id),
                        tile=built_tile,
                    )
                    builder.write_page_settings(
                        workspace,
                        dashboard_id=dashboard_id.strip(),
                        dashboard_title=dashboard_title.strip()
                        or builder.title_from_identifier(dashboard_id),
                        page_id=page_id.strip(),
                        page_title=page_title.strip() or builder.title_from_identifier(page_id),
                        filters=page_filters,
                        time_filter=page_time_filter,
                    )
                _complete_builder_apply(
                    status=draft_status,
                    scope="report",
                    label=title.strip() or humanize_identifier(str(built_tile.get("id", ""))),
                    message="Report tile and page settings applied.",
                )
            except Exception as exc:  # pragma: no cover - Streamlit display path
                _record_builder_apply_failed(exc)
                logger.exception(
                    "Failed to write tile definition: dashboard=%s page=%s tile=%s",
                    dashboard_id.strip(),
                    page_id.strip(),
                    built_tile.get("id"),
                )
                st.error(str(exc))

        _technical_yaml(
            "Generated report YAML",
            builder.tile_yaml(built_tile) if built_tile else "{}",
        )

    with st.expander("Report inventory", expanded=False, icon=":material/inventory_2:"):
        st.caption("Search configured reports by human title, metric, or chart type.")
        _report_inventory_tab(catalog)


def _compatible_tile_field_keys(chart_kind: str) -> set[str]:
    keys = set(builder.chart_field_controls(chart_kind))
    keys.update({"facet_column", "facets"})
    if "error_y" in keys:
        keys.update({"error_y_plus", "error_y_minus"})
    if "locations" in keys:
        keys.add("location")
    if chart_kind in {"calendar_heatmap", "donut"}:
        keys.update({"x", "y"})
    if chart_kind in {"gauge", "table"}:
        keys.add("group_by")
    return keys


def _preserve_untouched_tile_definition(
    seed_tile: dict[str, Any],
    built_tile: dict[str, Any],
) -> dict[str, Any]:
    """Keep exact tile YAML when the visual editor preserves effective behavior."""

    try:
        baseline = model.Tile.model_validate(seed_tile)
        candidate = model.Tile.model_validate(built_tile)
    except ValueError:
        return built_tile
    status = builder.builder_draft_status(
        f"tile-editor:{baseline.id}",
        baseline.model_dump(mode="json", by_alias=True, exclude_none=True),
        candidate.model_dump(mode="json", by_alias=True, exclude_none=True),
    )
    return seed_tile if not status.dirty else built_tile


def _page_settings_editor(
    *,
    dashboard_id: str,
    dashboard_title: str,
    page_id: str,
    page_title: str,
    page: model.DashboardPage | None,
    key_suffix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    with st.expander("Page filters and time range", expanded=False):
        rows = [item.model_dump(mode="python") for item in page.filters] if page is not None else []
        edited = st.data_editor(
            rows,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key=f"builder_page_filters_{key_suffix}",
            column_config={
                "field": st.column_config.TextColumn(
                    "Aggregate field",
                    required=True,
                    help=config_help.field_help("report.filter_field"),
                ),
                "label": st.column_config.TextColumn(
                    "Display label", help=config_help.field_help("report.filter_label")
                ),
                "display": st.column_config.SelectboxColumn(
                    "Placement",
                    options=["primary", "secondary"],
                    required=True,
                    help=config_help.field_help("report.filter_placement"),
                ),
                "scope": st.column_config.SelectboxColumn(
                    "Coverage",
                    options=["all_tiles", "compatible_tiles"],
                    required=True,
                    help=config_help.field_help("report.filter_scope"),
                ),
                "control": st.column_config.SelectboxColumn(
                    "Control",
                    options=["multiselect", "selectbox", "text"],
                    required=True,
                    help=config_help.field_help("report.filter_control"),
                ),
            },
        )
        filters = _page_filter_rows(edited)
        all_presets = [
            "last_7_days",
            "last_30_days",
            "last_90_days",
            "year_to_date",
            "custom",
            "all_time",
        ]
        seed_time = page.time_filter if page is not None else model.TimeFilterSpec()
        presets = st.multiselect(
            "Available time ranges",
            all_presets,
            default=list(seed_time.presets),
            format_func=lambda value: value.replace("_", " ").title(),
            key=f"builder_page_time_presets_{key_suffix}",
            help=config_help.field_help("report.available_ranges"),
        )
        default_options = presets or ["all_time"]
        default = st.selectbox(
            "Default time range",
            default_options,
            index=builder.option_index(default_options, seed_time.default),
            format_func=lambda value: value.replace("_", " ").title(),
            key=f"builder_page_time_default_{key_suffix}",
            help=config_help.field_help("report.default_range"),
        )
    ready = bool(dashboard_id.strip() and page_id.strip() and presets)
    return filters, {"default": default, "presets": presets}, ready


def _page_filter_rows(edited: Any) -> list[dict[str, Any]]:
    if hasattr(edited, "to_dict"):
        values = edited.to_dict(orient="records")
    elif isinstance(edited, list):
        values = edited
    else:
        values = []
    defaults = {
        "label": "",
        "display": "secondary",
        "scope": "compatible_tiles",
        "control": "multiselect",
    }
    return [
        {
            "field": str(row.get("field", "")).strip(),
            **{key: str(row.get(key) or value).strip() for key, value in defaults.items()},
        }
        for row in values
        if isinstance(row, Mapping) and str(row.get("field", "")).strip()
    ]


def _tile_field_controls(
    chart_kind: str,
    defaults: dict[str, Any],
    field_options: list[str],
    *,
    key_suffix: str,
    catalog: model.Catalog | None = None,
    metric_name: str | None = None,
) -> dict[str, Any]:
    fields = _field_controls_for_keys(
        builder.chart_field_controls(chart_kind),
        chart_kind,
        defaults,
        field_options,
        key_suffix,
        catalog,
        metric_name,
    )
    fields.update(_chart_setting_controls(chart_kind, defaults, field_options, key_suffix))
    return fields


def _field_controls_for_keys(
    keys: tuple[str, ...],
    chart_kind: str,
    defaults: dict[str, Any],
    field_options: list[str],
    key_suffix: str,
    catalog: model.Catalog | None,
    metric_name: str | None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in keys:
        if key == "stages":
            fields[key] = _stage_list_control(defaults.get("stages"), key_suffix)
        elif key in {"path", "columns", "group_by"}:
            fields[key] = st.multiselect(
                _field_label(key),
                field_options[1:],
                default=_default_multiselect_values(defaults.get(key), field_options),
                key=f"builder_tile_{key}_{key_suffix}",
                help=config_help.field_help("report.field"),
            )
        else:
            options = _tile_field_options(
                chart_kind,
                key,
                defaults,
                field_options,
                fields,
                catalog,
                metric_name,
            )
            fields[key] = st.selectbox(
                _field_label(key),
                options,
                index=builder.option_index(options, _tile_field_default(defaults, key)),
                key=f"builder_tile_{key}_{key_suffix}",
                help=config_help.field_help("report.field"),
            )
    return fields


def _field_label(key: str) -> str:
    return key.upper() if len(key) == 1 else key.title()


def _default_multiselect_values(value: Any, field_options: list[str]) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item) for item in value if isinstance(item, str) and item in field_options]


def _tile_field_options(
    chart_kind: str,
    key: str,
    defaults: dict[str, Any],
    field_options: list[str],
    selected_fields: dict[str, Any],
    catalog: model.Catalog | None,
    metric_name: str | None,
) -> list[str]:
    if chart_kind.startswith("descriptive_") and key == "property":
        properties = (
            builder.descriptive_property_options(catalog, metric_name)
            if catalog is not None and metric_name
            else []
        )
        return properties or field_options
    if chart_kind.startswith("descriptive_") and key == "score":
        property_name = str(selected_fields.get("property") or defaults.get("property") or "")
        scores = (
            builder.descriptive_score_options(catalog, metric_name, property_name)
            if catalog is not None and metric_name
            else []
        )
        return scores or ["Mean"]
    return field_options


def _stage_list_control(value: Any, key_suffix: str) -> list[str]:
    raw = st.text_input(
        "Stages",
        value=", ".join(_stage_values(value)),
        key=f"builder_tile_stages_{key_suffix}",
        help=config_help.field_help("report.stages"),
    )
    return builder.csv_text_to_list(raw)


def _stage_values(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    stages: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            name = item.get("name")
            if name not in (None, ""):
                stages.append(str(name))
        elif item not in (None, ""):
            stages.append(str(item))
    return stages


def _selectbox_fields(
    keys: tuple[str, ...],
    defaults: dict[str, Any],
    field_options: list[str],
    key_suffix: str,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in keys:
        fields[key] = st.selectbox(
            key.upper() if len(key) == 1 else key.title(),
            field_options,
            index=builder.option_index(field_options, _tile_field_default(defaults, key)),
            key=f"builder_tile_{key}_{key_suffix}",
            help=config_help.field_help("report.field"),
        )
    return fields


def _chart_setting_controls(  # noqa: PLR0912
    chart_kind: str,
    defaults: dict[str, Any],
    field_options: list[str],
    key_suffix: str,
) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    with st.expander("Chart Settings", expanded=False):
        description = st.text_area(
            "Description",
            value=str(defaults.get("description", "") or ""),
            height=80,
            key=f"builder_tile_description_{key_suffix}",
            help=config_help.field_help("report.description"),
        )
        if description.strip():
            settings["description"] = description.strip()

        format_options = ["", "percent", "integer", "number", "currency"]
        selected_format = st.selectbox(
            "Value Format",
            format_options,
            index=builder.option_index(format_options, defaults.get("value_format")),
            format_func=lambda value: "Default" if value == "" else value.title(),
            key=f"builder_tile_value_format_{key_suffix}",
            help=config_help.field_help("report.value_format"),
        )
        if selected_format:
            settings["value_format"] = selected_format

        if chart_kind in {"line", "stacked_area"}:
            scale_modes = ["absolute", "index_100", "percent_change"]
            scale_mode = st.selectbox(
                "Scale",
                scale_modes,
                index=builder.option_index(scale_modes, defaults.get("scale_mode") or "absolute"),
                format_func=lambda value: value.replace("_", " ").title(),
                key=f"builder_tile_scale_mode_{key_suffix}",
                help=config_help.field_help("report.scale"),
            )
            if scale_mode != "absolute":
                settings["scale_mode"] = scale_mode

        if chart_kind == "kpi_card":
            settings.update(_kpi_card_settings(defaults, key_suffix))

        if chart_kind == "gauge":
            settings.update(_gauge_reference_settings(defaults, key_suffix))

        show_trend_delta = st.checkbox(
            "Show Trend Delta",
            value=bool(defaults.get("show_trend_delta", defaults.get("trend_delta", False))),
            key=f"builder_tile_trend_delta_{key_suffix}",
            help=config_help.field_help("report.show_trend_delta"),
        )
        if show_trend_delta:
            settings["show_trend_delta"] = True

        goal_enabled = st.checkbox(
            "Goal Line",
            value=defaults.get("goal_line") not in (None, "", []),
            key=f"builder_tile_goal_enabled_{key_suffix}",
            help=config_help.field_help("report.goal_enabled"),
        )
        if goal_enabled:
            goal_value, goal_label, goal_color = _goal_line_defaults(defaults.get("goal_line"))
            settings["goal_line"] = {
                "value": st.number_input(
                    "Goal Value",
                    value=goal_value,
                    key=f"builder_tile_goal_value_{key_suffix}",
                    help=config_help.field_help("report.goal_value"),
                ),
                "label": st.text_input(
                    "Goal Label",
                    value=goal_label,
                    key=f"builder_tile_goal_label_{key_suffix}",
                    help=config_help.field_help("report.goal_label"),
                ),
                "color": st.color_picker(
                    "Goal Color",
                    value=goal_color,
                    key=f"builder_tile_goal_color_{key_suffix}",
                    help=config_help.field_help("report.goal_color"),
                ),
            }

        if chart_kind == "bar":
            barmode_options = ["group", "stack", "relative", "percent"]
            selected_barmode = st.selectbox(
                "Bar Mode",
                barmode_options,
                index=builder.option_index(barmode_options, defaults.get("barmode") or "group"),
                key=f"builder_tile_barmode_{key_suffix}",
                help=config_help.field_help("report.bar_mode"),
            )
            if selected_barmode != "group" or defaults.get("barmode"):
                settings["barmode"] = selected_barmode
            sort_by = st.selectbox(
                "Sort By",
                field_options,
                index=builder.option_index(field_options, defaults.get("sort_by")),
                key=f"builder_tile_sort_by_{key_suffix}",
                help=config_help.field_help("report.sort_by"),
            )
            if sort_by:
                settings["sort_by"] = sort_by
                settings["sort_direction"] = st.selectbox(
                    "Sort Direction",
                    ["desc", "asc"],
                    index=builder.option_index(
                        ["desc", "asc"], defaults.get("sort_direction") or "desc"
                    ),
                    key=f"builder_tile_sort_direction_{key_suffix}",
                    help=config_help.field_help("report.sort_direction"),
                )
            top_n = st.number_input(
                "Top N",
                min_value=0,
                value=int(defaults.get("top_n") or 0),
                step=1,
                key=f"builder_tile_top_n_{key_suffix}",
                help=config_help.field_help("report.top_n"),
            )
            if top_n:
                settings["top_n"] = int(top_n)

        rules_text = st.text_area(
            "Conditional Formatting YAML",
            value=_conditional_formatting_text(defaults.get("conditional_formatting")),
            height=120,
            placeholder='- column: CTR\n  operator: ">="\n  value: 0.12\n  color: "#2e7d32"',
            key=f"builder_tile_conditional_{key_suffix}",
            help=config_help.field_help("report.conditional_formatting"),
        )
        if rules_text.strip():
            try:
                rules = yaml.safe_load(rules_text)
                if isinstance(rules, list):
                    settings["conditional_formatting"] = rules
                else:
                    st.warning("Conditional formatting must be a YAML list.")
            except Exception as exc:
                logger.exception("Failed to parse conditional formatting YAML")
                st.warning(str(exc))
    return settings


def _kpi_card_settings(defaults: dict[str, Any], key_suffix: str) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    placement = st.selectbox(
        "Placement",
        ["content", "kpi_strip"],
        index=builder.option_index(
            ["content", "kpi_strip"], defaults.get("placement") or "content"
        ),
        format_func=lambda value: "KPI strip" if value == "kpi_strip" else "Report content",
        key=f"builder_tile_placement_{key_suffix}",
        help=config_help.field_help("report.placement"),
    )
    if placement != "kpi_strip":
        return settings
    settings["placement"] = placement
    raw_kpi = defaults.get("kpi")
    seed = dict(raw_kpi) if isinstance(raw_kpi, Mapping) else {}
    comparison = st.selectbox(
        "Comparison",
        ["none", "previous_period"],
        index=builder.option_index(["none", "previous_period"], seed.get("comparison") or "none"),
        format_func=lambda value: value.replace("_", " ").title(),
        key=f"builder_tile_kpi_comparison_{key_suffix}",
        help=config_help.field_help("report.comparison"),
    )
    comparison_period = st.selectbox(
        "Comparison period",
        ["day", "week", "month", "quarter", "year"],
        index=builder.option_index(
            ["day", "week", "month", "quarter", "year"],
            seed.get("comparison_period") or "month",
        ),
        disabled=comparison == "none",
        key=f"builder_tile_kpi_period_{key_suffix}",
        help=config_help.field_help("report.comparison_period"),
    )
    sparkline_options = ["", "daily", "weekly", "monthly"]
    sparkline_grain = st.selectbox(
        "Sparkline grain",
        sparkline_options,
        index=builder.option_index(sparkline_options, seed.get("sparkline_grain") or ""),
        format_func=lambda value: "None" if not value else value.title(),
        key=f"builder_tile_kpi_sparkline_{key_suffix}",
        help=config_help.field_help("report.sparkline_grain"),
    )
    sparkline_points = st.number_input(
        "Sparkline points",
        min_value=2,
        max_value=366,
        value=int(seed.get("sparkline_points") or 30),
        disabled=not sparkline_grain,
        key=f"builder_tile_kpi_points_{key_suffix}",
        help=config_help.field_help("report.sparkline_points"),
    )
    target_enabled = st.checkbox(
        "Target",
        value=seed.get("target") is not None,
        key=f"builder_tile_kpi_target_enabled_{key_suffix}",
        help=config_help.field_help("report.target_enabled"),
    )
    target = (
        st.number_input(
            "Target value",
            value=float(seed.get("target") or 0.0),
            key=f"builder_tile_kpi_target_{key_suffix}",
            help=config_help.field_help("report.target_value"),
        )
        if target_enabled
        else None
    )
    settings["kpi"] = {
        "comparison": comparison,
        "comparison_period": comparison_period,
        "sparkline_grain": sparkline_grain or None,
        "sparkline_points": int(sparkline_points),
        "target": target,
    }
    return settings


def _tile_field_default(defaults: dict[str, Any], key: str) -> Any:
    value = defaults.get(key)
    if value not in (None, ""):
        return value
    if key == "facet_col":
        value = defaults.get("facet_column")
        if value not in (None, ""):
            return value
    facets = defaults.get("facets")
    if isinstance(facets, dict):
        if key == "facet_row":
            return facets.get("row")
        if key == "facet_col":
            return facets.get("col", facets.get("column"))
    group_by = defaults.get("group_by")
    if isinstance(group_by, (list, tuple)):
        if key == "facet_row" and group_by:
            value = group_by[0]
        elif key == "facet_col" and len(group_by) > 1:
            value = group_by[1]
    return value


def _goal_line_defaults(raw: Any) -> tuple[float, str, str]:
    if isinstance(raw, dict):
        return (
            float(raw.get("value", 0.0) or 0.0),
            str(raw.get("label", "Goal")),
            str(raw.get("color", "#475569")),
        )
    if isinstance(raw, int | float):
        return float(raw), "Goal", "#475569"
    return 0.0, "Goal", "#475569"


def _gauge_reference_settings(defaults: dict[str, Any], key_suffix: str) -> dict[str, Any]:
    reference_value = _reference_number(defaults.get("reference"))
    reference_enabled = st.checkbox(
        "Reference",
        value=reference_value is not None,
        key=f"builder_tile_reference_enabled_{key_suffix}",
        help=config_help.field_help("report.reference_enabled"),
    )
    if reference_enabled:
        return {
            "reference": st.number_input(
                "Reference Value",
                value=reference_value or 0.0,
                key=f"builder_tile_reference_value_{key_suffix}",
                help=config_help.field_help("report.reference_value"),
            )
        }
    if isinstance(defaults.get("references"), Mapping):
        return {"references": defaults["references"]}
    return {}


def _reference_number(raw: Any) -> float | None:
    if isinstance(raw, (bool, Mapping)) or raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _conditional_formatting_text(raw: Any) -> str:
    if not raw:
        return ""
    return yaml.safe_dump(raw, sort_keys=False).strip()


def _preview_tile(workspace: Path, catalog: model.Catalog, tile: dict[str, Any]) -> None:
    try:
        parsed = model.Tile.model_validate(tile)
        rows = query_tile(workspace, catalog, parsed)
        figure = render_chart(rows, tile, theme={**dashboard_theme(), **catalog.dashboards.theme})
        st.plotly_chart(
            figure,
            width="stretch",
            theme=None,
            key=f"builder_tile_preview_{tile.get('id', 'tile')}",
        )
    except Exception as exc:  # pragma: no cover - Streamlit display path
        logger.exception("Failed to preview tile: tile=%s", tile.get("id"))
        st.warning(str(exc))


def _chat_metric_rows(catalog: model.Catalog) -> list[dict[str, str]]:
    processors = {processor.id: processor for processor in catalog.processors.processors}
    rows: list[dict[str, str]] = []
    for metric_name, metric in sorted(catalog.metrics.metrics.items(), key=lambda item: item[0]):
        metric_def = builder.metric_to_dict(metric)
        processor_id = str(metric_def.get("source", "") or "")
        processor = processors.get(processor_id)
        rows.append(
            {
                "Metric": metric_name,
                "Kind": str(metric_def.get("kind", "") or ""),
                "Processor": processor_id,
                "Group By": ", ".join(processor.group_by) if processor else "",
            }
        )
    return rows


def _chat_description_rows(
    catalog_keys: list[tuple[str, str]],
    descriptions: Mapping[str, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for key, item_type in catalog_keys:
        rows.append(
            {
                "Type": item_type,
                "Key": key,
                "Description": str(descriptions.get(key, "")),
            }
        )
        seen.add(key)
    for key, description in descriptions.items():
        if key in seen:
            continue
        rows.append({"Type": "Custom", "Key": key, "Description": description})
    return rows


def _chat_description_map(rows: Any) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for row in builder.normalize_editor_rows(rows):
        key = str(row.get("Key", "") or "").strip()
        description = str(row.get("Description", "") or "").strip()
        if key and description:
            descriptions[key] = description
    return dict(sorted(descriptions.items(), key=lambda item: item[0].casefold()))


@st.fragment()
def _chat_review(ctx: ValueStreamContext, save_slot: Any, draft_slot: Any) -> None:
    _claim_fragment_action_slots(save_slot, draft_slot)
    config_path, ai_config = load_llm_settings_config(ctx.workspace)
    chat_config_path, chat_config = load_chat_with_data_config(ctx.workspace)
    settings_label = "Configured" if ai_config.get("model") else "Session-only"
    components.metric_strip(
        [
            {"label": "Metrics", "value": len(ctx.catalog.metrics.metrics)},
            {"label": "Processors", "value": len(ctx.catalog.processors.processors)},
            {"label": "Catalog", "value": "Valid" if ctx.validation.ok else "Review"},
            {"label": "LLM Settings", "value": settings_label},
        ]
    )
    if config_path is not None:
        st.caption(f"Chat defaults loaded from `{config_path.name}`.")
    elif chat_config_path is not None:
        st.caption(f"Chat guidance loaded from `{chat_config_path.name}`.")
    st.caption(
        "Chat With Data plans questions against aggregate metrics in the active catalog. "
        "Raw source rows are not exposed to chat."
    )
    rows = _chat_metric_rows(ctx.catalog)
    if not rows:
        st.info("Add aggregate metrics before using Chat With Data.")
        _chat_settings_editor(
            ctx,
            chat_config,
            save_slot=save_slot,
            draft_slot=draft_slot,
        )
        return
    st.dataframe(rows, hide_index=True, width="stretch", height=420)
    if not ctx.validation.ok:
        st.warning("Resolve catalog validation issues before relying on chat answers.")
    _chat_settings_editor(
        ctx,
        chat_config,
        save_slot=save_slot,
        draft_slot=draft_slot,
    )


def _chat_settings_editor(
    ctx: ValueStreamContext,
    chat_config: Mapping[str, Any],
    *,
    save_slot: Any,
    draft_slot: Any,
) -> None:
    agent_prompt = str(chat_config.get("agent_prompt") or DEFAULT_CHAT_AGENT_PROMPT)
    metric_descriptions = {
        str(key): str(value)
        for key, value in dict(chat_config.get("metric_descriptions") or {}).items()
    }
    dataset_descriptions = {
        str(key): str(value)
        for key, value in dict(chat_config.get("dataset_descriptions") or {}).items()
    }
    components.sync_text_area("builder_chat_agent_prompt", agent_prompt)

    with components.bordered_panel(
        "Chat Prompt",
        "Edit chat-only guidance sent to the LLM planner. Governed query rules still apply.",
    ):
        edited_prompt = st.text_area(
            "Agent Prompt",
            key="builder_chat_agent_prompt",
            height=220,
            help=config_help.field_help("chat.agent_prompt"),
        )
        st.caption(
            "Use this for business context and terminology. It does not allow raw rows, SQL, "
            "Python execution, or arbitrary chart code."
        )

    dataset_rows = _chat_description_rows(
        [(source.id, "Dataset") for source in ctx.catalog.pipelines.sources],
        dataset_descriptions,
    )
    metric_keys = [
        *[(processor.id, "Processor") for processor in ctx.catalog.processors.processors],
        *[(name, "Metric") for name in sorted(ctx.catalog.metrics.metrics, key=str.casefold)],
    ]
    metric_rows = _chat_description_rows(metric_keys, metric_descriptions)

    with components.bordered_panel(
        "Chat Descriptions",
        "Correct dataset, processor, and metric descriptions for LLM planning only.",
    ):
        edited_dataset_rows = st.data_editor(
            dataset_rows,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key="builder_chat_dataset_descriptions",
            column_config={
                "Type": st.column_config.TextColumn(
                    "Type",
                    disabled=True,
                    width="small",
                    help=config_help.field_help("chat.description_type"),
                ),
                "Key": st.column_config.TextColumn(
                    "Key", width="medium", help=config_help.field_help("chat.description_key")
                ),
                "Description": st.column_config.TextColumn(
                    "Description",
                    width="large",
                    help=config_help.field_help("chat.description"),
                ),
            },
        )
        edited_metric_rows = st.data_editor(
            metric_rows,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key="builder_chat_metric_descriptions",
            column_config={
                "Type": st.column_config.TextColumn(
                    "Type",
                    disabled=True,
                    width="small",
                    help=config_help.field_help("chat.description_type"),
                ),
                "Key": st.column_config.TextColumn(
                    "Key", width="medium", help=config_help.field_help("chat.description_key")
                ),
                "Description": st.column_config.TextColumn(
                    "Description",
                    width="large",
                    help=config_help.field_help("chat.description"),
                ),
            },
        )

    proposed_chat_config = {
        "agent_prompt": edited_prompt,
        "dataset_descriptions": _chat_description_map(edited_dataset_rows),
        "metric_descriptions": _chat_description_map(edited_metric_rows),
    }
    baseline_chat_config = {
        "agent_prompt": agent_prompt,
        "dataset_descriptions": dataset_descriptions,
        "metric_descriptions": metric_descriptions,
    }
    draft_status = builder.builder_draft_status(
        "chat:guidance",
        baseline_chat_config,
        proposed_chat_config,
    )
    if _render_editor_primary_action(
        save_slot=save_slot,
        draft_slot=draft_slot,
        status=draft_status,
        valid=True,
        widget_prefixes=("builder_chat_",),
        help_text="Apply Chat With Data guidance to the active workspace.",
    ):
        try:
            write_chat_with_data_config(
                ctx.workspace,
                agent_prompt=edited_prompt,
                dataset_descriptions=proposed_chat_config["dataset_descriptions"],
                metric_descriptions=proposed_chat_config["metric_descriptions"],
            )
            _complete_builder_apply(
                status=draft_status,
                scope="chat",
                label="Chat guidance",
                message="Chat guidance applied to the workspace.",
            )
        except Exception as exc:  # pragma: no cover - Streamlit display path
            _record_builder_apply_failed(exc)
            logger.exception("Failed to write Chat With Data guidance")
            st.error(str(exc))


@st.fragment()
def _settings_builder(ctx: ValueStreamContext, save_slot: Any, draft_slot: Any) -> None:
    _claim_fragment_action_slots(save_slot, draft_slot)
    defaults = ctx.catalog.pipelines.defaults
    calendar = defaults.calendar
    known_grains = ["Day", "Month", "Quarter", "Year", "Summary"]
    grain_options = builder.dedupe([*known_grains, *calendar.grains])
    with components.bordered_panel(
        "Workspace Defaults",
        "Edit shared defaults used by source processing and generated reports.",
    ):
        workspace_name = st.text_input(
            "Workspace Name",
            value=ctx.catalog.pipelines.workspace,
            key="builder_settings_workspace",
            help=config_help.field_help("workspace.name"),
        )
        time_zone = st.text_input(
            "Time Zone",
            value=defaults.time_zone,
            key="builder_settings_time_zone",
            help=config_help.field_help("workspace.time_zone"),
        )
        selected_grains = st.multiselect(
            "Calendar Grains",
            grain_options,
            default=[grain for grain in calendar.grains if grain in grain_options],
            key="builder_settings_calendar_grains",
            help=config_help.field_help("workspace.calendar_grains"),
        )
        week_start = st.selectbox(
            "Week Start",
            ["monday", "sunday"],
            index=builder.option_index(["monday", "sunday"], calendar.week_start),
            key="builder_settings_week_start",
            help=config_help.field_help("workspace.week_start"),
        )

    theme_text = yaml.safe_dump(ctx.catalog.dashboards.theme, sort_keys=False)
    components.sync_text_area("builder_settings_theme_yaml", theme_text)
    theme: dict[str, Any] = {}
    theme_error: str | None = None
    with st.expander("Technical details · Dashboard theme YAML", expanded=False):
        st.caption("Edit permissive Plotly/dashboard theme tokens stored in dashboards.yaml.")
        raw_theme = st.text_area(
            "Theme YAML",
            key="builder_settings_theme_yaml",
            height=220,
            help=config_help.field_help("workspace.theme_yaml"),
        )
        try:
            parsed = yaml.safe_load(raw_theme) or {}
            if not isinstance(parsed, dict):
                raise ValueError("theme must be a YAML mapping")
            theme = parsed
        except Exception as exc:
            logger.exception("Failed to parse builder dashboard theme YAML")
            theme_error = str(exc)
            st.warning(theme_error)

    st.caption("These values remain a session draft until you apply them to the workspace.")
    if not selected_grains:
        st.warning("Select at least one calendar grain.")
    baseline_settings = {
        "workspace_name": ctx.catalog.pipelines.workspace,
        "time_zone": defaults.time_zone,
        "calendar_grains": list(calendar.grains),
        "week_start": calendar.week_start,
        "dashboard_theme": ctx.catalog.dashboards.theme,
    }
    proposed_settings = {
        "workspace_name": workspace_name,
        "time_zone": time_zone,
        "calendar_grains": selected_grains,
        "week_start": week_start,
        "dashboard_theme": theme,
    }
    draft_status = builder.builder_draft_status(
        "settings:workspace",
        baseline_settings,
        proposed_settings,
    )
    if _render_editor_primary_action(
        save_slot=save_slot,
        draft_slot=draft_slot,
        status=draft_status,
        valid=not theme_error and bool(selected_grains),
        widget_prefixes=("builder_settings_",),
        help_text=(
            "Validate and apply workspace defaults and dashboard theme. Data Load remains "
            "a separate action."
        ),
    ):
        try:
            with builder.validated_catalog_transaction(ctx.workspace):
                builder.write_workspace_settings(
                    ctx.workspace,
                    workspace_name=workspace_name,
                    time_zone=time_zone,
                    calendar_grains=selected_grains,
                    week_start=week_start,
                    dashboard_theme=theme,
                )
            _complete_builder_apply(
                status=draft_status,
                scope="workspace_settings",
                label="Workspace settings",
                source_ids=[source.id for source in ctx.catalog.pipelines.sources],
                message="Workspace settings applied. Use Data Load to refresh affected sources.",
            )
        except Exception as exc:  # pragma: no cover - Streamlit display path
            _record_builder_apply_failed(exc)
            logger.exception("Failed to write workspace settings")
            st.error(str(exc))


@st.fragment()
def _export_current_workspace(ctx: ValueStreamContext, save_slot: Any) -> None:
    save_slot.empty()
    save_slot.button(
        "Workspace current",
        icon=":material/check_circle:",
        disabled=True,
        width="stretch",
        key="builder_export_current",
    )
    st.write("### What happens next")
    _render_outcome_handoff()
    st.write("### Current workspace validation")
    components.render_validation_summary(ctx.validation.issues, ok=ctx.validation.ok)
    filenames = [
        "pipelines.yaml",
        "processors.yaml",
        "metrics.yaml",
        "dashboards.yaml",
    ]
    file_text: dict[str, str] = {}
    for filename in filenames:
        path = ctx.workspace / "catalog" / filename
        if path.exists():
            file_text[filename] = path.read_text(encoding="utf-8")

    st.write("### Download catalog files")
    st.caption("Downloads are available before the optional file previews.")
    download_cols = st.columns(2)
    for index, filename in enumerate(filenames):
        with download_cols[index % 2]:
            if filename in file_text:
                st.download_button(
                    f"Download {filename}",
                    data=file_text[filename],
                    file_name=filename,
                    mime="text/yaml",
                    key=f"builder_download_{filename}",
                    width="stretch",
                )
            else:
                st.warning(f"{filename} does not exist.")

    st.write("### Technical details")
    st.caption("File previews are collapsed so large catalogs do not bury the actions above.")
    for filename in filenames:
        with st.expander(filename, expanded=False):
            if filename in file_text:
                st.code(file_text[filename], language="yaml")
            else:
                st.warning(f"{filename} does not exist.")


def _metric_mode_options(metric_names: list[str]) -> list[str]:
    if metric_names:
        return [METRIC_ACTION_CREATE, METRIC_ACTION_EDIT]
    return [METRIC_ACTION_CREATE]


def _queue_metric_refresh(
    session_state: MutableMapping[str, Any],
    *,
    metric_id: str,
    metric_def: Mapping[str, Any],
    message: str,
    issues: list[str],
    materialization: recipe_library.RecipeMaterializationPlan | None = None,
) -> None:
    """Queue a full-catalog refresh and open the written metric afterwards."""
    session_state["builder_metric_pending_refresh"] = {
        "metric_id": metric_id,
        "source": str(metric_def.get("source", "") or ""),
        "kind": str(metric_def.get("kind", "") or ""),
        "message": message,
        "issues": list(issues),
        "materialization": asdict(materialization) if materialization else None,
    }


def _consume_pending_metric_refresh(
    session_state: MutableMapping[str, Any],
    metric_defs_by_name: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Restore the editor selection after a write-triggered full app rerun."""
    raw = session_state.pop("builder_metric_pending_refresh", None)
    if not isinstance(raw, Mapping):
        return {}
    feedback = dict(raw)
    metric_id = str(feedback.get("metric_id", "") or "")
    metric_def = metric_defs_by_name.get(metric_id)
    if metric_def is None:
        return feedback
    source = str(metric_def.get("source", feedback.get("source", "")) or "")
    kind = str(metric_def.get("kind", feedback.get("kind", "")) or "")
    session_state["builder_metric_mode"] = METRIC_ACTION_EDIT
    session_state["builder_metric_selected_id"] = metric_id
    if source:
        session_state["builder_metric_processor_edit"] = source
    if source and kind:
        session_state[f"builder_metric_kind_edit_{source}"] = kind
        session_state[f"builder_metric_select_{source}_{kind}"] = metric_id
    return feedback


def _render_metric_write_feedback(feedback: Mapping[str, Any]) -> None:
    message = str(feedback.get("message", "") or "Metric written.")
    issues = feedback.get("issues", [])
    issue_lines = [str(issue) for issue in issues] if isinstance(issues, list) else []
    if issue_lines:
        st.warning(f"{message} The catalog still needs attention.")
        st.code("\n".join(issue_lines), language="text")
    else:
        st.success(message, icon=":material/check_circle:")
    materialization = feedback.get("materialization")
    if not isinstance(materialization, Mapping):
        return
    source_id = str(materialization.get("source_id", "") or "")
    state_names = [str(value) for value in materialization.get("state_names", []) if str(value)]
    state_text = ", ".join(f"`{name}`" for name in state_names)
    st.warning(
        f"Metric configuration is complete, but {state_text or 'its new aggregate state'} "
        f"is not materialized yet. Run source `{source_id}` before using the metric in "
        "reports."
    )
    current_hash = str(materialization.get("current_computation_hash", "") or "")
    proposed_hash = str(materialization.get("proposed_computation_hash", "") or "")
    if current_hash and proposed_hash:
        st.caption(f"Processor computation hash: `{current_hash[:12]}` → `{proposed_hash[:12]}`.")
    st.link_button(
        "Open Data Load to materialize",
        "/data_load",
        icon=":material/database_upload:",
        type="primary",
    )


def _processor_choice_label(value: str, processors_by_id: dict[str, model.Processor]) -> str:
    if not value:
        return "Select processor"
    processor = processors_by_id[value]
    return _processor_choice_label_human(processor)


def _source_choice_label(source: model.Source) -> str:
    """Show the human source name before reader and technical identity."""
    label = source.description.strip() or humanize_identifier(source.id)
    return f"{label} · {humanize_identifier(source.reader.kind)}"


def _processor_choice_label_human(processor: model.Processor) -> str:
    """Show the human processor name before its technical kind."""
    label = processor.description.strip() or humanize_identifier(processor.id)
    return f"{label} · {humanize_identifier(processor.kind)}"


def _metric_processors_for_definitions(
    processors: list[model.Processor],
    metric_defs_by_name: Mapping[str, Mapping[str, Any]],
) -> list[model.Processor]:
    metric_sources = {
        str(metric_def.get("source", "") or "") for metric_def in metric_defs_by_name.values()
    }
    return [processor for processor in processors if processor.id in metric_sources]


def _metric_kinds_for_source(
    metric_defs_by_name: Mapping[str, Mapping[str, Any]], source: str
) -> list[str]:
    kinds = [
        str(metric_def.get("kind", "") or "")
        for metric_def in metric_defs_by_name.values()
        if metric_def.get("source") == source
    ]
    return sorted(builder.dedupe([kind for kind in kinds if kind]), key=builder.metric_kind_label)


def _metric_names_for_source_kind(
    metric_defs_by_name: Mapping[str, Mapping[str, Any]], source: str, kind: str
) -> list[str]:
    return sorted(
        [
            name
            for name, metric_def in metric_defs_by_name.items()
            if metric_def.get("source") == source and metric_def.get("kind") == kind
        ],
        key=str.casefold,
    )


def _dashboard_choice_label(value: str, dashboards_by_id: dict[str, model.Dashboard]) -> str:
    if value == NEW_DASHBOARD_KEY:
        return "New dashboard"
    dashboard = dashboards_by_id[value]
    return dashboard.title or dashboard.id


def _page_choice_label(value: str, pages_by_id: dict[str, model.DashboardPage]) -> str:
    if value == NEW_PAGE_KEY:
        return "New page"
    page = pages_by_id[value]
    return page.title or page.id


def _metric_choice_label(catalog: model.Catalog, metric_name: str) -> str:
    metric = catalog.metrics.metrics.get(metric_name)
    if metric is None:
        return metric_name
    metric_def = builder.metric_to_dict(metric)
    source = str(metric_def.get("source", "") or "unknown")
    kind = str(metric_def.get("kind", "") or "unknown")
    display = metric_def.get("display")
    display_label = str(display.get("label", "") or "") if isinstance(display, dict) else ""
    label = display_label or humanize_identifier(metric_name)
    return f"{label} · {builder.metric_kind_label(kind)} · {humanize_identifier(source)}"


def _stable_random_suffix(session_state: MutableMapping[str, Any], key: str) -> str:
    existing = session_state.get(key)
    if isinstance(existing, str) and existing:
        return existing
    suffix = secrets.token_hex(8)
    session_state[key] = suffix
    return suffix


def _generated_catalog_id(name: str, suffix: str, *, fallback: str) -> str:
    return builder.generated_catalog_id(name, suffix, fallback=fallback)


def _digest_state_label(processor: model.Processor, state_name: str) -> str:
    spec = model.effective_processor_states(processor).get(state_name)
    if spec is None:
        return state_name
    extra = dict(spec.model_extra or {})
    source = str(extra.get("source_column") or extra.get("score") or "")
    outcome = str(extra.get("outcome", "") or "")
    details = ", ".join(item for item in (source, outcome) if item)
    return f"{state_name} ({details})" if details else state_name


def _show_validation_after_write(workspace: Path, success_message: str) -> None:
    ok, issues = builder.validate_workspace(workspace)
    if ok:
        st.toast(success_message, icon=":material/check:")
        st.success(f"{success_message} Catalog validates.")
    else:
        st.warning(f"{success_message} Catalog needs attention.")
        st.code("\n".join(issues), language="text")


def _source_field_options(
    ctx: ValueStreamContext,
    source: model.Source | None,
    *,
    rename_capitalize: bool | None = None,
) -> list[str]:
    if source is None:
        return []
    use_rename_capitalize = (
        _source_has_transform(source, "rename_capitalize")
        if rename_capitalize is None
        else rename_capitalize
    )
    fields: list[str] = []
    fields.extend(_source_sample_columns(ctx, source, rename_capitalize=use_rename_capitalize))
    if source.schema_.timestamp_column:
        fields.append(
            _rename_capitalize_field(source.schema_.timestamp_column, use_rename_capitalize)
        )
    fields.extend(_rename_capitalize_fields(source.schema_.natural_key, use_rename_capitalize))
    fields.extend(_rename_capitalize_fields(source.schema_.drop_columns, use_rename_capitalize))
    fields.extend(_rename_capitalize_fields(builder.source_defaults(source), use_rename_capitalize))
    for transform in source.transforms:
        if isinstance(transform, model.RenameCapitalize):
            continue
        if isinstance(transform, model.ParseDatetime):
            fields.extend(_rename_capitalize_fields(transform.columns, use_rename_capitalize))
        elif isinstance(transform, model.DeriveCalendar):
            fields.append(_rename_capitalize_field(transform.from_, use_rename_capitalize))
            fields.extend(transform.outputs)
        elif isinstance(transform, model.DeriveActionId):
            fields.extend(_rename_capitalize_fields(transform.parts, use_rename_capitalize))
            fields.append("ActionID")
        elif isinstance(transform, model.DeriveColumn):
            fields.append(transform.output)
        elif isinstance(transform, model.Cast | model.DropColumns):
            fields.extend(_rename_capitalize_fields(transform.columns, use_rename_capitalize))
        elif isinstance(transform, model.Coalesce):
            fields.extend(_rename_capitalize_fields(transform.columns, use_rename_capitalize))
            fields.append(transform.output)
    for processor in processors_for_source(ctx, source.id):
        fields.extend(processor.group_by)
        if processor.time and processor.time.column:
            fields.append(processor.time.column)
        fields.extend(dimension_profile.processor_field_references(processor))
    return sorted(
        builder.dedupe([str(field) for field in fields if field]),
        key=lambda field: (field.casefold(), field),
    )


def _source_sample_columns(
    ctx: ValueStreamContext,
    source: model.Source,
    *,
    rename_capitalize: bool = False,
) -> list[str]:
    try:
        chunks = discover(ctx.workspace, source)
        if not chunks:
            return []
        frame = read(source.reader, chunks[0].files)
        return _rename_capitalize_fields(frame.collect_schema().names(), rename_capitalize)
    except Exception:
        logger.exception("Failed to inspect source sample columns: source=%s", source.id)
        return []
    finally:
        cleanup_temporaries()


def _rename_capitalize_field(field: str, enabled: bool) -> str:
    if not enabled or not field:
        return field
    return capitalize_fields([field])[0]


def _rename_capitalize_fields(fields: Any, enabled: bool) -> list[str]:
    values = [str(field) for field in fields if str(field)]
    if not enabled:
        return values
    return capitalize_fields(values)


def _sync_source_rename_capitalize_state(
    ctx: ValueStreamContext,
    source: model.Source,
    enabled: bool,
) -> None:
    state_key = f"builder_source_rename_capitalize_applied_{source.id}"
    previous = st.session_state.get(state_key)
    if previous is None:
        st.session_state[state_key] = enabled
        return
    if bool(previous) == enabled:
        return

    mapping = _source_rename_mapping(ctx, source, enabled)
    if mapping:
        field_remap.remap_state_field(f"builder_source_ts_{source.id}", mapping)
        field_remap.remap_state_field_list(f"builder_source_natural_{source.id}", mapping)
        field_remap.remap_state_field_list(f"builder_source_drop_{source.id}", mapping)
        field_remap.remap_state_rows(f"builder_source_defaults_{source.id}", mapping, ("Field",))
        field_remap.remap_state_rows(f"builder_source_filter_rows_{source.id}", mapping, ("Field",))
        field_remap.remap_state_raw_expression(f"builder_source_raw_filter_{source.id}", mapping)
        field_remap.remap_state_calculation_rows(f"builder_source_calcs_{source.id}", mapping)

    for editor_key in (
        f"builder_source_defaults_editor_{source.id}",
        f"builder_source_filter_editor_{source.id}",
        f"builder_source_raw_filter_{source.id}_editor",
        f"builder_source_calcs_editor_{source.id}",
    ):
        st.session_state.pop(editor_key, None)
    st.session_state[state_key] = enabled


def _source_rename_mapping(
    ctx: ValueStreamContext,
    source: model.Source,
    enabled: bool,
) -> dict[str, str]:
    raw_fields = _source_field_options(ctx, source, rename_capitalize=False)
    forward = {
        field: renamed
        for field, renamed in zip(raw_fields, capitalize_fields(raw_fields), strict=False)
        if field and renamed and field != renamed
    }
    if enabled:
        return forward
    return {renamed: field for field, renamed in forward.items()}


def _new_processor_template(ctx: ValueStreamContext) -> model.BinaryOutcomeProcessor:
    source = ctx.catalog.pipelines.sources[0]
    fields = _source_field_options(ctx, source)
    field_mapping = _source_rename_mapping(ctx, source, True)
    time_column = field_remap.remap_field_name(source.schema_.timestamp_column or "", field_mapping)
    outcome_column = _first_matching_field(fields, "Outcome")
    data: dict[str, Any] = {
        "id": _next_processor_id(ctx, source.id),
        "source": source.id,
        "kind": "binary_outcome",
        "description": "",
        "group_by": [],
        "time": {
            "column": time_column,
            "grains": ["Day", "Summary"] if time_column else ["Summary"],
        },
        "states": {
            "Count": {"type": "count"},
            "Positives": {"type": "count"},
            "Negatives": {"type": "count"},
        },
    }
    if outcome_column:
        data["outcome"] = {
            "column": outcome_column,
            "positive_values": [1, "Clicked", "Conversion"],
            "negative_values": [0, "Impression", "Pending"],
        }
    return model.BinaryOutcomeProcessor.model_validate(data)


def _next_processor_id(ctx: ValueStreamContext, source_id: str) -> str:
    existing = {processor.id for processor in ctx.catalog.processors.processors}
    base = f"{source_id}_processor"
    if base not in existing:
        return base
    index = 2
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def _first_matching_field(fields: list[str], target: str) -> str:
    folded = target.casefold()
    return next((field for field in fields if field.casefold() == folded), "")


def _build_source_definition(
    *,
    source: model.Source,
    source_id: str,
    description: str,
    reader_kind: str,
    file_pattern: str,
    group_by_filename: str | None,
    root: str,
    streaming: bool,
    hive_partitioning: bool,
    timestamp_column: str | None,
    natural_key: list[str],
    drop_columns: list[str],
    default_rows: list[dict[str, Any]],
    use_rename_capitalize: bool,
    filter_expression: dict[str, Any] | None,
    calculated_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_def = builder.source_to_dict(source)
    reader_def = dict(source_def.get("reader", {}))
    reader_def.update(
        {
            "kind": reader_kind,
            "file_pattern": file_pattern,
            "streaming": streaming,
        }
    )
    if group_by_filename:
        reader_def["group_by_filename"] = group_by_filename
    else:
        reader_def.pop("group_by_filename", None)
    if root:
        reader_def["root"] = root
    else:
        reader_def.pop("root", None)
        reader_def.pop("base_dir", None)
    if hive_partitioning:
        reader_def["hive_partitioning"] = True
    else:
        reader_def.pop("hive_partitioning", None)

    transforms = [
        transform
        for transform in source_def.get("transforms", [])
        if transform.get("kind") not in {"rename_capitalize", "defaults", "filter", "derive_column"}
    ]
    default_values = builder.build_default_values(default_rows)
    if use_rename_capitalize:
        transforms.insert(0, {"kind": "rename_capitalize"})
        if default_values:
            transforms.append({"kind": "defaults", "values": default_values})
    if filter_expression:
        transforms.append({"kind": "filter", "expression": filter_expression})
    transforms.extend(builder.build_derive_column_transforms(calculated_rows))

    source_def.update(
        {
            "id": source_id,
            "description": description,
            "reader": reader_def,
            "schema": {
                "timestamp_column": timestamp_column,
                "natural_key": natural_key,
                "drop_columns": drop_columns,
            },
            "defaults": {} if use_rename_capitalize else default_values,
            "transforms": transforms,
        }
    )
    candidate = model.Source.model_validate(source_def)
    equivalent = builder.builder_draft_status(
        f"source-editor:{source.id}",
        _source_editor_projection(source),
        _source_editor_projection(candidate),
    )
    if not equivalent.dirty:
        return builder.source_to_dict(source)
    return source_def


def _source_editor_projection(source: model.Source) -> dict[str, Any]:
    """Compare source-editor meaning without treating transform reordering as an edit."""

    source_def = builder.source_to_dict(source)
    editable_transform_kinds = {"rename_capitalize", "defaults", "filter", "derive_column"}
    return {
        "id": source.id,
        "description": source.description,
        "reader": source_def.get("reader", {}),
        "schema": source_def.get("schema", {}),
        "defaults": builder.source_defaults(source),
        "rename_capitalize": _source_has_transform(source, "rename_capitalize"),
        "filter": builder.first_filter_expression(source),
        "calculated_fields": builder.calculated_rows_from_source(source),
        "fixed_transforms": [
            transform
            for transform in source_def.get("transforms", [])
            if transform.get("kind") not in editable_transform_kinds
        ],
        "materialize_transforms": source_def.get("materialize_transforms", False),
        "debugging": source_def.get("debugging", False),
    }


def _source_has_transform(source: model.Source, kind: str) -> bool:
    return any(transform.kind == kind for transform in source.transforms)


def _preserve_untouched_processor_definition(
    processor: model.Processor,
    processor_def: dict[str, Any],
) -> dict[str, Any]:
    """Keep exact catalog YAML when the processor editor made no semantic change."""

    try:
        candidate = model.Processors.model_validate({"processors": [processor_def]}).processors[0]
    except ValueError:
        return processor_def
    status = builder.builder_draft_status(
        f"processor-editor:{processor.id}",
        _processor_editor_projection(processor),
        _processor_editor_projection(candidate),
    )
    if not status.dirty:
        return builder.processor_to_dict(processor)
    return processor_def


def _processor_editor_projection(processor: model.Processor) -> dict[str, Any]:
    """Normalize implicit processor defaults to the values exposed by the editor."""

    processor_def = builder.processor_to_dict(processor)
    raw_time = dict(processor_def.pop("time", {}) or {})
    fixed_time = {key: value for key, value in raw_time.items() if key not in {"column", "grains"}}
    processor_def["time"] = {
        "column": processor.time.column if processor.time else None,
        "grains": [builder.display_grain(grain) for grain in processor.grains],
        "fixed": fixed_time,
    }
    processor_def["states"] = {
        name: spec.model_dump(mode="json", by_alias=True, exclude_none=True)
        for name, spec in model.effective_processor_states(processor).items()
    }
    return processor_def


def _state_rows(
    processor: model.Processor,
    field_mapping: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    mapping = field_mapping or {}
    rows: list[dict[str, Any]] = []
    for name, spec in model.effective_processor_states(processor).items():
        extra = dict(spec.model_extra or {})
        rows.append(
            {
                "State": name,
                "Type": spec.type,
                "Source Column": field_remap.remap_field_name(
                    str(extra.get("source_column", "") or ""),
                    mapping,
                ),
                "Derived From": _state_derivation(processor, name, spec),
                "Enabled": True,
            }
        )
    return rows or [
        {
            "State": "Count",
            "Type": "count",
            "Source Column": "",
            "Derived From": "included rows",
            "Enabled": True,
        }
    ]


def _blank_state_row() -> dict[str, Any]:
    return {
        "State": "Count",
        "Type": "count",
        "Source Column": "",
        "Derived From": "included rows",
        "Enabled": True,
    }


def _state_derivation(  # noqa: PLR0911
    processor: model.Processor,
    state_name: str,
    spec: model.StateSpec,
) -> str:
    extra = dict(spec.model_extra or {})
    processor_extra = dict(processor.model_extra or {})
    outcome = processor_extra.get("outcome")
    outcome_column = "Outcome"
    positives: list[Any] = ["Clicked", "Conversion"]
    negatives: list[Any] = ["Impression", "Pending"]
    if isinstance(outcome, dict):
        outcome_column = str(outcome.get("column", outcome_column))
        positives = list(outcome.get("positive_values", positives))
        negatives = list(outcome.get("negative_values", negatives))
    if state_name == "Count":
        return f"{outcome_column} in {_compact_values([*positives, *negatives])}"
    if state_name == "Positives":
        return f"{outcome_column} in {_compact_values(positives)}"
    if state_name == "Negatives":
        return f"{outcome_column} in {_compact_values(negatives)}"
    source_column = str(extra.get("source_column", "") or "")
    if spec.type in {"cpc", "hll", "theta"}:
        return f"approx distinct {source_column}" if source_column else "approx distinct values"
    if spec.type in {"tdigest", "kll"}:
        outcome_role = str(extra.get("outcome", "") or "")
        suffix = f" for {outcome_role} outcomes" if outcome_role else ""
        return f"distribution of {source_column or state_name}{suffix}"
    if spec.type == "value_sum":
        return f"sum of {source_column or state_name}"
    if spec.type == "min":
        return f"minimum of {source_column or state_name}"
    if spec.type == "max":
        return f"maximum of {source_column or state_name}"
    if spec.type == "pooled_mean":
        return f"weighted mean of {source_column or state_name}"
    if spec.type == "pooled_variance":
        return f"pooled variance of {source_column or state_name}"
    return "included rows"


def _compact_values(values: list[Any]) -> str:
    if len(values) <= 4:
        return "[" + ", ".join(map(str, values)) + "]"
    return "[" + ", ".join(map(str, values[:4])) + ", ...]"


def _build_state_defs(
    processor: model.Processor,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    existing = {
        name: spec.model_dump(mode="json", exclude_none=True)
        for name, spec in model.effective_processor_states(processor).items()
    }
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row.get("Enabled", True):
            continue
        state_name = str(row.get("State", "")).strip()
        state_type = str(row.get("Type", "")).strip()
        if not state_name or not state_type:
            continue
        state_def = dict(existing.get(state_name, {}))
        state_def["type"] = state_type
        source_column = str(row.get("Source Column", "")).strip()
        if source_column:
            state_def["source_column"] = source_column
        else:
            state_def.pop("source_column", None)
        out[state_name] = state_def
    return out


def _selected_tile(
    tile_options: list[tuple[str, str, str, dict[str, Any]]],
    selected_tile_key: str | None,
) -> tuple[str, str, str, dict[str, Any]] | None:
    if not selected_tile_key:
        return None
    return next(
        (
            option
            for option in tile_options
            if f"{option[0]}/{option[1]}/{option[2]}" == selected_tile_key
        ),
        None,
    )
