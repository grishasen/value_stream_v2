"""Reports and dashboard page."""

from __future__ import annotations

import base64
import calendar
import csv
import datetime as dt
import functools
import importlib.util
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from io import StringIO
from typing import Any

import polars as pl
import streamlit as st
import yaml

from valuestream.charts import prepare_table_data, render_chart, table_row_colors
from valuestream.config import model
from valuestream.query import AggregateNotReadyError
from valuestream.ui import builder, components
from valuestream.ui.context import ValueStreamContext
from valuestream.ui.data import (
    FilterCapability,
    cached_metric_freshness,
    filter_capabilities_for_page,
    filter_columns_for_tile,
    grain_for_tile,
    parse_filter_text,
    partition_filters_for_tile,
    query_metric_cached,
    query_tile,
    tile_to_dict,
)
from valuestream.ui.freshness import freshness_label
from valuestream.ui.instrumentation import (
    AuthoringEvent,
    AuthoringOutcome,
    AuthoringStage,
    record_event,
    workflow_from_handoff,
)
from valuestream.ui.presentation import resolve_tile_presentation
from valuestream.ui.theme import dashboard_theme
from valuestream.utils.logger import get_logger

logger = get_logger(__name__)

FULL_WIDTH_CHARTS = {
    "calendar_heatmap",
    "calibration_curve",
    "clv_treemap",
    "cohort_heatmap",
    "descriptive_funnel",
    "descriptive_heatmap",
    "funnel",
    "gain_curve",
    "geo_map",
    "heatmap",
    "lift_curve",
    "precision_recall_curve",
    "rfm_density",
    "roc_curve",
    "sankey",
    "scatter",
    "table",
    "treemap",
}
REPORT_CHART_HEIGHT_FALLBACK_PX = 560
REPORT_CHART_HEIGHT_HERO_PX = 720
REPORT_CHART_HEIGHT_EXPANDED_PX = 880
REPORT_CHART_HEIGHT_MIN_PX = 420
REPORT_DATAFRAME_HEIGHT_PX = 360
REPORT_VIEW_PRESENTATION = "Presentation"
REPORT_VIEW_INSPECT = "Inspect"
TILE_LAYOUT_HERO = "hero"
TILE_LAYOUT_EXPANDED = "expanded"
TILE_LAYOUT_FULL = "full"
TILE_LAYOUT_HALF = "half"
TILE_LAYOUT_COMPACT = "compact"
ADVANCED_VALUE_FORMATS = ["", "percent", "integer", "number", "currency"]
ADVANCED_COLOR_SCALES = [
    "",
    "Viridis",
    "Cividis",
    "Plasma",
    "Inferno",
    "Magma",
    "Blues",
    "Greens",
    "Reds",
]
ADVANCED_FIELD_CONTROLS: dict[str, tuple[str, ...]] = {
    "line": ("x", "y", "color", "facet_row", "facet_col"),
    "stacked_area": ("x", "y", "color", "facet_row", "facet_col"),
    "bar": ("x", "y", "color", "facet_row", "facet_col"),
    "kpi_card": ("value",),
    "waterfall": ("x", "y", "color", "facet_row", "facet_col"),
    "pareto": ("x", "y", "color", "facet_row", "facet_col"),
    "treemap": ("path", "value", "color"),
    "heatmap": ("x", "y", "color"),
    "cohort_heatmap": ("x", "y", "color"),
    "scatter": (
        "x",
        "y",
        "color",
        "size",
        "animation_frame",
        "animation_group",
        "facet_row",
        "facet_col",
    ),
    "combo": ("x", "y", "y2", "color", "facet_row", "facet_col"),
    "interval": (
        "x",
        "y",
        "error_y",
        "error_y_lower",
        "error_y_upper",
        "color",
        "facet_row",
        "facet_col",
    ),
    "donut": ("names", "values", "color"),
    "geo_map": ("locations", "value", "lat", "lon", "color", "size"),
    "table": ("columns",),
    "calendar_heatmap": ("date", "value"),
    "bar_polar": ("r", "theta", "color"),
    "sankey": ("source", "target", "value"),
    "gauge": ("value", "facet_row", "facet_col"),
    "funnel": ("stages", "color", "facet_row", "facet_col"),
    "boxplot": ("x", "y", "color", "facet_row", "facet_col"),
    "histogram": ("property", "color", "facet_row", "facet_col"),
    "calibration_curve": ("color", "facet_row", "facet_col"),
    "roc_curve": ("color", "facet_row", "facet_col"),
    "precision_recall_curve": ("color", "facet_row", "facet_col"),
    "gain_curve": ("color", "facet_row", "facet_col"),
    "lift_curve": ("color", "facet_row", "facet_col"),
    "rfm_density": ("x", "y", "color"),
    "exposure": ("color",),
    "corr": ("x", "y", "color"),
    "model": ("color",),
    "descriptive_line": ("x", "property", "score", "color", "facet_row", "facet_col"),
    "descriptive_histogram": ("property", "color", "facet_row", "facet_col"),
    "descriptive_heatmap": ("x", "y", "property", "score"),
    "descriptive_funnel": ("x", "stages", "color", "facet_row", "facet_col"),
    "experiment_z_score": ("x", "y", "color", "facet_row", "facet_col"),
    "experiment_odds_ratio": ("x", "y", "color", "facet_row", "facet_col"),
    "clv_treemap": ("path", "value", "color"),
}
ADVANCED_FIELD_KEYS = {
    "animation_frame",
    "animation_group",
    "color",
    "columns",
    "date",
    "error_y",
    "error_y_minus",
    "error_y_plus",
    "error_y_lower",
    "error_y_upper",
    "facet_col",
    "facet_column",
    "facet_row",
    "facets",
    "lat",
    "location",
    "locations",
    "lon",
    "names",
    "path",
    "property",
    "r",
    "score",
    "size",
    "source",
    "stages",
    "target",
    "theta",
    "value",
    "values",
    "x",
    "y",
    "y2",
}
ADVANCED_SETTING_KEYS = {
    "barmode",
    "color_continuous_scale",
    "goal_line",
    "reference",
    "references",
    "scale_mode",
    "show_trend_delta",
    "sort_by",
    "sort_direction",
    "top_n",
    "value_format",
}


def render(ctx: ValueStreamContext) -> None:
    """Render configured dashboards as the Value Stream reports workspace."""
    handoff_workflow = workflow_from_handoff(st.query_params.get("from"))
    if not ctx.catalog.dashboards.dashboards:
        if handoff_workflow is not None:
            record_event(
                st.session_state,
                event=AuthoringEvent.FAILED,
                workflow=handoff_workflow,
                stage=AuthoringStage.REPORT,
                outcome=AuthoringOutcome.BLOCKED,
                once=True,
            )
        components.render_page_header("Reports", "No dashboards configured.", status="pending")
        st.info("No dashboards configured.")
        return
    if handoff_workflow is not None:
        record_event(
            st.session_state,
            event=AuthoringEvent.REPORT_OPENED,
            workflow=handoff_workflow,
            stage=AuthoringStage.REPORT,
            outcome=AuthoringOutcome.SUCCESS,
            once=True,
        )

    dashboard = _selected_dashboard(ctx)
    page = _selected_page(dashboard)
    freshnesses = _page_freshnesses(ctx, page)
    latest_data_date, latest_data_label = _latest_page_data_coverage(freshnesses)
    first_fresh = freshnesses[0] if freshnesses else None

    components.render_page_header(
        page.title,
        help=_page_help_text(dashboard, page, first_fresh),
    )

    dashboard, page, filters, start, end, view_mode, advanced_mode = _report_toolbar(
        ctx,
        dashboard,
        page,
        latest_data_date=latest_data_date,
        latest_data_label=latest_data_label,
    )
    inspect_mode = view_mode == REPORT_VIEW_INSPECT
    # ``freshnesses`` from above is still valid: the toolbar widgets read the
    # same session state on this rerun, so the selected page cannot differ.
    _page_status_banner(
        dashboard,
        page,
        freshnesses,
        filters=filters,
        start=start,
        end=end,
        view_mode=view_mode,
    )

    _kpi_strip(
        ctx,
        page,
        filters=filters,
        start=start,
        end=end,
    )
    _tile_grid(
        ctx,
        dashboard,
        page,
        filters=filters,
        start=start,
        end=end,
        advanced_mode=advanced_mode,
        inspect_mode=inspect_mode,
    )


def _report_toolbar(
    ctx: ValueStreamContext,
    dashboard: Any,
    page: Any,
    *,
    latest_data_date: dt.date | None = None,
    latest_data_label: str | None = None,
) -> tuple[Any, Any, dict[str, Any], dt.date | None, dt.date | None, str, bool]:
    report_pages = tuple(
        report_page
        for configured_dashboard in ctx.catalog.dashboards.dashboards
        for report_page in configured_dashboard.pages
    )
    with st.sidebar:
        dashboards = ctx.catalog.dashboards.dashboards
        dashboard = st.selectbox(
            "**Dashboard**",
            dashboards,
            index=_object_index(dashboards, dashboard),
            format_func=lambda item: item.title,
            key="reports_dashboard",
        )
        pages = dashboard.pages
        page = st.selectbox(
            "**Report**",
            pages,
            index=_object_index(pages, _selected_page(dashboard)),
            format_func=lambda item: item.title,
            key=f"reports_page_{dashboard.id}",
        )
    with components.card():
        filter_capabilities = filter_capabilities_for_page(ctx.catalog, page)
        action_cols = st.columns([0.54, 0.23, 0.23], vertical_alignment="bottom")
        view_mode = action_cols[0].segmented_control(
            "View",
            [REPORT_VIEW_PRESENTATION, REPORT_VIEW_INSPECT],
            default=REPORT_VIEW_PRESENTATION,
            key=f"reports_view_mode_{dashboard.id}_{page.id}",
            help="**Presentation** is the clean dashboard mode: it renders the report charts/KPIs for normal viewing. **Inspect** is the audit/debug mode. It renders the same charts from the same queries, but turns on extra per-tile details: freshness/row count/query timing plus tabs for Data Overview and the tile's YAML config.",
        )
        view_mode = str(view_mode or REPORT_VIEW_PRESENTATION)
        advanced_mode = action_cols[1].toggle(
            "Advanced",
            value=_is_descriptive_report_page(page),
            key=f"reports_advanced_mode_{dashboard.id}_{page.id}",
            help=(
                "Explore one-time chart changes in this browser session. "
                "These changes are not written to the catalog."
            ),
        )
        with action_cols[2].popover(
            "Filters",
            icon=":material/filter_list:",
            width="stretch",
        ):
            filters, start, end = _filter_controls(
                ctx,
                page,
                capabilities=filter_capabilities,
                latest_data_date=latest_data_date,
            )
            st.button(
                "Clear",
                icon=":material/close:",
                key=f"reports_clear_filters_{page.id}",
                width="stretch",
                on_click=_clear_filter_state,
                args=(page, report_pages),
            )

        _filter_chips(
            page,
            filters,
            start,
            end,
            capabilities=filter_capabilities,
            report_pages=report_pages,
        )
        active_preset = str(st.session_state.get(f"reports_time_preset_{page.id}") or "")
        notice = _latest_data_notice(
            active_preset,
            today=dt.date.today(),
            latest_data_date=latest_data_date,
            latest_data_label=latest_data_label,
        )
        if notice:
            st.caption(notice)
    return dashboard, page, filters, start, end, view_mode, bool(advanced_mode)


def _selected_dashboard(ctx: ValueStreamContext) -> Any:
    dashboards = ctx.catalog.dashboards.dashboards
    selected = st.session_state.get("reports_dashboard")
    return _selected_object(dashboards, selected)


def _selected_page(dashboard: Any) -> Any:
    pages = dashboard.pages
    selected = st.session_state.get(f"reports_page_{dashboard.id}")
    return _selected_object(pages, selected)


def _selected_object(options: list[Any], selected: Any) -> Any:
    if not options:
        return None
    selected_id = getattr(selected, "id", None)
    for option in options:
        if selected is option or getattr(option, "id", None) == selected_id:
            return option
    return options[0]


def _object_index(options: list[Any], selected: Any) -> int:
    selected_id = getattr(selected, "id", None)
    for idx, option in enumerate(options):
        if selected is option or getattr(option, "id", None) == selected_id:
            return idx
    return 0


def _filter_controls(
    ctx: ValueStreamContext,
    page: model.DashboardPage,
    *,
    capabilities: list[FilterCapability] | None = None,
    latest_data_date: dt.date | None = None,
) -> tuple[dict[str, Any], dt.date | None, dt.date | None]:
    filters: dict[str, Any] = {}
    today = dt.date.today()
    presets = list(page.time_filter.presets)
    preset_key = f"reports_time_preset_{page.id}"
    if st.session_state.get(preset_key) not in presets:
        st.session_state[preset_key] = page.time_filter.default
    preset = st.segmented_control(
        "Time range",
        presets,
        key=preset_key,
        format_func=_time_preset_label,
    )
    start, end = _relative_time_bounds(
        str(preset or ""),
        today=today,
        latest_data_date=latest_data_date,
    )
    if preset == "custom":
        selected_range = st.date_input(
            "Custom range",
            value=(today - dt.timedelta(days=30), today),
            key=f"reports_custom_range_{page.id}",
        )
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start, end = selected_range

    active_capabilities = capabilities or filter_capabilities_for_page(ctx.catalog, page)
    primary = [item for item in active_capabilities if item.display == "primary"][:3]
    secondary = [item for item in active_capabilities if item not in primary]
    for capability in primary:
        _filter_control(ctx, page, capability, filters)
    if secondary:
        with st.expander("More filters", expanded=False, icon=":material/tune:"):
            for capability in secondary:
                _filter_control(ctx, page, capability, filters)

    with st.expander("Advanced raw filters", expanded=False):
        raw = st.text_area(
            "Filters",
            placeholder="channel=Web\nplan=Basic",
            key=f"reports_raw_filters_{page.id}",
        )
        filters.update(parse_filter_text(raw))
    return filters, start, end


def _relative_time_bounds(
    preset: str,
    *,
    today: dt.date,
    latest_data_date: dt.date | None,
) -> tuple[dt.date | None, dt.date | None]:
    """Resolve relative report presets against the latest aggregate coverage."""

    anchor = min(today, latest_data_date) if latest_data_date is not None else today
    days = {
        "last_7_days": 7,
        "last_30_days": 30,
        "last_90_days": 90,
    }.get(preset)
    if days is not None:
        return anchor - dt.timedelta(days=days - 1), anchor
    if preset == "year_to_date":
        return dt.date(anchor.year, 1, 1), anchor
    return None, None


def _latest_data_notice(
    preset: str,
    *,
    today: dt.date,
    latest_data_date: dt.date | None,
    latest_data_label: str | None,
) -> str | None:
    """Explain when a relative range is anchored before today's date."""

    relative = preset in {"last_7_days", "last_30_days", "last_90_days", "year_to_date"}
    if not relative or latest_data_date is None or latest_data_date >= today:
        return None
    through = latest_data_label or latest_data_date.isoformat()
    return f"Showing latest available data (through {through})."


def _time_preset_label(value: str) -> str:
    return {
        "last_7_days": "Last 7 days",
        "last_30_days": "Last 30 days",
        "last_90_days": "Last 90 days",
        "year_to_date": "Year to date",
        "custom": "Custom",
        "all_time": "All time",
    }.get(value, value.replace("_", " ").capitalize())


def _filter_control(
    ctx: ValueStreamContext,
    page: model.DashboardPage,
    capability: FilterCapability,
    filters: dict[str, Any],
) -> None:
    field = capability.field
    key = f"reports_filter_{page.id}_{field}"
    coverage = (
        "Applies to every chart."
        if capability.applies_to_all
        else (
            f"Applies to {len(capability.supported_tile_ids)} of {len(page.tiles)} charts; "
            "unsupported charts will be marked."
        )
    )
    if capability.control == "text":
        raw = st.text_input(
            capability.label,
            key=key,
            placeholder="Type one or more comma-separated values",
            help=coverage,
        )
        selected = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        options = _filter_options(ctx, page, capability)
        if capability.control == "selectbox":
            value = st.selectbox(
                capability.label,
                options,
                index=None,
                placeholder="All — choose or type",
                accept_new_options=True,
                key=key,
                help=coverage,
            )
            selected = [value] if value not in (None, "") else []
        else:
            selected = st.multiselect(
                capability.label,
                options,
                placeholder="All — choose or type values",
                accept_new_options=True,
                key=key,
                help=coverage,
            )
    if selected:
        filters[field] = selected


def _clear_filter_state(page: Any, report_pages: Iterable[Any] | None = None) -> None:
    prefixes = (
        f"reports_filter_{page.id}_",
        f"reports_raw_filters_{page.id}",
        f"reports_custom_range_{page.id}",
        f"reports_active_filter_chips_{page.id}",
    )
    for key in list(st.session_state):
        if any(str(key).startswith(prefix) for prefix in prefixes):
            st.session_state.pop(key, None)
    _show_all_time(page, report_pages)


def _show_all_time(page: Any, report_pages: Iterable[Any] | None = None) -> None:
    """Reset every report page to its broadest authored time preset."""
    targets = {str(target.id): target for target in (report_pages or ())}
    targets.setdefault(str(page.id), page)
    for target in targets.values():
        time_filter = getattr(target, "time_filter", None)
        presets = list(getattr(time_filter, "presets", []) or [])
        default = str(getattr(time_filter, "default", "all_time"))
        st.session_state[f"reports_time_preset_{target.id}"] = (
            "all_time" if "all_time" in presets or not presets else default
        )
        st.session_state.pop(f"reports_custom_range_{target.id}", None)


def _filter_columns(ctx: ValueStreamContext, page: Any) -> list[str]:
    capabilities = filter_capabilities_for_page(ctx.catalog, page)
    if capabilities:
        return [item.field for item in capabilities]
    return list(
        dict.fromkeys(
            column for tile in page.tiles for column in filter_columns_for_tile(tile_to_dict(tile))
        )
    )


def _filter_options(
    ctx: ValueStreamContext,
    page: model.DashboardPage,
    capability: FilterCapability,
) -> list[str]:
    """Load suggestions from the first compatible plot that has values.

    A filter remains editable when no aggregate values are available.  Querying
    one compatible plot at a time avoids the previous eager union across every
    tile while retaining a fallback for an empty or not-yet-ready first plot.
    """

    supported_tiles = [tile for tile in page.tiles if tile.id in capability.supported_tile_ids]
    plot_tiles = [tile for tile in supported_tiles if tile.placement != "kpi_strip"]
    kpi_tiles = [tile for tile in supported_tiles if tile.placement == "kpi_strip"]
    for tile in [*plot_tiles, *kpi_tiles]:
        metric = ctx.catalog.metrics.metrics.get(tile.metric)
        if metric is None:
            continue
        processor = next(
            (
                candidate
                for candidate in ctx.catalog.processors.processors
                if candidate.id == metric.source
            ),
            None,
        )
        if processor is None:
            continue
        try:
            rows = query_metric_cached(
                ctx.workspace,
                ctx.catalog,
                tile.metric,
                group_by=[capability.field],
                grain="summary",
            )
        except AggregateNotReadyError as exc:
            logger.info(
                "Filter options are waiting for aggregate backfill: tile=%s metric=%s "
                "column=%s reason=%s",
                tile.id,
                tile.metric,
                capability.field,
                exc,
            )
            continue
        except Exception:
            logger.exception(
                "Failed to load filter options: tile=%s metric=%s column=%s",
                tile.id,
                tile.metric,
                capability.field,
            )
            continue
        if capability.field not in rows.columns:
            continue
        values = {
            str(value)
            for value in rows.get_column(capability.field).drop_nulls().unique().to_list()
            if str(value)
        }
        if values:
            return sorted(values, key=str.casefold)[:500]
    return []


def _filter_chip_value(value: Any) -> str:
    if isinstance(value, list):
        rendered = ", ".join(str(item) for item in value[:3])
        if len(value) > 3:
            rendered = f"{rendered} +{len(value) - 3}"
        return rendered
    return str(value)


def _filter_chip_labels(
    filters: Mapping[str, Any],
    start: dt.date | None,
    end: dt.date | None,
    *,
    time_preset: str | None = None,
) -> list[str]:
    """Summarize the active filter context as chip labels."""
    chips: list[str] = []
    if start and end:
        if time_preset:
            chips.append(f"{_time_preset_label(time_preset)} · {_compact_date_range(start, end)}")
        else:
            chips.append(f"Time: {start.isoformat()} to {end.isoformat()}")
    chips.extend(f"{key}: {_filter_chip_value(value)}" for key, value in filters.items())
    return chips


def _compact_date_range(start: dt.date, end: dt.date) -> str:
    """Render a concise, locale-independent date range for the active-time chip."""
    months = (
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    )
    start_month = months[start.month - 1]
    end_month = months[end.month - 1]
    separator = "\N{EN DASH}"
    if start.year != end.year:
        return (
            f"{start_month} {start.day}, {start.year}{separator}{end_month} {end.day}, {end.year}"
        )
    if start.month == end.month:
        return f"{start_month} {start.day}{separator}{end.day}, {end.year}"
    return f"{start_month} {start.day}{separator}{end_month} {end.day}, {end.year}"


def _apply_filter_chip_selection(
    widget_key: str,
    removable: Mapping[str, tuple[str, ...]],
) -> None:
    """Clear deselected filter widgets before Streamlit reruns the page."""
    selected = set(st.session_state.get(widget_key) or [])
    for label, widget_keys in removable.items():
        if label in selected:
            continue
        for widget_key_to_clear in widget_keys:
            st.session_state.pop(widget_key_to_clear, None)


def _filter_chips(
    page: Any,
    filters: Mapping[str, Any],
    start: dt.date | None,
    end: dt.date | None,
    *,
    capabilities: list[FilterCapability] | None = None,
    report_pages: Iterable[Any] | None = None,
) -> None:
    """Render active filters as chips; deselecting a chip clears that filter."""
    removable: dict[str, tuple[str, ...]] = {}
    inert: list[str] = []
    time_label: str | None = None
    if start and end:
        preset_key = f"reports_time_preset_{page.id}"
        time_preset = str(st.session_state.get(preset_key) or "custom")
        time_label = _filter_chip_labels(
            {},
            start,
            end,
            time_preset=time_preset,
        )[0]
    capability_by_field = {item.field: item for item in (capabilities or [])}
    for column, value in filters.items():
        capability = capability_by_field.get(column)
        display_label = capability.label if capability is not None else column
        coverage = ""
        if capability is not None and not capability.applies_to_all:
            coverage = (
                f" · {len(capability.supported_tile_ids)}/{len(getattr(page, 'tiles', []))} charts"
            )
        label = f"{display_label}: {_filter_chip_value(value)}{coverage}"
        widget_key = f"reports_filter_{page.id}_{column}"
        if st.session_state.get(widget_key):
            removable[label] = (widget_key,)
        else:
            # Filters typed into the raw text area cannot be cleared one by one.
            inert.append(label)
    if time_label is None and not removable and not inert:
        st.caption("All time · no filters")
        return
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        if time_label is not None:
            st.button(
                time_label,
                key=f"reports_time_chip_{page.id}",
                type="primary",
                icon=":material/calendar_today:",
                help="Active time range. Click to show all time.",
                on_click=_show_all_time,
                args=(page, report_pages),
            )
        if removable:
            chips_key = f"reports_active_filter_chips_{page.id}"
            st.pills(
                "Active report filters",
                list(removable),
                selection_mode="multi",
                default=list(removable),
                key=chips_key,
                label_visibility="collapsed",
                help="Deselect a chip to clear that filter.",
                on_change=_apply_filter_chip_selection,
                args=(chips_key, removable),
            )
    if inert:
        st.caption("Raw filters: " + " · ".join(inert))


def _page_freshnesses(ctx: ValueStreamContext, page: Any) -> list[Any]:
    out: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for tile in page.tiles:
        tile_dict = tile_to_dict(tile)
        grain = grain_for_tile(tile_dict)
        metric = str(tile_dict.get("metric", ""))
        key = (metric, grain)
        if not metric or key in seen:
            continue
        seen.add(key)
        out.append(cached_metric_freshness(ctx.workspace, ctx.catalog, metric, grain=grain))
    return out


def _latest_page_data_coverage(freshnesses: Iterable[Any]) -> tuple[dt.date | None, str | None]:
    """Return the latest usable date and its authored aggregate-period label."""

    candidates: list[tuple[dt.date, str]] = []
    for freshness in freshnesses:
        label = str(getattr(freshness, "latest_period", "") or "").strip()
        coverage_end = _aggregate_period_end(label)
        if coverage_end is not None:
            candidates.append((coverage_end, label))
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[0])


def _aggregate_period_end(value: str) -> dt.date | None:
    """Interpret aggregate period labels as inclusive coverage end dates."""

    text = value.strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError:
        pass
    try:
        month = dt.datetime.strptime(text[:7], "%Y-%m").date()
    except ValueError:
        month = None
    if month is not None:
        return dt.date(
            month.year,
            month.month,
            calendar.monthrange(month.year, month.month)[1],
        )
    try:
        year = int(text[:4])
    except ValueError:
        return None
    if text[:4] == text:
        return dt.date(year, 12, 31)
    return None


def _page_status_banner(
    dashboard: Any,
    page: Any,
    freshnesses: list[Any],
    *,
    filters: Mapping[str, Any],
    start: dt.date | None,
    end: dt.date | None,
    view_mode: str,
) -> None:
    coverage_page = _is_coverage_page(page)
    report_kind = "Report type coverage" if coverage_page else "Business report"
    details = [
        report_kind,
        f"{len(page.tiles)} tile(s)",
        f"{view_mode.lower()} view",
    ]
    if start and end:
        details.append(f"{start.isoformat()} to {end.isoformat()}")
    elif not filters:
        details.append("all time")
    if filters:
        details.append(f"{len(filters)} active filter(s)")

    if not freshnesses:
        st.warning(f"{' | '.join(details)}. No tile data found.")
        return

    stale = [fresh for fresh in freshnesses if not fresh.latest_period]
    statuses = {str(fresh.status) for fresh in freshnesses if getattr(fresh, "status", None)}
    latest_periods = sorted(
        {str(fresh.latest_period) for fresh in freshnesses if fresh.latest_period}, reverse=True
    )
    last_run = max(
        (
            fresh.last_run_finished_at
            for fresh in freshnesses
            if getattr(fresh, "last_run_finished_at", None) is not None
        ),
        default=None,
    )
    freshness_bits = [
        f"latest {latest_periods[0]}" if latest_periods else "no aggregate period",
        f"run {components.format_timestamp(last_run)}",
        f"status {', '.join(sorted(statuses)) or 'unknown'}",
    ]
    message = f"{' | '.join(details)}. {' | '.join(freshness_bits)}."
    if stale:
        st.warning(message)
    else:
        st.caption(message)


def _is_coverage_page(page: Any) -> bool:
    identifier = f"{getattr(page, 'id', '')} {getattr(page, 'title', '')}".casefold()
    return "coverage" in identifier or "report type" in identifier


def _is_descriptive_report_page(page: Any) -> bool:
    return any(
        tile.chart == "boxplot" or str(tile.chart).startswith("descriptive_") for tile in page.tiles
    )


@st.fragment(parallel=True)
def _kpi_strip(
    ctx: ValueStreamContext,
    page: Any,
    *,
    filters: dict[str, Any],
    start: dt.date | None,
    end: dt.date | None,
) -> None:
    kpi_tiles = [
        tile for tile in page.tiles if tile.chart == "kpi_card" and tile.placement == "kpi_strip"
    ]
    if not kpi_tiles:
        return
    items: list[components.MetricItem] = []
    for tile in kpi_tiles:
        try:
            tile_dict = resolve_tile_presentation(ctx.catalog, tile)
            bundle = _kpi_bundle(
                ctx,
                tile,
                filters=filters,
                start=start,
                end=end,
            )
            items.append(
                components.MetricItem(
                    tile.title,
                    _format_metric_value(
                        bundle.value,
                        str(tile_dict.get("value_format") or tile_dict.get("number_format") or ""),
                    ),
                    delta=(
                        _format_metric_value(
                            bundle.delta,
                            str(
                                tile_dict.get("value_format")
                                or tile_dict.get("number_format")
                                or ""
                            ),
                        )
                        if bundle.delta is not None
                        else None
                    ),
                    delta_description=bundle.delta_description,
                    delta_color=(
                        "inverse" if tile_dict.get("direction") == "lower_is_better" else "normal"
                    ),
                    chart_data=bundle.sparkline,
                    help=_kpi_help(tile_dict, bundle),
                )
            )
        except AggregateNotReadyError as exc:
            logger.info(
                "KPI tile is waiting for aggregate backfill: tile=%s metric=%s reason=%s",
                tile.id,
                tile.metric,
                exc,
            )
            items.append(
                components.MetricItem(
                    tile.title,
                    "not ready",
                    help=str(exc),
                )
            )
        except Exception:
            logger.exception(
                "Failed to summarize KPI tile: tile=%s metric=%s",
                tile.id,
                tile.metric,
            )
            items.append(components.MetricItem(tile.title, "n/a"))
    components.metric_strip(items, columns=min(len(items), 5) or None, key=f"reports_{page.id}")


@dataclass(frozen=True)
class KpiBundle:
    """Display-ready values for one explicitly configured KPI."""

    value: float | int | str
    delta: float | int | None = None
    delta_description: str | None = None
    sparkline: tuple[float, ...] | None = None
    period_description: str = "All time"


def _kpi_bundle(
    ctx: ValueStreamContext,
    tile: model.Tile,
    *,
    filters: Mapping[str, Any],
    start: dt.date | None,
    end: dt.date | None,
) -> KpiBundle:
    tile_dict = tile_to_dict(tile)
    value_column = str(tile_dict.get("value") or tile.metric)
    applied, _ = partition_filters_for_tile(ctx.catalog, tile, filters)
    query_filters = {**dict(tile_dict.get("filters") or {}), **applied}
    kpi = tile.kpi or model.KpiSpec()
    series = pl.DataFrame()
    if kpi.sparkline_grain or kpi.comparison == "previous_period":
        series = query_metric_cached(
            ctx.workspace,
            ctx.catalog,
            tile.metric,
            filters=query_filters,
            grain=kpi.sparkline_grain or "daily",
        )

    current_start, current_end = start, end
    period_description = "All time"
    if start is not None and end is not None:
        period_description = f"{start.isoformat()} to {end.isoformat()}"
    elif kpi.comparison == "previous_period":
        latest = _latest_series_date(series)
        if latest is not None:
            current_start, current_end = _calendar_period(latest, kpi.comparison_period)
            period_description = _period_description(
                current_start, current_end, kpi.comparison_period
            )

    current_rows = query_metric_cached(
        ctx.workspace,
        ctx.catalog,
        tile.metric,
        filters=query_filters,
        grain="summary",
        start=current_start,
        end=current_end,
    )
    value = _scalar_value(current_rows, value_column)
    delta: float | int | None = None
    delta_description: str | None = None
    if kpi.comparison == "previous_period" and current_start and current_end:
        day_count = (current_end - current_start).days + 1
        previous_end = current_start - dt.timedelta(days=1)
        previous_start = previous_end - dt.timedelta(days=day_count - 1)
        previous_rows = query_metric_cached(
            ctx.workspace,
            ctx.catalog,
            tile.metric,
            filters=query_filters,
            grain="summary",
            start=previous_start,
            end=previous_end,
        )
        previous_value = _scalar_value(previous_rows, value_column)
        if isinstance(value, int | float) and isinstance(previous_value, int | float):
            delta = value - previous_value
            delta_description = _comparison_period_label(previous_start, previous_end)
    elif kpi.target is not None and isinstance(value, int | float):
        delta = value - kpi.target
        delta_description = f"Target {kpi.target:g}"

    sparkline = _sparkline_values(series, value_column, kpi.sparkline_points)
    return KpiBundle(
        value=value,
        delta=delta,
        delta_description=delta_description,
        sparkline=sparkline,
        period_description=period_description,
    )


def _scalar_value(rows: pl.DataFrame, column: str) -> float | int | str:
    if rows.is_empty() or column not in rows.columns:
        return "n/a"
    values = rows.get_column(column).drop_nulls()
    if values.len() != 1:
        return "n/a"
    value = values.item()
    return value if isinstance(value, int | float) else "n/a"


def _sparkline_values(
    rows: pl.DataFrame,
    column: str,
    points: int,
) -> tuple[float, ...] | None:
    if rows.is_empty() or column not in rows.columns or not rows.schema[column].is_numeric():
        return None
    time_column = _series_time_column(rows)
    ordered = rows.sort(time_column) if time_column else rows
    values = ordered.get_column(column).drop_nulls().tail(points).to_list()
    return tuple(float(value) for value in values) if len(values) >= 2 else None


def _latest_series_date(rows: pl.DataFrame) -> dt.date | None:
    column = _series_time_column(rows)
    if column is None or rows.is_empty():
        return None
    values = rows.get_column(column).drop_nulls().cast(pl.String).to_list()
    parsed = [_parse_period_date(str(value)) for value in values]
    return max((value for value in parsed if value is not None), default=None)


def _series_time_column(rows: pl.DataFrame) -> str | None:
    return next(
        (
            candidate
            for candidate in ("Day", "day", "as_of_date", "Week", "Month", "month")
            if candidate in rows.columns
        ),
        None,
    )


def _parse_period_date(value: str) -> dt.date | None:
    for pattern in ("%Y-%m-%d", "%Y-%m"):
        try:
            parsed = dt.datetime.strptime(value[:10], pattern).date()
            if pattern == "%Y-%m":
                return dt.date(parsed.year, parsed.month, 1)
            return parsed
        except ValueError:
            continue
    return None


def _calendar_period(value: dt.date, period: str) -> tuple[dt.date, dt.date]:
    if period == "day":
        return value, value
    if period == "week":
        start = value - dt.timedelta(days=value.weekday())
        return start, start + dt.timedelta(days=6)
    if period == "month":
        return (
            dt.date(value.year, value.month, 1),
            dt.date(value.year, value.month, calendar.monthrange(value.year, value.month)[1]),
        )
    if period == "quarter":
        first_month = 3 * ((value.month - 1) // 3) + 1
        last_month = first_month + 2
        return (
            dt.date(value.year, first_month, 1),
            dt.date(
                value.year,
                last_month,
                calendar.monthrange(value.year, last_month)[1],
            ),
        )
    return dt.date(value.year, 1, 1), dt.date(value.year, 12, 31)


def _period_description(start: dt.date, end: dt.date, period: str) -> str:
    if period == "month":
        return start.strftime("%B %Y")
    if period == "quarter":
        return f"Q{((start.month - 1) // 3) + 1} {start.year}"
    if period == "year":
        return str(start.year)
    return f"{start.isoformat()} to {end.isoformat()}"


def _comparison_period_label(start: dt.date, end: dt.date) -> str:
    if start.year == end.year and start.month == end.month:
        return f"vs {start.strftime('%b')} {start.day}-{end.day}, {end.year}"
    return f"vs {start.strftime('%b')} {start.day}-{end.strftime('%b')} {end.day}, {end.year}"


def _kpi_help(tile_dict: Mapping[str, Any], bundle: KpiBundle) -> str:
    details = [bundle.period_description]
    if tile_dict.get("description"):
        details.append(str(tile_dict["description"]))
    if tile_dict.get("quality_label"):
        details.append(f"{tile_dict['quality_label']}: {tile_dict.get('quality_help', '')}")
    return "\n".join(details)


def _tile_grid(
    ctx: ValueStreamContext,
    dashboard: Any,
    page: Any,
    *,
    filters: dict[str, Any],
    start: dt.date | None,
    end: dt.date | None,
    advanced_mode: bool,
    inspect_mode: bool,
) -> None:
    normal_tiles = [tile for tile in page.tiles if tile.placement != "kpi_strip"]
    if len(normal_tiles) == 1:
        _tile_card(
            ctx,
            dashboard,
            page,
            normal_tiles[0],
            filters=filters,
            start=start,
            end=end,
            advanced_mode=advanced_mode,
            inspect_mode=inspect_mode,
            layout_mode=_tile_layout_mode(dashboard.id, page.id, normal_tiles[0], advanced_mode),
        )
        return
    idx = 0
    while idx < len(normal_tiles):
        tile = normal_tiles[idx]
        layout_mode = _tile_layout_mode(dashboard.id, page.id, tile, advanced_mode)
        if layout_mode == TILE_LAYOUT_COMPACT:
            compact_tiles = []
            while idx < len(normal_tiles) and len(compact_tiles) < 3:
                candidate = normal_tiles[idx]
                if (
                    _tile_layout_mode(dashboard.id, page.id, candidate, advanced_mode)
                    != TILE_LAYOUT_COMPACT
                ):
                    break
                compact_tiles.append(candidate)
                idx += 1
            cols = st.columns(len(compact_tiles))
            for col, compact_tile in zip(cols, compact_tiles, strict=False):
                with col:
                    _tile_card(
                        ctx,
                        dashboard,
                        page,
                        compact_tile,
                        filters=filters,
                        start=start,
                        end=end,
                        advanced_mode=advanced_mode,
                        inspect_mode=inspect_mode,
                        layout_mode=TILE_LAYOUT_COMPACT,
                    )
            continue
        if layout_mode in {TILE_LAYOUT_FULL, TILE_LAYOUT_HERO} or idx == len(normal_tiles) - 1:
            _tile_card(
                ctx,
                dashboard,
                page,
                tile,
                filters=filters,
                start=start,
                end=end,
                advanced_mode=advanced_mode,
                inspect_mode=inspect_mode,
                layout_mode=layout_mode,
            )
            idx += 1
            continue
        next_mode = _tile_layout_mode(dashboard.id, page.id, normal_tiles[idx + 1], advanced_mode)
        if next_mode not in {TILE_LAYOUT_HALF, TILE_LAYOUT_COMPACT}:
            _tile_card(
                ctx,
                dashboard,
                page,
                tile,
                filters=filters,
                start=start,
                end=end,
                advanced_mode=advanced_mode,
                inspect_mode=inspect_mode,
                layout_mode=layout_mode,
            )
            idx += 1
            continue
        cols = st.columns(2)
        for col in cols:
            with col:
                paired_tile = normal_tiles[idx]
                _tile_card(
                    ctx,
                    dashboard,
                    page,
                    paired_tile,
                    filters=filters,
                    start=start,
                    end=end,
                    advanced_mode=advanced_mode,
                    inspect_mode=inspect_mode,
                    layout_mode=_tile_layout_mode(
                        dashboard.id, page.id, paired_tile, advanced_mode
                    ),
                )
            idx += 1


def _page_help_text(dashboard: Any, page: Any, fresh: Any | None) -> str:
    lines = [
        f"Dashboard: {dashboard.title}",
        f"Tiles: {len(page.tiles)}",
    ]
    if fresh is None:
        lines.append("Run status: no tiles")
    else:
        lines.extend(_freshness_help_lines(fresh))
    return "\n".join(lines)


def _tile_help_text(
    configured_tile: Mapping[str, Any],
    tile_dict: Mapping[str, Any],
    grain: str,
    fresh: Any,
) -> str:
    freshness = "fresh" if fresh.latest_period else "stale"
    lines = [
        f"Metric: {configured_tile['metric']}",
        f"Chart: {tile_dict['chart']}",
        f"Grain: {grain}",
        f"Tile freshness: {freshness}",
        *_freshness_help_lines(fresh),
    ]
    if tile_dict.get("description"):
        lines.insert(0, str(tile_dict["description"]))
    if tile_dict.get("quality_label"):
        lines.append(f"{tile_dict['quality_label']}: {tile_dict.get('quality_help', '')}".rstrip())
    return "\n".join(lines)


def _freshness_help_lines(fresh: Any) -> list[str]:
    return [
        f"Run status: {fresh.status}",
        f"Latest aggregate: {fresh.latest_period or 'no aggregate'}",
        f"Last run: {components.format_timestamp(fresh.last_run_finished_at)}",
    ]


def _layout_tile_dict(
    dashboard_id: str,
    page_id: str,
    tile: Any,
    *,
    advanced_mode: bool,
) -> dict[str, Any]:
    tile_dict = _as_tile_dict(tile)
    if not advanced_mode:
        return tile_dict
    draft = st.session_state.get(_advanced_tile_state_key(dashboard_id, page_id, tile_dict))
    return dict(draft) if isinstance(draft, dict) else tile_dict


def _as_tile_dict(tile: Any) -> dict[str, Any]:
    if isinstance(tile, Mapping):
        return dict(tile)
    return tile_to_dict(tile)


def _is_full_width_tile(tile: Any) -> bool:
    tile_dict = _as_tile_dict(tile)
    chart = str(tile_dict.get("chart", "")).casefold()
    if chart in FULL_WIDTH_CHARTS:
        return True
    if _has_grouped_gauge_tile(tile):
        return True
    if chart == "line" and tile_dict.get("color"):
        return True
    facets = tile_dict.get("facets")
    return bool(
        tile_dict.get("facet_row")
        or tile_dict.get("facet_col")
        or tile_dict.get("facet_column")
        or (isinstance(facets, dict) and any(facets.get(key) for key in ("row", "col", "column")))
    )


def _has_grouped_gauge_tile(tile: Any) -> bool:
    tile_dict = _as_tile_dict(tile)
    if str(tile_dict.get("chart", "")).casefold() != "gauge":
        return False
    facets = tile_dict.get("facets")
    return bool(
        tile_dict.get("group_by")
        or tile_dict.get("facet_row")
        or tile_dict.get("facet_col")
        or tile_dict.get("facet_column")
        or (isinstance(facets, dict) and any(facets.get(key) for key in ("row", "col", "column")))
    )


def _tile_layout_mode(
    dashboard_id: str,
    page_id: str,
    tile: Any,
    advanced_mode: bool = False,
) -> str:
    tile_dict = _layout_tile_dict(dashboard_id, page_id, tile, advanced_mode=advanced_mode)
    if _tile_expanded(dashboard_id, page_id, tile_dict):
        return TILE_LAYOUT_FULL
    raw = str(
        tile_dict.get("layout") or tile_dict.get("width") or tile_dict.get("size") or ""
    ).casefold()
    if raw in {TILE_LAYOUT_HERO, "lead"}:
        return TILE_LAYOUT_HERO
    if raw in {TILE_LAYOUT_FULL, "wide"}:
        return TILE_LAYOUT_FULL
    if raw in {TILE_LAYOUT_COMPACT, "small", "third"}:
        return TILE_LAYOUT_COMPACT
    return TILE_LAYOUT_FULL if _is_full_width_tile(tile_dict) else TILE_LAYOUT_HALF


@st.fragment(parallel=True)
def _tile_card(
    ctx: ValueStreamContext,
    dashboard: Any,
    page: Any,
    tile: Any,
    *,
    filters: dict[str, Any],
    start: dt.date | None,
    end: dt.date | None,
    advanced_mode: bool,
    inspect_mode: bool,
    layout_mode: str,
) -> None:
    """Render one report tile.

    Runs as a parallel fragment: tiles on a page render concurrently on the
    initial pass, and per-tile interactions (Inspect/Expand toggles, advanced
    controls) rerun only this tile instead of the whole page.
    """
    configured_tile = tile_to_dict(tile)
    tile_dict = _layout_tile_dict(
        dashboard.id,
        page.id,
        configured_tile,
        advanced_mode=advanced_mode,
    )
    grain = grain_for_tile(tile_dict)
    fresh = cached_metric_freshness(
        ctx.workspace, ctx.catalog, configured_tile["metric"], grain=grain
    )
    with components.card():
        header_slot = st.empty()
        if advanced_mode:
            tile_dict = _advanced_tile_controls(
                ctx,
                dashboard_id=dashboard.id,
                page_id=page.id,
                base_tile=configured_tile,
            )
            grain = grain_for_tile(tile_dict)
            fresh = cached_metric_freshness(
                ctx.workspace,
                ctx.catalog,
                configured_tile["metric"],
                grain=grain,
            )
        tile_dict = resolve_tile_presentation(ctx.catalog, tile_dict)
        start_time = time.perf_counter()
        try:
            parsed_tile = model.Tile.model_validate(tile_dict)
            _, ignored_filters = partition_filters_for_tile(ctx.catalog, parsed_tile, filters)
            rows = query_tile(
                ctx.workspace, ctx.catalog, parsed_tile, filters=filters, start=start, end=end
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            is_native_table = str(tile_dict.get("chart", "")).casefold() == "table"
            table_data = _native_table_data(tile_dict, rows) if is_native_table else None
            if rows.is_empty():
                _render_empty_tile_result(
                    header_slot,
                    dashboard,
                    page,
                    configured_tile,
                    tile_dict,
                    grain=grain,
                    fresh=fresh,
                    rows=rows,
                    inspect_mode=inspect_mode,
                    filters=filters,
                    start=start,
                    end=end,
                    ignored_filters=ignored_filters,
                    elapsed_ms=elapsed_ms,
                )
                return
            figure = (
                None
                if is_native_table
                else render_chart(
                    rows,
                    tile_dict,
                    theme=dashboard_theme(
                        ctx.catalog.dashboards.theme,
                        tile_dict.get("theme"),
                    ),
                )
            )
            with header_slot.container():
                tile_inspect, expanded, show_data = _tile_header(
                    dashboard,
                    page,
                    configured_tile,
                    tile_dict,
                    grain,
                    fresh,
                    rows,
                    figure,
                    inspect_mode=inspect_mode,
                    download_rows=table_data[0] if table_data is not None else rows,
                )
            if ignored_filters:
                st.caption(
                    ":material/info: Active filters not applied to this chart: "
                    + ", ".join(ignored_filters)
                )
            if table_data is not None:
                _render_report_table(
                    *table_data,
                    key=f"reports_table_{dashboard.id}_{page.id}_{configured_tile['id']}",
                )
            else:
                chart_height = _report_chart_height(
                    figure,
                    layout_mode=TILE_LAYOUT_EXPANDED if expanded else layout_mode,
                )
                figure.update_layout(height=chart_height)
                st.plotly_chart(
                    figure,
                    width="stretch",
                    height=chart_height,
                    theme=None,
                    key=f"reports_chart_{dashboard.id}_{page.id}_{configured_tile['id']}",
                )
            if show_data:
                _tile_data_table(tile_dict, rows)
            if tile_inspect:
                st.caption(
                    f"{freshness_label(fresh)} | {rows.height:,} row(s) | {elapsed_ms:.0f} ms"
                )
                _tile_inspectors(
                    tile_dict,
                    rows,
                    include_data=not (show_data or is_native_table),
                )
        except AggregateNotReadyError as exc:
            logger.info(
                "Report tile is waiting for aggregate backfill: dashboard=%s page=%s "
                "tile=%s metric=%s reason=%s",
                dashboard.id,
                page.id,
                configured_tile.get("id"),
                configured_tile.get("metric"),
                exc,
            )
            with header_slot.container():
                st.markdown(f"**{configured_tile.get('title', 'Metric')}**")
                st.badge(
                    "Backfill required",
                    color="orange",
                    icon=":material/history:",
                )
            st.warning(str(exc), icon=":material/database:")
        except Exception as exc:  # pragma: no cover - Streamlit display path
            logger.exception(
                "Failed to render report tile: dashboard=%s page=%s tile=%s metric=%s chart=%s",
                dashboard.id,
                page.id,
                configured_tile.get("id"),
                configured_tile.get("metric"),
                tile_dict.get("chart"),
            )
            st.error(str(exc))
            with st.expander("Tile YAML", expanded=False):
                st.code(yaml.safe_dump(tile_dict, sort_keys=False), language="yaml")


def _render_empty_tile_result(
    header_slot: Any,
    dashboard: Any,
    page: Any,
    configured_tile: Mapping[str, Any],
    tile_dict: dict[str, Any],
    *,
    grain: str,
    fresh: Any,
    rows: pl.DataFrame,
    inspect_mode: bool,
    filters: Mapping[str, Any],
    start: dt.date | None,
    end: dt.date | None,
    ignored_filters: list[str] | tuple[str, ...],
    elapsed_ms: float,
) -> None:
    with header_slot.container():
        tile_inspect, _, _ = _tile_header(
            dashboard,
            page,
            configured_tile,
            tile_dict,
            grain,
            fresh,
            rows,
            None,
            inspect_mode=inspect_mode,
        )
    _render_empty_tile_recovery(
        page,
        configured_tile,
        filters=filters,
        start=start,
        end=end,
        ignored_filters=ignored_filters,
    )
    if tile_inspect:
        st.caption(f"{freshness_label(fresh)} | 0 row(s) | {elapsed_ms:.0f} ms")
        _tile_inspectors(tile_dict, rows)


def _empty_tile_message(
    *,
    filters: Mapping[str, Any],
    start: dt.date | None,
    end: dt.date | None,
    ignored_filters: list[str] | tuple[str, ...],
) -> tuple[str, bool]:
    """Describe an empty aggregate result and whether broadening can recover it."""

    active_scope: list[str] = []
    if start and end:
        active_scope.append(f"date range {start.isoformat()} to {end.isoformat()}")
    if filters:
        active_scope.append(f"{len(filters)} active filter(s)")
    if active_scope:
        ignored = (
            " Some active filters are unsupported by this chart: "
            + ", ".join(ignored_filters)
            + "."
            if ignored_filters
            else ""
        )
        return (
            "No aggregate rows match " + " and ".join(active_scope) + "." + ignored,
            True,
        )
    return (
        "Aggregate rows are not materialized for this metric and grain yet. "
        "Run Data Load or inspect freshness before retrying.",
        False,
    )


def _render_empty_tile_recovery(
    page: Any,
    tile: Mapping[str, Any],
    *,
    filters: Mapping[str, Any],
    start: dt.date | None,
    end: dt.date | None,
    ignored_filters: list[str] | tuple[str, ...],
) -> None:
    message, recoverable = _empty_tile_message(
        filters=filters,
        start=start,
        end=end,
        ignored_filters=ignored_filters,
    )
    if not recoverable:
        st.info(message, icon=":material/database:")
        return
    st.warning(message, icon=":material/filter_alt_off:")
    if st.button(
        "Show all available data",
        key=f"reports_show_all_{page.id}_{tile.get('id', 'tile')}",
        icon=":material/date_range:",
    ):
        _clear_filter_state(page)
        st.rerun(scope="app")


def _tile_header(
    dashboard: Any,
    page: Any,
    configured_tile: Mapping[str, Any],
    tile_dict: Mapping[str, Any],
    grain: str,
    fresh: Any,
    rows: pl.DataFrame,
    figure: Any | None,
    *,
    inspect_mode: bool,
    download_rows: pl.DataFrame | None = None,
) -> tuple[bool, bool, bool]:
    action_prefix = _tile_action_prefix(dashboard.id, page.id, configured_tile)
    inspect_key = f"{action_prefix}_inspect"
    expand_key = f"{action_prefix}_expanded"
    data_key = f"{action_prefix}_data"
    title_col, action_col = st.columns([0.94, 0.06], vertical_alignment="center")
    with title_col:
        st.subheader(
            str(tile_dict.get("title") or configured_tile["title"]),
            help=_tile_help_text(configured_tile, tile_dict, grain, fresh),
        )
        if tile_dict.get("quality_label"):
            st.badge(
                str(tile_dict["quality_label"]),
                color="gray",
                help=str(tile_dict.get("quality_help") or ""),
            )
    with action_col:
        actions_popover = st.popover(
            ":material/more_vert:",
            help="Actions",
            width="content",
            key=f"{action_prefix}_actions",
        )
    with actions_popover:
        local_inspect = st.toggle(
            "Inspect",
            value=bool(st.session_state.get(inspect_key, False)),
            key=inspect_key,
        )
        expanded = st.toggle(
            "Expand",
            value=bool(st.session_state.get(expand_key, False)),
            key=expand_key,
        )
        if figure is None:
            show_data = False
        else:
            show_data = st.toggle(
                "View data",
                value=bool(st.session_state.get(data_key, False)),
                key=data_key,
            )
        # ``data=`` callables defer generation until the user actually clicks
        # the button (Streamlit >= 1.58), so no CSV/PNG/HTML is built during
        # normal tile renders.
        st.download_button(
            "CSV",
            data=lambda: _rows_csv(download_rows if download_rows is not None else rows),
            file_name=f"{_download_file_stem(page, configured_tile)}.csv",
            mime="text/csv",
            icon=":material/download:",
            key=f"{action_prefix}_csv",
            width="stretch",
        )
        if figure is not None:
            if _png_export_available():
                st.download_button(
                    "PNG",
                    data=lambda: _figure_png_bytes(figure),
                    file_name=f"{_download_file_stem(page, configured_tile)}.png",
                    mime="image/png",
                    icon=":material/image:",
                    key=f"{action_prefix}_png",
                    width="stretch",
                )
            else:
                st.download_button(
                    "Chart HTML",
                    data=lambda: figure.to_html(include_plotlyjs="cdn", full_html=True),
                    file_name=f"{_download_file_stem(page, configured_tile)}.html",
                    mime="text/html",
                    icon=":material/image:",
                    key=f"{action_prefix}_html",
                    width="stretch",
                )
    return inspect_mode or local_inspect, expanded, show_data


def _tile_action_prefix(dashboard_id: str, page_id: str, tile: Mapping[str, Any]) -> str:
    return f"reports_tile_action_{dashboard_id}_{page_id}_{tile.get('id', 'tile')}"


def _tile_expanded(dashboard_id: str, page_id: str, tile: Mapping[str, Any]) -> bool:
    return bool(
        st.session_state.get(f"{_tile_action_prefix(dashboard_id, page_id, tile)}_expanded")
    )


def _rows_csv(rows: pl.DataFrame) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=rows.columns)
    writer.writeheader()
    for row in rows.iter_rows(named=True):
        writer.writerow({key: _csv_cell_value(value) for key, value in row.items()})
    return buffer.getvalue()


def _csv_cell_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes | bytearray):
        encoded = base64.b64encode(bytes(value)).decode("ascii")
        return f"base64:{encoded}"
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, Mapping | list | tuple):
        return json.dumps(_json_safe_value(value), sort_keys=True)
    return value


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes | bytearray):
        encoded = base64.b64encode(bytes(value)).decode("ascii")
        return f"base64:{encoded}"
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(nested) for key, nested in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe_value(nested) for nested in value]
    return str(value)


@functools.cache
def _png_export_available() -> bool:
    return importlib.util.find_spec("kaleido") is not None


def _figure_png_bytes(figure: Any) -> bytes:
    """Render a figure to PNG; called lazily when a download is clicked."""
    try:
        return bytes(figure.to_image(format="png", scale=2))
    except Exception:
        logger.exception("Failed to export chart PNG")
        raise


def _download_file_stem(page: Any, tile: Mapping[str, Any]) -> str:
    raw = f"{getattr(page, 'id', 'report')}_{tile.get('id', 'tile')}"
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in raw)


def _tile_inspectors(
    tile_dict: dict[str, Any],
    rows: pl.DataFrame,
    *,
    include_data: bool = True,
) -> None:
    if not include_data:
        st.code(yaml.safe_dump(tile_dict, sort_keys=False), language="yaml")
        return
    data_tab, yaml_tab = st.tabs(["Data Overview", "YAML"])
    with data_tab:
        _tile_data_table(tile_dict, rows)
    with yaml_tab:
        st.code(yaml.safe_dump(tile_dict, sort_keys=False), language="yaml")


def _tile_data_table(tile_dict: Mapping[str, Any], rows: pl.DataFrame) -> None:
    if rows.is_empty():
        st.info("No rows returned.")
        return
    displayed_rows = (
        prepare_table_data(rows, tile_dict)
        if str(tile_dict.get("chart", "")).casefold() == "table"
        else rows
    )
    displayed, column_config = _present_table_data(tile_dict, displayed_rows)
    st.dataframe(
        displayed,
        hide_index=True,
        width="stretch",
        height=REPORT_DATAFRAME_HEIGHT_PX,
        column_config=column_config or None,
        placeholder="—",
    )


def _native_table_data(
    tile_dict: Mapping[str, Any], rows: pl.DataFrame
) -> tuple[pl.DataFrame, dict[str, Any], list[str | None]]:
    prepared = prepare_table_data(rows, tile_dict)
    displayed, column_config = _present_table_data(tile_dict, prepared)
    return displayed, column_config, table_row_colors(prepared, tile_dict)


def _present_table_data(
    tile_dict: Mapping[str, Any], rows: pl.DataFrame
) -> tuple[pl.DataFrame, dict[str, Any]]:
    labels = tile_dict.get("labels")
    rename = {
        column: str(labels[column])
        for column in rows.columns
        if isinstance(labels, Mapping)
        and column in labels
        and labels[column]
        and str(labels[column]) not in rows.columns
    }
    displayed = rows.rename(rename)
    raw_config = _percent_column_config(dict(tile_dict), rows) or {}
    column_config = {rename.get(column, column): config for column, config in raw_config.items()}
    for column in rows.columns:
        if rows.schema[column].is_numeric():
            column_config.setdefault(
                rename.get(column, column),
                st.column_config.NumberColumn(format="localized"),
            )
    return displayed, column_config


def _render_report_table(
    displayed: pl.DataFrame,
    column_config: Mapping[str, Any],
    row_colors: list[str | None],
    *,
    key: str,
) -> None:
    if displayed.is_empty():
        st.info("No rows returned.")
        return
    rendered: Any = displayed
    if any(row_colors):
        row_styles = [f"background-color: {color}" if color else "" for color in row_colors]
        rendered = displayed.to_pandas().style.apply(
            lambda row: [row_styles[int(row.name)]] * len(row),
            axis=1,
        )
    st.dataframe(
        rendered,
        hide_index=True,
        width="stretch",
        height="auto",
        column_config=column_config or None,
        key=key,
        placeholder="—",
    )


def _advanced_tile_controls(
    ctx: ValueStreamContext,
    *,
    dashboard_id: str,
    page_id: str,
    base_tile: dict[str, Any],
) -> dict[str, Any]:
    state_key = _advanced_tile_state_key(dashboard_id, page_id, base_tile)
    prefix = _advanced_tile_control_prefix(dashboard_id, page_id, base_tile)
    base_signature_key = f"{prefix}_base_signature"
    base_signature = yaml.safe_dump(base_tile, sort_keys=True)
    if st.session_state.get(base_signature_key) != base_signature:
        st.session_state[state_key] = dict(base_tile)
        st.session_state[base_signature_key] = base_signature

    current = dict(st.session_state.get(state_key, base_tile))
    with st.expander("Advanced mode", expanded=False):
        st.caption("Session-only. Changes here reset when the session ends and never update YAML.")
        reset_col, status_col = st.columns([0.22, 0.78], vertical_alignment="center")
        if reset_col.button(
            "Reset",
            icon=":material/restart_alt:",
            key=f"{prefix}_reset",
            help="Return this tile to its configured report definition.",
        ):
            _clear_advanced_tile_state(prefix, state_key)
            st.rerun()
        if current != base_tile:
            status_col.caption("Using an unsaved advanced draft.")
        else:
            status_col.caption("Using the configured report tile.")

        fields_tab, style_tab, yaml_tab = st.tabs(["Fields", "Style", "YAML"])
        with fields_tab:
            draft = _advanced_field_controls(ctx, base_tile, current, prefix)
        with style_tab:
            draft = _advanced_style_controls(draft, prefix)
        with yaml_tab:
            st.code(yaml.safe_dump(draft, sort_keys=False), language="yaml")

    st.session_state[state_key] = draft
    return draft


def _advanced_field_controls(
    ctx: ValueStreamContext,
    base_tile: dict[str, Any],
    current: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    metric_name = str(base_tile["metric"])
    chart_choices = builder.chart_choices_for_metric(ctx.catalog, metric_name)
    if not chart_choices:
        chart_choices = [str(base_tile["chart"])]
    chart_key = f"{prefix}_chart"
    chart_index = _widget_option_index(
        chart_key,
        str(current.get("chart", base_tile["chart"])),
        chart_choices,
    )
    chart_kind = st.selectbox(
        "Chart",
        chart_choices,
        index=chart_index,
        key=chart_key,
    )

    field_options = ["", *builder.chart_field_options(ctx.catalog, metric_name)]
    seed = _advanced_tile_seed(ctx.catalog, base_tile, current, chart_kind)
    selected_fields: dict[str, Any] = {}
    controls = ADVANCED_FIELD_CONTROLS.get(
        chart_kind, ("x", "y", "color", "facet_row", "facet_col")
    )
    two_col_controls = [
        control for control in controls if control not in {"columns", "path", "stages"}
    ]
    for offset in range(0, len(two_col_controls), 2):
        columns = st.columns(2)
        for column, field_name in zip(columns, two_col_controls[offset : offset + 2], strict=False):
            with column:
                options = _advanced_field_options(
                    ctx.catalog,
                    metric_name,
                    str(chart_kind),
                    field_name,
                    seed,
                    selected_fields,
                    field_options,
                )
                selected_fields[field_name] = _advanced_field_widget(
                    field_name,
                    seed.get(field_name),
                    options,
                    prefix,
                )
    if "path" in controls:
        selected_fields["path"] = st.multiselect(
            "Path",
            field_options[1:],
            default=[
                str(value)
                for value in seed.get("path", [])
                if isinstance(value, str) and value in field_options
            ],
            key=f"{prefix}_path",
        )
    if "columns" in controls:
        selected_fields["columns"] = st.multiselect(
            "Columns",
            field_options[1:],
            default=[
                str(value)
                for value in seed.get("columns", [])
                if isinstance(value, str) and value in field_options
            ],
            key=f"{prefix}_columns",
        )
    if "stages" in controls:
        selected_fields["stages"] = _stage_list_control(seed.get("stages"), prefix)
    return _advanced_tile_from_fields(base_tile, chart_kind, selected_fields)


def _advanced_style_controls(  # noqa: PLR0912, PLR0915
    draft: dict[str, Any], prefix: str
) -> dict[str, Any]:
    out = dict(draft)
    value_format_key = f"{prefix}_value_format"
    value_format_index = _widget_option_index(
        value_format_key,
        str(out.get("value_format") or ""),
        ADVANCED_VALUE_FORMATS,
    )
    selected_format = st.selectbox(
        "Value Format",
        ADVANCED_VALUE_FORMATS,
        index=value_format_index,
        format_func=lambda value: "Default" if value == "" else str(value).title(),
        key=value_format_key,
    )
    _set_optional_field(out, "value_format", selected_format)

    if out.get("chart") in {"line", "stacked_area"}:
        scale_options = ["absolute", "index_100", "percent_change"]
        scale_key = f"{prefix}_scale_mode"
        scale_index = _widget_option_index(
            scale_key,
            str(out.get("scale_mode") or "absolute"),
            scale_options,
        )
        out["scale_mode"] = st.selectbox(
            "Scale",
            scale_options,
            index=scale_index,
            format_func=lambda value: value.replace("_", " ").title(),
            key=scale_key,
        )
    else:
        out.pop("scale_mode", None)

    out = _advanced_gauge_reference_controls(out, prefix)

    if out.get("chart") in {"heatmap", "descriptive_heatmap"}:
        color_scale_key = f"{prefix}_color_scale"
        color_scale_index = _widget_option_index(
            color_scale_key,
            str(out.get("color_continuous_scale") or ""),
            ADVANCED_COLOR_SCALES,
        )
        color_scale = st.selectbox(
            "Color Scale",
            ADVANCED_COLOR_SCALES,
            index=color_scale_index,
            format_func=lambda value: "Default" if value == "" else str(value),
            key=color_scale_key,
        )
        _set_optional_field(out, "color_continuous_scale", color_scale)
    else:
        out.pop("color_continuous_scale", None)

    show_trend_delta = st.checkbox(
        "Show Trend Delta",
        value=bool(out.get("show_trend_delta", out.get("trend_delta", False))),
        key=f"{prefix}_show_trend_delta",
    )
    if show_trend_delta:
        out["show_trend_delta"] = True
    else:
        out.pop("show_trend_delta", None)

    goal_enabled = st.checkbox(
        "Goal Line",
        value=out.get("goal_line") not in (None, "", []),
        key=f"{prefix}_goal_enabled",
    )
    if goal_enabled:
        goal_value, goal_label, goal_color = _goal_line_defaults(out.get("goal_line"))
        out["goal_line"] = {
            "value": st.number_input(
                "Goal Value",
                value=goal_value,
                key=f"{prefix}_goal_value",
            ),
            "label": st.text_input(
                "Goal Label",
                value=goal_label,
                key=f"{prefix}_goal_label",
            ),
            "color": st.color_picker(
                "Goal Color",
                value=goal_color,
                key=f"{prefix}_goal_color",
            ),
        }
    else:
        out.pop("goal_line", None)

    if out.get("chart") == "bar":
        bar_cols = st.columns(2)
        with bar_cols[0]:
            barmode_key = f"{prefix}_barmode"
            barmode_options = ["group", "stack", "relative", "percent"]
            barmode_index = _widget_option_index(
                barmode_key,
                str(out.get("barmode") or "group"),
                barmode_options,
            )
            barmode = st.selectbox(
                "Bar Mode",
                barmode_options,
                index=barmode_index,
                key=barmode_key,
            )
            _set_optional_field(out, "barmode", "" if barmode == "group" else barmode)
        with bar_cols[1]:
            top_n = st.number_input(
                "Top N",
                min_value=0,
                value=int(out.get("top_n") or 0),
                step=1,
                key=f"{prefix}_top_n",
            )
            _set_optional_field(out, "top_n", int(top_n) if top_n else None)
        sort_options = _advanced_sort_options(out)
        sort_by_key = f"{prefix}_sort_by"
        sort_by_index = _widget_option_index(
            sort_by_key, str(out.get("sort_by") or ""), sort_options
        )
        sort_by = st.selectbox(
            "Sort By",
            sort_options,
            index=sort_by_index,
            key=sort_by_key,
        )
        _set_optional_field(out, "sort_by", sort_by)
        if sort_by:
            sort_direction_key = f"{prefix}_sort_direction"
            sort_direction_index = _widget_option_index(
                sort_direction_key,
                str(out.get("sort_direction") or "desc"),
                ["desc", "asc"],
            )
            sort_direction = st.selectbox(
                "Sort Direction",
                ["desc", "asc"],
                index=sort_direction_index,
                key=sort_direction_key,
            )
            out["sort_direction"] = sort_direction
        else:
            out.pop("sort_direction", None)
    else:
        for key in ("barmode", "sort_by", "sort_direction", "top_n"):
            out.pop(key, None)
    return out


def _advanced_field_widget(
    field_name: str,
    value: Any,
    field_options: list[str],
    prefix: str,
) -> str:
    if field_name == "score":
        label = "Metric"
    elif field_name == "property":
        label = "Property"
    else:
        label = field_name.upper() if len(field_name) == 1 else field_name.replace("_", " ").title()
    key = f"{prefix}_{field_name}"
    default = str(value) if isinstance(value, str) and value in field_options else ""
    index = _widget_option_index(key, default, field_options)
    return st.selectbox(
        label,
        field_options,
        index=index,
        format_func=lambda item: "None" if item == "" else item,
        key=key,
    )


def _advanced_field_options(
    catalog: model.Catalog,
    metric_name: str,
    chart_kind: str,
    field_name: str,
    seed: Mapping[str, Any],
    selected_fields: Mapping[str, Any],
    field_options: list[str],
) -> list[str]:
    if not chart_kind.startswith("descriptive_"):
        return field_options
    if field_name == "property":
        properties = builder.descriptive_property_options(catalog, metric_name)
        return properties or field_options
    if field_name == "score":
        selected_property = str(selected_fields.get("property") or seed.get("property") or "")
        scores = builder.descriptive_score_options(catalog, metric_name, selected_property)
        return scores or ["Mean"]
    return field_options


def _stage_list_control(value: Any, prefix: str) -> list[str]:
    default = ", ".join(str(item) for item in value) if isinstance(value, list) else ""
    raw = st.text_input(
        "Stages",
        value=default,
        key=f"{prefix}_stages",
        help="Comma-separated stage names for funnel-style charts.",
    )
    return [item.strip() for item in raw.split(",") if item.strip()]


def _advanced_tile_seed(
    catalog: model.Catalog,
    base_tile: dict[str, Any],
    current_tile: dict[str, Any],
    chart_kind: str,
) -> dict[str, Any]:
    seed = builder.default_tile_fields(catalog, str(base_tile["metric"]), chart_kind)
    for source in (base_tile, current_tile):
        for key, value in source.items():
            if key in ADVANCED_FIELD_KEYS or key in ADVANCED_SETTING_KEYS:
                seed[key] = value
        facets = source.get("facets")
        if isinstance(facets, Mapping):
            if facets.get("row"):
                seed["facet_row"] = facets["row"]
            if facets.get("col") or facets.get("column"):
                seed["facet_col"] = facets.get("col", facets.get("column"))
    return seed


def _advanced_tile_from_fields(
    base_tile: dict[str, Any],
    chart_kind: str,
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(base_tile)
    out["chart"] = chart_kind
    for key in ADVANCED_FIELD_KEYS:
        out.pop(key, None)
    for key, value in fields.items():
        _set_optional_field(out, key, value)
    return out


def _set_optional_field(tile: dict[str, Any], key: str, value: Any) -> None:
    if value in (None, "", []):
        tile.pop(key, None)
    else:
        tile[key] = value


def _advanced_sort_options(tile: Mapping[str, Any]) -> list[str]:
    options = [""]
    for key in ("y", "value", "color", "x"):
        value = tile.get(key)
        if isinstance(value, str) and value and value not in options:
            options.append(value)
    return options


def _advanced_tile_state_key(dashboard_id: str, page_id: str, tile: Mapping[str, Any]) -> str:
    return f"reports_advanced_tile_{dashboard_id}_{page_id}_{tile.get('id', 'tile')}"


def _advanced_tile_control_prefix(dashboard_id: str, page_id: str, tile: Mapping[str, Any]) -> str:
    return f"reports_advanced_control_{dashboard_id}_{page_id}_{tile.get('id', 'tile')}"


def _clear_advanced_tile_state(prefix: str, state_key: str) -> None:
    for key in list(st.session_state):
        if str(key).startswith(prefix):
            st.session_state.pop(key, None)
    st.session_state.pop(state_key, None)


def _widget_option_index(key: str, value: str, options: list[str]) -> int:
    if st.session_state.get(key) not in (None, *options):
        st.session_state.pop(key, None)
    selected = st.session_state.get(key)
    if isinstance(selected, str) and selected in options:
        return options.index(selected)
    return _option_index(options, value)


def _option_index(options: list[str], value: Any) -> int:
    return options.index(value) if isinstance(value, str) and value in options else 0


def _goal_line_defaults(raw: Any) -> tuple[float, str, str]:
    if isinstance(raw, Mapping):
        return (
            float(raw.get("value", 0.0) or 0.0),
            str(raw.get("label", "Goal")),
            str(raw.get("color", "#475569")),
        )
    if isinstance(raw, int | float):
        return float(raw), "Goal", "#475569"
    return 0.0, "Goal", "#475569"


def _advanced_gauge_reference_controls(draft: dict[str, Any], prefix: str) -> dict[str, Any]:
    out = dict(draft)
    if out.get("chart") != "gauge":
        out.pop("reference", None)
        out.pop("references", None)
        return out

    reference_value = _reference_number(out.get("reference"))
    reference_enabled = st.checkbox(
        "Reference",
        value=reference_value is not None,
        key=f"{prefix}_reference_enabled",
    )
    if reference_enabled:
        out["reference"] = st.number_input(
            "Reference Value",
            value=reference_value or 0.0,
            key=f"{prefix}_reference_value",
        )
        out.pop("references", None)
    elif not isinstance(out.get("references"), Mapping):
        out.pop("reference", None)
    return out


def _reference_number(raw: Any) -> float | None:
    if isinstance(raw, (bool, Mapping)) or raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _format_metric_value(value: Any, value_format: str | None = None) -> str:
    return components.format_metric_value(value, value_format)


def _report_chart_height(
    figure: Any,
    *,
    layout_mode: str = TILE_LAYOUT_HALF,
) -> int:
    layout_height = getattr(getattr(figure, "layout", None), "height", None)
    chart_height = _layout_fallback_height(layout_mode)
    if isinstance(layout_height, int | float):
        return max(chart_height, int(layout_height))
    return chart_height


def _layout_fallback_height(layout_mode: str) -> int:
    if layout_mode == TILE_LAYOUT_HERO:
        return REPORT_CHART_HEIGHT_HERO_PX
    if layout_mode == TILE_LAYOUT_EXPANDED:
        return REPORT_CHART_HEIGHT_EXPANDED_PX
    if layout_mode == TILE_LAYOUT_FULL:
        return REPORT_CHART_HEIGHT_FALLBACK_PX
    if layout_mode == TILE_LAYOUT_COMPACT:
        return REPORT_CHART_HEIGHT_MIN_PX
    return REPORT_CHART_HEIGHT_FALLBACK_PX


def _percent_column_config(tile_dict: dict[str, Any], rows: pl.DataFrame) -> dict[str, Any] | None:
    """Return percent formatting for value columns when the tile asks for it."""
    if (
        str(tile_dict.get("value_format", tile_dict.get("number_format", ""))).casefold()
        != "percent"
    ):
        return None
    value_columns = _tile_value_columns(tile_dict, rows)
    if not value_columns:
        return None
    return {column: st.column_config.NumberColumn(format="percent") for column in value_columns}


def _tile_value_columns(tile_dict: dict[str, Any], rows: pl.DataFrame) -> list[str]:
    columns: list[str] = []
    for key in ("y", "value"):
        value = tile_dict.get(key)
        if isinstance(value, str):
            columns.append(value)
    metric = tile_dict.get("metric")
    if isinstance(metric, str):
        columns.append(metric)
    return [
        column
        for column in dict.fromkeys(columns)
        if column in rows.columns and rows.schema[column].is_numeric()
    ]
