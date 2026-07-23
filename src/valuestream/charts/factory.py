"""Plotly chart factory for dashboard tiles."""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from collections.abc import Iterable, Mapping
from itertools import pairwise
from typing import Any

import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import plotly.io as pio  # type: ignore[import-untyped]
import polars as pl
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from valuestream.charts import lttb
from valuestream.states import kll, tdigest
from valuestream.utils.timer import timed

MAX_POINTS = 50_000
SCATTER_SIZE_MAX = 72
LINE_AS_BAR_MAX_DISTINCT_X = 30
TABLE_MIN_HEIGHT_PX = 140
TABLE_MAX_HEIGHT_PX = 520
TABLE_ROW_HEIGHT_PX = 22
TABLE_CHROME_HEIGHT_PX = 72
_DESCRIPTIVE_QUANTILES = {
    "Median": 0.5,
    "p25": 0.25,
    "p50": 0.5,
    "p75": 0.75,
    "p90": 0.9,
    "p95": 0.95,
}
_TIME_COLUMNS = {
    "Day",
    "day",
    "Week",
    "week",
    "Month",
    "month",
    "Quarter",
    "quarter",
    "Year",
    "year",
    "period",
    "as_of_date",
}
_CURVE_CHARTS = {"roc_curve", "precision_recall_curve", "gain_curve", "lift_curve"}
_CURVE_LIST_COLUMNS = {"fpr", "tpr", "precision", "recall"}
_COLOR_VALUE_CHARTS = {
    "calendar_heatmap",
    "cohort_heatmap",
    "descriptive_heatmap",
    "heatmap",
    "treemap",
}
_TREEMAP_LIGHT_COLORSCALE = [
    "#334155",
    "#315B73",
    "#287C8E",
    "#2E9B88",
    "#6BBF72",
    "#B7D968",
]
_TREEMAP_DARK_COLORSCALE = [
    "#223046",
    "#1E4F66",
    "#1D7482",
    "#2CA58D",
    "#66C96E",
    "#C7E77A",
]
_HEATMAP_LIGHT_COLORSCALE = [
    "#2563EB",
    "#38BDF8",
    "#E0F2FE",
    "#FFF7ED",
    "#FDBA74",
    "#DC2626",
]
# Dark diverging scale: bright poles, near-surface midpoint; lightness is
# monotone toward the midpoint per arm so magnitude reads without false bands.
_HEATMAP_DARK_COLORSCALE = [
    "#5598E7",
    "#1C5CAB",
    "#26292C",
    "#7C2D12",
    "#EA580C",
    "#FCA5A5",
]

# Chart chrome that Plotly templates cannot express per-element. Light values
# keep the historical defaults; dark values are checked against the #020203
# surface (facet labels 6.4:1, goal lines 8.1:1, reference lines 7.1:1).
_CHART_INKS: dict[str, dict[str, str]] = {
    "light": {
        "facet_label": "#5d6470",
        "goal_line": "#475569",
        "reference_line": "darkred",
        "gauge_threshold": "#c62828",
        "table_header_fill": "#e2e8f0",
        "table_cell_fill": "#ffffff",
    },
    "dark": {
        "facet_label": "#8f8f8f",
        "goal_line": "#94a3b8",
        "reference_line": "#f26d6d",
        "gauge_threshold": "#f14d4c",
        "table_header_fill": "#101214",
        "table_cell_fill": "#050607",
    },
}


@timed
def render_chart(  # noqa: PLR0912, PLR0915
    rows: pl.DataFrame,
    tile: Mapping[str, Any],
    *,
    theme: Mapping[str, Any] | None = None,
    max_points: int = MAX_POINTS,
) -> go.Figure:
    """Render a tile's aggregate rows as a Plotly figure."""
    tile = dict(tile)
    rows = _apply_scale_mode(rows, tile)
    if tile.get("scale_mode") == "percent_change" and not tile.get("value_format"):
        tile["value_format"] = "percent"
        tile["y_axis_title"] = f"{tile.get('y_axis_title') or tile.get('y', 'Value')} change"
    elif tile.get("scale_mode") == "index_100":
        tile["y_axis_title"] = (
            f"{tile.get('y_axis_title') or tile.get('y', 'Value')} (index, first = 100)"
        )
    kind = str(tile["chart"])
    if kind == "line":
        fig = _line(rows, tile, max_points=max_points)
    elif kind == "stacked_area":
        fig = _stacked_area(rows, tile, max_points=max_points)
    elif kind == "bar":
        fig = _bar(rows, tile)
    elif kind == "kpi_card":
        fig = _kpi_card(rows, tile)
    elif kind == "waterfall":
        fig = _waterfall(rows, tile)
    elif kind == "pareto":
        fig = _pareto(rows, tile)
    elif kind == "treemap":
        fig = _treemap(rows, tile, theme or {})
    elif kind == "heatmap":
        fig = _heatmap(rows, tile, theme or {})
    elif kind == "cohort_heatmap":
        fig = _cohort_heatmap(rows, tile, theme or {})
    elif kind == "scatter":
        fig = _scatter(rows, tile, max_points=max_points)
    elif kind == "combo":
        fig = _combo(rows, tile)
    elif kind == "interval":
        fig = _interval(rows, tile)
    elif kind == "donut":
        fig = _donut(rows, tile)
    elif kind == "geo_map":
        fig = _geo_map(rows, tile)
    elif kind == "table":
        fig = _table(rows, tile, theme or {})
    elif kind == "calendar_heatmap":
        fig = _calendar_heatmap(rows, tile)
    elif kind == "bar_polar":
        fig = _bar_polar(rows, tile)
    elif kind == "sankey":
        fig = _sankey(rows, tile)
    elif kind == "gauge":
        fig = _gauge(rows, tile, theme or {})
    elif kind == "funnel":
        fig = _funnel(rows, tile)
    elif kind == "boxplot":
        fig = _boxplot(rows, tile, theme or {})
    elif kind == "histogram":
        fig = _histogram(rows, tile)
    elif kind == "calibration_curve":
        fig = _calibration_curve(rows, tile, theme or {})
    elif kind == "roc_curve":
        fig = _roc_curve(rows, tile, theme or {})
    elif kind == "precision_recall_curve":
        fig = _precision_recall_curve(rows, tile)
    elif kind == "gain_curve":
        fig = _gain_curve(rows, tile, theme or {})
    elif kind == "lift_curve":
        fig = _lift_curve(rows, tile, theme or {})
    elif kind == "rfm_density":
        fig = _rfm_density(rows, tile)
    elif kind == "exposure":
        fig = _exposure(rows, tile)
    elif kind == "corr":
        fig = _corr(rows, tile)
    elif kind == "model":
        fig = _model(rows, tile)
    elif kind.startswith("descriptive_"):
        fig = _descriptive(rows, tile, theme or {})
    elif kind == "experiment_z_score":
        fig = _experiment_z_score(rows, tile)
    elif kind == "experiment_odds_ratio":
        fig = _experiment_odds_ratio(rows, tile, theme or {})
    elif kind == "clv_treemap":
        fig = _clv_treemap(rows, tile)
    else:
        raise ValueError(f"unsupported chart kind {kind!r}")
    base = _theme_base(tile, theme or {})
    _apply_accessibility(fig, tile)
    _apply_label_overrides(fig, tile)
    _apply_goal_lines(fig, tile, base)
    _apply_value_format(fig, tile)
    _strip_facet_annotation_prefixes(fig, tile, base)
    _apply_theme(fig, theme or {}, tile)
    _apply_display_labels(fig, tile)
    _apply_semantic_category_colors(fig, tile, theme or {})
    # After _apply_theme so the top-margin reserved for the delta badge is not
    # clobbered by the theme's default margins.
    _apply_trend_delta(fig, rows, tile)
    return fig


def _apply_scale_mode(rows: pl.DataFrame, tile: Mapping[str, Any]) -> pl.DataFrame:
    """Normalize aggregate result rows for comparison without changing stored metrics."""

    mode = str(tile.get("scale_mode", "absolute"))
    if mode == "absolute" or rows.is_empty():
        return rows
    if str(tile.get("chart")) not in {"line", "stacked_area"}:
        return rows
    x = tile.get("x")
    y = tile.get("y")
    if not isinstance(x, str) or not isinstance(y, str) or {x, y} - set(rows.columns):
        return rows
    if not rows.schema[y].is_numeric():
        return rows
    groups = [
        str(value)
        for value in (
            tile.get("color"),
            _facet_row(tile),
            _facet_col(tile),
        )
        if isinstance(value, str) and value in rows.columns
    ]
    ordered = rows.sort([*groups, x] if groups else x)
    baseline = pl.col(y).first().over(groups) if groups else pl.col(y).first()
    valid = baseline.is_not_null() & (baseline != 0)
    if mode == "index_100":
        expression = pl.when(valid).then(pl.col(y) / baseline * 100.0).otherwise(None)
    elif mode == "percent_change":
        expression = pl.when(valid).then(pl.col(y) / baseline - 1.0).otherwise(None)
    else:
        return rows
    return ordered.with_columns(expression.alias(y))


def _line(rows: pl.DataFrame, tile: Mapping[str, Any], *, max_points: int) -> go.Figure:
    x = _field(tile, "x")
    y = _field(tile, "y")
    plotted = rows.sort(x) if x in rows.columns else rows
    if isinstance(y, str) and rows.height > max_points:
        plotted = _downsample_by_color(
            plotted, x=x, y=y, color=tile.get("color"), max_points=max_points
        )
    if _should_render_line_as_grouped_bar(rows, x):
        return _bar(plotted, {**tile, "barmode": tile.get("barmode", "group")})
    fig = px.line(
        plotted,
        x=x,
        y=y,
        color=tile.get("color"),
        facet_row=_facet_row(tile),
        facet_col=_facet_col(tile),
        log_x=bool(tile.get("log_x", False)),
        log_y=bool(tile.get("log_y", False)),
        title=str(tile.get("title", "")),
    )
    fig.update_layout(showlegend=bool(tile.get("showlegend", True)))
    return fig


def _should_render_line_as_grouped_bar(rows: pl.DataFrame, x: str) -> bool:
    if x not in rows.columns:
        return False
    return rows.get_column(x).drop_nulls().n_unique() < LINE_AS_BAR_MAX_DISTINCT_X


def _stacked_area(rows: pl.DataFrame, tile: Mapping[str, Any], *, max_points: int) -> go.Figure:
    x = _field(tile, "x")
    y = _field(tile, "y")
    plotted = rows.sort(x) if x in rows.columns else rows
    if isinstance(tile.get("color"), str) and rows.height > max_points:
        plotted = _downsample_by_color(
            plotted, x=x, y=y, color=tile.get("color"), max_points=max_points
        )
    groupnorm = str(tile.get("groupnorm", "")).casefold()
    if groupnorm in {"percent", "fraction"}:
        groupnorm = "percent"
    elif groupnorm in {"", "none"}:
        groupnorm = ""
    fig = px.area(
        plotted,
        x=x,
        y=y,
        color=tile.get("color"),
        facet_row=_facet_row(tile),
        facet_col=_facet_col(tile),
        groupnorm=groupnorm or None,
        title=str(tile.get("title", "")),
    )
    fig.update_layout(showlegend=bool(tile.get("showlegend", True)))
    return fig


def _bar(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    plotted = _sort_and_limit(rows, tile, default_sort=_field(tile, "y"))
    barmode, barnorm = _bar_mode(tile)
    fig = px.bar(
        plotted,
        x=_field(tile, "x"),
        y=_field(tile, "y"),
        color=tile.get("color"),
        facet_row=_facet_row(tile),
        facet_col=_facet_col(tile),
        barmode=barmode,
        title=str(tile.get("title", "")),
    )
    if barnorm:
        fig.update_layout(barnorm=barnorm)
    if tile.get("color") in (None, ""):
        colors = _conditional_colors(plotted, tile)
        if colors:
            fig.update_traces(marker_color=colors)
    fig.update_layout(showlegend=bool(tile.get("showlegend", True)))
    return fig


def _kpi_card(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    value_col = str(tile.get("value") or tile.get("y") or tile.get("metric") or "")
    if value_col not in rows.columns:
        value_col = _first_numeric_column(rows) or (rows.columns[-1] if rows.columns else "")
    value = _as_float(rows[value_col][0]) if value_col in rows.columns and rows.height else 0.0
    reference = _kpi_reference(rows, tile)
    mode = "number+delta" if reference is not None else "number"
    indicator: dict[str, Any] = {
        "mode": mode,
        "value": value or 0.0,
        "title": {"text": str(tile.get("title", ""))},
    }
    if reference is not None:
        indicator["delta"] = {"reference": reference}
    fig = go.Figure(go.Indicator(**indicator))
    fig.update_layout(height=int(tile.get("height", 260)))
    return fig


def _waterfall(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    plotted = _sort_and_limit(rows, tile, default_sort=_field(tile, "y"))
    x = _field(tile, "x")
    y = _field(tile, "y")
    measure_col = tile.get("measure")
    measures = (
        [str(value) for value in plotted[str(measure_col)].to_list()]
        if isinstance(measure_col, str) and measure_col in plotted.columns
        else ["relative"] * plotted.height
    )
    fig = go.Figure(
        go.Waterfall(
            x=plotted[x].to_list(),
            y=plotted[y].to_list(),
            measure=measures,
            connector={"line": {"color": str(tile.get("connector_color", "#64748b"))}},
        )
    )
    fig.update_layout(title=str(tile.get("title", "")))
    return fig


def _pareto(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    plotted = _sort_and_limit(
        rows,
        {**tile, "sort_by": tile.get("sort_by") or _field(tile, "y")},
        default_sort=_field(tile, "y"),
    )
    x = _field(tile, "x")
    y = _field(tile, "y")
    values = [_as_float(value) or 0.0 for value in plotted[y].to_list()]
    total = sum(values)
    cumulative: list[float] = []
    running = 0.0
    for value in values:
        running += value
        cumulative.append(running / total if total else 0.0)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=plotted[x].to_list(), y=values, name=str(y)), secondary_y=False)
    fig.add_trace(
        go.Scatter(
            x=plotted[x].to_list(),
            y=cumulative,
            mode="lines+markers",
            name=str(tile.get("cumulative_label", "Cumulative share")),
        ),
        secondary_y=True,
    )
    fig.update_layout(title=str(tile.get("title", "")), showlegend=True)
    fig.update_yaxes(title_text=str(tile.get("y_axis_title") or y), secondary_y=False)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", secondary_y=True)
    return fig


def _treemap(rows: pl.DataFrame, tile: Mapping[str, Any], theme: Mapping[str, Any]) -> go.Figure:
    path = [px.Constant("All"), *_treemap_path(tile)]
    return px.treemap(
        rows,
        path=path,
        values=_treemap_value_field(tile),
        color=_optional_text(tile.get("color")),
        color_continuous_scale=_treemap_color_scale(tile, theme),
        title=str(tile.get("title", "")),
    )


def _treemap_path(tile: Mapping[str, Any]) -> list[str]:
    raw_path = tile.get("path")
    if isinstance(raw_path, list | tuple):
        return [str(item) for item in raw_path if item not in (None, "", "---")]
    if raw_path not in (None, "", "---"):
        return [str(raw_path)]
    for key in ("x", "names"):
        value = tile.get(key)
        if value not in (None, "", "---"):
            return [str(value)]
    return []


def _treemap_value_field(tile: Mapping[str, Any]) -> str | None:
    for key in ("value", "values", "y"):
        value = tile.get(key)
        if value not in (None, "", "---"):
            return str(value)
    return None


def _treemap_color_scale(tile: Mapping[str, Any], theme: Mapping[str, Any]) -> Any:
    override = tile.get("color_continuous_scale")
    if override not in (None, "", "---"):
        return override
    return (
        _TREEMAP_DARK_COLORSCALE
        if _theme_base(tile, theme) == "dark"
        else _TREEMAP_LIGHT_COLORSCALE
    )


def _theme_base(tile: Mapping[str, Any], theme: Mapping[str, Any]) -> str:
    raw_tile_theme = tile.get("theme")
    tile_theme: Mapping[str, Any] = raw_tile_theme if isinstance(raw_tile_theme, Mapping) else {}
    for source in (tile_theme, theme):
        value = source.get("base", source.get("theme_base"))
        if str(value).casefold() == "dark":
            return "dark"
    template = str(tile_theme.get("template", theme.get("template", ""))).casefold()
    return "dark" if "dark" in template else "light"


def _reference_line_color(tile: Mapping[str, Any], theme: Mapping[str, Any]) -> str:
    return _CHART_INKS[_theme_base(tile, theme)]["reference_line"]


def _heatmap_color_scale(tile: Mapping[str, Any], theme: Mapping[str, Any]) -> Any:
    override = tile.get("color_continuous_scale")
    if override not in (None, "", "---"):
        return override
    return (
        _HEATMAP_DARK_COLORSCALE
        if _theme_base(tile, theme) == "dark"
        else _HEATMAP_LIGHT_COLORSCALE
    )


def _heatmap(rows: pl.DataFrame, tile: Mapping[str, Any], theme: Mapping[str, Any]) -> go.Figure:
    matrix = rows.pivot(
        index=_field(tile, "y"),
        on=_field(tile, "x"),
        values=_field(tile, "color"),
        aggregate_function="mean",
    )
    labels = matrix[:, 0].to_list()
    values = matrix.drop(matrix.columns[0])
    fig = go.Figure(
        go.Heatmap(
            z=values.to_numpy().tolist(),
            x=values.columns,
            y=labels,
            colorscale=_heatmap_color_scale(tile, theme),
        )
    )
    fig.update_layout(title=str(tile.get("title", "")))
    return fig


def _cohort_heatmap(
    rows: pl.DataFrame, tile: Mapping[str, Any], theme: Mapping[str, Any]
) -> go.Figure:
    return _heatmap(rows, tile, theme)


def _scatter(rows: pl.DataFrame, tile: Mapping[str, Any], *, max_points: int) -> go.Figure:
    plotted = rows.head(max_points) if rows.height > max_points else rows
    plotted, size_column = _scatter_size_column(plotted, tile)
    fig = px.scatter(
        plotted,
        x=_field(tile, "x"),
        y=_field(tile, "y"),
        color=tile.get("color"),
        size=size_column,
        animation_frame=tile.get("animation_frame"),
        animation_group=tile.get("animation_group"),
        facet_row=_facet_row(tile),
        facet_col=_facet_col(tile),
        size_max=SCATTER_SIZE_MAX,
        log_x=bool(tile.get("log_x", False)),
        log_y=bool(tile.get("log_y", False)),
        title=str(tile.get("title", "")),
    )
    if rows.height > max_points:
        fig.add_annotation(
            text=f"Showing first {max_points:,} aggregate rows",
            xref="paper",
            yref="paper",
            x=1,
            y=1,
            showarrow=False,
        )
    if tile.get("color") in (None, ""):
        colors = _conditional_colors(plotted, tile)
        if colors:
            fig.update_traces(marker_color=colors)
    return fig


def _scatter_size_column(
    rows: pl.DataFrame, tile: Mapping[str, Any]
) -> tuple[pl.DataFrame, str | None]:
    size = _optional_column(rows, tile.get("size"))
    if size is None:
        return rows, None
    marker_size = "__valuestream_marker_size"
    raw = pl.col(size).cast(pl.Float64, strict=False)
    sanitized = pl.when(raw.is_not_null() & raw.is_finite() & (raw >= 0.0)).then(raw).otherwise(0.0)
    scaled = pl.when(sanitized > 0.0).then((sanitized + 1.0).log()).otherwise(0.0)
    out = rows.with_columns(scaled.alias(marker_size))
    max_size = out[marker_size].max()
    if max_size is None or max_size <= 0:
        return rows, None
    return out, marker_size


def _combo(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    x = _field(tile, "x")
    y = _field(tile, "y")
    y2 = str(tile.get("y2") or tile.get("line_y") or "")
    if y2 not in rows.columns:
        raise ValueError("combo chart requires a secondary y field via 'y2' or 'line_y'")
    plotted = rows.sort(x) if x in rows.columns else rows
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    color = _optional_column(plotted, tile.get("color"))
    groups = plotted.partition_by(color, as_dict=True) if color else {None: plotted}
    for group, sub in groups.items():
        suffix = f" · {group}" if group is not None else ""
        fig.add_trace(
            go.Bar(x=sub[x].to_list(), y=sub[y].to_list(), name=f"{y}{suffix}"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=sub[x].to_list(),
                y=sub[y2].to_list(),
                mode="lines+markers",
                name=f"{y2}{suffix}",
            ),
            secondary_y=True,
        )
    fig.update_layout(title=str(tile.get("title", "")), barmode=str(tile.get("barmode", "group")))
    fig.update_yaxes(title_text=str(tile.get("y_axis_title") or y), secondary_y=False)
    fig.update_yaxes(title_text=str(tile.get("y2_axis_title") or y2), secondary_y=True)
    return fig


def _interval(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    plotted = rows.sort(_field(tile, "x")) if _field(tile, "x") in rows.columns else rows
    y = _field(tile, "y")
    lower = _optional_column(plotted, tile.get("error_y_lower"))
    upper = _optional_column(plotted, tile.get("error_y_upper"))
    error_y = _optional_column(plotted, tile.get("error_y") or tile.get("error_y_plus"))
    error_y_minus = _optional_column(plotted, tile.get("error_y_minus"))
    if lower and upper and y in plotted.columns:
        error_y = "__interval_error_plus"
        error_y_minus = "__interval_error_minus"
        plotted = plotted.with_columns(
            (pl.col(upper) - pl.col(y)).clip(lower_bound=0).alias(error_y),
            (pl.col(y) - pl.col(lower)).clip(lower_bound=0).alias(error_y_minus),
        )
    fig = px.scatter(
        plotted,
        x=_field(tile, "x"),
        y=y,
        color=tile.get("color"),
        error_y=error_y,
        error_y_minus=error_y_minus,
        facet_row=_facet_row(tile),
        facet_col=_facet_col(tile),
        title=str(tile.get("title", "")),
    )
    fig.update_traces(mode="markers")
    return fig


def _donut(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    names = str(tile.get("names") or tile.get("x") or "label")
    values = str(tile.get("values") or tile.get("y") or tile.get("value") or "value")
    plotted = _sort_and_limit(
        rows, {**tile, "sort_by": tile.get("sort_by") or values}, default_sort=values
    )
    fig = go.Figure(
        go.Pie(
            labels=plotted[names].to_list(),
            values=plotted[values].to_list(),
            hole=float(tile.get("hole", 0.45)),
        )
    )
    fig.update_layout(title=str(tile.get("title", "")))
    return fig


def _geo_map(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    value = str(tile.get("value") or tile.get("color") or tile.get("y") or rows.columns[-1])
    lat = tile.get("lat")
    lon = tile.get("lon")
    if (
        isinstance(lat, str)
        and isinstance(lon, str)
        and lat in rows.columns
        and lon in rows.columns
    ):
        return px.scatter_geo(
            rows,
            lat=lat,
            lon=lon,
            size=tile.get("size") if tile.get("size") in rows.columns else value,
            color=tile.get("color") if tile.get("color") in rows.columns else value,
            hover_name=tile.get("hover_name"),
            title=str(tile.get("title", "")),
        )
    locations = str(tile.get("locations") or tile.get("location") or tile.get("x") or "")
    if locations not in rows.columns:
        raise ValueError("geo_map requires either lat/lon fields or a locations field")
    return px.choropleth(
        rows,
        locations=locations,
        color=value if value in rows.columns else None,
        locationmode=str(tile.get("locationmode", "ISO-3")),
        hover_name=tile.get("hover_name"),
        title=str(tile.get("title", "")),
    )


def _table(rows: pl.DataFrame, tile: Mapping[str, Any], theme: Mapping[str, Any]) -> go.Figure:
    frame = prepare_table_data(rows, tile)
    selected = list(frame.columns)
    inks = _CHART_INKS[_theme_base(tile, theme)]
    table_font = _table_font(theme)
    fig = go.Figure(
        go.Table(
            header={
                "values": selected,
                "fill_color": str(tile.get("header_fill_color", inks["table_header_fill"])),
                "align": "left",
                "font": table_font,
            },
            cells={
                "values": [
                    [_table_cell_value(value) for value in frame[column].to_list()]
                    for column in selected
                ],
                "fill_color": _table_fill_colors(frame, tile, inks["table_cell_fill"]),
                "align": "left",
                "font": table_font,
            },
        )
    )
    table_height = min(
        TABLE_MAX_HEIGHT_PX,
        max(TABLE_MIN_HEIGHT_PX, TABLE_CHROME_HEIGHT_PX + TABLE_ROW_HEIGHT_PX * frame.height),
    )
    fig.update_layout(title=str(tile.get("title", "")), height=table_height)
    return fig


def prepare_table_data(rows: pl.DataFrame, tile: Mapping[str, Any]) -> pl.DataFrame:
    """Select and normalize table columns for every presentation surface."""

    columns = tile.get("columns")
    if isinstance(columns, list):
        selected = [str(column) for column in columns if str(column) in rows.columns]
    else:
        selected = list(rows.columns)
    if not selected:
        selected = list(rows.columns)
    return _expand_topk_table_rows(rows.select(selected))


def table_row_colors(rows: pl.DataFrame, tile: Mapping[str, Any]) -> list[str | None]:
    """Resolve optional conditional-formatting colors for prepared table rows."""

    raw_rules = tile.get("conditional_formatting")
    if not isinstance(raw_rules, list):
        return [None] * rows.height
    rules = [rule for rule in raw_rules if isinstance(rule, Mapping)]
    return [
        next((_rule_color(rule) for rule in rules if _rule_matches(rule, row)), None)
        for row in rows.iter_rows(named=True)
    ]


def _table_font(theme: Mapping[str, Any]) -> dict[str, Any]:
    raw_font = theme.get("font")
    font = (
        {
            key: raw_font[key]
            for key in ("family", "size", "color")
            if key in raw_font and raw_font[key] not in (None, "")
        }
        if isinstance(raw_font, Mapping)
        else {}
    )
    font.setdefault("family", "DM Sans, Inter, Segoe UI, system-ui, sans-serif")
    font.setdefault("size", 13)
    family = str(font["family"])
    if "sans-serif" not in family.casefold() and "serif" not in family.casefold():
        font["family"] = f"{family}, DM Sans, Segoe UI, system-ui, sans-serif"
    return font


def _expand_topk_table_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Expand one Top-K list column into ranked rows with uncertainty columns."""

    topk_columns = [
        column for column in frame.columns if _is_topk_table_column(frame[column].to_list())
    ]
    if len(topk_columns) != 1:
        return frame

    topk_column = topk_columns[0]
    used_columns = set(frame.columns)
    rank_column = _unique_table_column("Rank", used_columns)
    estimate_column = _unique_table_column("Estimate", used_columns)
    lower_column = _unique_table_column("Lower bound", used_columns)
    upper_column = _unique_table_column("Upper bound", used_columns)
    base_columns = [column for column in frame.columns if column != topk_column]
    expanded: list[dict[str, Any]] = []
    for row in frame.iter_rows(named=True):
        raw_items = row.get(topk_column)
        items = list(raw_items) if isinstance(raw_items, list | tuple) else []
        if not items:
            expanded.append(
                {
                    **{column: row.get(column) for column in base_columns},
                    rank_column: None,
                    topk_column: "",
                    estimate_column: None,
                    lower_column: None,
                    upper_column: None,
                }
            )
            continue
        for rank, item in enumerate(items, start=1):
            expanded.append(
                {
                    **{column: row.get(column) for column in base_columns},
                    rank_column: rank,
                    topk_column: str(item.get("item", "")),
                    estimate_column: item.get("estimate"),
                    lower_column: item.get("lower_bound"),
                    upper_column: item.get("upper_bound"),
                }
            )
    return pl.DataFrame(expanded).select(
        *base_columns,
        rank_column,
        topk_column,
        estimate_column,
        lower_column,
        upper_column,
    )


def _is_topk_table_column(values: list[Any]) -> bool:
    has_items = False
    for value in values:
        if value is None or value == []:
            continue
        if not isinstance(value, list | tuple) or not value:
            return False
        if not all(isinstance(item, Mapping) and _is_topk_item(item) for item in value):
            return False
        has_items = True
    return has_items


def _unique_table_column(preferred: str, used: set[str]) -> str:
    candidate = preferred
    suffix = 2
    while candidate in used:
        candidate = f"{preferred} {suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _table_cell_value(value: Any) -> Any:
    if value is None:
        out: Any = ""
    elif isinstance(value, dt.datetime | dt.date):
        out = value.isoformat()
    elif isinstance(value, Mapping):
        if _is_topk_item(value):
            out = _topk_item_label(value)
        else:
            out = json.dumps(_json_safe_table_value(value), sort_keys=True)
    elif isinstance(value, list | tuple):
        if not value:
            out = ""
        elif all(isinstance(item, Mapping) and _is_topk_item(item) for item in value):
            out = ", ".join(_topk_item_label(item) for item in value)
        else:
            out = ", ".join(str(_table_cell_value(item)) for item in value)
    else:
        out = value
    return out


def _is_topk_item(value: Mapping[str, Any]) -> bool:
    return "item" in value and "estimate" in value


def _topk_item_label(value: Mapping[str, Any]) -> str:
    item = str(value.get("item", ""))
    estimate = value.get("estimate")
    if estimate is None:
        return item
    return f"{item} ({estimate})"


def _json_safe_table_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe_table_value(nested) for key, nested in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe_table_value(nested) for nested in value]
    return str(value)


def _calendar_heatmap(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    date_col = str(tile.get("date") or tile.get("x") or "Day")
    value_col = str(tile.get("value") or tile.get("y") or tile.get("metric") or "")
    if date_col not in rows.columns:
        raise ValueError(f"calendar_heatmap date column {date_col!r} not present")
    if value_col not in rows.columns:
        value_col = _first_numeric_column(rows) or ""
    if not value_col:
        raise ValueError("calendar_heatmap requires a numeric value column")
    matrix, weeks, weekdays = _calendar_matrix(rows, date_col, value_col)
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=weeks,
            y=weekdays,
            colorscale=str(tile.get("color_continuous_scale", "Viridis")),
            hovertemplate="%{y}<br>%{x}<br>%{z}<extra></extra>",
        )
    )
    fig.update_layout(title=str(tile.get("title", "")))
    return fig


def _bar_polar(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    return px.bar_polar(
        rows,
        r=_field(tile, "r"),
        theta=_field(tile, "theta"),
        color=_field(tile, "color"),
        title=str(tile.get("title", "")),
    )


def _sankey(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    source_col = _field(tile, "source")
    target_col = _field(tile, "target")
    value_col = _field(tile, "value")
    labels = list(
        dict.fromkeys(
            [
                *[str(value) for value in rows[source_col].to_list()],
                *[str(value) for value in rows[target_col].to_list()],
            ]
        )
    )
    index = {label: pos for pos, label in enumerate(labels)}
    fig = go.Figure(
        go.Sankey(
            node={"label": labels},
            link={
                "source": [index[str(value)] for value in rows[source_col].to_list()],
                "target": [index[str(value)] for value in rows[target_col].to_list()],
                "value": [_as_float(value) or 0.0 for value in rows[value_col].to_list()],
            },
        )
    )
    fig.update_layout(title=str(tile.get("title", "")))
    return fig


def _gauge(rows: pl.DataFrame, tile: Mapping[str, Any], theme: Mapping[str, Any]) -> go.Figure:
    value_col = _field(tile, "value")
    dimension_fields = _gauge_dimension_fields(rows, tile, value_col)
    if len(dimension_fields) > 2:
        raise ValueError("Gauge plot type does not support more than two grouping columns.")

    threshold_color = _CHART_INKS[_theme_base(tile, theme)]["gauge_threshold"]
    default_reference = _gauge_average_reference(rows, value_col)
    axis_max = _gauge_axis_max(rows, value_col)
    if not dimension_fields:
        row = rows.row(0, named=True) if rows.height else {}
        value = _as_float(row.get(value_col)) or 0.0
        reference = _reference_for_row(row, tile, default=default_reference)
        return go.Figure(
            _gauge_indicator(
                value=value,
                reference=reference,
                title=str(tile.get("title", "")),
                tile=tile,
                axis_max=axis_max,
                threshold_color=threshold_color,
            )
        )

    row_count, col_count, row_lookup, col_lookup = _gauge_grid(rows, dimension_fields)
    subplot_titles = _gauge_subplot_titles(
        rows,
        dimension_fields,
        row_count,
        col_count,
        row_lookup,
        col_lookup,
    )
    fig = make_subplots(
        rows=row_count,
        cols=col_count,
        subplot_titles=subplot_titles,
        specs=[[{"type": "indicator"} for _ in range(col_count)] for _ in range(row_count)],
    )
    fig.update_layout(
        autosize=True,
        height=max(320, 320 * row_count),
        margin={"b": 10, "t": 180, "l": 10, "r": 10},
        title=str(tile.get("title", "")),
    )
    _lift_gauge_subplot_titles(fig, subplot_titles)
    for index, row in enumerate(rows.iter_rows(named=True)):
        value = _as_float(row.get(value_col)) or 0.0
        reference = _reference_for_row(row, tile, dimension_fields, default=default_reference)
        if len(dimension_fields) == 2:
            grid_row = row_lookup.get(str(row.get(dimension_fields[0])), 0)
            grid_col = col_lookup.get(str(row.get(dimension_fields[1])), 0)
        else:
            grid_row, grid_col = divmod(index, col_count)
        fig.add_trace(
            _gauge_indicator(
                value=value,
                reference=reference,
                title="",
                tile=tile,
                axis_max=axis_max,
                threshold_color=threshold_color,
            ),
            row=grid_row + 1,
            col=grid_col + 1,
        )
    return fig


def _gauge_indicator(
    *,
    value: float,
    reference: float,
    title: str,
    tile: Mapping[str, Any],
    axis_max: float,
    threshold_color: str = "#c62828",
) -> go.Indicator:
    axis: dict[str, Any] = {"range": [0, axis_max]}
    value_format = _format_spec(tile)
    if value_format is not None:
        axis["tickformat"] = value_format
    gauge: dict[str, Any] = {
        "axis": axis,
        "threshold": {
            "line": {"color": threshold_color, "width": 3},
            "thickness": 0.75,
            "value": reference,
        },
    }
    if reference and value < reference:
        gauge["bar"] = {"color": "#EC5300" if value < (0.75 * reference) else "#EC9B00"}
    indicator: dict[str, Any] = {
        "mode": "gauge+number+delta",
        "value": value,
        "delta": {"reference": reference},
        "gauge": gauge,
    }
    if title:
        indicator["title"] = {"text": title}
    return go.Indicator(**indicator)


def _gauge_dimension_fields(
    rows: pl.DataFrame,
    tile: Mapping[str, Any],
    value_col: str,
) -> list[str]:
    fields: list[str] = []
    for candidate in (_facet_row(tile), _facet_col(tile)):
        _append_gauge_dimension(fields, candidate, rows, value_col)
    if fields:
        return fields
    for candidate in _config_values(tile.get("group_by")):
        _append_gauge_dimension(fields, candidate, rows, value_col)
    return fields


def _append_gauge_dimension(
    fields: list[str],
    candidate: Any,
    rows: pl.DataFrame,
    value_col: str,
) -> None:
    text = _optional_text(candidate)
    if text is not None and text != value_col and text in rows.columns and text not in fields:
        fields.append(text)


def _gauge_grid(
    rows: pl.DataFrame,
    dimension_fields: list[str],
) -> tuple[int, int, dict[str, int], dict[str, int]]:
    if len(dimension_fields) == 2:
        row_values = _ordered_unique(rows, dimension_fields[0])
        col_values = _ordered_unique(rows, dimension_fields[1])
        row_lookup = {value: index for index, value in enumerate(row_values)}
        col_lookup = {value: index for index, value in enumerate(col_values)}
        return max(1, len(row_values)), max(1, len(col_values)), row_lookup, col_lookup
    count = max(1, rows.height)
    col_count = max(1, math.ceil(math.sqrt(count)))
    row_count = max(1, math.ceil(count / col_count))
    return row_count, col_count, {}, {}


def _gauge_subplot_titles(
    rows: pl.DataFrame,
    dimension_fields: list[str],
    row_count: int,
    col_count: int,
    row_lookup: Mapping[str, int],
    col_lookup: Mapping[str, int],
) -> list[str]:
    titles = [""] * (row_count * col_count)
    for index, row in enumerate(rows.iter_rows(named=True)):
        if len(dimension_fields) == 2:
            grid_row = row_lookup.get(str(row.get(dimension_fields[0])), 0)
            grid_col = col_lookup.get(str(row.get(dimension_fields[1])), 0)
        else:
            grid_row, grid_col = divmod(index, col_count)
        titles[grid_row * col_count + grid_col] = _gauge_title(row, dimension_fields)
    return titles


def _lift_gauge_subplot_titles(fig: go.Figure, subplot_titles: list[str]) -> None:
    labels = {title for title in subplot_titles if title}
    for annotation in fig.layout.annotations or ():
        if annotation.text in labels:
            annotation.update(yshift=24, yanchor="bottom")


def _gauge_average_reference(rows: pl.DataFrame, value_col: str) -> float:
    if value_col not in rows.columns:
        return 0.0
    values: list[float] = []
    for value in rows.get_column(value_col).drop_nulls().to_list():
        numeric = _as_float(value)
        if numeric is not None:
            values.append(numeric)
    return sum(values) / len(values) if values else 0.0


def _gauge_axis_max(rows: pl.DataFrame, value_col: str) -> float:
    if value_col not in rows.columns:
        return 1.0
    values: list[float] = []
    for value in rows.get_column(value_col).drop_nulls().to_list():
        numeric = _as_float(value)
        if numeric is not None:
            values.append(numeric)
    max_value = max(values, default=0.0)
    return max_value * 1.2 if max_value > 0 else 1.0


def _ordered_unique(rows: pl.DataFrame, column: str) -> list[str]:
    values: list[str] = []
    for value in rows.get_column(column).to_list():
        text = str(value)
        if text not in values:
            values.append(text)
    return values


def _ordered_values(rows: pl.DataFrame, column: str) -> list[Any]:
    values: list[Any] = []
    for value in rows.get_column(column).to_list():
        if all(str(value) != str(existing) for existing in values):
            values.append(value)
    return values


def _category_index(values: list[Any], value: Any) -> int:
    text = str(value)
    for index, candidate in enumerate(values):
        if str(candidate) == text:
            return index
    return 0


def _template_colorway(theme: Mapping[str, Any] | None = None) -> list[str]:
    template_name = (theme or {}).get("template") or pio.templates.default
    try:
        colorway = pio.templates[template_name].layout.colorway
    except (KeyError, TypeError, ValueError):
        colorway = None
    return list(colorway or px.colors.qualitative.Plotly)


def _gauge_title(row: Mapping[str, Any], dimension_fields: list[str]) -> str:
    return " ".join(str(row[field]) for field in dimension_fields if field in row)


def _funnel(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    stages = _funnel_stages(tile.get("stages", []))
    frame = _funnel_frame(rows, tile, stages)
    return px.funnel(
        frame,
        x="Count",
        y="stage",
        color=_optional_column(frame, tile.get("color")),
        facet_row=_optional_column(frame, _facet_row(tile)),
        facet_col=_optional_column(frame, _facet_col(tile)),
        title=str(tile.get("title", "")),
    )


def _funnel_stages(raw_stages: Any) -> list[str]:
    if not isinstance(raw_stages, Iterable) or isinstance(raw_stages, str):
        return []
    stages: list[str] = []
    for stage in raw_stages:
        if isinstance(stage, Mapping):
            name = _optional_text(stage.get("name"))
        else:
            name = _optional_text(stage)
        if name is not None:
            stages.append(name)
    return stages


def _funnel_frame(
    rows: pl.DataFrame,
    tile: Mapping[str, Any],
    stages: list[str],
) -> pl.DataFrame:
    if {"stage", "Count"} <= set(rows.columns):
        return rows
    stage_col = _optional_column(rows, tile.get("x"))
    count_col = _funnel_count_column(rows, tile, stage_col)
    if stage_col is not None and count_col is not None:
        return _categorical_funnel_rows(rows, stages, stage_col, count_col)
    return _funnel_stage_rows(rows, stages)


def _funnel_count_column(
    rows: pl.DataFrame,
    tile: Mapping[str, Any],
    stage_col: str | None,
) -> str | None:
    candidates = [
        tile.get("values"),
        tile.get("value"),
        f"{tile.get('property')}_Count" if tile.get("property") else None,
        f"{stage_col}_Count" if stage_col else None,
        tile.get("metric") if str(tile.get("metric", "")).endswith("_Count") else None,
        "Count",
    ]
    for candidate in candidates:
        column = _optional_column(rows, candidate)
        if column is not None:
            return column
    return None


def _categorical_funnel_rows(
    rows: pl.DataFrame,
    stages: list[str],
    stage_col: str,
    count_col: str,
) -> pl.DataFrame:
    out: list[dict[str, Any]] = []
    stage_order = {stage: index for index, stage in enumerate(stages)}
    for row in rows.iter_rows(named=True):
        stage = str(row[stage_col])
        if stages and stage not in stage_order:
            continue
        values = {column: row[column] for column in rows.columns if column != count_col}
        out.append({**values, "stage": stage, "Count": row[count_col]})
    if not out:
        return pl.DataFrame({"stage": [], "Count": []})
    frame = pl.DataFrame(out)
    if stages:
        frame = frame.with_columns(
            pl.col("stage").replace_strict(stage_order, default=len(stage_order)).alias("_order")
        ).sort("_order")
        return frame.drop("_order")
    return frame


def _boxplot(
    rows: pl.DataFrame, tile: Mapping[str, Any], theme: Mapping[str, Any] | None = None
) -> go.Figure:
    prop = tile.get("property") or _infer_quantile_property(rows)
    if prop is not None and f"{prop}_Median" in rows.columns:
        return _quantile_box(rows, str(prop), tile, theme)
    fig = px.box(
        rows,
        x=tile.get("x"),
        y=tile.get("y", prop),
        color=tile.get("color"),
        facet_row=_facet_row(tile),
        facet_col=_facet_col(tile),
        title=str(tile.get("title", "")),
    )
    fig.update_layout(boxmode="group")
    return fig


def _infer_quantile_property(rows: pl.DataFrame) -> str | None:
    """Return the sole quantile-suite property present in the frame, if any.

    Boxplot tiles authored without ``property`` would otherwise box the scalar
    metric value — one number per group — instead of the digest's quantile
    suite that the query layer already delivered alongside it.
    """

    candidates = {
        column.removesuffix("_Median") for column in rows.columns if column.endswith("_Median")
    }
    candidates = {
        candidate
        for candidate in candidates
        if f"{candidate}_p25" in rows.columns and f"{candidate}_p75" in rows.columns
    }
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def _histogram(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    column = str(tile.get("property", tile.get("x", tile.get("y", rows.columns[-1]))))
    return px.histogram(
        rows,
        x=column,
        color=tile.get("color"),
        facet_row=_facet_row(tile),
        facet_col=_facet_col(tile),
        title=tile.get("title"),
    )


def _calibration_curve(
    rows: pl.DataFrame, tile: Mapping[str, Any], theme: Mapping[str, Any]
) -> go.Figure:
    frame = _expand_calibration(rows)
    color = _optional_column(frame, tile.get("color"))
    facet_row = _facet_row(tile)
    facet_col = _facet_col(tile)
    sort_columns = [
        column
        for column in (facet_row, facet_col, color, "predicted")
        if column is not None and column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort(sort_columns)
    fig = px.line(
        frame,
        x="predicted",
        y="observed",
        color=color,
        facet_row=facet_row,
        facet_col=facet_col,
        title=str(tile.get("title", "")),
        height=_facet_height(frame, facet_row, per_row=400),
    )
    fig.update_traces(mode="lines+markers")
    fig.add_shape(
        type="line",
        line={"dash": "dash", "color": _reference_line_color(tile, theme)},
        row="all",
        col="all",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
    )
    fig.update_xaxes(title="", range=[0, 1], tickformat=_format_spec(tile))
    fig.update_yaxes(title="", range=[0, 1])
    _add_global_axis_labels(
        fig,
        x_label=str(tile.get("x_axis_title") or "Predicted propensity"),
        y_label=str(tile.get("y_axis_title") or "Observed rate"),
    )
    return fig


def _roc_curve(rows: pl.DataFrame, tile: Mapping[str, Any], theme: Mapping[str, Any]) -> go.Figure:
    frame = _expand_curve_pair(rows, x="fpr", y="tpr")
    fig = _curve_line(
        frame,
        tile,
        x="fpr",
        y="tpr",
        x_label="False positive rate",
        y_label="True positive rate",
    )
    fig.add_shape(
        type="line",
        line={"dash": "dash", "color": _reference_line_color(tile, theme)},
        row="all",
        col="all",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
    )
    fig.update_xaxes(range=[0, 1], tickformat=".2%")
    fig.update_yaxes(range=[0, 1], tickformat=".2%")
    return fig


def _precision_recall_curve(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    frame = _expand_curve_pair(rows, x="recall", y="precision")
    fig = _curve_line(
        frame,
        tile,
        x="recall",
        y="precision",
        x_label="Recall",
        y_label="Precision",
    )
    fig.update_xaxes(range=[0, 1], tickformat=".2%")
    fig.update_yaxes(range=[0, 1], tickformat=".2%")
    return fig


def _gain_curve(rows: pl.DataFrame, tile: Mapping[str, Any], theme: Mapping[str, Any]) -> go.Figure:
    frame = _expand_gain_lift(rows, include_lift=False)
    fig = _curve_line(
        frame,
        tile,
        x="sample_fraction",
        y="gain",
        x_label="Fraction of population",
        y_label="Gain",
    )
    fig.add_shape(
        type="line",
        line={"dash": "dash", "color": _reference_line_color(tile, theme)},
        row="all",
        col="all",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
    )
    fig.update_xaxes(range=[0, 1], tickformat=".2%")
    fig.update_yaxes(range=[0, 1], tickformat=".2%")
    return fig


def _lift_curve(rows: pl.DataFrame, tile: Mapping[str, Any], theme: Mapping[str, Any]) -> go.Figure:
    frame = _expand_gain_lift(rows, include_lift=True)
    fig = _curve_line(
        frame,
        tile,
        x="sample_fraction",
        y="lift",
        x_label="Fraction of population",
        y_label="Lift",
    )
    fig.add_shape(
        type="line",
        line={"dash": "dash", "color": _reference_line_color(tile, theme)},
        row="all",
        col="all",
        x0=0,
        y0=1,
        x1=1,
        y1=1,
    )
    max_lift = _max_numeric(frame, "lift")
    fig.update_xaxes(range=[0, 1], tickformat=".2%")
    fig.update_yaxes(range=[0, max(1.0, max_lift)])
    return fig


def _rfm_density(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    return px.density_heatmap(
        rows,
        x=str(tile.get("x", "recency")),
        y=str(tile.get("y", "frequency")),
        z=tile.get("z"),
        marginal_x="histogram",
        marginal_y="histogram",
        title=str(tile.get("title", "")),
    )


def _exposure(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    value = str(tile.get("value", "lifetime_value"))
    count = "customers_count" if "customers_count" in rows.columns else value
    sorted_df = rows.sort(value, descending=True).with_columns(
        (pl.col(count).cum_sum() / pl.col(count).sum()).alias("cum_customers"),
        (pl.col(value).cum_sum() / pl.col(value).sum()).alias("cum_value"),
    )
    return px.line(
        sorted_df,
        x="cum_customers",
        y="cum_value",
        title=str(tile.get("title", "")),
    )


def _corr(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    return px.scatter(
        rows,
        x=str(tile.get("x", "frequency")),
        y=str(tile.get("y", "monetary_value")),
        color=tile.get("color", "rfm_segment" if "rfm_segment" in rows.columns else None),
        title=str(tile.get("title", "")),
    )


def _model(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    horizon = float(tile.get("horizon", 30))
    tenure = pl.when(pl.col("tenure") <= 0).then(1.0).otherwise(pl.col("tenure"))
    modeled = rows.with_columns(
        ((pl.col("frequency").cast(pl.Float64) / tenure) * horizon).alias("predicted_purchases")
    )
    return px.scatter(
        modeled,
        x="frequency",
        y="predicted_purchases",
        color="rfm_segment" if "rfm_segment" in modeled.columns else None,
        title=str(tile.get("title", "")),
    )


def _descriptive(
    rows: pl.DataFrame, tile: Mapping[str, Any], theme: Mapping[str, Any]
) -> go.Figure:
    kind = str(tile["chart"])
    prop = str(tile.get("property", "value"))
    rows, metric_col = _with_descriptive_metric_column(rows, prop, str(tile.get("score", "Mean")))
    if kind == "descriptive_line":
        return _line(rows, {**tile, "y": metric_col}, max_points=MAX_POINTS)
    if kind == "descriptive_histogram":
        return _descriptive_histogram(
            rows,
            {
                **tile,
                "property": prop,
                "fallback_property": metric_col if metric_col in rows.columns else prop,
            },
            theme,
        )
    if kind == "descriptive_heatmap":
        return _heatmap(rows, {**tile, "color": metric_col}, theme)
    if kind == "descriptive_funnel":
        return _funnel(rows, tile)
    raise ValueError(f"unsupported descriptive chart kind {kind!r}")


def _with_descriptive_metric_column(
    rows: pl.DataFrame,
    prop: str,
    raw_score: str,
) -> tuple[pl.DataFrame, str]:
    score = _normalize_descriptive_score(raw_score)
    metric_col = f"{prop}_{score}"
    if metric_col in rows.columns:
        return rows, metric_col
    quantile = _DESCRIPTIVE_QUANTILES.get(raw_score) or _DESCRIPTIVE_QUANTILES.get(score)
    if quantile is None:
        return rows, metric_col
    for digest_col, quantile_fn in (
        (f"{prop}_tdigest", tdigest.quantile),
        (f"{prop}_kll", kll.quantile),
    ):
        if digest_col not in rows.columns:
            continue
        return (
            rows.with_columns(
                pl.col(digest_col)
                .map_elements(
                    lambda payload, fn=quantile_fn, q=quantile: float(fn(payload, q)),
                    return_dtype=pl.Float64,
                )
                .alias(metric_col)
            ),
            metric_col,
        )
    return rows, metric_col


def _normalize_descriptive_score(score: str) -> str:
    return "Median" if score == "p50" else score


def _experiment_z_score(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    x_col = _field(tile, "x")
    y_col = _field(tile, "y")
    facet_row = _optional_column(rows, _facet_row(tile))
    facet_col = _optional_column(rows, _facet_col(tile))
    frame = rows.drop_nulls([column for column in (x_col, y_col, facet_row, facet_col) if column])
    sort_columns = [column for column in (facet_col, facet_row, x_col) if column in frame.columns]
    if sort_columns and not frame.is_empty():
        frame = frame.sort(sort_columns, descending=True)
    fig = px.bar(
        frame,
        x=x_col,
        y=y_col,
        color=y_col,
        facet_row=facet_row,
        facet_col=facet_col,
        orientation="h",
        title=str(tile.get("title", "")),
        height=_experiment_height(frame, y_col, facet_row, base=_as_int(tile.get("height")) or 640),
    )
    fig.update_layout(
        showlegend=False,
        updatemenus=[
            {
                "buttons": [
                    {"args": [{"type": "bar"}], "label": "Bar", "method": "restyle"},
                    {
                        "args": [{"type": "scatter", "mode": "lines+markers"}],
                        "label": "Line",
                        "method": "restyle",
                    },
                ],
                "direction": "down",
                "pad": {"r": 10, "t": 20},
                "showactive": True,
                "x": 0,
                "xanchor": "left",
                "y": 1.1,
                "yanchor": "top",
            }
        ],
    )
    fig.add_vrect(x0=-1.96, x1=1.96, line_width=0, fillcolor="red", opacity=0.1)
    return fig


def _experiment_odds_ratio(
    rows: pl.DataFrame, tile: Mapping[str, Any], theme: Mapping[str, Any]
) -> go.Figure:
    configured_x = _field(tile, "x")
    y_col = _field(tile, "y")
    prefix = "g" if configured_x.startswith("g") else "chi2"
    x_col = f"{prefix}_odds_ratio_stat"
    ci_low = f"{prefix}_odds_ratio_ci_low"
    ci_high = f"{prefix}_odds_ratio_ci_high"
    facet_row = _optional_column(rows, _facet_row(tile))
    facet_col = _optional_column(rows, _facet_col(tile))
    required = [x_col, y_col, ci_low, ci_high]
    if not set(required) <= set(rows.columns):
        fig = px.scatter(
            rows,
            x=configured_x,
            y=y_col,
            facet_row=facet_row,
            facet_col=facet_col,
            title=str(tile.get("title", "")),
        )
        fig.add_vline(
            x=1, line_width=2, line_dash="dash", line_color=_reference_line_color(tile, theme)
        )
        fig.update_layout(showlegend=False)
        return fig

    frame = rows.drop_nulls([column for column in (*required, facet_row, facet_col) if column])
    if not frame.is_empty():
        frame = frame.with_columns(
            (pl.col(ci_high) - pl.col(x_col)).alias("_x_plus"),
            (pl.col(x_col) - pl.col(ci_low)).alias("_x_minus"),
            (
                pl.when((pl.col(ci_high) < 1) & (pl.col(ci_low) < 1))
                .then(pl.lit("Control"))
                .when((pl.col(ci_high) > 1) & (pl.col(ci_low) > 1))
                .then(pl.lit("Test"))
                .otherwise(pl.lit("N/A"))
            ).alias("_experiment_color"),
        )
    sort_columns = [column for column in (facet_col, facet_row, x_col) if column in frame.columns]
    if sort_columns and not frame.is_empty():
        frame = frame.sort(sort_columns)
    fig = px.scatter(
        frame,
        x=x_col,
        y=y_col,
        color="_experiment_color",
        color_discrete_map={"Control": "#e74c3c", "N/A": "#f1c40f", "Test": "#2ecc71"},
        category_orders={"_experiment_color": ["Control", "N/A", "Test"]},
        facet_row=facet_row,
        facet_col=facet_col,
        error_x="_x_plus",
        error_x_minus="_x_minus",
        title=str(tile.get("title", "")),
        height=_experiment_height(frame, y_col, facet_row, base=600, per_category=20),
    )
    fig.add_vline(
        x=1, line_width=2, line_dash="dash", line_color=_reference_line_color(tile, theme)
    )
    fig.update_layout(showlegend=False)
    return fig


def _experiment_height(
    rows: pl.DataFrame,
    y_col: str,
    facet_row: str | None,
    *,
    base: int,
    per_category: int = 10,
) -> int:
    if rows.is_empty() or y_col not in rows.columns:
        return base
    row_count = rows[facet_row].n_unique() if facet_row in rows.columns else 1
    return max(base, per_category * rows[y_col].n_unique() * row_count)


def _clv_treemap(rows: pl.DataFrame, tile: Mapping[str, Any]) -> go.Figure:
    return px.treemap(
        rows,
        path=[px.Constant("All"), "rfm_segment"],
        values="lifetime_value",
        title=str(tile.get("title", "")),
    )


def _downsample_by_color(
    rows: pl.DataFrame,
    *,
    x: str,
    y: str,
    color: Any,
    max_points: int,
) -> pl.DataFrame:
    if not isinstance(color, str) or color not in rows.columns:
        return lttb.downsample(rows, x=x, y=y, threshold=max_points)
    groups = rows.partition_by(color, as_dict=True)
    budget = max(3, max_points // max(len(groups), 1))
    return pl.concat(
        [lttb.downsample(group, x=x, y=y, threshold=budget) for group in groups.values()]
    )


def _funnel_stage_rows(rows: pl.DataFrame, stages: list[str]) -> pl.DataFrame:
    if not stages:
        stages = [
            column.removesuffix("_Count") for column in rows.columns if column.endswith("_Count")
        ]
    out: list[dict[str, Any]] = []
    group_columns = [
        column
        for column in rows.columns
        if column not in {f"{stage}_Count" for stage in stages}
        and not column.endswith("_Customers_cpc")
        and not column.endswith("_Customers_hll")
    ]
    for row in rows.iter_rows(named=True):
        groups = {column: row[column] for column in group_columns}
        for stage in stages:
            count_column = f"{stage}_Count"
            if count_column in row:
                out.append({**groups, "stage": stage, "Count": row[count_column]})
    return pl.DataFrame(out) if out else pl.DataFrame({"stage": [], "Count": []})


def _quantile_box(
    rows: pl.DataFrame,
    prop: str,
    tile: Mapping[str, Any],
    theme: Mapping[str, Any] | None = None,
) -> go.Figure:
    x_col = _optional_column(rows, tile.get("x"))
    color_col = _optional_column(rows, tile.get("color"))
    facet_row = _optional_column(rows, _facet_row(tile))
    facet_col = _optional_column(rows, _facet_col(tile))
    row_values = _ordered_values(rows, facet_row) if facet_row else [None]
    col_values = _ordered_values(rows, facet_col) if facet_col else [None]
    row_count = max(1, len(row_values))
    col_count = max(1, len(col_values))
    fig = make_subplots(
        rows=row_count,
        cols=col_count,
        shared_xaxes=True,
        shared_yaxes=False,
        vertical_spacing=_subplot_spacing(row_count, 0.05),
        horizontal_spacing=_subplot_spacing(col_count, 0.04),
    )
    color_values = _ordered_values(rows, color_col) if color_col else [prop]
    colorway = _template_colorway(theme)
    legend_shown: set[str] = set()
    for row in rows.iter_rows(named=True):
        row_value = row.get(facet_row) if facet_row else None
        col_value = row.get(facet_col) if facet_col else None
        subplot_row = _category_index(row_values, row_value) + 1
        subplot_col = _category_index(col_values, col_value) + 1
        color_value = row.get(color_col) if color_col else prop
        legend_key = str(color_value)
        color_index = _category_index(color_values, color_value)
        box_kwargs = _quantile_box_kwargs(
            row,
            prop=prop,
            x_col=x_col,
            color=colorway[color_index % len(colorway)],
            legend_key=legend_key,
            showlegend=legend_key not in legend_shown,
        )
        legend_shown.add(legend_key)
        fig.add_trace(go.Box(**box_kwargs), row=subplot_row, col=subplot_col)
    _add_quantile_box_facet_labels(
        fig,
        row_values=row_values,
        col_values=col_values,
        facet_row=facet_row,
        facet_col=facet_col,
    )
    fig.update_xaxes(tickfont={"size": 10})
    fig.update_layout(title=str(tile.get("title", "")), boxmode="group")
    return fig


def _descriptive_histogram(
    rows: pl.DataFrame, tile: Mapping[str, Any], theme: Mapping[str, Any] | None = None
) -> go.Figure:
    prop = str(tile.get("property", "value"))
    digest_col = f"{prop}_tdigest"
    if digest_col not in rows.columns:
        return _histogram(rows, {**tile, "property": tile.get("fallback_property", prop)})

    facet_row = _optional_column(rows, _facet_row(tile))
    facet_col = _optional_column(rows, _facet_col(tile))
    color_col = _optional_column(rows, tile.get("color"))
    row_values = _ordered_values(rows, facet_row) if facet_row else [None]
    col_values = _ordered_values(rows, facet_col) if facet_col else [None]
    row_count = max(1, len(row_values))
    col_count = max(1, len(col_values))
    fig = make_subplots(
        rows=row_count,
        cols=col_count,
        shared_xaxes=True,
        shared_yaxes=False,
        vertical_spacing=_subplot_spacing(row_count, 0.05),
        horizontal_spacing=_subplot_spacing(col_count, 0.04),
    )
    color_values = _ordered_values(rows, color_col) if color_col else [prop]
    colorway = _template_colorway(theme)
    legend_shown: set[str] = set()
    bins = _as_int(tile.get("bins")) or 100

    for row in rows.iter_rows(named=True):
        histogram = _tdigest_histogram(row.get(digest_col), bins=bins)
        if histogram is None:
            continue
        bin_edges, bin_counts = histogram
        bin_centers = [(left + right) / 2 for left, right in pairwise(bin_edges)]
        bin_widths = [right - left for left, right in pairwise(bin_edges)]
        row_value = row.get(facet_row) if facet_row else None
        col_value = row.get(facet_col) if facet_col else None
        subplot_row = _category_index(row_values, row_value) + 1
        subplot_col = _category_index(col_values, col_value) + 1
        color_value = row.get(color_col) if color_col else prop
        legend_key = str(color_value)
        color_index = _category_index(color_values, color_value)
        showlegend = bool(color_col) and legend_key not in legend_shown
        legend_shown.add(legend_key)
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=bin_counts,
                width=bin_widths,
                name=legend_key if color_col else " ",
                legendgroup=legend_key,
                marker_color=colorway[color_index % len(colorway)],
                opacity=0.68 if color_col else 0.9,
                showlegend=showlegend,
                hovertemplate="%{x}<br>mass=%{y:.2%}<extra>%{fullData.name}</extra>",
            ),
            row=subplot_row,
            col=subplot_col,
        )

    _add_quantile_box_facet_labels(
        fig,
        row_values=row_values,
        col_values=col_values,
        facet_row=facet_row,
        facet_col=facet_col,
    )
    fig.update_xaxes(tickfont={"size": 8})
    fig.update_yaxes(title_text="Probability mass")
    fig.update_layout(
        title=str(tile.get("title", "")),
        height=max(640, 350 * row_count) if facet_row else 640,
        barmode="overlay",
    )
    return fig


def _tdigest_histogram(
    payload: bytes | bytearray | memoryview | None,
    *,
    bins: int,
    value_range: tuple[float, float] | None = None,
) -> tuple[list[float], list[float]] | None:
    if not payload:
        return None
    sketch = tdigest.deserialize(payload)
    if sketch.get_total_weight() == 0:
        return None
    if value_range is None:
        value_range = (float(sketch.get_quantile(0)), float(sketch.get_quantile(1)))
    lower, upper = value_range
    if not math.isfinite(lower) or not math.isfinite(upper):
        return None
    if lower == upper:
        padding = abs(lower) * 0.05 or 0.5
        lower -= padding
        upper += padding
    bin_count = max(1, bins)
    width = (upper - lower) / bin_count
    bin_edges = [lower + (index * width) for index in range(bin_count + 1)]
    bin_counts = [
        float(sketch.get_cdf([right])[0] - sketch.get_cdf([left])[0])
        for left, right in pairwise(bin_edges)
    ]
    return bin_edges, bin_counts


def _subplot_spacing(count: int, default: float) -> float:
    if count <= 1:
        return 0.0
    return min(default, 0.8 / (count - 1))


def _quantile_box_kwargs(
    row: Mapping[str, Any],
    *,
    prop: str,
    x_col: str | None,
    color: str,
    legend_key: str,
    showlegend: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "q1": [row[f"{prop}_p25"]],
        "median": [row[f"{prop}_Median"]],
        "q3": [row[f"{prop}_p75"]],
        "name": legend_key,
        "legendgroup": legend_key,
        "offsetgroup": legend_key,
        "marker_color": color,
        "boxpoints": False,
        "showlegend": showlegend,
    }
    if x_col is not None:
        kwargs["x"] = [row[x_col]]
    lowerfence, upperfence = _quantile_box_fences(row, prop)
    if lowerfence is not None:
        kwargs["lowerfence"] = [lowerfence]
    if upperfence is not None:
        kwargs["upperfence"] = [upperfence]
    mean = row.get(f"{prop}_Mean")
    if mean is not None:
        kwargs["mean"] = [mean]
    notchspan = _quantile_box_notchspan(row, prop)
    if notchspan is not None:
        kwargs["notchspan"] = [notchspan]
    return kwargs


def _quantile_box_fences(row: Mapping[str, Any], prop: str) -> tuple[float | None, float | None]:
    q1 = _as_float(row.get(f"{prop}_p25"))
    q3 = _as_float(row.get(f"{prop}_p75"))
    minimum = _as_float(row.get(f"{prop}_Min"))
    maximum = _as_float(row.get(f"{prop}_Max"))
    if q1 is None or q3 is None:
        return minimum, maximum
    iqr = q3 - q1
    lowerfence = q1 - (1.5 * iqr)
    upperfence = q3 + (1.5 * iqr)
    if minimum is not None:
        lowerfence = max(minimum, lowerfence)
    if maximum is not None:
        upperfence = min(maximum, upperfence)
    return lowerfence, upperfence


def _quantile_box_notchspan(row: Mapping[str, Any], prop: str) -> float | None:
    q1 = _as_float(row.get(f"{prop}_p25"))
    q3 = _as_float(row.get(f"{prop}_p75"))
    count = _as_float(row.get(f"{prop}_Count"))
    if q1 is None or q3 is None or count is None or count <= 0:
        return None
    return 1.57 * ((q3 - q1) / math.sqrt(count))


def _add_quantile_box_facet_labels(
    fig: go.Figure,
    *,
    row_values: list[Any],
    col_values: list[Any],
    facet_row: str | None,
    facet_col: str | None,
) -> None:
    if facet_col is not None:
        col_count = len(col_values)
        for index, value in enumerate(col_values):
            fig.add_annotation(
                text=str(value),
                xref="paper",
                yref="paper",
                x=(index / col_count) + (0.5 / col_count),
                y=1.0,
                showarrow=False,
                font={"size": 14},
                xanchor="center",
                yanchor="bottom",
            )
    if facet_row is not None:
        row_count = len(row_values)
        delta = 1 / (2 * row_count)
        for index, value in enumerate(row_values):
            fig.add_annotation(
                text=str(value),
                xref="paper",
                yref="paper",
                x=1.02,
                y=(1 - ((index + 1) / row_count)) + delta,
                showarrow=False,
                font={"size": 14},
                xanchor="right",
                yanchor="middle",
                textangle=90,
            )


def _expand_calibration(rows: pl.DataFrame) -> pl.DataFrame:
    if {"predicted", "observed"} <= set(rows.columns):
        return rows
    struct_col = next(
        (column for column in rows.columns if rows.schema[column].base_type() == pl.Struct), None
    )
    if struct_col is None or rows.is_empty():
        return pl.DataFrame({"predicted": [0.0, 1.0], "observed": [0.0, 1.0]})
    expanded: list[dict[str, Any]] = []
    for row in rows.iter_rows(named=True):
        value = row.get(struct_col)
        if not isinstance(value, Mapping):
            continue
        group_values = {key: val for key, val in row.items() if key != struct_col}
        bins = list(value.get("bin", []))
        predicted = list(value.get("predicted", []))
        observed = list(value.get("observed", []))
        for index, (predicted_value, observed_value) in enumerate(
            zip(predicted, observed, strict=False)
        ):
            item = {
                **group_values,
                "predicted": predicted_value,
                "observed": observed_value,
            }
            if index < len(bins):
                item["bin"] = bins[index]
            expanded.append(item)
    if not expanded:
        return pl.DataFrame({"predicted": [0.0, 1.0], "observed": [0.0, 1.0]})
    return pl.DataFrame(expanded)


def _kpi_reference(rows: pl.DataFrame, tile: Mapping[str, Any]) -> float | None:
    raw = tile.get("reference", tile.get("delta_reference"))
    if isinstance(raw, str) and raw in rows.columns and rows.height:
        return _as_float(rows[raw][0])
    ref = _as_float(raw)
    if ref is not None:
        return ref
    goal = tile.get("goal_line")
    if isinstance(goal, Mapping):
        return _as_float(goal.get("value"))
    return _as_float(goal)


def _first_numeric_column(rows: pl.DataFrame) -> str | None:
    for column in rows.columns:
        if rows.schema[column].is_numeric():
            return column
    return None


def _table_fill_colors(
    rows: pl.DataFrame, tile: Mapping[str, Any], default_fill: str
) -> list[list[str]] | str:
    default = str(tile.get("cell_fill_color", default_fill))
    row_colors = table_row_colors(rows, tile)
    if not any(row_colors):
        return default
    return [[color or default for color in row_colors] for _column in rows.columns]


def _calendar_matrix(
    rows: pl.DataFrame,
    date_col: str,
    value_col: str,
) -> tuple[list[list[float | None]], list[str], list[str]]:
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    week_keys: list[dt.date] = []
    values: dict[tuple[str, dt.date], float] = {}
    for row in rows.iter_rows(named=True):
        date_value = _coerce_date(row.get(date_col))
        if date_value is None:
            continue
        week_start = date_value - dt.timedelta(days=date_value.weekday())
        if week_start not in week_keys:
            week_keys.append(week_start)
        weekday = weekdays[date_value.weekday()]
        values[(weekday, week_start)] = _as_float(row.get(value_col)) or 0.0
    week_keys = sorted(week_keys)
    if not week_keys:
        return [[None], [None], [None], [None], [None], [None], [None]], [""], weekdays
    matrix = [
        [values.get((weekday, week_start)) for week_start in week_keys] for weekday in weekdays
    ]
    return matrix, [week.isoformat() for week in week_keys], weekdays


def _coerce_date(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if value is None:
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _curve_line(
    frame: pl.DataFrame,
    tile: Mapping[str, Any],
    *,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
) -> go.Figure:
    color = _optional_column(frame, tile.get("color"))
    facet_row = _facet_row(tile)
    facet_col = _facet_col(tile)
    sort_columns = [
        column
        for column in (facet_row, facet_col, color, x)
        if column is not None and column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort(sort_columns)
    fig = px.line(
        frame,
        x=x,
        y=y,
        color=color,
        facet_row=facet_row,
        facet_col=facet_col,
        title=str(tile.get("title", "")),
        height=_facet_height(frame, facet_row, per_row=400),
    )
    fig.update_traces(mode="lines")
    fig.update_xaxes(title="")
    fig.update_yaxes(title="")
    _add_global_axis_labels(
        fig,
        x_label=str(tile.get("x_axis_title") or x_label),
        y_label=str(tile.get("y_axis_title") or y_label),
    )
    return fig


def _expand_curve_pair(rows: pl.DataFrame, *, x: str, y: str) -> pl.DataFrame:
    if rows.is_empty() or not {x, y} <= set(rows.columns):
        return pl.DataFrame({x: [0.0], y: [0.0]})
    expanded: list[dict[str, Any]] = []
    for row in rows.iter_rows(named=True):
        group_values = _curve_group_values(row)
        x_values = list(row.get(x) or [])
        y_values = list(row.get(y) or [])
        for x_value, y_value in zip(x_values, y_values, strict=False):
            expanded.append({**group_values, x: x_value, y: y_value})
    if not expanded:
        return pl.DataFrame({x: [0.0], y: [0.0]})
    return pl.DataFrame(expanded)


def _expand_gain_lift(rows: pl.DataFrame, *, include_lift: bool) -> pl.DataFrame:
    if rows.is_empty() or not {"fpr", "tpr", "pos_fraction"} <= set(rows.columns):
        schema = {"sample_fraction": [0.0], "gain": [0.0]}
        if include_lift:
            schema["lift"] = [1.0]
        return pl.DataFrame(schema)
    expanded: list[dict[str, Any]] = []
    for row in rows.iter_rows(named=True):
        group_values = _curve_group_values(row)
        pos_fraction = _as_float(row.get("pos_fraction")) or 0.0
        fpr_values = list(row.get("fpr") or [])
        tpr_values = list(row.get("tpr") or [])
        for fpr_value, tpr_value in zip(fpr_values, tpr_values, strict=False):
            fpr_float = _as_float(fpr_value) or 0.0
            tpr_float = _as_float(tpr_value) or 0.0
            sample_fraction = pos_fraction * tpr_float + (1.0 - pos_fraction) * fpr_float
            item = {
                **group_values,
                "sample_fraction": sample_fraction,
                "gain": tpr_float,
            }
            if include_lift:
                item["lift"] = tpr_float / sample_fraction if sample_fraction > 1e-6 else 0.0
            expanded.append(item)
    if not expanded:
        schema = {"sample_fraction": [0.0], "gain": [0.0]}
        if include_lift:
            schema["lift"] = [1.0]
        return pl.DataFrame(schema)
    return pl.DataFrame(expanded)


def _curve_group_values(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in _CURVE_LIST_COLUMNS and not isinstance(value, list)
    }


def _max_numeric(rows: pl.DataFrame, column: str) -> float:
    if column not in rows.columns or rows.is_empty():
        return 1.0
    values = [_as_float(value) for value in rows[column].drop_nulls().to_list()]
    return max((value for value in values if value is not None), default=1.0)


def _apply_accessibility(fig: go.Figure, tile: Mapping[str, Any]) -> None:
    if tile.get("description"):
        fig.update_layout(meta={"description": str(tile["description"])})
    if tile.get("show_chart_title"):
        fig.update_layout(title=str(tile.get("title", "")))
    else:
        fig.update_layout(title=None)


def _apply_label_overrides(fig: go.Figure, tile: Mapping[str, Any]) -> None:
    if tile.get("chart") == "calibration_curve" or tile.get("chart") in _CURVE_CHARTS:
        x_title = None
        y_title = None
    else:
        x_title = _axis_title(tile, "x", "x_axis_title")
        y_title = _axis_title(tile, "y", "y_axis_title")
    legend_title = _axis_title(tile, "color", "legend_title")
    standoff = _as_int(tile.get("axis_title_standoff"))
    axis_layout: dict[str, Any] = {"automargin": True}
    if standoff is not None:
        axis_layout["title_standoff"] = standoff
    if x_title and not _has_facets(tile):
        axis_layout["title_text"] = x_title
    fig.update_xaxes(**axis_layout)
    _apply_y_axis_labels(fig, tile, y_title=y_title, standoff=standoff)
    if _has_facets(tile):
        _apply_outer_facet_axis_titles(fig, x_title=x_title, y_title=y_title)
    if legend_title:
        fig.update_layout(legend_title_text=legend_title)


def _apply_y_axis_labels(
    fig: go.Figure,
    tile: Mapping[str, Any],
    *,
    y_title: str | None,
    standoff: int | None,
) -> None:
    """Apply primary and secondary labels without crossing dual-axis semantics."""

    axis_layout = {"automargin": True}
    if standoff is not None:
        axis_layout["title_standoff"] = standoff
    if y_title and not _has_facets(tile):
        axis_layout["title_text"] = y_title
    chart = str(tile.get("chart", ""))
    if chart in {"combo", "pareto"}:
        fig.update_yaxes(**axis_layout, secondary_y=False)
        secondary_axis_layout: dict[str, Any] = {"automargin": True}
        if standoff is not None:
            secondary_axis_layout["title_standoff"] = standoff
        if chart == "combo":
            secondary_field = "y2" if tile.get("y2") else "line_y"
            y2_title = _axis_title(tile, secondary_field, "y2_axis_title")
            if y2_title:
                secondary_axis_layout["title_text"] = y2_title
        fig.update_yaxes(**secondary_axis_layout, secondary_y=True)
    else:
        fig.update_yaxes(**axis_layout)


def _apply_theme(fig: go.Figure, theme: Mapping[str, Any], tile: Mapping[str, Any]) -> None:
    raw_tile_theme = tile.get("theme")
    tile_theme: Mapping[str, Any] = raw_tile_theme if isinstance(raw_tile_theme, Mapping) else {}
    merged = {**theme, **dict(tile_theme)}
    layout: dict[str, Any] = {}
    for key in (
        "template",
        "colorway",
        "font",
        "hoverlabel",
        "legend",
        "paper_bgcolor",
        "plot_bgcolor",
    ):
        if key in merged:
            layout[key] = merged[key]
    if "margins" in merged:
        layout["margin"] = merged["margins"]
    if layout:
        fig.update_layout(**layout)


def _apply_display_labels(fig: go.Figure, tile: Mapping[str, Any]) -> None:
    """Replace technical field identifiers in hover and table presentation."""

    raw_labels = tile.get("labels")
    if not isinstance(raw_labels, Mapping):
        return
    labels = {
        str(field): str(label) for field, label in raw_labels.items() if str(field) and str(label)
    }
    if not labels:
        return
    for trace in fig.data:
        if isinstance(trace, go.Table):
            values = list(trace.header.values) if trace.header.values is not None else []
            trace.header.values = [labels.get(str(value), value) for value in values]
        hovertemplate = getattr(trace, "hovertemplate", None)
        if isinstance(hovertemplate, str):
            rendered = hovertemplate
            for field, label in labels.items():
                rendered = rendered.replace(f"{field}=", f"{label}=")
                rendered = rendered.replace(f"{field}:", f"{label}:")
            trace.update(hovertemplate=rendered)
        name = getattr(trace, "name", None)
        if isinstance(name, str) and name in labels:
            trace.update(name=labels[name])


def _apply_semantic_category_colors(
    fig: go.Figure,
    tile: Mapping[str, Any],
    theme: Mapping[str, Any],
) -> None:
    """Keep category colors stable across charts, filters, and result ordering."""

    if tile.get("conditional_formatting") or str(tile.get("chart", "")).startswith("experiment_"):
        return
    color_field = tile.get("color")
    raw_maps = theme.get("category_colors")
    if not isinstance(color_field, str) or not isinstance(raw_maps, Mapping):
        return
    raw_category_map = raw_maps.get(color_field)
    if not isinstance(raw_category_map, Mapping):
        normalized = _label_key(color_field)
        raw_category_map = next(
            (
                value
                for key, value in raw_maps.items()
                if _label_key(str(key)) == normalized and isinstance(value, Mapping)
            ),
            None,
        )
    if not isinstance(raw_category_map, Mapping):
        return
    category_map = {str(key): str(value) for key, value in raw_category_map.items()}
    for trace in fig.data:
        if isinstance(trace, go.Pie):
            labels = (
                [str(value) for value in list(trace.labels)] if trace.labels is not None else []
            )
            if labels:
                existing = list(getattr(trace.marker, "colors", None) or [None] * len(labels))
                colors = [
                    category_map.get(label, existing[index] if index < len(existing) else None)
                    for index, label in enumerate(labels)
                ]
                trace.update(marker={"colors": colors})
            continue
        name = str(getattr(trace, "name", "") or "")
        color = category_map.get(name)
        if color is None:
            continue
        update: dict[str, Any] = {}
        if getattr(trace, "marker", None) is not None:
            update["marker_color"] = color
        if getattr(trace, "line", None) is not None:
            update["line_color"] = color
        if update:
            trace.update(**update)


def _label_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _apply_goal_lines(fig: go.Figure, tile: Mapping[str, Any], base: str) -> None:
    raw = tile.get("goal_lines", tile.get("goal_line"))
    if raw in (None, "", []):
        return
    default_color = _CHART_INKS[base]["goal_line"]
    rules = raw if isinstance(raw, list) else [raw]
    for rule in rules:
        if isinstance(rule, Mapping):
            raw_value = rule.get("value")
            label = str(rule.get("label", "Goal"))
            color = str(rule.get("color", default_color))
            dash = str(rule.get("dash", "dash"))
            axis = str(rule.get("axis", "y")).casefold()
        else:
            raw_value = rule
            label = "Goal"
            color = default_color
            dash = "dash"
            axis = "y"
        value = _as_float(raw_value)
        if value is None:
            continue
        if axis == "x":
            fig.add_vline(
                x=value,
                line_dash=dash,
                line_color=color,
                annotation_text=label,
                annotation_position="top",
            )
        else:
            fig.add_hline(
                y=value,
                line_dash=dash,
                line_color=color,
                annotation_text=label,
                annotation_position="top right",
            )


def _apply_value_format(fig: go.Figure, tile: Mapping[str, Any]) -> None:
    value_format = _format_spec(tile)
    if value_format is None:
        return
    if str(tile.get("chart", "")) in _COLOR_VALUE_CHARTS:
        _apply_color_value_format(fig, value_format)
        return
    fig.update_coloraxes(colorbar_tickformat=value_format)
    fig.update_yaxes(tickformat=value_format)
    for trace in fig.data:
        if getattr(trace, "type", None) == "indicator":
            trace.update(number={"valueformat": value_format}, delta={"valueformat": value_format})
            continue
        if hasattr(trace, "y"):
            if tile.get("chart") == "calibration_curve" or tile.get("chart") in _CURVE_CHARTS:
                trace.update(
                    hovertemplate=f"%{{x:{value_format}}}<br>%{{y:{value_format}}}<extra></extra>"
                )
            else:
                trace.update(hovertemplate=f"%{{x}}<br>%{{y:{value_format}}}<extra></extra>")


def _apply_color_value_format(fig: go.Figure, value_format: str) -> None:
    fig.update_coloraxes(colorbar_tickformat=value_format)
    for trace in fig.data:
        if hasattr(trace, "colorbar"):
            trace.update(colorbar={"tickformat": value_format})
        trace_type = getattr(trace, "type", None)
        hovertemplate = getattr(trace, "hovertemplate", None)
        if trace_type == "treemap":
            trace.update(
                hovertemplate=_format_hover_token(
                    hovertemplate,
                    "color",
                    value_format,
                )
            )
        elif trace_type == "heatmap":
            if isinstance(hovertemplate, str) and "%{z" in hovertemplate:
                trace.update(hovertemplate=_format_hover_token(hovertemplate, "z", value_format))
            else:
                trace.update(
                    hovertemplate=f"%{{y}}<br>%{{x}}<br>%{{z:{value_format}}}<extra></extra>"
                )


def _format_hover_token(hovertemplate: Any, token: str, value_format: str) -> str:
    if not isinstance(hovertemplate, str):
        return f"%{{{token}:{value_format}}}<extra></extra>"
    pattern = rf"%\{{{re.escape(token)}(?::[^}}]*)?\}}"
    return re.sub(pattern, f"%{{{token}:{value_format}}}", hovertemplate)


def _apply_trend_delta(fig: go.Figure, rows: pl.DataFrame, tile: Mapping[str, Any]) -> None:
    raw = tile.get("trend_delta", tile.get("show_trend_delta"))
    if not raw:
        return
    config: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    column = str(config.get("column") or tile.get("y") or tile.get("value") or "")
    if not column or column not in rows.columns:
        return
    values_frame = rows
    x = tile.get("x")
    if isinstance(x, str) and x in values_frame.columns:
        values_frame = values_frame.sort(x)
    values = [
        float(value)
        for value in values_frame.get_column(column).drop_nulls().to_list()
        if isinstance(value, int | float)
    ]
    if len(values) < 2:
        return
    first, latest = values[0], values[-1]
    absolute = latest - first
    relative = absolute / abs(first) if first else None
    label = str(config.get("label", "Delta"))
    text = _format_delta(label, absolute, relative, tile)
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=1,
        y=1,
        xanchor="right",
        yanchor="bottom",
        yshift=8,
        showarrow=False,
        font={"size": 12},
    )
    # The badge sits above the plot area; the template's 18px top margin clips it.
    current_top = getattr(fig.layout.margin, "t", None)
    fig.update_layout(margin_t=max(40, current_top if current_top is not None else 0))


def _sort_and_limit(
    rows: pl.DataFrame,
    tile: Mapping[str, Any],
    *,
    default_sort: str,
) -> pl.DataFrame:
    sort_by = str(tile.get("sort_by") or default_sort)
    top_n = _as_int(tile.get("top_n"))
    should_sort = bool(tile.get("sort_by")) or top_n is not None
    out = rows
    if should_sort and sort_by in out.columns:
        descending = str(tile.get("sort_direction", "desc")).casefold() != "asc"
        out = out.sort(sort_by, descending=descending)
    if top_n is not None and top_n > 0:
        out = out.head(top_n)
    return out


def _bar_mode(tile: Mapping[str, Any]) -> tuple[str, str | None]:
    raw = str(tile.get("barmode", "group")).casefold()
    if raw in {"percent", "stacked_percent", "100%", "100_percent"}:
        return "stack", "percent"
    if raw in {"stacked", "stack"}:
        return "stack", None
    return raw, str(tile.get("barnorm")) if tile.get("barnorm") else None


def _conditional_colors(rows: pl.DataFrame, tile: Mapping[str, Any]) -> list[str] | None:
    raw_rules = tile.get("conditional_formatting")
    if not isinstance(raw_rules, list):
        return None
    rules = [rule for rule in raw_rules if isinstance(rule, Mapping)]
    if not rules:
        return None
    default = str(tile.get("default_color", "#2563eb"))
    colors: list[str] = []
    for row in rows.iter_rows(named=True):
        colors.append(
            next((_rule_color(rule) for rule in rules if _rule_matches(rule, row)), default)
        )
    return colors


def _rule_matches(rule: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    column = str(rule.get("column", ""))
    if column not in row:
        return False
    actual = row[column]
    expected = rule.get("value")
    operator = str(rule.get("operator", rule.get("op", "=="))).strip()
    equality_ops: dict[str, bool] = {
        "==": actual == expected,
        "=": actual == expected,
        "!=": actual != expected,
    }
    if operator in equality_ops:
        return equality_ops[operator]
    actual_float = _as_float(actual)
    expected_float = _as_float(expected)
    if actual_float is None or expected_float is None:
        return False
    numeric_ops: dict[str, bool] = {
        ">": actual_float > expected_float,
        ">=": actual_float >= expected_float,
        "<": actual_float < expected_float,
        "<=": actual_float <= expected_float,
    }
    return numeric_ops.get(operator, False)


def _rule_color(rule: Mapping[str, Any]) -> str:
    return str(rule.get("color", "#2563eb"))


def _format_spec(tile: Mapping[str, Any]) -> str | None:
    raw = tile.get("value_format", tile.get("number_format"))
    if raw is None:
        return None
    value_format = str(raw).casefold()
    if value_format == "percent":
        return ".2%"
    if value_format == "integer":
        return ",.0f"
    if value_format == "currency":
        return "$,.2f"
    if value_format == "number":
        return ",.2f"
    return str(raw)


def _axis_title(tile: Mapping[str, Any], field_key: str, override_key: str) -> str | None:
    override = tile.get(override_key)
    if override not in (None, ""):
        return str(override)
    labels = tile.get("labels")
    field = tile.get(field_key)
    if isinstance(labels, Mapping) and field_key in labels and labels[field_key] not in (None, ""):
        return str(labels[field_key])
    if (
        isinstance(labels, Mapping)
        and isinstance(field, str)
        and field in labels
        and labels[field] not in (None, "")
    ):
        return str(labels[field])
    if isinstance(field, str) and field:
        return _friendly_label(field)
    return None


def _friendly_label(value: str) -> str:
    stripped = value.removeprefix("VS_").removeprefix("vs_")
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", stripped)
    words = [word for word in normalized.replace("-", "_").split("_") if word]
    if not words:
        return value
    acronym_words = {"ctr", "cvr", "clv", "rfm", "roc", "auc", "p95", "p99", "id"}
    small_words = {"a", "an", "and", "as", "by", "for", "in", "of", "on", "or", "per", "to", "vs"}
    rendered: list[str] = []
    for idx, word in enumerate(words):
        lower = word.casefold()
        if lower in acronym_words:
            rendered.append(lower.upper())
        elif idx > 0 and lower in small_words:
            rendered.append(lower)
        else:
            rendered.append(lower.capitalize())
    return " ".join(rendered)


def _format_delta(
    label: str,
    absolute: float,
    relative: float | None,
    tile: Mapping[str, Any],
) -> str:
    spec = _format_spec(tile)
    if spec == ".2%":
        absolute_text = f"{absolute:+.2%}"
    elif spec == ",.0f":
        absolute_text = f"{absolute:+,.0f}"
    elif spec == "$,.2f":
        absolute_text = f"${absolute:+,.2f}"
    else:
        absolute_text = f"{absolute:+,.2f}"
    if relative is None:
        return f"{label} {absolute_text}"
    return f"{label} {absolute_text} ({relative:+.1%})"


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _reference_for_row(
    row: Mapping[str, Any],
    tile: Mapping[str, Any],
    dimension_fields: Iterable[str] = (),
    *,
    default: float = 0.0,
) -> float:
    raw_references = tile.get("references")
    if raw_references is None:
        raw_references = tile.get("reference")
    scalar_reference = _as_float(raw_references)
    if scalar_reference is not None:
        return scalar_reference
    if not isinstance(raw_references, Mapping):
        return default
    for key in _reference_keys(row, dimension_fields):
        if key in raw_references:
            reference = _as_float(raw_references[key])
            return reference if reference is not None else default
    reference = _as_float(next(iter(raw_references.values()), default))
    return reference if reference is not None else default


def _reference_keys(row: Mapping[str, Any], dimension_fields: Iterable[str]) -> list[str]:
    keys: list[str] = []
    dimension_values = [str(row[field]) for field in dimension_fields if field in row]
    if dimension_values:
        keys.extend(
            [
                "_".join(dimension_values),
                " ".join(dimension_values),
                "|".join(dimension_values),
            ]
        )
        keys.extend(dimension_values)
    keys.extend(str(value) for value in row.values())
    return list(dict.fromkeys(keys))


def _config_values(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, str) or not isinstance(value, Iterable):
        return [value]
    return list(value)


def _optional_column(rows: pl.DataFrame, value: Any) -> str | None:
    text = _optional_text(value)
    if text is None or text not in rows.columns:
        return None
    return text


def _facet_height(rows: pl.DataFrame, facet_row: str | None, *, per_row: int) -> int:
    if facet_row is None or facet_row not in rows.columns:
        return 640
    return max(640, per_row * rows[facet_row].n_unique())


def _strip_facet_annotation_prefixes(fig: go.Figure, tile: Mapping[str, Any], base: str) -> None:
    facet_fields = {
        field for field in (_facet_row(tile), _facet_col(tile)) if field not in (None, "")
    }
    if not facet_fields:
        return
    label_color = _CHART_INKS[base]["facet_label"]
    fig.for_each_annotation(
        lambda annotation: _clean_facet_annotation(annotation, facet_fields, label_color)
    )


def _clean_facet_annotation(annotation: Any, facet_fields: set[str], label_color: str) -> None:
    if not _is_facet_annotation(annotation, facet_fields):
        return
    annotation.update(
        text=annotation.text.split("=", 1)[-1],
        font={"size": 12, "color": label_color},
    )


def _is_facet_annotation(annotation: Any, facet_fields: set[str]) -> bool:
    text = getattr(annotation, "text", None)
    if not isinstance(text, str):
        return False
    field, separator, _value = text.partition("=")
    return bool(separator) and field in facet_fields


def _has_facets(tile: Mapping[str, Any]) -> bool:
    return _facet_row(tile) is not None or _facet_col(tile) is not None


def _apply_outer_facet_axis_titles(
    fig: go.Figure,
    *,
    x_title: str | None,
    y_title: str | None,
) -> None:
    x_axes = _layout_axes(fig, "x")
    y_axes = _layout_axes(fig, "y")
    if x_title and x_axes:
        fig.update_xaxes(title_text=None)
        y_domains = [
            _axis_domain(_axis_from_anchor(fig, axis.anchor, "y")) for _name, axis in x_axes
        ]
        bottom = min((domain[0] for domain in y_domains if domain is not None), default=None)
        outer = [
            axis
            for (_name, axis), domain in zip(x_axes, y_domains, strict=True)
            if bottom is not None and domain is not None and math.isclose(domain[0], bottom)
        ]
        _set_title_on_centered_axis(outer, x_title)
    if y_title and y_axes:
        fig.update_yaxes(title_text=None)
        x_domains = [
            _axis_domain(_axis_from_anchor(fig, axis.anchor, "x")) for _name, axis in y_axes
        ]
        left = min((domain[0] for domain in x_domains if domain is not None), default=None)
        outer = [
            axis
            for (_name, axis), domain in zip(y_axes, x_domains, strict=True)
            if left is not None and domain is not None and math.isclose(domain[0], left)
        ]
        _set_title_on_centered_axis(outer, y_title)


def _set_title_on_centered_axis(axes: list[Any], title: str) -> None:
    """Title one axis per orientation: repeating it on every facet row/column
    stacks copies into overlapping, unreadable text."""
    if not axes:
        return

    def distance_from_center(axis: Any) -> float:
        domain = _axis_domain(axis)
        if domain is None:
            return 0.0
        return abs(((domain[0] + domain[1]) / 2) - 0.5)

    min(axes, key=distance_from_center).update(title_text=title)


def _layout_axes(fig: go.Figure, prefix: str) -> list[tuple[str, Any]]:
    names = [name for name in fig.layout if re.fullmatch(rf"{prefix}axis\d*", name)]
    return [(name, getattr(fig.layout, name)) for name in sorted(names, key=_axis_sort_key)]


def _axis_sort_key(name: str) -> int:
    suffix = name.removeprefix("xaxis").removeprefix("yaxis")
    return int(suffix or "1")


def _axis_from_anchor(fig: go.Figure, anchor: Any, prefix: str) -> Any | None:
    if not isinstance(anchor, str) or not anchor.startswith(prefix):
        return None
    suffix = anchor.removeprefix(prefix)
    return getattr(fig.layout, f"{prefix}axis{suffix}", None)


def _axis_domain(axis: Any | None) -> tuple[float, float] | None:
    domain = getattr(axis, "domain", None)
    if domain is None or len(domain) != 2:
        return None
    return (float(domain[0]), float(domain[1]))


def _add_global_axis_labels(fig: go.Figure, *, x_label: str, y_label: str) -> None:
    fig.add_annotation(
        showarrow=False,
        xanchor="center",
        xref="paper",
        x=0.5,
        yref="paper",
        y=-0.1,
        text=x_label,
    )
    fig.add_annotation(
        showarrow=False,
        xanchor="center",
        xref="paper",
        x=-0.04,
        yanchor="middle",
        yref="paper",
        y=0.5,
        textangle=90,
        text=y_label,
    )


def _field(tile: Mapping[str, Any], name: str) -> str:
    value = tile.get(name)
    if value is None:
        raise ValueError(f"tile {tile.get('id', '<unknown>')!r} missing required field {name!r}")
    return str(value)


def _optional_text(value: Any) -> str | None:
    if value in (None, "", "---"):
        return None
    return str(value)


def _facet_row(tile: Mapping[str, Any]) -> str | None:
    facets = tile.get("facets")
    if isinstance(facets, Mapping) and "row" in facets:
        return _optional_text(facets["row"])
    return _optional_text(tile.get("facet_row"))


def _facet_col(tile: Mapping[str, Any]) -> str | None:
    facets = tile.get("facets")
    if isinstance(facets, Mapping) and "col" in facets:
        return _optional_text(facets["col"])
    if isinstance(facets, Mapping) and "column" in facets:
        return _optional_text(facets["column"])
    return _optional_text(tile.get("facet_col", tile.get("facet_column")))


__all__ = ["MAX_POINTS", "render_chart"]
