"""Dashboard tile queries shared by the UI, the HTTP API, and the MCP server.

The Streamlit report page, the read-only API, and the MCP tools must answer a
tile question identically, so the tile-shaped query logic lives here rather
than in ``valuestream.ui``: chart-specific grain and group-by inference, the
combo secondary-metric join, the histogram property remap, and time-column
restoration. Nothing in this module imports Streamlit — the UI layers its own
caching on top by passing ``query_fn``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import polars as pl

from valuestream.config import model
from valuestream.config.report_fields import (
    distribution_property_metrics,
    metric_output_columns,
)
from valuestream.query import query_metric

QueryFn = Callable[..., pl.DataFrame]

_TIME_GRAINS = {
    "Day": "daily",
    "day": "daily",
    "as_of_date": "daily",
    "Week": "weekly",
    "week": "weekly",
    "Month": "monthly",
    "month": "monthly",
    "period": "monthly",
    "Quarter": "quarterly",
    "quarter": "quarterly",
    "Year": "yearly",
    "year": "yearly",
}
_TIME_GRAIN_RANK = {
    "daily": 0,
    "weekly": 1,
    "monthly": 2,
    "quarterly": 3,
    "yearly": 4,
    "summary": 5,
}
_TIME_COLUMNS = set(_TIME_GRAINS)
_TIME_COLUMN_EQUIVALENTS = {
    "daily": ("Day", "day", "as_of_date"),
    "weekly": ("Week", "week"),
    "monthly": ("Month", "month", "period"),
    "quarterly": ("Quarter", "quarter"),
    "yearly": ("Year", "year"),
}
_FACET_FIELDS = ("facet_col", "facet_row")
_CURVE_CHARTS = {"roc_curve", "precision_recall_curve", "gain_curve", "lift_curve"}
_CHART_DIMENSION_FIELDS = {
    "line": ("x", *_FACET_FIELDS, "color", "line_dash", "symbol"),
    "stacked_area": ("x", *_FACET_FIELDS, "color"),
    "bar": ("x", *_FACET_FIELDS, "color"),
    "kpi_card": ("group_by",),
    "waterfall": ("x",),
    "pareto": ("x", *_FACET_FIELDS, "color"),
    "treemap": ("path",),
    "heatmap": ("x", "y"),
    "scatter": ("animation_frame", "animation_group", *_FACET_FIELDS, "color"),
    "combo": ("x", *_FACET_FIELDS, "color"),
    "interval": ("x", *_FACET_FIELDS, "color"),
    "donut": ("names", "color"),
    "geo_map": ("locations", "lat", "lon", *_FACET_FIELDS, "color"),
    "table": ("group_by", "columns"),
    "bar_polar": ("theta", "color"),
    "sankey": ("path",),
    "gauge": ("facet_row", "facet_col", "group_by"),
    "funnel": ("x", *_FACET_FIELDS, "color"),
    "boxplot": ("x", *_FACET_FIELDS, "color"),
    "histogram": (*_FACET_FIELDS, "color"),
    "calibration_curve": (*_FACET_FIELDS, "color"),
    "roc_curve": (*_FACET_FIELDS, "color"),
    "precision_recall_curve": (*_FACET_FIELDS, "color"),
    "gain_curve": (*_FACET_FIELDS, "color"),
    "lift_curve": (*_FACET_FIELDS, "color"),
    "corr": ("color",),
    "descriptive_line": ("x", *_FACET_FIELDS, "color"),
    "experiment_z_score": ("x", "y", *_FACET_FIELDS, "color"),
    "experiment_odds_ratio": ("x", "y", *_FACET_FIELDS, "color"),
}
_DEFAULT_DIMENSION_FIELDS = (
    "group_by",
    "path",
    "x",
    "color",
    "line_dash",
    "symbol",
    *_FACET_FIELDS,
    "animation_frame",
    "animation_group",
)

_STATEFUL_CHARTS = {"funnel", "model"}


def query_tile(
    workspace_path: str | Path,
    catalog: model.Catalog,
    tile: model.Tile,
    *,
    filters: Mapping[str, Any] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    query_fn: QueryFn | None = None,
) -> pl.DataFrame:
    """Query aggregate data for a dashboard tile.

    ``query_fn`` defaults to the uncached governed query. The Streamlit app
    passes its cached wrapper so the UI keeps its memoization without this
    module depending on Streamlit.
    """

    run_query = query_fn if query_fn is not None else query_metric
    tile_dict = tile_to_dict(tile)
    query_metric_name = tile.metric
    if tile.chart == "histogram" and tile_dict.get("property") not in (None, ""):
        property_name = str(tile_dict["property"])
        distribution_metric_name = distribution_property_metrics(catalog, tile.metric).get(
            property_name,
            "",
        )
        query_metric_name = distribution_metric_name or query_metric_name
    canonical_tile = tile_dict
    tile_filters: dict[str, Any] = {}
    filter_columns = available_filter_columns_for_tile(catalog, tile)
    if filters:
        for key, value in filters.items():
            column = _canonical_column(key, filter_columns)
            if value not in (None, "") and column is not None:
                tile_filters[column] = value
    # Authored tile filters define the metric's valid population and therefore
    # take precedence over interactive page filters on the same field.
    tile_filters.update(dict(tile_dict.get("filters") or {}))
    tile_filters = _canonicalize_filter_keys(tile_filters, filter_columns)
    grain = grain_for_tile(canonical_tile)
    group_by = _processor_group_columns(
        catalog,
        query_metric_name,
        group_by_for_tile(canonical_tile),
    )
    rows = run_query(
        workspace_path,
        query_metric_name,
        group_by=list(group_by),
        filters=tile_filters,
        grain=grain,
        start=start,
        end=end,
        compare=None,
        include_state_columns=(
            tile.chart in _STATEFUL_CHARTS
            or tile.chart == "histogram"
            or tile.chart.startswith("descriptive_")
            or _tile_references_scalar_state(catalog, query_metric_name, canonical_tile)
        ),
        include_quantile_suite=tile.chart in {"boxplot", "combo"},
        include_curve_columns=tile.chart in _CURVE_CHARTS,
    )
    if tile.chart == "combo":
        rows = _join_combo_secondary_metric(
            workspace_path,
            catalog,
            tile=tile,
            canonical_tile=canonical_tile,
            tile_filters=tile_filters,
            grain=grain,
            group_by=group_by,
            start=start,
            end=end,
            primary_rows=rows,
            run_query=run_query,
        )
    return _restore_time_columns(rows, tile_dict)


def _join_combo_secondary_metric(
    workspace_path: str | Path,
    catalog: model.Catalog,
    *,
    tile: model.Tile,
    canonical_tile: Mapping[str, Any],
    tile_filters: Mapping[str, Any],
    grain: str,
    group_by: list[str],
    start: dt.date | None,
    end: dt.date | None,
    primary_rows: pl.DataFrame,
    run_query: QueryFn,
) -> pl.DataFrame:
    """Join the combo tile's compatible secondary metric onto its rows."""

    return join_secondary_metric(
        workspace_path,
        catalog,
        primary_metric=tile.metric,
        secondary_metric=str(canonical_tile.get("secondary_metric") or ""),
        primary_rows=primary_rows,
        group_by=group_by,
        filters=tile_filters,
        grain=grain,
        start=start,
        end=end,
        run_query=run_query,
    )


def join_secondary_metric(
    workspace_path: str | Path,
    catalog: model.Catalog,
    *,
    primary_metric: str,
    secondary_metric: str,
    primary_rows: pl.DataFrame,
    group_by: list[str],
    filters: Mapping[str, Any],
    grain: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
    run_query: QueryFn | None = None,
) -> pl.DataFrame:
    """Join a combo chart's secondary metric onto the primary metric's rows.

    A combo draws two metrics on one set of categories, so the second metric
    is queried over the same dimensions and filters and joined on the shared
    non-value columns. Used by both the dashboard tile path and the ad-hoc
    chart tools so a combo means the same thing on either surface.
    """

    query = run_query if run_query is not None else query_metric
    secondary = catalog.metrics.metrics.get(secondary_metric)
    if secondary is None:
        raise ValueError(f"Combo secondary metric {secondary_metric!r} is not configured.")
    secondary_outputs = metric_output_columns(secondary_metric, secondary)
    if secondary_outputs != [secondary_metric]:
        raise ValueError(
            f"Combo secondary metric {secondary_metric!r} must expose one scalar output."
        )
    secondary_rows = query(
        workspace_path,
        secondary_metric,
        group_by=list(group_by),
        filters=dict(filters),
        grain=grain,
        start=start,
        end=end,
        compare=None,
        include_state_columns=False,
        include_quantile_suite=False,
        include_curve_columns=False,
    )
    if secondary_metric not in secondary_rows.columns:
        raise ValueError(
            f"Combo secondary metric {secondary_metric!r} did not produce its scalar output."
        )

    primary = catalog.metrics.metrics[primary_metric]
    value_columns = {
        *metric_output_columns(primary_metric, primary),
        *secondary_outputs,
    }
    join_keys = [
        column
        for column in primary_rows.columns
        if column in secondary_rows.columns and column not in value_columns
    ]
    if not join_keys:
        if primary_rows.height <= 1 and secondary_rows.height <= 1:
            value = (
                secondary_rows.get_column(secondary_metric).item()
                if secondary_rows.height
                else None
            )
            return primary_rows.with_columns(pl.lit(value).alias(secondary_metric))
        raise ValueError(
            f"Combo metrics {primary_metric!r} and {secondary_metric!r} have no shared dimensions."
        )
    return primary_rows.join(
        secondary_rows.select([*join_keys, secondary_metric]),
        on=join_keys,
        how="left",
    )


def tile_to_dict(tile: model.Tile) -> dict[str, Any]:
    """Return the canonical strict tile payload."""

    return tile.model_dump(mode="python", exclude_none=True)


def available_filter_columns_for_tile(catalog: model.Catalog, tile: model.Tile) -> list[str]:
    """Return aggregate dimensions that can filter a tile without raw rows."""
    processor = _processor_for_metric(catalog, tile.metric)
    if processor is None:
        return filter_columns_for_tile(tile_to_dict(tile))
    return [column for column in processor.group_by if column not in _TIME_COLUMNS]


def available_filter_columns_for_page(catalog: model.Catalog, page: Any) -> list[str]:
    """Return all page-level aggregate filter columns in display order."""
    out: list[str] = []
    for tile in page.tiles:
        for column in available_filter_columns_for_tile(catalog, tile):
            if column not in out:
                out.append(column)
    return out


def grain_for_tile(tile: Mapping[str, Any]) -> str:
    """Choose the physical grain for a tile."""
    grain = _grain_for_dimension_fields(tile)
    if grain is not None:
        return grain
    if "grain" in tile:
        return model.normalize_grain_name(str(tile["grain"]))
    grain = _grain_for_value(tile.get("group_by"))
    if grain is not None:
        return grain
    return "summary"


def group_by_for_tile(tile: Mapping[str, Any]) -> list[str]:
    """Infer the dimensions represented by a tile's rendered marks."""
    candidates: list[str] = []
    for field in _dimension_fields_for_tile(tile):
        _append_dimensions(candidates, tile.get(field))
    return [candidate for candidate in candidates if candidate not in _TIME_COLUMNS]


def filter_columns_for_tile(tile: Mapping[str, Any]) -> list[str]:
    """Return tile dimensions that are useful as report filters."""
    candidates: list[str] = []
    _append_dimensions(candidates, tile.get("group_by"))
    for candidate in group_by_for_tile(tile):
        _append_dimensions(candidates, candidate)
    return [candidate for candidate in candidates if candidate not in _TIME_COLUMNS]


def _processor_group_columns(
    catalog: model.Catalog,
    metric_name: str,
    candidates: list[str],
) -> list[str]:
    metric = catalog.metrics.metrics.get(metric_name)
    if metric is None:
        return candidates
    processor = next(
        (candidate for candidate in catalog.processors.processors if candidate.id == metric.processor),
        None,
    )
    if processor is None:
        return candidates
    out: list[str] = []
    for candidate in candidates:
        column = _canonical_column(candidate, processor.group_by)
        if column is not None and column not in out:
            out.append(column)
    return out


def _processor_for_metric(catalog: model.Catalog, metric_name: str) -> model.Processor | None:
    metric = catalog.metrics.metrics.get(metric_name)
    if metric is None:
        return None
    return next(
        (candidate for candidate in catalog.processors.processors if candidate.id == metric.processor),
        None,
    )


def _tile_references_scalar_state(
    catalog: model.Catalog,
    metric_name: str,
    tile: Mapping[str, Any],
) -> bool:
    processor = _processor_for_metric(catalog, metric_name)
    if processor is None:
        return False
    scalar_states = {
        name
        for name, state in model.effective_processor_states(processor).items()
        if state.type in {"count", "value_sum", "min", "max", "pooled_mean", "pooled_variance"}
    }
    if not scalar_states:
        return False
    candidates: list[str] = []
    _append_dimensions(candidates, tile)
    return any(candidate in scalar_states for candidate in candidates)


def _append_dimensions(candidates: list[str], value: Any) -> None:
    values: Iterable[Any]
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]
    for item in values:
        if item in (None, "", "---"):
            continue
        candidate = str(item)
        if candidate not in candidates:
            candidates.append(candidate)


def _canonicalize_filter_keys(
    filters: Mapping[str, Any],
    columns: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in filters.items():
        out[_canonical_column(key, columns) or key] = value
    return out


def _canonical_column(candidate: str, columns: Iterable[str]) -> str | None:
    return candidate if candidate in columns else None


def _humanize_identifier(value: str) -> str:
    spaced = "".join(
        f" {character}"
        if index and character.isupper() and not value[index - 1].isupper()
        else character
        for index, character in enumerate(value.replace("_", " ").replace("-", " "))
    )
    return " ".join(spaced.split()).strip().capitalize()


def _grain_for_value(value: Any) -> str | None:
    candidates: list[str] = []
    _append_dimensions(candidates, value)
    return next(
        (_TIME_GRAINS[candidate] for candidate in candidates if candidate in _TIME_GRAINS), None
    )


def _grain_for_dimension_fields(tile: Mapping[str, Any]) -> str | None:
    grains = []
    for field in _dimension_fields_for_tile(tile):
        grain = _grain_for_value(tile.get(field))
        if grain is not None:
            grains.append(grain)
    if not grains:
        return None
    return min(grains, key=lambda grain: _TIME_GRAIN_RANK[grain])


def _dimension_fields_for_tile(tile: Mapping[str, Any]) -> tuple[str, ...]:
    fields = _CHART_DIMENSION_FIELDS.get(str(tile.get("chart", "")), _DEFAULT_DIMENSION_FIELDS)
    if str(tile.get("chart", "")).casefold() == "gauge" and _has_facet_dimensions(tile):
        return tuple(field for field in fields if field != "group_by")
    return fields


def _has_facet_dimensions(tile: Mapping[str, Any]) -> bool:
    return bool(tile.get("facet_row") or tile.get("facet_col"))


def _restore_time_columns(rows: pl.DataFrame, tile: Mapping[str, Any]) -> pl.DataFrame:
    out = rows
    candidates: list[str] = []
    for field in _dimension_fields_for_tile(tile):
        _append_dimensions(candidates, tile.get(field))
    for candidate in candidates:
        grain = _TIME_GRAINS.get(candidate)
        if grain is None or candidate in out.columns:
            continue
        source = next(
            (column for column in _TIME_COLUMN_EQUIVALENTS[grain] if column in out.columns),
            None,
        )
        if source is not None:
            out = out.with_columns(pl.col(source).alias(candidate))
    return out


__all__ = [
    "QueryFn",
    "available_filter_columns_for_page",
    "available_filter_columns_for_tile",
    "filter_columns_for_tile",
    "grain_for_tile",
    "group_by_for_tile",
    "join_secondary_metric",
    "query_tile",
    "tile_to_dict",
]
