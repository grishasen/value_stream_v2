"""Dashboard discovery for the read-only tool surfaces.

``catalog_chat_manifest`` describes metrics; it says nothing about the
dashboards an operator actually reads. A tool client needs the authored
dashboards, pages, filters, and tiles to answer "what does this report show"
the way the UI does, so this module renders that structure.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from valuestream.config import model
from valuestream.config.report_fields import metric_output_columns
from valuestream.reporting.tiles import (
    available_filter_columns_for_page,
    available_filter_columns_for_tile,
    grain_for_tile,
    group_by_for_tile,
    tile_to_dict,
)

# Authored tile keys that describe the mark rather than the query. They are
# reported verbatim so a client can reproduce the chart the operator sees.
_CHART_FIELDS = (
    "x",
    "y",
    "color",
    "facet_col",
    "facet_row",
    "line_dash",
    "symbol",
    "path",
    "names",
    "theta",
    "stages",
    "secondary_metric",
    "primary_mark",
    "shared_y_axis",
    "lower_output",
    "upper_output",
    "metric_output",
    "property",
    "value_format",
    "goal_line",
    "barmode",
    "sort_by",
    "sort_direction",
    "top_n",
    "x_axis_title",
    "y_axis_title",
)


def dashboard_manifest(
    catalog: model.Catalog,
    *,
    dashboard_id: str | None = None,
    include_tiles: bool = True,
) -> dict[str, Any]:
    """Return authored dashboards, pages, and tiles with their chart specs."""

    dashboards = [
        dashboard
        for dashboard in catalog.dashboards.dashboards
        if dashboard_id is None or dashboard.id == dashboard_id
    ]
    if dashboard_id is not None and not dashboards:
        known = ", ".join(item.id for item in catalog.dashboards.dashboards)
        raise ValueError(f"unknown dashboard {dashboard_id!r}; available: {known}")
    return {
        "dashboards": [
            _dashboard_payload(catalog, dashboard, include_tiles=include_tiles)
            for dashboard in dashboards
        ]
    }


def resolve_tile(
    catalog: model.Catalog,
    *,
    dashboard_id: str,
    page_id: str,
    tile_id: str,
) -> tuple[model.Dashboard, model.DashboardPage, model.Tile]:
    """Find one authored tile, naming the valid options when it is missing."""

    dashboard = next(
        (item for item in catalog.dashboards.dashboards if item.id == dashboard_id), None
    )
    if dashboard is None:
        known = ", ".join(item.id for item in catalog.dashboards.dashboards)
        raise ValueError(f"unknown dashboard {dashboard_id!r}; available: {known}")
    page = next((item for item in dashboard.pages if item.id == page_id), None)
    if page is None:
        known = ", ".join(item.id for item in dashboard.pages)
        raise ValueError(
            f"unknown page {page_id!r} on dashboard {dashboard_id!r}; available: {known}"
        )
    tile = next((item for item in page.tiles if item.id == tile_id), None)
    if tile is None:
        known = ", ".join(item.id for item in page.tiles)
        raise ValueError(f"unknown tile {tile_id!r} on page {page_id!r}; available: {known}")
    return dashboard, page, tile


def _dashboard_payload(
    catalog: model.Catalog,
    dashboard: model.Dashboard,
    *,
    include_tiles: bool,
) -> dict[str, Any]:
    return {
        "id": dashboard.id,
        "title": dashboard.title,
        "pages": [
            _page_payload(catalog, page, include_tiles=include_tiles) for page in dashboard.pages
        ],
    }


def _page_payload(
    catalog: model.Catalog,
    page: model.DashboardPage,
    *,
    include_tiles: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": page.id,
        "title": page.title,
        "filter_columns": available_filter_columns_for_page(catalog, page),
        "filters": [
            {
                "field": authored.field,
                "label": authored.label,
                "display": authored.display,
                "scope": authored.scope,
                "control": authored.control,
            }
            for authored in page.filters
        ],
        "time_filter": {
            "default": page.time_filter.default,
            "presets": list(page.time_filter.presets),
        },
        "tile_count": len(page.tiles),
    }
    if include_tiles:
        payload["tiles"] = [_tile_payload(catalog, tile) for tile in page.tiles]
    return payload


def _tile_payload(catalog: model.Catalog, tile: model.Tile) -> dict[str, Any]:
    tile_dict = tile_to_dict(tile)
    metric = catalog.metrics.metrics.get(tile.metric)
    payload: dict[str, Any] = {
        "id": tile.id,
        "title": tile.title,
        "metric": tile.metric,
        "chart": tile.chart,
        "grain": grain_for_tile(tile_dict),
        "group_by": group_by_for_tile(tile_dict),
        "filter_columns": available_filter_columns_for_tile(catalog, tile),
        "outputs": metric_output_columns(tile.metric, metric) if metric is not None else [],
        "chart_fields": _chart_fields(tile_dict),
    }
    if tile.description:
        payload["description"] = tile.description
    if tile_dict.get("filters"):
        payload["authored_filters"] = dict(tile_dict["filters"])
    if tile.kpi is not None:
        payload["kpi"] = {
            "comparison": tile.kpi.comparison,
            "comparison_period": tile.kpi.comparison_period,
            "sparkline_grain": tile.kpi.sparkline_grain,
            "sparkline_points": tile.kpi.sparkline_points,
            "target": tile.kpi.target,
        }
    return payload


def _chart_fields(tile_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: tile_dict[field]
        for field in _CHART_FIELDS
        if tile_dict.get(field) not in (None, "", [], {})
    }


__all__ = ["dashboard_manifest", "resolve_tile"]
