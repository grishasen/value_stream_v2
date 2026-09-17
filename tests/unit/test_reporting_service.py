"""The report service the UI, API, and MCP server share."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from valuestream.config import model
from valuestream.config.loader import load
from valuestream.reporting.kpi import (
    calendar_period,
    comparison_period_label,
    kpi_bundle,
    latest_series_date,
    parse_period_date,
    scalar_value,
    sparkline_values,
)
from valuestream.reporting.manifest import dashboard_manifest, resolve_tile
from valuestream.reporting.tiles import query_tile

DEMO_WS = Path("examples/demo")


def _catalog() -> model.Catalog:
    return load(DEMO_WS)


def _first_tile(catalog: model.Catalog) -> tuple[str, str, model.Tile]:
    dashboard = catalog.dashboards.dashboards[0]
    page = dashboard.pages[0]
    return dashboard.id, page.id, page.tiles[0]


class _Recorder:
    """Stand-in for query_metric that records how a tile was queried."""

    def __init__(self, frame: pl.DataFrame | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._frame = frame if frame is not None else pl.DataFrame({"value": [1.0]})

    def __call__(self, workspace: Any, metric_name: str, **options: Any) -> pl.DataFrame:
        self.calls.append({"workspace": workspace, "metric": metric_name, **options})
        return self._frame


@pytest.mark.unit
def test_dashboard_manifest_exposes_the_ids_the_tile_tools_take() -> None:
    catalog = _catalog()

    manifest = dashboard_manifest(catalog)

    assert manifest["dashboards"], "demo workspace should author at least one dashboard"
    dashboard = manifest["dashboards"][0]
    page = dashboard["pages"][0]
    tile = page["tiles"][0]
    assert {"id", "title", "pages"} <= set(dashboard)
    assert {"id", "title", "filter_columns", "filters", "time_filter", "tiles"} <= set(page)
    assert {"id", "metric", "chart", "grain", "group_by", "outputs", "chart_fields"} <= set(tile)
    # The ids must round-trip into resolve_tile.
    assert resolve_tile(
        catalog, dashboard_id=dashboard["id"], page_id=page["id"], tile_id=tile["id"]
    )


@pytest.mark.unit
def test_dashboard_manifest_can_skip_tiles_for_a_compact_index() -> None:
    catalog = _catalog()

    compact = dashboard_manifest(catalog, include_tiles=False)
    page = compact["dashboards"][0]["pages"][0]

    assert "tiles" not in page
    assert page["tile_count"] >= 1


@pytest.mark.unit
def test_dashboard_manifest_filters_to_one_dashboard() -> None:
    catalog = _catalog()
    wanted = catalog.dashboards.dashboards[0].id

    manifest = dashboard_manifest(catalog, dashboard_id=wanted)

    assert [item["id"] for item in manifest["dashboards"]] == [wanted]


@pytest.mark.unit
def test_unknown_ids_name_the_available_options() -> None:
    catalog = _catalog()
    dashboard_id, page_id, _tile = _first_tile(catalog)

    with pytest.raises(ValueError, match="available:"):
        dashboard_manifest(catalog, dashboard_id="nope")
    with pytest.raises(ValueError, match="unknown page 'nope'"):
        resolve_tile(catalog, dashboard_id=dashboard_id, page_id="nope", tile_id="x")
    with pytest.raises(ValueError, match="unknown tile 'nope'"):
        resolve_tile(catalog, dashboard_id=dashboard_id, page_id=page_id, tile_id="nope")


@pytest.mark.unit
def test_query_tile_uses_the_injected_query_function() -> None:
    catalog = _catalog()
    _dashboard_id, _page_id, tile = _first_tile(catalog)
    recorder = _Recorder()

    query_tile(DEMO_WS, catalog, tile, query_fn=recorder)

    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["metric"] == tile.metric
    # The tile's own grain and group-by inference drives the query, not defaults.
    assert "grain" in call
    assert "include_state_columns" in call


@pytest.mark.unit
def test_authored_tile_filters_win_over_caller_filters() -> None:
    # An authored filter defines the metric's valid population, so a page
    # filter on the same field must not be able to widen or replace it.
    catalog = _catalog()
    _dashboard_id, _page_id, template = _first_tile(catalog)
    tile = template.model_copy(update={"filters": {"Channel": "Web"}})
    recorder = _Recorder()

    query_tile(DEMO_WS, catalog, tile, filters={"Channel": "Mobile"}, query_fn=recorder)

    assert recorder.calls[0]["filters"]["Channel"] == "Web"


@pytest.mark.unit
def test_caller_filters_apply_on_fields_the_tile_does_not_pin() -> None:
    catalog = _catalog()
    _dashboard_id, _page_id, template = _first_tile(catalog)
    tile = template.model_copy(update={"filters": {"Channel": "Web"}})
    recorder = _Recorder()

    query_tile(DEMO_WS, catalog, tile, filters={"Issue": "Sales"}, query_fn=recorder)

    assert recorder.calls[0]["filters"] == {"Channel": "Web", "Issue": "Sales"}


@pytest.mark.unit
def test_kpi_bundle_reads_the_tile_output_column_not_the_first_column() -> None:
    catalog = _catalog()
    tile = next(
        (
            item
            for dashboard in catalog.dashboards.dashboards
            for page in dashboard.pages
            for item in page.tiles
            if item.chart == "kpi_card"
        ),
        None,
    )
    if tile is None:
        pytest.skip("demo workspace has no KPI tile")
    recorder = _Recorder(pl.DataFrame({tile.metric: [0.25]}))

    bundle = kpi_bundle(DEMO_WS, catalog, tile, query_fn=recorder)

    assert bundle.value_column
    assert recorder.calls


@pytest.mark.unit
def test_scalar_value_reports_n_a_rather_than_guessing() -> None:
    assert scalar_value(pl.DataFrame({"x": [1.0]}), "x") == 1.0
    assert scalar_value(pl.DataFrame({"x": [1.0, 2.0]}), "x") == "n/a"
    assert scalar_value(pl.DataFrame({"x": []}), "x") == "n/a"
    assert scalar_value(pl.DataFrame({"x": [1.0]}), "missing") == "n/a"


@pytest.mark.unit
def test_sparkline_needs_at_least_two_points() -> None:
    assert sparkline_values(pl.DataFrame({"Day": ["2024-01-01"], "x": [1.0]}), "x", 30) is None
    series = pl.DataFrame({"Day": ["2024-01-02", "2024-01-01"], "x": [2.0, 1.0]})
    # Sorted by its time column, not by row order.
    assert sparkline_values(series, "x", 30) == (1.0, 2.0)


@pytest.mark.unit
def test_period_helpers_cover_the_authored_comparison_periods() -> None:
    day = dt.date(2024, 5, 15)

    assert calendar_period(day, "day") == (day, day)
    assert calendar_period(day, "month") == (dt.date(2024, 5, 1), dt.date(2024, 5, 31))
    assert calendar_period(day, "quarter") == (dt.date(2024, 4, 1), dt.date(2024, 6, 30))
    assert calendar_period(day, "year") == (dt.date(2024, 1, 1), dt.date(2024, 12, 31))
    assert calendar_period(day, "week") == (dt.date(2024, 5, 13), dt.date(2024, 5, 19))
    assert comparison_period_label(dt.date(2024, 4, 1), dt.date(2024, 4, 30)).startswith("vs Apr")


@pytest.mark.unit
def test_period_dates_parse_daily_and_monthly_labels() -> None:
    assert parse_period_date("2024-05-15") == dt.date(2024, 5, 15)
    assert parse_period_date("2024-05") == dt.date(2024, 5, 1)
    assert parse_period_date("not-a-date") is None
    assert latest_series_date(pl.DataFrame({"Month": ["2024-01", "2024-03"]})) == dt.date(2024, 3, 1)
