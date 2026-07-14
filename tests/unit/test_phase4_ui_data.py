"""Phase 4 UI helper tests."""

from __future__ import annotations

import csv
import datetime as dt
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import streamlit as st
from plotly.graph_objects import Figure  # type: ignore[import-untyped]

import valuestream.ui.freshness as freshness_module
from valuestream.config import model
from valuestream.config.canonical import processor_computation_hash
from valuestream.config.validate import CatalogValidationResult
from valuestream.engine.ledger import insert_chunk, insert_run
from valuestream.query import AggregateNotReadyError
from valuestream.store.parquet import write_aggregate
from valuestream.ui import data as ui_data
from valuestream.ui.context import ValueStreamContext, load_context
from valuestream.ui.data import (
    _restore_time_columns,
    available_filter_columns_for_page,
    filter_capabilities_for_page,
    filter_columns_for_tile,
    grain_for_tile,
    group_by_for_tile,
    parse_filter_text,
    partition_filters_for_tile,
    query_tile,
)
from valuestream.ui.freshness import Freshness, metric_freshness
from valuestream.ui.pages.ai_config_studio import _natural_key_fields
from valuestream.ui.pages.config_builder import (
    _new_processor_template,
    _source_field_options,
    _state_rows,
)
from valuestream.ui.pages.reports import (
    ADVANCED_COLOR_SCALES,
    ADVANCED_FIELD_CONTROLS,
    REPORT_CHART_HEIGHT_FALLBACK_PX,
    REPORT_CHART_HEIGHT_HERO_PX,
    TILE_LAYOUT_HERO,
    _advanced_style_controls,
    _advanced_tile_from_fields,
    _advanced_tile_seed,
    _filter_chip_labels,
    _has_grouped_gauge_tile,
    _is_coverage_page,
    _is_descriptive_report_page,
    _is_full_width_tile,
    _kpi_bundle,
    _kpi_strip,
    _page_help_text,
    _page_status_banner,
    _report_chart_height,
    _rows_csv,
    _tile_help_text,
    _tile_value_columns,
)
from valuestream.ui.presentation import resolve_tile_presentation


@pytest.fixture(autouse=True)
def _clear_query_tile_cache() -> None:
    ui_data._cached_query_metric.clear()


@pytest.mark.unit
def test_tile_grain_inference() -> None:
    assert grain_for_tile({"chart": "line", "x": "Day"}) == "daily"
    assert grain_for_tile({"chart": "line", "x": "Month"}) == "monthly"
    assert (
        grain_for_tile({"chart": "scatter", "x": "Count", "animation_frame": "Month"}) == "monthly"
    )
    assert grain_for_tile({"chart": "bar", "x": "group_viz", "facet_col": "Quarter"}) == "quarterly"
    assert grain_for_tile({"chart": "line", "x": "Year", "facet_col": "Month"}) == "monthly"
    assert grain_for_tile({"chart": "bar", "x": "Quarter", "grain": "summary"}) == "quarterly"
    assert grain_for_tile({"chart": "gauge", "value": "CTR"}) == "summary"


@pytest.mark.unit
def test_tile_group_by_inference_uses_column_names() -> None:
    result = group_by_for_tile(
        {"x": "Day", "color": "Channel", "facets": {"row": "Placement"}},
    )

    assert result == ["Channel", "Placement"]


@pytest.mark.unit
def test_gauge_group_by_inference_prefers_facets_and_keeps_legacy_group_by() -> None:
    assert group_by_for_tile(
        {
            "chart": "gauge",
            "value": "CTR",
            "facet_row": "Channel",
            "facet_col": "Placement",
            "group_by": ["Stale"],
        }
    ) == ["Channel", "Placement"]
    assert group_by_for_tile({"chart": "gauge", "value": "CTR", "group_by": ["Channel"]}) == [
        "Channel"
    ]


@pytest.mark.unit
def test_tile_group_by_inference_ignores_metric_output_fields() -> None:
    result = group_by_for_tile(
        {"chart": "scatter", "x": "CTR", "y": "Lift", "color": "Channel"},
    )

    assert result == ["Channel"]


@pytest.mark.unit
def test_treemap_group_by_supports_legacy_x_y_shape() -> None:
    result = group_by_for_tile(
        {
            "chart": "treemap",
            "x": "CustomerSegment",
            "y": "VS_Interactions",
            "color": "Channel",
        }
    )

    assert result == ["CustomerSegment", "Channel"]


@pytest.mark.unit
def test_tile_group_by_inference_ignores_hidden_legacy_line_dimensions() -> None:
    result = group_by_for_tile(
        {
            "chart": "line",
            "group_by": [
                "Day",
                "Channel",
                "CustomerType",
                "Placement",
                "Issue",
                "AppliedModel",
            ],
            "x": "Day",
            "y": "CTR",
            "color": "AppliedModel",
            "facet_row": "Channel",
            "facet_col": "CustomerType",
        },
    )

    assert result == ["CustomerType", "Channel", "AppliedModel"]


@pytest.mark.unit
def test_new_marketing_chart_grouping_and_grain_inference() -> None:
    assert group_by_for_tile(
        {
            "chart": "sankey",
            "source": "FirstChannel",
            "target": "NextChannel",
            "value": "Count",
        }
    ) == ["FirstChannel", "NextChannel"]
    assert group_by_for_tile(
        {
            "chart": "combo",
            "x": "Day",
            "y": "Spend",
            "y2": "Revenue",
            "color": "Channel",
        }
    ) == ["Channel"]
    assert grain_for_tile({"chart": "calendar_heatmap", "date": "Day", "value": "Revenue"}) == (
        "daily"
    )


@pytest.mark.unit
def test_calibration_curve_group_by_ignores_stale_axes_but_preserves_facets() -> None:
    result = group_by_for_tile(
        {
            "chart": "calibration_curve",
            "x": "placement_type",
            "y": "MIL_Calibration",
            "color": "model_name",
            "facet_row": "region",
            "facet_column": "placement_type",
        }
    )

    assert result == ["placement_type", "region", "model_name"]


@pytest.mark.unit
def test_curve_tile_requests_curve_columns_and_color_grouping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _curve_catalog()
    tile = catalog.dashboards.dashboards[0].pages[0].tiles[0]
    captured: dict[str, object] = {}

    def fake_query_metric(*args: object, **kwargs: object) -> pl.DataFrame:
        captured["group_by"] = kwargs["group_by"]
        captured["include_curve_columns"] = kwargs["include_curve_columns"]
        return pl.DataFrame(
            {
                "placement_type": ["Hero"],
                "ROC_AUC": [0.9],
                "roc_auc": [0.9],
                "fpr": [[0.0, 1.0]],
                "tpr": [[0.0, 1.0]],
                "precision": [[1.0, 0.5]],
                "recall": [[0.0, 1.0]],
                "pos_fraction": [0.2],
            }
        )

    monkeypatch.setattr(ui_data, "query_metric", fake_query_metric)

    query_tile("workspace", catalog, tile)

    assert captured["group_by"] == ["placement_type"]
    assert captured["include_curve_columns"] is True


@pytest.mark.unit
def test_query_tile_requests_state_columns_for_selected_metric_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog(["Channel"])
    tile = model.Tile.model_validate(
        {
            "id": "ctr_scatter",
            "title": "CTR Scatter",
            "metric": "CTR",
            "chart": "scatter",
            "x": "CTR",
            "y": "CTR",
            "size": "Count",
            "color": "Channel",
        }
    )
    captured: dict[str, object] = {}

    def fake_query_metric(*args: object, **kwargs: object) -> pl.DataFrame:
        captured["include_state_columns"] = kwargs["include_state_columns"]
        return pl.DataFrame({"Channel": ["Web"], "CTR": [0.2], "Count": [10]})

    monkeypatch.setattr(ui_data, "query_metric", fake_query_metric)

    query_tile("workspace", catalog, tile)

    assert captured["include_state_columns"] is True


@pytest.mark.unit
def test_query_tile_requests_state_columns_for_descriptive_charts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog(["Channel"])
    tile = model.Tile.model_validate(
        {
            "id": "desc_line",
            "title": "Descriptive",
            "metric": "CTR",
            "chart": "descriptive_line",
            "x": "Month",
            "property": "Propensity",
            "score": "p50",
            "color": "Channel",
        }
    )
    captured: dict[str, object] = {}

    def fake_query_metric(*args: object, **kwargs: object) -> pl.DataFrame:
        captured["include_state_columns"] = kwargs["include_state_columns"]
        return pl.DataFrame({"Month": ["2026-01"], "Channel": ["Web"], "Propensity_Mean": [0.2]})

    monkeypatch.setattr(ui_data, "query_metric", fake_query_metric)

    query_tile("workspace", catalog, tile)

    assert captured["include_state_columns"] is True


@pytest.mark.unit
def test_filter_columns_include_hidden_legacy_dimensions() -> None:
    result = filter_columns_for_tile(
        {
            "chart": "line",
            "group_by": ["Day", "Channel", "Placement", "AppliedModel"],
            "x": "Day",
            "y": "CTR",
            "color": "AppliedModel",
            "facet_row": "Channel",
        },
    )

    assert result == ["Channel", "Placement", "AppliedModel"]


@pytest.mark.unit
def test_page_filter_columns_use_processor_group_by_columns() -> None:
    catalog = _catalog(["Channel", "Placement"])
    page = catalog.dashboards.dashboards[0].pages[0]

    result = available_filter_columns_for_page(catalog, page)

    assert result == ["Channel", "Placement"]


@pytest.mark.unit
def test_explicit_filter_capabilities_and_tile_partition_are_transparent() -> None:
    catalog = _catalog(["Channel", "Placement"])
    page = model.DashboardPage.model_validate(
        {
            **catalog.dashboards.dashboards[0].pages[0].model_dump(mode="python"),
            "filters": [
                {
                    "field": "Channel",
                    "label": "Business channel",
                    "display": "primary",
                    "scope": "all_tiles",
                    "control": "multiselect",
                }
            ],
        }
    )

    capabilities = filter_capabilities_for_page(catalog, page)
    applied, ignored = partition_filters_for_tile(
        catalog,
        page.tiles[0],
        {"Channel": ["Web"], "Unknown": ["x"]},
    )

    assert capabilities[0].label == "Business channel"
    assert capabilities[0].applies_to_all
    assert applied == {"Channel": ["Web"]}
    assert ignored == ("Unknown",)


@pytest.mark.unit
def test_presentation_resolver_applies_metric_display_defaults() -> None:
    catalog = _catalog(["Channel"])
    metric = catalog.metrics.metrics["CTR"]
    metric.display = model.MetricDisplaySpec(
        label="Engagement rate",
        unit="percent",
        value_format="percent",
        direction="higher_is_better",
    )

    resolved = resolve_tile_presentation(
        catalog,
        catalog.dashboards.dashboards[0].pages[0].tiles[0],
    )

    assert resolved["labels"]["CTR"] == "Engagement rate"
    assert resolved["value_format"] == "percent"
    assert resolved["direction"] == "higher_is_better"


@pytest.mark.unit
def test_query_tile_applies_only_supported_page_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _catalog(["Channel"])
    tile = catalog.dashboards.dashboards[0].pages[0].tiles[0]
    captured: dict[str, object] = {}

    def fake_query_metric(*args: object, **kwargs: object) -> pl.DataFrame:
        captured["filters"] = kwargs["filters"]
        return pl.DataFrame({"Day": ["2024-01-01"], "Channel": ["Web"], "CTR": [0.2]})

    monkeypatch.setattr(ui_data, "query_metric", fake_query_metric)

    query_tile(
        "workspace",
        catalog,
        tile,
        filters={"Channel": ["Web"], "Plan": ["Basic"]},
    )

    assert captured["filters"] == {"Channel": ["Web"]}


@pytest.mark.unit
def test_query_tile_caches_identical_metric_queries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = _catalog(["Channel"])
    tile = catalog.dashboards.dashboards[0].pages[0].tiles[0]
    calls: list[dict[str, object]] = []

    def fake_query_metric(*args: object, **kwargs: object) -> pl.DataFrame:
        calls.append(dict(kwargs))
        return pl.DataFrame({"Day": ["2024-01-01"], "Channel": ["Web"], "CTR": [0.2]})

    monkeypatch.setattr(ui_data, "query_metric", fake_query_metric)

    query_tile(tmp_path, catalog, tile, filters={"Channel": ["Web"]})
    query_tile(tmp_path, catalog, tile, filters={"Channel": ["Web"]})
    query_tile(tmp_path, catalog, tile, filters={"Channel": ["Mobile"]})

    assert [call["filters"] for call in calls] == [
        {"Channel": ["Web"]},
        {"Channel": ["Mobile"]},
    ]


@pytest.mark.unit
def test_query_tile_restores_readable_dimension_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog(["Channel", "PlacementType", "Issue", "Group"])
    tile = model.Tile.model_validate(
        {
            "id": "ctr_treemap",
            "title": "CTR treemap",
            "metric": "CTR",
            "chart": "treemap",
            "path": ["channel", "placement", "issue", "group"],
            "color": "CTR",
        }
    )
    captured: dict[str, object] = {}

    def fake_query_metric(*args: object, **kwargs: object) -> pl.DataFrame:
        captured["group_by"] = kwargs["group_by"]
        captured["filters"] = kwargs["filters"]
        return pl.DataFrame(
            {
                "Channel": ["Web"],
                "PlacementType": ["Hero"],
                "Issue": ["Retention"],
                "Group": ["Test"],
                "CTR": [0.2],
            }
        )

    monkeypatch.setattr(ui_data, "query_metric", fake_query_metric)

    rows = query_tile(
        "workspace",
        catalog,
        tile,
        filters={"channel": ["Web"], "placement": ["Hero"]},
    )

    assert captured["group_by"] == ["Channel", "PlacementType", "Issue", "Group"]
    assert captured["filters"] == {"Channel": ["Web"], "PlacementType": ["Hero"]}
    assert rows["channel"].to_list() == ["Web"]
    assert rows["placement"].to_list() == ["Hero"]
    assert rows["issue"].to_list() == ["Retention"]
    assert rows["group"].to_list() == ["Test"]


@pytest.mark.unit
def test_query_tile_uses_time_facet_to_select_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog(["group_viz"])
    tile = model.Tile.model_validate(
        {
            "id": "ctr_by_dimension",
            "title": "CTR By Dimension",
            "metric": "CTR",
            "chart": "bar",
            "x": "group_viz",
            "y": "CTR",
            "facet_col": "Quarter",
        }
    )
    captured: dict[str, object] = {}

    def fake_query_metric(*args: object, **kwargs: object) -> pl.DataFrame:
        captured["grain"] = kwargs["grain"]
        captured["group_by"] = kwargs["group_by"]
        return pl.DataFrame(
            {
                "Quarter": ["2026_Q2"],
                "group_viz": ["A"],
                "CTR": [0.2],
            }
        )

    monkeypatch.setattr(ui_data, "query_metric", fake_query_metric)

    rows = query_tile("workspace", catalog, tile)

    assert captured["grain"] == "quarterly"
    assert captured["group_by"] == ["group_viz"]
    assert rows["Quarter"].to_list() == ["2026_Q2"]


@pytest.mark.unit
def test_parse_filter_text() -> None:
    assert parse_filter_text("Channel=Web,Mobile\nPlan=Basic") == {
        "Channel": ["Web", "Mobile"],
        "Plan": "Basic",
    }


@pytest.mark.unit
def test_restore_time_columns_adds_requested_lowercase_alias() -> None:
    rows = pl.DataFrame({"Day": ["2024-01-01"], "CTR": [0.2]})

    restored = _restore_time_columns(rows, {"x": "day"})

    assert restored["day"].to_list() == ["2024-01-01"]


@pytest.mark.unit
def test_percent_display_columns_use_tile_value_fields() -> None:
    rows = pl.DataFrame({"placement_type": ["A"], "MIL_ROC_AUC": [0.8827]})

    columns = _tile_value_columns(
        {"metric": "MIL_ROC_AUC", "chart": "line", "x": "placement_type", "y": "MIL_ROC_AUC"},
        rows,
    )

    assert columns == ["MIL_ROC_AUC"]


@pytest.mark.unit
def test_report_chart_height_respects_taller_figure_layout_heights() -> None:
    default_figure = Figure()
    tall_figure = Figure()
    tall_figure.update_layout(height=1200)

    assert _report_chart_height(default_figure) == REPORT_CHART_HEIGHT_FALLBACK_PX
    assert _report_chart_height(tall_figure) == 1200


@pytest.mark.unit
def test_report_chart_height_has_hero_profile() -> None:
    figure = Figure()

    assert _report_chart_height(figure, layout_mode=TILE_LAYOUT_HERO) == REPORT_CHART_HEIGHT_HERO_PX


@pytest.mark.unit
def test_report_filter_chip_labels_summarize_active_context() -> None:
    labels = _filter_chip_labels(
        {"Channel": ["Web", "Mobile", "Email", "Branch"], "Plan": "Premium"},
        dt.date(2026, 5, 1),
        dt.date(2026, 5, 31),
        time_preset="last_30_days",
    )

    assert "Last 30 days · May 1\N{EN DASH}31, 2026" in labels
    assert "Channel: Web, Mobile, Email +1" in labels
    assert "Plan: Premium" in labels


@pytest.mark.unit
def test_report_filter_chip_deselection_clears_filter_state() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 — test-only dependency

    def app() -> None:
        # AppTest.from_function reruns the function source as a standalone
        # script, so the imports must live inside it.
        import streamlit as st  # noqa: PLC0415

        from valuestream.ui.pages import reports  # noqa: PLC0415

        class FakePage:
            id = "exec_summary"

        filters: dict[str, object] = {}
        selected = st.multiselect(
            "Channel",
            ["Web", "Mobile", "Email"],
            key="reports_filter_exec_summary_Channel",
        )
        if selected:
            filters["Channel"] = selected
        reports._filter_chips(FakePage(), filters, None, None)

    at = AppTest.from_function(app)
    at.run()
    assert [caption.value for caption in at.caption] == ["All time · no filters"]

    at.multiselect[0].set_value(["Web"]).run()
    chips = at.get("button_group")[0]
    assert chips.value == ["Channel: Web"]

    chips.set_value([]).run()
    assert at.multiselect[0].value == []
    assert not at.get("button_group")
    assert [caption.value for caption in at.caption] == ["All time · no filters"]
    assert not at.exception


@pytest.mark.unit
def test_report_time_chip_names_preset_and_clears_via_callback() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 — test-only dependency

    def app() -> None:
        import datetime as dt  # noqa: PLC0415

        import streamlit as st  # noqa: PLC0415

        from valuestream.ui.pages import reports  # noqa: PLC0415

        class FakeTimeFilter:
            default = "last_90_days"
            presets = ("last_90_days", "all_time")

        class FakePage:
            id = "exec_summary"
            time_filter = FakeTimeFilter()

        class OtherPage:
            id = "conversion"
            time_filter = FakeTimeFilter()

        key = "reports_time_preset_exec_summary"
        other_key = "reports_time_preset_conversion"
        st.session_state.setdefault(key, "last_90_days")
        st.session_state.setdefault(other_key, "last_90_days")
        preset = st.segmented_control(
            "Time range",
            FakePage.time_filter.presets,
            key=key,
            format_func=reports._time_preset_label,
        )
        if preset == "last_90_days":
            reports._filter_chips(
                FakePage(),
                {},
                dt.date(2026, 4, 16),
                dt.date(2026, 7, 14),
                report_pages=(FakePage(), OtherPage()),
            )
        else:
            reports._filter_chips(
                FakePage(),
                {},
                None,
                None,
                report_pages=(FakePage(), OtherPage()),
            )

    at = AppTest.from_function(app)
    at.run()
    assert at.button[0].label == "Last 90 days · Apr 16\N{EN DASH}Jul 14, 2026"

    at.button[0].click().run()

    assert at.get("button_group")[0].value == "all_time"
    assert at.session_state["reports_time_preset_conversion"] == "all_time"
    assert [caption.value for caption in at.caption] == ["All time · no filters"]
    assert not at.exception


@pytest.mark.unit
def test_report_status_banner_is_silent_when_page_is_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    infos: list[str] = []
    warnings: list[str] = []
    monkeypatch.setattr(st, "info", lambda message, *_, **__: infos.append(str(message)))
    monkeypatch.setattr(st, "warning", lambda message, *_, **__: warnings.append(str(message)))
    page = model.DashboardPage.model_validate(
        {
            "id": "funnel_report_type_coverage",
            "title": "Funnel Report Type Coverage",
            "tiles": [
                {
                    "id": "coverage",
                    "title": "Coverage",
                    "metric": "CTR",
                    "chart": "bar",
                }
            ],
        }
    )
    fresh = Freshness(
        latest_period="2026-04",
        last_created_at=None,
        last_run_finished_at=dt.datetime(2026, 6, 17, 9, 29),
        status="ok",
    )

    _page_status_banner(
        SimpleNamespace(title="Overview"),
        page,
        [fresh],
        filters={},
        start=None,
        end=None,
        view_mode="Presentation",
    )

    assert infos == []
    assert warnings == []


@pytest.mark.unit
def test_report_status_banner_warns_when_tile_data_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    infos: list[str] = []
    warnings: list[str] = []
    monkeypatch.setattr(st, "info", lambda message, *_, **__: infos.append(str(message)))
    monkeypatch.setattr(st, "warning", lambda message, *_, **__: warnings.append(str(message)))
    page = model.DashboardPage.model_validate(
        {
            "id": "business_overview",
            "title": "Business Overview",
            "tiles": [],
        }
    )

    _page_status_banner(
        SimpleNamespace(title="Overview"),
        page,
        [],
        filters={},
        start=None,
        end=None,
        view_mode="Presentation",
    )

    assert infos == []
    assert warnings == [
        "Business report | 0 tile(s) | presentation view | all time. No tile data found."
    ]


@pytest.mark.unit
def test_report_rows_csv_serializes_binary_and_nested_columns() -> None:
    rows = pl.DataFrame(
        {
            "label": ["curve"],
            "payload": pl.Series([b"\x00\xff"], dtype=pl.Binary),
            "points": [[0.1, 0.9]],
            "meta": [{"kind": "roc", "blob": b"\x01"}],
        }
    )

    rendered = _rows_csv(rows)
    parsed = next(csv.DictReader(StringIO(rendered)))

    assert parsed["label"] == "curve"
    assert parsed["payload"] == "base64:AP8="
    assert parsed["points"] == "[0.1, 0.9]"
    assert parsed["meta"] == '{"blob": "base64:AQ==", "kind": "roc"}'


@pytest.mark.unit
def test_report_page_help_text_carries_hidden_header_metadata() -> None:
    catalog = _catalog(["Channel"])
    dashboard = catalog.dashboards.dashboards[0]
    page = dashboard.pages[0]
    fresh = Freshness(
        latest_period="2026-04",
        last_created_at=None,
        last_run_finished_at=dt.datetime(2026, 6, 17, 9, 29),
        status="ok",
    )

    help_text = _page_help_text(dashboard, page, fresh)

    assert "Dashboard: Overview" in help_text
    assert "Tiles: 1" in help_text
    assert "Run status: ok" in help_text
    assert "Latest aggregate: 2026-04" in help_text
    assert "Last run: 2026-06-17 09:29" in help_text


@pytest.mark.unit
def test_metric_freshness_uses_query_grain_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _daily_only_catalog()
    processor = catalog.processors.processors[0]
    config_hash = processor_computation_hash(catalog, processor)
    run_id = "00000000-0000-0000-0000-000000000001"
    chunk_id = "2024-09-18"
    finished_at = dt.datetime(2026, 6, 25, 15, 23)
    source_file = tmp_path / "ih_20240918.parquet"
    source_file.write_text("placeholder", encoding="utf-8")

    insert_run(
        tmp_path,
        run_id=run_id,
        workspace="test",
        source_id="ih",
        config_hash=config_hash,
        started_at=finished_at - dt.timedelta(minutes=5),
        finished_at=finished_at,
        status="ok",
        rows_in=10,
        rows_kept=10,
        chunks_total=1,
        chunks_ok=1,
        chunks_failed=0,
    )
    insert_chunk(
        tmp_path,
        source_id="ih",
        chunk_id=chunk_id,
        files=[source_file],
        rows_in=10,
        rows_kept=10,
        started_at=finished_at - dt.timedelta(minutes=5),
        finished_at=finished_at,
        status="ok",
        error=None,
        pipeline_run_id=run_id,
    )
    write_aggregate(
        pl.DataFrame(
            {
                "Group": ["Alerts"],
                "Count": [10],
                "Positives": [2],
                "period": ["2024-09"],
                "created_at": [finished_at],
                "config_hash": [config_hash],
                "pipeline_run_id": [run_id],
                "chunk_id": [chunk_id],
            }
        ),
        tmp_path,
        source_id="ih",
        processor_id="engagement",
        grain="daily",
        run_id=run_id,
        chunk_id=chunk_id,
    )
    original_collect_all = pl.collect_all
    collect_all_calls = 0

    def counted_collect_all(lazy_frames: list[pl.LazyFrame]) -> list[pl.DataFrame]:
        nonlocal collect_all_calls
        collect_all_calls += 1
        return original_collect_all(lazy_frames)

    monkeypatch.setattr(freshness_module.pl, "collect_all", counted_collect_all)

    fresh = metric_freshness(tmp_path, catalog, "CTR", grain="summary")

    assert collect_all_calls == 1
    assert fresh.latest_period == "2024-09"
    assert fresh.last_created_at == finished_at
    assert fresh.last_run_finished_at == finished_at
    assert fresh.status == "ok"


@pytest.mark.unit
def test_report_tile_help_text_carries_hidden_tile_metadata() -> None:
    tile = {
        "id": "ctr",
        "title": "CTR",
        "metric": "CTR",
        "chart": "bar",
    }
    fresh = Freshness(
        latest_period=None,
        last_created_at=None,
        last_run_finished_at=None,
        status="not run",
    )

    help_text = _tile_help_text(tile, tile, "summary", fresh)

    assert "Metric: CTR" in help_text
    assert "Chart: bar" in help_text
    assert "Grain: summary" in help_text
    assert "Tile freshness: stale" in help_text
    assert "Run status: not run" in help_text


@pytest.mark.unit
def test_report_coverage_pages_are_classified_as_non_business_surfaces() -> None:
    page = model.DashboardPage.model_validate(
        {"id": "numeric_report_type_coverage", "title": "Numeric Report Type Coverage", "tiles": []}
    )
    business_page = model.DashboardPage.model_validate(
        {"id": "funnel_and_response", "title": "Funnel and Response", "tiles": []}
    )

    assert _is_coverage_page(page)
    assert not _is_coverage_page(business_page)


@pytest.mark.unit
def test_descriptive_report_pages_default_to_advanced_mode() -> None:
    descriptive_page = model.DashboardPage.model_validate(
        {
            "id": "descriptive",
            "title": "Descriptive",
            "tiles": [
                {
                    "id": "quartiles",
                    "title": "Quartiles",
                    "metric": "ResponseP50",
                    "chart": "descriptive_boxplot",
                    "property": "ResponseTime",
                }
            ],
        }
    )
    business_page = model.DashboardPage.model_validate(
        {
            "id": "business",
            "title": "Business",
            "tiles": [
                {
                    "id": "daily_ctr",
                    "title": "Daily CTR",
                    "metric": "CTR",
                    "chart": "line",
                    "x": "Day",
                    "y": "CTR",
                }
            ],
        }
    )

    assert _is_descriptive_report_page(descriptive_page)
    assert not _is_descriptive_report_page(business_page)


@pytest.mark.unit
def test_report_grid_promotes_faceted_tiles_to_full_width() -> None:
    faceted = model.Tile.model_validate(
        {
            "id": "daily_ctr",
            "title": "Daily CTR",
            "metric": "CTR",
            "chart": "line",
            "x": "Day",
            "y": "CTR",
            "facets": {"row": "channel", "col": "placement"},
        }
    )
    simple = model.Tile.model_validate(
        {
            "id": "daily_ctr",
            "title": "Daily CTR",
            "metric": "CTR",
            "chart": "line",
            "x": "Day",
            "y": "CTR",
        }
    )
    color_series = model.Tile.model_validate(
        {
            "id": "daily_ctr_by_channel",
            "title": "Daily CTR by channel",
            "metric": "CTR",
            "chart": "line",
            "x": "Day",
            "y": "CTR",
            "color": "Channel",
        }
    )

    assert _is_full_width_tile(faceted)
    assert _is_full_width_tile(color_series)
    assert not _is_full_width_tile(simple)


@pytest.mark.unit
def test_grouped_gauge_tiles_are_chart_grid_tiles() -> None:
    grouped = model.Tile.model_validate(
        {
            "id": "ctr_gauge",
            "title": "CTR Gauge",
            "metric": "CTR",
            "chart": "gauge",
            "value": "CTR",
            "facet_row": "Channel",
        }
    )
    simple = model.Tile.model_validate(
        {
            "id": "ctr_gauge",
            "title": "CTR Gauge",
            "metric": "CTR",
            "chart": "gauge",
            "value": "CTR",
        }
    )

    assert _has_grouped_gauge_tile(grouped)
    assert _is_full_width_tile(grouped)
    assert not _has_grouped_gauge_tile(simple)


@pytest.mark.unit
def test_derived_kpi_strip_is_hidden_for_single_report_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = model.DashboardPage.model_validate(
        {
            "id": "single",
            "title": "Single",
            "tiles": [
                {
                    "id": "novelty",
                    "title": "Novelty",
                    "metric": "Novelty",
                    "chart": "treemap",
                    "value": "Novelty",
                }
            ],
        }
    )
    rendered: list[list[object]] = []

    def fail_query(*_: object, **__: object) -> pl.DataFrame:
        raise AssertionError("single report pages should not query derived KPI strip values")

    monkeypatch.setattr("valuestream.ui.pages.reports.query_tile", fail_query)
    monkeypatch.setattr(
        "valuestream.ui.pages.reports.components.metric_strip",
        lambda items, **_: rendered.append(list(items)),
    )

    # ``_kpi_strip`` is an ``st.fragment``; call the wrapped body directly in
    # bare test mode (the fragment wrapper is a no-op without a Streamlit run).
    _kpi_strip.__wrapped__(
        SimpleNamespace(workspace=Path("."), catalog=None),
        page,
        filters={},
        start=None,
        end=None,
    )

    assert rendered == []


@pytest.mark.unit
def test_explicit_kpi_strip_is_visible_for_authored_kpi_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = model.DashboardPage.model_validate(
        {
            "id": "multi",
            "title": "Multi",
            "tiles": [
                {
                    "id": "ctr",
                    "title": "CTR",
                    "metric": "CTR",
                    "chart": "kpi_card",
                    "value": "CTR",
                    "placement": "kpi_strip",
                },
                {
                    "id": "ctr_secondary",
                    "title": "Secondary CTR",
                    "metric": "CTR",
                    "chart": "kpi_card",
                    "value": "CTR",
                    "placement": "kpi_strip",
                },
            ],
        }
    )
    rendered: list[list[object]] = []

    def fake_query_metric(*_: object, **__: object) -> pl.DataFrame:
        return pl.DataFrame({"CTR": [0.3]})

    monkeypatch.setattr("valuestream.ui.pages.reports.query_metric_cached", fake_query_metric)
    monkeypatch.setattr(
        "valuestream.ui.pages.reports.components.metric_strip",
        lambda items, **_: rendered.append(list(items)),
    )

    _kpi_strip.__wrapped__(
        SimpleNamespace(workspace=Path("."), catalog=_catalog([])),
        page,
        filters={},
        start=None,
        end=None,
    )

    assert len(rendered) == 1
    assert len(rendered[0]) == 2


@pytest.mark.unit
def test_kpi_strip_surfaces_backfill_required_without_error_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = model.DashboardPage.model_validate(
        {
            "id": "multi",
            "title": "Multi",
            "tiles": [
                {
                    "id": "unique_channels",
                    "title": "Unique channels",
                    "metric": "CTR",
                    "chart": "kpi_card",
                    "value": "CTR",
                    "placement": "kpi_strip",
                }
            ],
        }
    )
    rendered: list[list[object]] = []

    def not_ready(*_: object, **__: object) -> pl.DataFrame:
        raise AggregateNotReadyError("run ingestion or backfill/reprocess existing data")

    monkeypatch.setattr("valuestream.ui.pages.reports.query_metric_cached", not_ready)
    monkeypatch.setattr(
        "valuestream.ui.pages.reports.components.metric_strip",
        lambda items, **_: rendered.append(list(items)),
    )

    _kpi_strip.__wrapped__(
        SimpleNamespace(workspace=Path("."), catalog=_catalog([])),
        page,
        filters={},
        start=None,
        end=None,
    )

    assert len(rendered) == 1
    item = rendered[0][0]
    assert item.value == "not ready"
    assert item.help == "run ingestion or backfill/reprocess existing data"


@pytest.mark.unit
def test_kpi_bundle_uses_complete_latest_period_and_equal_previous_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog(["Channel"])
    tile = model.Tile.model_validate(
        {
            "id": "ctr_kpi",
            "title": "CTR",
            "metric": "CTR",
            "chart": "kpi_card",
            "value": "CTR",
            "placement": "kpi_strip",
            "kpi": {
                "comparison": "previous_period",
                "comparison_period": "month",
                "sparkline_grain": "daily",
                "sparkline_points": 3,
            },
        }
    )
    calls: list[tuple[str, dt.date | None, dt.date | None]] = []

    def fake_query_metric(*_: object, **kwargs: object) -> pl.DataFrame:
        grain = str(kwargs["grain"])
        start = kwargs.get("start")
        end = kwargs.get("end")
        calls.append((grain, start, end))
        if grain == "daily":
            return pl.DataFrame(
                {
                    "Day": ["2026-05-13", "2026-05-14", "2026-05-15"],
                    "CTR": [0.2, 0.25, 0.3],
                }
            )
        return pl.DataFrame({"CTR": [0.3 if start == dt.date(2026, 5, 1) else 0.2]})

    monkeypatch.setattr("valuestream.ui.pages.reports.query_metric_cached", fake_query_metric)

    bundle = _kpi_bundle(
        SimpleNamespace(workspace=Path("."), catalog=catalog),
        tile,
        filters={},
        start=None,
        end=None,
    )

    assert bundle.value == pytest.approx(0.3)
    assert bundle.delta == pytest.approx(0.1)
    assert bundle.sparkline == pytest.approx((0.2, 0.25, 0.3))
    assert bundle.period_description == "May 2026"
    assert ("summary", dt.date(2026, 5, 1), dt.date(2026, 5, 31)) in calls
    assert ("summary", dt.date(2026, 3, 31), dt.date(2026, 4, 30)) in calls


@pytest.mark.unit
def test_advanced_tile_from_fields_keeps_metric_locked_and_rebuilds_chart_fields() -> None:
    base_tile = {
        "id": "daily_ctr",
        "title": "Daily CTR",
        "metric": "CTR",
        "chart": "line",
        "x": "Day",
        "y": "CTR",
        "color": "Channel",
        "filters": {"Channel": ["Web"]},
        "facets": {"row": "Placement"},
    }

    draft = _advanced_tile_from_fields(
        base_tile,
        "bar",
        {
            "x": "Channel",
            "y": "CTR",
            "color": "",
            "facet_row": "Placement",
        },
    )

    assert draft["metric"] == "CTR"
    assert draft["chart"] == "bar"
    assert draft["x"] == "Channel"
    assert draft["y"] == "CTR"
    assert draft["facet_row"] == "Placement"
    assert draft["filters"] == {"Channel": ["Web"]}
    assert "color" not in draft
    assert "facets" not in draft


@pytest.mark.unit
def test_advanced_tile_seed_merges_configured_and_current_session_values() -> None:
    catalog = _catalog(["Channel", "Placement"])
    base_tile = {
        "id": "daily_ctr",
        "title": "Daily CTR",
        "metric": "CTR",
        "chart": "line",
        "x": "Day",
        "y": "CTR",
        "facets": {"row": "Placement", "col": "Channel"},
        "value_format": "percent",
    }
    current_tile = {
        **base_tile,
        "chart": "bar",
        "x": "Channel",
        "color": "Placement",
        "top_n": 8,
    }

    seed = _advanced_tile_seed(catalog, base_tile, current_tile, "bar")

    assert seed["x"] == "Channel"
    assert seed["y"] == "CTR"
    assert seed["color"] == "Placement"
    assert seed["facet_row"] == "Placement"
    assert seed["facet_col"] == "Channel"
    assert seed["value_format"] == "percent"
    assert seed["top_n"] == 8


@pytest.mark.unit
def test_reports_advanced_style_controls_include_gauge_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(st, "selectbox", lambda _label, options, *, index=0, **__: options[index])
    monkeypatch.setattr(st, "checkbox", lambda label, **__: label == "Reference")
    monkeypatch.setattr(st, "number_input", lambda _label, *, value=0.0, **__: value)

    draft = _advanced_style_controls({"chart": "gauge", "reference": 0.12}, "gauge")

    assert draft["reference"] == 0.12


@pytest.mark.unit
def test_reports_heatmap_style_controls_keep_default_color_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selectbox_calls: list[tuple[str, list[str], int]] = []

    def fake_selectbox(
        label: str,
        options: list[str],
        *,
        index: int = 0,
        **_: object,
    ) -> str:
        selectbox_calls.append((label, options, index))
        return options[index]

    monkeypatch.setattr(st, "selectbox", fake_selectbox)
    monkeypatch.setattr(st, "checkbox", lambda *_args, **_kwargs: False)

    draft = _advanced_style_controls({"chart": "descriptive_heatmap"}, "heat")

    assert "color_continuous_scale" not in draft
    assert ADVANCED_COLOR_SCALES[0] == ""
    assert ("Color Scale", ADVANCED_COLOR_SCALES, 0) in selectbox_calls


@pytest.mark.unit
def test_reports_advanced_editor_exposes_polar_radius_field() -> None:
    catalog = _catalog(["Channel", "Placement"])
    base_tile = {
        "id": "daily_ctr",
        "title": "Daily CTR",
        "metric": "CTR",
        "chart": "line",
        "x": "Day",
        "y": "CTR",
    }

    seed = _advanced_tile_seed(catalog, base_tile, base_tile, "bar_polar")

    assert ADVANCED_FIELD_CONTROLS["bar_polar"] == ("r", "theta", "color")
    assert seed["r"] == "CTR"
    assert seed["theta"] == "Channel"
    assert seed["color"] == "Placement"
    assert "size" not in ADVANCED_FIELD_CONTROLS["line"]
    assert "animation_frame" in ADVANCED_FIELD_CONTROLS["scatter"]
    assert "animation_group" in ADVANCED_FIELD_CONTROLS["scatter"]
    assert "size" in ADVANCED_FIELD_CONTROLS["scatter"]
    assert "size" in ADVANCED_FIELD_CONTROLS["geo_map"]
    assert ADVANCED_FIELD_CONTROLS["gauge"] == ("value", "facet_row", "facet_col")


@pytest.mark.unit
def test_load_context_bootstraps_missing_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "AI Config Drafts"

    ctx = load_context(workspace)

    assert ctx.workspace == workspace.resolve()
    assert ctx.catalog.pipelines.workspace == "ai_config_drafts"
    assert ctx.validation.ok
    assert (workspace / "catalog" / "pipelines.yaml").is_file()


@pytest.mark.unit
def test_source_field_options_include_processor_extra_columns(tmp_path: Path) -> None:
    catalog = _catalog_with_processor_extras()
    ctx = ValueStreamContext(
        workspace=tmp_path,
        catalog=catalog,
        validation=CatalogValidationResult(ok=True),
        catalog_hash="hash",
    )

    fields = _source_field_options(ctx, catalog.pipelines.sources[0])

    assert "Outcome" in fields
    assert "final_propensity" in fields


@pytest.mark.unit
def test_source_field_options_include_discovered_csv_columns(tmp_path: Path) -> None:
    (tmp_path / "sample.csv").write_text(
        "interaction_id,Outcome,final_propensity\n1,1,0.9\n",
        encoding="utf-8",
    )
    catalog = _catalog_with_raw_csv_source()
    ctx = ValueStreamContext(
        workspace=tmp_path,
        catalog=catalog,
        validation=CatalogValidationResult(ok=True),
        catalog_hash="hash",
    )

    fields = _source_field_options(ctx, catalog.pipelines.sources[0])

    assert "Outcome" in fields
    assert "final_propensity" in fields


@pytest.mark.unit
def test_new_processor_template_uses_fresh_id_and_known_outcome(tmp_path: Path) -> None:
    catalog = _catalog_with_processor_extras()
    ctx = ValueStreamContext(
        workspace=tmp_path,
        catalog=catalog,
        validation=CatalogValidationResult(ok=True),
        catalog_hash="hash",
    )

    processor = _new_processor_template(ctx)

    assert processor.id == "ih_processor"
    assert processor.source == "ih"
    assert processor.kind == "binary_outcome"
    assert processor.model_extra["outcome"]["column"] == "Outcome"


@pytest.mark.unit
def test_state_rows_explain_binary_outcome_derivations() -> None:
    processor = model.BinaryOutcomeProcessor.model_validate(
        {
            "id": "engagement",
            "source": "ih",
            "kind": "binary_outcome",
            "outcome": {
                "column": "Outcome",
                "positive_values": [1],
                "negative_values": [0],
            },
        }
    )

    rows = {row["State"]: row for row in _state_rows(processor)}

    assert rows["Count"]["Derived From"] == "Outcome in [1, 0]"
    assert rows["Positives"]["Derived From"] == "Outcome in [1]"
    assert rows["Negatives"]["Derived From"] == "Outcome in [0]"


@pytest.mark.unit
def test_ai_studio_natural_key_uses_raw_subject_source_column() -> None:
    assert _natural_key_fields("SubjectID", {"SubjectID": "interaction_id"}) == ["interaction_id"]
    assert _natural_key_fields("SubjectID", {"SubjectID": "SubjectID"}) == ["SubjectID"]


def _catalog(group_by: list[str]) -> model.Catalog:
    return model.Catalog.model_validate(
        {
            "pipelines": {
                "workspace": "test",
                "sources": [
                    {
                        "id": "ih",
                        "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                    }
                ],
            },
            "processors": {
                "processors": [
                    {
                        "id": "engagement",
                        "source": "ih",
                        "kind": "binary_outcome",
                        "group_by": group_by,
                        "states": {
                            "Count": {"type": "count"},
                            "Positives": {"type": "count"},
                        },
                    }
                ]
            },
            "metrics": {
                "metrics": {
                    "CTR": {
                        "source": "engagement",
                        "kind": "formula",
                        "expression": {
                            "op": "safe_div",
                            "num": {"col": "Positives"},
                            "den": {"col": "Count"},
                        },
                    }
                }
            },
            "dashboards": {
                "dashboards": [
                    {
                        "id": "overview",
                        "title": "Overview",
                        "pages": [
                            {
                                "id": "engagement",
                                "title": "Engagement",
                                "tiles": [
                                    {
                                        "id": "ctr",
                                        "title": "CTR",
                                        "metric": "CTR",
                                        "chart": "line",
                                        "x": "Day",
                                        "y": "CTR",
                                        "color": "Channel",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
        }
    )


def _daily_only_catalog() -> model.Catalog:
    return model.Catalog.model_validate(
        {
            "pipelines": {
                "workspace": "test",
                "sources": [
                    {
                        "id": "ih",
                        "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                    }
                ],
            },
            "processors": {
                "processors": [
                    {
                        "id": "engagement",
                        "source": "ih",
                        "kind": "binary_outcome",
                        "group_by": ["Group"],
                        "time": {"column": "OutcomeTime", "grains": ["Day"]},
                        "outcome": {
                            "column": "Outcome",
                            "positive_values": ["Clicked"],
                            "negative_values": ["Impression"],
                        },
                        "states": {
                            "Count": {"type": "count"},
                            "Positives": {"type": "count"},
                        },
                    }
                ]
            },
            "metrics": {
                "metrics": {
                    "CTR": {
                        "source": "engagement",
                        "kind": "formula",
                        "expression": {
                            "op": "safe_div",
                            "num": {"col": "Positives"},
                            "den": {"col": "Count"},
                        },
                    }
                }
            },
            "dashboards": {"dashboards": []},
        }
    )


def _curve_catalog() -> model.Catalog:
    return model.Catalog.model_validate(
        {
            "pipelines": {
                "workspace": "test",
                "sources": [
                    {
                        "id": "ih",
                        "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                    }
                ],
            },
            "processors": {
                "processors": [
                    {
                        "id": "scores",
                        "source": "ih",
                        "kind": "score_distribution",
                        "group_by": ["placement_type"],
                        "score_properties": ["Propensity"],
                        "outcome": {
                            "column": "Outcome",
                            "positive_values": ["Clicked"],
                            "negative_values": ["Impression"],
                        },
                    }
                ]
            },
            "metrics": {
                "metrics": {
                    "ROC_AUC": {
                        "source": "scores",
                        "kind": "curve_from_digests",
                        "positive_state": "Propensity_tdigest_positives",
                        "negative_state": "Propensity_tdigest_negatives",
                        "output": "roc_auc",
                    }
                }
            },
            "dashboards": {
                "dashboards": [
                    {
                        "id": "overview",
                        "title": "Overview",
                        "pages": [
                            {
                                "id": "curves",
                                "title": "Curves",
                                "tiles": [
                                    {
                                        "id": "roc",
                                        "title": "ROC",
                                        "metric": "ROC_AUC",
                                        "chart": "roc_curve",
                                        "color": "placement_type",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
        }
    )


def _catalog_with_processor_extras() -> model.Catalog:
    return model.Catalog.model_validate(
        {
            "pipelines": {
                "workspace": "test",
                "sources": [
                    {
                        "id": "ih",
                        "reader": {"kind": "csv", "file_pattern": "*.csv"},
                        "schema": {
                            "timestamp_column": "OutcomeTime",
                            "natural_key": ["interaction_id"],
                        },
                    }
                ],
            },
            "processors": {
                "processors": [
                    {
                        "id": "ih_engagement",
                        "source": "ih",
                        "kind": "binary_outcome",
                        "group_by": ["group_viz"],
                        "time": {"column": "OutcomeTime", "grains": ["Day", "Summary"]},
                        "states": {
                            "Count": {"type": "count"},
                            "ScoreDigest": {
                                "type": "tdigest",
                                "source_column": "final_propensity",
                            },
                        },
                        "outcome": {
                            "column": "Outcome",
                            "positive_values": [1],
                            "negative_values": [0],
                        },
                    }
                ]
            },
            "metrics": {"metrics": {}},
            "dashboards": {"dashboards": []},
        }
    )


def _catalog_with_raw_csv_source() -> model.Catalog:
    return model.Catalog.model_validate(
        {
            "pipelines": {
                "workspace": "test",
                "sources": [
                    {
                        "id": "ih",
                        "reader": {"kind": "csv", "file_pattern": "*.csv"},
                        "schema": {
                            "timestamp_column": None,
                            "natural_key": ["interaction_id"],
                        },
                    }
                ],
            },
            "processors": {"processors": []},
            "metrics": {"metrics": {}},
            "dashboards": {"dashboards": []},
        }
    )
