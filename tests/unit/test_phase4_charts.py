"""Phase 4 chart factory tests."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import plotly.io as pio  # type: ignore[import-untyped]
import polars as pl
import pytest
from plotly.graph_objects import Figure  # type: ignore[import-untyped]

from valuestream.charts import render_chart
from valuestream.states import tdigest

SUPPORTED_CHART_CASES = [
    ({"chart": "line", "title": "Line", "x": "Day", "y": "CTR", "color": "Channel"}, None),
    (
        {"chart": "stacked_area", "title": "Area", "x": "Day", "y": "CTR", "color": "Channel"},
        None,
    ),
    ({"chart": "bar", "title": "Bar", "x": "Channel", "y": "CTR"}, None),
    ({"chart": "kpi_card", "title": "KPI", "value": "CTR", "reference": 0.25}, None),
    ({"chart": "waterfall", "title": "Waterfall", "x": "Channel", "y": "Revenue"}, "marketing"),
    ({"chart": "pareto", "title": "Pareto", "x": "Campaign", "y": "Revenue"}, "marketing"),
    (
        {"chart": "treemap", "title": "Tree", "path": ["Channel", "Placement"], "color": "CTR"},
        None,
    ),
    (
        {"chart": "heatmap", "title": "Heat", "x": "Channel", "y": "Placement", "color": "CTR"},
        None,
    ),
    (
        {
            "chart": "cohort_heatmap",
            "title": "Cohort",
            "x": "Month",
            "y": "Cohort",
            "color": "Retention",
        },
        "marketing",
    ),
    ({"chart": "scatter", "title": "Scatter", "x": "frequency", "y": "monetary_value"}, None),
    ({"chart": "combo", "title": "Combo", "x": "Day", "y": "Spend", "y2": "Revenue"}, "marketing"),
    (
        {
            "chart": "interval",
            "title": "Interval",
            "x": "Campaign",
            "y": "Lift",
            "error_y": "StdErr",
        },
        "marketing",
    ),
    ({"chart": "donut", "title": "Donut", "names": "Channel", "values": "Revenue"}, "marketing"),
    (
        {
            "chart": "geo_map",
            "title": "Geo",
            "locations": "CountryCode",
            "value": "Revenue",
            "locationmode": "ISO-3",
        },
        "marketing",
    ),
    ({"chart": "table", "title": "Table", "columns": ["Campaign", "Revenue"]}, "marketing"),
    (
        {"chart": "calendar_heatmap", "title": "Calendar", "date": "Day", "value": "Revenue"},
        "marketing",
    ),
    (
        {
            "chart": "bar_polar",
            "title": "Polar",
            "r": "CTR",
            "theta": "Channel",
            "color": "Placement",
        },
        None,
    ),
    (
        {
            "chart": "sankey",
            "title": "Sankey",
            "source": "SourceStage",
            "target": "TargetStage",
            "value": "FlowValue",
        },
        "marketing",
    ),
    ({"chart": "gauge", "title": "Gauge", "value": "CTR", "references": {"Web": 0.4}}, None),
    (
        {
            "chart": "funnel",
            "title": "Funnel",
            "stages": ["Impression", "Clicked", "Conversion"],
            "color": "Channel",
        },
        None,
    ),
    ({"chart": "boxplot", "title": "Box", "x": "Channel", "property": "ResponseTime"}, "box"),
    ({"chart": "histogram", "title": "Hist", "property": "monetary_value"}, None),
    ({"chart": "calibration_curve", "title": "Calibration"}, "calibration"),
    ({"chart": "roc_curve", "title": "ROC", "color": "Channel"}, "curve"),
    ({"chart": "precision_recall_curve", "title": "PR", "color": "Channel"}, "curve"),
    ({"chart": "gain_curve", "title": "Gain", "color": "Channel"}, "curve"),
    ({"chart": "lift_curve", "title": "Lift", "color": "Channel"}, "curve"),
    ({"chart": "rfm_density", "title": "RFM"}, None),
    ({"chart": "exposure", "title": "Exposure"}, None),
    ({"chart": "corr", "title": "Corr"}, None),
    ({"chart": "model", "title": "Model"}, None),
    (
        {
            "chart": "descriptive_line",
            "title": "DLine",
            "x": "Day",
            "property": "ResponseTime",
            "score": "Mean",
        },
        "descriptive",
    ),
    (
        {
            "chart": "descriptive_boxplot",
            "title": "DBox",
            "x": "Channel",
            "property": "ResponseTime",
        },
        "box",
    ),
    (
        {
            "chart": "descriptive_histogram",
            "title": "DHist",
            "property": "ResponseTime",
            "score": "Mean",
        },
        "descriptive",
    ),
    (
        {
            "chart": "descriptive_heatmap",
            "title": "DHeat",
            "x": "Channel",
            "y": "Placement",
            "property": "ResponseTime",
            "score": "Mean",
        },
        "descriptive",
    ),
    (
        {
            "chart": "descriptive_funnel",
            "title": "DFunnel",
            "stages": ["Impression", "Clicked"],
            "color": "Channel",
        },
        None,
    ),
    (
        {"chart": "experiment_z_score", "title": "Z", "x": "z_score", "y": "ExperimentName"},
        "experiment",
    ),
    (
        {
            "chart": "experiment_odds_ratio",
            "title": "OR",
            "x": "g_odds_ratio_stat",
            "y": "ExperimentName",
        },
        "experiment",
    ),
    ({"chart": "clv_treemap", "title": "CLV Tree"}, None),
]


@pytest.mark.unit
@pytest.mark.parametrize(("tile", "frame_name"), SUPPORTED_CHART_CASES)
def test_chart_factory_renders_supported_kind(tile: dict[str, Any], frame_name: str | None) -> None:
    figure = render_chart(_frame(frame_name), {"id": "tile", "metric": "Metric", **tile})

    assert isinstance(figure, Figure)
    assert figure.to_dict()["data"]


@pytest.mark.unit
@pytest.mark.parametrize(("tile", "frame_name"), SUPPORTED_CHART_CASES)
def test_supported_charts_validate_plotly_6_json(
    tile: dict[str, Any], frame_name: str | None
) -> None:
    figure = render_chart(_frame(frame_name), {"id": "tile", "metric": "Metric", **tile})

    assert pio.to_json(figure, validate=True)


@pytest.mark.unit
def test_combo_chart_preserves_secondary_axis_semantics() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Day": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
                "Clicked_Count": [12, 15],
                "Impression_Count": [120, 150],
            }
        ),
        {
            "id": "clicks_and_impressions",
            "metric": "FunnelClicks",
            "chart": "combo",
            "x": "Day",
            "y": "Clicked_Count",
            "y2": "Impression_Count",
            "labels": {
                "Clicked_Count": "Clicked Count",
                "Impression_Count": "Impression Count",
            },
            "y_axis_title": "Clicked Count (clicks)",
            "y2_axis_title": "Impressions",
        },
    )

    assert figure.layout.yaxis.title.text == "Clicked Count (clicks)"
    assert figure.layout.yaxis2.title.text == "Impressions"


@pytest.mark.unit
def test_table_chart_expands_topk_items_into_ranked_rows() -> None:
    rows = pl.DataFrame(
        {
            "Channel": ["Web"],
            "Issue": ["Cards"],
            "Top_Actions": [
                [
                    {
                        "item": "Retention",
                        "estimate": 12,
                        "lower_bound": 11,
                        "upper_bound": 13,
                    },
                    {
                        "item": "CrossSell",
                        "estimate": 7,
                        "lower_bound": 7,
                        "upper_bound": 8,
                    },
                ]
            ],
        }
    )

    figure = render_chart(
        rows,
        {
            "id": "top_actions",
            "metric": "Top_Actions",
            "chart": "table",
            "columns": ["Channel", "Issue", "Top_Actions"],
        },
        theme={"font": {"family": "Inter", "size": 14}},
    )

    table = figure.data[0]
    assert list(table.header.values) == [
        "Channel",
        "Issue",
        "Rank",
        "Top_Actions",
        "Estimate",
        "Lower bound",
        "Upper bound",
    ]
    assert list(table.cells.values[0]) == ["Web", "Web"]
    assert list(table.cells.values[2]) == [1, 2]
    assert list(table.cells.values[3]) == ["Retention", "CrossSell"]
    assert list(table.cells.values[4]) == [12, 7]
    assert list(table.cells.values[5]) == [11, 7]
    assert list(table.cells.values[6]) == [13, 8]
    expected_font = "Inter, DM Sans, Segoe UI, system-ui, sans-serif"
    assert table.header.font.family == expected_font
    assert table.cells.font.family == expected_font
    assert table.header.font.size == 14
    assert figure.layout.height == 140


@pytest.mark.unit
def test_theme_background_overrides_builtin_plotly_template_background() -> None:
    frame = pl.DataFrame(
        {
            "Day": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "Interactions": [120, 140],
        }
    )

    figure = render_chart(
        frame,
        {
            "id": "interactions_trend",
            "metric": "Interactions",
            "chart": "line",
            "title": "Interactions Trend",
            "x": "Day",
            "y": "Interactions",
        },
        theme={
            "template": "plotly_white",
            "paper_bgcolor": "#f5f3ee",
            "plot_bgcolor": "#f5f3ee",
        },
    )

    assert figure.layout.paper_bgcolor == "#f5f3ee"
    assert figure.layout.plot_bgcolor == "#f5f3ee"


@pytest.mark.unit
def test_line_downsampling_caps_large_frames() -> None:
    frame = pl.DataFrame(
        {
            "Day": [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(1_000)],
            "CTR": [float(i % 100) / 100 for i in range(1_000)],
        }
    )

    figure = render_chart(
        frame,
        {"id": "line", "metric": "CTR", "chart": "line", "title": "Line", "x": "Day", "y": "CTR"},
        max_points=100,
    )

    assert len(figure.data[0]["x"]) <= 100


@pytest.mark.unit
def test_line_chart_sorts_rows_by_x_axis() -> None:
    days = [dt.date(2024, 1, day) for day in (3, 1, 2)]
    frame = pl.DataFrame({"Day": days, "CTR": [0.3, 0.1, 0.2]})

    figure = render_chart(
        frame,
        {"id": "line", "metric": "CTR", "chart": "line", "title": "Line", "x": "Day", "y": "CTR"},
    )

    assert [dt.date.fromisoformat(str(value)[:10]) for value in figure.data[0]["x"]] == sorted(days)


@pytest.mark.unit
def test_sparse_line_chart_renders_as_grouped_bar() -> None:
    days = [dt.date(2024, 1, day) for day in range(1, 30)]
    frame = pl.DataFrame({"Day": days, "CTR": [float(day) / 100 for day in range(1, 30)]})

    figure = render_chart(
        frame,
        {"id": "line", "metric": "CTR", "chart": "line", "title": "Line", "x": "Day", "y": "CTR"},
    )

    assert figure.data[0].type == "bar"
    assert figure.layout.barmode == "group"
    assert [dt.date.fromisoformat(str(value)[:10]) for value in figure.data[0]["x"]] == days


@pytest.mark.unit
def test_line_chart_keeps_line_trace_at_distinct_threshold() -> None:
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=offset) for offset in range(30)]
    frame = pl.DataFrame({"Day": days, "CTR": [float(offset) / 100 for offset in range(30)]})

    figure = render_chart(
        frame,
        {"id": "line", "metric": "CTR", "chart": "line", "title": "Line", "x": "Day", "y": "CTR"},
    )

    assert figure.data[0].type == "scatter"


@pytest.mark.unit
def test_bar_chart_sorts_and_limits_rows() -> None:
    frame = pl.DataFrame(
        {
            "Channel": ["A", "B", "C"],
            "CTR": [0.2, 0.5, 0.3],
        }
    )

    figure = render_chart(
        frame,
        {
            "id": "bar",
            "metric": "CTR",
            "chart": "bar",
            "title": "Bar",
            "x": "Channel",
            "y": "CTR",
            "sort_by": "CTR",
            "sort_direction": "desc",
            "top_n": 2,
        },
    )

    assert list(figure.data[0]["x"]) == ["B", "C"]


@pytest.mark.unit
def test_bar_chart_supports_percent_stacking() -> None:
    figure = render_chart(
        _base_frame(),
        {
            "id": "bar",
            "metric": "CTR",
            "chart": "bar",
            "title": "Bar",
            "x": "Channel",
            "y": "CTR",
            "color": "Placement",
            "barmode": "percent",
        },
    )

    assert figure.layout.barmode == "stack"
    assert figure.layout.barnorm == "percent"


@pytest.mark.unit
def test_scatter_ignores_invalid_marker_size_values() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "CTR": [0.2, 0.3, 0.4],
                "Lift": [0.1, 0.2, 0.3],
                "Count": [10.0, float("nan"), 5.0],
            }
        ),
        {
            "id": "scatter",
            "metric": "Lift",
            "chart": "scatter",
            "title": "Scatter",
            "x": "CTR",
            "y": "Lift",
            "size": "Count",
        },
    )

    sizes = list(figure.data[0].marker.size)

    assert all(math.isfinite(size) and size >= 0 for size in sizes)
    assert sizes[0] == pytest.approx(math.log1p(10.0))
    assert sizes[1] == 0
    assert sizes[2] == pytest.approx(math.log1p(5.0))


@pytest.mark.unit
def test_chart_settings_add_goal_line_percent_format_and_trend_delta() -> None:
    figure = render_chart(
        _base_frame(),
        {
            "id": "line",
            "metric": "CTR",
            "chart": "line",
            "title": "Line",
            "x": "Day",
            "y": "CTR",
            "value_format": "percent",
            "goal_line": {"value": 0.25, "label": "Target"},
            "show_trend_delta": True,
        },
    )

    assert figure.layout.yaxis.tickformat == ".2%"
    assert figure.layout.shapes
    assert any("Delta" in annotation.text for annotation in figure.layout.annotations)


@pytest.mark.unit
def test_facet_annotation_prefixes_are_stripped_but_values_remain() -> None:
    figure = render_chart(
        _base_frame(),
        {
            "id": "line",
            "metric": "CTR",
            "chart": "line",
            "title": "Line",
            "x": "Day",
            "y": "CTR",
            "facet_row": "Channel",
            "facet_col": "Placement",
            "show_trend_delta": True,
        },
    )

    annotation_texts = [annotation.text for annotation in figure.layout.annotations]
    x_titles = [
        getattr(figure.layout, name).title.text
        for name in figure.layout
        if name.startswith("xaxis")
    ]
    y_titles = [
        getattr(figure.layout, name).title.text
        for name in figure.layout
        if name.startswith("yaxis")
    ]

    assert {"Web", "Mobile", "Hero", "Sidebar"} <= set(annotation_texts)
    assert all(not text.startswith(("Channel=", "Placement=")) for text in annotation_texts)
    # One centered title per orientation: stamping the title on every facet
    # row/column stacks copies into overlapping, unreadable text.
    assert x_titles.count("Day") == 1
    assert y_titles.count("CTR") == 1
    assert any("Delta" in text for text in annotation_texts)


@pytest.mark.unit
def test_gauge_renders_faceted_indicator_grid() -> None:
    figure = render_chart(
        _base_frame(),
        {
            "id": "gauge",
            "metric": "CTR",
            "chart": "gauge",
            "title": "Gauge",
            "value": "CTR",
            "facet_row": "Channel",
            "facet_col": "Placement",
            "reference": {
                "Web_Hero": 0.25,
                "Web_Sidebar": 0.35,
                "Mobile_Hero": 0.45,
            },
            "value_format": "percent",
        },
    )

    assert len(figure.data) == 3
    annotation_texts = {annotation.text for annotation in figure.layout.annotations}
    assert {
        "Web Hero",
        "Web Sidebar",
        "Mobile Hero",
    } <= annotation_texts
    gauge_title_annotations = [
        annotation
        for annotation in figure.layout.annotations
        if annotation.text in {"Web Hero", "Web Sidebar", "Mobile Hero"}
    ]
    assert all(annotation.yshift == 24 for annotation in gauge_title_annotations)
    assert all(annotation.yanchor == "bottom" for annotation in gauge_title_annotations)
    assert all(trace.type == "indicator" for trace in figure.data)
    assert all(trace.number.valueformat == ".2%" for trace in figure.data)
    assert {trace.gauge.threshold.value for trace in figure.data} == {0.25, 0.35, 0.45}
    assert all(trace.gauge.axis.range[1] == pytest.approx(0.48) for trace in figure.data)
    assert figure.layout.height == 640


@pytest.mark.unit
def test_gauge_defaults_reference_line_to_average_value() -> None:
    figure = render_chart(
        _base_frame(),
        {
            "id": "gauge",
            "metric": "CTR",
            "chart": "gauge",
            "title": "Gauge",
            "value": "CTR",
            "facet_row": "Channel",
            "facet_col": "Placement",
        },
    )

    assert all(trace.gauge.threshold.line.color == "#c62828" for trace in figure.data)
    assert all(trace.gauge.threshold.value == pytest.approx(0.3) for trace in figure.data)
    assert all(trace.delta.reference == pytest.approx(0.3) for trace in figure.data)
    assert all(trace.gauge.axis.range[1] == pytest.approx(0.48) for trace in figure.data)


@pytest.mark.unit
def test_calibration_curve_percent_format_applies_to_both_rate_axes() -> None:
    figure = render_chart(
        _frame("calibration"),
        {
            "id": "calibration",
            "metric": "MIL_Calibration",
            "chart": "calibration_curve",
            "title": "Calibration",
            "value_format": "percent",
        },
    )

    assert figure.layout.xaxis.tickformat == ".2%"
    assert figure.layout.yaxis.tickformat == ".2%"
    assert "%{x:.2%}" in str(figure.data[0].hovertemplate)
    assert "%{y:.2%}" in str(figure.data[0].hovertemplate)


@pytest.mark.unit
def test_calibration_curve_supports_legacy_facet_column_alias() -> None:
    rows = pl.DataFrame(
        {
            "Segment": ["A", "B"],
            "Calibration": [
                {
                    "bin": [0.0, 1.0],
                    "predicted": [0.1, 0.9],
                    "observed": [0.2, 0.8],
                },
                {
                    "bin": [0.0, 1.0],
                    "predicted": [0.2, 0.8],
                    "observed": [0.3, 0.7],
                },
            ],
        }
    )

    figure = render_chart(
        rows,
        {
            "id": "calibration",
            "metric": "MIL_Calibration",
            "chart": "calibration_curve",
            "title": "Calibration",
            "facet_column": "Segment",
            "value_format": "percent",
        },
    )

    trace_x_values = {
        tuple(round(float(value), 1) for value in trace.x)
        for trace in figure.data
        if hasattr(trace, "x")
    }

    assert trace_x_values == {(0.1, 0.9), (0.2, 0.8)}
    assert figure.layout.xaxis.tickformat == ".2%"
    assert figure.layout.xaxis2.tickformat == ".2%"


@pytest.mark.unit
def test_gain_and_lift_curves_derive_population_fraction_from_curve_arrays() -> None:
    rows = _curve_frame()

    gain = render_chart(
        rows,
        {
            "id": "gain",
            "metric": "ROC_AUC",
            "chart": "gain_curve",
            "title": "Gain",
            "color": "Channel",
        },
    )
    lift = render_chart(
        rows,
        {
            "id": "lift",
            "metric": "ROC_AUC",
            "chart": "lift_curve",
            "title": "Lift",
            "color": "Channel",
        },
    )

    web_gain = next(trace for trace in gain.data if trace.name == "Web")
    web_lift = next(trace for trace in lift.data if trace.name == "Web")

    assert pytest.approx(web_gain["x"][1]) == 0.35
    assert pytest.approx(web_gain["y"][1]) == 0.8
    assert pytest.approx(web_lift["y"][1]) == 0.8 / 0.35
    assert lift.layout.yaxis.range[1] >= 2.0


@pytest.mark.unit
def test_pareto_chart_adds_cumulative_share_axis() -> None:
    figure = render_chart(
        _marketing_frame(),
        {
            "id": "pareto",
            "metric": "Revenue",
            "chart": "pareto",
            "title": "Pareto",
            "x": "Campaign",
            "y": "Revenue",
        },
    )

    assert len(figure.data) == 2
    assert list(figure.data[1]["y"])[-1] == pytest.approx(1.0)
    assert figure.layout.yaxis2.tickformat == ".0%"


@pytest.mark.unit
def test_sankey_chart_maps_source_target_labels_to_link_indices() -> None:
    figure = render_chart(
        _marketing_frame(),
        {
            "id": "sankey",
            "metric": "FlowValue",
            "chart": "sankey",
            "title": "Sankey",
            "source": "SourceStage",
            "target": "TargetStage",
            "value": "FlowValue",
        },
    )

    labels = list(figure.data[0]["node"]["label"])

    assert {"Email", "Landing", "Signup"} <= set(labels)
    assert len(figure.data[0]["link"]["source"]) == _marketing_frame().height


@pytest.mark.unit
def test_calendar_heatmap_buckets_dates_by_weekday_and_week_start() -> None:
    figure = render_chart(
        _marketing_frame(),
        {
            "id": "calendar",
            "metric": "Revenue",
            "chart": "calendar_heatmap",
            "title": "Calendar",
            "date": "Day",
            "value": "Revenue",
        },
    )

    assert "Mon" in figure.data[0]["y"]
    assert "2024-01-01" in figure.data[0]["x"]


@pytest.mark.unit
def test_chart_settings_apply_axis_and_legend_label_overrides() -> None:
    figure = render_chart(
        _base_frame(),
        {
            "id": "line",
            "metric": "CTR",
            "chart": "line",
            "title": "Line",
            "x": "Channel",
            "y": "CTR",
            "color": "Placement",
            "labels": {"Channel": "Channel Name", "CTR": "CTR", "Placement": "Placement"},
            "y_axis_title": "CTR (%)",
            "legend_title": "Placement Type",
            "axis_title_standoff": 18,
        },
    )

    assert figure.layout.xaxis.title.text == "Channel Name"
    assert figure.layout.yaxis.title.text == "CTR (%)"
    assert figure.layout.legend.title.text == "Placement Type"
    assert figure.layout.xaxis.title.standoff == 18
    assert figure.layout.yaxis.title.standoff == 18


@pytest.mark.unit
def test_chart_title_is_suppressed_when_tile_header_owns_title() -> None:
    figure = render_chart(
        _base_frame(),
        {
            "id": "line",
            "metric": "CTR",
            "chart": "line",
            "title": "Line",
            "x": "Channel",
            "y": "CTR",
        },
    )

    assert figure.layout.title.text is None


@pytest.mark.unit
def test_chart_defaults_to_friendly_axis_and_legend_labels() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "CustomerSegment": ["A", "B"],
                "VS_Click_to_Conversion_Dropoff": [0.2, 0.3],
                "Offer_Type": ["Web", "Email"],
            }
        ),
        {
            "id": "bar",
            "metric": "VS_Click_to_Conversion_Dropoff",
            "chart": "bar",
            "title": "Dropoff",
            "x": "CustomerSegment",
            "y": "VS_Click_to_Conversion_Dropoff",
            "color": "Offer_Type",
        },
    )

    assert figure.layout.xaxis.title.text == "Customer Segment"
    assert figure.layout.yaxis.title.text == "Click to Conversion Dropoff"
    assert figure.layout.legend.title.text == "Offer Type"


@pytest.mark.unit
def test_treemap_supports_legacy_x_y_tile_shape() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "CustomerSegment": ["A", "B"],
                "Channel": ["Web", "Email"],
                "VS_Interactions": [10, 15],
            }
        ),
        {
            "id": "treemap",
            "metric": "VS_Interactions",
            "chart": "treemap",
            "title": "Interactions by Segment",
            "x": "CustomerSegment",
            "y": "VS_Interactions",
            "color": "Channel",
        },
    )

    assert figure.data[0]["type"] == "treemap"
    assert "A" in figure.data[0]["labels"]


@pytest.mark.unit
def test_treemap_uses_theme_aware_default_colorscales() -> None:
    rows = pl.DataFrame(
        {
            "Channel": ["Web", "Email"],
            "Placement": ["Hero", "Flex"],
            "CTR": [0.1, 0.2],
        }
    )
    tile = {
        "id": "treemap",
        "metric": "CTR",
        "chart": "treemap",
        "title": "CTR Treemap",
        "path": ["Channel", "Placement"],
        "color": "CTR",
    }

    light = render_chart(rows, tile, theme={"base": "light"})
    dark = render_chart(rows, tile, theme={"base": "dark"})

    assert light.layout.coloraxis.colorscale[0][1] == "#334155"
    assert light.layout.coloraxis.colorscale[-1][1] == "#B7D968"
    assert dark.layout.coloraxis.colorscale[0][1] == "#223046"
    assert dark.layout.coloraxis.colorscale[-1][1] == "#C7E77A"


@pytest.mark.unit
def test_treemap_respects_explicit_color_scale_override() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Channel": ["Web", "Email"],
                "Placement": ["Hero", "Flex"],
                "CTR": [0.1, 0.2],
            }
        ),
        {
            "id": "treemap",
            "metric": "CTR",
            "chart": "treemap",
            "title": "CTR Treemap",
            "path": ["Channel", "Placement"],
            "color": "CTR",
            "color_continuous_scale": "Cividis",
        },
        theme={"base": "dark"},
    )

    assert figure.layout.coloraxis.colorscale[0][1] == "#00224e"


@pytest.mark.unit
def test_treemap_applies_percent_format_to_colorbar_and_hover() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Channel": ["Web", "Email"],
                "Placement": ["Hero", "Flex"],
                "CTR": [0.02, 0.1],
            }
        ),
        {
            "id": "treemap",
            "metric": "CTR",
            "chart": "treemap",
            "title": "CTR Treemap",
            "path": ["Channel", "Placement"],
            "color": "CTR",
            "value_format": "percent",
        },
    )

    assert figure.layout.coloraxis.colorbar.tickformat == ".2%"
    assert "CTR=%{color:.2%}" in str(figure.data[0].hovertemplate)


@pytest.mark.unit
def test_heatmap_uses_theme_aware_hot_cold_colorscales() -> None:
    rows = pl.DataFrame(
        {
            "Channel": ["Web", "Web", "Email", "Email"],
            "Placement": ["Hero", "Flex", "Hero", "Flex"],
            "CTR": [0.01, 0.04, 0.08, 0.12],
        }
    )
    tile = {
        "id": "heatmap",
        "metric": "CTR",
        "chart": "heatmap",
        "title": "CTR Heatmap",
        "x": "Channel",
        "y": "Placement",
        "color": "CTR",
    }

    light = render_chart(rows, tile, theme={"base": "light"})
    dark = render_chart(rows, tile, theme={"base": "dark"})

    assert light.data[0].colorscale[0][1] == "#2563EB"
    assert light.data[0].colorscale[-1][1] == "#DC2626"
    assert dark.data[0].colorscale[0][1] == "#5598E7"
    assert dark.data[0].colorscale[-1][1] == "#FCA5A5"


@pytest.mark.unit
def test_heatmap_respects_explicit_color_scale_override() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Channel": ["Web", "Email"],
                "Placement": ["Hero", "Flex"],
                "CTR": [0.1, 0.2],
            }
        ),
        {
            "id": "heatmap",
            "metric": "CTR",
            "chart": "heatmap",
            "title": "CTR Heatmap",
            "x": "Channel",
            "y": "Placement",
            "color": "CTR",
            "color_continuous_scale": "Cividis",
        },
        theme={"base": "dark"},
    )

    assert figure.data[0].colorscale[0][1] == "#00224e"


@pytest.mark.unit
def test_heatmap_applies_percent_format_to_colorbar_and_hover_z() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Channel": ["Web", "Email"],
                "Placement": ["Hero", "Flex"],
                "CTR": [0.02, 0.1],
            }
        ),
        {
            "id": "heatmap",
            "metric": "CTR",
            "chart": "heatmap",
            "title": "CTR Heatmap",
            "x": "Channel",
            "y": "Placement",
            "color": "CTR",
            "value_format": "percent",
        },
    )

    assert figure.data[0].colorbar.tickformat == ".2%"
    assert "%{z:.2%}" in str(figure.data[0].hovertemplate)
    assert "%{y:.2%}" not in str(figure.data[0].hovertemplate)


@pytest.mark.unit
def test_calendar_heatmap_applies_percent_format_to_colorbar_and_hover_z() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Day": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
                "CTR": [0.02, 0.1],
            }
        ),
        {
            "id": "calendar",
            "metric": "CTR",
            "chart": "calendar_heatmap",
            "title": "CTR Calendar",
            "date": "Day",
            "value": "CTR",
            "value_format": "percent",
        },
    )

    assert figure.data[0].colorbar.tickformat == ".2%"
    assert "%{z:.2%}" in str(figure.data[0].hovertemplate)


@pytest.mark.unit
def test_descriptive_heatmap_uses_same_colorscale_as_regular_heatmap() -> None:
    rows = pl.DataFrame(
        {
            "Channel": ["Web", "Email"],
            "Placement": ["Hero", "Flex"],
            "ResponseTime_Mean": [0.1, 0.2],
        }
    )
    descriptive = render_chart(
        rows,
        {
            "id": "descriptive_heatmap",
            "metric": "ResponseTime",
            "chart": "descriptive_heatmap",
            "title": "Response Time Heatmap",
            "x": "Channel",
            "y": "Placement",
            "property": "ResponseTime",
            "score": "Mean",
        },
        theme={"base": "dark"},
    )
    regular = render_chart(
        rows,
        {
            "id": "heatmap",
            "metric": "ResponseTime",
            "chart": "heatmap",
            "title": "Response Time Heatmap",
            "x": "Channel",
            "y": "Placement",
            "color": "ResponseTime_Mean",
        },
        theme={"base": "dark"},
    )

    assert descriptive.data[0].colorscale == regular.data[0].colorscale
    assert descriptive.data[0].colorscale[0][1] == "#5598E7"


@pytest.mark.unit
def test_descriptive_line_can_render_p50_from_digest_state() -> None:
    rows = pl.DataFrame(
        {
            "Month": ["2026-01", "2026-02"],
            "Propensity_tdigest": [
                tdigest.build([0.1, 0.2, 0.3]),
                tdigest.build([0.6, 0.7, 0.8]),
            ],
        }
    )

    figure = render_chart(
        rows,
        {
            "id": "descriptive_line",
            "metric": "PropensityP50",
            "chart": "descriptive_line",
            "title": "P50",
            "x": "Month",
            "property": "Propensity",
            "score": "p50",
        },
    )

    assert figure.data[0].type == "bar"
    assert list(figure.data[0]["y"]) == pytest.approx([0.2, 0.7], abs=0.1)


@pytest.mark.unit
def test_colored_boxplots_render_in_group_mode() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Month": ["2026-01", "2026-01", "2026-02", "2026-02"],
                "Issue": ["Acquisition", "Activation", "Acquisition", "Activation"],
                "Propensity": [0.1, 0.2, 0.3, 0.4],
            }
        ),
        {
            "id": "box",
            "metric": "Propensity",
            "chart": "boxplot",
            "title": "Propensity",
            "x": "Month",
            "property": "Propensity",
            "color": "Issue",
        },
    )

    assert figure.layout.boxmode == "group"


@pytest.mark.unit
def test_colored_quantile_boxplots_render_in_group_mode() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Month": ["2026-01", "2026-01", "2026-02", "2026-02"],
                "Issue": ["Acquisition", "Activation", "Acquisition", "Activation"],
                "Propensity_p25": [0.1, 0.2, 0.3, 0.4],
                "Propensity_Median": [0.2, 0.3, 0.4, 0.5],
                "Propensity_p75": [0.3, 0.4, 0.5, 0.6],
                "Propensity_Min": [0.0, 0.1, 0.2, 0.3],
                "Propensity_Max": [0.4, 0.5, 0.6, 0.7],
            }
        ),
        {
            "id": "descriptive_box",
            "metric": "Propensity",
            "chart": "descriptive_boxplot",
            "title": "Quartiles",
            "x": "Month",
            "property": "Propensity",
            "color": "Issue",
        },
    )

    assert figure.layout.boxmode == "group"


@pytest.mark.unit
def test_quantile_boxplots_render_faceted_grouped_subplots() -> None:
    rows = pl.DataFrame(
        {
            "Month": ["2026-01"] * 8,
            "Issue": [
                "Acquisition",
                "Activation",
                "Acquisition",
                "Activation",
                "Acquisition",
                "Activation",
                "Acquisition",
                "Activation",
            ],
            "Channel": ["Web", "Web", "Web", "Web", "Mobile", "Mobile", "Mobile", "Mobile"],
            "CustomerType": [
                "Known",
                "Known",
                "Anonymous",
                "Anonymous",
                "Known",
                "Known",
                "Anonymous",
                "Anonymous",
            ],
            "Propensity_Count": [100, 120, 90, 110, 80, 95, 70, 85],
            "Propensity_Mean": [0.18, 0.22, 0.2, 0.24, 0.19, 0.23, 0.21, 0.25],
            "Propensity_p25": [0.1, 0.12, 0.11, 0.13, 0.12, 0.14, 0.13, 0.15],
            "Propensity_Median": [0.2, 0.22, 0.21, 0.23, 0.22, 0.24, 0.23, 0.25],
            "Propensity_p75": [0.3, 0.32, 0.31, 0.33, 0.32, 0.34, 0.33, 0.35],
            "Propensity_Min": [0.0, 0.02, 0.01, 0.03, 0.02, 0.04, 0.03, 0.05],
            "Propensity_Max": [0.5, 0.52, 0.51, 0.53, 0.52, 0.54, 0.53, 0.55],
        }
    )

    figure = render_chart(
        rows,
        {
            "id": "descriptive_box",
            "metric": "Propensity",
            "chart": "descriptive_boxplot",
            "title": "Quartiles",
            "x": "Month",
            "property": "Propensity",
            "color": "Issue",
            "facet_row": "Channel",
            "facet_col": "CustomerType",
        },
    )

    assert figure.layout.boxmode == "group"
    assert len(figure.data) == rows.height
    assert len({trace.xaxis for trace in figure.data}) == 4
    assert len({trace.yaxis for trace in figure.data}) == 4
    assert {trace.offsetgroup for trace in figure.data} == {"Acquisition", "Activation"}
    assert sorted(trace.name for trace in figure.data if trace.showlegend) == [
        "Acquisition",
        "Activation",
    ]
    assert {"Web", "Mobile", "Known", "Anonymous"} <= {
        annotation.text for annotation in figure.layout.annotations
    }


@pytest.mark.unit
def test_descriptive_histogram_renders_tdigest_bins_with_facets() -> None:
    rows = pl.DataFrame(
        {
            "Issue": ["Acquisition", "Activation", "Acquisition", "Activation"],
            "Channel": ["Web", "Web", "Mobile", "Mobile"],
            "CustomerType": ["Known", "Anonymous", "Known", "Anonymous"],
            "Propensity_tdigest": [
                tdigest.build([0.1, 0.2, 0.3]),
                tdigest.build([0.4, 0.5, 0.6]),
                tdigest.build([0.2, 0.3, 0.4]),
                tdigest.build([0.6, 0.7, 0.8]),
            ],
        }
    )

    figure = render_chart(
        rows,
        {
            "id": "descriptive_hist",
            "metric": "Propensity",
            "chart": "descriptive_histogram",
            "title": "Distribution",
            "property": "Propensity",
            "color": "Issue",
            "facet_row": "Channel",
            "facet_col": "CustomerType",
        },
    )

    assert figure.layout.barmode == "overlay"
    assert len(figure.data) == rows.height
    assert all(trace.type == "bar" for trace in figure.data)
    assert all(len(trace.x) == 100 for trace in figure.data)
    assert all(sum(trace.y) > 0 for trace in figure.data)
    assert len({trace.xaxis for trace in figure.data}) == 4
    assert len({trace.yaxis for trace in figure.data}) == 4
    assert sorted(trace.name for trace in figure.data if trace.showlegend) == [
        "Acquisition",
        "Activation",
    ]
    assert {"Web", "Mobile", "Known", "Anonymous"} <= {
        annotation.text for annotation in figure.layout.annotations
    }


@pytest.mark.unit
def test_descriptive_funnel_uses_x_count_rows_with_facets() -> None:
    rows = pl.DataFrame(
        {
            "Outcome": [
                "Impression",
                "Clicked",
                "Conversion",
                "Impression",
                "Clicked",
                "Conversion",
            ],
            "Channel": ["Web", "Web", "Web", "Mobile", "Mobile", "Mobile"],
            "Issue": [
                "Acquisition",
                "Acquisition",
                "Acquisition",
                "Activation",
                "Activation",
                "Activation",
            ],
            "Outcome_Count": [100, 60, 20, 80, 50, 10],
            "Outcome_Mean": [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        }
    )

    figure = render_chart(
        rows,
        {
            "id": "describe_outcome_funnel",
            "metric": "Outcome_Mean",
            "chart": "descriptive_funnel",
            "title": "Outcome Funnel",
            "stages": ["Impression", "Clicked", "Conversion"],
            "x": "Outcome",
            "property": "Outcome",
            "color": "Issue",
            "facet_col": "Channel",
        },
    )

    assert {trace.type for trace in figure.data} == {"funnel"}
    assert len({trace.xaxis for trace in figure.data}) == 2
    assert {trace.name for trace in figure.data} == {"Acquisition", "Activation"}
    assert sorted(value for trace in figure.data for value in trace.x) == [10, 20, 50, 60, 80, 100]


@pytest.mark.unit
def test_experiment_z_score_renders_horizontal_bar_with_significance_band() -> None:
    figure = render_chart(
        _experiment_frame(),
        {
            "id": "experiment_z",
            "metric": "Experiment_Significance",
            "chart": "experiment_z_score",
            "title": "Experiment Z",
            "x": "z_score",
            "y": "ExperimentName",
            "facet_col": "Channel",
        },
    )

    assert {trace.orientation for trace in figure.data} == {"h"}
    assert not figure.layout.showlegend
    assert figure.layout.updatemenus[0].buttons[0].label == "Bar"
    assert figure.layout.updatemenus[0].buttons[1].label == "Line"
    assert any(shape.x0 == -1.96 and shape.x1 == 1.96 for shape in figure.layout.shapes)
    assert {"Web", "Mobile"} <= {annotation.text for annotation in figure.layout.annotations}


@pytest.mark.unit
def test_experiment_odds_ratio_renders_ci_errors_and_significance_buckets() -> None:
    figure = render_chart(
        _experiment_frame(),
        {
            "id": "experiment_odds",
            "metric": "Experiment_Significance",
            "chart": "experiment_odds_ratio",
            "title": "Experiment Odds",
            "x": "g_odds_ratio_stat",
            "y": "ExperimentName",
            "facet_row": "Channel",
            "facet_col": "CustomerType",
        },
    )

    assert not figure.layout.showlegend
    assert {trace.name for trace in figure.data} == {"Control", "N/A", "Test"}
    assert any(shape.x0 == 1 and shape.x1 == 1 for shape in figure.layout.shapes)
    assert all(trace.error_x.array is not None for trace in figure.data)
    assert all(trace.error_x.arrayminus is not None for trace in figure.data)
    assert {"Web", "Mobile", "Known", "Anonymous"} <= {
        annotation.text for annotation in figure.layout.annotations
    }


@pytest.mark.unit
def test_conditional_formatting_colors_bar_marks() -> None:
    figure = render_chart(
        _base_frame(),
        {
            "id": "bar",
            "metric": "CTR",
            "chart": "bar",
            "title": "Bar",
            "x": "Channel",
            "y": "CTR",
            "conditional_formatting": [
                {"column": "CTR", "operator": ">=", "value": 0.3, "color": "#2e7d32"},
                {"column": "CTR", "operator": "<", "value": 0.3, "color": "#c62828"},
            ],
        },
    )

    assert list(figure.data[0]["marker"]["color"]) == ["#c62828", "#2e7d32", "#2e7d32"]


def _base_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Day": [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
            "Channel": ["Web", "Web", "Mobile"],
            "Placement": ["Hero", "Sidebar", "Hero"],
            "CTR": [0.2, 0.3, 0.4],
            "Impression_Count": [100, 80, 60],
            "Clicked_Count": [20, 24, 24],
            "Conversion_Count": [4, 6, 8],
            "frequency": [1, 2, 3],
            "recency": [5, 3, 1],
            "tenure": [10, 10, 10],
            "monetary_value": [50.0, 100.0, 150.0],
            "lifetime_value": [100.0, 200.0, 300.0],
            "customers_count": [1, 1, 1],
            "rfm_segment": ["Top Spender", "Repeat Customer", "Premium Customer"],
        }
    )


def _frame(name: str | None) -> pl.DataFrame:  # noqa: PLR0911
    if name == "box":
        return _box_frame()
    if name == "calibration":
        return _calibration_frame()
    if name == "curve":
        return _curve_frame()
    if name == "marketing":
        return _marketing_frame()
    if name == "descriptive":
        return _descriptive_frame()
    if name == "experiment":
        return _experiment_frame()
    return _base_frame()


def _box_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Channel": ["Web", "Mobile"],
            "ResponseTime_p25": [1.0, 2.0],
            "ResponseTime_Median": [2.0, 3.0],
            "ResponseTime_p75": [3.0, 4.0],
            "ResponseTime_Min": [0.5, 1.5],
            "ResponseTime_Max": [4.0, 5.0],
        }
    )


@pytest.mark.unit
def test_comparison_scale_and_semantic_colors_are_partitioned_by_series() -> None:
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=index) for index in range(31)]
    rows = pl.DataFrame(
        {
            "Day": [*days, *days],
            "Channel": ["Web"] * 31 + ["Mobile"] * 31,
            "CTR": [float(index + 1) for index in range(31)]
            + [float((index + 1) * 2) for index in range(31)],
        }
    )

    figure = render_chart(
        rows,
        {
            "metric": "CTR",
            "chart": "line",
            "x": "Day",
            "y": "CTR",
            "color": "Channel",
            "scale_mode": "index_100",
        },
        theme={"category_colors": {"Channel": {"Web": "#2563EB", "Mobile": "#14B8A6"}}},
    )

    traces = {trace.name: trace for trace in figure.data}
    assert float(traces["Web"].y[0]) == pytest.approx(100.0)
    assert float(traces["Mobile"].y[0]) == pytest.approx(100.0)
    assert traces["Web"].line.color == "#2563EB"
    assert traces["Mobile"].line.color == "#14B8A6"


@pytest.mark.unit
def test_interval_chart_accepts_absolute_confidence_bounds() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Channel": ["Web"],
                "Effect": [0.1],
                "Low": [-0.05],
                "High": [0.25],
            }
        ),
        {
            "metric": "Effect",
            "chart": "interval",
            "x": "Channel",
            "y": "Effect",
            "error_y_lower": "Low",
            "error_y_upper": "High",
        },
    )

    assert list(figure.data[0].error_y.array) == pytest.approx([0.15])
    assert list(figure.data[0].error_y.arrayminus) == pytest.approx([0.15])


def _calibration_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Calibration": [
                {
                    "bin": [0.0, 0.5, 1.0],
                    "predicted": [0.1, 0.5, 0.9],
                    "observed": [0.0, 0.6, 1.0],
                }
            ]
        }
    )


def _curve_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Channel": ["Web", "Mobile"],
            "ROC_AUC": [0.9, 0.8],
            "roc_auc": [0.9, 0.8],
            "average_precision": [0.7, 0.6],
            "fpr": [[0.0, 0.2, 1.0], [0.0, 0.4, 1.0]],
            "tpr": [[0.0, 0.8, 1.0], [0.0, 0.7, 1.0]],
            "precision": [[1.0, 0.8, 0.5], [1.0, 0.7, 0.5]],
            "recall": [[0.0, 0.8, 1.0], [0.0, 0.7, 1.0]],
            "pos_fraction": [0.25, 0.3],
        }
    )


def _marketing_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Day": [
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 8),
            ],
            "Month": ["2024-01", "2024-01", "2024-02"],
            "Cohort": ["2023-12", "2023-12", "2024-01"],
            "Channel": ["Email", "Search", "Social"],
            "Campaign": ["Welcome", "Brand", "Retarget"],
            "CountryCode": ["USA", "DEU", "FRA"],
            "SourceStage": ["Email", "Email", "Landing"],
            "TargetStage": ["Landing", "Signup", "Signup"],
            "Revenue": [100.0, 60.0, 40.0],
            "Spend": [40.0, 35.0, 20.0],
            "Retention": [0.4, 0.35, 0.25],
            "Lift": [0.1, 0.2, 0.15],
            "StdErr": [0.02, 0.03, 0.025],
            "FlowValue": [80.0, 30.0, 50.0],
        }
    )


def _descriptive_frame() -> pl.DataFrame:
    return _base_frame().with_columns(pl.Series("ResponseTime_Mean", [2.0, 3.0, 4.0]))


def _experiment_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ExperimentName": ["A", "B", "C"],
            "Channel": ["Web", "Web", "Mobile"],
            "CustomerType": ["Known", "Anonymous", "Known"],
            "z_score": [1.2, 2.4, -2.1],
            "g_odds_ratio_stat": [1.2, 0.7, 1.0],
            "g_odds_ratio_ci_low": [1.05, 0.5, 0.8],
            "g_odds_ratio_ci_high": [1.5, 0.9, 1.2],
            "chi2_odds_ratio_stat": [1.15, 0.75, 1.1],
            "chi2_odds_ratio_ci_low": [1.01, 0.55, 0.9],
            "chi2_odds_ratio_ci_high": [1.4, 0.95, 1.3],
        }
    )
