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
from valuestream.ui import theme as ui_theme

SUPPORTED_CHART_CASES = [
    (
        {"chart": "line", "title": "Line", "metric": "CTR", "x": "Day", "color": "Channel"},
        None,
    ),
    (
        {
            "chart": "stacked_area",
            "title": "Area",
            "metric": "CTR",
            "x": "Day",
            "color": "Channel",
        },
        None,
    ),
    ({"chart": "bar", "title": "Bar", "metric": "CTR", "x": "Channel"}, None),
    ({"chart": "kpi_card", "title": "KPI", "metric": "CTR", "reference": 0.25}, None),
    (
        {"chart": "waterfall", "title": "Waterfall", "metric": "Revenue", "x": "Channel"},
        "marketing",
    ),
    (
        {
            "chart": "pareto",
            "title": "Pareto",
            "metric": "Revenue",
            "metric_output": "Revenue",
            "x": "Campaign",
        },
        "marketing",
    ),
    (
        {
            "chart": "treemap",
            "title": "Tree",
            "metric": "CTR",
            "path": ["Channel", "Placement"],
        },
        None,
    ),
    (
        {
            "chart": "heatmap",
            "title": "Heat",
            "metric": "CTR",
            "metric_output": "CTR",
            "x": "Channel",
            "y": "Placement",
        },
        None,
    ),
    (
        {
            "chart": "heatmap",
            "title": "Cohort",
            "metric": "Retention",
            "metric_output": "Retention",
            "x": "Month",
            "y": "Cohort",
        },
        "marketing",
    ),
    ({"chart": "scatter", "title": "Scatter", "x": "frequency", "y": "monetary_value"}, None),
    (
        {
            "chart": "combo",
            "title": "Combo",
            "metric": "Spend",
            "metric_output": "Spend",
            "secondary_metric": "Revenue",
            "x": "Day",
        },
        "marketing",
    ),
    (
        {
            "chart": "interval",
            "title": "Interval",
            "metric": "Lift",
            "metric_output": "Lift",
            "x": "Campaign",
        },
        "marketing",
    ),
    ({"chart": "donut", "title": "Donut", "metric": "Revenue", "names": "Channel"}, "marketing"),
    (
        {
            "chart": "geo_map",
            "title": "Geo",
            "metric": "Revenue",
            "metric_output": "Revenue",
            "locations": "CountryCode",
            "locationmode": "ISO-3",
        },
        "marketing",
    ),
    ({"chart": "table", "title": "Table", "columns": ["Campaign", "Revenue"]}, "marketing"),
    (
        {
            "chart": "heatmap",
            "title": "Calendar",
            "metric": "Revenue",
            "metric_output": "Revenue",
            "x": "Day",
        },
        "marketing",
    ),
    (
        {
            "chart": "bar_polar",
            "title": "Polar",
            "metric": "CTR",
            "theta": "Channel",
            "color": "Placement",
        },
        None,
    ),
    (
        {
            "chart": "sankey",
            "title": "Sankey",
            "metric": "FlowValue",
            "metric_output": "FlowValue",
            "path": ["SourceStage", "TargetStage"],
        },
        "marketing",
    ),
    (
        {"chart": "gauge", "title": "Gauge", "metric": "CTR", "references": {"Web": 0.4}},
        None,
    ),
    (
        {
            "chart": "funnel",
            "title": "Funnel",
            "stages": ["Impression", "Clicked", "Conversion"],
            "color": "Channel",
        },
        None,
    ),
    ({"chart": "boxplot", "title": "Box", "x": "Channel"}, "box"),
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
            "chart": "heatmap",
            "title": "DHeat",
            "metric": "ResponseTime_Mean",
            "metric_output": "ResponseTime_Mean",
            "x": "Channel",
            "y": "Placement",
        },
        "descriptive",
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
def test_purchase_frequency_projection_uses_run_rate_and_clear_axis_labels() -> None:
    figure = render_chart(
        _base_frame(),
        {
            "id": "purchase_projection",
            "title": "Purchase frequency projection",
            "metric": "CLV",
            "chart": "model",
            "horizon": 20,
        },
    )

    projected = sorted(float(value) for trace in figure.data for value in trace.y)
    assert projected == pytest.approx([2.0, 4.0, 6.0])
    assert figure.layout.xaxis.title.text == "Historical purchase frequency"
    assert figure.layout.yaxis.title.text == "Projected purchases (20-day horizon)"
    assert figure.layout.legend.title.text == "RFM segment"


@pytest.mark.unit
def test_kpi_card_uses_selected_metric_output() -> None:
    figure = render_chart(
        pl.DataFrame({"Channel": ["Web"], "CTR": [0.25]}),
        {
            "id": "ctr",
            "title": "CTR",
            "metric": "Experiment",
            "metric_output": "CTR",
            "chart": "kpi_card",
        },
    )

    assert figure.data[0].value == pytest.approx(0.25)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("chart_kind", "x", "extra"),
    [
        ("line", "Day", {}),
        ("bar", "Channel", {}),
        ("stacked_area", "Day", {"color": "Channel"}),
        ("waterfall", "Channel", {}),
    ],
)
def test_metric_owned_y_charts_use_selected_metric_output(
    chart_kind: str,
    x: str,
    extra: dict[str, str],
) -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Day": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
                "Channel": ["Web", "Mobile"],
                "CTR": [0.2, 0.4],
                "Count": [20, 40],
            }
        ),
        {
            "id": f"ctr_{chart_kind}",
            "title": "CTR",
            "metric": "Experiment",
            "metric_output": "CTR",
            "chart": chart_kind,
            "x": x,
            **extra,
        },
    )

    plotted = sorted(float(value) for trace in figure.data for value in trace.y)
    assert plotted == pytest.approx([0.2, 0.4])


@pytest.mark.unit
def test_pareto_uses_selected_metric_output() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Channel": ["Web", "Mobile"],
                "CTR": [0.2, 0.4],
                "Count": [20, 40],
            }
        ),
        {
            "id": "ctr_pareto",
            "title": "CTR",
            "metric": "Experiment",
            "metric_output": "CTR",
            "chart": "pareto",
            "x": "Channel",
        },
    )

    plotted = sorted(float(value) for value in figure.data[0].y)
    assert plotted == pytest.approx([0.2, 0.4])


@pytest.mark.unit
def test_polar_bar_uses_selected_metric_output() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Channel": ["Web", "Mobile"],
                "CTR": [0.2, 0.4],
                "Count": [20, 40],
            }
        ),
        {
            "id": "ctr_polar",
            "title": "CTR",
            "metric": "Experiment",
            "metric_output": "CTR",
            "chart": "bar_polar",
            "theta": "Channel",
            "color": "Channel",
        },
    )

    plotted = sorted(float(value) for trace in figure.data for value in trace.r)
    assert plotted == pytest.approx([0.2, 0.4])


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
            "metric_output": "Clicked_Count",
            "chart": "combo",
            "x": "Day",
            "secondary_metric": "Impression_Count",
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
    assert list(figure.data[0].y) == [12, 15]
    assert list(figure.data[1].y) == [120, 150]


@pytest.mark.unit
def test_combo_chart_can_render_two_lines_on_one_shared_axis() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "ExposureFrequency": [1, 2, 3],
                "ComparableCTR": [0.021, 0.014, 0.009],
                "RunnerExpectedCTR": [0.006, 0.0062, 0.0065],
            }
        ),
        {
            "id": "frequency_response",
            "metric": "ComparableCTR",
            "metric_output": "ComparableCTR",
            "chart": "combo",
            "x": "ExposureFrequency",
            "secondary_metric": "RunnerExpectedCTR",
            "primary_mark": "line",
            "shared_y_axis": True,
            "y_axis_title": "Response probability",
        },
    )

    assert [trace.type for trace in figure.data] == ["scatter", "scatter"]
    assert all(trace.yaxis == "y" for trace in figure.data)
    assert figure.layout.yaxis.title.text == "Response probability"
    assert "yaxis2" not in figure.layout
    assert list(figure.data[0].y) == pytest.approx([0.021, 0.014, 0.009])
    assert list(figure.data[1].y) == pytest.approx([0.006, 0.0062, 0.0065])


@pytest.mark.unit
def test_combo_chart_separates_facet_columns_without_cross_facet_lines() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "ExposureFrequency": [1, 1, 2, 2, 3, 3],
                "CustomerType": ["Known", "Anonymous"] * 3,
                "ComparableCTR": [0.021, 0.012, 0.014, 0.011, 0.009, 0.008],
                "RunnerExpectedCTR": [0.006, 0.005, 0.0062, 0.0052, 0.0065, 0.0055],
            }
        ),
        {
            "id": "frequency_response",
            "metric": "ComparableCTR",
            "metric_output": "ComparableCTR",
            "chart": "combo",
            "x": "ExposureFrequency",
            "secondary_metric": "RunnerExpectedCTR",
            "primary_mark": "line",
            "shared_y_axis": True,
            "facet_col": "CustomerType",
            "y_axis_title": "Response probability",
        },
        theme={"colorway": ["#0072B2", "#D55E00"]},
    )

    assert len(figure.data) == 4
    assert len({trace.xaxis for trace in figure.data}) == 2
    assert all(list(trace.x) == [1, 2, 3] for trace in figure.data)
    assert [trace.showlegend for trace in figure.data] == [True, True, False, False]
    assert [trace.legendgroup for trace in figure.data] == [
        "ComparableCTR",
        "RunnerExpectedCTR",
        "ComparableCTR",
        "RunnerExpectedCTR",
    ]
    assert [trace.line.color for trace in figure.data] == [
        "#0072B2",
        "#D55E00",
        "#0072B2",
        "#D55E00",
    ]
    assert {annotation.text for annotation in figure.layout.annotations} == {
        "Known",
        "Anonymous",
    }
    assert [
        getattr(figure.layout, name).title.text
        for name in figure.layout
        if name.startswith("yaxis")
    ].count("Response probability") == 1


@pytest.mark.unit
def test_combo_chart_facets_preserve_primary_and_secondary_axis_semantics() -> None:
    rows: list[dict[str, object]] = []
    for channel in ("Web", "Mobile"):
        for customer_type in ("Known", "Anonymous"):
            for exposure in (1, 2):
                rows.append(
                    {
                        "ExposureFrequency": exposure,
                        "Channel": channel,
                        "CustomerType": customer_type,
                        "Clicks": exposure + (1 if channel == "Web" else 2),
                        "Impressions": 100 * exposure,
                    }
                )

    figure = render_chart(
        pl.DataFrame(rows),
        {
            "id": "clicks_and_impressions",
            "metric": "Clicks",
            "metric_output": "Clicks",
            "chart": "combo",
            "x": "ExposureFrequency",
            "secondary_metric": "Impressions",
            "facet_row": "Channel",
            "facet_col": "CustomerType",
            "y_axis_title": "Clicks",
            "y2_axis_title": "Impressions",
        },
    )

    assert len(figure.data) == 8
    assert len({trace.xaxis for trace in figure.data}) == 4
    assert len({trace.yaxis for trace in figure.data}) == 8
    for index in range(0, len(figure.data), 2):
        primary = figure.data[index]
        secondary = figure.data[index + 1]
        assert list(primary.x) == [1, 2]
        assert primary.xaxis == secondary.xaxis
        assert primary.yaxis != secondary.yaxis
    assert {"Web", "Mobile", "Known", "Anonymous"} <= {
        annotation.text for annotation in figure.layout.annotations
    }
    titled_y_axes = [
        axis
        for name in figure.layout
        if name.startswith("yaxis")
        and (axis := getattr(figure.layout, name)).title.text is not None
    ]
    assert [axis.title.text for axis in titled_y_axes].count("Clicks") == 1
    assert [axis.title.text for axis in titled_y_axes].count("Impressions") == 1
    assert next(axis for axis in titled_y_axes if axis.title.text == "Clicks").overlaying is None
    assert next(axis for axis in titled_y_axes if axis.title.text == "Impressions").side == "right"


@pytest.mark.unit
def test_combo_chart_deduplicates_color_series_across_facets() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "ExposureFrequency": [1, 1, 1, 1, 2, 2, 2, 2],
                "CustomerType": ["Known", "Known", "Anonymous", "Anonymous"] * 2,
                "Channel": ["Web", "Mobile"] * 4,
                "CTR": [0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45],
                "ExpectedCTR": [0.05, 0.06, 0.07, 0.08, 0.055, 0.065, 0.075, 0.085],
            }
        ),
        {
            "id": "colored_frequency_response",
            "metric": "CTR",
            "metric_output": "CTR",
            "chart": "combo",
            "x": "ExposureFrequency",
            "secondary_metric": "ExpectedCTR",
            "primary_mark": "line",
            "shared_y_axis": True,
            "color": "Channel",
            "facet_col": "CustomerType",
        },
        theme={"colorway": ["#111111", "#222222", "#333333", "#444444"]},
    )

    visible = [trace for trace in figure.data if trace.showlegend]
    assert len(figure.data) == 8
    assert [trace.name for trace in visible] == [
        "CTR · Web",
        "ExpectedCTR · Web",
        "CTR · Mobile",
        "ExpectedCTR · Mobile",
    ]
    colors_by_group: dict[str, set[str]] = {}
    for trace in figure.data:
        colors_by_group.setdefault(str(trace.legendgroup), set()).add(str(trace.line.color))
    assert all(len(colors) == 1 for colors in colors_by_group.values())


@pytest.mark.unit
def test_combo_chart_uses_the_effective_template_palette_across_facets() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "ExposureFrequency": [1, 1, 2, 2],
                "CustomerType": ["Known", "Anonymous"] * 2,
                "CTR": [0.1, 0.2, 0.15, 0.25],
                "ExpectedCTR": [0.05, 0.06, 0.055, 0.065],
            }
        ),
        {
            "id": "frequency_response",
            "metric": "CTR",
            "metric_output": "CTR",
            "chart": "combo",
            "x": "ExposureFrequency",
            "secondary_metric": "ExpectedCTR",
            "primary_mark": "line",
            "shared_y_axis": True,
            "facet_col": "CustomerType",
        },
        theme={"template": "seaborn"},
    )

    expected = list(pio.templates["seaborn"].layout.colorway)[:2]
    assert [trace.line.color for trace in figure.data] == [*expected, *expected]


@pytest.mark.unit
def test_combo_chart_keeps_null_and_literal_none_facets_separate() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "ExposureFrequency": [1, 1, 2, 2],
                "CustomerType": [None, "None", None, "None"],
                "CTR": [0.1, 0.2, 0.15, 0.25],
                "ExpectedCTR": [0.05, 0.06, 0.055, 0.065],
            }
        ),
        {
            "id": "frequency_response",
            "metric": "CTR",
            "metric_output": "CTR",
            "chart": "combo",
            "x": "ExposureFrequency",
            "secondary_metric": "ExpectedCTR",
            "primary_mark": "line",
            "shared_y_axis": True,
            "facet_col": "CustomerType",
        },
    )

    assert len(figure.data) == 4
    assert len({trace.xaxis for trace in figure.data}) == 2
    assert all(list(trace.x) == [1, 2] for trace in figure.data)
    assert {annotation.text for annotation in figure.layout.annotations} == {"(null)", "None"}


@pytest.mark.unit
def test_combo_chart_rejects_a_missing_configured_facet_column() -> None:
    with pytest.raises(ValueError, match="facet column 'CustomerType' is not available"):
        render_chart(
            pl.DataFrame(
                {
                    "ExposureFrequency": [1, 2],
                    "CTR": [0.1, 0.2],
                    "ExpectedCTR": [0.05, 0.06],
                }
            ),
            {
                "id": "frequency_response",
                "metric": "CTR",
                "metric_output": "CTR",
                "chart": "combo",
                "x": "ExposureFrequency",
                "secondary_metric": "ExpectedCTR",
                "facet_col": "CustomerType",
            },
        )


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
    expected_font = (
        'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
        '"Helvetica Neue", Arial, sans-serif'
    )
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
            "metric_output": "Interactions",
            "chart": "line",
            "title": "Interactions Trend",
            "x": "Day",
        },
        theme={
            "template": "plotly_white",
            "colorway": ["#4B73F0", "#22C7F3"],
            "paper_bgcolor": "#f5f3ee",
            "plot_bgcolor": "#f5f3ee",
        },
    )

    assert list(figure.layout.colorway) == ["#4B73F0", "#22C7F3"]
    assert figure.layout.paper_bgcolor == "#f5f3ee"
    assert figure.layout.plot_bgcolor == "#f5f3ee"


@pytest.mark.unit
def test_funnel_qualitative_palette_handles_plotly_array_stages() -> None:
    figure = render_chart(
        _marketing_frame(),
        {
            "id": "funnel",
            "metric": "Interactions",
            "chart": "funnel",
            "title": "Funnel",
            "stages": ["Impression", "Clicked", "Conversion"],
            "color": "Channel",
        },
        theme={"colorway": ["#4B73F0", "#22C7F3"]},
    )

    assert {trace.type for trace in figure.data} == {"funnel"}
    assert all(len(trace.marker.color) == max(1, len(trace.y)) for trace in figure.data)


@pytest.mark.unit
def test_grouped_report_uses_the_app_dark_chart_palette_and_surface() -> None:
    frame = pl.DataFrame(
        {
            "Day": [
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 2),
            ],
            "Channel": ["Web", "Web", "Mobile", "Mobile"],
            "Interactions": [120, 140, 90, 110],
        }
    )
    previous_default = pio.templates.default
    try:
        ui_theme.init_plotly_theme.cache_clear()
        ui_theme.init_plotly_theme()
        figure = render_chart(
            frame,
            {
                "id": "interactions_trend",
                "metric": "Interactions",
                "metric_output": "Interactions",
                "chart": "line",
                "title": "Interactions Trend",
                "x": "Day",
                "color": "Channel",
            },
            theme=ui_theme.dashboard_theme({"base": "dark"}),
        )
    finally:
        pio.templates.default = previous_default

    assert [trace.marker.color for trace in figure.data] == ["#4B73F0", "#22C7F3"]
    assert figure.layout.paper_bgcolor == "#162438"
    assert figure.layout.plot_bgcolor == "#162438"


@pytest.mark.unit
def test_grouped_report_can_use_the_true_light_chart_palette_and_surface() -> None:
    frame = pl.DataFrame(
        {
            "Day": [
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 2),
            ],
            "Channel": ["Web", "Web", "Mobile", "Mobile"],
            "Interactions": [120, 140, 90, 110],
        }
    )
    previous_default = pio.templates.default
    try:
        ui_theme.init_plotly_theme.cache_clear()
        ui_theme.init_plotly_theme()
        tile_theme = {"base": "light", "template": "valuestream_light"}
        figure = render_chart(
            frame,
            {
                "id": "interactions_trend",
                "metric": "Interactions",
                "metric_output": "Interactions",
                "chart": "line",
                "title": "Interactions Trend",
                "x": "Day",
                "color": "Channel",
                "theme": tile_theme,
            },
            theme=ui_theme.dashboard_theme(tile_theme),
        )
    finally:
        pio.templates.default = previous_default

    assert [trace.marker.color for trace in figure.data] == ["#0072B2", "#D55E00"]
    assert figure.layout.paper_bgcolor == "#ffffff"
    assert figure.layout.plot_bgcolor == "#ffffff"
    assert figure.layout.font.color == "#17202a"


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
        {
            "id": "line",
            "metric": "CTR",
            "metric_output": "CTR",
            "chart": "line",
            "title": "Line",
            "x": "Day",
        },
        max_points=100,
    )

    assert len(figure.data[0]["x"]) <= 100


@pytest.mark.unit
def test_line_chart_sorts_rows_by_x_axis() -> None:
    days = [dt.date(2024, 1, day) for day in (3, 1, 2)]
    frame = pl.DataFrame({"Day": days, "CTR": [0.3, 0.1, 0.2]})

    figure = render_chart(
        frame,
        {
            "id": "line",
            "metric": "CTR",
            "metric_output": "CTR",
            "chart": "line",
            "title": "Line",
            "x": "Day",
        },
    )

    assert [dt.date.fromisoformat(str(value)[:10]) for value in figure.data[0]["x"]] == sorted(days)


@pytest.mark.unit
def test_sparse_line_chart_renders_as_grouped_bar() -> None:
    days = [dt.date(2024, 1, day) for day in range(1, 30)]
    frame = pl.DataFrame({"Day": days, "CTR": [float(day) / 100 for day in range(1, 30)]})

    figure = render_chart(
        frame,
        {
            "id": "line",
            "metric": "CTR",
            "metric_output": "CTR",
            "chart": "line",
            "title": "Line",
            "x": "Day",
        },
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
        {
            "id": "line",
            "metric": "CTR",
            "metric_output": "CTR",
            "chart": "line",
            "title": "Line",
            "x": "Day",
        },
    )

    assert figure.data[0].type == "scatter"


@pytest.mark.unit
def test_line_chart_maps_color_and_style_to_independent_dimensions() -> None:
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=offset) for offset in range(30)]
    frame = pl.DataFrame(
        [
            {
                "Day": day,
                "Channel": channel,
                "CustomerType": customer_type,
                "CTR": 0.01 * (index + 1),
            }
            for index, day in enumerate(days)
            for channel in ("Web", "Mobile")
            for customer_type in ("Known", "Anonymous")
        ]
    )

    figure = render_chart(
        frame,
        {
            "id": "line",
            "metric": "CTR",
            "metric_output": "CTR",
            "chart": "line",
            "title": "CTR by channel and customer type",
            "x": "Day",
            "color": "Channel",
            "line_dash": "CustomerType",
            "symbol": "CustomerType",
        },
        theme={
            "category_colors": {
                "Channel": {
                    "Web": "#2563EB",
                    "Mobile": "#14B8A6",
                }
            }
        },
    )

    assert len(figure.data) == 4
    web = [trace for trace in figure.data if "Web" in str(trace.name)]
    mobile = [trace for trace in figure.data if "Mobile" in str(trace.name)]
    known = [trace for trace in figure.data if "Known" in str(trace.name)]
    anonymous = [trace for trace in figure.data if "Anonymous" in str(trace.name)]
    assert {trace.line.color for trace in web} == {"#2563EB"}
    assert {trace.line.color for trace in mobile} == {"#14B8A6"}
    assert len({trace.line.dash for trace in known}) == 1
    assert len({trace.line.dash for trace in anonymous}) == 1
    assert {trace.line.dash for trace in known} != {trace.line.dash for trace in anonymous}
    assert len({trace.marker.symbol for trace in known}) == 1
    assert len({trace.marker.symbol for trace in anonymous}) == 1
    assert {trace.marker.symbol for trace in known} != {
        trace.marker.symbol for trace in anonymous
    }
    assert all("markers" in str(trace.mode) for trace in figure.data)


@pytest.mark.unit
def test_line_scale_mode_partitions_by_color_and_line_style() -> None:
    frame = pl.DataFrame(
        {
            "Day": [
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 2),
            ],
            "Channel": ["Web"] * 4,
            "CustomerType": ["Known", "Known", "Anonymous", "Anonymous"],
            "CTR": [0.1, 0.2, 0.4, 0.2],
        }
    )

    figure = render_chart(
        frame,
        {
            "id": "line",
            "metric": "CTR",
            "metric_output": "CTR",
            "chart": "line",
            "title": "Indexed CTR",
            "x": "Day",
            "color": "Channel",
            "line_dash": "CustomerType",
            "scale_mode": "index_100",
        },
    )

    traces = {str(trace.name): list(trace.y) for trace in figure.data}
    assert next(values for name, values in traces.items() if "Known" in name) == [100.0, 200.0]
    assert next(values for name, values in traces.items() if "Anonymous" in name) == [
        100.0,
        50.0,
    ]


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
            "metric_output": "CTR",
            "chart": "bar",
            "title": "Bar",
            "x": "Channel",
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
            "metric_output": "CTR",
            "chart": "bar",
            "title": "Bar",
            "x": "Channel",
            "color": "Placement",
            "barmode": "percent",
        },
    )

    assert figure.layout.barmode == "stack"
    assert figure.layout.barnorm == "percent"


@pytest.mark.unit
def test_month_axis_uses_exact_periods_instead_of_thirty_day_date_ticks() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Month": [
                    "2024-06",
                    "2024-06",
                    "2024-07",
                    "2024-07",
                    "2024-08",
                    "2024-08",
                    "2024-09",
                    "2024-09",
                ],
                "Channel": ["Mobile", "Web"] * 4,
                "Lift": [0.6, 1.15, 0.46, 0.57, 0.48, 0.89, 0.63, 0.89],
            }
        ),
        {
            "id": "monthly_lift",
            "metric": "ModelControlLift",
            "metric_output": "Lift",
            "chart": "bar",
            "title": "Monthly model-control lift",
            "x": "Month",
            "color": "Channel",
            "value_format": "percent",
        },
    )

    assert figure.layout.xaxis.type == "category"
    assert list(figure.layout.xaxis.tickvals) == [
        "2024-06",
        "2024-07",
        "2024-08",
        "2024-09",
    ]
    assert list(figure.layout.xaxis.ticktext) == [
        "June",
        "July",
        "August",
        "September",
    ]


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
            "metric_output": "CTR",
            "chart": "line",
            "title": "Line",
            "x": "Day",
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
            "metric_output": "CTR",
            "chart": "line",
            "title": "Line",
            "x": "Day",
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
            "metric_output": "CTR",
            "chart": "gauge",
            "title": "Gauge",
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
def test_calibration_curve_supports_canonical_facet_column() -> None:
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
            "facet_col": "Segment",
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
            "metric_output": "Revenue",
        },
    )

    assert len(figure.data) == 2
    assert list(figure.data[1]["y"])[-1] == pytest.approx(1.0)
    assert figure.layout.yaxis2.tickformat == ".0%"


@pytest.mark.unit
def test_sankey_chart_maps_path_labels_to_link_indices() -> None:
    figure = render_chart(
        _marketing_frame(),
        {
            "id": "sankey",
            "metric": "FlowValue",
            "metric_output": "FlowValue",
            "chart": "sankey",
            "title": "Sankey",
            "path": ["SourceStage", "TargetStage"],
        },
    )

    labels = list(figure.data[0]["node"]["label"])

    assert {"Email", "Landing", "Signup"} <= set(labels)
    assert len(figure.data[0]["link"]["source"]) == _marketing_frame().height
    assert list(figure.data[0]["link"]["value"]) == pytest.approx([80.0, 30.0, 50.0])


@pytest.mark.unit
def test_sankey_chart_connects_every_adjacent_step_in_ordered_path() -> None:
    figure = render_chart(
        _marketing_frame(),
        {
            "id": "multi_step_sankey",
            "metric": "FlowValue",
            "metric_output": "FlowValue",
            "chart": "sankey",
            "title": "Multi-step Sankey",
            "path": ["SourceStage", "TargetStage", "Campaign"],
        },
    )

    trace = figure.data[0]

    assert list(trace["node"]["customdata"]) == [
        "SourceStage",
        "SourceStage",
        "TargetStage",
        "TargetStage",
        "Campaign",
        "Campaign",
        "Campaign",
    ]
    assert len(trace["link"]["source"]) == _marketing_frame().height * 2
    assert list(trace["link"]["value"]) == pytest.approx(
        [80.0, 80.0, 30.0, 30.0, 50.0, 50.0]
    )


@pytest.mark.unit
def test_daily_heatmap_buckets_dates_by_weekday_and_week_start() -> None:
    figure = render_chart(
        _marketing_frame(),
        {
            "id": "calendar",
            "metric": "Revenue",
            "metric_output": "Revenue",
            "chart": "heatmap",
            "title": "Calendar",
            "x": "Day",
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
            "metric_output": "CTR",
            "chart": "line",
            "title": "Line",
            "x": "Channel",
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
            "metric_output": "CTR",
            "chart": "line",
            "title": "Line",
            "x": "Channel",
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
            "metric_output": "VS_Click_to_Conversion_Dropoff",
            "chart": "bar",
            "title": "Dropoff",
            "x": "CustomerSegment",
            "color": "Offer_Type",
        },
    )

    assert figure.layout.xaxis.title.text == "Customer Segment"
    assert figure.layout.yaxis.title.text == "Click to Conversion Dropoff"
    assert figure.layout.legend.title.text == "Offer Type"


@pytest.mark.unit
def test_treemap_renders_dimension_path() -> None:
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
            "metric_output": "VS_Interactions",
            "chart": "treemap",
            "title": "Interactions by Segment",
            "path": ["CustomerSegment", "Channel"],
        },
    )

    assert figure.data[0]["type"] == "treemap"
    assert "A" in figure.data[0]["labels"]


@pytest.mark.unit
def test_treemap_uses_selected_metric_output() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Channel": ["Web", "Mobile"],
                "CTR": [0.2, 0.4],
                "Count": [20, 40],
            }
        ),
        {
            "id": "ctr_treemap",
            "title": "CTR",
            "metric": "Experiment",
            "metric_output": "CTR",
            "chart": "treemap",
            "path": ["Channel"],
        },
    )

    colors = [float(value) for value in figure.data[0].marker.colors]
    assert min(colors) == pytest.approx(0.2)
    assert max(colors) == pytest.approx(0.4)


@pytest.mark.unit
def test_donut_uses_selected_metric_output() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Channel": ["Web", "Mobile"],
                "CTR": [0.2, 0.4],
                "Count": [20, 40],
            }
        ),
        {
            "id": "ctr_mix",
            "title": "CTR mix",
            "metric": "Experiment",
            "metric_output": "CTR",
            "chart": "donut",
            "names": "Channel",
        },
    )

    assert sorted(float(value) for value in figure.data[0].values) == pytest.approx([0.2, 0.4])


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
        "metric_output": "CTR",
        "chart": "treemap",
        "title": "CTR Treemap",
        "path": ["Channel", "Placement"],
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
            "metric_output": "CTR",
            "chart": "treemap",
            "title": "CTR Treemap",
            "path": ["Channel", "Placement"],
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
            "metric_output": "CTR",
            "chart": "treemap",
            "title": "CTR Treemap",
            "path": ["Channel", "Placement"],
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
        "metric_output": "CTR",
        "chart": "heatmap",
        "title": "CTR Heatmap",
        "x": "Channel",
        "y": "Placement",
    }

    light = render_chart(rows, tile, theme={"base": "light"})
    dark = render_chart(rows, tile, theme={"base": "dark"})

    assert light.data[0].colorscale[0][1] == "#2563EB"
    assert light.data[0].colorscale[-1][1] == "#DC2626"
    assert dark.data[0].colorscale[0][1] == "#5598E7"
    assert dark.data[0].colorscale[-1][1] == "#FCA5A5"


@pytest.mark.unit
def test_heatmap_intensity_uses_selected_metric_output() -> None:
    rows = pl.DataFrame(
        {
            "Channel": ["Web", "Email"],
            "Placement": ["Hero", "Flex"],
            "CTR": [0.1, 0.2],
            "Count": [100, 200],
        }
    )

    figure = render_chart(
        rows,
        {
            "id": "heatmap",
            "metric": "Experiment",
            "metric_output": "CTR",
            "chart": "heatmap",
            "title": "CTR Heatmap",
            "x": "Channel",
            "y": "Placement",
        },
    )

    assert {
        value
        for row in figure.data[0].z
        for value in row
        if value is not None and not math.isnan(value)
    } == {0.1, 0.2}


@pytest.mark.unit
def test_unified_heatmap_without_y_uses_daily_calendar_layout() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Day": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
                "CTR": [0.02, 0.1],
                "Count": [20, 100],
            }
        ),
        {
            "id": "calendar",
            "metric": "CTR",
            "metric_output": "CTR",
            "chart": "heatmap",
            "title": "CTR Calendar",
            "x": "Day",
        },
    )

    assert "Mon" in figure.data[0]["y"]
    assert "2024-01-01" in figure.data[0]["x"]
    assert {value for row in figure.data[0].z for value in row if value is not None} == {
        0.02,
        0.1,
    }


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
            "metric_output": "CTR",
            "chart": "heatmap",
            "title": "CTR Heatmap",
            "x": "Channel",
            "y": "Placement",
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
            "metric_output": "CTR",
            "chart": "heatmap",
            "title": "CTR Heatmap",
            "x": "Channel",
            "y": "Placement",
            "value_format": "percent",
        },
    )

    assert figure.data[0].colorbar.tickformat == ".2%"
    assert "%{z:.2%}" in str(figure.data[0].hovertemplate)
    assert "%{y:.2%}" not in str(figure.data[0].hovertemplate)


@pytest.mark.unit
def test_daily_heatmap_applies_percent_format_to_colorbar_and_hover_z() -> None:
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
            "metric_output": "CTR",
            "chart": "heatmap",
            "title": "CTR Calendar",
            "x": "Day",
            "value_format": "percent",
        },
    )

    assert figure.data[0].colorbar.tickformat == ".2%"
    assert "%{z:.2%}" in str(figure.data[0].hovertemplate)


@pytest.mark.unit
def test_metric_output_heatmap_uses_theme_aware_colorscale() -> None:
    rows = pl.DataFrame(
        {
            "Channel": ["Web", "Email"],
            "Placement": ["Hero", "Flex"],
            "ResponseTime_Mean": [0.1, 0.2],
        }
    )
    figure = render_chart(
        rows,
        {
            "id": "heatmap",
            "metric": "ResponseTimeMean",
            "metric_output": "ResponseTime_Mean",
            "chart": "heatmap",
            "title": "Response Time Heatmap",
            "x": "Channel",
            "y": "Placement",
        },
        theme={"base": "dark"},
    )

    assert figure.data[0].colorscale[0][1] == "#5598E7"


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
def test_descriptive_line_normalizes_uppercase_quantile_score() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Month": ["2026-01", "2026-02"],
                "ResponseTime_p95": [1.2, 1.8],
            }
        ),
        {
            "id": "response_time",
            "metric": "ResponseTimeP95",
            "chart": "descriptive_line",
            "title": "P95 response time",
            "x": "Month",
            "property": "ResponseTime",
            "score": "P95",
        },
    )

    assert list(figure.data[0]["y"]) == [1.2, 1.8]


@pytest.mark.unit
def test_colored_boxplots_render_in_group_mode() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Month": ["2026-01", "2026-01", "2026-02", "2026-02"],
                "Issue": ["Acquisition", "Activation", "Acquisition", "Activation"],
                "Propensity_p25": [0.1, 0.2, 0.3, 0.4],
                "Propensity_Median": [0.2, 0.3, 0.4, 0.5],
                "Propensity_p75": [0.3, 0.4, 0.5, 0.6],
            }
        ),
        {
            "id": "box",
            "metric": "Propensity",
            "chart": "boxplot",
            "title": "Propensity",
            "x": "Month",
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
            "chart": "boxplot",
            "title": "Quartiles",
            "x": "Month",
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
            "chart": "boxplot",
            "title": "Quartiles",
            "x": "Month",
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
def test_histogram_renders_tdigest_bins_with_facets() -> None:
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
            "id": "digest_hist",
            "metric": "Propensity",
            "chart": "histogram",
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
def test_funnel_uses_categorical_count_rows_with_facets() -> None:
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
            "id": "outcome_funnel",
            "metric": "OutcomeCounts",
            "metric_output": "Outcome_Count",
            "chart": "funnel",
            "title": "Outcome Funnel",
            "stages": ["Impression", "Clicked", "Conversion"],
            "x": "Outcome",
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
            "value_format": "number",
        },
    )

    assert {trace.orientation for trace in figure.data} == {"h"}
    assert not figure.layout.showlegend
    assert figure.layout.hovermode == "closest"
    assert figure.layout.xaxis.tickformat == ",.2f"
    assert figure.layout.yaxis.tickformat is None
    assert all("%{x:,.2f}" in str(trace.hovertemplate) for trace in figure.data)
    assert all("%{y:" not in str(trace.hovertemplate) for trace in figure.data)
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
            "metric_output": "CTR",
            "chart": "bar",
            "title": "Bar",
            "x": "Channel",
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
            "metric_output": "CTR",
            "chart": "line",
            "x": "Day",
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
            "metric_output": "Effect",
            "chart": "interval",
            "x": "Channel",
            "lower_output": "Low",
            "upper_output": "High",
        },
    )

    assert list(figure.data[0].error_y.array) == pytest.approx([0.15])
    assert list(figure.data[0].error_y.arrayminus) == pytest.approx([0.15])


@pytest.mark.unit
def test_interval_chart_plots_relative_lift_confidence_bounds() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Channel": ["Web"],
                "Lift": [0.5],
                "Lift_CI_Low": [-0.084],
                "Lift_CI_High": [1.456],
            }
        ),
        {
            "metric": "ExperimentLift",
            "metric_output": "Lift",
            "chart": "interval",
            "x": "Channel",
            "lower_output": "Lift_CI_Low",
            "upper_output": "Lift_CI_High",
        },
    )

    assert list(figure.data[0].error_y.array) == pytest.approx([0.956])
    assert list(figure.data[0].error_y.arrayminus) == pytest.approx([0.584])


@pytest.mark.unit
def test_interval_chart_accepts_null_confidence_bounds() -> None:
    figure = render_chart(
        pl.DataFrame(
            {
                "Channel": ["Web"],
                "Effect": pl.Series([None], dtype=pl.Float64),
                "Low": pl.Series([None], dtype=pl.Float64),
                "High": pl.Series([None], dtype=pl.Float64),
            }
        ),
        {
            "metric": "Effect",
            "metric_output": "Effect",
            "chart": "interval",
            "x": "Channel",
            "lower_output": "Low",
            "upper_output": "High",
        },
    )

    assert len(figure.data) == 1
    assert math.isnan(float(figure.data[0].error_y.array[0]))
    assert math.isnan(float(figure.data[0].error_y.arrayminus[0]))


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


@pytest.mark.unit
def test_boxplot_without_property_infers_quantile_suite_columns() -> None:
    """A boxplot tile authored without ``property`` still gets quantile boxes.

    Boxing the scalar metric column instead collapses every statistic to the
    per-group median, which renders as flat single-value boxes.
    """
    rows = pl.DataFrame(
        {
            "Year": [2022, 2022, 2023, 2023],
            "Issue": ["Acquisition", "Activation", "Acquisition", "Activation"],
            "dist_metric": [0.2, 0.3, 0.4, 0.5],
            "FinalPropensity_p25": [0.1, 0.2, 0.3, 0.4],
            "FinalPropensity_Median": [0.2, 0.3, 0.4, 0.5],
            "FinalPropensity_p75": [0.3, 0.4, 0.5, 0.6],
            "FinalPropensity_Min": [0.0, 0.1, 0.2, 0.3],
            "FinalPropensity_Max": [0.4, 0.5, 0.6, 0.7],
        }
    )
    figure = render_chart(
        rows,
        {
            "id": "dist_box",
            "metric": "dist_metric",
            "chart": "boxplot",
            "title": "Distribution",
            "x": "Year",
            "color": "Issue",
        },
    )

    boxes = [trace for trace in figure.data if trace.type == "box"]
    assert boxes
    first = boxes[0]
    # Quantile-suite path: explicit q1/median/q3 pulled from the suite columns.
    assert first.median is not None
    assert first.q1 is not None
    assert first.q3 is not None
    assert float(first.q1[0]) != float(first.q3[0])


@pytest.mark.unit
def test_boxplot_requires_one_complete_distribution_suite() -> None:
    with pytest.raises(ValueError, match="exactly one complete"):
        render_chart(
            pl.DataFrame(
                {
                    "FinalPropensity_p25": [0.1],
                    "FinalPropensity_p75": [0.3],
                }
            ),
            {
                "id": "box",
                "metric": "distribution",
                "chart": "boxplot",
                "title": "Distribution",
            },
        )
