"""Chart recipe metadata used by the UI layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChartRecipe:
    kind: str
    allowed_processor_kinds: tuple[str, ...]
    default_x: str | None = None
    default_y: str | None = None


_AGGREGATE_CHART_PROCESSORS = (
    "binary_outcome",
    "score_distribution",
    "numeric_distribution",
    "entity_lifecycle",
    "entity_set",
    "funnel",
    "snapshot",
)

METRIC_OWNED_Y_CHARTS = frozenset({"line", "bar", "stacked_area", "waterfall", "combo", "pareto"})
METRIC_OWNED_RADIAL_CHARTS = frozenset({"bar_polar"})
METRIC_OWNED_COLOR_CHARTS = frozenset({"treemap"})
METRIC_OWNED_VALUES_CHARTS = frozenset({"donut", "treemap"})
METRIC_OWNED_VALUE_CHARTS = frozenset(
    {"kpi_card", "gauge", "sankey", *METRIC_OWNED_Y_CHARTS, *METRIC_OWNED_RADIAL_CHARTS}
)
HEATMAP_CHARTS = frozenset({"heatmap"})
METRIC_OWNED_INTENSITY_CHARTS = HEATMAP_CHARTS
METRIC_OUTPUT_SELECTABLE_CHARTS = frozenset(
    {
        *METRIC_OWNED_VALUE_CHARTS,
        *METRIC_OWNED_VALUES_CHARTS,
        *METRIC_OWNED_COLOR_CHARTS,
        *METRIC_OWNED_INTENSITY_CHARTS,
    }
)
INTERVAL_METRIC_OUTPUT_FIELDS = frozenset(
    {"metric_output", "lower_output", "upper_output"}
)


CHART_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "line": ("x",),
    "stacked_area": ("x", "color"),
    "bar": ("x",),
    # KPI cards display the selected metric's primary output. A separate value
    # role would allow dimensions or unrelated aggregate states to be selected.
    "kpi_card": (),
    # Waterfall changes come from the selected metric. Plotly assigns semantic
    # increasing/decreasing/total colors, so a separate color role is invalid.
    "waterfall": ("x",),
    # Pareto bars and their cumulative share use the selected metric's output.
    "pareto": ("x",),
    # Treemap color comes from the selected metric.
    "treemap": ("path",),
    # Heatmap intensity is supplied by the selected metric. With no Y axis a
    # daily X axis renders as a calendar; otherwise X/Y render a matrix.
    "heatmap": ("x",),
    "scatter": ("x", "y"),
    # The selected metric supplies Y. ``secondary_metric`` names a compatible secondary
    # metric, not another arbitrary aggregate output column.
    "combo": ("x", "secondary_metric"),
    "interval": ("x", "metric_output"),
    # Donut slice values come from the selected metric.
    "donut": ("names",),
    "geo_map": ("locations",),
    "table": (),
    # Polar radius comes from the selected metric.
    "bar_polar": ("theta",),
    # Sankey links connect each adjacent pair in an ordered dimension path.
    # Link values come from the selected metric.
    "sankey": ("path",),
    # Gauges, like KPI cards, display the selected metric's primary output.
    "gauge": (),
    "funnel": ("stages",),
    "boxplot": (),
    "histogram": ("property",),
    "calibration_curve": (),
    "roc_curve": (),
    "precision_recall_curve": (),
    "gain_curve": (),
    "lift_curve": (),
    "rfm_density": (),
    "exposure": (),
    "corr": ("x", "y"),
    "model": (),
    "descriptive_line": ("x", "property", "score"),
    "experiment_z_score": ("x", "y"),
    "experiment_odds_ratio": ("x", "y"),
    "clv_treemap": ("path",),
}


CHART_OPTIONAL_FIELDS: dict[str, tuple[str, ...]] = {
    "line": ("color", "line_dash", "symbol", "facet_row", "facet_col"),
    "stacked_area": ("facet_row", "facet_col"),
    "bar": ("color", "facet_row", "facet_col"),
    "waterfall": (),
    "pareto": ("color", "facet_row", "facet_col"),
    "treemap": (),
    "heatmap": ("y",),
    "scatter": ("color", "size", "animation_frame", "animation_group", "facet_row", "facet_col"),
    "combo": ("color", "facet_row", "facet_col"),
    "interval": ("lower_output", "upper_output", "color", "facet_row", "facet_col"),
    "donut": ("color",),
    "geo_map": ("lat", "lon", "color", "size"),
    "table": ("columns", "group_by"),
    "bar_polar": ("color",),
    "gauge": ("facet_row", "facet_col"),
    "funnel": ("x", "facet_row", "facet_col"),
    "boxplot": ("x", "color", "facet_row", "facet_col"),
    "histogram": ("color", "facet_row", "facet_col"),
    "calibration_curve": ("color", "facet_row", "facet_col"),
    "roc_curve": ("color", "facet_row", "facet_col"),
    "precision_recall_curve": ("color", "facet_row", "facet_col"),
    "gain_curve": ("color", "facet_row", "facet_col"),
    "lift_curve": ("color", "facet_row", "facet_col"),
    "rfm_density": ("x", "y", "color"),
    "exposure": ("color",),
    "model": ("color",),
    "descriptive_line": ("color", "facet_row", "facet_col"),
    "experiment_z_score": ("color", "facet_row", "facet_col"),
    "experiment_odds_ratio": ("color", "facet_row", "facet_col"),
    "clv_treemap": (),
}


# Each tuple is one required canonical role.
TILE_REQUIRED_ALTERNATIVES: dict[str, tuple[tuple[str, ...], ...]] = {
    "line": (("x",),),
    "stacked_area": (("x",), ("color",)),
    "bar": (("x",),),
    "kpi_card": (),
    "waterfall": (("x",),),
    "pareto": (("x",),),
    "treemap": (("path",),),
    "heatmap": (("x",),),
    "scatter": (("x",), ("y",)),
    "combo": (("x",), ("secondary_metric",)),
    "interval": (("x",), ("metric_output",)),
    "donut": (("names",),),
    "geo_map": (("locations", "lat"),),
    "bar_polar": (("theta",),),
    "sankey": (("path",),),
    "gauge": (),
    "funnel": (("stages",),),
    "boxplot": (),
    "histogram": (("property",),),
    "corr": (("x",), ("y",)),
    "descriptive_line": (("x",), ("property",), ("score",)),
    "experiment_z_score": (("x",), ("y",)),
    "experiment_odds_ratio": (("x",), ("y",)),
    "clv_treemap": (("path",),),
}


def chart_field_controls(chart_kind: str) -> tuple[str, ...]:
    """Return the canonical ordered field controls for one chart kind."""

    required = CHART_REQUIRED_FIELDS.get(chart_kind)
    if required is None:
        return ()
    metric_output = ("metric_output",) if chart_kind in METRIC_OUTPUT_SELECTABLE_CHARTS else ()
    return tuple(
        dict.fromkeys((*metric_output, *required, *CHART_OPTIONAL_FIELDS.get(chart_kind, ())))
    )


_TILE_RUNTIME_FIELDS = {
    "animation_frame",
    "animation_group",
    "bins",
    "color",
    "columns",
    "conditional_formatting",
    "facet_col",
    "facet_row",
    "filters",
    "goal_line",
    "group_by",
    "height",
    "hole",
    "horizon",
    "labels",
    "lat",
    "line_dash",
    "locationmode",
    "locations",
    "log_x",
    "log_y",
    "lower_output",
    "lon",
    "metric_output",
    "names",
    "path",
    "property",
    "reference",
    "references",
    "scale_mode",
    "score",
    "secondary_metric",
    "show_trend_delta",
    "showlegend",
    "size",
    "sort_by",
    "sort_direction",
    "stages",
    "symbol",
    "theme",
    "theta",
    "top_n",
    "upper_output",
    "value_format",
    "width",
    "x",
    "x_axis_title",
    "y",
    "y2_axis_title",
    "y_axis_title",
}
SUPPORTED_TILE_FIELDS = frozenset(
    {
        "id",
        "title",
        "metric",
        "chart",
        "description",
        "placement",
        "kpi",
        *_TILE_RUNTIME_FIELDS,
        *(field for fields in CHART_REQUIRED_FIELDS.values() for field in fields),
        *(field for fields in CHART_OPTIONAL_FIELDS.values() for field in fields),
    }
)


RECIPES: dict[str, ChartRecipe] = {
    "line": ChartRecipe("line", _AGGREGATE_CHART_PROCESSORS),
    "stacked_area": ChartRecipe("stacked_area", _AGGREGATE_CHART_PROCESSORS),
    "bar": ChartRecipe("bar", _AGGREGATE_CHART_PROCESSORS),
    "kpi_card": ChartRecipe("kpi_card", _AGGREGATE_CHART_PROCESSORS),
    "waterfall": ChartRecipe("waterfall", _AGGREGATE_CHART_PROCESSORS),
    "pareto": ChartRecipe("pareto", _AGGREGATE_CHART_PROCESSORS),
    "treemap": ChartRecipe("treemap", _AGGREGATE_CHART_PROCESSORS),
    "heatmap": ChartRecipe("heatmap", _AGGREGATE_CHART_PROCESSORS),
    "scatter": ChartRecipe("scatter", _AGGREGATE_CHART_PROCESSORS),
    "combo": ChartRecipe("combo", _AGGREGATE_CHART_PROCESSORS),
    "interval": ChartRecipe("interval", _AGGREGATE_CHART_PROCESSORS),
    "donut": ChartRecipe("donut", _AGGREGATE_CHART_PROCESSORS),
    "geo_map": ChartRecipe("geo_map", _AGGREGATE_CHART_PROCESSORS),
    "table": ChartRecipe("table", _AGGREGATE_CHART_PROCESSORS),
    "bar_polar": ChartRecipe("bar_polar", ("binary_outcome",)),
    "sankey": ChartRecipe("sankey", _AGGREGATE_CHART_PROCESSORS),
    "gauge": ChartRecipe("gauge", _AGGREGATE_CHART_PROCESSORS),
    "funnel": ChartRecipe("funnel", ("funnel", "numeric_distribution")),
    "boxplot": ChartRecipe("boxplot", ("numeric_distribution", "score_distribution")),
    "histogram": ChartRecipe("histogram", ("numeric_distribution", "entity_lifecycle")),
    "calibration_curve": ChartRecipe("calibration_curve", ("score_distribution",)),
    "roc_curve": ChartRecipe("roc_curve", ("score_distribution",)),
    "precision_recall_curve": ChartRecipe("precision_recall_curve", ("score_distribution",)),
    "gain_curve": ChartRecipe("gain_curve", ("score_distribution",)),
    "lift_curve": ChartRecipe("lift_curve", ("score_distribution",)),
    "rfm_density": ChartRecipe("rfm_density", ("entity_lifecycle",), "recency", "frequency"),
    "exposure": ChartRecipe("exposure", ("entity_lifecycle",)),
    "corr": ChartRecipe("corr", ("entity_lifecycle",), "frequency", "monetary_value"),
    "model": ChartRecipe("model", ("entity_lifecycle",)),
    "descriptive_line": ChartRecipe("descriptive_line", ("numeric_distribution",)),
    "experiment_z_score": ChartRecipe("experiment_z_score", ("binary_outcome",)),
    "experiment_odds_ratio": ChartRecipe("experiment_odds_ratio", ("binary_outcome",)),
    "clv_treemap": ChartRecipe("clv_treemap", ("entity_lifecycle",)),
}


__all__ = [
    "CHART_OPTIONAL_FIELDS",
    "CHART_REQUIRED_FIELDS",
    "HEATMAP_CHARTS",
    "INTERVAL_METRIC_OUTPUT_FIELDS",
    "METRIC_OUTPUT_SELECTABLE_CHARTS",
    "METRIC_OWNED_COLOR_CHARTS",
    "METRIC_OWNED_INTENSITY_CHARTS",
    "METRIC_OWNED_RADIAL_CHARTS",
    "METRIC_OWNED_VALUES_CHARTS",
    "METRIC_OWNED_VALUE_CHARTS",
    "METRIC_OWNED_Y_CHARTS",
    "RECIPES",
    "SUPPORTED_TILE_FIELDS",
    "TILE_REQUIRED_ALTERNATIVES",
    "ChartRecipe",
    "chart_field_controls",
]
