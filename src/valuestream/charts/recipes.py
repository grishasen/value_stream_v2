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


CHART_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "line": ("x", "y"),
    "stacked_area": ("x", "y", "color"),
    "bar": ("x", "y"),
    "kpi_card": ("value",),
    "waterfall": ("x", "y"),
    "pareto": ("x", "y"),
    "treemap": ("path", "color"),
    "heatmap": ("x", "y", "color"),
    "cohort_heatmap": ("x", "y", "color"),
    "scatter": ("x", "y"),
    "combo": ("x", "y", "y2"),
    "interval": ("x", "y"),
    "donut": ("names", "values"),
    "geo_map": ("locations", "value"),
    "table": (),
    "calendar_heatmap": ("date", "value"),
    "bar_polar": ("r", "theta", "color"),
    "sankey": ("source", "target", "value"),
    "gauge": ("value",),
    "funnel": ("stages", "color"),
    "boxplot": ("x",),
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
    "descriptive_histogram": ("property",),
    "descriptive_heatmap": ("x", "y", "property", "score"),
    "descriptive_funnel": ("x", "color", "stages"),
    "experiment_z_score": ("x", "y"),
    "experiment_odds_ratio": ("x", "y"),
    "clv_treemap": ("path",),
}


CHART_OPTIONAL_FIELDS: dict[str, tuple[str, ...]] = {
    "line": ("color", "facet_row", "facet_col"),
    "stacked_area": ("facet_row", "facet_col"),
    "bar": ("color", "facet_row", "facet_col"),
    "waterfall": ("color", "facet_row", "facet_col"),
    "pareto": ("color", "facet_row", "facet_col"),
    "treemap": (),
    "heatmap": (),
    "cohort_heatmap": (),
    "scatter": ("color", "size", "animation_frame", "animation_group", "facet_row", "facet_col"),
    "combo": ("color", "facet_row", "facet_col"),
    "interval": (
        "error_y",
        "error_y_lower",
        "error_y_upper",
        "color",
        "facet_row",
        "facet_col",
    ),
    "donut": ("color",),
    "geo_map": ("lat", "lon", "color", "size"),
    "table": ("columns", "group_by"),
    "gauge": ("facet_row", "facet_col"),
    "funnel": ("facet_row", "facet_col"),
    "boxplot": ("color", "facet_row", "facet_col"),
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
    "descriptive_histogram": ("color", "facet_row", "facet_col"),
    "descriptive_funnel": ("facet_row", "facet_col"),
    "experiment_z_score": ("color", "facet_row", "facet_col"),
    "experiment_odds_ratio": ("color", "facet_row", "facet_col"),
    "clv_treemap": ("value", "color"),
}


# Each tuple is one required role. Alternatives are accepted legacy/runtime aliases.
TILE_REQUIRED_ALTERNATIVES: dict[str, tuple[tuple[str, ...], ...]] = {
    "line": (("x",), ("y",)),
    "stacked_area": (("x",), ("y",), ("color",)),
    "bar": (("x",), ("y",)),
    "kpi_card": (("value", "y"),),
    "waterfall": (("x",), ("y",)),
    "pareto": (("x",), ("y",)),
    "treemap": (("path", "x", "names"),),
    "heatmap": (("x",), ("y",), ("color",)),
    "cohort_heatmap": (("x",), ("y",), ("color",)),
    "scatter": (("x",), ("y",)),
    "combo": (("x",), ("y",), ("y2", "line_y")),
    "interval": (("x",), ("y",)),
    "donut": (("names", "x"), ("values", "value", "y")),
    "geo_map": (("locations", "location", "lat"), ("value", "color", "y")),
    "calendar_heatmap": (("date", "x"), ("value", "y")),
    "bar_polar": (("r",), ("theta",), ("color",)),
    "sankey": (("source",), ("target",), ("value",)),
    "gauge": (("value", "y"),),
    "funnel": (("stages",), ("color",)),
    "boxplot": (("x",),),
    "histogram": (("property", "x", "y"),),
    "corr": (("x",), ("y",)),
    "descriptive_line": (("x",), ("property",), ("score",)),
    "descriptive_histogram": (("property", "x", "y"),),
    "descriptive_heatmap": (("x",), ("y",), ("property",), ("score",)),
    "descriptive_funnel": (("x",), ("color",), ("stages",)),
    "experiment_z_score": (("x",), ("y",)),
    "experiment_odds_ratio": (("x",), ("y",)),
    "clv_treemap": (("path", "x", "names"),),
}


def chart_field_controls(chart_kind: str) -> tuple[str, ...]:
    """Return the canonical ordered field controls for one chart kind."""

    required = CHART_REQUIRED_FIELDS.get(chart_kind)
    if required is None:
        return ("x", "y", "color", "facet_row", "facet_col")
    return tuple(dict.fromkeys((*required, *CHART_OPTIONAL_FIELDS.get(chart_kind, ()))))


_TILE_RUNTIME_FIELDS = {
    "animation_frame", "animation_group", "axis_title_standoff", "barmode", "barnorm",
    "bins", "cell_fill_color", "color_continuous_scale", "conditional_formatting",
    "connector_color", "cumulative_label", "date", "default_color", "delta_reference",
    "direction", "error_y", "error_y_lower", "error_y_minus", "error_y_plus",
    "error_y_upper", "facet_col", "facet_column", "facet_row", "facets", "filters",
    "goal_line", "goal_lines", "grain", "group_by", "groupnorm", "header_fill_color",
    "fallback_property", "field", "height", "hole", "horizon", "hover_name", "labels",
    "lat", "layout", "legend_title", "line_y",
    "location", "locationmode", "locations", "log_x", "log_y", "lon", "measure",
    "names", "number_format", "path", "property", "quality_help", "quality_label",
    "r", "reference", "references", "scale_mode", "score", "show_chart_title",
    "show_trend_delta", "showlegend", "size", "sort_by", "sort_direction", "source",
    "stages", "summary_aggregation", "target", "theme", "theta", "top_n", "trend_delta", "value",
    "value_format", "values", "width", "x", "x_axis_title", "y", "y2",
    "y2_axis_title", "y_axis_title", "z",
}
SUPPORTED_TILE_FIELDS = frozenset(
    {
        "id", "title", "metric", "chart", "description", "placement", "kpi",
        *_TILE_RUNTIME_FIELDS,
        *(field for fields in CHART_REQUIRED_FIELDS.values() for field in fields),
        *(field for fields in CHART_OPTIONAL_FIELDS.values() for field in fields),
    }
)


RECIPES: dict[str, ChartRecipe] = {
    "line": ChartRecipe("line", _AGGREGATE_CHART_PROCESSORS),
    "stacked_area": ChartRecipe(
        "stacked_area", _AGGREGATE_CHART_PROCESSORS
    ),
    "bar": ChartRecipe("bar", _AGGREGATE_CHART_PROCESSORS),
    "kpi_card": ChartRecipe("kpi_card", _AGGREGATE_CHART_PROCESSORS),
    "waterfall": ChartRecipe("waterfall", _AGGREGATE_CHART_PROCESSORS),
    "pareto": ChartRecipe("pareto", _AGGREGATE_CHART_PROCESSORS),
    "treemap": ChartRecipe("treemap", _AGGREGATE_CHART_PROCESSORS),
    "heatmap": ChartRecipe("heatmap", _AGGREGATE_CHART_PROCESSORS),
    "cohort_heatmap": ChartRecipe("cohort_heatmap", _AGGREGATE_CHART_PROCESSORS),
    "scatter": ChartRecipe("scatter", _AGGREGATE_CHART_PROCESSORS),
    "combo": ChartRecipe("combo", _AGGREGATE_CHART_PROCESSORS),
    "interval": ChartRecipe("interval", _AGGREGATE_CHART_PROCESSORS),
    "donut": ChartRecipe("donut", _AGGREGATE_CHART_PROCESSORS),
    "geo_map": ChartRecipe("geo_map", _AGGREGATE_CHART_PROCESSORS),
    "table": ChartRecipe("table", _AGGREGATE_CHART_PROCESSORS),
    "calendar_heatmap": ChartRecipe(
        "calendar_heatmap", ("binary_outcome", "score_distribution", "snapshot")
    ),
    "bar_polar": ChartRecipe("bar_polar", ("binary_outcome",)),
    "sankey": ChartRecipe("sankey", _AGGREGATE_CHART_PROCESSORS),
    "gauge": ChartRecipe("gauge", _AGGREGATE_CHART_PROCESSORS),
    "funnel": ChartRecipe("funnel", ("funnel",)),
    "boxplot": ChartRecipe("boxplot", ("numeric_distribution",)),
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
    "descriptive_histogram": ChartRecipe("descriptive_histogram", ("numeric_distribution",)),
    "descriptive_heatmap": ChartRecipe("descriptive_heatmap", ("numeric_distribution",)),
    "descriptive_funnel": ChartRecipe("descriptive_funnel", ("numeric_distribution",)),
    "experiment_z_score": ChartRecipe("experiment_z_score", ("binary_outcome",)),
    "experiment_odds_ratio": ChartRecipe("experiment_odds_ratio", ("binary_outcome",)),
    "clv_treemap": ChartRecipe("clv_treemap", ("entity_lifecycle",)),
}


__all__ = [
    "CHART_OPTIONAL_FIELDS",
    "CHART_REQUIRED_FIELDS",
    "RECIPES",
    "SUPPORTED_TILE_FIELDS",
    "TILE_REQUIRED_ALTERNATIVES",
    "ChartRecipe",
    "chart_field_controls",
]
