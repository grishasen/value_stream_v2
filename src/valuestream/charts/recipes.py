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


RECIPES: dict[str, ChartRecipe] = {
    "line": ChartRecipe("line", ("binary_outcome", "score_distribution", "snapshot")),
    "stacked_area": ChartRecipe(
        "stacked_area", ("binary_outcome", "score_distribution", "snapshot")
    ),
    "bar": ChartRecipe("bar", ("binary_outcome", "snapshot")),
    "kpi_card": ChartRecipe("kpi_card", _AGGREGATE_CHART_PROCESSORS),
    "waterfall": ChartRecipe("waterfall", _AGGREGATE_CHART_PROCESSORS),
    "pareto": ChartRecipe("pareto", _AGGREGATE_CHART_PROCESSORS),
    "treemap": ChartRecipe("treemap", ("binary_outcome", "score_distribution")),
    "heatmap": ChartRecipe("heatmap", ("binary_outcome", "score_distribution")),
    "cohort_heatmap": ChartRecipe("cohort_heatmap", ("binary_outcome", "snapshot")),
    "scatter": ChartRecipe("scatter", ("binary_outcome", "score_distribution")),
    "combo": ChartRecipe("combo", _AGGREGATE_CHART_PROCESSORS),
    "interval": ChartRecipe("interval", _AGGREGATE_CHART_PROCESSORS),
    "donut": ChartRecipe("donut", _AGGREGATE_CHART_PROCESSORS),
    "geo_map": ChartRecipe("geo_map", ("binary_outcome", "score_distribution", "snapshot")),
    "table": ChartRecipe("table", _AGGREGATE_CHART_PROCESSORS),
    "calendar_heatmap": ChartRecipe(
        "calendar_heatmap", ("binary_outcome", "score_distribution", "snapshot")
    ),
    "bar_polar": ChartRecipe("bar_polar", ("binary_outcome",)),
    "sankey": ChartRecipe("sankey", _AGGREGATE_CHART_PROCESSORS),
    "gauge": ChartRecipe("gauge", ("binary_outcome", "snapshot")),
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
    "descriptive_boxplot": ChartRecipe("descriptive_boxplot", ("numeric_distribution",)),
    "descriptive_histogram": ChartRecipe("descriptive_histogram", ("numeric_distribution",)),
    "descriptive_heatmap": ChartRecipe("descriptive_heatmap", ("numeric_distribution",)),
    "descriptive_funnel": ChartRecipe("descriptive_funnel", ("numeric_distribution",)),
    "experiment_z_score": ChartRecipe("experiment_z_score", ("binary_outcome",)),
    "experiment_odds_ratio": ChartRecipe("experiment_odds_ratio", ("binary_outcome",)),
    "clv_treemap": ChartRecipe("clv_treemap", ("entity_lifecycle",)),
}


__all__ = ["RECIPES", "ChartRecipe"]
