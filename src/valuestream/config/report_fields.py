"""Shared report-field discovery for editors and catalog validation."""

from __future__ import annotations

from collections.abc import Iterable

from valuestream.config import model

CALENDAR_FIELDS = ("Hour", "Day", "Week", "Month", "Quarter", "Year")
_TIME_FIELDS_BY_GRAIN = {
    "hourly": ("Hour", "Day", "Week", "Month", "Quarter", "Year"),
    "daily": ("Day", "Week", "Month", "Quarter", "Year"),
    "weekly": ("Week", "Month", "Quarter", "Year"),
    "monthly": ("Month", "Quarter", "Year"),
    "quarterly": ("Quarter", "Year"),
    "yearly": ("Year",),
    "summary": (),
}
SCALAR_STATE_TYPES = frozenset(
    {"count", "value_sum", "min", "max", "pooled_mean", "pooled_variance"}
)
DESCRIPTIVE_SCORES = ("Count", "Sum", "Mean", "Var", "Min", "Max", "p25", "p50", "p75", "p90", "p95")
_LIFECYCLE_OUTPUT_COLUMNS = (
    "customers_count",
    "unique_holdings",
    "lifetime_value",
    "frequency",
    "recency",
    "monetary_value",
    "rfm_segment",
    "rfm_score",
)
_VARIANT_OUTPUT_COLUMNS = (
    "Count",
    "Positives",
    "Negatives",
    "CTR",
    "TestCTR",
    "ControlCTR",
    "TestSampleSize",
    "ControlSampleSize",
    "AbsoluteRateDifference",
    "AbsoluteRateDifference_CI_Low",
    "AbsoluteRateDifference_CI_High",
    "Lift",
    "Lift_Z_Score",
    "Lift_P_Val",
    "StdErr",
)
_CONTINGENCY_OUTPUT_COLUMNS = (
    "Count",
    "Positives",
    "Negatives",
    "chi2_stat",
    "chi2_dof",
    "chi2_p_val",
    "chi2_odds_ratio_stat",
    "chi2_odds_ratio_ci_low",
    "chi2_odds_ratio_ci_high",
    "g_stat",
    "g_dof",
    "g_p_val",
    "g_odds_ratio_stat",
    "g_odds_ratio_ci_low",
    "g_odds_ratio_ci_high",
    "z_score",
    "z_p_val",
)


def metric_output_columns(  # noqa: PLR0911
    metric_name: str, metric: model.Metric
) -> list[str]:
    """Return the user-facing columns produced by one metric."""

    if isinstance(metric, model.LifecycleSummaryMetric):
        return list(metric.outputs or _LIFECYCLE_OUTPUT_COLUMNS)
    if isinstance(metric, model.VariantCompareMetric):
        return _dedupe([*_VARIANT_OUTPUT_COLUMNS, *metric.outputs])
    if isinstance(metric, model.ContingencyTestMetric):
        return _dedupe([*_CONTINGENCY_OUTPUT_COLUMNS, *metric.outputs])
    if isinstance(metric, model.ProportionTestMetric):
        return _dedupe(
            ["Count", "Positives", "Negatives", "z_score", "z_p_val", *metric.outputs]
        )
    if isinstance(metric, model.CurveFromDigestsMetric):
        return [
            metric_name,
            metric.output,
            "roc_auc",
            "average_precision",
            "tpr",
            "fpr",
            "precision",
            "recall",
            "pos_fraction",
        ]
    if isinstance(metric, model.CalibrationFromDigestsMetric):
        return [metric_name, "bin", "predicted", "observed"]
    if isinstance(metric, model.TopKItemsMetric):
        return [metric_name, "item", "count"]
    if isinstance(metric, model.FunnelDropoffMetric):
        return [metric_name]
    return [metric_name]


def descriptive_properties(processor: model.Processor | None) -> list[str]:
    """Return numeric properties addressable by descriptive report charts."""

    if processor is None:
        return []
    properties = (
        list(processor.properties)
        if isinstance(processor, model.NumericDistributionProcessor)
        else [item.column for item in processor.score_properties]
        if isinstance(processor, model.ScoreDistributionProcessor)
        else []
    )
    for state in model.effective_processor_states(processor).values():
        property_name = state.model_dump().get("source_column")
        if property_name and property_name not in properties:
            properties.append(property_name)
    return properties


def distribution_property_metrics(
    catalog: model.Catalog,
    metric_name: str,
) -> dict[str, str]:
    """Map boxplot properties to queryable distribution metrics.

    Only explicit distribution metrics over unconditioned t-digest and KLL
    states from the selected metric's processor qualify.
    """

    metric = catalog.metrics.metrics.get(metric_name)
    if metric is None:
        return {}
    processor = next(
        (
            candidate
            for candidate in catalog.processors.processors
            if candidate.id == metric.processor
        ),
        None,
    )
    if processor is None:
        return {}

    states = model.effective_processor_states(processor)
    property_metrics: dict[str, str] = {}
    for candidate_name, candidate in catalog.metrics.metrics.items():
        if not isinstance(candidate, model.DistributionMetric):
            continue
        if candidate.processor != processor.id:
            continue
        state = states.get(candidate.state)
        if state is None or state.type not in {"tdigest", "kll"}:
            continue
        outcome = str(getattr(state, "outcome", None) or "").casefold()
        if outcome in {"positive", "negative"}:
            continue
        source_column = getattr(state, "source_column", None)
        if source_column:
            property_metrics[source_column] = candidate_name
    return property_metrics


def report_field_options(catalog: model.Catalog, metric_name: str) -> list[str]:
    """Return every field a report tile may address for one metric."""

    metric = catalog.metrics.metrics.get(metric_name)
    processor_by_id = {processor.id: processor for processor in catalog.processors.processors}
    processor = processor_by_id.get(metric.processor) if metric is not None else None
    fields: list[str] = report_axis_options(catalog, metric_name)
    if processor is not None:
        fields.extend(
            name
            for name, state in model.effective_processor_states(processor).items()
            if state.type in SCALAR_STATE_TYPES
        )
        for property_name in descriptive_properties(processor):
            fields.append(property_name)
            fields.extend(
                f"{property_name}_{score}" for score in (*DESCRIPTIVE_SCORES, "Median")
            )
    fields.extend(_metric_and_dependency_outputs(catalog, metric_name))
    return _dedupe(fields)


def report_axis_options(catalog: model.Catalog, metric_name: str) -> list[str]:
    """Return persisted time grains and dimensions usable as chart axes."""

    metric = catalog.metrics.metrics.get(metric_name)
    processor_by_id = {processor.id: processor for processor in catalog.processors.processors}
    processor = processor_by_id.get(metric.processor) if metric is not None else None
    if processor is None:
        return []
    configured_calendar_fields = set(catalog.pipelines.defaults.calendar.grains)
    time_fields = [
        field
        for field in _TIME_FIELDS_BY_GRAIN[processor.time.grain]
        if field in configured_calendar_fields
    ]
    return _dedupe([*time_fields, *processor.group_by])


def _metric_and_dependency_outputs(catalog: model.Catalog, metric_name: str) -> list[str]:
    outputs: list[str] = []
    pending = [metric_name]
    visited: set[str] = set()
    while pending:
        current = pending.pop(0)
        if current in visited:
            continue
        visited.add(current)
        metric = catalog.metrics.metrics.get(current)
        if metric is None:
            continue
        outputs.extend(metric_output_columns(current, metric))
        pending.extend(metric.depends_on)
    return outputs


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


__all__ = [
    "CALENDAR_FIELDS",
    "DESCRIPTIVE_SCORES",
    "SCALAR_STATE_TYPES",
    "descriptive_properties",
    "distribution_property_metrics",
    "metric_output_columns",
    "report_axis_options",
    "report_field_options",
]
