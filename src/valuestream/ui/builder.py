"""Catalog Builder helpers for the Streamlit UI."""

from __future__ import annotations

import ast as py_ast
import math
import os
import re
import secrets
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import polars as pl
import yaml

from valuestream.charts.recipes import RECIPES
from valuestream.config import model
from valuestream.config.loader import CatalogLoadError, load
from valuestream.config.validate import validate_catalog
from valuestream.expr import parser as expr_parser

FILTER_OPERATORS = [
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "contains",
    "starts with",
    "ends with",
    "in",
    "not in",
    "is null",
    "is not null",
]

CALCULATION_MODES = [
    "AST YAML",
    "Polars",
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "Safe Divide",
    "Concat",
    "Coalesce",
    "Date Diff Seconds",
    "Date Part Year",
    "Date Part Month",
    "Date Part Quarter",
    "Date Part Day",
]

STATE_TYPES = [
    "count",
    "value_sum",
    "min",
    "max",
    "pooled_mean",
    "pooled_variance",
    "tdigest",
    "kll",
    "cpc",
    "hll",
    "theta",
    "topk",
]

SCALAR_STATE_TYPES = ("count", "value_sum", "min", "max", "pooled_mean", "pooled_variance")
DIGEST_STATE_TYPES = ("tdigest", "kll")

METRIC_KIND_LABELS = {
    "formula": "Formula / state passthrough",
    "approx_distinct_count": "Approx distinct count",
    "topk_items": "Top-K frequent items",
    "tdigest_quantile": "Digest quantile",
    "variant_compare": "Variant comparison",
    "curve_from_digests": "ROC / average precision",
    "calibration_from_digests": "Calibration curve",
    "contingency_test": "Contingency test",
    "proportion_test": "Proportion test",
    "lifecycle_summary": "Lifecycle summary",
    "set_op": "Set operation",
    "funnel_dropoff": "Funnel drop-off",
}

METRIC_KIND_HELP = {
    "formula": "Use scalar states such as Count, Positives, Mean, or MRR.",
    "approx_distinct_count": "Estimate the cardinality of a CPC, HLL, or Theta state.",
    "topk_items": "Return frequent values from a Top-K sketch state.",
    "tdigest_quantile": "Read a percentile from a t-digest or KLL state.",
    "variant_compare": "Compare test and control outcome rates by a variant column.",
    "curve_from_digests": "Compute ROC AUC or average precision from positive and negative score digests.",
    "calibration_from_digests": "Build calibration bins from one score property's positive and negative digests.",
    "contingency_test": "Run chi-square, G, or z tests across variant outcome counts.",
    "proportion_test": "Expose proportion-test outputs for binary outcome counts.",
    "lifecycle_summary": "Produce RFM and lifetime-value summary columns.",
    "set_op": "Estimate union, intersection, or difference across theta-set states.",
    "funnel_dropoff": "Compute drop-off count or rate between funnel stages.",
}

_METRIC_KIND_ORDER = (
    "curve_from_digests",
    "calibration_from_digests",
    "tdigest_quantile",
    "lifecycle_summary",
    "funnel_dropoff",
    "set_op",
    "topk_items",
    "variant_compare",
    "contingency_test",
    "proportion_test",
    "approx_distinct_count",
    "formula",
)

_LIFECYCLE_OUTPUT_COLUMNS = [
    "customers_count",
    "unique_holdings",
    "lifetime_value",
    "frequency",
    "recency",
    "monetary_value",
    "rfm_segment",
    "rfm_score",
]
_VARIANT_OUTPUT_COLUMNS = [
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
]
_CONTINGENCY_OUTPUT_COLUMNS = [
    "Count",
    "Positives",
    "Negatives",
    "chi2_stat",
    "chi2_dof",
    "chi2_p_val",
    "g_stat",
    "g_dof",
    "g_p_val",
    "z_score",
    "z_p_val",
]
DESCRIPTIVE_SCALAR_SUFFIXES = ("Count", "Sum", "Mean", "Var", "Min", "Max")
DESCRIPTIVE_QUANTILE_SCORES = ("p25", "p50", "p75", "p90", "p95")
_DESCRIPTIVE_STATE_SUFFIXES = (
    *DESCRIPTIVE_SCALAR_SUFFIXES,
    "Median",
    *DESCRIPTIVE_QUANTILE_SCORES,
    "tdigest",
    "kll",
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
    "boxplot": ("x", "y"),
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
    "descriptive_boxplot": ("x", "property"),
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
    "descriptive_boxplot": ("color", "facet_row", "facet_col"),
    "descriptive_histogram": ("color", "facet_row", "facet_col"),
    "descriptive_funnel": ("facet_row", "facet_col"),
    "experiment_z_score": ("color", "facet_row", "facet_col"),
    "experiment_odds_ratio": ("color", "facet_row", "facet_col"),
    "clv_treemap": ("value", "color"),
}

CHART_SETTING_FIELDS = {
    "description",
    "placement",
    "kpi",
    "scale_mode",
    "value_format",
    "goal_line",
    "reference",
    "references",
    "show_trend_delta",
    "trend_delta",
    "labels",
    "x_axis_title",
    "y_axis_title",
    "legend_title",
    "axis_title_standoff",
    "sort_by",
    "sort_direction",
    "top_n",
    "barmode",
    "barnorm",
    "conditional_formatting",
    "default_color",
}


def chart_field_controls(chart_kind: str) -> tuple[str, ...]:
    """Return ordered tile field controls, always starting with required fields."""
    required = CHART_REQUIRED_FIELDS.get(chart_kind)
    if required is None:
        return ("x", "y", "color", "facet_row", "facet_col")
    return tuple(_dedupe([*required, *CHART_OPTIONAL_FIELDS.get(chart_kind, ())]))


MINIMUM_CATALOG_FILES = ("pipelines.yaml", "processors.yaml", "metrics.yaml", "dashboards.yaml")


def blank_default_row() -> dict[str, Any]:
    """Return one empty default-value editor row."""
    return {"Field": "", "Default Value": "", "Enabled": True}


def blank_filter_row() -> dict[str, Any]:
    """Return one empty filter editor row."""
    return {"Field": "", "Operator": "==", "Value": "", "Enabled": True}


def blank_calculated_row() -> dict[str, Any]:
    """Return one empty derived-field editor row."""
    return {
        "Name": "",
        "Mode": "AST YAML",
        "Left": "",
        "Right Kind": "Field",
        "Right": "",
        "Expression": "",
        "Enabled": True,
    }


def normalize_editor_rows(frame: Any) -> list[dict[str, Any]]:
    """Convert Streamlit/Pandas/Polars editor output into plain row dicts."""
    if hasattr(frame, "to_dicts"):
        rows = frame.to_dicts()
    elif hasattr(frame, "to_dict"):
        rows = frame.to_dict("records")
    else:
        rows = list(frame or [])
    return [
        {key: ("" if _is_missing_editor_value(value) else value) for key, value in row.items()}
        for row in rows
    ]


def editor_frame(
    rows: list[dict[str, Any]],
    columns: list[str],
    blank_row_factory: Callable[[], dict[str, Any]],
) -> pl.DataFrame:
    """Return a stable typed frame for Streamlit editable row tables."""
    editor_rows = normalize_editor_rows(rows) or [blank_row_factory()]
    return pl.DataFrame(
        {
            column: [
                bool(row.get(column, False))
                if column == "Enabled"
                else _editor_text_value(row.get(column, ""))
                for row in editor_rows
            ]
            for column in columns
        }
    )


def _editor_text_value(value: Any) -> str:
    if _is_missing_editor_value(value):
        return ""
    return str(value)


def default_rows_from_values(values: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a default-values mapping into editor rows."""
    rows = [
        {"Field": key, "Default Value": value, "Enabled": True}
        for key, value in sorted(values.items(), key=lambda item: str(item[0]).casefold())
    ]
    return rows or [blank_default_row()]


def default_rows_with_fields(
    default_rows: list[dict[str, Any]],
    fields: Iterable[Any],
) -> list[dict[str, Any]]:
    """Append blank default-value rows for selected fields not already present."""
    rows = normalize_editor_rows(default_rows)
    field_names = [str(field or "").strip() for field in fields]
    field_names = [field for field in field_names if field]
    if field_names:
        rows = [row for row in rows if _default_row_has_content(row)]
    existing = {str(row.get("Field", "")).strip() for row in rows}
    for field in field_names:
        if field in existing:
            continue
        row = blank_default_row()
        row["Field"] = field
        rows.append(row)
        existing.add(field)
    return rows or [blank_default_row()]


def _default_row_has_content(row: dict[str, Any]) -> bool:
    return bool(
        str(row.get("Field", "")).strip()
        or str(row.get("Default Value", "")).strip()
        or row.get("Enabled") is False
    )


def build_default_values(default_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a source defaults map from editor rows."""
    out: dict[str, Any] = {}
    for row in default_rows:
        if not row.get("Enabled", True):
            continue
        field = str(row.get("Field", "")).strip()
        if field:
            out[field] = _safe_literal(row.get("Default Value", ""))
    return out


def filter_rows_from_expression(expression: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Best-effort conversion from a simple AST expression to editable rule rows."""
    if expression is None:
        return [blank_filter_row()]
    if expression.get("op") == "and" and isinstance(expression.get("args"), list):
        rows: list[dict[str, Any]] = []
        for arg in expression["args"]:
            parsed = _filter_row_from_expression(arg)
            if parsed is None:
                return None
            rows.append(parsed)
        return rows or [blank_filter_row()]
    parsed = _filter_row_from_expression(expression)
    return [parsed] if parsed is not None else None


def compile_filter_rows(filter_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Compile rule rows into the Value Stream expression AST dictionary."""
    expressions = [_compile_filter_row(row) for row in filter_rows]
    expressions = [expression for expression in expressions if expression is not None]
    if not expressions:
        return None
    if len(expressions) == 1:
        return expressions[0]
    return {"op": "and", "args": expressions}


def parse_expression_yaml(text: str) -> dict[str, Any]:
    """Parse and validate one YAML/JSON expression AST."""
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("expression must be a YAML mapping")
    return expr_parser.to_dict(expr_parser.parse(loaded))


def calculated_rows_from_source(source: model.Source) -> list[dict[str, Any]]:
    """Return editor rows for source-level derived columns."""
    rows: list[dict[str, Any]] = []
    for transform in source.transforms:
        if not isinstance(transform, model.DeriveColumn):
            continue
        expression = expr_parser.to_dict(transform.expression)
        mode = "Polars" if set(expression) == {"polars"} else "AST YAML"
        rows.append(
            {
                "Name": transform.output,
                "Mode": mode,
                "Left": "",
                "Right Kind": "Field",
                "Right": "",
                "Expression": (
                    str(expression["polars"])
                    if mode == "Polars"
                    else yaml.safe_dump(expression, sort_keys=False).strip()
                ),
                "Enabled": True,
            }
        )
    return rows or [blank_calculated_row()]


def calculated_rows_for_editor(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize calculated-field rows to the current editor column shape."""
    normalized: list[dict[str, Any]] = []
    for row in rows or [blank_calculated_row()]:
        mode = str(row.get("Mode", "") or "AST YAML").strip()
        normalized.append(
            {
                "Name": str(row.get("Name", "") or ""),
                "Mode": mode,
                "Left": str(row.get("Left", "") or ""),
                "Right Kind": str(row.get("Right Kind", "") or "Field"),
                "Right": str(row.get("Right", "") or ""),
                "Expression": str(row.get("Expression") or row.get("Expression YAML", "") or ""),
                "Enabled": bool(row.get("Enabled", True)),
            }
        )
    return normalized


def build_derive_column_transforms(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build ``derive_column`` transforms from calculated-field editor rows."""
    transforms: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("Enabled", True):
            continue
        name = str(row.get("Name", "")).strip()
        expression = _calculated_expression(row)
        if not name or expression is None:
            continue
        transforms.append(
            {
                "kind": "derive_column",
                "output": name,
                "expression": expression,
            }
        )
    return transforms


def _calculated_expression(row: dict[str, Any]) -> dict[str, Any] | None:
    mode = str(row.get("Mode", "") or "AST YAML").strip()
    expression_text = str(row.get("Expression") or row.get("Expression YAML", "")).strip()
    if mode == "Polars":
        return {"polars": expression_text} if expression_text else None
    if mode == "AST YAML":
        return parse_expression_yaml(expression_text) if expression_text else None
    return _builder_calculation_expression(mode, row)


def _builder_calculation_expression(mode: str, row: dict[str, Any]) -> dict[str, Any] | None:  # noqa: PLR0911
    left = str(row.get("Left", "")).strip()
    right = str(row.get("Right", "")).strip()
    right_kind = str(row.get("Right Kind", "Field")).strip()
    if not left:
        return None
    left_expr = {"col": left}
    if mode in {"Date Part Year", "Date Part Month", "Date Part Quarter", "Date Part Day"}:
        return {
            "op": "date_part",
            "unit": mode.removeprefix("Date Part ").casefold(),
            "arg": left_expr,
        }
    if not right:
        return None
    right_expr = {"lit": _safe_literal(right)} if right_kind == "Literal" else {"col": right}
    if mode == "Add":
        return {"op": "add", "args": [left_expr, right_expr]}
    if mode == "Subtract":
        return {"op": "sub", "args": [left_expr, right_expr]}
    if mode == "Multiply":
        return {"op": "mul", "args": [left_expr, right_expr]}
    if mode == "Divide":
        return {"op": "div", "args": [left_expr, right_expr]}
    if mode == "Safe Divide":
        return {"op": "safe_div", "num": left_expr, "den": right_expr}
    if mode == "Concat":
        return {"op": "concat", "args": [left_expr, right_expr], "sep": ""}
    if mode == "Coalesce":
        return {"op": "coalesce", "args": [left_expr, right_expr]}
    if mode == "Date Diff Seconds":
        return {"op": "date_diff", "unit": "seconds", "end": left_expr, "start": right_expr}
    return None


def expression_yaml(expression: Any) -> str:
    """Render an expression AST as YAML."""
    if expression is None:
        return ""
    if hasattr(expression, "model_dump"):
        expression = expr_parser.to_dict(expression)
    return yaml.safe_dump(expression, sort_keys=False).strip()


def source_to_dict(source: model.Source) -> dict[str, Any]:
    """Serialize a source model to a YAML-ready dict."""
    return source.model_dump(mode="json", by_alias=True, exclude_none=True)


def processor_to_dict(processor: model.Processor) -> dict[str, Any]:
    """Serialize a processor model to user-facing YAML-ready dict."""
    data = processor.model_dump(mode="json", by_alias=True, exclude_none=True)
    group_by = data.pop("group_by", None)
    if group_by:
        data["dimensions"] = group_by
    if not processor.states:
        data.pop("states", None)
    return data


def metric_to_dict(metric: model.Metric) -> dict[str, Any]:
    """Serialize a metric model to a concise YAML-ready dict."""
    data = metric.model_dump(mode="json", by_alias=True, exclude_none=True)
    if not data.get("description"):
        data.pop("description", None)
    if not data.get("depends_on"):
        data.pop("depends_on", None)
    return data


def source_defaults(source: model.Source) -> dict[str, Any]:
    """Return effective source defaults from source-level and defaults transforms."""
    values = dict(source.defaults)
    for transform in source.transforms:
        if isinstance(transform, model.Defaults):
            values.update(transform.values)
    return values


def first_filter_expression(source: model.Source | model.Processor) -> dict[str, Any] | None:
    """Return the first source or processor filter expression as a plain dict."""
    transforms = getattr(source, "transforms", None)
    if transforms is not None:
        for transform in transforms:
            if isinstance(transform, model.FilterTransform):
                return expr_parser.to_dict(transform.expression)
        return None
    filter_expression = getattr(source, "filter", None)
    if filter_expression is not None:
        return expr_parser.to_dict(filter_expression)
    return None


def chart_recipe_summary(
    catalog: model.Catalog,
    metric_name: str,
    chart_kind: str,
) -> dict[str, Any]:
    """Return compact chart-recipe metadata for the Builder UI."""
    processor = processor_for_metric(catalog, metric_name)
    recipe = RECIPES.get(chart_kind)
    return {
        "Required Fields": ", ".join(CHART_REQUIRED_FIELDS.get(chart_kind, ())) or "none",
        "Compatible Processors": ", ".join(recipe.allowed_processor_kinds) if recipe else "unknown",
        "Processor": f"{processor.id} ({processor.kind})" if processor else "unknown",
        "Default Grain": _default_grain(processor),
        "Output Columns": ", ".join(
            metric_output_columns(metric_name, catalog.metrics.metrics[metric_name])
        ),
    }


def state_columns(processor: model.Processor) -> list[str]:
    """Return public state columns available to formula metrics."""
    return list(model.effective_processor_states(processor))


def state_columns_by_type(processor: model.Processor, *state_types: str) -> list[str]:
    """Return processor state columns matching one or more state types."""
    wanted = set(state_types)
    return [
        name
        for name, spec in model.effective_processor_states(processor).items()
        if spec.type in wanted
    ]


def scalar_state_columns(processor: model.Processor) -> list[str]:
    """Return state columns that can safely participate in formulas."""
    return state_columns_by_type(processor, *SCALAR_STATE_TYPES)


def metric_kind_label(kind: str) -> str:
    """Return a user-facing label for a metric kind."""
    return METRIC_KIND_LABELS.get(kind, kind.replace("_", " ").title())


def metric_kind_help(kind: str) -> str:
    """Return a short description for a metric kind."""
    return METRIC_KIND_HELP.get(kind, "")


def metric_kind_options(processor: model.Processor) -> list[str]:
    """Return executable metric kinds that fit a processor's states and semantics."""
    states = model.effective_processor_states(processor)
    state_types = {name: spec.type for name, spec in states.items()}
    scalar_states = [name for name, kind in state_types.items() if kind in SCALAR_STATE_TYPES]
    cardinality_states = [
        name for name, kind in state_types.items() if kind in {"cpc", "hll", "theta"}
    ]
    theta_states = [name for name, kind in state_types.items() if kind == "theta"]
    topk_states = [name for name, kind in state_types.items() if kind == "topk"]
    digest_states = [name for name, kind in state_types.items() if kind in DIGEST_STATE_TYPES]
    kinds: list[str] = []

    def add(kind: str, enabled: bool) -> None:
        if enabled and kind not in kinds:
            kinds.append(kind)

    extra = dict(processor.model_extra or {})
    has_outcome_counts = {"Positives", "Negatives"} <= set(states)
    has_variant = bool(extra.get("variant_column")) and has_outcome_counts
    stages = funnel_stage_names(processor)

    if processor.kind == "score_distribution":
        add("curve_from_digests", len(digest_states) >= 2)
        add("calibration_from_digests", len(digest_states) >= 2)
        add("tdigest_quantile", bool(digest_states))
        add("approx_distinct_count", bool(cardinality_states))
        add("formula", bool(scalar_states))
    elif processor.kind == "numeric_distribution":
        add("tdigest_quantile", bool(digest_states))
        add("approx_distinct_count", bool(cardinality_states))
        add("formula", bool(scalar_states))
    elif processor.kind == "binary_outcome":
        add("variant_compare", has_variant)
        add("contingency_test", has_variant)
        add("proportion_test", has_outcome_counts)
        add("approx_distinct_count", bool(cardinality_states))
        add("formula", bool(scalar_states))
    elif processor.kind == "entity_lifecycle":
        add("lifecycle_summary", True)
        add("approx_distinct_count", bool(cardinality_states))
        add("formula", bool(scalar_states))
    elif processor.kind == "entity_set":
        add("set_op", len(theta_states) >= 2)
        add("approx_distinct_count", bool(cardinality_states))
        add("formula", bool(scalar_states))
    elif processor.kind == "funnel":
        add("funnel_dropoff", len(stages) >= 2)
        add("approx_distinct_count", bool(cardinality_states))
        add("formula", bool(scalar_states))
    elif processor.kind == "snapshot":
        add("approx_distinct_count", bool(cardinality_states))
        add("formula", bool(scalar_states))

    add("topk_items", bool(topk_states))
    return sorted(kinds, key=_METRIC_KIND_ORDER.index)


def default_metric_kind(processor: model.Processor) -> str | None:
    """Return the preferred metric kind for a processor."""
    options = metric_kind_options(processor)
    return options[0] if options else None


def default_metric_name(processor: model.Processor, metric_kind: str) -> str:
    """Return a concise default metric id for a processor and metric kind."""
    suffix = {
        "formula": "metric",
        "approx_distinct_count": "unique",
        "topk_items": "topk",
        "tdigest_quantile": "median",
        "variant_compare": "lift",
        "curve_from_digests": "roc_auc",
        "calibration_from_digests": "calibration",
        "contingency_test": "significance",
        "lifecycle_summary": "summary",
        "set_op": "overlap",
        "funnel_dropoff": "dropoff",
    }.get(metric_kind, metric_kind)
    return f"{processor.id}_{suffix}"


def default_curve_digest_states(
    processor: model.Processor,
    *,
    final: bool = False,
) -> tuple[str, str] | None:
    """Return the best positive/negative digest-state pair for curve metrics."""
    digest_states = state_columns_by_type(processor, "tdigest")
    if not digest_states:
        return None
    pairs = digest_state_pair_options(processor)
    if pairs:
        return pairs[0][1], pairs[0][2]
    positives = _states_with_outcome(processor, "positive")
    negatives = _states_with_outcome(processor, "negative")
    if positives and negatives:
        return positives[0], negatives[0]
    named_positive = next((state for state in digest_states if "positive" in state.lower()), "")
    named_negative = next((state for state in digest_states if "negative" in state.lower()), "")
    if named_positive and named_negative:
        return named_positive, named_negative
    return None


def digest_state_pair_options(processor: model.Processor) -> list[tuple[str, str, str]]:
    """Return selectable positive/negative t-digest pairs grouped by score property."""
    states = {
        name: spec.model_dump(mode="json", exclude_none=True)
        for name, spec in model.effective_processor_states(processor).items()
    }
    processor_def = {**dict(processor.model_extra or {}), "states": states}
    return digest_pair_options_from_definition(processor_def)


def digest_pair_options_from_definition(
    processor_def: dict[str, Any],
) -> list[tuple[str, str, str]]:
    """Return positive/negative t-digest pairs from a processor definition dict."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for state_name, spec in state_spec_definitions(processor_def).items():
        if str(spec.get("type", "") or "") != "tdigest":
            continue
        role = _digest_outcome_role(state_name, spec)
        if role not in {"positive", "negative"}:
            continue
        property_name = _digest_source_property(processor_def, state_name, spec)
        family = _digest_state_family(state_name, role, property_name)
        buckets = grouped.setdefault(property_name, [])
        bucket = next((item for item in buckets if item.get("family") == family), None)
        if bucket is None:
            bucket = {"family": family}
            buckets.append(bucket)
        bucket[role] = state_name
    pairs: list[tuple[str, str, str]] = []
    for property_name, buckets in grouped.items():
        for bucket in buckets:
            positive_state = bucket.get("positive")
            negative_state = bucket.get("negative")
            if not positive_state or not negative_state:
                continue
            family = bucket.get("family", "")
            label = (
                f"{property_name} ({family})"
                if len(buckets) > 1 and family and family != property_name
                else property_name
            )
            pairs.append((label, positive_state, negative_state))
    return pairs


def state_spec_definitions(processor_def: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return state specs by name from dict- or list-form processor states."""
    specs: dict[str, dict[str, Any]] = {}
    raw_states = processor_def.get("states")
    if isinstance(raw_states, dict):
        for name, spec in raw_states.items():
            specs[str(name)] = dict(spec) if isinstance(spec, dict) else {"type": spec}
    elif isinstance(raw_states, list):
        for spec in raw_states:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name") or spec.get("id") or spec.get("state") or spec.get("output")
            if not name:
                continue
            specs[str(name)] = {
                key: value
                for key, value in spec.items()
                if key not in {"name", "id", "state", "output", "enabled"}
            }
    return specs


def score_properties_from_definition(processor_def: dict[str, Any]) -> list[str]:
    """Return score property names from a score-distribution definition dict."""
    properties = string_list(processor_def.get("score_properties"))
    if properties:
        return _dedupe(properties)
    score_columns = processor_def.get("score_columns")
    if isinstance(score_columns, dict):
        legacy = _dedupe([str(value) for value in score_columns.values() if value])
        if legacy:
            return legacy
    if isinstance(score_columns, list):
        columns = _dedupe(
            [
                str(item.get("column", "") or "")
                for item in score_columns
                if isinstance(item, dict) and item.get("column")
            ]
        )
        if columns:
            return columns
    scores = processor_def.get("scores")
    if isinstance(scores, dict):
        legacy = _dedupe([str(value) for value in scores.values() if value])
        if legacy:
            return legacy
    return ["Propensity"]


def funnel_stage_names(processor: model.Processor) -> list[str]:
    """Return configured or inferred stage names for a funnel processor."""
    extra = dict(processor.model_extra or {})
    raw_stages = extra.get("stages", [])
    stages: list[str] = []
    if isinstance(raw_stages, list):
        for item in raw_stages:
            if isinstance(item, dict) and item.get("name"):
                stages.append(str(item["name"]))
    if stages:
        return _dedupe(stages)
    suffix = "_Count"
    return [
        name[: -len(suffix)]
        for name in state_columns_by_type(processor, "count")
        if name.endswith(suffix)
    ]


def merge_stage_definitions(
    existing_stages: Any,
    stage_names: list[str],
) -> list[dict[str, Any]]:
    """Merge edited stage names with existing funnel stage definitions.

    Stages that keep their name retain their full definition (including the
    ``when`` expression and any extra keys); new names produce name-only
    stages that still need a ``when`` expression before the funnel can run.
    """
    existing: dict[str, dict[str, Any]] = {}
    if isinstance(existing_stages, list):
        for item in existing_stages:
            if isinstance(item, dict) and item.get("name"):
                existing[str(item["name"])] = dict(item)
            elif isinstance(item, str):
                existing.setdefault(item, {"name": item})
    return [{**existing.get(name, {}), "name": name} for name in _dedupe(stage_names)]


def stage_names_missing_when(stages: Any) -> list[str]:
    """Return names of funnel stages that lack a ``when`` expression."""
    if not isinstance(stages, list):
        return []
    missing: list[str] = []
    for item in stages:
        if isinstance(item, dict) and item.get("name") and not item.get("when"):
            missing.append(str(item["name"]))
        elif isinstance(item, str):
            missing.append(item)
    return _dedupe(missing)


def processor_for_metric(catalog: model.Catalog, metric_name: str) -> model.Processor | None:
    """Resolve the processor that backs a metric."""
    metric = catalog.metrics.metrics.get(metric_name)
    if metric is None:
        return None
    return next(
        (processor for processor in catalog.processors.processors if processor.id == metric.source),
        None,
    )


def chart_choices_for_metric(catalog: model.Catalog, metric_name: str) -> list[str]:
    """Return chart kinds compatible with the metric's processor."""
    processor = processor_for_metric(catalog, metric_name)
    if processor is None:
        return sorted(RECIPES)
    metric = catalog.metrics.metrics[metric_name]
    choices = [
        name for name, recipe in RECIPES.items() if processor.kind in recipe.allowed_processor_kinds
    ]
    if metric.kind != "calibration_from_digests":
        choices = [name for name in choices if name != "calibration_curve"]
    curve_charts = {"roc_curve", "precision_recall_curve", "gain_curve", "lift_curve"}
    if metric.kind != "curve_from_digests":
        choices = [name for name in choices if name not in curve_charts]
    return sorted(choices)


def metric_output_columns(metric_name: str, metric: model.Metric) -> list[str]:
    """Best-effort output column names for a metric."""
    if isinstance(metric, model.LifecycleSummaryMetric):
        return metric.outputs or list(_LIFECYCLE_OUTPUT_COLUMNS)
    if isinstance(metric, model.VariantCompareMetric):
        return metric.outputs or list(_VARIANT_OUTPUT_COLUMNS)
    if isinstance(metric, model.ContingencyTestMetric):
        return metric.outputs or list(_CONTINGENCY_OUTPUT_COLUMNS)
    if isinstance(metric, model.ProportionTestMetric):
        return metric.outputs or [metric_name]
    if isinstance(metric, model.FunnelDropoffMetric):
        return [metric_name]
    return [metric_name]


def _scalar_state_columns(processor: model.Processor | None) -> list[str]:
    if processor is None:
        return []
    return [
        name
        for name, state in model.effective_processor_states(processor).items()
        if state.type in SCALAR_STATE_TYPES
    ]


def _descriptive_property_from_state(state_name: str) -> str | None:
    for suffix in _DESCRIPTIVE_STATE_SUFFIXES:
        marker = f"_{suffix}"
        if state_name.endswith(marker):
            return state_name[: -len(marker)]
    return None


def _preferred_size_column(processor: model.Processor | None, outputs: list[str]) -> str | None:
    candidates = [*_scalar_state_columns(processor), *outputs]
    for name in ("Count", "Positives", "Negatives"):
        if name in candidates:
            return name
    return None


def chart_field_options(catalog: model.Catalog, metric_name: str) -> list[str]:
    """Return columns that are sensible for chart field selectors."""
    processor = processor_for_metric(catalog, metric_name)
    metric = catalog.metrics.metrics.get(metric_name)
    fields: list[str] = ["Day", "Month", "Quarter", "Year"]
    if processor is not None:
        fields.extend(processor.group_by)
        fields.extend(_scalar_state_columns(processor))
    if metric is not None:
        fields.extend(metric_output_columns(metric_name, metric))
    return _dedupe(fields)


def descriptive_property_options(catalog: model.Catalog, metric_name: str) -> list[str]:
    """Return numeric distribution properties available to descriptive charts."""
    processor = processor_for_metric(catalog, metric_name)
    if processor is None:
        return []
    extra = dict(processor.model_extra or {})
    properties = [str(item) for item in extra.get("properties", []) if item not in (None, "")]
    for state_name in model.effective_processor_states(processor):
        prop = _descriptive_property_from_state(state_name)
        if prop and prop not in properties:
            properties.append(prop)
    return properties


def descriptive_score_options(
    catalog: model.Catalog,
    metric_name: str,
    property_name: str,
) -> list[str]:
    """Return descriptive statistic suffixes for one score/property column."""
    processor = processor_for_metric(catalog, metric_name)
    if processor is None or not property_name:
        return []
    states = model.effective_processor_states(processor)
    scores = [
        suffix for suffix in DESCRIPTIVE_SCALAR_SUFFIXES if f"{property_name}_{suffix}" in states
    ]
    if f"{property_name}_Median" in states:
        scores.append("p50")
    if f"{property_name}_tdigest" in states or f"{property_name}_kll" in states:
        for score in DESCRIPTIVE_QUANTILE_SCORES:
            if score not in scores:
                scores.append(score)
    return scores or ["Mean"]


def _default_descriptive_property(metric: model.Metric, properties: list[str]) -> str:
    if isinstance(metric, model.TdigestQuantileMetric):
        prop = _descriptive_property_from_state(metric.state)
        if prop and prop in properties:
            return prop
    return properties[0] if properties else "value"


def _default_descriptive_score(metric: model.Metric, scores: list[str]) -> str:
    if isinstance(metric, model.TdigestQuantileMetric):
        quantile_scores = {
            0.25: "p25",
            0.5: "p50",
            0.75: "p75",
            0.9: "p90",
            0.95: "p95",
        }
        score = quantile_scores.get(float(metric.quantile))
        if score in scores:
            return score
    return "Mean" if "Mean" in scores else scores[0] if scores else "Mean"


def default_tile_fields(  # noqa: PLR0912, PLR0915
    catalog: model.Catalog, metric_name: str, chart_kind: str
) -> dict[str, Any]:
    """Return default chart fields for a metric/chart pair."""
    metric = catalog.metrics.metrics[metric_name]
    processor = processor_for_metric(catalog, metric_name)
    outputs = metric_output_columns(metric_name, metric)
    group_by: list[str] = []
    grains: list[str] = []
    if processor is not None:
        group_by = list(processor.group_by)
        grains = processor.grains
    time_x = "Day" if "daily" in grains else "Month" if "monthly" in grains else None
    first_dim = group_by[0] if group_by else None
    second_dim = group_by[1] if len(group_by) > 1 else first_dim
    first_output = outputs[0]
    second_output = outputs[1] if len(outputs) > 1 else first_output

    fields: dict[str, Any]
    if chart_kind == "line":
        fields = {"x": time_x or first_dim or "Month", "y": first_output, "color": first_dim}
    elif chart_kind == "stacked_area":
        fields = {"x": time_x or "Month", "y": first_output, "color": first_dim}
    elif chart_kind == "bar":
        fields = {"x": first_dim or "Month", "y": first_output}
    elif chart_kind == "kpi_card":
        fields = {"value": first_output}
    elif chart_kind in {"waterfall", "pareto"}:
        fields = {"x": first_dim or "Month", "y": first_output}
    elif chart_kind == "heatmap":
        fields = {
            "x": first_dim or "Month",
            "y": second_dim or first_output,
            "color": first_output,
        }
    elif chart_kind == "cohort_heatmap":
        fields = {"x": time_x or "Month", "y": first_dim or "Cohort", "color": first_output}
    elif chart_kind == "scatter":
        fields = {"x": first_output, "y": second_output, "color": first_dim}
        size = _preferred_size_column(processor, outputs)
        if size:
            fields["size"] = size
    elif chart_kind == "combo":
        fields = {"x": time_x or first_dim or "Month", "y": first_output, "y2": second_output}
    elif chart_kind == "interval":
        fields = {"x": first_dim or time_x or "Month", "y": first_output}
    elif chart_kind == "donut":
        fields = {"names": first_dim or "Segment", "values": first_output}
    elif chart_kind == "geo_map":
        fields = {"locations": first_dim or "Country", "value": first_output}
    elif chart_kind == "table":
        fields = {"columns": [*(group_by[:3]), first_output]}
    elif chart_kind == "calendar_heatmap":
        fields = {"date": time_x or "Day", "value": first_output}
    elif chart_kind == "bar_polar":
        theta = first_dim or time_x or "Month"
        fields = {"r": first_output, "theta": theta, "color": second_dim or theta}
    elif chart_kind == "treemap":
        fields = {"path": group_by[:3] or [first_output], "color": first_output}
    elif chart_kind == "sankey":
        fields = {
            "source": first_dim or "source",
            "target": second_dim or "target",
            "value": first_output,
        }
    elif chart_kind == "gauge":
        fields = {"value": first_output}
        if first_dim:
            fields["facet_row"] = first_dim
        if len(group_by) > 1:
            fields["facet_col"] = group_by[1]
    elif chart_kind.startswith("descriptive_"):
        properties = descriptive_property_options(catalog, metric_name)
        prop = _default_descriptive_property(metric, properties)
        scores = descriptive_score_options(catalog, metric_name, prop)
        score = _default_descriptive_score(metric, scores)
        if chart_kind == "descriptive_line":
            fields = {"x": time_x or first_dim or "Month", "property": prop, "score": score}
            if first_dim:
                fields["color"] = first_dim
        elif chart_kind == "descriptive_boxplot":
            fields = {"x": first_dim or time_x or "Month", "property": prop}
        elif chart_kind == "descriptive_histogram":
            fields = {"property": prop}
        elif chart_kind == "descriptive_heatmap":
            fields = {
                "x": first_dim or time_x or "Month",
                "y": second_dim or first_dim or "Month",
                "property": prop,
                "score": score,
            }
        elif chart_kind == "descriptive_funnel":
            fields = {"x": first_dim or time_x or "Month", "color": second_dim or first_dim}
        else:
            fields = {"property": prop, "score": score}
    elif chart_kind == "calibration_curve":
        fields = {}
    elif chart_kind in {"roc_curve", "precision_recall_curve", "gain_curve", "lift_curve"}:
        fields = {"color": first_dim} if first_dim else {}
    elif chart_kind == "rfm_density":
        fields = {"x": "recency", "y": "frequency"}
    elif chart_kind == "corr":
        fields = {"x": "frequency", "y": "monetary_value"}
    else:
        fields = {"y": first_output}
    return fields


def build_formula_metric(
    processor_id: str,
    numerator: str,
    denominator: str | None = None,
) -> dict[str, Any]:
    """Create a formula metric definition."""
    expression: dict[str, Any]
    if denominator:
        expression = {
            "op": "safe_div",
            "num": {"col": numerator},
            "den": {"col": denominator},
        }
    else:
        expression = {"col": numerator}
    return {"source": processor_id, "kind": "formula", "expression": expression}


def build_approx_distinct_metric(processor_id: str, state: str) -> dict[str, Any]:
    """Create an approximate-distinct-count metric definition."""
    return {"source": processor_id, "kind": "approx_distinct_count", "state": state}


def build_topk_items_metric(
    processor_id: str,
    state: str,
    *,
    limit: int = 10,
    error_type: str = "NO_FALSE_POSITIVES",
) -> dict[str, Any]:
    """Create a Top-K frequent-items metric definition."""
    metric: dict[str, Any] = {
        "source": processor_id,
        "kind": "topk_items",
        "state": state,
        "limit": limit,
    }
    if error_type and error_type != "NO_FALSE_POSITIVES":
        metric["error_type"] = error_type
    return metric


def build_tdigest_quantile_metric(
    processor_id: str,
    state: str,
    quantile: float,
) -> dict[str, Any]:
    """Create a digest-quantile metric definition."""
    return {
        "source": processor_id,
        "kind": "tdigest_quantile",
        "state": state,
        "quantile": quantile,
    }


def build_curve_from_digests_metric(
    processor_id: str,
    positive_state: str,
    negative_state: str,
    output: str = "roc_auc",
) -> dict[str, Any]:
    """Create a ROC/Average Precision metric from positive and negative digests."""
    return {
        "source": processor_id,
        "kind": "curve_from_digests",
        "positive_state": positive_state,
        "negative_state": negative_state,
        "output": output,
    }


def build_calibration_from_digests_metric(
    processor_id: str,
    positive_state: str,
    negative_state: str,
) -> dict[str, Any]:
    """Create a calibration metric from final-score positive and negative digests."""
    return {
        "source": processor_id,
        "kind": "calibration_from_digests",
        "positive_state": positive_state,
        "negative_state": negative_state,
    }


def build_variant_compare_metric(
    processor_id: str,
    variant_column: str,
    *,
    test_role: str = "Test",
    control_role: str = "Control",
    confidence_level: float = 0.95,
    outputs: list[str] | None = None,
) -> dict[str, Any]:
    """Create a test-vs-control comparison metric definition."""
    metric: dict[str, Any] = {
        "source": processor_id,
        "kind": "variant_compare",
        "variant_column": variant_column,
        "test_role": test_role,
        "control_role": control_role,
        "confidence_level": confidence_level,
    }
    if outputs:
        metric["outputs"] = outputs
    return metric


def build_contingency_test_metric(
    processor_id: str,
    variant_column: str,
    *,
    tests: list[str] | None = None,
    outputs: list[str] | None = None,
) -> dict[str, Any]:
    """Create a contingency-test metric definition."""
    metric: dict[str, Any] = {
        "source": processor_id,
        "kind": "contingency_test",
        "variant_column": variant_column,
        "tests": tests or ["chi2", "g", "z"],
    }
    if outputs:
        metric["outputs"] = outputs
    return metric


def build_proportion_test_metric(
    processor_id: str,
    variant_column: str = "ModelControlGroup",
    *,
    test_role: str = "Test",
    control_role: str = "Control",
    outputs: list[str] | None = None,
) -> dict[str, Any]:
    """Create a proportion-test metric definition."""
    metric: dict[str, Any] = {
        "source": processor_id,
        "kind": "proportion_test",
        "variant_column": variant_column,
        "test_role": test_role,
        "control_role": control_role,
    }
    if outputs:
        metric["outputs"] = outputs
    return metric


def build_lifecycle_summary_metric(
    processor_id: str,
    *,
    outputs: list[str] | None = None,
) -> dict[str, Any]:
    """Create a lifecycle summary metric definition."""
    metric: dict[str, Any] = {"source": processor_id, "kind": "lifecycle_summary"}
    if outputs:
        metric["outputs"] = outputs
    return metric


def build_set_op_metric(
    processor_id: str,
    op: str,
    states: list[str],
) -> dict[str, Any]:
    """Create a theta-set operation metric definition."""
    return {"source": processor_id, "kind": "set_op", "op": op, "states": states}


def build_funnel_dropoff_metric(
    processor_id: str,
    from_stage: str,
    to_stage: str,
    output: str = "rate",
) -> dict[str, Any]:
    """Create a funnel drop-off metric definition."""
    return {
        "source": processor_id,
        "kind": "funnel_dropoff",
        "from_stage": from_stage,
        "to_stage": to_stage,
        "output": output,
    }


def build_tile(
    *,
    tile_id: str,
    title: str,
    metric_name: str,
    chart_kind: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    """Create a dashboard tile definition."""
    tile: dict[str, Any] = {
        "id": tile_id,
        "title": title,
        "metric": metric_name,
        "chart": chart_kind,
    }
    for key, value in fields.items():
        if value not in (None, "", []):
            tile[key] = value
    return tile


def generated_catalog_id(name: str, suffix: str, *, fallback: str) -> str:
    """Build a YAML-safe dashboard/page/tile id from a display name and suffix."""
    stem = _catalog_id_stem(name, fallback=fallback)
    return f"{stem}_{suffix}"


def random_catalog_id(name: str, *, fallback: str) -> str:
    """Build a YAML-safe dashboard/page/tile id with an 8-byte random suffix."""
    return generated_catalog_id(name, secrets.token_hex(8), fallback=fallback)


def _catalog_id_stem(name: str, *, fallback: str) -> str:
    stem = _catalog_id_slug(name)[:20].strip("_") or fallback
    if not stem[0].isalpha():
        stem = f"{fallback}_{stem}"[:20].strip("_") or fallback
    return stem


def _catalog_id_slug(value: str) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    return "_".join("".join(chars).split("_")).strip("_")


def metric_yaml(metric_name: str, metric_def: dict[str, Any]) -> str:
    """Render a metric draft as YAML."""
    return yaml.safe_dump({"metrics": {metric_name: metric_def}}, sort_keys=False)


def tile_yaml(tile: dict[str, Any]) -> str:
    """Render a tile draft as YAML."""
    return yaml.safe_dump({"tiles": [tile]}, sort_keys=False)


CATALOG_FILENAMES = ("pipelines.yaml", "processors.yaml", "metrics.yaml", "dashboards.yaml")


@contextmanager
def catalog_transaction(workspace: str | Path) -> Iterator[None]:
    """Restore every catalog file if a multi-file authoring write fails midway."""
    root = Path(workspace)
    paths = [root / "catalog" / filename for filename in CATALOG_FILENAMES]
    with _configuration_file_transaction(paths):
        yield


@contextmanager
def workspace_configuration_transaction(workspace: str | Path) -> Iterator[None]:
    """Restore catalog and workspace AI config when a complete apply fails."""

    root = Path(workspace)
    paths = [
        *(root / "catalog" / filename for filename in CATALOG_FILENAMES),
        root / "ai.yaml",
    ]
    with _configuration_file_transaction(paths):
        yield


@contextmanager
def _configuration_file_transaction(paths: Iterable[Path]) -> Iterator[None]:
    """Restore ``paths`` to their exact pre-write contents on any exception."""

    unique_paths = list(dict.fromkeys(paths))
    snapshots = {
        path: path.read_text(encoding="utf-8") if path.exists() else None
        for path in unique_paths
    }
    try:
        yield
    except BaseException:
        for path, content in snapshots.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(content, encoding="utf-8")
        raise


def _replace_or_append(items: list[Any], definition: dict[str, Any]) -> None:
    """Replace the entry with a matching id in place, or append a new one."""
    for index, item in enumerate(items):
        if isinstance(item, dict) and item.get("id") == definition["id"]:
            items[index] = definition
            return
    items.append(definition)


def write_metric_definition(
    workspace: str | Path,
    metric_name: str,
    metric_def: dict[str, Any],
) -> None:
    """Add or replace one metric in ``metrics.yaml``."""
    path = _catalog_file(workspace, "metrics.yaml")
    data = _read_yaml(path)
    metrics = data.setdefault("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("metrics.yaml must contain a mapping at `metrics`")
    metrics[metric_name] = metric_def
    _write_yaml(path, data)


def write_source_definition(
    workspace: str | Path,
    source_def: dict[str, Any],
) -> None:
    """Add or replace one source in ``pipelines.yaml``."""
    if not source_def.get("id"):
        raise ValueError("source definition must include `id`")
    path = _catalog_file(workspace, "pipelines.yaml")
    data = _read_yaml(path)
    sources = data.setdefault("sources", [])
    if not isinstance(sources, list):
        raise ValueError("pipelines.yaml must contain a list at `sources`")
    _replace_or_append(sources, source_def)
    _write_yaml(path, data)


def write_processor_definition(
    workspace: str | Path,
    processor_def: dict[str, Any],
) -> None:
    """Add or replace one processor in ``processors.yaml``."""
    if not processor_def.get("id"):
        raise ValueError("processor definition must include `id`")
    path = _catalog_file(workspace, "processors.yaml")
    data = _read_yaml(path)
    processors = data.setdefault("processors", [])
    if not isinstance(processors, list):
        raise ValueError("processors.yaml must contain a list at `processors`")
    _replace_or_append(processors, processor_def)
    _write_yaml(path, data)


def write_tile_definition(
    workspace: str | Path,
    *,
    dashboard_id: str,
    dashboard_title: str,
    page_id: str,
    page_title: str,
    tile: dict[str, Any],
) -> None:
    """Append or replace one tile in ``dashboards.yaml``."""
    path = _catalog_file(workspace, "dashboards.yaml")
    data = _read_yaml(path)
    dashboards = data.setdefault("dashboards", [])
    if not isinstance(dashboards, list):
        raise ValueError("dashboards.yaml must contain a list at `dashboards`")
    dashboard = _find_or_create_dashboard(dashboards, dashboard_id, dashboard_title)
    page = _find_or_create_page(dashboard, page_id, page_title)
    tiles = page.setdefault("tiles", [])
    if not isinstance(tiles, list):
        raise ValueError("dashboard page must contain a list at `tiles`")
    _replace_or_append(tiles, tile)
    _write_yaml(path, data)


def write_page_settings(
    workspace: str | Path,
    *,
    dashboard_id: str,
    dashboard_title: str,
    page_id: str,
    page_title: str,
    filters: list[dict[str, Any]] | None = None,
    time_filter: dict[str, Any] | None = None,
) -> None:
    """Update one page's authored settings without replacing its tiles."""

    path = _catalog_file(workspace, "dashboards.yaml")
    data = _read_yaml(path)
    dashboards = data.setdefault("dashboards", [])
    if not isinstance(dashboards, list):
        raise ValueError("dashboards.yaml must contain a list at `dashboards`")
    dashboard = _find_or_create_dashboard(dashboards, dashboard_id, dashboard_title)
    page = _find_or_create_page(dashboard, page_id, page_title)
    page["title"] = page_title
    if filters:
        page["filters"] = filters
    else:
        page.pop("filters", None)
    if time_filter:
        page["time_filter"] = time_filter
    else:
        page.pop("time_filter", None)
    model.Dashboards.model_validate(data)
    _write_yaml(path, data)


def write_dashboards_definition(
    workspace: str | Path,
    dashboards_def: dict[str, Any],
) -> None:
    """Replace ``dashboards.yaml`` with one structurally validated full definition."""

    model.Dashboards.model_validate(dashboards_def)
    _write_yaml(_catalog_file(workspace, "dashboards.yaml"), dashboards_def)


def write_workspace_settings(
    workspace: str | Path,
    *,
    workspace_name: str,
    time_zone: str,
    calendar_grains: list[str],
    week_start: str,
    dashboard_theme: dict[str, Any],
) -> None:
    """Update workspace defaults in ``pipelines.yaml`` and theme in ``dashboards.yaml``."""
    if not calendar_grains:
        raise ValueError("select at least one calendar grain")
    if week_start not in {"monday", "sunday"}:
        raise ValueError("week_start must be `monday` or `sunday`")

    pipelines_path = _catalog_file(workspace, "pipelines.yaml")
    pipelines = _read_yaml(pipelines_path)
    pipelines["workspace"] = workspace_name.strip() or _workspace_name(Path(workspace))
    defaults = pipelines.setdefault("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("pipelines.yaml must contain a mapping at `defaults`")
    defaults["time_zone"] = time_zone.strip() or "UTC"
    calendar = defaults.setdefault("calendar", {})
    if not isinstance(calendar, dict):
        raise ValueError("pipelines.yaml defaults must contain a mapping at `calendar`")
    calendar["grains"] = calendar_grains
    calendar["week_start"] = week_start
    _write_yaml(pipelines_path, pipelines)

    dashboards_path = _catalog_file(workspace, "dashboards.yaml")
    dashboards = _read_yaml(dashboards_path)
    dashboards["theme"] = dashboard_theme
    _write_yaml(dashboards_path, dashboards)


def delete_tile_definition(
    workspace: str | Path,
    *,
    dashboard_id: str,
    page_id: str,
    tile_id: str,
) -> bool:
    """Remove a tile from ``dashboards.yaml`` and return whether it existed."""
    path = _catalog_file(workspace, "dashboards.yaml")
    data = _read_yaml(path)
    dashboards = data.get("dashboards", [])
    if not isinstance(dashboards, list):
        raise ValueError("dashboards.yaml must contain a list at `dashboards`")
    for dashboard in dashboards:
        if dashboard.get("id") != dashboard_id:
            continue
        pages = dashboard.get("pages", [])
        if not isinstance(pages, list):
            continue
        for page in pages:
            if page.get("id") != page_id:
                continue
            tiles = page.get("tiles", [])
            if not isinstance(tiles, list):
                continue
            before = len(tiles)
            page["tiles"] = [tile for tile in tiles if tile.get("id") != tile_id]
            deleted = len(page["tiles"]) != before
            if deleted:
                _write_yaml(path, data)
            return deleted
    return False


def validate_workspace(workspace: str | Path) -> tuple[bool, list[str]]:
    """Load and validate a workspace after a builder change."""
    ensure_minimum_workspace(workspace)
    try:
        catalog = load(workspace)
    except CatalogLoadError as exc:
        return False, [str(exc)]
    result = validate_catalog(catalog)
    return result.ok, [f"{issue.location}: {issue.message}" for issue in result.issues]


def require_valid_workspace(workspace: str | Path) -> None:
    """Raise with validation details so an enclosing authoring transaction rolls back."""

    ok, issues = validate_workspace(workspace)
    if ok:
        return
    details = "\n".join(f"- {issue}" for issue in issues)
    raise ValueError(f"Workspace catalog validation failed; changes were rolled back:\n{details}")


def ensure_minimum_workspace(workspace: str | Path) -> Path:
    """Create ``workspace/catalog`` and any missing minimum catalog YAML files."""
    ws = Path(workspace)
    catalog_dir = ws / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    workspace_name = _workspace_name(ws)
    for filename in MINIMUM_CATALOG_FILES:
        path = catalog_dir / filename
        if path.exists() and path.read_text(encoding="utf-8").strip():
            continue
        _write_yaml(path, _minimum_catalog_data(filename, workspace_name))
    return ws


def _find_or_create_dashboard(
    dashboards: list[dict[str, Any]],
    dashboard_id: str,
    dashboard_title: str,
) -> dict[str, Any]:
    for dashboard in dashboards:
        if dashboard.get("id") == dashboard_id:
            return dashboard
    dashboard = {
        "id": dashboard_id,
        "title": dashboard_title,
        "layout": "tabs",
        "pages": [],
    }
    dashboards.append(dashboard)
    return dashboard


def _find_or_create_page(
    dashboard: dict[str, Any],
    page_id: str,
    page_title: str,
) -> dict[str, Any]:
    pages = cast(list[dict[str, Any]], dashboard.setdefault("pages", []))
    if not isinstance(pages, list):
        raise ValueError("dashboard must contain a list at `pages`")
    for page in pages:
        if page.get("id") == page_id:
            return page
    page = {"id": page_id, "title": page_title, "tiles": []}
    pages.append(page)
    return page


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a mapping")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace one YAML file after serializing it completely."""

    payload = yaml.safe_dump(data, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _catalog_file(workspace: str | Path, filename: str) -> Path:
    return ensure_minimum_workspace(workspace) / "catalog" / filename


def _minimum_catalog_data(filename: str, workspace_name: str) -> dict[str, Any]:
    if filename == "pipelines.yaml":
        return {
            "version": 1,
            "workspace": workspace_name,
            "sources": [],
        }
    if filename == "processors.yaml":
        return {"processors": []}
    if filename == "metrics.yaml":
        return {"metrics": {}}
    if filename == "dashboards.yaml":
        return {"theme": {}, "dashboards": []}
    raise ValueError(f"unsupported catalog file: {filename}")


def _workspace_name(workspace: Path) -> str:
    name = workspace.name or "workspace"
    normalized = re.sub(r"[^a-z0-9_-]+", "_", name.casefold()).strip("_-")
    if not normalized or not normalized[0].isalpha():
        normalized = f"workspace_{normalized}" if normalized else "workspace"
    return normalized


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _safe_literal(value: Any) -> Any:  # noqa: PLR0911
    if isinstance(value, bool) or value is None or isinstance(value, int | float):
        return value
    text = str(value).strip()
    if text == "":
        return ""
    if text.casefold() == "true":
        return True
    if text.casefold() == "false":
        return False
    if text.casefold() in {"null", "none"}:
        return None
    try:
        return py_ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _split_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = py_ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, list | tuple):
        return [_safe_literal(item) for item in parsed]
    return [_safe_literal(item) for item in text.split(",") if str(item).strip()]


def _compile_filter_row(row: dict[str, Any]) -> dict[str, Any] | None:  # noqa: PLR0911
    if not row.get("Enabled", True):
        return None
    field = str(row.get("Field", "")).strip()
    operator = str(row.get("Operator", "")).strip()
    raw_value = row.get("Value", "")
    if not field or not operator:
        return None
    if operator == "is null":
        return {"op": "is_null", "column": field}
    if operator == "is not null":
        return {"op": "not_null", "column": field}
    if operator == "in":
        return {"op": "in", "column": field, "values": _split_values(raw_value)}
    if operator == "not in":
        return {"op": "not_in", "column": field, "values": _split_values(raw_value)}
    if operator == "contains":
        return {"op": "matches", "column": field, "pattern": str(raw_value)}
    if operator == "starts with":
        return {"op": "starts_with", "column": field, "value": str(raw_value)}
    if operator == "ends with":
        return {"op": "ends_with", "column": field, "value": str(raw_value)}
    op_map = {
        "==": "eq",
        "!=": "ne",
        ">": "gt",
        ">=": "ge",
        "<": "lt",
        "<=": "le",
    }
    if operator not in op_map:
        raise ValueError(f"unsupported filter operator: {operator}")
    return {"op": op_map[operator], "column": field, "value": _safe_literal(raw_value)}


def _filter_row_from_expression(expression: dict[str, Any]) -> dict[str, Any] | None:  # noqa: PLR0911
    op = expression.get("op")
    if op in {"eq", "ne", "gt", "ge", "lt", "le"} and "column" in expression:
        operator_map = {"eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}
        return {
            "Field": expression.get("column", ""),
            "Operator": operator_map[str(op)],
            "Value": expression.get("value", ""),
            "Enabled": True,
        }
    if op == "in":
        return {
            "Field": expression.get("column", ""),
            "Operator": "in",
            "Value": ", ".join(map(str, expression.get("values", []))),
            "Enabled": True,
        }
    if op == "not_in":
        return {
            "Field": expression.get("column", ""),
            "Operator": "not in",
            "Value": ", ".join(map(str, expression.get("values", []))),
            "Enabled": True,
        }
    if op == "is_null":
        return {
            "Field": expression.get("column", ""),
            "Operator": "is null",
            "Value": "",
            "Enabled": True,
        }
    if op == "not_null":
        return {
            "Field": expression.get("column", ""),
            "Operator": "is not null",
            "Value": "",
            "Enabled": True,
        }
    if op == "matches":
        return {
            "Field": expression.get("column", ""),
            "Operator": "contains",
            "Value": expression.get("pattern", ""),
            "Enabled": True,
        }
    if op == "starts_with":
        return {
            "Field": expression.get("column", ""),
            "Operator": "starts with",
            "Value": expression.get("value", ""),
            "Enabled": True,
        }
    if op == "ends_with":
        return {
            "Field": expression.get("column", ""),
            "Operator": "ends with",
            "Value": expression.get("value", ""),
            "Enabled": True,
        }
    return None


def _is_missing_editor_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(value)
    except TypeError:
        return False


def _states_with_outcome(processor: model.Processor, outcome: str) -> list[str]:
    return [
        name
        for name, spec in model.effective_processor_states(processor).items()
        if spec.type == "tdigest"
        and str(dict(spec.model_extra or {}).get("outcome", "")).casefold() == outcome.casefold()
    ]


def _digest_outcome_role(state_name: str, spec: dict[str, Any]) -> str:
    outcome = str(spec.get("outcome", "") or "").casefold()
    if outcome in {"positive", "negative"}:
        return outcome
    lowered = state_name.casefold()
    if "positive" in lowered or "positives" in lowered:
        return "positive"
    if "negative" in lowered or "negatives" in lowered:
        return "negative"
    return ""


def _digest_source_property(
    processor_def: dict[str, Any], state_name: str, spec: dict[str, Any]
) -> str:
    score_property = str(spec.get("score_property", "") or "")
    if score_property:
        return score_property
    source_column = str(spec.get("source_column", "") or "")
    if source_column:
        return source_column
    score = str(spec.get("score", "") or "")
    if score:
        raw_score_columns = processor_def.get("score_columns")
        score_columns = dict(raw_score_columns) if isinstance(raw_score_columns, dict) else {}
        score_properties = [
            str(item)
            for item in processor_def.get("score_properties", []) or []
            if str(item).strip()
        ]
        if score == "primary":
            return str(
                score_columns.get("primary", "")
                or (score_properties[0] if score_properties else "primary")
            )
        if score == "calibrated":
            return str(
                score_columns.get("calibrated", "")
                or (score_properties[1] if len(score_properties) > 1 else "")
                or (score_properties[0] if score_properties else "")
                or "calibrated"
            )
        return score
    return _digest_property_from_state_name(state_name)


def _digest_property_from_state_name(state_name: str) -> str:
    out = state_name
    for token in (
        "tdigest_",
        "_tdigest",
        "_positives",
        "_positive",
        "_negatives",
        "_negative",
    ):
        out = out.replace(token, "")
    return out or state_name


def _digest_state_family(state_name: str, role: str, property_name: str) -> str:
    lowered_role = role.casefold()
    out = state_name
    for token in (
        f"_{lowered_role}s",
        f"_{lowered_role}",
        f"{lowered_role}s_",
        f"{lowered_role}_",
    ):
        out = out.replace(token, "_")
    out = out.strip("_")
    return out if out != state_name else property_name


def _default_grain(processor: model.Processor | None) -> str:
    if processor is None:
        return "summary"
    grains = processor.grains
    if "daily" in grains:
        return "daily"
    if "monthly" in grains:
        return "monthly"
    return grains[0] if grains else "summary"


# ---------------------------------------------------------------------------
# Shared page helpers — used by both the Config Builder and AI Studio pages.
# ---------------------------------------------------------------------------


def dedupe(values: list[str]) -> list[str]:
    """Return values with duplicates and empties removed, preserving order."""
    return _dedupe(values)


def string_list(value: Any) -> list[str]:
    """Coerce a YAML list or comma-separated string into a list of strings."""
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return csv_text_to_list(value)
    return []


def csv_text_to_list(raw: str) -> list[str]:
    """Split comma-separated text into trimmed, non-empty values."""
    return [part.strip() for part in raw.split(",") if part.strip()]


def float_in_range(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Coerce a value to a float clamped to [minimum, maximum]."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def option_index(options: Iterable[str], value: Any) -> int:
    """Return the selectbox index of a value, defaulting to the first option."""
    choices = list(options)
    normalized = str(value or "").strip()
    return choices.index(normalized) if normalized in choices else 0


def display_grain(grain: Any) -> str:
    """Map normalized aggregate grain names back to their display labels."""
    mapping = {
        "daily": "Day",
        "monthly": "Month",
        "quarterly": "Quarter",
        "yearly": "Year",
        "summary": "Summary",
    }
    text = str(grain or "")
    return mapping.get(text.casefold(), text)


def title_from_identifier(value: str, *, fallback: str = "Untitled") -> str:
    """Turn an identifier like `vs_engagement_CTR` into a display title."""
    words = str(value or "").replace("_", " ").replace("-", " ").strip().split()
    return " ".join(word if word.isupper() else word.capitalize() for word in words) or fallback


def widget_key_fragment(value: str, *, fallback: str = "value") -> str:
    """Return a session-state-safe key fragment for a display value."""
    fragment = "".join(char if char.isalnum() else "_" for char in value)
    return fragment[:80] or fallback


def operand_states(metric_def: dict[str, Any]) -> list[str]:
    """Return states referenced by a set_op metric's operands."""
    operands = metric_def.get("operands")
    if not isinstance(operands, list):
        return []
    states: list[str] = []
    for operand in operands:
        if isinstance(operand, dict) and operand.get("state"):
            states.append(str(operand["state"]))
    return states


def digest_pair_option_index(
    options: list[tuple[str, str, str]],
    positive_state: str,
    negative_state: str,
) -> int:
    """Return the option index matching a positive/negative digest pair."""
    for index, option in enumerate(options):
        if option[1] == positive_state and option[2] == negative_state:
            return index
    return len(options) - 1 if options else 0


def digest_pair_option_label(option: tuple[str, str, str]) -> str:
    """Return the display label for a digest pair option."""
    property_name, positive_state, negative_state = option
    if not positive_state or not negative_state:
        return property_name
    return f"{property_name} ({positive_state} / {negative_state})"


__all__ = [
    "CALCULATION_MODES",
    "CHART_OPTIONAL_FIELDS",
    "CHART_REQUIRED_FIELDS",
    "CHART_SETTING_FIELDS",
    "FILTER_OPERATORS",
    "METRIC_KIND_HELP",
    "METRIC_KIND_LABELS",
    "SCALAR_STATE_TYPES",
    "STATE_TYPES",
    "blank_calculated_row",
    "blank_default_row",
    "blank_filter_row",
    "build_approx_distinct_metric",
    "build_calibration_from_digests_metric",
    "build_contingency_test_metric",
    "build_curve_from_digests_metric",
    "build_default_values",
    "build_derive_column_transforms",
    "build_formula_metric",
    "build_funnel_dropoff_metric",
    "build_lifecycle_summary_metric",
    "build_proportion_test_metric",
    "build_set_op_metric",
    "build_tdigest_quantile_metric",
    "build_tile",
    "build_topk_items_metric",
    "build_variant_compare_metric",
    "calculated_rows_for_editor",
    "calculated_rows_from_source",
    "chart_choices_for_metric",
    "chart_field_controls",
    "chart_field_options",
    "chart_recipe_summary",
    "compile_filter_rows",
    "csv_text_to_list",
    "dedupe",
    "default_curve_digest_states",
    "default_metric_kind",
    "default_metric_name",
    "default_rows_from_values",
    "default_rows_with_fields",
    "default_tile_fields",
    "delete_tile_definition",
    "descriptive_property_options",
    "descriptive_score_options",
    "digest_pair_option_index",
    "digest_pair_option_label",
    "digest_pair_options_from_definition",
    "digest_state_pair_options",
    "display_grain",
    "editor_frame",
    "ensure_minimum_workspace",
    "expression_yaml",
    "filter_rows_from_expression",
    "first_filter_expression",
    "float_in_range",
    "funnel_stage_names",
    "generated_catalog_id",
    "merge_stage_definitions",
    "metric_kind_help",
    "metric_kind_label",
    "metric_kind_options",
    "metric_to_dict",
    "metric_yaml",
    "normalize_editor_rows",
    "operand_states",
    "option_index",
    "parse_expression_yaml",
    "processor_for_metric",
    "processor_to_dict",
    "random_catalog_id",
    "scalar_state_columns",
    "score_properties_from_definition",
    "source_defaults",
    "source_to_dict",
    "stage_names_missing_when",
    "state_columns",
    "state_columns_by_type",
    "state_spec_definitions",
    "string_list",
    "tile_yaml",
    "title_from_identifier",
    "validate_workspace",
    "widget_key_fragment",
    "write_dashboards_definition",
    "write_metric_definition",
    "write_page_settings",
    "write_processor_definition",
    "write_source_definition",
    "write_tile_definition",
    "write_workspace_settings",
]
