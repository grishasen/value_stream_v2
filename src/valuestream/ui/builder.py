"""Catalog Builder helpers for the Streamlit UI."""

from __future__ import annotations

import ast as py_ast
import copy
import math
import os
import re
import secrets
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl
import yaml

from valuestream.charts.recipes import (
    CHART_OPTIONAL_FIELDS,
    CHART_REQUIRED_FIELDS,
    RECIPES,
)
from valuestream.charts.recipes import (
    chart_field_controls as _recipe_chart_field_controls,
)
from valuestream.config import canonical as config_canonical
from valuestream.config import model
from valuestream.config.loader import CatalogLoadError, load
from valuestream.config.report_fields import (
    metric_output_columns as _report_metric_output_columns,
)
from valuestream.config.report_fields import (
    report_field_options,
)
from valuestream.config.validate import validate_catalog
from valuestream.expr import parser as expr_parser
from valuestream.expr import translator as expr_translator

FILTER_OPERATORS = [
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "between",
    "contains",
    "starts with",
    "ends with",
    "in",
    "not in",
    "is null",
    "is not null",
]

_DATE_DIFF_MODE_UNITS = {
    f"Date Diff {unit.title()}": unit
    for unit in ("seconds", "minutes", "hours", "days", "months", "years")
}
_DATE_PART_MODE_UNITS = {
    f"Date Part {unit.title()}": unit
    for unit in ("year", "month", "quarter", "day", "hour", "weekday")
}

CALCULATION_MODES = [
    "AST YAML",
    "Polars",
    "Copy Field",
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "Safe Divide",
    "Absolute Value",
    "Round",
    "Concat",
    "Coalesce",
    *_DATE_DIFF_MODE_UNITS,
    *_DATE_PART_MODE_UNITS,
]

VISUAL_CASE_MAX_BRANCHES = 8

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

BUILDER_DRAFTS_KEY = "builder_unapplied_drafts"
_RUN_DATA_SCOPES = frozenset(
    {
        "source",
        "processor",
        "dimensions",
        "workspace_settings",
        "exploration",
        "recipe_with_state",
    }
)


@dataclass(frozen=True)
class BuilderDraftStatus:
    """Canonical dirty-state summary for one step-local Builder object."""

    key: str
    baseline_hash: str
    draft_hash: str
    revision: str
    dirty: bool
    baseline_payload: Any
    draft_payload: Any


@dataclass(frozen=True)
class BuilderApplyOutcome:
    """Outcome-first handoff shown after one configuration apply."""

    label: str
    action: str
    message: str
    source_ids: tuple[str, ...] = ()

    @property
    def requires_data_run(self) -> bool:
        """Return whether the applied configuration changes computation state."""
        return self.action == "run_data"


@dataclass(frozen=True)
class CalculatedExpressionValidation:
    """Human-facing validation result for one calculated-field expression."""

    valid: bool
    messages: tuple[str, ...] = ()
    technical_details: str = ""


def builder_draft_status(
    key: str,
    baseline: Any,
    draft: Any,
) -> BuilderDraftStatus:
    """Compare one proposed object with its applied canonical representation."""
    baseline_payload = config_canonical.canonicalize(baseline)
    draft_payload = config_canonical.canonicalize(draft)
    baseline_hash = config_canonical.config_hash(baseline_payload)
    draft_hash = config_canonical.config_hash(draft_payload)
    return BuilderDraftStatus(
        key=key,
        baseline_hash=baseline_hash,
        draft_hash=draft_hash,
        revision=draft_hash[:12],
        dirty=baseline_hash != draft_hash,
        baseline_payload=baseline_payload,
        draft_payload=draft_payload,
    )


def builder_template_draft_status(
    session_state: MutableMapping[str, Any],
    key: str,
    baseline_key: str,
    draft: Any,
    *,
    baseline_draft: Any | None = None,
) -> BuilderDraftStatus:
    """Compare a create-mode editor with its first untouched template.

    Create editors do not have an applied catalog object to use as a baseline.
    Capturing the fully rendered template on first entry keeps the editor clean
    until the user actually changes one of its values.  The baseline lives
    beside the editor widgets so Discard and post-apply cleanup can remove it
    with the same prefix.

    ``baseline_draft`` lets editors whose form only becomes complete after a
    required selection register a pristine template instead of the live draft.
    Without it, values typed before that selection would be captured into the
    baseline and the finished draft would compare clean, hiding Apply.
    """

    if baseline_key not in session_state:
        template = draft if baseline_draft is None else baseline_draft
        session_state[baseline_key] = copy.deepcopy(config_canonical.canonicalize(template))
    return builder_draft_status(
        key,
        session_state[baseline_key],
        draft,
    )


def update_builder_draft_registry(
    session_state: MutableMapping[str, Any],
    status: BuilderDraftStatus,
    *,
    widget_prefixes: Iterable[str] = (),
) -> bool:
    """Persist one canonical draft plus restorable widget shadows."""
    raw = session_state.get(BUILDER_DRAFTS_KEY)
    registry = dict(raw) if isinstance(raw, dict) else {}
    existed = status.key in registry
    if status.dirty:
        prefixes = tuple(widget_prefixes)
        widget_state = {
            str(widget_key): copy.deepcopy(value)
            for widget_key, value in session_state.items()
            if prefixes and any(str(widget_key).startswith(prefix) for prefix in prefixes)
        }
        registry[status.key] = {
            "revision": status.revision,
            "baseline_hash": status.baseline_hash,
            "draft_hash": status.draft_hash,
            "draft_payload": copy.deepcopy(status.draft_payload),
            "widget_state": widget_state,
        }
    else:
        registry.pop(status.key, None)
    session_state[BUILDER_DRAFTS_KEY] = registry
    return existed


def registered_builder_draft(
    session_state: MutableMapping[str, Any],
    key: str,
) -> dict[str, Any] | None:
    """Return one registered Builder draft without exposing the live registry."""
    raw = session_state.get(BUILDER_DRAFTS_KEY)
    if not isinstance(raw, dict):
        return None
    entry = raw.get(key)
    return copy.deepcopy(entry) if isinstance(entry, dict) else None


def restore_builder_draft(
    session_state: MutableMapping[str, Any],
    key: str,
) -> bool:
    """Restore a registered draft's shadow values before widgets are rendered."""
    entry = registered_builder_draft(session_state, key)
    if entry is None:
        return False
    widget_state = entry.get("widget_state")
    if not isinstance(widget_state, dict) or not widget_state:
        return False
    for widget_key, value in widget_state.items():
        session_state[str(widget_key)] = copy.deepcopy(value)
    return True


def discard_builder_draft(
    session_state: MutableMapping[str, Any],
    key: str,
    *,
    widget_prefixes: Iterable[str] = (),
    preserve_widget_keys: Iterable[str] = (),
) -> None:
    """Discard one registered proposal and its step-local widget state."""
    raw = session_state.get(BUILDER_DRAFTS_KEY)
    registry = dict(raw) if isinstance(raw, dict) else {}
    registry.pop(key, None)
    session_state[BUILDER_DRAFTS_KEY] = registry
    prefixes = tuple(widget_prefixes)
    preserved = {str(widget_key) for widget_key in preserve_widget_keys}
    if not prefixes:
        return
    for widget_key in list(session_state):
        if widget_key == BUILDER_DRAFTS_KEY or str(widget_key) in preserved:
            continue
        if any(str(widget_key).startswith(prefix) for prefix in prefixes):
            session_state.pop(widget_key, None)


def builder_apply_outcome(
    label: str,
    *,
    source_ids: Iterable[str] = (),
    requires_data_run: bool = False,
) -> BuilderApplyOutcome:
    """Classify an applied Builder object as report-ready or requiring data."""
    sources = tuple(sorted({str(source_id) for source_id in source_ids if str(source_id)}))
    if requires_data_run:
        source_text = (
            f" Run {', '.join(sources)} from Data Load to publish matching aggregates."
            if sources
            else " Run the affected source from Data Load to publish matching aggregates."
        )
        return BuilderApplyOutcome(
            label=label,
            action="run_data",
            message=(
                "The workspace configuration is valid, but its aggregate computation "
                f"contract changed.{source_text}"
            ),
            source_ids=sources,
        )
    return BuilderApplyOutcome(
        label=label,
        action="open_report",
        message="The workspace configuration is valid and existing aggregates can be used now.",
    )


def builder_requires_data_run(
    scope: str,
    baseline: Any,
    draft: Any,
) -> bool:
    """Compare only the configuration contract that can change persisted aggregates."""
    if scope not in _RUN_DATA_SCOPES:
        return False
    if scope == "recipe_with_state":
        return True
    if scope == "source":
        baseline_contract = _source_computation_projection(baseline)
        draft_contract = _source_computation_projection(draft)
    elif scope == "processor":
        baseline_contract = _processor_computation_projection(baseline)
        draft_contract = _processor_computation_projection(draft)
    elif scope == "dimensions":
        baseline_contract = _dimension_computation_projection(baseline)
        draft_contract = _dimension_computation_projection(draft)
    elif scope == "workspace_settings":
        baseline_contract = _workspace_computation_projection(baseline)
        draft_contract = _workspace_computation_projection(draft)
    else:
        return False
    return config_canonical.config_hash(baseline_contract) != config_canonical.config_hash(
        draft_contract
    )


def _source_computation_projection(value: Any) -> Any:
    if not isinstance(value, dict):
        return config_canonical.canonicalize(value)
    payload = copy.deepcopy(value)
    for key in ("description", "debugging", "materialize_transforms"):
        payload.pop(key, None)
    reader = payload.get("reader")
    if isinstance(reader, dict):
        reader.pop("debugging", None)
        reader.pop("streaming", None)
    return config_canonical.canonicalize(payload)


def _processor_computation_projection(value: Any) -> Any:
    if not isinstance(value, dict):
        return config_canonical.canonicalize(value)
    payload = copy.deepcopy(value)
    for key in ("description", "sketch_build_mode"):
        payload.pop(key, None)
    return config_canonical.canonicalize(payload)


def _dimension_computation_projection(value: Any) -> Any:
    """Project a workspace-dimensions draft onto its computation contract.

    The common-dimension list itself is authoring metadata; only the
    processors extended with new group-by fields change persisted rows.
    """
    if not isinstance(value, dict):
        return config_canonical.canonicalize(value)
    processors = value.get("processors")
    if not isinstance(processors, dict):
        processors = {}
    return {
        str(processor_id): _processor_computation_projection(processor_def)
        for processor_id, processor_def in processors.items()
    }


def _workspace_computation_projection(value: Any) -> Any:
    if not isinstance(value, dict):
        return config_canonical.canonicalize(value)
    return config_canonical.canonicalize(
        {key: value.get(key) for key in ("time_zone", "calendar_grains", "week_start")}
    )


METRIC_KIND_LABELS = {
    "formula": "Formula / state passthrough",
    "approx_distinct_count": "Approx distinct count",
    "topk_items": "Top-K frequent items",
    "tdigest_quantile": "Digest quantile / distribution",
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
    "tdigest_quantile": (
        "Read a percentile from a t-digest or KLL state, or omit the quantile "
        "for a distribution metric that feeds boxplots."
    ),
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

DESCRIPTIVE_SCALAR_SUFFIXES = ("Count", "Sum", "Mean", "Var", "Min", "Max")
DESCRIPTIVE_QUANTILE_SCORES = ("p25", "p50", "p75", "p90", "p95")
_DESCRIPTIVE_STATE_SUFFIXES = (
    *DESCRIPTIVE_SCALAR_SUFFIXES,
    "Median",
    *DESCRIPTIVE_QUANTILE_SCORES,
    "tdigest",
    "kll",
)

CHART_DISPLAY_LABELS: dict[str, str] = {
    "bar": "Bar",
    "bar_polar": "Polar bar",
    "boxplot": "Box plot",
    "calendar_heatmap": "Calendar heatmap",
    "calibration_curve": "Calibration curve",
    "clv_treemap": "CLV treemap",
    "cohort_heatmap": "Cohort heatmap",
    "combo": "Combo",
    "corr": "Correlation",
    "descriptive_funnel": "Descriptive funnel",
    "descriptive_heatmap": "Descriptive heatmap",
    "descriptive_histogram": "Descriptive histogram",
    "descriptive_line": "Descriptive line",
    "donut": "Donut",
    "experiment_odds_ratio": "Experiment odds ratio",
    "experiment_z_score": "Experiment z-score",
    "exposure": "Exposure",
    "funnel": "Funnel",
    "gain_curve": "Gain curve",
    "gauge": "Gauge",
    "geo_map": "Geographic map",
    "heatmap": "Heatmap",
    "histogram": "Histogram",
    "interval": "Interval",
    "kpi_card": "KPI card",
    "lift_curve": "Lift curve",
    "line": "Line",
    "model": "Model performance",
    "pareto": "Pareto",
    "precision_recall_curve": "Precision-recall curve",
    "rfm_density": "RFM density",
    "roc_curve": "ROC curve",
    "sankey": "Sankey",
    "scatter": "Scatter",
    "stacked_area": "Stacked area",
    "table": "Table",
    "treemap": "Treemap",
    "waterfall": "Waterfall",
}

CHART_DISPLAY_PURPOSES: dict[str, str] = {
    "bar": "Compare magnitudes across discrete categories.",
    "bar_polar": "Compare cyclical or directional categories around a circle.",
    "boxplot": "Compare medians, spread, and outliers between groups.",
    "calendar_heatmap": "Reveal daily activity and seasonality on a calendar grid.",
    "calibration_curve": "Compare predicted probabilities with observed outcomes.",
    "clv_treemap": "Show customer-value hierarchy through nested area.",
    "cohort_heatmap": "Compare retention or behavior across cohort periods.",
    "combo": "Place two measures on coordinated bar and line axes.",
    "corr": "Scan the strength and direction of pairwise relationships.",
    "descriptive_funnel": "Follow aggregate descriptive measures through ordered stages.",
    "descriptive_heatmap": "Compare an aggregate statistic across two dimensions.",
    "descriptive_histogram": "Show the distribution of an aggregate numeric property.",
    "descriptive_line": "Track an aggregate statistic over time or ordered categories.",
    "donut": "Show a small set of parts as shares of a whole.",
    "experiment_odds_ratio": "Compare experiment effects with confidence intervals.",
    "experiment_z_score": "Compare standardized experiment effects around zero.",
    "exposure": "Show how populations progress through exposure levels.",
    "funnel": "Show volume retained through ordered journey stages.",
    "gain_curve": "Show cumulative positives captured as coverage increases.",
    "gauge": "Show a current value against a reference or operating range.",
    "geo_map": "Compare a measure across geographic locations.",
    "heatmap": "Compare intensity across two categorical dimensions.",
    "histogram": "Show how numeric observations are distributed across bins.",
    "interval": "Compare estimates together with their uncertainty bounds.",
    "kpi_card": "Present one decision-ready value with optional comparison context.",
    "lift_curve": "Show improvement over random selection as coverage increases.",
    "line": "Track one or more measures across time or an ordered axis.",
    "model": "Summarize model performance across thresholds or score bands.",
    "pareto": "Rank contributors and show their cumulative share.",
    "precision_recall_curve": "Show the precision-recall tradeoff across thresholds.",
    "rfm_density": "Map customer density across recency and frequency/value space.",
    "roc_curve": "Show true-positive versus false-positive performance by threshold.",
    "sankey": "Show volume flowing between stages or categories.",
    "scatter": "Explore the relationship between two numeric measures.",
    "stacked_area": "Show total change over time together with category composition.",
    "table": "Inspect exact ranked or operational values in native tabular form.",
    "treemap": "Show hierarchical composition through nested area.",
    "waterfall": "Explain how positive and negative contributions build a total.",
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
    return _recipe_chart_field_controls(chart_kind)


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


def editor_row_enabled(value: Any) -> bool:
    """Normalize a grid checkbox without treating missing values as excluded.

    Streamlit reports a newly appended checkbox as ``None`` on some reruns.
    Missing/blank values therefore use the visible editor default (enabled),
    while explicit false values remain excluded.  String handling is strict so
    values such as ``"False"`` are not accidentally truthy.
    """

    if _is_missing_editor_value(value) or value == "":
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return True


def normalize_editor_rows(frame: Any) -> list[dict[str, Any]]:
    """Convert Streamlit/Pandas/Polars editor output into plain row dicts."""
    if hasattr(frame, "to_dicts"):
        rows = frame.to_dicts()
    elif hasattr(frame, "to_dict"):
        rows = frame.to_dict("records")
    else:
        rows = list(frame or [])
    return [
        {
            key: (
                editor_row_enabled(value)
                if key == "Enabled"
                else ""
                if _is_missing_editor_value(value)
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]


def editor_frame(
    rows: list[dict[str, Any]],
    columns: list[str],
    blank_row_factory: Callable[[], dict[str, Any]],
) -> pl.DataFrame:
    """Return a stable typed frame for Streamlit editable row tables."""
    del blank_row_factory  # kept for API compatibility with existing editor callers
    editor_rows = normalize_editor_rows(rows)
    series = []
    for column in columns:
        values = [
            editor_row_enabled(row.get(column))
            if column == "Enabled"
            else _editor_text_value(row.get(column, ""))
            for row in editor_rows
        ]
        dtype = pl.Boolean if column == "Enabled" else pl.String
        series.append(pl.Series(column, values, dtype=dtype))
    return pl.DataFrame(series)


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
    return rows


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
    return rows


def _default_row_has_content(row: dict[str, Any]) -> bool:
    return bool(
        str(row.get("Field", "")).strip()
        or str(row.get("Default Value", "")).strip()
        or not editor_row_enabled(row.get("Enabled"))
    )


def build_default_values(default_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a source defaults map from editor rows."""
    out: dict[str, Any] = {}
    for row in default_rows:
        if not editor_row_enabled(row.get("Enabled")):
            continue
        field = str(row.get("Field", "")).strip()
        if field:
            out[field] = _safe_literal(row.get("Default Value", ""))
    return out


def filter_rows_from_expression(expression: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Best-effort conversion from a simple AST expression to editable rule rows."""
    if expression is None:
        return []
    if expression.get("op") == "and" and isinstance(expression.get("args"), list):
        rows: list[dict[str, Any]] = []
        for arg in expression["args"]:
            parsed = _filter_row_from_expression(arg)
            if parsed is None:
                return None
            rows.append(parsed)
        return rows
    parsed = _filter_row_from_expression(expression)
    return [parsed] if parsed is not None else None


def compile_filter_rows(filter_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Compile rule rows into the Value Stream expression AST dictionary."""
    return compile_condition_rows(filter_rows, combine="all")


def compile_condition_rows(
    condition_rows: list[dict[str, Any]],
    *,
    combine: str = "all",
) -> dict[str, Any] | None:
    """Compile predicate rows into one boolean AST joined with and/or.

    ``combine="all"`` joins rows with ``and``; ``combine="any"`` joins with
    ``or``. Rows without a field or operator, and disabled rows, are skipped.
    Returns ``None`` when no complete row remains.
    """
    expressions = [_compile_filter_row(row) for row in condition_rows]
    expressions = [expression for expression in expressions if expression is not None]
    if not expressions:
        return None
    if len(expressions) == 1:
        return expressions[0]
    return {"op": "or" if combine == "any" else "and", "args": expressions}


_CONDITION_REF_PATTERN = re.compile(r"^[Ee]([1-9]\d*)$")
_FORMULA_KEYWORD_PATTERN = re.compile(r"\b(AND|OR|NOT)\b", re.IGNORECASE)


def label_condition_rows(condition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return condition rows with ``Ref`` set to E1..En in visible order."""
    labeled = []
    for index, row in enumerate(condition_rows):
        updated = dict(row)
        updated["Ref"] = f"E{index + 1}"
        labeled.append(updated)
    return labeled


def compile_condition_formula(
    condition_rows: list[dict[str, Any]],
    formula: str,
) -> dict[str, Any]:
    """Compile condition rows joined by a boolean formula into one AST.

    The formula references rows as ``E1``..``En`` and combines them with
    ``AND`` / ``OR`` / ``NOT`` (any case) and parentheses, e.g.
    ``(NOT E1 AND E2) OR E3``. Unlike the basic combine modes, every
    referenced row must exist, be enabled, and be complete — silent skipping
    would change the formula's meaning. Raises ``ValueError`` with
    remediation text on any problem.
    """
    text = str(formula or "").strip()
    if not text:
        raise ValueError("enter a logic formula, for example (E1 AND E2) OR NOT E3")
    normalized = _FORMULA_KEYWORD_PATTERN.sub(lambda match: match.group(1).lower(), text)
    try:
        parsed = py_ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"the logic formula is not valid: {exc.msg}") from exc
    return _compile_formula_node(parsed.body, condition_rows)


def _compile_formula_node(
    node: py_ast.AST,
    condition_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(node, py_ast.BoolOp):
        op = "and" if isinstance(node.op, py_ast.And) else "or"
        return {
            "op": op,
            "args": [_compile_formula_node(value, condition_rows) for value in node.values],
        }
    if isinstance(node, py_ast.UnaryOp) and isinstance(node.op, py_ast.Not):
        return {"op": "not", "arg": _compile_formula_node(node.operand, condition_rows)}
    if isinstance(node, py_ast.Name):
        return _compile_formula_ref(node.id, condition_rows)
    raise ValueError(
        "logic formulas support only condition references (E1, E2, ...), "
        "AND, OR, NOT, and parentheses"
    )


def _compile_formula_ref(name: str, condition_rows: list[dict[str, Any]]) -> dict[str, Any]:
    match = _CONDITION_REF_PATTERN.match(name)
    if match is None:
        raise ValueError(f"unknown token {name!r}: condition references look like E1, E2, ...")
    index = int(match.group(1))
    if index > len(condition_rows):
        raise ValueError(
            f"E{index} does not exist: the table has {len(condition_rows)} condition row(s)"
        )
    row = condition_rows[index - 1]
    if not editor_row_enabled(row.get("Enabled")):
        raise ValueError(f"E{index} is disabled: enable the row or remove it from the formula")
    predicate = _compile_filter_row(row)
    if predicate is None:
        raise ValueError(f"E{index} is incomplete: choose a field and an operator")
    return predicate


def condition_state_from_expression(expression: Any) -> dict[str, Any] | None:
    """Decompile a filter AST into rows plus Basic/Advanced logic state.

    Returns ``{"rows", "mode", "combine", "formula"}`` where flat
    conjunctions stay ``Basic`` (with ``combine`` AND/OR) and anything with
    ``not`` or mixed nesting becomes ``Advanced`` with a canonical formula
    string. ``formula`` is always rendered so a Basic seed can prefill the
    Advanced input. Returns ``None`` when any leaf predicate cannot map to a
    condition row (callers fall back to the raw-AST editor).
    """
    if expression in (None, {}):
        return {"rows": [], "mode": "Basic", "combine": "AND", "formula": ""}
    rows: list[dict[str, Any]] = []
    skeleton = _condition_skeleton(expression, rows)
    if skeleton is None:
        return None
    formula = _render_condition_formula(skeleton)
    combine = _basic_combine_for_skeleton(skeleton)
    return {
        "rows": label_condition_rows(rows),
        "mode": "Basic" if combine is not None else "Advanced",
        "combine": combine or "AND",
        "formula": formula,
    }


def _condition_skeleton(  # noqa: PLR0911
    expression: Any,
    rows: list[dict[str, Any]],
) -> tuple[str, Any] | None:
    """Collect leaf rows and return the boolean structure over their indexes."""
    if not isinstance(expression, dict):
        return None
    op = expression.get("op")
    if op in {"and", "or"} and isinstance(expression.get("args"), list):
        children = []
        for arg in expression["args"]:
            child = _condition_skeleton(arg, rows)
            if child is None:
                return None
            children.append(child)
        if len(children) < 2:
            return None
        return (str(op), children)
    if op == "not":
        child = _condition_skeleton(expression.get("arg"), rows)
        return None if child is None else ("not", child)
    row = _filter_row_from_expression(expression)
    if row is None:
        return None
    rows.append(row)
    return ("ref", len(rows))


def _basic_combine_for_skeleton(skeleton: tuple[str, Any]) -> str | None:
    """Return AND/OR when the skeleton is one flat conjunction, else None."""
    kind = skeleton[0]
    if kind == "ref":
        return "AND"
    if kind in {"and", "or"} and all(child[0] == "ref" for child in skeleton[1]):
        return "AND" if kind == "and" else "OR"
    return None


def _render_condition_formula(skeleton: tuple[str, Any]) -> str:
    """Render a skeleton as canonical formula text that reparses identically.

    Any nested and/or child is parenthesized — even under the same operator —
    so ``(E1 AND E2) AND E3`` survives the round trip instead of flattening.
    """
    kind = skeleton[0]
    if kind == "ref":
        return f"E{skeleton[1]}"
    if kind == "not":
        child = skeleton[1]
        rendered = _render_condition_formula(child)
        return f"NOT ({rendered})" if child[0] in {"and", "or"} else f"NOT {rendered}"
    parts = []
    for child in skeleton[1]:
        rendered = _render_condition_formula(child)
        parts.append(f"({rendered})" if child[0] in {"and", "or"} else rendered)
    return (" AND " if kind == "and" else " OR ").join(parts)


def case_value_expression(kind: str, value: Any) -> dict[str, Any]:
    """Build a case then/else value atom from visual-builder inputs."""
    if str(kind).strip().casefold() == "field":
        field = str(value or "").strip()
        if not field:
            raise ValueError("a field name is required when the value kind is Field")
        return {"col": field}
    return {"lit": _safe_literal(value)}


def compile_case_expression(
    branches: list[dict[str, Any]],
    *,
    else_kind: str,
    else_value: Any,
) -> dict[str, Any]:
    """Compile visual-builder branches into one ``case`` expression AST.

    Each branch is ``{"conditions": rows, "combine": "all"|"any",
    "then_kind": "Literal"|"Field", "then_value": scalar}``. Branches are
    evaluated in order and the first matching condition wins; the ``else``
    value is required because the expression grammar requires it.
    """
    when: list[dict[str, Any]] = []
    for number, branch in enumerate(branches, start=1):
        condition = compile_condition_rows(
            list(branch.get("conditions", [])),
            combine=str(branch.get("combine", "all")),
        )
        if condition is None:
            raise ValueError(
                f"branch {number} needs at least one complete condition row (field and operator)"
            )
        try:
            then_value = case_value_expression(
                str(branch.get("then_kind", "Literal")),
                branch.get("then_value", ""),
            )
        except ValueError as exc:
            raise ValueError(f"branch {number}: {exc}") from exc
        when.append({"cond": condition, "then": then_value})
    if not when:
        raise ValueError("add at least one branch")
    try:
        else_expression = case_value_expression(str(else_kind), else_value)
    except ValueError as exc:
        raise ValueError(f"else value: {exc}") from exc
    return {"op": "case", "when": when, "else": else_expression}


def _literal_editor_text(value: Any) -> str | None:
    """Format a ``lit`` scalar as editor text that survives ``_safe_literal``.

    Returns ``None`` when no text round-trips to ``value`` — e.g. the string
    ``"3.5"`` would re-parse as a float — so callers can refuse a lossy
    conversion and keep the raw AST instead.
    """
    if value is None:
        text = "null"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    return text if _safe_literal(text) == value else None


def case_value_from_expression(expression: Any) -> tuple[str, str] | None:
    """Reverse of :func:`case_value_expression`: atom to ``(kind, editor text)``."""
    if not isinstance(expression, dict):
        return None
    if set(expression) == {"col"}:
        column = str(expression["col"]).strip()
        return None if not column else ("Field", column)
    if set(expression) == {"lit"}:
        text = _literal_editor_text(expression["lit"])
        return None if text is None else ("Literal", text)
    return None


def condition_rows_from_expression(
    expression: Any,
) -> tuple[list[dict[str, Any]], str] | None:
    """Reverse of :func:`compile_condition_rows`: flat and/or to ``(rows, combine)``.

    Only one level of ``and``/``or`` over row-mappable predicates is
    representable; anything nested or beyond the row operators returns
    ``None``.
    """
    if not isinstance(expression, dict):
        return None
    op = expression.get("op")
    if op in {"and", "or"} and isinstance(expression.get("args"), list):
        rows: list[dict[str, Any]] = []
        for arg in expression["args"]:
            row = _filter_row_from_expression(arg) if isinstance(arg, dict) else None
            if row is None:
                return None
            rows.append(row)
        return rows, ("any" if op == "or" else "all")
    row = _filter_row_from_expression(expression)
    return None if row is None else ([row], "all")


def visual_case_state_from_expression(text: str) -> dict[str, Any] | None:  # noqa: PLR0911
    """Decompile AST YAML into visual case-builder state.

    Returns ``{"shape": "case" | "condition", "branches": [...], "else_kind",
    "else_value"}`` where each branch is ``{"rows", "combine", "then_kind",
    "then_value"}`` — the exact inputs :func:`compile_case_expression` and
    :func:`compile_condition_rows` consume. ``when_then`` normalizes to a
    one-branch case. Returns ``None`` for empty or invalid text and for
    expressions beyond the builder (nested logic, computed values, more than
    ``VISUAL_CASE_MAX_BRANCHES`` branches).
    """
    stripped = str(text or "").strip()
    if not stripped:
        return None
    try:
        expression = parse_expression_yaml(stripped)
    except (yaml.YAMLError, ValueError):
        return None
    op = expression.get("op")
    if op == "when_then":
        branch = _case_branch_state(expression.get("cond"), expression.get("then"))
        return _case_state([branch], expression.get("else"))
    if op == "case":
        when = expression.get("when")
        if not isinstance(when, list) or not 1 <= len(when) <= VISUAL_CASE_MAX_BRANCHES:
            return None
        branches = [
            _case_branch_state(item.get("cond"), item.get("then"))
            if isinstance(item, dict)
            else None
            for item in when
        ]
        return _case_state(branches, expression.get("else"))
    condition = condition_rows_from_expression(expression)
    if condition is None:
        return None
    rows, combine = condition
    return {
        "shape": "condition",
        "branches": [{"rows": rows, "combine": combine, "then_kind": "Literal", "then_value": ""}],
        "else_kind": "Literal",
        "else_value": "",
    }


def _case_branch_state(cond: Any, then: Any) -> dict[str, Any] | None:
    condition = condition_rows_from_expression(cond)
    value = case_value_from_expression(then)
    if condition is None or value is None:
        return None
    rows, combine = condition
    return {"rows": rows, "combine": combine, "then_kind": value[0], "then_value": value[1]}


def _case_state(
    branches: list[dict[str, Any] | None],
    else_expression: Any,
) -> dict[str, Any] | None:
    else_value = case_value_from_expression(else_expression)
    if else_value is None or any(branch is None for branch in branches):
        return None
    return {
        "shape": "case",
        "branches": branches,
        "else_kind": else_value[0],
        "else_value": else_value[1],
    }


def parse_expression_yaml(text: str) -> dict[str, Any]:
    """Parse and validate one YAML/JSON expression AST."""
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("expression must be a YAML mapping")
    return expr_parser.to_dict(expr_parser.parse(loaded))


def calculated_expression_example(mode: str) -> str:
    """Return one copy-ready example for the focused expression editor."""

    if mode == "Polars":
        return 'pl.col("Revenue") - pl.col("Cost")'
    return (
        "op: case\n"
        "when:\n"
        "  - cond: {op: gt, column: Revenue, value: 100}\n"
        "    then: {lit: High}\n"
        "else: {lit: Standard}"
    )


def validate_calculated_expression(mode: str, text: str) -> CalculatedExpressionValidation:
    """Validate custom calculated-field text with concise remediation messages."""

    expression_text = str(text or "").strip()
    if not expression_text:
        return CalculatedExpressionValidation(
            valid=False,
            messages=(f"Enter a {mode} expression before applying it.",),
        )
    if mode == "Polars":
        return _validate_polars_calculated_expression(expression_text)
    return _validate_ast_calculated_expression(expression_text)


def _validate_polars_calculated_expression(text: str) -> CalculatedExpressionValidation:
    try:
        expr_translator.translate(expr_parser.parse({"polars": text}))
    except (expr_parser.ParseError, expr_translator.TranslationError) as exc:
        return CalculatedExpressionValidation(
            valid=False,
            messages=(_friendly_polars_expression_error(exc),),
            technical_details=str(exc),
        )
    return CalculatedExpressionValidation(valid=True)


def _validate_ast_calculated_expression(text: str) -> CalculatedExpressionValidation:
    """Parse AST YAML while retaining structured errors for friendly mapping."""

    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = (
            f" near line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
        )
        return CalculatedExpressionValidation(
            valid=False,
            messages=(
                f"Expression YAML is not valid{location}. Check indentation, colons, and brackets.",
            ),
            technical_details=str(exc),
        )
    if not isinstance(loaded, dict):
        return CalculatedExpressionValidation(
            valid=False,
            messages=("Expression YAML must be a mapping such as `col: Revenue` or `op: add`.",),
            technical_details=f"Parsed YAML value has type {type(loaded).__name__}: {loaded!r}",
        )
    try:
        expr_parser.parse(loaded)
    except expr_parser.ParseError as exc:
        return CalculatedExpressionValidation(
            valid=False,
            messages=_friendly_ast_expression_errors(exc, loaded),
            technical_details=str(exc),
        )
    return CalculatedExpressionValidation(valid=True)


def _friendly_polars_expression_error(exc: ValueError) -> str:
    detail = str(exc)
    lowered = detail.casefold()
    if "invalid polars expression syntax" in lowered:
        return "Polars expression syntax is invalid. Check brackets, quotes, and operators."
    if "unsupported name" in lowered:
        return "Only the `pl` namespace is available in a Polars expression."
    if "private" in lowered:
        return "Private Polars attributes and methods are not allowed."
    if (
        "unsupported polars function" in lowered
        or "unsupported polars expression method" in lowered
    ):
        return "That Polars function or method is not supported by calculated fields."
    if "must evaluate to a polars.expr" in lowered:
        return 'The expression must return a Polars expression such as `pl.col("Revenue")`.'
    return "Polars could not validate this expression. Review the supported syntax and example."


def _friendly_ast_expression_errors(
    exc: expr_parser.ParseError,
    loaded: dict[str, Any],
) -> tuple[str, ...]:
    cause = exc.__cause__
    raw_errors = cause.errors() if cause is not None and hasattr(cause, "errors") else []
    messages: list[str] = []
    for error in raw_errors:
        location = tuple(error.get("loc", ()))
        path = _expression_error_path(location, loaded)
        error_type = str(error.get("type", ""))
        leaf = str(location[-1]) if location else ""
        if leaf == "cond" and error_type in {"union_tag_not_found", "union_tag_invalid"}:
            message = (
                f"`{path}` must be a condition such as `{{op: gt, column: Revenue, value: 100}}`."
            )
        elif leaf == "else" and error_type == "missing":
            message = "`else` is required for a `case` expression."
        elif leaf == "otherwise":
            message = "`otherwise` is not supported for `case`; use `else:` instead."
        elif error_type in {"union_tag_not_found", "union_tag_invalid"}:
            message = (
                f"`{path}` must start with `col`, `lit`, `param`, `polars`, or a supported `op`."
            )
        elif error_type == "missing":
            message = f"`{path}` is required."
        elif error_type == "extra_forbidden":
            message = f"`{path}` is not supported here; remove it or check the example."
        elif error_type in {"too_short", "list_too_short"}:
            message = f"`{path}` needs more items for this operation."
        elif error_type == "literal_error":
            message = f"`{path}` contains an unsupported value."
        else:
            message = f"`{path}` is invalid. Check the expression example and required fields."
        if message not in messages:
            messages.append(message)
    return tuple(messages) or (
        "Expression structure is invalid. Check the example and required fields.",
    )


def _expression_error_path(location: tuple[Any, ...], loaded: dict[str, Any]) -> str:
    parts = list(location)
    if parts and parts[0] == loaded.get("op"):
        parts.pop(0)
    path = ""
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += ("." if path else "") + str(part)
    return path or "expression"


def calculated_rows_from_source(source: model.Source) -> list[dict[str, Any]]:
    """Return editor rows for source-level derived columns."""
    rows: list[dict[str, Any]] = []
    for transform in source.transforms:
        if not isinstance(transform, model.DeriveColumn):
            continue
        expression = expr_parser.to_dict(transform.expression)
        recognized = calculation_mode_from_expression(expression)
        if recognized is not None:
            rows.append({"Name": transform.output, **recognized, "Expression": "", "Enabled": True})
            continue
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
    return rows


def calculated_rows_for_editor(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize calculated-field rows to the current editor column shape."""
    normalized: list[dict[str, Any]] = []
    for row in rows:
        mode = str(row.get("Mode", "") or "AST YAML").strip()
        normalized.append(
            {
                "Name": str(row.get("Name", "") or ""),
                "Mode": mode,
                "Left": str(row.get("Left", "") or ""),
                "Right Kind": str(row.get("Right Kind", "") or "Field"),
                "Right": str(row.get("Right", "") or ""),
                "Expression": str(row.get("Expression") or row.get("Expression YAML", "") or ""),
                "Enabled": editor_row_enabled(row.get("Enabled")),
            }
        )
    return normalized


def build_derive_column_transforms(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build ``derive_column`` transforms from calculated-field editor rows."""
    transforms: list[dict[str, Any]] = []
    for row in rows:
        if not editor_row_enabled(row.get("Enabled")):
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


def _builder_calculation_expression(  # noqa: PLR0911, PLR0912
    mode: str,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    left = str(row.get("Left", "")).strip()
    right = str(row.get("Right", "")).strip()
    right_kind = str(row.get("Right Kind", "Field")).strip()
    if not left:
        return None
    left_expr = {"col": left}
    if mode == "Copy Field":
        return left_expr
    if mode == "Absolute Value":
        return {"op": "abs", "arg": left_expr}
    if mode in _DATE_PART_MODE_UNITS:
        return {"op": "date_part", "unit": _DATE_PART_MODE_UNITS[mode], "arg": left_expr}
    if mode == "Round":
        if not right:
            return {"op": "round", "arg": left_expr}
        ndigits = _safe_literal(right)
        if isinstance(ndigits, bool) or not isinstance(ndigits, int):
            return None
        return {"op": "round", "arg": left_expr, "ndigits": ndigits}
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
    if mode in _DATE_DIFF_MODE_UNITS:
        return {
            "op": "date_diff",
            "unit": _DATE_DIFF_MODE_UNITS[mode],
            "end": left_expr,
            "start": right_expr,
        }
    return None


_ARITHMETIC_OP_MODES = {"add": "Add", "sub": "Subtract", "mul": "Multiply", "div": "Divide"}
_DATE_DIFF_UNIT_MODES = {unit: mode for mode, unit in _DATE_DIFF_MODE_UNITS.items()}
_DATE_PART_UNIT_MODES = {unit: mode for mode, unit in _DATE_PART_MODE_UNITS.items()}


def _column_operand(expression: Any) -> str | None:
    """Return the column name when ``expression`` is a bare ``{col: ...}`` atom."""
    if isinstance(expression, dict) and set(expression) == {"col"}:
        column = str(expression["col"])
        if column and column == column.strip():
            return column
    return None


def _right_operand(expression: Any) -> tuple[str, str] | None:
    """Map an operand atom to grid ``(Right Kind, Right)``; None when not exact.

    A blank right text is rejected because the grid treats an empty Right as
    an incomplete row, so it could never compile back to ``expression``.
    """
    operand = case_value_from_expression(expression)
    if operand is None or not operand[1]:
        return None
    return operand


def calculation_mode_from_expression(  # noqa: PLR0911, PLR0912
    expression: Any,
) -> dict[str, str] | None:
    """Recognize the simple-mode row whose compile is exactly ``expression``.

    Returns ``{"Mode", "Left", "Right Kind", "Right"}`` when one of the
    Left/Right grid modes reproduces ``expression`` verbatim, so catalog loads
    can show the friendly mode instead of raw AST YAML. Extra keys, nested
    operands, and literals that would not survive the editor-text round trip
    return ``None`` and stay AST YAML.
    """
    if not isinstance(expression, dict):
        return None
    left = _column_operand(expression)
    if left is not None:
        return _mode_row("Copy Field", left)
    keys = set(expression)
    op = expression.get("op")
    left = _column_operand(expression.get("arg"))
    if op == "abs" and keys == {"op", "arg"} and left is not None:
        return _mode_row("Absolute Value", left)
    if op == "round" and keys in ({"op", "arg"}, {"op", "arg", "ndigits"}) and left is not None:
        ndigits = expression.get("ndigits")
        if "ndigits" not in expression:
            return _mode_row("Round", left)
        if isinstance(ndigits, int) and not isinstance(ndigits, bool):
            return _mode_row("Round", left, right=str(ndigits), right_kind="Literal")
        return None
    if op == "date_part" and keys == {"op", "unit", "arg"} and left is not None:
        mode = _DATE_PART_UNIT_MODES.get(str(expression.get("unit")))
        return None if mode is None else _mode_row(mode, left)
    if op in _ARITHMETIC_OP_MODES and keys == {"op", "args"}:
        return _binary_mode_row(_ARITHMETIC_OP_MODES[str(op)], expression.get("args"))
    if op == "coalesce" and keys == {"op", "args"}:
        return _binary_mode_row("Coalesce", expression.get("args"))
    if op == "concat" and keys == {"op", "args", "sep"} and expression.get("sep") == "":
        return _binary_mode_row("Concat", expression.get("args"))
    if op == "safe_div" and keys == {"op", "num", "den"}:
        return _binary_mode_row("Safe Divide", [expression.get("num"), expression.get("den")])
    if op == "date_diff" and keys == {"op", "unit", "end", "start"}:
        mode = _DATE_DIFF_UNIT_MODES.get(str(expression.get("unit")))
        if mode is None:
            return None
        return _binary_mode_row(mode, [expression.get("end"), expression.get("start")])
    return None


def _binary_mode_row(mode: str, args: Any) -> dict[str, str] | None:
    if not isinstance(args, list) or len(args) != 2:
        return None
    left = _column_operand(args[0])
    right = _right_operand(args[1])
    if left is None or right is None:
        return None
    return _mode_row(mode, left, right=right[1], right_kind=right[0])


def _mode_row(
    mode: str,
    left: str,
    *,
    right: str = "",
    right_kind: str = "Field",
) -> dict[str, str]:
    return {"Mode": mode, "Left": left, "Right Kind": right_kind, "Right": right}


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
    data = metric.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    )
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


def chart_kind_label(chart_kind: str) -> str:
    """Return the shared chart-catalog display label for a technical kind."""

    return CHART_DISPLAY_LABELS.get(chart_kind, title_from_identifier(chart_kind))


def chart_kind_purpose(chart_kind: str) -> str:
    """Return the shared plain-language purpose for a technical chart kind."""

    return CHART_DISPLAY_PURPOSES.get(
        chart_kind,
        "Present the selected metric in a configured report tile.",
    )


def chart_kind_selector_label(chart_kind: str) -> str:
    """Format a chart select option without replacing its stored technical value."""

    if chart_kind == "All":
        return "All chart types"
    return chart_kind_label(chart_kind)


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
    # Variant metrics may name their column per metric (any group-by dimension
    # qualifies), so a processor-level variant_column is a default, not a gate.
    has_variant = has_outcome_counts and bool(
        extra.get("variant_column") or getattr(processor, "group_by", None)
    )
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


def stage_condition_rows(when: Any) -> tuple[list[dict[str, Any]], str, bool]:
    """Best-effort conversion from a stage ``when`` AST to editable rule rows.

    Returns ``(rows, combine, representable)`` where ``combine`` is ``"all"``
    for ``and``-joined (or single) conditions and ``"any"`` for ``or``-joined
    single conditions. Expressions the rules form cannot represent return
    ``([], "all", False)`` so callers can fall back to the raw-AST editor.
    """
    if when in (None, {}):
        return [], "all", True
    if not isinstance(when, dict):
        return [], "all", False
    rows = filter_rows_from_expression(when)
    if rows is not None:
        return rows, "all", True
    if when.get("op") == "or" and isinstance(when.get("args"), list):
        or_rows: list[dict[str, Any]] = []
        for arg in when["args"]:
            parsed = _filter_row_from_expression(arg) if isinstance(arg, dict) else None
            if parsed is None:
                return [], "all", False
            or_rows.append(parsed)
        return or_rows, "any", True
    return [], "all", False


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
    # Quantile boxes read the digest behind a tdigest_quantile metric; other
    # metric kinds have no distribution to draw.
    if metric.kind != "tdigest_quantile":
        choices = [name for name in choices if name != "boxplot"]
    return sorted(choices)


def metric_output_columns(metric_name: str, metric: model.Metric) -> list[str]:
    """Best-effort output column names for a metric."""
    return _report_metric_output_columns(metric_name, metric)


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
    return report_field_options(catalog, metric_name)


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


def stable_catalog_id(
    name: str,
    *,
    fallback: str,
    parent_id: str = "",
    existing_ids: Iterable[str] = (),
    preferred_id: str = "",
) -> str:
    """Build a readable, deterministic catalog id with numeric collision suffixes.

    A valid existing id wins so changing a display title cannot break references.
    New ids include the artifact type and a bounded parent hint, which keeps page
    and tile anchors understandable without leaking random hashes into the UI.
    """

    used = {str(value) for value in existing_ids if str(value)}
    preferred = str(preferred_id).strip()
    if preferred and catalog_id_is_safe(preferred) and preferred not in used:
        return preferred

    semantic = _catalog_id_slug(name) or fallback
    parent = _catalog_id_slug(parent_id)
    parts = [parent[:32].strip("_") if parent else "", _catalog_id_slug(fallback), semantic]
    base = "_".join(part for part in parts if part)[:64].strip("_") or fallback
    if not base[0].isalpha():
        base = f"{fallback}_{base}"[:64].strip("_") or fallback
    candidate = base
    counter = 2
    while candidate in used:
        suffix = f"_{counter}"
        candidate = f"{base[: 64 - len(suffix)].rstrip('_')}{suffix}"
        counter += 1
    return candidate


def _catalog_id_stem(name: str, *, fallback: str) -> str:
    stem = _catalog_id_slug(name)[:20].strip("_") or fallback
    if not stem[0].isalpha():
        stem = f"{fallback}_{stem}"[:20].strip("_") or fallback
    return stem


def _catalog_id_slug(value: str) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    return "_".join("".join(chars).split("_")).strip("_")


def catalog_id_is_safe(value: str) -> bool:
    """Return whether an id is ASCII, letter-prefixed, and YAML-reference safe."""

    return (
        bool(value)
        and value[0].isascii()
        and value[0].isalpha()
        and all(char.isascii() and (char.isalnum() or char == "_") for char in value)
    )


def metric_yaml(metric_name: str, metric_def: dict[str, Any]) -> str:
    """Render a metric draft as YAML."""
    return yaml.safe_dump({"metrics": {metric_name: metric_def}}, sort_keys=False)


def tile_yaml(tile: dict[str, Any]) -> str:
    """Render a tile draft as YAML."""
    return yaml.safe_dump({"tiles": [tile]}, sort_keys=False)


CATALOG_FILENAMES = ("pipelines.yaml", "processors.yaml", "metrics.yaml", "dashboards.yaml")


@dataclass(frozen=True, slots=True)
class SourceCascadePlan:
    """Catalog definitions removed together with one source."""

    source_id: str
    processor_ids: tuple[str, ...]
    metric_ids: tuple[str, ...]
    tile_locations: tuple[str, ...]
    page_filter_locations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProcessorCascadePlan:
    """Catalog definitions removed together with one processor."""

    processor_id: str
    metric_ids: tuple[str, ...]
    tile_locations: tuple[str, ...]
    page_filter_locations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MetricDeletePlan:
    """Exact dependencies considered before deleting one metric."""

    metric_id: str
    dependent_metric_ids: tuple[str, ...]
    tile_locations: tuple[str, ...]
    page_filter_locations: tuple[str, ...]


def _transitive_metric_dependants(
    catalog: model.Catalog,
    metric_ids: set[str],
) -> set[str]:
    """Return ``metric_ids`` plus every metric that transitively depends on them."""

    closure = set(metric_ids)
    while True:
        dependants = {
            name
            for name, metric in catalog.metrics.metrics.items()
            if name not in closure
            and any(dependency in closure for dependency in metric.depends_on)
        }
        if not dependants:
            return closure
        closure.update(dependants)


def _dashboard_dependency_locations(
    catalog: model.Catalog,
    *,
    metric_ids: set[str],
    processor_ids: set[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return tile and newly unsupported-filter paths for a metric removal."""

    removed_processors = processor_ids or set()
    remaining_metrics = {
        name: metric for name, metric in catalog.metrics.metrics.items() if name not in metric_ids
    }
    remaining_processors = {
        processor.id: processor
        for processor in catalog.processors.processors
        if processor.id not in removed_processors
    }
    tile_locations: list[str] = []
    page_filter_locations: list[str] = []
    for dashboard in catalog.dashboards.dashboards:
        for page in dashboard.pages:
            remaining_tiles = [tile for tile in page.tiles if tile.metric not in metric_ids]
            tile_locations.extend(
                f"{dashboard.id}/{page.id}/{tile.id}"
                for tile in page.tiles
                if tile.metric in metric_ids
            )
            page_filter_locations.extend(
                f"{dashboard.id}/{page.id}/{filter_spec.field}"
                for filter_spec in page.filters
                if not _page_filter_has_support(
                    filter_spec.field,
                    remaining_tiles,
                    remaining_metrics,
                    remaining_processors,
                )
            )
    sort_key = str.casefold
    return (
        tuple(sorted(tile_locations, key=sort_key)),
        tuple(sorted(page_filter_locations, key=sort_key)),
    )


def source_cascade_plan(catalog: model.Catalog, source_id: str) -> SourceCascadePlan:
    """Return the complete, deterministic catalog cascade for ``source_id``."""

    normalized_source_id = str(source_id).strip()
    if normalized_source_id not in {source.id for source in catalog.pipelines.sources}:
        raise ValueError(f"unknown source {normalized_source_id!r}")

    processor_ids = {
        processor.id
        for processor in catalog.processors.processors
        if processor.source == normalized_source_id
    }
    metric_ids = {
        name for name, metric in catalog.metrics.metrics.items() if metric.source in processor_ids
    }
    metric_ids = _transitive_metric_dependants(catalog, metric_ids)
    tile_locations, page_filter_locations = _dashboard_dependency_locations(
        catalog,
        metric_ids=metric_ids,
        processor_ids=processor_ids,
    )

    sort_key = str.casefold
    return SourceCascadePlan(
        source_id=normalized_source_id,
        processor_ids=tuple(sorted(processor_ids, key=sort_key)),
        metric_ids=tuple(sorted(metric_ids, key=sort_key)),
        tile_locations=tile_locations,
        page_filter_locations=page_filter_locations,
    )


def processor_cascade_plan(
    catalog: model.Catalog,
    processor_id: str,
) -> ProcessorCascadePlan:
    """Return the complete, deterministic catalog cascade for one processor."""

    normalized_processor_id = str(processor_id).strip()
    if normalized_processor_id not in {processor.id for processor in catalog.processors.processors}:
        raise ValueError(f"unknown processor {normalized_processor_id!r}")
    direct_metrics = {
        name
        for name, metric in catalog.metrics.metrics.items()
        if metric.source == normalized_processor_id
    }
    metric_ids = _transitive_metric_dependants(catalog, direct_metrics)
    tile_locations, page_filter_locations = _dashboard_dependency_locations(
        catalog,
        metric_ids=metric_ids,
        processor_ids={normalized_processor_id},
    )
    return ProcessorCascadePlan(
        processor_id=normalized_processor_id,
        metric_ids=tuple(sorted(metric_ids, key=str.casefold)),
        tile_locations=tile_locations,
        page_filter_locations=page_filter_locations,
    )


def metric_delete_plan(catalog: model.Catalog, metric_id: str) -> MetricDeletePlan:
    """Return exact blockers and report dependencies for one metric deletion."""

    normalized_metric_id = str(metric_id).strip()
    if normalized_metric_id not in catalog.metrics.metrics:
        raise ValueError(f"unknown metric {normalized_metric_id!r}")
    closure = _transitive_metric_dependants(catalog, {normalized_metric_id})
    dependent_metric_ids = closure - {normalized_metric_id}
    tile_locations, page_filter_locations = _dashboard_dependency_locations(
        catalog,
        metric_ids={normalized_metric_id},
    )
    return MetricDeletePlan(
        metric_id=normalized_metric_id,
        dependent_metric_ids=tuple(sorted(dependent_metric_ids, key=str.casefold)),
        tile_locations=tile_locations,
        page_filter_locations=page_filter_locations,
    )


def delete_source_cascade(
    workspace: str | Path,
    source_id: str,
) -> SourceCascadePlan:
    """Delete one source and every catalog definition that depends on it.

    Aggregate files and run metadata intentionally remain untouched. The
    catalog and chat-description writes share one rollback boundary and the
    resulting workspace is validated before the deletion is committed.
    """

    catalog = load(workspace)
    plan = source_cascade_plan(catalog, source_id)
    processor_ids = set(plan.processor_ids)
    metric_ids = set(plan.metric_ids)

    with workspace_configuration_transaction(workspace):
        pipelines_path = _catalog_file(workspace, "pipelines.yaml")
        pipelines = _read_yaml(pipelines_path)
        sources = pipelines.get("sources", [])
        if not isinstance(sources, list):
            raise ValueError("pipelines.yaml must contain a list at `sources`")
        pipelines["sources"] = [source for source in sources if source.get("id") != plan.source_id]

        processors_path = _catalog_file(workspace, "processors.yaml")
        processors = _read_yaml(processors_path)
        processor_defs = processors.get("processors", [])
        if not isinstance(processor_defs, list):
            raise ValueError("processors.yaml must contain a list at `processors`")
        processors["processors"] = [
            processor for processor in processor_defs if processor.get("id") not in processor_ids
        ]

        metrics_path = _catalog_file(workspace, "metrics.yaml")
        metrics = _read_yaml(metrics_path)
        metric_defs = metrics.get("metrics", {})
        if not isinstance(metric_defs, dict):
            raise ValueError("metrics.yaml must contain a mapping at `metrics`")
        metrics["metrics"] = {
            name: definition for name, definition in metric_defs.items() if name not in metric_ids
        }

        dashboards_path = _catalog_file(workspace, "dashboards.yaml")
        dashboards = _read_yaml(dashboards_path)
        _remove_source_dashboard_dependencies(
            dashboards,
            metric_ids=metric_ids,
            remaining_metrics={
                name: metric
                for name, metric in catalog.metrics.metrics.items()
                if name not in metric_ids
            },
            remaining_processors={
                processor.id: processor
                for processor in catalog.processors.processors
                if processor.id not in processor_ids
            },
        )

        model.Pipelines.model_validate(pipelines)
        model.Processors.model_validate(processors)
        model.Metrics.model_validate(metrics)
        model.Dashboards.model_validate(dashboards)

        # Remove downstream definitions first so readers never observe a
        # dangling tile/metric/processor reference during this process.
        _write_yaml(dashboards_path, dashboards)
        _write_yaml(metrics_path, metrics)
        _write_yaml(processors_path, processors)
        _write_yaml(pipelines_path, pipelines)
        _remove_source_chat_descriptions(
            workspace,
            source_id=plan.source_id,
            processor_ids=processor_ids,
            metric_ids=metric_ids,
        )
        require_valid_workspace(workspace)
    return plan


def delete_processor_cascade(
    workspace: str | Path,
    processor_id: str,
    *,
    source_columns_by_id: Mapping[str, Iterable[str]] | None = None,
) -> ProcessorCascadePlan:
    """Delete exactly one processor and every catalog definition that depends on it.

    Persisted aggregates and run metadata intentionally remain untouched. The
    catalog and Chat-description writes share one rollback boundary and the
    resulting workspace is validated before commit. ``source_columns_by_id``
    provides the same observed-schema context authoring transactions use, so a
    delete is not rolled back over columns the remaining catalog can no longer
    seed by declaration alone.
    """

    catalog = load(workspace)
    plan = processor_cascade_plan(catalog, processor_id)
    metric_ids = set(plan.metric_ids)

    with workspace_configuration_transaction(workspace):
        processors_path = _catalog_file(workspace, "processors.yaml")
        processors = _read_yaml(processors_path)
        processor_defs = processors.get("processors", [])
        if not isinstance(processor_defs, list):
            raise ValueError("processors.yaml must contain a list at `processors`")
        processors["processors"] = [
            processor for processor in processor_defs if processor.get("id") != plan.processor_id
        ]

        metrics_path = _catalog_file(workspace, "metrics.yaml")
        metrics = _read_yaml(metrics_path)
        metric_defs = metrics.get("metrics", {})
        if not isinstance(metric_defs, dict):
            raise ValueError("metrics.yaml must contain a mapping at `metrics`")
        metrics["metrics"] = {
            name: definition for name, definition in metric_defs.items() if name not in metric_ids
        }

        dashboards_path = _catalog_file(workspace, "dashboards.yaml")
        dashboards = _read_yaml(dashboards_path)
        _remove_source_dashboard_dependencies(
            dashboards,
            metric_ids=metric_ids,
            remaining_metrics={
                name: metric
                for name, metric in catalog.metrics.metrics.items()
                if name not in metric_ids
            },
            remaining_processors={
                processor.id: processor
                for processor in catalog.processors.processors
                if processor.id != plan.processor_id
            },
        )

        model.Processors.model_validate(processors)
        model.Metrics.model_validate(metrics)
        model.Dashboards.model_validate(dashboards)

        _write_yaml(dashboards_path, dashboards)
        _write_yaml(metrics_path, metrics)
        _write_yaml(processors_path, processors)
        _remove_chat_descriptions(
            workspace,
            processor_ids={plan.processor_id},
            metric_ids=metric_ids,
        )
        require_valid_workspace(workspace, source_columns_by_id=source_columns_by_id)
    return plan


def delete_metric_definition(
    workspace: str | Path,
    metric_id: str,
    *,
    cascade_tiles: bool,
    source_columns_by_id: Mapping[str, Iterable[str]] | None = None,
) -> MetricDeletePlan:
    """Delete exactly one metric after explicit handling of its dependencies.

    Metrics that depend on the target are blockers rather than implicit cascade
    targets. Report tiles may be removed only when ``cascade_tiles`` is true.
    """

    catalog = load(workspace)
    plan = metric_delete_plan(catalog, metric_id)
    if plan.dependent_metric_ids:
        names = ", ".join(plan.dependent_metric_ids)
        raise ValueError(f"metric {plan.metric_id!r} is required by dependent metric(s): {names}")
    if plan.tile_locations and not cascade_tiles:
        raise ValueError(
            f"metric {plan.metric_id!r} is used by report tiles; choose the tile cascade explicitly"
        )

    with workspace_configuration_transaction(workspace):
        metrics_path = _catalog_file(workspace, "metrics.yaml")
        metrics = _read_yaml(metrics_path)
        metric_defs = metrics.get("metrics", {})
        if not isinstance(metric_defs, dict):
            raise ValueError("metrics.yaml must contain a mapping at `metrics`")
        metrics["metrics"] = {
            name: definition for name, definition in metric_defs.items() if name != plan.metric_id
        }

        dashboards_path = _catalog_file(workspace, "dashboards.yaml")
        dashboards = _read_yaml(dashboards_path)
        if cascade_tiles:
            _remove_source_dashboard_dependencies(
                dashboards,
                metric_ids={plan.metric_id},
                remaining_metrics={
                    name: metric
                    for name, metric in catalog.metrics.metrics.items()
                    if name != plan.metric_id
                },
                remaining_processors={
                    processor.id: processor for processor in catalog.processors.processors
                },
            )

        model.Metrics.model_validate(metrics)
        model.Dashboards.model_validate(dashboards)
        _write_yaml(dashboards_path, dashboards)
        _write_yaml(metrics_path, metrics)
        _remove_chat_descriptions(workspace, metric_ids={plan.metric_id})
        require_valid_workspace(workspace, source_columns_by_id=source_columns_by_id)
    return plan


def delete_report_page(
    workspace: str | Path,
    dashboard_id: str,
    page_id: str,
    *,
    source_columns_by_id: Mapping[str, Iterable[str]] | None = None,
) -> None:
    """Delete one dashboard page (and its tiles); drop the dashboard when empty."""

    with workspace_configuration_transaction(workspace):
        dashboards_path = _catalog_file(workspace, "dashboards.yaml")
        data = _read_yaml(dashboards_path)
        dashboards = data.get("dashboards", [])
        if not isinstance(dashboards, list):
            raise ValueError("dashboards.yaml must contain a list at `dashboards`")
        dashboard = next(
            (entry for entry in dashboards if entry.get("id") == dashboard_id), None
        )
        if dashboard is None:
            raise ValueError(f"dashboard {dashboard_id!r} was not found")
        pages = dashboard.get("pages", [])
        if not isinstance(pages, list) or not any(
            page.get("id") == page_id for page in pages
        ):
            raise ValueError(f"page {page_id!r} was not found in dashboard {dashboard_id!r}")
        dashboard["pages"] = [page for page in pages if page.get("id") != page_id]
        if not dashboard["pages"]:
            data["dashboards"] = [
                entry for entry in dashboards if entry.get("id") != dashboard_id
            ]
        model.Dashboards.model_validate(data)
        _write_yaml(dashboards_path, data)
        require_valid_workspace(workspace, source_columns_by_id=source_columns_by_id)


def delete_dashboard(
    workspace: str | Path,
    dashboard_id: str,
    *,
    source_columns_by_id: Mapping[str, Iterable[str]] | None = None,
) -> None:
    """Delete one dashboard with all of its pages and tiles."""

    with workspace_configuration_transaction(workspace):
        dashboards_path = _catalog_file(workspace, "dashboards.yaml")
        data = _read_yaml(dashboards_path)
        dashboards = data.get("dashboards", [])
        if not isinstance(dashboards, list):
            raise ValueError("dashboards.yaml must contain a list at `dashboards`")
        if not any(entry.get("id") == dashboard_id for entry in dashboards):
            raise ValueError(f"dashboard {dashboard_id!r} was not found")
        data["dashboards"] = [
            entry for entry in dashboards if entry.get("id") != dashboard_id
        ]
        model.Dashboards.model_validate(data)
        _write_yaml(dashboards_path, data)
        require_valid_workspace(workspace, source_columns_by_id=source_columns_by_id)


def _dimension_key(value: str) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _page_filter_has_support(
    field: str,
    tiles: Iterable[model.Tile],
    metrics: dict[str, model.Metric],
    processors: dict[str, model.Processor],
) -> bool:
    field_key = _dimension_key(field)
    for tile in tiles:
        metric = metrics.get(tile.metric)
        processor = processors.get(metric.source) if metric is not None else None
        if processor is not None and any(
            _dimension_key(column) == field_key for column in processor.group_by
        ):
            return True
    return False


def _remove_source_dashboard_dependencies(
    dashboards: dict[str, Any],
    *,
    metric_ids: set[str],
    remaining_metrics: dict[str, model.Metric],
    remaining_processors: dict[str, model.Processor],
) -> None:
    dashboard_defs = dashboards.get("dashboards", [])
    if not isinstance(dashboard_defs, list):
        raise ValueError("dashboards.yaml must contain a list at `dashboards`")
    for dashboard in dashboard_defs:
        pages = dashboard.get("pages", [])
        if not isinstance(pages, list):
            raise ValueError("dashboard must contain a list at `pages`")
        for page in pages:
            tiles = page.get("tiles", [])
            if not isinstance(tiles, list):
                raise ValueError("dashboard page must contain a list at `tiles`")
            page["tiles"] = [tile for tile in tiles if tile.get("metric") not in metric_ids]
            filters = page.get("filters", [])
            if not isinstance(filters, list):
                raise ValueError("dashboard page must contain a list at `filters`")
            remaining_tile_models = [model.Tile.model_validate(tile) for tile in page["tiles"]]
            page["filters"] = [
                filter_def
                for filter_def in filters
                if _page_filter_has_support(
                    str(filter_def.get("field", "")),
                    remaining_tile_models,
                    remaining_metrics,
                    remaining_processors,
                )
            ]
            if not page["filters"]:
                page.pop("filters", None)


def _remove_source_chat_descriptions(
    workspace: str | Path,
    *,
    source_id: str,
    processor_ids: set[str],
    metric_ids: set[str],
) -> None:
    _remove_chat_descriptions(
        workspace,
        source_ids={source_id},
        processor_ids=processor_ids,
        metric_ids=metric_ids,
    )


def _remove_chat_descriptions(
    workspace: str | Path,
    *,
    source_ids: set[str] | None = None,
    processor_ids: set[str] | None = None,
    metric_ids: set[str] | None = None,
) -> None:
    path = Path(workspace) / "ai.yaml"
    if not path.exists():
        return
    source_ids = source_ids or set()
    processor_ids = processor_ids or set()
    metric_ids = metric_ids or set()
    data = _read_yaml(path)
    blocks: list[dict[str, Any]] = []
    top_level = data.get("chat_with_data")
    if isinstance(top_level, dict):
        blocks.append(top_level)
    ai = data.get("ai")
    nested = ai.get("chat_with_data") if isinstance(ai, dict) else None
    if isinstance(nested, dict):
        blocks.append(nested)

    changed = False
    for block in blocks:
        dataset_descriptions = block.get("dataset_descriptions")
        if isinstance(dataset_descriptions, dict):
            for identifier in source_ids:
                if identifier in dataset_descriptions:
                    dataset_descriptions.pop(identifier)
                    changed = True
        metric_descriptions = block.get("metric_descriptions")
        if isinstance(metric_descriptions, dict):
            for identifier in processor_ids | metric_ids:
                if identifier in metric_descriptions:
                    metric_descriptions.pop(identifier)
                    changed = True
    if changed:
        _write_yaml(path, data)


@contextmanager
def catalog_transaction(workspace: str | Path) -> Iterator[None]:
    """Restore every catalog file if a multi-file authoring write fails midway."""
    root = Path(workspace)
    paths = [root / "catalog" / filename for filename in CATALOG_FILENAMES]
    with _configuration_file_transaction(paths):
        yield


@contextmanager
def validated_catalog_transaction(
    workspace: str | Path,
    *,
    source_columns_by_id: Mapping[str, Iterable[str]] | None = None,
) -> Iterator[None]:
    """Commit Builder catalog writes only when the resulting workspace is valid.

    ``source_columns_by_id`` supplies observed physical source columns so
    expression validation accepts data-only fields the editors offered.
    """

    with catalog_transaction(workspace):
        yield
        require_valid_workspace(workspace, source_columns_by_id=source_columns_by_id)


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
    snapshots = {path: path.read_bytes() if path.exists() else None for path in unique_paths}
    try:
        yield
    except BaseException:
        for path, content in snapshots.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
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


def rename_processor_definition(
    workspace: str | Path,
    old_id: str,
    processor_def: dict[str, Any],
    *,
    source_columns_by_id: Mapping[str, Iterable[str]] | None = None,
) -> None:
    """Rename one processor: rewrite it in place and retarget metric sources.

    Editing an ID previously wrote the definition under the new id and left
    the old one behind as a copy. A rename keeps exactly one definition,
    updates every metric whose ``source`` referenced the old id, and validates
    the workspace inside one rollback boundary.
    """

    new_id = str(processor_def.get("id", "")).strip()
    if not new_id:
        raise ValueError("processor definition must include `id`")
    if new_id == old_id:
        write_processor_definition(workspace, processor_def)
        return

    with workspace_configuration_transaction(workspace):
        processors_path = _catalog_file(workspace, "processors.yaml")
        processors = _read_yaml(processors_path)
        processor_defs = processors.get("processors", [])
        if not isinstance(processor_defs, list):
            raise ValueError("processors.yaml must contain a list at `processors`")
        if any(entry.get("id") == new_id for entry in processor_defs):
            raise ValueError(
                f"processor id {new_id!r} already exists; pick a different id to rename "
                f"{old_id!r}"
            )
        replaced = False
        for index, entry in enumerate(processor_defs):
            if entry.get("id") == old_id:
                processor_defs[index] = processor_def
                replaced = True
                break
        if not replaced:
            raise ValueError(f"processor {old_id!r} was not found")
        processors["processors"] = processor_defs

        metrics_path = _catalog_file(workspace, "metrics.yaml")
        metrics = _read_yaml(metrics_path)
        metric_defs = metrics.get("metrics", {})
        if not isinstance(metric_defs, dict):
            raise ValueError("metrics.yaml must contain a mapping at `metrics`")
        for definition in metric_defs.values():
            if isinstance(definition, dict) and definition.get("source") == old_id:
                definition["source"] = new_id

        model.Processors.model_validate(processors)
        model.Metrics.model_validate(metrics)
        _write_yaml(processors_path, processors)
        _write_yaml(metrics_path, metrics)
        require_valid_workspace(workspace, source_columns_by_id=source_columns_by_id)


def write_pipelines_definition(
    workspace: str | Path,
    pipelines_def: dict[str, Any],
) -> None:
    """Replace ``pipelines.yaml`` with one structurally validated full definition."""

    model.Pipelines.model_validate(pipelines_def)
    _write_yaml(_catalog_file(workspace, "pipelines.yaml"), pipelines_def)


def workspace_dimension_defaults(catalog: model.Catalog) -> list[str]:
    """Return the workspace-level common business dimensions.

    When ``defaults.dimensions`` is not set in ``pipelines.yaml``, the list is
    restored from the processors instead: the dimensions every processor with
    a group-by shares, in the first processor's order. Existing workspaces
    therefore surface their de-facto common list rather than an empty
    selector; the derived list is only persisted when the user applies it.
    """

    explicit = dedupe(
        [
            str(field).strip()
            for field in catalog.pipelines.defaults.dimensions
            if str(field).strip()
        ]
    )
    if explicit:
        return explicit
    return _shared_processor_dimensions(catalog)


def _shared_processor_dimensions(catalog: model.Catalog) -> list[str]:
    """Dimensions common to every processor group-by, in first-seen order.

    Processors without any group-by are skipped — they have nothing to share
    and would otherwise erase the common list.
    """

    shared: list[str] | None = None
    for processor in catalog.processors.processors:
        fields = dedupe([str(field).strip() for field in processor.group_by if str(field).strip()])
        if not fields:
            continue
        if shared is None:
            shared = fields
            continue
        keys = {field.casefold() for field in fields}
        shared = [field for field in shared if field.casefold() in keys]
        if not shared:
            return []
    return shared or []


def write_workspace_dimensions(
    workspace: str | Path,
    dimensions: list[str],
) -> None:
    """Update the workspace-level common dimensions in ``pipelines.yaml``.

    An empty selection removes the key so untouched workspaces keep their
    exact prior file contents.
    """
    path = _catalog_file(workspace, "pipelines.yaml")
    data = _read_yaml(path)
    defaults = data.setdefault("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("pipelines.yaml must contain a mapping at `defaults`")
    cleaned = dedupe([str(field).strip() for field in dimensions if str(field).strip()])
    if cleaned:
        defaults["dimensions"] = cleaned
    elif "dimensions" in defaults:
        del defaults["dimensions"]
    if not defaults:
        data.pop("defaults", None)
    _write_yaml(path, data)


def write_processors_definition(
    workspace: str | Path,
    processors_def: dict[str, Any],
) -> None:
    """Replace ``processors.yaml`` with one structurally validated full definition."""

    model.Processors.model_validate(processors_def)
    _write_yaml(_catalog_file(workspace, "processors.yaml"), processors_def)


def write_metrics_definition(
    workspace: str | Path,
    metrics_def: dict[str, Any],
) -> None:
    """Replace ``metrics.yaml`` with one structurally validated full definition."""

    model.Metrics.model_validate(metrics_def)
    _write_yaml(_catalog_file(workspace, "metrics.yaml"), metrics_def)


def write_tile_definition(
    workspace: str | Path,
    *,
    dashboard_id: str,
    dashboard_title: str,
    dashboard_layout: str | None = None,
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
    dashboard = _find_or_create_dashboard(
        dashboards,
        dashboard_id,
        dashboard_title,
        layout=dashboard_layout,
    )
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


def validate_report_candidate(
    catalog: model.Catalog,
    *,
    dashboard_id: str,
    dashboard_title: str,
    dashboard_layout: str,
    page_id: str,
    page_title: str,
    filters: list[dict[str, Any]],
    time_filter: dict[str, Any],
    tile: dict[str, Any],
    source_columns_by_id: Mapping[str, Iterable[str]] | None = None,
) -> tuple[bool, list[str]]:
    """Validate one proposed report edit against a complete in-memory catalog."""

    missing = [field for field in ("id", "title", "metric", "chart") if not tile.get(field)]
    if missing:
        return False, [
            "Tile YAML replaces the whole tile definition; required field(s) missing: "
            + ", ".join(f"`{name}`" for name in missing)
        ]
    try:
        payload = catalog.model_dump(mode="json", by_alias=True, exclude_none=True)
        dashboards_payload = cast(dict[str, Any], payload["dashboards"])
        dashboards = cast(list[dict[str, Any]], dashboards_payload["dashboards"])
        dashboard = _find_or_create_dashboard(
            dashboards,
            dashboard_id,
            dashboard_title,
            layout=dashboard_layout,
        )
        page = _find_or_create_page(dashboard, page_id, page_title)
        page["title"] = page_title
        page["filters"] = filters
        page["time_filter"] = time_filter
        tiles = page.setdefault("tiles", [])
        if not isinstance(tiles, list):
            raise ValueError("dashboard page must contain a list at `tiles`")
        _replace_or_append(tiles, tile)
        candidate = model.Catalog.model_validate(payload)
    except (KeyError, TypeError, ValueError) as exc:
        return False, [str(exc)]
    result = validate_catalog(candidate, source_columns_by_id=source_columns_by_id)
    errors = [issue for issue in result.issues if issue.severity == "error"]
    return not errors, [f"{issue.location}: {issue.message}" for issue in errors]


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


def validate_workspace(
    workspace: str | Path,
    *,
    source_columns_by_id: Mapping[str, Iterable[str]] | None = None,
) -> tuple[bool, list[str]]:
    """Load and validate a workspace after a builder change."""
    ensure_minimum_workspace(workspace)
    try:
        catalog = load(workspace)
    except CatalogLoadError as exc:
        return False, [str(exc)]
    result = validate_catalog(catalog, source_columns_by_id=source_columns_by_id)
    return result.ok, [f"{issue.location}: {issue.message}" for issue in result.issues]


def require_valid_workspace(
    workspace: str | Path,
    *,
    source_columns_by_id: Mapping[str, Iterable[str]] | None = None,
) -> None:
    """Raise with validation details so an enclosing authoring transaction rolls back."""

    ok, issues = validate_workspace(workspace, source_columns_by_id=source_columns_by_id)
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
    *,
    layout: str | None = None,
) -> dict[str, Any]:
    if layout is not None and layout not in {"tabs", "grid", "stacked"}:
        raise ValueError("dashboard layout must be `tabs`, `grid`, or `stacked`")
    for dashboard in dashboards:
        if dashboard.get("id") == dashboard_id:
            dashboard["title"] = dashboard_title
            if layout is not None:
                dashboard["layout"] = layout
            return dashboard
    dashboard = {
        "id": dashboard_id,
        "title": dashboard_title,
        "layout": layout or "tabs",
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
            page["title"] = page_title
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
    if not editor_row_enabled(row.get("Enabled")):
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
    if operator == "between":
        bounds = _split_values(raw_value)
        if len(bounds) != 2:
            raise ValueError(
                f"between on {field!r} needs exactly two comma-separated values: low, high"
            )
        return {"op": "between", "column": field, "low": bounds[0], "high": bounds[1]}
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
    if op == "between":
        return {
            "Field": expression.get("column", ""),
            "Operator": "between",
            "Value": f"{expression.get('low', '')}, {expression.get('high', '')}",
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
    "BUILDER_DRAFTS_KEY",
    "CALCULATION_MODES",
    "CHART_DISPLAY_LABELS",
    "CHART_DISPLAY_PURPOSES",
    "CHART_OPTIONAL_FIELDS",
    "CHART_REQUIRED_FIELDS",
    "CHART_SETTING_FIELDS",
    "FILTER_OPERATORS",
    "METRIC_KIND_HELP",
    "METRIC_KIND_LABELS",
    "SCALAR_STATE_TYPES",
    "STATE_TYPES",
    "VISUAL_CASE_MAX_BRANCHES",
    "BuilderApplyOutcome",
    "BuilderDraftStatus",
    "CalculatedExpressionValidation",
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
    "builder_apply_outcome",
    "builder_draft_status",
    "builder_requires_data_run",
    "builder_template_draft_status",
    "calculated_expression_example",
    "calculated_rows_for_editor",
    "calculated_rows_from_source",
    "calculation_mode_from_expression",
    "case_value_expression",
    "case_value_from_expression",
    "catalog_id_is_safe",
    "chart_choices_for_metric",
    "chart_field_controls",
    "chart_field_options",
    "chart_kind_label",
    "chart_kind_purpose",
    "chart_kind_selector_label",
    "chart_recipe_summary",
    "compile_case_expression",
    "compile_condition_formula",
    "compile_condition_rows",
    "compile_filter_rows",
    "condition_rows_from_expression",
    "condition_state_from_expression",
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
    "discard_builder_draft",
    "display_grain",
    "editor_frame",
    "editor_row_enabled",
    "ensure_minimum_workspace",
    "expression_yaml",
    "filter_rows_from_expression",
    "first_filter_expression",
    "float_in_range",
    "funnel_stage_names",
    "generated_catalog_id",
    "label_condition_rows",
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
    "registered_builder_draft",
    "restore_builder_draft",
    "scalar_state_columns",
    "score_properties_from_definition",
    "source_defaults",
    "source_to_dict",
    "stable_catalog_id",
    "stage_condition_rows",
    "stage_names_missing_when",
    "state_columns",
    "state_columns_by_type",
    "state_spec_definitions",
    "string_list",
    "tile_yaml",
    "title_from_identifier",
    "update_builder_draft_registry",
    "validate_calculated_expression",
    "validate_report_candidate",
    "validate_workspace",
    "visual_case_state_from_expression",
    "widget_key_fragment",
    "workspace_dimension_defaults",
    "write_dashboards_definition",
    "write_metric_definition",
    "write_metrics_definition",
    "write_page_settings",
    "write_pipelines_definition",
    "write_processor_definition",
    "write_processors_definition",
    "write_source_definition",
    "write_tile_definition",
    "write_workspace_dimensions",
    "write_workspace_settings",
]
