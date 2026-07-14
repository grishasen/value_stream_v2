"""LLM intent planning and governed chat tools for aggregate data."""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import polars as pl

from valuestream.ai.studio import AICallSettings, call_litellm
from valuestream.config import model
from valuestream.query import query_metric
from valuestream.ui.builder import chart_choices_for_metric
from valuestream.ui.freshness import freshness_label, metric_freshness
from valuestream.utils.logger import get_logger

ResponseKind = Literal["text", "table", "chart", "clarify", "sql"]
ChartKind = Literal[
    "line",
    "bar",
    "stacked_area",
    "scatter",
    "heatmap",
    "donut",
    "table",
    "kpi_card",
    "roc_curve",
    "precision_recall_curve",
    "calibration_curve",
]
# Chart kinds the chat intent shape (x / y / color / facet) can express and the
# shared dashboard chart factory can render.
CHAT_CHART_KINDS: tuple[str, ...] = (
    "line",
    "bar",
    "stacked_area",
    "scatter",
    "heatmap",
    "donut",
    "table",
    "kpi_card",
    "roc_curve",
    "precision_recall_curve",
    "calibration_curve",
)
# Always offered regardless of the processor's chart recipes.
_BASE_CHART_KINDS = ("line", "bar", "table", "kpi_card")
_CURVE_CHART_KINDS = frozenset({"roc_curve", "precision_recall_curve", "calibration_curve"})
_VALUE_FORMATS = ("percent", "integer", "number", "currency")
_COMPARE_ALIASES = {
    "prior_period": "prior_period",
    "previous_period": "prior_period",
    "prior": "prior_period",
    "period_over_period": "prior_period",
    "pop": "prior_period",
}
_COMPARE_SUFFIXES = ("_prev", "_delta", "_pct_change")
_NARRATIVE_ROW_CAP = 50

logger = get_logger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TIME_COLUMN_BY_GRAIN = {
    "daily": "Day",
    "monthly": "Month",
    "quarterly": "Quarter",
    "yearly": "Year",
}
_TIME_COLUMNS = {
    "Day",
    "day",
    "Week",
    "week",
    "Month",
    "month",
    "Quarter",
    "quarter",
    "Year",
    "year",
    "period",
}
_MAX_CHAT_ROWS = 500
_METRIC_KIND_EXPLANATIONS = {
    "formula": "Derived from aggregate state columns or dependency metrics using the expression AST.",
    "approx_distinct_count": (
        "Approximate distinct count from a CPC, HLL, or Theta sketch state."
    ),
    "topk_items": "Top-K items from a processor top-k state.",
    "tdigest_quantile": "Quantile or percentile derived from a t-digest state.",
    "variant_compare": "Compares test and control roles across a variant column.",
    "curve_from_digests": "Model-quality curve metric reconstructed from positive and negative score digests.",
    "calibration_from_digests": "Calibration bins reconstructed from positive and negative score digests.",
    "contingency_test": "Statistical test across variants, usually chi-square, G-test, or z-test.",
    "proportion_test": "Proportion comparison derived from positive and negative aggregate counts.",
    "lifecycle_summary": "Customer lifecycle summary metric from entity lifecycle aggregates.",
    "set_op": "Set algebra metric over sketch states.",
    "funnel_dropoff": "Funnel transition count or rate between configured stages.",
}
_PROCESSOR_KIND_EXPLANATIONS = {
    "binary_outcome": "Counts positive and negative outcomes for rates, lift, and tests.",
    "numeric_distribution": "Stores numeric distribution states such as count, mean, variance, and quantile sketches.",
    "score_distribution": "Stores score distributions split by outcome for ROC, precision/recall, and calibration.",
    "entity_lifecycle": "Aggregates lifecycle state per entity for CLV/RFM-style summaries.",
    "entity_set": "Stores entity sketches for distinct counts and set operations.",
    "funnel": "Counts configured funnel stages and transitions.",
    "snapshot": "Stores periodic or accumulating state snapshots.",
}


@dataclass(frozen=True)
class ChartIntent:
    """Validated chart request over aggregate query output."""

    kind: ChartKind
    x: str | None = None
    y: str | None = None
    color: str | None = None
    facet_col: str | None = None
    value_format: str | None = None


@dataclass(frozen=True)
class ChatIntent:
    """Validated natural-language query intent."""

    question: str
    metric: str
    response: ResponseKind
    group_by: list[str]
    filters: dict[str, Any]
    grain: str
    start: str | None = None
    end: str | None = None
    chart: ChartIntent | None = None
    limit: int = 100
    having: dict[str, Any] = field(default_factory=dict)
    order_by: list[str] = field(default_factory=list)
    top_n: int | None = None
    top_n_by: str | None = None
    compare: str | None = None
    quantiles: bool = False
    clarify: str | None = None
    sql: str | None = None


@dataclass(frozen=True)
class ChatQueryResult:
    """Executed chat intent with rows and provenance text."""

    intent: ChatIntent
    rows: pl.DataFrame
    query_summary: str
    freshness: str


def catalog_chat_manifest(
    catalog: model.Catalog,
    *,
    chat_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return compact catalog metadata for LLM prompts and MCP clients."""

    dataset_descriptions = _description_map(chat_config, "dataset_descriptions")
    metric_descriptions = _description_map(chat_config, "metric_descriptions")
    processors = {processor.id: processor for processor in catalog.processors.processors}
    sources = {source.id: source for source in catalog.pipelines.sources}
    metrics: list[dict[str, Any]] = []
    for metric_name, metric in sorted(catalog.metrics.metrics.items()):
        processor = processors.get(metric.source)
        source = sources.get(processor.source) if processor is not None else None
        metric_chat_description = _first_description(
            metric_descriptions,
            metric_name,
            metric.kind,
            metric.source,
            processor.kind if processor is not None else "",
            processor.source if processor is not None else "",
        )
        metrics.append(
            {
                "name": metric_name,
                "description": metric.description,
                "chat_description": metric_chat_description,
                "kind": metric.kind,
                "kind_explanation": _METRIC_KIND_EXPLANATIONS.get(metric.kind, ""),
                "processor": metric.source,
                "processor_description": processor.description if processor is not None else "",
                "processor_kind": processor.kind if processor is not None else "",
                "dataset": processor.source if processor is not None else "",
                "dataset_description": source.description if source is not None else "",
                "dataset_chat_description": _first_description(
                    dataset_descriptions,
                    processor.source if processor is not None else "",
                    source.id if source is not None else "",
                ),
                "dimensions": list(processor.group_by) if processor is not None else [],
                "time_axes": _query_time_axes(processor) if processor is not None else [],
                "outputs": metric_output_columns(metric_name, metric),
                "chart_kinds": allowed_chart_kinds(catalog, metric_name),
                "configuration": _metric_configuration(metric),
            }
        )
    return {
        "workspace": catalog.pipelines.workspace,
        "datasets": [
            _dataset_manifest(source, dataset_descriptions=dataset_descriptions)
            for source in catalog.pipelines.sources
        ],
        "processors": [
            _processor_manifest(
                processor,
                sources,
                metric_descriptions=metric_descriptions,
                dataset_descriptions=dataset_descriptions,
            )
            for processor in sorted(catalog.processors.processors, key=lambda item: item.id)
        ],
        "metrics": metrics,
        "supported_charts": list(CHAT_CHART_KINDS),
        "query_features": {
            "filter_operators": [
                "eq",
                "ne",
                "gt",
                "gte",
                "lt",
                "lte",
                "in",
                "not_in",
                "contains",
                "starts_with",
                "ends_with",
                "is_null",
                "not_null",
            ],
            "filter_spec": 'scalar, list, or {"op": ">=", "value": 3} / {"op": "in", "values": [..]}',
            "having": "same operator specs applied to metric output columns after aggregation",
            "order_by": 'column names, "-" prefix for descending',
            "top_n": "keep the N largest rows by top_n_by (a metric output column)",
            "compare": 'set "prior_period" to add *_prev, *_delta, *_pct_change columns over the time axis',
            "quantiles": "set true to include Median/p25/p75/p90/p95 columns for digest metrics",
        },
        "chart_contract": {
            "required_fields": ["kind", "x", "y"],
            "optional_fields": ["color", "facet_col", "value_format"],
            "time_axes": ["Day", "Month", "Quarter", "Year"],
            "value_formats": list(_VALUE_FORMATS),
            "axis_rules": [
                "Choose chart.kind only from the metric's own chart_kinds list.",
                "Metric output columns belong in chart.y only.",
                "chart.x may be a time column or business dimension present in the query result.",
                "For time-trend charts, use Day/Month/Quarter/Year as chart.x.",
                "For dimension comparisons, use a business dimension such as Issue or Channel as chart.x.",
                "Business dimensions may be used for group_by, chart.x, chart.color, and chart.facet_col.",
                "For heatmap use chart.x and chart.color as the two dimensions and chart.y as the metric value.",
                "For donut use chart.x as the category and chart.y as the metric value.",
                "roc_curve, precision_recall_curve, and calibration_curve need only the metric and an optional chart.color.",
                "Set chart.value_format to percent for rates, integer for counts, or currency for money.",
            ],
        },
    }


def _dataset_manifest(
    source: model.Source,
    *,
    dataset_descriptions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    schema = source.schema_
    return _compact_manifest(
        {
            "id": source.id,
            "description": source.description,
            "chat_description": _first_description(dataset_descriptions or {}, source.id),
            "reader": _reader_manifest(source.reader),
            "timestamp_column": schema.timestamp_column,
            "natural_key": schema.natural_key,
            "dropped_columns": schema.drop_columns,
            "defaults": source.defaults,
            "transforms": [_transform_manifest(transform) for transform in source.transforms],
            "materialize_transforms": source.materialize_transforms,
        }
    )


def _reader_manifest(reader: model.Reader) -> dict[str, Any]:
    payload = _jsonable(reader)
    # Do not expose permissive reader extras such as local filesystem roots in prompts.
    keys = ("kind", "file_pattern", "group_by_filename", "streaming", "delimiter", "sheet")
    return _compact_manifest({key: payload.get(key) for key in keys if isinstance(payload, dict)})


def _transform_manifest(transform: model.Transform) -> dict[str, Any]:
    payload = _jsonable(transform)
    if not isinstance(payload, dict):
        return {"kind": str(transform)}
    kind = str(payload.get("kind", ""))
    keep_by_kind = {
        "parse_datetime": ("kind", "columns", "format"),
        "derive_calendar": ("kind", "from", "outputs"),
        "derive_action_id": ("kind", "parts", "sep"),
        "derive_column": ("kind", "output", "expression"),
        "filter": ("kind", "expression"),
        "dedup": ("kind", "keys"),
        "defaults": ("kind", "values"),
        "cast": ("kind", "columns"),
        "drop_columns": ("kind", "columns"),
        "coalesce": ("kind", "output", "columns"),
    }
    keys = keep_by_kind.get(kind, ("kind",))
    return _compact_manifest({key: payload.get(key) for key in keys})


def _processor_manifest(
    processor: model.Processor,
    sources: Mapping[str, model.Source],
    *,
    metric_descriptions: Mapping[str, str] | None = None,
    dataset_descriptions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    source = sources.get(processor.source)
    states = model.effective_processor_states(processor)
    time_payload = (
        {
            "column": processor.time.column,
            "query_time_axes": _query_time_axes(processor),
        }
        if processor.time is not None
        else {"query_time_axes": _query_time_axes(processor)}
    )
    config = _processor_configuration(processor)
    time_columns = _query_time_axes(processor)
    return _compact_manifest(
        {
            "id": processor.id,
            "description": processor.description,
            "chat_description": _first_description(
                metric_descriptions or {},
                processor.id,
                processor.kind,
                processor.source,
            ),
            "kind": processor.kind,
            "kind_explanation": _PROCESSOR_KIND_EXPLANATIONS.get(processor.kind, ""),
            "dataset": processor.source,
            "dataset_description": source.description if source is not None else "",
            "dataset_chat_description": _first_description(
                dataset_descriptions or {},
                processor.source,
                source.id if source is not None else "",
            ),
            "dimensions": list(processor.group_by),
            "time": time_payload,
            "available_query_fields": {
                "dimensions": list(processor.group_by),
                "time_columns": time_columns,
                "state_columns": list(states),
            },
            "states": [_state_manifest(name, state) for name, state in sorted(states.items())],
            "configuration": config,
        }
    )


def _query_time_axes(processor: model.Processor) -> list[str]:
    """Return time bucket columns query clients may request for this processor."""

    axes_by_grain = {
        "daily": ["Day", "Month", "Quarter", "Year"],
        "weekly": ["Week", "Month", "Quarter", "Year"],
        "monthly": ["Month", "Quarter", "Year"],
        "quarterly": ["Quarter", "Year"],
        "yearly": ["Year"],
    }
    out: list[str] = []
    for grain in processor.grains:
        for axis in axes_by_grain.get(model.normalize_grain_name(grain), []):
            if axis not in out:
                out.append(axis)
    return out


def _state_manifest(name: str, state: model.StateSpec) -> dict[str, Any]:
    payload = _jsonable(state)
    if isinstance(payload, dict):
        return _compact_manifest({"name": name, **payload})
    return {"name": name}


def _metric_configuration(metric: model.Metric) -> dict[str, Any]:
    payload = _jsonable(metric)
    if not isinstance(payload, dict):
        return {}
    return _compact_manifest(
        {
            key: value
            for key, value in payload.items()
            if key not in {"source", "kind", "description"}
        }
    )


def _processor_configuration(processor: model.Processor) -> dict[str, Any]:
    payload = _jsonable(processor)
    if not isinstance(payload, dict):
        return {}
    return _compact_manifest(
        {
            key: value
            for key, value in payload.items()
            if key not in {"id", "source", "kind", "description", "group_by", "time", "states"}
        }
    )


def _description_map(config: Mapping[str, Any] | None, key: str) -> dict[str, str]:
    if config is None:
        return {}
    raw = config.get(key)
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, str] = {}
    for raw_key, raw_value in raw.items():
        item_key = str(raw_key or "").strip()
        item_value = str(raw_value or "").strip()
        if item_key and item_value:
            out[item_key.casefold()] = item_value
    return out


def _first_description(descriptions: Mapping[str, str], *keys: str) -> str:
    for key in keys:
        normalized = str(key or "").strip().casefold()
        if normalized and normalized in descriptions:
            return descriptions[normalized]
    return ""


def _agent_prompt(config: Mapping[str, Any] | None) -> str:
    if config is None:
        return "No additional workspace-specific chat guidance was configured."
    prompt = str(config.get("agent_prompt") or "").strip()
    return prompt or "No additional workspace-specific chat guidance was configured."


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _compact_manifest(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def metric_output_columns(metric_name: str, metric: model.Metric) -> list[str]:
    """Return user-facing output columns normally produced by a metric."""

    kind = metric.kind
    simple_metric_kinds = {
        "formula",
        "approx_distinct_count",
        "tdigest_quantile",
        "set_op",
        "funnel_dropoff",
    }
    configured_output_defaults = {
        "variant_compare": ["Count", "Positives", "Negatives", metric_name],
        "contingency_test": ["chi2_stat", "chi2_p_val", "z_score", "z_p_val"],
        "lifecycle_summary": ["customers_count", "lifetime_value"],
    }
    static_outputs = {
        "calibration_from_digests": ["bin", "predicted", "observed"],
        "topk_items": ["item", "count"],
    }
    if kind in simple_metric_kinds:
        return [metric_name]
    if kind in configured_output_defaults:
        outputs = list(getattr(metric, "outputs", []) or [])
        return outputs or configured_output_defaults[kind]
    if kind == "curve_from_digests":
        return [getattr(metric, "output", metric_name), "fpr", "tpr", "precision", "recall"]
    return static_outputs.get(kind, [metric_name])


def allowed_chart_kinds(catalog: model.Catalog, metric_name: str) -> list[str]:
    """Return chat-renderable chart kinds compatible with a metric's processor.

    Intersects the processor's recipe-compatible chart kinds with the subset
    the chat intent shape can express, and always keeps the base kinds so a
    plain trend/comparison/summary is available for every metric.
    """

    if metric_name not in catalog.metrics.metrics:
        return list(_BASE_CHART_KINDS)
    compatible = set(chart_choices_for_metric(catalog, metric_name))
    kinds = [kind for kind in CHAT_CHART_KINDS if kind in compatible or kind in _BASE_CHART_KINDS]
    return kinds or list(_BASE_CHART_KINDS)


def chart_tile_from_intent(intent: ChatIntent) -> dict[str, Any]:
    """Map a validated chart intent to a chart-factory tile spec.

    The chat intent carries generic ``x``/``y``/``color``/``facet_col`` fields;
    this translates them into the kind-specific keys that
    :func:`valuestream.charts.render_chart` expects.
    """

    chart = intent.chart
    kind = chart.kind if chart is not None else "bar"
    tile: dict[str, Any] = {
        "id": f"chat_{_slug(intent.metric) or 'result'}",
        "title": _chart_title(intent),
        "metric": intent.metric,
        "chart": kind,
    }
    if chart is None:
        return tile
    if chart.value_format:
        tile["value_format"] = chart.value_format
    if kind in _CURVE_CHART_KINDS:
        if chart.color:
            tile["color"] = chart.color
        return tile
    if kind == "kpi_card":
        tile["value"] = chart.y or intent.metric
        return tile
    if kind == "donut":
        tile["names"] = chart.x or chart.color
        tile["values"] = chart.y or intent.metric
        return tile
    if kind == "heatmap":
        # Two dimensions form the axes; the metric value is the color scale.
        tile["x"] = chart.x
        tile["y"] = chart.color or chart.facet_col
        tile["color"] = chart.y or intent.metric
        return tile
    for key, value in (
        ("x", chart.x),
        ("y", chart.y),
        ("color", chart.color),
        ("facet_col", chart.facet_col),
    ):
        if value:
            tile[key] = value
    return tile


def _chart_title(intent: ChatIntent) -> str:
    chart = intent.chart
    if chart is not None and chart.x:
        return f"{intent.metric} by {chart.x}"
    return intent.metric or "Result"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def chat_pin_tile(intent: ChatIntent, *, tile_id: str) -> dict[str, Any]:
    """Return a dashboard tile spec for pinning a chat answer.

    Chart answers reuse :func:`chart_tile_from_intent`; table and text answers
    pin as a table so the same governed query can be re-run from a dashboard.
    """

    if intent.chart is not None:
        tile = chart_tile_from_intent(intent)
    else:
        tile = {
            "id": tile_id,
            "title": intent.metric or "Result",
            "metric": intent.metric,
            "chart": "table",
        }
    tile["id"] = tile_id
    return tile


def chat_starter_questions(catalog: model.Catalog, *, limit: int = 4) -> list[str]:
    """Return a few example questions grounded in the catalog's metrics.

    Deterministic starter prompts help first-time users discover what the
    aggregate catalog can answer without having to guess metric names.
    """

    processors = {processor.id: processor for processor in catalog.processors.processors}
    questions: list[str] = []
    for metric_name, metric in sorted(catalog.metrics.metrics.items()):
        label = _title_from_identifier(metric_name)
        processor = processors.get(metric.source)
        dimensions = list(processor.group_by) if processor is not None else []
        time_axes = _query_time_axes(processor) if processor is not None else []
        if dimensions:
            questions.append(f"Compare {label} by {dimensions[0]}")
        elif time_axes:
            questions.append(f"Show {label} by {time_axes[0].lower()}")
        else:
            questions.append(f"What is the overall {label}?")
        if len(questions) >= limit:
            break
    return questions


def _title_from_identifier(value: str) -> str:
    words = value.replace("_", " ").replace("-", " ").split()
    cleaned = [word for word in words if word.upper() != "VS"]
    return " ".join(cleaned) or value


def prompt_for_chat_intent(
    catalog: model.Catalog,
    question: str,
    *,
    history: list[Mapping[str, Any]] | None = None,
    chat_config: Mapping[str, Any] | None = None,
    sql_schema: str | None = None,
) -> str:
    """Build a JSON-only prompt that maps a user question to a query intent."""

    manifest = catalog_chat_manifest(catalog, chat_config=chat_config)
    agent_prompt = _agent_prompt(chat_config)
    recent = _compact_history(history or [])
    today = dt.date.today().isoformat()
    response_kinds = "text|table|chart|clarify" + ("|sql" if sql_schema else "")
    sql_rules = (
        """
- response "sql" is a last resort for questions the intent shape cannot express
  (joins across metrics, window functions, arbitrary SQL aggregation). Set "sql"
  to one read-only SELECT statement over the governed tables listed under
  "Governed SQL tables". Never write DDL/DML, never reference files, and prefer
  a metric intent whenever one can answer the question."""
        if sql_schema
        else ""
    )
    sql_section = f"\nGoverned SQL tables:\n{sql_schema}\n" if sql_schema else ""
    sql_json_line = '\n  "sql": null,' if sql_schema else ""
    return f"""
Today is {today}.

You translate analyst questions into a Value Stream aggregate query intent.
Return only one JSON object. Do not include markdown.

Workspace chat guidance:
{agent_prompt}

Rules:
- The workspace chat guidance and chat descriptions help choose metrics and context. The rules below override them when there is any conflict.
- Use only metric names, dimensions, time axes, filters, and output columns from the catalog.
- Use the chat descriptions, metric descriptions, kind explanations, and configuration to choose the right metric.
- Use datasets and processors only to understand what each metric measures; do not output dataset or processor ids.
- Datasets/sources are not queryable directly. The output metric must always be a metric from the catalog.
- group_by must contain only business dimensions, not time columns.
- Do not choose or return an aggregate grain. Value Stream chooses the physical aggregate automatically.
- Use time_axis Day, Month, Quarter, or Year only when the question asks for a time bucket or trend.
- Use response "chart" when the user asks to plot, chart, visualize, trend, or compare visually.
- Use response "clarify" with a short "clarify" question when the request is ambiguous, spans data
  the catalog does not have, or could map to several very different metrics. Do not guess in
  those cases. For minor field-name mismatches, still pick the closest catalog field.{sql_rules}
- For chart requests, choose chart.kind ONLY from that metric's own "chart_kinds" list in the catalog.
- Prefer line for time trends, bar for category comparisons, kpi_card for a single summary number,
  donut for share-of-total, heatmap for two-dimension breakdowns, and roc_curve/precision_recall_curve/
  calibration_curve only for model-quality metrics that list them.
- Set chart.value_format to percent for rates, integer for counts, currency for money, else null.
- Set chart.x to the explicit requested axis when available. It may be a time column or business dimension.
- For time-trend requests, set time_axis and chart.x to Day/Month/Quarter/Year.
- For dimension-comparison requests, set chart.x to a business dimension such as Issue, Channel, or CustomerSegment.
- Put metric output columns in chart.y only. Never use a metric output column as chart.x.
- Put business dimensions in group_by, chart.color, or chart.facet_col, not in chart.x for time trends.
- When a time trend is requested "by" a business dimension, include that dimension in group_by and chart.color.
- For comparison charts with multiple business dimensions, use one dimension as chart.x and the others as chart.color or chart.facet_col.
- Always fill every chart field explicitly. Use null only when color or facet_col is not needed.
- Use ISO dates for start and end when the question asks for a time range; otherwise use null.
- filters accept scalars, lists, or operator objects such as {{"op": ">=", "value": 3}},
  {{"op": "not_in", "values": ["X"]}}, or {{"op": "contains", "value": "text"}}.
- Use "having" with the same operator objects to filter on metric output values after
  aggregation, for example {{"CTR": {{"op": ">", "value": 0.05}}}}.
- Use "order_by" for requested sort orders; prefix a column with "-" for descending.
- Use "top_n" plus "top_n_by" for top-N/bottom-N questions; "top_n" keeps the largest values.
- Use "compare": "prior_period" for change/growth/period-over-period questions. It requires a
  time_axis and adds _prev, _delta, and _pct_change columns to the metric outputs.
- Set "quantiles": true when the user asks about percentiles or the distribution of a
  digest-backed metric.
- If a requested field is not available, choose the closest available catalog field and keep the request valid.

JSON shape:
{{
  "metric": "metric name",
  "response": "{response_kinds}",
  "group_by": ["dimension"],
  "filters": {{"Dimension": "value, list, or operator object"}},
  "having": {{}},
  "order_by": [],
  "top_n": null,
  "top_n_by": null,
  "compare": null,
  "quantiles": false,
  "time_axis": "Day|Month|Quarter|Year|null",
  "start": null,
  "end": null,
  "chart": {{
    "kind": "one of the metric's chart_kinds",
    "x": "column or null",
    "y": "metric output column or null",
    "color": "dimension or null",
    "facet_col": "dimension or null",
    "value_format": "percent|integer|number|currency|null"
  }},
  "clarify": null,{sql_json_line}
  "limit": 100
}}

Catalog:
{json.dumps(manifest, indent=2, sort_keys=True)}
{sql_section}
Recent conversation:
{recent}

User question:
{question}
"""


def plan_chat_intent(
    settings: AICallSettings,
    catalog: model.Catalog,
    question: str,
    *,
    history: list[Mapping[str, Any]] | None = None,
    chat_config: Mapping[str, Any] | None = None,
    sql_schema: str | None = None,
) -> tuple[ChatIntent, str]:
    """Ask the configured model for a structured intent and validate it."""

    logger.info(
        "Chat intent planning started: model=%s provider=%s api_base=%s question=%r sql_enabled=%s",
        settings.model,
        settings.custom_llm_provider or "",
        settings.api_base or "",
        _preview(question),
        bool(sql_schema),
    )
    prompt = prompt_for_chat_intent(
        catalog,
        question,
        history=history,
        chat_config=chat_config,
        sql_schema=sql_schema,
    )
    sql_clause = (
        "Governed read-only SQL over the listed tables is allowed as a last resort. "
        if sql_schema
        else "Never invent SQL. "
    )
    raw = call_litellm(
        settings,
        prompt,
        system_prompt=(
            "You are a Value Stream chat planner. Return valid compact JSON only. "
            "Never choose aggregate grains or invent raw data, Python, or filesystem access. "
            + sql_clause
        ),
    )
    logger.debug("Chat intent raw model response: response=%r", _preview(raw, limit=1000))
    intent = parse_chat_intent(raw, catalog, question=question, allow_sql=bool(sql_schema))
    logger.info(
        "Chat intent planning completed: metric=%s response=%s grain=%s group_by=%s chart=%s",
        intent.metric,
        intent.response,
        intent.grain,
        intent.group_by,
        _chart_log_payload(intent.chart),
    )
    return intent, raw


def parse_chat_intent(
    text: str,
    catalog: model.Catalog,
    *,
    question: str = "",
    allow_payload_grain: bool = False,
    allow_sql: bool = False,
) -> ChatIntent:
    """Parse and validate model-produced chat intent JSON."""

    payload = _extract_json_payload(text)
    logger.debug("Parsing chat intent payload: payload=%s", _json_log_payload(payload))
    if "intent" in payload and isinstance(payload["intent"], dict):
        payload = payload["intent"]
    response_raw = str(payload.get("response") or "").strip().lower()
    clarify_text = _optional_text(payload.get("clarify") or payload.get("clarification"))
    if response_raw == "clarify" or (clarify_text and not _optional_text(payload.get("metric"))):
        if not clarify_text:
            raise ValueError("clarify response requires a `clarify` question for the user")
        logger.info("Chat intent asks for clarification: question=%r", _preview(clarify_text))
        return _non_query_intent(question, "clarify", clarify=clarify_text)
    sql_text = _optional_text(payload.get("sql"))
    if response_raw == "sql" or (sql_text and not _optional_text(payload.get("metric"))):
        if not allow_sql:
            raise ValueError(
                "SQL answers are not enabled; enable governed SQL or ask a metric question"
            )
        if not sql_text:
            raise ValueError("sql response requires a `sql` SELECT statement")
        logger.info("Chat intent escalated to governed SQL: sql=%r", _preview(sql_text))
        return _non_query_intent(question, "sql", sql=sql_text)
    metric_name = _resolve_metric_name(payload.get("metric"), catalog)
    metric = catalog.metrics.metrics[metric_name]
    processor = _processor_for_metric(catalog, metric)
    response = _normalize_response(payload.get("response"), payload.get("chart"))
    raw_chart = payload.get("chart")
    chart_payload: dict[str, Any] = raw_chart if isinstance(raw_chart, dict) else {}
    outputs = metric_output_columns(metric_name, metric)
    group_by = _normalize_group_by(payload.get("group_by"), processor)
    time_axis = _infer_time_axis(payload, chart_payload, question)
    grain = (
        _normalize_explicit_grain(payload.get("grain")) if allow_payload_grain else None
    ) or _grain_from_time_axis(time_axis)
    chart = _normalize_chart(
        chart_payload,
        response,
        grain,
        time_axis,
        group_by,
        processor,
        outputs,
        allowed_chart_kinds(catalog, metric_name),
    )
    group_by = _with_chart_group_by(group_by, chart, processor)
    filters = _normalize_filters(payload.get("filters"), processor)
    limit = _normalize_limit(payload.get("limit"))
    compare = _normalize_compare(payload.get("compare"), grain)
    having = _normalize_having(payload.get("having"), outputs, compare)
    order_by = _normalize_order_by(
        payload.get("order_by"),
        outputs=outputs,
        group_by=group_by,
        time_axis=time_axis,
        compare=compare,
    )
    top_n = _normalize_top_n(payload.get("top_n"))
    top_n_by = (
        _normalize_output_column(payload.get("top_n_by"), outputs) if top_n is not None else None
    )
    intent = ChatIntent(
        question=question,
        metric=metric_name,
        response=response,
        group_by=group_by,
        filters=filters,
        grain=grain,
        start=_optional_text(payload.get("start")),
        end=_optional_text(payload.get("end")),
        chart=chart,
        limit=limit,
        having=having,
        order_by=order_by,
        top_n=top_n,
        top_n_by=top_n_by,
        compare=compare,
        quantiles=bool(payload.get("quantiles")),
    )
    logger.debug(
        "Validated chat intent: metric=%s response=%s grain=%s group_by=%s filters=%s "
        "having=%s order_by=%s top_n=%s compare=%s chart=%s limit=%s",
        intent.metric,
        intent.response,
        intent.grain,
        intent.group_by,
        list(intent.filters),
        list(intent.having),
        intent.order_by,
        intent.top_n,
        intent.compare,
        _chart_log_payload(intent.chart),
        intent.limit,
    )
    return intent


def _non_query_intent(
    question: str,
    response: ResponseKind,
    *,
    clarify: str | None = None,
    sql: str | None = None,
) -> ChatIntent:
    return ChatIntent(
        question=question,
        metric="",
        response=response,
        group_by=[],
        filters={},
        grain="summary",
        clarify=clarify,
        sql=sql,
    )


def chart_intent_from_parameters(
    catalog: model.Catalog,
    *,
    metric: str,
    chart_kind: str,
    x: str,
    y: str,
    group_by: list[str],
    filters: dict[str, Any] | None = None,
    grain: str = "summary",
    start: str | None = None,
    end: str | None = None,
    color: str | None = None,
    facet_col: str | None = None,
    having: dict[str, Any] | None = None,
    order_by: list[str] | None = None,
    top_n: int | None = None,
    top_n_by: str | None = None,
    compare: str | None = None,
    quantiles: bool = False,
    value_format: str | None = None,
    limit: int = 100,
    question: str = "MCP chart request",
) -> ChatIntent:
    """Build a validated chart intent from explicit tool parameters."""

    payload = {
        "metric": metric,
        "response": "chart",
        "group_by": group_by,
        "filters": filters or {},
        "having": having or {},
        "order_by": order_by or [],
        "top_n": top_n,
        "top_n_by": top_n_by,
        "compare": compare,
        "quantiles": quantiles,
        "grain": grain,
        "start": start,
        "end": end,
        "chart": {
            "kind": chart_kind,
            "x": x,
            "y": y,
            "color": color,
            "facet_col": facet_col,
            "value_format": value_format,
        },
        "limit": limit,
    }
    logger.debug(
        "Building chart intent from explicit parameters: payload=%s", _json_log_payload(payload)
    )
    return parse_chat_intent(
        json.dumps(payload),
        catalog,
        question=question,
        allow_payload_grain=True,
    )


def execute_chat_intent(
    workspace_path: str | Path,
    catalog: model.Catalog,
    intent: ChatIntent,
) -> ChatQueryResult:
    """Execute a validated chat intent through the aggregate query layer."""

    if intent.response in {"clarify", "sql"}:
        raise ValueError(
            f"intent with response {intent.response!r} cannot be executed as a metric query"
        )
    logger.info(
        "Executing chat aggregate query: metric=%s grain=%s group_by=%s filters=%s start=%s "
        "end=%s having=%s order_by=%s top_n=%s compare=%s limit=%s",
        intent.metric,
        intent.grain,
        intent.group_by,
        list(intent.filters),
        intent.start,
        intent.end,
        list(intent.having),
        intent.order_by,
        intent.top_n,
        intent.compare,
        intent.limit,
    )
    rows = query_metric(
        workspace_path,
        intent.metric,
        group_by=intent.group_by,
        filters=intent.filters,
        grain=intent.grain,
        start=intent.start,
        end=intent.end,
        having=intent.having,
        order_by=intent.order_by,
        top_n=intent.top_n,
        top_n_by=intent.top_n_by,
        compare=intent.compare,
        include_quantile_suite=intent.quantiles,
        include_curve_columns=True,
    )
    if rows.height > intent.limit:
        rows = rows.head(intent.limit)
    fresh = metric_freshness(workspace_path, catalog, intent.metric, grain=intent.grain)
    logger.info(
        "Executed chat aggregate query: metric=%s grain=%s rows=%s columns=%s freshness=%s",
        intent.metric,
        intent.grain,
        rows.height,
        rows.columns,
        freshness_label(fresh),
    )
    return ChatQueryResult(
        intent=intent,
        rows=rows,
        query_summary=_query_summary(intent),
        freshness=freshness_label(fresh),
    )


def dimension_values(
    workspace_path: str | Path,
    catalog: model.Catalog,
    metric_name: str,
    column: str,
    *,
    grain: str = "summary",
    limit: int = 50,
) -> list[Any]:
    """Return aggregate-backed distinct values for one metric dimension."""

    metric_name = _resolve_metric_name(metric_name, catalog)
    metric = catalog.metrics.metrics[metric_name]
    processor = _processor_for_metric(catalog, metric)
    column = _resolve_column_name(column, processor.group_by)
    rows = query_metric(workspace_path, metric_name, group_by=[column], grain=grain)
    if column not in rows.columns:
        return []
    values = rows.get_column(column).drop_nulls().unique(maintain_order=True).head(limit).to_list()
    logger.debug(
        "Loaded chat dimension values: metric=%s column=%s grain=%s count=%s",
        metric_name,
        column,
        grain,
        len(values),
    )
    return values


def _extract_json_payload(text: str) -> dict[str, Any]:
    match = _JSON_BLOCK_RE.search(text)
    payload = match.group(1) if match else text
    payload = payload.strip()
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response did not contain a JSON object") from None
        loaded = json.loads(payload[start : end + 1])
    if not isinstance(loaded, dict):
        raise ValueError("LLM response JSON must be an object")
    return loaded


def _resolve_metric_name(value: object, catalog: model.Catalog) -> str:
    raw = str(value or "").strip()
    if raw in catalog.metrics.metrics:
        return raw
    normalized = _key(raw)
    for metric_name in catalog.metrics.metrics:
        if _key(metric_name) == normalized:
            return metric_name
    options = ", ".join(sorted(catalog.metrics.metrics)[:12])
    raise ValueError(f"unknown metric {raw!r}; available metrics include: {options}")


def _processor_for_metric(catalog: model.Catalog, metric: model.Metric) -> model.Processor:
    processor = next(
        (candidate for candidate in catalog.processors.processors if candidate.id == metric.source),
        None,
    )
    if processor is None:
        raise ValueError(f"metric references unknown processor {metric.source!r}")
    return processor


def _infer_time_axis(
    payload: Mapping[str, Any],
    chart_payload: Mapping[str, Any],
    question: str,
) -> str | None:
    for value in (
        payload.get("time_axis"),
        payload.get("time_bucket"),
        payload.get("time_column"),
        chart_payload.get("x"),
    ):
        axis = _normalize_time_axis(value)
        if axis is not None:
            return axis
    raw_group_by = payload.get("group_by")
    if isinstance(raw_group_by, list):
        for value in raw_group_by:
            axis = _normalize_time_axis(value)
            if axis is not None:
                return axis
    return _time_axis_from_question(question)


def _normalize_time_axis(value: object) -> str | None:
    raw = _optional_text(value)
    if raw is None or not _is_time_column(raw):
        return None
    axis = _canonical_time_column(raw)
    return axis if axis in {"Day", "Week", "Month", "Quarter", "Year"} else None


def _time_axis_from_question(question: str) -> str | None:
    normalized = f" {_key_with_spaces(question)} "
    patterns = (
        ("Day", (" day ", " daily ", " by day ", " per day ")),
        ("Week", (" week ", " weekly ", " by week ", " per week ")),
        ("Month", (" month ", " monthly ", " by month ", " per month ")),
        ("Quarter", (" quarter ", " quarterly ", " by quarter ", " per quarter ")),
        ("Year", (" year ", " yearly ", " annual ", " annually ", " by year ", " per year ")),
    )
    for axis, needles in patterns:
        if any(needle in normalized for needle in needles):
            return axis
    return None


def _grain_from_time_axis(axis: str | None) -> str:
    if axis is None:
        return "summary"
    for grain, column in _TIME_COLUMN_BY_GRAIN.items():
        if column == axis:
            return grain
    if axis == "Week":
        return "weekly"
    return "summary"


def _normalize_explicit_grain(value: object) -> str | None:
    raw = _optional_text(value)
    if raw is None:
        return None
    grain = model.normalize_grain_name(raw)
    if grain in {"daily", "weekly", "monthly", "quarterly", "yearly", "summary"}:
        return grain
    raise ValueError(f"unsupported query grain {raw!r}")


def _normalize_response(value: object, chart_payload: object) -> ResponseKind:
    response = str(value or "").strip().lower()
    if response in {"chart", "plot", "visual", "visualization"} or chart_payload:
        return "chart"
    if response in {"table", "data", "rows"}:
        return "table"
    return "text"


def _normalize_group_by(value: object, processor: model.Processor) -> list[str]:
    values = value if isinstance(value, list) else []
    out: list[str] = []
    for item in values:
        raw = str(item or "").strip()
        if not raw or _is_time_column(raw):
            continue
        column = _resolve_column_name(raw, processor.group_by)
        if column not in out:
            out.append(column)
    return out


def _normalize_filters(value: object, processor: model.Processor) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = _resolve_column_name(str(raw_key), processor.group_by)
        if raw_value in (None, ""):
            continue
        if isinstance(raw_value, list):
            out[key] = [item for item in raw_value if item not in (None, "")]
        else:
            out[key] = raw_value
    return out


def _normalize_chart(
    value: Mapping[str, Any],
    response: ResponseKind,
    grain: str,
    time_axis: str | None,
    group_by: list[str],
    processor: model.Processor,
    outputs: list[str],
    allowed_kinds: list[str],
) -> ChartIntent | None:
    if response != "chart":
        return None
    kind = _normalize_chart_kind(value.get("kind"), allowed_kinds, time_axis, grain)
    value_format = _normalize_value_format(value.get("value_format"))
    if kind in _CURVE_CHART_KINDS:
        # Curves read fixed columns (fpr/tpr, predicted/observed); only an
        # optional color dimension is meaningful.
        color = _normalize_dimension_chart_column(value.get("color"), processor)
        return ChartIntent(kind=kind, color=color, value_format=value_format)  # type: ignore[arg-type]
    requested_x = _normalize_chart_x_value(value.get("x"), processor)
    x = _normalize_chart_x(
        requested_x,
        kind=kind,
        grain=grain,
        time_axis=time_axis,
        group_by=group_by,
    )
    y = _normalize_output_column(value.get("y"), outputs)
    color = _normalize_dimension_chart_column(value.get("color"), processor)
    facet_col = _normalize_dimension_chart_column(value.get("facet_col"), processor)
    color = _default_chart_color(
        color,
        x=x,
        group_by=group_by,
        facet_col=facet_col,
    )
    return ChartIntent(
        kind=kind,  # type: ignore[arg-type]
        x=x,
        y=y,
        color=color,
        facet_col=facet_col,
        value_format=value_format,
    )


def _normalize_chart_kind(
    value: object,
    allowed_kinds: list[str],
    time_axis: str | None,
    grain: str,
) -> str:
    requested = str(value or "").strip().lower()
    if requested in allowed_kinds:
        return requested
    prefer_line = time_axis is not None or grain in _TIME_COLUMN_BY_GRAIN
    for candidate in (
        "line" if prefer_line else "bar",
        "bar" if prefer_line else "line",
        "table",
        "kpi_card",
    ):
        if candidate in allowed_kinds:
            if requested:
                logger.info(
                    "Chat chart kind %r not allowed for metric; defaulted to %s (allowed=%s)",
                    requested,
                    candidate,
                    allowed_kinds,
                )
            return candidate
    return allowed_kinds[0] if allowed_kinds else "bar"


def _normalize_value_format(value: object) -> str | None:
    raw = _optional_text(value)
    if raw is None:
        return None
    normalized = raw.lower()
    return normalized if normalized in _VALUE_FORMATS else None


def _with_chart_group_by(
    group_by: list[str],
    chart: ChartIntent | None,
    processor: model.Processor,
) -> list[str]:
    if chart is None:
        return group_by
    out = list(group_by)
    for candidate in (chart.x, chart.color, chart.facet_col):
        if candidate is None or _is_time_column(candidate):
            continue
        try:
            column = _resolve_column_name(candidate, processor.group_by)
        except ValueError:
            continue
        if column not in out:
            out.append(column)
    return out


def _normalize_chart_x_value(value: object, processor: model.Processor) -> str | None:
    raw = _optional_text(value)
    if raw is None:
        return None
    if _is_time_column(raw):
        return _canonical_time_column(raw)
    try:
        return _resolve_column_name(raw, processor.group_by)
    except ValueError:
        logger.warning(
            "Ignoring invalid chart x-axis: raw=%r processor=%s allowed_dimensions=%s",
            raw,
            processor.id,
            processor.group_by,
        )
        return None


def _normalize_dimension_chart_column(value: object, processor: model.Processor) -> str | None:
    raw = _optional_text(value)
    if raw is None:
        return None
    if _is_time_column(raw):
        return None
    try:
        return _resolve_column_name(raw, processor.group_by)
    except ValueError:
        logger.warning(
            "Ignoring non-dimension chart field: raw=%r processor=%s",
            raw,
            processor.id,
        )
        return None


def _normalize_chart_x(
    requested_x: str | None,
    *,
    kind: str,
    grain: str,
    time_axis: str | None,
    group_by: list[str],
) -> str | None:
    time_x = time_axis or _TIME_COLUMN_BY_GRAIN.get(grain)
    if requested_x is not None:
        return requested_x
    if kind == "line" and time_x is not None:
        logger.info(
            "Defaulted missing/invalid line chart x-axis to time column: normalized_x=%s grain=%s",
            time_x,
            grain,
        )
        return time_x
    fallback = group_by[0] if group_by else time_x
    if fallback is not None:
        logger.info(
            "Defaulted missing/invalid chart x-axis: normalized_x=%s grain=%s kind=%s group_by=%s",
            fallback,
            grain,
            kind,
            group_by,
        )
    return fallback


def _default_chart_color(
    color: str | None,
    *,
    x: str | None,
    group_by: list[str],
    facet_col: str | None,
) -> str | None:
    if color is not None:
        return color
    for candidate in group_by:
        if candidate in (x, facet_col):
            continue
        logger.info(
            "Defaulted missing chart color to grouped dimension: color=%s x=%s group_by=%s",
            candidate,
            x,
            group_by,
        )
        return candidate
    return None


def _normalize_output_column(value: object, outputs: list[str]) -> str | None:
    raw = _optional_text(value)
    if raw is None:
        return outputs[0] if outputs else None
    normalized = _key(raw)
    for output in outputs:
        if _key(output) == normalized:
            return output
    return outputs[0] if outputs else raw


def _normalize_limit(value: object) -> int:
    limit = _coerce_int(value)
    if limit is None:
        return 100
    return min(max(limit, 1), _MAX_CHAT_ROWS)


def _normalize_top_n(value: object) -> int | None:
    top_n = _coerce_int(value)
    if top_n is None:
        return None
    return min(max(top_n, 1), _MAX_CHAT_ROWS)


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _normalize_compare(value: object, grain: str) -> str | None:
    raw = _optional_text(value)
    if raw is None:
        return None
    compare = _COMPARE_ALIASES.get(_key_with_spaces(raw).replace(" ", "_"))
    if compare is None:
        raise ValueError(f"unsupported compare mode {raw!r}; use 'prior_period'")
    if grain == "summary":
        raise ValueError(
            "compare='prior_period' needs a time axis; ask for the metric by "
            "Day, Month, Quarter, or Year"
        )
    return compare


def _normalize_having(
    value: object,
    outputs: list[str],
    compare: str | None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = list(outputs)
    if compare:
        allowed.extend(f"{output}{suffix}" for output in outputs for suffix in _COMPARE_SUFFIXES)
    out: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        if raw_value in (None, "", {}):
            continue
        out[_resolve_column_name(str(raw_key), allowed)] = raw_value
    return out


def _normalize_order_by(
    value: object,
    *,
    outputs: list[str],
    group_by: list[str],
    time_axis: str | None,
    compare: str | None,
) -> list[str]:
    if not isinstance(value, list):
        return []
    allowed = [*outputs, *group_by]
    if time_axis:
        allowed.append(time_axis)
    if compare:
        allowed.extend(f"{output}{suffix}" for output in outputs for suffix in _COMPARE_SUFFIXES)
    out: list[str] = []
    for item in value:
        raw = str(item or "").strip()
        if not raw:
            continue
        descending = raw.startswith("-")
        name = _resolve_column_name(raw[1:].strip() if descending else raw, allowed)
        spec = f"-{name}" if descending else name
        if spec not in out:
            out.append(spec)
    return out


def _resolve_column_name(value: object, options: list[str]) -> str:
    raw = str(value or "").strip()
    if raw in options:
        return raw
    normalized = _key(raw)
    for option in options:
        if _key(option) == normalized:
            return option
    raise ValueError(f"column {raw!r} is not available; use one of: {', '.join(options)}")


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"null", "none"}:
        return None
    return raw


def _canonical_time_column(value: str) -> str:
    normalized = _key(value)
    mapping = {
        "day": "Day",
        "week": "Week",
        "month": "Month",
        "period": "Month",
        "quarter": "Quarter",
        "year": "Year",
    }
    return mapping.get(normalized, value)


def _is_time_column(value: str) -> bool:
    return value in _TIME_COLUMNS or _key(value) in {_key(column) for column in _TIME_COLUMNS}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _key_with_spaces(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _compact_history(history: list[Mapping[str, Any]]) -> str:
    if not history:
        return "No previous conversation."
    lines: list[str] = []
    for message in history[-6:]:
        role = str(message.get("role", "user"))
        content = str(message.get("content") or message.get("summary") or "")
        if content:
            lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines) if lines else "No previous conversation."


def _query_summary(intent: ChatIntent) -> str:
    parts = [
        f"query_metric(workspace, {intent.metric!r}",
        f"group_by={intent.group_by!r}",
        f"filters={intent.filters!r}",
        f"grain={intent.grain!r}",
    ]
    if intent.start:
        parts.append(f"start={intent.start!r}")
    if intent.end:
        parts.append(f"end={intent.end!r}")
    if intent.having:
        parts.append(f"having={intent.having!r}")
    if intent.order_by:
        parts.append(f"order_by={intent.order_by!r}")
    if intent.top_n is not None:
        parts.append(f"top_n={intent.top_n!r}")
        if intent.top_n_by:
            parts.append(f"top_n_by={intent.top_n_by!r}")
    if intent.compare:
        parts.append(f"compare={intent.compare!r}")
    if intent.quantiles:
        parts.append("include_quantile_suite=True")
    return ", ".join(parts) + ")"


def overall_metric_value(
    workspace_path: str | Path,
    intent: ChatIntent,
) -> tuple[str, Any] | None:
    """Return the governed overall (summary-grain) value for an executed intent.

    Used for verbal answers instead of averaging grouped rows, which would
    weight every group equally regardless of volume.
    """

    if intent.response in {"clarify", "sql"} or not intent.metric:
        return None
    try:
        rows = query_metric(
            workspace_path,
            intent.metric,
            group_by=[],
            filters=intent.filters,
            grain="summary",
            start=intent.start,
            end=intent.end,
        )
    except Exception:
        logger.exception("Failed to load overall metric value: metric=%s", intent.metric)
        return None
    if rows.height != 1:
        return None
    value_column = next(
        (column for column in rows.columns if rows.schema[column].is_numeric()),
        None,
    )
    if value_column is None:
        return None
    return value_column, rows.get_column(value_column).item()


def prompt_for_chat_narrative(result: ChatQueryResult) -> str:
    """Build a grounded prompt that narrates governed aggregate rows."""

    intent = result.intent
    rows = result.rows.head(_NARRATIVE_ROW_CAP)
    return f"""
You summarize governed aggregate query results for a business analyst.
Write 2-4 plain sentences. Use only the numbers in the rows below; never
invent values, totals, or trends that are not visible in the rows. Mention
the most relevant comparison or extreme if one stands out. Do not output
markdown tables or code.

User question:
{intent.question}

Executed governed query:
{result.query_summary}

Data freshness: {result.freshness}
Returned rows ({result.rows.height} total, first {rows.height} shown) as JSON:
{json.dumps(rows.to_dicts(), default=str)}
"""


def narrate_chat_result(
    settings: AICallSettings,
    result: ChatQueryResult,
) -> str:
    """Ask the configured model for a short grounded narrative of the result."""

    prompt = prompt_for_chat_narrative(result)
    raw = call_litellm(
        settings,
        prompt,
        system_prompt=(
            "You are a careful data analyst. Only reference numbers present in "
            "the provided rows. Answer in plain prose."
        ),
    )
    return " ".join(raw.split()).strip()


def _preview(value: str, *, limit: int = 240) -> str:
    single_line = " ".join(value.split())
    if len(single_line) <= limit:
        return single_line
    return f"{single_line[:limit]}..."


def _chart_log_payload(chart: ChartIntent | None) -> dict[str, Any] | None:
    if chart is None:
        return None
    return {
        "kind": chart.kind,
        "x": chart.x,
        "y": chart.y,
        "color": chart.color,
        "facet_col": chart.facet_col,
        "value_format": chart.value_format,
    }


def _json_log_payload(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(payload, sort_keys=True)[:2000]
    except TypeError:
        return repr(payload)[:2000]


__all__ = [
    "CHAT_CHART_KINDS",
    "ChartIntent",
    "ChatIntent",
    "ChatQueryResult",
    "allowed_chart_kinds",
    "catalog_chat_manifest",
    "chart_intent_from_parameters",
    "chart_tile_from_intent",
    "chat_pin_tile",
    "chat_starter_questions",
    "dimension_values",
    "execute_chat_intent",
    "metric_output_columns",
    "narrate_chat_result",
    "overall_metric_value",
    "parse_chat_intent",
    "plan_chat_intent",
    "prompt_for_chat_intent",
    "prompt_for_chat_narrative",
]
