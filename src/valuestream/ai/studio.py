"""AI support for the Streamlit configuration studio.

The AI boundary is deliberately YAML-shaped because Value Stream's source of
truth is the workspace catalog: ``pipelines.yaml``, ``processors.yaml``,
``metrics.yaml``, and ``dashboards.yaml``.
"""

from __future__ import annotations

import hashlib
import re
import time
import traceback
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, get_args

import polars as pl
import yaml
from litellm import completion as litellm_completion
from pydantic import ValidationError

from valuestream.charts.recipes import TILE_REQUIRED_ALTERNATIVES
from valuestream.config import model
from valuestream.config.validate import validate_catalog
from valuestream.utils.logger import get_logger

_CATALOG_SECTION_KEYS = ("pipelines", "processors", "metrics", "dashboards")
_SECTION_KEYS = ("processors", "metrics", "dashboards", "chat_with_data")
_FENCED_BLOCK_RE = re.compile(r"```(?:yaml|yml)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_REPAIRABLE_MISSING_METRIC_FIELDS = {
    "approx_distinct_count": {"state"},
    "tdigest_quantile": {"state", "quantile"},
    "variant_compare": {"variant_column", "test_role", "control_role"},
    "curve_from_digests": {"positive_state", "negative_state"},
    "calibration_from_digests": {"positive_state", "negative_state"},
    "contingency_test": {"variant_column", "tests"},
    "set_op": {"op"},
    "funnel_dropoff": {"from_stage", "to_stage"},
}
_REPAIRABLE_PROCESSOR_STATES_RE = re.compile(
    r"^processors\.processors\.\d+\.[^.]+\.states: Input should be a valid dictionary$"
)
_REPAIRABLE_CONTINGENCY_TEST_RE = re.compile(
    r"^metrics\.metrics\.[^.]+\.contingency_test\.tests\.\d+: "
    r"Input should be 'chi2', 'g' or 'z'$"
)
_LOW_CARDINALITY_LIMIT = 50
_DRAFT_DIAGNOSTIC_MAX_ISSUES = 8
_DRAFT_DIAGNOSTIC_MAX_ISSUE_CHARS = 512
_DRAFT_DIAGNOSTIC_MAX_POSITION = 1_000_000
_DRAFT_DIAGNOSTIC_ISSUE_AREAS = (
    "source",
    "processor",
    "metric",
    "report",
    "chat",
    "field_contract",
    "other",
)
logger = get_logger(__name__)

_APPLICATION_STRUCTURE_DICTIONARY: dict[str, Any] = {
    "data_flow": [
        "pipelines.yaml defines source readers, source schemas, defaults, and transforms.",
        "processors.yaml binds processors to pipeline source ids and stores mergeable state columns.",
        "metrics.yaml derives query-time metrics from one processor's state columns.",
        "dashboards.yaml reports metrics through dashboards, pages, and tiles.",
    ],
    "file_ownership": {
        "pipelines.yaml": "Current source definitions are authoritative; do not replace them in AI drafts.",
        "processors.yaml": "AI draft may replace complete processors section when requested.",
        "metrics.yaml": "AI draft may replace complete metrics section when requested.",
        "dashboards.yaml": "AI draft may replace complete dashboards section when requested.",
    },
    "reference_graph": [
        "processor.source -> pipelines.sources[].id",
        "metric.source -> processors.processors[].id",
        "tile.metric -> metrics.metrics keys",
        "metric formula expression columns -> effective state columns on metric.source plus depends_on metrics",
    ],
}

_CATALOG_SCHEMA_DICTIONARY: dict[str, Any] = {
    "processors.yaml": {
        "shape": {"processors": ["Processor"]},
        "state_type_enum": [
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
        ],
        "processor_fields": {
            "id": "stable snake_case id, unique within processors.yaml",
            "source": "existing pipelines source id",
            "kind": "one processor kind from processor_kind_dictionary",
            "description": "optional short business description",
            "dimensions": "list of approved low-cardinality source fields; alias for group_by",
            "time": {
                "column": "approved timestamp field when time-based",
                "grains": ["Day", "Month", "Quarter", "Year", "Summary"],
                "aggregation_levels": "optional logical grain -> physical grain map",
            },
            "states": (
                "optional mapping of state_name -> {type, source_column?, ...}; not a list; "
                "every type must be one of state_type_enum (value_sum not sum, "
                "pooled_mean not mean, pooled_variance not var)"
            ),
            "filter": "optional expression AST over approved source fields",
        },
    },
    "metrics.yaml": {
        "shape": {"metrics": {"metric_id": "Metric"}},
        "metric_fields": {
            "source": "existing processor id",
            "kind": "one metric kind from metric_kind_dictionary",
            "description": "optional short business description",
            "depends_on": "optional list of existing metric ids used by formula expressions",
            "display": {
                "label": "optional friendly business label",
                "unit": "optional unit such as orders, EUR, or seconds",
                "value_format": "percent, integer, number, or currency",
                "direction": "higher_is_better, lower_is_better, or neutral",
            },
        },
    },
    "dashboards.yaml": {
        "shape": {
            "theme": "optional mapping",
            "dashboards": [
                {
                    "id": "stable snake_case id",
                    "title": "display title",
                    "layout": "tabs, grid, or stacked",
                    "pages": [
                        {
                            "id": "stable snake_case id",
                            "title": "display title",
                            "filters": "optional list of {field, label, display, scope, control}",
                            "time_filter": "optional {default, presets} using supported presets",
                            "tiles": ["Tile"],
                        }
                    ],
                }
            ],
        },
        "tile_fields": {
            "id": "stable snake_case id, unique on page",
            "title": "display title",
            "metric": "existing metric id",
            "chart": "one chart kind from chart_required_field_dictionary",
            "description": "optional plain-language report context",
            "placement": "content or kpi_strip; kpi_strip requires chart=kpi_card",
            "kpi": "optional {comparison, comparison_period, sparkline_grain, sparkline_points, target}",
            "scale_mode": "absolute, index_100, or percent_change for line/stacked_area",
            "value_format": "optional tile override for metric display.value_format",
            "filters": "optional mapping of approved low-cardinality field -> allowed values",
            "chart_data_fields": (
                "x, y, value, color, and the other keys listed in "
                "chart_required_field_dictionary are top-level tile keys beside "
                "id/title/metric/chart; never nest them under chart_specific_fields, "
                "options, or any other mapping"
            ),
        },
    },
}

_PROCESSOR_KIND_DICTIONARY: dict[str, Any] = {
    "binary_outcome": {
        "purpose": "engagement, conversion, experiment, or response rates from positive/negative outcomes",
        "key_fields": [
            "outcome.column",
            "outcome.positive_values",
            "outcome.negative_values",
            "variant_column when creating experiment metrics; never repeat it in "
            "dimensions/group_by because the processor persists it automatically",
            "value_aggs for revenue or monetary sums",
            "touchpoint for conversion touchpoint counts",
        ],
        "default_states": ["Count", "Positives", "Negatives"],
        "compatible_metrics": [
            "formula",
            "variant_compare",
            "contingency_test",
            "approx_distinct_count",
        ],
    },
    "numeric_distribution": {
        "purpose": "descriptive distributions and percentiles for approved numeric properties",
        "key_fields": ["properties", "quantile_engine", "sketch_build_mode"],
        "defaults": {"sketch_build_mode": "bulk"},
        "authoring": (
            "declare numeric source columns in properties and omit states; the engine "
            "derives every state below automatically, so never author states such as "
            "<Property>_Sum, <Property>_Mean, or <Property>_Var manually"
        ),
        "derived_states": [
            "<Property>_Count",
            "<Property>_Sum",
            "<Property>_Mean",
            "<Property>_Var",
            "<Property>_Min",
            "<Property>_Max",
            "<Property>_tdigest or <Property>_kll",
        ],
        "compatible_metrics": ["tdigest_quantile", "formula"],
    },
    "score_distribution": {
        "purpose": "model score quality, ROC/PR curves, calibration, and score summaries",
        "key_fields": ["score_properties", "outcome", "sketch_build_mode"],
        "defaults": {"sketch_build_mode": "bulk"},
        "authoring": (
            "declare score_properties and outcome and omit states; the default states "
            "below are derived automatically, and explicit states would replace them "
            "and break curve metrics"
        ),
        "default_states": [
            "Count",
            "<ScoreProperty>_tdigest_positives",
            "<ScoreProperty>_tdigest_negatives",
        ],
        "compatible_metrics": [
            "curve_from_digests",
            "calibration_from_digests",
            "tdigest_quantile",
            "formula",
        ],
    },
    "entity_lifecycle": {
        "purpose": "customer/product lifecycle, RFM, retention, and CLV-style summaries",
        "key_fields": ["entities.subject", "purchase_date", "value_column", "holding_column"],
        "compatible_metrics": ["lifecycle_summary", "approx_distinct_count", "formula"],
    },
    "entity_set": {
        "purpose": "unique reach, audience overlap, cohorts, retention, and set algebra",
        "key_fields": ["states with cpc, hll, or theta source_column"],
        "compatible_metrics": ["set_op", "approx_distinct_count", "formula"],
    },
    "funnel": {
        "purpose": "ordered journey stages and stage-to-stage drop-off",
        "key_fields": ["stages: [{name, when}] where when is a Boolean expression AST"],
        "compatible_metrics": ["funnel_dropoff", "formula"],
    },
    "snapshot": {
        "purpose": "periodic or accumulating status snapshots",
        "key_fields": ["snapshot_kind: periodic or accumulating", "cadence"],
        "compatible_metrics": ["formula", "approx_distinct_count"],
    },
}

_METRIC_KIND_DICTIONARY: dict[str, Any] = {
    "formula": {
        "requires": ["expression"],
        "expression_columns": "processor scalar state columns plus depends_on metric ids",
        "outputs": ["metric_id"],
    },
    "approx_distinct_count": {
        "requires": ["state"],
        "state_type": "cpc, hll, or theta",
        "outputs": ["metric_id"],
    },
    "tdigest_quantile": {
        "requires": ["state", "quantile"],
        "state_type": "tdigest or kll",
        "quantile_range": "0 <= quantile <= 1",
        "outputs": ["metric_id", "<Property>_Median/p25/p75/p90/p95/Min/Max when queried"],
    },
    "variant_compare": {
        "requires": ["variant_column", "test_role", "control_role"],
        "optional": {"confidence_level": "0 < value < 1; default 0.95"},
        "processor_needs": ["Positives", "Negatives", "variant_column"],
        "outputs": [
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
        ],
    },
    "curve_from_digests": {
        "requires": ["positive_state", "negative_state"],
        "state_type": "tdigest",
        "output": "roc_auc or average_precision",
        "curve_outputs": ["fpr", "tpr", "precision", "recall", "pos_fraction"],
    },
    "calibration_from_digests": {
        "requires": ["positive_state", "negative_state"],
        "state_type": "tdigest",
        "outputs": ["bin", "predicted", "observed"],
    },
    "contingency_test": {
        "requires": ["variant_column", "tests"],
        "allowed_tests": ["chi2", "g", "z"],
        "processor_needs": ["Positives", "Negatives", "variant_column"],
        "outputs": [
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
        ],
    },
    "proportion_test": {
        "requires": ["outputs optional"],
        "processor_needs": ["Positives", "Negatives"],
        "outputs": ["metric_id"],
    },
    "lifecycle_summary": {
        "requires": ["entity_lifecycle processor states"],
        "outputs": [
            "customers_count",
            "unique_holdings",
            "lifetime_value",
            "frequency",
            "recency",
            "monetary_value",
            "rfm_segment",
            "rfm_score",
        ],
    },
    "set_op": {
        "requires": ["op", "states or operands"],
        "allowed_ops": ["union", "intersection", "a_not_b", "diff"],
        "state_type": "theta",
        "outputs": ["metric_id"],
    },
    "funnel_dropoff": {
        "requires": ["from_stage", "to_stage"],
        "output": "rate or count",
        "outputs": ["metric_id"],
    },
}

_CHART_REQUIRED_FIELD_DICTIONARY: dict[str, Any] = {
    "authoring_rules": [
        "Chart data fields (x, y, value, color, ...) are top-level tile keys beside "
        "id/title/metric/chart; never nest them under another mapping.",
        "required_fields_by_chart lists mandatory tile keys per chart; 'a|b' means "
        "at least one of those alternative keys must be set.",
        "x and date keys usually take a time grain (Day, Month, Quarter, Year) or an "
        "approved low-cardinality dimension.",
        "y and value keys take one metric output column; a formula metric outputs a "
        "column named after the metric id, so y or value is the metric id itself.",
        "For variant_compare or contingency_test metrics, y/value must name one listed "
        "metric output such as Lift or CTR, and x is usually the variant column.",
        "color, names, and path keys take approved low-cardinality dimensions.",
    ],
    "required_fields_by_chart": {
        chart: ["|".join(group) for group in TILE_REQUIRED_ALTERNATIVES.get(chart, ())]
        for chart in get_args(model.Tile.model_fields["chart"].annotation)
    },
    "tile_examples": {
        "kpi_card": {
            "id": "engagement_ctr_kpi",
            "title": "Engagement CTR",
            "metric": "engagement_ctr",
            "chart": "kpi_card",
            "placement": "kpi_strip",
            "value": "engagement_ctr",
        },
        "line": {
            "id": "engagement_ctr_trend",
            "title": "CTR Trend by Day",
            "metric": "engagement_ctr",
            "chart": "line",
            "x": "Day",
            "y": "engagement_ctr",
        },
        "bar": {
            "id": "conversion_rate_by_channel",
            "title": "Conversion Rate by Channel",
            "metric": "conversion_rate",
            "chart": "bar",
            "x": "Channel",
            "y": "conversion_rate",
        },
        "kpi_card_from_variant_compare": {
            "id": "click_lift_kpi",
            "title": "Click Lift",
            "metric": "click_experiment_lift",
            "chart": "kpi_card",
            "placement": "kpi_strip",
            "value": "Lift",
        },
    },
}

_EXPRESSION_AST_DICTIONARY: dict[str, Any] = {
    "syntax_rules": [
        "Expressions are nested YAML/JSON mappings, not SQL or function-call strings.",
        "Use only the atoms and operator forms below; every expression-valued slot may nest.",
        "String concatenation is supported by op: concat with args and optional sep; "
        "concat(...) is not valid AST syntax.",
    ],
    "atoms": {
        "column_or_state": {"col": "ColumnOrState"},
        "literal": {"lit": "scalar"},
        "workspace_parameter": {"param": "Name"},
    },
    "operator_forms": {
        "unary": {
            "ops": ["not", "neg", "abs", "sqrt", "exp", "ceil", "floor"],
            "shape": {"op": "<one of ops>", "arg": "<expression>"},
        },
        "log": {
            "ops": ["log"],
            "shape": {"op": "log", "arg": "<expression>", "base": "<optional number>"},
        },
        "round": {
            "ops": ["round"],
            "shape": {
                "op": "round",
                "arg": "<expression>",
                "ndigits": "<optional integer>",
            },
        },
        "cast": {
            "ops": ["cast"],
            "shape": {"op": "cast", "arg": "<expression>", "dtype": "<dtype>"},
        },
        "logical": {
            "ops": ["and", "or"],
            "shape": {"op": "<one of ops>", "args": ["<expression>", "<expression>"]},
        },
        "arithmetic": {
            "ops": ["add", "sub", "mul", "div"],
            "shape": {"op": "<one of ops>", "args": ["<expression>", "<expression>"]},
        },
        "safe_division": {
            "ops": ["safe_div"],
            "shape": {"op": "safe_div", "num": "<expression>", "den": "<expression>"},
        },
        "concatenation": {
            "ops": ["concat"],
            "shape": {
                "op": "concat",
                "args": ["<string expression>", "<string expression>", "<optional more>"],
                "sep": "<optional string; empty by default>",
            },
        },
        "horizontal_min_max": {
            "ops": ["least", "greatest"],
            "shape": {"op": "<one of ops>", "args": ["<expression>", "<expression>"]},
        },
        "first_non_null": {
            "ops": ["coalesce"],
            "shape": {"op": "coalesce", "args": ["<expression>", "<optional more>"]},
        },
        "comparison_to_literal": {
            "ops": ["eq", "ne", "lt", "le", "gt", "ge"],
            "shape": {"op": "<one of ops>", "column": "<column>", "value": "<scalar>"},
        },
        "comparison_of_expressions": {
            "ops": ["eq", "ne", "lt", "le", "gt", "ge"],
            "shape": {"op": "<one of ops>", "args": ["<expression>", "<expression>"]},
        },
        "set_membership": {
            "ops": ["in", "not_in"],
            "shape": {
                "op": "<one of ops>",
                "column": "<column>",
                "values": ["<scalar>", "<optional more>"],
            },
        },
        "range": {
            "ops": ["between"],
            "shape": {
                "op": "between",
                "column": "<column>",
                "low": "<scalar>",
                "high": "<scalar>",
            },
        },
        "null_check": {
            "ops": ["is_null", "not_null"],
            "shape": {"op": "<one of ops>", "column": "<column>"},
        },
        "regex_match": {
            "ops": ["matches"],
            "shape": {"op": "matches", "column": "<column>", "pattern": "<regex>"},
        },
        "string_affix": {
            "ops": ["starts_with", "ends_with"],
            "shape": {"op": "<one of ops>", "column": "<column>", "value": "<string>"},
        },
        "multi_branch_conditional": {
            "ops": ["case"],
            "shape": {
                "op": "case",
                "when": [{"cond": "<expression>", "then": "<expression>"}],
                "else": "<expression>",
            },
        },
        "binary_conditional": {
            "ops": ["when_then"],
            "shape": {
                "op": "when_then",
                "cond": "<expression>",
                "then": "<expression>",
                "else": "<expression>",
            },
        },
        "date_truncation": {
            "ops": ["date_trunc"],
            "shape": {"op": "date_trunc", "unit": "<date_trunc unit>", "arg": "<expression>"},
        },
        "date_difference": {
            "ops": ["date_diff"],
            "shape": {
                "op": "date_diff",
                "unit": "<date_diff unit>",
                "end": "<expression>",
                "start": "<expression>",
            },
        },
        "date_part": {
            "ops": ["date_part"],
            "shape": {"op": "date_part", "unit": "<date_part unit>", "arg": "<expression>"},
        },
        "current_time": {"ops": ["now"], "shape": {"op": "now"}},
        "datetime_format": {
            "ops": ["strftime"],
            "shape": {"op": "strftime", "arg": "<expression>", "format": "<string>"},
        },
        "datetime_parse": {
            "ops": ["strptime"],
            "shape": {"op": "strptime", "arg": "<expression>", "format": "<string>"},
        },
    },
    "allowed_values": {
        "dtype": [
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "Float32",
            "Float64",
            "String",
            "Date",
            "Datetime",
            "Boolean",
        ],
        "date_trunc_unit": ["day", "month", "quarter", "year", "hour", "week_iso"],
        "date_diff_unit": ["seconds", "minutes", "hours", "days", "months", "years"],
        "date_part_unit": ["year", "month", "day", "quarter", "hour", "weekday"],
    },
    "examples": {
        "safe_ratio": {
            "op": "safe_div",
            "num": {"col": "Positives"},
            "den": {"col": "Count"},
        },
        "concatenate_fields": {
            "op": "concat",
            "args": [{"col": "Issue"}, {"col": "Group"}, {"col": "Name"}],
            "sep": "/",
        },
        "concatenate_after_cast": {
            "op": "concat",
            "args": [
                {"col": "Channel"},
                {"op": "cast", "arg": {"col": "Rank"}, "dtype": "String"},
            ],
            "sep": "-",
        },
    },
}


def catalog_prompt_dictionaries() -> dict[str, Any]:
    """Return the shared catalog dictionaries embedded in AI prompts."""

    return {
        "application_structure": _APPLICATION_STRUCTURE_DICTIONARY,
        "catalog_schema": _CATALOG_SCHEMA_DICTIONARY,
        "processor_kinds": _PROCESSOR_KIND_DICTIONARY,
        "metric_kinds": _METRIC_KIND_DICTIONARY,
        "chart_required_fields": _CHART_REQUIRED_FIELD_DICTIONARY,
        "expression_ast": _EXPRESSION_AST_DICTIONARY,
    }


def prompt_draft_sections(
    draft: dict[str, Any],
    *,
    hidden_fields: list[str] | None = None,
    preserve_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Return prompt-safe draft sections with unapproved field names removed."""

    prompt_draft = _prompt_draft(draft)
    if not hidden_fields:
        return prompt_draft
    redacted = redact_hidden_field_mentions(
        prompt_draft,
        hidden_fields,
        preserve_fields=preserve_fields,
    )
    return redacted if isinstance(redacted, dict) else {}


@dataclass(frozen=True)
class AICallSettings:
    """Runtime settings for a LiteLLM chat completion call."""

    model: str
    api_key: str = ""
    api_base: str = ""
    custom_llm_provider: str = ""
    temperature: float | None = None
    reasoning_effort: str = ""
    verbosity: str = ""
    timeout_seconds: int = 90


class AIProviderFailureCategory(StrEnum):
    """Privacy-safe categories shared by AI provider call surfaces."""

    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    PROVIDER = "provider"
    RESPONSE_VALIDATION = "response_validation"
    INTERNAL = "internal"


_AI_PROVIDER_CATEGORY_RETRYABLE: dict[AIProviderFailureCategory, bool] = {
    AIProviderFailureCategory.CONFIGURATION: False,
    AIProviderFailureCategory.AUTHENTICATION: False,
    AIProviderFailureCategory.AUTHORIZATION: False,
    AIProviderFailureCategory.RATE_LIMIT: True,
    AIProviderFailureCategory.TIMEOUT: True,
    AIProviderFailureCategory.NETWORK: True,
    AIProviderFailureCategory.PROVIDER: True,
    AIProviderFailureCategory.RESPONSE_VALIDATION: True,
    AIProviderFailureCategory.INTERNAL: False,
}


@dataclass(frozen=True)
class AIProviderFailureClassification:
    """Bounded provider-failure metadata that is safe for logs and UI state."""

    category: AIProviderFailureCategory
    retryable: bool


class AIProviderCallError(RuntimeError):
    """A provider failure safe to show in UI and record in caller logs."""

    def __init__(
        self,
        *,
        call_id: str,
        error_type: str,
        permission_denied: bool,
        category: AIProviderFailureCategory | str | None = None,
    ) -> None:
        self.call_id = call_id
        self.error_type = error_type
        self.permission_denied = permission_denied
        self.category = _coerce_ai_provider_failure_category(
            category,
            permission_denied=permission_denied,
        )
        self.retryable = _AI_PROVIDER_CATEGORY_RETRYABLE[self.category]
        detail = " due to insufficient permissions" if permission_denied else f" ({error_type})"
        super().__init__(f"AI provider call failed{detail}. Reference: {call_id}.")


class AIProviderResponseValidationError(ValueError):
    """A provider response did not satisfy the bounded completion contract."""


AI_PROVIDER_PREFLIGHT_MAX_TIMEOUT_SECONDS = 5
_AI_PROVIDER_LOCAL_NAMES = frozenset({"local", "lm_studio", "ollama", "vllm"})


@dataclass(frozen=True)
class AIProviderPreflightReceipt:
    """Privacy-safe receipt for one successful provider capability check."""

    reference: str
    timeout_seconds: int


def validate_ai_provider_settings(settings: AICallSettings) -> None:
    """Validate required provider settings locally without network work."""

    if ai_provider_settings_configured(settings):
        return
    reference = uuid.uuid4().hex[:12]
    logger.warning(
        "LLM provider configuration blocked: reference=%s failure_category=configuration "
        "retryable=False",
        reference,
    )
    raise AIProviderCallError(
        call_id=reference,
        error_type="ProviderConfigurationError",
        permission_denied=False,
        category=AIProviderFailureCategory.CONFIGURATION,
    )


def ai_provider_settings_configured(settings: AICallSettings) -> bool:
    """Return whether the settings can reach a hosted or local provider."""

    model_name = settings.model.strip()
    provider_name = (settings.custom_llm_provider or model_name.partition("/")[0]).casefold()
    return bool(
        model_name
        and (settings.api_key or settings.api_base or provider_name in _AI_PROVIDER_LOCAL_NAMES)
    )


def ai_provider_preflight_cache_key(
    settings: AICallSettings,
    *,
    capability: str = "chat_completion",
) -> str:
    """Return a secret-free fingerprint for a session preflight cache."""

    credential_digest = (
        hashlib.sha256(settings.api_key.encode("utf-8")).hexdigest() if settings.api_key else ""
    )
    payload = "\x1f".join(
        (
            capability,
            settings.model,
            settings.custom_llm_provider,
            settings.api_base,
            credential_digest,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preflight_ai_provider(
    settings: AICallSettings,
    *,
    call: Callable[..., str] | None = None,
) -> AIProviderPreflightReceipt:
    """Run one independent, bounded READY check for a configured provider."""

    validate_ai_provider_settings(settings)
    reference = uuid.uuid4().hex[:12]
    timeout_seconds = min(
        max(1, int(settings.timeout_seconds)),
        AI_PROVIDER_PREFLIGHT_MAX_TIMEOUT_SECONDS,
    )
    preflight_settings = replace(settings, timeout_seconds=timeout_seconds)
    provider_call = call or call_litellm
    logger.info(
        "LLM provider preflight started: reference=%s timeout_seconds=%s",
        reference,
        timeout_seconds,
    )
    response: Any = None
    failure: AIProviderCallError | None = None
    try:
        response = provider_call(
            preflight_settings,
            "Reply with READY.",
            system_prompt=(
                "This is a capability preflight. Do not request or infer data. "
                "Reply with READY only."
            ),
        )
    except AIProviderCallError as exc:
        logger.warning(
            "LLM provider preflight failed: reference=%s failure_category=%s retryable=%s",
            exc.call_id,
            exc.category,
            exc.retryable,
        )
        raise
    except PermissionError:
        # Caller-owned local policy gates, including Studio sharing consent, are
        # deliberately not reclassified as provider failures.
        raise
    except Exception as exc:
        classification = classify_ai_provider_failure(exc)
        failure = AIProviderCallError(
            call_id=reference,
            error_type=_safe_error_type_for_log(exc),
            permission_denied=classification.category
            in {
                AIProviderFailureCategory.AUTHENTICATION,
                AIProviderFailureCategory.AUTHORIZATION,
            },
            category=classification.category,
        )

    if failure is not None:
        logger.warning(
            "LLM provider preflight failed: reference=%s failure_category=%s retryable=%s",
            failure.call_id,
            failure.category,
            failure.retryable,
        )
        raise failure
    if not isinstance(response, str) or response.strip().casefold() != "ready":
        failure = AIProviderCallError(
            call_id=reference,
            error_type="AIProviderPreflightResponseError",
            permission_denied=False,
            category=AIProviderFailureCategory.RESPONSE_VALIDATION,
        )
        logger.warning(
            "LLM provider preflight failed: reference=%s failure_category=%s retryable=%s",
            failure.call_id,
            failure.category,
            failure.retryable,
        )
        raise failure
    logger.info("LLM provider preflight completed: reference=%s status=ready", reference)
    return AIProviderPreflightReceipt(reference=reference, timeout_seconds=timeout_seconds)


@dataclass(frozen=True)
class DraftAttemptDiagnostic:
    """Bounded, user-safe evidence for one catalog generation attempt."""

    attempt: int
    role: str
    stage: str
    issues: tuple[str, ...]
    issue_count: int
    issue_areas: tuple[tuple[str, int], ...]
    sections: tuple[str, ...]
    response_chars: int
    error_type: str = ""
    line: int | None = None
    column: int | None = None


@dataclass(frozen=True)
class DraftCandidateResult:
    """Result of a bounded generate, parse, merge, and validate operation.

    Invalid model output is deliberately not exposed as a candidate. Callers can
    keep the previously accepted revision in place and show ``issues`` as a safe
    retry receipt instead of accidentally placing broken YAML into review state.
    """

    draft: dict[str, Any] | None
    issues: tuple[str, ...]
    attempts: int
    last_response: str = ""
    failure_stage: str = ""
    attempt_diagnostics: tuple[DraftAttemptDiagnostic, ...] = ()
    reference: str = ""

    @property
    def ok(self) -> bool:
        """Return whether a validated candidate is available."""

        return self.draft is not None and not self.issues


def generate_schema_preview(
    frame: pl.DataFrame,
    approved_fields: list[str],
    example_fields: list[str],
    *,
    max_examples: int = 8,
) -> list[dict[str, Any]]:
    """Return a compact schema preview suitable for prompts."""

    approved = [field for field in approved_fields if field in frame.columns]
    example_set = set(example_fields)
    rows: list[dict[str, Any]] = []
    for field in approved:
        series = frame.get_column(field)
        row: dict[str, Any] = {
            "column": field,
            "dtype": str(frame.schema.get(field, "")),
            "nulls": int(series.null_count()),
            "unique": int(series.n_unique()),
        }
        if field in example_set:
            examples = [
                _jsonable(value)
                for value in series.drop_nulls().unique(maintain_order=True).head(max_examples)
            ]
            row["examples"] = examples
        rows.append(row)
    return rows


def prompt_for_config_draft(
    *,
    file_name: str,
    approved_schema: list[dict[str, Any]],
    approved_fields: list[str],
    hidden_fields: list[str],
    baseline_draft: dict[str, Any],
    user_goals: str = "",
) -> str:
    """Build the first-pass AI prompt for processors, metrics, and dashboards."""

    return _catalog_prompt(
        task=(
            "Create a richer Value Stream catalog draft from this approved schema. "
            "Create as many distinct valid processors as possible from the approved schema "
            "and business requirements. "
            "Return only YAML for processors, metrics, dashboards, and optional chat_with_data."
        ),
        file_name=file_name,
        approved_schema=approved_schema,
        approved_fields=approved_fields,
        hidden_fields=hidden_fields,
        current_draft=baseline_draft,
        user_goals=user_goals,
        extra_rules=[
            "Keep source definitions from pipelines.yaml unchanged.",
            "You may replace processors, metrics, and dashboards.",
            "You may add chat_with_data.agent_prompt and chat_with_data metric/dataset descriptions for Chat With Data guidance.",
            "Use only approved fields as source dimensions, filters, chart axes, or tile settings.",
            "Every metric.source must reference a processor id in processors.yaml.",
            "Every dashboard tile.metric must reference a metric id in metrics.yaml.",
            "Add metric display metadata when the business label, unit, format, or favorable direction is clear.",
            "Use explicit kpi_strip placement only for scalar kpi_card tiles; never infer KPI cards from arbitrary charts.",
            "Author page filters only for processor aggregate dimensions and declare compatible_tiles when coverage is partial.",
            "Do not stop after a minimal baseline: include every useful, non-duplicative "
            "processor whose required fields and valid state definitions are available.",
        ],
    )


def prompt_for_report_refresh(
    *,
    file_name: str,
    approved_schema: list[dict[str, Any]],
    approved_fields: list[str],
    hidden_fields: list[str],
    current_draft: dict[str, Any],
    user_goals: str = "",
) -> str:
    """Build a report-only refresh prompt from the current draft."""

    return _catalog_prompt(
        task=(
            "Refresh only dashboards.yaml for the current processors and metrics. "
            "Return only YAML containing dashboards."
        ),
        file_name=file_name,
        approved_schema=approved_schema,
        approved_fields=approved_fields,
        hidden_fields=hidden_fields,
        current_draft=current_draft,
        user_goals=user_goals,
        extra_rules=[
            "Do not change pipelines, processors, or metrics.",
            "Every tile.metric must reference an existing metric id from the current draft.",
            "Choose chart kinds compatible with the metric outputs and processor dimensions.",
            "Use only approved fields or known metric output columns in tile settings.",
            "Use kpi_strip only with scalar kpi_card tiles and configure comparison or target behavior explicitly.",
            "Preserve or author page filters and time_filter presets from aggregate-backed dimensions.",
            "Use scale_mode only on line or stacked_area comparison charts.",
        ],
    )


def prompt_for_draft_refinement(
    *,
    file_name: str,
    approved_schema: list[dict[str, Any]],
    approved_fields: list[str],
    hidden_fields: list[str],
    current_draft: dict[str, Any],
    instruction: str,
    user_goals: str = "",
) -> str:
    """Build a free-form revision prompt against the current draft."""

    return _catalog_prompt(
        task=(
            "Revise this Value Stream catalog draft according to the change request. "
            "Return only YAML for the sections you need to replace."
        ),
        file_name=file_name,
        approved_schema=approved_schema,
        approved_fields=approved_fields,
        hidden_fields=hidden_fields,
        current_draft=current_draft,
        user_goals=user_goals,
        change_request=instruction,
        extra_rules=[
            "Apply only the requested change; keep unrelated processors, metrics, and tiles unchanged.",
            "Keep existing ids stable unless the change request explicitly renames them.",
            "Every returned section must be complete for that YAML file.",
            "Do not change pipelines.",
            "When the request cannot be satisfied with approved fields, choose the closest valid alternative instead of inventing fields.",
        ],
    )


def prompt_for_repair(
    *,
    file_name: str,
    approved_schema: list[dict[str, Any]],
    approved_fields: list[str],
    hidden_fields: list[str],
    current_draft: dict[str, Any],
    validation_issues: list[str],
    validation_trace: str = "",
) -> str:
    """Build a targeted repair prompt for invalid generated sections."""

    return _catalog_prompt(
        task=(
            "Repair the invalid processors, metrics, and dashboards in this draft. "
            "Return only YAML for the sections you need to replace."
        ),
        file_name=file_name,
        approved_schema=approved_schema,
        approved_fields=approved_fields,
        hidden_fields=hidden_fields,
        current_draft=current_draft,
        validation_issues=validation_issues,
        validation_trace=validation_trace,
        extra_rules=[
            "Prioritize fixing the listed validation issues.",
            "Do not change pipelines unless a processor source reference is impossible to repair.",
            "Keep valid ids stable where possible.",
            "Every returned section must be complete for that YAML file.",
        ],
    )


def parse_ai_yaml_sections(text: str) -> dict[str, Any]:
    """Parse AI response text into normalized catalog sections."""

    payload = _extract_yaml_payload(text)
    loaded = yaml.safe_load(payload)
    if not isinstance(loaded, dict):
        raise ValueError("AI response must be a YAML mapping")

    normalized: dict[str, Any] = {}
    for raw_key, value in loaded.items():
        key = str(raw_key).removesuffix(".yaml")
        if key == "ai" and isinstance(value, dict) and "chat_with_data" in value:
            normalized["chat_with_data"] = _normalize_chat_with_data(value["chat_with_data"])
            continue
        if key not in _SECTION_KEYS:
            continue
        if key == "processors":
            normalized["processors"] = _normalize_processors(value)
        elif key == "metrics":
            normalized["metrics"] = _normalize_metrics(value)
        elif key == "dashboards":
            normalized["dashboards"] = _normalize_dashboards(value)
        elif key == "chat_with_data":
            normalized["chat_with_data"] = _normalize_chat_with_data(value)
    if not normalized:
        expected = ", ".join(_SECTION_KEYS)
        raise ValueError(f"AI response must include one of: {expected}")
    return normalized


def merge_draft_sections(base_draft: dict[str, Any], sections: dict[str, Any]) -> dict[str, Any]:
    """Return a draft with normalized AI sections overlaid."""

    merged = _deepcopy_yaml(base_draft)
    for section in _SECTION_KEYS:
        if section in sections:
            merged[section] = _deepcopy_yaml(sections[section])
    return merged


def generate_validated_candidate(
    *,
    base_draft: dict[str, Any],
    prompt: str,
    call: Callable[[str], str],
    repair_prompt: Callable[[dict[str, Any], list[str], str], str],
    max_repairs: int = 2,
    validate: Callable[[dict[str, Any]], tuple[bool, list[str]]] | None = None,
    operation: str = "catalog_candidate",
) -> DraftCandidateResult:
    """Generate one valid catalog candidate with at most two internal repairs.

    The operation boundary is intentionally small and deterministic: every model
    response is parsed, merged onto the same accepted base, and fully validated
    before it can leave this function. A repair response may replace complete
    sections of the latest parseable attempt, while an unparseable response falls
    back to the accepted base. Provider exceptions are allowed to propagate so the
    UI can preserve their safe call reference and offer an explicit retry.
    """

    repairs_remaining = min(max(int(max_repairs), 0), 2)
    candidate_validator = validate or validate_draft_catalog
    current_prompt = prompt
    repair_base = _deepcopy_yaml(base_draft)
    last_response = ""
    issues: list[str] = []
    failure_stage = ""
    attempt_diagnostics: list[DraftAttemptDiagnostic] = []
    reference = uuid.uuid4().hex[:12]
    safe_operation = _safe_candidate_operation_for_log(operation)
    max_attempts = repairs_remaining + 1
    logger.info(
        "AI draft candidate started: reference=%s operation=%s max_attempts=%s",
        reference,
        safe_operation,
        max_attempts,
    )

    for attempt in range(1, max_attempts + 1):
        role = "generation" if attempt == 1 else "repair"
        last_response = call(current_prompt)
        try:
            sections = parse_ai_yaml_sections(last_response)
        except (TypeError, ValueError, yaml.YAMLError) as exc:
            failure_stage = "parse"
            issues = [
                "The model response was not valid catalog YAML. Return complete catalog sections."
            ]
            candidate = repair_base
            trace = "Model response parsing failed before catalog validation."
            line, column = _safe_parse_position(exc)
            diagnostic = _draft_attempt_diagnostic(
                attempt=attempt,
                role=role,
                stage="parse",
                issues=issues,
                sections=(),
                response=last_response,
                error_type=_safe_error_type_for_log(exc),
                line=line,
                column=column,
            )
            attempt_diagnostics.append(diagnostic)
            _log_draft_attempt(reference, safe_operation, diagnostic)
        else:
            candidate = merge_draft_sections(repair_base, sections)
            ok, issues = candidate_validator(candidate)
            if ok:
                diagnostic = _draft_attempt_diagnostic(
                    attempt=attempt,
                    role=role,
                    stage="validated",
                    issues=[],
                    sections=tuple(key for key in _SECTION_KEYS if key in sections),
                    response=last_response,
                )
                attempt_diagnostics.append(diagnostic)
                _log_draft_attempt(reference, safe_operation, diagnostic)
                return DraftCandidateResult(
                    draft=candidate,
                    issues=(),
                    attempts=attempt,
                    last_response=last_response,
                    attempt_diagnostics=tuple(attempt_diagnostics),
                    reference=reference,
                )
            failure_stage = "validation"
            repair_base = candidate
            trace = validation_trace_for_repair(candidate)
            diagnostic = _draft_attempt_diagnostic(
                attempt=attempt,
                role=role,
                stage="validation",
                issues=issues,
                sections=tuple(key for key in _SECTION_KEYS if key in sections),
                response=last_response,
            )
            attempt_diagnostics.append(diagnostic)
            _log_draft_attempt(reference, safe_operation, diagnostic)

        if attempt > repairs_remaining:
            break
        current_prompt = repair_prompt(candidate, list(issues), trace)

    final_diagnostic = attempt_diagnostics[-1]
    logger.warning(
        "AI draft candidate exhausted: reference=%s operation=%s attempts=%s "
        "final_stage=%s issue_count=%s issue_areas=%s",
        reference,
        safe_operation,
        max_attempts,
        failure_stage,
        final_diagnostic.issue_count,
        ",".join(f"{area}:{count}" for area, count in final_diagnostic.issue_areas) or "none",
    )
    return DraftCandidateResult(
        draft=None,
        issues=tuple(issues),
        attempts=max_attempts,
        last_response=last_response,
        failure_stage=failure_stage,
        attempt_diagnostics=tuple(attempt_diagnostics),
        reference=reference,
    )


def _draft_attempt_diagnostic(
    *,
    attempt: int,
    role: str,
    stage: str,
    issues: list[str],
    sections: tuple[str, ...],
    response: Any,
    error_type: str = "",
    line: int | None = None,
    column: int | None = None,
) -> DraftAttemptDiagnostic:
    return DraftAttemptDiagnostic(
        attempt=attempt,
        role=role,
        stage=stage,
        issues=_bounded_draft_diagnostic_issues(issues),
        issue_count=len(issues),
        issue_areas=_draft_issue_area_counts(issues),
        sections=tuple(section for section in _SECTION_KEYS if section in sections),
        response_chars=len(response) if isinstance(response, str) else 0,
        error_type=error_type,
        line=line,
        column=column,
    )


def _bounded_draft_diagnostic_issues(issues: list[str]) -> tuple[str, ...]:
    bounded: list[str] = []
    for issue in issues[:_DRAFT_DIAGNOSTIC_MAX_ISSUES]:
        text = " ".join(str(issue).split())
        if len(text) > _DRAFT_DIAGNOSTIC_MAX_ISSUE_CHARS:
            text = text[: _DRAFT_DIAGNOSTIC_MAX_ISSUE_CHARS - 1] + "…"
        bounded.append(text)
    return tuple(bounded)


def _draft_issue_area_counts(issues: list[str]) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = dict.fromkeys(_DRAFT_DIAGNOSTIC_ISSUE_AREAS, 0)
    for issue in issues:
        counts[_draft_issue_area(str(issue))] += 1
    return tuple((area, counts[area]) for area in _DRAFT_DIAGNOSTIC_ISSUE_AREAS if counts[area])


def _draft_issue_area(issue: str) -> str:
    normalized = issue.casefold()
    if any(
        token in normalized
        for token in (
            "effective approved schema",
            "field-name contract",
            "stale raw field",
            "approved field",
        )
    ):
        area = "field_contract"
    elif normalized.startswith(("pipelines", "sources[", "source ")):
        area = "source"
    elif normalized.startswith(("processors", "processor ")):
        area = "processor"
    elif normalized.startswith(("metrics", "metric ")):
        area = "metric"
    elif normalized.startswith(("dashboards", "tiles", "report ", "set_dashboards")):
        area = "report"
    elif normalized.startswith(("chat_with_data", "chat ")):
        area = "chat"
    else:
        area = "other"
    return area


def _safe_parse_position(exc: BaseException) -> tuple[int | None, int | None]:
    mark = _safe_exception_attribute(exc, "problem_mark")
    return (
        _safe_one_based_position(_safe_exception_attribute(mark, "line")),
        _safe_one_based_position(_safe_exception_attribute(mark, "column")),
    )


def _safe_one_based_position(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return min(value + 1, _DRAFT_DIAGNOSTIC_MAX_POSITION)


def _safe_candidate_operation_for_log(operation: str) -> str:
    value = str(operation).strip().casefold()
    return value if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value) else "catalog_candidate"


def _log_draft_attempt(
    reference: str,
    operation: str,
    diagnostic: DraftAttemptDiagnostic,
) -> None:
    log = logger.info if diagnostic.stage == "validated" else logger.debug
    log(
        "AI draft candidate attempt: reference=%s operation=%s attempt=%s role=%s "
        "stage=%s status=%s response_chars=%s sections=%s issue_count=%s "
        "issue_areas=%s error_type=%s line=%s column=%s",
        reference,
        operation,
        diagnostic.attempt,
        diagnostic.role,
        diagnostic.stage,
        "success" if diagnostic.stage == "validated" else "failed",
        diagnostic.response_chars,
        ",".join(diagnostic.sections) or "none",
        diagnostic.issue_count,
        ",".join(f"{area}:{count}" for area, count in diagnostic.issue_areas) or "none",
        diagnostic.error_type or "none",
        diagnostic.line if diagnostic.line is not None else "none",
        diagnostic.column if diagnostic.column is not None else "none",
    )


def filter_draft_by_selection(
    draft: dict[str, Any],
    *,
    selected_processors: list[str] | None = None,
    selected_metrics: list[str] | None = None,
    selected_tiles: list[str] | None = None,
) -> dict[str, Any]:
    """Filter pending draft objects and keep cross-references consistent."""

    filtered = _deepcopy_yaml(draft)
    processor_ids = set(
        processor_ids_for_draft(filtered) if selected_processors is None else selected_processors
    )
    metric_names = set(
        metric_names_for_draft(filtered) if selected_metrics is None else selected_metrics
    )
    tile_key_set = set(tile_keys(filtered) if selected_tiles is None else selected_tiles)

    processors = filtered.get("processors", {}).get("processors", [])
    if isinstance(processors, list):
        filtered["processors"]["processors"] = [
            processor
            for processor in processors
            if isinstance(processor, dict) and processor.get("id") in processor_ids
        ]

    metrics = filtered.get("metrics", {}).get("metrics", {})
    if isinstance(metrics, dict):
        filtered["metrics"]["metrics"] = {
            name: metric_def
            for name, metric_def in metrics.items()
            if name in metric_names
            and isinstance(metric_def, dict)
            and metric_def.get("source") in processor_ids
        }
        metric_names = set(filtered["metrics"]["metrics"])

    dashboards = filtered.get("dashboards", {}).get("dashboards", [])
    if isinstance(dashboards, list):
        filtered["dashboards"]["dashboards"] = _filter_dashboards(
            dashboards,
            metric_names=metric_names,
            tile_key_set=tile_key_set,
        )
    return filtered


def section_name_diff(
    base: dict[str, Any], proposed: dict[str, Any]
) -> dict[str, dict[str, list[str]]]:
    """Return added/changed/removed/unchanged ids for major draft sections."""

    return {
        "processors": _name_diff(_processors_by_id(base), _processors_by_id(proposed)),
        "metrics": _name_diff(_metrics_by_name(base), _metrics_by_name(proposed)),
        "tiles": _name_diff(_tiles_by_key(base), _tiles_by_key(proposed)),
    }


def validate_draft_catalog(
    draft: dict[str, Any],
    *,
    source_columns_by_id: Mapping[str, Iterable[str]] | None = None,
) -> tuple[bool, list[str]]:
    """Validate an in-memory draft catalog against optional observed source columns."""

    try:
        catalog = model.Catalog.model_validate(_catalog_sections_for_validation(draft))
    except ValidationError as exc:
        return False, [
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
        ]
    except Exception as exc:
        error_type = _safe_error_type_for_log(exc)
        return False, [f"Catalog validation could not complete ({error_type})."]
    result = validate_catalog(catalog, source_columns_by_id=source_columns_by_id)
    return result.ok, [f"{issue.location}: {issue.message}" for issue in result.issues]


def _catalog_sections_for_validation(draft: dict[str, Any]) -> dict[str, Any]:
    return {
        section: _deepcopy_yaml(draft[section])
        for section in _CATALOG_SECTION_KEYS
        if section in draft
    }


def validation_trace_for_repair(draft: dict[str, Any]) -> str:
    """Return a traceback for repair prompts when validation raises.

    Normal catalog validation findings are returned by :func:`validate_draft_catalog`.
    This helper is repair-prompt-only diagnostic context for cases where structural
    model validation or semantic validation raises an exception with a useful stack.
    """

    try:
        catalog = model.Catalog.model_validate(_deepcopy_yaml(draft))
    except ValidationError as exc:
        return _format_validation_exception("Catalog model validation failed", exc)
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        return _format_validation_exception("Catalog model validation crashed", exc)

    try:
        validate_catalog(catalog)
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        return _format_validation_exception("Catalog semantic validation crashed", exc)
    return ""


def classify_draft_validation_issues(issues: list[str]) -> tuple[list[str], list[str]]:
    """Split draft issues into blocking and repairable-in-review lists.

    AI drafts often omit kind-specific metric knobs while preserving enough
    structure for users to continue into metric review. Those drafts still
    cannot be exported or applied, but accepting them into the editable draft
    lets the next review/repair step correct the missing fields.
    """

    blocking: list[str] = []
    repairable: list[str] = []
    for issue in issues:
        if _is_repairable_ai_draft_issue(issue):
            repairable.append(issue)
        else:
            blocking.append(issue)
    return blocking, repairable


def draft_object_counts(draft: dict[str, Any]) -> dict[str, int]:
    """Return compact object counts for a draft."""

    return {
        "Sources": len(draft.get("pipelines", {}).get("sources", []) or []),
        "Processors": len(processor_ids_for_draft(draft)),
        "Metrics": len(metric_names_for_draft(draft)),
        "Dashboards": len(draft.get("dashboards", {}).get("dashboards", []) or []),
        "Tiles": len(tile_keys(draft)),
    }


def processor_ids_for_draft(draft: dict[str, Any]) -> list[str]:
    """Return processor ids in draft order."""

    processors = draft.get("processors", {}).get("processors", [])
    if not isinstance(processors, list):
        return []
    return [str(item.get("id")) for item in processors if isinstance(item, dict) and item.get("id")]


def metric_names_for_draft(draft: dict[str, Any]) -> list[str]:
    """Return metric names in draft order."""

    metrics = draft.get("metrics", {}).get("metrics", {})
    return list(metrics) if isinstance(metrics, dict) else []


def tile_keys(draft: dict[str, Any]) -> list[str]:
    """Return stable tile keys as ``dashboard/page/tile``."""

    return list(_tiles_by_key(draft))


def call_litellm(
    settings: AICallSettings,
    prompt: str,
    *,
    system_prompt: str = (
        "You are a careful data product configuration assistant. "
        "Return concise valid YAML only when asked for YAML."
    ),
) -> str:
    """Call the configured model through LiteLLM and return message content."""

    call_id = uuid.uuid4().hex[:12]
    request_kwargs: dict[str, Any] = {
        "model": settings.model,
        "request_timeout": settings.timeout_seconds,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ],
    }
    if settings.temperature is not None:
        request_kwargs["temperature"] = settings.temperature
    if settings.reasoning_effort:
        request_kwargs["reasoning_effort"] = settings.reasoning_effort
    if settings.verbosity:
        request_kwargs["verbosity"] = settings.verbosity
    if settings.api_key:
        request_kwargs["api_key"] = settings.api_key
    if settings.api_base:
        request_kwargs["api_base"] = settings.api_base
    if settings.custom_llm_provider:
        request_kwargs["custom_llm_provider"] = settings.custom_llm_provider

    log_settings = _litellm_log_settings(settings)
    logger.info("LLM call started: call_id=%s settings=%s", call_id, log_settings)
    started_at = time.perf_counter()
    try:
        response = litellm_completion(**request_kwargs)
        content = _litellm_message_content(response)
    except Exception as exc:
        error_type = _safe_error_type_for_log(exc)
        classification = classify_ai_provider_failure(exc)
        logger.error(
            "LLM call failed: call_id=%s model=%s duration_ms=%.2f status=error "
            "failure_category=%s retryable=%s error_type=%s",
            call_id,
            log_settings["model"],
            (time.perf_counter() - started_at) * 1000,
            classification.category,
            classification.retryable,
            error_type,
        )
        failure = AIProviderCallError(
            call_id=call_id,
            error_type=error_type,
            permission_denied=classification.category
            in {
                AIProviderFailureCategory.AUTHENTICATION,
                AIProviderFailureCategory.AUTHORIZATION,
            },
            category=classification.category,
        )
    else:
        response_metadata = _litellm_response_log_metadata(response)
        logger.info(
            "LLM call completed: call_id=%s model=%s duration_ms=%.2f metadata=%s",
            call_id,
            log_settings["model"],
            (time.perf_counter() - started_at) * 1000,
            response_metadata,
        )
        return content

    # Raise outside the provider exception handler so caller tracebacks cannot
    # retain or re-log the raw exception text, request body, credentials, or paths.
    raise failure


def _litellm_log_settings(settings: AICallSettings) -> dict[str, Any]:
    return {
        "model": _safe_model_for_log(settings.model),
        "has_api_base": bool(settings.api_base),
        "has_custom_llm_provider": bool(settings.custom_llm_provider),
        "temperature": settings.temperature,
        "has_reasoning_effort": bool(settings.reasoning_effort),
        "has_verbosity": bool(settings.verbosity),
        "timeout_seconds": settings.timeout_seconds,
        "has_api_key": bool(settings.api_key),
    }


def _safe_model_for_log(model: str) -> str:
    """Return a useful model identifier without exposing filesystem locations."""

    value = model.strip()
    normalized = value.replace("\\", "/")
    local_path_markers = ("/Users/", "/home/", "/private/", "/tmp/", "/var/folders/")
    if (
        not value
        or value.startswith(("/", "~", "./", "../", "file:", "http://", "https://"))
        or re.match(r"^[A-Za-z]:[\\/]", value)
        or any(marker in normalized for marker in local_path_markers)
        or any(character in value for character in "\r\n\t")
    ):
        return "<redacted-model>"
    return value[:128]


def _litellm_response_log_metadata(response: Any) -> dict[str, Any]:
    """Return non-content response metadata suitable for routine logs."""

    metadata: dict[str, Any] = {"status": "success"}
    choices = _field(response, "choices")
    if choices:
        finish_reason = _safe_status_for_log(_field(choices[0], "finish_reason"))
        if finish_reason:
            metadata["status"] = finish_reason

    usage = _field(response, "usage")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        count = _field(usage, key)
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            metadata[key] = count
    return metadata


def _safe_status_for_log(value: Any) -> str:
    """Return a bounded provider status without accepting arbitrary text."""

    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower()
    if normalized in {
        "stop",
        "length",
        "tool_calls",
        "function_call",
        "content_filter",
        "cancelled",
        "timeout",
        "error",
    }:
        return normalized
    return "other"


def _safe_error_type_for_log(exc: Exception) -> str:
    """Return a bounded exception class name without provider-controlled text."""

    name = type(exc).__name__
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name):
        return name
    return "ProviderError"


def classify_ai_provider_failure(exc: Exception) -> AIProviderFailureClassification:
    """Classify a failure without inspecting provider-controlled message text."""

    status_code = _provider_failure_status_code(exc)
    error_code = _provider_failure_code(exc)
    type_names = _provider_failure_type_names(exc)
    category = _provider_failure_category(status_code, error_code, type_names)

    return AIProviderFailureClassification(
        category=category,
        retryable=_AI_PROVIDER_CATEGORY_RETRYABLE[category],
    )


_PROVIDER_FAILURE_EXACT_STATUS: dict[int, AIProviderFailureCategory] = {
    401: AIProviderFailureCategory.AUTHENTICATION,
    403: AIProviderFailureCategory.AUTHORIZATION,
    408: AIProviderFailureCategory.TIMEOUT,
    429: AIProviderFailureCategory.RATE_LIMIT,
    504: AIProviderFailureCategory.TIMEOUT,
}
_PROVIDER_CONFIGURATION_STATUS = frozenset({400, 404, 405, 409, 410, 412, 415, 422})
_PROVIDER_FAILURE_CATEGORY_ORDER = (
    AIProviderFailureCategory.AUTHENTICATION,
    AIProviderFailureCategory.AUTHORIZATION,
    AIProviderFailureCategory.RATE_LIMIT,
    AIProviderFailureCategory.TIMEOUT,
    AIProviderFailureCategory.NETWORK,
    AIProviderFailureCategory.RESPONSE_VALIDATION,
    AIProviderFailureCategory.CONFIGURATION,
    AIProviderFailureCategory.PROVIDER,
)
_PROVIDER_FAILURE_CODES: dict[AIProviderFailureCategory, frozenset[str]] = {
    AIProviderFailureCategory.CONFIGURATION: frozenset(
        {
            "bad_request",
            "configuration_error",
            "context_length_exceeded",
            "invalid_model",
            "invalid_request",
            "model_not_found",
            "unsupported_parameter",
        }
    ),
    AIProviderFailureCategory.AUTHENTICATION: frozenset(
        {
            "authentication_error",
            "invalid_api_key",
            "invalid_authentication",
            "unauthorized",
        }
    ),
    AIProviderFailureCategory.AUTHORIZATION: frozenset(
        {
            "authorization_error",
            "forbidden",
            "insufficient_permissions",
            "permission_denied",
        }
    ),
    AIProviderFailureCategory.RATE_LIMIT: frozenset(
        {
            "insufficient_quota",
            "rate_limit",
            "rate_limit_error",
            "rate_limit_exceeded",
            "too_many_requests",
        }
    ),
    AIProviderFailureCategory.TIMEOUT: frozenset({"request_timeout", "timed_out", "timeout"}),
    AIProviderFailureCategory.NETWORK: frozenset(
        {
            "api_connection_error",
            "connection_error",
            "connection_refused",
            "network_error",
        }
    ),
    AIProviderFailureCategory.PROVIDER: frozenset(
        {"api_error", "provider_error", "server_error", "service_unavailable"}
    ),
    AIProviderFailureCategory.RESPONSE_VALIDATION: frozenset(
        {"invalid_response", "response_validation_error"}
    ),
    AIProviderFailureCategory.INTERNAL: frozenset(),
}


_PROVIDER_FAILURE_TYPE_NAMES: dict[AIProviderFailureCategory, frozenset[str]] = {
    AIProviderFailureCategory.CONFIGURATION: frozenset(
        {
            "badrequesterror",
            "configurationerror",
            "configerror",
            "contentpolicyviolationerror",
            "contextwindowexceedederror",
            "invalidrequesterror",
            "notfounderror",
            "unprocessableentityerror",
            "unsupportedparamserror",
        }
    ),
    AIProviderFailureCategory.AUTHENTICATION: frozenset(
        {"authenticationerror", "unauthorizederror"}
    ),
    AIProviderFailureCategory.AUTHORIZATION: frozenset(
        {"authorizationerror", "permissiondeniederror"}
    ),
    AIProviderFailureCategory.RATE_LIMIT: frozenset({"ratelimiterror", "toomanyrequestserror"}),
    AIProviderFailureCategory.TIMEOUT: frozenset(
        {
            "apitimeouterror",
            "connecttimeout",
            "pooltimeout",
            "readtimeout",
            "requesttimeouterror",
            "timeout",
            "timeouterror",
            "writetimeout",
        }
    ),
    AIProviderFailureCategory.NETWORK: frozenset(
        {
            "apiconnectionerror",
            "clientconnectionerror",
            "connecterror",
            "connectionabortederror",
            "connectionerror",
            "connectionrefusederror",
            "connectionreseterror",
            "networkerror",
            "serverconnectionerror",
        }
    ),
    AIProviderFailureCategory.PROVIDER: frozenset(
        {
            "apierror",
            "badgatewayerror",
            "internalservererror",
            "serviceunavailableerror",
        }
    ),
    AIProviderFailureCategory.RESPONSE_VALIDATION: frozenset(
        {
            "aiproviderresponsevalidationerror",
            "jsondecodeerror",
            "responsevalidationerror",
            "validationerror",
        }
    ),
    AIProviderFailureCategory.INTERNAL: frozenset(),
}


def _provider_failure_category(
    status_code: int | None,
    error_code: str,
    type_names: frozenset[str],
) -> AIProviderFailureCategory:
    if status_code in _PROVIDER_FAILURE_EXACT_STATUS:
        return _PROVIDER_FAILURE_EXACT_STATUS[status_code]
    for category in _PROVIDER_FAILURE_CATEGORY_ORDER:
        if error_code in _PROVIDER_FAILURE_CODES[category]:
            return category
    for category in _PROVIDER_FAILURE_CATEGORY_ORDER:
        if type_names & _PROVIDER_FAILURE_TYPE_NAMES[category]:
            return category
    if status_code in _PROVIDER_CONFIGURATION_STATUS:
        return AIProviderFailureCategory.CONFIGURATION
    if status_code is not None and 500 <= status_code <= 599:
        return AIProviderFailureCategory.PROVIDER
    return AIProviderFailureCategory.INTERNAL


def _coerce_ai_provider_failure_category(
    category: AIProviderFailureCategory | str | None,
    *,
    permission_denied: bool,
) -> AIProviderFailureCategory:
    if category is None:
        return (
            AIProviderFailureCategory.AUTHORIZATION
            if permission_denied
            else AIProviderFailureCategory.PROVIDER
        )
    try:
        return AIProviderFailureCategory(category)
    except ValueError:
        return (
            AIProviderFailureCategory.AUTHORIZATION
            if permission_denied
            else AIProviderFailureCategory.PROVIDER
        )


def _provider_failure_type_names(exc: Exception) -> frozenset[str]:
    """Return bounded class names from the exception hierarchy."""

    names: set[str] = set()
    for cls in type(exc).__mro__:
        name = cls.__name__
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name):
            names.add(name.casefold())
    return frozenset(names)


def _provider_failure_status_code(exc: Exception) -> int | None:
    """Return a bounded HTTP status without reading exception text or bodies."""

    owners = [exc]
    response = _safe_exception_attribute(exc, "response")
    if response is not None:
        owners.append(response)
    for owner in owners:
        for attribute in ("status_code", "http_status", "status"):
            value = _safe_exception_attribute(owner, attribute)
            if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
                return value
            if isinstance(value, str) and re.fullmatch(r"[1-5][0-9]{2}", value):
                return int(value)
    return None


def _provider_failure_code(exc: Exception) -> str:
    """Return an allowlist-shaped provider code without reading error messages."""

    for attribute in ("code", "error_code"):
        value = _safe_exception_attribute(exc, attribute)
        if not isinstance(value, str):
            continue
        normalized = value.strip().casefold().replace("-", "_").replace(".", "_")
        if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", normalized):
            return normalized
    return ""


def _safe_exception_attribute(value: Any, attribute: str) -> Any:
    try:
        return getattr(value, attribute, None)
    except Exception:  # pragma: no cover - defensive third-party exception behavior
        return None


def _litellm_message_content(response: Any) -> str:
    choices = _field(response, "choices")
    if not choices:
        raise AIProviderResponseValidationError("LiteLLM response did not include choices")
    first = choices[0]
    message = _field(first, "message")
    content = _field(message, "content")
    if content is None:
        raise AIProviderResponseValidationError("LiteLLM response did not include message content")
    return str(content)


def _field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _catalog_prompt(
    *,
    task: str,
    file_name: str,
    approved_schema: list[dict[str, Any]],
    approved_fields: list[str],
    hidden_fields: list[str],
    current_draft: dict[str, Any],
    extra_rules: list[str],
    user_goals: str = "",
    change_request: str = "",
    validation_issues: list[str] | None = None,
    validation_trace: str = "",
) -> str:
    hidden_names = {field.casefold() for field in hidden_fields}
    prompt_approved_fields = [
        field for field in approved_fields if field.casefold() not in hidden_names
    ]
    prompt_approved_schema = [
        row
        for row in approved_schema
        if str(row.get("column") or "").casefold() not in hidden_names
    ]
    prompt_approved_schema = redact_hidden_field_mentions(
        prompt_approved_schema,
        hidden_fields,
        preserve_fields=prompt_approved_fields,
    )
    safe_file_name = redact_hidden_field_mentions(
        file_name, hidden_fields, preserve_fields=prompt_approved_fields
    )
    safe_user_goals = redact_hidden_field_mentions(
        user_goals, hidden_fields, preserve_fields=prompt_approved_fields
    )
    safe_change_request = redact_hidden_field_mentions(
        change_request, hidden_fields, preserve_fields=prompt_approved_fields
    )
    safe_validation_issues = redact_hidden_field_mentions(
        validation_issues or [],
        hidden_fields,
        preserve_fields=prompt_approved_fields,
    )
    safe_validation_trace = redact_hidden_field_mentions(
        validation_trace, hidden_fields, preserve_fields=prompt_approved_fields
    )
    safe_extra_rules = redact_hidden_field_mentions(
        extra_rules, hidden_fields, preserve_fields=prompt_approved_fields
    )
    goals_text = ""
    if safe_user_goals.strip():
        goals_text = (
            "Business requirements from the user. Satisfy each requirement with concrete "
            "processors, metrics, and report tiles where the approved schema allows it; "
            "skip requirements the schema cannot support:\n"
            f"{safe_user_goals.strip()}\n\n"
        )
    change_text = ""
    if safe_change_request.strip():
        change_text = f"Change request from the user:\n{safe_change_request.strip()}\n\n"
    issue_text = ""
    if safe_validation_issues:
        issue_text = "\nValidation errors to fix:\n" + yaml.safe_dump(
            safe_validation_issues, sort_keys=False
        )
    if safe_validation_trace.strip():
        issue_text += "\nValidation exception traceback, if available:\n"
        issue_text += safe_validation_trace.strip()
        issue_text += "\n"
    rules = "\n".join(f"- {rule}" for rule in safe_extra_rules)
    field_roles = _approved_field_role_dictionary(
        prompt_approved_schema,
        approved_fields=prompt_approved_fields,
    )
    hard_rules = "\n".join(f"- {rule}" for rule in _hard_output_rules())
    final_self_check = "\n".join(
        f"{index}. {rule}" for index, rule in enumerate(_self_check_rules(), start=1)
    )
    return (
        f"{task}\n\n"
        f"{goals_text}"
        f"{change_text}"
        f"Source sample: {safe_file_name}\n\n"
        "Application structure dictionary:\n"
        f"{yaml.safe_dump(_APPLICATION_STRUCTURE_DICTIONARY, sort_keys=False)}\n"
        "Catalog schema dictionary:\n"
        f"{yaml.safe_dump(_CATALOG_SCHEMA_DICTIONARY, sort_keys=False)}\n"
        "Processor kind dictionary:\n"
        f"{yaml.safe_dump(_PROCESSOR_KIND_DICTIONARY, sort_keys=False)}\n"
        "Metric kind dictionary:\n"
        f"{yaml.safe_dump(_METRIC_KIND_DICTIONARY, sort_keys=False)}\n"
        "Chart required-field dictionary:\n"
        f"{yaml.safe_dump(_CHART_REQUIRED_FIELD_DICTIONARY, sort_keys=False)}\n"
        "Expression AST dictionary:\n"
        f"{yaml.safe_dump(_EXPRESSION_AST_DICTIONARY, sort_keys=False)}\n"
        "Approved field role dictionary:\n"
        f"{yaml.safe_dump(field_roles, sort_keys=False)}\n"
        f"Rules:\n{rules}\n\n"
        f"Hard output rules:\n{hard_rules}\n\n"
        f"Approved fields:\n{yaml.safe_dump(prompt_approved_fields, sort_keys=False)}\n"
        f"Hidden field count: {len(hidden_fields)}\n"
        f"Approved schema preview:\n{yaml.safe_dump(prompt_approved_schema, sort_keys=False)}\n"
        "Current draft:\n"
        f"{yaml.safe_dump(prompt_draft_sections(current_draft, hidden_fields=hidden_fields, preserve_fields=prompt_approved_fields), sort_keys=False)}"
        f"{issue_text}\n\n"
        f"Final self-check before returning:\n{final_self_check}\n\n"
        "Return valid YAML only. Do not wrap the answer in prose or Markdown fences."
    )


def _approved_field_role_dictionary(
    approved_schema: list[dict[str, Any]],
    *,
    approved_fields: list[str],
) -> dict[str, list[str]]:
    schema_by_field = {
        str(row.get("column")): row
        for row in approved_schema
        if isinstance(row, dict) and row.get("column") is not None
    }
    roles: dict[str, list[str]] = {
        "safe_dimension_candidates": [],
        "time_candidates": [],
        "numeric_property_candidates": [],
        "outcome_candidates": [],
        "experiment_candidates": [],
        "score_candidates": [],
        "clv_candidates": [],
        "avoid_for_group_by_or_filters": [],
    }
    for field in approved_fields:
        row = schema_by_field.get(field, {})
        dtype = str(row.get("dtype", "")).casefold()
        unique = _safe_int(row.get("unique"))
        normalized = field.casefold()
        if _is_time_field(field, dtype):
            roles["time_candidates"].append(field)
        if _is_numeric_dtype(dtype) and not _looks_like_identifier(field):
            roles["numeric_property_candidates"].append(field)
        if _is_safe_dimension(field, dtype, unique):
            roles["safe_dimension_candidates"].append(field)
        else:
            roles["avoid_for_group_by_or_filters"].append(field)
        if _matches_any(normalized, ("outcome", "response", "conversion", "click", "accept")):
            roles["outcome_candidates"].append(field)
        if _matches_any(normalized, ("experiment", "variant", "control", "test", "modelcontrol")):
            roles["experiment_candidates"].append(field)
        if _matches_any(normalized, ("score", "propensity", "probability", "prediction", "rank")):
            roles["score_candidates"].append(field)
        if _matches_any(
            normalized,
            ("clv", "lifetime", "purchase", "holding", "monetary", "revenue", "customer"),
        ):
            roles["clv_candidates"].append(field)
    return {key: _dedupe(values) for key, values in roles.items()}


def redact_hidden_field_mentions(
    value: Any,
    hidden_fields: list[str],
    *,
    preserve_fields: list[str] | None = None,
) -> Any:
    """Recursively replace unapproved field-name mentions in dynamic prompt data.

    A hidden name is redacted even inside a longer identifier, because generated
    object ids such as ``CustomerID_metric`` derive from the hidden field. The
    exception is an identifier that belongs to a preserved (approved) field:
    hidden ``Outcome`` must not corrupt approved ``OutcomeTime`` — a different
    field — or the model would receive a schema it cannot satisfy.
    """

    preserved = {field.casefold() for field in (preserve_fields or []) if field}
    rules = tuple(
        (
            re.compile(rf"[A-Za-z0-9_]*{re.escape(field)}[A-Za-z0-9_]*", re.IGNORECASE),
            re.compile(re.escape(field), re.IGNORECASE),
            # Preserved names that contain this hidden name: an identifier built
            # from one of them mentions the hidden name only via the approved
            # field, so it must stay intact.
            tuple(name for name in preserved if field.casefold() in name),
        )
        for field in sorted({field for field in hidden_fields if field}, key=len, reverse=True)
    )

    def redact_token(token: str, mention: re.Pattern[str], covering: tuple[str, ...]) -> str:
        folded = token.casefold()
        if folded in preserved or any(name in folded for name in covering):
            return token
        return mention.sub("<hidden-field>", token)

    def redact_text(text: str) -> str:
        for token_pattern, mention, covering in rules:
            text = token_pattern.sub(
                lambda match, mention=mention, covering=covering: redact_token(
                    match.group(0), mention, covering
                ),
                text,
            )
        return text

    def redact(item: Any) -> Any:
        if isinstance(item, str):
            return redact_text(item)
        if isinstance(item, dict):
            cleaned: dict[Any, Any] = {}
            for key, nested_item in item.items():
                safe_key = redact(key) if isinstance(key, str) else key
                cleaned[safe_key] = redact(nested_item)
            return cleaned
        if isinstance(item, list):
            return [redact(nested_item) for nested_item in item]
        if isinstance(item, tuple):
            return tuple(redact(nested_item) for nested_item in item)
        return item

    return redact(value)


def _hard_output_rules() -> tuple[str, ...]:
    return (
        "Keep pipelines.yaml source definitions unchanged unless the task explicitly asks for pipelines.",
        "Return only complete top-level sections named processors, metrics, dashboards, and/or chat_with_data.",
        "Use dimensions as the authoring key for processor group_by fields; values must be approved fields.",
        "For time-based processors and dashboards, use available Day, Month, Quarter, and Year fields where relevant.",
        "Use processor time.column only when the timestamp field is approved or already present in the current draft.",
        "Do not emit legacy TOML-only settings such as metrics.global_filters; this catalog has no metrics.global_filters.",
        "Keep filters, tile facets, and processor dimensions limited to safe low-cardinality business dimensions.",
        "Do not use hidden fields, IDs, raw timestamps, free text, or high-cardinality fields as filters or chart facets.",
        "Every processor id, metric id, dashboard id, page id, and tile id must be stable, unique, and YAML-safe.",
        "Every processor source must reference an existing source id from the current draft pipelines.",
        "Every explicit processor state type must be one of count, value_sum, min, max, pooled_mean, "
        "pooled_variance, tdigest, kll, cpc, hll, theta, or topk; sum, mean, avg, var, and variance "
        "are invalid state types.",
        "numeric_distribution and score_distribution processors derive per-property states from "
        "properties/score_properties automatically; never author states such as <Property>_Sum, "
        "<Property>_Mean, or <Property>_Var manually.",
        "Never repeat a processor's variant_column in dimensions/group_by; the processor persists "
        "the variant column automatically.",
        "Every metric source must reference a processor id returned in processors or already present in current draft.",
        "Every metric must use only states that exist on its source processor.",
        "Every formula metric expression must reference only scalar state columns or declared depends_on metrics.",
        "Every dashboard tile.metric must reference a metric id returned in metrics or already present in current draft.",
        "Every tile must set the chart's required data fields from chart_required_field_dictionary "
        "as top-level tile keys; kpi_card needs value, line and bar need x and y.",
        "Every chart field must be an approved field, a time field, or a known output of that tile's metric.",
        "Choose chart kinds compatible with the metric's processor and output shape.",
        "Keep metrics and dashboards internally consistent; remove tiles/pages that depend on missing fields.",
        "If experiment fields or variant roles are unavailable, omit experiment processors, metrics, and reports.",
        "If CLV/lifecycle fields are unavailable, omit entity_lifecycle processors, CLV metrics, and CLV reports.",
        "If numeric descriptive properties are unavailable, omit numeric_distribution processors and descriptive reports.",
        "Set sketch_build_mode to bulk for every numeric_distribution and score_distribution processor; legacy is an explicit rollback escape hatch.",
        "Maximize useful processor coverage while keeping the catalog coherent; never add "
        "speculative or invalid objects merely to increase counts.",
    )


def _self_check_rules() -> tuple[str, ...]:
    return (
        "Every processor dimensions/group_by field exists in approved fields.",
        "Every processor filter, funnel stage condition, and metric expression uses valid expression AST structure.",
        "Every metric source exists in processors.",
        "Every metric state reference exists on the source processor with the required state type.",
        "Every explicit processor state type is a valid state_type_enum value, and no "
        "numeric_distribution or score_distribution processor hand-authors derived states.",
        "No processor repeats its variant_column in dimensions/group_by.",
        "Every metric depends_on value exists in metrics.",
        "Every report/dashboard tile metric exists in metrics.",
        "Every tile defines its chart's required data fields as top-level tile keys: kpi_card and "
        "gauge define value, line/bar/scatter define x and y, per chart_required_field_dictionary.",
        "Every report/dashboard tile field exists in approved fields, time fields, or known metric output columns.",
        "No experiment, CLV, model-score, or descriptive object is present without the required approved fields.",
        "No legacy TOML-only keys are present.",
        "Output valid YAML only.",
    )


def _is_time_field(field: str, dtype: str) -> bool:
    return field in {"Day", "Month", "Quarter", "Year"} or "date" in dtype or "time" in dtype


def _is_numeric_dtype(dtype: str) -> bool:
    return any(token in dtype for token in ("int", "float", "decimal"))


def _is_safe_dimension(field: str, dtype: str, unique: int | None) -> bool:
    if _looks_like_identifier(field) or _is_time_field(field, dtype):
        return False
    if unique is not None and unique > _LOW_CARDINALITY_LIMIT:
        return False
    return (
        unique is not None or "str" in dtype or "cat" in dtype or "bool" in dtype or "enum" in dtype
    )


def _looks_like_identifier(field: str) -> bool:
    normalized = field.casefold()
    return normalized.endswith(("id", "_id")) or "uuid" in normalized


def _matches_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_validation_exception(label: str, exc: BaseException) -> str:
    return (
        f"{label}: {type(exc).__name__}: {exc}\n"
        + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    ).strip()


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _prompt_draft(draft: dict[str, Any]) -> dict[str, Any]:
    out = {
        "pipelines": draft.get("pipelines", {}),
        "processors": draft.get("processors", {}),
        "metrics": draft.get("metrics", {}),
        "dashboards": draft.get("dashboards", {}),
    }
    if isinstance(draft.get("chat_with_data"), dict):
        out["chat_with_data"] = draft["chat_with_data"]
    return out


def _is_repairable_ai_draft_issue(issue: str) -> bool:
    return (
        _is_repairable_missing_metric_field(issue)
        or _REPAIRABLE_PROCESSOR_STATES_RE.match(issue) is not None
        or _REPAIRABLE_CONTINGENCY_TEST_RE.match(issue) is not None
    )


def _is_repairable_missing_metric_field(issue: str) -> bool:
    if not issue.endswith(": Field required"):
        return False
    location = issue.split(": ", 1)[0]
    parts = location.split(".")
    if len(parts) < 5 or parts[:2] != ["metrics", "metrics"]:
        return False
    for index, part in enumerate(parts[2:], start=2):
        fields = _REPAIRABLE_MISSING_METRIC_FIELDS.get(part)
        if fields is not None and index + 1 < len(parts):
            return parts[index + 1] in fields
    return False


def _extract_yaml_payload(text: str) -> str:
    match = _FENCED_BLOCK_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _normalize_processors(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get("processors"), list):
        return {"processors": value["processors"]}
    if isinstance(value, list):
        return {"processors": value}
    raise ValueError("processors section must be a list or mapping with `processors`")


def _normalize_metrics(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get("metrics"), dict):
        return {"metrics": value["metrics"]}
    if isinstance(value, dict):
        return {"metrics": value}
    raise ValueError("metrics section must be a mapping")


def _normalize_dashboards(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get("dashboards"), list):
        return {"theme": value.get("theme", {}), "dashboards": value["dashboards"]}
    if isinstance(value, list):
        return {"theme": {}, "dashboards": value}
    raise ValueError("dashboards section must be a list or mapping with `dashboards`")


def _normalize_chat_with_data(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("chat_with_data section must be a mapping")
    agent_prompt = str(value.get("agent_prompt") or "").strip()
    dataset_descriptions = _string_map(value.get("dataset_descriptions") or {})
    metric_descriptions = _string_map(value.get("metric_descriptions") or {})
    return _without_empty(
        {
            "agent_prompt": agent_prompt,
            "dataset_descriptions": dataset_descriptions,
            "metric_descriptions": metric_descriptions,
        }
    )


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "").strip()
        text = str(raw_value or "").strip()
        if key and text:
            out[key] = text
    return dict(sorted(out.items(), key=lambda item: item[0].casefold()))


def _without_empty(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value not in (None, "", [], {})}


def _filter_dashboards(
    dashboards: list[Any],
    *,
    metric_names: set[str],
    tile_key_set: set[str],
) -> list[dict[str, Any]]:
    filtered_dashboards: list[dict[str, Any]] = []
    for dashboard in dashboards:
        if not isinstance(dashboard, dict):
            continue
        dashboard_copy = dict(dashboard)
        pages: list[dict[str, Any]] = []
        for page in dashboard.get("pages", []) or []:
            if not isinstance(page, dict):
                continue
            page_copy = dict(page)
            tiles = []
            for tile in page.get("tiles", []) or []:
                if not isinstance(tile, dict):
                    continue
                key = f"{dashboard.get('id')}/{page.get('id')}/{tile.get('id')}"
                if key in tile_key_set and tile.get("metric") in metric_names:
                    tiles.append(tile)
            if tiles:
                page_copy["tiles"] = tiles
                pages.append(page_copy)
        if pages:
            dashboard_copy["pages"] = pages
            filtered_dashboards.append(dashboard_copy)
    return filtered_dashboards


def _processors_by_id(draft: dict[str, Any]) -> dict[str, Any]:
    processors = draft.get("processors", {}).get("processors", [])
    if not isinstance(processors, list):
        return {}
    return {
        str(processor["id"]): processor
        for processor in processors
        if isinstance(processor, dict) and processor.get("id")
    }


def _metrics_by_name(draft: dict[str, Any]) -> dict[str, Any]:
    metrics = draft.get("metrics", {}).get("metrics", {})
    return dict(metrics) if isinstance(metrics, dict) else {}


def _tiles_by_key(draft: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    dashboards = draft.get("dashboards", {}).get("dashboards", [])
    if not isinstance(dashboards, list):
        return out
    for dashboard in dashboards:
        if not isinstance(dashboard, dict):
            continue
        for page in dashboard.get("pages", []) or []:
            if not isinstance(page, dict):
                continue
            for tile in page.get("tiles", []) or []:
                if isinstance(tile, dict) and tile.get("id"):
                    out[f"{dashboard.get('id')}/{page.get('id')}/{tile.get('id')}"] = tile
    return out


def _name_diff(base: dict[str, Any], proposed: dict[str, Any]) -> dict[str, list[str]]:
    base_names = set(base)
    proposed_names = set(proposed)
    changed = sorted(
        name
        for name in base_names & proposed_names
        if _deepcopy_yaml(base[name]) != _deepcopy_yaml(proposed[name])
    )
    unchanged = sorted((base_names & proposed_names) - set(changed))
    return {
        "added": sorted(proposed_names - base_names),
        "changed": changed,
        "removed": sorted(base_names - proposed_names),
        "unchanged": unchanged,
    }


def _deepcopy_yaml(value: Any) -> Any:
    return yaml.safe_load(yaml.safe_dump(value, sort_keys=False))


def _jsonable(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
