"""Shared per-kind processor and metric parameter forms.

Both configuration UIs render the same editing surface through this module:
the Config Builder page edits the active catalog and the AI Configuration
Studio edits a draft dict. Callers pass plain-dict definitions plus widget
key prefixes; the forms return the kind-specific fields to merge into the
definition, or ``None`` when the form cannot produce a valid definition yet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import streamlit as st
import yaml

from valuestream.ui import builder, components, config_help

PROCESSOR_KIND_OPTIONS = (
    "binary_outcome",
    "numeric_distribution",
    "score_distribution",
    "entity_lifecycle",
    "entity_set",
    "funnel",
    "snapshot",
)
PROCESSOR_GRAIN_OPTIONS = ("Day", "Month", "Quarter", "Year", "Summary")
QUANTILE_ENGINE_OPTIONS = ("tdigest", "kll")
SNAPSHOT_KIND_OPTIONS = ("periodic", "accumulating")
SNAPSHOT_CADENCE_OPTIONS = ("", "daily", "weekly", "monthly")
TOPK_ERROR_TYPE_OPTIONS = ("NO_FALSE_POSITIVES", "NO_FALSE_NEGATIVES")
CONTINGENCY_TEST_OPTIONS = ("chi2", "g", "z")
SET_OP_OPTIONS = ("intersection", "union", "a_not_b", "diff")
FUNNEL_OUTPUT_OPTIONS = ("rate", "count")
CURVE_OUTPUT_OPTIONS = ("roc_auc", "average_precision")
LIFECYCLE_OUTPUT_OPTIONS = (
    "customers_count",
    "unique_holdings",
    "lifetime_value",
    "frequency",
    "recency",
    "monetary_value",
    "rfm_segment",
    "rfm_score",
)

SUBJECT_PREFERRED_FIELDS = ["SubjectID", "CustomerID", "customer_id", "interaction_id"]
OUTCOME_PREFERRED_FIELDS = ["Outcome", "pyOutcome", "outcome"]

# Keys written by processor_kind_fields; callers strip these from the base
# definition before merging so cleared values actually disappear.
PROCESSOR_KIND_MANAGED_FIELDS = frozenset(
    {
        "entities",
        "outcome",
        "outcome_column",
        "positive_values",
        "negative_values",
        "variant_column",
        "properties",
        "quantile_engine",
        "score_properties",
        "stages",
        "snapshot_kind",
        "cadence",
    }
)

METRIC_BASE_FIELDS = frozenset({"source", "kind", "description", "depends_on", "display"})


# ---------------------------------------------------------------------------
# Small widget helpers.
# ---------------------------------------------------------------------------


def with_current(options: list[str], current: str | list[str]) -> list[str]:
    """Return options extended with the current value(s) so they stay selectable."""
    values = [current] if isinstance(current, str) else list(current)
    extra = [value for value in values if value and value not in options]
    return [*options, *extra]


def select_or_text(
    label: str,
    options: list[str],
    current: Any,
    *,
    key: str,
    help_key: str,
    format_option: Callable[[str], str] | None = None,
) -> str:
    """Render a selectbox over options, or a text input when there are none."""
    current_text = str(current or "").strip()
    choices = with_current(options, current_text)
    if not choices:
        return st.text_input(
            label,
            value=current_text,
            key=key,
            help=config_help.field_help(help_key),
        ).strip()
    choices = ["", *choices]

    def _label(value: str) -> str:
        if not value:
            return "Select..."
        return format_option(value) if format_option else value

    return st.selectbox(
        label,
        choices,
        index=builder.option_index(choices, current_text),
        format_func=_label,
        key=key,
        help=config_help.field_help(help_key),
    )


def csv_list_field(label: str, value: Any, *, key: str, help_key: str) -> list[str]:
    """Render a comma-separated list editor and return the parsed values."""
    raw = st.text_input(
        label,
        value=", ".join(builder.string_list(value)),
        key=key,
        help=config_help.field_help(help_key),
    )
    return builder.csv_text_to_list(raw)


def first_preferred_field(fields: list[str], targets: list[str]) -> str:
    """Return the first field that case-insensitively matches a preferred name."""
    for target in targets:
        folded = target.casefold()
        match = next((field_ for field_ in fields if field_.casefold() == folded), "")
        if match:
            return match
    return ""


def stage_names_from_definition(raw_stages: Any) -> list[str]:
    """Return stage names from a funnel processor's ``stages`` definition."""
    stages: list[str] = []
    if isinstance(raw_stages, list):
        for item in raw_stages:
            if isinstance(item, dict) and item.get("name"):
                stages.append(str(item["name"]))
            elif isinstance(item, str):
                stages.append(item)
    return builder.dedupe(stages)


# ---------------------------------------------------------------------------
# Processor kind forms.
# ---------------------------------------------------------------------------


def processor_kind_fields(  # noqa: PLR0911 — dispatch table over the processor-kind union
    processor_def: dict[str, Any],
    kind: str,
    *,
    field_options: list[str],
    numeric_field_options: list[str] | None = None,
    key_prefix: str,
) -> dict[str, Any]:
    """Render kind-specific processor controls and return the edited fields."""
    numeric_options = numeric_field_options or field_options
    if kind == "binary_outcome":
        return _binary_outcome_fields(processor_def, field_options, key_prefix)
    if kind == "score_distribution":
        return _score_distribution_fields(processor_def, field_options, numeric_options, key_prefix)
    if kind == "numeric_distribution":
        return _numeric_distribution_fields(processor_def, numeric_options, key_prefix)
    if kind in {"entity_lifecycle", "entity_set"}:
        st.write("### Entity Settings")
        settings: dict[str, Any] = {}
        subject_col, _ = st.columns(2, gap="small")
        with subject_col:
            subject = _subject_field(processor_def, field_options, key_prefix)
        if subject:
            settings["entities"] = {"subject": subject}
        return settings
    if kind == "funnel":
        return _funnel_fields(processor_def, key_prefix)
    if kind == "snapshot":
        return _snapshot_fields(processor_def, key_prefix)
    return {}


def _subject_field(
    processor_def: dict[str, Any],
    field_options: list[str],
    key_prefix: str,
) -> str:
    raw_entities = processor_def.get("entities")
    entities: dict[str, Any] = raw_entities if isinstance(raw_entities, dict) else {}
    current = str(entities.get("subject", "") or "") or first_preferred_field(
        field_options, SUBJECT_PREFERRED_FIELDS
    )
    return select_or_text(
        "Subject Entity Field",
        field_options,
        current,
        key=f"{key_prefix}_entity_subject",
        help_key="processor.subject_field",
    ).strip()


def _outcome_fields(
    processor_def: dict[str, Any],
    field_options: list[str],
    key_prefix: str,
    *,
    positive_defaults: list[str],
    negative_defaults: list[str],
) -> dict[str, Any]:
    raw_outcome = processor_def.get("outcome")
    outcome: dict[str, Any] = raw_outcome if isinstance(raw_outcome, dict) else {}
    outcome_col, positive_col, negative_col = st.columns(
        [1.2, 1, 1],
        gap="xsmall",
        vertical_alignment="bottom",
    )
    with outcome_col:
        column = select_or_text(
            "Outcome Column",
            field_options,
            str(outcome.get("column", processor_def.get("outcome_column", "")) or "")
            or first_preferred_field(field_options, OUTCOME_PREFERRED_FIELDS),
            key=f"{key_prefix}_outcome_column",
            help_key="processor.outcome_column",
        ).strip()
    with positive_col:
        positive_values = csv_list_field(
            "Positive Values",
            builder.string_list(
                outcome.get("positive_values", processor_def.get("positive_values"))
            )
            or positive_defaults,
            key=f"{key_prefix}_positive_values",
            help_key="processor.positive_values",
        )
    with negative_col:
        negative_values = csv_list_field(
            "Negative Values",
            builder.string_list(
                outcome.get("negative_values", processor_def.get("negative_values"))
            )
            or negative_defaults,
            key=f"{key_prefix}_negative_values",
            help_key="processor.negative_values",
        )
    if not column:
        return {}
    return {
        "outcome": {
            "column": column,
            "positive_values": positive_values,
            "negative_values": negative_values,
        }
    }


def _binary_outcome_fields(
    processor_def: dict[str, Any],
    field_options: list[str],
    key_prefix: str,
) -> dict[str, Any]:
    st.write("### Binary Outcome Processor Settings")
    settings: dict[str, Any] = {}
    subject_col, variant_col = st.columns(2, gap="xsmall", vertical_alignment="bottom")
    with subject_col:
        subject = _subject_field(processor_def, field_options, key_prefix)
    if subject:
        settings["entities"] = {"subject": subject}
    with variant_col:
        variant_column = select_or_text(
            "Variant Column",
            field_options,
            str(processor_def.get("variant_column", "") or ""),
            key=f"{key_prefix}_variant_column",
            help_key="processor.variant_column",
        ).strip()
    settings.update(
        _outcome_fields(
            processor_def,
            field_options,
            key_prefix,
            positive_defaults=["Clicked", "Conversion"],
            negative_defaults=["Impression", "Pending"],
        )
    )
    if variant_column:
        settings["variant_column"] = variant_column
    return settings


def _score_distribution_fields(
    processor_def: dict[str, Any],
    field_options: list[str],
    numeric_options: list[str],
    key_prefix: str,
) -> dict[str, Any]:
    st.write("### Score Processor Settings")
    property_source_options = numeric_options or field_options
    current_properties = _score_properties_for_editor(processor_def, property_source_options)
    property_choices = with_current(property_source_options, current_properties)
    properties_col, subject_col = st.columns([2, 1], gap="xsmall", vertical_alignment="bottom")
    with properties_col:
        properties = st.multiselect(
            "Score Properties",
            property_choices,
            default=[item for item in current_properties if item in property_choices],
            accept_new_options=True,
            key=f"{key_prefix}_score_properties",
            help=config_help.field_help("processor.score_properties"),
        )
    settings: dict[str, Any] = {}
    if properties:
        settings["score_properties"] = builder.dedupe([str(item) for item in properties])
    with subject_col:
        subject = _subject_field(processor_def, field_options, key_prefix)
    if subject:
        settings["entities"] = {"subject": subject}
    settings.update(
        _outcome_fields(
            processor_def,
            field_options,
            key_prefix,
            positive_defaults=["Clicked"],
            negative_defaults=["Impression", "Pending"],
        )
    )
    return settings


def _score_properties_for_editor(
    processor_def: dict[str, Any],
    field_options: list[str],
) -> list[str]:
    configured = builder.score_properties_from_definition(processor_def)
    if _has_configured_score_properties(processor_def):
        return configured
    preferred = builder.dedupe(
        [
            first_preferred_field(
                field_options, ["Propensity", "propensity", "score", "model_score"]
            ),
            first_preferred_field(
                field_options,
                ["FinalPropensity", "final_propensity", "calibrated_score"],
            ),
        ]
    )
    return preferred or configured


def _has_configured_score_properties(processor_def: dict[str, Any]) -> bool:
    if builder.string_list(processor_def.get("score_properties")):
        return True
    for key in ("score_columns", "scores"):
        value = processor_def.get(key)
        if isinstance(value, dict) and any(str(item).strip() for item in value.values()):
            return True
        if isinstance(value, list) and any(str(item).strip() for item in value):
            return True
    return False


def _numeric_distribution_fields(
    processor_def: dict[str, Any],
    numeric_options: list[str],
    key_prefix: str,
) -> dict[str, Any]:
    st.write("### Distribution Processor Settings")
    current = builder.string_list(processor_def.get("properties"))
    choices = with_current(numeric_options, current)
    properties_col, engine_col = st.columns([3, 1], gap="xsmall", vertical_alignment="bottom")
    properties = properties_col.multiselect(
        "Numeric Properties",
        choices,
        default=[item for item in current if item in choices],
        accept_new_options=True,
        key=f"{key_prefix}_numeric_properties",
        help=config_help.field_help("processor.numeric_properties"),
    )
    engine = engine_col.selectbox(
        "Quantile Engine",
        list(QUANTILE_ENGINE_OPTIONS),
        index=builder.option_index(QUANTILE_ENGINE_OPTIONS, processor_def.get("quantile_engine")),
        key=f"{key_prefix}_quantile_engine",
        help=config_help.field_help("processor.quantile_engine"),
    )
    settings: dict[str, Any] = {"quantile_engine": engine}
    if properties:
        settings["properties"] = list(properties)
    return settings


def _funnel_fields(processor_def: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    st.write("### Funnel Processor Settings")
    stage_names = stage_names_from_definition(processor_def.get("stages"))
    stages_col, _ = st.columns(2, gap="xsmall")
    raw = stages_col.text_input(
        "Stages",
        value=", ".join(stage_names),
        key=f"{key_prefix}_stages",
        help=config_help.field_help("processor.stages"),
    )
    merged_stages = builder.merge_stage_definitions(
        processor_def.get("stages"),
        builder.csv_text_to_list(raw),
    )
    missing_when = builder.stage_names_missing_when(merged_stages)
    if missing_when:
        st.warning(
            "Stage(s) without a `when` expression: "
            f"{', '.join(missing_when)}. The funnel cannot run until each stage "
            "has a Boolean `when` expression (edit the stage YAML directly)."
        )
    return {"stages": merged_stages}


def _snapshot_fields(processor_def: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    st.write("### Snapshot Processor Settings")
    kind_col, cadence_col = st.columns(2, gap="xsmall", vertical_alignment="bottom")
    snapshot_kind = kind_col.selectbox(
        "Snapshot Kind",
        list(SNAPSHOT_KIND_OPTIONS),
        index=builder.option_index(SNAPSHOT_KIND_OPTIONS, processor_def.get("snapshot_kind")),
        key=f"{key_prefix}_snapshot_kind",
        help=config_help.field_help("processor.snapshot_kind"),
    )
    cadence = cadence_col.selectbox(
        "Cadence",
        list(SNAPSHOT_CADENCE_OPTIONS),
        index=builder.option_index(SNAPSHOT_CADENCE_OPTIONS, processor_def.get("cadence")),
        format_func=lambda value: value or "None",
        key=f"{key_prefix}_cadence",
        help=config_help.field_help("processor.cadence"),
    )
    settings: dict[str, Any] = {"snapshot_kind": snapshot_kind}
    if cadence:
        settings["cadence"] = cadence
    return settings


# ---------------------------------------------------------------------------
# Metric kind forms.
# ---------------------------------------------------------------------------


@dataclass
class MetricFormContext:
    """Providers a metric form needs from either the catalog or a draft."""

    state_options: Callable[[set[str]], list[str]]
    digest_pairs: list[tuple[str, str, str]] = field(default_factory=list)
    funnel_stages: list[str] = field(default_factory=list)
    default_variant_column: str = ""
    variant_roles: dict[str, str] = field(default_factory=dict)
    state_label: Callable[[str], str] | None = None
    default_digest_pair: Callable[[bool], tuple[str, str] | None] | None = None


def metric_kind_fields(  # noqa: PLR0911, PLR0912 — dispatch table over the metric-kind union
    kind: str,
    seed: dict[str, Any],
    ctx: MetricFormContext,
    *,
    key_prefix: str,
) -> dict[str, Any] | None:
    """Render kind-specific metric controls and return the edited fields.

    Returns ``None`` when required inputs are missing so callers can keep the
    previous definition instead of writing a broken one.
    """
    if kind == "formula":
        return _formula_fields(seed, ctx, key_prefix)
    if kind == "approx_distinct_count":
        state = _state_field(
            "Cardinality Sketch State",
            seed.get("state"),
            ctx,
            {"cpc", "hll", "theta"},
            f"{key_prefix}_cardinality_state",
            help_key="metric.state",
        )
        if not state:
            st.warning("Approx distinct count requires a CPC, HLL, or Theta state.")
            return None
        return {"state": state}
    if kind == "topk_items":
        return _topk_fields(seed, ctx, key_prefix)
    if kind == "tdigest_quantile":
        return _quantile_fields(seed, ctx, key_prefix)
    if kind == "curve_from_digests":
        pair = _digest_pair_fields(seed, ctx, key_prefix, final=False)
        if pair is None:
            return None
        pair["output"] = st.selectbox(
            "Output",
            list(CURVE_OUTPUT_OPTIONS),
            index=builder.option_index(CURVE_OUTPUT_OPTIONS, seed.get("output")),
            key=f"{key_prefix}_curve_output",
            help=config_help.field_help("metric.output"),
        )
        return pair
    if kind == "calibration_from_digests":
        return _digest_pair_fields(seed, ctx, key_prefix, final=True)
    if kind == "variant_compare":
        return _variant_compare_fields(seed, ctx, key_prefix)
    if kind == "contingency_test":
        return _contingency_fields(seed, ctx, key_prefix)
    if kind == "proportion_test":
        return _variant_compare_fields(seed, ctx, f"{key_prefix}_proportion")
    if kind == "lifecycle_summary":
        return _lifecycle_fields(seed, key_prefix)
    if kind == "set_op":
        return _set_op_fields(seed, ctx, key_prefix)
    if kind == "funnel_dropoff":
        return _funnel_dropoff_fields(seed, ctx, key_prefix)
    # Unknown kinds have no visual controls; keep their kind-specific fields
    # intact instead of dropping them on apply.
    return {key: value for key, value in seed.items() if key not in METRIC_BASE_FIELDS}


def is_simple_formula(expression: Any) -> bool:
    """Return whether the num/den form can represent an expression losslessly."""
    if expression in (None, {}):
        return True
    if not isinstance(expression, dict):
        return False
    if set(expression) == {"col"}:
        return True
    if expression.get("op") == "safe_div" and set(expression) <= {"op", "num", "den"}:
        num = expression.get("num")
        den = expression.get("den")
        return (
            isinstance(num, dict)
            and set(num) == {"col"}
            and isinstance(den, dict)
            and set(den) == {"col"}
        )
    return False


def _formula_fields(
    seed: dict[str, Any],
    ctx: MetricFormContext,
    key_prefix: str,
) -> dict[str, Any] | None:
    seed_expression = seed.get("expression") if isinstance(seed.get("expression"), dict) else None
    simple = is_simple_formula(seed_expression)
    mode_options = ["Numerator / Denominator", "Expression YAML"]
    mode_key = f"{key_prefix}_formula_mode"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = mode_options[0] if simple else mode_options[1]
    mode = st.segmented_control(
        "Formula Mode",
        mode_options,
        default=st.session_state[mode_key],
        key=f"{mode_key}_control",
        help=config_help.field_help("metric.formula_mode"),
    )
    st.session_state[mode_key] = mode or st.session_state[mode_key]
    scalar_states = ctx.state_options(set(builder.SCALAR_STATE_TYPES))
    if st.session_state[mode_key] == "Expression YAML":
        raw_key = f"{key_prefix}_formula_expression"
        default_text = (
            builder.expression_yaml(seed_expression)
            or yaml.safe_dump(
                {"col": scalar_states[0] if scalar_states else "Count"},
                sort_keys=False,
            ).strip()
        )
        components.sync_text_area(raw_key, default_text)
        raw = st.text_area(
            "Expression YAML",
            key=raw_key,
            height=180,
            help=config_help.field_help("metric.expression"),
        )
        try:
            parsed = yaml.safe_load(raw) or {}
        except yaml.YAMLError as exc:
            st.warning(f"Expression YAML could not be parsed: {exc}")
            return None
        if not isinstance(parsed, dict) or not parsed:
            st.warning("Expression must be a non-empty YAML mapping.")
            return None
        return {"expression": parsed}
    if not scalar_states:
        st.warning("This processor has no scalar states for formula metrics.")
        return None
    if not simple:
        st.warning(
            "This metric uses a compound expression that the numerator/denominator "
            "form cannot represent. Use Expression YAML mode to keep it; applying "
            "in this mode replaces the expression."
        )
    default_numerator, default_denominator = _formula_defaults(seed)
    numerator = st.selectbox(
        "Numerator",
        scalar_states,
        index=builder.option_index(scalar_states, default_numerator or "Positives"),
        key=f"{key_prefix}_formula_num",
        help=config_help.field_help("metric.numerator"),
    )
    denominator_options = ["", *scalar_states]
    denominator = st.selectbox(
        "Denominator",
        denominator_options,
        index=builder.option_index(denominator_options, default_denominator),
        format_func=lambda value: value or "None",
        key=f"{key_prefix}_formula_den",
        help=config_help.field_help("metric.denominator"),
    )
    if denominator:
        return {
            "expression": {
                "op": "safe_div",
                "num": {"col": numerator},
                "den": {"col": denominator},
            }
        }
    return {"expression": {"col": numerator}}


def _formula_defaults(seed: dict[str, Any]) -> tuple[str, str]:
    expression = seed.get("expression")
    if not isinstance(expression, dict):
        return "", "Count"
    if "col" in expression:
        return str(expression.get("col", "") or ""), ""
    if expression.get("op") == "safe_div":
        return _expression_column(expression.get("num")), _expression_column(expression.get("den"))
    return "", "Count"


def _expression_column(expression: Any) -> str:
    if isinstance(expression, dict) and expression.get("col"):
        return str(expression["col"])
    return ""


def _state_field(
    label: str,
    current: Any,
    ctx: MetricFormContext,
    state_types: set[str],
    key: str,
    *,
    help_key: str,
) -> str:
    return select_or_text(
        label,
        ctx.state_options(state_types),
        current,
        key=key,
        help_key=help_key,
        format_option=ctx.state_label,
    ).strip()


def _topk_fields(
    seed: dict[str, Any],
    ctx: MetricFormContext,
    key_prefix: str,
) -> dict[str, Any] | None:
    state = _state_field(
        "Top-K State",
        seed.get("state"),
        ctx,
        {"topk"},
        f"{key_prefix}_topk_state",
        help_key="metric.state",
    )
    limit = st.number_input(
        "Item Limit",
        min_value=1,
        max_value=100,
        value=int(seed.get("limit") or 10),
        step=1,
        key=f"{key_prefix}_topk_limit",
        help=config_help.field_help("metric.topk_limit"),
    )
    error_type = st.selectbox(
        "Error Type",
        list(TOPK_ERROR_TYPE_OPTIONS),
        index=builder.option_index(TOPK_ERROR_TYPE_OPTIONS, seed.get("error_type")),
        key=f"{key_prefix}_topk_error_type",
        help=config_help.field_help("metric.topk_error_type"),
    )
    if not state:
        st.warning("Top-K items require a Top-K sketch state.")
        return None
    fields: dict[str, Any] = {"state": state, "limit": int(limit)}
    if error_type != "NO_FALSE_POSITIVES" or seed.get("error_type"):
        fields["error_type"] = error_type
    return fields


def _quantile_fields(
    seed: dict[str, Any],
    ctx: MetricFormContext,
    key_prefix: str,
) -> dict[str, Any] | None:
    state = _state_field(
        "Digest State",
        seed.get("state"),
        ctx,
        {"tdigest", "kll"},
        f"{key_prefix}_quantile_state",
        help_key="metric.state",
    )
    quantile = st.number_input(
        "Quantile",
        min_value=0.0,
        max_value=1.0,
        value=builder.float_in_range(seed.get("quantile"), default=0.5, minimum=0.0, maximum=1.0),
        step=0.05,
        key=f"{key_prefix}_quantile_value",
        help=config_help.field_help("metric.quantile"),
    )
    if not state:
        st.warning("Digest quantiles require a t-digest or KLL state.")
        return None
    return {"state": state, "quantile": float(quantile)}


def _digest_pair_fields(
    seed: dict[str, Any],
    ctx: MetricFormContext,
    key_prefix: str,
    *,
    final: bool,
) -> dict[str, Any] | None:
    digest_states = ctx.state_options({"tdigest"})
    pairs = ctx.digest_pairs
    default_pair: tuple[str, str] | None = None
    if ctx.default_digest_pair is not None:
        default_pair = ctx.default_digest_pair(final)
    if default_pair is None and pairs:
        default_pair = (pairs[0][1], pairs[0][2])
    positive_default = str(
        seed.get("positive_state")
        or (default_pair[0] if default_pair else "")
        or (digest_states[0] if digest_states else "")
    )
    negative_default = str(
        seed.get("negative_state")
        or (default_pair[1] if default_pair else "")
        or (digest_states[1] if len(digest_states) > 1 else "")
    )
    custom_pair = ("Custom digest states", "", "")
    selected_pair = custom_pair
    if pairs:
        pair_options = [*pairs, custom_pair]
        selected_pair = st.selectbox(
            "Digest Property",
            pair_options,
            index=builder.digest_pair_option_index(
                pair_options, positive_default, negative_default
            ),
            format_func=builder.digest_pair_option_label,
            key=f"{key_prefix}_digest_property",
            help=config_help.field_help("metric.digest_property"),
        )
        if selected_pair != custom_pair:
            positive_default = selected_pair[1]
            negative_default = selected_pair[2]
    elif len(digest_states) >= 2:
        st.info("No positive/negative digest metadata found; choose digest states manually.")
    pair_key = builder.widget_key_fragment("|".join(selected_pair))
    positive_state = _state_field(
        "Positive Digest",
        positive_default,
        ctx,
        {"tdigest"},
        f"{key_prefix}_digest_pos_{pair_key}",
        help_key="metric.positive_digest",
    )
    negative_state = _state_field(
        "Negative Digest",
        negative_default,
        ctx,
        {"tdigest"},
        f"{key_prefix}_digest_neg_{pair_key}",
        help_key="metric.negative_digest",
    )
    if not positive_state or not negative_state:
        st.warning("This metric requires positive and negative t-digest states.")
        return None
    return {"positive_state": positive_state, "negative_state": negative_state}


def _variant_compare_fields(
    seed: dict[str, Any],
    ctx: MetricFormContext,
    key_prefix: str,
) -> dict[str, Any] | None:
    roles = ctx.variant_roles
    variant_column = st.text_input(
        "Variant Column",
        value=str(seed.get("variant_column") or ctx.default_variant_column or ""),
        key=f"{key_prefix}_variant_column",
        help=config_help.field_help("metric.variant_column"),
    ).strip()
    test_role = st.text_input(
        "Test Role",
        value=str(seed.get("test_role") or roles.get("Test", "Test") or "Test"),
        key=f"{key_prefix}_test_role",
        help=config_help.field_help("metric.test_role"),
    ).strip()
    control_role = st.text_input(
        "Control Role",
        value=str(seed.get("control_role") or roles.get("Control", "Control") or "Control"),
        key=f"{key_prefix}_control_role",
        help=config_help.field_help("metric.control_role"),
    ).strip()
    confidence_level = st.number_input(
        "Confidence Level",
        min_value=0.5,
        max_value=0.999,
        value=float(seed.get("confidence_level") or 0.95),
        step=0.01,
        key=f"{key_prefix}_confidence_level",
        help=config_help.field_help("metric.confidence_level"),
    )
    outputs = csv_list_field(
        "Outputs",
        seed.get("outputs"),
        key=f"{key_prefix}_variant_outputs",
        help_key="metric.outputs",
    )
    if not variant_column:
        st.warning("Variant comparison requires a variant column.")
        return None
    fields: dict[str, Any] = {
        "variant_column": variant_column,
        "test_role": test_role or "Test",
        "control_role": control_role or "Control",
        "confidence_level": float(confidence_level),
    }
    if outputs:
        fields["outputs"] = outputs
    return fields


def _contingency_fields(
    seed: dict[str, Any],
    ctx: MetricFormContext,
    key_prefix: str,
) -> dict[str, Any] | None:
    variant_column = st.text_input(
        "Variant Column",
        value=str(seed.get("variant_column") or ctx.default_variant_column or ""),
        key=f"{key_prefix}_contingency_variant",
        help=config_help.field_help("metric.variant_column"),
    ).strip()
    current_tests = [
        test for test in builder.string_list(seed.get("tests")) if test in CONTINGENCY_TEST_OPTIONS
    ]
    tests = st.multiselect(
        "Tests",
        list(CONTINGENCY_TEST_OPTIONS),
        default=current_tests or list(CONTINGENCY_TEST_OPTIONS),
        key=f"{key_prefix}_contingency_tests",
        help=config_help.field_help("metric.tests"),
    )
    outputs = csv_list_field(
        "Outputs",
        seed.get("outputs"),
        key=f"{key_prefix}_contingency_outputs",
        help_key="metric.outputs",
    )
    if not variant_column:
        st.warning("Contingency tests require a variant column.")
        return None
    fields: dict[str, Any] = {"variant_column": variant_column, "tests": tests or ["chi2"]}
    if outputs:
        fields["outputs"] = outputs
    return fields


def _lifecycle_fields(seed: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    default_outputs = [
        output
        for output in (
            builder.string_list(seed.get("outputs"))
            or ["frequency", "monetary_value", "rfm_segment", "rfm_score"]
        )
        if output in LIFECYCLE_OUTPUT_OPTIONS
    ]
    outputs = st.multiselect(
        "Output Columns",
        list(LIFECYCLE_OUTPUT_OPTIONS),
        default=default_outputs,
        key=f"{key_prefix}_lifecycle_outputs",
        help=config_help.field_help("metric.lifecycle_outputs"),
    )
    return {"outputs": outputs} if outputs else {}


def _set_op_fields(
    seed: dict[str, Any],
    ctx: MetricFormContext,
    key_prefix: str,
) -> dict[str, Any] | None:
    states = ctx.state_options({"theta"})
    if len(states) < 2:
        st.warning("Set operations require at least two theta states.")
        return None
    op = st.selectbox(
        "Operation",
        list(SET_OP_OPTIONS),
        index=builder.option_index(SET_OP_OPTIONS, seed.get("op")),
        key=f"{key_prefix}_set_op",
        help=config_help.field_help("metric.set_operation"),
    )
    default_states = (
        builder.string_list(seed.get("states")) or builder.operand_states(seed) or states[:2]
    )
    selected = st.multiselect(
        "Theta States",
        states,
        default=[state for state in default_states if state in states],
        key=f"{key_prefix}_set_states",
        help=config_help.field_help("metric.theta_states"),
    )
    if op in {"a_not_b", "diff"} and len(selected) != 2:
        st.warning("Difference metrics require exactly two theta states.")
        return None
    if not selected:
        st.warning("Choose at least one theta state.")
        return None
    return {"op": op, "states": list(selected)}


def _funnel_dropoff_fields(
    seed: dict[str, Any],
    ctx: MetricFormContext,
    key_prefix: str,
) -> dict[str, Any] | None:
    stages = ctx.funnel_stages
    if len(stages) < 2:
        st.warning("Funnel drop-off requires at least two configured stages.")
        return None
    from_stage = st.selectbox(
        "From Stage",
        stages,
        index=builder.option_index(stages, str(seed.get("from_stage", "") or "")),
        key=f"{key_prefix}_funnel_from",
        help=config_help.field_help("metric.from_stage"),
    )
    to_stage_default = str(seed.get("to_stage", "") or "")
    to_stage = st.selectbox(
        "To Stage",
        stages,
        index=builder.option_index(stages, to_stage_default)
        if to_stage_default
        else (1 if len(stages) > 1 else 0),
        key=f"{key_prefix}_funnel_to",
        help=config_help.field_help("metric.to_stage"),
    )
    output = st.selectbox(
        "Output",
        list(FUNNEL_OUTPUT_OPTIONS),
        index=builder.option_index(FUNNEL_OUTPUT_OPTIONS, str(seed.get("output", "") or "")),
        key=f"{key_prefix}_funnel_output",
        help=config_help.field_help("metric.funnel_output"),
    )
    return {"from_stage": from_stage, "to_stage": to_stage, "output": output}


__all__ = [
    "CONTINGENCY_TEST_OPTIONS",
    "CURVE_OUTPUT_OPTIONS",
    "FUNNEL_OUTPUT_OPTIONS",
    "LIFECYCLE_OUTPUT_OPTIONS",
    "METRIC_BASE_FIELDS",
    "OUTCOME_PREFERRED_FIELDS",
    "PROCESSOR_GRAIN_OPTIONS",
    "PROCESSOR_KIND_MANAGED_FIELDS",
    "PROCESSOR_KIND_OPTIONS",
    "QUANTILE_ENGINE_OPTIONS",
    "SET_OP_OPTIONS",
    "SNAPSHOT_CADENCE_OPTIONS",
    "SNAPSHOT_KIND_OPTIONS",
    "SUBJECT_PREFERRED_FIELDS",
    "TOPK_ERROR_TYPE_OPTIONS",
    "MetricFormContext",
    "csv_list_field",
    "first_preferred_field",
    "is_simple_formula",
    "metric_kind_fields",
    "processor_kind_fields",
    "select_or_text",
    "stage_names_from_definition",
    "with_current",
]
