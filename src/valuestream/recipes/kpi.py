"""Versioned KPI recipe artifacts and deterministic catalog instantiation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from valuestream.config import model
from valuestream.utils.names import dedupe_strings as _dedupe_strings

_METRIC_ADAPTER = TypeAdapter(model.Metric)
_PROCESSOR_ADAPTER = TypeAdapter(model.Processor)
_STATE_ADAPTER = TypeAdapter(model.StateSpec)
_TILE_ADAPTER = TypeAdapter(model.Tile)
_PLACEHOLDER = re.compile(r"^\$\{([a-z][a-z0-9_]*)\}$")
# Explicit ``states`` replace the kind defaults for these processor kinds, so
# recipe state additions must first pin the defaults to keep them computed.
_REPLACING_STATE_KINDS = frozenset(
    {"binary_outcome", "score_distribution", "snapshot", "entity_set"}
)

RecipeProcessorKind = Literal[
    "binary_outcome",
    "numeric_distribution",
    "score_distribution",
    "entity_lifecycle",
    "entity_set",
    "funnel",
    "snapshot",
]
RecipeStateType = Literal[
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


class _RecipeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RecipeParameter(_RecipeModel):
    """One bounded value selected while a recipe is installed."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str
    description: str = ""
    unit: Literal["number", "percent"] = "number"
    default: float
    minimum: float
    maximum: float
    step: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _range_is_valid(self) -> RecipeParameter:
        values = (self.minimum, self.default, self.maximum, self.step)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("recipe parameter bounds must be finite")
        if self.minimum > self.maximum:
            raise ValueError("recipe parameter minimum cannot exceed maximum")
        if not self.minimum <= self.default <= self.maximum:
            raise ValueError("recipe parameter default must be inside its bounds")
        return self


class RecipeInput(_RecipeModel):
    """One processor capability that must be bound before instantiation."""

    role: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str
    description: str = ""
    source: Literal["state", "stage"] = "state"
    state_types: tuple[RecipeStateType, ...] = ()
    state_attributes: dict[str, str] = Field(default_factory=dict)
    absent_state_attributes: tuple[str, ...] = ()
    requires_where: bool = False
    state_template: dict[str, Any] = Field(default_factory=dict)
    proposed_name: str = Field(default="", pattern=r"^$|^[A-Za-z][A-Za-z0-9_]*$")
    preferred_names: tuple[str, ...] = ()
    preferred_state_types: tuple[RecipeStateType, ...] = ()
    selection: Literal["automatic", "choice", "field_algorithm"] = "choice"
    require_preferred: bool = False
    same_attribute_as: str | None = None
    match_attribute: str | None = None
    different_from: str | None = None

    @model_validator(mode="after")
    def _paired_input_is_complete(self) -> RecipeInput:
        if bool(self.same_attribute_as) != bool(self.match_attribute):
            raise ValueError("same_attribute_as and match_attribute must be configured together")
        if self.source == "stage" and self.selection == "field_algorithm":
            raise ValueError("stage inputs cannot use field_algorithm selection")
        if self.source == "stage" and self.requires_where:
            raise ValueError("stage inputs cannot require a where predicate")
        if self.source == "stage" and self.state_template:
            raise ValueError("stage inputs cannot define a state template")
        if self.proposed_name and not self.state_template:
            raise ValueError("proposed_name requires a state_template")
        if self.require_preferred and not self.preferred_names:
            raise ValueError("require_preferred needs at least one preferred name")
        if not set(self.preferred_state_types) <= set(self.state_types):
            raise ValueError("preferred_state_types must be included in state_types")
        if self.different_from == self.role:
            raise ValueError("different_from cannot reference the same input role")
        return self


class RecipeMethod(_RecipeModel):
    calculation: str
    accuracy: Literal["exact", "approximate", "statistical"]
    algorithm: str
    caveat: str = ""


class RecipeReport(_RecipeModel):
    chart: str = "kpi_card"
    placement: Literal["content", "kpi_strip"] = "kpi_strip"
    kpi: model.KpiSpec | None = None


class KpiRecipe(_RecipeModel):
    """An inert business definition plus a validated metric template."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    version: int = Field(ge=1)
    title: str
    domain: str
    summary: str
    business_questions: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    maturity: Literal["draft", "reviewed", "certified"] = "reviewed"
    processor_kinds: tuple[RecipeProcessorKind, ...]
    parameters: tuple[RecipeParameter, ...] = ()
    inputs: tuple[RecipeInput, ...] = ()
    default_metric_id: str
    metric: model.Metric
    method: RecipeMethod
    report: RecipeReport = Field(default_factory=RecipeReport)

    @model_validator(mode="after")
    def _roles_are_unique_and_referenced(self) -> KpiRecipe:
        roles = [item.role for item in self.inputs]
        if len(roles) != len(set(roles)):
            raise ValueError("recipe input roles must be unique")
        parameter_names = [item.name for item in self.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("recipe parameter names must be unique")
        parameter_defaults = {item.name: item.default for item in self.parameters}
        role_set = set(roles)
        for item in self.inputs:
            if item.same_attribute_as and item.same_attribute_as not in role_set:
                raise ValueError(
                    f"input {item.role!r} references unknown role {item.same_attribute_as!r}"
                )
            if item.different_from and item.different_from not in role_set:
                raise ValueError(
                    f"input {item.role!r} references unknown role {item.different_from!r}"
                )
            template_parameters = _placeholder_names(item.state_template)
            if unknown := template_parameters - set(parameter_names):
                raise ValueError(
                    f"input {item.role!r} state template references unknown parameter(s): "
                    f"{', '.join(sorted(unknown))}"
                )
            if item.state_template:
                state = _STATE_ADAPTER.validate_python(
                    _substitute(item.state_template, parameter_defaults)
                )
                if item.state_types and state.type not in item.state_types:
                    raise ValueError(
                        f"input {item.role!r} state template type {state.type!r} is not accepted"
                    )
                if item.requires_where and getattr(state, "where", None) is None:
                    raise ValueError(
                        f"input {item.role!r} state template requires a where predicate"
                    )
        template = _metric_template(self)
        placeholders = _placeholder_names(template)
        allowed = {"processor_id", "metric_id", "entity_column", *role_set}
        if unknown := placeholders - allowed:
            raise ValueError(
                f"metric template references unknown placeholder(s): {', '.join(sorted(unknown))}"
            )
        if unused := role_set - placeholders:
            raise ValueError(f"recipe input role(s) are unused: {', '.join(sorted(unused))}")
        sample_bindings = dict.fromkeys(allowed, "bound")
        _METRIC_ADAPTER.validate_python(_substitute(template, sample_bindings))
        tile_sample: dict[str, Any] = {
            "id": "recipe_tile",
            "title": self.title,
            "metric": "recipe_metric",
            "chart": self.report.chart,
            "placement": self.report.placement,
            "kpi": self.report.kpi or None,
        }
        _TILE_ADAPTER.validate_python(tile_sample)
        return self


class KpiRecipeLibrary(_RecipeModel):
    schema_version: int = Field(default=1, ge=1)
    recipes: tuple[KpiRecipe, ...]

    @model_validator(mode="after")
    def _recipe_ids_are_unique(self) -> KpiRecipeLibrary:
        ids = [recipe.id for recipe in self.recipes]
        if len(ids) != len(set(ids)):
            raise ValueError("recipe ids must be unique")
        return self


class RecipeReadiness(_RecipeModel):
    recipe_id: str
    processor_id: str
    status: Literal["ready", "mapping_required", "backfill_required", "incompatible"]
    input_options: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    resolved_inputs: dict[str, str] = Field(default_factory=dict)
    messages: tuple[str, ...] = ()


class RecipeBindingOption(_RecipeModel):
    """Business-facing description of one internal state or stage binding."""

    value: str
    label: str
    field: str = ""
    scope: str = ""
    algorithm: str
    state_type: str = ""
    technical_detail: str
    configured: bool = True
    state_definition: dict[str, Any] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def load_builtin_kpi_recipes() -> KpiRecipeLibrary:
    """Load and validate the packaged KPI recipe library once per process."""

    recipe_path = resources.files("valuestream.recipes").joinpath("kpis.yaml")
    payload = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    return KpiRecipeLibrary.model_validate(payload)


def resolve_recipe_parameters(
    recipe: KpiRecipe,
    values: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Resolve defaults and validate one recipe's editable installation parameters."""

    provided = dict(values or {})
    known = {item.name for item in recipe.parameters}
    if unknown := set(provided) - known:
        raise ValueError(f"unknown recipe parameter(s): {', '.join(sorted(unknown))}")
    resolved: dict[str, float] = {}
    for item in recipe.parameters:
        value = float(provided.get(item.name, item.default))
        if not math.isfinite(value):
            raise ValueError(f"{item.label} must be finite")
        if not item.minimum <= value <= item.maximum:
            raise ValueError(
                f"{item.label} must be between {item.minimum:g} and {item.maximum:g}"
            )
        resolved[item.name] = value
    return resolved


def recipe_readiness(
    recipe: KpiRecipe,
    processor: model.Processor,
    *,
    parameter_values: Mapping[str, float] | None = None,
) -> RecipeReadiness:
    """Resolve unambiguous inputs and describe remaining authoring work."""

    if processor.kind not in recipe.processor_kinds:
        return RecipeReadiness(
            recipe_id=recipe.id,
            processor_id=processor.id,
            status="incompatible",
            messages=(
                f"Requires {', '.join(recipe.processor_kinds)}; {processor.id} is "
                f"{processor.kind}.",
            ),
        )

    resolved_parameters = resolve_recipe_parameters(recipe, parameter_values)
    options: dict[str, tuple[str, ...]] = {}
    resolved: dict[str, str] = {}
    messages: list[str] = []
    missing = False
    ambiguous = False
    for item in recipe.inputs:
        choices = tuple(_input_options(item, processor, resolved_parameters))
        preferred = _preferred_choice(item.preferred_names, choices)
        if item.require_preferred:
            choices = (preferred,) if preferred else ()
        options[item.role] = choices
        if not choices:
            missing = True
            messages.append(_missing_input_message(item))
            continue
        if preferred:
            resolved[item.role] = preferred
        elif preferred_type := _preferred_state_type_choice(item, processor, choices):
            resolved[item.role] = preferred_type
        elif len(choices) == 1:
            resolved[item.role] = choices[0]
        else:
            ambiguous = True
            messages.append(f"Choose {item.label.casefold()} from {len(choices)} candidates.")

    status: Literal["ready", "mapping_required", "backfill_required", "incompatible"]
    if missing:
        status = "backfill_required"
    elif ambiguous:
        status = "mapping_required"
    else:
        status = "ready"
    return RecipeReadiness(
        recipe_id=recipe.id,
        processor_id=processor.id,
        status=status,
        input_options=options,
        resolved_inputs=resolved,
        messages=tuple(messages),
    )


def _missing_input_message(item: RecipeInput) -> str:
    if item.require_preferred and item.preferred_names:
        names = " or ".join(repr(name) for name in item.preferred_names)
        predicate = " with a where predicate" if item.requires_where else ""
        return (
            f"{item.label} requires a processor state named {names}{predicate}; add or "
            "rename a matching state, then backfill aggregates."
        )
    return f"{item.label} is not available and requires aggregate backfill."


def instantiate_metric(
    recipe: KpiRecipe,
    processor: model.Processor,
    metric_id: str,
    bindings: dict[str, str],
    *,
    parameter_values: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Materialize and validate one ordinary ``metrics.yaml`` definition."""

    resolved_parameters = resolve_recipe_parameters(recipe, parameter_values)
    readiness = recipe_readiness(
        recipe,
        processor,
        parameter_values=resolved_parameters,
    )
    if readiness.status == "incompatible":
        raise ValueError(readiness.messages[0])
    expected_roles = {item.role for item in recipe.inputs}
    missing_roles = expected_roles - set(bindings)
    if missing_roles:
        raise ValueError(f"missing recipe input binding(s): {', '.join(sorted(missing_roles))}")
    for item in recipe.inputs:
        selected = bindings[item.role]
        if selected not in readiness.input_options.get(item.role, ()):
            raise ValueError(f"{selected!r} is not a valid binding for {item.label}")
    _validate_paired_bindings(recipe, processor, bindings)

    entity_column = (
        processor.keys.customer_id
        if isinstance(processor, model.EntityLifecycleProcessor)
        else ""
    )
    values = {
        "processor_id": processor.id,
        "metric_id": metric_id,
        "entity_column": entity_column,
        **bindings,
    }
    metric_def = _substitute(_metric_template(recipe), values)
    if not isinstance(metric_def, dict):  # pragma: no cover - schema guards this shape
        raise TypeError("recipe metric template must materialize to a mapping")
    display = dict(metric_def.get("display") or {})
    display["label"] = _metric_label_from_id(metric_id)
    metric_def["display"] = display
    metric_def["recipe"] = {
        "id": recipe.id,
        "version": recipe.version,
        **({"parameters": resolved_parameters} if resolved_parameters else {}),
    }
    validated = _METRIC_ADAPTER.validate_python(metric_def)
    return validated.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
        exclude_defaults=True,
    )


def instantiate_tile(
    recipe: KpiRecipe,
    processor: model.Processor,
    metric_id: str,
    tile_id: str,
    bindings: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Materialize and validate a recommended report tile for a recipe metric."""

    del processor, bindings
    value_format = recipe.metric.display.value_format if recipe.metric.display else None
    raw: dict[str, Any] = {
        "id": tile_id,
        "title": recipe.title,
        "metric": metric_id,
        "description": recipe.summary,
        "chart": recipe.report.chart,
        "placement": recipe.report.placement,
    }
    if value_format:
        raw["value_format"] = value_format
    if recipe.report.kpi:
        raw["kpi"] = recipe.report.kpi
    return _TILE_ADAPTER.validate_python(raw).model_dump(
        mode="json", exclude_none=True, exclude_defaults=True
    )


def unique_artifact_id(preferred: str, existing: set[str]) -> str:
    """Return a deterministic unused catalog id without changing its base spelling."""

    if preferred not in existing:
        return preferred
    index = 2
    while f"{preferred}_{index}" in existing:
        index += 1
    return f"{preferred}_{index}"


def _metric_label_from_id(metric_id: str) -> str:
    """Derive the default recipe-created display name from its stable ID."""

    label = re.sub(r"[_-]+", " ", metric_id).strip()
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label)
    label = re.sub(r"\s+", " ", label)
    if label.startswith("VS "):
        label = label[3:]
    return label[:1].upper() + label[1:] if label else metric_id


def recipe_binding_options(
    item: RecipeInput,
    processor: model.Processor,
    values: tuple[str, ...] | list[str] | None = None,
    *,
    proposal_fields: tuple[str, ...] | list[str] = (),
    parameter_values: Mapping[str, float] | None = None,
) -> list[RecipeBindingOption]:
    """Describe configured and safely configurable field/algorithm bindings."""

    resolved_parameters = dict(parameter_values or {})
    choices = (
        list(values)
        if values is not None
        else _input_options(item, processor, resolved_parameters)
    )
    if item.source == "stage":
        return [
            RecipeBindingOption(
                value=choice,
                label=choice,
                field=choice,
                algorithm="Funnel stage",
                technical_detail=f"Configured funnel stage: {choice}",
            )
            for choice in choices
        ]

    states = model.effective_processor_states(processor)
    out: list[RecipeBindingOption] = []
    for choice in choices:
        spec = states.get(choice)
        if spec is None:
            continue
        field = _state_business_field(processor, choice, spec)
        scope = _state_business_scope(processor, choice)
        algorithm = recipe_algorithm_label(spec.type)
        subject = field or _state_business_label(choice)
        parameters = _state_parameter_summary(spec)
        technical = f"Existing aggregate state: {choice} · {spec.type}"
        if parameters:
            technical += f" · {parameters}"
        out.append(
            RecipeBindingOption(
                value=choice,
                label=" · ".join(value for value in (subject, scope, algorithm) if value),
                field=field,
                scope=scope,
                algorithm=algorithm,
                state_type=spec.type,
                technical_detail=technical,
            )
        )

    if item.state_template:
        if out:
            return out
        state_definition = _state_template_definition(item, resolved_parameters)
        spec = _STATE_ADAPTER.validate_python(state_definition)
        existing = set(model.effective_processor_states(processor)) | set(processor.states)
        state_name = unique_artifact_id(
            item.proposed_name or f"{item.role}_{spec.type}",
            existing,
        )
        field = _state_business_field(processor, state_name, spec)
        parameters = _state_parameter_summary(spec)
        technical = f"Proposed aggregate state: {state_name} · {spec.type}"
        if parameters:
            technical += f" · {parameters}"
        return [
            RecipeBindingOption(
                value=state_name,
                label=" · ".join(
                    value for value in (field, recipe_algorithm_label(spec.type)) if value
                ),
                field=field,
                algorithm=recipe_algorithm_label(spec.type),
                state_type=spec.type,
                technical_detail=technical,
                configured=False,
                state_definition=state_definition,
            )
        ]

    if item.source != "state" or not proposal_fields or item.requires_where:
        return out

    fields = _dedupe_strings([*proposal_fields, *processor_recipe_fields(processor)])
    for field in fields:
        for state_type in item.state_types:
            if any(option.field == field and option.state_type == state_type for option in out):
                continue
            state_name = _proposed_state_name(
                processor,
                field=field,
                state_type=state_type,
                attributes=item.state_attributes,
            )
            state_definition = _proposed_state_definition(item, field, state_type)
            spec = _STATE_ADAPTER.validate_python(state_definition)
            parameters = _state_parameter_summary(spec)
            technical = f"Proposed aggregate state: {state_name} · {state_type}"
            if parameters:
                technical += f" · {parameters}"
            out.append(
                RecipeBindingOption(
                    value=state_name,
                    label=" · ".join(
                        value for value in (field, recipe_algorithm_label(state_type)) if value
                    ),
                    field=field,
                    algorithm=recipe_algorithm_label(state_type),
                    state_type=state_type,
                    technical_detail=technical,
                    configured=False,
                    state_definition=state_definition,
                )
            )
    field_order = {field: index for index, field in enumerate(fields)}
    type_order = {state_type: index for index, state_type in enumerate(item.state_types)}
    return sorted(
        out,
        key=lambda option: (
            field_order.get(option.field, len(field_order)),
            type_order.get(option.state_type, len(type_order)),
            not option.configured,
            option.value.casefold(),
        ),
    )


def processor_recipe_fields(processor: model.Processor) -> list[str]:
    """Return deterministic processor-owned fields eligible for new recipe states."""

    fields = [*processor.group_by, processor.time.property]
    if isinstance(processor, model.BinaryOutcomeProcessor):
        fields.extend([processor.outcome.column, *processor.dedup_keys])
        fields.extend(filter(None, [processor.variant_column]))
        if processor.entities:
            fields.append(processor.entities.subject)
    elif isinstance(processor, model.NumericDistributionProcessor):
        fields.extend(processor.properties)
    elif isinstance(processor, model.ScoreDistributionProcessor):
        fields.extend([item.column for item in processor.score_properties])
        fields.extend([processor.outcome.column, *processor.dedup_keys])
        if processor.entities:
            fields.append(processor.entities.subject)
    elif isinstance(processor, model.EntityLifecycleProcessor):
        fields.extend(processor.keys.model_dump().values())
    elif isinstance(processor, model.EntitySetProcessor):
        fields.append(processor.entity)
        if processor.entities:
            fields.append(processor.entities.subject)
    elif isinstance(processor, model.FunnelProcessor):
        fields.extend(filter(None, [processor.entity]))
    elif isinstance(processor, model.SnapshotProcessor):
        fields.extend(filter(None, [processor.entity, processor.as_of_property]))
        fields.extend(item.property for item in processor.milestones)
    for name, spec in model.effective_processor_states(processor).items():
        field = _state_business_field(processor, name, spec)
        if field:
            fields.append(field)
    return _dedupe_strings(fields)


def processor_with_recipe_states(
    processor: model.Processor,
    state_additions: dict[str, dict[str, Any]],
) -> model.Processor:
    """Return a validated processor with proposed recipe states configured."""

    if not state_additions:
        return processor
    data = processor.model_dump(mode="python", by_alias=True, exclude_none=True)
    configured_states = {
        name: spec.model_dump(mode="python", by_alias=True, exclude_none=True)
        for name, spec in processor.states.items()
    }
    if not configured_states and processor.kind in _REPLACING_STATE_KINDS:
        configured_states = {
            name: spec.model_dump(mode="python", by_alias=True, exclude_none=True)
            for name, spec in model.effective_processor_states(processor).items()
        }
    for name, definition in state_additions.items():
        configured_states[name] = _STATE_ADAPTER.validate_python(definition).model_dump(
            mode="python",
            by_alias=True,
            exclude_none=True,
        )
    data["states"] = configured_states
    return _PROCESSOR_ADAPTER.validate_python(data)


def _proposed_state_definition(
    item: RecipeInput,
    field: str,
    state_type: str,
) -> dict[str, Any]:
    definition: dict[str, Any] = {
        "type": state_type,
        "source_column": field,
        **model.DEFAULT_STATE_PARAMETERS.get(state_type, {}),
        **item.state_attributes,
    }
    if item.state_attributes.get("outcome") and "score_property" not in definition:
        definition["score_property"] = field
    return definition


def _proposed_state_name(
    processor: model.Processor,
    *,
    field: str,
    state_type: str,
    attributes: dict[str, str],
) -> str:
    field_fragment = re.sub(r"[^A-Za-z0-9]+", "_", field).strip("_") or "Field"
    if field_fragment[0].isdigit():
        field_fragment = f"Field_{field_fragment}"
    outcome = attributes.get("outcome", "")
    outcome_suffix = {"positive": "positives", "negative": "negatives"}.get(
        outcome.casefold(), outcome
    )
    base = f"{field_fragment}_{state_type}"
    if outcome_suffix:
        base += f"_{outcome_suffix}"
    existing = set(model.effective_processor_states(processor)) | set(processor.states)
    return unique_artifact_id(base, existing)


def _input_options(
    item: RecipeInput,
    processor: model.Processor,
    parameter_values: Mapping[str, float] | None = None,
) -> list[str]:
    if item.source == "stage":
        if not isinstance(processor, model.FunnelProcessor):
            return []
        by_stage = {
            state.stage: name
            for name, state in processor.states.items()
            if isinstance(state, model.CountState) and state.stage
        }
        return [
            by_stage[stage.name]
            for stage in processor.stages
            if stage.name in by_stage
        ]

    target_definition = (
        _state_template_definition(item, dict(parameter_values or {}))
        if item.state_template
        else None
    )
    out: list[str] = []
    wanted_types = set(item.state_types)
    for name, spec in model.effective_processor_states(processor).items():
        if wanted_types and spec.type not in wanted_types:
            continue
        if target_definition is not None:
            if _normalized_state_definition(spec) == target_definition:
                out.append(name)
            continue
        if not _state_attributes_match(
            spec,
            item.state_attributes,
            item.absent_state_attributes,
        ):
            continue
        if item.requires_where and getattr(spec, "where", None) is None:
            continue
        out.append(name)
    return out


def _state_template_definition(
    item: RecipeInput,
    parameter_values: Mapping[str, float],
) -> dict[str, Any]:
    definition = _substitute(item.state_template, dict(parameter_values))
    spec = _STATE_ADAPTER.validate_python(definition)
    return _normalized_state_definition(spec)


def _normalized_state_definition(spec: model.StateSpec) -> dict[str, Any]:
    return spec.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
        exclude_defaults=True,
    )


def _state_attributes_match(
    spec: model.StateSpec,
    expected: dict[str, str],
    absent: tuple[str, ...],
) -> bool:
    values = spec.model_dump(mode="python", exclude_none=True)
    return not any(key in values for key in absent) and all(
        str(values.get(key, "")).casefold() == value.casefold() for key, value in expected.items()
    )


def _preferred_choice(preferred_names: tuple[str, ...], choices: tuple[str, ...]) -> str:
    by_casefold = {choice.casefold(): choice for choice in choices}
    for name in preferred_names:
        if match := by_casefold.get(name.casefold()):
            return match
    return ""


def _preferred_state_type_choice(
    item: RecipeInput,
    processor: model.Processor,
    choices: tuple[str, ...],
) -> str:
    if not item.preferred_state_types or item.source != "state":
        return ""
    states = model.effective_processor_states(processor)
    for state_type in item.preferred_state_types:
        matching = [choice for choice in choices if states[choice].type == state_type]
        if len(matching) == 1:
            return matching[0]
    return ""


def _validate_paired_bindings(
    recipe: KpiRecipe,
    processor: model.Processor,
    bindings: dict[str, str],
) -> None:
    for item in recipe.inputs:
        if item.different_from and bindings[item.role] == bindings[item.different_from]:
            raise ValueError(
                f"{item.label} must be different from {item.different_from.replace('_', ' ')}"
            )
        if not item.same_attribute_as or not item.match_attribute:
            continue
        left_value = recipe_binding_attribute(processor, bindings[item.role], item.match_attribute)
        right_value = recipe_binding_attribute(
            processor, bindings[item.same_attribute_as], item.match_attribute
        )
        if left_value and right_value and left_value != right_value:
            raise ValueError(
                f"{item.label} must use the same {item.match_attribute.replace('_', ' ')} "
                f"as {item.same_attribute_as.replace('_', ' ')}"
            )


def recipe_binding_attribute(
    processor: model.Processor,
    state_name: str,
    attribute: str,
) -> str:
    """Return normalized state metadata used to pair related recipe inputs."""

    spec = model.effective_processor_states(processor).get(state_name)
    if spec is None:
        return ""
    if attribute == "type":
        return spec.type
    if attribute == "score_property":
        return str(
            getattr(spec, "score_property", None) or getattr(spec, "source_column", None) or ""
        )
    return str(getattr(spec, attribute, "") or "")


def _state_business_field(
    processor: model.Processor,
    state_name: str,
    spec: model.StateSpec,
) -> str:
    for key in ("source_column", "score_property"):
        if value := getattr(spec, key, None):
            return str(value)
    return ""


def _state_business_scope(
    processor: model.Processor,
    state_name: str,
) -> str:
    del processor, state_name
    return ""


def _state_business_label(state_name: str) -> str:
    semantic_names = {
        "count": "All observations",
        "positives": "Positive outcomes",
        "negatives": "Negative outcomes",
    }
    if label := semantic_names.get(state_name.casefold()):
        return label
    return state_name.replace("_", " ")


def recipe_algorithm_label(state_type: str) -> str:
    """Return the business-facing name for one persisted state algorithm."""

    return {
        "count": "Exact count",
        "value_sum": "Exact sum",
        "min": "Exact minimum",
        "max": "Exact maximum",
        "pooled_mean": "Pooled mean",
        "pooled_variance": "Pooled variance",
        "tdigest": "t-digest",
        "kll": "KLL",
        "cpc": "CPC",
        "hll": "HLL",
        "theta": "Theta",
        "topk": "Frequent items",
    }.get(state_type, state_type.replace("_", " ").title())


def _state_parameter_summary(spec: model.StateSpec) -> str:
    values = {
        key: getattr(spec, key)
        for key in ("lg_k", "k", "lg_max_map_size")
        if getattr(spec, key, None) is not None
    }
    parts = [
        f"{key}={values[key]}"
        for key in ("lg_k", "k", "lg_max_map_size")
        if key in values
    ]
    where = getattr(spec, "where", None)
    if where is not None:
        rendered = yaml.safe_dump(
            where.model_dump(mode="json", by_alias=True, exclude_none=True),
            default_flow_style=True,
            sort_keys=False,
            width=1000,
        ).strip()
        parts.append(f"where={rendered}")
    return " · ".join(parts)


def _substitute(value: Any, bindings: Mapping[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _substitute(item, bindings) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, bindings) for item in value]
    if isinstance(value, str) and (match := _PLACEHOLDER.fullmatch(value)):
        key = match.group(1)
        if key not in bindings:
            raise ValueError(f"recipe template references unbound value {key!r}")
        return bindings[key]
    return value


def _placeholder_names(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(_placeholder_names(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_placeholder_names(item) for item in value), set())
    if isinstance(value, str) and (match := _PLACEHOLDER.fullmatch(value)):
        return {match.group(1)}
    return set()


def _metric_template(recipe: KpiRecipe) -> dict[str, Any]:
    return recipe.metric.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
        exclude_defaults=True,
    )


__all__ = [
    "KpiRecipe",
    "KpiRecipeLibrary",
    "RecipeBindingOption",
    "RecipeInput",
    "RecipeParameter",
    "RecipeReadiness",
    "instantiate_metric",
    "instantiate_tile",
    "load_builtin_kpi_recipes",
    "processor_recipe_fields",
    "processor_with_recipe_states",
    "recipe_algorithm_label",
    "recipe_binding_attribute",
    "recipe_binding_options",
    "recipe_readiness",
    "resolve_recipe_parameters",
    "unique_artifact_id",
]
