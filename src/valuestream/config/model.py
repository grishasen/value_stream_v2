"""Typed model for the catalog YAML files.

Maps to the shapes in ``REPLACEMENT_DESIGN.md`` §7 and the per-processor
schemas in ``docs/reference/processors.md``. Each top-level YAML file
(``pipelines.yaml``, ``processors.yaml``, ``metrics.yaml``, and
``dashboards.yaml``) is one Pydantic model; the :class:`Catalog` aggregates
all four for a workspace.

Phase 0's goal is structural validation: the loader can confirm that a
YAML file matches its shape, that ``col`` references in expressions
resolve, and that every Tile binds to a known Metric. Per-kind processor
algorithms ship in Phase 1.

Discriminated unions:
* :class:`Transform` — 11 built-in transform kinds (``rename_capitalize``,
  ``parse_datetime``, ``derive_calendar``, ``derive_action_id``,
  ``derive_column``, ``filter``, ``dedup``, ``defaults``, ``cast``,
  ``drop_columns``, ``coalesce``).
* :class:`Reader` — 4 built-in reader kinds.
* :class:`Processor` — 7 built-in processor kinds.
* :class:`Metric` — 9 built-in metric kinds.

Kind-specific fields not enumerated here are accepted via
``model_config(extra='allow')`` so that Phase-1 processor implementations
can tighten their own model without breaking Phase 0 validators.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from valuestream.expr.ast import Dtype, Expr
from valuestream.utils.names import dedupe_strings as _dedupe

# ---------------------------------------------------------------------------
# Common building blocks.
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    """Base for strictly-typed config models — forbids unknown fields."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class _PermissiveModel(BaseModel):
    """Base for kind-specialized config models — allows extra fields.

    Kept permissive at Phase 0 because per-kind processors and metrics
    grow extra config knobs in Phase 1+. Tightening this is a per-kind
    concern when those processors land.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ---------------------------------------------------------------------------
# pipelines.yaml
# ---------------------------------------------------------------------------


class Calendar(_StrictModel):
    grains: list[str] = Field(
        default_factory=lambda: ["Day", "Month", "Quarter", "Year", "Summary"]
    )
    week_start: Literal["monday", "sunday"] = "monday"


class WorkspaceDefaults(_StrictModel):
    time_zone: str = "UTC"
    calendar: Calendar = Field(default_factory=Calendar)


class SourceSchema(_StrictModel):
    timestamp_column: str | None = None
    natural_key: list[str] = Field(default_factory=list)
    drop_columns: list[str] = Field(default_factory=list)


# Reader kinds — discriminated on ``kind``.
class _ReaderBase(_PermissiveModel):
    kind: str
    file_pattern: str
    group_by_filename: str | None = None
    streaming: bool = False


class PegaDsExportReader(_ReaderBase):
    kind: Literal["pega_ds_export"]


class ParquetReader(_ReaderBase):
    kind: Literal["parquet"]


class CsvReader(_ReaderBase):
    kind: Literal["csv"]
    delimiter: str | None = None


class XlsxReader(_ReaderBase):
    kind: Literal["xlsx"]
    sheet: str | int | None = None


Reader = Annotated[
    PegaDsExportReader | ParquetReader | CsvReader | XlsxReader,
    Field(discriminator="kind"),
]


# Transform kinds — discriminated on ``kind``.
class _TransformBase(_StrictModel):
    kind: str


class RenameCapitalize(_TransformBase):
    kind: Literal["rename_capitalize"]


class ParseDatetime(_TransformBase):
    kind: Literal["parse_datetime"]
    columns: list[str]
    format: str


class DeriveCalendar(_TransformBase):
    kind: Literal["derive_calendar"]
    from_: str = Field(alias="from")
    outputs: list[str] = Field(default_factory=lambda: ["Day", "Month", "Year", "Quarter"])


class DeriveActionId(_TransformBase):
    kind: Literal["derive_action_id"]
    parts: list[str]
    sep: str = "/"


class DeriveColumn(_TransformBase):
    kind: Literal["derive_column"]
    output: str
    expression: Expr


class FilterTransform(_TransformBase):
    kind: Literal["filter"]
    expression: Expr


class Dedup(_TransformBase):
    kind: Literal["dedup"]
    keys: list[str]


class Defaults(_TransformBase):
    kind: Literal["defaults"]
    values: dict[str, Any]


class Cast(_TransformBase):
    kind: Literal["cast"]
    columns: dict[str, Dtype]


class DropColumns(_TransformBase):
    kind: Literal["drop_columns"]
    columns: list[str]


class Coalesce(_TransformBase):
    kind: Literal["coalesce"]
    output: str
    columns: list[str]


Transform = Annotated[
    RenameCapitalize
    | ParseDatetime
    | DeriveCalendar
    | DeriveActionId
    | DeriveColumn
    | FilterTransform
    | Dedup
    | Defaults
    | Cast
    | DropColumns
    | Coalesce,
    Field(discriminator="kind"),
]


class Source(_StrictModel):
    id: str
    description: str = ""
    reader: Reader
    schema_: SourceSchema = Field(default_factory=SourceSchema, alias="schema")
    transforms: list[Transform] = Field(default_factory=list)
    defaults: dict[str, Any] = Field(default_factory=dict)
    materialize_transforms: bool = False
    debugging: bool = False


class Pipelines(_StrictModel):
    """Top-level shape for ``pipelines.yaml``."""

    version: int = 1
    workspace: str
    defaults: WorkspaceDefaults = Field(default_factory=WorkspaceDefaults)
    sources: list[Source]


# ---------------------------------------------------------------------------
# processors.yaml
# ---------------------------------------------------------------------------


class StateSpec(_PermissiveModel):
    """A state column on a processor's aggregate output.

    Kind-specific fields (``source_column``, ``lg_k``, ``per_property``,
    etc.) are tracked via ``extra='allow'`` for Phase 0.
    """

    type: Literal[
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


class ProcessorTime(_PermissiveModel):
    """Processor time contract.

    ``column`` is the source timestamp used to derive calendar fields.
    ``grains`` are user-facing calendar grains. They are normalized to the
    internal aggregate names on the parent processor. ``aggregation_levels``
    controls the physical row level stored for each grain.
    """

    column: str | None = None
    grains: list[str] = Field(default_factory=lambda: ["Day", "Month", "Summary"])
    aggregation_levels: dict[str, str] = Field(
        default_factory=lambda: {
            "Day": "Day",
            "Month": "Month",
            "Summary": "Month",
        }
    )

    @model_validator(mode="after")
    def _validate_aggregation_levels(self) -> ProcessorTime:
        normalize_aggregation_levels(self.aggregation_levels)
        return self


_GRAIN_ALIASES: dict[str, str] = {
    "hour": "hourly",
    "hourly": "hourly",
    "day": "daily",
    "daily": "daily",
    "week": "weekly",
    "weekly": "weekly",
    "month": "monthly",
    "monthly": "monthly",
    "quarter": "quarterly",
    "quarterly": "quarterly",
    "year": "yearly",
    "yearly": "yearly",
    "summary": "summary",
    "all": "summary",
    "all_time": "summary",
    "all-time": "summary",
}

_DEFAULT_AGGREGATION_LEVELS: dict[str, str] = {
    "daily": "daily",
    "monthly": "monthly",
    "summary": "monthly",
}

_ALLOWED_AGGREGATION_LEVELS: dict[str, set[str]] = {
    "daily": {"hourly", "daily"},
    "monthly": {"daily", "weekly", "monthly"},
    "summary": {"monthly", "quarterly", "yearly"},
}


def normalize_grain_name(grain: str) -> str:
    """Normalize public grain names to the physical aggregate identifier."""

    key = grain.strip().replace(" ", "_").lower()
    return _GRAIN_ALIASES.get(key, key)


def normalize_grains(grains: list[str]) -> list[str]:
    """Normalize grain names while preserving order and removing duplicates."""

    out: list[str] = []
    for grain in grains:
        normalized = normalize_grain_name(str(grain))
        if normalized not in out:
            out.append(normalized)
    return out


def normalize_aggregation_levels(levels: dict[str, str] | None) -> dict[str, str]:
    """Normalize and fill physical aggregation levels for logical grains."""

    out = dict(_DEFAULT_AGGREGATION_LEVELS)
    for raw_grain, raw_level in (levels or {}).items():
        grain = normalize_grain_name(str(raw_grain))
        level = normalize_grain_name(str(raw_level))
        allowed = _ALLOWED_AGGREGATION_LEVELS.get(grain)
        if allowed is not None and level not in allowed:
            allowed_values = ", ".join(sorted(allowed))
            raise ValueError(
                f"aggregation level {raw_level!r} is not valid for grain "
                f"{raw_grain!r}; expected one of: {allowed_values}"
            )
        out[grain] = level
    return out


class _ProcessorBase(_PermissiveModel):
    id: str
    source: str
    kind: str
    description: str = ""
    group_by: list[str] = Field(default_factory=list)
    time: ProcessorTime | None = None
    states: dict[str, StateSpec] = Field(default_factory=dict)
    filter: Expr | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_authoring_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            dimensions = data.pop("dimensions", None)
            if dimensions is not None and "group_by" not in data:
                data["group_by"] = dimensions
            if data.get("kind") == "score_distribution":
                score_alias = data.pop("scores", None)
                if score_alias is not None and "score_columns" not in data:
                    data["score_columns"] = score_alias
                if "score_properties" not in data:
                    score_properties = _score_properties_from_legacy(data.get("score_columns"))
                    if score_properties:
                        data["score_properties"] = score_properties
            legacy = {"extra_dimensions", "grains"} & set(data)
            if legacy:
                names = ", ".join(sorted(legacy))
                raise ValueError(
                    f"legacy processor field(s) are not supported: {names}; "
                    "use dimensions and time.grains"
                )
        return data

    @property
    def grains(self) -> list[str]:
        raw_grains = self.time.grains if self.time is not None else ["Day", "Month", "Summary"]
        return normalize_grains([str(grain) for grain in raw_grains])

    @property
    def aggregation_levels(self) -> dict[str, str]:
        raw = self.time.aggregation_levels if self.time is not None else None
        return normalize_aggregation_levels(raw)

    def aggregation_level_for(self, grain: str) -> str:
        normalized = normalize_grain_name(grain)
        return self.aggregation_levels.get(normalized, normalized)


class BinaryOutcomeProcessor(_ProcessorBase):
    kind: Literal["binary_outcome"]


class NumericDistributionProcessor(_ProcessorBase):
    kind: Literal["numeric_distribution"]
    sketch_build_mode: Literal["legacy", "bulk"] = "legacy"


class ScoreDistributionProcessor(_ProcessorBase):
    kind: Literal["score_distribution"]
    sketch_build_mode: Literal["legacy", "bulk"] = "legacy"


class EntityLifecycleProcessor(_ProcessorBase):
    kind: Literal["entity_lifecycle"]


class EntitySetProcessor(_ProcessorBase):
    kind: Literal["entity_set"]


class FunnelProcessor(_ProcessorBase):
    kind: Literal["funnel"]


class SnapshotProcessor(_ProcessorBase):
    kind: Literal["snapshot"]
    snapshot_kind: Literal["periodic", "accumulating"]
    cadence: Literal["daily", "weekly", "monthly"] | None = None


Processor = Annotated[
    BinaryOutcomeProcessor
    | NumericDistributionProcessor
    | ScoreDistributionProcessor
    | EntityLifecycleProcessor
    | EntitySetProcessor
    | FunnelProcessor
    | SnapshotProcessor,
    Field(discriminator="kind"),
]


class Processors(_StrictModel):
    """Top-level shape for ``processors.yaml``."""

    processors: list[Processor]


DEFAULT_STATE_PARAMETERS: dict[str, dict[str, int]] = {
    "cpc": {"lg_k": 11},
    "hll": {"lg_k": 12},
    "theta": {"lg_k": 12},
    "tdigest": {"k": 500},
    "kll": {"k": 200},
    "topk": {"lg_max_map_size": 10},
}


def effective_processor_states(processor: Processor) -> dict[str, StateSpec]:
    """Return the state columns the ingestion engine computes for a processor.

    Single source of truth shared by the engine processors, catalog
    validation, the KPI recipe library, and the UI. Explicit ``states``
    replace the kind defaults for ``binary_outcome``, ``score_distribution``,
    ``snapshot``, and ``entity_set`` processors, and merge with the derived
    states for ``numeric_distribution``, ``entity_lifecycle``, and ``funnel``
    processors.
    """
    if isinstance(processor, NumericDistributionProcessor):
        return _numeric_distribution_states(processor)
    if isinstance(processor, EntityLifecycleProcessor):
        return _entity_lifecycle_states(processor)
    if isinstance(processor, FunnelProcessor):
        return _funnel_states(processor)
    if processor.states:
        return processor.states
    return _default_kind_states(processor)


def _default_kind_states(processor: Processor) -> dict[str, StateSpec]:
    if isinstance(processor, BinaryOutcomeProcessor):
        states = {
            "Count": StateSpec.model_validate({"type": "count"}),
            "Positives": StateSpec.model_validate({"type": "count"}),
            "Negatives": StateSpec.model_validate({"type": "count"}),
        }
        subject = _entity_subject(processor)
        if subject:
            states["UniqueSubjects_cpc"] = _sketch_state("cpc", subject)
        return states
    if isinstance(processor, ScoreDistributionProcessor):
        return _score_distribution_states(processor)
    if isinstance(processor, SnapshotProcessor):
        return {"Count": StateSpec.model_validate({"type": "count"})}
    if isinstance(processor, EntitySetProcessor):
        entity = entity_set_column(processor)
        return {
            "ActiveUsers_cpc": _sketch_state("cpc", entity),
            "ActiveUsers_theta": _sketch_state("theta", entity),
        }
    return {}


def entity_lifecycle_keys(processor: Processor) -> dict[str, str]:
    """Return the configured lifecycle key columns with engine defaults."""
    raw = _processor_extra(processor).get("keys", {})
    keys = raw if isinstance(raw, dict) else {}
    return {
        "customer_id": str(keys.get("customer_id", "CustomerID")),
        "order_id": str(keys.get("order_id", "OrderID")),
        "monetary": str(keys.get("monetary", "Monetary")),
        "purchase_date": str(keys.get("purchase_date", "PurchaseDate")),
    }


def funnel_stage_names(processor: Processor) -> list[str]:
    """Return configured funnel stage names in stage order."""
    raw = _processor_extra(processor).get("stages", [])
    if not isinstance(raw, list):
        return []
    return [str(item["name"]) for item in raw if isinstance(item, dict) and item.get("name")]


def entity_set_column(processor: Processor) -> str:
    """Return the entity column an entity_set processor sketches by default."""
    return str(_processor_extra(processor).get("entity", "CustomerID"))


def _sketch_state(state_type: str, source_column: str) -> StateSpec:
    return StateSpec.model_validate(
        {
            "type": state_type,
            "source_column": source_column,
            **DEFAULT_STATE_PARAMETERS[state_type],
        }
    )


def _entity_lifecycle_states(processor: EntityLifecycleProcessor) -> dict[str, StateSpec]:
    keys = entity_lifecycle_keys(processor)
    states = {
        "unique_holdings": StateSpec.model_validate(
            {"type": "count", "source_column": keys["order_id"]}
        ),
        "lifetime_value": StateSpec.model_validate(
            {"type": "value_sum", "source_column": keys["monetary"]}
        ),
        "MinPurchasedDate": StateSpec.model_validate(
            {"type": "min", "source_column": keys["purchase_date"]}
        ),
        "MaxPurchasedDate": StateSpec.model_validate(
            {"type": "max", "source_column": keys["purchase_date"]}
        ),
    }
    if not {"UniquePurchasers_cpc", "UniquePurchasers_hll"} & processor.states.keys():
        states["UniquePurchasers_cpc"] = _sketch_state("cpc", keys["customer_id"])
    states.update(processor.states)
    return states


def _funnel_states(processor: FunnelProcessor) -> dict[str, StateSpec]:
    entity = _processor_extra(processor).get("entity")
    states: dict[str, StateSpec] = {}
    for stage_name in funnel_stage_names(processor):
        states[f"{stage_name}_Count"] = StateSpec.model_validate({"type": "count"})
        if entity is None:
            continue
        cardinality_names = {f"{stage_name}_Customers_cpc", f"{stage_name}_Customers_hll"}
        if not cardinality_names & processor.states.keys():
            states[f"{stage_name}_Customers_cpc"] = _sketch_state("cpc", str(entity))
    states.update(processor.states)
    return states


def _numeric_distribution_states(processor: NumericDistributionProcessor) -> dict[str, StateSpec]:
    properties = _extra_list(processor, "properties")
    engine = str(_processor_extra(processor).get("quantile_engine", "tdigest"))
    states: dict[str, StateSpec] = {}
    for prop in properties:
        for name, spec in _numeric_distribution_templates(engine).items():
            states[name.replace("{prop}", prop)] = _expand_property_spec(spec, prop)
    for name, spec in processor.states.items():
        if _uses_property_template(name, spec):
            for prop in properties:
                states[name.replace("{prop}", prop)] = _expand_property_spec(spec, prop)
        else:
            states[name] = spec
    return states


def _numeric_distribution_templates(engine: str) -> dict[str, StateSpec]:
    sketch_type = "kll" if engine == "kll" else "tdigest"
    return {
        "{prop}_Count": StateSpec.model_validate({"type": "count", "per_property": True}),
        "{prop}_Sum": StateSpec.model_validate({"type": "value_sum", "per_property": True}),
        "{prop}_Mean": StateSpec.model_validate(
            {"type": "pooled_mean", "per_property": True, "weight": "{prop}_Count"}
        ),
        "{prop}_Var": StateSpec.model_validate({"type": "pooled_variance", "per_property": True}),
        "{prop}_Min": StateSpec.model_validate({"type": "min", "per_property": True}),
        "{prop}_Max": StateSpec.model_validate({"type": "max", "per_property": True}),
        f"{{prop}}_{sketch_type}": StateSpec.model_validate(
            {"type": sketch_type, "per_property": True}
        ),
    }


def _expand_property_spec(spec: StateSpec, prop: str) -> StateSpec:
    payload = spec.model_dump(by_alias=True)
    for key, value in list(payload.items()):
        if isinstance(value, str):
            payload[key] = value.replace("{prop}", prop)
    return StateSpec.model_validate(payload)


def _uses_property_template(name: str, spec: StateSpec) -> bool:
    if "{prop}" in name:
        return True
    return any(
        isinstance(value, str) and "{prop}" in value
        for value in spec.model_dump(mode="python").values()
    )


def _score_distribution_states(processor: ScoreDistributionProcessor) -> dict[str, StateSpec]:
    properties = _score_properties(processor)
    subject = _entity_subject(processor) or "CustomerID"
    unique_state = "UniqueCustomers_cpc" if subject == "CustomerID" else "UniqueSubjects_cpc"
    states = {
        "Count": StateSpec.model_validate({"type": "count"}),
        "personalization": StateSpec.model_validate({"type": "pooled_mean", "weight": "Count"}),
        "novelty": StateSpec.model_validate({"type": "pooled_mean", "weight": "Count"}),
        unique_state: _sketch_state("cpc", subject),
    }
    for prop in properties:
        for outcome in ("positive", "negative"):
            states[f"{prop}_tdigest_{outcome}s"] = StateSpec.model_validate(
                {
                    "type": "tdigest",
                    "source_column": prop,
                    "outcome": outcome,
                    "score_property": prop,
                    **DEFAULT_STATE_PARAMETERS["tdigest"],
                }
            )
    return states


def _score_properties(processor: ScoreDistributionProcessor) -> list[str]:
    extra = _processor_extra(processor)
    properties = _extra_list(processor, "score_properties")
    if properties:
        return _dedupe(properties)
    legacy = _score_properties_from_legacy(extra.get("score_columns"))
    return legacy or ["Propensity"]


def _score_properties_from_legacy(value: Any) -> list[str]:
    if isinstance(value, dict):
        return _dedupe([str(item) for item in value.values() if str(item).strip()])
    if isinstance(value, list):
        return _dedupe([str(item) for item in value if str(item).strip()])
    return []


def _entity_subject(processor: Processor) -> str:
    entities = _processor_extra(processor).get("entities")
    if isinstance(entities, dict):
        subject = entities.get("subject")
        if subject:
            return str(subject)
    return ""


def _processor_extra(processor: Processor) -> dict[str, Any]:
    return dict(processor.model_extra or {})


def _extra_list(processor: Processor, key: str) -> list[str]:
    raw = _processor_extra(processor).get(key, [])
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


# ---------------------------------------------------------------------------
# metrics.yaml
# ---------------------------------------------------------------------------


class MetricDisplaySpec(_StrictModel):
    """Presentation metadata shared by reports and catalog authoring surfaces."""

    label: str = ""
    unit: str = ""
    value_format: Literal["percent", "integer", "number", "currency"] | None = None
    direction: Literal["higher_is_better", "lower_is_better", "neutral"] = "neutral"


class _MetricBase(_PermissiveModel):
    """Base shape for every derived-metric kind.

    Per docs/reference/processors.md §1, metrics carry ``kind``, a ``source``
    pointing to a Processor, and any kind-specific knobs.
    """

    source: str
    kind: str
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    display: MetricDisplaySpec | None = None


class FormulaMetric(_MetricBase):
    kind: Literal["formula"]
    expression: Expr


class ApproxDistinctCountMetric(_MetricBase):
    kind: Literal["approx_distinct_count"]
    state: str


class TopKItemsMetric(_MetricBase):
    kind: Literal["topk_items"]
    state: str
    limit: int = 10
    error_type: Literal["NO_FALSE_POSITIVES", "NO_FALSE_NEGATIVES"] = "NO_FALSE_POSITIVES"


class TdigestQuantileMetric(_MetricBase):
    kind: Literal["tdigest_quantile"]
    state: str
    quantile: float


class VariantCompareMetric(_MetricBase):
    kind: Literal["variant_compare"]
    variant_column: str
    test_role: str
    control_role: str
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    outputs: list[str] = Field(default_factory=list)


class CurveFromDigestsMetric(_MetricBase):
    kind: Literal["curve_from_digests"]
    positive_state: str
    negative_state: str
    output: Literal["roc_auc", "average_precision"] = "roc_auc"


class CalibrationFromDigestsMetric(_MetricBase):
    kind: Literal["calibration_from_digests"]
    positive_state: str
    negative_state: str


class ContingencyTestMetric(_MetricBase):
    kind: Literal["contingency_test"]
    variant_column: str
    tests: list[Literal["chi2", "g", "z"]]
    outputs: list[str] = Field(default_factory=list)


class ProportionTestMetric(_MetricBase):
    kind: Literal["proportion_test"]
    variant_column: str = "ModelControlGroup"
    test_role: str = "Test"
    control_role: str = "Control"
    outputs: list[str] = Field(default_factory=list)


class LifecycleSummaryMetric(_MetricBase):
    kind: Literal["lifecycle_summary"]
    outputs: list[str] = Field(default_factory=list)


class SetOpOperand(_PermissiveModel):
    state: str
    time_window: dict[str, Any] | None = None


class SetOpMetric(_MetricBase):
    kind: Literal["set_op"]
    op: Literal["union", "intersection", "a_not_b", "diff"]
    operands: list[SetOpOperand] = Field(default_factory=list)
    states: list[str] = Field(default_factory=list)
    output: Literal["count"] = "count"


class FunnelDropoffMetric(_MetricBase):
    kind: Literal["funnel_dropoff"]
    from_stage: str
    to_stage: str
    output: Literal["rate", "count"] = "rate"


Metric = Annotated[
    FormulaMetric
    | ApproxDistinctCountMetric
    | TopKItemsMetric
    | TdigestQuantileMetric
    | VariantCompareMetric
    | CurveFromDigestsMetric
    | CalibrationFromDigestsMetric
    | ContingencyTestMetric
    | ProportionTestMetric
    | LifecycleSummaryMetric
    | SetOpMetric
    | FunnelDropoffMetric,
    Field(discriminator="kind"),
]


class Metrics(_StrictModel):
    """Top-level shape for ``metrics.yaml``."""

    metrics: dict[str, Metric]


# ---------------------------------------------------------------------------
# dashboards.yaml
# ---------------------------------------------------------------------------


class PageFilterSpec(_StrictModel):
    """One aggregate-backed page filter exposed by the Reports surface."""

    field: str
    label: str = ""
    display: Literal["primary", "secondary"] = "secondary"
    scope: Literal["all_tiles", "compatible_tiles"] = "compatible_tiles"
    control: Literal["multiselect", "selectbox", "text"] = "multiselect"


class TimeFilterSpec(_StrictModel):
    """Available and default time-range presets for one dashboard page."""

    default: Literal[
        "last_7_days",
        "last_30_days",
        "last_90_days",
        "year_to_date",
        "custom",
        "all_time",
    ] = "all_time"
    presets: list[
        Literal[
            "last_7_days",
            "last_30_days",
            "last_90_days",
            "year_to_date",
            "custom",
            "all_time",
        ]
    ] = Field(
        default_factory=lambda: [
            "last_30_days",
            "last_90_days",
            "year_to_date",
            "all_time",
        ]
    )

    @model_validator(mode="after")
    def _default_must_be_available(self) -> TimeFilterSpec:
        if not self.presets:
            raise ValueError("time_filter.presets must contain at least one preset")
        if self.default not in self.presets:
            raise ValueError("time_filter.default must be included in time_filter.presets")
        return self


class KpiSpec(_StrictModel):
    """Explicit scalar-card comparison, target, and sparkline behavior."""

    comparison: Literal["none", "previous_period"] = "none"
    comparison_period: Literal["day", "week", "month", "quarter", "year"] = "month"
    sparkline_grain: Literal["daily", "weekly", "monthly"] | None = None
    sparkline_points: int = Field(default=30, ge=2, le=366)
    target: float | None = None


class Tile(_PermissiveModel):
    """One tile on a dashboard page.

    Required and optional fields per chart kind are documented in
    ``docs/CHART_CATALOG.md`` §3. We allow extras here so chart-specific
    keys (``x``, ``y``, ``color``, ``facets``, ``path``, ``references``,
    ``value``) don't need their own per-kind schemas at Phase 0.
    """

    id: str
    title: str
    metric: str
    description: str = ""
    placement: Literal["content", "kpi_strip"] = "content"
    kpi: KpiSpec | None = None
    scale_mode: Literal["absolute", "index_100", "percent_change"] = "absolute"
    value_format: Literal["percent", "integer", "number", "currency"] | None = None
    chart: Literal[
        "line",
        "stacked_area",
        "bar",
        "kpi_card",
        "waterfall",
        "pareto",
        "treemap",
        "heatmap",
        "cohort_heatmap",
        "scatter",
        "combo",
        "interval",
        "donut",
        "geo_map",
        "table",
        "calendar_heatmap",
        "bar_polar",
        "sankey",
        "gauge",
        "funnel",
        "boxplot",
        "histogram",
        "calibration_curve",
        "roc_curve",
        "precision_recall_curve",
        "gain_curve",
        "lift_curve",
        "rfm_density",
        "exposure",
        "corr",
        "model",
        "descriptive_line",
        "descriptive_boxplot",
        "descriptive_histogram",
        "descriptive_heatmap",
        "descriptive_funnel",
        "experiment_z_score",
        "experiment_odds_ratio",
        "clv_treemap",
    ]


class DashboardPage(_StrictModel):
    id: str
    title: str
    filters: list[PageFilterSpec] = Field(default_factory=list)
    time_filter: TimeFilterSpec = Field(default_factory=TimeFilterSpec)
    tiles: list[Tile]


class Dashboard(_StrictModel):
    id: str
    title: str
    layout: Literal["tabs", "grid", "stacked"] = "tabs"
    pages: list[DashboardPage]


class Dashboards(_StrictModel):
    """Top-level shape for ``dashboards.yaml``."""

    theme: dict[str, Any] = Field(default_factory=dict)
    dashboards: list[Dashboard]


# ---------------------------------------------------------------------------
# Catalog — the assembled config for a workspace.
# ---------------------------------------------------------------------------


class Catalog(_StrictModel):
    """Aggregate of the four catalog YAML files for one workspace."""

    pipelines: Pipelines
    processors: Processors
    metrics: Metrics
    dashboards: Dashboards


__all__ = [
    "DEFAULT_STATE_PARAMETERS",
    "ApproxDistinctCountMetric",
    "BinaryOutcomeProcessor",
    "Calendar",
    "CalibrationFromDigestsMetric",
    "Cast",
    "Catalog",
    "Coalesce",
    "ContingencyTestMetric",
    "CsvReader",
    "CurveFromDigestsMetric",
    "Dashboard",
    "DashboardPage",
    "Dashboards",
    "Dedup",
    "Defaults",
    "DeriveActionId",
    "DeriveCalendar",
    "DeriveColumn",
    "DropColumns",
    "EntityLifecycleProcessor",
    "EntitySetProcessor",
    "FilterTransform",
    "FormulaMetric",
    "FunnelDropoffMetric",
    "FunnelProcessor",
    "KpiSpec",
    "LifecycleSummaryMetric",
    "Metric",
    "MetricDisplaySpec",
    "Metrics",
    "NumericDistributionProcessor",
    "PageFilterSpec",
    "ParquetReader",
    "ParseDatetime",
    "PegaDsExportReader",
    "Pipelines",
    "Processor",
    "ProcessorTime",
    "Processors",
    "ProportionTestMetric",
    "Reader",
    "RenameCapitalize",
    "ScoreDistributionProcessor",
    "SetOpMetric",
    "SetOpOperand",
    "SnapshotProcessor",
    "Source",
    "SourceSchema",
    "StateSpec",
    "TdigestQuantileMetric",
    "Tile",
    "TimeFilterSpec",
    "TopKItemsMetric",
    "Transform",
    "VariantCompareMetric",
    "WorkspaceDefaults",
    "XlsxReader",
    "effective_processor_states",
    "entity_lifecycle_keys",
    "entity_set_column",
    "funnel_stage_names",
    "normalize_grain_name",
    "normalize_grains",
]
