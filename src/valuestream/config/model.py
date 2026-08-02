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
* :class:`Processor` — 8 built-in processor kinds.
* :class:`Metric` — 9 built-in metric kinds.

Catalog version 2 is intentionally strict: every accepted field is part of
the public contract and unknown or retired fields fail validation.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from valuestream.expr.ast import Dtype, Expr
from valuestream.expr.references import column_references

# ---------------------------------------------------------------------------
# Common building blocks.
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    """Base for strictly-typed config models — forbids unknown fields."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


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
    # Common business dimensions. Authoring metadata only: seeds processor
    # group-by selections and never reaches the engine, so it is excluded
    # from computation hashes (see canonical._workspace_computation_defaults).
    dimensions: list[str] = Field(default_factory=list)


class SourceSchema(_StrictModel):
    timestamp_column: str | None = None
    natural_key: list[str] = Field(default_factory=list)
    drop_columns: list[str] = Field(default_factory=list)


# Reader kinds — discriminated on ``kind``.
class _ReaderBase(_StrictModel):
    kind: str
    file_pattern: str
    group_by_filename: str | None = None
    streaming: bool = False
    root: str | None = None
    background: bool = False
    hive_partitioning: bool = False
    separator: str | None = None
    infer_schema_length: int | None = Field(default=None, ge=0)
    try_parse_dates: bool = True
    archive_temp_dir: str | None = None


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
    outputs: list[str] = Field(default_factory=lambda: ["Day", "Week", "Month", "Quarter", "Year"])


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

    catalog_version: Literal[2]
    workspace: str
    defaults: WorkspaceDefaults = Field(default_factory=WorkspaceDefaults)
    sources: list[Source]


# ---------------------------------------------------------------------------
# processors.yaml
# ---------------------------------------------------------------------------


class CountState(_StrictModel):
    type: Literal["count"]
    source_column: str | None = None
    distinct: bool = False
    outcome: Literal["positive", "negative"] | None = None
    stage: str | None = None
    where: Expr | None = None

    @model_validator(mode="after")
    def _one_count_selector(self) -> CountState:
        selectors = [
            self.source_column is not None,
            self.outcome is not None,
            self.stage is not None,
        ]
        if sum(selectors) > 1:
            raise ValueError("count state accepts only one of source_column, outcome, or stage")
        if self.distinct and self.source_column is None:
            raise ValueError("distinct count state requires source_column")
        return self


class ValueSumState(_StrictModel):
    type: Literal["value_sum"]
    source_column: str
    where: Expr | None = None


class MinState(_StrictModel):
    type: Literal["min"]
    source_column: str


class MaxState(_StrictModel):
    type: Literal["max"]
    source_column: str


class PooledMeanState(_StrictModel):
    type: Literal["pooled_mean"]
    source_column: str | None = None
    weight: str
    recipe: Literal["personalization", "novelty"] | None = None

    @model_validator(mode="after")
    def _mean_source_or_recipe(self) -> PooledMeanState:
        if (self.source_column is None) == (self.recipe is None):
            raise ValueError("pooled_mean state requires exactly one of source_column or recipe")
        return self


class PooledVarianceState(_StrictModel):
    type: Literal["pooled_variance"]
    source_column: str
    mean: str
    weight: str


class _DigestState(_StrictModel):
    source_column: str
    k: int | None = Field(default=None, ge=8)
    score_property: str | None = None
    outcome: Literal["positive", "negative"] | None = None

    @model_validator(mode="after")
    def _conditioned_digest_has_property(self) -> _DigestState:
        if self.outcome and not self.score_property:
            raise ValueError("outcome-conditioned sketch states require score_property")
        return self


class TDigestState(_DigestState):
    type: Literal["tdigest"]


class KllState(_DigestState):
    type: Literal["kll"]


class _CardinalityState(_StrictModel):
    source_column: str
    lg_k: int | None = Field(default=None, ge=4, le=26)
    stage: str | None = None
    where: Expr | None = None


class CpcState(_CardinalityState):
    type: Literal["cpc"]


class HllState(_CardinalityState):
    type: Literal["hll"]


class ThetaState(_CardinalityState):
    type: Literal["theta"]


class TopKState(_StrictModel):
    type: Literal["topk"]
    source_column: str
    lg_max_map_size: int | None = Field(default=None, ge=3)


StateSpec = Annotated[
    CountState
    | ValueSumState
    | MinState
    | MaxState
    | PooledMeanState
    | PooledVarianceState
    | TDigestState
    | KllState
    | CpcState
    | HllState
    | ThetaState
    | TopKState,
    Field(discriminator="type"),
]


class ProcessorCalendar(_StrictModel):
    timezone: str = "UTC"
    week_start: Literal["monday", "sunday"] = "monday"
    fiscal_year_start_month: int = Field(default=1, ge=1, le=12)


class ProcessorTime(_StrictModel):
    """Canonical semantic time contract for one processor."""

    property: str
    grain: Literal["hourly", "daily", "weekly", "monthly", "quarterly", "yearly", "summary"]
    calendar: ProcessorCalendar = Field(default_factory=ProcessorCalendar)

    @property
    def column(self) -> str:
        """Internal ingestion name for the configured timestamp property."""

        return self.property


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


class OutcomeSpec(_StrictModel):
    column: str
    positive_values: list[Any]
    negative_values: list[Any]


FREQUENCY_RESPONSE_VIRTUAL_COLUMNS = frozenset(
    {
        "ClickedContact",
        "ComparableContact",
        "ComparableClick",
        "RunnerAvailable",
        "RunnerPropensity",
        "PriorityComparableContact",
        "FocalPriorityComparable",
        "RunnerPriorityComparable",
    }
)
_FREQUENCY_RESPONSE_RESERVED_COLUMNS = frozenset(
    {
        "Day",
        "pipeline_run_id",
        "chunk_id",
        "period",
        "created_at",
        "config_hash",
        *FREQUENCY_RESPONSE_VIRTUAL_COLUMNS,
    }
)


class FrequencyResponseColumns(_StrictModel):
    """Raw source-column bindings for frequency-response materialization."""

    customer: str = Field(min_length=1)
    interaction: str = Field(min_length=1)
    action: str = Field(min_length=1)
    placement: str = Field(min_length=1)
    rank: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    propensity: str = Field(min_length=1)
    priority: str | None = Field(default=None, min_length=1)


class EntitiesSpec(_StrictModel):
    subject: str


class ScorePropertySpec(_StrictModel):
    column: str
    role: Literal["primary", "calibrated", "auxiliary"] = "auxiliary"


class FunnelStageSpec(_StrictModel):
    name: str
    when: Expr


class LifecycleKeysSpec(_StrictModel):
    customer_id: str
    order_id: str
    monetary: str
    purchase_date: str


class SnapshotMilestoneSpec(_StrictModel):
    name: str
    property: str


class _ProcessorBase(_StrictModel):
    id: str
    source: str
    kind: str
    description: str = ""
    group_by: list[str] = Field(default_factory=list)
    time: ProcessorTime
    states: dict[str, StateSpec] = Field(min_length=1)
    filter: Expr | None = None

    @property
    def grains(self) -> list[str]:
        """Return the single physical materialization configured for this processor."""

        return [self.time.grain]

    @property
    def aggregation_levels(self) -> dict[str, str]:
        base_grain = self.time.grain
        levels = {base_grain: base_grain}
        if base_grain != "summary":
            levels["summary"] = (
                "monthly" if base_grain in {"hourly", "daily", "weekly", "monthly"} else base_grain
            )
        return levels

    def aggregation_level_for(self, grain: str) -> str:
        normalized = normalize_grain_name(grain)
        return self.aggregation_levels.get(normalized, normalized)


class BinaryOutcomeProcessor(_ProcessorBase):
    kind: Literal["binary_outcome"]
    outcome: OutcomeSpec
    dedup_keys: list[str] = Field(default_factory=list)
    entities: EntitiesSpec | None = None
    variant_column: str | None = None


class FrequencyResponseCheckpoint(_StrictModel):
    """Execution strategy for bounded frequency-response contact state."""

    mode: Literal["source_scan", "persistent_sharded"] = "source_scan"
    shards: int = Field(default=64, ge=1, le=4096)
    retention_days: int | None = Field(default=None, ge=1, le=3650)


def _minimum_frequency_checkpoint_retention_days(
    window_hours: int,
    partition_lag_hours: int,
) -> int:
    """Return source days needed by the exact window plus partition lag."""

    return (window_hours + partition_lag_hours + 23) // 24


class FrequencyResponseProcessor(_ProcessorBase):
    """Frequency-window response states over a marked current chunk plus lookback."""

    kind: Literal["frequency_response"]
    columns: FrequencyResponseColumns
    alternative_group_by: list[str]
    positive_values: list[Any] = Field(min_length=1)
    exposure_values: list[Any] = Field(min_length=1)
    candidate_values: list[Any] = Field(min_length=1)
    window_hours: int = Field(default=168, gt=0)
    partition_lag_hours: int = Field(default=0, ge=0)
    max_frequency: int = Field(default=7, gt=0)
    frequency_column: str = Field(default="ExposureBucket", min_length=1)
    checkpoint: FrequencyResponseCheckpoint = Field(default_factory=FrequencyResponseCheckpoint)

    @model_validator(mode="after")
    def _frequency_contract_is_mergeable(  # noqa: PLR0912
        self,
    ) -> FrequencyResponseProcessor:
        if self.time.grain != "daily":
            raise ValueError("frequency_response requires time.grain 'daily'")
        if self.frequency_column not in self.group_by:
            raise ValueError("frequency_response frequency_column must be present in group_by")
        if len(self.alternative_group_by) != len(set(self.alternative_group_by)):
            raise ValueError("frequency_response alternative_group_by columns must be unique")
        invalid_alternative_columns = sorted(
            column
            for column in self.alternative_group_by
            if not column.strip()
            or column != column.strip()
            or column in _FREQUENCY_RESPONSE_RESERVED_COLUMNS
            or column == self.frequency_column
            or column.startswith("__valuestream_")
        )
        if invalid_alternative_columns:
            raise ValueError(
                "frequency_response alternative_group_by must contain raw source columns: "
                + ", ".join(repr(column) for column in invalid_alternative_columns)
            )
        mandatory_alternative_columns = {self.columns.customer, self.columns.interaction}
        redundant_alternative_columns = sorted(
            mandatory_alternative_columns.intersection(self.alternative_group_by)
        )
        if redundant_alternative_columns:
            raise ValueError(
                "frequency_response alternative_group_by must not repeat mandatory "
                "customer/interaction columns: " + ", ".join(redundant_alternative_columns)
            )
        if self.columns.customer == self.columns.interaction:
            raise ValueError(
                "frequency_response customer and interaction bindings must be distinct"
            )
        minimum_retention_days = _minimum_frequency_checkpoint_retention_days(
            self.window_hours,
            self.partition_lag_hours,
        )
        if (
            self.checkpoint.mode == "persistent_sharded"
            and self.checkpoint.retention_days is not None
            and self.checkpoint.retention_days < minimum_retention_days
        ):
            raise ValueError(
                "frequency_response checkpoint.retention_days must be at least "
                f"{minimum_retention_days} for the configured window and partition lag"
            )
        if (
            self.frequency_column in _FREQUENCY_RESPONSE_RESERVED_COLUMNS
            or self.frequency_column.startswith("__valuestream_")
        ):
            raise ValueError("frequency_column must not use a reserved derived column name")
        invalid_group_columns = sorted(
            column
            for column in self.group_by
            if column in _FREQUENCY_RESPONSE_RESERVED_COLUMNS - {"Day"}
            or column.startswith("__valuestream_")
        )
        if invalid_group_columns:
            raise ValueError(
                "frequency_response reserved derived columns cannot be used in group_by: "
                + ", ".join(invalid_group_columns)
            )
        raw_columns = set(self.columns.model_dump(exclude_none=True).values())
        invalid_input_bindings = sorted(
            column
            for column in {*raw_columns, self.time.property}
            if column.startswith("__valuestream_")
        )
        if invalid_input_bindings:
            raise ValueError(
                "frequency_response raw input bindings cannot use internal column names: "
                + ", ".join(invalid_input_bindings)
            )
        if self.frequency_column in raw_columns:
            raise ValueError("frequency_column must be a derived column, not a raw input binding")
        invalid_filter_columns = sorted(
            column
            for column in column_references(self.filter)
            if column
            in {
                *FREQUENCY_RESPONSE_VIRTUAL_COLUMNS,
                "Day",
                self.frequency_column,
            }
            or column.startswith("__valuestream_")
        )
        if invalid_filter_columns:
            raise ValueError(
                "frequency_response filter runs before derived columns and cannot reference: "
                + ", ".join(invalid_filter_columns)
            )
        colliding_states = sorted(
            name
            for name in self.states
            if name
            in set(self.group_by)
            | (_FREQUENCY_RESPONSE_RESERVED_COLUMNS - FREQUENCY_RESPONSE_VIRTUAL_COLUMNS)
            or name.startswith("__valuestream_")
        )
        if colliding_states:
            raise ValueError(
                "frequency_response state names collide with dimensions or reserved columns: "
                + ", ".join(colliding_states)
            )
        for name, state in self.states.items():
            source_column = getattr(state, "source_column", None)
            internal_state_inputs = sorted(
                {
                    *(
                        {source_column}
                        if isinstance(source_column, str)
                        and source_column.startswith("__valuestream_")
                        else set()
                    ),
                    *(
                        column
                        for column in column_references(getattr(state, "where", None))
                        if column.startswith("__valuestream_")
                    ),
                }
            )
            if internal_state_inputs:
                raise ValueError(
                    f"frequency_response state {name!r} cannot read internal columns: "
                    + ", ".join(internal_state_inputs)
                )
            if state.type not in {"count", "value_sum"}:
                raise ValueError(f"frequency_response state {name!r} must use count or value_sum")
            if isinstance(state, CountState) and (state.outcome is not None or state.stage):
                raise ValueError(
                    f"frequency_response state {name!r} must bind a virtual/source column "
                    "instead of an outcome or stage selector"
                )
        return self

    @property
    def checkpoint_retention_days(self) -> int:
        """Return the bounded number of source days retained in rolling state."""

        if self.checkpoint.retention_days is not None:
            return self.checkpoint.retention_days
        return _minimum_frequency_checkpoint_retention_days(
            self.window_hours,
            self.partition_lag_hours,
        )

    @property
    def alternative_group_columns(self) -> list[str]:
        """Return source columns that define one selected-action comparison."""

        return list(
            dict.fromkeys(
                [
                    self.columns.customer,
                    self.columns.interaction,
                    *self.alternative_group_by,
                ]
            )
        )


class NumericDistributionProcessor(_ProcessorBase):
    kind: Literal["numeric_distribution"]
    properties: list[str] = Field(min_length=1)
    quantile_engine: Literal["tdigest", "kll"] = "tdigest"


class ScoreDistributionProcessor(_ProcessorBase):
    kind: Literal["score_distribution"]
    score_properties: list[ScorePropertySpec] = Field(min_length=1)
    outcome: OutcomeSpec
    dedup_keys: list[str] = Field(default_factory=list)
    entities: EntitiesSpec | None = None


class EntityLifecycleProcessor(_ProcessorBase):
    kind: Literal["entity_lifecycle"]
    keys: LifecycleKeysSpec
    lifespan_years: float | None = Field(default=None, gt=0)


class EntitySetProcessor(_ProcessorBase):
    kind: Literal["entity_set"]
    entity: str
    entities: EntitiesSpec | None = None


class FunnelProcessor(_ProcessorBase):
    kind: Literal["funnel"]
    stages: list[FunnelStageSpec] = Field(min_length=1)
    entity: str | None = None


class SnapshotProcessor(_ProcessorBase):
    kind: Literal["snapshot"]
    snapshot_kind: Literal["periodic", "accumulating"]
    cadence: Literal["daily", "weekly", "monthly"] | None = None
    entity: str
    as_of_property: str | None = None
    milestones: list[SnapshotMilestoneSpec] = Field(default_factory=list)


Processor = Annotated[
    BinaryOutcomeProcessor
    | FrequencyResponseProcessor
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

    catalog_version: Literal[2]
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
    """Return the processor's explicit catalog output contract."""

    return processor.states


def entity_lifecycle_keys(processor: Processor) -> dict[str, str]:
    """Return the configured lifecycle key columns."""

    if not isinstance(processor, EntityLifecycleProcessor):
        return {}
    return processor.keys.model_dump()


def funnel_stage_names(processor: Processor) -> list[str]:
    """Return configured funnel stage names in stage order."""
    if not isinstance(processor, FunnelProcessor):
        return []
    return [stage.name for stage in processor.stages]


def entity_set_column(processor: Processor) -> str:
    """Return the explicitly configured entity column."""

    if not isinstance(processor, EntitySetProcessor):
        raise TypeError("entity_set_column requires an entity_set processor")
    return processor.entity


def processor_settings(processor: Processor) -> dict[str, Any]:
    """Return only kind-specific, explicitly typed processor settings."""

    payload = processor.model_dump(mode="python", by_alias=True, exclude_none=True)
    for field_name in (
        "id",
        "source",
        "kind",
        "description",
        "group_by",
        "time",
        "states",
        "filter",
    ):
        payload.pop(field_name, None)
    return payload


def state_settings(spec: StateSpec) -> dict[str, Any]:
    """Return explicitly typed settings for one state, excluding its discriminator."""

    payload = spec.model_dump(mode="python", by_alias=True, exclude_none=True)
    payload.pop("type", None)
    return payload


# ---------------------------------------------------------------------------
# metrics.yaml
# ---------------------------------------------------------------------------


class MetricDisplaySpec(_StrictModel):
    """Presentation metadata shared by reports and catalog authoring surfaces."""

    label: str = ""
    unit: str = ""
    value_format: Literal["percent", "integer", "number", "currency"] | None = None
    direction: Literal["higher_is_better", "lower_is_better", "neutral"] = "neutral"


class MetricRecipeSpec(_StrictModel):
    id: str
    version: int = Field(ge=1)
    parameters: dict[str, float] = Field(default_factory=dict)


class _MetricBase(_StrictModel):
    """Base shape for every catalog-v2 metric."""

    processor: str
    kind: str
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    display: MetricDisplaySpec | None = None
    recipe: MetricRecipeSpec | None = None


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


class DistributionMetric(_MetricBase):
    """Full distribution exposed by one t-digest or KLL state."""

    kind: Literal["distribution"]
    state: str


class QuantileMetric(_MetricBase):
    """One explicitly selected quantile from a t-digest or KLL state."""

    kind: Literal["quantile"]
    state: str
    quantile: float = Field(ge=0.0, le=1.0)


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


LifecycleOutput = Literal[
    "customers_count",
    "unique_holdings",
    "lifetime_value",
    "frequency",
    "recency",
    "monetary_value",
    "rfm_segment",
    "rfm_score",
]


class LifecycleSummaryMetric(_MetricBase):
    kind: Literal["lifecycle_summary"]
    holdings_state: str
    monetary_state: str
    first_purchase_state: str
    last_purchase_state: str
    outputs: list[LifecycleOutput] = Field(default_factory=list)
    segment_preset: Literal["default", "rfm", "lifecycle"] = "default"
    entity_column: str


class SetOpTimeWindow(_StrictModel):
    last: str | None = None
    between: tuple[str, str] | None = None

    @model_validator(mode="after")
    def _exactly_one_window(self) -> SetOpTimeWindow:
        if (self.last is None) == (self.between is None):
            raise ValueError("set-operation window requires exactly one of last or between")
        return self


class SetOpOperand(_StrictModel):
    state: str
    time_window: SetOpTimeWindow | None = None


class SetOpMetric(_MetricBase):
    kind: Literal["set_op"]
    operation: Literal["union", "intersection", "minus"]
    operands: list[SetOpOperand] = Field(min_length=2)

    @model_validator(mode="after")
    def _minus_has_two_operands(self) -> SetOpMetric:
        if self.operation == "minus" and len(self.operands) != 2:
            raise ValueError("minus requires exactly two operands")
        return self


class FunnelDropoffMetric(_MetricBase):
    kind: Literal["funnel_dropoff"]
    from_state: str
    to_state: str
    output: Literal["rate", "count"] = "rate"


Metric = Annotated[
    FormulaMetric
    | ApproxDistinctCountMetric
    | TopKItemsMetric
    | DistributionMetric
    | QuantileMetric
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
_METRIC_ADAPTER = TypeAdapter(Metric)


def validate_metric(value: Any) -> Metric:
    """Validate one metric definition against the canonical metric union."""

    return _METRIC_ADAPTER.validate_python(value)


class Metrics(_StrictModel):
    """Top-level shape for ``metrics.yaml``."""

    catalog_version: Literal[2]
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


class _TileBase(_StrictModel):
    id: str
    title: str
    metric: str
    description: str = ""
    filters: dict[str, Any] | None = None
    placement: Literal["content", "kpi_strip"] = "content"
    kpi: KpiSpec | None = None
    scale_mode: Literal["absolute", "index_100", "percent_change"] = "absolute"
    value_format: Literal["percent", "integer", "number", "currency"] | None = None
    metric_output: str | None = None
    group_by: list[str] | None = None
    goal_line: float | dict[str, Any] | None = None
    show_trend_delta: bool = False
    sort_by: str | None = None
    sort_direction: Literal["ascending", "descending"] | None = None
    top_n: int | None = Field(default=None, ge=1)
    labels: dict[str, str] | None = None
    theme: dict[str, Any] | None = None
    conditional_formatting: list[dict[str, Any]] | None = None
    bins: int | None = Field(default=None, ge=1)
    hole: float | None = Field(default=None, ge=0.0, le=1.0)
    height: int | None = Field(default=None, ge=1)
    width: int | None = Field(default=None, ge=1)
    log_x: bool = False
    log_y: bool = False
    showlegend: bool | None = None
    horizon: int | None = Field(default=None, ge=1)
    x_axis_title: str | None = None
    y_axis_title: str | None = None


class _FacetedTile(_TileBase):
    color: str | None = None
    facet_row: str | None = None
    facet_col: str | None = None


class LineTile(_FacetedTile):
    chart: Literal["line"]
    x: str
    line_dash: str | None = None
    symbol: str | None = None


class MetricAxisTile(_FacetedTile):
    chart: Literal["bar", "pareto"]
    x: str


class StackedAreaTile(_FacetedTile):
    chart: Literal["stacked_area"]
    x: str
    color: str


class WaterfallTile(_TileBase):
    chart: Literal["waterfall"]
    x: str


class ScalarTile(_TileBase):
    chart: Literal["kpi_card"]


class GaugeTile(_FacetedTile):
    chart: Literal["gauge"]
    reference: float | str | None = None
    references: dict[str, float] | None = None


class TreemapTile(_TileBase):
    chart: Literal["treemap", "clv_treemap"]
    path: list[str] = Field(min_length=1)


class HeatmapTile(_TileBase):
    chart: Literal["heatmap"]
    x: str
    y: str | None = None


class ScatterTile(_FacetedTile):
    chart: Literal["scatter"]
    x: str
    y: str
    size: str | None = None
    animation_frame: str | None = None
    animation_group: str | None = None


class ComboTile(_FacetedTile):
    chart: Literal["combo"]
    x: str
    secondary_metric: str
    primary_mark: Literal["bar", "line"] = "bar"
    shared_y_axis: bool = False
    y2_axis_title: str | None = None


class IntervalTile(_FacetedTile):
    chart: Literal["interval"]
    x: str
    metric_output: str
    lower_output: str | None = None
    upper_output: str | None = None

    @model_validator(mode="after")
    def _paired_bounds(self) -> IntervalTile:
        if (self.lower_output is None) != (self.upper_output is None):
            raise ValueError("interval lower_output and upper_output must be configured together")
        return self


class DonutTile(_TileBase):
    chart: Literal["donut"]
    names: str
    color: str | None = None
    hole: float | None = Field(default=None, ge=0.0, le=1.0)


class GeoMapTile(_TileBase):
    chart: Literal["geo_map"]
    locations: str | None = None
    lat: str | None = None
    lon: str | None = None
    color: str | None = None
    size: str | None = None
    locationmode: str | None = None

    @model_validator(mode="after")
    def _location_role(self) -> GeoMapTile:
        if not self.locations and not self.lat:
            raise ValueError("geo_map requires locations or lat")
        return self


class TableTile(_TileBase):
    chart: Literal["table"]
    columns: list[str] | None = None


class PolarTile(_TileBase):
    chart: Literal["bar_polar"]
    theta: str
    color: str | None = None


class SankeyTile(_TileBase):
    chart: Literal["sankey"]
    path: list[str] = Field(min_length=2)


class FunnelTile(_FacetedTile):
    chart: Literal["funnel"]
    stages: list[str] = Field(min_length=1)
    x: str | None = None


class DistributionTile(_FacetedTile):
    chart: Literal["boxplot"]
    x: str | None = None


class HistogramTile(_FacetedTile):
    chart: Literal["histogram"]
    property: str


class CurveTile(_FacetedTile):
    chart: Literal[
        "calibration_curve",
        "roc_curve",
        "precision_recall_curve",
        "gain_curve",
        "lift_curve",
    ]


class RfmDensityTile(_TileBase):
    chart: Literal["rfm_density"]
    x: str | None = None
    y: str | None = None
    color: str | None = None


class ExposureTile(_TileBase):
    chart: Literal["exposure"]
    color: str | None = None


class CorrelationTile(_TileBase):
    chart: Literal["corr"]
    x: str
    y: str
    color: str | None = None


class ModelTile(_TileBase):
    chart: Literal["model"]
    color: str | None = None


class DescriptiveLineTile(_FacetedTile):
    chart: Literal["descriptive_line"]
    x: str
    property: str
    score: str


class ExperimentTile(_FacetedTile):
    chart: Literal["experiment_z_score", "experiment_odds_ratio"]
    x: str
    y: str


Tile = Annotated[
    LineTile
    | MetricAxisTile
    | StackedAreaTile
    | WaterfallTile
    | ScalarTile
    | GaugeTile
    | TreemapTile
    | HeatmapTile
    | ScatterTile
    | ComboTile
    | IntervalTile
    | DonutTile
    | GeoMapTile
    | TableTile
    | PolarTile
    | SankeyTile
    | FunnelTile
    | DistributionTile
    | HistogramTile
    | CurveTile
    | RfmDensityTile
    | ExposureTile
    | CorrelationTile
    | ModelTile
    | DescriptiveLineTile
    | ExperimentTile,
    Field(discriminator="chart"),
]
_TILE_ADAPTER = TypeAdapter(Tile)


def validate_tile(value: Any) -> Tile:
    """Validate one chart-specific tile definition."""

    return _TILE_ADAPTER.validate_python(value)


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

    catalog_version: Literal[2]
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
    "FREQUENCY_RESPONSE_VIRTUAL_COLUMNS",
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
    "DistributionMetric",
    "DropColumns",
    "EntityLifecycleProcessor",
    "EntitySetProcessor",
    "FilterTransform",
    "FormulaMetric",
    "FrequencyResponseColumns",
    "FrequencyResponseProcessor",
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
    "ProcessorCalendar",
    "ProcessorTime",
    "Processors",
    "ProportionTestMetric",
    "QuantileMetric",
    "Reader",
    "RenameCapitalize",
    "ScoreDistributionProcessor",
    "SetOpMetric",
    "SetOpOperand",
    "SetOpTimeWindow",
    "SnapshotProcessor",
    "Source",
    "SourceSchema",
    "StateSpec",
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
    "processor_settings",
    "state_settings",
    "validate_tile",
]
