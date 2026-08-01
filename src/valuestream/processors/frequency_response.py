"""Frequency-window response processor over a marked current chunk plus lookback."""

from __future__ import annotations

from typing import Any

import polars as pl

import valuestream.processors.processors_helper as p3
from valuestream.config import model
from valuestream.config.canonical import processor_config_hash
from valuestream.expr.references import column_references
from valuestream.expr.translator import translate
from valuestream.processors import grain_levels
from valuestream.processors.context import (
    PROVENANCE_COLUMNS,
    TARGET_CHUNK_COLUMN,
    ChunkContext,
)
from valuestream.processors.outcomes import compatible_values, is_in_values
from valuestream.utils.timer import timed

VIRTUAL_COLUMNS = model.FREQUENCY_RESPONSE_VIRTUAL_COLUMNS
DERIVED_GROUP_COLUMNS = frozenset({"Day"})

_RANK_COLUMN = "__valuestream_frequency_rank"
_ROW_ORDER_COLUMN = "__valuestream_frequency_row_order"
_POSITIVE_COLUMN = "__valuestream_frequency_positive"
_EXPOSED_COLUMN = "__valuestream_frequency_exposed"
_SEEN_CURRENT_COLUMN = "__valuestream_frequency_seen_current"
_SEEN_HISTORY_COLUMN = "__valuestream_frequency_seen_history"
_CONTACT_ORDER_COLUMN = "__valuestream_frequency_contact_order"
_EXPOSURE_SEQUENCE_COLUMN = "__valuestream_frequency_exposure_sequence"
_WINDOW_START_COLUMN = "__valuestream_frequency_window_start"
_BOUNDARY_TIME_COLUMN = "__valuestream_frequency_boundary_time"
_BOUNDARY_SEQUENCE_COLUMN = "__valuestream_frequency_boundary_sequence"
_RUNNER_PROPENSITY_COLUMN = "__valuestream_runner_propensity"
_RUNNER_PRIORITY_COLUMN = "__valuestream_runner_priority"
_RUNNER_AVAILABLE_COLUMN = "__valuestream_runner_available"
_CHECKPOINT_PARTITION_ORDER_COLUMN = "__valuestream_frequency_checkpoint_partition_order"
_CHECKPOINT_LOCAL_ORDER_COLUMN = "__valuestream_frequency_checkpoint_local_order"


def required_input_columns(config: model.FrequencyResponseProcessor) -> frozenset[str]:
    """Return raw transformed-frame inputs required by ``config``.

    Processor-created contact markers, the number-of-impressions bucket, and the
    daily calendar key are intentionally excluded.
    """

    virtual = {*VIRTUAL_COLUMNS, config.frequency_column, *DERIVED_GROUP_COLUMNS}
    required = {
        config.time.property,
        TARGET_CHUNK_COLUMN,
        *config.columns.model_dump(exclude_none=True).values(),
        *config.alternative_group_by,
        *(column for column in config.group_by if column not in virtual),
    }
    required.update(column_references(config.filter))
    for spec in config.states.values():
        source_column = getattr(spec, "source_column", None)
        if source_column and source_column not in virtual:
            required.add(source_column)
        required.update(column_references(getattr(spec, "where", None)) - virtual)
    return frozenset(str(column) for column in required if str(column).strip())


def required_history_input_columns(
    config: model.FrequencyResponseProcessor,
) -> frozenset[str]:
    """Return the narrow transformed history schema needed for exposure counts.

    Historical alternatives, reporting dimensions, and state inputs never
    contribute to the target contact. Raw processor-filter columns remain
    required because that filter applies before history is reduced to exposed
    rank-one contacts.
    """

    columns = config.columns
    required = {
        config.time.property,
        columns.customer,
        columns.interaction,
        columns.action,
        columns.placement,
        columns.rank,
        columns.outcome,
        *column_references(config.filter),
    }
    return frozenset(str(column) for column in required if str(column).strip())


def validate_current_input_schema(
    config: model.FrequencyResponseProcessor,
    schema: pl.Schema,
) -> None:
    """Validate the unmarked transformed target schema for one processor."""

    missing = sorted((required_input_columns(config) - {TARGET_CHUNK_COLUMN}) - set(schema.names()))
    if missing:
        raise ValueError(
            f"frequency_response processor {config.id!r} requires missing input "
            f"column(s): {', '.join(missing)}"
        )
    if config.frequency_column in schema.names():
        raise ValueError(
            f"frequency_response processor {config.id!r} cannot derive frequency column "
            f"{config.frequency_column!r} because it already exists in the transformed "
            "source schema"
        )
    decision_dtype = schema[config.time.property]
    if decision_dtype.base_type() != pl.Datetime:
        raise TypeError(
            f"frequency_response processor {config.id!r} requires datetime decision "
            f"column {config.time.property!r}, got {decision_dtype}"
        )
    columns = config.columns
    rank_dtype = schema[columns.rank]
    if not rank_dtype.is_integer():
        raise TypeError(
            f"frequency_response processor {config.id!r} requires integer rank "
            f"column {columns.rank!r}, got {rank_dtype}"
        )
    for role, column in (
        ("propensity", columns.propensity),
        ("priority", columns.priority),
    ):
        if column is None:
            continue
        dtype = schema[column]
        if not dtype.is_numeric():
            raise TypeError(
                f"frequency_response processor {config.id!r} requires numeric {role} "
                f"column {column!r}, got {dtype}"
            )


class FrequencyResponseProcessor:
    """Build mergeable response/opportunity states by number of impressions."""

    def __init__(self, config: model.Processor, *, computation_hash: str | None = None) -> None:
        if not isinstance(config, model.FrequencyResponseProcessor):
            raise TypeError(f"expected frequency_response processor, got {config.kind!r}")
        self.config = config
        self.config_hash = computation_hash or processor_config_hash(config)

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def source_id(self) -> str:
        return self.config.source

    @property
    def group_by_columns(self) -> list[str]:
        return list(self.config.group_by)

    @property
    def state_specs(self) -> dict[str, model.StateSpec]:
        return model.effective_processor_states(self.config)

    @timed
    def chunk_aggregate(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.DataFrame:
        """Return one daily partial for the marked target chunk."""

        return self.chunk_aggregate_lazy(frame, ctx).collect()

    @timed
    def chunk_aggregate_lazy(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.LazyFrame:
        """Build the deterministic lookback/contact aggregation as one lazy plan."""

        source_schema = frame.collect_schema()
        self._validate_input_schema(source_schema)
        contacts = self._prepare_contacts(frame, source_schema)
        return self._aggregate_contacts_lazy(contacts, ctx)

    def checkpoint_contacts_lazy(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        """Prepare one current source chunk for immutable checkpoint storage.

        The checkpoint retains filtered/projected current candidate rows because
        cross-partition contact normalization must see duplicates before collapsing
        them. A separate history payload is derived from this prepared target
        payload after it has been streamed to checkpoint storage.
        """

        source_schema = frame.collect_schema()
        validate_current_input_schema(self.config, source_schema)
        marked = frame.with_columns(pl.lit(True).alias(TARGET_CHUNK_COLUMN))
        marked_schema = marked.collect_schema()
        return self._prepare_checkpoint_rows(marked, marked_schema)

    def checkpoint_history_contacts_lazy(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        """Project staged target candidates to the exact bounded-history payload."""

        columns = self.config.columns
        history_columns = list(
            dict.fromkeys(
                [
                    columns.customer,
                    columns.interaction,
                    columns.action,
                    columns.placement,
                    self.config.time.property,
                    _ROW_ORDER_COLUMN,
                    _RANK_COLUMN,
                    _POSITIVE_COLUMN,
                    _EXPOSED_COLUMN,
                ]
            )
        )
        schema = frame.collect_schema()
        missing = sorted(set(history_columns) - set(schema.names()))
        if missing:
            raise ValueError(
                f"frequency_response processor {self.id!r} target checkpoint is missing "
                f"history column(s): {', '.join(missing)}"
            )
        return frame.filter(pl.col(_EXPOSED_COLUMN) & (pl.col(_RANK_COLUMN) == 1)).select(
            history_columns
        )

    @timed
    def checkpoint_aggregate_lazy(
        self,
        current: pl.LazyFrame,
        history: list[pl.LazyFrame],
        ctx: ChunkContext,
    ) -> pl.LazyFrame:
        """Aggregate one customer shard from prepared daily checkpoints."""

        columns = self.config.columns
        prepared: list[pl.LazyFrame] = []
        for partition_order, historical in enumerate(history):
            prepared.append(
                historical.filter(
                    pl.col(_EXPOSED_COLUMN) & (pl.col(_RANK_COLUMN) == 1)
                ).with_columns(
                    pl.lit(False).alias(TARGET_CHUNK_COLUMN),
                    pl.lit(partition_order).alias(_CHECKPOINT_PARTITION_ORDER_COLUMN),
                )
            )
        prepared.append(
            current.with_columns(
                pl.lit(True).alias(TARGET_CHUNK_COLUMN),
                pl.lit(len(history)).alias(_CHECKPOINT_PARTITION_ORDER_COLUMN),
            )
        )
        contacts = (
            pl.concat(prepared, how="diagonal_relaxed")
            .rename({_ROW_ORDER_COLUMN: _CHECKPOINT_LOCAL_ORDER_COLUMN})
            .sort([_CHECKPOINT_PARTITION_ORDER_COLUMN, _CHECKPOINT_LOCAL_ORDER_COLUMN])
            .with_row_index(_ROW_ORDER_COLUMN)
            .drop(_CHECKPOINT_PARTITION_ORDER_COLUMN, _CHECKPOINT_LOCAL_ORDER_COLUMN)
        )
        checkpoint_schema = contacts.collect_schema()
        missing = sorted(
            {
                columns.customer,
                columns.interaction,
                columns.action,
                columns.placement,
                columns.propensity,
                *self.config.alternative_group_by,
                self.config.time.property,
                TARGET_CHUNK_COLUMN,
                _RANK_COLUMN,
                _ROW_ORDER_COLUMN,
                _POSITIVE_COLUMN,
                _EXPOSED_COLUMN,
            }
            - set(checkpoint_schema.names())
        )
        if missing:
            raise ValueError(
                f"frequency_response processor {self.id!r} checkpoint is missing "
                f"column(s): {', '.join(missing)}"
            )
        normalized = self._normalize_contacts(contacts)
        return self._aggregate_contacts_lazy(normalized, ctx)

    def _prepare_contacts(self, frame: pl.LazyFrame, source_schema: pl.Schema) -> pl.LazyFrame:
        """Filter, classify, project, and normalize transformed source contacts."""

        return self._normalize_contacts(self._prepare_checkpoint_rows(frame, source_schema))

    def _prepare_checkpoint_rows(
        self,
        frame: pl.LazyFrame,
        source_schema: pl.Schema,
    ) -> pl.LazyFrame:
        """Return the narrow classified rows persisted by a contact checkpoint."""

        source = frame
        if self.config.filter is not None:
            source = source.filter(translate(self.config.filter))

        columns = self.config.columns
        outcome_dtype = source_schema[columns.outcome]
        positive_values = compatible_values(self.config.positive_values, outcome_dtype)
        exposure_values = compatible_values(self.config.exposure_values, outcome_dtype)
        candidate_values = compatible_values(
            _dedupe_values(
                [
                    *self.config.candidate_values,
                    *self.config.exposure_values,
                    *self.config.positive_values,
                ]
            ),
            outcome_dtype,
        )
        source = (
            source.filter(is_in_values(columns.outcome, candidate_values))
            .with_columns(
                pl.col(TARGET_CHUNK_COLUMN).fill_null(False).alias(TARGET_CHUNK_COLUMN),
                pl.col(columns.rank).cast(pl.Int64).alias(_RANK_COLUMN),
                is_in_values(columns.outcome, positive_values).alias(_POSITIVE_COLUMN),
            )
            .with_columns(
                (pl.col(_POSITIVE_COLUMN) | is_in_values(columns.outcome, exposure_values)).alias(
                    _EXPOSED_COLUMN
                )
            )
            # Rows missing a contact identity or decision time can never
            # contribute after normalization. Drop them before checkpoint
            # sharding as well, so a null customer is excluded rather than
            # turning persistent mode into an input error.
            .filter(
                pl.all_horizontal(
                    pl.col(column).is_not_null()
                    for column in (
                        columns.customer,
                        columns.interaction,
                        columns.action,
                        columns.placement,
                        columns.rank,
                        self.config.time.property,
                    )
                )
            )
            .filter(
                pl.col(TARGET_CHUNK_COLUMN)
                | ((pl.col(_RANK_COLUMN) == 1) & pl.col(_EXPOSED_COLUMN))
            )
            .with_row_index(_ROW_ORDER_COLUMN)
        )

        projected = set(required_input_columns(self.config))
        projected.update(
            {
                columns.outcome,
                columns.rank,
                self.config.time.property,
                TARGET_CHUNK_COLUMN,
            }
        )
        return source.select(
            *[name for name in source_schema.names() if name in projected],
            _ROW_ORDER_COLUMN,
            _RANK_COLUMN,
            _POSITIVE_COLUMN,
            _EXPOSED_COLUMN,
        )

    def _aggregate_contacts_lazy(
        self,
        contacts: pl.LazyFrame,
        ctx: ChunkContext,
    ) -> pl.LazyFrame:
        """Build daily additive states from normalized current/history contacts."""

        contact_schema = contacts.collect_schema()
        exposures = self._with_exposure_frequency(contacts)
        runners = self._runner_candidates(contacts)

        focal = exposures.filter(
            pl.col(_SEEN_CURRENT_COLUMN)
            & ~pl.col(_SEEN_HISTORY_COLUMN)
            & pl.col(_EXPOSED_COLUMN)
            & (pl.col(_RANK_COLUMN) == 1)
        ).join(
            runners,
            on=self.config.alternative_group_columns,
            how="left",
            nulls_equal=True,
        )
        focal = self._with_virtual_columns(focal)
        focal = focal.with_columns(
            self._decision_day_expr(contact_schema[self.config.time.property]).alias("Day")
        )

        group_keys = list(dict.fromkeys([*self.group_by_columns, "Day"]))
        enriched_columns = set(focal.collect_schema().names())
        missing_groups = sorted(set(group_keys) - enriched_columns)
        if missing_groups:
            raise ValueError(
                f"frequency_response processor {self.id!r} cannot derive group-by "
                f"column(s): {', '.join(missing_groups)}"
            )
        grouped = focal.group_by(group_keys).agg(self._agg_exprs(enriched_columns))
        return p3.with_provenance(
            grouped,
            self.config_hash,
            ctx,
            period=pl.col("Day").dt.strftime("%Y-%m"),
        )

    @timed
    def compact(self, frame: pl.DataFrame, target_grain: str, ctx: ChunkContext) -> pl.DataFrame:
        """Project or merge the configured daily state contract."""

        if frame.is_empty():
            return frame
        target_grain = grain_levels.normalize_target_grain(
            self.config, target_grain, "frequency_response"
        )
        working = frame.drop(
            [
                column
                for column in PROVENANCE_COLUMNS
                if column != "period" and column in frame.columns
            ]
        )
        group_columns = list(dict.fromkeys([*self.group_by_columns, "Day", "period"]))
        merged = p3.compact_state_frame(
            working,
            self.state_specs,
            group_columns,
            self.merge,
            identity_level=True,
        )
        return p3.with_static_provenance(merged, self.config_hash, ctx)

    @timed
    def merge(self, frame: pl.DataFrame, group_columns: list[str] | None = None) -> pl.DataFrame:
        """Merge count/value-sum partials with the generic state algebra."""

        if frame.is_empty():
            return frame
        return p3.merge_state_frame(frame, self.state_specs, group_columns)

    @timed
    def merge_for_query(self, frame: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
        """Merge query rows and retain the current computation hash."""

        return p3.merge_for_query(self.merge, frame, group_columns, self.config_hash)

    def _validate_input_schema(self, schema: pl.Schema) -> None:
        validate_current_input_schema(self.config, schema)
        if TARGET_CHUNK_COLUMN not in schema.names():
            raise ValueError(
                f"frequency_response processor {self.id!r} requires missing input "
                f"column(s): {TARGET_CHUNK_COLUMN}"
            )
        if schema[TARGET_CHUNK_COLUMN] != pl.Boolean:
            raise TypeError(
                f"frequency_response processor {self.id!r} requires boolean {TARGET_CHUNK_COLUMN!r}"
            )

    def _decision_day_expr(self, dtype: pl.DataType) -> pl.Expr:
        timestamp = pl.col(self.config.time.property)
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
            timestamp = timestamp.dt.replace_time_zone("UTC")
        return timestamp.dt.convert_time_zone(self.config.time.calendar.timezone).dt.date()

    def _normalize_contacts(self, source: pl.LazyFrame) -> pl.LazyFrame:
        columns = self.config.columns
        time_column = self.config.time.property
        contact_keys = [
            columns.customer,
            columns.interaction,
            columns.action,
            columns.placement,
            _RANK_COLUMN,
        ]
        source = source.filter(
            pl.all_horizontal(pl.col(key).is_not_null() for key in contact_keys),
            pl.col(time_column).is_not_null(),
        )
        sort_columns = [
            *contact_keys,
            _POSITIVE_COLUMN,
            _EXPOSED_COLUMN,
            TARGET_CHUNK_COLUMN,
            time_column,
            _ROW_ORDER_COLUMN,
        ]
        source = source.sort(
            sort_columns,
            descending=[False] * len(contact_keys) + [True, True, True, False, False],
        )
        source_columns = source.collect_schema().names()
        passthrough = [
            name
            for name in source_columns
            if name
            not in {
                *contact_keys,
                TARGET_CHUNK_COLUMN,
                _ROW_ORDER_COLUMN,
                _POSITIVE_COLUMN,
                _EXPOSED_COLUMN,
                time_column,
                columns.propensity,
                *(set() if columns.priority is None else {columns.priority}),
            }
        ]
        aggregations: list[pl.Expr] = [
            pl.col(time_column).min().alias(time_column),
            pl.col(columns.propensity).drop_nulls().first().alias(columns.propensity),
            pl.col(_POSITIVE_COLUMN).max().alias(_POSITIVE_COLUMN),
            pl.col(_EXPOSED_COLUMN).max().alias(_EXPOSED_COLUMN),
            pl.col(TARGET_CHUNK_COLUMN).any().alias(_SEEN_CURRENT_COLUMN),
            (~pl.col(TARGET_CHUNK_COLUMN)).any().alias(_SEEN_HISTORY_COLUMN),
            pl.col(_ROW_ORDER_COLUMN).min().alias(_CONTACT_ORDER_COLUMN),
            *(pl.col(name).first().alias(name) for name in passthrough),
        ]
        if columns.priority is not None:
            aggregations.append(
                pl.col(columns.priority).drop_nulls().first().alias(columns.priority)
            )
        return source.group_by(contact_keys, maintain_order=True).agg(aggregations)

    def _with_exposure_frequency(self, contacts: pl.LazyFrame) -> pl.LazyFrame:
        columns = self.config.columns
        time_column = self.config.time.property
        exposure_keys = [columns.customer, columns.action, columns.placement]
        ordered = contacts.filter(pl.col(_EXPOSED_COLUMN) & (pl.col(_RANK_COLUMN) == 1)).sort(
            [*exposure_keys, time_column, columns.interaction, _CONTACT_ORDER_COLUMN]
        )
        ordered = ordered.with_columns(
            pl.col(time_column)
            .cum_count()
            .over(exposure_keys)
            .cast(pl.Int64)
            .alias(_EXPOSURE_SEQUENCE_COLUMN)
        )
        left = ordered.with_columns(
            (pl.col(time_column) - pl.duration(hours=self.config.window_hours)).alias(
                _WINDOW_START_COLUMN
            )
        ).sort([*exposure_keys, _WINDOW_START_COLUMN, columns.interaction, _CONTACT_ORDER_COLUMN])
        right = ordered.select(
            *exposure_keys,
            pl.col(time_column).alias(_BOUNDARY_TIME_COLUMN),
            pl.col(_EXPOSURE_SEQUENCE_COLUMN).alias(_BOUNDARY_SEQUENCE_COLUMN),
        ).sort([*exposure_keys, _BOUNDARY_TIME_COLUMN, _BOUNDARY_SEQUENCE_COLUMN])
        return (
            left.join_asof(
                right,
                left_on=_WINDOW_START_COLUMN,
                right_on=_BOUNDARY_TIME_COLUMN,
                by=exposure_keys,
                strategy="backward",
                allow_exact_matches=True,
                check_sortedness=False,
            )
            .with_columns(
                (pl.col(_EXPOSURE_SEQUENCE_COLUMN) - pl.col(_BOUNDARY_SEQUENCE_COLUMN).fill_null(0))
                .clip(upper_bound=self.config.max_frequency)
                .cast(pl.Int64)
                .alias(self.config.frequency_column)
            )
            .drop(
                _WINDOW_START_COLUMN,
                _BOUNDARY_TIME_COLUMN,
                _BOUNDARY_SEQUENCE_COLUMN,
            )
        )

    def _runner_candidates(self, contacts: pl.LazyFrame) -> pl.LazyFrame:
        columns = self.config.columns
        runner_keys = self.config.alternative_group_columns
        sort_columns = list(
            dict.fromkeys(
                [
                    *runner_keys,
                    "__valuestream_runner_rank_class",
                    _RANK_COLUMN,
                    columns.action,
                    columns.placement,
                    self.config.time.property,
                    _CONTACT_ORDER_COLUMN,
                ]
            )
        )
        runner = (
            contacts.filter(pl.col(_SEEN_CURRENT_COLUMN) & (pl.col(_RANK_COLUMN) > 1))
            .with_columns(
                pl.when(pl.col(_RANK_COLUMN) == 2)
                .then(0)
                .otherwise(1)
                .alias("__valuestream_runner_rank_class")
            )
            .sort(sort_columns)
            .group_by(runner_keys, maintain_order=True)
            .agg(
                pl.col(columns.propensity)
                .first()
                .cast(pl.Float64)
                .alias(_RUNNER_PROPENSITY_COLUMN),
                pl.lit(True).alias(_RUNNER_AVAILABLE_COLUMN),
                *(
                    [
                        pl.col(columns.priority)
                        .first()
                        .cast(pl.Float64)
                        .alias(_RUNNER_PRIORITY_COLUMN)
                    ]
                    if columns.priority is not None
                    else []
                ),
            )
        )
        if columns.priority is None:
            runner = runner.with_columns(
                pl.lit(None, dtype=pl.Float64).alias(_RUNNER_PRIORITY_COLUMN)
            )
        return runner

    def _with_virtual_columns(self, focal: pl.LazyFrame) -> pl.LazyFrame:
        columns = self.config.columns
        contact = pl.struct(
            columns.customer,
            columns.interaction,
            columns.action,
            columns.placement,
            _RANK_COLUMN,
        )
        runner_available = pl.col(_RUNNER_AVAILABLE_COLUMN).fill_null(False)
        comparable = runner_available & pl.col(_RUNNER_PROPENSITY_COLUMN).is_not_null()
        focal_priority = (
            pl.col(columns.priority).cast(pl.Float64)
            if columns.priority is not None
            else pl.lit(None, dtype=pl.Float64)
        )
        priority_comparable = (
            runner_available
            & focal_priority.is_not_null()
            & pl.col(_RUNNER_PRIORITY_COLUMN).is_not_null()
        )
        return focal.with_columns(
            pl.when(pl.col(_POSITIVE_COLUMN)).then(contact).otherwise(None).alias("ClickedContact"),
            pl.when(comparable).then(contact).otherwise(None).alias("ComparableContact"),
            pl.when(comparable & pl.col(_POSITIVE_COLUMN))
            .then(contact)
            .otherwise(None)
            .alias("ComparableClick"),
            pl.when(runner_available).then(contact).otherwise(None).alias("RunnerAvailable"),
            pl.col(_RUNNER_PROPENSITY_COLUMN).alias("RunnerPropensity"),
            pl.when(priority_comparable)
            .then(contact)
            .otherwise(None)
            .alias("PriorityComparableContact"),
            pl.when(priority_comparable)
            .then(focal_priority)
            .otherwise(None)
            .alias("FocalPriorityComparable"),
            pl.when(priority_comparable)
            .then(pl.col(_RUNNER_PRIORITY_COLUMN))
            .otherwise(None)
            .alias("RunnerPriorityComparable"),
        )

    def _agg_exprs(self, existing: set[str]) -> list[pl.Expr]:
        expressions: list[pl.Expr] = []
        for name, spec in self.state_specs.items():
            raw_extra = p3.spec_extra(spec)
            source_column = getattr(spec, "source_column", None)
            if source_column and source_column not in existing:
                raise ValueError(
                    f"frequency_response processor {self.id!r} state {name!r} "
                    f"requires missing enriched column {source_column!r}"
                )
            if spec.type == "count":
                expressions.append(p3.count_expr(raw_extra, alias=name))
            elif spec.type == "value_sum":
                assert source_column is not None
                expressions.append(p3.value_sum_expr(source_column, raw_extra, alias=name))
            else:  # Pydantic guards this, retained for defensive runtime errors.
                raise ValueError(
                    f"frequency_response processor {self.id!r} cannot build state type "
                    f"{spec.type!r}"
                )
        return expressions


def _dedupe_values(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


__all__ = [
    "DERIVED_GROUP_COLUMNS",
    "PROVENANCE_COLUMNS",
    "TARGET_CHUNK_COLUMN",
    "VIRTUAL_COLUMNS",
    "ChunkContext",
    "FrequencyResponseProcessor",
    "required_history_input_columns",
    "required_input_columns",
    "validate_current_input_schema",
]
