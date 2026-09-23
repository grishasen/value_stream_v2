"""Engagement by number of impressions over a marked current chunk plus lookback."""

from __future__ import annotations

import polars as pl

import valuestream.processors.processors_helper as p3
from valuestream.config import model
from valuestream.config.canonical import processor_config_hash
from valuestream.expr.references import column_references
from valuestream.expr.translator import translate
from valuestream.processors import frequency_fields, grain_levels
from valuestream.processors.context import (
    PROVENANCE_COLUMNS,
    TARGET_CHUNK_COLUMN,
    ChunkContext,
)
from valuestream.processors.outcomes import compatible_values, is_in_values
from valuestream.store.processor_state import HistoryProjectionSpec
from valuestream.utils.timer import timed

SCOPE_RANK_COLUMN = model.FREQUENCY_SCOPE_RANK_COLUMN
PRIOR_POSITIVE_COLUMN = model.FREQUENCY_PRIOR_POSITIVE_COLUMN
DERIVED_GROUP_COLUMNS = model.FREQUENCY_RESPONSE_DERIVED_COLUMNS

_RANK_COLUMN = frequency_fields.RANK_COLUMN
_ROW_ORDER_COLUMN = frequency_fields.ROW_ORDER_COLUMN
_DECISION_DAY_COLUMN = frequency_fields.DECISION_DAY_COLUMN
_PRIOR_EXPOSURES_COLUMN = frequency_fields.PRIOR_EXPOSURES_COLUMN
_PRIOR_POSITIVES_COLUMN = frequency_fields.PRIOR_POSITIVES_COLUMN
_POSITIVE_COLUMN = frequency_fields.POSITIVE_COLUMN
_SEEN_CURRENT_COLUMN = frequency_fields.SEEN_CURRENT_COLUMN
_SEEN_HISTORY_COLUMN = frequency_fields.SEEN_HISTORY_COLUMN
_CONTACT_ORDER_COLUMN = frequency_fields.CONTACT_ORDER_COLUMN
_EXPOSURE_SEQUENCE_COLUMN = frequency_fields.EXPOSURE_SEQUENCE_COLUMN
_POSITIVE_SEQUENCE_COLUMN = frequency_fields.POSITIVE_SEQUENCE_COLUMN
_WINDOW_START_COLUMN = frequency_fields.WINDOW_START_COLUMN
_BOUNDARY_TIME_COLUMN = frequency_fields.BOUNDARY_TIME_COLUMN
_BOUNDARY_SEQUENCE_COLUMN = frequency_fields.BOUNDARY_SEQUENCE_COLUMN
_BOUNDARY_POSITIVE_COLUMN = frequency_fields.BOUNDARY_POSITIVE_COLUMN
_CHECKPOINT_PARTITION_ORDER_COLUMN = frequency_fields.CHECKPOINT_PARTITION_ORDER_COLUMN
_CHECKPOINT_LOCAL_ORDER_COLUMN = frequency_fields.CHECKPOINT_LOCAL_ORDER_COLUMN


def required_input_columns(config: model.FrequencyResponseProcessor) -> frozenset[str]:
    """Return raw transformed-frame inputs required by ``config``.

    The number-of-impressions bucket, scope rank, prior-positive flag, and the
    daily calendar key are derived here and therefore excluded.
    """

    derived = config.derived_columns
    required = {
        config.time.property,
        TARGET_CHUNK_COLUMN,
        *config.columns.model_dump().values(),
        config.outcome.column,
        *config.scope_by,
        *(column for column in config.group_by if column not in derived),
    }
    required.update(column_references(config.filter))
    return frozenset(str(column) for column in required if str(column).strip())


def required_history_input_columns(
    config: model.FrequencyResponseProcessor,
) -> frozenset[str]:
    """Return the narrow transformed history schema a later target can depend on.

    History contributes impressions, earlier positive outcomes, and the ranks of
    a decision that straddles partitions; reporting dimensions never do. Raw
    processor-filter columns remain required because that filter applies
    before history is reduced to contacts.
    """

    required = {
        config.time.property,
        *config.columns.model_dump().values(),
        config.outcome.column,
        *config.scope_by,
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
    colliding = sorted(
        column for column in config.derived_columns - {"Day"} if column in schema.names()
    )
    if colliding:
        raise ValueError(
            f"frequency_response processor {config.id!r} cannot derive "
            f"{', '.join(repr(column) for column in colliding)} because the transformed "
            "source schema already contains it"
        )
    decision_dtype = schema[config.time.property]
    if decision_dtype.base_type() != pl.Datetime:
        raise TypeError(
            f"frequency_response processor {config.id!r} requires datetime decision "
            f"column {config.time.property!r}, got {decision_dtype}"
        )
    rank_dtype = schema[config.columns.rank]
    if not rank_dtype.is_integer():
        raise TypeError(
            f"frequency_response processor {config.id!r} requires integer rank "
            f"column {config.columns.rank!r}, got {rank_dtype}"
        )


class FrequencyResponseProcessor:
    """Build mergeable engagement states by number of impressions of the same action."""

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

    @property
    def _contact_keys(self) -> list[str]:
        return [*self.config.contact_identity_columns, _RANK_COLUMN]

    @timed
    def chunk_aggregate(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.DataFrame:
        """Return one daily partial for the marked target chunk."""

        return self.chunk_aggregate_lazy(frame, ctx).collect()

    @timed
    def chunk_aggregate_lazy(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.LazyFrame:
        """Build the deterministic lookback/contact aggregation as one lazy plan."""

        source_schema = frame.collect_schema()
        self._validate_input_schema(source_schema)
        if self.config.window_granularity == "daily":
            return self._chunk_aggregate_daily_lazy(frame, source_schema, ctx)
        contacts = self._prepare_contacts(frame, source_schema)
        return self._aggregate_contacts_lazy(contacts, ctx)

    def _chunk_aggregate_daily_lazy(
        self,
        frame: pl.LazyFrame,
        source_schema: pl.Schema,
        ctx: ChunkContext,
    ) -> pl.LazyFrame:
        """Aggregate one marked chunk with day-granular impression history.

        Unlike the exact path, history rows never merge with current contacts:
        they only contribute per-day impression and positive counters. The
        current day itself is still normalized and sequenced exactly.
        """

        time_column = self.config.time.property
        prepared = self._prepare_checkpoint_rows(frame, source_schema)
        canonical = self._canonicalize_dictionary_columns(
            self._canonicalize_decision_time(prepared, source_schema[time_column])
        )
        dated = canonical.with_columns(pl.col(time_column).dt.date().alias(_DECISION_DAY_COLUMN))
        current = dated.filter(pl.col(TARGET_CHUNK_COLUMN)).drop(_DECISION_DAY_COLUMN)
        history = dated.filter(~pl.col(TARGET_CHUNK_COLUMN))
        counters = self._daily_history_counters(history)
        contacts = self._normalize_contacts(current)
        return self._aggregate_daily_contacts_lazy(contacts, counters, ctx)

    def checkpoint_contacts_lazy(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        """Prepare one current source chunk for rolling checkpoint staging.

        The temporary current table retains the filtered, projected impression
        rows because cross-partition contact normalization must see duplicates
        before collapsing them. Their normalized projection is committed as
        bounded history only after the target calculation succeeds.
        """

        source_schema = frame.collect_schema()
        validate_current_input_schema(self.config, source_schema)
        marked = frame.with_columns(pl.lit(True).alias(TARGET_CHUNK_COLUMN))
        marked_schema = marked.collect_schema()
        prepared = self._prepare_checkpoint_rows(marked, marked_schema)
        canonical = self._canonicalize_decision_time(
            prepared,
            source_schema[self.config.time.property],
        )
        if self.config.window_granularity == "daily":
            # Day-granular history is keyed by the canonical UTC calendar day,
            # so it must be a physical staged column for the commit projection.
            canonical = canonical.with_columns(
                pl.col(self.config.time.property).dt.date().alias(_DECISION_DAY_COLUMN)
            )
        return canonical

    def checkpoint_history_contacts_lazy(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        """Project staged target rows to the exact bounded-history payload.

        Mirrors the SQL commit projection: rows collapse to one row per contact
        with the earliest decision time and source order and an or-combined
        positive flag. MIN/BOOL_OR are associative, so per-day normalized rows
        merge across days exactly like the raw rows they replace.
        """

        projection = self.checkpoint_history_projection()
        schema = frame.collect_schema()
        missing = sorted(set(projection.columns) - set(schema.names()))
        if missing:
            raise ValueError(
                f"frequency_response processor {self.id!r} target checkpoint is missing "
                f"history column(s): {', '.join(missing)}"
            )
        aggregations: list[pl.Expr] = []
        for column in projection.columns:
            if column in projection.min_columns:
                aggregations.append(pl.col(column).min().alias(column))
            elif column in projection.bool_or_columns:
                aggregations.append(pl.col(column).any().alias(column))
        return (
            frame.group_by(list(projection.key_columns), maintain_order=True)
            .agg(aggregations)
            .select(projection.columns)
        )

    def checkpoint_history_projection(self) -> HistoryProjectionSpec:
        """Return the persisted normalized history contract."""

        identity = self.config.contact_identity_columns
        if self.config.window_granularity == "daily":
            # Day-granular history needs only contact identity and response per
            # calendar day: the SQL plan reduces it to per-day counters, and
            # exact time-of-day, rank, and source order never affect a later
            # target.
            return HistoryProjectionSpec(
                columns=tuple(dict.fromkeys([*identity, _DECISION_DAY_COLUMN, _POSITIVE_COLUMN])),
                bool_or_columns=(_POSITIVE_COLUMN,),
            )
        return HistoryProjectionSpec(
            columns=tuple(
                dict.fromkeys(
                    [
                        *identity,
                        self.config.time.property,
                        _ROW_ORDER_COLUMN,
                        _RANK_COLUMN,
                        _POSITIVE_COLUMN,
                    ]
                )
            ),
            min_columns=(self.config.time.property, _ROW_ORDER_COLUMN),
            bool_or_columns=(_POSITIVE_COLUMN,),
        )

    @timed
    def checkpoint_aggregate_lazy(
        self,
        current: pl.LazyFrame,
        history: list[pl.LazyFrame],
        ctx: ChunkContext,
    ) -> pl.LazyFrame:
        """Reference aggregation over prepared current and history chunks."""

        if self.config.window_granularity == "daily":
            return self._checkpoint_aggregate_daily_lazy(current, history, ctx)
        prepared: list[pl.LazyFrame] = [
            historical.with_columns(
                pl.lit(False).alias(TARGET_CHUNK_COLUMN),
                pl.lit(partition_order).alias(_CHECKPOINT_PARTITION_ORDER_COLUMN),
            )
            for partition_order, historical in enumerate(history)
        ]
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
        contacts = self._canonicalize_dictionary_columns(contacts)
        checkpoint_schema = contacts.collect_schema()
        missing = sorted(
            {
                *self.config.contact_identity_columns,
                self.config.time.property,
                TARGET_CHUNK_COLUMN,
                _RANK_COLUMN,
                _ROW_ORDER_COLUMN,
                _POSITIVE_COLUMN,
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

    def _checkpoint_aggregate_daily_lazy(
        self,
        current: pl.LazyFrame,
        history: list[pl.LazyFrame],
        ctx: ChunkContext,
    ) -> pl.LazyFrame:
        """Reference daily aggregation over a staged current chunk and history.

        ``history`` frames must carry the day-granular projection columns
        (contact identity, canonical decision day, and positive flag).
        """

        marked = current.with_columns(pl.lit(True).alias(TARGET_CHUNK_COLUMN))
        schema = marked.collect_schema()
        if _DECISION_DAY_COLUMN in schema.names():
            marked = marked.drop(_DECISION_DAY_COLUMN)
        contacts = self._normalize_contacts(self._canonicalize_dictionary_columns(marked))
        counters: pl.LazyFrame | None = None
        if history:
            counters = self._daily_history_counters(
                self._canonicalize_dictionary_columns(pl.concat(history, how="diagonal_relaxed"))
            )
        return self._aggregate_daily_contacts_lazy(contacts, counters, ctx)

    def _daily_history_counters(self, history: pl.LazyFrame) -> pl.LazyFrame:
        """Reduce history rows to per-day impression and positive counters."""

        contact_identity = [*self.config.contact_identity_columns, _DECISION_DAY_COLUMN]
        counter_keys = [*self.config.exposure_key_columns, _DECISION_DAY_COLUMN]
        return (
            history.group_by(contact_identity)
            .agg(pl.col(_POSITIVE_COLUMN).any())
            .group_by(counter_keys)
            .agg(
                pl.len().cast(pl.Int64).alias(_PRIOR_EXPOSURES_COLUMN),
                pl.col(_POSITIVE_COLUMN).sum().cast(pl.Int64).alias(_PRIOR_POSITIVES_COLUMN),
            )
        )

    def _aggregate_daily_contacts_lazy(
        self,
        contacts: pl.LazyFrame,
        counters: pl.LazyFrame | None,
        ctx: ChunkContext,
    ) -> pl.LazyFrame:
        """Build daily additive states from current contacts and day counters."""

        exposures = self._with_scope_rank(self._with_daily_exposure_frequency(contacts, counters))
        focal = exposures.filter(pl.col(_SEEN_CURRENT_COLUMN) & ~pl.col(_SEEN_HISTORY_COLUMN))
        return self.aggregate_focal_lazy(focal, ctx)

    def _with_daily_exposure_frequency(
        self,
        contacts: pl.LazyFrame,
        counters: pl.LazyFrame | None,
    ) -> pl.LazyFrame:
        """Bucket = prior full-day impressions plus the exact intra-day sequence."""

        columns = self.config.columns
        time_column = self.config.time.property
        exposure_keys = self.config.exposure_key_columns
        window_days = self.config.window_days
        day_keys = [*exposure_keys, _DECISION_DAY_COLUMN]
        ordered = (
            contacts.with_columns(pl.col(time_column).dt.date().alias(_DECISION_DAY_COLUMN))
            .sort([*day_keys, time_column, columns.interaction, _CONTACT_ORDER_COLUMN])
            .with_columns(
                pl.col(time_column)
                .cum_count()
                .over(day_keys)
                .cast(pl.Int64)
                .alias(_EXPOSURE_SEQUENCE_COLUMN),
                pl.col(_POSITIVE_COLUMN)
                .cast(pl.Int64)
                .cum_sum()
                .over(day_keys)
                .alias(_POSITIVE_SEQUENCE_COLUMN),
            )
        )
        earlier_positives = pl.col(_POSITIVE_SEQUENCE_COLUMN) - pl.col(_POSITIVE_COLUMN).cast(
            pl.Int64
        )
        sequence = pl.col(_EXPOSURE_SEQUENCE_COLUMN)
        derived_columns = [
            _DECISION_DAY_COLUMN,
            _EXPOSURE_SEQUENCE_COLUMN,
            _POSITIVE_SEQUENCE_COLUMN,
        ]
        if counters is None:
            return ordered.with_columns(
                sequence.clip(upper_bound=self.config.max_frequency)
                .cast(pl.Int64)
                .alias(self.config.frequency_column),
                (earlier_positives > 0).alias(PRIOR_POSITIVE_COLUMN),
            ).drop(derived_columns)
        counter_day = "__valuestream_frequency_counter_day"
        # Date columns compare as days-since-epoch integers, mirroring the SQL
        # ``counter_day > focal_day - window AND counter_day < focal_day``.
        prior = (
            ordered.select(day_keys)
            .unique()
            .join(
                counters.rename({_DECISION_DAY_COLUMN: counter_day}),
                on=exposure_keys,
                how="inner",
            )
            .filter(
                (
                    pl.col(counter_day).cast(pl.Int32)
                    > pl.col(_DECISION_DAY_COLUMN).cast(pl.Int32) - window_days
                )
                & (pl.col(counter_day) < pl.col(_DECISION_DAY_COLUMN))
            )
            .group_by(day_keys)
            .agg(
                pl.col(_PRIOR_EXPOSURES_COLUMN).sum(),
                pl.col(_PRIOR_POSITIVES_COLUMN).sum(),
            )
        )
        return (
            ordered.join(prior, on=day_keys, how="left")
            .with_columns(
                (sequence + pl.col(_PRIOR_EXPOSURES_COLUMN).fill_null(0))
                .clip(upper_bound=self.config.max_frequency)
                .cast(pl.Int64)
                .alias(self.config.frequency_column),
                ((earlier_positives + pl.col(_PRIOR_POSITIVES_COLUMN).fill_null(0)) > 0).alias(
                    PRIOR_POSITIVE_COLUMN
                ),
            )
            .drop([*derived_columns, _PRIOR_EXPOSURES_COLUMN, _PRIOR_POSITIVES_COLUMN])
        )

    def _prepare_contacts(self, frame: pl.LazyFrame, source_schema: pl.Schema) -> pl.LazyFrame:
        """Filter, classify, project, and normalize transformed source contacts."""

        prepared = self._prepare_checkpoint_rows(frame, source_schema)
        canonical = self._canonicalize_decision_time(
            prepared,
            source_schema[self.config.time.property],
        )
        return self._normalize_contacts(self._canonicalize_dictionary_columns(canonical))

    def _canonicalize_decision_time(
        self,
        frame: pl.LazyFrame,
        dtype: pl.DataType,
    ) -> pl.LazyFrame:
        """Represent decision instants identically in source-scan and DuckDB modes."""

        time_column = self.config.time.property
        timestamp = pl.col(time_column)
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
            # Convert aware instants to naive UTC before Arrow streaming; the
            # aggregation tail interprets this canonical representation as UTC.
            timestamp = timestamp.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        # Sub-second precision is not semantic for frequency windows. Whole
        # seconds are exact in both engines, so `time - INTERVAL` arithmetic and
        # ASOF comparisons in DuckDB match Polars bit-for-bit, while raw inputs
        # of any datetime unit canonicalize to one identical representation.
        return frame.with_columns(
            timestamp.dt.truncate("1s").cast(pl.Datetime("us")).alias(time_column)
        )

    @staticmethod
    def _canonicalize_dictionary_columns(frame: pl.LazyFrame) -> pl.LazyFrame:
        """Match the Arrow/DuckDB VARCHAR representation of dictionary columns."""

        schema = frame.collect_schema()
        expressions = [
            pl.col(name).cast(pl.String).alias(name)
            for name, dtype in schema.items()
            if dtype == pl.Categorical or isinstance(dtype, pl.Enum)
        ]
        return frame.with_columns(*expressions) if expressions else frame

    def _prepare_checkpoint_rows(
        self,
        frame: pl.LazyFrame,
        source_schema: pl.Schema,
    ) -> pl.LazyFrame:
        """Return the narrow classified impression rows persisted by a checkpoint."""

        source = frame
        if self.config.filter is not None:
            source = source.filter(translate(self.config.filter))

        columns = self.config.columns
        outcome = self.config.outcome
        outcome_dtype = source_schema[outcome.column]
        positive_values = compatible_values(outcome.positive_values, outcome_dtype)
        negative_values = compatible_values(outcome.negative_values, outcome_dtype)
        source = (
            source.filter(is_in_values(outcome.column, [*positive_values, *negative_values]))
            .with_columns(
                pl.col(TARGET_CHUNK_COLUMN).fill_null(False).alias(TARGET_CHUNK_COLUMN),
                pl.col(columns.rank).cast(pl.Int64).alias(_RANK_COLUMN),
                is_in_values(outcome.column, positive_values).alias(_POSITIVE_COLUMN),
            )
            # Rows missing a contact identity, scope value, or decision time can
            # never contribute after normalization. Drop them before checkpoint
            # sharding as well, so a null customer is excluded rather than
            # turning persistent mode into an input error.
            .filter(
                pl.all_horizontal(
                    pl.col(column).is_not_null()
                    for column in (
                        *self.config.contact_identity_columns,
                        columns.rank,
                        self.config.time.property,
                    )
                )
            )
        )
        if self.config.customer_sample is not None:
            # Deterministic customer subsample, applied identically in
            # source-scan and rolling modes and to current and history rows
            # alike, so a sampled customer's contact set stays complete.
            source = source.filter(
                pl.col(columns.customer)
                .cast(pl.String)
                .hash(*model.FREQUENCY_CUSTOMER_SAMPLE_SEEDS)
                % pl.lit(model.FREQUENCY_CUSTOMER_SAMPLE_MODULUS, dtype=pl.UInt64)
                < pl.lit(
                    self.config.customer_sample.sample_threshold,
                    dtype=pl.UInt64,
                )
            )
        source = source.with_row_index(_ROW_ORDER_COLUMN)

        projected = set(required_input_columns(self.config))
        projected.update(
            {
                outcome.column,
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
        )

    def _aggregate_contacts_lazy(
        self,
        contacts: pl.LazyFrame,
        ctx: ChunkContext,
    ) -> pl.LazyFrame:
        """Build daily additive states from normalized current/history contacts."""

        exposures = self._with_scope_rank(self._with_exposure_frequency(contacts))
        focal = exposures.filter(pl.col(_SEEN_CURRENT_COLUMN) & ~pl.col(_SEEN_HISTORY_COLUMN))
        return self.aggregate_focal_lazy(focal, ctx)

    def aggregate_focal_lazy(
        self,
        focal: pl.LazyFrame,
        ctx: ChunkContext,
    ) -> pl.LazyFrame:
        """Aggregate already-enriched focal contacts into daily additive states."""

        focal_schema = focal.collect_schema()
        time_column = self.config.time.property
        if time_column not in focal_schema.names():
            raise ValueError(
                f"frequency_response processor {self.id!r} focal rows are missing "
                f"decision-time column {time_column!r}"
            )
        focal = focal.with_columns(self._decision_day_expr(focal_schema[time_column]).alias("Day"))

        group_keys = list(dict.fromkeys([*self.group_by_columns, "Day"]))
        enriched_columns = set(focal.collect_schema().names())
        missing_groups = sorted(set(group_keys) - enriched_columns)
        if missing_groups:
            raise ValueError(
                f"frequency_response processor {self.id!r} cannot derive group-by "
                f"column(s): {', '.join(missing_groups)}"
            )
        grouped = focal.group_by(group_keys).agg(self._agg_exprs())
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
        """Merge count partials with the generic state algebra."""

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
        """Collapse repeated outcome rows to one contact; a positive outcome wins."""

        time_column = self.config.time.property
        contact_keys = self._contact_keys
        source = source.filter(
            pl.all_horizontal(pl.col(key).is_not_null() for key in contact_keys),
            pl.col(time_column).is_not_null(),
        )
        sort_columns = [
            *contact_keys,
            _POSITIVE_COLUMN,
            TARGET_CHUNK_COLUMN,
            time_column,
            _ROW_ORDER_COLUMN,
        ]
        source = source.sort(
            sort_columns,
            descending=[False] * len(contact_keys) + [True, True, False, False],
        )
        passthrough = [
            name
            for name in source.collect_schema().names()
            if name
            not in {
                *contact_keys,
                TARGET_CHUNK_COLUMN,
                _ROW_ORDER_COLUMN,
                _POSITIVE_COLUMN,
                time_column,
            }
        ]
        return source.group_by(contact_keys, maintain_order=True).agg(
            pl.col(time_column).min().alias(time_column),
            pl.col(_POSITIVE_COLUMN).max().alias(_POSITIVE_COLUMN),
            pl.col(TARGET_CHUNK_COLUMN).any().alias(_SEEN_CURRENT_COLUMN),
            (~pl.col(TARGET_CHUNK_COLUMN)).any().alias(_SEEN_HISTORY_COLUMN),
            pl.col(_ROW_ORDER_COLUMN).min().alias(_CONTACT_ORDER_COLUMN),
            *(pl.col(name).first().alias(name) for name in passthrough),
        )

    def _with_exposure_frequency(self, contacts: pl.LazyFrame) -> pl.LazyFrame:
        """Number of impressions and earlier positives in ``(t - window, t]``.

        Both come from the same running sequence of the action's impressions to
        the customer inside the scope, minus the running totals at the last
        impression on or before the window start.
        """

        columns = self.config.columns
        time_column = self.config.time.property
        time_dtype = contacts.collect_schema()[time_column]
        time_unit = time_dtype.time_unit if isinstance(time_dtype, pl.Datetime) else None
        exposure_keys = self.config.exposure_key_columns
        ordered = contacts.sort(
            [*exposure_keys, time_column, columns.interaction, _CONTACT_ORDER_COLUMN]
        ).with_columns(
            pl.col(time_column)
            .cum_count()
            .over(exposure_keys)
            .cast(pl.Int64)
            .alias(_EXPOSURE_SEQUENCE_COLUMN),
            pl.col(_POSITIVE_COLUMN)
            .cast(pl.Int64)
            .cum_sum()
            .over(exposure_keys)
            .alias(_POSITIVE_SEQUENCE_COLUMN),
        )
        left = ordered.with_columns(
            (
                pl.col(time_column)
                - pl.duration(hours=self.config.window_hours, time_unit=time_unit)
            ).alias(_WINDOW_START_COLUMN)
        ).sort([*exposure_keys, _WINDOW_START_COLUMN, columns.interaction, _CONTACT_ORDER_COLUMN])
        right = ordered.select(
            *exposure_keys,
            pl.col(time_column).alias(_BOUNDARY_TIME_COLUMN),
            pl.col(_EXPOSURE_SEQUENCE_COLUMN).alias(_BOUNDARY_SEQUENCE_COLUMN),
            pl.col(_POSITIVE_SEQUENCE_COLUMN).alias(_BOUNDARY_POSITIVE_COLUMN),
        ).sort([*exposure_keys, _BOUNDARY_TIME_COLUMN, _BOUNDARY_SEQUENCE_COLUMN])
        earlier_positives = (
            pl.col(_POSITIVE_SEQUENCE_COLUMN)
            - pl.col(_POSITIVE_COLUMN).cast(pl.Int64)
            - pl.col(_BOUNDARY_POSITIVE_COLUMN).fill_null(0)
        )
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
                .alias(self.config.frequency_column),
                (earlier_positives > 0).alias(PRIOR_POSITIVE_COLUMN),
            )
            .drop(
                _WINDOW_START_COLUMN,
                _BOUNDARY_TIME_COLUMN,
                _BOUNDARY_SEQUENCE_COLUMN,
                _BOUNDARY_POSITIVE_COLUMN,
                _EXPOSURE_SEQUENCE_COLUMN,
                _POSITIVE_SEQUENCE_COLUMN,
            )
        )

    def _with_scope_rank(self, contacts: pl.LazyFrame) -> pl.LazyFrame:
        """Rank each shown action among the decision's actions inside its scope."""

        return contacts.with_columns(
            pl.col(_RANK_COLUMN)
            .rank("dense")
            .over(self.config.rank_partition_columns)
            .cast(pl.Int64)
            .clip(upper_bound=self.config.max_rank)
            .alias(SCOPE_RANK_COLUMN)
        )

    def _agg_exprs(self) -> list[pl.Expr]:
        positive = pl.col(_POSITIVE_COLUMN)
        expressions: list[pl.Expr] = []
        for name, spec in self.state_specs.items():
            selector = getattr(spec, "outcome", None)
            if spec.type != "count" or selector not in {"positive", "negative"}:
                # The model rejects this; retained for defensive runtime errors.
                raise ValueError(
                    f"frequency_response processor {self.id!r} cannot build state {name!r}"
                )
            counted = positive if selector == "positive" else ~positive
            expressions.append(counted.sum().cast(pl.Int64).alias(name))
        return expressions


__all__ = [
    "DERIVED_GROUP_COLUMNS",
    "PRIOR_POSITIVE_COLUMN",
    "PROVENANCE_COLUMNS",
    "SCOPE_RANK_COLUMN",
    "TARGET_CHUNK_COLUMN",
    "ChunkContext",
    "FrequencyResponseProcessor",
    "required_history_input_columns",
    "required_input_columns",
    "validate_current_input_schema",
]
