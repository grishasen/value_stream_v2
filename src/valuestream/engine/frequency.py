"""Native DuckDB execution over rolling frequency-response checkpoint state."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from types import TracebackType

import duckdb
import polars as pl
import pyarrow as pa

from valuestream.config import model
from valuestream.processors.context import TARGET_CHUNK_COLUMN
from valuestream.processors.frequency_fields import (
    BOUNDARY_POSITIVE_COLUMN,
    BOUNDARY_SEQUENCE_COLUMN,
    BOUNDARY_TIME_COLUMN,
    CHECKPOINT_LOCAL_ORDER_COLUMN,
    CHECKPOINT_SHARD_COLUMN,
    CONTACT_ORDER_COLUMN,
    DECISION_DAY_COLUMN,
    EXPOSURE_SEQUENCE_COLUMN,
    POSITIVE_COLUMN,
    POSITIVE_SEQUENCE_COLUMN,
    PRIOR_EXPOSURES_COLUMN,
    PRIOR_POSITIVES_COLUMN,
    RANK_COLUMN,
    ROW_ORDER_COLUMN,
    SCOPE_RANK_SEQUENCE_COLUMN,
    SEEN_CURRENT_COLUMN,
    SEEN_HISTORY_COLUMN,
    WINDOW_START_COLUMN,
)

_DEFAULT_HISTORY_TABLE = "history"
_DEFAULT_CURRENT_TABLE = "current"


class DuckDBFrequencySession:
    """Stream one focal shard from a staged rolling checkpoint through SQL.

    The returned Polars lazy frame owns a DuckDB Arrow stream.  It therefore has
    to be consumed before this context manager exits and before the rolling store
    is mutated.  The supplied connection remains owned by the rolling store and
    is deliberately not closed by this session.
    """

    def __init__(
        self,
        *,
        config: model.FrequencyResponseProcessor,
        connection: duckdb.DuckDBPyConnection,
        current_chunk_id: str,
        history_table: str = _DEFAULT_HISTORY_TABLE,
        current_table: str = _DEFAULT_CURRENT_TABLE,
        shard_column: str = CHECKPOINT_SHARD_COLUMN,
        chunk_id_column: str = "__valuestream_checkpoint_chunk_id",
        batch_size: int = 250_000,
    ) -> None:
        if not current_chunk_id:
            raise ValueError("frequency checkpoint current chunk ID must not be empty")
        if not history_table or not current_table:
            raise ValueError("frequency checkpoint table names must not be empty")
        if not shard_column:
            raise ValueError("frequency checkpoint shard column must not be empty")
        if not chunk_id_column:
            raise ValueError("frequency checkpoint chunk ID column must not be empty")
        if batch_size < 1:
            raise ValueError("frequency checkpoint batch_size must be positive")
        self.config = config
        self.connection = connection
        self.current_chunk_id = current_chunk_id
        self.history_table = history_table
        self.current_table = current_table
        self.shard_column = shard_column
        self.chunk_id_column = chunk_id_column
        self.batch_size = batch_size
        self._entered = False
        self._current_columns: tuple[str, ...] = ()
        self._history_columns: tuple[str, ...] = ()

    def __enter__(self) -> DuckDBFrequencySession:
        if self._entered:
            raise RuntimeError("DuckDB frequency session is already open")
        # TIMESTAMPTZ -> Arrow conversion otherwise inherits the operator's
        # local zone, which can change timestamp dtypes across workers.
        self.connection.execute("SET TimeZone='UTC'")
        self._history_columns = self._relation_columns(self.history_table)
        self._current_columns = self._relation_columns(self.current_table)
        self._validate_checkpoint_schemas()
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self._entered = False
        self._current_columns = ()
        self._history_columns = ()

    def focal_lazy(self, shard_id: int) -> pl.LazyFrame:
        """Return exact focal rows for one persisted customer-hash shard."""

        relation = self.connection.sql(self._focal_shard_sql(shard_id))
        return relation.pl(lazy=True, batch_size=self.batch_size)

    def focal_table(self, shard_id: int) -> pa.Table:
        """Materialize one shard's exact focal rows as an Arrow table.

        Unlike :meth:`focal_lazy`, the result does not borrow a DuckDB stream:
        once returned, the connection is free for the next query, so a caller
        can fetch the next shard while a previous shard's Polars tail runs.
        ``to_arrow_table`` (not ``.arrow()``) keeps the schema when a shard
        yields zero focal rows.
        """

        return self.connection.sql(self._focal_shard_sql(shard_id)).to_arrow_table()

    def _focal_shard_sql(self, shard_id: int) -> str:
        if not self._entered:
            raise RuntimeError("DuckDB frequency session must be entered before querying")
        if isinstance(shard_id, bool) or not isinstance(shard_id, int) or shard_id < 0:
            raise ValueError("frequency checkpoint shard_id must be a non-negative integer")
        return _focal_sql(
            self.config,
            current_columns=self._current_columns,
            history_columns=self._history_columns,
            history_table=self.history_table,
            current_table=self.current_table,
            shard_column=self.shard_column,
            chunk_id_column=self.chunk_id_column,
            current_chunk_id=self.current_chunk_id,
            shard_id=shard_id,
        )

    def _relation_columns(
        self,
        table: str,
    ) -> tuple[str, ...]:
        rows = self.connection.sql(f"DESCRIBE SELECT * FROM {_quote_identifier(table)}").fetchall()
        return tuple(str(row[0]) for row in rows)

    def _validate_checkpoint_schemas(self) -> None:
        identity = self.config.contact_identity_columns
        common = {self.shard_column, self.chunk_id_column, *identity, POSITIVE_COLUMN}
        current_required = {
            *common,
            self.config.time.property,
            ROW_ORDER_COLUMN,
            RANK_COLUMN,
        }
        missing_current = sorted(current_required - set(self._current_columns))
        if missing_current:
            raise ValueError(
                f"frequency_response processor {self.config.id!r} target checkpoint is "
                f"missing column(s): {', '.join(missing_current)}"
            )
        if self.config.window_granularity == "daily":
            history_required = {*common, DECISION_DAY_COLUMN}
        else:
            history_required = current_required
        missing_history = sorted(history_required - set(self._history_columns))
        if missing_history:
            raise ValueError(
                f"frequency_response processor {self.config.id!r} rolling history is "
                f"missing column(s): {', '.join(missing_history)}"
            )


def _focal_sql(
    config: model.FrequencyResponseProcessor,
    *,
    current_columns: Sequence[str],
    history_columns: Sequence[str],
    history_table: str,
    current_table: str,
    shard_column: str,
    chunk_id_column: str,
    current_chunk_id: str,
    shard_id: int,
) -> str:
    """Build the exact contact-normalization, window, and scope-rank SQL plan."""

    if config.window_granularity == "daily":
        return _daily_focal_sql(
            config,
            current_columns=current_columns,
            history_table=history_table,
            current_table=current_table,
            shard_column=shard_column,
            chunk_id_column=chunk_id_column,
            current_chunk_id=current_chunk_id,
            shard_id=shard_id,
        )
    relation_specs = [
        (history_table, tuple(history_columns), False),
        (current_table, tuple(current_columns), True),
    ]
    union_members = [
        _checkpoint_member_sql(
            table,
            columns,
            is_current=is_current,
            shard_column=shard_column,
            chunk_id_column=chunk_id_column,
            current_chunk_id=current_chunk_id,
            shard_id=shard_id,
        )
        for table, columns, is_current in relation_specs
    ]
    union_sql = "\nUNION ALL BY NAME\n".join(f"({member})" for member in union_members)
    union_columns = _union_columns(
        [column for _table, columns, _is_current in relation_specs for column in columns],
        shard_column=shard_column,
        chunk_id_column=chunk_id_column,
    )

    time_column = config.time.property
    contact_keys = [*config.contact_identity_columns, RANK_COLUMN]
    normalized_select = _normalized_select_sql(
        config,
        contact_keys,
        _passthrough_columns(config, union_columns, contact_keys),
    )
    exposure_keys = config.exposure_key_columns
    boundary_join = "\n        AND ".join(
        f"{_qualified_identifier('windowed_contact', key)} = "
        f"{_qualified_identifier('boundary', key)}"
        for key in exposure_keys
    )
    non_null_contacts = "\n      AND ".join(
        f"{_quote_identifier(column)} IS NOT NULL" for column in [*contact_keys, time_column]
    )
    sequence = _quote_identifier(EXPOSURE_SEQUENCE_COLUMN)
    positives = _quote_identifier(POSITIVE_SEQUENCE_COLUMN)
    scope_rank = _quote_identifier(SCOPE_RANK_SEQUENCE_COLUMN)
    positive = _quote_identifier(POSITIVE_COLUMN)
    need_prior = model.FREQUENCY_PRIOR_POSITIVE_COLUMN in config.group_by
    need_rank = model.FREQUENCY_SCOPE_RANK_COLUMN in config.group_by
    boundary_positive = (
        f",\n        MAX({_qualified_identifier('ordered_contact', POSITIVE_SEQUENCE_COLUMN)})"
        f" AS {_quote_identifier(BOUNDARY_POSITIVE_COLUMN)}"
        if need_prior
        else ""
    )
    excluded = [_quote_identifier(WINDOW_START_COLUMN), sequence]
    if need_prior:
        excluded.append(positives)
    if need_rank:
        excluded.append(scope_rank)
    derived = [
        f"CAST(LEAST(windowed_contact.{sequence} - "
        f"COALESCE(boundary.{_quote_identifier(BOUNDARY_SEQUENCE_COLUMN)}, 0), "
        f"{config.max_frequency}) AS BIGINT) AS {_quote_identifier(config.frequency_column)}"
    ]
    if need_prior:
        derived.append(
            f"(windowed_contact.{positives} - CAST(windowed_contact.{positive} AS BIGINT) "
            f"- COALESCE(boundary.{_quote_identifier(BOUNDARY_POSITIVE_COLUMN)}, 0)) > 0 "
            f"AS {_quote_identifier(model.FREQUENCY_PRIOR_POSITIVE_COLUMN)}"
        )
    if need_rank:
        derived.append(
            f"CAST(LEAST(windowed_contact.{scope_rank}, {config.max_rank}) AS BIGINT) "
            f"AS {_quote_identifier(model.FREQUENCY_SCOPE_RANK_COLUMN)}"
        )

    return f"""
WITH checkpoint_union AS (
{_indent(union_sql, 4)}
),
sequenced_contacts AS (
    SELECT
        * EXCLUDE (
            {_quote_identifier(chunk_id_column)},
            {_quote_identifier(CHECKPOINT_LOCAL_ORDER_COLUMN)}
        ),
        CAST(
            ROW_NUMBER() OVER (
                ORDER BY
                    {_quote_identifier(chunk_id_column)},
                    {_quote_identifier(CHECKPOINT_LOCAL_ORDER_COLUMN)}
            ) - 1
            AS BIGINT
        ) AS {_quote_identifier(ROW_ORDER_COLUMN)}
    FROM checkpoint_union
),
normalized_contacts AS MATERIALIZED (
    SELECT
{_indent(normalized_select, 8)}
    FROM sequenced_contacts
    WHERE {non_null_contacts}
    GROUP BY {_identifier_csv(contact_keys)}
),
ordered_exposures AS MATERIALIZED (
    SELECT
        contact.*,
{_indent(_running_totals_sql(config, partition_by=_identifier_csv(exposure_keys)), 8)}
    FROM normalized_contacts AS contact
),
boundary_points AS (
    SELECT
        {_qualified_identifier_csv("ordered_contact", exposure_keys)},
        {_qualified_identifier("ordered_contact", time_column)}
            AS {_quote_identifier(BOUNDARY_TIME_COLUMN)},
        MAX({_qualified_identifier("ordered_contact", EXPOSURE_SEQUENCE_COLUMN)})
            AS {_quote_identifier(BOUNDARY_SEQUENCE_COLUMN)}{boundary_positive}
    FROM ordered_exposures AS ordered_contact
    GROUP BY {_identifier_csv([*exposure_keys, time_column])}
),
windowed_exposures AS (
    SELECT
        ordered_contact.*,
        {_qualified_identifier("ordered_contact", time_column)}
            - INTERVAL {config.window_hours} HOUR
            AS {_quote_identifier(WINDOW_START_COLUMN)}
    FROM ordered_exposures AS ordered_contact
),
exposures AS (
    SELECT
        windowed_contact.* EXCLUDE ({", ".join(excluded)}),
        {", ".join(derived)}
    FROM windowed_exposures AS windowed_contact
    ASOF LEFT JOIN boundary_points AS boundary
        ON {boundary_join}
        AND windowed_contact.{_quote_identifier(WINDOW_START_COLUMN)}
            >= boundary.{_quote_identifier(BOUNDARY_TIME_COLUMN)}
)
SELECT *
FROM exposures
WHERE {_quote_identifier(SEEN_CURRENT_COLUMN)}
  AND NOT {_quote_identifier(SEEN_HISTORY_COLUMN)}
""".strip()


def _daily_focal_sql(
    config: model.FrequencyResponseProcessor,
    *,
    current_columns: Sequence[str],
    history_table: str,
    current_table: str,
    shard_column: str,
    chunk_id_column: str,
    current_chunk_id: str,
    shard_id: int,
) -> str:
    """Build the day-granular frequency SQL plan.

    History never joins the contact union: it is reduced to per-day impression
    and positive counters, and each focal contact adds its exact intra-day
    sequence to the counters of the previous ``window_days - 1`` calendar days.
    """

    current_member = _checkpoint_member_sql(
        current_table,
        tuple(current_columns),
        is_current=True,
        shard_column=shard_column,
        chunk_id_column=chunk_id_column,
        current_chunk_id=current_chunk_id,
        shard_id=shard_id,
    )
    union_columns = _union_columns(
        current_columns,
        shard_column=shard_column,
        chunk_id_column=chunk_id_column,
    )
    time_column = config.time.property
    contact_keys = [*config.contact_identity_columns, RANK_COLUMN]
    normalized_select = _normalized_select_sql(
        config,
        contact_keys,
        [
            column
            for column in _passthrough_columns(config, union_columns, contact_keys)
            if column != DECISION_DAY_COLUMN
        ],
    )
    exposure_keys = config.exposure_key_columns
    day = _quote_identifier(DECISION_DAY_COLUMN)
    prior_exposures = _quote_identifier(PRIOR_EXPOSURES_COLUMN)
    prior_positives = _quote_identifier(PRIOR_POSITIVES_COLUMN)
    positive = _quote_identifier(POSITIVE_COLUMN)
    sequence = _quote_identifier(EXPOSURE_SEQUENCE_COLUMN)
    positives = _quote_identifier(POSITIVE_SEQUENCE_COLUMN)
    scope_rank = _quote_identifier(SCOPE_RANK_SEQUENCE_COLUMN)
    history_identity = _identifier_csv([*config.contact_identity_columns, DECISION_DAY_COLUMN])
    counter_keys = _identifier_csv([*exposure_keys, DECISION_DAY_COLUMN])
    focal_keys = _qualified_identifier_csv("focal", exposure_keys)
    prior_join = "\n        AND ".join(
        f"{_qualified_identifier('focal', key)} = {_qualified_identifier('counter', key)}"
        for key in exposure_keys
    )
    exposure_prior_join = "\n    AND ".join(
        f"{_qualified_identifier('ordered_contact', key)} = {_qualified_identifier('prior', key)}"
        for key in exposure_keys
    )
    non_null_contacts = "\n      AND ".join(
        f"{_quote_identifier(column)} IS NOT NULL" for column in [*contact_keys, time_column]
    )
    running_partition = ", ".join(
        [
            _identifier_csv(exposure_keys),
            f"CAST({_quote_identifier(time_column)} AS DATE)",
        ]
    )
    need_prior = model.FREQUENCY_PRIOR_POSITIVE_COLUMN in config.group_by
    need_rank = model.FREQUENCY_SCOPE_RANK_COLUMN in config.group_by
    history_positive = f",\n        BOOL_OR({positive}) AS {positive}" if need_prior else ""
    counter_positive = (
        f",\n        CAST(count_if({positive}) AS BIGINT) AS {prior_positives}"
        if need_prior
        else ""
    )
    window_positive = (
        f",\n        CAST(SUM(counter.{prior_positives}) AS BIGINT) AS {prior_positives}"
        if need_prior
        else ""
    )
    excluded = [sequence]
    if need_prior:
        excluded.append(positives)
    if need_rank:
        excluded.append(scope_rank)
    derived = [
        f"CAST(LEAST(ordered_contact.{sequence} + COALESCE(prior.{prior_exposures}, 0), "
        f"{config.max_frequency}) AS BIGINT) AS {_quote_identifier(config.frequency_column)}"
    ]
    if need_prior:
        derived.append(
            f"(ordered_contact.{positives} - CAST(ordered_contact.{positive} AS BIGINT) "
            f"+ COALESCE(prior.{prior_positives}, 0)) > 0 "
            f"AS {_quote_identifier(model.FREQUENCY_PRIOR_POSITIVE_COLUMN)}"
        )
    if need_rank:
        derived.append(
            f"CAST(LEAST(ordered_contact.{scope_rank}, {config.max_rank}) AS BIGINT) "
            f"AS {_quote_identifier(model.FREQUENCY_SCOPE_RANK_COLUMN)}"
        )

    return f"""
WITH checkpoint_union AS (
{_indent(f"({current_member})", 4)}
),
sequenced_contacts AS (
    SELECT
        * EXCLUDE (
            {_quote_identifier(chunk_id_column)},
            {_quote_identifier(CHECKPOINT_LOCAL_ORDER_COLUMN)}
        ),
        CAST(
            ROW_NUMBER() OVER (
                ORDER BY
                    {_quote_identifier(chunk_id_column)},
                    {_quote_identifier(CHECKPOINT_LOCAL_ORDER_COLUMN)}
            ) - 1
            AS BIGINT
        ) AS {_quote_identifier(ROW_ORDER_COLUMN)}
    FROM checkpoint_union
),
normalized_contacts AS MATERIALIZED (
    SELECT
{_indent(normalized_select, 8)}
    FROM sequenced_contacts
    WHERE {non_null_contacts}
    GROUP BY {_identifier_csv(contact_keys)}
),
ordered_exposures AS MATERIALIZED (
    SELECT
        contact.*,
{_indent(_running_totals_sql(config, partition_by=running_partition), 8)}
    FROM normalized_contacts AS contact
),
history_contacts AS (
    SELECT
        {history_identity}{history_positive}
    FROM {_quote_identifier(history_table)}
    WHERE {_quote_identifier(shard_column)} = {shard_id}
      AND {_quote_identifier(chunk_id_column)} < {_quote_literal(current_chunk_id)}
    GROUP BY {history_identity}
),
daily_counters AS (
    SELECT
        {counter_keys},
        CAST(count(*) AS BIGINT) AS {prior_exposures}{counter_positive}
    FROM history_contacts
    GROUP BY {counter_keys}
),
focal_days AS (
    SELECT DISTINCT
        {_qualified_identifier_csv("ordered_contact", exposure_keys)},
        CAST({_qualified_identifier("ordered_contact", time_column)} AS DATE) AS {day}
    FROM ordered_exposures AS ordered_contact
),
window_prior AS (
    SELECT
        {focal_keys},
        focal.{day},
        CAST(SUM(counter.{prior_exposures}) AS BIGINT) AS {prior_exposures}{window_positive}
    FROM focal_days AS focal
    JOIN daily_counters AS counter
        ON {prior_join}
        AND counter.{day} > focal.{day} - {config.window_days}
        AND counter.{day} < focal.{day}
    GROUP BY {focal_keys}, focal.{day}
),
exposures AS (
    SELECT
        ordered_contact.* EXCLUDE ({", ".join(excluded)}),
        {", ".join(derived)}
    FROM ordered_exposures AS ordered_contact
    LEFT JOIN window_prior AS prior
        ON {exposure_prior_join}
        AND CAST({_qualified_identifier("ordered_contact", time_column)} AS DATE) = prior.{day}
)
SELECT *
FROM exposures
WHERE {_quote_identifier(SEEN_CURRENT_COLUMN)}
  AND NOT {_quote_identifier(SEEN_HISTORY_COLUMN)}
""".strip()


def _running_totals_sql(config: model.FrequencyResponseProcessor, *, partition_by: str) -> str:
    """Window expressions shared by both plans, omitting unpublished dimensions."""

    order = ", ".join(
        f"{_qualified_identifier('contact', column)} ASC NULLS FIRST"
        for column in [config.time.property, config.columns.interaction, CONTACT_ORDER_COLUMN]
    )
    running = f"PARTITION BY {partition_by} ORDER BY {order}"
    expressions = [
        f"CAST(ROW_NUMBER() OVER ({running}) AS BIGINT)\n"
        f"    AS {_quote_identifier(EXPOSURE_SEQUENCE_COLUMN)}"
    ]
    if model.FREQUENCY_PRIOR_POSITIVE_COLUMN in config.group_by:
        positive = _qualified_identifier("contact", POSITIVE_COLUMN)
        expressions.append(
            f"CAST(SUM(CAST({positive} AS BIGINT)) OVER (\n"
            f"    {running}\n"
            "    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW\n"
            f") AS BIGINT) AS {_quote_identifier(POSITIVE_SEQUENCE_COLUMN)}"
        )
    if model.FREQUENCY_SCOPE_RANK_COLUMN in config.group_by:
        rank_partition = _qualified_identifier_csv("contact", config.rank_partition_columns)
        expressions.append(
            f"CAST(DENSE_RANK() OVER (\n"
            f"    PARTITION BY {rank_partition}\n"
            f"    ORDER BY {_qualified_identifier('contact', RANK_COLUMN)}\n"
            f") AS BIGINT) AS {_quote_identifier(SCOPE_RANK_SEQUENCE_COLUMN)}"
        )
    return ",\n".join(expressions)


def _union_columns(
    columns: Iterable[str],
    *,
    shard_column: str,
    chunk_id_column: str,
) -> list[str]:
    union_columns = _ordered_unique(
        column
        for column in columns
        if column not in {shard_column, chunk_id_column, TARGET_CHUNK_COLUMN, ROW_ORDER_COLUMN}
    )
    union_columns.extend([TARGET_CHUNK_COLUMN, ROW_ORDER_COLUMN])
    return union_columns


def _passthrough_columns(
    config: model.FrequencyResponseProcessor,
    union_columns: Sequence[str],
    contact_keys: Sequence[str],
) -> list[str]:
    excluded = {
        *contact_keys,
        TARGET_CHUNK_COLUMN,
        ROW_ORDER_COLUMN,
        POSITIVE_COLUMN,
        config.time.property,
    }
    return [column for column in union_columns if column not in excluded]


def _checkpoint_member_sql(
    table: str,
    columns: Sequence[str],
    *,
    is_current: bool,
    shard_column: str,
    chunk_id_column: str,
    current_chunk_id: str,
    shard_id: int,
) -> str:
    excluded = [shard_column, ROW_ORDER_COLUMN]
    if TARGET_CHUNK_COLUMN in columns:
        excluded.append(TARGET_CHUNK_COLUMN)
    return f"""
SELECT
    * EXCLUDE ({_identifier_csv(excluded)}),
    CAST({_quote_identifier(ROW_ORDER_COLUMN)} AS BIGINT)
        AS {_quote_identifier(CHECKPOINT_LOCAL_ORDER_COLUMN)},
    {"TRUE" if is_current else "FALSE"} AS {_quote_identifier(TARGET_CHUNK_COLUMN)}
FROM {_quote_identifier(table)}
WHERE {_quote_identifier(shard_column)} = {shard_id}
  AND {_quote_identifier(chunk_id_column)} {"=" if is_current else "<"}
      {_quote_literal(current_chunk_id)}
""".strip()


def _normalized_select_sql(
    config: model.FrequencyResponseProcessor,
    contact_keys: Sequence[str],
    passthrough: Sequence[str],
) -> str:
    time_column = config.time.property
    first_order = ", ".join(
        [
            f"{_quote_identifier(POSITIVE_COLUMN)} DESC",
            f"{_quote_identifier(TARGET_CHUNK_COLUMN)} DESC",
            f"{_quote_identifier(time_column)} ASC",
            f"{_quote_identifier(ROW_ORDER_COLUMN)} ASC",
        ]
    )
    expressions = [
        *_quote_identifiers(contact_keys),
        f"MIN({_quote_identifier(time_column)}) AS {_quote_identifier(time_column)}",
        f"BOOL_OR({_quote_identifier(POSITIVE_COLUMN)}) AS {_quote_identifier(POSITIVE_COLUMN)}",
        (
            f"BOOL_OR({_quote_identifier(TARGET_CHUNK_COLUMN)}) "
            f"AS {_quote_identifier(SEEN_CURRENT_COLUMN)}"
        ),
        (
            f"BOOL_OR(NOT {_quote_identifier(TARGET_CHUNK_COLUMN)}) "
            f"AS {_quote_identifier(SEEN_HISTORY_COLUMN)}"
        ),
        (
            f"MIN({_quote_identifier(ROW_ORDER_COLUMN)}) "
            f"AS {_quote_identifier(CONTACT_ORDER_COLUMN)}"
        ),
        *(
            f"FIRST({_quote_identifier(column)} ORDER BY {first_order}) "
            f"AS {_quote_identifier(column)}"
            for column in passthrough
        ),
    ]
    return ",\n".join(expressions)


def _qualified_identifier(alias: str, column: str) -> str:
    return f"{_quote_identifier(alias)}.{_quote_identifier(column)}"


def _qualified_identifier_csv(alias: str, columns: Sequence[str]) -> str:
    return ", ".join(_qualified_identifier(alias, column) for column in columns)


def _identifier_csv(columns: Sequence[str]) -> str:
    return ", ".join(_quote_identifiers(columns))


def _quote_identifiers(columns: Sequence[str]) -> list[str]:
    return [_quote_identifier(column) for column in columns]


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _indent(value: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in value.splitlines())


__all__ = ["DuckDBFrequencySession"]
