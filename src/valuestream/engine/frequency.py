"""Native DuckDB execution over rolling frequency-response checkpoint state."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from types import TracebackType

import duckdb
import polars as pl

from valuestream.config import model
from valuestream.processors.context import TARGET_CHUNK_COLUMN
from valuestream.processors.frequency_fields import (
    BOUNDARY_SEQUENCE_COLUMN,
    BOUNDARY_TIME_COLUMN,
    CHECKPOINT_LOCAL_ORDER_COLUMN,
    CHECKPOINT_SHARD_COLUMN,
    CONTACT_ORDER_COLUMN,
    EXPOSED_COLUMN,
    EXPOSURE_SEQUENCE_COLUMN,
    POSITIVE_COLUMN,
    RANK_COLUMN,
    ROW_ORDER_COLUMN,
    RUNNER_AVAILABLE_COLUMN,
    RUNNER_PRIORITY_COLUMN,
    RUNNER_PROPENSITY_COLUMN,
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

        if not self._entered:
            raise RuntimeError("DuckDB frequency session must be entered before querying")
        if isinstance(shard_id, bool) or not isinstance(shard_id, int) or shard_id < 0:
            raise ValueError("frequency checkpoint shard_id must be a non-negative integer")
        sql = _focal_sql(
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
        relation = self.connection.sql(sql)
        return relation.pl(lazy=True, batch_size=self.batch_size)

    def _relation_columns(
        self,
        table: str,
    ) -> tuple[str, ...]:
        rows = self.connection.sql(f"DESCRIBE SELECT * FROM {_quote_identifier(table)}").fetchall()
        return tuple(str(row[0]) for row in rows)

    def _validate_checkpoint_schemas(self) -> None:
        columns = self.config.columns
        common = {
            self.shard_column,
            self.chunk_id_column,
            columns.customer,
            columns.interaction,
            columns.action,
            columns.placement,
            self.config.time.property,
            ROW_ORDER_COLUMN,
            RANK_COLUMN,
            POSITIVE_COLUMN,
            EXPOSED_COLUMN,
        }
        current_required = {
            *common,
            columns.propensity,
            *self.config.alternative_group_by,
        }
        if columns.priority is not None:
            current_required.add(columns.priority)
        missing_current = sorted(current_required - set(self._current_columns))
        if missing_current:
            raise ValueError(
                f"frequency_response processor {self.config.id!r} target checkpoint is "
                f"missing column(s): {', '.join(missing_current)}"
            )
        missing_history = sorted(common - set(self._history_columns))
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
    """Build the exact contact-normalization, window, and runner SQL plan."""

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

    union_columns = _ordered_unique(
        column
        for _table, columns, _is_current in relation_specs
        for column in columns
        if column
        not in {
            shard_column,
            chunk_id_column,
            TARGET_CHUNK_COLUMN,
            ROW_ORDER_COLUMN,
        }
    )
    union_columns.extend([TARGET_CHUNK_COLUMN, ROW_ORDER_COLUMN])

    columns = config.columns
    time_column = config.time.property
    contact_keys = [
        columns.customer,
        columns.interaction,
        columns.action,
        columns.placement,
        RANK_COLUMN,
    ]
    excluded_passthrough = {
        *contact_keys,
        TARGET_CHUNK_COLUMN,
        ROW_ORDER_COLUMN,
        POSITIVE_COLUMN,
        EXPOSED_COLUMN,
        time_column,
        columns.propensity,
        *(set() if columns.priority is None else {columns.priority}),
    }
    passthrough = [column for column in union_columns if column not in excluded_passthrough]
    normalized_select = _normalized_select_sql(config, contact_keys, passthrough)

    exposure_keys = [columns.customer, columns.action, columns.placement]
    exposure_partition = _identifier_csv(exposure_keys)
    exposure_order = _qualified_order(
        "contact",
        [time_column, columns.interaction, CONTACT_ORDER_COLUMN],
    )
    boundary_groups = _identifier_csv([*exposure_keys, time_column])
    boundary_select_keys = _qualified_identifier_csv("ordered_contact", exposure_keys)
    boundary_join = "\n            AND ".join(
        f"{_qualified_identifier('windowed_contact', key)} = "
        f"{_qualified_identifier('boundary', key)}"
        for key in exposure_keys
    )

    runner_keys = config.alternative_group_columns
    runner_partition = _identifier_csv(runner_keys)
    runner_sort_columns = _ordered_unique(
        [
            *runner_keys,
            RANK_COLUMN,
            columns.action,
            columns.placement,
            time_column,
            CONTACT_ORDER_COLUMN,
        ]
    )
    runner_order_parts = [
        f"CASE WHEN {_quote_identifier(RANK_COLUMN)} = 2 THEN 0 ELSE 1 END ASC",
        *(
            f"{_quote_identifier(column)} ASC NULLS FIRST"
            for column in runner_sort_columns
            if column not in runner_keys
        ),
    ]
    runner_priority = (
        f"CAST({_quote_identifier(columns.priority)} AS DOUBLE)"
        if columns.priority is not None
        else "NULL::DOUBLE"
    )
    runner_join = "\n        AND ".join(
        f"{_qualified_identifier('exposure', key)} IS NOT DISTINCT FROM "
        f"{_qualified_identifier('runner', key)}"
        for key in runner_keys
    )
    non_null_contacts = "\n          AND ".join(
        f"{_quote_identifier(column)} IS NOT NULL" for column in [*contact_keys, time_column]
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
        CAST(
            ROW_NUMBER() OVER (
                PARTITION BY {exposure_partition}
                ORDER BY {exposure_order}
            ) AS BIGINT
        ) AS {_quote_identifier(EXPOSURE_SEQUENCE_COLUMN)}
    FROM normalized_contacts AS contact
    WHERE {_qualified_identifier("contact", EXPOSED_COLUMN)}
      AND {_qualified_identifier("contact", RANK_COLUMN)} = 1
),
boundary_points AS (
    SELECT
        {boundary_select_keys},
        {_qualified_identifier("ordered_contact", time_column)}
            AS {_quote_identifier(BOUNDARY_TIME_COLUMN)},
        MAX({_qualified_identifier("ordered_contact", EXPOSURE_SEQUENCE_COLUMN)})
            AS {_quote_identifier(BOUNDARY_SEQUENCE_COLUMN)}
    FROM ordered_exposures AS ordered_contact
    GROUP BY {boundary_groups}
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
        windowed_contact.* EXCLUDE ({_quote_identifier(WINDOW_START_COLUMN)}),
        CAST(
            LEAST(
                {_qualified_identifier("windowed_contact", EXPOSURE_SEQUENCE_COLUMN)}
                    - COALESCE(
                        {_qualified_identifier("boundary", BOUNDARY_SEQUENCE_COLUMN)},
                        0
                    ),
                {config.max_frequency}
            ) AS BIGINT
        ) AS {_quote_identifier(config.frequency_column)}
    FROM windowed_exposures AS windowed_contact
    ASOF LEFT JOIN boundary_points AS boundary
        ON {boundary_join}
        AND {_qualified_identifier("windowed_contact", WINDOW_START_COLUMN)}
            >= {_qualified_identifier("boundary", BOUNDARY_TIME_COLUMN)}
),
runners AS (
    SELECT
        {_identifier_csv(runner_keys)},
        CAST({_quote_identifier(columns.propensity)} AS DOUBLE)
            AS {_quote_identifier(RUNNER_PROPENSITY_COLUMN)},
        {runner_priority} AS {_quote_identifier(RUNNER_PRIORITY_COLUMN)},
        TRUE AS {_quote_identifier(RUNNER_AVAILABLE_COLUMN)}
    FROM normalized_contacts
    WHERE {_quote_identifier(SEEN_CURRENT_COLUMN)}
      AND {_quote_identifier(RANK_COLUMN)} > 1
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY {runner_partition}
        ORDER BY {", ".join(runner_order_parts)}
    ) = 1
)
SELECT
    exposure.*,
    runner.{_quote_identifier(RUNNER_PROPENSITY_COLUMN)},
    runner.{_quote_identifier(RUNNER_PRIORITY_COLUMN)},
    runner.{_quote_identifier(RUNNER_AVAILABLE_COLUMN)}
FROM exposures AS exposure
LEFT JOIN runners AS runner
    ON {runner_join}
WHERE exposure.{_quote_identifier(SEEN_CURRENT_COLUMN)}
  AND NOT exposure.{_quote_identifier(SEEN_HISTORY_COLUMN)}
  AND exposure.{_quote_identifier(EXPOSED_COLUMN)}
  AND exposure.{_quote_identifier(RANK_COLUMN)} = 1
""".strip()


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
    columns = config.columns
    time_column = config.time.property
    first_order = ", ".join(
        [
            f"{_quote_identifier(POSITIVE_COLUMN)} DESC",
            f"{_quote_identifier(EXPOSED_COLUMN)} DESC",
            f"{_quote_identifier(TARGET_CHUNK_COLUMN)} DESC",
            f"{_quote_identifier(time_column)} ASC",
            f"{_quote_identifier(ROW_ORDER_COLUMN)} ASC",
        ]
    )
    expressions = [
        *_quote_identifiers(contact_keys),
        f"MIN({_quote_identifier(time_column)}) AS {_quote_identifier(time_column)}",
        (
            f"FIRST({_quote_identifier(columns.propensity)} ORDER BY {first_order}) "
            f"FILTER (WHERE {_quote_identifier(columns.propensity)} IS NOT NULL) "
            f"AS {_quote_identifier(columns.propensity)}"
        ),
        f"BOOL_OR({_quote_identifier(POSITIVE_COLUMN)}) AS {_quote_identifier(POSITIVE_COLUMN)}",
        f"BOOL_OR({_quote_identifier(EXPOSED_COLUMN)}) AS {_quote_identifier(EXPOSED_COLUMN)}",
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
    if columns.priority is not None:
        expressions.append(
            f"FIRST({_quote_identifier(columns.priority)} ORDER BY {first_order}) "
            f"FILTER (WHERE {_quote_identifier(columns.priority)} IS NOT NULL) "
            f"AS {_quote_identifier(columns.priority)}"
        )
    return ",\n".join(expressions)


def _qualified_identifier(alias: str, column: str) -> str:
    return f"{_quote_identifier(alias)}.{_quote_identifier(column)}"


def _qualified_identifier_csv(alias: str, columns: Sequence[str]) -> str:
    return ", ".join(_qualified_identifier(alias, column) for column in columns)


def _qualified_order(alias: str, columns: Sequence[str]) -> str:
    return ", ".join(
        f"{_qualified_identifier(alias, column)} ASC NULLS FIRST" for column in columns
    )


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
