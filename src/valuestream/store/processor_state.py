"""Mutable rolling state for bounded frequency-response processing.

The rolling database is processor-internal, rebuildable state. It is kept
outside ``aggregates/`` and is never query-visible. One stable database is
addressed by the source and processor identities. Its durable history
contains only the exact bounded closure needed by the next target; a complete
current chunk exists as a temporary table until the target aggregate succeeds.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal, TypeVar, cast
from urllib.parse import quote
from uuid import uuid4

import duckdb
import polars as pl

from valuestream.utils.logger import get_logger

CHECKPOINT_SCHEMA_REVISION = 7
SHARD_HASH_REVISION = 1
SHARD_HASH_ALGORITHM = "polars.Expr.hash"
SHARD_HASH_SEEDS = (
    0x243F6A8885A308D3,
    0x13198A2E03707344,
    0xA4093822299F31D0,
    0x082EFA98EC4E6C89,
)
SHARD_COLUMN = "__valuestream_checkpoint_shard"
CHUNK_ID_COLUMN = "__valuestream_checkpoint_chunk_id"
ROLLING_DATABASE_FILENAME = "rolling.duckdb"
METADATA_TABLE = "checkpoint_metadata"
JOURNAL_TABLE = "chunk_journal"
HISTORY_TABLE = "history"
CURRENT_TABLE = "current"
CHECKPOINT_MAINTENANCE_INTERVAL_DAYS = 30

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_FrameT = TypeVar("_FrameT", pl.DataFrame, pl.LazyFrame)
_CollectEngine = Literal["auto", "streaming"]
_RESERVED_COLUMNS = frozenset({SHARD_COLUMN, CHUNK_ID_COLUMN})
logger = get_logger(__name__)


class CheckpointValidationError(ValueError):
    """Raised when rolling state is incomplete, corrupt, or incompatible."""


class _CheckpointCompatibilityError(CheckpointValidationError):
    """Raised when valid processor state belongs to an unsupported contract."""


@dataclass(frozen=True, slots=True)
class HistoryProjectionSpec:
    """SQL-compatible narrow history projection and inclusion predicate."""

    columns: tuple[str, ...]
    rank_column: str
    exposed_column: str

    def __post_init__(self) -> None:
        if not isinstance(self.columns, tuple):
            raise TypeError("checkpoint history columns must be a tuple")
        if not self.columns or any(not isinstance(name, str) or not name for name in self.columns):
            raise ValueError("checkpoint history columns must be non-empty strings")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("checkpoint history columns must be unique")
        reserved = sorted(set(self.columns) & _RESERVED_COLUMNS)
        if reserved:
            raise ValueError(
                "checkpoint history columns use reserved column(s): " + ", ".join(reserved)
            )
        for label, name in {
            "rank_column": self.rank_column,
            "exposed_column": self.exposed_column,
        }.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"checkpoint history {label} must be a non-empty string")
            if name not in self.columns:
                raise ValueError(f"checkpoint history {label} {name!r} must appear in columns")


@dataclass(frozen=True, slots=True)
class CheckpointJournalEntry:
    """One committed chunk in rolling-history order."""

    sequence: int
    chunk_id: str
    raw_fingerprint: str


def rolling_checkpoint_path(
    workspace_path: str | Path,
    *,
    source_id: str,
    processor_id: str,
) -> Path:
    """Return the stable rolling database path for one source processor."""

    _validate_path_identity(source_id=source_id, processor_id=processor_id)
    return (
        Path(workspace_path)
        / ".valuestream"
        / "state"
        / "frequency_response"
        / f"source={_component(source_id)}"
        / f"processor={_component(processor_id)}"
        / ROLLING_DATABASE_FILENAME
    )


def assign_customer_shard(
    frame: _FrameT,
    *,
    customer_column: str,
    shard_count: int,
) -> _FrameT:
    """Add the deterministic internal customer-hash shard column."""

    _validate_shard_request(frame, customer_column=customer_column, shard_count=shard_count)
    shard = (
        pl.col(customer_column).hash(*SHARD_HASH_SEEDS) % pl.lit(shard_count, dtype=pl.UInt64)
    ).cast(pl.UInt32)
    return frame.with_columns(shard.alias(SHARD_COLUMN))


class RollingCheckpoint:
    """Own one mutable rolling frequency checkpoint connection.

    ``stage_current`` materializes a complete target payload as a DuckDB TEMP
    table through the Arrow C Stream interface. Callers may query it together
    with ``history`` and then either commit its narrow history projection or
    abort it. No staged current rows survive connection close.
    """

    def __init__(
        self,
        workspace_path: str | Path,
        *,
        source_id: str,
        processor_id: str,
        config_hash: str,
        customer_column: str,
        customer_dtype: str,
        shard_count: int,
        retention_days: int,
        history_projection: HistoryProjectionSpec,
        force: bool = False,
    ) -> None:
        _validate_identity(
            source_id=source_id,
            processor_id=processor_id,
            config_hash=config_hash,
        )
        if not isinstance(customer_column, str) or not customer_column:
            raise ValueError("checkpoint customer column must be a non-empty string")
        if not isinstance(customer_dtype, str) or not customer_dtype:
            raise ValueError("checkpoint customer dtype must be a non-empty string")
        _validate_shard_count(shard_count)
        if (
            isinstance(retention_days, bool)
            or not isinstance(retention_days, int)
            or retention_days < 1
        ):
            raise ValueError("checkpoint retention_days must be a positive integer")
        if not isinstance(history_projection, HistoryProjectionSpec):
            raise TypeError("checkpoint history_projection must be a HistoryProjectionSpec")
        if customer_column not in history_projection.columns:
            raise ValueError(
                f"checkpoint history columns must include customer column {customer_column!r}"
            )
        if not isinstance(force, bool):
            raise TypeError("checkpoint force must be a boolean")

        self.path = rolling_checkpoint_path(
            workspace_path,
            source_id=source_id,
            processor_id=processor_id,
        )
        self.source_id = source_id
        self.processor_id = processor_id
        self.config_hash = config_hash
        self.customer_column = customer_column
        self.customer_dtype = customer_dtype
        self.shard_count = shard_count
        self.retention_days = retention_days
        self.history_projection = history_projection
        self.force = force
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._effective_customer_dtype = customer_dtype
        self._effective_customer_storage_dtype = "Null"
        self._staged_chunk_id: str | None = None
        self._staged_raw_fingerprint: str | None = None
        self._staged_shard_ids: tuple[int, ...] = ()

    def __enter__(self) -> RollingCheckpoint:
        if self._connection is not None:
            raise RuntimeError("rolling checkpoint is already open")
        self._prepare_database_path()
        try:
            connection = self._open_database()
        except _CheckpointCompatibilityError as exc:
            connection = self._replace_incompatible_database(exc)
        except CheckpointValidationError:
            raise
        except (duckdb.Error, OSError, ValueError) as exc:
            raise CheckpointValidationError(
                f"cannot open rolling checkpoint {self.path}: {exc}"
            ) from exc
        self._connection = connection
        return self

    def _prepare_database_path(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not self.path.is_file():
            raise CheckpointValidationError(f"rolling checkpoint path is not a file: {self.path}")
        wal_path = self.path.with_name(f"{self.path.name}.wal")
        if not self.force and not self.path.exists() and wal_path.exists():
            raise CheckpointValidationError(
                f"rolling checkpoint has an orphan WAL without its database: {wal_path}"
            )
        if self.force:
            _remove_database_files(self.path)

    def _open_database(self) -> duckdb.DuckDBPyConnection:
        is_new = not self.path.exists()
        connection = duckdb.connect(str(self.path))
        try:
            _configure_connection(connection)
            if is_new:
                self._initialize(connection)
            else:
                self._validate_existing(connection)
        except Exception:
            connection.close()
            raise
        return connection

    def _replace_incompatible_database(
        self,
        reason: _CheckpointCompatibilityError,
    ) -> duckdb.DuckDBPyConnection:
        logger.info(
            "Replacing incompatible rolling checkpoint with the current schema: "
            "source=%s processor=%s path=%s reason=%s",
            self.source_id,
            self.processor_id,
            self.path,
            reason,
        )
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            self._initialize_database_file(temporary)
            # Opening the old database has already replayed its WAL and closing
            # it released the writer. Never let that WAL attach to the new file.
            with suppress(FileNotFoundError):
                self.path.with_name(f"{self.path.name}.wal").unlink()
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
            return self._open_database()
        except (duckdb.Error, OSError, ValueError) as exc:
            raise CheckpointValidationError(
                f"cannot rebuild rolling checkpoint {self.path}: {exc}"
            ) from exc
        finally:
            _remove_database_files(temporary)

    def _initialize_database_file(self, path: Path) -> None:
        connection = duckdb.connect(str(path))
        try:
            _configure_connection(connection)
            self._initialize(connection)
            self._validate_existing(connection)
        finally:
            connection.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """The borrowed DuckDB connection used by native frequency SQL."""

        if self._connection is None:
            raise RuntimeError("rolling checkpoint is not open")
        return self._connection

    @property
    def journal(self) -> tuple[CheckpointJournalEntry, ...]:
        """Committed chunks in rolling-history order."""

        return self._journal_entries(self.connection)

    @property
    def journal_entries(self) -> tuple[CheckpointJournalEntry, ...]:
        """Alias for ``journal`` for explicit call sites."""

        return self.journal

    @property
    def staged_chunk_id(self) -> str | None:
        """Currently staged chunk, if any."""

        return self._staged_chunk_id

    def stage_current(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        *,
        chunk_id: str,
        raw_fingerprint: str,
        engine: _CollectEngine = "auto",
    ) -> tuple[int, ...]:
        """Stream one complete current chunk into a temporary DuckDB table."""

        if self._staged_chunk_id is not None:
            raise RuntimeError(
                f"rolling checkpoint already has staged chunk {self._staged_chunk_id!r}"
            )
        _validate_chunk_identity(chunk_id, raw_fingerprint)
        _validate_shard_request(
            frame,
            customer_column=self.customer_column,
            shard_count=self.shard_count,
        )
        schema = frame.schema if isinstance(frame, pl.DataFrame) else frame.collect_schema()
        missing = sorted(set(self.history_projection.columns) - set(schema.names()))
        if missing:
            raise ValueError(
                "checkpoint current frame is missing history column(s): " + ", ".join(missing)
            )
        if schema[self.history_projection.exposed_column] != pl.Boolean:
            raise TypeError(
                f"checkpoint history exposed column {self.history_projection.exposed_column!r} "
                "must be Boolean"
            )
        incoming_dtype = str(schema[self.customer_column])
        if not _customer_dtypes_compatible(self._effective_customer_dtype, incoming_dtype):
            raise CheckpointValidationError(
                f"checkpoint customer dtype {incoming_dtype!r} does not match "
                f"{self._effective_customer_dtype!r}"
            )

        lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
        sharded = assign_customer_shard(
            lazy,
            customer_column=self.customer_column,
            shard_count=self.shard_count,
        )
        stream = sharded.collect_batches(
            maintain_order=True,
            lazy=True,
            engine=engine,
        )
        connection = self.connection
        registered = False
        committed_logical_dtype: str | None = None
        committed_storage_dtype: str | None = None
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.register("_valuestream_checkpoint_input", stream)
            registered = True
            connection.execute(f"DROP TABLE IF EXISTS {_quote_identifier(CURRENT_TABLE)}")
            connection.execute(
                f"CREATE TEMP TABLE {_quote_identifier(CURRENT_TABLE)} AS "
                "SELECT *, "
                f"{_quote_literal(chunk_id)}::VARCHAR AS {_quote_identifier(CHUNK_ID_COLUMN)} "
                "FROM _valuestream_checkpoint_input "
                f"ORDER BY {_quote_identifier(SHARD_COLUMN)}"
            )
            shards = _table_shards(
                connection,
                CURRENT_TABLE,
                shard_count=self.shard_count,
            )
            current_rows = sum(rows for _shard_id, rows in shards)
            null_customers = _null_count(connection, CURRENT_TABLE, self.customer_column)
            if null_customers:
                raise ValueError(
                    f"checkpoint customer column {self.customer_column!r} cannot contain nulls"
                )
            committed_logical_dtype, committed_storage_dtype = self._stage_customer_dtype_updates(
                connection,
                incoming_dtype=incoming_dtype,
                current_rows=current_rows,
            )
            self._ensure_history_table(connection, current_rows=current_rows)
            connection.execute("COMMIT")
            if committed_logical_dtype is not None:
                self._effective_customer_dtype = committed_logical_dtype
            if committed_storage_dtype is not None:
                self._effective_customer_storage_dtype = committed_storage_dtype
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            if registered:
                connection.unregister("_valuestream_checkpoint_input")

        self._staged_chunk_id = chunk_id
        self._staged_raw_fingerprint = raw_fingerprint
        self._staged_shard_ids = tuple(shard_id for shard_id, _rows in shards)
        return self._staged_shard_ids

    def commit_staged(self) -> None:
        """Atomically promote staged narrow history, journal it, and prune retention."""

        chunk_id, raw_fingerprint = self._require_staged()
        connection = self.connection
        maintenance_due = False
        connection.execute("BEGIN TRANSACTION")
        try:
            existing = self._journal_entries(connection)
            if existing and _chunk_date(chunk_id) <= _chunk_date(existing[-1].chunk_id):
                raise CheckpointValidationError(
                    "rolling checkpoint chunks must commit in increasing calendar order; "
                    f"got {chunk_id!r} after {existing[-1].chunk_id!r}"
                )
            selected = [
                *self.history_projection.columns,
                CHUNK_ID_COLUMN,
                SHARD_COLUMN,
            ]
            connection.execute(
                f"INSERT INTO {_quote_identifier(HISTORY_TABLE)} "
                f"SELECT {_identifier_csv(selected)} "
                f"FROM {_quote_identifier(CURRENT_TABLE)} "
                f"WHERE {_quote_identifier(self.history_projection.rank_column)} = 1 "
                f"AND {_quote_identifier(self.history_projection.exposed_column)} IS TRUE"
            )
            sequence = existing[-1].sequence + 1 if existing else 0
            connection.execute(
                f"INSERT INTO {_quote_identifier(JOURNAL_TABLE)} "
                "(sequence, chunk_id, raw_fingerprint) VALUES (?, ?, ?)",
                [sequence, chunk_id, raw_fingerprint],
            )
            self._prune_retention(connection)
            maintenance_due = (sequence + 1) % CHECKPOINT_MAINTENANCE_INTERVAL_DAYS == 0
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        self._drop_current()
        if maintenance_due:
            _checkpoint_connection(connection)

    def abort_staged(self) -> None:
        """Discard the temporary current table without changing durable history."""

        if self._connection is None:
            self._clear_staged()
            return
        self._drop_current()

    def reconcile_history(
        self,
        expected: Sequence[tuple[str, str]],
    ) -> tuple[str, ...]:
        """Reconcile durable history and return the missing expected suffix.

        Rows outside ``expected`` are removed even when the retained journal is
        otherwise reusable. If the retained entries are not an exact prefix, or
        any retained fingerprint changed, history and journal are reset in the
        same transaction and every expected chunk is reported missing.
        """

        if self._staged_chunk_id is not None:
            raise RuntimeError("cannot reconcile history while a current chunk is staged")
        normalized = _normalize_expected_history(expected)
        expected_ids = [chunk_id for chunk_id, _fingerprint in normalized]
        expected_set = set(expected_ids)
        connection = self.connection
        connection.execute("BEGIN TRANSACTION")
        try:
            existing = self._journal_entries(connection)
            outside = [entry for entry in existing if entry.chunk_id not in expected_set]
            if outside:
                outside_ids = [entry.chunk_id for entry in outside]
                if _table_exists(connection, HISTORY_TABLE):
                    _delete_chunks(connection, HISTORY_TABLE, outside_ids)
                _delete_chunks(connection, JOURNAL_TABLE, outside_ids)
            retained = self._journal_entries(connection)
            retained_pairs = tuple((entry.chunk_id, entry.raw_fingerprint) for entry in retained)
            prefix = normalized[: len(retained_pairs)]
            reusable = len(retained_pairs) <= len(normalized) and retained_pairs == prefix
            if not reusable:
                self._reset_in_transaction(connection)
                missing = tuple(expected_ids)
            else:
                missing = tuple(expected_ids[len(retained_pairs) :])
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        return missing

    def reset(self) -> None:
        """Atomically reset durable history and its journal."""

        self.abort_staged()
        connection = self.connection
        connection.execute("BEGIN TRANSACTION")
        try:
            self._reset_in_transaction(connection)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def prune_retention(self) -> None:
        """Apply the configured retention bound without staging a target."""

        if self._staged_chunk_id is not None:
            raise RuntimeError("cannot prune retention while a current chunk is staged")
        connection = self.connection
        connection.execute("BEGIN TRANSACTION")
        try:
            self._prune_retention(connection)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def close(self) -> None:
        """Discard staging, checkpoint durable state, and close the connection."""

        if self._connection is None:
            return
        connection = self._connection
        try:
            self.abort_staged()
            _checkpoint_connection(connection)
        finally:
            connection.close()
            self._connection = None
            self._clear_staged()

    def _initialize(self, connection: duckdb.DuckDBPyConnection) -> None:
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(_metadata_table_sql())
            connection.execute(_journal_table_sql())
            metadata_values = [
                True,
                CHECKPOINT_SCHEMA_REVISION,
                SHARD_HASH_REVISION,
                SHARD_HASH_ALGORITHM,
                *SHARD_HASH_SEEDS,
                pl.__version__,
                duckdb.__version__,
                self.source_id,
                self.processor_id,
                self.config_hash,
                self.customer_column,
                self.customer_dtype,
                "Null",
                self.shard_count,
                _projection_json(self.history_projection),
            ]
            connection.execute(
                f"INSERT INTO {_quote_identifier(METADATA_TABLE)} VALUES "
                f"({', '.join('?' for _ in metadata_values)})",
                metadata_values,
            )
            connection.execute("COMMIT")
            _checkpoint_connection(connection)
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def _validate_existing(self, connection: duckdb.DuckDBPyConnection) -> None:
        try:
            tables = _persistent_table_names(connection)
            allowed = {METADATA_TABLE, JOURNAL_TABLE, HISTORY_TABLE}
            unexpected = sorted(tables - allowed)
            missing = sorted({METADATA_TABLE, JOURNAL_TABLE} - tables)
            if missing:
                raise CheckpointValidationError(
                    "rolling checkpoint is missing table(s): " + ", ".join(missing)
                )
            if unexpected:
                raise CheckpointValidationError(
                    "rolling checkpoint has unexpected table(s): " + ", ".join(unexpected)
                )
            self._validate_schema_revision(connection)
            _validate_table_schema(
                connection,
                METADATA_TABLE,
                _metadata_schema(),
            )
            _validate_table_schema(
                connection,
                JOURNAL_TABLE,
                _journal_schema(),
            )
            self._validate_metadata(connection)
            journal = self._validate_journal(connection)
            if journal and HISTORY_TABLE not in tables:
                raise CheckpointValidationError(
                    "rolling checkpoint has committed journal entries but no history table"
                )
            if HISTORY_TABLE in tables:
                self._validate_history(connection)
        except CheckpointValidationError:
            raise
        except (duckdb.Error, OSError, ValueError) as exc:
            raise CheckpointValidationError(
                f"cannot validate rolling checkpoint {self.path}: {exc}"
            ) from exc

    def _validate_metadata(self, connection: duckdb.DuckDBPyConnection) -> None:
        rows = connection.execute(f"SELECT * FROM {_quote_identifier(METADATA_TABLE)}").fetchall()
        if len(rows) != 1:
            raise CheckpointValidationError("rolling checkpoint metadata must contain one row")
        row = rows[0]
        names = [name for name, _dtype in _metadata_schema()]
        values = dict(zip(names, row, strict=True))
        identity_expected: Mapping[str, object] = {
            "singleton": True,
            "source_id": self.source_id,
            "processor_id": self.processor_id,
        }
        compatibility_expected: Mapping[str, object] = {
            "schema_revision": CHECKPOINT_SCHEMA_REVISION,
            "shard_hash_revision": SHARD_HASH_REVISION,
            "shard_hash_algorithm": SHARD_HASH_ALGORITHM,
            "seed_0": SHARD_HASH_SEEDS[0],
            "seed_1": SHARD_HASH_SEEDS[1],
            "seed_2": SHARD_HASH_SEEDS[2],
            "seed_3": SHARD_HASH_SEEDS[3],
            "polars_version": pl.__version__,
            "config_hash": self.config_hash,
            "customer_column": self.customer_column,
            "shard_count": self.shard_count,
            "history_projection": _projection_json(self.history_projection),
        }
        for name, expected_value in identity_expected.items():
            if values[name] != expected_value:
                raise CheckpointValidationError(
                    f"rolling checkpoint {name} {values[name]!r} does not match {expected_value!r}"
                )
        for name, expected_value in compatibility_expected.items():
            if values[name] != expected_value:
                raise _CheckpointCompatibilityError(
                    f"rolling checkpoint {name} {values[name]!r} does not match {expected_value!r}"
                )
        stored_dtype = str(values["customer_dtype"])
        stored_storage_dtype = str(values["customer_storage_dtype"])
        history_rows = (
            _table_row_count(connection, HISTORY_TABLE)
            if _table_exists(connection, HISTORY_TABLE)
            else 0
        )
        if self.customer_dtype in {stored_dtype, "Null"}:
            self._effective_customer_dtype = stored_dtype
            self._effective_customer_storage_dtype = stored_storage_dtype
        elif stored_dtype == "Null" and history_rows == 0:
            connection.execute("BEGIN TRANSACTION")
            try:
                self._write_customer_dtypes(
                    connection,
                    logical_dtype=self.customer_dtype,
                )
                connection.execute("COMMIT")
                self._effective_customer_dtype = self.customer_dtype
                self._effective_customer_storage_dtype = stored_storage_dtype
            except Exception:
                connection.execute("ROLLBACK")
                raise
        else:
            raise CheckpointValidationError(
                f"rolling checkpoint customer dtype {stored_dtype!r} does not match "
                f"{self.customer_dtype!r}"
            )

    @staticmethod
    def _validate_schema_revision(connection: duckdb.DuckDBPyConnection) -> None:
        metadata_names = {name for name, _dtype in _table_schema(connection, METADATA_TABLE)}
        if "schema_revision" not in metadata_names:
            return
        rows = connection.execute(
            f"SELECT schema_revision FROM {_quote_identifier(METADATA_TABLE)}"
        ).fetchall()
        if len(rows) != 1:
            return
        revision = int(rows[0][0])
        if revision != CHECKPOINT_SCHEMA_REVISION:
            raise _CheckpointCompatibilityError(
                f"rolling checkpoint schema revision {revision} is unsupported; "
                f"only {CHECKPOINT_SCHEMA_REVISION} is supported"
            )

    def _validate_journal(
        self,
        connection: duckdb.DuckDBPyConnection,
    ) -> tuple[CheckpointJournalEntry, ...]:
        entries = self._journal_entries(connection)
        sequences = [entry.sequence for entry in entries]
        chunk_ids = [entry.chunk_id for entry in entries]
        if len(set(sequences)) != len(sequences) or sequences != sorted(sequences):
            raise CheckpointValidationError(
                "rolling checkpoint journal sequences must be unique and ordered"
            )
        if len(set(chunk_ids)) != len(chunk_ids):
            raise CheckpointValidationError("rolling checkpoint journal chunk IDs must be unique")
        for entry in entries:
            _validate_chunk_identity(entry.chunk_id, entry.raw_fingerprint)
        chunk_dates = [_chunk_date(entry.chunk_id) for entry in entries]
        if chunk_dates != sorted(chunk_dates) or len(set(chunk_dates)) != len(chunk_dates):
            raise CheckpointValidationError(
                "rolling checkpoint journal chunks must use increasing calendar order"
            )
        return entries

    def _validate_history(self, connection: duckdb.DuckDBPyConnection) -> None:
        expected_names = [
            *self.history_projection.columns,
            CHUNK_ID_COLUMN,
            SHARD_COLUMN,
        ]
        actual_schema = _table_schema(connection, HISTORY_TABLE)
        if [name for name, _dtype in actual_schema] != expected_names:
            raise CheckpointValidationError(
                "rolling checkpoint history columns do not match configured projection"
            )
        customer = _quote_identifier(self.customer_column)
        chunk = _quote_identifier(CHUNK_ID_COLUMN)
        shard = _quote_identifier(SHARD_COLUMN)
        rank = _quote_identifier(self.history_projection.rank_column)
        exposed = _quote_identifier(self.history_projection.exposed_column)
        validation = connection.execute(
            "SELECT count(*), "
            f"coalesce(count_if(history.{customer} IS NULL), 0), "
            f"coalesce(count_if(history.{chunk} IS NULL), 0), "
            f"coalesce(count_if(history.{shard} IS NULL), 0), "
            f"coalesce(count_if(history.{shard} < 0 OR history.{shard} >= ?), 0), "
            f"coalesce(count_if(history.{rank} IS NULL OR history.{rank} <> 1 "
            f"OR history.{exposed} IS DISTINCT FROM TRUE), 0), "
            "coalesce(count_if(journal.chunk_id IS NULL), 0) "
            f"FROM {_quote_identifier(HISTORY_TABLE)} AS history "
            f"LEFT JOIN {_quote_identifier(JOURNAL_TABLE)} AS journal "
            f"ON history.{chunk} = journal.chunk_id",
            [self.shard_count],
        ).fetchone()
        if validation is None:  # pragma: no cover - aggregate always returns one row
            raise RuntimeError("checkpoint history validation query returned no row")
        (
            rows,
            null_customers,
            null_chunks,
            null_shards,
            invalid_shards,
            invalid_roles,
            orphan_rows,
        ) = (int(value) for value in validation)
        if not rows:
            return
        null_counts = {
            self.customer_column: null_customers,
            CHUNK_ID_COLUMN: null_chunks,
            SHARD_COLUMN: null_shards,
        }
        for column, null_count in null_counts.items():
            if null_count:
                raise CheckpointValidationError(
                    f"rolling checkpoint history column {column!r} contains nulls"
                )
        if invalid_shards:
            raise CheckpointValidationError("rolling checkpoint history has invalid shard IDs")
        if invalid_roles:
            raise CheckpointValidationError(
                "rolling checkpoint history contains non-rank-1 or unexposed rows"
            )
        if orphan_rows:
            raise CheckpointValidationError(
                "rolling checkpoint history contains a chunk absent from the journal"
            )
        actual_dtype = _polars_dtype_for_column(
            connection,
            HISTORY_TABLE,
            self.customer_column,
        )
        if actual_dtype != self._effective_customer_storage_dtype:
            raise CheckpointValidationError(
                f"rolling checkpoint history customer storage dtype {actual_dtype!r} does not "
                f"match {self._effective_customer_storage_dtype!r}"
            )

    def _ensure_history_table(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        current_rows: int,
    ) -> None:
        selected = [
            *self.history_projection.columns,
            CHUNK_ID_COLUMN,
            SHARD_COLUMN,
        ]
        desired_schema = _selected_schema(connection, CURRENT_TABLE, selected)
        if not _table_exists(connection, HISTORY_TABLE):
            _create_empty_history(connection, selected)
            return
        actual_schema = _table_schema(connection, HISTORY_TABLE)
        actual_names = [name for name, _dtype in actual_schema]
        if actual_names != selected:
            raise CheckpointValidationError(
                "rolling checkpoint history columns do not match staged projection"
            )
        if actual_schema == desired_schema or current_rows == 0:
            return
        history_rows = _table_row_count(connection, HISTORY_TABLE)
        if not history_rows:
            connection.execute(f"DROP TABLE {_quote_identifier(HISTORY_TABLE)}")
            _create_empty_history(connection, selected)
            return
        _promote_history_schema(connection, selected)

    def _prune_retention(self, connection: duckdb.DuckDBPyConnection) -> None:
        entries = self._journal_entries(connection)
        if not entries:
            return
        cutoff = _chunk_date(entries[-1].chunk_id) - dt.timedelta(days=self.retention_days - 1)
        obsolete = [entry for entry in entries if _chunk_date(entry.chunk_id) < cutoff]
        if not obsolete:
            return
        obsolete_ids = [entry.chunk_id for entry in obsolete]
        _delete_chunks(connection, HISTORY_TABLE, obsolete_ids)
        _delete_chunks(connection, JOURNAL_TABLE, obsolete_ids)

    def _reset_in_transaction(self, connection: duckdb.DuckDBPyConnection) -> None:
        if _table_exists(connection, HISTORY_TABLE):
            connection.execute(f"DROP TABLE {_quote_identifier(HISTORY_TABLE)}")
        connection.execute(f"DELETE FROM {_quote_identifier(JOURNAL_TABLE)}")

    def _drop_current(self) -> None:
        if self._connection is not None:
            self._connection.execute(f"DROP TABLE IF EXISTS {_quote_identifier(CURRENT_TABLE)}")
        self._clear_staged()

    def _clear_staged(self) -> None:
        self._staged_chunk_id = None
        self._staged_raw_fingerprint = None
        self._staged_shard_ids = ()

    def _require_staged(self) -> tuple[str, str]:
        if self._staged_chunk_id is None or self._staged_raw_fingerprint is None:
            raise RuntimeError("rolling checkpoint has no staged current chunk")
        return self._staged_chunk_id, self._staged_raw_fingerprint

    def _journal_entries(
        self,
        connection: duckdb.DuckDBPyConnection,
    ) -> tuple[CheckpointJournalEntry, ...]:
        rows = connection.execute(
            f"SELECT sequence, chunk_id, raw_fingerprint "
            f"FROM {_quote_identifier(JOURNAL_TABLE)} ORDER BY sequence"
        ).fetchall()
        return tuple(
            CheckpointJournalEntry(
                sequence=int(sequence),
                chunk_id=str(chunk_id),
                raw_fingerprint=str(raw_fingerprint),
            )
            for sequence, chunk_id, raw_fingerprint in rows
        )

    def _write_customer_dtypes(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        logical_dtype: str | None = None,
        storage_dtype: str | None = None,
    ) -> None:
        assignments: list[str] = []
        values: list[str] = []
        if logical_dtype is not None:
            assignments.append("customer_dtype = ?")
            values.append(logical_dtype)
        if storage_dtype is not None:
            assignments.append("customer_storage_dtype = ?")
            values.append(storage_dtype)
        if not assignments:  # pragma: no cover - private call invariant
            return
        connection.execute(
            f"UPDATE {_quote_identifier(METADATA_TABLE)} SET {', '.join(assignments)} "
            "WHERE singleton",
            values,
        )

    def _stage_customer_dtype_updates(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        incoming_dtype: str,
        current_rows: int,
    ) -> tuple[str | None, str | None]:
        """Write pending logical/physical dtype metadata inside the stage transaction."""

        if not current_rows:
            return None, None
        actual_storage_dtype = _polars_dtype_for_column(
            connection,
            CURRENT_TABLE,
            self.customer_column,
        )
        if self._effective_customer_dtype == "Null":
            self._write_customer_dtypes(
                connection,
                logical_dtype=incoming_dtype,
                storage_dtype=actual_storage_dtype,
            )
            return incoming_dtype, actual_storage_dtype
        if self._effective_customer_storage_dtype == "Null":
            self._write_customer_dtypes(
                connection,
                storage_dtype=actual_storage_dtype,
            )
            return None, actual_storage_dtype
        if actual_storage_dtype != self._effective_customer_storage_dtype:
            raise CheckpointValidationError(
                f"checkpoint customer storage dtype {actual_storage_dtype!r} does not match "
                f"{self._effective_customer_storage_dtype!r}"
            )
        return None, None


def _metadata_schema() -> tuple[tuple[str, str], ...]:
    return (
        ("singleton", "BOOLEAN"),
        ("schema_revision", "INTEGER"),
        ("shard_hash_revision", "INTEGER"),
        ("shard_hash_algorithm", "VARCHAR"),
        ("seed_0", "UBIGINT"),
        ("seed_1", "UBIGINT"),
        ("seed_2", "UBIGINT"),
        ("seed_3", "UBIGINT"),
        ("polars_version", "VARCHAR"),
        ("duckdb_version", "VARCHAR"),
        ("source_id", "VARCHAR"),
        ("processor_id", "VARCHAR"),
        ("config_hash", "VARCHAR"),
        ("customer_column", "VARCHAR"),
        ("customer_dtype", "VARCHAR"),
        ("customer_storage_dtype", "VARCHAR"),
        ("shard_count", "INTEGER"),
        ("history_projection", "VARCHAR"),
    )


def _journal_schema() -> tuple[tuple[str, str], ...]:
    return (
        ("sequence", "BIGINT"),
        ("chunk_id", "VARCHAR"),
        ("raw_fingerprint", "VARCHAR"),
    )


def _metadata_table_sql() -> str:
    columns = [
        "singleton BOOLEAN PRIMARY KEY CHECK (singleton)",
        "schema_revision INTEGER NOT NULL",
        "shard_hash_revision INTEGER NOT NULL",
        "shard_hash_algorithm VARCHAR NOT NULL",
        "seed_0 UBIGINT NOT NULL",
        "seed_1 UBIGINT NOT NULL",
        "seed_2 UBIGINT NOT NULL",
        "seed_3 UBIGINT NOT NULL",
        "polars_version VARCHAR NOT NULL",
        "duckdb_version VARCHAR NOT NULL",
        "source_id VARCHAR NOT NULL",
        "processor_id VARCHAR NOT NULL",
        "config_hash VARCHAR NOT NULL",
        "customer_column VARCHAR NOT NULL",
        "customer_dtype VARCHAR NOT NULL",
        "customer_storage_dtype VARCHAR NOT NULL",
        "shard_count INTEGER NOT NULL",
        "history_projection VARCHAR NOT NULL",
    ]
    return f"CREATE TABLE {_quote_identifier(METADATA_TABLE)} ({', '.join(columns)})"


def _journal_table_sql() -> str:
    return (
        f"CREATE TABLE {_quote_identifier(JOURNAL_TABLE)} ("
        "sequence BIGINT PRIMARY KEY, "
        "chunk_id VARCHAR UNIQUE NOT NULL, "
        "raw_fingerprint VARCHAR NOT NULL)"
    )


def _projection_json(spec: HistoryProjectionSpec) -> str:
    return json.dumps(
        {
            "columns": list(spec.columns),
            "rank_column": spec.rank_column,
            "exposed_column": spec.exposed_column,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _create_empty_history(
    connection: duckdb.DuckDBPyConnection,
    selected: Sequence[str],
) -> None:
    connection.execute(
        f"CREATE TABLE {_quote_identifier(HISTORY_TABLE)} AS "
        f"SELECT {_identifier_csv(selected)} FROM {_quote_identifier(CURRENT_TABLE)} "
        "WHERE FALSE"
    )


def _promote_history_schema(
    connection: duckdb.DuckDBPyConnection,
    selected: Sequence[str],
) -> None:
    """Widen durable history with Polars diagonal-relaxed type semantics."""

    history_empty = _empty_polars_select(connection, HISTORY_TABLE, selected)
    current_empty = _empty_polars_select(connection, CURRENT_TABLE, selected)
    try:
        promoted_schema = pl.concat(
            [history_empty.lazy(), current_empty.lazy()],
            how="diagonal_relaxed",
        ).collect_schema()
    except pl.exceptions.PolarsError as exc:
        raise CheckpointValidationError(
            "rolling checkpoint history schema cannot be promoted to staged input"
        ) from exc
    if history_empty.schema == promoted_schema:
        return

    replacement = "__valuestream_checkpoint_history_retyped"
    registered = "_valuestream_checkpoint_promoted_schema"
    connection.register(registered, pl.DataFrame(schema=promoted_schema))
    try:
        connection.execute(
            f"CREATE TABLE {_quote_identifier(replacement)} AS "
            f"SELECT * FROM {_quote_identifier(registered)}"
        )
        connection.execute(
            f"INSERT INTO {_quote_identifier(replacement)} "
            f"SELECT {_identifier_csv(selected)} FROM {_quote_identifier(HISTORY_TABLE)}"
        )
        connection.execute(f"DROP TABLE {_quote_identifier(HISTORY_TABLE)}")
        connection.execute(
            f"ALTER TABLE {_quote_identifier(replacement)} "
            f"RENAME TO {_quote_identifier(HISTORY_TABLE)}"
        )
    except (duckdb.Error, ValueError) as exc:
        raise CheckpointValidationError(
            "rolling checkpoint history values cannot be promoted to staged input"
        ) from exc
    finally:
        connection.unregister(registered)


def _empty_polars_select(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    columns: Sequence[str],
) -> pl.DataFrame:
    arrow = connection.execute(
        f"SELECT {_identifier_csv(columns)} FROM {_quote_identifier(table)} LIMIT 0"
    ).to_arrow_table()
    return cast(pl.DataFrame, pl.from_arrow(arrow))


def _selected_schema(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    columns: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    rows = connection.execute(
        f"DESCRIBE SELECT {_identifier_csv(columns)} FROM {_quote_identifier(table)}"
    ).fetchall()
    return tuple((str(row[0]), str(row[1])) for row in rows)


def _table_schema(
    connection: duckdb.DuckDBPyConnection,
    table: str,
) -> tuple[tuple[str, str], ...]:
    rows = connection.execute(f"DESCRIBE SELECT * FROM {_quote_identifier(table)}").fetchall()
    return tuple((str(row[0]), str(row[1])) for row in rows)


def _validate_table_schema(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    expected: tuple[tuple[str, str], ...],
) -> None:
    actual = _table_schema(connection, table)
    if actual != expected:
        raise CheckpointValidationError(
            f"rolling checkpoint table {table!r} schema does not match revision "
            f"{CHECKPOINT_SCHEMA_REVISION}"
        )


def _persistent_table_names(connection: duckdb.DuckDBPyConnection) -> set[str]:
    rows = connection.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_catalog = current_database() AND table_schema = 'main'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _table_exists(connection: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = connection.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_catalog = current_database() AND table_schema = 'main' "
        "AND table_name = ?",
        [table],
    ).fetchone()
    return row is not None and int(row[0]) == 1


def _table_row_count(connection: duckdb.DuckDBPyConnection, table: str) -> int:
    row = connection.execute(f"SELECT count(*) FROM {_quote_identifier(table)}").fetchone()
    if row is None:  # pragma: no cover - aggregate always returns one row
        raise RuntimeError(f"checkpoint table {table!r} row-count query returned no row")
    return int(row[0])


def _table_shards(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    *,
    shard_count: int,
) -> tuple[tuple[int, int], ...]:
    rows = connection.execute(
        f"SELECT {_quote_identifier(SHARD_COLUMN)}, count(*) "
        f"FROM {_quote_identifier(table)} "
        f"GROUP BY {_quote_identifier(SHARD_COLUMN)} "
        f"ORDER BY {_quote_identifier(SHARD_COLUMN)}"
    ).fetchall()
    shards: list[tuple[int, int]] = []
    for raw_shard_id, raw_rows in rows:
        if raw_shard_id is None:
            raise ValueError("checkpoint shard column cannot contain nulls")
        shard_id = int(raw_shard_id)
        if not 0 <= shard_id < shard_count:
            raise CheckpointValidationError(
                f"checkpoint shard {shard_id} is outside configured count {shard_count}"
            )
        shards.append((shard_id, int(raw_rows)))
    return tuple(shards)


def _null_count(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    column: str,
) -> int:
    row = connection.execute(
        f"SELECT count(*) FROM {_quote_identifier(table)} WHERE {_quote_identifier(column)} IS NULL"
    ).fetchone()
    if row is None:  # pragma: no cover - aggregate always returns one row
        raise RuntimeError("checkpoint null-count query returned no row")
    return int(row[0])


def _polars_dtype_for_column(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    column: str,
) -> str:
    arrow = connection.execute(
        f"SELECT {_quote_identifier(column)} FROM {_quote_identifier(table)} LIMIT 0"
    ).to_arrow_table()
    frame = cast(pl.DataFrame, pl.from_arrow(arrow))
    return str(frame.schema[column])


def _delete_chunks(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    chunk_ids: Sequence[str],
) -> None:
    if not chunk_ids:
        return
    column = CHUNK_ID_COLUMN if table == HISTORY_TABLE else "chunk_id"
    placeholders = ", ".join("?" for _ in chunk_ids)
    connection.execute(
        f"DELETE FROM {_quote_identifier(table)} "
        f"WHERE {_quote_identifier(column)} IN ({placeholders})",
        list(chunk_ids),
    )


def _normalize_expected_history(
    expected: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in expected:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("checkpoint expected history entries must be (chunk_id, fingerprint)")
        chunk_id, raw_fingerprint = item
        _validate_chunk_identity(chunk_id, raw_fingerprint)
        if chunk_id in seen:
            raise ValueError(f"checkpoint expected history repeats chunk {chunk_id!r}")
        seen.add(chunk_id)
        normalized.append((chunk_id, raw_fingerprint))
    return tuple(normalized)


def _validate_shard_request(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    customer_column: str,
    shard_count: int,
) -> None:
    _validate_shard_count(shard_count)
    columns = frame.columns if isinstance(frame, pl.DataFrame) else frame.collect_schema().names()
    if customer_column not in columns:
        raise ValueError(f"checkpoint frame is missing customer column {customer_column!r}")
    reserved = sorted(set(columns) & _RESERVED_COLUMNS)
    if reserved:
        raise ValueError("checkpoint frame uses reserved column(s): " + ", ".join(reserved))


def _validate_shard_count(shard_count: int) -> None:
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or not 1 <= shard_count <= 4096
    ):
        raise ValueError("checkpoint shard count must be a positive integer no greater than 4096")


def _validate_identity(
    *,
    source_id: str,
    processor_id: str,
    config_hash: str,
) -> None:
    _validate_path_identity(source_id=source_id, processor_id=processor_id)
    if not isinstance(config_hash, str) or _HASH_PATTERN.fullmatch(config_hash) is None:
        raise ValueError("checkpoint config_hash must be a full lowercase sha256 digest")


def _validate_path_identity(*, source_id: str, processor_id: str) -> None:
    for name, value in {
        "source_id": source_id,
        "processor_id": processor_id,
    }.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"checkpoint {name} must be a non-empty string")


def _validate_chunk_identity(chunk_id: str, raw_fingerprint: str) -> None:
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("checkpoint chunk_id must be a non-empty string")
    _chunk_date(chunk_id)
    if not isinstance(raw_fingerprint, str) or _HASH_PATTERN.fullmatch(raw_fingerprint) is None:
        raise ValueError("checkpoint raw_fingerprint must be a full lowercase sha256 digest")


def _chunk_date(chunk_id: str) -> dt.date:
    try:
        parsed = dt.date.fromisoformat(chunk_id)
    except ValueError as exc:
        raise ValueError(
            f"checkpoint chunk_id must be an ISO YYYY-MM-DD calendar date; got {chunk_id!r}"
        ) from exc
    if parsed.isoformat() != chunk_id:
        raise ValueError(
            f"checkpoint chunk_id must be an ISO YYYY-MM-DD calendar date; got {chunk_id!r}"
        )
    return parsed


def _customer_dtypes_compatible(left: str, right: str) -> bool:
    return left in {right, "Null"} or right == "Null"


def _configure_connection(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("SET TimeZone = 'UTC'")


def _checkpoint_connection(connection: duckdb.DuckDBPyConnection) -> None:
    """Fold DuckDB WAL/free blocks without closing the long-lived writer."""

    connection.execute("CHECKPOINT")


def _remove_database_files(path: Path) -> None:
    for candidate in (path, path.with_name(f"{path.name}.wal")):
        with suppress(FileNotFoundError):
            candidate.unlink()


def _fsync_directory(path: Path) -> None:
    """Durably publish a same-directory replacement where supported."""

    if os.name == "nt":  # Opening directories as file descriptors is unsupported on Windows.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _component(value: str) -> str:
    return quote(value, safe="-._~")


def _identifier_csv(columns: Sequence[str]) -> str:
    return ", ".join(_quote_identifier(column) for column in columns)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


__all__ = [
    "CHECKPOINT_SCHEMA_REVISION",
    "CHUNK_ID_COLUMN",
    "CURRENT_TABLE",
    "HISTORY_TABLE",
    "JOURNAL_TABLE",
    "METADATA_TABLE",
    "ROLLING_DATABASE_FILENAME",
    "SHARD_COLUMN",
    "SHARD_HASH_ALGORITHM",
    "SHARD_HASH_REVISION",
    "SHARD_HASH_SEEDS",
    "CheckpointJournalEntry",
    "CheckpointValidationError",
    "HistoryProjectionSpec",
    "RollingCheckpoint",
    "assign_customer_shard",
    "rolling_checkpoint_path",
]
