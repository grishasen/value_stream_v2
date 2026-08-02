"""Current-schema rolling DuckDB processor-state tests."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import polars as pl
import pytest

from valuestream.store import processor_state
from valuestream.store.processor_state import (
    CHECKPOINT_SCHEMA_REVISION,
    CHUNK_ID_COLUMN,
    CURRENT_TABLE,
    HISTORY_TABLE,
    JOURNAL_TABLE,
    METADATA_TABLE,
    ROLLING_DATABASE_FILENAME,
    SHARD_COLUMN,
    CheckpointJournalEntry,
    CheckpointValidationError,
    HistoryProjectionSpec,
    RollingCheckpoint,
    assign_customer_shard,
    rolling_checkpoint_path,
)

CONFIG_HASH = "a" * 64
PATH_IDENTITY = {
    "source_id": "ih/source",
    "processor_id": "frequency response",
}
HISTORY_SPEC = HistoryProjectionSpec(
    columns=("CustomerID", "DecisionTime", "LocalOrder", "Rank", "Exposed"),
    rank_column="Rank",
    exposed_column="Exposed",
)


def _contacts(*, customer_dtype: pl.DataType = pl.String) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "CustomerID": pl.Series(
                ["customer-a", "customer-b", "customer-a", "customer-c"],
                dtype=customer_dtype,
            ),
            "ActionName": ["A", "A", "B", "C"],
            "DecisionTime": [1, 2, 3, 4],
            "LocalOrder": [0, 1, 2, 3],
            "Rank": [1, 2, 1, 1],
            "Exposed": [True, True, False, True],
        }
    )


def _rolling(
    tmp_path: Path,
    *,
    config_hash: str = CONFIG_HASH,
    customer_dtype: str = "String",
    shard_count: int = 8,
    retention_days: int = 7,
    history_projection: HistoryProjectionSpec = HISTORY_SPEC,
    force: bool = False,
) -> RollingCheckpoint:
    return RollingCheckpoint(
        tmp_path,
        **PATH_IDENTITY,
        config_hash=config_hash,
        customer_column="CustomerID",
        customer_dtype=customer_dtype,
        shard_count=shard_count,
        retention_days=retention_days,
        history_projection=history_projection,
        force=force,
    )


def _fingerprint(index: int) -> str:
    return f"{index:064x}"


def _stage_and_commit(
    checkpoint: RollingCheckpoint,
    chunk_id: str,
    fingerprint: str,
    frame: pl.DataFrame | None = None,
) -> None:
    checkpoint.stage_current(
        _contacts() if frame is None else frame,
        chunk_id=chunk_id,
        raw_fingerprint=fingerprint,
    )
    checkpoint.commit_staged()


def _persistent_tables(connection: duckdb.DuckDBPyConnection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_catalog = current_database() AND table_schema = 'main'"
        ).fetchall()
    }


@pytest.mark.unit
def test_rolling_path_is_stable_and_contains_only_source_and_processor(tmp_path: Path) -> None:
    path = rolling_checkpoint_path(tmp_path, **PATH_IDENTITY)

    assert path == (
        tmp_path
        / ".valuestream"
        / "state"
        / "frequency_response"
        / "source=ih%2Fsource"
        / "processor=frequency%20response"
        / ROLLING_DATABASE_FILENAME
    )
    assert not any(
        part.startswith(("schema=", "hash=", "polars=", "config=", "layout="))
        for part in path.parts
    )

    with _rolling(tmp_path):
        pass
    assert path.is_file()
    assert not (tmp_path / "aggregates").exists()


@pytest.mark.unit
def test_native_polars_customer_shards_are_deterministic() -> None:
    frame = _contacts()
    eager = assign_customer_shard(frame, customer_column="CustomerID", shard_count=8)
    repeated = assign_customer_shard(frame, customer_column="CustomerID", shard_count=8)
    lazy = assign_customer_shard(
        frame.lazy(), customer_column="CustomerID", shard_count=8
    ).collect()

    assert eager.get_column(SHARD_COLUMN).to_list() == repeated.get_column(SHARD_COLUMN).to_list()
    assert eager.get_column(SHARD_COLUMN).to_list() == lazy.get_column(SHARD_COLUMN).to_list()
    assert (
        eager.filter(pl.col("CustomerID") == "customer-a").get_column(SHARD_COLUMN).n_unique() == 1
    )


@pytest.mark.unit
def test_open_initializes_metadata_and_reopens_same_journal(tmp_path: Path) -> None:
    with _rolling(tmp_path) as checkpoint:
        assert checkpoint.connection is not None
        assert checkpoint.journal == ()
        assert _persistent_tables(checkpoint.connection) == {METADATA_TABLE, JOURNAL_TABLE}
        _stage_and_commit(checkpoint, "2026-07-30", _fingerprint(1))
        expected = (
            CheckpointJournalEntry(
                sequence=0,
                chunk_id="2026-07-30",
                raw_fingerprint=_fingerprint(1),
            ),
        )
        assert checkpoint.journal_entries == expected

    with _rolling(tmp_path) as reopened:
        assert reopened.journal == expected
        assert _persistent_tables(reopened.connection) == {
            METADATA_TABLE,
            JOURNAL_TABLE,
            HISTORY_TABLE,
        }


@pytest.mark.unit
def test_stage_current_uses_arrow_stream_and_keeps_complete_temp_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    original_collect_batches = pl.LazyFrame.collect_batches

    def track_collect_batches(self: pl.LazyFrame, **kwargs: object):
        calls.append(kwargs)
        result = original_collect_batches(self, **kwargs)
        assert hasattr(result, "__arrow_c_stream__")
        return result

    def fail_collect(*args: object, **kwargs: object) -> None:
        raise AssertionError("whole-frame LazyFrame.collect must not be used")

    def fail_sink_parquet(*args: object, **kwargs: object) -> None:
        raise AssertionError("rolling state must not stage Parquet")

    monkeypatch.setattr(pl.LazyFrame, "collect_batches", track_collect_batches)
    monkeypatch.setattr(pl.LazyFrame, "collect", fail_collect)
    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", fail_sink_parquet)

    with _rolling(tmp_path) as checkpoint:
        shard_ids = checkpoint.stage_current(
            _contacts().lazy(),
            chunk_id="2026-07-31",
            raw_fingerprint=_fingerprint(2),
            engine="streaming",
        )
        columns = [
            str(row[0])
            for row in checkpoint.connection.execute(
                f'DESCRIBE SELECT * FROM "{CURRENT_TABLE}"'
            ).fetchall()
        ]
        rows = checkpoint.connection.execute(f'SELECT * FROM "{CURRENT_TABLE}"').fetchall()
        physical_shards = (
            checkpoint.connection.execute(f'SELECT "{SHARD_COLUMN}" FROM "{CURRENT_TABLE}"')
            .fetchnumpy()[SHARD_COLUMN]
            .tolist()
        )

        assert shard_ids == tuple(sorted(set(physical_shards)))
        assert physical_shards == sorted(physical_shards)
        assert len(rows) == _contacts().height
        assert columns == [*_contacts().columns, SHARD_COLUMN, CHUNK_ID_COLUMN]
        assert checkpoint.staged_chunk_id == "2026-07-31"
        assert checkpoint.journal == ()

    assert calls == [{"maintain_order": True, "lazy": True, "engine": "streaming"}]


@pytest.mark.unit
def test_commit_promotes_only_rank1_exposed_history_and_journals(tmp_path: Path) -> None:
    with _rolling(tmp_path) as checkpoint:
        checkpoint.stage_current(
            _contacts(),
            chunk_id="2026-07-31",
            raw_fingerprint=_fingerprint(3),
        )
        checkpoint.commit_staged()

        rows = checkpoint.connection.execute(
            f'SELECT "CustomerID", "Rank", "Exposed", "{CHUNK_ID_COLUMN}" '
            f'FROM "{HISTORY_TABLE}" ORDER BY "CustomerID"'
        ).fetchall()
        assert rows == [
            ("customer-a", 1, True, "2026-07-31"),
            ("customer-c", 1, True, "2026-07-31"),
        ]
        assert checkpoint.journal[0].chunk_id == "2026-07-31"
        assert checkpoint.staged_chunk_id is None
        with pytest.raises(duckdb.CatalogException):
            checkpoint.connection.execute(f'SELECT * FROM "{CURRENT_TABLE}"')


@pytest.mark.unit
def test_abort_discards_temp_current_without_mutating_history(tmp_path: Path) -> None:
    with _rolling(tmp_path) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-29", _fingerprint(1))
        before = checkpoint.connection.execute(
            f'SELECT * FROM "{HISTORY_TABLE}" ORDER BY "{SHARD_COLUMN}"'
        ).fetchall()
        checkpoint.stage_current(
            _contacts(),
            chunk_id="2026-07-30",
            raw_fingerprint=_fingerprint(2),
        )
        checkpoint.abort_staged()

        assert checkpoint.journal[0].chunk_id == "2026-07-29"
        assert (
            checkpoint.connection.execute(
                f'SELECT * FROM "{HISTORY_TABLE}" ORDER BY "{SHARD_COLUMN}"'
            ).fetchall()
            == before
        )


@pytest.mark.unit
def test_reconcile_reuses_prefix_deletes_outside_closure_and_returns_suffix(
    tmp_path: Path,
) -> None:
    entries = [
        ("2026-07-28", _fingerprint(1)),
        ("2026-07-29", _fingerprint(2)),
        ("2026-07-30", _fingerprint(3)),
    ]
    with _rolling(tmp_path) as checkpoint:
        for chunk_id, fingerprint in entries:
            _stage_and_commit(checkpoint, chunk_id, fingerprint)

        expected = [
            ("2026-07-29", _fingerprint(2)),
            ("2026-07-30", _fingerprint(3)),
            ("2026-07-31", _fingerprint(4)),
        ]
        assert checkpoint.reconcile_history(expected) == ("2026-07-31",)
        assert [entry.chunk_id for entry in checkpoint.journal] == [
            "2026-07-29",
            "2026-07-30",
        ]
        history_chunks = checkpoint.connection.execute(
            f'SELECT DISTINCT "{CHUNK_ID_COLUMN}" FROM "{HISTORY_TABLE}" '
            f'ORDER BY "{CHUNK_ID_COLUMN}"'
        ).fetchall()
        assert history_chunks == [("2026-07-29",), ("2026-07-30",)]

        _stage_and_commit(checkpoint, "2026-07-31", _fingerprint(4))
        assert checkpoint.reconcile_history(expected[:2]) == ()
        assert [entry.chunk_id for entry in checkpoint.journal] == [
            "2026-07-29",
            "2026-07-30",
        ]


@pytest.mark.unit
def test_reconcile_fingerprint_mismatch_atomically_resets(tmp_path: Path) -> None:
    with _rolling(tmp_path) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-29", _fingerprint(1))
        _stage_and_commit(checkpoint, "2026-07-30", _fingerprint(2))
        expected = (
            ("2026-07-29", _fingerprint(1)),
            ("2026-07-30", _fingerprint(9)),
        )

        assert checkpoint.reconcile_history(expected) == (
            "2026-07-29",
            "2026-07-30",
        )
        assert checkpoint.journal == ()
        assert HISTORY_TABLE not in _persistent_tables(checkpoint.connection)


@pytest.mark.unit
def test_reconcile_nonprefix_history_resets_instead_of_using_suffix(tmp_path: Path) -> None:
    with _rolling(tmp_path) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-30", _fingerprint(2))
        expected = (
            ("2026-07-29", _fingerprint(1)),
            ("2026-07-30", _fingerprint(2)),
        )

        assert checkpoint.reconcile_history(expected) == (
            "2026-07-29",
            "2026-07-30",
        )
        assert checkpoint.journal == ()


@pytest.mark.unit
def test_retention_prunes_oldest_history_and_journal_chunks(tmp_path: Path) -> None:
    with _rolling(tmp_path, retention_days=2) as checkpoint:
        for index, chunk_id in enumerate(
            ("2026-07-28", "2026-07-29", "2026-07-30"),
            start=1,
        ):
            _stage_and_commit(checkpoint, chunk_id, _fingerprint(index))

        assert [entry.chunk_id for entry in checkpoint.journal] == [
            "2026-07-29",
            "2026-07-30",
        ]
        chunks = checkpoint.connection.execute(
            f'SELECT DISTINCT "{CHUNK_ID_COLUMN}" FROM "{HISTORY_TABLE}" '
            f'ORDER BY "{CHUNK_ID_COLUMN}"'
        ).fetchall()
        assert chunks == [("2026-07-29",), ("2026-07-30",)]


@pytest.mark.unit
def test_retention_uses_calendar_cutoff_when_source_days_are_missing(tmp_path: Path) -> None:
    with _rolling(tmp_path, retention_days=7) as checkpoint:
        for index, chunk_id in enumerate(
            ("2026-07-01", "2026-07-13", "2026-07-14", "2026-07-20"),
            start=1,
        ):
            _stage_and_commit(checkpoint, chunk_id, _fingerprint(index))

        assert [entry.chunk_id for entry in checkpoint.journal] == [
            "2026-07-14",
            "2026-07-20",
        ]


@pytest.mark.unit
def test_commits_must_follow_increasing_calendar_order(tmp_path: Path) -> None:
    with _rolling(tmp_path) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-30", _fingerprint(1))
        checkpoint.stage_current(
            _contacts(),
            chunk_id="2026-07-29",
            raw_fingerprint=_fingerprint(2),
        )
        with pytest.raises(CheckpointValidationError, match="increasing calendar order"):
            checkpoint.commit_staged()
        checkpoint.abort_staged()


@pytest.mark.unit
def test_retention_can_be_pruned_without_staging_a_target(tmp_path: Path) -> None:
    with _rolling(tmp_path, retention_days=7) as checkpoint:
        for index, chunk_id in enumerate(
            ("2026-07-28", "2026-07-29", "2026-07-30"),
            start=1,
        ):
            _stage_and_commit(checkpoint, chunk_id, _fingerprint(index))

    with _rolling(tmp_path, retention_days=2) as checkpoint:
        checkpoint.prune_retention()

        assert [entry.chunk_id for entry in checkpoint.journal] == [
            "2026-07-29",
            "2026-07-30",
        ]
        chunks = checkpoint.connection.execute(
            f'SELECT DISTINCT "{CHUNK_ID_COLUMN}" FROM "{HISTORY_TABLE}" '
            f'ORDER BY "{CHUNK_ID_COLUMN}"'
        ).fetchall()
        assert chunks == [("2026-07-29",), ("2026-07-30",)]


@pytest.mark.unit
def test_thirty_daily_commits_prune_to_seven_and_checkpoint_without_closing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _rolling(tmp_path, retention_days=7) as checkpoint:
        connection = checkpoint.connection
        checkpoint_calls: list[duckdb.DuckDBPyConnection] = []
        original_checkpoint = processor_state._checkpoint_connection

        def track_checkpoint(candidate: duckdb.DuckDBPyConnection) -> None:
            checkpoint_calls.append(candidate)
            original_checkpoint(candidate)

        monkeypatch.setattr(processor_state, "_checkpoint_connection", track_checkpoint)
        first_day = dt.date(2026, 7, 1)
        chunk_ids = [(first_day + dt.timedelta(days=offset)).isoformat() for offset in range(30)]
        for index, chunk_id in enumerate(chunk_ids, start=1):
            _stage_and_commit(checkpoint, chunk_id, _fingerprint(index))

        assert checkpoint_calls == [connection]
        assert checkpoint.connection is connection
        assert connection.execute("SELECT 1").fetchone() == (1,)
        assert [entry.chunk_id for entry in checkpoint.journal] == chunk_ids[-7:]
        history_chunks = connection.execute(
            f'SELECT DISTINCT "{CHUNK_ID_COLUMN}" FROM "{HISTORY_TABLE}" '
            f'ORDER BY "{CHUNK_ID_COLUMN}"'
        ).fetchall()
        assert history_chunks == [(chunk_id,) for chunk_id in chunk_ids[-7:]]


@pytest.mark.unit
def test_history_schema_promotes_non_customer_columns_across_chunks(tmp_path: Path) -> None:
    history_projection = HistoryProjectionSpec(
        columns=(
            "CustomerID",
            "ActionName",
            "DecisionTime",
            "LocalOrder",
            "Rank",
            "Exposed",
        ),
        rank_column="Rank",
        exposed_column="Exposed",
    )
    action_values = [1, 2, 3, 4]
    first = _contacts().with_columns(pl.Series("ActionName", action_values, dtype=pl.Int32))
    second = _contacts().with_columns(pl.Series("ActionName", action_values, dtype=pl.Int64))
    third = _contacts().with_columns(
        pl.Series("ActionName", [str(value) for value in action_values])
    )

    with _rolling(tmp_path, history_projection=history_projection) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-28", _fingerprint(1), first)
        _stage_and_commit(checkpoint, "2026-07-29", _fingerprint(2), second)
        schema = pl.from_arrow(
            checkpoint.connection.execute(
                f'SELECT "ActionName" FROM "{HISTORY_TABLE}" LIMIT 0'
            ).to_arrow_table()
        ).schema
        assert schema["ActionName"] == pl.Int64

        replacement = "__valuestream_checkpoint_history_retyped"
        checkpoint.connection.execute(f'CREATE TEMP TABLE "{replacement}" (blocked INTEGER)')
        _stage_and_commit(checkpoint, "2026-07-30", _fingerprint(3), first)
        checkpoint.connection.execute(f'DROP TABLE "{replacement}"')

        _stage_and_commit(checkpoint, "2026-07-31", _fingerprint(4), third)
        recovered = pl.from_arrow(
            checkpoint.connection.execute(
                f'SELECT "ActionName" FROM "{HISTORY_TABLE}" ORDER BY "{CHUNK_ID_COLUMN}"'
            ).to_arrow_table()
        )
        assert isinstance(recovered, pl.DataFrame)
        assert recovered.schema["ActionName"] == pl.String
        assert set(recovered.get_column("ActionName")) == {"1", "4"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "customer_type",
    [pl.Categorical, pl.Enum(["customer-a", "customer-b", "customer-c"])],
)
def test_dictionary_customer_type_keeps_logical_and_physical_types_separate(
    tmp_path: Path,
    customer_type: pl.DataType,
) -> None:
    frame = _contacts(customer_dtype=customer_type)
    logical_dtype = str(customer_type)
    expected_shards = (
        assign_customer_shard(frame, customer_column="CustomerID", shard_count=8)
        .filter((pl.col("Rank") == 1) & pl.col("Exposed"))
        .select("CustomerID", SHARD_COLUMN)
        .sort("CustomerID")
        .rows()
    )
    with _rolling(tmp_path, customer_dtype=logical_dtype) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-29", _fingerprint(1), frame)
        metadata = checkpoint.connection.execute(
            f'SELECT customer_dtype, customer_storage_dtype FROM "{METADATA_TABLE}"'
        ).fetchone()
        assert metadata == (logical_dtype, "String")
        persisted = pl.from_arrow(
            checkpoint.connection.execute(
                f'SELECT "CustomerID", "{SHARD_COLUMN}" FROM "{HISTORY_TABLE}" '
                'ORDER BY "CustomerID"'
            ).to_arrow_table()
        )
        assert isinstance(persisted, pl.DataFrame)
        assert persisted.schema["CustomerID"] == pl.String
        assert persisted.rows() == expected_shards

    with _rolling(tmp_path, customer_dtype=logical_dtype) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-30", _fingerprint(2), frame)
        assert len(checkpoint.journal) == 2


@pytest.mark.unit
def test_dictionary_customer_logical_dtype_drift_is_rejected(tmp_path: Path) -> None:
    categorical = _contacts(customer_dtype=pl.Categorical)
    with _rolling(tmp_path, customer_dtype="Categorical") as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-29", _fingerprint(1), categorical)

    with (
        pytest.raises(CheckpointValidationError, match="customer dtype"),
        _rolling(tmp_path, customer_dtype="String"),
    ):
        pass


@pytest.mark.unit
def test_enum_customer_category_order_drift_is_rejected(tmp_path: Path) -> None:
    first_type = pl.Enum(["customer-a", "customer-b", "customer-c"])
    second_type = pl.Enum(["customer-c", "customer-b", "customer-a"])
    with _rolling(tmp_path, customer_dtype=str(first_type)) as checkpoint:
        _stage_and_commit(
            checkpoint,
            "2026-07-29",
            _fingerprint(1),
            _contacts(customer_dtype=first_type),
        )

    with (
        pytest.raises(CheckpointValidationError, match="customer dtype"),
        _rolling(tmp_path, customer_dtype=str(second_type)),
    ):
        pass


@pytest.mark.unit
@pytest.mark.parametrize("customer_type", [pl.String, pl.Categorical])
def test_empty_null_customer_day_does_not_lock_dynamic_history_type(
    tmp_path: Path,
    customer_type: pl.DataType,
) -> None:
    empty = pl.DataFrame(
        schema={
            "CustomerID": pl.Null,
            "ActionName": pl.String,
            "DecisionTime": pl.Int64,
            "LocalOrder": pl.Int64,
            "Rank": pl.Int64,
            "Exposed": pl.Boolean,
        }
    )
    with _rolling(tmp_path, customer_dtype="Null") as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-29", _fingerprint(1), empty)
        assert checkpoint.journal[0].chunk_id == "2026-07-29"

    with _rolling(tmp_path, customer_dtype=str(customer_type)) as checkpoint:
        _stage_and_commit(
            checkpoint,
            "2026-07-30",
            _fingerprint(2),
            _contacts(customer_dtype=customer_type),
        )
        assert (
            checkpoint.connection.execute(f'SELECT count(*) FROM "{HISTORY_TABLE}"').fetchone()[0]
            == 2
        )


@pytest.mark.unit
def test_failed_stage_does_not_leak_uncommitted_customer_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _rolling(tmp_path, customer_dtype="Null") as checkpoint:
        original = checkpoint._ensure_history_table

        def fail_history(*args: object, **kwargs: object) -> None:
            raise RuntimeError("injected history failure")

        monkeypatch.setattr(checkpoint, "_ensure_history_table", fail_history)
        with pytest.raises(RuntimeError, match="injected history failure"):
            checkpoint.stage_current(
                _contacts(customer_dtype=pl.Categorical),
                chunk_id="2026-07-29",
                raw_fingerprint=_fingerprint(1),
            )
        metadata = checkpoint.connection.execute(
            f'SELECT customer_dtype, customer_storage_dtype FROM "{METADATA_TABLE}"'
        ).fetchone()
        assert metadata == ("Null", "Null")

        monkeypatch.setattr(checkpoint, "_ensure_history_table", original)
        _stage_and_commit(checkpoint, "2026-07-30", _fingerprint(2))
        assert checkpoint.journal[0].chunk_id == "2026-07-30"


@pytest.mark.unit
def test_nonempty_customer_dtype_drift_is_rejected(tmp_path: Path) -> None:
    with _rolling(tmp_path) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-29", _fingerprint(1))

    with (
        pytest.raises(CheckpointValidationError, match="customer dtype"),
        _rolling(tmp_path, customer_dtype="Int64"),
    ):
        pass


@pytest.mark.unit
def test_reset_clears_history_and_journal_but_preserves_identity(tmp_path: Path) -> None:
    with _rolling(tmp_path) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-29", _fingerprint(1))
        checkpoint.reset()

        assert checkpoint.journal == ()
        assert HISTORY_TABLE not in _persistent_tables(checkpoint.connection)
        assert METADATA_TABLE in _persistent_tables(checkpoint.connection)

    with _rolling(tmp_path) as reopened:
        assert reopened.journal == ()


@pytest.mark.unit
def test_corrupt_database_fails_loudly_and_force_recreates(tmp_path: Path) -> None:
    path = rolling_checkpoint_path(tmp_path, **PATH_IDENTITY)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a DuckDB database")

    with pytest.raises(CheckpointValidationError, match="cannot open"), _rolling(tmp_path):
        pass

    with _rolling(tmp_path, force=True) as recovered:
        assert recovered.journal == ()
        assert _persistent_tables(recovered.connection) == {METADATA_TABLE, JOURNAL_TABLE}


@pytest.mark.unit
def test_orphan_wal_is_corruption_but_force_can_recover(tmp_path: Path) -> None:
    path = rolling_checkpoint_path(tmp_path, **PATH_IDENTITY)
    path.parent.mkdir(parents=True)
    path.with_name(f"{path.name}.wal").write_bytes(b"orphan")

    with pytest.raises(CheckpointValidationError, match="orphan WAL"), _rolling(tmp_path):
        pass
    with _rolling(tmp_path, force=True) as recovered:
        assert recovered.journal == ()


@pytest.mark.unit
def test_config_change_reinitializes_same_stable_database(tmp_path: Path) -> None:
    with _rolling(tmp_path) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-30", _fingerprint(1))
        path = checkpoint.path

    replacement_hash = "f" * 64
    with _rolling(tmp_path, config_hash=replacement_hash) as rebuilt:
        assert rebuilt.path == path
        assert rebuilt.journal == ()
        assert _persistent_tables(rebuilt.connection) == {METADATA_TABLE, JOURNAL_TABLE}
        metadata = rebuilt.connection.execute(
            f'SELECT schema_revision, config_hash FROM "{METADATA_TABLE}"'
        ).fetchone()
        assert metadata == (CHECKPOINT_SCHEMA_REVISION, replacement_hash)


@pytest.mark.unit
def test_failed_compatibility_rebuild_preserves_old_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _rolling(tmp_path) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-30", _fingerprint(1))
        old_journal = checkpoint.journal
        path = checkpoint.path

    replacement_hash = "f" * 64
    replacement = _rolling(tmp_path, config_hash=replacement_hash)

    def fail_initialize(_connection: duckdb.DuckDBPyConnection) -> None:
        raise ValueError("injected initialization failure")

    monkeypatch.setattr(replacement, "_initialize", fail_initialize)
    with pytest.raises(CheckpointValidationError, match="cannot rebuild"), replacement:
        pass

    with _rolling(tmp_path) as original:
        assert original.path == path
        assert original.journal == old_journal
    assert not list(path.parent.glob(f".{path.name}.*.tmp*"))

    with _rolling(tmp_path, config_hash=replacement_hash) as rebuilt:
        assert rebuilt.path == path
        assert rebuilt.journal == ()


@pytest.mark.unit
def test_shard_count_change_reinitializes_same_stable_database(tmp_path: Path) -> None:
    with _rolling(tmp_path, shard_count=8) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-30", _fingerprint(1))
        path = checkpoint.path

    with _rolling(tmp_path, shard_count=4) as rebuilt:
        assert rebuilt.path == path
        assert rebuilt.journal == ()
        assert rebuilt.connection.execute(
            f'SELECT shard_count FROM "{METADATA_TABLE}"'
        ).fetchone() == (4,)


@pytest.mark.unit
def test_unsupported_schema_revision_reinitializes_to_current_only(tmp_path: Path) -> None:
    assert CHECKPOINT_SCHEMA_REVISION == 7
    with _rolling(tmp_path) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-30", _fingerprint(1))
        path = checkpoint.path
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            f'UPDATE "{METADATA_TABLE}" SET schema_revision = ?',
            [CHECKPOINT_SCHEMA_REVISION - 1],
        )

    with _rolling(tmp_path) as rebuilt:
        assert rebuilt.path == path
        assert rebuilt.journal == ()
        assert _persistent_tables(rebuilt.connection) == {METADATA_TABLE, JOURNAL_TABLE}
        assert rebuilt.connection.execute(
            f'SELECT schema_revision FROM "{METADATA_TABLE}"'
        ).fetchone() == (CHECKPOINT_SCHEMA_REVISION,)


@pytest.mark.unit
def test_polars_version_change_reinitializes_same_stable_database(tmp_path: Path) -> None:
    with _rolling(tmp_path) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-30", _fingerprint(1))
        path = checkpoint.path
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            f'UPDATE "{METADATA_TABLE}" SET polars_version = ?',
            ["incompatible-version"],
        )

    with _rolling(tmp_path) as rebuilt:
        assert rebuilt.path == path
        assert rebuilt.journal == ()
        assert rebuilt.connection.execute(
            f'SELECT polars_version FROM "{METADATA_TABLE}"'
        ).fetchone() == (pl.__version__,)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["source_id", "processor_id"])
def test_tampered_path_identity_metadata_is_rejected(tmp_path: Path, field: str) -> None:
    with _rolling(tmp_path) as checkpoint:
        path = checkpoint.path
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            f'UPDATE "{METADATA_TABLE}" SET "{field}" = ?',
            ["tampered"],
        )

    with pytest.raises(CheckpointValidationError, match=field), _rolling(tmp_path):
        pass


@pytest.mark.unit
def test_committed_journal_without_history_table_is_rejected(tmp_path: Path) -> None:
    with _rolling(tmp_path) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-30", _fingerprint(1))
        path = checkpoint.path
    with duckdb.connect(str(path)) as connection:
        connection.execute(f'DROP TABLE "{HISTORY_TABLE}"')

    with (
        pytest.raises(CheckpointValidationError, match=r"journal entries.*no history"),
        _rolling(tmp_path),
    ):
        pass


@pytest.mark.unit
def test_timestamp_round_trip_is_pinned_to_utc(tmp_path: Path) -> None:
    frame = _contacts().with_columns(
        pl.Series(
            "DecisionTime",
            [dt.datetime(2026, 7, 31, hour, tzinfo=dt.UTC) for hour in range(4)],
            dtype=pl.Datetime("us", "UTC"),
        )
    )
    with _rolling(tmp_path) as checkpoint:
        _stage_and_commit(checkpoint, "2026-07-31", _fingerprint(1), frame)
        arrow = checkpoint.connection.execute(
            f'SELECT "DecisionTime" FROM "{HISTORY_TABLE}"'
        ).to_arrow_table()
        recovered = pl.from_arrow(arrow)
        assert isinstance(recovered, pl.DataFrame)
        assert recovered.schema["DecisionTime"] == pl.Datetime("us", "UTC")


@pytest.mark.unit
def test_context_close_aborts_uncommitted_current(tmp_path: Path) -> None:
    with _rolling(tmp_path) as checkpoint:
        checkpoint.stage_current(
            _contacts(),
            chunk_id="2026-07-31",
            raw_fingerprint=_fingerprint(1),
        )

    with _rolling(tmp_path) as reopened:
        assert reopened.journal == ()
        assert CURRENT_TABLE not in _persistent_tables(reopened.connection)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pl.DataFrame({"Other": ["x"]}), "missing customer"),
        (
            _contacts().with_columns(pl.lit(0).alias(SHARD_COLUMN)),
            "reserved column",
        ),
        (
            _contacts().with_columns(pl.lit(None).cast(pl.String).alias("CustomerID")),
            "cannot contain nulls",
        ),
        (_contacts().drop("LocalOrder"), "missing history"),
        (_contacts().with_columns(pl.col("Exposed").cast(pl.Int64)), "must be Boolean"),
    ],
)
def test_stage_rejects_invalid_current_frames(
    tmp_path: Path,
    frame: pl.DataFrame,
    message: str,
) -> None:
    with (
        _rolling(tmp_path) as checkpoint,
        pytest.raises((ValueError, TypeError), match=message),
    ):
        checkpoint.stage_current(
            frame,
            chunk_id="2026-07-31",
            raw_fingerprint=_fingerprint(1),
        )


@pytest.mark.unit
def test_expected_history_requires_unique_valid_entries(tmp_path: Path) -> None:
    with _rolling(tmp_path) as checkpoint:
        with pytest.raises(ValueError, match="repeats chunk"):
            checkpoint.reconcile_history(
                [
                    ("2026-07-30", _fingerprint(1)),
                    ("2026-07-30", _fingerprint(1)),
                ]
            )
        with pytest.raises(ValueError, match="raw_fingerprint"):
            checkpoint.reconcile_history([("2026-07-30", "short")])


@pytest.mark.unit
def test_hashes_and_constructor_bounds_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="config_hash"):
        _rolling(tmp_path, config_hash="short")
    with pytest.raises(ValueError, match="source_id"):
        rolling_checkpoint_path(
            tmp_path,
            source_id="",
            processor_id=PATH_IDENTITY["processor_id"],
        )
    with pytest.raises(ValueError, match="shard count"):
        RollingCheckpoint(
            tmp_path,
            **PATH_IDENTITY,
            config_hash=CONFIG_HASH,
            customer_column="CustomerID",
            customer_dtype="String",
            shard_count=0,
            retention_days=7,
            history_projection=HISTORY_SPEC,
        )
    with pytest.raises(ValueError, match="retention_days"):
        RollingCheckpoint(
            tmp_path,
            **PATH_IDENTITY,
            config_hash=CONFIG_HASH,
            customer_column="CustomerID",
            customer_dtype="String",
            shard_count=8,
            retention_days=0,
            history_projection=HISTORY_SPEC,
        )
