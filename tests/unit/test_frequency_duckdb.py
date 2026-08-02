"""Differential contracts for native SQL over rolling frequency state."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from valuestream.config import model
from valuestream.engine.frequency import DuckDBFrequencySession
from valuestream.processors.context import ChunkContext
from valuestream.processors.frequency_fields import (
    CHECKPOINT_SHARD_COLUMN,
    EXPOSED_COLUMN,
    RANK_COLUMN,
)
from valuestream.processors.frequency_response import FrequencyResponseProcessor

_CHUNK_ID_COLUMN = "__valuestream_checkpoint_chunk_id"
_SEGMENT_COLUMN = 'Comparison "Group'


def _config() -> model.FrequencyResponseProcessor:
    return model.FrequencyResponseProcessor.model_validate(
        {
            "id": "frequency",
            "source": "interaction_history",
            "kind": "frequency_response",
            "group_by": ["Day", "Placement", _SEGMENT_COLUMN, "ExposureBucket"],
            "time": {"property": "DecisionTime", "grain": "daily"},
            "columns": {
                "customer": "CustomerID",
                "interaction": "InteractionID",
                "action": "ActionID",
                "placement": "Placement",
                "rank": "Rank",
                "outcome": "Outcome",
                "propensity": "Propensity",
                "priority": "Priority",
            },
            "alternative_group_by": ["Placement", _SEGMENT_COLUMN],
            "positive_values": ["Clicked"],
            "exposure_values": ["Impression"],
            "candidate_values": ["Pending"],
            "window_hours": 168,
            "max_frequency": 7,
            "states": {
                "Contacts": {"type": "count"},
                "Clicks": {"type": "count", "source_column": "ClickedContact"},
                "ComparableContacts": {
                    "type": "count",
                    "source_column": "ComparableContact",
                },
                "RunnerAvailableContacts": {
                    "type": "count",
                    "source_column": "RunnerAvailable",
                },
                "RunnerPropensitySum": {
                    "type": "value_sum",
                    "source_column": "RunnerPropensity",
                },
                "RunnerPrioritySum": {
                    "type": "value_sum",
                    "source_column": "RunnerPriorityComparable",
                },
            },
        }
    )


def _row(
    customer: str,
    interaction: str,
    decision_time: dt.datetime,
    *,
    action: str = "A",
    placement: str = "Hero",
    rank: int = 1,
    outcome: str = "Impression",
    propensity: float | None = 0.5,
    priority: float | None = 1.0,
    segment: str | None = "A",
) -> dict[str, Any]:
    return {
        "CustomerID": customer,
        "InteractionID": interaction,
        "ActionID": action,
        "Placement": placement,
        "Rank": rank,
        "Outcome": outcome,
        "Propensity": propensity,
        "Priority": priority,
        _SEGMENT_COLUMN: segment,
        "DecisionTime": decision_time,
    }


def _ctx(chunk_id: str = "2024-01-08") -> ChunkContext:
    return ChunkContext(
        pipeline_run_id="run",
        chunk_id=chunk_id,
        created_at=dt.datetime(2024, 1, 8, 14, tzinfo=dt.UTC),
    )


def _prepared_history(
    processor: FrequencyResponseProcessor,
    rows: list[dict[str, Any]],
) -> pl.DataFrame:
    target = processor.checkpoint_contacts_lazy(pl.DataFrame(rows).lazy())
    return processor.checkpoint_history_contacts_lazy(target).collect()


def _prepared_current(
    processor: FrequencyResponseProcessor,
    rows: list[dict[str, Any]],
) -> pl.DataFrame:
    return processor.checkpoint_contacts_lazy(pl.DataFrame(rows).lazy()).collect()


def _routed(
    frame: pl.DataFrame,
    *,
    chunk_id: str,
    shard_column: str,
    chunk_id_column: str,
) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.col("CustomerID") == "other-shard").then(1).otherwise(0).alias(shard_column),
        pl.lit(chunk_id).alias(chunk_id_column),
    )


def _create_rolling_tables(
    connection: duckdb.DuckDBPyConnection,
    *,
    history: Sequence[tuple[str, pl.DataFrame]],
    current: Sequence[tuple[str, pl.DataFrame]],
    empty_history: pl.DataFrame,
    history_table: str = "history",
    current_table: str = "current",
    shard_column: str = CHECKPOINT_SHARD_COLUMN,
    chunk_id_column: str = _CHUNK_ID_COLUMN,
) -> None:
    history_frames = [
        _routed(
            frame,
            chunk_id=chunk_id,
            shard_column=shard_column,
            chunk_id_column=chunk_id_column,
        )
        for chunk_id, frame in history
    ]
    current_frames = [
        _routed(
            frame,
            chunk_id=chunk_id,
            shard_column=shard_column,
            chunk_id_column=chunk_id_column,
        )
        for chunk_id, frame in current
    ]
    history_rows = (
        pl.concat(history_frames, how="diagonal_relaxed")
        if history_frames
        else _routed(
            empty_history,
            chunk_id="",
            shard_column=shard_column,
            chunk_id_column=chunk_id_column,
        )
    )
    current_rows = pl.concat(current_frames, how="diagonal_relaxed")
    quoted_history = _quote_identifier(history_table)
    quoted_current = _quote_identifier(current_table)
    connection.register("rolling_history_input", history_rows)
    connection.register("rolling_current_input", current_rows)
    try:
        connection.execute(f"CREATE TABLE {quoted_history} AS SELECT * FROM rolling_history_input")
        connection.execute(
            f"CREATE TEMP TABLE {quoted_current} AS SELECT * FROM rolling_current_input"
        )
    finally:
        connection.unregister("rolling_history_input")
        connection.unregister("rolling_current_input")


def _old_checkpoint_result(
    processor: FrequencyResponseProcessor,
    current: pl.DataFrame,
    history: Sequence[pl.DataFrame],
    *,
    chunk_id: str = "2024-01-08",
) -> pl.DataFrame:
    current_shard = current.filter(pl.col("CustomerID") != "other-shard")
    history_shard = [
        frame.filter(pl.col("CustomerID") != "other-shard").lazy() for frame in history
    ]
    return (
        processor.checkpoint_aggregate_lazy(
            current_shard.lazy(),
            history_shard,
            _ctx(chunk_id),
        )
        .collect()
        .sort(["Day", "Placement", _SEGMENT_COLUMN, "ExposureBucket"], nulls_last=False)
    )


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


@pytest.mark.unit
def test_rolling_frequency_sql_matches_polars_for_exact_window_and_runner_semantics() -> None:
    config = _config()
    processor = FrequencyResponseProcessor(config, computation_hash="hash")
    target_chunk = "2024-01-08"
    target_time = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    oldest_rows = [
        # Both exact-boundary contacts are outside strict (t-W, t].
        _row("strict", "boundary-a", target_time - dt.timedelta(hours=168)),
        _row("strict", "boundary-b", target_time - dt.timedelta(hours=168)),
        _row(
            "strict",
            "inside",
            target_time - dt.timedelta(hours=168) + dt.timedelta(seconds=1),
        ),
        # A later DecisionTime in history must not enter an earlier target prefix.
        _row("late", "future", target_time + dt.timedelta(hours=2)),
        _row("nullable", "old", target_time - dt.timedelta(hours=1), segment=None),
        _row("other-shard", "old", target_time - dt.timedelta(hours=1)),
    ]
    newest_rows = [
        # Cross-chunk overlap must mark this contact as historical.
        _row("overlap", "same", target_time - dt.timedelta(minutes=5)),
    ]
    current_rows = [
        _row("strict", "target", target_time, outcome="Clicked"),
        # No exact rank 2 in this group: choose the smallest available rank > 2.
        _row(
            "strict",
            "target",
            target_time,
            action="fallback",
            rank=3,
            outcome="Pending",
            propensity=0.25,
            priority=0.4,
        ),
        _row("late", "target", target_time),
        _row(
            "late",
            "target",
            target_time,
            action="runner",
            rank=2,
            outcome="Pending",
            propensity=0.4,
            priority=0.5,
        ),
        _row("nullable", "target", target_time, segment=None),
        _row(
            "nullable",
            "target",
            target_time,
            action="runner",
            rank=2,
            outcome="Pending",
            propensity=0.3,
            priority=0.6,
            segment=None,
        ),
        _row("placement", "target", target_time, segment="B"),
        # Exact rank 2 is in another placement and cannot win this configured group.
        _row(
            "placement",
            "target",
            target_time,
            action="other-placement",
            placement="Other",
            rank=2,
            outcome="Pending",
            propensity=0.1,
            segment="B",
        ),
        _row(
            "placement",
            "target",
            target_time,
            action="same-placement",
            rank=3,
            outcome="Pending",
            propensity=0.7,
            priority=0.8,
            segment="B",
        ),
        # Current values win, but the historical overlap suppresses focal output.
        _row("overlap", "same", target_time, outcome="Clicked", propensity=0.9),
        _row("other-shard", "target", target_time, outcome="Clicked"),
    ]
    oldest = _prepared_history(processor, oldest_rows)
    newest = _prepared_history(processor, newest_rows)
    current = _prepared_current(processor, current_rows)
    expected = _old_checkpoint_result(processor, current, [oldest, newest])

    # The physical insertion order is deliberately newest-first. SQL must
    # reconstruct deterministic chunk/local order rather than trust table order.
    history_table = 'hist"ory'
    current_table = 'cur"rent'
    shard_column = 'shard"id'
    chunk_column = 'chunk"id'
    with duckdb.connect(":memory:") as connection:
        _create_rolling_tables(
            connection,
            history=[
                ("2024-01-07", newest),
                ("2024-01-01", oldest),
                # Rows at or after the target chunk are never historical input.
                (target_chunk, oldest),
                ("2024-01-09", oldest),
            ],
            current=[(target_chunk, current)],
            empty_history=oldest.head(0),
            history_table=history_table,
            current_table=current_table,
            shard_column=shard_column,
            chunk_id_column=chunk_column,
        )
        with DuckDBFrequencySession(
            config=config,
            connection=connection,
            current_chunk_id=target_chunk,
            history_table=history_table,
            current_table=current_table,
            shard_column=shard_column,
            chunk_id_column=chunk_column,
            batch_size=2,
        ) as session:
            focal = session.focal_lazy(0)
            assert focal.collect_schema()["DecisionTime"] == pl.Datetime("ns")
            actual = (
                processor.aggregate_focal_lazy(focal, _ctx())
                .collect()
                .sort(
                    ["Day", "Placement", _SEGMENT_COLUMN, "ExposureBucket"],
                    nulls_last=False,
                )
            )

        # The SQL session borrows rather than closes the rolling-store connection.
        assert connection.execute("SELECT 1").fetchone() == (1,)

    assert_frame_equal(actual, expected)
    totals = actual.select(
        pl.col("Contacts").sum(),
        pl.col("Clicks").sum(),
        pl.col("ComparableContacts").sum(),
        pl.col("RunnerAvailableContacts").sum(),
        pl.col("RunnerPropensitySum").sum(),
    ).row(0)
    assert totals[:4] == (4, 1, 4, 4)
    assert totals[4] == pytest.approx(1.65)
    strict = actual.filter((pl.col(_SEGMENT_COLUMN) == "A") & (pl.col("ExposureBucket") == 2))
    assert strict.select(pl.col("Contacts").sum()).item() == 1


@pytest.mark.unit
def test_rolling_frequency_sql_preserves_aware_nanoseconds_at_strict_boundary() -> None:
    config = _config()
    processor = FrequencyResponseProcessor(config, computation_hash="hash")
    target_chunk = "2024-01-08"
    target_ns = int(dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC).timestamp()) * 1_000_000_000
    window_ns = 168 * 60 * 60 * 1_000_000_000

    history_rows = [
        _row("customer", "boundary", dt.datetime(2024, 1, 1, tzinfo=dt.UTC)),
        _row("customer", "inside", dt.datetime(2024, 1, 1, tzinfo=dt.UTC)),
    ]
    history_source = pl.DataFrame(history_rows).with_columns(
        pl.Series(
            "DecisionTime",
            [target_ns - window_ns, target_ns - window_ns + 100],
            dtype=pl.Datetime("ns", "UTC"),
        )
    )
    current_source = pl.DataFrame(
        [_row("customer", "target", dt.datetime(2024, 1, 8, tzinfo=dt.UTC))]
    ).with_columns(
        pl.Series(
            "DecisionTime",
            [target_ns],
            dtype=pl.Datetime("ns", "UTC"),
        )
    )
    prepared_history_target = processor.checkpoint_contacts_lazy(history_source.lazy()).collect()
    history = processor.checkpoint_history_contacts_lazy(prepared_history_target.lazy()).collect()
    current = processor.checkpoint_contacts_lazy(current_source.lazy()).collect()
    assert current.schema["DecisionTime"] == pl.Datetime("ns")
    assert current.get_column("DecisionTime").cast(pl.Int64).item() == target_ns
    expected = _old_checkpoint_result(processor, current, [history])

    with duckdb.connect(":memory:") as connection:
        _create_rolling_tables(
            connection,
            history=[("2024-01-01", history)],
            current=[(target_chunk, current)],
            empty_history=history.head(0),
        )
        with DuckDBFrequencySession(
            config=config,
            connection=connection,
            current_chunk_id=target_chunk,
        ) as session:
            focal = session.focal_lazy(0)
            assert focal.collect_schema()["DecisionTime"] == pl.Datetime("ns")
            actual = processor.aggregate_focal_lazy(focal, _ctx()).collect()

    assert_frame_equal(actual, expected, check_row_order=False)
    assert actual.select(pl.col("ExposureBucket").max()).item() == 2


@pytest.mark.unit
def test_rolling_frequency_sql_matches_source_storage_types_for_dictionary_dimensions() -> None:
    config = _config()
    processor = FrequencyResponseProcessor(config, computation_hash="hash")
    target_chunk = "2024-01-08"
    target_time = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    source = pl.DataFrame(
        [
            _row("customer", "target", target_time, outcome="Clicked"),
            _row(
                "customer",
                "target",
                target_time,
                action="runner",
                rank=2,
                outcome="Pending",
            ),
        ]
    ).with_columns(
        pl.col("CustomerID").cast(pl.Categorical),
        pl.col("Placement").cast(pl.Enum(["Hero"])),
        pl.col(_SEGMENT_COLUMN).cast(pl.Categorical),
    )
    current = processor.checkpoint_contacts_lazy(source.lazy()).collect()
    empty_history = processor.checkpoint_history_contacts_lazy(current.lazy()).collect().head(0)
    expected = _old_checkpoint_result(processor, current, [], chunk_id=target_chunk)

    with duckdb.connect(":memory:") as connection:
        _create_rolling_tables(
            connection,
            history=[],
            current=[(target_chunk, current)],
            empty_history=empty_history,
        )
        with DuckDBFrequencySession(
            config=config,
            connection=connection,
            current_chunk_id=target_chunk,
        ) as session:
            actual = (
                processor.aggregate_focal_lazy(session.focal_lazy(0), _ctx())
                .collect()
                .sort(["Day", "Placement", _SEGMENT_COLUMN, "ExposureBucket"])
            )

    assert actual.schema["Placement"] == pl.String
    assert actual.schema[_SEGMENT_COLUMN] == pl.String
    assert_frame_equal(actual, expected)


@pytest.mark.unit
def test_rolling_frequency_session_bootstraps_empty_history_and_filters_staged_chunk() -> None:
    config = _config()
    processor = FrequencyResponseProcessor(config, computation_hash="hash")
    target_chunk = "2024-01-08"
    target_time = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    current = _prepared_current(
        processor,
        [_row("customer", "target", target_time, outcome="Clicked")],
    )
    stray = _prepared_current(
        processor,
        [_row("stray", "other", target_time, outcome="Clicked")],
    )
    empty_history = processor.checkpoint_history_contacts_lazy(current.lazy()).collect().head(0)
    expected = _old_checkpoint_result(processor, current, [], chunk_id=target_chunk)

    with duckdb.connect(":memory:") as connection:
        _create_rolling_tables(
            connection,
            history=[],
            current=[("2024-01-07", stray), (target_chunk, current)],
            empty_history=empty_history,
        )
        session = DuckDBFrequencySession(
            config=config,
            connection=connection,
            current_chunk_id=target_chunk,
        )
        with pytest.raises(RuntimeError, match="must be entered"):
            session.focal_lazy(0)
        with session:
            actual = processor.aggregate_focal_lazy(session.focal_lazy(0), _ctx()).collect()
            with pytest.raises(ValueError, match="non-negative"):
                session.focal_lazy(-1)
        assert connection.execute("SELECT count(*) FROM history").fetchone() == (0,)

    assert_frame_equal(actual, expected, check_row_order=False)


@pytest.mark.unit
def test_processor_exposes_exact_history_projection_contract() -> None:
    processor = FrequencyResponseProcessor(_config(), computation_hash="hash")

    columns, rank_column, exposed_column = processor.checkpoint_history_projection()

    assert rank_column == RANK_COLUMN
    assert exposed_column == EXPOSED_COLUMN
    assert columns[-4:] == (
        "__valuestream_frequency_row_order",
        RANK_COLUMN,
        "__valuestream_frequency_positive",
        EXPOSED_COLUMN,
    )
