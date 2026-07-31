"""Focused semantics for the typed ``frequency_response`` processor."""

from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from valuestream.config import model
from valuestream.config.validate import _processor_source_columns
from valuestream.processors.context import TARGET_CHUNK_COLUMN, ChunkContext
from valuestream.processors.frequency_response import (
    FrequencyResponseProcessor,
    required_history_input_columns,
    required_input_columns,
)


def _config(**overrides: Any) -> model.FrequencyResponseProcessor:
    payload: dict[str, Any] = {
        "id": "frequency",
        "source": "interaction_history",
        "kind": "frequency_response",
        "group_by": ["Day", "ExposureBucket"],
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
        "positive_values": ["Clicked"],
        "exposure_values": ["Impression"],
        "candidate_values": ["Pending"],
        "states": {
            "Contacts": {"type": "count"},
            "Clicks": {"type": "count", "source_column": "ClickedContact"},
            "ComparableContacts": {
                "type": "count",
                "source_column": "ComparableContact",
            },
            "ComparableClicks": {
                "type": "count",
                "source_column": "ComparableClick",
            },
            "RunnerAvailableContacts": {
                "type": "count",
                "source_column": "RunnerAvailable",
            },
            "RunnerPropensitySum": {
                "type": "value_sum",
                "source_column": "RunnerPropensity",
            },
            "PriorityComparableContacts": {
                "type": "count",
                "source_column": "PriorityComparableContact",
            },
            "FocalPrioritySum": {
                "type": "value_sum",
                "source_column": "FocalPriorityComparable",
            },
            "RunnerPrioritySum": {
                "type": "value_sum",
                "source_column": "RunnerPriorityComparable",
            },
        },
    }
    payload.update(overrides)
    return model.FrequencyResponseProcessor.model_validate(payload)


def _ctx(chunk: str = "20240108") -> ChunkContext:
    return ChunkContext(
        pipeline_run_id="run",
        chunk_id=chunk,
        created_at=dt.datetime(2024, 1, 8, 13, tzinfo=dt.UTC),
    )


def _row(
    *,
    customer: str,
    interaction: str,
    decision_time: dt.datetime,
    target: bool,
    action: str = "A",
    placement: str = "Hero",
    rank: int = 1,
    outcome: str = "Impression",
    propensity: float | None = 0.5,
    priority: float | None = 1.0,
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
        "DecisionTime": decision_time,
        TARGET_CHUNK_COLUMN: target,
    }


def _aggregate(
    rows: list[dict[str, Any]], config: model.FrequencyResponseProcessor | None = None
) -> pl.DataFrame:
    processor = FrequencyResponseProcessor(config or _config(), computation_hash="hash")
    return processor.chunk_aggregate(pl.DataFrame(rows).lazy(), _ctx()).sort(
        ["Day", "ExposureBucket"]
    )


@pytest.mark.unit
def test_model_requires_daily_bucketed_mergeable_states() -> None:
    config = _config()

    assert config.partition_lag_hours == 0
    assert config.checkpoint.mode == "source_scan"
    assert config.checkpoint.shards == 64
    assert config.checkpoint.retention_days is None
    assert config.checkpoint_retention_days == 8
    assert set(config.positive_values).isdisjoint(config.exposure_values)
    assert set(config.candidate_values).isdisjoint(
        [*config.positive_values, *config.exposure_values]
    )
    with pytest.raises(ValidationError, match=r"time\.grain 'daily'"):
        _config(time={"property": "DecisionTime", "grain": "hourly"})
    with pytest.raises(ValidationError, match="frequency_column must be present in group_by"):
        _config(group_by=["Day"])
    with pytest.raises(ValidationError, match="must use count or value_sum"):
        _config(states={"Minimum": {"type": "min", "source_column": "Propensity"}})
    with pytest.raises(ValidationError, match="reserved derived column"):
        _config(frequency_column="Day")
    with pytest.raises(ValidationError, match="cannot be used in group_by"):
        _config(group_by=["Day", "ExposureBucket", "ClickedContact"])
    with pytest.raises(ValidationError, match="cannot be used in group_by"):
        _config(group_by=["Day", "ExposureBucket", "__valuestream_private"])
    with pytest.raises(ValidationError, match="state names collide"):
        _config(states={"ExposureBucket": {"type": "count"}})
    with pytest.raises(ValidationError, match="partition_lag_hours"):
        _config(partition_lag_hours=-1)
    with pytest.raises(ValidationError, match="shards"):
        _config(checkpoint={"mode": "persistent_sharded", "shards": 0})
    with pytest.raises(ValidationError, match=r"retention_days must be at least 8"):
        _config(
            checkpoint={
                "mode": "persistent_sharded",
                "shards": 8,
                "retention_days": 7,
            }
        )
    columns = config.columns.model_dump()
    columns["rank"] = "__valuestream_frequency_rank"
    with pytest.raises(
        ValidationError,
        match=r"raw input bindings.*__valuestream_frequency_rank",
    ):
        _config(columns=columns)
    with pytest.raises(ValidationError, match="cannot read internal columns"):
        _config(
            states={
                "Private": {
                    "type": "value_sum",
                    "source_column": "__valuestream_runner_propensity",
                }
            }
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("filter_expression", "derived_column"),
    [
        ({"op": "eq", "column": "ExposureBucket", "value": 1}, "ExposureBucket"),
        ({"op": "not_null", "column": "ClickedContact"}, "ClickedContact"),
        (
            {"polars": 'pl.col("RunnerPropensity").is_not_null()'},
            "RunnerPropensity",
        ),
    ],
)
def test_processor_filter_cannot_reference_post_filter_derived_columns(
    filter_expression: dict[str, Any],
    derived_column: str,
) -> None:
    with pytest.raises(ValidationError, match=rf"before derived columns.*{derived_column}"):
        _config(filter=filter_expression)


@pytest.mark.unit
def test_required_inputs_exclude_derived_bucket_and_virtual_state_fields() -> None:
    config = _config(
        group_by=["Day", "Channel", "ExposureBucket"],
        filter={"op": "eq", "column": "Market", "value": "DE"},
    )
    required = required_input_columns(config)
    history_required = required_history_input_columns(config)
    catalog_source_columns = _processor_source_columns(config)

    assert {
        "DecisionTime",
        "Channel",
        "Market",
        "Priority",
        TARGET_CHUNK_COLUMN,
    } <= required
    assert "Day" not in required
    assert "ExposureBucket" not in required
    assert not model.FREQUENCY_RESPONSE_VIRTUAL_COLUMNS.intersection(required)
    assert {"DecisionTime", "Channel", "Priority"} <= catalog_source_columns
    assert "ExposureBucket" not in catalog_source_columns
    assert not model.FREQUENCY_RESPONSE_VIRTUAL_COLUMNS.intersection(catalog_source_columns)
    assert history_required == {
        "CustomerID",
        "InteractionID",
        "ActionID",
        "Placement",
        "Rank",
        "Outcome",
        "DecisionTime",
        "Market",
    }


@pytest.mark.unit
def test_runtime_rejects_existing_bucket_and_non_numeric_bindings() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    valid = _row(customer="c1", interaction="i1", decision_time=when, target=True)

    with pytest.raises(ValueError, match=r"ExposureBucket.*already exists"):
        _aggregate([{**valid, "ExposureBucket": 99}])

    with pytest.raises(TypeError, match=r"integer rank.*Float64"):
        _aggregate([{**valid, "Rank": 1.5}])

    with pytest.raises(TypeError, match=r"numeric propensity.*String"):
        _aggregate([{**valid, "Propensity": "malformed"}])

    with pytest.raises(TypeError, match=r"numeric priority.*String"):
        _aggregate([{**valid, "Priority": "malformed"}])


@pytest.mark.unit
def test_strict_168_hour_boundary_is_excluded_and_frequency_is_capped_at_seven() -> None:
    target_time = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = [
        _row(
            customer="boundary",
            interaction="boundary-old",
            decision_time=target_time - dt.timedelta(hours=168),
            target=False,
        ),
        _row(
            customer="boundary",
            interaction="boundary-target",
            decision_time=target_time,
            target=True,
        ),
        _row(
            customer="capped",
            interaction="capped-boundary",
            decision_time=target_time - dt.timedelta(hours=168),
            target=False,
        ),
        *[
            _row(
                customer="capped",
                interaction=f"capped-{index}",
                decision_time=target_time - dt.timedelta(hours=hours),
                target=False,
            )
            for index, hours in enumerate((167, 150, 120, 90, 60, 30, 1), start=1)
        ],
        _row(
            customer="capped",
            interaction="capped-target",
            decision_time=target_time,
            target=True,
        ),
    ]

    out = _aggregate(rows)

    assert out.select("ExposureBucket", "Contacts").rows() == [(1, 1), (7, 1)]


@pytest.mark.unit
def test_decision_day_uses_the_configured_calendar_timezone_and_compact_keeps_day_only() -> None:
    config = _config(
        time={
            "property": "DecisionTime",
            "grain": "daily",
            "calendar": {"timezone": "America/New_York"},
        }
    )
    processor = FrequencyResponseProcessor(config, computation_hash="hash")
    out = processor.chunk_aggregate(
        pl.DataFrame(
            [
                _row(
                    customer="c1",
                    interaction="i1",
                    decision_time=dt.datetime(2024, 1, 8, 0, 30, tzinfo=dt.UTC),
                    target=True,
                )
            ]
        ).lazy(),
        _ctx(),
    )

    compacted = processor.compact(out, "daily", _ctx())

    assert out["Day"].to_list() == [dt.date(2024, 1, 7)]
    assert "Month" not in compacted.columns
    assert compacted["Day"].to_list() == [dt.date(2024, 1, 7)]


@pytest.mark.unit
def test_clicked_wins_contact_normalization_and_history_overlap_is_not_targeted() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = [
        _row(customer="c1", interaction="new", decision_time=when, target=True),
        _row(
            customer="c1",
            interaction="new",
            decision_time=when,
            target=True,
            outcome="Clicked",
        ),
        _row(customer="c2", interaction="overlap", decision_time=when, target=False),
        _row(
            customer="c2",
            interaction="overlap",
            decision_time=when,
            target=True,
            outcome="Clicked",
        ),
    ]

    out = _aggregate(rows)

    assert out.select("Contacts", "Clicks").row(0) == (1, 1)


@pytest.mark.unit
def test_checkpoint_candidates_match_source_scan_for_cross_day_duplicates_and_late_time() -> None:
    processor = FrequencyResponseProcessor(
        _config(checkpoint={"mode": "persistent_sharded", "shards": 8}),
        computation_hash="hash",
    )
    target_time = dt.datetime(2024, 1, 8, 8, tzinfo=dt.UTC)
    history_rows = [
        _row(
            customer="c1",
            interaction="old",
            decision_time=target_time - dt.timedelta(hours=1),
            target=False,
        ),
        # A lagged partition can contain a later DecisionTime. It must not enter
        # the prefix count for the chronologically earlier target contact.
        _row(
            customer="c1",
            interaction="future",
            decision_time=target_time + dt.timedelta(hours=2),
            target=False,
        ),
        _row(
            customer="c2",
            interaction="overlap",
            decision_time=target_time - dt.timedelta(minutes=5),
            target=False,
            propensity=None,
        ),
    ]
    current_rows = [
        _row(
            customer="c1",
            interaction="target",
            decision_time=target_time,
            target=True,
            outcome="Clicked",
        ),
        _row(
            customer="c1",
            interaction="target",
            decision_time=target_time,
            target=True,
            action="runner",
            rank=3,
            outcome="Pending",
            propensity=0.25,
            priority=0.5,
        ),
        _row(
            customer="c2",
            interaction="overlap",
            decision_time=target_time,
            target=True,
            outcome="Clicked",
            propensity=0.7,
        ),
        {
            **_row(
                customer="discarded",
                interaction="null-customer",
                decision_time=target_time,
                target=True,
            ),
            "CustomerID": None,
        },
    ]
    expected = processor.chunk_aggregate(
        pl.DataFrame([*history_rows, *current_rows]).lazy(),
        _ctx(),
    ).sort(["Day", "ExposureBucket"])
    history_checkpoint = processor.checkpoint_contacts_lazy(
        pl.DataFrame(history_rows).drop(TARGET_CHUNK_COLUMN).lazy()
    )
    current_checkpoint = processor.checkpoint_contacts_lazy(
        pl.DataFrame(current_rows).drop(TARGET_CHUNK_COLUMN).lazy()
    )

    actual = (
        processor.checkpoint_aggregate_lazy(
            current_checkpoint,
            [history_checkpoint],
            _ctx(),
        )
        .collect()
        .sort(["Day", "ExposureBucket"])
    )

    assert actual.equals(expected)


@pytest.mark.unit
def test_runner_is_rank_two_then_minimum_above_one_across_the_interaction() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = [
        _row(customer="c1", interaction="exact", decision_time=when, target=True),
        _row(
            customer="c1",
            interaction="exact",
            decision_time=when,
            target=True,
            action="B",
            placement="DifferentPlacement",
            rank=2,
            outcome="Pending",
            propensity=0.2,
            priority=0.4,
        ),
        _row(
            customer="c1",
            interaction="exact",
            decision_time=when,
            target=True,
            action="C",
            rank=3,
            outcome="Pending",
            propensity=0.9,
            priority=0.9,
        ),
        _row(customer="c2", interaction="fallback", decision_time=when, target=True),
        _row(
            customer="c2",
            interaction="fallback",
            decision_time=when,
            target=True,
            action="B",
            rank=3,
            outcome="Pending",
            propensity=0.3,
            priority=0.6,
        ),
        _row(
            customer="c2",
            interaction="fallback",
            decision_time=when,
            target=True,
            action="C",
            rank=4,
            outcome="Pending",
            propensity=0.4,
            priority=0.8,
        ),
        _row(customer="c3", interaction="missing", decision_time=when, target=True),
    ]

    row = (
        _aggregate(rows)
        .select(
            pl.col("Contacts").sum(),
            pl.col("RunnerAvailableContacts").sum(),
            pl.col("ComparableContacts").sum(),
            pl.col("RunnerPropensitySum").sum(),
            pl.col("RunnerPrioritySum").sum(),
        )
        .row(0)
    )

    assert row[:3] == (3, 2, 2)
    assert row[3] == pytest.approx(0.5)
    assert row[4] == pytest.approx(1.0)


@pytest.mark.unit
def test_frequency_keys_isolate_customer_action_and_placement() -> None:
    old = dt.datetime(2024, 1, 8, 10, tzinfo=dt.UTC)
    current = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = [
        _row(customer="c1", interaction="old", decision_time=old, target=False),
        _row(customer="c1", interaction="same", decision_time=current, target=True),
        _row(
            customer="c1",
            interaction="action",
            decision_time=current,
            target=True,
            action="OtherAction",
        ),
        _row(
            customer="c1",
            interaction="placement",
            decision_time=current,
            target=True,
            placement="OtherPlacement",
        ),
        _row(customer="c2", interaction="customer", decision_time=current, target=True),
    ]

    out = _aggregate(rows)

    assert out.select("ExposureBucket", "Contacts").rows() == [(1, 3), (2, 1)]


@pytest.mark.unit
def test_comparable_clicks_and_propensity_share_the_same_eligible_population() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows: list[dict[str, Any]] = []
    for customer, interaction, outcome, runner_propensity in (
        ("c1", "clicked-comparable", "Clicked", 0.2),
        ("c2", "impression-comparable", "Impression", 0.3),
        ("c3", "clicked-null", "Clicked", None),
    ):
        rows.extend(
            [
                _row(
                    customer=customer,
                    interaction=interaction,
                    decision_time=when,
                    target=True,
                    outcome=outcome,
                ),
                _row(
                    customer=customer,
                    interaction=interaction,
                    decision_time=when,
                    target=True,
                    action="Runner",
                    rank=2,
                    outcome="Pending",
                    propensity=runner_propensity,
                ),
            ]
        )
    rows.append(_row(customer="c4", interaction="no-runner", decision_time=when, target=True))

    row = (
        _aggregate(rows)
        .select(
            pl.col("Contacts").sum(),
            pl.col("Clicks").sum(),
            pl.col("RunnerAvailableContacts").sum(),
            pl.col("ComparableContacts").sum(),
            pl.col("ComparableClicks").sum(),
            pl.col("RunnerPropensitySum").sum(),
        )
        .row(0)
    )

    assert row[:5] == (4, 2, 3, 2, 1)
    assert row[5] == pytest.approx(0.5)


@pytest.mark.unit
def test_chunk_partials_merge_to_the_combined_result() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    first = [
        _row(
            customer="c1",
            interaction="i1",
            decision_time=when,
            target=True,
            outcome="Clicked",
        ),
        _row(
            customer="c1",
            interaction="i1",
            decision_time=when,
            target=True,
            action="Runner",
            rank=2,
            outcome="Pending",
            propensity=0.2,
        ),
    ]
    second = [
        _row(customer="c2", interaction="i2", decision_time=when, target=True),
        _row(
            customer="c2",
            interaction="i2",
            decision_time=when,
            target=True,
            action="Runner",
            rank=2,
            outcome="Pending",
            propensity=0.3,
        ),
    ]
    processor = FrequencyResponseProcessor(_config(), computation_hash="hash")
    partials = [
        processor.chunk_aggregate(pl.DataFrame(rows).lazy(), _ctx(chunk))
        for rows, chunk in ((first, "first"), (second, "second"))
    ]
    merged = processor.merge(
        pl.concat(partials).drop("pipeline_run_id", "chunk_id", "created_at", "config_hash"),
        group_columns=["Day", "ExposureBucket", "period"],
    ).sort(["Day", "ExposureBucket"])
    combined = (
        processor.chunk_aggregate(pl.DataFrame([*first, *second]).lazy(), _ctx())
        .drop("pipeline_run_id", "chunk_id", "created_at", "config_hash")
        .sort(["Day", "ExposureBucket"])
    )

    assert merged.select(combined.columns).equals(combined)
