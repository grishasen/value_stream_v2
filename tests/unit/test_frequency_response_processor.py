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
        "alternative_group_by": ["Placement"],
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
    comparison_group: str | None = "default",
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
        "ComparisonGroup": comparison_group,
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
    assert config.checkpoint_retention_days == 7
    assert config.alternative_group_by == ["Placement"]
    assert config.alternative_group_columns == ["CustomerID", "InteractionID", "Placement"]
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
    payload = config.model_dump(mode="python")
    payload.pop("alternative_group_by")
    with pytest.raises(ValidationError, match=r"(?s)alternative_group_by.*Field required"):
        model.FrequencyResponseProcessor.model_validate(payload)
    interaction_wide = _config(alternative_group_by=[])
    assert interaction_wide.alternative_group_columns == ["CustomerID", "InteractionID"]
    multi_field = _config(alternative_group_by=["Placement", "ComparisonGroup"])
    assert multi_field.alternative_group_columns == [
        "CustomerID",
        "InteractionID",
        "Placement",
        "ComparisonGroup",
    ]
    with pytest.raises(ValidationError, match="columns must be unique"):
        _config(alternative_group_by=["Placement", "Placement"])
    for mandatory_column in ("CustomerID", "InteractionID"):
        with pytest.raises(ValidationError, match="must not repeat mandatory"):
            _config(alternative_group_by=[mandatory_column])
    for invalid_column in ("", " Day", "Day", "ExposureBucket", "ClickedContact", "config_hash"):
        with pytest.raises(ValidationError, match="must contain raw source columns"):
            _config(alternative_group_by=[invalid_column])
    duplicate_columns = config.columns.model_dump()
    duplicate_columns["interaction"] = duplicate_columns["customer"]
    with pytest.raises(ValidationError, match="customer and interaction bindings must be distinct"):
        _config(columns=duplicate_columns)
    with pytest.raises(ValidationError, match="shards"):
        _config(checkpoint={"mode": "persistent_sharded", "shards": 0})
    with pytest.raises(ValidationError, match=r"retention_days must be at least 7"):
        _config(
            checkpoint={
                "mode": "persistent_sharded",
                "shards": 8,
                "retention_days": 6,
            }
        )
    exact_minimum = _config(
        checkpoint={
            "mode": "persistent_sharded",
            "shards": 8,
            "retention_days": 7,
        }
    )
    assert exact_minimum.checkpoint_retention_days == 7
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
    ("window_hours", "partition_lag_hours", "expected_days"),
    [
        (1, 0, 1),
        (24, 0, 1),
        (25, 0, 2),
        (167, 0, 7),
        (168, 0, 7),
        (169, 0, 8),
        (168, 24, 8),
        (168, 25, 9),
    ],
)
def test_checkpoint_retention_is_exact_source_day_closure(
    window_hours: int,
    partition_lag_hours: int,
    expected_days: int,
) -> None:
    defaulted = _config(
        window_hours=window_hours,
        partition_lag_hours=partition_lag_hours,
        checkpoint={"mode": "persistent_sharded"},
    )
    exact = _config(
        window_hours=window_hours,
        partition_lag_hours=partition_lag_hours,
        checkpoint={
            "mode": "persistent_sharded",
            "retention_days": expected_days,
        },
    )

    assert defaulted.checkpoint_retention_days == expected_days
    assert exact.checkpoint_retention_days == expected_days
    if expected_days > 1:
        with pytest.raises(
            ValidationError,
            match=rf"retention_days must be at least {expected_days}",
        ):
            _config(
                window_hours=window_hours,
                partition_lag_hours=partition_lag_hours,
                checkpoint={
                    "mode": "persistent_sharded",
                    "retention_days": expected_days - 1,
                },
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
        alternative_group_by=["Placement", "ComparisonSegment"],
        filter={"op": "eq", "column": "Market", "value": "DE"},
    )
    required = required_input_columns(config)
    history_required = required_history_input_columns(config)
    catalog_source_columns = _processor_source_columns(config)

    assert {
        "DecisionTime",
        "Channel",
        "ComparisonSegment",
        "Market",
        "Priority",
        TARGET_CHUNK_COLUMN,
    } <= required
    assert "Day" not in required
    assert "ExposureBucket" not in required
    assert not model.FREQUENCY_RESPONSE_VIRTUAL_COLUMNS.intersection(required)
    assert {"DecisionTime", "Channel", "ComparisonSegment", "Priority"} <= (catalog_source_columns)
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
def test_raw_decision_time_grouping_is_canonical_and_matches_checkpoint_mode() -> None:
    config = _config(group_by=["Day", "DecisionTime", "ExposureBucket"])
    processor = FrequencyResponseProcessor(config, computation_hash="hash")
    when = dt.datetime(2024, 1, 8, 0, 30, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    rows = [
        _row(customer="c1", interaction="i1", decision_time=when, target=True),
        _row(
            customer="c1",
            interaction="i1",
            decision_time=when,
            target=True,
            action="B",
            rank=2,
            outcome="Pending",
        ),
    ]
    source = pl.DataFrame(rows)

    expected = processor.chunk_aggregate(source.lazy(), _ctx()).sort(
        ["Day", "DecisionTime", "ExposureBucket"]
    )
    current = processor.checkpoint_contacts_lazy(source.drop(TARGET_CHUNK_COLUMN).lazy())
    actual = (
        processor.checkpoint_aggregate_lazy(current, [], _ctx())
        .collect()
        .sort(["Day", "DecisionTime", "ExposureBucket"])
    )

    assert expected.schema["DecisionTime"] == pl.Datetime("us")
    assert actual.equals(expected)


@pytest.mark.unit
def test_dictionary_dimensions_match_duckdb_varchar_semantics() -> None:
    config = _config(group_by=["Day", "Placement", "ExposureBucket"])
    processor = FrequencyResponseProcessor(config, computation_hash="hash")
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    source = pl.DataFrame(
        [
            _row(customer="c1", interaction="i1", decision_time=when, target=True),
            _row(
                customer="c1",
                interaction="i1",
                decision_time=when,
                target=True,
                action="B",
                rank=2,
                outcome="Pending",
            ),
        ]
    ).with_columns(
        pl.col("CustomerID").cast(pl.Categorical),
        pl.col("Placement").cast(pl.Enum(["Hero"])),
    )

    expected = processor.chunk_aggregate(source.lazy(), _ctx()).sort(
        ["Day", "Placement", "ExposureBucket"]
    )
    current = processor.checkpoint_contacts_lazy(source.drop(TARGET_CHUNK_COLUMN).lazy())
    actual = (
        processor.checkpoint_aggregate_lazy(current, [], _ctx())
        .collect()
        .sort(["Day", "Placement", "ExposureBucket"])
    )

    assert expected.schema["Placement"] == pl.String
    assert actual.equals(expected)


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
@pytest.mark.parametrize(
    "alternative_group_by",
    [
        [],
        ["Placement"],
    ],
)
def test_checkpoint_candidates_match_source_scan_for_cross_day_duplicates_and_late_time(
    alternative_group_by: list[str],
) -> None:
    processor = FrequencyResponseProcessor(
        _config(
            alternative_group_by=alternative_group_by,
            checkpoint={"mode": "persistent_sharded", "shards": 8},
        ),
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
            action="global-runner",
            placement="DifferentPlacement",
            rank=2,
            outcome="Pending",
            propensity=0.15,
            priority=0.4,
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
    history_target_checkpoint = processor.checkpoint_contacts_lazy(
        pl.DataFrame(history_rows).drop(TARGET_CHUNK_COLUMN).lazy()
    )
    history_checkpoint = processor.checkpoint_history_contacts_lazy(history_target_checkpoint)
    current_checkpoint = processor.checkpoint_contacts_lazy(
        pl.DataFrame(current_rows).drop(TARGET_CHUNK_COLUMN).lazy()
    )

    history_schema = set(history_checkpoint.collect_schema().names())
    assert history_checkpoint.collect().height == 3
    assert {
        "Propensity",
        "Priority",
        "Outcome",
        "Rank",
        TARGET_CHUNK_COLUMN,
    }.isdisjoint(history_schema)

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
@pytest.mark.parametrize(
    ("alternative_group_by", "expected_propensity", "expected_priority"),
    [
        ([], 0.5, 1.0),
        (["Placement"], 1.2, 1.5),
    ],
)
def test_selected_rank_two_uses_configured_group_and_fallback(
    alternative_group_by: list[str],
    expected_propensity: float,
    expected_priority: float,
) -> None:
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
        _aggregate(rows, _config(alternative_group_by=alternative_group_by))
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
    assert row[3] == pytest.approx(expected_propensity)
    assert row[4] == pytest.approx(expected_priority)


@pytest.mark.unit
@pytest.mark.parametrize(
    "alternative_group_by",
    [
        [],
        ["Placement"],
    ],
)
def test_alternative_group_never_crosses_interactions(
    alternative_group_by: list[str],
) -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = [
        _row(customer="c1", interaction="focal", decision_time=when, target=True),
        _row(
            customer="c1",
            interaction="runner",
            decision_time=when,
            target=True,
            action="B",
            rank=2,
            outcome="Pending",
            propensity=0.4,
        ),
    ]

    result = _aggregate(rows, _config(alternative_group_by=alternative_group_by)).select(
        pl.col("ComparableContacts").sum(),
        pl.col("RunnerPropensitySum").sum(),
    )

    assert result.row(0) == (0, 0.0)


@pytest.mark.unit
def test_multiple_alternative_group_fields_constrain_runner_selection() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = [
        _row(
            customer="c1",
            interaction="decision",
            decision_time=when,
            target=True,
            comparison_group="A",
        ),
        _row(
            customer="c1",
            interaction="decision",
            decision_time=when,
            target=True,
            action="B",
            rank=2,
            outcome="Pending",
            propensity=0.2,
            comparison_group="B",
        ),
        _row(
            customer="c1",
            interaction="decision",
            decision_time=when,
            target=True,
            action="C",
            rank=3,
            outcome="Pending",
            propensity=0.7,
            comparison_group="A",
        ),
    ]

    result = _aggregate(
        rows,
        _config(alternative_group_by=["Placement", "ComparisonGroup"]),
    ).select(
        pl.col("ComparableContacts").sum(),
        pl.col("RunnerPropensitySum").sum(),
    )

    assert result.row(0) == (1, 0.7)


@pytest.mark.unit
def test_null_alternative_group_values_compare_as_one_group() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows: list[dict[str, Any]] = []
    for customer, comparison_group in (("c1", None), ("c2", "known")):
        rows.extend(
            [
                _row(
                    customer=customer,
                    interaction=f"{customer}-decision",
                    decision_time=when,
                    target=True,
                    comparison_group=comparison_group,
                ),
                _row(
                    customer=customer,
                    interaction=f"{customer}-decision",
                    decision_time=when,
                    target=True,
                    action="B",
                    rank=2,
                    outcome="Pending",
                    propensity=0.3,
                    comparison_group=comparison_group,
                ),
            ]
        )

    result = _aggregate(
        rows,
        _config(alternative_group_by=["Placement", "ComparisonGroup"]),
    ).select(pl.col("ComparableContacts").sum())

    assert result.item() == 2


@pytest.mark.unit
def test_missing_alternative_group_input_fails_before_processing() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    row = _row(customer="c1", interaction="i1", decision_time=when, target=True)
    row.pop("ComparisonGroup")

    with pytest.raises(ValueError, match=r"missing input column\(s\): ComparisonGroup"):
        _aggregate([row], _config(alternative_group_by=["ComparisonGroup"]))


@pytest.mark.unit
def test_placement_scoped_alternative_does_not_cross_placements() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = [
        _row(customer="c1", interaction="decision", decision_time=when, target=True),
        _row(
            customer="c1",
            interaction="decision",
            decision_time=when,
            target=True,
            action="B",
            placement="DifferentPlacement",
            rank=2,
            outcome="Pending",
            propensity=0.2,
        ),
    ]

    result = _aggregate(rows).select(
        pl.col("Contacts").sum(),
        pl.col("RunnerAvailableContacts").sum(),
        pl.col("ComparableContacts").sum(),
        pl.col("RunnerPropensitySum").sum(),
    )

    assert result.row(0) == (1, 0, 0, 0.0)


@pytest.mark.unit
def test_alternative_group_never_crosses_customers() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = [
        _row(customer="c1", interaction="shared", decision_time=when, target=True),
        _row(customer="c2", interaction="shared", decision_time=when, target=True),
        _row(
            customer="c2",
            interaction="shared",
            decision_time=when,
            target=True,
            action="B",
            rank=2,
            outcome="Pending",
            propensity=0.4,
        ),
    ]

    result = _aggregate(rows).select(
        pl.col("Contacts").sum(),
        pl.col("RunnerAvailableContacts").sum(),
        pl.col("ComparableContacts").sum(),
        pl.col("RunnerPropensitySum").sum(),
    )

    assert result.row(0) == (2, 1, 1, 0.4)


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


@pytest.mark.unit
def test_daily_granularity_validates_whole_days_and_zero_lag() -> None:
    with pytest.raises(ValidationError, match="divisible by 24"):
        _config(window_granularity="daily", window_hours=100)
    with pytest.raises(ValidationError, match="partition_lag_hours 0"):
        _config(window_granularity="daily", partition_lag_hours=24)
    config = _config(window_granularity="daily")
    assert config.window_days == 7


@pytest.mark.unit
def test_customer_sample_fraction_must_be_representable() -> None:
    with pytest.raises(ValidationError, match="representable"):
        _config(customer_sample={"fraction": 1e-9})
    with pytest.raises(ValidationError):
        _config(customer_sample={"fraction": 0.0})
    with pytest.raises(ValidationError):
        _config(customer_sample={"fraction": 1.5})
    config = _config(customer_sample={"fraction": 0.25})
    assert config.customer_sample is not None
    assert config.customer_sample.sample_threshold == 250_000


@pytest.mark.unit
def test_customer_sampling_is_deterministic_and_keeps_whole_customers() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = []
    for index in range(40):
        customer = f"customer-{index:02d}"
        rows.append(
            _row(customer=customer, interaction=f"i{index}a", decision_time=when, target=True)
        )
        rows.append(
            _row(
                customer=customer,
                interaction=f"i{index}a",
                decision_time=when,
                target=True,
                outcome="Clicked",
            )
        )
    frame = pl.DataFrame(rows)
    config = _config(customer_sample={"fraction": 0.5})
    processor = FrequencyResponseProcessor(config, computation_hash="hash")

    assert config.customer_sample is not None
    membership = (
        frame.select(pl.col("CustomerID").unique().sort())
        .with_columns(
            (
                pl.col("CustomerID").hash(*model.FREQUENCY_CUSTOMER_SAMPLE_SEEDS)
                % pl.lit(model.FREQUENCY_CUSTOMER_SAMPLE_MODULUS, dtype=pl.UInt64)
                < pl.lit(config.customer_sample.sample_threshold, dtype=pl.UInt64)
            ).alias("sampled")
        )
    )
    expected = set(membership.filter(pl.col("sampled"))["CustomerID"].to_list())
    assert 0 < len(expected) < 40

    prepared = processor.checkpoint_contacts_lazy(
        frame.drop(TARGET_CHUNK_COLUMN).lazy()
    ).collect()
    assert set(prepared["CustomerID"].to_list()) == expected
    # Every row of a sampled customer survives; repeated runs are identical.
    assert prepared.height == 2 * len(expected)
    again = processor.checkpoint_contacts_lazy(frame.drop(TARGET_CHUNK_COLUMN).lazy()).collect()
    assert again.equals(prepared)

    aggregated = processor.chunk_aggregate(frame.lazy(), _ctx())
    assert aggregated["Contacts"].sum() == len(expected)


@pytest.mark.unit
def test_daily_bucket_adds_prior_full_day_counters_to_exact_intraday_sequence() -> None:
    target_day = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = [
        # Two distinct prior-day exposures (one is a duplicated contact whose
        # rows must count once) inside the 7-day window.
        _row(
            customer="c",
            interaction="d7-a",
            decision_time=dt.datetime(2024, 1, 2, 9, tzinfo=dt.UTC),
            target=False,
        ),
        _row(
            customer="c",
            interaction="d7-a",
            decision_time=dt.datetime(2024, 1, 2, 9, tzinfo=dt.UTC),
            target=False,
            outcome="Clicked",
        ),
        _row(
            customer="c",
            interaction="d6-b",
            decision_time=dt.datetime(2024, 1, 3, 23, tzinfo=dt.UTC),
            target=False,
        ),
        # A day outside the last 7 calendar days contributes nothing.
        _row(
            customer="c",
            interaction="old",
            decision_time=dt.datetime(2024, 1, 1, 23, 59, tzinfo=dt.UTC),
            target=False,
        ),
        _row(customer="c", interaction="target", decision_time=target_day, target=True),
    ]
    config = _config(window_granularity="daily")
    aggregated = _aggregate(rows, config)

    # Prior counters: 2024-01-02 and 2024-01-03 => 2; intra-day sequence adds 1.
    assert aggregated["ExposureBucket"].to_list() == [3]
    assert aggregated["Contacts"].sum() == 1


@pytest.mark.unit
def test_daily_and_exact_agree_on_midnight_aligned_exposures() -> None:
    def midnight(day: int) -> dt.datetime:
        return dt.datetime(2024, 1, day, tzinfo=dt.UTC)

    rows = [
        # Exactly 7 days before the target: excluded by both modes.
        _row(customer="c", interaction="boundary", decision_time=midnight(1), target=False),
        _row(customer="c", interaction="inside", decision_time=midnight(2), target=False),
        _row(customer="c", interaction="later", decision_time=midnight(5), target=False),
        _row(customer="c", interaction="target", decision_time=midnight(8), target=True),
        _row(
            customer="c",
            interaction="target",
            decision_time=midnight(8),
            target=True,
            action="runner",
            rank=2,
            outcome="Pending",
            propensity=0.4,
        ),
        _row(customer="other", interaction="solo", decision_time=midnight(8), target=True),
    ]
    exact = _aggregate(rows, _config())
    daily = _aggregate(rows, _config(window_granularity="daily"))

    assert exact.equals(daily)
    assert exact.filter(pl.col("Contacts") > 0)["ExposureBucket"].sort().to_list() == [1, 3]
