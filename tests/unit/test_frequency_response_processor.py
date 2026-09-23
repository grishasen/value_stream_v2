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

_STATES = {
    "Positives": {"type": "count", "outcome": "positive"},
    "Negatives": {"type": "count", "outcome": "negative"},
}


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
            "rank": "Rank",
        },
        "outcome": {
            "column": "Outcome",
            "positive_values": ["Clicked"],
            "negative_values": ["Impression", "Pending"],
        },
        "scope_by": ["Placement"],
        "states": dict(_STATES),
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
    channel: str = "Web",
    rank: int = 1,
    outcome: str = "Impression",
) -> dict[str, Any]:
    return {
        "CustomerID": customer,
        "InteractionID": interaction,
        "ActionID": action,
        "Placement": placement,
        "Channel": channel,
        "Rank": rank,
        "Outcome": outcome,
        "DecisionTime": decision_time,
        TARGET_CHUNK_COLUMN: target,
    }


def _aggregate(
    rows: list[dict[str, Any]], config: model.FrequencyResponseProcessor | None = None
) -> pl.DataFrame:
    config = config or _config()
    processor = FrequencyResponseProcessor(config, computation_hash="hash")
    keys = [column for column in config.group_by if column != "Day"]
    return processor.chunk_aggregate(pl.DataFrame(rows).lazy(), _ctx()).sort(["Day", *keys])


@pytest.mark.unit
def test_model_contract_is_daily_bucketed_and_scoped() -> None:
    config = _config()

    assert config.partition_lag_hours == 0
    assert config.max_frequency == 7
    assert config.max_rank == 3
    assert config.checkpoint.mode == "source_scan"
    assert config.checkpoint.shards == 64
    assert config.checkpoint.retention_days is None
    assert config.checkpoint_retention_days == 7
    assert config.exposure_key_columns == ["CustomerID", "ActionID", "Placement"]
    assert config.rank_partition_columns == ["CustomerID", "InteractionID", "Placement"]
    assert config.contact_identity_columns == [
        "CustomerID",
        "InteractionID",
        "ActionID",
        "Placement",
    ]
    assert config.derived_columns == {"Day", "ExposureBucket", "ScopeRank", "PriorPositive"}
    everywhere = _config(scope_by=[])
    assert everywhere.exposure_key_columns == ["CustomerID", "ActionID"]
    assert everywhere.rank_partition_columns == ["CustomerID", "InteractionID"]
    channel_placement = _config(scope_by=["Channel", "Placement"])
    assert channel_placement.exposure_key_columns == [
        "CustomerID",
        "ActionID",
        "Channel",
        "Placement",
    ]
    grouped = _config(group_by=["Day", "ExposureBucket", "ScopeRank", "PriorPositive"])
    assert grouped.group_by[-2:] == ["ScopeRank", "PriorPositive"]

    with pytest.raises(ValidationError, match=r"time\.grain 'daily'"):
        _config(time={"property": "DecisionTime", "grain": "hourly"})
    with pytest.raises(ValidationError, match="frequency_column must be present in group_by"):
        _config(group_by=["Day"])
    with pytest.raises(ValidationError, match="reserved derived column"):
        _config(frequency_column="Day")
    with pytest.raises(ValidationError, match="reserved derived column"):
        _config(frequency_column="ScopeRank", group_by=["Day", "ScopeRank"])
    with pytest.raises(ValidationError, match="cannot be used in group_by"):
        _config(group_by=["Day", "ExposureBucket", "config_hash"])
    with pytest.raises(ValidationError, match="cannot be used in group_by"):
        _config(group_by=["Day", "ExposureBucket", "__valuestream_private"])
    with pytest.raises(ValidationError, match="state names collide"):
        _config(states={"ExposureBucket": {"type": "count", "outcome": "positive"}})
    with pytest.raises(ValidationError, match="partition_lag_hours"):
        _config(partition_lag_hours=-1)
    with pytest.raises(ValidationError, match="max_rank"):
        _config(max_rank=0)
    payload = config.model_dump(mode="python")
    payload.pop("scope_by")
    with pytest.raises(ValidationError, match=r"(?s)scope_by.*Field required"):
        model.FrequencyResponseProcessor.model_validate(payload)
    with pytest.raises(ValidationError, match="scope_by columns must be unique"):
        _config(scope_by=["Placement", "Placement"])
    for bound_column in ("CustomerID", "InteractionID", "ActionID", "Rank", "Outcome"):
        with pytest.raises(ValidationError, match="must not repeat the customer"):
            _config(scope_by=[bound_column])
    for invalid_column in ("", " Day", "Day", "ExposureBucket", "ScopeRank", "config_hash"):
        with pytest.raises(ValidationError, match="must contain raw source columns"):
            _config(scope_by=[invalid_column])
    duplicate_columns = config.columns.model_dump()
    duplicate_columns["interaction"] = duplicate_columns["customer"]
    with pytest.raises(ValidationError, match="bindings must be distinct"):
        _config(columns=duplicate_columns)
    with pytest.raises(ValidationError, match="at least one positive and one negative"):
        _config(
            outcome={"column": "Outcome", "positive_values": ["Clicked"], "negative_values": []}
        )
    with pytest.raises(ValidationError, match="both positive and negative"):
        _config(
            outcome={
                "column": "Outcome",
                "positive_values": ["Clicked"],
                "negative_values": ["Impression", "Clicked"],
            }
        )
    with pytest.raises(ValidationError, match="shards"):
        _config(checkpoint={"mode": "persistent_sharded", "shards": 0})
    with pytest.raises(ValidationError, match=r"retention_days must be at least 7"):
        _config(checkpoint={"mode": "persistent_sharded", "shards": 8, "retention_days": 6})
    exact_minimum = _config(
        checkpoint={"mode": "persistent_sharded", "shards": 8, "retention_days": 7}
    )
    assert exact_minimum.checkpoint_retention_days == 7
    columns = config.columns.model_dump()
    columns["rank"] = "__valuestream_frequency_rank"
    with pytest.raises(ValidationError, match=r"raw input bindings.*__valuestream_frequency_rank"):
        _config(columns=columns)
    with pytest.raises(ValidationError, match=r"raw input bindings.*PriorPositive"):
        _config(outcome={**config.outcome.model_dump(), "column": "PriorPositive"})


@pytest.mark.unit
def test_states_are_canonical_for_the_kind() -> None:
    definitions = model.frequency_response_state_definitions()

    assert definitions == _STATES
    assert list(model.frequency_response_states()) == ["Positives", "Negatives"]
    assert set(_config(states=definitions).states) == {"Positives", "Negatives"}
    # A catalog may publish part of the contract without publishing all of it.
    assert set(_config(states={"Positives": _STATES["Positives"]}).states) == {"Positives"}

    with pytest.raises(ValidationError, match="retired rank-2 opportunity contract"):
        _config(states={"Responses": {"type": "count"}})
    with pytest.raises(ValidationError, match="retired rank-2 opportunity contract"):
        _config(states={"RunnerPropensitySum": {"type": "value_sum", "source_column": "Rank"}})
    with pytest.raises(ValidationError, match="not part of the kind's canonical contract"):
        _config(states={"MyOwnCounter": {"type": "count", "outcome": "positive"}})
    with pytest.raises(ValidationError, match=r"'Positives' is canonical"):
        _config(states={"Positives": {"type": "count", "source_column": "ClickedContact"}})
    with pytest.raises(ValidationError, match=r"'Negatives' is canonical"):
        _config(states={"Negatives": {"type": "count", "outcome": "positive"}})
    with pytest.raises(ValidationError, match=r"'Positives' is canonical"):
        _config(
            states={
                "Positives": {
                    "type": "count",
                    "outcome": "positive",
                    "where": {"op": "eq", "column": "Channel", "value": "Web"},
                }
            }
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"alternative_group_by": ["Placement"]}, r"alternative_group_by \(replaced by scope_by"),
        ({"positive_values": ["Clicked"]}, r"positive_values \(replaced by outcome"),
        ({"exposure_values": ["Impression"]}, r"exposure_values \(replaced by outcome"),
        ({"candidate_values": ["Pending"]}, r"candidate_values \(replaced by outcome"),
    ],
)
def test_retired_rank_two_settings_fail_with_a_migration_hint(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _config(**overrides)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("binding", "message"),
    [
        ("placement", "list the placement field in scope_by"),
        ("outcome", r"moved to outcome\.column"),
        ("propensity", "no longer used"),
        ("priority", "no longer used"),
    ],
)
def test_retired_column_bindings_fail_with_a_migration_hint(binding: str, message: str) -> None:
    columns = {**_config().columns.model_dump(), binding: "Legacy"}
    with pytest.raises(ValidationError, match=message):
        _config(columns=columns)


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
        ({"op": "eq", "column": "ScopeRank", "value": 1}, "ScopeRank"),
        ({"polars": 'pl.col("PriorPositive")'}, "PriorPositive"),
    ],
)
def test_processor_filter_cannot_reference_post_filter_derived_columns(
    filter_expression: dict[str, Any],
    derived_column: str,
) -> None:
    with pytest.raises(ValidationError, match=rf"before derived columns.*{derived_column}"):
        _config(filter=filter_expression)


@pytest.mark.unit
def test_required_inputs_exclude_derived_dimensions() -> None:
    config = _config(
        group_by=["Day", "Name", "ExposureBucket", "ScopeRank", "PriorPositive"],
        scope_by=["Channel", "Placement"],
        filter={"op": "eq", "column": "Market", "value": "DE"},
    )
    required = required_input_columns(config)
    history_required = required_history_input_columns(config)
    catalog_source_columns = _processor_source_columns(config)

    assert {
        "DecisionTime",
        "Name",
        "Channel",
        "Placement",
        "Outcome",
        "Rank",
        "Market",
        TARGET_CHUNK_COLUMN,
    } <= required
    assert not config.derived_columns.intersection(required)
    assert {"DecisionTime", "Name", "Channel", "Placement", "Outcome"} <= catalog_source_columns
    assert not config.derived_columns.intersection(catalog_source_columns)
    assert history_required == {
        "CustomerID",
        "InteractionID",
        "ActionID",
        "Rank",
        "Outcome",
        "Channel",
        "Placement",
        "DecisionTime",
        "Market",
    }


@pytest.mark.unit
def test_runtime_rejects_existing_derived_columns_and_non_integer_rank() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    valid = _row(customer="c1", interaction="i1", decision_time=when, target=True)

    with pytest.raises(ValueError, match=r"'ExposureBucket'.*already contains"):
        _aggregate([{**valid, "ExposureBucket": 99}])
    with pytest.raises(ValueError, match=r"'ScopeRank'.*already contains"):
        _aggregate([{**valid, "ScopeRank": 1}])
    with pytest.raises(TypeError, match=r"integer rank.*Float64"):
        _aggregate([{**valid, "Rank": 1.5}])
    missing = {key: value for key, value in valid.items() if key != "Placement"}
    with pytest.raises(ValueError, match=r"requires missing input column\(s\): Placement"):
        _aggregate([missing])


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

    assert out.select("ExposureBucket", "Negatives").rows() == [(1, 1), (7, 1)]


@pytest.mark.unit
def test_impressions_at_every_rank_count_for_the_same_action() -> None:
    """The action's earlier impressions count wherever arbitration ranked it."""

    target_time = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = [
        _row(
            customer="c",
            interaction=f"earlier-{rank}",
            decision_time=target_time - dt.timedelta(hours=rank),
            target=False,
            rank=rank,
        )
        for rank in (2, 4, 6)
    ]
    rows.append(
        _row(customer="c", interaction="target", decision_time=target_time, target=True, rank=6)
    )

    out = _aggregate(rows)

    assert out.select("ExposureBucket", "Positives", "Negatives").rows() == [(4, 0, 1)]


@pytest.mark.unit
def test_scope_by_isolates_counting_and_ranking() -> None:
    earlier = dt.datetime(2024, 1, 8, 10, tzinfo=dt.UTC)
    target_time = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = [
        _row(customer="c", interaction="web", decision_time=earlier, target=False, channel="Web"),
        _row(
            customer="c",
            interaction="flex",
            decision_time=earlier,
            target=False,
            placement="Flex",
        ),
        _row(
            customer="c",
            interaction="target",
            decision_time=target_time,
            target=True,
            channel="Mobile",
        ),
    ]
    group_by = ["Day", "ExposureBucket"]

    everywhere = _aggregate(rows, _config(scope_by=[], group_by=group_by))
    placement = _aggregate(rows, _config(scope_by=["Placement"], group_by=group_by))
    channel_placement = _aggregate(
        rows, _config(scope_by=["Channel", "Placement"], group_by=group_by)
    )

    assert everywhere["ExposureBucket"].to_list() == [3]
    assert placement["ExposureBucket"].to_list() == [2]
    assert channel_placement["ExposureBucket"].to_list() == [1]


@pytest.mark.unit
def test_scope_rank_reranks_shown_actions_inside_the_scope_and_caps() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = [
        # One banner shown at arbitration rank 6 is the placement's first action.
        _row(customer="hero", interaction="i", decision_time=when, target=True, rank=6),
        # A carousel that does not own rank 1: recorded 2, 3, 4, 5 become 1, 2, 3, 3+.
        *[
            _row(
                customer="flex",
                interaction="i",
                decision_time=when,
                target=True,
                action=f"A{rank}",
                placement="Flex",
                rank=rank,
                outcome="Clicked" if rank == 3 else "Impression",
            )
            for rank in (2, 3, 4, 5)
        ],
        _row(customer="flex", interaction="i", decision_time=when, target=True, action="H"),
    ]
    config = _config(group_by=["Day", "Placement", "ExposureBucket", "ScopeRank"])

    out = _aggregate(rows, config)

    assert out.select("Placement", "ScopeRank", "Positives", "Negatives").rows() == [
        ("Flex", 1, 0, 1),
        ("Flex", 2, 1, 0),
        ("Flex", 3, 0, 2),
        ("Hero", 1, 0, 2),
    ]


@pytest.mark.unit
def test_prior_positive_marks_impressions_after_an_earlier_response_in_the_window() -> None:
    target_time = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = [
        _row(
            customer="responded",
            interaction="click",
            decision_time=target_time - dt.timedelta(hours=2),
            target=False,
            outcome="Clicked",
        ),
        _row(customer="responded", interaction="target", decision_time=target_time, target=True),
        # A click outside the strict window no longer counts.
        _row(
            customer="expired",
            interaction="click",
            decision_time=target_time - dt.timedelta(hours=168),
            target=False,
            outcome="Clicked",
        ),
        _row(customer="expired", interaction="target", decision_time=target_time, target=True),
        # The contact's own click and a later click are not "prior".
        _row(
            customer="self",
            interaction="target",
            decision_time=target_time,
            target=True,
            outcome="Clicked",
        ),
        _row(
            customer="later",
            interaction="target",
            decision_time=target_time,
            target=True,
        ),
        _row(
            customer="later",
            interaction="click",
            decision_time=target_time + dt.timedelta(hours=1),
            target=False,
            outcome="Clicked",
        ),
    ]
    config = _config(group_by=["Day", "ExposureBucket", "PriorPositive"])

    out = _aggregate(rows, config)

    assert out.select("ExposureBucket", "PriorPositive", "Positives", "Negatives").rows() == [
        (1, False, 1, 2),
        (2, True, 0, 1),
    ]


@pytest.mark.unit
def test_every_configured_negative_value_is_an_impression_and_positive_wins() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = [
        # A sent email is recorded as Pending and later as Clicked: one contact.
        _row(
            customer="email", interaction="sent", decision_time=when, target=True, outcome="Pending"
        ),
        _row(
            customer="email", interaction="sent", decision_time=when, target=True, outcome="Clicked"
        ),
        _row(
            customer="email",
            interaction="other",
            decision_time=when,
            target=True,
            action="B",
            outcome="Pending",
        ),
        # An unclassified outcome is not an impression.
        _row(
            customer="email",
            interaction="skip",
            decision_time=when,
            target=True,
            action="C",
            outcome="Conversion",
        ),
    ]

    out = _aggregate(rows)

    assert out.select(pl.col("Positives").sum(), pl.col("Negatives").sum()).row(0) == (1, 1)


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
def test_positive_wins_contact_normalization_and_history_overlap_is_not_targeted() -> None:
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

    assert out.select("Positives", "Negatives").row(0) == (1, 0)


@pytest.mark.unit
@pytest.mark.parametrize("scope_by", [[], ["Placement"], ["Channel", "Placement"]])
def test_checkpoint_matches_source_scan_for_cross_day_duplicates_and_late_time(
    scope_by: list[str],
) -> None:
    processor = FrequencyResponseProcessor(
        _config(
            scope_by=scope_by,
            group_by=["Day", "ExposureBucket", "ScopeRank", "PriorPositive"],
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
            outcome="Clicked",
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
            action="other-placement",
            placement="DifferentPlacement",
            rank=2,
            outcome="Pending",
        ),
        _row(
            customer="c1",
            interaction="target",
            decision_time=target_time,
            target=True,
            action="second",
            rank=3,
            outcome="Pending",
        ),
        _row(
            customer="c2",
            interaction="overlap",
            decision_time=target_time,
            target=True,
            outcome="Clicked",
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
    sort = ["Day", "ExposureBucket", "ScopeRank", "PriorPositive"]
    expected = processor.chunk_aggregate(
        pl.DataFrame([*history_rows, *current_rows]).lazy(),
        _ctx(),
    ).sort(sort)
    history_target_checkpoint = processor.checkpoint_contacts_lazy(
        pl.DataFrame(history_rows).drop(TARGET_CHUNK_COLUMN).lazy()
    )
    history_checkpoint = processor.checkpoint_history_contacts_lazy(history_target_checkpoint)
    current_checkpoint = processor.checkpoint_contacts_lazy(
        pl.DataFrame(current_rows).drop(TARGET_CHUNK_COLUMN).lazy()
    )

    history_schema = set(history_checkpoint.collect_schema().names())
    assert history_checkpoint.collect().height == 3
    assert {"Outcome", "Rank", TARGET_CHUNK_COLUMN}.isdisjoint(history_schema)

    actual = (
        processor.checkpoint_aggregate_lazy(
            current_checkpoint,
            [history_checkpoint],
            _ctx(),
        )
        .collect()
        .sort(sort)
    )

    assert actual.equals(expected)
    assert actual.select(pl.col("Positives").sum(), pl.col("Negatives").sum()).row(0) == (1, 2)


@pytest.mark.unit
def test_frequency_keys_isolate_customer_action_and_scope() -> None:
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

    assert out.select("ExposureBucket", "Negatives").rows() == [(1, 3), (2, 1)]


@pytest.mark.unit
def test_chunk_partials_merge_to_the_combined_result() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    first = [
        _row(customer="c1", interaction="i1", decision_time=when, target=True, outcome="Clicked"),
        _row(
            customer="c1",
            interaction="i1",
            decision_time=when,
            target=True,
            action="Second",
            rank=2,
            outcome="Pending",
        ),
    ]
    second = [
        _row(customer="c2", interaction="i2", decision_time=when, target=True),
        _row(
            customer="c2",
            interaction="i2",
            decision_time=when,
            target=True,
            action="Second",
            rank=2,
            outcome="Clicked",
        ),
    ]
    config = _config(group_by=["Day", "ExposureBucket", "ScopeRank"])
    processor = FrequencyResponseProcessor(config, computation_hash="hash")
    partials = [
        processor.chunk_aggregate(pl.DataFrame(rows).lazy(), _ctx(chunk))
        for rows, chunk in ((first, "first"), (second, "second"))
    ]
    merged = processor.merge(
        pl.concat(partials).drop("pipeline_run_id", "chunk_id", "created_at", "config_hash"),
        group_columns=["Day", "ExposureBucket", "ScopeRank", "period"],
    ).sort(["Day", "ExposureBucket", "ScopeRank"])
    combined = (
        processor.chunk_aggregate(pl.DataFrame([*first, *second]).lazy(), _ctx())
        .drop("pipeline_run_id", "chunk_id", "created_at", "config_hash")
        .sort(["Day", "ExposureBucket", "ScopeRank"])
    )

    assert merged.select(combined.columns).equals(combined)
    assert combined.select("ScopeRank", "Positives", "Negatives").rows() == [(1, 1, 1), (2, 1, 1)]


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
    with pytest.raises(ValidationError):
        _config(customer_sample={"fraction": 1e-9})
    with pytest.raises(ValidationError, match="multiple of"):
        _config(customer_sample={"fraction": 1.49e-6})
    with pytest.raises(ValidationError):
        _config(customer_sample={"fraction": 0.0})
    with pytest.raises(ValidationError):
        _config(customer_sample={"fraction": 1.5})
    minimum = _config(customer_sample={"fraction": 1e-6})
    assert minimum.customer_sample is not None
    assert minimum.customer_sample.sample_threshold == 1
    config = _config(customer_sample={"fraction": 0.25})
    assert config.customer_sample is not None
    assert config.customer_sample.sample_threshold == 250_000


@pytest.mark.unit
def test_checkpoint_memory_limit_requires_positive_absolute_size() -> None:
    for value in (
        "1K",
        "1KB",
        "1KiB",
        "2M",
        "2MB",
        "2MiB",
        "3G",
        "3GB",
        "3GiB",
        "4T",
        "4TB",
        "4TiB",
        "0.5 gb",
    ):
        assert _config(checkpoint={"memory_limit": value}).checkpoint.memory_limit == value

    for value in ("0GB", "0.0MiB", "80%", "1B", "1", "-1GB"):
        with pytest.raises(ValidationError):
            _config(checkpoint={"memory_limit": value})


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
    membership = frame.select(pl.col("CustomerID").unique().sort()).with_columns(
        (
            pl.col("CustomerID").cast(pl.String).hash(*model.FREQUENCY_CUSTOMER_SAMPLE_SEEDS)
            % pl.lit(model.FREQUENCY_CUSTOMER_SAMPLE_MODULUS, dtype=pl.UInt64)
            < pl.lit(config.customer_sample.sample_threshold, dtype=pl.UInt64)
        ).alias("sampled")
    )
    expected = set(membership.filter(pl.col("sampled"))["CustomerID"].to_list())
    assert 0 < len(expected) < 40

    prepared = processor.checkpoint_contacts_lazy(frame.drop(TARGET_CHUNK_COLUMN).lazy()).collect()
    assert set(prepared["CustomerID"].to_list()) == expected
    # Every row of a sampled customer survives; repeated runs are identical.
    assert prepared.height == 2 * len(expected)
    again = processor.checkpoint_contacts_lazy(frame.drop(TARGET_CHUNK_COLUMN).lazy()).collect()
    assert again.equals(prepared)

    aggregated = processor.chunk_aggregate(frame.lazy(), _ctx())
    assert aggregated["Positives"].sum() == len(expected)
    assert aggregated["Negatives"].sum() == 0


@pytest.mark.unit
def test_customer_sampling_matches_text_integer_and_dictionary_ids_that_render_alike() -> None:
    when = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    config = _config(customer_sample={"fraction": 0.5})
    processor = FrequencyResponseProcessor(config, computation_hash="hash")

    strings = pl.DataFrame(
        [
            _row(customer=str(index), interaction=f"i{index}", decision_time=when, target=True)
            for index in range(10)
        ]
    ).drop(TARGET_CHUNK_COLUMN)
    integers = strings.with_columns(pl.col("CustomerID").cast(pl.Int64))
    categoricals = strings.with_columns(pl.col("CustomerID").cast(pl.Categorical))

    def selected(frame: pl.DataFrame) -> set[str]:
        prepared = processor.checkpoint_contacts_lazy(frame.lazy()).collect()
        return set(prepared["CustomerID"].cast(pl.String).to_list())

    expected = selected(strings)
    assert expected
    assert expected != set(strings["CustomerID"].to_list())
    assert selected(integers) == expected
    assert selected(categoricals) == expected


@pytest.mark.unit
def test_daily_bucket_adds_prior_full_day_counters_to_exact_intraday_sequence() -> None:
    target_day = dt.datetime(2024, 1, 8, 12, tzinfo=dt.UTC)
    rows = [
        # Two distinct prior-day impressions (one is a duplicated contact whose
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
            rank=3,
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
    config = _config(
        window_granularity="daily", group_by=["Day", "ExposureBucket", "PriorPositive"]
    )
    aggregated = _aggregate(rows, config)

    # Prior counters: 2024-01-02 and 2024-01-03 => 2; intra-day sequence adds 1.
    assert aggregated.select("ExposureBucket", "PriorPositive", "Negatives").rows() == [
        (3, True, 1)
    ]


@pytest.mark.unit
def test_daily_intrachunk_sequence_resets_for_each_utc_decision_day() -> None:
    rows = [
        _row(
            customer="c",
            interaction="day-one",
            decision_time=dt.datetime(2024, 1, 7, 23, 59, tzinfo=dt.UTC),
            target=True,
        ),
        _row(
            customer="c",
            interaction="day-two",
            decision_time=dt.datetime(2024, 1, 8, 0, 1, tzinfo=dt.UTC),
            target=True,
        ),
    ]

    aggregated = _aggregate(rows, _config(window_granularity="daily"))

    assert aggregated.select("Day", "ExposureBucket", "Negatives").rows() == [
        (dt.date(2024, 1, 7), 1, 1),
        (dt.date(2024, 1, 8), 1, 1),
    ]


@pytest.mark.unit
def test_daily_and_exact_agree_on_midnight_aligned_exposures() -> None:
    def midnight(day: int) -> dt.datetime:
        return dt.datetime(2024, 1, day, tzinfo=dt.UTC)

    rows = [
        # Exactly 7 days before the target: excluded by both modes.
        _row(customer="c", interaction="boundary", decision_time=midnight(1), target=False),
        _row(
            customer="c",
            interaction="inside",
            decision_time=midnight(2),
            target=False,
            outcome="Clicked",
        ),
        _row(customer="c", interaction="later", decision_time=midnight(5), target=False),
        _row(customer="c", interaction="target", decision_time=midnight(8), target=True),
        _row(
            customer="c",
            interaction="target",
            decision_time=midnight(8),
            target=True,
            action="second",
            rank=2,
            outcome="Pending",
        ),
        _row(customer="other", interaction="solo", decision_time=midnight(8), target=True),
    ]
    group_by = ["Day", "ExposureBucket", "ScopeRank", "PriorPositive"]
    exact = _aggregate(rows, _config(group_by=group_by))
    daily = _aggregate(rows, _config(window_granularity="daily", group_by=group_by))

    assert exact.equals(daily)
    assert exact.select("ExposureBucket", "ScopeRank", "PriorPositive", "Negatives").rows() == [
        (1, 1, False, 1),
        (1, 2, False, 1),
        (3, 1, True, 1),
    ]
