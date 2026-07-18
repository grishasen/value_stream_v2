"""Focused unit tests for the Phase 1 binary_outcome processor."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from valuestream.config import model
from valuestream.processors.binary_outcome import BinaryOutcomeProcessor, ChunkContext
from valuestream.states import hll, topk


def _ctx() -> ChunkContext:
    return ChunkContext(
        pipeline_run_id="00000000-0000-0000-0000-000000000001",
        chunk_id="20240101",
        created_at=dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
    )


def _processor(config: dict[str, object]) -> BinaryOutcomeProcessor:
    return BinaryOutcomeProcessor(model.BinaryOutcomeProcessor.model_validate(config))


def test_rejects_non_binary_processor() -> None:
    processor = model.NumericDistributionProcessor.model_validate(
        {"id": "p", "source": "ih", "kind": "numeric_distribution"}
    )
    with pytest.raises(TypeError):
        BinaryOutcomeProcessor(processor)


def test_default_states_and_default_outcome_without_time_column() -> None:
    processor = _processor({"id": "p", "source": "ih", "kind": "binary_outcome"})
    frame = pl.DataFrame(
        {
            "Outcome": ["Clicked", "Impression"],
            "Channel": ["Web", "Web"],
            "Group": ["Cards", "Cards"],
        }
    )

    out = processor.chunk_aggregate(frame.lazy(), _ctx())

    assert {"Count", "Positives", "Negatives"} <= set(out.columns)
    assert out["period"].to_list() == ["ALL"]
    assert out["Positives"].to_list() == [1]


def test_mixed_default_outcome_values_match_integer_outcome_column() -> None:
    processor = _processor(
        {
            "id": "p",
            "source": "ih",
            "kind": "binary_outcome",
            "outcome": {
                "column": "Outcome",
                "positive_values": [1, "Clicked"],
                "negative_values": [0, "Impression"],
            },
        }
    )
    frame = pl.DataFrame({"Outcome": [1, 0, 0], "Channel": ["Web", "Web", "Web"]})

    out = processor.chunk_aggregate(frame.lazy(), _ctx())

    assert out.select("Count", "Positives", "Negatives").row(0) == (3, 1, 2)


def test_filter_dedup_variant_group_by_and_value_states() -> None:
    processor = _processor(
        {
            "id": "p",
            "source": "ih",
            "kind": "binary_outcome",
            "group_by": ["Channel", "Group", "Segment"],
            "variant_column": "Variant",
            "dedup_keys": ["InteractionID"],
            "filter": {"op": "eq", "column": "Channel", "value": "Web"},
            "outcome": {
                "column": "Outcome",
                "positive_values": ["Clicked"],
                "negative_values": ["Impression"],
            },
            "states": {
                "Count": {"type": "count"},
                "Positives": {"type": "count"},
                "Negatives": {"type": "count"},
                "Revenue": {"type": "value_sum", "source_column": "Revenue"},
                "MinScore": {"type": "min", "source_column": "Score"},
                "MaxScore": {"type": "max", "source_column": "Score"},
                "Customers": {"type": "hll", "source_column": "CustomerID", "lg_k": 12},
            },
        }
    )
    frame = pl.DataFrame(
        {
            "day": [dt.date(2024, 1, 1)] * 4,
            "Channel": ["Web", "Web", "Web", "Mobile"],
            "Group": ["Cards", "Cards", "Cards", "Cards"],
            "Variant": ["Test", "Test", "Test", "Test"],
            "Segment": ["Mass", "Mass", "Mass", "Mass"],
            "InteractionID": ["i1", "i1", "i2", "i3"],
            "Outcome": ["Impression", "Clicked", "Impression", "Clicked"],
            "Revenue": [1.0, 5.0, 2.0, 100.0],
            "Score": [0.1, 0.8, 0.3, 0.9],
            "CustomerID": ["c1", "c1", "c2", "c3"],
        }
    )

    out = processor.chunk_aggregate(frame.lazy(), _ctx())

    assert out.select("Count", "Positives", "Negatives", "Revenue").row(0) == (2, 1, 1, 7.0)
    assert out.select("MinScore", "MaxScore").row(0) == (0.3, 0.8)
    assert out.select("Variant", "Segment").row(0) == ("Test", "Mass")
    assert hll.estimate(out["Customers"][0]) == pytest.approx(2, rel=0.02)


def test_duplicate_variant_group_key_is_defensively_deduplicated() -> None:
    processor = _processor(
        {
            "id": "p",
            "source": "ih",
            "kind": "binary_outcome",
            "group_by": ["Variant"],
            "variant_column": "Variant",
            "outcome": {
                "column": "Outcome",
                "positive_values": ["Clicked"],
                "negative_values": ["Impression"],
            },
        }
    )
    frame = pl.DataFrame(
        {
            "Variant": ["Test", "Control"],
            "Outcome": ["Clicked", "Impression"],
        }
    )

    out = processor.chunk_aggregate(frame.lazy(), _ctx())

    assert out.columns.count("Variant") == 1
    assert out.select("Count").sum().item() == 2


def test_topk_recipe_state_builds_from_any_configured_source_field() -> None:
    processor = _processor(
        {
            "id": "p",
            "source": "ih",
            "kind": "binary_outcome",
            "states": {
                "Count": {"type": "count"},
                "Positives": {"type": "count"},
                "Negatives": {"type": "count"},
                "Category_topk": {
                    "type": "topk",
                    "source_column": "Category",
                    "lg_max_map_size": 10,
                },
            },
        }
    )
    frame = pl.DataFrame(
        {
            "Outcome": ["Clicked", "Impression", "Clicked"],
            "Category": ["A", "B", "A"],
        }
    )

    out = processor.chunk_aggregate(frame.lazy(), _ctx())

    assert topk.frequent_items(out["Category_topk"][0])[0]["item"] == "A"


def test_compact_and_merge_edge_branches() -> None:
    processor = _processor(
        {
            "id": "p",
            "source": "ih",
            "kind": "binary_outcome",
            "group_by": ["Channel"],
            "states": {
                "Count": {"type": "count"},
                "Positives": {"type": "count"},
                "Negatives": {"type": "count"},
            },
        }
    )
    daily = pl.DataFrame(
        {
            "Channel": ["Web"],
            "day": [dt.date(2024, 1, 1)],
            "period": ["2024-01"],
            "Count": [2],
            "Positives": [1],
            "Negatives": [1],
            "pipeline_run_id": ["run"],
            "chunk_id": ["chunk"],
            "created_at": [dt.datetime(2024, 1, 1)],
            "config_hash": ["hash"],
        }
    )

    assert processor.compact(pl.DataFrame(), "summary", _ctx()).is_empty()
    assert processor.compact(daily, "daily", _ctx()).select("Count", "Positives").row(0) == (2, 1)
    assert processor.compact(daily, "summary", _ctx())["period"].to_list() == ["2024-01"]
    with pytest.raises(ValueError, match="unsupported compact grain"):
        processor.compact(daily, "weekly", _ctx())

    merged = processor.merge(daily.drop(["Channel", "day", "period"]), group_columns=[])
    assert merged.select("Count", "Positives", "Negatives").row(0) == (2, 1, 1)


def test_daily_compaction_fast_path_restamps_provenance() -> None:
    processor = _processor(
        {
            "id": "p",
            "source": "ih",
            "kind": "binary_outcome",
            "group_by": ["Channel"],
            "states": {
                "Count": {"type": "count"},
                "Positives": {"type": "count"},
                "Negatives": {"type": "count"},
            },
        }
    )
    daily = pl.DataFrame(
        {
            "Channel": ["Web"],
            "day": [dt.date(2024, 1, 1)],
            "period": ["stale"],
            "Count": [2],
            "Positives": [1],
            "Negatives": [999],
            "pipeline_run_id": ["old-run"],
            "chunk_id": ["old-chunk"],
            "created_at": [dt.datetime(2023, 1, 1, tzinfo=dt.UTC)],
            "config_hash": ["old-hash"],
        }
    )
    ctx = ChunkContext(
        pipeline_run_id="new-run",
        chunk_id="new-chunk",
        created_at=dt.datetime(2024, 1, 2, tzinfo=dt.UTC),
    )

    compacted = processor.compact(daily, "daily", ctx)

    assert compacted.select("Count", "Positives", "Negatives").row(0) == (2, 1, 1)
    assert compacted["period"].to_list() == ["2024-01"]
    assert compacted["pipeline_run_id"].to_list() == ["new-run"]
    assert compacted["chunk_id"].to_list() == ["new-chunk"]
    assert compacted["created_at"].to_list() == [ctx.created_at]
    assert compacted["config_hash"].to_list() == [processor.config_hash]
