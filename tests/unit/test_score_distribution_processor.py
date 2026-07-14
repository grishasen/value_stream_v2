"""Focused unit tests for the Phase 2 score_distribution processor."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from valuestream.algorithms.curves import curve_from_digests
from valuestream.config import model
from valuestream.processors.binary_outcome import ChunkContext
from valuestream.processors.score_distribution import ScoreDistributionProcessor
from valuestream.states import cpc, kll, tdigest, topk


def _ctx() -> ChunkContext:
    return ChunkContext(
        pipeline_run_id="00000000-0000-0000-0000-000000000003",
        chunk_id="20240101",
        created_at=dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
    )


def _processor() -> ScoreDistributionProcessor:
    return ScoreDistributionProcessor(
        model.ScoreDistributionProcessor.model_validate(
            {
                "id": "scores",
                "source": "ih",
                "kind": "score_distribution",
                "group_by": ["Channel"],
                "score_properties": ["Propensity", "FinalPropensity"],
                "outcome": {
                    "column": "Outcome",
                    "positive_values": ["Clicked"],
                    "negative_values": ["Impression"],
                },
                "dedup_keys": ["InteractionID"],
            }
        )
    )


def test_rejects_non_score_processor() -> None:
    processor = model.BinaryOutcomeProcessor.model_validate(
        {"id": "p", "source": "ih", "kind": "binary_outcome"}
    )
    with pytest.raises(TypeError):
        ScoreDistributionProcessor(processor)


def test_chunk_aggregate_builds_score_states() -> None:
    processor = _processor()
    frame = pl.DataFrame(
        {
            "day": [dt.date(2024, 1, 1)] * 4,
            "Channel": ["Web", "Web", "Web", "Web"],
            "Outcome": ["Impression", "Clicked", "Impression", "Clicked"],
            "Propensity": [0.1, 0.9, 0.2, 0.8],
            "FinalPropensity": [0.15, 0.85, 0.25, 0.75],
            "CustomerID": ["c1", "c2", "c3", "c4"],
            "InteractionID": ["i1", "i2", "i3", "i4"],
            "Name": ["A", "A", "B", "C"],
        }
    )

    out = processor.chunk_aggregate(frame.lazy(), _ctx())
    curve = curve_from_digests(
        out["Propensity_tdigest_positives"][0],
        out["Propensity_tdigest_negatives"][0],
    )

    assert out["Count"].to_list() == [4]
    assert curve.roc_auc > 0.95
    assert cpc.estimate(out["UniqueCustomers_cpc"][0]) == pytest.approx(4, rel=0.02)
    assert 0 <= out["personalization"][0] <= 1
    assert out["period"].to_list() == ["2024-01"]


def test_mixed_default_outcome_values_match_integer_outcome_column() -> None:
    processor = ScoreDistributionProcessor(
        model.ScoreDistributionProcessor.model_validate(
            {
                "id": "scores",
                "source": "ih",
                "kind": "score_distribution",
                "score_properties": ["final_propensity"],
                "outcome": {
                    "column": "Outcome",
                    "positive_values": [1, "Clicked"],
                    "negative_values": [0, "Impression"],
                },
            }
        )
    )
    frame = pl.DataFrame(
        {
            "Outcome": [0, 1, 0, 1],
            "final_propensity": [0.1, 0.9, 0.2, 0.8],
            "SubjectID": ["c1", "c2", "c3", "c4"],
        }
    )

    out = processor.chunk_aggregate(frame.lazy(), _ctx())
    curve = curve_from_digests(
        out["final_propensity_tdigest_positives"][0],
        out["final_propensity_tdigest_negatives"][0],
    )

    assert out["Count"].to_list() == [4]
    assert curve.roc_auc > 0.95


def test_merge_weighted_means_and_sketches() -> None:
    processor = _processor()
    left = processor.chunk_aggregate(
        pl.DataFrame(
            {
                "Channel": ["Web", "Web"],
                "Outcome": ["Impression", "Clicked"],
                "Propensity": [0.1, 0.9],
                "FinalPropensity": [0.1, 0.9],
                "CustomerID": ["c1", "c2"],
                "InteractionID": ["i1", "i2"],
                "Name": ["A", "B"],
            }
        ).lazy(),
        _ctx(),
    )
    right = processor.chunk_aggregate(
        pl.DataFrame(
            {
                "Channel": ["Web", "Web"],
                "Outcome": ["Impression", "Clicked"],
                "Propensity": [0.2, 0.8],
                "FinalPropensity": [0.2, 0.8],
                "CustomerID": ["c3", "c4"],
                "InteractionID": ["i3", "i4"],
                "Name": ["C", "D"],
            }
        ).lazy(),
        _ctx(),
    )

    merged = processor.merge(pl.concat([left, right]))

    assert merged["Count"].to_list() == [4]
    assert (
        curve_from_digests(
            merged["Propensity_tdigest_positives"][0],
            merged["Propensity_tdigest_negatives"][0],
        ).roc_auc
        > 0.95
    )


def test_chunk_aggregate_rejects_missing_configured_score_column() -> None:
    processor = _processor()
    frame = pl.DataFrame(
        {
            "Channel": ["Web", "Web"],
            "Outcome": ["Impression", "Clicked"],
            "FinalPropensity": [0.1, 0.9],
            "InteractionID": ["i1", "i2"],
        }
    )

    with pytest.raises(ValueError, match="requires missing score column 'Propensity'"):
        processor.chunk_aggregate(frame.lazy(), _ctx())


def test_chunk_aggregate_uses_explicit_tdigest_source_columns() -> None:
    processor = ScoreDistributionProcessor(
        model.ScoreDistributionProcessor.model_validate(
            {
                "id": "scores",
                "source": "ih",
                "kind": "score_distribution",
                "group_by": ["Channel"],
                "score_properties": ["IgnoredPrimary", "IgnoredCalibrated"],
                "outcome": {
                    "column": "Outcome",
                    "positive_values": ["Clicked"],
                    "negative_values": ["Impression"],
                },
                "states": {
                    "Count": {"type": "count"},
                    "custom_positives": {
                        "type": "tdigest",
                        "source_column": "TransformedScore",
                        "outcome": "positive",
                    },
                    "custom_negatives": {
                        "type": "tdigest",
                        "source_column": "TransformedScore",
                        "outcome": "negative",
                    },
                },
            }
        )
    )
    frame = pl.DataFrame(
        {
            "Channel": ["Web", "Web"],
            "Outcome": ["Impression", "Clicked"],
            "TransformedScore": [0.1, 0.9],
        }
    )

    out = processor.chunk_aggregate(frame.lazy(), _ctx())
    curve = curve_from_digests(out["custom_positives"][0], out["custom_negatives"][0])

    assert curve.roc_auc > 0.95


def test_generic_tdigest_state_infers_its_source_and_uses_all_outcomes() -> None:
    processor = ScoreDistributionProcessor(
        model.ScoreDistributionProcessor.model_validate(
            {
                "id": "scores",
                "source": "ih",
                "kind": "score_distribution",
                "outcome": {
                    "column": "Outcome",
                    "positive_values": ["Clicked"],
                    "negative_values": ["Impression"],
                },
                "states": {
                    "Count": {"type": "count"},
                    "Priority_tdigest": {"type": "tdigest"},
                },
            }
        )
    )
    frame = pl.DataFrame(
        {
            "Outcome": ["Impression", "Clicked", "Impression", "Clicked"],
            "Propensity": [0.1, 0.9, 0.2, 0.8],
            "Priority": [1.0, 2.0, 3.0, 100.0],
        }
    )

    out = processor.chunk_aggregate(frame.lazy(), _ctx())

    assert tdigest.quantile(out["Priority_tdigest"][0], 0.5) == pytest.approx(2.5, abs=0.6)


def test_explicit_kll_and_topk_recipe_states_are_materialized() -> None:
    processor = ScoreDistributionProcessor(
        model.ScoreDistributionProcessor.model_validate(
            {
                "id": "scores",
                "source": "ih",
                "kind": "score_distribution",
                "outcome": {
                    "column": "Outcome",
                    "positive_values": ["Clicked"],
                    "negative_values": ["Impression"],
                },
                "states": {
                    "Count": {"type": "count"},
                    "Priority_kll": {
                        "type": "kll",
                        "source_column": "Priority",
                        "k": 200,
                    },
                    "Category_topk": {
                        "type": "topk",
                        "source_column": "Category",
                        "lg_max_map_size": 10,
                    },
                },
            }
        )
    )
    frame = pl.DataFrame(
        {
            "Outcome": ["Impression", "Clicked", "Impression", "Clicked"],
            "Priority": [1.0, 2.0, 3.0, 100.0],
            "Category": ["A", "A", "B", "A"],
        }
    )

    out = processor.chunk_aggregate(frame.lazy(), _ctx())

    assert kll.quantile(out["Priority_kll"][0], 0.5) == pytest.approx(3.0)
    assert topk.frequent_items(out["Category_topk"][0])[0]["item"] == "A"
