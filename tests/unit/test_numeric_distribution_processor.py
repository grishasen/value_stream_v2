"""Focused unit tests for the Phase 2 numeric_distribution processor."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from valuestream.config import model
from valuestream.processors.binary_outcome import ChunkContext
from valuestream.processors.numeric_distribution import NumericDistributionProcessor
from valuestream.states import cpc, kll, tdigest


def _ctx() -> ChunkContext:
    return ChunkContext(
        pipeline_run_id="00000000-0000-0000-0000-000000000002",
        chunk_id="20240101",
        created_at=dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
    )


def _processor() -> NumericDistributionProcessor:
    return NumericDistributionProcessor(
        model.NumericDistributionProcessor.model_validate(
            {
                "id": "descriptive",
                "source": "ih",
                "kind": "numeric_distribution",
                "group_by": ["Channel"],
                "properties": ["Propensity"],
            }
        )
    )


def test_rejects_non_numeric_processor() -> None:
    processor = model.BinaryOutcomeProcessor.model_validate(
        {"id": "p", "source": "ih", "kind": "binary_outcome"}
    )
    with pytest.raises(TypeError):
        NumericDistributionProcessor(processor)


def test_chunk_aggregate_builds_descriptive_states() -> None:
    processor = _processor()
    frame = pl.DataFrame(
        {
            "day": [dt.date(2024, 1, 1)] * 3,
            "Channel": ["Web", "Web", "Web"],
            "Propensity": [0.1, 0.2, 0.9],
        }
    )

    out = processor.chunk_aggregate(frame.lazy(), _ctx())

    assert out.select("Propensity_Count", "Propensity_Sum").row(0) == (3, pytest.approx(1.2))
    assert out["Propensity_Mean"][0] == pytest.approx(0.4)
    assert out["Propensity_Min"][0] == pytest.approx(0.1)
    assert out["Propensity_Max"][0] == pytest.approx(0.9)
    assert tdigest.quantile(out["Propensity_tdigest"][0], 0.5) == pytest.approx(0.2, abs=0.1)
    assert out["period"].to_list() == ["2024-01"]


def test_merge_uses_pooled_variance() -> None:
    processor = _processor()
    partials = pl.DataFrame(
        {
            "Channel": ["Web", "Web"],
            "Propensity_Count": [2, 2],
            "Propensity_Sum": [3.0, 7.0],
            "Propensity_Mean": [1.5, 3.5],
            "Propensity_Var": [0.5, 0.5],
            "Propensity_Min": [1.0, 3.0],
            "Propensity_Max": [2.0, 4.0],
            "Propensity_tdigest": [tdigest.build([1.0, 2.0]), tdigest.build([3.0, 4.0])],
        }
    )

    merged = processor.merge(partials)

    assert merged["Propensity_Count"].to_list() == [4]
    assert merged["Propensity_Mean"].to_list() == [pytest.approx(2.5)]
    assert merged["Propensity_Var"].to_list() == [pytest.approx(1.6666666667)]
    assert merged["Propensity_Min"].to_list() == [1.0]
    assert merged["Propensity_Max"].to_list() == [4.0]


def test_partial_explicit_states_do_not_become_group_columns() -> None:
    processor = NumericDistributionProcessor(
        model.NumericDistributionProcessor.model_validate(
            {
                "id": "response_time",
                "source": "ih",
                "kind": "numeric_distribution",
                "group_by": ["Channel"],
                "properties": ["ResponseTime"],
                "states": {"ResponseTime_tdigest": {"type": "tdigest"}},
            }
        )
    )
    partials = pl.DataFrame(
        {
            "Channel": ["Web", "Web"],
            "ResponseTime_Count": [2, 3],
            "ResponseTime_Sum": [3.0, 9.0],
            "ResponseTime_Mean": [1.5, 3.0],
            "ResponseTime_Var": [0.5, 1.0],
            "ResponseTime_Min": [1.0, 2.0],
            "ResponseTime_Max": [2.0, 4.0],
            "ResponseTime_tdigest": [
                tdigest.build([1.0, 2.0]),
                tdigest.build([2.0, 3.0, 4.0]),
            ],
        }
    )

    merged = processor.merge(partials)

    assert merged.columns.count("ResponseTime_Count") == 1
    assert merged["ResponseTime_Count"].to_list() == [5]
    assert "ResponseTime_Count" not in [
        column
        for column in merged.columns
        if column not in processor.state_specs and column != "Channel"
    ]


def test_explicit_recipe_sketches_build_alongside_default_distribution_states() -> None:
    processor = NumericDistributionProcessor(
        model.NumericDistributionProcessor.model_validate(
            {
                "id": "response_time",
                "source": "ih",
                "kind": "numeric_distribution",
                "group_by": ["Channel"],
                "properties": ["ResponseTime"],
                "states": {
                    "Channel_cpc": {
                        "type": "cpc",
                        "source_column": "Channel",
                        "lg_k": 11,
                    },
                    "ResponseTime_kll": {
                        "type": "kll",
                        "source_column": "ResponseTime",
                        "k": 200,
                    },
                },
            }
        )
    )
    frame = pl.DataFrame(
        {
            "Channel": ["Web", "Web", "Web"],
            "ResponseTime": [1.0, 2.0, 100.0],
        }
    )

    out = processor.chunk_aggregate(frame.lazy(), _ctx())

    assert cpc.estimate(out["Channel_cpc"][0]) == pytest.approx(1, rel=0.02)
    assert kll.quantile(out["ResponseTime_kll"][0], 0.5) == pytest.approx(2.0)
    assert "ResponseTime_tdigest" in out.columns


@pytest.mark.parametrize("quantile_engine", ["tdigest", "kll"])
def test_bulk_sketch_mode_matches_legacy_distribution_semantics(
    quantile_engine: str,
) -> None:
    base_config = {
        "id": "descriptive",
        "source": "ih",
        "kind": "numeric_distribution",
        "group_by": ["Channel"],
        "properties": ["Value"],
        "quantile_engine": quantile_engine,
        "sketch_build_mode": "legacy",
    }
    legacy = NumericDistributionProcessor(
        model.NumericDistributionProcessor.model_validate(base_config)
    )
    bulk = NumericDistributionProcessor(
        model.NumericDistributionProcessor.model_validate(
            {**base_config, "sketch_build_mode": "bulk"}
        )
    )
    values = [float(value) for value in range(128)]
    frame = pl.DataFrame(
        {
            "Channel": ["Web"] * 64 + ["Mobile"] * 64,
            "Value": values,
        }
    )

    legacy_out = legacy.chunk_aggregate(frame.lazy(), _ctx()).sort("Channel")
    bulk_out = bulk.chunk_aggregate(frame.lazy(), _ctx()).sort("Channel")

    assert bulk_out.select("Channel", "Value_Count", "Value_Sum").equals(
        legacy_out.select("Channel", "Value_Count", "Value_Sum")
    )
    state_name = f"Value_{quantile_engine}"
    for legacy_payload, bulk_payload in zip(
        legacy_out[state_name], bulk_out[state_name], strict=True
    ):
        if quantile_engine == "tdigest":
            assert tdigest.weight(bulk_payload) == tdigest.weight(legacy_payload)
            quantile = tdigest.quantile
        else:
            assert kll.count(bulk_payload) == kll.count(legacy_payload)
            quantile = kll.quantile
        assert quantile(bulk_payload, 0.5) == pytest.approx(quantile(legacy_payload, 0.5), abs=1.0)
