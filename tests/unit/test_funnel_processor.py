"""Focused tests for configurable funnel recipe sketch states."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from valuestream.config import model
from valuestream.processors.binary_outcome import ChunkContext
from valuestream.processors.funnel import FunnelProcessor
from valuestream.states import cpc, topk


def _ctx() -> ChunkContext:
    return ChunkContext(
        pipeline_run_id="00000000-0000-0000-0000-000000000009",
        chunk_id="20240101",
        created_at=dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
    )


def test_unscoped_recipe_sketches_build_without_a_funnel_entity() -> None:
    processor = FunnelProcessor(
        model.FunnelProcessor.model_validate(
            {
                "id": "journey",
                "source": "events",
                "kind": "funnel",
                "group_by": ["Region"],
                "stages": [
                    {
                        "name": "Started",
                        "when": {"col": "Started"},
                    }
                ],
                "states": {
                    "Region_cpc": {
                        "type": "cpc",
                        "source_column": "Region",
                        "lg_k": 11,
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
            "Region": ["North", "North", "North"],
            "Started": [True, False, True],
            "Category": ["A", "B", "A"],
        }
    )

    out = processor.chunk_aggregate(frame.lazy(), _ctx())

    assert out["Started_Count"].to_list() == [2]
    assert cpc.estimate(out["Region_cpc"][0]) == pytest.approx(1, rel=0.02)
    assert topk.frequent_items(out["Category_topk"][0])[0]["item"] == "A"
