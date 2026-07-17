"""Unit tests for shared processor helpers."""

from __future__ import annotations

import polars as pl
import pytest

import valuestream.processors.processors_helper as p3
from valuestream.config import model
from valuestream.states import tdigest


@pytest.mark.unit
def test_merge_state_frame_combines_pooled_variance() -> None:
    # Two partials for group "A": [1, 3] and [5, 7]. Combined sample variance of
    # [1, 3, 5, 7] is 20/3; the helper must reproduce that, not drop the column.
    frame = pl.DataFrame(
        {
            "g": ["A", "A"],
            "X_Count": [2, 2],
            "X_Mean": [2.0, 6.0],
            "X_Var": [2.0, 2.0],
        }
    )
    specs = {
        "X_Count": model.StateSpec.model_validate({"type": "count"}),
        "X_Mean": model.StateSpec.model_validate({"type": "pooled_mean", "weight": "X_Count"}),
        "X_Var": model.StateSpec.model_validate({"type": "pooled_variance"}),
    }

    merged = p3.merge_state_frame(frame, specs, ["g"])

    assert merged["X_Count"].to_list() == [4]
    assert merged["X_Mean"].to_list() == [pytest.approx(4.0)]
    assert merged["X_Var"].to_list() == [pytest.approx(20.0 / 3.0)]


@pytest.mark.unit
def test_merge_state_frame_pooled_variance_requires_companions() -> None:
    frame = pl.DataFrame({"g": ["A", "A"], "X_Var": [2.0, 2.0]})
    specs = {"X_Var": model.StateSpec.model_validate({"type": "pooled_variance"})}

    with pytest.raises(ValueError, match="pooled_variance state 'X_Var' requires companion"):
        p3.merge_state_frame(frame, specs, ["g"])


@pytest.mark.unit
def test_compact_state_frame_projects_unique_identity_groups_without_merging() -> None:
    payload = tdigest.build([1.0, 2.0, 3.0])
    frame = pl.DataFrame(
        {
            "g": ["A", "B"],
            "Count": [3, 3],
            "X_tdigest": [payload, payload],
        }
    )
    specs = {
        "Count": model.StateSpec.model_validate({"type": "count"}),
        "X_tdigest": model.StateSpec.model_validate({"type": "tdigest"}),
    }

    def unexpected_merge(*_args: object, **_kwargs: object) -> pl.DataFrame:
        raise AssertionError("unique identity groups must not be merged")

    compacted = p3.compact_state_frame(
        frame,
        specs,
        ["g"],
        unexpected_merge,
        identity_level=True,
    )

    assert compacted.columns == ["g", "Count", "X_tdigest"]
    assert compacted["X_tdigest"].to_list() == [payload, payload]


@pytest.mark.unit
def test_compact_state_frame_preserves_singleton_pooled_state_semantics() -> None:
    frame = pl.DataFrame(
        {
            "g": ["A", "B"],
            "Count": [3, 0],
            "X_Mean": [0.1, None],
            "X_Var": [0.02, None],
        }
    )
    specs = {
        "Count": model.StateSpec.model_validate({"type": "count"}),
        "X_Mean": model.StateSpec.model_validate({"type": "pooled_mean", "weight": "Count"}),
        "X_Var": model.StateSpec.model_validate(
            {"type": "pooled_variance", "mean": "X_Mean", "weight": "Count"}
        ),
    }
    expected = p3.merge_state_frame(frame, specs, ["g"]).sort("g")

    compacted = p3.compact_state_frame(
        frame,
        specs,
        ["g"],
        p3.merge_state_frame,
        identity_level=True,
    ).sort("g")

    assert compacted.equals(expected)


@pytest.mark.unit
def test_compact_state_frame_falls_back_for_duplicate_identity_groups() -> None:
    payload = tdigest.build([1.0, 2.0])
    frame = pl.DataFrame(
        {
            "g": ["A", "A"],
            "Count": [2, 2],
            "X_tdigest": [payload, payload],
        }
    )
    specs = {
        "Count": model.StateSpec.model_validate({"type": "count"}),
        "X_tdigest": model.StateSpec.model_validate({"type": "tdigest"}),
    }
    calls = 0

    def merge(frame: pl.DataFrame, *, group_columns: list[str]) -> pl.DataFrame:
        nonlocal calls
        calls += 1
        return p3.merge_state_frame(frame, specs, group_columns)

    compacted = p3.compact_state_frame(
        frame,
        specs,
        ["g"],
        merge,
        identity_level=True,
    )

    assert calls == 1
    assert compacted["Count"].to_list() == [4]
    assert tdigest.weight(compacted["X_tdigest"][0]) == 4


@pytest.mark.unit
def test_compact_state_frame_uses_merge_for_coarser_level() -> None:
    frame = pl.DataFrame({"g": ["A", "B"], "Count": [2, 3]})
    specs = {"Count": model.StateSpec.model_validate({"type": "count"})}
    calls = 0

    def merge(frame: pl.DataFrame, *, group_columns: list[str]) -> pl.DataFrame:
        nonlocal calls
        calls += 1
        return p3.merge_state_frame(frame, specs, group_columns)

    compacted = p3.compact_state_frame(
        frame,
        specs,
        ["g"],
        merge,
        identity_level=False,
    )

    assert calls == 1
    assert compacted.sort("g")["Count"].to_list() == [2, 3]
