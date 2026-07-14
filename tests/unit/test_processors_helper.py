"""Unit tests for shared processor helpers."""

from __future__ import annotations

import polars as pl
import pytest

import valuestream.processors.processors_helper as p3
from valuestream.config import model


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
