"""Focused tests for CPC cardinality state helpers."""

from __future__ import annotations

import polars as pl
import pytest

from valuestream.config import model
from valuestream.query import executor
from valuestream.states import cpc, theta


def test_cpc_build_estimate_and_bounds() -> None:
    payload = cpc.build(["a", "b", "b", "c", None])

    assert cpc.estimate(payload) == pytest.approx(3, rel=0.02)
    lower, upper = cpc.bounds(payload)
    assert lower <= 3 <= upper


def test_cpc_merge_and_empty_payloads() -> None:
    merged = cpc.merge(
        [
            cpc.build(["a", "b"]),
            None,
            cpc.build(["b", "c"]),
        ]
    )

    assert cpc.estimate(merged) == pytest.approx(3, rel=0.02)
    assert cpc.estimate(None) == 0.0
    assert cpc.bounds(None) == (0.0, 0.0)


def test_approx_distinct_metric_dispatches_to_cpc_state() -> None:
    state_name = "UniqueCustomers_cpc"
    metric = model.ApproxDistinctCountMetric.model_validate(
        {"source": "engagement", "kind": "approx_distinct_count", "state": state_name}
    )
    frame = pl.DataFrame({state_name: [cpc.build(["a", "b", "c"])]})

    result = executor._derive_metric(
        frame,
        "UniqueCustomers",
        metric,
        {"UniqueCustomers": metric},
        state_specs={state_name: model.StateSpec.model_validate({"type": "cpc"})},
    )

    assert result["UniqueCustomers"][0] == pytest.approx(3, rel=0.02)


def test_approx_distinct_metric_dispatches_to_theta_state() -> None:
    state_name = "UniqueCustomers_theta"
    metric = model.ApproxDistinctCountMetric.model_validate(
        {"source": "engagement", "kind": "approx_distinct_count", "state": state_name}
    )
    frame = pl.DataFrame({state_name: [theta.build(["a", "b", "c"])]})

    result = executor._derive_metric(
        frame,
        "UniqueCustomers",
        metric,
        {"UniqueCustomers": metric},
        state_specs={state_name: model.StateSpec.model_validate({"type": "theta"})},
    )

    assert result["UniqueCustomers"][0] == pytest.approx(3, rel=0.02)
