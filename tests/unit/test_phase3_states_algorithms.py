"""Phase 3 state and RFM helper coverage."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from valuestream.algorithms import rfm
from valuestream.states import hll, theta, topk


@pytest.mark.unit
def test_theta_set_algebra_exact_for_small_sets() -> None:
    left = theta.build(["a", "b", "c"])
    right = theta.build(["b", "c", "d"])

    assert theta.estimate(theta.merge([left, right])) == pytest.approx(4.0)
    assert theta.estimate(theta.intersect([left, right])) == pytest.approx(2.0)
    assert theta.estimate(theta.a_not_b(left, right)) == pytest.approx(1.0)
    lower, upper = theta.bounds(left)
    assert lower <= 3 <= upper


@pytest.mark.unit
def test_hll_bounds_cover_small_exact_cardinality() -> None:
    payload = hll.build(["a", "b", "c"])

    lower, upper = hll.bounds(payload)

    assert lower <= 3 <= upper


@pytest.mark.unit
def test_topk_merge_returns_frequent_items() -> None:
    first = topk.build(["web", "web", "mobile"])
    second = topk.build(["web", "branch", "branch"])

    merged = topk.merge([first, second])
    items = topk.frequent_items(merged)

    assert topk.weight(merged) == 6
    assert items[0]["item"] in {"web", "branch"}
    assert {item["item"]: item["estimate"] for item in items}["web"] == 3


@pytest.mark.unit
def test_rfm_helper_adds_segment_columns() -> None:
    frame = pl.DataFrame(
        {
            "CustomerID": ["c1", "c2", "c3", "c4"],
            "unique_holdings": [4, 3, 2, 1],
            "lifetime_value": [400.0, 240.0, 120.0, 20.0],
            "MinPurchasedDate": [
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 2),
                dt.datetime(2024, 1, 3),
                dt.datetime(2024, 1, 4),
            ],
            "MaxPurchasedDate": [
                dt.datetime(2024, 1, 9),
                dt.datetime(2024, 1, 8),
                dt.datetime(2024, 1, 7),
                dt.datetime(2024, 1, 6),
            ],
        }
    )

    out = rfm.with_rfm(frame)

    assert {"rfm_seg", "rfm_segment", "rfm_score"} <= set(out.columns)
    assert out["rfm_segment"].to_list() == [
        rfm.segment_name(code) for code in out["rfm_seg"].to_list()
    ]
    assert out["rfm_score"].min() >= 1.0
    assert out["rfm_score"].max() <= 4.0
