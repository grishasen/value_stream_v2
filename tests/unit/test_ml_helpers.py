"""Focused tests for ML helper metrics."""

from __future__ import annotations

from math import log2

import polars as pl
import pytest

from valuestream.algorithms import ml_helpers


def test_personalization_uses_average_cosine_without_pair_matrix() -> None:
    customer_ids = ["c1", "c1", "c2", "c2", "c3"]
    action_names = ["A", "B", "A", "B", "C"]

    assert ml_helpers.personalization(customer_ids, action_names) == pytest.approx(2 / 3)


def test_personalization_counts_repeated_recommendations_like_vector_space() -> None:
    customer_ids = ["c1", "c1", "c2"]
    action_names = ["A", "A", "A"]

    assert ml_helpers.personalization(customer_ids, action_names) == pytest.approx(0.0)


def test_novelty_uses_reference_formula() -> None:
    customer_ids = ["c1", "c2", "c3", "c3"]
    interaction_ids = ["i1", "i2", "i3", "i3"]
    action_names = ["A", "A", "B", "C"]

    expected = (
        2 * -(log2(2 / 3) + 1e-10) + 1 * -(log2(1 / 3) + 1e-10) + 1 * -(log2(1 / 3) + 1e-10)
    ) / (3 * 2)
    assert ml_helpers.novelty(customer_ids, interaction_ids, action_names) == pytest.approx(
        expected
    )


def test_native_group_paths_match_scalar_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    customer_ids = pl.Series([f"c{index}" for index in range(1_024)])
    interaction_ids = pl.Series([f"i{index // 4}" for index in range(1_024)])
    action_names = pl.Series([f"action-{(index * 7) % 19}" for index in range(1_024)])

    native_personalization = ml_helpers.personalization(customer_ids, action_names)
    native_novelty = ml_helpers.novelty(customer_ids, interaction_ids, action_names)
    monkeypatch.setattr(ml_helpers, "_NATIVE_GROUP_MIN_ROWS", 10_000)

    assert native_personalization == pytest.approx(
        ml_helpers.personalization(customer_ids, action_names), abs=1e-12
    )
    assert native_novelty == pytest.approx(
        ml_helpers.novelty(customer_ids, interaction_ids, action_names), abs=1e-10
    )
