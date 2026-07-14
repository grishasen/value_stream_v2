"""Focused tests for ML helper metrics."""

from __future__ import annotations

from math import log2

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
