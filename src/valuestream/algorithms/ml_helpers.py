"""Small Phase 2 ML helper metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sized
from itertools import islice
from math import fsum, log2, sqrt
from typing import Any, Protocol, cast

_PERSONALIZATION_FULL_LIMIT = 50_000
_PERSONALIZATION_SAMPLE_LIMIT = 100_000
_PERSONALIZATION_SAMPLE_SIZE = 50_000
_NOVELTY_FULL_LIMIT = 50_000
_NOVELTY_SAMPLE_LIMIT = 100_000
_NOVELTY_SAMPLE_SIZE = 50_000


class _SizedIterable(Sized, Iterable[Any], Protocol):
    pass


def personalization(customer_ids: _SizedIterable, action_names: _SizedIterable) -> float:
    """Return recommendation personalization without building the user-pair matrix."""
    start, stop = _sample_window(
        customer_ids,
        action_names,
        full_limit=_PERSONALIZATION_FULL_LIMIT,
        sample_limit=_PERSONALIZATION_SAMPLE_LIMIT,
        sample_size=_PERSONALIZATION_SAMPLE_SIZE,
    )
    per_customer: dict[Any, Counter[str]] = defaultdict(Counter)
    for customer_id, action_name in zip(
        _sampled_values(customer_ids, start, stop),
        _sampled_values(action_names, start, stop),
        strict=False,
    ):
        if customer_id is not None and action_name is not None:
            per_customer[customer_id][str(action_name)] += 1

    normalized_action_sums: dict[str, float] = defaultdict(float)
    vector_count = 0
    for action_counts in per_customer.values():
        norm = sqrt(fsum(count * count for count in action_counts.values()))
        if norm == 0:
            continue
        vector_count += 1
        inv_norm = 1.0 / norm
        for action_name, count in action_counts.items():
            normalized_action_sums[action_name] += count * inv_norm

    if vector_count <= 1:
        return 0.0

    total_similarity = fsum(value * value for value in normalized_action_sums.values())
    off_diagonal_similarity = total_similarity - vector_count
    avg_similarity = off_diagonal_similarity / (vector_count * (vector_count - 1))
    avg_similarity = max(0.0, min(1.0, avg_similarity))
    return 1.0 - avg_similarity


def novelty(
    customer_ids: _SizedIterable, interaction_ids: _SizedIterable, action_names: _SizedIterable
) -> float:
    """Return information-theoretic novelty for customer/interaction/action rows."""
    start, stop = _sample_window(
        customer_ids,
        interaction_ids,
        action_names,
        full_limit=_NOVELTY_FULL_LIMIT,
        sample_limit=_NOVELTY_SAMPLE_LIMIT,
        sample_size=_NOVELTY_SAMPLE_SIZE,
    )
    users: set[Any] = set()
    counts: Counter[str] = Counter()
    actions_per_interaction: dict[Any, int] = defaultdict(int)
    for customer_id, interaction_id, action_name in zip(
        _sampled_values(customer_ids, start, stop),
        _sampled_values(interaction_ids, start, stop),
        _sampled_values(action_names, start, stop),
        strict=False,
    ):
        if customer_id is not None:
            users.add(customer_id)
        if action_name is not None:
            counts[str(action_name)] += 1
            if interaction_id is not None:
                actions_per_interaction[interaction_id] += 1

    unique_users = len(users)
    if unique_users == 0:
        return 0.0
    total_self_info = fsum(
        count * -(log2(count / unique_users) + 1e-10) for count in counts.values()
    )
    max_rec_length = max(actions_per_interaction.values(), default=0)
    if max_rec_length == 0:
        return 0.0
    return total_self_info / (unique_users * max_rec_length)


def _sample_window(
    *columns: _SizedIterable, full_limit: int, sample_limit: int, sample_size: int
) -> tuple[int, int]:
    if not columns:
        return 0, 0
    height = min(len(column) for column in columns)
    if height < full_limit:
        return 0, height
    start = round(height / 2)
    if height < sample_limit:
        return start, height
    return start, min(height, start + sample_size)


def _sampled_values(column: _SizedIterable, start: int, stop: int) -> Iterable[Any]:
    if start == 0 and stop == len(column):
        return column
    try:
        return cast(Iterable[Any], column[start:stop])  # type: ignore[index]
    except TypeError:
        return islice(column, start, stop)


__all__ = ["novelty", "personalization"]
