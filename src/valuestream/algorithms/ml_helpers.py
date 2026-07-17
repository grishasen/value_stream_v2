"""Small Phase 2 ML helper metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sized
from itertools import islice
from math import fsum, log2, sqrt
from typing import Any, Protocol, cast

import polars as pl

_PERSONALIZATION_FULL_LIMIT = 50_000
_PERSONALIZATION_SAMPLE_LIMIT = 100_000
_PERSONALIZATION_SAMPLE_SIZE = 50_000
_NOVELTY_FULL_LIMIT = 50_000
_NOVELTY_SAMPLE_LIMIT = 100_000
_NOVELTY_SAMPLE_SIZE = 50_000
_NATIVE_GROUP_MIN_ROWS = 256


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
    if _native_personalization_is_worthwhile(customer_ids, action_names, start, stop):
        native = _personalization_native(customer_ids, action_names, start, stop)
        if native is not None:
            return native
    return _personalization_scalar(customer_ids, action_names, start, stop)


def _personalization_scalar(
    customer_ids: _SizedIterable,
    action_names: _SizedIterable,
    start: int,
    stop: int,
) -> float:
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


def _personalization_native(
    customer_ids: _SizedIterable,
    action_names: _SizedIterable,
    start: int,
    stop: int,
) -> float | None:
    frame = _sampled_frame(
        ("customer", customer_ids),
        ("action", action_names),
        start=start,
        stop=stop,
    )
    if frame.schema["action"] not in {pl.String, pl.Categorical, pl.Enum}:
        return None
    counts = (
        frame.drop_nulls(["customer", "action"])
        .with_columns(pl.col("action").cast(pl.String))
        .group_by("customer", "action")
        .len(name="count")
    )
    vector_count = counts.get_column("customer").n_unique()
    if vector_count <= 1:
        return 0.0
    action_sums = (
        counts.with_columns(
            pl.col("count").cast(pl.Float64).pow(2).sum().over("customer").sqrt().alias("__norm")
        )
        .with_columns((pl.col("count").cast(pl.Float64) / pl.col("__norm")).alias("__normalized"))
        .group_by("action")
        .agg(pl.col("__normalized").sum().alias("__sum"))
    )
    total_similarity = float(action_sums.select(pl.col("__sum").pow(2).sum()).item())
    off_diagonal_similarity = total_similarity - vector_count
    avg_similarity = off_diagonal_similarity / (vector_count * (vector_count - 1))
    return _stable_float(1.0 - max(0.0, min(1.0, avg_similarity)))


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
    if (
        stop - start >= _NATIVE_GROUP_MIN_ROWS
        and isinstance(customer_ids, pl.Series)
        and isinstance(interaction_ids, pl.Series)
        and isinstance(action_names, pl.Series)
    ):
        native = _novelty_native(customer_ids, interaction_ids, action_names, start, stop)
        if native is not None:
            return native
    return _novelty_scalar(customer_ids, interaction_ids, action_names, start, stop)


def _novelty_scalar(
    customer_ids: _SizedIterable,
    interaction_ids: _SizedIterable,
    action_names: _SizedIterable,
    start: int,
    stop: int,
) -> float:
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


def _novelty_native(
    customer_ids: _SizedIterable,
    interaction_ids: _SizedIterable,
    action_names: _SizedIterable,
    start: int,
    stop: int,
) -> float | None:
    frame = _sampled_frame(
        ("customer", customer_ids),
        ("interaction", interaction_ids),
        ("action", action_names),
        start=start,
        stop=stop,
    )
    if frame.schema["action"] not in {pl.String, pl.Categorical, pl.Enum}:
        return None
    frame = frame.with_columns(pl.col("action").cast(pl.String))
    unique_users = frame.get_column("customer").drop_nulls().n_unique()
    if unique_users == 0:
        return 0.0
    action_counts = frame.filter(pl.col("action").is_not_null()).group_by("action").len()
    count = pl.col("len").cast(pl.Float64)
    total_self_info = float(
        action_counts.select((count * -((count / unique_users).log(base=2) + 1e-10)).sum()).item()
    )
    interaction_counts = frame.drop_nulls(["interaction", "action"]).group_by("interaction").len()
    max_rec_length = interaction_counts.get_column("len").max()
    if not max_rec_length:
        return 0.0
    return _stable_float(total_self_info / (unique_users * int(max_rec_length)))


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


def _sampled_frame(
    *columns: tuple[str, _SizedIterable],
    start: int,
    stop: int,
) -> pl.DataFrame:
    length = max(0, stop - start)
    series: list[pl.Series] = []
    for name, column in columns:
        if isinstance(column, pl.Series):
            series.append(column.slice(start, length).rename(name))
        else:
            series.append(pl.Series(name, list(_sampled_values(column, start, stop)), strict=False))
    return pl.DataFrame(series)


def _native_personalization_is_worthwhile(
    customer_ids: _SizedIterable,
    action_names: _SizedIterable,
    start: int,
    stop: int,
) -> bool:
    length = stop - start
    if (
        length < _NATIVE_GROUP_MIN_ROWS
        or not isinstance(customer_ids, pl.Series)
        or not isinstance(action_names, pl.Series)
    ):
        return False
    # The scalar Counter path is faster for a small, repeatedly updated user
    # population. Native grouped reductions win once per-customer dictionaries
    # become numerous enough to dominate Python allocation and hashing.
    return customer_ids.slice(start, length).drop_nulls().n_unique() * 8 >= length


def _stable_float(value: float) -> float:
    """Normalize native parallel reductions to the output-contract precision."""
    return float(f"{value:.12g}")


__all__ = ["novelty", "personalization"]
