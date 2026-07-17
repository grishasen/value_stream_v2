"""Apache DataSketches frequent-string helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, TypedDict

from datasketches import (  # type: ignore[import-untyped]
    frequent_items_error_type,
    frequent_strings_sketch,
)


class FrequentItem(TypedDict):
    """One frequent-items estimate."""

    item: str
    estimate: int
    lower_bound: int
    upper_bound: int


def build(values: Iterable[Any], *, lg_max_map_size: int = 10) -> bytes:
    """Return a serialized frequent-string sketch for non-null ``values``."""
    sketch = frequent_strings_sketch(lg_max_map_size)
    for value in values:
        if value is not None:
            sketch.update(str(value))
    return bytes(sketch.serialize())


def build_strings(values: Iterable[str], *, lg_max_map_size: int = 10) -> bytes:
    """Build top-K state from the original stream already normalized to strings."""

    sketch = frequent_strings_sketch(lg_max_map_size)
    for value in values:
        sketch.update(value)
    return bytes(sketch.serialize())


def merge(
    sketches: Iterable[bytes | bytearray | memoryview | None],
    *,
    lg_max_map_size: int = 10,
) -> bytes:
    """Merge frequent-string sketches and return a serialized sketch."""
    merged = frequent_strings_sketch(lg_max_map_size)
    for payload in sketches:
        if payload:
            merged.merge(frequent_strings_sketch.deserialize(bytes(payload)))
    return bytes(merged.serialize())


def frequent_items(
    payload: bytes | bytearray | memoryview | None,
    *,
    error_type: Literal["NO_FALSE_POSITIVES", "NO_FALSE_NEGATIVES"] = "NO_FALSE_POSITIVES",
) -> list[FrequentItem]:
    """Return frequent item estimates from a serialized sketch."""
    if not payload:
        return []
    sketch = frequent_strings_sketch.deserialize(bytes(payload))
    mode = getattr(frequent_items_error_type, error_type)
    return [
        {
            "item": str(item),
            "estimate": int(estimate),
            "lower_bound": int(lower_bound),
            "upper_bound": int(upper_bound),
        }
        for item, estimate, lower_bound, upper_bound in sketch.get_frequent_items(mode)
    ]


def weight(payload: bytes | bytearray | memoryview | None) -> int:
    """Return the total update weight represented by ``payload``."""
    if not payload:
        return 0
    return int(frequent_strings_sketch.deserialize(bytes(payload)).total_weight)


__all__ = ["FrequentItem", "build", "build_strings", "frequent_items", "merge", "weight"]
