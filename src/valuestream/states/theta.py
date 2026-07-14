"""Apache DataSketches theta helpers.

Theta sketches are the set-algebra companion to HLL: they can estimate
unions, intersections, and A-not-B differences from serialized state blobs.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from datasketches import (  # type: ignore[import-untyped]
    compact_theta_sketch,
    theta_a_not_b,
    theta_intersection,
    theta_union,
    update_theta_sketch,
)


def build(values: Iterable[Any], *, lg_k: int = 12) -> bytes:
    """Return a compact theta sketch for non-null ``values``."""
    sketch = update_theta_sketch(lg_k)
    for value in values:
        if value is not None:
            sketch.update(str(value))
    return bytes(sketch.compact().serialize())


def merge(sketches: Iterable[bytes | bytearray | memoryview | None], *, lg_k: int = 12) -> bytes:
    """Union serialized theta sketches and return a compact serialized sketch."""
    union = theta_union(lg_k)
    for payload in sketches:
        if payload:
            union.update(_deserialize(payload))
    return bytes(union.get_result().serialize())


def intersect(
    sketches: Iterable[bytes | bytearray | memoryview | None],
    *,
    lg_k: int = 12,
) -> bytes:
    """Intersect serialized theta sketches and return a compact serialized sketch."""
    intersection = theta_intersection()
    updated = False
    for payload in sketches:
        if payload:
            intersection.update(_deserialize(payload))
            updated = True
    if not updated:
        return build([], lg_k=lg_k)
    return bytes(intersection.get_result().serialize())


def a_not_b(
    left: bytes | bytearray | memoryview | None,
    right: bytes | bytearray | memoryview | None,
    *,
    lg_k: int = 12,
) -> bytes:
    """Return the theta sketch for ``left`` minus ``right``."""
    left_sketch = _deserialize(left) if left else _deserialize(build([], lg_k=lg_k))
    right_sketch = _deserialize(right) if right else _deserialize(build([], lg_k=lg_k))
    return bytes(theta_a_not_b().compute(left_sketch, right_sketch).serialize())


def estimate(payload: bytes | bytearray | memoryview | None) -> float:
    """Return the theta cardinality estimate for ``payload``."""
    if not payload:
        return 0.0
    return float(_deserialize(payload).get_estimate())


def _deserialize(payload: bytes | bytearray | memoryview) -> compact_theta_sketch:
    return compact_theta_sketch.deserialize(bytes(payload))


__all__ = ["a_not_b", "build", "estimate", "intersect", "merge"]
