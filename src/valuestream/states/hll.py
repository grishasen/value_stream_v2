"""Apache DataSketches HLL helpers.

The rest of the code treats HLL state as opaque ``bytes``. These helpers are
the only place that knows how to build, merge, and estimate sketches.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from datasketches import hll_sketch, hll_union, tgt_hll_type  # type: ignore[import-untyped]


def build(values: Iterable[Any], *, lg_k: int = 12) -> bytes:
    """Return a compact HLL_8 sketch for non-null ``values``."""
    sketch = hll_sketch(lg_k, tgt_hll_type.HLL_8)
    for value in values:
        if value is not None:
            sketch.update(str(value))
    return bytes(sketch.serialize_compact())


def merge(sketches: Iterable[bytes | bytearray | memoryview | None], *, lg_k: int = 12) -> bytes:
    """Union serialized HLL sketches and return a compact serialized sketch."""
    union = hll_union(lg_k)
    for payload in sketches:
        if payload:
            union.update(hll_sketch.deserialize(bytes(payload)))
    return bytes(union.get_result(tgt_hll_type.HLL_8).serialize_compact())


def estimate(payload: bytes | bytearray | memoryview | None) -> float:
    """Return the HLL cardinality estimate for ``payload``."""
    if not payload:
        return 0.0
    return float(hll_sketch.deserialize(bytes(payload)).get_estimate())


def bounds(
    payload: bytes | bytearray | memoryview | None,
    *,
    num_std_dev: int = 2,
) -> tuple[float, float]:
    """Return lower and upper cardinality bounds."""
    if not payload:
        return 0.0, 0.0
    sketch = hll_sketch.deserialize(bytes(payload))
    return (
        float(sketch.get_lower_bound(num_std_dev)),
        float(sketch.get_upper_bound(num_std_dev)),
    )


__all__ = ["bounds", "build", "estimate", "merge"]
