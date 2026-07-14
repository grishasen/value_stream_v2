"""Apache DataSketches CPC helpers.

The rest of the code treats CPC state as opaque ``bytes``. These helpers are
the only place that knows how to build, merge, and estimate sketches.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from datasketches import cpc_sketch, cpc_union  # type: ignore[import-untyped]


def build(values: Iterable[Any], *, lg_k: int = 11) -> bytes:
    """Return a serialized CPC sketch for non-null ``values``."""
    sketch = cpc_sketch(lg_k)
    for value in values:
        if value is not None:
            sketch.update(str(value))
    return bytes(sketch.serialize())


def merge(sketches: Iterable[bytes | bytearray | memoryview | None], *, lg_k: int = 11) -> bytes:
    """Union serialized CPC sketches and return a serialized sketch."""
    union = cpc_union(lg_k)
    for payload in sketches:
        if payload:
            union.update(cpc_sketch.deserialize(bytes(payload)))
    return bytes(union.get_result().serialize())


def estimate(payload: bytes | bytearray | memoryview | None) -> float:
    """Return the CPC cardinality estimate for ``payload``."""
    if not payload:
        return 0.0
    return float(cpc_sketch.deserialize(bytes(payload)).get_estimate())


def bounds(
    payload: bytes | bytearray | memoryview | None,
    *,
    kappa: int = 2,
) -> tuple[float, float]:
    """Return approximate lower and upper confidence bounds."""
    if not payload:
        return 0.0, 0.0
    sketch = cpc_sketch.deserialize(bytes(payload))
    return float(sketch.get_lower_bound(kappa)), float(sketch.get_upper_bound(kappa))


__all__ = ["bounds", "build", "estimate", "merge"]
