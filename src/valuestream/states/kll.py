"""Apache DataSketches KLL helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from datasketches import kll_floats_sketch  # type: ignore[import-untyped]


def build(values: Iterable[Any], *, k: int = 200) -> bytes:
    """Return a serialized KLL sketch for non-null numeric values."""
    sketch = kll_floats_sketch(k)
    for value in values:
        if value is not None:
            sketch.update(float(value))
    return bytes(sketch.serialize())


def merge(sketches: Iterable[bytes | bytearray | memoryview | None], *, k: int = 200) -> bytes:
    """Merge serialized KLL sketches and return a serialized sketch."""
    merged = kll_floats_sketch(k)
    for payload in sketches:
        if payload:
            merged.merge(kll_floats_sketch.deserialize(bytes(payload)))
    return bytes(merged.serialize())


def quantile(payload: bytes | bytearray | memoryview | None, q: float) -> float:
    """Return quantile ``q`` from ``payload``; empty sketches return 0.0."""
    if not payload:
        return 0.0
    sketch = kll_floats_sketch.deserialize(bytes(payload))
    if sketch.n == 0:
        return 0.0
    return float(sketch.get_quantile(q))


__all__ = ["build", "merge", "quantile"]
