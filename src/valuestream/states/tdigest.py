"""Apache DataSketches t-digest helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from datasketches import tdigest_double  # type: ignore[import-untyped]


def build(values: Iterable[Any], *, k: int = 500) -> bytes:
    """Return a serialized t-digest for non-null numeric values."""
    sketch = tdigest_double(k)
    for value in values:
        if value is not None:
            sketch.update(float(value))
    return bytes(sketch.serialize())


def merge(sketches: Iterable[bytes | bytearray | memoryview | None], *, k: int = 500) -> bytes:
    """Merge serialized t-digests and return a serialized digest."""
    merged = tdigest_double(k)
    for payload in sketches:
        if payload:
            merged.merge(tdigest_double.deserialize(bytes(payload)))
    return bytes(merged.serialize())


def quantile(payload: bytes | bytearray | memoryview | None, q: float) -> float:
    """Return quantile ``q`` from ``payload``; empty digests return 0.0."""
    if not payload:
        return 0.0
    sketch = tdigest_double.deserialize(bytes(payload))
    if sketch.get_total_weight() == 0:
        return 0.0
    return float(sketch.get_quantile(q))


def cdf(payload: bytes | bytearray | memoryview | None, x: float) -> float:
    """Return CDF(x) from ``payload``; empty digests return 0.0."""
    if not payload:
        return 0.0
    sketch = tdigest_double.deserialize(bytes(payload))
    if sketch.get_total_weight() == 0:
        return 0.0
    return float(sketch.get_cdf([x])[0])


def weight(payload: bytes | bytearray | memoryview | None) -> float:
    """Return total digest weight."""
    if not payload:
        return 0.0
    return float(tdigest_double.deserialize(bytes(payload)).get_total_weight())


def deserialize(payload: bytes | bytearray | memoryview) -> tdigest_double:
    """Deserialize a t-digest payload."""
    return tdigest_double.deserialize(bytes(payload))


__all__ = ["build", "cdf", "deserialize", "merge", "quantile", "weight"]
