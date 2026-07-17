"""Shared numeric-array preparation for native DataSketches bulk updates."""

from __future__ import annotations

from collections.abc import Iterable, Sized
from typing import Any

import numpy as np

_BULK_UPDATE_MIN_VALUES = 32


def bulk_numeric_array(
    values: Iterable[Any],
    *,
    dtype: Any,
) -> np.ndarray[Any, Any] | None:
    """Return null-free writable numeric storage, or ``None`` for a tiny input.

    Polars group callbacks provide ``Series`` objects, whose ``to_numpy`` path
    avoids one Python call per value.  Tiny inputs retain the scalar update
    path because array setup costs more than the native bulk call saves.

    Polars numeric ``Series.to_numpy`` represents nulls as NaN, so nulls are
    removed before conversion instead of being conflated with genuine NaN
    values. Genuine NaNs are deliberately preserved: DataSketches ignores
    them in both its scalar and bulk numeric update paths.
    """

    prepared = values
    if callable(drop_nulls := getattr(values, "drop_nulls", None)):
        prepared = drop_nulls()
    if isinstance(prepared, Sized) and len(prepared) < _BULK_UPDATE_MIN_VALUES:
        return None
    if isinstance(prepared, np.ndarray):
        raw = prepared
        if raw.dtype == object:
            raw = np.fromiter(
                (float(value) for value in raw.reshape(-1) if value is not None),
                dtype=dtype,
            )
    elif callable(to_numpy := getattr(prepared, "to_numpy", None)):
        raw = to_numpy()
    else:
        raw = np.fromiter(
            (float(value) for value in prepared if value is not None),
            dtype=dtype,
        )
    array = np.asarray(raw, dtype=dtype).reshape(-1)
    return np.require(array, dtype=dtype, requirements=["C", "W"])


__all__ = ["bulk_numeric_array"]
