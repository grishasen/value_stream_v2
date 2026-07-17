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
    """Return a contiguous writable array, or ``None`` for a tiny input.

    Polars group callbacks provide ``Series`` objects, whose ``to_numpy`` path
    avoids one Python call per value.  Tiny inputs retain the scalar update
    path because array setup costs more than the native bulk call saves.
    """

    if isinstance(values, Sized) and len(values) < _BULK_UPDATE_MIN_VALUES:
        return None
    if isinstance(values, np.ndarray):
        raw = values
    elif callable(to_numpy := getattr(values, "to_numpy", None)):
        raw = to_numpy()
    else:
        raw = np.fromiter(
            (float(value) for value in values if value is not None),
            dtype=dtype,
        )
    array = np.asarray(raw, dtype=dtype).reshape(-1)
    return np.require(array, dtype=dtype, requirements=["C", "W"])


__all__ = ["bulk_numeric_array"]
