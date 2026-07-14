"""Outcome-value helpers shared by outcome-based processors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class Outcome:
    """Resolved positive/negative outcome configuration for a processor."""

    column: str
    positive_values: list[Any]
    negative_values: list[Any]


def parse_outcome(extra: dict[str, Any]) -> Outcome:
    """Return the configured ``outcome`` block, falling back to demo defaults."""
    raw = extra.get("outcome")
    if isinstance(raw, dict):
        return Outcome(
            column=str(raw.get("column", "Outcome")),
            positive_values=list(raw.get("positive_values", ["Clicked", "Conversion"])),
            negative_values=list(raw.get("negative_values", ["Impression", "Pending"])),
        )
    return Outcome(
        column="Outcome",
        positive_values=["Clicked", "Conversion"],
        negative_values=["Impression", "Pending"],
    )


def compatible_values(values: list[Any], dtype: pl.DataType) -> list[Any]:
    """Return configured values that can be compared with a Polars column dtype."""
    if dtype == pl.String:
        return [str(value) for value in values]
    if dtype == pl.Boolean:
        return [_coerce_bool(value) for value in values if _coerce_bool(value) is not None]
    if dtype.is_integer():
        return [_coerce_int(value) for value in values if _coerce_int(value) is not None]
    if dtype.is_float():
        return [_coerce_float(value) for value in values if _coerce_float(value) is not None]
    return values


def is_in_values(column: str, values: list[Any]) -> pl.Expr:
    """Return a membership expression that is false for an empty value list."""
    if not values:
        return pl.lit(False)
    return pl.col(column).is_in(values)


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        folded = value.strip().casefold()
        if folded in {"true", "1", "yes"}:
            return True
        if folded in {"false", "0", "no"}:
            return False
    return None


def _coerce_int(value: Any) -> int | None:
    result: int | None = None
    if isinstance(value, bool):
        result = int(value)
    elif isinstance(value, int):
        result = value
    elif isinstance(value, float) and value.is_integer():
        result = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if text:
            try:
                number = float(text)
            except ValueError:
                number = None
            if number is not None and number.is_integer():
                result = int(number)
    return result


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


__all__ = ["Outcome", "compatible_values", "is_in_values", "parse_outcome"]
