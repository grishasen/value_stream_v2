"""Column naming helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable

_CAPITALIZE_END_WORDS = [
    "ID",
    "Key",
    "Name",
    "Treatment",
    "Count",
    "Category",
    "Class",
    "Time",
    "DateTime",
    "UpdateTime",
    "Version",
    "Rate",
    "Ratio",
    "Negatives",
    "Positives",
    "Threshold",
    "Error",
    "Importance",
    "Type",
    "Percentage",
    "Index",
    "Symbol",
    "ResponseCount",
    "ConfigurationName",
    "Configuration",
]


def dedupe_strings(values: Iterable[str]) -> list[str]:
    """Return non-empty values with duplicates removed, preserving order."""
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def capitalize_fields(fields: str | Iterable[str]) -> list[str]:
    """Apply legacy Pega-aware column capitalization."""
    original = [fields] if isinstance(fields, str) else list(fields)

    renamed = [re.sub(r"^p([xyz])", "", field) for field in original]
    seen = set(original)
    for index, item in enumerate(renamed):
        if item in seen:
            renamed[index] = original[index]

    for word in _CAPITALIZE_END_WORDS:
        renamed = [re.sub(word + r"\b", word, field, flags=re.I) for field in renamed]
        renamed = [field[:1].upper() + field[1:] for field in renamed]
    return renamed


__all__ = ["capitalize_fields", "dedupe_strings"]
