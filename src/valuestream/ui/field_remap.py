"""Field-name remapping helpers shared by the Config Builder and AI Studio.

When the ``rename_capitalize`` source transform is toggled, editor state that
references source fields (filters, defaults, calculated fields, raw expression
YAML) must be remapped between the raw and Pega-aware capitalized schemas.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Pure remapping helpers.
# ---------------------------------------------------------------------------


def remap_field_name(field: str, mapping: dict[str, str]) -> str:
    """Remap one field name."""
    return mapping.get(field, field)


def remap_field_list(fields: list[str], mapping: dict[str, str]) -> list[str]:
    """Remap a list of field names."""
    return [mapping.get(str(field), str(field)) for field in fields]


def remap_default_values(values: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Remap the keys of a default-values mapping."""
    return {mapping.get(str(field), str(field)): value for field, value in values.items()}


def remap_expression_fields(value: Any, mapping: dict[str, str]) -> Any:
    """Remap ``col``/``column`` references inside an expression AST."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"column", "col"} and isinstance(item, str):
                out[key] = mapping.get(item, item)
            else:
                out[key] = remap_expression_fields(item, mapping)
        return out
    if isinstance(value, list):
        return [remap_expression_fields(item, mapping) for item in value]
    return value


def remap_expression_text(text: str, mapping: dict[str, str], *, mode: str) -> str:
    """Remap field references inside Polars or AST-YAML expression text."""
    if mode == "Polars":
        out = text
        for old, new in mapping.items():
            out = out.replace(f'pl.col("{old}")', f'pl.col("{new}")')
            out = out.replace(f"pl.col('{old}')", f"pl.col('{new}')")
        return out
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    remapped = remap_expression_fields(parsed, mapping)
    return yaml.safe_dump(remapped, sort_keys=False).strip() if remapped is not None else text


def remap_calculation_row_values(
    rows: list[dict[str, Any]],
    mapping: dict[str, str],
) -> list[dict[str, Any]]:
    """Remap field references inside calculated-field editor rows."""
    remapped: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        left = str(item.get("Left", "") or "")
        if left in mapping:
            item["Left"] = mapping[left]
        right = str(item.get("Right", "") or "")
        if str(item.get("Right Kind", "Field") or "Field") == "Field" and right in mapping:
            item["Right"] = mapping[right]
        expression = str(item.get("Expression", "") or "")
        if expression:
            item["Expression"] = remap_expression_text(
                expression,
                mapping,
                mode=str(item.get("Mode", "") or ""),
            )
        remapped.append(item)
    return remapped


# ---------------------------------------------------------------------------
# Session-state remapping helpers.
# ---------------------------------------------------------------------------


def remap_state_field(key: str, mapping: dict[str, str]) -> None:
    """Remap a single field name stored in session state."""
    value = st.session_state.get(key)
    if isinstance(value, str) and value in mapping:
        st.session_state[key] = mapping[value]


def remap_state_field_list(key: str, mapping: dict[str, str]) -> None:
    """Remap a list of field names stored in session state."""
    value = st.session_state.get(key)
    if isinstance(value, list):
        st.session_state[key] = [mapping.get(str(item), str(item)) for item in value]


def remap_state_rows(key: str, mapping: dict[str, str], field_keys: tuple[str, ...]) -> None:
    """Remap named columns of editor rows stored in session state."""
    rows = st.session_state.get(key)
    if not isinstance(rows, list):
        return
    remapped: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for field_key in field_keys:
            value = str(item.get(field_key, "") or "")
            if value in mapping:
                item[field_key] = mapping[value]
        remapped.append(item)
    st.session_state[key] = remapped


def remap_state_calculation_rows(key: str, mapping: dict[str, str]) -> None:
    """Remap calculated-field editor rows stored in session state."""
    rows = st.session_state.get(key)
    if not isinstance(rows, list):
        return
    st.session_state[key] = remap_calculation_row_values(rows, mapping)


def remap_state_raw_expression(key: str, mapping: dict[str, str]) -> None:
    """Remap raw AST-YAML expression text stored in session state."""
    value = st.session_state.get(key)
    if isinstance(value, str) and value.strip():
        st.session_state[key] = remap_expression_text(value, mapping, mode="AST YAML")


__all__ = [
    "remap_calculation_row_values",
    "remap_default_values",
    "remap_expression_fields",
    "remap_expression_text",
    "remap_field_list",
    "remap_field_name",
    "remap_state_calculation_rows",
    "remap_state_field",
    "remap_state_field_list",
    "remap_state_raw_expression",
    "remap_state_rows",
]
