"""Static column-reference discovery for the expression AST."""

from __future__ import annotations

import ast as py_ast
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel


def column_references(value: BaseModel | Mapping[str, Any] | None) -> frozenset[str]:
    """Return literal column names referenced by a typed or serialized expression.

    The closed expression AST uses ``col`` and ``column`` fields. Advanced
    ``polars`` nodes are also inspected for literal ``pl.col(...)`` calls so
    configuration guards apply consistently to both expression forms.
    """

    payload: Any = (
        value.model_dump(mode="python", by_alias=True) if isinstance(value, BaseModel) else value
    )
    return frozenset(_mapping_references(payload))


def _mapping_references(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        columns = {
            str(item).strip()
            for key, item in value.items()
            if key in {"col", "column"} and isinstance(item, str) and str(item).strip()
        }
        polars_text = value.get("polars")
        if isinstance(polars_text, str):
            columns.update(_polars_column_references(polars_text))
        for item in value.values():
            columns.update(_mapping_references(item))
        return columns
    if isinstance(value, list | tuple):
        nested_columns: set[str] = set()
        for item in value:
            nested_columns.update(_mapping_references(item))
        return nested_columns
    return set()


def _polars_column_references(text: str) -> set[str]:
    try:
        parsed = py_ast.parse(text, mode="eval")
    except SyntaxError:
        # Expression translation owns syntax diagnostics. Reference discovery
        # remains best-effort so it does not replace the authoritative error.
        return set()
    columns: set[str] = set()
    for node in py_ast.walk(parsed):
        if not isinstance(node, py_ast.Call) or not isinstance(node.func, py_ast.Attribute):
            continue
        if (
            node.func.attr != "col"
            or not isinstance(node.func.value, py_ast.Name)
            or node.func.value.id != "pl"
            or not node.args
        ):
            continue
        first = node.args[0]
        values = first.elts if isinstance(first, py_ast.List | py_ast.Tuple) else [first]
        columns.update(
            str(value.value).strip()
            for value in values
            if isinstance(value, py_ast.Constant)
            and isinstance(value.value, str)
            and str(value.value).strip()
        )
    return columns


__all__ = ["column_references"]
