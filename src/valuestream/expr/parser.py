"""Parse YAML/JSON dicts into typed :class:`~valuestream.expr.ast.Expr` instances.

The parser is a thin wrapper around a :class:`pydantic.TypeAdapter` — Pydantic
does the actual structural validation against the discriminated union defined
in :mod:`valuestream.expr.ast`. This module's job is to:

* expose a ``parse`` function that accepts a JSON-like Python value,
* accept YAML text via ``parse_yaml`` (using PyYAML's ``safe_load``),
* round-trip back to a dict via ``to_dict`` for canonical hashing, and
* surface validation errors as :class:`ParseError` with the AST path.
"""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import TypeAdapter, ValidationError

from valuestream.expr.ast import Expr

_adapter: TypeAdapter[Expr] = TypeAdapter(Expr)


class ParseError(ValueError):
    """Raised when a value does not validate against the AST grammar.

    The original :class:`pydantic.ValidationError` is preserved on
    ``__cause__`` so callers can inspect detailed per-error info.
    """


def parse(value: Any) -> Expr:
    """Validate ``value`` against the AST grammar and return a typed node."""
    try:
        return _adapter.validate_python(value)
    except ValidationError as exc:
        raise ParseError(_format_error(exc)) from exc


def parse_yaml(text: str) -> Expr:
    """Load YAML text and parse it. Equivalent to ``parse(yaml.safe_load(text))``."""
    loaded = yaml.safe_load(text)
    return parse(loaded)


def to_dict(node: Expr) -> dict[str, Any]:
    """Serialize an AST node back to a plain dict.

    Uses serialization aliases (so ``else_`` becomes ``else``), keeps
    ``None`` values (so ``{"lit": None}`` round-trips cleanly), and
    recursively serializes children. Every AST node serializes to a dict —
    atoms via their single field (``col``/``lit``/``param``), op nodes via
    the ``op`` discriminator.

    The canonical-hashing path (:mod:`valuestream.config.canonical`)
    separately drops ``None`` values for normalization; ``to_dict`` itself
    is identity-preserving.
    """
    dumped: dict[str, Any] = _adapter.dump_python(node, by_alias=True, exclude_unset=True)
    return dumped


def _format_error(exc: ValidationError) -> str:
    """Render a Pydantic ValidationError as a readable, path-tagged message."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"  {loc}: {err['msg']} (type={err['type']})")
    body = "\n".join(parts) if parts else "  <no details>"
    return f"expression failed validation:\n{body}"


__all__ = ["ParseError", "parse", "parse_yaml", "to_dict"]
