"""Identifier helpers: pipeline-run UUIDs and identifier validation."""

from __future__ import annotations

import re
import uuid

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_WORKSPACE_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


def new_pipeline_run_id() -> str:
    """Return a fresh UUIDv4 pipeline-run id, hex-encoded with dashes."""
    return str(uuid.uuid4())


def is_ident(value: str) -> bool:
    """``True`` if ``value`` matches the AST ``ident`` production."""
    return bool(_IDENT_RE.match(value))


def is_snake_case(value: str) -> bool:
    """``True`` if ``value`` is snake_case (used for source/processor ids)."""
    return bool(_SNAKE_RE.match(value))


def is_workspace_id(value: str) -> bool:
    """``True`` if ``value`` is a valid workspace id."""
    return bool(_WORKSPACE_RE.match(value))
