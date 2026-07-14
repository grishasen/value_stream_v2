"""Regenerate ``schemas/expr.json`` from the AST model.

The on-disk schema is the single source of truth for external validators
(catalog YAML loaders, the Builder UI, third-party SDKs). It is checked in
so reviewers can diff schema changes alongside AST changes. A test in
``tests/unit/test_expr_parser.py`` enforces parity between the file and the
model — if you edit ``ast.py``, regenerate via this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from valuestream.expr.ast import Expr

_TITLE = "Value Stream Expression AST"
_SCHEMA_URL = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_ID = "https://valuestream.dev/schemas/expr.json"


def generate_schema() -> dict[str, Any]:
    """Return the JSON Schema for ``Expr`` as a dict."""
    schema = TypeAdapter(Expr).json_schema(mode="validation")
    schema["$schema"] = _SCHEMA_URL
    schema["$id"] = _SCHEMA_ID
    schema["title"] = _TITLE
    return schema


def write_schema(path: Path | str) -> None:
    """Write the schema to ``path`` as pretty-printed, key-sorted JSON."""
    schema = generate_schema()
    Path(path).write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    # Invoked by hand to regenerate schemas/expr.json after AST edits.
    repo_root = Path(__file__).resolve().parents[3]
    write_schema(repo_root / "schemas" / "expr.json")
    print(f"wrote {repo_root / 'schemas' / 'expr.json'}")
