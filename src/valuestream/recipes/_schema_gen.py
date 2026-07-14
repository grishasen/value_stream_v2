"""Regenerate the checked-in JSON Schema for KPI recipe libraries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from valuestream.recipes.kpi import KpiRecipeLibrary

_SCHEMA_URL = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_ID = "https://valuestream.dev/schemas/kpi-recipes.json"


def generate_schema() -> dict[str, Any]:
    """Return the validation schema for a versioned KPI recipe library."""

    schema = TypeAdapter(KpiRecipeLibrary).json_schema(mode="validation")
    schema["$schema"] = _SCHEMA_URL
    schema["$id"] = _SCHEMA_ID
    schema["title"] = "Value Stream KPI Recipe Library"
    return schema


def write_schema(path: Path | str) -> None:
    """Write the generated schema to ``path``."""

    Path(path).write_text(
        json.dumps(generate_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[3]
    target = repo_root / "schemas" / "kpi-recipes.json"
    write_schema(target)
    print(f"wrote {target}")
