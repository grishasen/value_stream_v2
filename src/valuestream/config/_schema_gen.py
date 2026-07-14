"""Regenerate JSON schemas under ``schemas/`` from the config models.

One schema per top-level YAML file plus a ``catalog.json`` for the
assembled :class:`~valuestream.config.model.Catalog`. The schemas are checked
in so reviewers can diff schema changes alongside model changes; a
parity test in ``tests/unit/test_config_loader.py`` enforces that the
files on disk match the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from valuestream.config import model

_SCHEMA_URL = "https://json-schema.org/draft/2020-12/schema"

_TARGETS: tuple[tuple[str, str, type], ...] = (
    ("pipelines.json", "Value Stream Pipelines", model.Pipelines),
    ("processors.json", "Value Stream Processors", model.Processors),
    ("metrics.json", "Value Stream Metrics", model.Metrics),
    ("dashboards.json", "Value Stream Dashboards", model.Dashboards),
    ("catalog.json", "Value Stream Catalog", model.Catalog),
)


def generate_all() -> dict[str, dict[str, Any]]:
    """Return ``{filename: schema_dict}`` for every catalog model."""
    out: dict[str, dict[str, Any]] = {}
    for filename, title, model_cls in _TARGETS:
        schema = TypeAdapter(model_cls).json_schema(mode="validation")
        schema["$schema"] = _SCHEMA_URL
        schema["$id"] = f"https://valuestream.dev/schemas/{filename}"
        schema["title"] = title
        out[filename] = schema
    return out


def write_all(schemas_dir: Path | str) -> None:
    """Write each generated schema to ``<schemas_dir>/<filename>``."""
    schemas_dir = Path(schemas_dir)
    schemas_dir.mkdir(parents=True, exist_ok=True)
    for filename, schema in generate_all().items():
        (schemas_dir / filename).write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[3]
    write_all(repo_root / "schemas")
    print(f"wrote {len(_TARGETS)} schemas to {repo_root / 'schemas'}")
