"""Load a workspace's catalog YAML files into typed :class:`~valuestream.config.model.Catalog`.

Reads the four files under ``<workspace>/catalog/``, validates each
against its JSON Schema (generated from the Pydantic model), then
materializes the typed model. Validation errors carry the file name and
the field path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter, ValidationError
from yaml.constructor import ConstructorError

from valuestream.config import model
from valuestream.utils.timer import timed

_CATALOG_FILES: tuple[tuple[str, str, type], ...] = (
    ("pipelines.yaml", "pipelines", model.Pipelines),
    ("processors.yaml", "processors", model.Processors),
    ("metrics.yaml", "metrics", model.Metrics),
    ("dashboards.yaml", "dashboards", model.Dashboards),
)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class CatalogLoadError(ValueError):
    """Raised when a workspace catalog fails to load.

    ``file`` is the YAML path that triggered the error (may be ``None``
    for cross-file errors). ``issues`` mirrors Pydantic's structured
    validation output.
    """

    def __init__(
        self, message: str, *, file: Path | None = None, issues: list[Any] | None = None
    ) -> None:
        self.file = file
        self.issues = issues or []
        super().__init__(message)


@timed
def load(workspace_dir: str | Path) -> model.Catalog:
    """Load and validate ``<workspace_dir>/catalog/*.yaml``.

    Returns the typed :class:`~valuestream.config.model.Catalog`. Raises
    :class:`CatalogLoadError` if any file is missing, malformed, or
    fails validation.
    """
    ws = Path(workspace_dir)
    catalog_dir = ws / "catalog"
    if not catalog_dir.is_dir():
        raise CatalogLoadError(
            f"workspace catalog directory not found: {catalog_dir}", file=catalog_dir
        )

    parts: dict[str, Any] = {}
    for filename, key, model_cls in _CATALOG_FILES:
        path = catalog_dir / filename
        if not path.is_file():
            raise CatalogLoadError(f"missing catalog file: {path}", file=path)
        parts[key] = _load_one(path, model_cls)

    try:
        return model.Catalog.model_validate(parts)
    except ValidationError as exc:
        raise CatalogLoadError(
            "catalog cross-file validation failed:\n" + _format_errors(exc),
            issues=list(exc.errors()),
        ) from exc


def _load_one(path: Path, model_cls: type) -> Any:
    """Load one YAML file, parse, and validate against its model."""
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise CatalogLoadError(f"YAML parse error in {path.name}: {exc}", file=path) from exc

    if data is None:
        raise CatalogLoadError(f"empty catalog file: {path.name}", file=path)
    if not isinstance(data, dict):
        raise CatalogLoadError(
            f"top-level YAML in {path.name} must be a mapping, got {type(data).__name__}",
            file=path,
        )

    try:
        return TypeAdapter(model_cls).validate_python(data)
    except ValidationError as exc:
        raise CatalogLoadError(
            f"{path.name}: validation failed\n" + _format_errors(exc),
            file=path,
            issues=list(exc.errors()),
        ) from exc


def _format_errors(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"  {loc}: {err['msg']} (type={err['type']})")
    return "\n".join(parts) if parts else "  <no details>"


__all__ = ["CatalogLoadError", "load"]
