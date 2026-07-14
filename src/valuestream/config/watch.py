"""Mtime-based catalog reloading for long-lived servers.

The MCP server and the read-only API load the catalog once and keep it for
manifest and chart-intent validation. When an operator edits the catalog YAML
(for example through the Config Builder), those cached copies would otherwise
desync from what ``query_metric`` sees, since ``query_metric`` reloads on every
call. :class:`CatalogCache` closes that gap by reloading only when a catalog
file's modification time changes.
"""

from __future__ import annotations

from pathlib import Path

from valuestream.config.loader import load
from valuestream.config.model import Catalog

_CATALOG_FILES = ("pipelines.yaml", "processors.yaml", "metrics.yaml", "dashboards.yaml")
_AI_FILES = ("ai.yaml",)


class CatalogCache:
    """Return a catalog, reloading it when its source files change on disk."""

    def __init__(self, workspace_path: str | Path) -> None:
        self._workspace = Path(workspace_path)
        self._catalog: Catalog | None = None
        self._signature: tuple[tuple[str, int], ...] | None = None

    @property
    def workspace(self) -> Path:
        return self._workspace

    def get(self) -> Catalog:
        """Return the current catalog, reloading only when files changed."""
        signature = self._signature_now()
        if self._catalog is None or signature != self._signature:
            self._catalog = load(self._workspace)
            self._signature = signature
        return self._catalog

    def _signature_now(self) -> tuple[tuple[str, int], ...]:
        catalog_dir = self._workspace / "catalog"
        entries: list[tuple[str, int]] = []
        for name in (*_CATALOG_FILES, *_AI_FILES):
            path = catalog_dir / name if name in _CATALOG_FILES else self._workspace / name
            entries.append((name, path.stat().st_mtime_ns if path.exists() else 0))
        return tuple(entries)


__all__ = ["CatalogCache"]
