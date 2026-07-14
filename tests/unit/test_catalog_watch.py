"""Tests for the mtime-based catalog cache used by long-lived servers."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from valuestream.config.watch import CatalogCache

_DEMO = Path("examples/demo")


def _seed(workspace: Path) -> None:
    shutil.copytree(_DEMO / "catalog", workspace / "catalog")


@pytest.mark.unit
def test_catalog_cache_returns_same_object_until_files_change(tmp_path: Path) -> None:
    _seed(tmp_path)
    cache = CatalogCache(tmp_path)

    first = cache.get()
    assert cache.get() is first  # unchanged files -> cached instance


@pytest.mark.unit
def test_catalog_cache_reloads_after_metric_edit(tmp_path: Path) -> None:
    _seed(tmp_path)
    cache = CatalogCache(tmp_path)
    before = cache.get()
    assert "VS_Interactions" in before.metrics.metrics

    metrics_path = tmp_path / "catalog" / "metrics.yaml"
    text = metrics_path.read_text(encoding="utf-8")
    metrics_path.write_text(
        text + "\n  Pinned_Copy:\n    source: ih_engagement\n    kind: formula\n"
        "    expression: {col: Count}\n",
        encoding="utf-8",
    )

    after = cache.get()
    assert after is not before
    assert "Pinned_Copy" in after.metrics.metrics
