"""Tests for the Data Load page."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from valuestream.ui.pages import data_load


@pytest.mark.unit
def test_ordered_sources_reverses_without_mutating_catalog_order() -> None:
    sources = [
        SimpleNamespace(id="interaction_history"),
        SimpleNamespace(id="product_holdings"),
    ]

    ordered = data_load._ordered_sources(sources)

    assert [source.id for source in ordered] == ["product_holdings", "interaction_history"]
    assert [source.id for source in sources] == ["interaction_history", "product_holdings"]


@pytest.mark.unit
def test_aggregate_inventory_is_limited_to_selected_sources(tmp_path: Path) -> None:
    selected = tmp_path / "aggregates" / "selected" / "processor" / "daily"
    other = tmp_path / "aggregates" / "other" / "processor" / "daily"
    selected.mkdir(parents=True)
    other.mkdir(parents=True)
    (selected / "one.parquet").write_bytes(b"1234")
    (selected / "two.parquet").write_bytes(b"12")
    (other / "three.parquet").write_bytes(b"ignored")

    files, bytes_used = data_load._aggregate_inventory(tmp_path, ["selected"])

    assert files == 2
    assert bytes_used == 6
    assert data_load._format_bytes(bytes_used) == "6 B"
