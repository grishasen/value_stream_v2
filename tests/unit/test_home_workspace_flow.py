"""Focused tests for Home's flag-aware Workspace Flow copy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from valuestream.ui.pages import home


@pytest.fixture
def flow_context(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(
        home,
        "catalog_counts",
        lambda _ctx: {"Sources": 1, "Processors": 1, "Metrics": 2, "Dashboards": 1},
    )
    return SimpleNamespace(validation=SimpleNamespace(ok=True))


@pytest.mark.unit
def test_workspace_flow_lists_build_and_keeps_settings_catalog_focused(
    monkeypatch: pytest.MonkeyPatch,
    flow_context: SimpleNamespace,
) -> None:
    monkeypatch.setattr(home, "authoring_v2_enabled", lambda: True)

    cards = home._workspace_flow_cards(flow_context)  # type: ignore[arg-type]
    descriptions = {title: description for title, description, _status in cards}

    assert [title for title, _description, _status in cards] == [
        "Reports",
        "Chat With Data",
        "Build",
        "Settings",
        "Data Integration",
    ]
    assert descriptions["Build"] == "Guided authoring: Configuration Builder and AI Studio."
    assert descriptions["Settings"] == "Review the applied catalog and workspace configuration."


@pytest.mark.unit
def test_workspace_flow_preserves_legacy_settings_copy_when_build_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    flow_context: SimpleNamespace,
) -> None:
    monkeypatch.setattr(home, "authoring_v2_enabled", lambda: False)

    cards = home._workspace_flow_cards(flow_context)  # type: ignore[arg-type]
    descriptions = {title: description for title, description, _status in cards}

    assert "Build" not in descriptions
    assert descriptions["Settings"] == "Review catalog, config builders, and AI-assisted drafts."
