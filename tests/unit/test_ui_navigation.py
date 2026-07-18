"""Focused tests for the Streamlit shell navigation metadata."""

from __future__ import annotations

import logging

import pytest
from streamlit.testing.v1 import AppTest

from valuestream.ui import shell
from valuestream.ui.pages import build as build_page


def test_navigation_sections_group_pages_in_declared_order() -> None:
    pages = [
        shell.NavigationPage("Home", "Home", ":material/home:", lambda: None),
        shell.NavigationPage("Workspace", "Reports", ":material/area_chart:", lambda: None),
        shell.NavigationPage("Workspace", "Chat With Data", ":material/chat:", lambda: None),
    ]

    sections = shell._navigation_sections(pages)

    assert list(sections) == ["Home", "Workspace"]
    assert [page.title for page in sections["Workspace"]] == ["Reports", "Chat With Data"]


def test_navigation_search_matches_section_or_title() -> None:
    pages = [
        shell.NavigationPage("Home", "Home", ":material/home:", lambda: None),
        shell.NavigationPage("Workspace", "Reports", ":material/area_chart:", lambda: None),
        shell.NavigationPage("Workspace", "Chat With Data", ":material/chat:", lambda: None),
        shell.NavigationPage("Settings", "Catalog", ":material/database:", lambda: None),
    ]

    by_section = shell._navigation_sections(pages, query="workspace")
    by_title = shell._navigation_sections(pages, query="chat")

    assert [page.title for page in by_section["Workspace"]] == ["Reports", "Chat With Data"]
    assert list(by_title) == ["Workspace"]
    assert [page.title for page in by_title["Workspace"]] == ["Chat With Data"]


def test_active_navigation_section_prefers_selected_page_title() -> None:
    sections = {
        "Home": [shell.NavigationPage("Home", "Home", ":material/home:", lambda: None)],
        "Settings": [
            shell.NavigationPage("Settings", "Catalog", ":material/database:", lambda: None)
        ],
    }

    active = shell._active_navigation_section(sections, "Catalog", fallback="Home")

    assert active == "Settings"


def test_active_navigation_section_uses_fallback_or_first_section() -> None:
    sections = {
        "Home": [shell.NavigationPage("Home", "Home", ":material/home:", lambda: None)],
        "Settings": [
            shell.NavigationPage("Settings", "Catalog", ":material/database:", lambda: None)
        ],
    }

    assert shell._active_navigation_section(sections, "", fallback="Settings") == "Settings"
    assert shell._active_navigation_section(sections, "", fallback="Missing") == "Home"


def test_navigation_exposes_top_level_build_section(monkeypatch) -> None:
    monkeypatch.setattr(shell, "authoring_v2_enabled", lambda: True)
    context = object()

    pages = shell._navigation_pages(context)  # type: ignore[arg-type]

    build_pages = [page.title for page in pages if page.section == "Build"]
    assert build_pages == ["Build", "Configuration Builder", "AI Configuration Studio"]
    assert not [
        page for page in pages if page.section == "Settings" and "Configuration" in page.title
    ]


def test_navigation_flag_restores_legacy_authoring_group(monkeypatch) -> None:
    monkeypatch.setattr(shell, "authoring_v2_enabled", lambda: False)
    context = object()

    pages = shell._navigation_pages(context)  # type: ignore[arg-type]

    assert not [page for page in pages if page.section == "Build"]
    settings = [page.title for page in pages if page.section == "Settings"]
    assert settings == ["Configuration Builder", "AI Configuration Studio", "Catalog"]


def test_build_landing_starts_both_authoring_paths_in_the_main_canvas() -> None:
    def app() -> None:
        from types import SimpleNamespace  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import build  # noqa: PLC0415 - isolated AppTest source

        build.render(SimpleNamespace())

    rendered = AppTest.from_function(app).run()

    assert not rendered.exception
    assert rendered.title[0].value == "Build"
    assert {button.label for button in rendered.button} >= {
        "Start from sample",
        "Configure manually",
    }
    page_text = " ".join(item.value for item in [*rendered.markdown, *rendered.caption])
    assert "Applying configuration never starts data processing" in page_text


def test_build_sample_choice_enters_studio_before_sample_without_new_journey(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(build_page.ai_config_studio, "render", lambda _ctx: None)

    def app() -> None:
        from types import SimpleNamespace  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import build  # noqa: PLC0415 - isolated AppTest source

        build.render(SimpleNamespace())

    rendered = AppTest.from_function(app).run()
    journey = rendered.session_state["vs_authoring_journey_id"]
    rendered = (
        next(button for button in rendered.button if button.label == "Start from sample")
        .click()
        .run()
    )

    assert not rendered.exception
    assert rendered.session_state["vs_authoring_journey_id"] == journey
    studio_entered = caplog.text.index("workflow=ai_studio event=entered")
    sample_chosen = caplog.text.index("workflow=ai_studio event=sample_chosen")
    assert studio_entered < sample_chosen
    assert "event=abandoned" not in caplog.text
