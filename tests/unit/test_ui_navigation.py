"""Focused tests for the Streamlit shell navigation metadata."""

from __future__ import annotations

from valuestream.ui import shell


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
