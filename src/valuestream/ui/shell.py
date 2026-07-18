"""Streamlit app shell for Value Stream."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import streamlit as st

from valuestream.ui.context import ValueStreamContext
from valuestream.ui.feature_flags import authoring_v2_enabled
from valuestream.ui.instrumentation import abandon_active_journey, workflow_from_handoff
from valuestream.ui.pages import (
    ai_config_studio,
    build,
    catalog,
    chat,
    config_builder,
    data_load,
    home,
    ops,
    reports,
)
from valuestream.ui.theme import apply_app_chrome_tuning, init_plotly_theme


@dataclass(frozen=True)
class NavigationPage:
    """Page metadata used to build and filter the Streamlit navigation."""

    section: str
    title: str
    icon: str
    target: Callable[[], None]
    default: bool = False
    indent: bool = False


def parse_args() -> argparse.Namespace:
    """Parse Streamlit passthrough CLI arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace", default="examples/demo")
    parser.add_argument("--logging-config", "--logging_config", default=None)
    return parser.parse_known_args()[0]


def configure_page() -> None:
    """Configure Streamlit before rendering pages."""
    st.set_page_config(
        page_title="Value Stream",
        page_icon=":material/analytics:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    init_plotly_theme()
    apply_app_chrome_tuning()


def render_navigation(ctx: ValueStreamContext) -> None:
    """Render the app navigation and dispatch the selected page."""
    pages = _navigation_pages(ctx)
    all_sections = _navigation_sections(pages, query="")
    streamlit_sections = _streamlit_navigation(all_sections)
    page_lookup = _page_lookup(streamlit_sections, all_sections)
    pg = st.navigation(streamlit_sections, position="hidden")

    with st.sidebar:
        with st.container(key="vs_brand"):
            st.markdown("Value Stream")
        st.caption(f"Workspace · {ctx.catalog.pipelines.workspace}")
        with st.popover(
            "Workspace details",
            icon=":material/info:",
            width="stretch",
        ):
            st.caption(str(ctx.workspace))
            st.caption(f"Catalog `{ctx.catalog_hash}`")
        query = st.text_input(
            "Search",
            key="navigation_search",
            placeholder="Search pages",
            label_visibility="collapsed",
            icon=":material/search:",
        )

        sections = _navigation_sections(pages, query=query)
        if not sections:
            st.caption("No matching pages.")
        else:
            _render_sidebar_links(
                sections,
                page_lookup,
                selected_title=str(getattr(pg, "title", "")),
            )

    _record_authoring_navigation_exit(str(getattr(pg, "title", "")))
    pg.run()


def _record_authoring_navigation_exit(selected_title: str) -> None:
    """Capture observable in-app abandonment without browser unload hooks."""

    if selected_title in {"Build", "Configuration Builder", "AI Configuration Studio"}:
        return
    if selected_title in {"Reports", "Data Load"} and workflow_from_handoff(
        st.query_params.get("from")
    ):
        return
    abandon_active_journey(st.session_state)


def _navigation_pages(ctx: ValueStreamContext) -> list[NavigationPage]:
    pages = [
        NavigationPage("Home", "Home", ":material/home:", lambda: home.render(ctx), default=True),
        NavigationPage("Analysis", "Reports", ":material/area_chart:", lambda: reports.render(ctx)),
        NavigationPage("Analysis", "Chat With Data", ":material/chat:", lambda: chat.render(ctx)),
        NavigationPage("Settings", "Catalog", ":material/database:", lambda: catalog.render(ctx)),
        NavigationPage(
            "Data Integration",
            "Data Load",
            ":material/database_upload:",
            lambda: data_load.render(ctx),
        ),
        NavigationPage(
            "Data Integration",
            "Pipelines / Ops",
            ":material/monitoring:",
            lambda: ops.render(ctx),
        ),
    ]
    authoring_section = "Build" if authoring_v2_enabled() else "Settings"
    authoring_pages = (
        [
            NavigationPage(
                authoring_section,
                "Build",
                ":material/construction:",
                lambda: build.render(ctx),
            )
        ]
        if authoring_v2_enabled()
        else []
    )
    # Under the Build landing page, the two authoring tools read as its
    # children; without the landing page they stay flat section entries.
    tools_indented = authoring_v2_enabled()
    authoring_pages.extend(
        [
            NavigationPage(
                authoring_section,
                "Configuration Builder",
                ":material/build:",
                lambda: config_builder.render(ctx),
                indent=tools_indented,
            ),
            NavigationPage(
                authoring_section,
                "AI Configuration Studio",
                ":material/network_intelligence:",
                lambda: ai_config_studio.render(ctx),
                indent=tools_indented,
            ),
        ]
    )
    return [*pages[:3], *authoring_pages, *pages[3:]]


def _navigation_sections(
    pages: list[NavigationPage],
    *,
    query: str | None = None,
) -> dict[str, list[NavigationPage]]:
    normalized_query = (query or "").strip().casefold()
    sections: dict[str, list[NavigationPage]] = {}
    for page in pages:
        if normalized_query and not _page_matches(page, normalized_query):
            continue
        sections.setdefault(page.section, []).append(page)
    return sections


def _streamlit_navigation(sections: dict[str, list[NavigationPage]]) -> dict[str, list[Any]]:
    return {
        section: [
            _page(page.target, page.title, page.icon, default=page.default)
            for page in section_pages
        ]
        for section, section_pages in sections.items()
    }


def _page_lookup(
    streamlit_sections: dict[str, list[Any]],
    navigation_sections: dict[str, list[NavigationPage]],
) -> dict[tuple[str, str], Any]:
    lookup: dict[tuple[str, str], Any] = {}
    for section, navigation_pages in navigation_sections.items():
        for navigation_page, streamlit_page in zip(
            navigation_pages,
            streamlit_sections[section],
            strict=True,
        ):
            lookup[(section, navigation_page.title)] = streamlit_page
    return lookup


def _render_sidebar_links(
    sections: dict[str, list[NavigationPage]],
    page_lookup: dict[tuple[str, str], Any],
    *,
    selected_title: str = "",
) -> None:
    active_section = _active_navigation_section(
        sections,
        selected_title,
        fallback=str(st.session_state.get("current_section_selected", "")),
    )
    if active_section:
        st.session_state["current_section_selected"] = active_section
    for section, section_pages in sections.items():
        expander_key = f"nav_section_{_url_path(section)}_{_url_path(active_section)}"
        with st.expander(section, expanded=section == active_section, key=expander_key):
            for page in section_pages:
                container = st
                if page.indent:
                    # Children of a section landing page read as a submenu:
                    # a spacer column insets the link and its highlight pill.
                    _, container = st.columns([0.08, 0.92], gap="small")
                container.page_link(
                    page_lookup[(section, page.title)],
                    label=page.title,
                    icon=page.icon,
                    width="stretch",
                )


def _active_navigation_section(
    sections: dict[str, list[NavigationPage]],
    selected_title: str,
    *,
    fallback: str = "",
) -> str:
    for section, section_pages in sections.items():
        if any(page.title == selected_title for page in section_pages):
            return section
    if fallback in sections:
        return fallback
    return next(iter(sections), "")


def _page_matches(page: NavigationPage, query: str) -> bool:
    searchable = f"{page.section} {page.title}".casefold()
    return query in searchable


def _page(
    target: Callable[[], None],
    title: str,
    icon: str,
    *,
    default: bool = False,
) -> Any:
    return st.Page(target, title=title, icon=icon, default=default, url_path=_url_path(title))


def _url_path(title: str) -> str:
    return title.casefold().replace(" / ", "_").replace(" ", "_").replace("-", "_")


__all__ = ["configure_page", "parse_args", "render_navigation"]
