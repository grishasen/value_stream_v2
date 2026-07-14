"""Guardrails for Streamlit UI implementation choices."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import plotly.io as pio  # type: ignore[import-untyped]
import pytest

from valuestream.ui import config_help, theme

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_ROOT = PROJECT_ROOT / "src" / "valuestream" / "ui"
HTML_MARKERS = ("<style", "<script", "<div", "<span", "<iframe", "<html")
CONFIG_EDITOR_FILES = (
    UI_ROOT / "forms.py",
    UI_ROOT / "pages" / "config_builder.py",
    UI_ROOT / "pages" / "ai_config_studio.py",
    UI_ROOT / "recipe_library.py",
)
CONFIG_FIELD_WIDGETS = {
    "checkbox",
    "color_picker",
    "date_input",
    "file_uploader",
    "multiselect",
    "number_input",
    "segmented_control",
    "select_slider",
    "selectbox",
    "slider",
    "text_area",
    "text_input",
    "time_input",
    "toggle",
}


def _qualname(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _qualname(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _streamlit_aliases(tree: ast.AST) -> tuple[set[str], set[str], set[str]]:
    streamlit_aliases = {"streamlit"}
    component_module_aliases = {"streamlit.components.v1"}
    direct_html_call_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "streamlit":
                    streamlit_aliases.add(alias.asname or alias.name)
                elif alias.name == "streamlit.components.v1":
                    component_module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module in {
            "streamlit",
            "streamlit.components.v1",
        }:
            for alias in node.names:
                if alias.name == "html":
                    direct_html_call_names.add(alias.asname or alias.name)
    return streamlit_aliases, component_module_aliases, direct_html_call_names


def _constant_text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _contains_style_template(node: ast.AST | None) -> bool:
    if node is None:
        return False
    rendered_text = _constant_text(node)
    if rendered_text is not None:
        return "<style" in rendered_text.casefold()
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "replace"
    ):
        return _contains_style_template(node.func.value)
    return False


def _is_allowed_theme_style(path: Path, call_name: str, first_arg: ast.AST | None) -> bool:
    return (
        path.relative_to(UI_ROOT).as_posix() == "theme.py"
        and call_name.endswith(".markdown")
        and _contains_style_template(first_arg)
    )


@pytest.mark.unit
def test_streamlit_ui_uses_native_components_instead_of_rendered_html() -> None:
    violations: list[str] = []

    for path in sorted(UI_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        streamlit_aliases, component_module_aliases, direct_html_call_names = _streamlit_aliases(
            tree
        )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            call_name = _qualname(node.func)
            first_arg = node.args[0] if node.args else None
            rendered_text = _constant_text(first_arg) if first_arg is not None else None
            allowed_theme_style = _is_allowed_theme_style(path, call_name, first_arg)
            for keyword in node.keywords:
                if (
                    keyword.arg == "unsafe_allow_html"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    and not allowed_theme_style
                ):
                    violations.append(
                        f"{path.relative_to(UI_ROOT)}:{node.lineno} uses unsafe_allow_html=True"
                    )

            if any(call_name == f"{alias}.html" for alias in streamlit_aliases):
                violations.append(f"{path.relative_to(UI_ROOT)}:{node.lineno} uses st.html")
            if any(call_name == f"{alias}.html" for alias in component_module_aliases):
                violations.append(
                    f"{path.relative_to(UI_ROOT)}:{node.lineno} uses streamlit.components HTML"
                )
            if call_name in direct_html_call_names:
                violations.append(
                    f"{path.relative_to(UI_ROOT)}:{node.lineno} uses imported Streamlit HTML"
                )

            if (
                any(call_name == f"{alias}.markdown" for alias in streamlit_aliases)
                and rendered_text
                and any(marker in rendered_text.casefold() for marker in HTML_MARKERS)
                and not allowed_theme_style
            ):
                violations.append(
                    f"{path.relative_to(UI_ROOT)}:{node.lineno} passes HTML to st.markdown"
                )

    assert not violations, "Use native Streamlit components instead:\n" + "\n".join(violations)


@pytest.mark.unit
def test_config_editor_fields_and_table_columns_have_help() -> None:
    missing_help: list[str] = []

    for path in CONFIG_EDITOR_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            is_field_widget = node.func.attr in CONFIG_FIELD_WIDGETS
            is_editor_column = node.func.attr.endswith("Column") and "column_config" in _qualname(
                node.func
            )
            if not is_field_widget and not is_editor_column:
                continue
            if not any(keyword.arg == "help" for keyword in node.keywords):
                missing_help.append(f"{path.relative_to(UI_ROOT)}:{node.lineno}")

    assert not missing_help, "Configuration fields require help tooltips:\n" + "\n".join(
        missing_help
    )


@pytest.mark.unit
def test_config_help_catalog_includes_examples_for_concrete_values() -> None:
    assert len(config_help.FIELD_HELP) >= 150
    assert all(text.strip() for text in config_help.FIELD_HELP.values())
    for key in (
        "source.id",
        "processor.id",
        "processor.group_by",
        "processor.subject_field",
        "metric.id",
        "metric.depends_on",
        "metric.state",
        "report.tile_title",
        "workspace.time_zone",
        "recipe.algorithm",
    ):
        assert "**Example:**" in config_help.field_help(key)


@pytest.mark.unit
def test_dark_theme_uses_accessible_surface_ladder() -> None:
    config = tomllib.loads((UI_ROOT / ".streamlit" / "config.toml").read_text())
    dark_theme = config["theme"]["dark"]
    dark_sidebar = config["theme"]["dark"]["sidebar"]
    theme_source = (UI_ROOT / "theme.py").read_text(encoding="utf-8")

    expected_theme_tokens = {
        "primaryColor": "#45BA50",
        "backgroundColor": "#080A09",
        "secondaryBackgroundColor": "#171C18",
        "textColor": "#F0F3F0",
        "borderColor": "#343C36",
        "dataframeHeaderBackgroundColor": "#171C18",
        "blueColor": "#00B1D8",
        "violetColor": "#AD87ED",
        "greenColor": "#45BA50",
        "orangeColor": "#FF8B25",
        "redColor": "#F14D4C",
        "grayColor": "#A7AFA9",
    }
    for key, value in expected_theme_tokens.items():
        assert dark_theme[key] == value

    assert dark_sidebar["backgroundColor"] == "#080A09"
    assert dark_sidebar["secondaryBackgroundColor"] == "#171C18"

    for token in (
        '"cream": "#080a09"',
        '"card": "#111512"',
        '"soft": "#171c18"',
        '"border": "#343c36"',
        '"green": "#45ba50"',
        '"muted": "#a7afa9"',
    ):
        assert token in theme_source


@pytest.mark.unit
def test_plotly_template_uses_distinct_report_colorways() -> None:
    previous_default = pio.templates.default
    try:
        theme.init_plotly_theme.cache_clear()
        theme.init_plotly_theme()
        light_layout = pio.templates["valuestream_light"].layout
        light_colorway = list(light_layout.colorway)
        light_paper_bgcolor = light_layout.paper_bgcolor
        light_plot_bgcolor = light_layout.plot_bgcolor

        dark_layout = pio.templates["valuestream_dark"].layout
        dark_colorway = list(dark_layout.colorway)
        dark_paper_bgcolor = dark_layout.paper_bgcolor
        dark_plot_bgcolor = dark_layout.plot_bgcolor
    finally:
        pio.templates.default = previous_default

    assert light_colorway == theme.PLOTLY_LIGHT_COLORWAY
    assert dark_colorway == theme.PLOTLY_DARK_COLORWAY
    assert light_colorway[:2] == ["#0072B2", "#D55E00"]
    assert dark_colorway[:2] == ["#56B4E9", "#F2C14E"]
    assert light_paper_bgcolor == "#f5f3ee"
    assert light_plot_bgcolor == "#f5f3ee"
    assert dark_paper_bgcolor == "#080a09"
    assert dark_plot_bgcolor == "#080a09"


@pytest.mark.unit
def test_dashboard_theme_carries_app_background(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(theme, "_active_theme_base", lambda: "light")
    light = theme.dashboard_theme()

    monkeypatch.setattr(theme, "_active_theme_base", lambda: "dark")
    dark = theme.dashboard_theme()

    assert light["paper_bgcolor"] == "#f5f3ee"
    assert light["plot_bgcolor"] == "#f5f3ee"
    assert dark["paper_bgcolor"] == "#080a09"
    assert dark["plot_bgcolor"] == "#080a09"


@pytest.mark.unit
def test_theme_covers_surface_gaps_native_theming_cannot_express() -> None:
    theme_source = (UI_ROOT / "theme.py").read_text(encoding="utf-8")

    for selector in (
        'div[data-testid="stMainBlockContainer"].block-container',
        "padding-left: clamp(0.5rem, 0.8vw, 1rem) !important",
        "max-width: 100rem !important",
        'div[data-testid="stBadge"]',
        'div[data-testid="stSegmentedControl"] button[aria-checked="false"]',
        'div[class*="st-key-vs_metric_grid_"] div[data-testid="stHorizontalBlock"]',
        'div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stDataFrame"])',
        'div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"]',
    ):
        assert selector in theme_source

    # Widget input surfaces come from secondaryBackgroundColor in config.toml;
    # BaseWeb-internal selectors are brittle and must not come back.
    for forbidden in (
        'data-baseweb="input"',
        'data-baseweb="base-input"',
        'data-baseweb="select"',
        "st-emotion-cache",
    ):
        assert forbidden not in theme_source


@pytest.mark.unit
def test_summary_metrics_use_card_helper() -> None:
    direct_metric_calls: list[str] = []
    for relative_path in ("pages/data_load.py", "pages/home.py"):
        path = UI_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _qualname(node.func).endswith(".metric"):
                direct_metric_calls.append(f"{relative_path}:{node.lineno}")

    components_source = (UI_ROOT / "components.py").read_text(encoding="utf-8")
    reports_source = (UI_ROOT / "pages" / "reports.py").read_text(encoding="utf-8")

    assert not direct_metric_calls, (
        "Summary metrics should use components.metric_cards:\n" + "\n".join(direct_metric_calls)
    )
    assert "def metric_cards" in components_source
    assert "metric_cards(items, columns=columns)" in components_source
    assert "components.metric_strip(items," in reports_source
