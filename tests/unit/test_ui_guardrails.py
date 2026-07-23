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


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


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
def test_configuration_editors_use_one_truthful_action_vocabulary() -> None:
    builder_source = (UI_ROOT / "pages" / "config_builder.py").read_text(encoding="utf-8")
    studio_source = (UI_ROOT / "pages" / "ai_config_studio.py").read_text(encoding="utf-8")
    combined = builder_source + studio_source

    for action in ("Jump to step", "Back", "Continue", "Apply to workspace"):
        assert action in builder_source
        assert action in studio_source
    assert "Export current workspace" in builder_source
    for obsolete in (
        "Save Draft & Run Source",
        "Apply Draft & Run Source",
        "Save draft",
        "Save & Export",
    ):
        assert obsolete not in combined
    assert "run_source(" not in builder_source
    assert "run_source(" not in studio_source


@pytest.mark.unit
def test_ambiguous_configuration_controls_keep_targeted_help() -> None:
    builder_source = (UI_ROOT / "pages" / "config_builder.py").read_text(encoding="utf-8")
    studio_source = (UI_ROOT / "pages" / "ai_config_studio.py").read_text(encoding="utf-8")

    for key in ("source.reader", "processor.kind", "metric.kind", "report.chart"):
        assert f'config_help.field_help("{key}")' in builder_source
    for key in (
        "ai.model",
        "ai.timeout",
        "source.reader",
        "mapping.subject",
        "processor.kind",
        "metric.kind",
    ):
        assert f'config_help.field_help("{key}")' in studio_source


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
        "primaryColor": "#365EDB",
        "backgroundColor": "#0B1220",
        "secondaryBackgroundColor": "#1C2C43",
        "textColor": "#F7FAFF",
        "borderColor": "#7890AA",
        "dataframeHeaderBackgroundColor": "#223750",
        "blueColor": "#4B73F0",
        "violetColor": "#AD87ED",
        "greenColor": "#45D6A5",
        "orangeColor": "#F2C14E",
        "yellowColor": "#F2C14E",
        "redColor": "#FF8A80",
        "grayColor": "#B8C4D2",
    }
    for key, value in expected_theme_tokens.items():
        assert dark_theme[key] == value

    assert dark_sidebar["backgroundColor"] == "#0F1A2A"
    assert dark_sidebar["secondaryBackgroundColor"] == "#1C2C43"

    for token in (
        '"cream": "#0b1220"',
        '"sidebar": "#0f1a2a"',
        '"card": "#162438"',
        '"raised": "#223750"',
        '"soft": "#1c2c43"',
        '"border": "#3c5573"',
        '"input-border": "#7890aa"',
        '"action": "#365edb"',
        '"accent": "#22c7f3"',
        '"verified": "#45d6a5"',
        '"danger": "#ff8a80"',
        '"muted": "#b8c4d2"',
    ):
        assert token in theme_source


@pytest.mark.unit
def test_dark_review_theme_is_deterministic() -> None:
    assert theme._active_theme_base() == "dark"


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
    assert dark_colorway[:6] == [
        "#4B73F0",
        "#22C7F3",
        "#45D6A5",
        "#F2C14E",
        "#FF8A80",
        "#AD87ED",
    ]
    assert light_paper_bgcolor == "#ffffff"
    assert light_plot_bgcolor == "#ffffff"
    assert dark_paper_bgcolor == "#162438"
    assert dark_plot_bgcolor == "#162438"


@pytest.mark.unit
def test_dashboard_theme_carries_app_background(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(theme, "_active_theme_base", lambda: "light")
    light = theme.dashboard_theme()

    monkeypatch.setattr(theme, "_active_theme_base", lambda: "dark")
    dark = theme.dashboard_theme()

    assert light["paper_bgcolor"] == "#ffffff"
    assert light["plot_bgcolor"] == "#ffffff"
    assert dark["paper_bgcolor"] == "#162438"
    assert dark["plot_bgcolor"] == "#162438"
    assert dark["colorway"][:2] == ["#4B73F0", "#22C7F3"]


@pytest.mark.unit
def test_primary_dark_chart_colors_meet_non_text_contrast() -> None:
    for color in theme.PLOTLY_DARK_COLORWAY[:6]:
        assert _contrast_ratio(color, "#162438") >= 3.0


@pytest.mark.unit
def test_authoring_theme_tokens_meet_wcag_aa_contrast() -> None:
    required_pairs = (
        ("#17202A", "#F7F9FC"),
        ("#52606D", "#F7F9FC"),
        ("#FFFFFF", "#275DAD"),
        ("#8C1D18", "#FFFFFF"),
        ("#F7FAFF", "#0B1220"),
        ("#B8C4D2", "#0B1220"),
        ("#FFFFFF", "#365EDB"),
        ("#FFB4AD", "#162438"),
    )
    assert all(
        _contrast_ratio(foreground, background) >= 4.5 for foreground, background in required_pairs
    )

    # WCAG 1.4.11 non-text contrast: the field boundary against both the
    # input fill (secondaryBackgroundColor) and the card behind it.
    component_boundary_pairs = (
        ("#7C8CA0", "#EEF3F8"),
        ("#7C8CA0", "#FFFFFF"),
        ("#7890AA", "#1C2C43"),
        ("#7890AA", "#162438"),
    )
    assert all(
        _contrast_ratio(foreground, background) >= 3.0
        for foreground, background in component_boundary_pairs
    )


@pytest.mark.unit
def test_authoring_theme_has_visible_focus_and_reduced_motion() -> None:
    theme_source = (UI_ROOT / "theme.py").read_text(encoding="utf-8")
    config_source = (UI_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")

    assert ":focus-visible" in theme_source
    assert "@media (prefers-reduced-motion: reduce)" in theme_source
    assert "Playfair" not in theme_source + config_source
    assert "DM Sans" not in theme_source + config_source
    assert 'buttonRadius = "0.625rem"' in config_source


@pytest.mark.unit
def test_theme_covers_surface_gaps_native_theming_cannot_express() -> None:
    theme_source = (UI_ROOT / "theme.py").read_text(encoding="utf-8")

    for selector in (
        'div[data-testid="stMainBlockContainer"].block-container',
        "padding-left: clamp(0.75rem, 1.25vw, 1.5rem) !important",
        "max-width: 108rem !important",
        'div[data-testid="stBadge"]',
        'div[class*="st-key-vs_nav_active_"] a',
        'div[data-testid="stSegmentedControl"] button[aria-checked="false"]',
        'div[data-testid="stTextInputRootElement"]:focus-within',
        'body > div:has([data-testid="stSelectboxVirtualDropdown"])',
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


@pytest.mark.unit
def test_app_entry_pins_arrow_memory_pool_before_page_imports() -> None:
    """Arrow's bundled mimalloc segfaults in per-thread heap init on macOS
    arm64 when Table.to_pandas runs on Streamlit's short-lived script threads;
    app.py must select the system pool before any UI import can allocate."""

    path = UI_ROOT / "app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    setdefault_line: int | None = None
    first_ui_import_line: int | None = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and _qualname(node.func.value).endswith("os.environ")
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "ARROW_DEFAULT_MEMORY_POOL"
        ):
            setdefault_line = node.lineno
        if (
            isinstance(node, ast.ImportFrom)
            and str(node.module).startswith("valuestream")
            and (first_ui_import_line is None or node.lineno < first_ui_import_line)
        ):
            first_ui_import_line = node.lineno

    assert setdefault_line is not None, "app.py must default ARROW_DEFAULT_MEMORY_POOL"
    assert first_ui_import_line is not None
    assert setdefault_line < first_ui_import_line, (
        "ARROW_DEFAULT_MEMORY_POOL must be set before valuestream imports so the "
        "default is in place before the first Arrow allocation"
    )
