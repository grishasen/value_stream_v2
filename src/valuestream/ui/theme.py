"""Streamlit and Plotly theme helpers."""

from __future__ import annotations

from functools import lru_cache

import plotly.graph_objects as go  # type: ignore[import-untyped]
import plotly.io as pio  # type: ignore[import-untyped]
import streamlit as st

_CHROME_TOKENS: dict[str, dict[str, str]] = {
    "light": {
        "cream": "#f7f9fc",
        "card": "#ffffff",
        "ink": "#17202a",
        "muted": "#52606d",
        "border": "#d5dee8",
        "action": "#275dad",
        "verified": "#0f766e",
        "sage": "#e7eef8",
        "soft": "#eef3f8",
        "color-scheme": "light",
        "primary-fg": "#ffffff",
        "shadow": "rgba(23, 32, 42, 0.06)",
        "metric-delta-bg": "#e7f4f1",
    },
    "dark": {
        "cream": "#0b1017",
        "card": "#121a24",
        "ink": "#f3f7fb",
        "muted": "#a9b5c2",
        "border": "#304052",
        "action": "#6ea8fe",
        "verified": "#4fd1c5",
        "sage": "#18263a",
        "soft": "#18212c",
        "color-scheme": "dark",
        "primary-fg": "#08101c",
        "shadow": "rgba(0, 0, 0, 0.32)",
        "metric-delta-bg": "rgba(79, 209, 197, 0.16)",
    },
}

PLOTLY_DARK_COLORWAY = [
    "#56B4E9",
    "#F2C14E",
    "#45D6A5",
    "#F17CB0",
    "#FF8B5C",
    "#B89CFF",
    "#7BDFF2",
    "#F6AE2D",
    "#9CE37D",
    "#E76F91",
    "#F4D35E",
    "#8ECAE6",
]

PLOTLY_LIGHT_COLORWAY = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#6F63B5",
    "#2F6B3C",
    "#8B5E3C",
    "#3A7D8C",
    "#A23B72",
    "#B36B00",
    "#4D648D",
]


def _active_theme_base() -> str:
    """Resolve the theme actually rendered in the user's browser session.

    ``st.context.theme.type`` reflects the per-session theme (including the
    browser's ``prefers-color-scheme`` resolution); the server-level
    ``theme.base`` option is only a fallback outside a script run.
    """
    base: str | None
    try:
        base = st.context.theme.type
    except Exception:  # pragma: no cover - bare-mode/test fallback
        base = None
    base = base or st.get_option("theme.base") or "light"
    return "dark" if base == "dark" else "light"


def apply_app_chrome_tuning() -> None:
    """Apply the small CSS layer Streamlit theming cannot express.

    ``st.context.theme`` resolves the per-session theme server-side, so only
    the active theme's variables are emitted; a rerun re-emits them if the
    browser theme changes.
    """
    active_css_vars = _css_variables(_CHROME_TOKENS[_active_theme_base()])
    st.markdown(
        """
        <style>
        :root {
__VS_ACTIVE_CSS_VARS__
        }

        .stApp {
            background: var(--vs-cream);
            color: var(--vs-ink);
            color-scheme: var(--vs-color-scheme);
        }

        div[data-testid="stAppViewContainer"],
        div[data-testid="stHeader"] {
            background: var(--vs-cream);
        }

        .block-container {
            padding-top: 0.75rem;
            padding-bottom: 2.5rem;
            padding-left: clamp(0.5rem, 0.8vw, 1rem) !important;
            padding-right: clamp(0.5rem, 0.8vw, 1rem) !important;
            max-width: 100rem !important;
            width: 100% !important;
            margin-left: auto;
            margin-right: auto;
        }

        div[data-testid="stMainBlockContainer"].block-container {
            padding-top: 0.75rem !important;
            padding-left: clamp(0.5rem, 0.8vw, 1rem) !important;
            padding-right: clamp(0.5rem, 0.8vw, 1rem) !important;
            max-width: 100rem !important;
            width: 100% !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        .block-container h1,
        .block-container h2 {
            color: var(--vs-ink);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-weight: 650;
            letter-spacing: -0.025em;
            padding-top: 0;
        }

        .block-container h1 {
            font-size: clamp(1.85rem, 2.5vw, 2.35rem);
            line-height: 1.12;
        }

        .block-container h2 {
            font-size: clamp(1.4rem, 2vw, 1.8rem);
            line-height: 1.18;
        }

        div[data-testid="stHeading"] h1,
        div[data-testid="stHeading"] h2 {
            padding-top: 0 !important;
        }

        .block-container h3,
        .block-container h4,
        .block-container h5 {
            color: var(--vs-ink);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-weight: 650;
            letter-spacing: -0.01em;
        }

        .block-container p,
        .block-container label,
        .block-container [data-testid="stCaptionContainer"] {
            color: var(--vs-muted);
        }

        .block-container [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            opacity: 1 !important;
        }

        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: var(--vs-muted) !important;
        }

        div[data-testid="stBadge"],
        div[data-testid="stBadge"] * {
            opacity: 1 !important;
        }

        div[data-testid="stBadge"] p,
        div[data-testid="stBadge"] [data-testid="stIconMaterial"],
        button[aria-label^="Help"] {
            color: var(--vs-ink) !important;
        }

        [data-testid="stMarkdownContainer"] {
            color: inherit;
        }

        button[kind="primary"] p,
        button[kind="primary"] [data-testid="stMarkdownContainer"],
        button[kind="primary"] [data-testid="stIconMaterial"] {
            color: var(--vs-primary-fg) !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
            background: var(--vs-cream);
            padding-top: 2rem;
            border-right: 1px solid var(--vs-border);
        }

        section[data-testid="stSidebar"] div[data-testid="stSidebarHeader"] {
            height: 2rem !important;
            margin-bottom: 0 !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stLogoSpacer"] {
            height: 0 !important;
        }

        section[data-testid="stSidebar"] div[class*="st-key-vs_brand"] p {
            color: var(--vs-ink);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-weight: 700;
            font-size: 1.4rem;
            letter-spacing: -0.025em;
            line-height: 1.15;
            margin-bottom: 0.2rem;
        }

        section[data-testid="stSidebar"] details,
        section[data-testid="stSidebar"] details > summary,
        section[data-testid="stSidebar"] div[data-testid="stExpanderDetails"] {
            background: var(--vs-card) !important;
            border-color: var(--vs-border) !important;
            color: var(--vs-ink) !important;
        }

        section[data-testid="stSidebar"] details > summary p,
        section[data-testid="stSidebar"] details > summary [data-testid="stIconMaterial"] {
            color: var(--vs-muted) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--vs-card);
            border-color: var(--vs-border);
            border-radius: 0.75rem;
            box-shadow: 0 1px 2px var(--vs-shadow);
        }

        div[data-testid="stMetric"] label p {
            color: var(--vs-ink);
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--vs-ink);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-weight: 600;
        }

        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            background: var(--vs-metric-delta-bg);
            border-radius: 999px;
            color: var(--vs-verified);
            display: inline-flex;
            padding: 0.1rem 0.45rem;
            width: fit-content;
        }

        button[kind="secondary"] {
            background: var(--vs-card) !important;
            border-color: var(--vs-border) !important;
            color: var(--vs-ink) !important;
        }

        div[data-testid="stSegmentedControl"] button[aria-checked="false"],
        div[data-testid="stButtonGroup"] button[aria-checked="false"] {
            background: var(--vs-card) !important;
            border-color: var(--vs-border) !important;
            color: var(--vs-muted) !important;
        }

        div[data-testid="stSegmentedControl"] button[aria-checked="true"],
        div[data-testid="stButtonGroup"] button[aria-checked="true"] {
            background: var(--vs-sage) !important;
            border-color: var(--vs-action) !important;
            color: var(--vs-ink) !important;
        }

        button:focus-visible,
        input:focus-visible,
        textarea:focus-visible,
        [role="button"]:focus-visible,
        [role="option"]:focus-visible {
            outline: 3px solid var(--vs-action) !important;
            outline-offset: 2px !important;
        }

        /*
         * Select and multiselect menus are mounted in a body-level portal.
         * Streamlit popovers use the same default layer, so a menu opened from
         * inside a popover can otherwise render behind the popover contents.
         */
        body > div:has([data-testid="stSelectboxVirtualDropdown"]) {
            z-index: 1000070 !important;
        }

        div[class*="st-key-vs_metric_grid_"] div[data-testid="stHorizontalBlock"] {
            display: grid !important;
            grid-template-columns: repeat(var(--vs-grid-columns, 4), minmax(0, 1fr));
            gap: 1rem;
        }

        div[class*="st-key-vs_metric_grid_4_"] { --vs-grid-columns: 4; }
        div[class*="st-key-vs_metric_grid_5_"] { --vs-grid-columns: 5; }
        div[class*="st-key-vs_metric_grid_6_"] { --vs-grid-columns: 6; }

        div[class*="st-key-vs_metric_grid_"] div[data-testid="stColumn"] {
            width: 100% !important;
            min-width: 0 !important;
            flex: none !important;
        }

        @media (max-width: 1440px) {
            div[class*="st-key-vs_metric_grid_5_"] div[data-testid="stHorizontalBlock"],
            div[class*="st-key-vs_metric_grid_6_"] div[data-testid="stHorizontalBlock"] {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
        }

        @media (max-width: 1280px) {
            div[class*="st-key-vs_metric_grid_4_"] div[data-testid="stHorizontalBlock"] {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 760px) {
            div[class*="st-key-vs_metric_grid_"] div[data-testid="stHorizontalBlock"] {
                grid-template-columns: 1fr;
            }
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stDataFrame"]),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stTable"]) {
            background: var(--vs-soft);
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stDataFrame"] > div,
        div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"],
        div[data-testid="stTable"],
        div[data-testid="stTable"] table {
            background: var(--vs-soft) !important;
            color: var(--vs-ink);
        }

        div[data-testid="stTable"] th,
        div[data-testid="stTable"] td {
            background: var(--vs-card) !important;
            border-color: var(--vs-border) !important;
            color: var(--vs-ink) !important;
        }

        div[data-testid="stTable"] th {
            background: var(--vs-soft) !important;
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
                transition-duration: 0.01ms !important;
            }
        }
        </style>
        """.replace("__VS_ACTIVE_CSS_VARS__", active_css_vars),
        unsafe_allow_html=True,
    )


def _css_variables(tokens: dict[str, str]) -> str:
    return "\n".join(f"            --vs-{name}: {value};" for name, value in tokens.items())


@lru_cache(maxsize=1)
def init_plotly_theme() -> None:
    """Register the light and dark Plotly templates used by Value Stream charts.

    Both variants are registered once per process; the session-correct variant
    is selected per figure through :func:`dashboard_theme`, because the browser
    theme differs between sessions while Plotly's default template is global.
    """
    for base in ("light", "dark"):
        tokens = _CHROME_TOKENS[base]
        colorway = PLOTLY_DARK_COLORWAY if base == "dark" else PLOTLY_LIGHT_COLORWAY
        source_name = "plotly_dark" if base == "dark" else "plotly_white"
        grid_color = tokens["border"] if base == "dark" else tokens["soft"]
        template = go.layout.Template(pio.templates[source_name])
        template.layout.update(
            colorway=colorway,
            font={"family": "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"},
            margin={"l": 40, "r": 18, "t": 18, "b": 76},
            hovermode="x unified",
            hoverlabel=_hoverlabel(tokens),
            paper_bgcolor=tokens["cream"],
            plot_bgcolor=tokens["cream"],
            font_color=tokens["ink"],
            xaxis={"gridcolor": grid_color, "zerolinecolor": grid_color},
            yaxis={"gridcolor": grid_color, "zerolinecolor": grid_color},
        )
        pio.templates[f"valuestream_{base}"] = template
    # Alias kept for catalogs that persisted the pre-split template name; the
    # process default is only a fallback for theme-less renders.
    server_base = "dark" if (st.get_option("theme.base") or "light") == "dark" else "light"
    pio.templates["valuestream"] = pio.templates[f"valuestream_{server_base}"]
    pio.templates.default = f"valuestream_{server_base}"


def _hoverlabel(tokens: dict[str, str]) -> dict[str, object]:
    return {
        "bgcolor": tokens["card"],
        "bordercolor": tokens["border"],
        "font": {"color": tokens["ink"]},
    }


def dashboard_theme() -> dict[str, object]:
    """Return a default chart theme overlay for catalog dashboard rendering.

    Font and hoverlabel are set explicitly on every figure (not only in the
    template): Streamlit patches ``layout.font`` client-side with the light
    theme's text color even in dark mode and with ``theme=None``, which made
    hover tooltips render dark-on-dark. Explicit figure-level values keep the
    hover layer readable regardless of that patch.
    """
    base = _active_theme_base()
    tokens = _CHROME_TOKENS[base]
    background = tokens["cream"]
    return {
        "base": base,
        "template": f"valuestream_{base}",
        "paper_bgcolor": background,
        "plot_bgcolor": background,
        "font": {"color": tokens["ink"]},
        "hoverlabel": _hoverlabel(tokens),
        "margins": {"l": 40, "r": 18, "t": 18, "b": 76},
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.16,
            "xanchor": "center",
            "x": 0.5,
        },
    }
