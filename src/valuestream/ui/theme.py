"""Streamlit and Plotly theme helpers."""

from __future__ import annotations

from functools import lru_cache

import plotly.graph_objects as go  # type: ignore[import-untyped]
import plotly.io as pio  # type: ignore[import-untyped]
import streamlit as st

_CHROME_TOKENS: dict[str, dict[str, str]] = {
    "light": {
        "cream": "#f7f9fc",
        "sidebar": "#f2f5f9",
        "card": "#ffffff",
        "raised": "#e7eef8",
        "ink": "#17202a",
        "muted": "#52606d",
        "border": "#d5dee8",
        # Field boundary: 3.08:1 on the #eef3f8 input fill, 3.43:1 on card
        # white (WCAG 1.4.11). Mirrors borderColor in .streamlit/config.toml.
        "input-border": "#7c8ca0",
        "action": "#275dad",
        "action-hover": "#1e4a8f",
        "accent": "#0072b2",
        "attention": "#b45309",
        "attention-soft": "#fff7ed",
        "verified": "#0f766e",
        "danger": "#b3261e",
        "danger-soft": "#fceeee",
        "danger-ink": "#8c1d18",
        "sage": "#e7eef8",
        "soft": "#eef3f8",
        "color-scheme": "light",
        "primary-fg": "#ffffff",
        "shadow": "rgba(23, 32, 42, 0.06)",
        "surface-shadow": "rgba(23, 32, 42, 0.10)",
        "glow": "rgba(0, 114, 178, 0.14)",
        "metric-delta-bg": "#e7f4f1",
    },
    "dark": {
        "cream": "#0b1220",
        "sidebar": "#0f1a2a",
        "card": "#162438",
        "raised": "#223750",
        "ink": "#f7faff",
        "muted": "#b8c4d2",
        "border": "#3c5573",
        # High-contrast field boundary across #1c2c43 inputs and #162438 cards.
        "input-border": "#7890aa",
        "action": "#365edb",
        "action-hover": "#4b73f0",
        "accent": "#22c7f3",
        "attention": "#f2c14e",
        "attention-soft": "rgba(245, 158, 11, 0.12)",
        "verified": "#45d6a5",
        "danger": "#ff8a80",
        "danger-soft": "#3b1f26",
        "danger-ink": "#ffb4ad",
        "sage": "#1a3348",
        "soft": "#1c2c43",
        "color-scheme": "dark",
        "primary-fg": "#ffffff",
        "shadow": "rgba(0, 0, 0, 0.42)",
        "surface-shadow": "rgba(0, 0, 0, 0.55)",
        "glow": "rgba(34, 199, 243, 0.18)",
        "metric-delta-bg": "rgba(69, 214, 165, 0.16)",
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
    """Return the theme rendered by the isolated dark-theme review branch.

    Streamlit uses its light rendering engine for startup compatibility, but
    both native theme tables and the application chrome carry the same dark
    tokens. Keeping the branch dark-only makes the visual review deterministic.
    """
    return "dark"


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
            font-family: "Avenir Next", Avenir, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        div[data-testid="stAppViewContainer"] {
            background: var(--vs-cream);
        }

        div[data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 3rem;
            padding-left: clamp(0.75rem, 1.25vw, 1.5rem) !important;
            padding-right: clamp(0.75rem, 1.25vw, 1.5rem) !important;
            max-width: 108rem !important;
            width: 100% !important;
            margin-left: auto;
            margin-right: auto;
        }

        div[data-testid="stMainBlockContainer"].block-container {
            padding-top: 1rem !important;
            padding-left: clamp(0.75rem, 1.25vw, 1.5rem) !important;
            padding-right: clamp(0.75rem, 1.25vw, 1.5rem) !important;
            max-width: 108rem !important;
            width: 100% !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        .block-container h1,
        .block-container h2 {
            color: var(--vs-ink);
            font-family: "Avenir Next", Avenir, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-weight: 700;
            letter-spacing: -0.035em;
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
            font-family: "Avenir Next", Avenir, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-weight: 650;
            letter-spacing: -0.02em;
        }

        .block-container p,
        .block-container label,
        .block-container [data-testid="stCaptionContainer"] {
            color: var(--vs-muted);
        }

        /* Field labels read as ink, one clear step above muted captions. */
        .block-container [data-testid="stWidgetLabel"] p,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        div[data-testid="stDialog"] [data-testid="stWidgetLabel"] p {
            color: var(--vs-ink);
            font-weight: 600;
        }

        .block-container [data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] p,
        .block-container [data-testid="stToggle"] [data-testid="stMarkdownContainer"] p {
            color: var(--vs-ink);
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
            background: var(--vs-sidebar);
            padding-top: 2.25rem;
            border-right: 1px solid var(--vs-border);
            box-shadow: 0.7rem 0 2.2rem -1.8rem var(--vs-surface-shadow);
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
            font-family: "Avenir Next", Avenir, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-weight: 750;
            font-size: 1.45rem;
            letter-spacing: -0.035em;
            line-height: 1.15;
            margin-bottom: 0.2rem;
        }

        section[data-testid="stSidebar"] details,
        section[data-testid="stSidebar"] details > summary,
        section[data-testid="stSidebar"] div[data-testid="stExpanderDetails"] {
            background: var(--vs-soft) !important;
            border-color: var(--vs-border) !important;
            color: var(--vs-ink) !important;
        }

        section[data-testid="stSidebar"] details > summary p,
        section[data-testid="stSidebar"] details > summary [data-testid="stIconMaterial"] {
            color: var(--vs-muted) !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stPageLink"] a {
            border: 1px solid transparent;
            border-radius: 0.7rem;
            color: var(--vs-muted);
            min-height: 2.4rem;
            transition:
                background-color 140ms ease,
                border-color 140ms ease,
                color 140ms ease;
        }

        section[data-testid="stSidebar"] div[data-testid="stPageLink"] a:hover {
            background: var(--vs-raised);
            border-color: var(--vs-border);
            color: var(--vs-ink);
        }

        /* The keyed active-page container carries the cyan signal rail. */
        section[data-testid="stSidebar"] div[class*="st-key-vs_nav_active_"] a {
            background: var(--vs-raised) !important;
            border-color: var(--vs-border) !important;
            border-left: 0.22rem solid var(--vs-accent) !important;
            box-shadow: 0 0.55rem 1.25rem -0.8rem var(--vs-glow);
            color: var(--vs-ink) !important;
            opacity: 1 !important;
        }

        section[data-testid="stSidebar"] div[class*="st-key-vs_nav_active_"] a p {
            color: var(--vs-ink) !important;
            font-weight: 700;
        }

        section[data-testid="stSidebar"] div[class*="st-key-vs_nav_active_"] a [data-testid="stIconMaterial"] {
            color: var(--vs-accent) !important;
            opacity: 1 !important;
        }

        div[data-testid="stTextInputRootElement"],
        div[data-testid="stTextAreaRootElement"],
        div[data-testid="stSelectbox"] [role="combobox"],
        div[data-testid="stNumberInputContainer"],
        div[data-testid="stDateInput"] > div {
            background: var(--vs-soft) !important;
            border-color: var(--vs-input-border) !important;
            border-radius: 0.7rem !important;
            box-shadow: none !important;
        }

        input,
        textarea,
        div[data-testid="stSelectbox"] [role="combobox"] {
            color: var(--vs-ink) !important;
            caret-color: var(--vs-accent);
        }

        input::placeholder,
        textarea::placeholder {
            color: var(--vs-muted) !important;
            opacity: 0.86;
        }

        div[data-testid="stTextInputRootElement"]:focus-within,
        div[data-testid="stTextAreaRootElement"]:focus-within,
        div[data-testid="stSelectbox"]:focus-within [role="combobox"],
        div[data-testid="stNumberInputContainer"]:focus-within {
            border-color: var(--vs-accent) !important;
            box-shadow: 0 0 0 0.15rem var(--vs-glow) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--vs-card);
            border-color: var(--vs-border);
            border-radius: 1rem;
            box-shadow: 0 0.85rem 2rem -1.35rem var(--vs-surface-shadow);
        }

        div[data-testid="stMetric"] label p {
            color: var(--vs-ink);
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--vs-ink);
            font-family: "Avenir Next", Avenir, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-weight: 650;
        }

        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            background: var(--vs-metric-delta-bg);
            border-radius: 999px;
            color: var(--vs-verified);
            display: inline-flex;
            padding: 0.1rem 0.45rem;
            width: fit-content;
        }

        div[data-testid="stAlert"] {
            background: var(--vs-soft);
            border-color: var(--vs-border);
            border-radius: 0.8rem;
            color: var(--vs-ink);
        }

        div[data-testid="stAlert"] p {
            color: var(--vs-ink) !important;
        }

        div[data-testid="stTabs"] [role="tablist"] {
            background: var(--vs-soft);
            border: 1px solid var(--vs-border);
            border-radius: 0.75rem;
            padding: 0.2rem;
        }

        div[data-testid="stTabs"] [role="tab"] {
            border-radius: 0.55rem;
            color: var(--vs-muted);
        }

        div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            background: var(--vs-raised);
            color: var(--vs-ink);
        }

        div[data-testid="stPopoverBody"],
        div[data-testid="stDialog"],
        div[role="dialog"] {
            background: var(--vs-card) !important;
            border-color: var(--vs-border) !important;
            color: var(--vs-ink) !important;
        }

        button[kind="secondary"] {
            background: var(--vs-soft) !important;
            border-color: var(--vs-border) !important;
            color: var(--vs-ink) !important;
        }

        button[kind="primary"]:not(:disabled) {
            background-color: var(--vs-action) !important;
            border-color: var(--vs-action) !important;
            box-shadow: 0 0.5rem 1.25rem -0.65rem var(--vs-glow);
        }

        button[kind="primary"]:hover:not(:disabled),
        button[kind="primary"]:active:not(:disabled) {
            background-color: var(--vs-action-hover) !important;
            border-color: var(--vs-action-hover) !important;
            box-shadow: 0 0.65rem 1.5rem -0.65rem var(--vs-glow);
        }

        button[kind="secondary"]:hover:not(:disabled) {
            background: var(--vs-raised) !important;
            border-color: var(--vs-accent) !important;
        }

        button {
            border-radius: 0.65rem !important;
            font-weight: 600 !important;
            transition:
                background-color 140ms ease,
                border-color 140ms ease,
                box-shadow 140ms ease;
        }

        /*
         * Destructive actions share the builder_delete_ key prefix; cancel
         * buttons inside the same dialogs carry _cancel_ and stay neutral.
         * Triggers are secondary (danger outline); the cascade confirm is
         * primary (danger fill) and must not fall back to the blue hover.
         */
        div[class*="st-key-builder_delete_"]:not([class*="_cancel_"]) button[kind="secondary"],
        div[class*="st-key-builder_delete_"]:not([class*="_cancel_"]) button[kind="secondary"] p,
        div[class*="st-key-builder_delete_"]:not([class*="_cancel_"]) button[kind="secondary"] [data-testid="stIconMaterial"] {
            border-color: var(--vs-danger) !important;
            color: var(--vs-danger-ink) !important;
        }

        div[class*="st-key-builder_delete_"] button[kind="primary"]:not(:disabled),
        div[class*="st-key-builder_delete_"] button[kind="primary"]:hover:not(:disabled),
        div[class*="st-key-builder_delete_"] button[kind="primary"]:active:not(:disabled) {
            background-color: var(--vs-danger) !important;
            border-color: var(--vs-danger) !important;
        }

        /*
         * Create/edit mode selectors for processors and metrics are paired
         * full-width buttons at standard button height. The active mode is
         * the filled (primary) button. Create is green (verified), edit is
         * blue (action); key slugs carry _mode_button_create_ / _mode_button_edit_.
         */
        div[class*="_mode_button_create_"] button[kind="secondary"]:not(:disabled),
        div[class*="_mode_button_create_"] button[kind="secondary"]:not(:disabled) p,
        div[class*="_mode_button_create_"] button[kind="secondary"]:not(:disabled) [data-testid="stIconMaterial"] {
            border-color: var(--vs-verified) !important;
            color: var(--vs-verified) !important;
        }
        div[class*="_mode_button_create_"] button[kind="primary"]:not(:disabled),
        div[class*="_mode_button_create_"] button[kind="primary"]:hover:not(:disabled),
        div[class*="_mode_button_create_"] button[kind="primary"]:active:not(:disabled) {
            background-color: var(--vs-verified) !important;
            border-color: var(--vs-verified) !important;
            color: var(--vs-primary-fg) !important;
        }
        div[class*="_mode_button_edit_"] button[kind="secondary"]:not(:disabled),
        div[class*="_mode_button_edit_"] button[kind="secondary"]:not(:disabled) p,
        div[class*="_mode_button_edit_"] button[kind="secondary"]:not(:disabled) [data-testid="stIconMaterial"] {
            border-color: var(--vs-action) !important;
            color: var(--vs-action) !important;
        }
        div[class*="_mode_button_edit_"] button[kind="primary"]:not(:disabled),
        div[class*="_mode_button_edit_"] button[kind="primary"]:hover:not(:disabled),
        div[class*="_mode_button_edit_"] button[kind="primary"]:active:not(:disabled) {
            background-color: var(--vs-action) !important;
            border-color: var(--vs-action) !important;
            color: var(--vs-primary-fg) !important;
        }

        div[class*="st-key-vs_source_picker"] div[data-testid="stLinkButton"] a,
        div[class*="st-key-vs_source_picker"] div[data-testid="stLinkButton"] a p,
        div[class*="st-key-vs_source_picker"] div[data-testid="stLinkButton"] a [data-testid="stIconMaterial"],
        div[class*="st-key-vs_processor_picker"] div[data-testid="stLinkButton"] a,
        div[class*="st-key-vs_processor_picker"] div[data-testid="stLinkButton"] a p,
        div[class*="st-key-vs_processor_picker"] div[data-testid="stLinkButton"] a [data-testid="stIconMaterial"] {
            border-color: var(--vs-action) !important;
            color: var(--vs-action) !important;
        }

        div[data-testid="stSegmentedControl"] button[aria-checked="false"],
        div[data-testid="stButtonGroup"] button[aria-checked="false"] {
            background: var(--vs-card) !important;
            border-color: var(--vs-border) !important;
            color: var(--vs-muted) !important;
        }

        div[data-testid="stSegmentedControl"] button[aria-checked="true"],
        div[data-testid="stButtonGroup"] button[aria-checked="true"] {
            background: var(--vs-action) !important;
            border-color: var(--vs-action) !important;
            color: var(--vs-primary-fg) !important;
        }

        div[data-testid="stSegmentedControl"] button[aria-checked="true"] p,
        div[data-testid="stButtonGroup"] button[aria-checked="true"] p,
        div[data-testid="stSegmentedControl"] button[aria-checked="true"] [data-testid="stIconMaterial"],
        div[data-testid="stButtonGroup"] button[aria-checked="true"] [data-testid="stIconMaterial"] {
            color: var(--vs-primary-fg) !important;
        }

        div[class*="st-key-vs_ai_copilot_primary"] {
            background: var(--vs-sage);
            border-color: var(--vs-action) !important;
            border-left: 0.35rem solid var(--vs-action) !important;
            border-radius: 0.8rem;
            box-shadow: 0 0.35rem 1rem var(--vs-shadow);
            margin: 0.35rem 0 1rem;
            padding: 0.25rem 0.35rem 0.15rem;
        }

        div[class*="st-key-vs_ai_copilot_primary"] [data-testid="stChatInput"] {
            background: var(--vs-card);
            border-radius: 0.7rem;
        }

        div[class*="st-key-vs_ai_copilot_primary"] h3 {
            margin-bottom: 0.15rem;
        }

        div[class*="st-key-vs_ai_sharing_consent"] {
            background: var(--vs-attention-soft);
            border: 1px solid var(--vs-attention);
            border-left-width: 0.3rem;
            border-radius: 0.7rem;
            box-shadow: 0 1px 2px var(--vs-shadow);
            padding: 0.75rem 1rem 0.7rem;
        }

        div[class*="st-key-vs_ai_sharing_consent"]:has(input:checked) {
            background: var(--vs-metric-delta-bg);
            border-color: var(--vs-verified);
        }

        div[class*="st-key-vs_ai_sharing_consent"] [data-testid="stCheckbox"] label p {
            color: var(--vs-ink) !important;
            font-weight: 650;
        }

        div[class*="st-key-vs_ai_sharing_consent"] [data-testid="stCaptionContainer"] p {
            color: var(--vs-ink) !important;
        }

        div[class*="st-key-vs_processor_picker"] {
            background: var(--vs-sage);
            border: 1px solid var(--vs-action);
            border-left-width: 0.35rem;
            border-radius: 0.8rem;
            box-shadow: 0 1px 2px var(--vs-shadow);
            margin: 0.35rem 0 0.75rem;
            padding: 0.65rem 0.85rem 0.85rem;
        }

        div[class*="st-key-vs_processor_picker"] [data-testid="stWidgetLabel"] p {
            color: var(--vs-ink) !important;
            font-weight: 650;
        }

        div[class*="st-key-vs_source_picker"] {
            background: var(--vs-sage);
            border: 1px solid var(--vs-action);
            border-left-width: 0.35rem;
            border-radius: 0.8rem;
            box-shadow: 0 1px 2px var(--vs-shadow);
            margin: 0.35rem 0 0.75rem;
            padding: 0.65rem 0.85rem 0.85rem;
        }

        div[class*="st-key-vs_source_picker"] [data-testid="stWidgetLabel"] p {
            color: var(--vs-ink) !important;
            font-weight: 650;
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
            border-color: var(--vs-input-border) !important;
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
    server_base = _active_theme_base()
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
