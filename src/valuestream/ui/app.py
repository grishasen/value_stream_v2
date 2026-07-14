"""Streamlit app entry point."""

from __future__ import annotations

from valuestream.ui.context import load_context
from valuestream.ui.shell import configure_page, parse_args, render_navigation
from valuestream.utils import logger as log_utils


def main() -> None:
    """Render the Streamlit app."""
    args = parse_args()
    log_utils.configure(config_path=args.logging_config)
    configure_page()
    ctx = load_context(args.workspace)
    render_navigation(ctx)


if __name__ == "__main__":
    main()
