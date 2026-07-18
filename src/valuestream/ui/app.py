"""Streamlit app entry point."""

from __future__ import annotations

import os

# Arrow's bundled mimalloc segfaults during per-thread heap initialization
# (mi_thread_init -> mi_heap_main, pyarrow 25.0 on macOS arm64) when
# Table.to_pandas runs on Streamlit's short-lived script-runner threads —
# every dataframe/data_editor render of a polars frame takes that path. The
# variable is read lazily on the first Arrow allocation, so it must be set
# before any page code runs. Operators can still override the pool choice.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

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
