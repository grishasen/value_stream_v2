"""Rendered chart images and interactive HTML for the read-only tool surfaces.

A tool client that asks what a chart shows often wants to see it, not only the
rows behind it. This renders a tile through the same chart factory the report
page uses, so a PNG returned over MCP is the chart an operator would see.

PNG export needs the optional ``kaleido`` package (``uv sync --extra viz``);
interactive HTML needs nothing beyond Plotly and is always available.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import polars as pl

from valuestream.charts import render_chart
from valuestream.utils.logger import get_logger

logger = get_logger(__name__)

# Bounds keep a rendered PNG readable without letting a caller ask for a
# multi-megabyte image that no client will display.
_MIN_PIXELS = 200
_MAX_PIXELS = 2400
_MAX_SCALE = 3
DEFAULT_WIDTH = 1000
DEFAULT_HEIGHT = 600
DEFAULT_SCALE = 2

RENDER_MODES = ("none", "png", "html", "html_cdn", "spec")

_PNG_HINT = (
    "PNG rendering requires the optional `viz` dependencies (kaleido). Install them with "
    "`uv sync --extra viz`, or use render=\"html\" for an interactive chart instead."
)


def png_available() -> bool:
    """Return whether the optional PNG export dependency is installed."""

    return importlib.util.find_spec("kaleido") is not None


def figure_for_tile(
    rows: pl.DataFrame,
    tile: dict[str, Any],
    *,
    theme: dict[str, Any] | None = None,
) -> Any:
    """Render a tile spec to a Plotly figure through the shared chart factory."""

    return render_chart(rows, tile, theme=theme)


def figure_png(
    figure: Any,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    scale: int = DEFAULT_SCALE,
) -> bytes:
    """Return a PNG of a figure, or explain that the extra is missing."""

    if not png_available():
        raise RuntimeError(_PNG_HINT)
    return bytes(
        figure.to_image(
            format="png",
            width=_clamp(width, _MIN_PIXELS, _MAX_PIXELS),
            height=_clamp(height, _MIN_PIXELS, _MAX_PIXELS),
            scale=_clamp(scale, 1, _MAX_SCALE),
        )
    )


def figure_html(figure: Any, *, title: str = "", embed_plotlyjs: bool = True) -> str:
    """Return an interactive HTML page for a figure.

    Plotly's JavaScript is embedded by default, so the page opens offline and
    keeps working once the CDN version it was built against moves on. That
    costs about 4 MB of disk against 13 KB for the CDN variant — which is a
    file-size question, not a response-size one, because the page is written to
    disk and only its path is returned. Pass ``embed_plotlyjs=False`` when a
    small file matters more than working without a network.
    """

    html = str(
        figure.to_html(include_plotlyjs=True if embed_plotlyjs else "cdn", full_html=True)
    )
    if title:
        html = html.replace("<head>", f"<head><title>{_escape(title)}</title>", 1)
    return html


def figure_spec(figure: Any) -> dict[str, Any]:
    """Return the Plotly figure spec, for clients that render Plotly themselves."""

    return dict(json.loads(str(figure.to_json())))


def write_render(
    content: bytes | str,
    *,
    directory: Path,
    stem: str,
    suffix: str,
) -> Path:
    """Write a rendered chart next to its siblings and return the path."""

    directory.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%S")
    path = directory / f"{_slug(stem)}-{stamp}.{suffix}"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    logger.info("Wrote rendered chart: path=%s bytes=%s", path, path.stat().st_size)
    return path


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(int(value), high))


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower() or "chart"


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


__all__ = [
    "DEFAULT_HEIGHT",
    "DEFAULT_SCALE",
    "DEFAULT_WIDTH",
    "RENDER_MODES",
    "figure_for_tile",
    "figure_html",
    "figure_png",
    "figure_spec",
    "png_available",
    "write_render",
]
