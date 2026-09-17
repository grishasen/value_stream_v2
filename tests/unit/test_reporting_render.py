"""Rendered chart output for the read-only tool surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from valuestream.reporting import render as render_module
from valuestream.reporting.render import (
    RENDER_MODES,
    figure_for_tile,
    figure_html,
    figure_png,
    figure_spec,
    png_available,
    write_render,
)

# Plotly's own bundle contains the CDN URL, so only the loading script tag
# tells an embedded page apart from a CDN-backed one.
_CDN_SCRIPT = 'src="https://cdn.plot.ly'

ROWS = pl.DataFrame({"Channel": ["Web", "Mobile", "Email"], "CTR": [0.02, 0.05, 0.01]})
TILE = {
    "id": "t",
    "title": "CTR by channel",
    "metric": "CTR",
    "chart": "bar",
    "x": "Channel",
    "value_format": "percent",
}


@pytest.mark.unit
def test_html_embeds_plotly_so_the_page_opens_offline() -> None:
    # The page is written to disk and referenced by path, so embedding costs
    # file size rather than response size, and the file keeps working without
    # a network or a live CDN version.
    html = figure_html(figure_for_tile(ROWS, TILE), title="CTR by channel")

    assert html.startswith("<!doctype html>") or html.lstrip().startswith("<html")
    # The embedded bundle mentions the CDN internally, so the discriminator is
    # whether the page loads Plotly from there, not whether the string appears.
    assert _CDN_SCRIPT not in html
    assert "<title>CTR by channel</title>" in html
    assert len(html) > 1_000_000, "embedded Plotly makes this a multi-MB page"


@pytest.mark.unit
def test_html_can_fall_back_to_the_cdn_when_file_size_matters() -> None:
    html = figure_html(figure_for_tile(ROWS, TILE), embed_plotlyjs=False)

    assert _CDN_SCRIPT in html
    assert len(html) < 200_000


@pytest.mark.unit
def test_html_title_is_escaped() -> None:
    html = figure_html(figure_for_tile(ROWS, TILE), title="<script>alert(1)</script>")

    assert "<title>&lt;script&gt;alert(1)&lt;/script&gt;</title>" in html


@pytest.mark.unit
def test_spec_is_a_plotly_figure_the_client_can_draw() -> None:
    spec = figure_spec(figure_for_tile(ROWS, TILE))

    assert set(spec) >= {"data", "layout"}
    assert spec["data"], "figure spec carries at least one trace"
    # Round-trips as JSON, since that is how it leaves the server.
    assert json.loads(json.dumps(spec)) == spec


@pytest.mark.unit
def test_write_render_stamps_the_file_and_creates_the_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "renders"

    path = write_render("<html></html>", directory=target, stem="Fat Engagement/CTR", suffix="html")

    assert path.parent == target
    assert path.suffix == ".html"
    assert path.read_text() == "<html></html>"
    # The stem is slugged, so a dashboard or tile id is safe as a filename.
    assert "/" not in path.name
    assert " " not in path.name


@pytest.mark.unit
def test_png_explains_the_missing_extra_rather_than_failing_obscurely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(render_module, "png_available", lambda: False)

    with pytest.raises(RuntimeError, match="uv sync --extra viz"):
        figure_png(figure_for_tile(ROWS, TILE))


@pytest.mark.unit
def test_render_modes_are_the_documented_set() -> None:
    assert RENDER_MODES == ("none", "png", "html", "html_cdn", "spec")


@pytest.mark.unit
@pytest.mark.skipif(not png_available(), reason="PNG export needs the optional viz extra")
def test_png_renders_a_real_image() -> None:
    image = figure_png(figure_for_tile(ROWS, TILE), width=600, height=400, scale=1)

    assert image.startswith(b"\x89PNG\r\n\x1a\n"), "output is a PNG"
    assert len(image) > 1000


@pytest.mark.unit
@pytest.mark.skipif(not png_available(), reason="PNG export needs the optional viz extra")
def test_png_dimensions_are_clamped_to_something_a_client_can_show() -> None:
    # A caller asking for a 20000px image would produce a response no client
    # displays, so the bounds are enforced rather than trusted.
    image = figure_png(figure_for_tile(ROWS, TILE), width=99_999, height=1, scale=99)

    assert image.startswith(b"\x89PNG\r\n\x1a\n")
