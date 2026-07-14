"""Reusable Streamlit UI primitives for Value Stream."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import streamlit as st

from valuestream.config.validate import CatalogIssue
from valuestream.engine import ChunkProgress, ChunkProgressCallback
from valuestream.ui import builder

STATUS_BADGES: dict[str, tuple[str, str]] = {
    "ok": ("green", ":material/check_circle:"),
    "ready": ("green", ":material/check_circle:"),
    "fresh": ("green", ":material/check_circle:"),
    "success": ("green", ":material/check_circle:"),
    "partial": ("orange", ":material/rate_review:"),
    "stale": ("orange", ":material/schedule:"),
    "warning": ("orange", ":material/warning:"),
    "failed": ("red", ":material/error:"),
    "blocked": ("red", ":material/error:"),
    "error": ("red", ":material/error:"),
    "pending": ("gray", ":material/hourglass_empty:"),
    "unknown": ("gray", ":material/help:"),
}


@dataclass(frozen=True)
class MetricItem:
    """One compact metric-strip item."""

    label: str
    value: str | int | float
    delta: str | int | float | None = None
    help: str | None = None
    chart_data: tuple[float, ...] | None = None
    chart_type: Literal["line", "bar", "area"] = "line"
    delta_description: str | None = None
    delta_color: Literal["normal", "inverse", "off"] = "normal"


def render_page_header(
    title: str,
    caption: str = "",
    *,
    status: str | None = None,
    status_label: str | None = None,
    help: str | None = None,
    status_help: str | None = None,
) -> None:
    """Render a compact page header with an optional badge."""
    if status is None:
        st.title(title, help=help)
        if caption:
            st.caption(caption)
        return

    title_col, status_col = st.columns([0.78, 0.22], vertical_alignment="center")
    with title_col:
        st.title(title, help=help)
        if caption:
            st.caption(caption)
    with status_col:
        status_badge(status_label or status.title(), status, help=status_help)


def status_badge(label: str, status: str, *, help: str | None = None) -> None:
    """Render a standard status badge."""
    color, icon = STATUS_BADGES.get(status.lower(), STATUS_BADGES["unknown"])
    st.badge(label, color=cast(Any, color), icon=icon, help=help)


def status_color(status: str) -> str:
    """Return the Streamlit badge color for a status."""
    return STATUS_BADGES.get(status.lower(), STATUS_BADGES["unknown"])[0]


@contextmanager
def card() -> Iterator[None]:
    """Render a standard Value Stream card surface.

    Streamlit 1.58 does not expose a native ``st.card`` or ``st.cards`` API.
    Keep card usage centralized so pages can move to a native card primitive
    when Streamlit adds one.
    """
    with st.container(border=True, gap="xsmall"):
        yield


def metric_cards(
    items: Iterable[MetricItem | dict[str, Any]],
    *,
    columns: int | None = None,
    key: str | None = None,
) -> None:
    """Render metric cards, optionally in a breakpoint-aware grid.

    A keyed grid receives the ``vs_metric_grid`` CSS contract from
    :mod:`valuestream.ui.theme`. Unkeyed calls retain Streamlit's native column
    behavior for compact, page-specific compositions.
    """
    normalized = [item if isinstance(item, MetricItem) else MetricItem(**item) for item in items]
    if not normalized:
        return
    column_count = columns or len(normalized)

    def render_cards() -> None:
        cols = st.columns(column_count)
        for col, item in zip(cols, normalized, strict=False):
            with col, card():
                st.metric(
                    item.label,
                    item.value,
                    item.delta,
                    delta_color=item.delta_color,
                    help=item.help,
                    chart_data=item.chart_data,
                    chart_type=item.chart_type,
                    delta_description=item.delta_description,
                )

    if key is None:
        render_cards()
        return
    safe_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", key).strip("_") or "default"
    with st.container(key=f"vs_metric_grid_{column_count}_{safe_key}"):
        render_cards()


def metric_strip(
    items: Iterable[MetricItem | dict[str, Any]],
    *,
    columns: int | None = None,
    key: str | None = None,
) -> None:
    """Render a responsive row of metric cards."""
    if key is None:
        metric_cards(items, columns=columns)
    else:
        metric_cards(items, columns=columns, key=key)


def key_value_strip(items: Iterable[MetricItem | dict[str, Any]]) -> None:
    """Render metadata with standard Streamlit table components."""
    normalized = [item if isinstance(item, MetricItem) else MetricItem(**item) for item in items]
    if not normalized:
        return
    frame = pd.DataFrame([{"Setting": item.label, "Value": item.value} for item in normalized])
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        height=min(248, 38 + 35 * len(frame)),
        column_config={
            "Setting": st.column_config.TextColumn("Setting", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="large"),
        },
    )


@contextmanager
def bordered_panel(title: str, caption: str = "") -> Iterator[None]:
    """Render a titled bordered panel."""
    with card():
        st.write(f"### {title}")
        if caption:
            st.caption(caption)
        yield


def render_validation_summary(issues: list[CatalogIssue], *, ok: bool) -> None:
    """Render catalog validation counts and details."""
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity != "error"]
    with card():
        st.write("### Catalog Health")
        cols = st.columns(3)
        cols[0].metric("Status", "OK" if ok else "Needs attention")
        cols[1].metric("Errors", len(errors))
        cols[2].metric("Warnings", len(warnings))
        if not issues:
            st.success("No validation issues found.")
            return
        with st.expander("Validation Details", expanded=not ok):
            for issue in issues:
                message = f"`{issue.location}`: {issue.message}"
                if issue.severity == "error":
                    st.error(message)
                else:
                    st.warning(message)


def dataframe_with_search(
    rows: list[dict[str, Any]] | pl.DataFrame | pd.DataFrame,
    *,
    key: str,
    search_columns: Iterable[str] | None = None,
    height: int | Literal["auto", "stretch", "content"] = "auto",
    column_order: Iterable[str] | None = None,
    column_config: dict[str, Any] | None = None,
    static: bool = False,
) -> None:
    """Render a dataframe with a lightweight text search above it.

    Polars frames stay Polars end-to-end: Streamlit >= 1.57 renders them via
    a direct Arrow conversion, so no pandas round-trip is needed.
    """
    if isinstance(rows, pl.DataFrame):
        frame = rows
    elif isinstance(rows, pd.DataFrame):
        frame = pl.from_pandas(rows)
    else:
        frame = pl.DataFrame(rows, strict=False)
    query = st.text_input("Search", key=f"{key}_search", placeholder="Filter table")
    if query and frame.height:
        columns = [
            column for column in (search_columns or frame.columns) if column in frame.columns
        ]
        matchers = [
            pl.col(column)
            .cast(pl.String, strict=False)
            .str.contains(f"(?i){re.escape(query)}")
            .fill_null(value=False)
            for column in columns
        ]
        if matchers:
            frame = frame.filter(pl.any_horizontal(matchers))
    if static:
        st.table(frame, hide_index=True, width="stretch")
        return
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        height=height,
        column_order=column_order,
        column_config=column_config,
    )


def chunk_progress_indicator(*, include_source: bool = True) -> ChunkProgressCallback:
    """Render and return a callback for live source-run chunk progress."""
    progress_bar = st.progress(0.0, text="Waiting for chunks...")
    detail = st.empty()

    def update(progress: ChunkProgress) -> None:
        if progress.chunks_total <= 0:
            return
        verb = "Skipping" if progress.status == "skipped" else "Processing"
        source = f"{progress.source_id} · " if include_source else ""
        label = (
            f"{verb} {source}chunk {progress.chunk_order}/{progress.chunks_total}: "
            f"{progress.chunk_name}"
        )
        progress_bar.progress(
            min(progress.chunk_order / progress.chunks_total, 1.0),
            text=label,
        )
        detail.caption(
            f"Chunk `{progress.chunk_name}` · order {progress.chunk_order} of "
            f"{progress.chunks_total} · {len(progress.files)} file(s)"
        )

    return update


def format_count(value: int | float | None) -> str:
    """Format counts for compact metric display."""
    if value is None:
        return "0"
    return f"{int(value):,}"


def format_compact_number(value: int | float | None) -> str:
    """Format a large summary value without clipping compact metric cards."""
    if value is None:
        return "0"
    number = float(value)
    absolute = abs(number)
    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if absolute >= threshold:
            scaled = number / threshold
            digits = 0 if abs(scaled) >= 100 else 1 if abs(scaled) >= 10 else 2
            formatted = f"{scaled:.{digits}f}"
            if "." in formatted:
                formatted = formatted.rstrip("0").rstrip(".")
            return formatted + suffix
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.2f}".rstrip("0").rstrip(".")


def format_metric_value(value: Any, value_format: str | None = None) -> str:
    """Format a metric summary using catalog semantics and compact defaults."""
    if value is None:
        return "n/a"
    normalized = str(value_format or "").strip().casefold()
    if isinstance(value, bool):
        return str(value)
    if not isinstance(value, int | float):
        return str(value)
    number = float(value)
    formatted: str | None = None
    if normalized:
        if normalized == "percent":
            formatted = f"{number:.2%}"
        elif normalized == "integer":
            formatted = f"{round(number):,}"
        elif normalized == "currency":
            formatted = f"${number:,.2f}"
        elif normalized == "number":
            formatted = f"{number:,.2f}"
    if formatted is not None:
        return formatted
    if isinstance(value, float) and abs(number) <= 1:
        return f"{number:.2%}"
    return format_compact_number(number)


def format_timestamp(value: Any) -> str:
    """Format a timestamp-like value for UI captions."""
    if value in (None, ""):
        return "not run"
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat(sep=" ", timespec="minutes"))
        except TypeError:
            return str(value.isoformat())
    return str(value)


def frame_records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    """Return display-safe records from a Polars frame."""
    return frame.to_dicts() if not frame.is_empty() else []


def sync_text_area(key: str, text: str) -> None:
    """Reset a text widget's session value whenever its source text changes."""
    signature_key = f"{key}_signature"
    if st.session_state.get(signature_key) != text:
        st.session_state[key] = text
        st.session_state[signature_key] = text


def add_default_fields_from_picker(rows_key: str, picker_key: str, editor_key: str) -> None:
    """Append picked fields to a default-values editor and reset the picker."""
    st.session_state[rows_key] = builder.default_rows_with_fields(
        st.session_state.get(rows_key, []),
        st.session_state.get(picker_key, []),
    )
    st.session_state[picker_key] = []
    st.session_state.pop(editor_key, None)
