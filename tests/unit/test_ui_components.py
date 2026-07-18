"""Focused tests for reusable Streamlit UI components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from valuestream.engine import ChunkProgress
from valuestream.ui import components


class _ProgressRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[float, str | None]] = []

    def progress(self, value: float, *, text: str | None = None) -> None:
        self.calls.append((value, text))


class _CaptionRecorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def caption(self, text: str) -> None:
        self.calls.append(text)


@pytest.mark.unit
def test_chunk_progress_indicator_shows_monotonic_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_bar = _ProgressRecorder()
    detail = _CaptionRecorder()
    clock = iter([100.0, 165.9])
    initial: dict[str, Any] = {}

    def progress(value: float, *, text: str | None = None) -> _ProgressRecorder:
        initial.update(value=value, text=text)
        return progress_bar

    monkeypatch.setattr(components.st, "progress", progress)
    monkeypatch.setattr(components.st, "empty", lambda: detail)
    monkeypatch.setattr(components.time, "perf_counter", lambda: next(clock))

    update = components.chunk_progress_indicator(include_source=False)
    update(
        ChunkProgress(
            source_id="ih",
            chunk_id="2024-08-30",
            chunk_name="2024-08-30",
            chunk_order=20,
            chunks_total=799,
            status="processing",
            files=(Path("chunk.parquet"),),
        )
    )

    assert initial == {"value": 0.0, "text": "Waiting for chunks... · Elapsed 00:00:00"}
    assert progress_bar.calls == [
        (
            20 / 799,
            "Processing chunk 20/799: 2024-08-30 · Elapsed 00:01:05",
        )
    ]
    assert detail.calls == ["Chunk `2024-08-30` · order 20 of 799 · 1 file(s)"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (-1.0, "00:00:00"),
        (3_661.9, "01:01:01"),
        (360_000.0, "100:00:00"),
    ],
)
def test_format_elapsed(seconds: float, expected: str) -> None:
    assert components._format_elapsed(seconds) == expected


@pytest.mark.unit
def test_key_value_strip_normalizes_mixed_values_for_arrow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def dataframe(frame: pd.DataFrame, **kwargs: Any) -> None:
        captured.update(frame=frame, kwargs=kwargs)

    monkeypatch.setattr(components.st, "dataframe", dataframe)

    components.key_value_strip(
        [
            {"label": "Status", "value": "Ready"},
            {"label": "Sources", "value": 2},
            {"label": "Ratio", "value": 0.75},
        ]
    )

    assert captured["frame"].to_dict(orient="records") == [
        {"Setting": "Status", "Value": "Ready"},
        {"Setting": "Sources", "Value": "2"},
        {"Setting": "Ratio", "Value": "0.75"},
    ]
    assert all(str(dtype) == "string" for dtype in captured["frame"].dtypes)
