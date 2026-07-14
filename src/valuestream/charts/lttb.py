"""Largest-triangle-three-buckets downsampling for line charts."""

from __future__ import annotations

from typing import Any

import polars as pl


def downsample(frame: pl.DataFrame, *, x: str, y: str, threshold: int) -> pl.DataFrame:
    """Return at most ``threshold`` rows, preserving line-chart shape."""
    if frame.height <= threshold or threshold < 3:
        return frame
    if x not in frame.columns or y not in frame.columns:
        return frame.head(threshold)

    working = frame.sort(x)
    xs = _numeric_axis(working[x].to_list())
    ys = [float(value) if value is not None else 0.0 for value in working[y].to_list()]
    every = (len(xs) - 2) / (threshold - 2)
    selected = [0]
    a = 0

    for i in range(threshold - 2):
        start = int((i + 1) * every) + 1
        end = int((i + 2) * every) + 1
        end = min(end, len(xs))
        next_bucket = range(start, end)
        if not next_bucket:
            continue
        avg_x = sum(xs[idx] for idx in next_bucket) / len(next_bucket)
        avg_y = sum(ys[idx] for idx in next_bucket) / len(next_bucket)

        range_start = int(i * every) + 1
        range_end = int((i + 1) * every) + 1
        candidates = range(range_start, min(range_end, len(xs) - 1))
        best_idx = range_start
        best_area = -1.0
        for idx in candidates:
            area = abs((xs[a] - avg_x) * (ys[idx] - ys[a]) - (xs[a] - xs[idx]) * (avg_y - ys[a]))
            if area > best_area:
                best_area = area
                best_idx = idx
        selected.append(best_idx)
        a = best_idx

    selected.append(len(xs) - 1)
    return working[selected]


def _numeric_axis(values: list[Any]) -> list[float]:
    out: list[float] = []
    for index, value in enumerate(values):
        if hasattr(value, "timestamp"):
            out.append(float(value.timestamp()))
        elif hasattr(value, "toordinal"):
            out.append(float(value.toordinal()))
        else:
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                out.append(float(index))
    return out


__all__ = ["downsample"]
