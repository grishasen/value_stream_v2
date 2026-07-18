"""Bounded AI Studio Parquet preview qualification gates."""

from __future__ import annotations

import ctypes
import gc
import json
import math
import os
import resource
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pyarrow.parquet as pq
import pytest

from valuestream.ai import validate_draft_catalog
from valuestream.config.loader import load
from valuestream.ui import builder
from valuestream.ui.pages.ai_config_studio import _apply_draft, _read_workspace_sample

MIB = 1024 * 1024
PREVIEW_ROWS = 1_000
SYNTHETIC_CYCLES = 5
RELEASE_CYCLES = 3
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RELEASE_FIXTURE = (
    REPOSITORY_ROOT / "examples/test_ai_studio/data/Month=08/Day=2024-08-31/"
    "934be6678a7948e7b10c1cca2f5299fa-0.parquet"
)
EXPECTED_RELEASE_FIXTURE_BYTES = 280_045_584


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _current_rss_bytes() -> int:
    if sys.platform.startswith("linux"):
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    if sys.platform == "darwin":

        class MachTaskBasicInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64),
                ("user_time_seconds", ctypes.c_int32),
                ("user_time_microseconds", ctypes.c_int32),
                ("system_time_seconds", ctypes.c_int32),
                ("system_time_microseconds", ctypes.c_int32),
                ("policy", ctypes.c_int32),
                ("suspend_count", ctypes.c_int32),
            ]

        system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        system.mach_task_self.restype = ctypes.c_uint32
        system.task_info.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        info = MachTaskBasicInfo()
        count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
        status = system.task_info(
            system.mach_task_self(),
            20,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if status != 0:
            raise OSError(f"task_info failed with Mach status {status}")
        return int(info.resident_size)
    return _peak_rss_bytes()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return ordered[index]


def _logical_row_groups_for_limit(path: Path, limit: int) -> tuple[int, int]:
    metadata = pq.ParquetFile(path).metadata
    remaining = limit
    touched = 0
    for index in range(metadata.num_row_groups):
        if remaining <= 0:
            break
        touched += 1
        remaining -= metadata.row_group(index).num_rows
    return touched, metadata.num_row_groups


def _preview_profile(
    path: Path,
    *,
    columns: list[str],
    cycles: int,
) -> dict[str, Any]:
    row_groups_touched, total_row_groups = _logical_row_groups_for_limit(path, PREVIEW_ROWS)
    gc.collect()
    baseline_peak_rss = _peak_rss_bytes()
    baseline_current_rss = _current_rss_bytes()
    samples: list[dict[str, Any]] = []
    returned_rows = 0
    returned_columns: list[str] = []
    frame_bytes = 0
    for cycle in range(cycles):
        gc.collect()
        before_peak_rss = _peak_rss_bytes()
        before_current_rss = _current_rss_bytes()
        started = time.perf_counter()
        frame = _read_workspace_sample(
            path,
            limit=PREVIEW_ROWS,
            columns=columns,
        )
        elapsed_seconds = time.perf_counter() - started
        after_peak_rss = _peak_rss_bytes()
        loaded_current_rss = _current_rss_bytes()
        returned_rows = frame.height
        returned_columns = frame.columns
        frame_bytes = frame.estimated_size()
        del frame
        gc.collect()
        post_gc_current_rss = _current_rss_bytes()
        samples.append(
            {
                "cycle": cycle + 1,
                "cache_state": "cold_candidate" if cycle == 0 else "warm_candidate",
                "elapsed_seconds": elapsed_seconds,
                "peak_rss_before_bytes": before_peak_rss,
                "peak_rss_after_bytes": after_peak_rss,
                "peak_rss_delta_bytes": max(0, after_peak_rss - before_peak_rss),
                "current_rss_before_bytes": before_current_rss,
                "current_rss_loaded_bytes": loaded_current_rss,
                "post_gc_current_rss_bytes": post_gc_current_rss,
                "post_gc_retained_growth_bytes": max(
                    0,
                    post_gc_current_rss - baseline_current_rss,
                ),
                "rows": returned_rows,
                "columns": len(returned_columns),
            }
        )

    elapsed = [float(sample["elapsed_seconds"]) for sample in samples]
    return {
        "source": path.name,
        "source_bytes": path.stat().st_size,
        "requested_rows": PREVIEW_ROWS,
        "returned_rows": returned_rows,
        "selected_columns": columns,
        "returned_columns": returned_columns,
        "frame_bytes": frame_bytes,
        "dynamic_peak_rss_limit_bytes": 2 * frame_bytes + 128 * MIB,
        "preview_cycles": cycles,
        "row_groups_touched": row_groups_touched,
        "total_row_groups": total_row_groups,
        "row_group_measurement": "logical minimum from Parquet footer",
        "rss_measurement": "peak via getrusage; current via procfs or Mach task_info",
        "peak_rss_growth_bytes": max(
            0,
            max(int(sample["peak_rss_after_bytes"]) for sample in samples) - baseline_peak_rss,
        ),
        "elapsed_p50_seconds": _percentile(elapsed, 0.50),
        "elapsed_p95_seconds": _percentile(elapsed, 0.95),
        "post_gc_retained_growth_bytes": max(
            int(sample["post_gc_retained_growth_bytes"]) for sample in samples
        ),
        "cache_cycles": samples,
    }


def _canonical_authoring_draft() -> dict[str, Any]:
    metric_names = [
        "Studio_CTR",
        "Studio_Count",
        "Studio_Positive_Outcomes",
        "Studio_Negative_Outcomes",
    ]
    metrics = {
        "Studio_CTR": {
            "source": "engagement",
            "kind": "formula",
            "expression": {
                "op": "safe_div",
                "num": {"col": "Positives"},
                "den": {"col": "Count"},
            },
        },
        "Studio_Count": {
            "source": "engagement",
            "kind": "formula",
            "expression": {"col": "Count"},
        },
        "Studio_Positive_Outcomes": {
            "source": "engagement",
            "kind": "formula",
            "expression": {"col": "Positives"},
        },
        "Studio_Negative_Outcomes": {
            "source": "engagement",
            "kind": "formula",
            "expression": {"col": "Negatives"},
        },
    }
    tiles = [
        {
            "id": f"tile_{metric.casefold()}_{variant}",
            "title": f"{metric} {variant}",
            "metric": metric,
            "chart": "line" if variant == 1 else "bar",
            "x": "Day" if variant == 1 else "Channel",
            "y": metric,
            "color": "Channel" if variant == 1 else "",
        }
        for metric in metric_names
        for variant in (1, 2)
    ]
    return {
        "pipelines": {
            "version": 1,
            "workspace": "ai_studio_benchmark",
            "sources": [
                {
                    "id": "ih",
                    "reader": {"kind": "csv", "file_pattern": "*.csv"},
                    "schema": {
                        "timestamp_column": "OutcomeTime",
                        "natural_key": ["CustomerID"],
                    },
                }
            ],
        },
        "processors": {
            "processors": [
                {
                    "id": "engagement",
                    "source": "ih",
                    "kind": "binary_outcome",
                    "dimensions": ["Channel"],
                    "time": {"column": "OutcomeTime", "grains": ["Day", "Summary"]},
                    "outcome": {
                        "column": "Outcome",
                        "positive_values": ["Clicked"],
                        "negative_values": ["Impression"],
                    },
                }
            ]
        },
        "metrics": {"metrics": metrics},
        "dashboards": {
            "dashboards": [
                {
                    "id": "overview",
                    "title": "Overview",
                    "pages": [
                        {
                            "id": "outcomes",
                            "title": "Outcomes",
                            "tiles": tiles,
                        }
                    ],
                }
            ]
        },
    }


@pytest.mark.bench
def test_synthetic_multi_row_group_preview_is_bounded(
    tmp_path: Path,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    path = tmp_path / "ai-studio-preview.parquet"
    rows = 50_000
    pl.DataFrame(
        {
            "row_id": range(rows),
            "metric": [float(index % 1_001) / 10 for index in range(rows)],
            "group": [index % 17 for index in range(rows)],
            "channel": ["Web", "Email", "Mobile", "Agent"] * (rows // 4),
            "outcome": [index % 3 == 0 for index in range(rows)],
            "unused_payload": [f"payload-{index % 101:03d}" for index in range(rows)],
        }
    ).write_parquet(path, row_group_size=2_048)

    selected_columns = ["row_id", "metric", "channel"]
    profile = _preview_profile(
        path,
        columns=selected_columns,
        cycles=SYNTHETIC_CYCLES,
    )
    record_testsuite_property("ai_studio_preview_profile", json.dumps(profile, sort_keys=True))

    assert profile["returned_rows"] == PREVIEW_ROWS
    assert profile["returned_columns"] == selected_columns
    assert profile["row_groups_touched"] == 1
    assert profile["total_row_groups"] > profile["row_groups_touched"]
    assert profile["elapsed_p95_seconds"] <= 5.0
    assert profile["peak_rss_growth_bytes"] <= 256 * MIB
    assert profile["peak_rss_growth_bytes"] <= profile["dynamic_peak_rss_limit_bytes"]
    assert profile["preview_cycles"] == 5
    assert profile["post_gc_retained_growth_bytes"] <= 64 * MIB


@pytest.mark.bench
def test_canonical_draft_validation_and_apply_are_bounded(
    tmp_path: Path,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    draft = _canonical_authoring_draft()
    builder.ensure_minimum_workspace(tmp_path)

    validation_started = time.perf_counter()
    valid, issues = validate_draft_catalog(draft)
    validation_seconds = time.perf_counter() - validation_started
    assert valid, issues

    apply_started = time.perf_counter()
    _apply_draft(SimpleNamespace(workspace=tmp_path), draft)
    apply_seconds = time.perf_counter() - apply_started

    catalog = load(tmp_path)
    tile_count = sum(
        len(page.tiles) for dashboard in catalog.dashboards.dashboards for page in dashboard.pages
    )
    profile = {
        "validation_seconds": validation_seconds,
        "apply_seconds": apply_seconds,
        "sources": len(catalog.pipelines.sources),
        "processors": len(catalog.processors.processors),
        "metrics": len(catalog.metrics.metrics),
        "tiles": tile_count,
        "budget_seconds_each": 2.0,
    }
    record_testsuite_property(
        "ai_studio_validation_apply_profile",
        json.dumps(profile, sort_keys=True),
    )

    assert profile["sources"] == 1
    assert profile["processors"] == 1
    assert profile["metrics"] == 4
    assert profile["tiles"] == 8
    assert validation_seconds <= 2.0
    assert apply_seconds <= 2.0


@pytest.mark.bench
@pytest.mark.slow
def test_release_fixture_preview_profile(
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    if not RELEASE_FIXTURE.is_file():
        pytest.skip("release qualification fixture is not present")
    assert RELEASE_FIXTURE.stat().st_size == EXPECTED_RELEASE_FIXTURE_BYTES

    selected_columns = ["pxInteractionID", "pxOutcomeTime", "pyChannel"]
    profile = _preview_profile(
        RELEASE_FIXTURE,
        columns=selected_columns,
        cycles=RELEASE_CYCLES,
    )
    record_testsuite_property(
        "ai_studio_release_preview_profile",
        json.dumps(profile, sort_keys=True),
    )

    assert profile["returned_rows"] == PREVIEW_ROWS
    assert profile["returned_columns"] == selected_columns
    assert profile["row_groups_touched"] == 1
    assert profile["elapsed_p95_seconds"] <= 5.0
    assert profile["peak_rss_growth_bytes"] <= 512 * MIB
    assert profile["peak_rss_growth_bytes"] <= profile["dynamic_peak_rss_limit_bytes"]
    assert profile["post_gc_retained_growth_bytes"] <= 64 * MIB
