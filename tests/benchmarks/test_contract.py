"""Unit tests for the benchmark result contract."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import polars as pl
import pytest

from tests.benchmarks.contracts import SUITE_PROCESSORS, summarize_samples
from tests.benchmarks.run_ingestion import _prepare_workspace
from tests.benchmarks.worker import _output_contract, _update_exact_frame_digest
from valuestream.config.loader import load
from valuestream.states import cpc, tdigest


@pytest.mark.unit
def test_summarize_samples_reports_variation_and_determinism() -> None:
    samples = [
        {
            "wall_seconds": 10.0,
            "cpu_seconds": 20.0,
            "rows_per_second": 100.0,
            "cpu_seconds_per_million_rows": 200.0,
            "peak_rss_bytes": 1000,
            "output_digest": "same",
            "exact_output_digest": "same",
            "representation_digest": "raw-a",
            "approximate_state_probes": {},
            "unclassified_binary_states": [],
        },
        {
            "wall_seconds": 11.0,
            "cpu_seconds": 21.0,
            "rows_per_second": 90.0,
            "cpu_seconds_per_million_rows": 210.0,
            "peak_rss_bytes": 1200,
            "output_digest": "same",
            "exact_output_digest": "same",
            "representation_digest": "raw-b",
            "approximate_state_probes": {},
            "unclassified_binary_states": [],
        },
    ]

    summary = summarize_samples(samples)

    assert summary["samples"] == 2
    assert summary["wall_seconds_mean"] == pytest.approx(10.5)
    assert summary["wall_seconds_cv"] > 0
    assert summary["peak_rss_bytes_max"] == 1200
    assert summary["outputs_deterministic"] is True
    assert summary["outputs_equivalent"] is True
    assert summary["exact_outputs_deterministic"] is True
    assert summary["representations_stable"] is False
    assert summary["output_digest"] == "same"


@pytest.mark.unit
def test_summarize_samples_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_samples([])


@pytest.mark.unit
def test_prepare_workspace_builds_both_processor_profiles(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "examples" / "fat"
    for suite, processors in SUITE_PROCESSORS.items():
        scratch = tmp_path / suite
        _prepare_workspace(source, scratch, "ih", processors)

        catalog = load(scratch)

        actual = {
            processor.id for processor in catalog.processors.processors if processor.source == "ih"
        }
        assert actual == set(processors)
        assert catalog.metrics.metrics == {}
        assert catalog.dashboards.dashboards == []


@pytest.mark.unit
def test_output_digest_excludes_unstable_run_provenance(tmp_path: Path) -> None:
    base = tmp_path / "aggregates" / "ih" / "engagement" / "daily" / "period=2026-07"
    base.mkdir(parents=True)
    frame = pl.DataFrame(
        {
            "pipeline_run_id": ["run-a"],
            "chunk_id": ["chunk"],
            "created_at": [dt.datetime(2026, 7, 16, tzinfo=dt.UTC)],
            "period": ["2026-07"],
            "config_hash": ["hash"],
            "Count": [1],
            "state": [cpc.build(["a", "b", "c"])],
        }
    )
    frame.write_parquet(base / "part-run-a-chunk.parquet")
    state_types = {"engagement": {"state": "cpc"}}
    first = _output_contract(tmp_path, "ih", state_types=state_types)
    frame.with_columns(
        pl.lit("run-b").alias("pipeline_run_id"),
        pl.lit(dt.datetime(2026, 7, 17, tzinfo=dt.UTC)).alias("created_at"),
    ).write_parquet(base / "part-run-b-chunk.parquet")
    (base / "part-run-a-chunk.parquet").unlink()

    second = _output_contract(tmp_path, "ih", state_types=state_types)

    assert first["exact_digest"] == second["exact_digest"]
    assert first["representation_digest"] == second["representation_digest"]


@pytest.mark.unit
def test_exact_digest_normalizes_insignificant_float_noise(tmp_path: Path) -> None:
    base = tmp_path / "aggregates" / "ih" / "conversion" / "daily" / "period=2026-07"
    base.mkdir(parents=True)
    path = base / "part.parquet"
    frame = pl.DataFrame(
        {
            "chunk_id": ["chunk"],
            "period": ["2026-07"],
            "config_hash": ["hash"],
            "Revenue": [1.0],
        }
    )
    frame.write_parquet(path)
    state_types = {"conversion": {"Revenue": "value_sum"}}
    baseline = _output_contract(tmp_path, "ih", state_types=state_types)

    frame.with_columns(pl.lit(1.0 + 1e-13).alias("Revenue")).write_parquet(path)
    insignificant = _output_contract(tmp_path, "ih", state_types=state_types)
    frame.with_columns(pl.lit(1.001).alias("Revenue")).write_parquet(path)
    material = _output_contract(tmp_path, "ih", state_types=state_types)

    assert baseline["exact_digest"] == insignificant["exact_digest"]
    assert baseline["exact_digest"] != material["exact_digest"]


@pytest.mark.unit
def test_exact_digest_uses_logical_values_not_frame_buffer_layout() -> None:
    contiguous = pl.DataFrame(
        {
            "label": ["alpha", "beta", "gamma"],
            "value": [1.0, float("nan"), None],
        }
    )
    multi_chunk = pl.concat(
        [contiguous.slice(0, 1), contiguous.slice(1, 2)],
        rechunk=False,
    )
    first = hashlib.sha256()
    second = hashlib.sha256()

    _update_exact_frame_digest(first, "processor/daily/period=2026-07", contiguous)
    _update_exact_frame_digest(second, "processor/daily/period=2026-07", multi_chunk)

    assert first.hexdigest() == second.hexdigest()


@pytest.mark.unit
def test_serialized_sketch_bytes_are_diagnostic_not_correctness_gate(tmp_path: Path) -> None:
    base = tmp_path / "aggregates" / "ih" / "engagement" / "daily" / "period=2026-07"
    base.mkdir(parents=True)
    values = [f"customer-{index}" for index in range(5_000)]
    numeric_values = [((index * 7_919) % 10_007) / 10_007 for index in range(5_000)]
    frame = pl.DataFrame(
        {
            "pipeline_run_id": ["run"],
            "chunk_id": ["chunk"],
            "created_at": [dt.datetime(2026, 7, 16, tzinfo=dt.UTC)],
            "period": ["2026-07"],
            "config_hash": ["hash"],
            "Count": [len(values)],
            "UniqueCustomers_cpc": [cpc.build(values)],
            "Score_tdigest": [tdigest.build(numeric_values)],
        }
    )
    path = base / "part.parquet"
    frame.write_parquet(path)
    state_types = {
        "engagement": {
            "UniqueCustomers_cpc": "cpc",
            "Score_tdigest": "tdigest",
        }
    }
    first = _output_contract(tmp_path, "ih", state_types=state_types)

    frame.with_columns(
        pl.lit(cpc.build(reversed(values))).alias("UniqueCustomers_cpc"),
        pl.lit(tdigest.build(reversed(numeric_values))).alias("Score_tdigest"),
    ).write_parquet(path)
    second = _output_contract(tmp_path, "ih", state_types=state_types)

    assert first["exact_digest"] == second["exact_digest"]
    assert first["representation_digest"] != second["representation_digest"]
    summary = summarize_samples([_contract_sample(first), _contract_sample(second)])
    assert summary["representations_stable"] is False
    assert summary["approximate_outputs_equivalent"] is True
    assert summary["outputs_equivalent"] is True


@pytest.mark.unit
def test_approximate_contract_reports_semantic_divergence(tmp_path: Path) -> None:
    base = tmp_path / "aggregates" / "ih" / "engagement" / "daily" / "period=2026-07"
    base.mkdir(parents=True)
    pl.DataFrame(
        {
            "chunk_id": ["chunk"],
            "period": ["2026-07"],
            "config_hash": ["hash"],
            "Count": [1],
            "UniqueCustomers_cpc": [cpc.build(["customer"])],
        }
    ).write_parquet(base / "part.parquet")
    output = _output_contract(
        tmp_path,
        "ih",
        state_types={"engagement": {"UniqueCustomers_cpc": "cpc"}},
    )
    changed = deepcopy(output)
    probes = changed["approximate_state_probes"]["engagement.UniqueCustomers_cpc"]["samples"][0][
        "probes"
    ]
    probes.update({"estimate": 10_000.0, "lower": 9_000.0, "upper": 11_000.0})

    summary = summarize_samples([_contract_sample(output), _contract_sample(changed)])

    assert summary["exact_outputs_deterministic"] is True
    assert summary["approximate_outputs_equivalent"] is False
    assert summary["outputs_equivalent"] is False
    assert summary["approximate_comparison_issues"]


@pytest.mark.integration
def test_benchmark_worker_runs_in_clean_scratch_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    catalog = workspace / "catalog"
    data = workspace / "data"
    catalog.mkdir(parents=True)
    data.mkdir()
    (catalog / "pipelines.yaml").write_text(
        """
version: 1
workspace: benchmark_test
sources:
  - id: ih
    reader:
      kind: parquet
      root: data
      file_pattern: "*.parquet"
      group_by_filename: '(\\d{8})'
    schema:
      timestamp_column: OutcomeTime
      natural_key: [InteractionID, ActionID, Rank]
    transforms:
      - kind: derive_calendar
        from: OutcomeTime
        outputs: [Day, Month]
""",
        encoding="utf-8",
    )
    processors = []
    for processor_id in SUITE_PROCESSORS["full_current"]:
        processors.append(
            f"""
  - id: {processor_id}
    source: ih
    kind: binary_outcome
    group_by: [Channel]
    time:
      column: OutcomeTime
      grains: [Day]
    outcome:
      column: Outcome
      positive_values: [Clicked]
      negative_values: [Impression]
    states:
      Count: {{type: count}}
      Positives: {{type: count}}
      Negatives: {{type: count}}
      UniqueCustomers_cpc: {{type: cpc, source_column: CustomerID, lg_k: 11}}
"""
        )
    (catalog / "processors.yaml").write_text(
        "processors:\n" + "".join(processors), encoding="utf-8"
    )
    (catalog / "metrics.yaml").write_text("metrics: {}\n", encoding="utf-8")
    (catalog / "dashboards.yaml").write_text("dashboards: []\n", encoding="utf-8")
    row_count = 512
    pl.DataFrame(
        {
            "OutcomeTime": [dt.datetime(2026, 7, 16, tzinfo=dt.UTC)] * row_count,
            "InteractionID": [f"i{index}" for index in range(row_count)],
            "ActionID": [f"a{index}" for index in range(row_count)],
            "CustomerID": [f"c{index}" for index in range(row_count)],
            "Rank": [1] * row_count,
            "Channel": ["Web"] * row_count,
            "Outcome": ["Clicked"] * row_count,
        }
    ).write_parquet(data / "ih_20260716.parquet")

    output = tmp_path / "benchmark.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.benchmarks.run_ingestion",
            "--workspace",
            str(workspace),
            "--source",
            "ih",
            "--suite",
            "full_current",
            "--warmups",
            "0",
            "--repeats",
            "2",
            "--parallel",
            "1",
            "--scratch-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    sample = payload["suites"]["full_current"]["samples"][0]

    assert sample["rows_in"] == row_count
    assert sample["chunks_ok"] == 1
    assert sample["aggregate_files"] == len(SUITE_PROCESSORS["full_current"])
    assert sample["output_digest"]
    assert sample["exact_output_digest"] == sample["output_digest"]
    assert payload["suites"]["full_current"]["summary"]["outputs_equivalent"] is True


def _contract_sample(output: dict[str, object]) -> dict[str, object]:
    return {
        "wall_seconds": 1.0,
        "cpu_seconds": 1.0,
        "rows_per_second": 1.0,
        "cpu_seconds_per_million_rows": 1.0,
        "peak_rss_bytes": 1,
        "output_digest": output["exact_digest"],
        "exact_output_digest": output["exact_digest"],
        "representation_digest": output["representation_digest"],
        "approximate_state_probes": output["approximate_state_probes"],
        "unclassified_binary_states": output["unclassified_binary_states"],
    }
