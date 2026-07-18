"""Headless deterministic AI Studio release-journey qualification."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import polars as pl
import pyarrow.parquet as pq
import pytest

from valuestream.ai import (
    apply_draft_operations,
    draft_patches,
    tile_keys,
    validate_draft_catalog,
)
from valuestream.ai.chat import (
    deterministic_chat_starters,
    execute_deterministic_chat_query,
)
from valuestream.config.canonical import catalog_config_hash, processor_computation_hash
from valuestream.config.loader import load
from valuestream.engine import clean_rebuild, run_source
from valuestream.query import query_metric, query_metric_result
from valuestream.ui import builder
from valuestream.ui.pages import ai_config_studio as studio

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FULL_FIXTURE_RELATIVE = Path(
    "examples/test_ai_studio/data/Month=08/Day=2024-08-31/"
    "934be6678a7948e7b10c1cca2f5299fa-0.parquet"
)
FULL_FIXTURE_SHA256 = "ff54074bfffff95b38f8768a23671fdf9b9b443698f5bab275d82a858e4247c5"
FULL_FIXTURE_ROWS = 2_733_856
FULL_FIXTURE_OUTCOMES = {
    "Impression": 1_951_721,
    "NoConversion": 704_751,
    "Clicked": 48_810,
    "Conversion": 28_574,
}

COMPACT_ROWS = 2_400
COMPACT_OUTCOME_PATTERN = (
    *(["Impression"] * 10),
    *(["NoConversion"] * 5),
    *(["Clicked"] * 3),
    *(["Conversion"] * 2),
)
COMPACT_OUTCOMES = {
    "Impression": 1_200,
    "NoConversion": 600,
    "Clicked": 360,
    "Conversion": 240,
}
COMPACT_CHANNELS = ("Web", "Mobile", "Email")
EXPECTED_CTR = 360 / COMPACT_ROWS
EXPECTED_REACH = 900


def _pega_timestamp(value: dt.datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%S.") + f"{value.microsecond // 1_000:03d} GMT"


def _write_compact_production_fixture(workspace: Path) -> Path:
    relative = Path("data/Month=08/Day=2024-08-31/compact-ai-studio-20240831000000.parquet")
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    base = dt.datetime(2024, 8, 31, 8, 0)
    outcome_times = [base + dt.timedelta(seconds=index) for index in range(COMPACT_ROWS)]
    decision_times = [
        value - dt.timedelta(seconds=(index % 60) + 1) for index, value in enumerate(outcome_times)
    ]
    outcomes = [
        COMPACT_OUTCOME_PATTERN[index % len(COMPACT_OUTCOME_PATTERN)]
        for index in range(COMPACT_ROWS)
    ]
    channels = [COMPACT_CHANNELS[index % len(COMPACT_CHANNELS)] for index in range(COMPACT_ROWS)]
    propensities = [
        None if index % 11 == 0 else ((index % 997) + 1) / 1_000 for index in range(COMPACT_ROWS)
    ]
    subject_ids = [f"C-{index % EXPECTED_REACH:04d}" for index in range(COMPACT_ROWS)]
    interaction_ids = [f"I-{index:06d}" for index in range(COMPACT_ROWS)]
    treatments = [f"treatment-{index:06d}" for index in range(COMPACT_ROWS)]
    frame = pl.DataFrame(
        {
            "ModelControlGroup": [
                "Test" if index % 5 else "Control" for index in range(COMPACT_ROWS)
            ],
            "pxInteractionID": interaction_ids,
            "InteractionID": interaction_ids,
            "pxFactID": [f"F-{index:06d}" for index in range(COMPACT_ROWS)],
            "pxOutcomeTime": [_pega_timestamp(value) for value in outcome_times],
            "OutcomeTime": [_pega_timestamp(value) for value in outcome_times],
            "pxDecisionTime": [_pega_timestamp(value) for value in decision_times],
            "DecisionTime": [_pega_timestamp(value) for value in decision_times],
            "CustomerID": subject_ids,
            "pySubjectID": subject_ids,
            "SubjectID": subject_ids,
            "pyName": [f"Offer {index % 23}" for index in range(COMPACT_ROWS)],
            "Name": [f"Offer {index % 23}" for index in range(COMPACT_ROWS)],
            "pyChannel": channels,
            "Channel": channels,
            "pyDirection": [
                "Inbound" if index % 2 else "Outbound" for index in range(COMPACT_ROWS)
            ],
            "Direction": ["Inbound" if index % 2 else "Outbound" for index in range(COMPACT_ROWS)],
            "pyIssue": ["Sales" if index % 2 else "Service" for index in range(COMPACT_ROWS)],
            "Issue": ["Sales" if index % 2 else "Service" for index in range(COMPACT_ROWS)],
            "pyGroup": ["Cards" if index % 2 else "Loans" for index in range(COMPACT_ROWS)],
            "Group": ["Cards" if index % 2 else "Loans" for index in range(COMPACT_ROWS)],
            "pyTreatment": treatments,
            "Treatment": treatments,
            "pyPropensity": propensities,
            "Propensity": propensities,
            "FinalPropensity": [((index % 991) + 1) / 1_000 for index in range(COMPACT_ROWS)],
            "pyOutcome": outcomes,
            "Outcome": outcomes,
            "pxRank": [(index % 5) + 1 for index in range(COMPACT_ROWS)],
            "Rank": [(index % 5) + 1 for index in range(COMPACT_ROWS)],
            "PlacementType": [None if index % 17 == 0 else "Hero" for index in range(COMPACT_ROWS)],
        }
    )
    frame.write_parquet(path, row_group_size=256)
    return path


def _configure_no_provider_studio(
    workspace: Path,
    sample_path: Path,
    preview: pl.DataFrame,
) -> pl.DataFrame:
    relative = sample_path.relative_to(workspace).as_posix()
    plan = studio._sample_source_plan(
        sample_path.name,
        preview.columns,
        workspace_relative=relative,
    )
    assert plan.reader_kind == "parquet"
    assert plan.timestamp_format == "%Y%m%dT%H%M%S%.3f %Z"
    studio.st.session_state.clear()
    studio.st.session_state.update(
        {
            studio.AI_CALLS_ENABLED_STATE_KEY: False,
            "ai_studio_active_workspace_name": "ai_studio_release",
            # Use the canonical Pega source id while retaining the preview-inferred
            # reader contract. The generic Parquet preview intentionally suggests
            # ``sample`` because source naming is an authoring choice.
            "ai_studio_source_id": "ih",
            "ai_studio_reader_kind": plan.reader_kind,
            "ai_studio_reader_root": plan.root,
            "ai_studio_file_pattern": plan.file_pattern,
            "ai_studio_group_pattern": plan.group_pattern,
            "ai_studio_streaming": False,
            "ai_studio_hive_partitioning": False,
            "ai_studio_timestamp_format": plan.timestamp_format,
            "ai_studio_rename_capitalize": False,
            "ai_studio_subject": "SubjectID",
            "ai_studio_outcome_time": "OutcomeTime",
            "ai_studio_decision_time": "DecisionTime",
            "ai_studio_day_column": "",
            "ai_studio_month_column": "",
            "ai_studio_quarter_column": "",
            "ai_studio_year_column": "",
            "ai_studio_outcome_column": "Outcome",
            "ai_studio_defaults": [],
            "ai_studio_filter_mode": "Rules",
            "ai_studio_filter_rows": [],
            "ai_studio_calculations": [],
            "ai_studio_group_by_fields": [
                "Channel",
                "Direction",
                "Issue",
                "Group",
                "ModelControlGroup",
            ],
        }
    )
    schema = studio._schema_sample(preview)
    working, error = studio._working_sample(schema)
    assert error is None
    return working


def _mutate_draft(base: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    source_id = base["pipelines"]["sources"][0]["id"]
    processor = copy.deepcopy(base["processors"]["processors"][0])
    processor["states"] = {
        "Count": {"type": "count"},
        "Positives": {"type": "count"},
        "Negatives": {"type": "count"},
        "UniqueSubjects_cpc": {
            "type": "cpc",
            "source_column": "SubjectID",
            "lg_k": 11,
        },
    }
    dashboard = base["dashboards"]["dashboards"][0]
    outcomes_page = next(page for page in dashboard["pages"] if page["title"] == "Outcomes")
    scratch_tile = {
        "id": "scratch_count",
        "title": "Scratch count",
        "metric": "Studio_Count",
        "chart": "bar",
        "x": "Channel",
        "y": "Studio_Count",
    }
    operations = [
        {
            "op": "set_source_default",
            "source": source_id,
            "field": "PlacementType",
            "value": "Unknown",
        },
        {
            "op": "set_source_default",
            "source": source_id,
            "field": "PlacementType",
            "value": "N/A",
        },
        {
            "op": "set_source_default",
            "source": source_id,
            "field": "Propensity",
            "value": 0.0,
        },
        {
            "op": "set_source_default",
            "source": source_id,
            "field": "TemporaryDefault",
            "value": 1,
        },
        {
            "op": "remove_source_default",
            "source": source_id,
            "field": "TemporaryDefault",
        },
        {
            "op": "set_source_filter",
            "source": source_id,
            "expression": {"op": "not_null", "column": "Channel"},
        },
        {
            "op": "set_source_filter",
            "source": source_id,
            "expression": {
                "op": "in",
                "column": "Channel",
                "values": list(COMPACT_CHANNELS),
            },
        },
        {"op": "remove_source_filter", "source": source_id},
        {
            "op": "set_source_filter",
            "source": source_id,
            "expression": {"op": "not_null", "column": "Outcome"},
        },
        {
            "op": "set_calculated_field",
            "source": source_id,
            "name": "IsClicked",
            "expression": {"op": "eq", "column": "Outcome", "value": "Clicked"},
        },
        {
            "op": "set_calculated_field",
            "source": source_id,
            "name": "PropensityPct",
            "expression": {
                "op": "mul",
                "args": [{"col": "Propensity"}, {"lit": 100.0}],
            },
        },
        {
            "op": "set_calculated_field",
            "source": source_id,
            "name": "ResponseSeconds",
            "expression": {
                "op": "date_diff",
                "unit": "seconds",
                "end": {"col": "OutcomeTime"},
                "start": {"col": "DecisionTime"},
            },
        },
        {
            "op": "set_calculated_field",
            "source": source_id,
            "name": "ScratchCalculation",
            "expression": {"lit": 1},
        },
        {
            "op": "set_calculated_field",
            "source": source_id,
            "name": "ScratchCalculation",
            "expression": {"lit": 2},
        },
        {
            "op": "remove_calculated_field",
            "source": source_id,
            "name": "ScratchCalculation",
        },
        {
            "op": "set_processor",
            "previous_id": processor["id"],
            "processor": processor,
        },
        {"op": "remove_metric", "name": "Studio_Negative_Outcomes"},
        {
            "op": "install_recipe",
            "recipe_id": "audience.unique_entities",
            "processor": processor["id"],
            "metric_id": "VS_Unique_Entities",
            "bindings": {"cardinality_state": "UniqueSubjects_cpc"},
            "dashboard": dashboard["id"],
            "page": outcomes_page["id"],
            "tile_id": "unique_entities",
        },
        {
            "op": "set_metric",
            "name": "Scratch_Count",
            "metric": {
                "source": processor["id"],
                "kind": "formula",
                "expression": {"col": "Count"},
            },
        },
        {
            "op": "set_metric",
            "name": "Scratch_Count",
            "metric": {
                "source": processor["id"],
                "kind": "formula",
                "expression": {"col": "Positives"},
            },
        },
        {"op": "remove_metric", "name": "Scratch_Count"},
        {
            "op": "set_tile",
            "dashboard": dashboard["id"],
            "page": outcomes_page["id"],
            "tile": scratch_tile,
        },
        {
            "op": "set_tile",
            "dashboard": dashboard["id"],
            "page": outcomes_page["id"],
            "tile": {**scratch_tile, "title": "Modified scratch count"},
        },
        {
            "op": "remove_tile",
            "dashboard": dashboard["id"],
            "page": outcomes_page["id"],
            "id": scratch_tile["id"],
        },
    ]
    return apply_draft_operations(base, operations)


def _assert_reference_metrics(workspace: Path) -> dict[str, pl.DataFrame]:
    overall_count = query_metric(workspace, "Studio_Count", grain="summary")
    overall_ctr = query_metric(workspace, "Studio_CTR", grain="summary")
    overall_reach = query_metric(workspace, "VS_Unique_Entities", grain="summary")
    channel_count = query_metric(
        workspace,
        "Studio_Count",
        group_by=["Channel"],
        grain="summary",
    ).sort("Channel")
    channel_ctr = query_metric(
        workspace,
        "Studio_CTR",
        group_by=["Channel"],
        grain="summary",
    ).sort("Channel")
    channel_reach = query_metric(
        workspace,
        "VS_Unique_Entities",
        group_by=["Channel"],
        grain="summary",
    ).sort("Channel")

    assert overall_count["Studio_Count"].to_list() == [COMPACT_ROWS]
    assert overall_ctr["Studio_CTR"].to_list() == [pytest.approx(EXPECTED_CTR)]
    assert overall_reach["VS_Unique_Entities"][0] == pytest.approx(EXPECTED_REACH, rel=0.03)
    assert channel_count["Studio_Count"].to_list() == [800, 800, 800]
    assert channel_ctr["Studio_CTR"].to_list() == [
        pytest.approx(EXPECTED_CTR),
        pytest.approx(EXPECTED_CTR),
        pytest.approx(EXPECTED_CTR),
    ]
    assert channel_reach["VS_Unique_Entities"].to_list() == pytest.approx(
        [300, 300, 300],
        rel=0.04,
    )
    return {
        "count": overall_count,
        "ctr": overall_ctr,
        "reach": overall_reach,
        "channel_count": channel_count,
        "channel_ctr": channel_ctr,
        "channel_reach": channel_reach,
    }


@pytest.mark.e2e
@pytest.mark.integration
def test_no_provider_release_journey_is_executable_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(studio.st, "session_state", {})
    workspace = tmp_path / "ai-studio-release"
    builder.ensure_minimum_workspace(workspace)
    sample_path = _write_compact_production_fixture(workspace)
    metadata = pq.ParquetFile(sample_path).metadata
    assert metadata.num_rows == COMPACT_ROWS
    assert metadata.num_row_groups > 1

    source_profile = (
        pl.scan_parquet(sample_path)
        .select(
            pl.col("Outcome").value_counts(sort=True).alias("outcomes"),
            pl.col("Propensity").null_count().alias("null_propensity"),
            pl.col("Treatment").n_unique().alias("unique_treatments"),
        )
        .collect()
    )
    assert (
        dict(
            zip(
                source_profile["outcomes"].struct.field("Outcome").to_list(),
                source_profile["outcomes"].struct.field("count").to_list(),
                strict=True,
            )
        )
        == COMPACT_OUTCOMES
    )
    assert source_profile["null_propensity"][0] > 0
    assert source_profile["unique_treatments"][0] == COMPACT_ROWS

    preview = studio._read_workspace_sample(sample_path, limit=1_000)
    assert preview.height == 1_000
    assert {
        "Outcome",
        "pyOutcome",
        "pxOutcomeTime",
        "OutcomeTime",
        "DecisionTime",
        "Propensity",
        "Channel",
        "Direction",
        "Issue",
        "Group",
        "ModelControlGroup",
        "Treatment",
    } <= set(preview.columns)
    assert studio._default_outcome_column(preview) == "Outcome"
    assert studio._default_time_column(preview.columns, "OutcomeTime") == "OutcomeTime"
    assert studio._default_time_column(preview.columns, "DecisionTime") == "DecisionTime"

    try:
        working = _configure_no_provider_studio(workspace, sample_path, preview)
        assert {"Day", "Month", "Quarter", "Year", "ResponseTime"} <= set(working.columns)
        assert working["Propensity"].null_count() > 0
        base = studio._build_draft_catalog(working, working.columns)
        valid, issues = validate_draft_catalog(base)
        assert valid, issues

        draft, summaries = _mutate_draft(base)
        assert any(summary.startswith("Added") for summary in summaries)
        assert any(summary.startswith("Updated") for summary in summaries)
        assert any(summary.startswith("Removed") for summary in summaries)
        patches = draft_patches(base, draft)
        assert {patch.section for patch in patches} >= {
            "source_defaults",
            "source_filters",
            "calculated_fields",
            "processors",
            "metrics",
            "tiles",
        }

        source = draft["pipelines"]["sources"][0]
        assert source["defaults"] == {"PlacementType": "N/A", "Propensity": 0.0}
        derived_outputs = {
            transform["output"]
            for transform in source["transforms"]
            if transform.get("kind") == "derive_column"
        }
        assert {"IsClicked", "PropensityPct", "ResponseSeconds"} <= derived_outputs
        assert "ScratchCalculation" not in derived_outputs
        assert len([item for item in source["transforms"] if item.get("kind") == "filter"]) == 1

        metrics = draft["metrics"]["metrics"]
        assert len(metrics) == 4
        assert "Studio_Negative_Outcomes" not in metrics
        assert metrics["VS_Unique_Entities"]["recipe"] == {
            "id": "audience.unique_entities",
            "version": 1,
        }
        dashboard = draft["dashboards"]["dashboards"][0]
        assert len(dashboard["pages"]) == 3
        assert len(tile_keys(draft)) >= 6
        assert {page["title"] for page in dashboard["pages"]} == {
            "Engagement",
            "Volume",
            "Outcomes",
        }
        valid, issues = validate_draft_catalog(draft)
        assert valid, issues

        studio._apply_draft(SimpleNamespace(workspace=workspace), draft)
    finally:
        studio.st.session_state.clear()

    applied = load(workspace)
    assert len(applied.pipelines.sources) == 1
    assert len(applied.processors.processors) == 1
    assert len(applied.metrics.metrics) == 4

    first = run_source(workspace, "ih")
    assert first.status == "ok"
    assert first.chunks_total == 1
    assert first.chunks_ok == 1
    assert first.chunks_skipped == 0
    assert first.rows_in == COMPACT_ROWS
    assert first.rows_kept == COMPACT_ROWS
    before_metrics = _assert_reference_metrics(workspace)

    immediate = run_source(workspace, "ih")
    assert immediate.status == "ok"
    assert immediate.chunks_ok == 0
    assert immediate.chunks_skipped == 1
    assert immediate.rows_in == 0
    assert immediate.rows_kept == 0

    before_files = set((workspace / "aggregates").glob("*/*/*/period=*/*.parquet"))
    rebuild = clean_rebuild(workspace, source_ids=["ih"])
    after_files = set((workspace / "aggregates").glob("*/*/*/period=*/*.parquet"))
    assert rebuild.source_ids == ("ih",)
    assert rebuild.chunks_rebuilt == 1
    assert rebuild.vacuum.files_deleted == len(before_files)
    assert before_files.isdisjoint(after_files)
    assert len(after_files) == len(before_files)
    after_metrics = _assert_reference_metrics(workspace)
    for name in before_metrics:
        assert before_metrics[name].to_dicts() == after_metrics[name].to_dicts()

    applied = load(workspace)
    processor = applied.processors.processors[0]
    expected_catalog_hash = catalog_config_hash(applied)
    expected_computation_hash = processor_computation_hash(applied, processor)
    query_result = query_metric_result(workspace, "Studio_CTR", grain="summary")
    assert query_result.provenance.catalog_hash == expected_catalog_hash
    assert query_result.provenance.computation_hash == expected_computation_hash
    assert query_result.provenance.pipeline_run_ids == (rebuild.runs[0].run_id,)
    assert query_result.provenance.chunk_ids

    forbidden_raw_columns = {
        "CustomerID",
        "SubjectID",
        "InteractionID",
        "Treatment",
        "Outcome",
        "Propensity",
        "DecisionTime",
        "IsClicked",
        "PropensityPct",
        "ResponseSeconds",
        "ResponseTime",
    }
    for aggregate_path in after_files:
        schema = set(pl.scan_parquet(aggregate_path).collect_schema().names())
        assert {
            "pipeline_run_id",
            "chunk_id",
            "period",
            "created_at",
            "config_hash",
        } <= schema
        assert forbidden_raw_columns.isdisjoint(schema)
        provenance = (
            pl.scan_parquet(aggregate_path)
            .select("pipeline_run_id", "config_hash")
            .unique()
            .collect()
        )
        assert provenance["pipeline_run_id"].to_list() == [rebuild.runs[0].run_id]
        assert provenance["config_hash"].to_list() == [expected_computation_hash]

    with duckdb.connect(str(workspace / "meta" / "lineage.duckdb"), read_only=True) as connection:
        lineage = connection.execute(
            """
            SELECT partial_path, config_hash, rows
            FROM lineage
            WHERE CAST(pipeline_run_id AS VARCHAR) = ?
            ORDER BY partial_path
            """,
            [rebuild.runs[0].run_id],
        ).fetchall()
    assert {Path(path) for path, _, _ in lineage} == after_files
    assert all(
        config_hash == expected_computation_hash and rows > 0 for _, config_hash, rows in lineage
    )

    with duckdb.connect(
        str(workspace / "meta" / "config_versions.duckdb"),
        read_only=True,
    ) as connection:
        config_hashes = {
            row[0]
            for row in connection.execute("SELECT config_hash FROM config_versions").fetchall()
        }
    assert expected_computation_hash in config_hashes

    starters = deterministic_chat_starters(applied)
    assert [starter.key for starter in starters] == [
        "count",
        "rate",
        "unique",
        "channel",
        "date_range",
    ]
    chat_results = {
        starter.key: execute_deterministic_chat_query(workspace, applied, starter)
        for starter in starters
    }
    assert all(result.query_summary.startswith("query_metric(") for result in chat_results.values())
    assert all(result.freshness for result in chat_results.values())
    assert chat_results["count"].rows.height == 1
    assert chat_results["rate"].rows.height == 1
    assert chat_results["unique"].rows.height == 1
    assert chat_results["channel"].rows.height == len(COMPACT_CHANNELS)
    assert chat_results["date_range"].rows.to_dicts() == [
        {
            "Available from": "2024-08-31",
            "Available through": "2024-08-31",
            "Grain": "daily",
        }
    ]


@pytest.mark.e2e
@pytest.mark.slow
def test_full_release_fixture_manifest_is_exact_and_read_only() -> None:
    path = REPOSITORY_ROOT / FULL_FIXTURE_RELATIVE
    if not path.is_file():
        pytest.skip("canonical AI Studio release fixture is not present")
    before = path.stat()

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    metadata = pq.ParquetFile(path).metadata
    outcome_rows = pl.scan_parquet(path).group_by("pyOutcome").len().sort("pyOutcome").collect()

    after = path.stat()
    assert path.relative_to(REPOSITORY_ROOT) == FULL_FIXTURE_RELATIVE
    assert digest.hexdigest() == FULL_FIXTURE_SHA256
    assert metadata.num_rows == FULL_FIXTURE_ROWS
    assert (
        dict(zip(outcome_rows["pyOutcome"], outcome_rows["len"], strict=True))
        == FULL_FIXTURE_OUTCOMES
    )
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
