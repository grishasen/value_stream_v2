"""Phase 6 migration and legacy aggregate backfill tests."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest
import yaml
from click.testing import CliRunner

from valuestream.cli import main
from valuestream.config.loader import load
from valuestream.config.migration import migrate_toml
from valuestream.config.validate import validate_catalog
from valuestream.query import query_metric
from valuestream.store.backfill import backfill_from_legacy_db


def _legacy_engagement_toml() -> str:
    return """
variant = "demo"

[sources.ih]
reader = "parquet"
file_pattern = "data/*.parquet"
group_by_filename = "(\\\\d{8})"
timestamp_column = "OutcomeTime"
datetime_format = "%Y-%m-%d %H:%M:%S"
natural_key = ["InteractionID", "ActionID", "Rank"]

[metrics.engagement]
group_by = ["Day", "Month", "Year", "Quarter", "Channel", "PlacementType", "Issue", "Group"]
filter = '(pl.col("ModelControlGroup").is_in(["Test","Control"]))'
scores = ["CTR", "Lift", "Lift_Z_Score", "Lift_P_Val", "Positives", "Negatives", "Count"]
positive_model_response = ["Clicked"]
negative_model_response = ["Impression", "Pending"]
variant_column = "ModelControlGroup"

[reports.overview]
title = "engagement overview"
metric = "CTR"
"""


@pytest.mark.integration
def test_migrate_toml_writes_valid_catalog_and_structured_report(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.toml"
    legacy.write_text(_legacy_engagement_toml(), encoding="utf-8")
    catalog_dir = tmp_path / "workspace" / "catalog"

    report = migrate_toml(legacy, catalog_dir)

    assert report.ok
    assert {path.name for path in report.generated_files} == {
        "pipelines.yaml",
        "processors.yaml",
        "metrics.yaml",
        "dashboards.yaml",
        "migration_report.md",
    }
    catalog = load(tmp_path / "workspace")
    assert catalog.pipelines.workspace == "demo"
    ih_source = catalog.pipelines.sources[0]
    parse_datetime = next(t for t in ih_source.transforms if t.kind == "parse_datetime")
    assert parse_datetime.columns == ["OutcomeTime", "DecisionTime"]
    processor = catalog.processors.processors[0]
    assert processor.id == "engagement"
    assert processor.kind == "binary_outcome"
    assert processor.group_by == ["Channel", "PlacementType", "Issue", "Group"]
    assert processor.time is not None
    assert processor.time.grains == ["Day", "Month", "Summary"]
    assert processor.filter is not None
    assert {"CTR", "Lift"} <= set(catalog.metrics.metrics)

    report_text = (catalog_dir / "migration_report.md").read_text(encoding="utf-8")
    assert "`metrics.engagement.scores`" in report_text
    assert "`sources.ih.group_by_filename`" in report_text
    assert "_No gaps detected._" in report_text


@pytest.mark.integration
def test_migrate_toml_preserves_one_report_per_dashboard_page(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.toml"
    legacy.write_text(
        """
variant = "demo"

[ih]
file_type = "parquet"
file_pattern = "**/*.parquet"
ih_group_pattern = "Day=(.*)/"
streaming = true

[ih.extensions.default_values]
Revenue = 0.0

[metrics]
global_filters = ["Channel", "Year"]

[metrics.engagement]
group_by = ["Day", "Month", "Channel", "Issue", "Group"]
scores = ["CTR", "Count"]
positive_model_response = ["Clicked"]
negative_model_response = ["Impression", "Pending"]

[metrics.model_ml_scores]
group_by = ["Day", "Channel", "PlacementType"]
scores = ["roc_auc", "personalization"]
positive_model_response = ["Clicked"]
negative_model_response = ["Impression", "Pending"]

[reports.heatmap_month_group_ctr]
metric = "engagement"
type = "heatmap"
description = "[BIZ] Monthly CTR Heatmap"
group_by = ["Month", "Group"]
x = "Month"
y = "Group"
color = "CTR"

[reports.daily_model_roc_auc_place]
metric = "model_ml_scores"
type = "line"
description = "[ML] Daily model ROC AUC By Placement"
group_by = ["Day", "Channel", "PlacementType"]
x = "Day"
y = "roc_auc"
color = "PlacementType"
facet_row = "Channel"
facet_column = "PlacementType"
""",
        encoding="utf-8",
    )
    catalog_dir = tmp_path / "workspace" / "catalog"

    report = migrate_toml(legacy, catalog_dir)

    catalog = load(tmp_path / "workspace")
    assert validate_catalog(catalog).ok
    assert catalog.pipelines.sources[0].reader.kind == "parquet"
    assert catalog.pipelines.sources[0].reader.group_by_filename == "Day=(.*)/"
    assert not report.ok
    assert "metrics.global_filters" in {item["legacy"] for item in report.gaps}
    dashboard = catalog.dashboards.dashboards[0]
    assert [page.id for page in dashboard.pages] == [
        "heatmap_month_group_ctr",
        "daily_model_roc_auc_place",
    ]
    first_tile = dashboard.pages[0].tiles[0]
    assert first_tile.title == "[BIZ] Monthly CTR Heatmap"
    assert first_tile.metric == "CTR"
    assert first_tile.chart == "heatmap"
    assert first_tile.model_extra["group_by"] == ["Month", "Group"]
    assert first_tile.model_extra["color"] == "CTR"
    second_tile = dashboard.pages[1].tiles[0]
    assert second_tile.metric == "roc_auc"
    assert second_tile.chart == "line"
    assert second_tile.model_extra["facet_col"] == "PlacementType"


@pytest.mark.integration
def test_migrate_toml_preserves_lift_report_binding_and_experiment_name(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.toml"
    legacy.write_text(
        """
[metrics.engagement]
group_by = ["Month", "Channel"]
scores = ["CTR", "Lift", "Lift_Z_Score", "Lift_P_Val"]

[metrics.experiment]
group_by = ["Channel"]
scores = ["z_score", "g_stat", "chi2_stat"]
experiment_name = "ExperimentName"
experiment_group = "ExperimentGroup"

[reports.monthly_lift_significance]
metric = "engagement"
type = "line"
group_by = ["Month", "Channel"]
x = "Month"
y = "Lift_Z_Score"
""",
        encoding="utf-8",
    )

    migrate_toml(legacy, tmp_path / "workspace" / "catalog")

    catalog = load(tmp_path / "workspace")
    experiment = next(p for p in catalog.processors.processors if p.id == "experiment")
    tile = catalog.dashboards.dashboards[0].pages[0].tiles[0]
    assert "ExperimentName" in experiment.group_by
    assert tile.metric == "Lift"


@pytest.mark.integration
def test_migrate_cli_reports_manual_gaps_without_silently_dropping_fields(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.toml"
    legacy.write_text(
        """
variant = "demo"

[metrics.engagement]
group_by = ["Channel"]
filter = "legacy_lambda(row)"
scores = ["CTR"]
custom_weighting = "ignored_by_old_dashboard"
""",
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["migrate", "--from", str(legacy), "--to", str(tmp_path / "workspace" / "catalog")],
    )

    assert result.exit_code == 0, result.output
    assert "needs review" in result.output
    report_text = (tmp_path / "workspace" / "catalog" / "migration_report.md").read_text(
        encoding="utf-8"
    )
    assert "metrics.engagement.filter" in report_text
    assert "metrics.engagement.custom_weighting" in report_text


@pytest.mark.integration
def test_migrate_toml_handles_metrics_level_list_settings(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.toml"
    legacy.write_text(
        """
[metrics]
global_filters = ["Channel", "Year"]

[metrics.engagement]
group_by = ["Channel"]
scores = ["CTR"]
""",
        encoding="utf-8",
    )
    catalog_dir = tmp_path / "workspace" / "catalog"

    report = migrate_toml(legacy, catalog_dir)

    assert not report.ok
    catalog = load(tmp_path / "workspace")
    assert catalog.processors.processors[0].id == "engagement"
    report_text = (catalog_dir / "migration_report.md").read_text(encoding="utf-8")
    assert "metrics.global_filters" in report_text
    assert "legacy global filters need manual dashboard filter wiring" in report_text


@pytest.mark.integration
def test_migrate_cli_logs_stacktrace_when_caught_error(tmp_path: Path) -> None:
    legacy = tmp_path / "broken.toml"
    legacy.write_text("[metrics.engagement\n", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["migrate", "--from", str(legacy), "--to", str(tmp_path / "workspace" / "catalog")],
    )

    assert result.exit_code == 1
    log_output = result.stderr or result.output
    assert "migration_failed" in log_output
    assert "Traceback (most recent call last)" in log_output
    assert "TOMLDecodeError" in log_output


@pytest.mark.integration
def test_backfill_legacy_duckdb_tables_into_queryable_parquet(tmp_path: Path) -> None:
    _write_backfill_catalog(tmp_path)
    legacy_db = tmp_path / "legacy.duckdb"
    _write_legacy_db(legacy_db)

    result = backfill_from_legacy_db(tmp_path, legacy_db)

    assert result.skipped == []
    assert len(result.tables) == 1
    assert result.rows == 2
    ctr = query_metric(tmp_path, "CTR", group_by=["Channel"], grain="summary")
    web = ctr.filter(pl.col("Channel") == "Web")
    mobile = ctr.filter(pl.col("Channel") == "Mobile")
    assert web["CTR"][0] == pytest.approx(3 / 5)
    assert mobile["CTR"][0] == pytest.approx(1 / 4)


@pytest.mark.integration
def test_backfill_cli_summarizes_imported_tables(tmp_path: Path) -> None:
    _write_backfill_catalog(tmp_path)
    legacy_db = tmp_path / "legacy.duckdb"
    _write_legacy_db(legacy_db)
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["backfill", "--workspace", str(tmp_path), "--from-legacy-db", str(legacy_db)],
    )

    assert result.exit_code == 0, result.output
    assert "backfilled 1 table(s)" in result.output
    assert "aggregate_ih_engagement_summary" in result.output


@pytest.mark.integration
def test_backfill_cli_logs_stacktrace_when_caught_error(tmp_path: Path) -> None:
    legacy_db = tmp_path / "legacy.duckdb"
    legacy_db.write_bytes(b"not used because workspace is invalid")
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["backfill", "--workspace", str(tmp_path), "--from-legacy-db", str(legacy_db)],
    )

    assert result.exit_code == 1
    log_output = result.stderr or result.output
    assert "backfill_failed" in log_output
    assert "Traceback (most recent call last)" in log_output
    assert "catalog directory" in log_output


def _write_backfill_catalog(workspace: Path) -> None:
    catalog = workspace / "catalog"
    catalog.mkdir(parents=True)
    (catalog / "pipelines.yaml").write_text(
        """
version: 1
workspace: phase6_backfill
sources:
  - id: ih
    reader:
      kind: parquet
      file_pattern: "data/*.parquet"
    schema:
      timestamp_column: OutcomeTime
      natural_key: [InteractionID]
""",
        encoding="utf-8",
    )
    (catalog / "processors.yaml").write_text(
        """
processors:
  - id: engagement
    source: ih
    kind: binary_outcome
    group_by: [Channel]
    time:
      column: OutcomeTime
      grains: [Summary]
    outcome:
      column: Outcome
      positive_values: [Clicked]
      negative_values: [Impression, Pending]
    states:
      Count: {type: count}
      Positives: {type: count}
      Negatives: {type: count}
""",
        encoding="utf-8",
    )
    (catalog / "metrics.yaml").write_text(
        yaml.safe_dump(
            {
                "metrics": {
                    "CTR": {
                        "source": "engagement",
                        "kind": "formula",
                        "expression": {
                            "op": "safe_div",
                            "num": {"col": "Positives"},
                            "den": {
                                "op": "add",
                                "args": [{"col": "Positives"}, {"col": "Negatives"}],
                            },
                        },
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (catalog / "dashboards.yaml").write_text("dashboards: []\n", encoding="utf-8")


def _write_legacy_db(path: Path) -> None:
    frame = pl.DataFrame(
        {
            "Channel": ["Mobile", "Web"],
            "Count": [4, 5],
            "Positives": [1, 3],
            "Negatives": [3, 2],
        }
    )
    with duckdb.connect(str(path)) as conn:
        conn.register("legacy_summary", frame)
        conn.execute("CREATE TABLE aggregate_ih_engagement_summary AS SELECT * FROM legacy_summary")
