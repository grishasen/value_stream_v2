"""Phase 2 vertical slice: numeric + score distributions -> derived metrics."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from valuestream.engine import run_source
from valuestream.query import query_metric


def _write_catalog(ws: Path) -> None:
    catalog = ws / "catalog"
    catalog.mkdir(parents=True)
    (catalog / "pipelines.yaml").write_text(
        """
version: 1
workspace: phase2_test
sources:
  - id: ih
    reader:
      kind: parquet
      file_pattern: "data/*.parquet"
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
    (catalog / "processors.yaml").write_text(
        """
processors:
  - id: descriptive
    source: ih
    kind: numeric_distribution
    group_by: [Channel]
    time:
      column: OutcomeTime
      grains: [Day, Month, Summary]
    properties: [Propensity, FinalPropensity]
  - id: scores
    source: ih
    kind: score_distribution
    group_by: [Channel]
    time:
      column: OutcomeTime
      grains: [Day, Month, Summary]
    score_properties: [Propensity, FinalPropensity]
    outcome:
      column: Outcome
      positive_values: [Clicked]
      negative_values: [Impression]
    dedup_keys: [InteractionID, ActionID, Rank]
""",
        encoding="utf-8",
    )
    (catalog / "metrics.yaml").write_text(
        """
metrics:
  MedianPropensity:
    source: descriptive
    kind: tdigest_quantile
    state: Propensity_tdigest
    quantile: 0.5
  ROC_AUC:
    source: scores
    kind: curve_from_digests
    positive_state: Propensity_tdigest_positives
    negative_state: Propensity_tdigest_negatives
    output: roc_auc
  AvgPrecision:
    source: scores
    kind: curve_from_digests
    positive_state: Propensity_tdigest_positives
    negative_state: Propensity_tdigest_negatives
    output: average_precision
  Calibration:
    source: scores
    kind: calibration_from_digests
    positive_state: FinalPropensity_tdigest_positives
    negative_state: FinalPropensity_tdigest_negatives
""",
        encoding="utf-8",
    )
    (catalog / "dashboards.yaml").write_text("dashboards: []\n", encoding="utf-8")


def _write_data(ws: Path) -> None:
    data = ws / "data"
    data.mkdir()
    rows = [
        _row("2024-01-01", "Web", "Impression", 0.10, 0.12, "c1", "i1"),
        _row("2024-01-01", "Web", "Clicked", 0.90, 0.88, "c2", "i2"),
        _row("2024-01-01", "Mobile", "Impression", 0.20, 0.25, "c3", "i3"),
        _row("2024-01-01", "Mobile", "Clicked", 0.80, 0.76, "c4", "i4"),
        _row("2024-01-02", "Web", "Impression", 0.15, 0.18, "c5", "i5"),
        _row("2024-01-02", "Web", "Clicked", 0.95, 0.92, "c6", "i6"),
    ]
    pl.DataFrame(rows).write_parquet(data / "ih_20240101000000.parquet")


def _row(
    day: str,
    channel: str,
    outcome: str,
    propensity: float,
    final_propensity: float,
    customer: str,
    interaction: str,
) -> dict[str, object]:
    return {
        "OutcomeTime": dt.datetime.fromisoformat(day + "T10:00:00"),
        "Channel": channel,
        "Outcome": outcome,
        "Propensity": propensity,
        "FinalPropensity": final_propensity,
        "CustomerID": customer,
        "InteractionID": interaction,
        "ActionID": "action",
        "Rank": 1,
        "Name": "Offer",
    }


@pytest.mark.integration
def test_phase2_numeric_and_score_metrics_query(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    _write_data(tmp_path)

    run = run_source(tmp_path, "ih")
    median = query_metric(tmp_path, "MedianPropensity", group_by=["Channel"], grain="summary")
    auc = query_metric(tmp_path, "ROC_AUC", group_by=["Channel"], grain="summary")
    ap = query_metric(tmp_path, "AvgPrecision", group_by=["Channel"], grain="summary")
    calibration = query_metric(tmp_path, "Calibration", group_by=["Channel"], grain="summary")

    assert run.status == "ok"
    assert median.columns == ["Channel", "MedianPropensity"]
    assert median.filter(pl.col("Channel") == "Web")["MedianPropensity"][0] == pytest.approx(
        0.9, abs=0.2
    )
    assert auc.filter(pl.col("Channel") == "Web")["ROC_AUC"][0] > 0.95
    assert ap.filter(pl.col("Channel") == "Web")["AvgPrecision"][0] > 0.9
    assert calibration.schema["Calibration"].base_type() == pl.Struct


@pytest.mark.integration
def test_curve_metric_can_include_curve_arrays(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    _write_data(tmp_path)
    run_source(tmp_path, "ih")

    rows = query_metric(
        tmp_path,
        "ROC_AUC",
        group_by=["Channel"],
        grain="summary",
        include_curve_columns=True,
    )

    assert {"ROC_AUC", "roc_auc", "average_precision", "fpr", "tpr", "pos_fraction"} <= set(
        rows.columns
    )
    assert rows.schema["fpr"].base_type() == pl.List
    assert rows.filter(pl.col("Channel") == "Web")["ROC_AUC"][0] == pytest.approx(
        rows.filter(pl.col("Channel") == "Web")["roc_auc"][0]
    )
    assert len(rows.filter(pl.col("Channel") == "Web")["fpr"][0]) > 1


@pytest.mark.integration
def test_quantile_query_can_include_boxplot_suite(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    _write_data(tmp_path)
    run_source(tmp_path, "ih")

    rows = query_metric(
        tmp_path,
        "MedianPropensity",
        group_by=["Channel"],
        grain="summary",
        include_quantile_suite=True,
    )

    assert {"Propensity_p25", "Propensity_Median", "Propensity_p75"} <= set(rows.columns)
    assert {"Propensity_Min", "Propensity_Max"} <= set(rows.columns)
