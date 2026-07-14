"""Tests for the Pega dummy Parquet data generator."""

from __future__ import annotations

import datetime as dt
import json
import zipfile
from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner

from valuestream.cli import main
from valuestream.generators import (
    CUSTOMER_SEGMENTS,
    PegaDummyGenerationConfig,
    generate_pega_dummy_data,
)


def _series_mean(series: pl.Series) -> float:
    value = series.mean()
    assert isinstance(value, int | float)
    return float(value)


def _write_source_zip(path: Path) -> None:
    rows = [
        {
            "pySubjectType": "ANB-CDH-Data-Customer",
            "pxInteractionID": "-1120802700555964389",
            "CustomerID": "C-3144",
            "pySubjectID": "C-3144",
            "pyChannel": "Web",
            "pyDirection": "Inbound",
            "pyOutcome": "Impression",
            "pyGroup": "Mortgages",
            "pxOutcomeTime": "20260522T091752.763 GMT",
            "pxDecisionTime": "20260522T091751.200 GMT",
            "pxUpdateDateTime": 1779441472764,
            "pxStreamPosition": "0",
            "pxStreamPartition": "0",
            "pyPropensity": 0.017857142857142856,
            "FinalPropensity": 0.017857142857142856,
            "pyPropensitySource": "Web_Click_Through_Rate_GB_Customer",
            "BundleHead": False,
        },
        {
            "pySubjectType": "ANB-CDH-Data-Customer",
            "pxInteractionID": "-1120802700555964210",
            "CustomerID": "C-4792",
            "pySubjectID": "C-4792",
            "pyChannel": "Email",
            "pyDirection": "Outbound",
            "pyOutcome": "Pending",
            "pyGroup": "RealTime",
            "pxOutcomeTime": "20260522T092252.235 GMT",
            "pxDecisionTime": "20260522T092252.190 GMT",
            "pxUpdateDateTime": 1779441772236,
            "pxStreamPosition": "1",
            "pxStreamPartition": "1",
            "pyPropensity": 0.5,
            "FinalPropensity": 0.5,
            "pyPropensitySource": "Email_Click_Through_Rate_GB_Customer",
            "BundleHead": False,
        },
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.json", "\n".join(json.dumps(row) for row in rows))


@pytest.mark.unit
def test_generate_pega_dummy_data_preserves_source_columns_and_adds_requested_fields(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    out = tmp_path / "out"
    _write_source_zip(source)

    report = generate_pega_dummy_data(
        PegaDummyGenerationConfig(
            source_path=source,
            output_dir=out,
            start_date=dt.date(2026, 1, 1),
            end_date=dt.date(2026, 1, 2),
            rows_per_day=7,
            batch_size=3,
            customer_count=25,
            seed=7,
        )
    )

    assert report.rows == 14
    assert [path.name for path in report.files] == [
        "pega_interactions_20260101.parquet",
        "pega_interactions_20260102.parquet",
    ]
    frame = pl.concat([pl.read_parquet(path) for path in report.files])

    assert frame.height == 14
    assert frame.columns == [
        "pySubjectType",
        "pxInteractionID",
        "CustomerID",
        "pySubjectID",
        "pyChannel",
        "pyDirection",
        "pyOutcome",
        "pyGroup",
        "pxOutcomeTime",
        "pxDecisionTime",
        "pxUpdateDateTime",
        "pxStreamPosition",
        "pxStreamPartition",
        "pyPropensity",
        "FinalPropensity",
        "pyPropensitySource",
        "BundleHead",
        "IsProspect",
        "CustomerSegment",
    ]
    assert set(frame["pyChannel"].unique()).issubset({"Web", "Email"})
    assert set(frame["CustomerSegment"].unique()).issubset(set(CUSTOMER_SEGMENTS))
    assert frame["IsProspect"].dtype == pl.Boolean
    assert frame["pxOutcomeTime"].str.slice(0, 8).unique().sort().to_list() == [
        "20260101",
        "20260102",
    ]
    assert set(frame["pyOutcome"].unique()) <= {"Clicked", "Conversion", "Impression", "Pending"}
    assert frame["CustomerID"].to_list() == frame["pySubjectID"].to_list()
    assert frame["pxInteractionID"].n_unique() == 14


@pytest.mark.unit
def test_generate_pega_dummy_correlates_propensity_fields_with_binary_outcome(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    out = tmp_path / "out"
    _write_source_zip(source)

    report = generate_pega_dummy_data(
        PegaDummyGenerationConfig(
            source_path=source,
            output_dir=out,
            start_date=dt.date(2026, 1, 1),
            end_date=dt.date(2026, 1, 1),
            rows_per_day=1_000,
            batch_size=100,
            positive_rate=0.5,
            seed=3,
        )
    )
    frame = pl.read_parquet(report.files[0])
    positives = frame.filter(pl.col("pyOutcome").is_in(["Clicked", "Conversion"]))
    negatives = frame.filter(pl.col("pyOutcome").is_in(["Impression", "Pending"]))

    assert set(frame["pyOutcome"].unique()) == {"Clicked", "Conversion", "Impression", "Pending"}
    assert set(frame.filter(pl.col("pyOutcome") == "Pending")["pyDirection"].unique()) == {
        "Outbound"
    }
    assert set(frame.filter(pl.col("pyOutcome") == "Impression")["pyDirection"].unique()) == {
        "Inbound"
    }
    assert _series_mean(positives["pyPropensity"]) > _series_mean(negatives["pyPropensity"]) + 0.35
    assert _series_mean(positives["FinalPropensity"]) > (
        _series_mean(negatives["FinalPropensity"]) + 0.35
    )
    assert frame["pyPropensity"].to_list() == frame["FinalPropensity"].to_list()
    assert set(frame["pyPropensitySource"].unique()) == {
        "Email_Click_Through_Rate_GB_Customer",
        "Web_Click_Through_Rate_GB_Customer",
    }


@pytest.mark.unit
def test_generate_pega_dummy_refuses_to_overwrite_existing_files(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    out = tmp_path / "out"
    out.mkdir()
    _write_source_zip(source)
    (out / "pega_interactions_20260101.parquet").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        generate_pega_dummy_data(
            PegaDummyGenerationConfig(
                source_path=source,
                output_dir=out,
                start_date=dt.date(2026, 1, 1),
                end_date=dt.date(2026, 1, 1),
                rows_per_day=1,
            )
        )


@pytest.mark.unit
def test_generate_pega_dummy_cli_accepts_days_timeframe(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    out = tmp_path / "out"
    _write_source_zip(source)

    result = CliRunner().invoke(
        main,
        [
            "generate-pega-dummy",
            "--source",
            str(source),
            "--output-dir",
            str(out),
            "--start-date",
            "2026-01-01",
            "--days",
            "2",
            "--rows-per-day",
            "2",
            "--batch-size",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "generated 4 row(s)" in result.output
    assert pl.read_parquet(out / "pega_interactions_20260102.parquet").height == 2
