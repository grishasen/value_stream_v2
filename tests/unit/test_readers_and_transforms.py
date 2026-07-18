"""Unit coverage for Phase 1 readers and transforms."""

from __future__ import annotations

import datetime as dt
import gzip
import json
import tarfile
import zipfile
from pathlib import Path

import polars as pl
import pytest

from valuestream.config import model
from valuestream.readers import cleanup_temporaries, read
from valuestream.transforms import apply_transforms


def _source(transforms: list[dict[str, object]]) -> model.Source:
    return model.Source.model_validate(
        {
            "id": "s",
            "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
            "transforms": transforms,
        }
    )


@pytest.mark.unit
def test_parquet_csv_and_pega_json_readers(tmp_path: Path) -> None:
    parquet = tmp_path / "data.parquet"
    csv = tmp_path / "data.csv"
    json_file = tmp_path / "data.json"
    pl.DataFrame({"A": [1, 2]}).write_parquet(parquet)
    csv.write_text("A,B\n1,x\n2,y\n", encoding="utf-8")
    json_file.write_text(json.dumps([{"A": 1}, {"A": 2}]), encoding="utf-8")

    assert (
        read(model.ParquetReader(kind="parquet", file_pattern="*.parquet"), [parquet])
        .collect()
        .height
        == 2
    )
    assert read(model.CsvReader(kind="csv", file_pattern="*.csv"), [csv]).collect().height == 2
    assert (
        read(model.PegaDsExportReader(kind="pega_ds_export", file_pattern="*.json"), [json_file])
        .collect()
        .height
        == 2
    )


@pytest.mark.unit
def test_pega_archive_readers_normalize_supported_payloads(tmp_path: Path) -> None:
    zip_file = tmp_path / "data.zip"
    gzip_file = tmp_path / "data.json.gz"
    tar_file = tmp_path / "data.tar.gz"
    with zipfile.ZipFile(zip_file, "w") as zf:
        zf.writestr("a.json", json.dumps([{"A": 1}, {"A": 2}]))
        zf.writestr("b.ndjson", '{"A":3}\n')
    with gzip.open(gzip_file, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps([{"A": 4}, {"A": 5}]))
    member = tmp_path / "member.ndjson"
    member.write_text('{"A":6}\n{"A":7}\n', encoding="utf-8")
    with tarfile.open(tar_file, "w:gz") as tf:
        tf.add(member, arcname="nested/member.ndjson")

    reader = model.PegaDsExportReader(kind="pega_ds_export", file_pattern="*")
    frame = read(reader, [zip_file, gzip_file, tar_file]).collect().sort("A")

    assert frame["A"].to_list() == [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.unit
def test_pega_reader_cleans_normalized_temp_files(tmp_path: Path) -> None:
    json_file = tmp_path / "data.json"
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    json_file.write_text(json.dumps([{"A": 1}, {"A": 2}]), encoding="utf-8")
    reader = model.PegaDsExportReader(
        kind="pega_ds_export",
        file_pattern="*.json",
        archive_temp_dir=str(temp_root),
    )

    frame = read(reader, [json_file])

    try:
        assert frame.collect().height == 2
    finally:
        cleanup_temporaries()
    assert list(temp_root.glob("dataset_export_*")) == []


@pytest.mark.unit
def test_rename_capitalize_transform() -> None:
    source = _source([{"kind": "rename_capitalize"}])
    out = apply_transforms(
        pl.DataFrame(
            {
                "channel": ["Web"],
                "pxDecisionTime": ["2024"],
                "pyName": ["Offer"],
                "pzConfigurationName": ["Config"],
                "pxInteractionID": ["i-1"],
            }
        ).lazy(),
        source,
    ).collect()

    assert out.columns == [
        "Channel",
        "DecisionTime",
        "Name",
        "ConfigurationName",
        "InteractionID",
    ]


@pytest.mark.unit
def test_rename_capitalize_preserves_existing_name_collision() -> None:
    source = _source([{"kind": "rename_capitalize"}])
    out = apply_transforms(
        pl.DataFrame({"pxName": ["raw"], "Name": ["canonical"]}).lazy(),
        source,
    ).collect()

    assert out.columns == ["PxName", "Name"]


@pytest.mark.unit
def test_transform_pipeline_catalog() -> None:
    source = _source(
        [
            {"kind": "defaults", "values": {"FillMe": "fallback"}},
            {"kind": "parse_datetime", "columns": ["Start", "End"], "format": "%Y-%m-%d %H:%M:%S"},
            {
                "kind": "derive_calendar",
                "from": "End",
                "outputs": ["Day", "Month", "Year", "Quarter", "Week"],
            },
            {"kind": "derive_action_id", "parts": ["Issue", "Group", "Name"], "sep": "/"},
            {
                "kind": "derive_column",
                "output": "ResponseSeconds",
                "expression": {
                    "op": "date_diff",
                    "unit": "seconds",
                    "end": {"col": "End"},
                    "start": {"col": "Start"},
                },
            },
            {
                "kind": "derive_column",
                "output": "Gross",
                "expression": {"polars": 'pl.col("Amount").cast(pl.Float64) * pl.lit(2.0)'},
            },
            {"kind": "filter", "expression": {"op": "not_null", "column": "Channel"}},
            {"kind": "dedup", "keys": ["InteractionID"]},
            {"kind": "cast", "columns": {"Amount": "Float64"}},
            {"kind": "drop_columns", "columns": ["Raw"]},
            {"kind": "coalesce", "output": "DisplayName", "columns": ["MissingName", "Name"]},
        ]
    )
    frame = pl.DataFrame(
        {
            "Start": ["2024-01-01 10:00:00", "2024-01-01 10:00:00"],
            "End": ["2024-01-01 10:01:00", "2024-01-01 10:02:00"],
            "Channel": ["Web", "Web"],
            "InteractionID": ["i1", "i1"],
            "Issue": ["Sales", "Sales"],
            "Group": ["Cards", "Cards"],
            "Name": ["Offer", "Offer"],
            "MissingName": [None, None],
            "Amount": ["1.5", "2.5"],
            "Raw": ["x", "y"],
        }
    )

    out = apply_transforms(frame.lazy(), source).collect()

    assert out.height == 1
    assert out["Day"].to_list() == [dt.date(2024, 1, 1)]
    assert out["Month"].to_list() == ["2024-01"]
    assert out["ActionID"].to_list() == ["Sales/Cards/Offer"]
    assert out["ResponseSeconds"].to_list() == [60]
    assert out["Gross"].to_list() == [3.0]
    assert out["Amount"].dtype == pl.Float64
    assert "Raw" not in out.columns
    assert out["DisplayName"].to_list() == ["Offer"]
    assert out["FillMe"].to_list() == ["fallback"]


@pytest.mark.unit
def test_parse_datetime_skips_columns_the_reader_already_typed() -> None:
    """CSV date inference or parquet schemas can pre-type timestamp columns.

    The generated Studio source applies ``parse_datetime`` to columns the
    preview saw as strings; the runtime reader may deliver them as datetimes
    already, and the same config must work for both reads.
    """

    source = _source(
        [
            {"kind": "parse_datetime", "columns": ["OutcomeTime", "DecisionTime"], "format": "%+"},
            {
                "kind": "derive_column",
                "output": "ResponseSeconds",
                "expression": {
                    "op": "date_diff",
                    "unit": "seconds",
                    "end": {"col": "OutcomeTime"},
                    "start": {"col": "DecisionTime"},
                },
            },
        ]
    )
    already_typed = dt.datetime(2026, 7, 1, 9, 0, tzinfo=dt.UTC)
    frame = pl.DataFrame(
        {
            "OutcomeTime": [already_typed],
            "DecisionTime": ["2026-07-01T08:59:00Z"],
        }
    ).lazy()

    out = apply_transforms(frame, source).collect()

    assert out.get_column("OutcomeTime").dtype.is_temporal()
    assert out.get_column("DecisionTime").dtype.is_temporal()
    assert out.get_column("ResponseSeconds").to_list() == [60]
