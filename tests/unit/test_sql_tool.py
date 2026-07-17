"""Governed SQL tool validation and execution tests."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import duckdb
import polars as pl
import pytest

from valuestream.ai import sql_tool
from valuestream.store.meta import meta_dir


@pytest.mark.unit
def test_validate_sql_accepts_select_and_with_statements() -> None:
    assert sql_tool.validate_sql("SELECT 1") == "SELECT 1"
    assert sql_tool.validate_sql("  with t as (select 1 as x) select * from t; ").startswith(
        "with t"
    )


@pytest.mark.unit
def test_validate_sql_rejects_non_select_statements() -> None:
    for sql in (
        "",
        "DROP TABLE x",
        "INSERT INTO x VALUES (1)",
        "SELECT 1; SELECT 2",
        "SELECT 1 -- sneaky",
        "SELECT /* hidden */ 1",
        "ATTACH 'other.duckdb' AS other",
        "SELECT * FROM read_parquet('/etc/anything')",
        "SELECT * FROM read_csv_auto('x.csv')",
        "COPY (SELECT 1) TO 'out.csv'",
        "PRAGMA database_list",
        "SET enable_external_access=true",
        "CALL pragma_version()",
    ):
        with pytest.raises(ValueError, match=r"SQL|statement|allowed|governed"):
            sql_tool.validate_sql(sql)


def _seed_export_db(workspace: Path) -> None:
    meta = meta_dir(workspace)
    meta.mkdir(parents=True, exist_ok=True)
    db_path = meta / "metric_export_summary.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE "CTR" AS
            SELECT * FROM (VALUES
                ('Web', 0.5, encode('sketch-a')),
                ('Mobile', 0.25, encode('sketch-b')),
                ('Store', 0.75, encode('sketch-c'))
            ) AS t(Channel, CTR, State_blob)
            """
        )


@pytest.mark.unit
def test_run_sql_query_masks_blobs_and_caps_rows(tmp_path: Path) -> None:
    _seed_export_db(tmp_path)

    result = sql_tool.run_sql_query(
        tmp_path,
        'SELECT Channel, CTR, State_blob FROM metrics_summary."CTR" ORDER BY CTR DESC',
        limit=2,
    )

    assert result.truncated is True
    assert result.row_count == 2
    assert result.masked_columns == ["State_blob"]
    assert result.rows.columns == ["Channel", "CTR"]
    assert result.rows.get_column("Channel").to_list() == ["Store", "Web"]


@pytest.mark.unit
def test_run_sql_query_logs_only_safe_metadata(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    workspace = tmp_path / "PRIVATE-WORKSPACE-SECRET"
    _seed_export_db(workspace)
    sql_secret = "PRIVATE-CUSTOMER-42"
    sql = f"SELECT Channel, CTR FROM metrics_summary.\"CTR\" WHERE Channel = '{sql_secret}'"
    caplog.set_level(logging.INFO, logger=sql_tool.__name__)

    sql_tool.run_sql_query(workspace, sql)

    assert re.search(r"query_id=[0-9a-f]{12}", caplog.text)
    assert "statement=select" in caplog.text
    assert f"sql_length={len(sql)}" in caplog.text
    assert "column_count=2" in caplog.text
    assert sql_secret not in caplog.text
    assert str(workspace) not in caplog.text


@pytest.mark.unit
def test_run_sql_query_cannot_read_ungoverned_files(tmp_path: Path) -> None:
    _seed_export_db(tmp_path)
    raw_source = tmp_path / "raw-source.parquet"
    pl.DataFrame({"secret": ["not-an-aggregate"]}).write_parquet(raw_source)

    with pytest.raises(duckdb.PermissionException, match="file system operations are disabled"):
        sql_tool.run_sql_query(tmp_path, f"SELECT * FROM '{raw_source}'")


@pytest.mark.unit
def test_run_sql_query_requires_governed_databases(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no governed DuckDB aggregates"):
        sql_tool.run_sql_query(tmp_path, "SELECT 1")


@pytest.mark.unit
def test_list_sql_tables_and_schema_summary_exclude_blob_columns(tmp_path: Path) -> None:
    _seed_export_db(tmp_path)

    tables = sql_tool.list_sql_tables(tmp_path)

    assert len(tables) == 1
    table = tables[0]
    assert table.name == 'metrics_summary."CTR"'
    assert table.kind == "table"
    assert [name for name, _ in table.columns] == ["Channel", "CTR"]

    summary = sql_tool.sql_schema_summary(tmp_path)
    assert 'metrics_summary."CTR"' in summary
    assert "State_blob" not in summary


@pytest.mark.unit
def test_schema_summary_reports_missing_databases(tmp_path: Path) -> None:
    assert "No governed SQL tables" in sql_tool.sql_schema_summary(tmp_path)
