"""Materialized DuckDB exports for metric tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from valuestream.config import model
from valuestream.query import query_metric
from valuestream.store.meta import meta_dir
from valuestream.utils.timer import timed


@dataclass(frozen=True)
class ExportedMetricTable:
    metric_name: str
    table_name: str
    source_id: str
    processor_id: str
    grain: str
    rows: int


@dataclass(frozen=True)
class SkippedMetricTable:
    metric_name: str
    reason: str


@dataclass(frozen=True)
class MetricDuckDBExportResult:
    path: Path
    grain: str
    tables: list[ExportedMetricTable]
    skipped: list[SkippedMetricTable]

    @property
    def rows(self) -> int:
        return sum(table.rows for table in self.tables)


def metric_export_db_path(workspace_path: str | Path, grain: str) -> Path:
    """Return the default DuckDB export path for one metric grain."""
    return meta_dir(workspace_path) / f"metric_export_{model.normalize_grain_name(grain)}.duckdb"


@timed
def export_metric_tables_to_duckdb(
    workspace_path: str | Path,
    catalog: model.Catalog,
    *,
    grain: str,
    output_path: str | Path | None = None,
    replace: bool = True,
) -> MetricDuckDBExportResult:
    """Export each metric as one materialized DuckDB table at ``grain``."""
    workspace = Path(workspace_path)
    normalized_grain = model.normalize_grain_name(grain)
    path = (
        Path(output_path)
        if output_path is not None
        else metric_export_db_path(workspace, normalized_grain)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace and path.exists():
        path.unlink()

    tables: list[ExportedMetricTable] = []
    skipped: list[SkippedMetricTable] = []
    used_names: set[str] = set()
    with duckdb.connect(str(path)) as conn:
        for metric_name, metric_def in catalog.metrics.metrics.items():
            processor = _processor_for_metric(catalog, metric_def)
            if processor is None:
                skipped.append(
                    SkippedMetricTable(
                        metric_name,
                        f"metric references unknown processor {metric_def.source!r}",
                    )
                )
                continue
            table_name = _unique_table_name(
                f"metric_{_safe_identifier(metric_name)}_{normalized_grain}",
                used_names,
            )
            try:
                rows = query_metric(
                    workspace,
                    metric_name,
                    group_by=list(processor.group_by),
                    grain=normalized_grain,
                    include_quantile_suite=True,
                )
            except Exception as exc:  # pragma: no cover - exercised through CLI/reporting paths
                skipped.append(SkippedMetricTable(metric_name, str(exc)))
                continue
            exported = _with_export_columns(
                rows,
                metric_name=metric_name,
                source_id=processor.source,
                processor_id=processor.id,
                grain=normalized_grain,
            )
            _replace_table(conn, table_name, exported)
            tables.append(
                ExportedMetricTable(
                    metric_name=metric_name,
                    table_name=table_name,
                    source_id=processor.source,
                    processor_id=processor.id,
                    grain=normalized_grain,
                    rows=exported.height,
                )
            )
        _write_manifest(conn, tables, skipped)
    return MetricDuckDBExportResult(
        path=path, grain=normalized_grain, tables=tables, skipped=skipped
    )


def _replace_table(conn: duckdb.DuckDBPyConnection, table_name: str, rows: pl.DataFrame) -> None:
    conn.register("metric_export_rows", rows.to_arrow())
    try:
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM metric_export_rows')
    finally:
        conn.unregister("metric_export_rows")


def _write_manifest(
    conn: duckdb.DuckDBPyConnection,
    tables: list[ExportedMetricTable],
    skipped: list[SkippedMetricTable],
) -> None:
    rows: list[dict[str, Any]] = [
        {
            "metric_name": table.metric_name,
            "table_name": table.table_name,
            "source_id": table.source_id,
            "processor_id": table.processor_id,
            "grain": table.grain,
            "rows": table.rows,
            "status": "exported",
            "reason": "",
        }
        for table in tables
    ]
    rows.extend(
        {
            "metric_name": item.metric_name,
            "table_name": "",
            "source_id": "",
            "processor_id": "",
            "grain": "",
            "rows": 0,
            "status": "skipped",
            "reason": item.reason,
        }
        for item in skipped
    )
    manifest = pl.DataFrame(
        rows,
        schema={
            "metric_name": pl.String,
            "table_name": pl.String,
            "source_id": pl.String,
            "processor_id": pl.String,
            "grain": pl.String,
            "rows": pl.Int64,
            "status": pl.String,
            "reason": pl.String,
        },
    )
    _replace_table(conn, "valuestream_metric_export_manifest", manifest)


def _with_export_columns(
    rows: pl.DataFrame,
    *,
    metric_name: str,
    source_id: str,
    processor_id: str,
    grain: str,
) -> pl.DataFrame:
    return rows.with_columns(
        pl.lit(metric_name).alias("_valuestream_metric"),
        pl.lit(source_id).alias("_valuestream_source"),
        pl.lit(processor_id).alias("_valuestream_processor"),
        pl.lit(grain).alias("_valuestream_grain"),
    )


def _processor_for_metric(catalog: model.Catalog, metric: model.Metric) -> model.Processor | None:
    return next(
        (processor for processor in catalog.processors.processors if processor.id == metric.source),
        None,
    )


def _unique_table_name(base: str, used: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _safe_identifier(value: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in value]
    out = "".join(chars).strip("_") or "x"
    return out if out[0].isalpha() else f"x_{out}"


__all__ = [
    "ExportedMetricTable",
    "MetricDuckDBExportResult",
    "SkippedMetricTable",
    "export_metric_tables_to_duckdb",
    "metric_export_db_path",
]
