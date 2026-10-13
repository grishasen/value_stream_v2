"""Bounded tabular sample reading shared by Studio and MCP authoring."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def read_tabular_sample(
    path: Path,
    *,
    limit: int,
    columns: list[str] | tuple[str, ...] | None = None,
    try_parse_dates: bool = False,
) -> pl.DataFrame:
    """Read only a bounded CSV/Parquet preview, without copying the whole file."""

    if not 1 <= limit <= 100_000:
        raise ValueError("sample limit must be between 1 and 100000")
    if path.suffix.casefold() == ".parquet":
        preview = pl.scan_parquet(path)
        if columns is not None:
            preview = preview.select(list(columns))
        return preview.head(limit).collect()
    if path.suffix.casefold() == ".csv":
        return pl.read_csv(
            path,
            infer_schema_length=500,
            n_rows=limit,
            try_parse_dates=try_parse_dates,
            columns=list(columns) if columns is not None else None,
        ).head(limit)
    raise ValueError("MCP catalog authoring supports CSV and Parquet samples")
