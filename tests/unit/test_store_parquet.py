"""Aggregate parquet store tests."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from valuestream.store.parquet import write_aggregate


@pytest.mark.unit
def test_write_aggregate_rejects_null_period(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "period": [None],
            "Count": [1],
        }
    )

    with pytest.raises(ValueError, match="null period"):
        write_aggregate(
            frame,
            tmp_path,
            source_id="ih",
            processor_id="engagement",
            grain="daily",
            run_id="run",
            chunk_id="chunk",
        )
