"""RFM segmentation helpers for lifecycle aggregates."""

from __future__ import annotations

from typing import Literal

import polars as pl

SegmentPreset = Literal["default", "retail_banking", "telco", "e_commerce"]

DEFAULT_SEGMENTS: dict[str, tuple[str, ...]] = {
    "Premium Customer": ("334", "443", "444", "344", "434", "433", "343", "333"),
    "Repeat Customer": ("244", "234", "232", "332", "143", "233", "243", "242"),
    "Top Spender": (
        "424",
        "414",
        "144",
        "314",
        "324",
        "124",
        "224",
        "423",
        "413",
        "133",
        "323",
        "313",
        "134",
    ),
    "At Risk Customer": (
        "422",
        "223",
        "212",
        "122",
        "222",
        "132",
        "322",
        "312",
        "412",
        "123",
        "214",
    ),
    "Inactive Customer": ("411", "111", "113", "114", "112", "211", "311"),
}

SEGMENT_PRESETS: dict[SegmentPreset, dict[str, tuple[str, ...]]] = {
    "default": DEFAULT_SEGMENTS,
    "retail_banking": DEFAULT_SEGMENTS,
    "telco": DEFAULT_SEGMENTS,
    "e_commerce": DEFAULT_SEGMENTS,
}

_CODE_TO_SEGMENT: dict[SegmentPreset, dict[str, str]] = {
    preset: {code: segment for segment, codes in mapping.items() for code in codes}
    for preset, mapping in SEGMENT_PRESETS.items()
}


def segment_name(code: str | None, *, preset: SegmentPreset = "default") -> str:
    """Return the configured segment name for an RFM code."""
    if code is None:
        return "Unknown"
    return _CODE_TO_SEGMENT[preset].get(code, "Unknown")


def with_rfm(
    frame: pl.DataFrame,
    *,
    segment_preset: SegmentPreset = "default",
) -> pl.DataFrame:
    """Add RFM columns to a merged lifecycle aggregate frame."""
    if frame.is_empty():
        return frame
    observation_end = frame.select(pl.col("MaxPurchasedDate").max()).item()
    out = frame.with_columns(
        (pl.col("unique_holdings").fill_null(0).cast(pl.Int64) - 1)
        .clip(lower_bound=0)
        .alias("frequency"),
        (pl.lit(observation_end) - pl.col("MinPurchasedDate")).dt.total_days().alias("tenure"),
        (pl.col("MaxPurchasedDate") - pl.col("MinPurchasedDate"))
        .dt.total_days()
        .alias("__recency_raw"),
        (
            pl.col("lifetime_value").fill_null(0.0)
            / pl.when(pl.col("unique_holdings") == 0).then(1).otherwise(pl.col("unique_holdings"))
        ).alias("monetary_value"),
    )
    out = out.with_columns(
        (pl.col("tenure") - pl.col("__recency_raw")).fill_null(0).alias("recency"),
        pl.when(pl.col("frequency") == 0)
        .then(0.0)
        .otherwise(pl.col("monetary_value"))
        .alias("monetary_value"),
    )
    out = out.with_columns(
        pl.col("frequency")
        .qcut(4, labels=["1", "2", "3", "4"], allow_duplicates=True)
        .cast(pl.String)
        .alias("f_quartile"),
        pl.col("monetary_value")
        .qcut(4, labels=["1", "2", "3", "4"], allow_duplicates=True)
        .cast(pl.String)
        .alias("m_quartile"),
        pl.col("recency")
        .qcut(4, labels=["4", "3", "2", "1"], allow_duplicates=True)
        .cast(pl.String)
        .alias("r_quartile"),
    )
    out = out.with_columns(
        pl.concat_str(["r_quartile", "f_quartile", "m_quartile"]).alias("rfm_seg")
    )
    segment_code = pl.col("rfm_seg")
    return out.with_columns(
        pl.when(segment_code.is_null())
        .then(None)
        .otherwise(
            segment_code.replace_strict(
                _CODE_TO_SEGMENT[segment_preset],
                default="Unknown",
                return_dtype=pl.String,
            )
        )
        .alias("rfm_segment"),
        (
            (
                pl.col("r_quartile").cast(pl.Float64)
                + pl.col("f_quartile").cast(pl.Float64)
                + pl.col("m_quartile").cast(pl.Float64)
            )
            / 3.0
        ).alias("rfm_score"),
    ).drop("__recency_raw")


__all__ = ["DEFAULT_SEGMENTS", "SEGMENT_PRESETS", "SegmentPreset", "segment_name", "with_rfm"]
