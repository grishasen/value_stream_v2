"""Calendar grain helpers for aggregate materialization."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from valuestream.config import model

PROVENANCE_COLUMNS = {"pipeline_run_id", "chunk_id", "period", "created_at", "config_hash"}
TIME_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "hourly": ("Hour", "hour"),
    "daily": ("Day", "day", "as_of_date"),
    "weekly": ("Week", "week"),
    "monthly": ("Month", "month"),
    "quarterly": ("Quarter", "quarter"),
    "yearly": ("Year", "year"),
}
TIME_LEVELS = ("hourly", "daily", "weekly", "monthly", "quarterly", "yearly")
SUPPORTED_TARGET_GRAINS = {*TIME_LEVELS, "summary"}
TIME_COLUMNS = {column for candidates in TIME_COLUMN_CANDIDATES.values() for column in candidates}
_LEVEL_RANK = {level: index for index, level in enumerate(TIME_LEVELS)}


@dataclass(frozen=True)
class CompactionPlan:
    """Prepared frame and grouping columns for one target aggregate grain."""

    frame: pl.DataFrame
    group_columns: list[str]


def finest_configured_level(config: model.Processor) -> str:
    """Return the finest physical level needed by a processor."""

    levels = [config.aggregation_level_for(grain) for grain in config.grains]
    return min(levels, key=lambda level: _LEVEL_RANK.get(level, 99), default="daily")


def aggregate_grain_candidates(config: model.Processor, grain: str) -> list[str]:
    """Return stored grains to try for a requested grain, finest-eligible first.

    The requested grain is preferred, followed by any configured grain whose
    physical level is finer-or-equal to the request (coarsest first), so a
    query can roll a finer aggregate up to the requested grain when an exact
    physical aggregate is absent.
    """

    grain = model.normalize_grain_name(grain)
    requested_level = config.aggregation_level_for(grain)
    requested_rank = TIME_LEVELS.index(requested_level)
    candidates = [grain, *config.grains, "monthly", "daily", "summary"]
    available: list[tuple[int, str]] = []
    for candidate in candidates:
        normalized = model.normalize_grain_name(candidate)
        level = config.aggregation_level_for(normalized)
        if level not in TIME_LEVELS:
            continue
        rank = TIME_LEVELS.index(level)
        if rank <= requested_rank and normalized not in {item[1] for item in available}:
            available.append((rank, normalized))
    ordered = [
        candidate for _, candidate in sorted(available, key=lambda item: item[0], reverse=True)
    ]
    return [grain, *[candidate for candidate in ordered if candidate != grain]]


def normalize_target_grain(config: model.Processor, target_grain: str, processor_kind: str) -> str:
    """Normalize and validate one configured target aggregate grain."""

    normalized = model.normalize_grain_name(target_grain)
    if normalized not in SUPPORTED_TARGET_GRAINS or normalized not in config.grains:
        raise ValueError(f"unsupported compact grain for {processor_kind}: {normalized}")
    return normalized


def chunk_time_group_columns(existing: set[str], config: model.Processor) -> list[str]:
    """Return calendar columns to preserve in the base chunk aggregate."""

    finest = finest_configured_level(config)
    min_rank = _LEVEL_RANK.get(finest, _LEVEL_RANK["daily"])
    columns: list[str] = []
    for level in TIME_LEVELS[min_rank:]:
        column = column_for_level(existing, level)
        if column is not None and column not in columns:
            columns.append(column)
    if not columns:
        column = column_for_level(existing, "daily")
        if column is not None:
            columns.append(column)
    return columns


def base_period_expr(existing: set[str], config: model.Processor) -> pl.Expr:
    """Return a partition period for the base chunk aggregate."""

    return period_expr_for_level(existing, finest_configured_level(config))


def prepare_compaction(
    frame: pl.DataFrame,
    *,
    config: model.Processor,
    state_specs: dict[str, model.StateSpec],
    target_grain: str,
) -> CompactionPlan:
    """Prepare rows for merging into the target grain's physical level."""

    if frame.is_empty():
        return CompactionPlan(frame=frame, group_columns=[])

    target_grain = model.normalize_grain_name(target_grain)
    level = config.aggregation_level_for(target_grain)
    working = frame.drop(
        [column for column in PROVENANCE_COLUMNS - {"period"} if column in frame.columns]
    )
    working = with_calendar_columns(working)
    existing = set(working.columns)
    working = working.with_columns(period_expr_for_level(existing, level).alias("period"))
    existing = set(working.columns)

    time_columns = columns_for_physical_level(existing, level)
    state_columns = set(state_specs)
    business_columns = [
        column
        for column in working.columns
        if column not in state_columns
        and column not in PROVENANCE_COLUMNS
        and column not in TIME_COLUMNS
    ]
    group_columns = [*business_columns, *time_columns]
    if "period" in working.columns and "period" not in group_columns:
        group_columns.append("period")
    return CompactionPlan(frame=working, group_columns=_dedupe(group_columns))


def with_calendar_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Derive common calendar columns from existing finer columns when possible."""

    out = frame
    existing = set(out.columns)
    day_col = column_for_level(existing, "daily")
    month_col = column_for_level(existing, "monthly")
    expressions: list[pl.Expr] = []

    if "Month" not in existing and day_col is not None:
        expressions.append(pl.col(day_col).cast(pl.Date).dt.strftime("%Y-%m").alias("Month"))
    if "Year" not in existing:
        if day_col is not None:
            expressions.append(pl.col(day_col).cast(pl.Date).dt.year().cast(pl.Int16).alias("Year"))
        elif month_col is not None:
            expressions.append(pl.col(month_col).cast(pl.String).str.slice(0, 4).alias("Year"))
    if "Quarter" not in existing:
        if day_col is not None:
            day = pl.col(day_col).cast(pl.Date)
            expressions.append(
                pl.concat_str(
                    [day.dt.year().cast(pl.String), pl.lit("_Q"), day.dt.quarter().cast(pl.String)]
                ).alias("Quarter")
            )
        elif month_col is not None:
            month_num = (
                pl.col(month_col).cast(pl.String).str.slice(5, 2).cast(pl.Int16, strict=False)
            )
            quarter = ((month_num - 1) // 3) + 1
            expressions.append(
                pl.concat_str(
                    [
                        pl.col(month_col).cast(pl.String).str.slice(0, 4),
                        pl.lit("_Q"),
                        quarter.cast(pl.String),
                    ]
                ).alias("Quarter")
            )
    if "Week" not in existing and day_col is not None:
        expressions.append(pl.col(day_col).cast(pl.Date).dt.strftime("%G-W%V").alias("Week"))
    if expressions:
        out = out.with_columns(expressions)
    return out


def column_for_level(existing: set[str], level: str) -> str | None:
    """Return the preferred column name for a calendar level."""

    for candidate in TIME_COLUMN_CANDIDATES.get(level, ()):
        if candidate in existing:
            return candidate
    if level == "monthly" and "period" in existing:
        return "period"
    return None


def columns_for_physical_level(existing: set[str], level: str) -> list[str]:
    """Return time columns that identify rows at ``level``."""

    columns: list[str] = []
    column = column_for_level(existing, level)
    if column is not None:
        columns.append(column)
    if level in {"hourly", "daily", "weekly"}:
        month = column_for_level(existing, "monthly")
        if month is not None:
            columns.append(month)
    if level == "quarterly":
        year = column_for_level(existing, "yearly")
        if year is not None:
            columns.append(year)
    return _dedupe(columns)


def period_expr_for_level(existing: set[str], level: str) -> pl.Expr:
    """Return the hive partition period expression for a physical level."""

    builder = _PERIOD_BUILDERS.get(level)
    if builder is None:
        return pl.lit("ALL")
    return builder(existing)


def _hourly_period_expr(existing: set[str]) -> pl.Expr:
    day = column_for_level(existing, "daily")
    if day is not None:
        return pl.col(day).cast(pl.String)
    hour = column_for_level(existing, "hourly")
    if hour is not None:
        return pl.col(hour).cast(pl.String).str.slice(0, 10)
    return pl.lit("ALL")


def _daily_period_expr(existing: set[str]) -> pl.Expr:
    month = column_for_level(existing, "monthly")
    if month is not None:
        return pl.col(month).cast(pl.String)
    day = column_for_level(existing, "daily")
    if day is not None:
        return pl.col(day).cast(pl.String).str.slice(0, 7)
    return pl.lit("ALL")


def _single_level_period_expr(existing: set[str], level: str) -> pl.Expr:
    column = column_for_level(existing, level)
    if column is not None:
        return pl.col(column).cast(pl.String)
    return pl.lit("ALL")


_PERIOD_BUILDERS = {
    "hourly": _hourly_period_expr,
    "daily": _daily_period_expr,
    "weekly": lambda existing: _single_level_period_expr(existing, "weekly"),
    "monthly": lambda existing: _single_level_period_expr(existing, "monthly"),
    "quarterly": lambda existing: _single_level_period_expr(existing, "quarterly"),
    "yearly": lambda existing: _single_level_period_expr(existing, "yearly"),
}


def _dedupe(columns: list[str]) -> list[str]:
    out: list[str] = []
    for column in columns:
        if column not in out:
            out.append(column)
    return out


__all__ = [
    "CompactionPlan",
    "aggregate_grain_candidates",
    "base_period_expr",
    "chunk_time_group_columns",
    "column_for_level",
    "finest_configured_level",
    "period_expr_for_level",
    "prepare_compaction",
    "with_calendar_columns",
]
