"""KPI card values shared by the report page and the read-only tool surfaces.

A KPI tile is not just a metric query: it resolves the tile's output column,
picks the comparison window from the authored ``kpi`` spec, and returns the
previous-period delta and sparkline alongside the value. Reproducing that
arithmetic per surface is how the UI and the tool layer drift apart, so it
lives here. Nothing in this module imports Streamlit.
"""

from __future__ import annotations

import calendar
import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from valuestream.config import model
from valuestream.config.report_fields import metric_output_columns
from valuestream.query import query_metric
from valuestream.reporting.tiles import QueryFn, tile_to_dict

_SERIES_TIME_COLUMNS = ("Day", "day", "as_of_date", "Week", "Month", "month")


@dataclass(frozen=True)
class KpiBundle:
    """Display-ready values for one explicitly configured KPI."""

    value: float | int | str
    delta: float | int | None = None
    delta_description: str | None = None
    sparkline: tuple[float, ...] | None = None
    period_description: str = "All time"
    value_column: str = ""


def kpi_bundle(
    workspace_path: str | Path,
    catalog: model.Catalog,
    tile: model.Tile,
    *,
    filters: Mapping[str, Any] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    query_fn: QueryFn | None = None,
) -> KpiBundle:
    """Return the value, comparison delta, and sparkline for one KPI tile."""

    run_query = query_fn if query_fn is not None else query_metric
    tile_dict = tile_to_dict(tile)
    metric = catalog.metrics.metrics[tile.metric]
    output_columns = metric_output_columns(tile.metric, metric)
    selected_output = str(tile_dict.get("metric_output") or "")
    value_column = selected_output if selected_output in output_columns else output_columns[0]
    # Fixed tile filters define the KPI population; interactive page filters
    # may narrow other fields but cannot replace that authored scope.
    query_filters = {**dict(filters or {}), **dict(tile_dict.get("filters") or {})}
    kpi = tile.kpi or model.KpiSpec()

    series = pl.DataFrame()
    if kpi.sparkline_grain or kpi.comparison == "previous_period":
        series = run_query(
            workspace_path,
            tile.metric,
            filters=query_filters,
            grain=kpi.sparkline_grain or "daily",
        )

    current_start, current_end = start, end
    period_description = "All time"
    if start is not None and end is not None:
        period_description = f"{start.isoformat()} to {end.isoformat()}"
    elif kpi.comparison == "previous_period":
        latest = latest_series_date(series)
        if latest is not None:
            current_start, current_end = calendar_period(latest, kpi.comparison_period)
            period_description = period_label(current_start, current_end, kpi.comparison_period)

    current_rows = run_query(
        workspace_path,
        tile.metric,
        filters=query_filters,
        grain="summary",
        start=current_start,
        end=current_end,
    )
    value = scalar_value(current_rows, value_column)
    delta: float | int | None = None
    delta_description: str | None = None
    if kpi.comparison == "previous_period" and current_start and current_end:
        day_count = (current_end - current_start).days + 1
        previous_end = current_start - dt.timedelta(days=1)
        previous_start = previous_end - dt.timedelta(days=day_count - 1)
        previous_rows = run_query(
            workspace_path,
            tile.metric,
            filters=query_filters,
            grain="summary",
            start=previous_start,
            end=previous_end,
        )
        previous_value = scalar_value(previous_rows, value_column)
        if isinstance(value, int | float) and isinstance(previous_value, int | float):
            delta = value - previous_value
            delta_description = comparison_period_label(previous_start, previous_end)
    elif kpi.target is not None and isinstance(value, int | float):
        delta = value - kpi.target
        delta_description = f"Target {kpi.target:g}"

    return KpiBundle(
        value=value,
        delta=delta,
        delta_description=delta_description,
        sparkline=sparkline_values(series, value_column, kpi.sparkline_points),
        period_description=period_description,
        value_column=value_column,
    )


def scalar_value(rows: pl.DataFrame, column: str) -> float | int | str:
    """Return the one numeric value a KPI query produced, or ``"n/a"``."""

    if rows.is_empty() or column not in rows.columns:
        return "n/a"
    values = rows.get_column(column).drop_nulls()
    if values.len() != 1:
        return "n/a"
    value = values.item()
    return value if isinstance(value, int | float) else "n/a"


def sparkline_values(
    rows: pl.DataFrame,
    column: str,
    points: int,
) -> tuple[float, ...] | None:
    """Return the trailing sparkline series, or None when it cannot be drawn."""

    if rows.is_empty() or column not in rows.columns or not rows.schema[column].is_numeric():
        return None
    time_column = series_time_column(rows)
    ordered = rows.sort(time_column) if time_column else rows
    values = ordered.get_column(column).drop_nulls().tail(points).to_list()
    return tuple(float(value) for value in values) if len(values) >= 2 else None


def latest_series_date(rows: pl.DataFrame) -> dt.date | None:
    """Return the newest period present in a time series."""

    column = series_time_column(rows)
    if column is None or rows.is_empty():
        return None
    values = rows.get_column(column).drop_nulls().cast(pl.String).to_list()
    parsed = [parse_period_date(str(value)) for value in values]
    return max((value for value in parsed if value is not None), default=None)


def series_time_column(rows: pl.DataFrame) -> str | None:
    """Return the time column a series is ordered by, if it has one."""

    return next((column for column in _SERIES_TIME_COLUMNS if column in rows.columns), None)


def parse_period_date(value: str) -> dt.date | None:
    """Parse a daily or monthly aggregate period label into a date."""

    for pattern in ("%Y-%m-%d", "%Y-%m"):
        try:
            parsed = dt.datetime.strptime(value[:10], pattern).date()
        except ValueError:
            continue
        if pattern == "%Y-%m":
            return dt.date(parsed.year, parsed.month, 1)
        return parsed
    return None


def calendar_period(value: dt.date, period: str) -> tuple[dt.date, dt.date]:
    """Return the calendar period bounds containing ``value``."""

    if period == "day":
        return value, value
    if period == "week":
        start = value - dt.timedelta(days=value.weekday())
        return start, start + dt.timedelta(days=6)
    if period == "month":
        return (
            dt.date(value.year, value.month, 1),
            dt.date(value.year, value.month, calendar.monthrange(value.year, value.month)[1]),
        )
    if period == "quarter":
        first_month = 3 * ((value.month - 1) // 3) + 1
        last_month = first_month + 2
        return (
            dt.date(value.year, first_month, 1),
            dt.date(value.year, last_month, calendar.monthrange(value.year, last_month)[1]),
        )
    return dt.date(value.year, 1, 1), dt.date(value.year, 12, 31)


def period_label(start: dt.date, end: dt.date, period: str) -> str:
    """Describe a KPI's current period in words."""

    if period == "month":
        return start.strftime("%B %Y")
    if period == "quarter":
        return f"Q{((start.month - 1) // 3) + 1} {start.year}"
    if period == "year":
        return str(start.year)
    return f"{start.isoformat()} to {end.isoformat()}"


def comparison_period_label(start: dt.date, end: dt.date) -> str:
    """Describe the period a KPI delta is measured against."""

    if start.year == end.year and start.month == end.month:
        return f"vs {start.strftime('%b')} {start.day}-{end.day}, {end.year}"
    return f"vs {start.strftime('%b')} {start.day}-{end.strftime('%b')} {end.day}, {end.year}"


__all__ = [
    "KpiBundle",
    "calendar_period",
    "comparison_period_label",
    "kpi_bundle",
    "latest_series_date",
    "parse_period_date",
    "period_label",
    "scalar_value",
    "series_time_column",
    "sparkline_values",
]
