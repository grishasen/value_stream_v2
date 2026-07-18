"""Focused coverage for data-aware report time presets."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

from valuestream.ui.pages.reports import (
    _empty_tile_message,
    _latest_data_notice,
    _latest_page_data_coverage,
    _relative_time_bounds,
)


@pytest.mark.unit
def test_latest_page_data_coverage_uses_end_of_latest_aggregate_month() -> None:
    latest_date, label = _latest_page_data_coverage(
        [
            SimpleNamespace(latest_period="2024-08"),
            SimpleNamespace(latest_period="2024-09"),
            SimpleNamespace(latest_period=None),
        ]
    )

    assert latest_date == dt.date(2024, 9, 30)
    assert label == "2024-09"


@pytest.mark.unit
def test_relative_time_presets_anchor_to_historical_data_coverage() -> None:
    start, end = _relative_time_bounds(
        "last_30_days",
        today=dt.date(2026, 7, 18),
        latest_data_date=dt.date(2024, 9, 30),
    )

    assert start == dt.date(2024, 9, 1)
    assert end == dt.date(2024, 9, 30)


@pytest.mark.unit
def test_relative_time_presets_do_not_move_past_today_for_future_data() -> None:
    start, end = _relative_time_bounds(
        "last_7_days",
        today=dt.date(2026, 7, 18),
        latest_data_date=dt.date(2026, 8, 31),
    )

    assert start == dt.date(2026, 7, 12)
    assert end == dt.date(2026, 7, 18)


@pytest.mark.unit
def test_year_to_date_uses_latest_data_year_but_custom_is_not_clamped() -> None:
    assert _relative_time_bounds(
        "year_to_date",
        today=dt.date(2026, 7, 18),
        latest_data_date=dt.date(2024, 9, 30),
    ) == (dt.date(2024, 1, 1), dt.date(2024, 9, 30))
    assert _relative_time_bounds(
        "custom",
        today=dt.date(2026, 7, 18),
        latest_data_date=dt.date(2024, 9, 30),
    ) == (None, None)


@pytest.mark.unit
def test_latest_data_notice_only_describes_clamped_relative_ranges() -> None:
    assert (
        _latest_data_notice(
            "last_30_days",
            today=dt.date(2026, 7, 18),
            latest_data_date=dt.date(2024, 9, 30),
            latest_data_label="2024-09",
        )
        == "Showing latest available data (through 2024-09)."
    )
    assert (
        _latest_data_notice(
            "custom",
            today=dt.date(2026, 7, 18),
            latest_data_date=dt.date(2024, 9, 30),
            latest_data_label="2024-09",
        )
        is None
    )


@pytest.mark.unit
def test_empty_filtered_range_offers_explicit_show_all_recovery() -> None:
    message, recoverable = _empty_tile_message(
        filters={"Channel": ["Mobile"]},
        start=dt.date(2024, 8, 1),
        end=dt.date(2024, 8, 31),
        ignored_filters=["Region"],
    )

    assert recoverable
    assert "date range 2024-08-01 to 2024-08-31" in message
    assert "1 active filter(s)" in message
    assert "unsupported by this chart: Region" in message


@pytest.mark.unit
def test_empty_all_time_result_is_distinct_from_filter_mismatch() -> None:
    message, recoverable = _empty_tile_message(
        filters={},
        start=None,
        end=None,
        ignored_filters=[],
    )

    assert not recoverable
    assert "not materialized" in message
    assert "Data Load" in message
