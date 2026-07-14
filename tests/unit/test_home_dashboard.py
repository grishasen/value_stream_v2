"""Focused tests for Home dashboard summary helpers."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import polars as pl

from valuestream.ui import components
from valuestream.ui.pages import home


def test_dashboard_inventory_counts_pages_and_tiles() -> None:
    ctx = SimpleNamespace(
        catalog=SimpleNamespace(
            dashboards=SimpleNamespace(
                dashboards=[
                    SimpleNamespace(
                        pages=[
                            SimpleNamespace(tiles=[object(), object()]),
                            SimpleNamespace(tiles=[object()]),
                        ]
                    ),
                    SimpleNamespace(pages=[SimpleNamespace(tiles=[])]),
                ]
            )
        )
    )

    assert home._dashboard_inventory(ctx) == (3, 3)


def test_validation_issue_counts_split_errors_and_warnings() -> None:
    ctx = SimpleNamespace(
        validation=SimpleNamespace(
            issues=[
                SimpleNamespace(severity="error"),
                SimpleNamespace(severity="warning"),
                SimpleNamespace(severity="info"),
            ]
        )
    )

    assert home._validation_issue_counts(ctx) == {"errors": 1, "warnings": 2}


def test_recent_runs_display_prioritizes_operational_columns() -> None:
    frame = pl.DataFrame(
        {
            "id": ["d253d960-5f3d-4d2c-9b12-e604b5c8d731"],
            "source_id": ["ih"],
            "status": ["ok"],
            "rows_kept": [120_000_000],
            "finished_at": [dt.datetime(2026, 7, 9, 23, 6)],
            "chunks_total": [120],
        }
    )

    display = home._recent_runs_display(frame)

    assert display.columns == ["Source", "Status", "Rows kept", "Finished", "Run"]
    assert display.row(0, named=True)["Run"] == "d253d960"
    assert display.row(0, named=True)["Status"] == "Ok"
    assert display.row(0, named=True)["Rows kept"] == "120,000,000"


def test_compact_metric_formatting_prevents_card_clipping() -> None:
    assert components.format_compact_number(120_000_000) == "120M"
    assert components.format_compact_number(50_519.0) == "50.5K"
    assert components.format_metric_value(349_321) == "349K"
    assert components.format_metric_value(0.08, "percent") == "8.00%"
    assert components.format_metric_value(6_000_000.0, "integer") == "6,000,000"
