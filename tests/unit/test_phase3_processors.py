"""Focused Phase 3 processor tests."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from valuestream.config import model
from valuestream.processors.binary_outcome import ChunkContext
from valuestream.processors.snapshot import SnapshotProcessor
from valuestream.states import hll


@pytest.mark.unit
def test_periodic_snapshot_does_not_group_by_entity() -> None:
    processor = SnapshotProcessor(
        model.SnapshotProcessor.model_validate(
            {
                "id": "daily_customers",
                "source": "events",
                "kind": "snapshot",
                "snapshot_kind": "periodic",
                "cadence": "daily",
                "group_by": ["Channel", "Issue"],
                "entity": "CustomerID",
                "as_of_column": "Day",
                "states": {
                    "Count": {"type": "count"},
                    "ActiveCustomers_hll": {
                        "type": "hll",
                        "source_column": "CustomerID",
                        "lg_k": 12,
                    },
                },
            }
        )
    )
    ctx = ChunkContext("run", "chunk", dt.datetime(2024, 1, 2, tzinfo=dt.UTC))

    out = processor.chunk_aggregate(
        pl.DataFrame(
            {
                "Day": [dt.date(2024, 1, 1), dt.date(2024, 1, 1), dt.date(2024, 1, 1)],
                "Channel": ["Web", "Web", "Web"],
                "Issue": ["Cards", "Cards", "Cards"],
                "CustomerID": ["c1", "c2", "c1"],
            }
        ).lazy(),
        ctx,
    )

    assert out.height == 1
    assert "CustomerID" not in out.columns
    assert out["Count"][0] == 3
    assert hll.estimate(out["ActiveCustomers_hll"][0]) == pytest.approx(2.0, rel=0.02)


@pytest.mark.unit
def test_accumulating_snapshot_latest_entity_wins() -> None:
    processor = SnapshotProcessor(
        model.SnapshotProcessor.model_validate(
            {
                "id": "tickets",
                "source": "ticket_source",
                "kind": "snapshot",
                "snapshot_kind": "accumulating",
                "group_by": ["Team"],
                "entity": "TicketID",
                "milestones": [
                    {"name": "created_at", "column": "CreatedAt"},
                    {"name": "resolved_at", "column": "ResolvedAt"},
                ],
                "states": {
                    "OpenTickets": {
                        "type": "count",
                        "where": {"op": "is_null", "column": "ResolvedAt"},
                    },
                    "ResolvedTickets": {
                        "type": "count",
                        "where": {"op": "not_null", "column": "ResolvedAt"},
                    },
                },
            }
        )
    )
    ctx = ChunkContext("run", "chunk", dt.datetime(2024, 1, 3, tzinfo=dt.UTC))
    first = processor.chunk_aggregate(
        pl.DataFrame(
            {
                "TicketID": ["t1", "t2"],
                "Team": ["Support", "Support"],
                "CreatedAt": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1)],
                "ResolvedAt": [None, None],
            },
            schema={
                "TicketID": pl.String,
                "Team": pl.String,
                "CreatedAt": pl.Datetime("us"),
                "ResolvedAt": pl.Datetime("us"),
            },
        ).lazy(),
        ctx,
    )
    second = processor.chunk_aggregate(
        pl.DataFrame(
            {
                "TicketID": ["t1", "t2"],
                "Team": ["Support", "Support"],
                "CreatedAt": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1)],
                "ResolvedAt": [dt.datetime(2024, 1, 2), None],
            },
            schema={
                "TicketID": pl.String,
                "Team": pl.String,
                "CreatedAt": pl.Datetime("us"),
                "ResolvedAt": pl.Datetime("us"),
            },
        ).lazy(),
        ctx,
    )

    merged = processor.merge_for_query(pl.concat([first, second]), ["Team"])

    assert merged["OpenTickets"][0] == 1
    assert merged["ResolvedTickets"][0] == 1
