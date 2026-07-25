"""The model layer and the engine must agree on effective processor states.

``model.effective_processor_states`` feeds catalog validation, the KPI recipe
library, and every UI surface, while the engine's ``state_specs`` drives what
ingestion actually computes. Any divergence makes recipes and validation lie
about which aggregates exist.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter

from valuestream.config import model
from valuestream.processors.registry import create_processor

_PROCESSOR_ADAPTER = TypeAdapter(model.Processor)

_PARITY_CONFIGS: list[dict[str, Any]] = [
    {
        "id": "engagement",
        "source": "events",
        "kind": "binary_outcome",
        "time": {"property": "OutcomeTime", "grain": "daily"},
        "states": {
            "Count": {"type": "count"},
            "Positives": {"type": "count", "outcome": "positive"},
            "Negatives": {"type": "count", "outcome": "negative"},
        },
        "outcome": {
            "column": "Outcome",
            "positive_values": [1],
            "negative_values": [0],
        },
        "entities": {"subject": "SubjectID"},
    },
    {
        "id": "latency",
        "source": "events",
        "kind": "numeric_distribution",
        "time": {"property": "EventTime", "grain": "daily"},
        "properties": ["ResponseTime", "Cost"],
        "states": {
            "ResponseTime_tdigest": {
                "type": "tdigest",
                "source_column": "ResponseTime",
            },
            "Cost_tdigest": {"type": "tdigest", "source_column": "Cost"},
        },
    },
    {
        "id": "scores",
        "source": "events",
        "kind": "score_distribution",
        "time": {"property": "OutcomeTime", "grain": "daily"},
        "score_properties": [{"column": "Propensity", "role": "primary"}],
        "states": {
            "Count": {"type": "count"},
            "Propensity_tdigest": {
                "type": "tdigest",
                "source_column": "Propensity",
            },
        },
        "outcome": {
            "column": "Outcome",
            "positive_values": [1],
            "negative_values": [0],
        },
    },
    {
        "id": "lifecycle",
        "source": "orders",
        "kind": "entity_lifecycle",
        "time": {"property": "PurchasedDate", "grain": "daily"},
        "keys": {
            "customer_id": "CustomerID",
            "order_id": "OrderID",
            "monetary": "Amount",
            "purchase_date": "PurchasedDate",
        },
        "states": {
            "unique_holdings": {
                "type": "count",
                "source_column": "OrderID",
                "distinct": True,
            },
            "lifetime_value": {
                "type": "value_sum",
                "source_column": "Amount",
            },
            "MaxPurchasedDate": {
                "type": "max",
                "source_column": "PurchasedDate",
            },
        },
    },
    {
        "id": "cohort",
        "source": "events",
        "kind": "entity_set",
        "time": {"property": "EventTime", "grain": "daily"},
        "entity": "SubjectID",
        "states": {
            "Customers_theta": {
                "type": "theta",
                "source_column": "SubjectID",
            }
        },
    },
    {
        "id": "funnel",
        "source": "events",
        "kind": "funnel",
        "time": {"property": "EventTime", "grain": "daily"},
        "stages": [
            {"name": "Impression", "when": {"col": "Impression"}},
            {"name": "Conversion", "when": {"col": "Conversion"}},
        ],
        "states": {
            "Impression_Count": {"type": "count", "stage": "Impression"},
            "Conversion_Count": {"type": "count", "stage": "Conversion"},
        },
    },
    {
        "id": "book",
        "source": "holdings",
        "kind": "snapshot",
        "time": {"property": "SnapshotDate", "grain": "daily"},
        "snapshot_kind": "periodic",
        "entity": "AccountID",
        "states": {"Balance_Sum": {"type": "value_sum", "source_column": "Balance"}},
    },
]


@pytest.mark.unit
@pytest.mark.parametrize("config", _PARITY_CONFIGS, ids=[item["id"] for item in _PARITY_CONFIGS])
def test_engine_state_specs_match_model_effective_states(config: dict[str, Any]) -> None:
    processor = _PROCESSOR_ADAPTER.validate_python(config)
    runtime = create_processor(processor)

    assert runtime.state_specs == model.effective_processor_states(processor)
