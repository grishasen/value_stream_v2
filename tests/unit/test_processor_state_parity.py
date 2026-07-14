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
    {"id": "engagement", "source": "events", "kind": "binary_outcome"},
    {
        "id": "engagement_entities",
        "source": "events",
        "kind": "binary_outcome",
        "entities": {"subject": "SubjectID"},
    },
    {
        "id": "engagement_explicit",
        "source": "events",
        "kind": "binary_outcome",
        "states": {"Accepted": {"type": "count"}},
    },
    {
        "id": "latency",
        "source": "events",
        "kind": "numeric_distribution",
        "properties": ["ResponseTime", "Cost"],
    },
    {
        "id": "latency_kll",
        "source": "events",
        "kind": "numeric_distribution",
        "properties": ["ResponseTime"],
        "quantile_engine": "kll",
        "states": {"{prop}_topk": {"type": "topk", "source_column": "{prop}"}},
    },
    {"id": "scores", "source": "events", "kind": "score_distribution"},
    {
        "id": "scores_properties",
        "source": "events",
        "kind": "score_distribution",
        "score_properties": ["Propensity", "Priority"],
    },
    {"id": "lifecycle", "source": "orders", "kind": "entity_lifecycle"},
    {
        "id": "lifecycle_custom",
        "source": "orders",
        "kind": "entity_lifecycle",
        "keys": {"customer_id": "BuyerID", "monetary": "Amount"},
        "states": {"Channel_topk": {"type": "topk", "source_column": "Channel"}},
    },
    {"id": "cohort", "source": "events", "kind": "entity_set"},
    {
        "id": "cohort_explicit",
        "source": "events",
        "kind": "entity_set",
        "entity": "SubjectID",
        "states": {"Weekly_theta": {"type": "theta", "source_column": "SubjectID"}},
    },
    {
        "id": "funnel",
        "source": "events",
        "kind": "funnel",
        "stages": [
            {"name": "Impression", "when": {"col": "Impression"}},
            {"name": "Conversion", "when": {"col": "Conversion"}},
        ],
    },
    {
        "id": "funnel_entity",
        "source": "events",
        "kind": "funnel",
        "entity": "SubjectID",
        "stages": [
            {"name": "Impression", "when": {"col": "Impression"}},
            {"name": "Conversion", "when": {"col": "Conversion"}},
        ],
        "states": {"Impression_Customers_hll": {"type": "hll", "source_column": "SubjectID"}},
    },
    {"id": "book", "source": "holdings", "kind": "snapshot", "snapshot_kind": "periodic"},
    {
        "id": "book_explicit",
        "source": "holdings",
        "kind": "snapshot",
        "snapshot_kind": "accumulating",
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


@pytest.mark.unit
def test_model_layer_reports_engine_derived_funnel_and_lifecycle_states() -> None:
    funnel = _PROCESSOR_ADAPTER.validate_python(
        {
            "id": "funnel",
            "source": "events",
            "kind": "funnel",
            "entity": "SubjectID",
            "stages": [
                {"name": "Impression", "when": {"col": "Impression"}},
                {"name": "Conversion", "when": {"col": "Conversion"}},
            ],
        }
    )
    lifecycle = _PROCESSOR_ADAPTER.validate_python(
        {"id": "lifecycle", "source": "orders", "kind": "entity_lifecycle"}
    )

    funnel_states = model.effective_processor_states(funnel)
    lifecycle_states = model.effective_processor_states(lifecycle)

    assert {
        "Impression_Count",
        "Conversion_Count",
        "Impression_Customers_cpc",
        "Conversion_Customers_cpc",
    } <= set(funnel_states)
    assert {
        "unique_holdings",
        "lifetime_value",
        "MinPurchasedDate",
        "MaxPurchasedDate",
        "UniquePurchasers_cpc",
    } <= set(lifecycle_states)
