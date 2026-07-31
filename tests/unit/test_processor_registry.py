"""Runtime processor registry coverage."""

from __future__ import annotations

import pytest

from valuestream.config import model
from valuestream.processors.binary_outcome import BinaryOutcomeProcessor
from valuestream.processors.frequency_response import FrequencyResponseProcessor
from valuestream.processors.registry import create_processor, processor_kinds, register_processor


@pytest.mark.unit
def test_registry_constructs_builtin_processor() -> None:
    config = model.BinaryOutcomeProcessor.model_validate(
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
        }
    )

    processor = create_processor(config, computation_hash="computed")

    assert isinstance(processor, BinaryOutcomeProcessor)
    assert processor.config_hash == "computed"
    assert "binary_outcome" in processor_kinds()


@pytest.mark.unit
def test_registry_rejects_duplicate_kind_without_explicit_replace() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_processor("binary_outcome", BinaryOutcomeProcessor)


@pytest.mark.unit
def test_registry_constructs_frequency_response_processor() -> None:
    config = model.FrequencyResponseProcessor.model_validate(
        {
            "id": "frequency",
            "source": "events",
            "kind": "frequency_response",
            "group_by": ["Day", "ExposureBucket"],
            "time": {"property": "DecisionTime", "grain": "daily"},
            "columns": {
                "customer": "CustomerID",
                "interaction": "InteractionID",
                "action": "ActionID",
                "placement": "Placement",
                "rank": "Rank",
                "outcome": "Outcome",
                "propensity": "Propensity",
            },
            "positive_values": ["Clicked"],
            "exposure_values": ["Impression"],
            "candidate_values": ["Pending"],
            "states": {"Contacts": {"type": "count"}},
        }
    )

    processor = create_processor(config, computation_hash="computed")

    assert isinstance(processor, FrequencyResponseProcessor)
    assert processor.config_hash == "computed"
    assert "frequency_response" in processor_kinds()
