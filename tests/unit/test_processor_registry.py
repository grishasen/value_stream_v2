"""Runtime processor registry coverage."""

from __future__ import annotations

import pytest

from valuestream.config import model
from valuestream.processors.binary_outcome import BinaryOutcomeProcessor
from valuestream.processors.registry import create_processor, processor_kinds, register_processor


@pytest.mark.unit
def test_registry_constructs_builtin_processor() -> None:
    config = model.BinaryOutcomeProcessor.model_validate(
        {"id": "engagement", "source": "events", "kind": "binary_outcome"}
    )

    processor = create_processor(config, computation_hash="computed")

    assert isinstance(processor, BinaryOutcomeProcessor)
    assert processor.config_hash == "computed"
    assert "binary_outcome" in processor_kinds()


@pytest.mark.unit
def test_registry_rejects_duplicate_kind_without_explicit_replace() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_processor("binary_outcome", BinaryOutcomeProcessor)
