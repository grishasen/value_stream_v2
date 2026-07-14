"""Runtime processor registry and factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from valuestream.config import model
from valuestream.processors.binary_outcome import BinaryOutcomeProcessor
from valuestream.processors.entity_lifecycle import EntityLifecycleProcessor
from valuestream.processors.entity_set import EntitySetProcessor
from valuestream.processors.funnel import FunnelProcessor
from valuestream.processors.numeric_distribution import NumericDistributionProcessor
from valuestream.processors.score_distribution import ScoreDistributionProcessor
from valuestream.processors.snapshot import SnapshotProcessor

ProcessorRuntime: TypeAlias = (
    BinaryOutcomeProcessor
    | NumericDistributionProcessor
    | ScoreDistributionProcessor
    | EntityLifecycleProcessor
    | EntitySetProcessor
    | FunnelProcessor
    | SnapshotProcessor
)
ProcessorFactory: TypeAlias = Callable[..., ProcessorRuntime]

_PROCESSOR_FACTORIES: dict[str, ProcessorFactory] = {
    "binary_outcome": BinaryOutcomeProcessor,
    "numeric_distribution": NumericDistributionProcessor,
    "score_distribution": ScoreDistributionProcessor,
    "entity_lifecycle": EntityLifecycleProcessor,
    "entity_set": EntitySetProcessor,
    "funnel": FunnelProcessor,
    "snapshot": SnapshotProcessor,
}


def register_processor(
    kind: str,
    factory: ProcessorFactory,
    *,
    replace: bool = False,
) -> None:
    """Register one runtime processor factory."""

    normalized = kind.strip()
    if not normalized:
        raise ValueError("processor kind must not be empty")
    if normalized in _PROCESSOR_FACTORIES and not replace:
        raise ValueError(f"processor kind {normalized!r} is already registered")
    _PROCESSOR_FACTORIES[normalized] = factory


def create_processor(
    config: model.Processor,
    *,
    computation_hash: str | None = None,
) -> ProcessorRuntime:
    """Construct the registered runtime for a typed processor config."""

    factory = _PROCESSOR_FACTORIES.get(config.kind)
    if factory is None:
        supported = ", ".join(sorted(_PROCESSOR_FACTORIES))
        raise ValueError(
            f"no runtime registered for processor kind {config.kind!r}; supported: {supported}"
        )
    return factory(config, computation_hash=computation_hash)


def processor_kinds() -> tuple[str, ...]:
    """Return registered runtime kinds in stable order."""

    return tuple(sorted(_PROCESSOR_FACTORIES))


__all__ = [
    "ProcessorFactory",
    "ProcessorRuntime",
    "create_processor",
    "processor_kinds",
    "register_processor",
]
