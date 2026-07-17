"""Canonical form and hashing for catalog configs.

Per ``docs/EXPRESSION_DSL.md`` §7, the canonical form of a YAML config:

* sorts dict keys recursively,
* normalizes numeric scalars so ``1`` and ``1.0`` hash the same when the
  value is integral,
* unquotes string scalars (we serialize as JSON, so quoting is uniform),
* rewrites ``op: when_then`` chains as the equivalent ``op: case`` form,
* drops ``None`` valued optional fields so default and explicit-null
  spellings collide.

The output is a deterministic UTF-8 byte sequence; ``config_hash`` is the
sha256 of that sequence. Two YAML files that parse to the same AST
produce the same hash.

The serialization format is canonical JSON (RFC 8785-style: sorted keys,
no whitespace, no extra precision). JSON is a strict YAML 1.2 subset, so
this is consistent with the spec language about "canonicalized YAML"
while removing PyYAML's quoting and float-formatting variability.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from valuestream.config import model
from valuestream.utils.hashing import sha256_hex

COMPUTATION_SCHEMA_VERSION = 2
_SCORE_BOUNDED_SAMPLE_ORDER_REVISION = 1


def canonicalize(value: Any) -> Any:
    """Recursively normalize ``value`` into a canonical Python representation.

    * ``dict``: keys sorted, ``None`` values dropped, values canonicalized.
    * ``list``: elements canonicalized in place.
    * Numeric scalars: integral floats collapse to ``int`` (``1.0`` → ``1``).
    * ``op: when_then`` dicts rewrite to ``op: case`` with one branch.
    * Pydantic models dump to dicts via ``model_dump(by_alias=True,
      exclude_none=True)`` first.
    """
    if isinstance(value, BaseModel):
        return canonicalize(value.model_dump(by_alias=True, exclude_none=True))
    if isinstance(value, dict):
        rewritten = _maybe_rewrite_when_then(value)
        return {
            k: canonicalize(v)
            for k, v in sorted(rewritten.items(), key=lambda kv: kv[0])
            if v is not None
        }
    if isinstance(value, list):
        return [canonicalize(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # Collapse integral floats. ``int(1.0) == 1``; ``int(1.5) == 1`` so
        # we only collapse when the round-trip is lossless.
        as_int = int(value)
        if as_int == value:
            return as_int
        return value
    return value


def _maybe_rewrite_when_then(value: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ``{op: when_then, cond, then, else}`` to the equivalent ``op: case``."""
    if value.get("op") == "when_then" and {"cond", "then", "else"} <= value.keys():
        return {
            "op": "case",
            "when": [{"cond": value["cond"], "then": value["then"]}],
            "else": value["else"],
        }
    return value


def serialize(value: Any) -> bytes:
    """Return the canonical JSON byte sequence for ``value``."""
    return json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def config_hash(value: Any) -> str:
    """Return the lowercase 64-char sha256 hex digest of ``value``'s canonical form."""
    return sha256_hex(serialize(value))


def processor_config_hash(processor: model.Processor) -> str:
    """Hash one processor model without its upstream source contract.

    Direct processor construction uses this as a compatibility fallback. The
    ingestion/query paths persist :func:`processor_computation_hash` instead.
    """
    return config_hash(processor)


def processor_computation_hash(
    catalog: model.Catalog,
    processor: model.Processor,
) -> str:
    """Hash every declarative input that can affect one processor's rows."""

    return config_hash(processor_computation_config(catalog, processor))


def processor_computation_config(
    catalog: model.Catalog,
    processor: model.Processor,
) -> dict[str, Any]:
    """Return the canonicalizable computation contract for one processor."""

    source = next(
        (candidate for candidate in catalog.pipelines.sources if candidate.id == processor.source),
        None,
    )
    if source is None:
        raise ValueError(
            f"processor {processor.id!r} references unknown source {processor.source!r}"
        )
    return {
        "computation_schema_version": COMPUTATION_SCHEMA_VERSION,
        "workspace_defaults": catalog.pipelines.defaults,
        "source": _source_computation_fields(source),
        "processor": _processor_computation_fields(processor),
    }


def source_computation_hash(catalog: model.Catalog, source_id: str) -> str:
    """Hash source ingestion behavior and every processor fed by the source."""

    return config_hash(source_computation_config(catalog, source_id))


def source_computation_config(catalog: model.Catalog, source_id: str) -> dict[str, Any]:
    """Return the canonicalizable computation contract for one source run."""

    source = next(
        (candidate for candidate in catalog.pipelines.sources if candidate.id == source_id), None
    )
    if source is None:
        raise ValueError(f"unknown source {source_id!r}")
    processors = sorted(
        (
            _processor_computation_fields(processor)
            for processor in catalog.processors.processors
            if processor.source == source_id
        ),
        key=lambda processor: str(processor["id"]),
    )
    return {
        "computation_schema_version": COMPUTATION_SCHEMA_VERSION,
        "workspace_defaults": catalog.pipelines.defaults,
        "source": _source_computation_fields(source),
        "processors": processors,
    }


def _source_computation_fields(source: model.Source) -> dict[str, Any]:
    payload = source.model_dump(by_alias=True, exclude_none=True)
    for field in ("description", "debugging", "materialize_transforms"):
        payload.pop(field, None)
    reader = payload.get("reader")
    if isinstance(reader, dict):
        reader.pop("debugging", None)
        reader.pop("streaming", None)
    return payload


def _processor_computation_fields(processor: model.Processor) -> dict[str, Any]:
    payload = processor.model_dump(by_alias=True, exclude_none=True)
    for field in ("description", "sketch_build_mode"):
        payload.pop(field, None)
    if isinstance(processor, model.ScoreDistributionProcessor) and {
        "personalization",
        "novelty",
    }.intersection(model.effective_processor_states(processor)):
        payload["__valuestream_algorithm_revision"] = {
            "bounded_ml_source_order": _SCORE_BOUNDED_SAMPLE_ORDER_REVISION
        }
    return payload


def catalog_config_hash(catalog: model.Catalog) -> str:
    """Hash the whole assembled catalog — used as the workspace-level identifier."""
    return config_hash(catalog)


__all__ = [
    "COMPUTATION_SCHEMA_VERSION",
    "canonicalize",
    "catalog_config_hash",
    "config_hash",
    "processor_computation_config",
    "processor_computation_hash",
    "processor_config_hash",
    "serialize",
    "source_computation_config",
    "source_computation_hash",
]
