# ADR 0002 — YAML Catalog as the Source of Behavior

**Status:** Accepted (backfilled 2026-07-13 from the replacement design)

## Context

The legacy dashboard's metrics and pages were partly hand-coded, so business
changes required code changes, and there was no single reviewable artifact
that defined what the system computed.

## Decision

All behavior is defined by a declarative workspace catalog — `pipelines.yaml`,
`processors.yaml`, `metrics.yaml`, `dashboards.yaml` — validated against JSON
Schemas and Pydantic models. The loader rejects duplicate keys and IDs,
validates references and dependency cycles, and computes canonical hashes used
for audit and reprocessing decisions. The Builder and AI Configuration Studio
are editors over the same YAML, never a second configuration store.

## Consequences

- Configuration is reviewable in source control and diffable across
  environments; catalog hashes make "what changed" answerable.
- New metrics, tiles, and dashboards ship without code changes; genuinely new
  computation kinds still require code plus schema plus docs.
- Validation becomes the universal gate: every editing path ends with
  `valuestream validate`.
- The catalog schema itself must be versioned and documented
  ([catalog schemas](../../reference/catalog-schemas.md)).

See [replacement design §7](../../design/replacement-design.md) for the DSL.
