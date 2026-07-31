# ADR 0001 — Aggregate-First Storage

**Status:** Accepted (backfilled 2026-07-13; clarified by ADR 0007 on 2026-07-31)

## Context

The legacy CDH Value Dashboard queried raw event exports repeatedly. Large
file exports were slow to re-query, raw rows accumulated without bound, and
every report carried the cost and exposure of raw-event access. Most business
questions were answered from a small set of grouped statistics.

## Decision

Raw event rows are reduced to small, mergeable sufficient statistics (counts,
sums, sketches, digests, funnel states, snapshots) during ingestion. Those
aggregates are the only persisted business/query contract: every read surface —
reports, chat, CLI, SDK, API, MCP, and SQL export — reads through the aggregate
query layer.

Aggregate-first is a strong storage and serving preference, chosen to minimize
retained data and the amount processed while rendering reports. It is not a
rigid ban on processor-internal state. As specified by
[ADR 0007](0007-bounded-lookback-processors.md), a bounded processor may retain
minimal identity-level state when it is needed to compute exact aggregates
efficiently. Such state must be versioned, bounded, non-queryable, and
rebuildable from the authoritative source; it does not become a second source
of business truth.

## Consequences

- Queries stay fast and cheap regardless of source volume; storage is compact.
- Distinct counts, quantiles, and model-quality curves are approximate
  (bounded-error sketches) and are labeled as such in reports.
- Adding a new group-by dimension or changing outcome definitions usually
  requires authoritative source replay, because checkpoints are not a general
  raw-event history or query substrate.
- Raw-event SQL and a raw event warehouse are permanently out of scope.
- The aggregate-only **read-surface** contract remains a security boundary.
  Processor checkpoints that retain identity-level state expand the
  data-at-rest boundary and require explicit access and retention controls
  ([Security](../../guides/operations/security.md)).

See [replacement design §4–6](../../design/replacement-design.md) for the full
rationale and storage layout.
