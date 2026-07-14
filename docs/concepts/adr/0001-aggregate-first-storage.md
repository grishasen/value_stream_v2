# ADR 0001 — Aggregate-First Storage

**Status:** Accepted (backfilled 2026-07-13 from the replacement design)

## Context

The legacy CDH Value Dashboard queried raw event exports repeatedly. Large
file exports were slow to re-query, raw rows accumulated without bound, and
every report carried the cost and exposure of raw-event access. Most business
questions were answered from a small set of grouped statistics.

## Decision

Raw event rows are reduced to small, mergeable sufficient statistics (counts,
sums, sketches, digests, funnel states, snapshots) during a single chunk pass,
and only those aggregates are persisted. Raw rows never survive the chunk
pass; every read surface — reports, chat, CLI, SDK, API, MCP, SQL export —
reads through the aggregate query layer.

## Consequences

- Queries stay fast and cheap regardless of source volume; storage is compact.
- Distinct counts, quantiles, and model-quality curves are approximate
  (bounded-error sketches) and are labeled as such in reports.
- Adding a new group-by dimension or changing outcome definitions usually
  requires raw source replay, because the raw rows are gone.
- Raw-event SQL and a raw event warehouse are permanently out of scope.
- The aggregate-only contract doubles as the security boundary
  ([Security](../../guides/operations/security.md)).

See [replacement design §4–6](../../design/replacement-design.md) for the full
rationale and storage layout.
