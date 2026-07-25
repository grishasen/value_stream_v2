# ADR 0006 — Strict Catalog Version 2

**Status:** Accepted (2026-07-25)

## Context

The catalog accepted missing versions, unknown fields, renamed fields, chart
aliases, inferred processor states, and fallback report outputs. Those
compatibility paths made invalid authoring appear to work until ingestion or
rendering, and made the same YAML behave differently depending on runtime
inference.

## Decision

All four catalog files require `catalog_version: 2` and use strict,
discriminated models with unknown fields forbidden.

- Processors declare only `group_by` dimensions and an explicit typed `states`
  output contract. Time columns at or above the processor's base grain may be
  included directly in `group_by`.
- State names are identifiers only. Computation semantics live in typed fields:
  counts use an optional `source_column`, `distinct`, `outcome`, or `stage`;
  pooled states name their weight/mean companions; and sketches name their
  source column. Runtimes never infer behavior from suffixes or familiar state
  names.
- Score processors use `score_properties`.
- Metrics reference a `processor`. Distribution and quantile metrics are
  distinct kinds. Set operations use typed operands and the operation name
  `minus`. Lifecycle summaries explicitly bind their holdings, monetary,
  first-purchase, and last-purchase states.
- Tiles use chart-specific contracts. Scalar measure roles come from the
  selected metric; `metric_output` is required only when a multi-output metric
  does not have one unambiguous scalar output.
- Dimensions and metric outputs must match exact configured names. No aliases,
  inferred page filters, first-numeric output selection, or legacy catalog
  migration path remains.

## Consequences

- Invalid catalogs fail at load time with an exact field path.
- Builder output, query planning, and report rendering share one canonical
  contract.
- Existing pre-v2 catalogs must be rewritten before use; the application does
  not translate or backfill them.
- Adding a processor state, metric kind, or chart role requires an explicit
  schema and corresponding runtime support.
