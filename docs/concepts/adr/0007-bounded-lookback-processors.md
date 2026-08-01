# ADR 0007 — Bounded Lookback Processors and Rebuildable Checkpoints

**Status:** Accepted (2026-07-31; amended 2026-07-31 and 2026-08-01)

## Context

Most Value Stream processors reduce one source chunk independently. That is
the cheapest and simplest aggregate-first contract, but it cannot assign an
event to an immutable rolling-frequency bucket: the bucket for a contact today
depends on earlier contacts for the same customer, action, and placement.
Daily counts cannot reconstruct that identity-level order after ingestion.

The aggregate-first boundary exists to keep long-term storage small and to keep
report rendering independent of raw source volume. It is an architectural
preference and a query-serving contract, not a prohibition on every internal
identity-level artifact. Requiring the exact calculation to reread overlapping
source closures for every target day preserves aggregate-only persistence, but
duplicates source I/O and makes ingestion unnecessarily slow.

A pre-enriched source or a report-visible customer contact table would create a
second source of business truth. A bounded, processor-owned checkpoint is
different when it is only an internal acceleration structure: its contents are
minimal, versioned, immutable, and completely rebuildable from the authoritative
source files. Reports never read it.

The chunk ledger also needs more information than the target day's own file
fingerprint. If day D contributes to a seven-day frequency calculation for day
D+1 through D+7, correcting D must invalidate those dependent targets.

## Decision

Value Stream supports an explicitly bounded lookback contract for processor
kinds whose typed configuration declares one. The first such kind is
`frequency_response`.

- The discovered daily chunk remains the target unit of idempotency, output,
  provenance, and progress reporting.
- For a target day, the runner plans a dependency closure containing the target
  files and the preceding calendar partitions needed to cover the configured
  window plus an explicit partition-lag allowance. The selected execution mode
  supplies either marked transformed rows or the matching prepared-contact
  checkpoint shards.
- Other processors bound to the same source continue to receive only the
  target chunk. The lookback contract does not change their populations.
- The ledger fingerprints and records the complete dependency file set for the
  target. A changed dependency therefore reprocesses every affected target and
  no targets outside the bounded window.
- The report/query contract remains aggregate-first. The processor publishes
  only grouped, mergeable daily states, and every query surface reads those
  aggregates through the governed query layer.
- The initial contract is daily and requires ISO `YYYY-MM-DD` chunk ids so the
  dependency closure is a calendar calculation rather than an assumption
  about filename sort order.

A bounded processor may choose one of two exact execution strategies:

1. `source_scan` constructs an ephemeral transformed dependency frame for each
   target and persists no processor state. This remains the compatibility
   default and the correctness reference.
2. `persistent_sharded` filters, classifies, and projects each source chunk
   once per processor into a processor-owned checkpoint, uses a partitioned
   streaming sink to route it by a deterministic customer hash, and reads only
   matching shards for the target and bounded history. Candidate rows are
   retained before cross-partition contact collapse so duplicate/outcome
   precedence remains exactly equivalent to `source_scan`.
   The checkpoint may retain identity-level keys and values needed for exact
   ordering, deduplication, exposure frequency, grouping/state calculation, and
   within-interaction runner selection.

The strategy is operational rather than semantic. `checkpoint.mode`,
`checkpoint.shards`, and `checkpoint.retention_days` remain in the full catalog
hash for authorship and audit, but are excluded from processor and source
computation hashes. Consequently, changing checkpoint execution or storage
policy does not republish otherwise identical aggregates. Counts are exact
across layouts; ordinary floating-point sums retain their normal machine-
precision reduction tolerance.

Persistent processor state is permitted only under these constraints:

- Interaction History remains authoritative. A checkpoint is not a Source,
  aggregate, business record, publication marker, or queryable table. Deleting
  it may make ingestion slower, but cannot change a reported result.
- The stored schema is the smallest prepared candidate state required by the
  processor; unrelated source columns and outcomes are excluded. State is
  bounded by the declared dependency semantics and its retention policy,
  rather than becoming an unbounded raw-event archive.
- Every generation is identity-addressed by source, processor semantic
  computation identity, independent checkpoint-layout identity, chunk id, and
  raw-file fingerprint. The layout identity currently covers shard count;
  explicit state-schema and sharding revisions plus the hashing-runtime version
  cover format and routing compatibility. Writes are atomic and ordinary-run
  generations are immutable. A changed source file, processor semantic
  contract, checkpoint layout, state schema, sharding algorithm, or incompatible
  runtime builds a new generation instead of mutating one in place. An explicit
  forced run rebuilds and safely replaces the same address as well, so a
  metadata-preserving source-file replacement cannot force reuse of stale
  acceleration state.
- Every existing manifest and shard is size-, row-count-, schema-, file-set-,
  and SHA-256-validated once in the parent run before workers open individual
  shards. The recorded customer identity dtype must agree across a target's
  complete closure; dtype drift fails that target instead of routing equal
  logical keys to different shards silently.
- Checkpoints do not create chunk-ledger `ok` rows and are not covered by query
  publication. The target aggregate still commits through the normal lineage
  and chunk-ledger barrier, whose fingerprint covers the complete raw
  dependency closure.
- Stale or incomplete generations are independently vacuumable. A current
  generation may always be reconstructed by replaying the authoritative IH
  chunks, so processor-state backup is optional unless an operator values faster
  recovery over storage.
- `checkpoint.retention_days` bounds the number of daily checkpoint partitions
  retained per processor. It defaults to
  `ceil((window_hours + partition_lag_hours) / 24) + 1` and cannot be configured
  below that active closure. Retention is applied after a terminal ingestion
  run and by workspace vacuum. An older correction can rebuild an evicted
  partition from IH for that replay and discard it again afterward.
- `checkpoint.mode` selects the execution path, `checkpoint.shards` selects the
  checkpoint layout, and `checkpoint.retention_days` selects lifecycle policy.
  None changes aggregate identity. A shard-layout change is built lazily for
  the bounded closure of the next new or invalidated target; vacuum may remove
  the previous layout immediately. A retention-only change applies at vacuum
  without rebuilding state. `--force` remains an explicit request to rebuild
  aggregates and state regardless of idempotent hashes.
- Customer-hash sharding is an execution index, not anonymization. Exact
  processing can retain the original customer key inside a shard, so workspace
  access controls, encryption/retention policy, and upstream tokenization or
  HMAC remain necessary where identity data is sensitive.

Sketches or customer sampling may be used only when a KPI explicitly declares
approximate semantics. They are not substituted silently for the exact
frequency-response contract: sketches cannot recover per-customer event order
or join a focal action to the available runner within the same interaction.

For `frequency_response`, the exposure window is `(decision_time - W,
decision_time]`, with `W=168h` by default. Frequencies are capped into the
configured terminal bucket. This precise boundary is part of the processor
computation hash.

`partition_lag_hours` is a separate dependency-planning parameter, defaulting
to zero. It covers sources partitioned by a timestamp that can trail decision
time: the runner reads `ceil((W + partition_lag_hours) / 24)` prior calendar
partitions, but the processor still applies exactly `(decision_time - W,
decision_time]`. Correctness therefore requires the partition-date displacement
to stay within the configured allowance. Over-padding changes I/O and
invalidation fan-out, never exposure membership.

## Consequences

- Frequency-response curves can be produced directly from the existing
  Interaction History source without a loader, secondary source, or
  pre-enriched business table.
- Source corrections invalidate the dependent daily aggregates automatically;
  interrupted-run recovery verifies the same raw dependency fingerprints.
- `source_scan` trades repeated lookback I/O for zero retained identity state.
  `persistent_sharded` trades governed local storage for one checkpoint-
  preparation pass per source generation and bounded one-shard-at-a-time
  processing.
- Checkpoint tuning does not replay historical aggregates. Missing state for a
  new layout is reconstructed lazily from only the authoritative chunks needed
  by the next bounded target, while unchanged targets remain skipped.
- Chunks remain independently retryable and may run in parallel because
  workers read published generations and do not share mutable per-customer
  state. Checkpoint preparation errors are mapped only to target chunks whose
  dependency closures use the failed generation; independent chunks continue.
- Persistent mode increases data-at-rest and compliance scope. Operators must
  apply the same workspace isolation and retention controls used for sensitive
  source-derived data and must not treat hash sharding as de-identification.
- A source whose chunk ids are not daily ISO dates must be repartitioned or use
  another processor. The engine fails before writing rather than guessing.
- Outcome concepts absent from the source remain absent. In particular, an
  Impression can be an exposure proxy, but the processor cannot invent
  viewability or dismiss/irritation events.
