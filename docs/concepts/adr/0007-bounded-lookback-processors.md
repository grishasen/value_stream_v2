# ADR 0007 — Bounded Lookback Processors and Rebuildable Checkpoints

**Status:** Accepted (2026-07-31; amended 2026-07-31, 2026-08-01, and 2026-08-02)

The bounded-state and exactness decision remains current. The current and only
supported schema-revision-7 physical contract is the bounded rolling DuckDB
state defined in
[ADR 0008](0008-bounded-rolling-duckdb-state.md).

## Context

Most Value Stream processors reduce one source chunk independently. That is
the cheapest and simplest aggregate-first contract, but it cannot assign an
event to an immutable rolling number-of-impressions bucket: the bucket for a contact today
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
minimal, versioned, transactionally reconciled, bounded, and completely
rebuildable from the authoritative source files. Reports never read it.

The chunk ledger also needs more information than the target day's own file
fingerprint. If day D contributes to a seven-day number-of-impressions calculation for day
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
  rolling state partitioned by logical customer shard.
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
   once per target into a temporary current relation inside one long-lived
   processor-owned DuckDB writer. The complete current payload arrives through
   the Arrow C Stream interface and is not persisted. Exact SQL combines it
   with bounded rolling history one logical customer shard at a time, then
   persists only exposed rank-1 identity, time, classification, local order,
   chunk id, and shard for later targets. Historical alternatives and
   target-only grouping/state values are excluded. Frequency targets execute
   oldest-to-newest in one process so the rolling state has one writer. ADR
   0008 defines the schema-revision-7 physical, reconciliation, maintenance,
   and WAL contract.

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
- Persisted history uses the smallest prepared state required by later targets:
  exposed rank-1 identity, time, classification, deterministic order, source
  chunk id, and logical customer shard. Unrelated source columns, rank>1
  alternatives, and target-only state/grouping values are excluded. Live rows
  are bounded by the declared retention policy rather than becoming an
  unbounded raw-event archive.
- One stable `rolling.duckdb` is addressed only by source and processor at
  `.valuestream/state/frequency_response/source=<source>/processor=<processor>/rolling.duckdb`.
  Schema and hashing revisions, Polars version, processor computation hash,
  logical shard count, history projection, and customer dtype are compatibility
  metadata rather than directory levels; DuckDB version is audit-only. Chunk
  ids and raw fingerprints are stored in a transactional journal inside the
  database.
- Before each pending target, the writer reconciles that journal with the
  target's expected ordered history closure and authoritative fingerprints. An
  expired prefix or state-ahead suffix outside that closure is removed. The
  remaining journal must be an exact fingerprinted prefix; its missing suffix
  is filled from IH, which covers intermediate chunks skipped by aggregate-
  ledger reuse. Fingerprint or order/non-prefix mismatch resets and rebuilds
  the required closure. Structurally valid but incompatible state is replaced
  at the same stable path and rebuilt from IH; old schemas are not migrated.
  Corrupt, identity-tampered, or customer-dtype-drifted state fails closed;
  `--force` replaces it and rebuilds from IH oldest-to-newest.
- Appending one target's narrow history, recording its fingerprint, and pruning
  expired rows are one DuckDB transaction. That acceleration-state commit may
  precede aggregate publication, but it never creates a chunk-ledger `ok` row
  or authorizes query visibility. The aggregate still commits through normal
  Parquet, lineage, run, and chunk-ledger barriers whose fingerprint covers the
  complete raw dependency closure.
- One long-lived writer owns the rolling database. Persistent frequency targets
  are processed in ascending ISO-date order and the source chunk process pool
  is capped at one. A DuckDB WAL is expected during the writer session and
  crash recovery; it must not be treated as an incomplete-generation file or
  deleted independently. The connection remains open for the source run and is
  closed once at the source-run boundary.
- `checkpoint.retention_days` bounds the number of source-day journal entries
  retained in each processor's rolling state. It defaults to
  `ceil((window_hours + partition_lag_hours) / 24)` and cannot be configured
  below that active closure. A 168-hour window with zero lag therefore retains
  only the last seven calendar source days; missing dates can leave fewer than
  seven stored chunk entries. Expired history/journal rows are deleted in every
  chronological source-day commit. Every 30 commits, DuckDB `CHECKPOINT`
  runs on the same open connection to fold the WAL and partially reclaim or
  reuse deleted-row space; the maintenance step does not close the writer or
  guarantee complete compaction or an immediate file-size reduction.
- `checkpoint.mode` selects the execution path, `checkpoint.shards` selects the
  logical SQL partitioning, and `checkpoint.retention_days` selects lifecycle
  policy. None changes aggregate identity or the stable database path. An
  incompatible shard or processor contract reinitializes that database from
  IH; a retention-only change is applied when the next pending target opens and
  commits through the writer.
  `--force` explicitly rebuilds aggregates and rolling state regardless of
  idempotent hashes.
- Customer-hash sharding is an execution index, not anonymization. Exact
  processing can retain the original customer key inside a shard, so workspace
  access controls, encryption/retention policy, and upstream tokenization or
  HMAC remain necessary where identity data is sensitive.

Sketches or customer sampling may be used only when a KPI explicitly declares
approximate semantics. They are not substituted silently for the exact
frequency-response contract: sketches cannot recover per-customer event order
or join a selected rank-1 action to the selected rank-2 action available in its
configured comparison group.

`frequency_response.alternative_group_by` makes that comparison group an
explicit semantic part of the processor. Customer and interaction are implicit
mandatory base keys: customer preserves source-scan/checkpoint equivalence and
shard isolation, while interaction keeps ranked candidates inside one decision.
The configured value is a zero-or-more list of physical transformed source
columns appended to them. The default `[Placement]` therefore means customer +
interaction + placement. Multiple additional fields are allowed, and null
values match null values within a configured group. Because this field changes
computed populations, it remains in processor/source computation hashes; it is
not checkpoint storage tuning.

For `frequency_response`, the impression-count window is `(decision_time - W,
decision_time]`, with `W=168h` by default. The number of impressions is capped
into the configured terminal bucket. This precise boundary is part of the
processor computation hash.

`partition_lag_hours` is a separate dependency-planning parameter, defaulting
to zero. It covers sources partitioned by a timestamp that can trail decision
time: the runner reads `ceil((W + partition_lag_hours) / 24)` prior calendar
partitions, but the processor still applies exactly `(decision_time - W,
decision_time]`. Correctness therefore requires the partition-date displacement
to stay within the configured allowance. Over-padding changes I/O and
invalidation fan-out, never impression-count membership.

## Consequences

- Response-by-number-of-impressions curves can be produced directly from the existing
  Interaction History source without a loader, secondary source, or
  pre-enriched business table.
- Source corrections invalidate the dependent daily aggregates automatically;
  interrupted-run recovery verifies the same raw dependency fingerprints.
- `source_scan` trades repeated lookback I/O for zero retained identity state.
  `persistent_sharded` trades governed local storage and sequential target
  execution for one reusable rolling DuckDB history and bounded one-shard-at-a-
  time SQL processing.
- Checkpoint tuning does not replay historical aggregates. Incompatible state
  at the stable source/processor path is reconstructed lazily from only the
  authoritative chunks needed by the next bounded target, while unchanged
  targets remain skipped.
- Chunks remain independently retryable, but a source containing persistent
  frequency state processes its targets oldest-to-newest in one process because
  the rolling database has one writer. Other processor semantics and eligible
  source runs remain unchanged.
- Persistent mode increases data-at-rest and compliance scope. Operators must
  apply the same workspace isolation and retention controls used for sensitive
  source-derived data and must not treat hash sharding as de-identification.
- A source whose chunk ids are not daily ISO dates must be repartitioned or use
  another processor. The engine fails before writing rather than guessing.
- Outcome concepts absent from the source remain absent. In particular, an
  Impression can be an exposure proxy, but the processor cannot invent
  viewability or dismiss/irritation events.
