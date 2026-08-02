# ADR 0008 — Bounded Rolling DuckDB Processor State

**Status:** Accepted (2026-08-02)

## Context

[ADR 0007](0007-bounded-lookback-processors.md) permits a bounded processor to
retain minimal, rebuildable identity state without weakening Value Stream's
aggregate-first query contract. The first persistent implementation used
per-day Parquet customer shards. A per-day immutable-DuckDB layout was
considered, but target calculation would still have to open every retained day
and rebuild the same union for each target.

Frequency targets already form a strict temporal sequence. After target D is
calculated, only exposed rank-1 contacts from D can affect D+1 and later; the
complete rank>1 candidate population is needed only while D itself is current.
Persisting a complete target role per day therefore spends I/O and storage on
data that has no historical use. A rolling relation can retain only the bounded
history that the next target needs and let DuckDB reuse one open catalog and
buffer pool across the source run.

The rolling design introduces mutable acceleration state, so it must not become
a publication marker or a second business source. It needs a transactional
fingerprint journal, deterministic processing order, bounded retention, and an
unambiguous rebuild path from authoritative Interaction History.

## Decision

`frequency_response.checkpoint.mode: persistent_sharded` remains the catalog
spelling for compatibility. The current and only supported checkpoint schema
is revision 8. It selects **bounded rolling DuckDB state**, not immutable
per-day shard files.

There is one stable database path for each source and processor:

```text
.valuestream/state/frequency_response/source=<source>/processor=<processor>/rolling.duckdb
```

The path deliberately has no schema, hashing, Polars-version, processor-config,
or layout level. Schema revision, hashing revision and seeds, Polars version,
processor computation hash, logical shard count, history projection, customer
dtype, and DuckDB version are recorded inside the database metadata. The
sharding contract uses `pl.Expr.hash`, so Polars version participates in
compatibility and a change rebuilds the same stable database. DuckDB version is
audit metadata; DuckDB itself validates whether it can open the storage file.
Raw-file fingerprints and chunk ids live in the transactional journal rather
than in the path. The state contains:

- bounded exposed rank-1 history normalized to one row per contact and source
  chunk (earliest decision time and source order, or-combined outcome flags —
  MIN/BOOL_OR are associative, so normalized rows combine exactly like the raw
  rows they replace), including its source `chunk_id` and persisted logical
  customer shard. Under `window_granularity: daily` the projection instead
  keeps one row per contact, canonical UTC decision day, and source chunk while
  dropping time-of-day, source order, and outcome detail. The per-target SQL
  first deduplicates that daily identity globally across retained source chunks
  and then reduces it to per-day exposure counters instead of joining history
  into the contact union; and
- a transactional journal mapping every retained/processed ISO-date chunk to
  the authoritative raw fingerprint, alongside database-level state-schema
  metadata needed for validation.

It does not persist the complete current candidate payload. For each target,
the transformed, filtered, classified current payload is streamed through the
Arrow C Stream interface into a temporary DuckDB relation owned by the current
writer session. The existing exact SQL combines that temporary relation with
the retained history one logical shard at a time and performs:

- cross-partition contact normalization and deterministic duplicate/outcome
  precedence;
- the strict `(decision_time - window, decision_time]` number-of-impressions
  count and terminal-bucket cap; and
- decision-local selected rank-2 action resolution, including configured
  `alternative_group_by` scope and the smallest-rank-greater-than-one fallback.

DuckDB streams the enriched selected rank-1 target rows to the existing Polars
tail for virtual state columns, state-level `where`, configured count/value-sum
aggregation, grouping, and provenance. `source_scan` remains the correctness
reference; both paths must produce equivalent aggregate states, subject only to
ordinary floating-point reduction tolerance.

The DuckDB connection never crosses a thread boundary. A single queried shard
uses the batched lazy Arrow-to-Polars stream. When several shards are queried,
the owning thread may fetch the next shard into a detached Arrow table while a
single worker applies the Polars tail to the preceding detached table. This
keeps overlap without concurrent connection access and bounds detached focal
data to two complete shard results. DuckDB `memory_limit` does not cover those
detached Arrow/Polars allocations; logical shard sizing is the explicit bound.

Before either execution path performs contact normalization, decision time is
canonicalized to timezone-naive UTC truncated to whole seconds. Sub-second
precision is deliberately not semantic for frequency windows: whole-second
instants are exact in both engines, so DuckDB interval arithmetic and ASOF
comparisons match Polars bit-for-bit instead of relying on sub-microsecond
timestamp behavior. Calendar-day derivation interprets that representation as
UTC before converting to the configured reporting timezone. This also makes
grouping or state predicates on the raw decision-time field identical between
`source_scan` and rolling SQL.
Other dictionary-backed projected fields are canonicalized to strings in the
source-scan relational tail to match DuckDB's Arrow `VARCHAR` representation.
The customer key is still hashed in its original logical dtype before crossing
that boundary.

A source coordinator holds one long-lived DuckDB writer session for each
persistent frequency processor. Frequency targets run in ascending ISO-date
order through those processor-specific databases. The source's chunk process
pool is capped at one even when the operator requests a larger `--parallel`
value; ordinary processor semantics and other source runs are unchanged. The
ordered sessions avoid concurrent mutation and reuse each open database,
catalog, and buffer pool. The connection remains open across the complete
source run and closes once at the source-run boundary, rather than reopening
the database for every target.

For a successfully calculated target, appending its narrow exposed-rank-1
history, recording its raw fingerprint, and pruning expired history/journal
rows occur in one DuckDB transaction. That state transaction is an ingestion
acceleration commit, not an aggregate or chunk-ledger commit. It may finish
before the normal aggregate Parquet/lineage/chunk publication barrier. A later
aggregate write failure therefore cannot expose a report row, and a state row
cannot authorize query visibility or idempotent reuse.

Before **each pending target**, the session reconciles its journal with that
target's expected ordered history closure and authoritative raw fingerprints.
This is necessary because the aggregate ledger may skip an already-published
intermediate chunk that rolling history still needs. Entries outside the exact
closure—an expired prefix or a state-ahead suffix during replay/retry—are
removed. The retained entries must then be an exact fingerprinted prefix; the
missing suffix is prepared from authoritative IH in date order without
publishing aggregates. A fingerprint or order/non-prefix mismatch resets and
rebuilds that closure. Structurally valid but incompatible acceleration
state—including an unsupported schema revision, changed processor computation
hash, hashing contract, logical shard count, or history projection—is replaced
at the same stable path and rebuilt from authoritative IH; checkpoint schemas
are not migrated. The replacement is initialized and validated in a sibling
temporary database before an atomic same-directory swap, so initialization
failure leaves the prior rebuildable state intact. Corruption,
source/processor identity tampering, and
customer-dtype drift still fail closed. `--force` replaces such state and
rebuilds it from authoritative IH oldest-to-newest. Source corrections continue
to invalidate the normal bounded target closure through the chunk ledger;
rolling-state reconciliation is an additional acceleration-state safety check,
not a replacement for that fingerprint.

The database retains at most `checkpoint.retention_days` source-day journal
entries. The configured value must remain at least
`ceil((window_hours + partition_lag_hours) / 24)`. Thus the default 168-hour
window with zero partition lag retains only the last seven calendar source
days after each chronological target commit; missing dates can leave fewer than
seven stored chunk entries. Expired history and journal rows are deleted in
that same transaction after every committed source day. The processor still
applies the exact timestamp interval `(decision_time - window, decision_time]`;
source-day retention is only the bounded physical closure.

Every 30 committed source days, the writer executes DuckDB `CHECKPOINT` on the
same open connection. This folds the expected WAL and partially reclaims or
reuses logically deleted-row space without discarding the catalog or buffer
pool. The connection is not closed for maintenance; normal close happens once
at the source-run boundary and performs the final checkpoint. Logical deletion
and checkpointing do not promise complete compaction or immediate file-size
shrinkage.
Retention is also applied when an existing database is opened, even if the
aggregate ledger skips every discovered target. `--force` opens and replaces
each configured rolling database even when the source currently discovers no
chunks, so an empty forced run cannot leave stale rolling history behind.

A WAL is expected while the long-lived writer is active and after a crash that
requires DuckDB recovery. It is part of the database's transactional behavior,
not an incomplete-generation signal. Clean close/checkpoint normally folds it
into `rolling.duckdb`; operators and vacuum must not delete a live WAL
independently.

Earlier per-day checkpoints, nested identity layouts, and checkpoint schema
revisions before 8 are unsupported acceleration artifacts. They are not
migrated; the rolling database is reconstructed from authoritative IH and
obsolete layouts may be vacuumed. Every field under `checkpoint` remains
execution/storage policy outside processor and source computation hashes. The
shard count still selects the logical partitioning used to bound SQL working
data; `threads` and a positive absolute `memory_limit` tune the rolling DuckDB
connection. None selects another database path.

This decision changes only ingestion acceleration state. Published business
aggregates remain immutable, hive-partitioned Parquet. Reports, API/MCP,
governed SQL, aggregate views, and query planning cannot address
`.valuestream/state/` or `rolling.duckdb`.

Schema validation treats the normalized projection as a physical invariant,
not an optimization hint. Every persisted projection column must be non-null,
and `projection key + chunk_id + shard` must be unique. A reopened database
that violates either condition is corrupt acceleration state and fails closed
rather than silently inflating a daily counter.

## Consequences

- A source run opens one rolling database per persistent frequency processor
  instead of opening several state files for every day and target. Only
  historically useful exposed rank-1 rows persist.
- Native DuckDB SQL still performs exact normalization, windowing, and selected
  rank-2 joins; the configurable state tail remains in Polars.
- Deterministic oldest-to-newest processing and one writer remove checkpoint
  write races, but a source containing a persistent frequency processor cannot
  use chunk-process parallelism. Increasing `--parallel` still benefits other
  eligible sources/runs.
- The transactional journal makes state-ahead-of-aggregate retries detectable
  without treating state as published data. Out-of-closure rows are trimmed;
  every retained non-prefix or fingerprint mismatch resets the required
  closure. Valid incompatible state is rebuilt automatically; corrupt state
  fails closed and `--force` rebuilds it.
- Retention bounds live rows, not necessarily the physical database high-water
  mark. Logical pruning occurs after every source-day commit, and checkpointing
  on the same connection every 30 commits reuses free space.
- DuckDB WAL files are normal operational state and require the same workspace
  access, backup, and crash-recovery treatment as `rolling.duckdb`.
- Logical customer sharding is routing, not anonymization. The retained key and
  exact event order remain sensitive source-derived data requiring workspace
  controls, bounded retention, and upstream tokenization/HMAC where applicable.
  Metadata distinguishes the source's logical customer dtype from DuckDB's
  physical storage dtype, so dictionary-backed `Categorical`/`Enum` keys remain
  valid without weakening logical dtype-drift detection.
