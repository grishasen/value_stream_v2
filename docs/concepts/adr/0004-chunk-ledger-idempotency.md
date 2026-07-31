# ADR 0004 — Chunk Ledger with Computation Hashes

**Status:** Accepted (backfilled 2026-07-13 from the replacement design)

## Context

Ingestion consumes batches of files that arrive repeatedly and partially, jobs
get re-run by schedulers and humans, and configuration changes over time.
Without a deterministic unit of processing, re-runs double-count and partial
failures leave the store inconsistent.

## Decision

The chunk — a filename-derived group of input files — is the idempotent unit
of ingestion. A metadata ledger (DuckDB) records each chunk's status, its
input-file fingerprint, and the source computation hash (which incorporates
upstream catalog behavior). A completed chunk is skipped only when both hash
and fingerprint match; `--force` overrides. Writes use immutable run-specific
files plus atomic rename, and query visibility is gated by successful chunk
and run ledger rows.

The run row is created as `running` before the first chunk and acts as the
outer publication barrier. Within a chunk, atomic aggregate files are followed
by one complete lineage transaction; only then is `chunks.status='ok'`
inserted as the durable chunk commit marker. The existing run row is finally
updated to `ok`, `partial`, or `failed`.

For each `(source, chunk)`, publication selects one globally latest successful
attempt from terminal runs. A later successful attempt supersedes the prior one
even when no processor emits aggregate rows; the ledger row therefore acts as
an empty-result tombstone without persisting a data row. A failed attempt does
not replace the last success. Idempotent skip checks use the same latest-success
rule, so rolling configuration back to an older hash causes a recomputation
instead of making a previously superseded attempt visible again.

After acquiring a source lock, a new invocation treats older `running` rows for
that source as interrupted. It publishes a stale run as `partial` only after
verifying retained chunks against current input fingerprints, source and
processor computation hashes, lineage, physical files, and embedded
provenance. Unverified chunks are made failed/invisible and processed again.
This permits recovery without introducing a second chunk identity. Optional
processor checkpoints are independently validated acceleration state; they do
not create ledger rows or participate in query publication.

## Consequences

- `valuestream run` is safe for cron and humans alike; re-runs are no-ops
  unless inputs or config actually changed.
- A failed replacement leaves the previous successful version visible —
  readers never observe partial state.
- A successful empty replacement hides the previous non-empty version without
  requiring a synthetic aggregate row.
- A hard-terminated run reuses already committed and verified chunks on the
  next invocation instead of replaying them from raw input.
- Config changes automatically invalidate exactly the chunks whose
  computation they affect.
- Superseded aggregate files accumulate and require `vacuum`.
- Everything hinges on stable filename grouping (`group_by_filename`), which
  is why [probe](../../reference/cli.md#probe) exists.
