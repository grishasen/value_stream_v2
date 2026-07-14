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

## Consequences

- `valuestream run` is safe for cron and humans alike; re-runs are no-ops
  unless inputs or config actually changed.
- A failed replacement leaves the previous successful version visible —
  readers never observe partial state.
- Config changes automatically invalidate exactly the chunks whose
  computation they affect.
- Superseded aggregate files accumulate and require `vacuum`.
- Everything hinges on stable filename grouping (`group_by_filename`), which
  is why [probe](../../reference/cli.md#probe) exists.
