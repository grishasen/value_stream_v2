# ADR 0010 — Sample-Backed MCP Catalog Authoring

**Status:** Accepted (2026-10-13)

## Context

The aggregate MCP surface requires a prepared catalog. A client should also be
able to create a usable workspace from a source sample and business goals,
using the same configuration contracts as AI Configuration Studio.

## Decision

- Keep authoring absent by default; enable it explicitly with
  `--enable-authoring` on the shared stdio/HTTP factory. The REST API stays
  read-only. Authoring can start with an absent workspace/catalog, without
  writing anything at startup; partial/invalid existing catalogs still fail.
- Let the MCP client model generate proposals using Studio prompt dictionaries.
  The server makes no provider calls and requires no model credentials.
- Share bounded CSV/Parquet reads, Studio catalog/sample-field validators,
  object diff, and rollback-protected application. Studio retains its UI gates.
- Constrain MCP samples to existing files inside workspace `data/`. Return
  schema/counts by default, examples only for explicitly selected fields.
  Never retain sample rows or publish preview aggregates. Reader configuration
  stays bound to the exact sample; transforms and analytics remain editable.
- Create a deterministic complete baseline that counts all records, with no
  inferred outcome or temporal semantics. Preserve existing unsampled sources
  and their artifacts when adding a new sampled source.
- Retain at most 32 process-local drafts, with a one-hour lifetime. Use opaque
  ids, compare revisions on update, and invalidate review after every update.
  These are not per-user sessions or permissions.
- Preview full configuration and changes before issuing a review token for a
  valid revision. The client obtains user authorization before applying.
- Apply revalidates and checks the sample metadata fingerprint and exact base
  configuration hash. Compare configuration and write inside the shared
  workspace lock, atomically replace individual YAML files, and restore their
  original bytes on an exception. This does not provide an atomic snapshot to
  concurrent readers or exclude unrelated manual writes.
- Return a receipt with config hash and whether a data run is required.
  Idempotent retries return the successful receipt; ingestion remains separate.

## Consequences

Existing query clients retain their tool contracts and aggregate-first serving.
Authoring is an explicit sample-access exception, visible in deployment and
security documentation. All HTTP token holders can apply configuration when
enabled. Restart/expiry requires recreating uncommitted drafts; there is no raw
sample upload, remote sample staging, or draft database. JSON/archive authoring
remains a Studio workflow. Bounded validation cannot guarantee that unseen
production rows satisfy the same schema or business assumptions.
