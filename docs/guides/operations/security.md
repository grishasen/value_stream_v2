# Security

Value Stream's security posture in one page: what the surfaces expose, how
access is controlled, and what never leaves the aggregate layer. Intended for
operators deciding how to host the API/MCP surfaces and for reviewers
assessing exposure.

## The Aggregate-Only Contract

Every read surface — Reports, Chat With Data, CLI `query`, the Python SDK,
DuckDB export, MCP, and the HTTP API — reads through the same governed
aggregate query layer. None of them expose raw source rows, raw aggregate
parquet paths, or filesystem access. Raw source rows never enter the governed
query store; it contains only mergeable aggregate statistics with provenance
columns.

Aggregate-first is the read-surface contract and preferred storage shape, not
a claim that every ingestion implementation is stateless. A bounded processor
may persist a minimal internal checkpoint when necessary for efficient exact
aggregation. That namespace is not registered with the query planner, DuckDB
views, SQL allowlist, API, MCP, Chat, SDK, or Reports.

## Processor Checkpoint Boundary

`frequency_response` can opt into `checkpoint.mode: persistent_sharded`. Treat
the resulting checkpoint as sensitive source-derived data:

- It contains only filtered candidate rows and fields required to repeat exact
  bounded history: exposed rank-1 contact identity, time, classification,
  deterministic order, chunk id, and logical shard. The complete current
  candidate payload is temporary, but the rolling history may retain an exact
  customer key.
- Customer hashing routes records to logical shards. It is not encryption,
  anonymization, or a substitute for upstream tokenization/HMAC.
- One schema-revision-7 `rolling.duckdb` uses the stable path
  `.valuestream/state/frequency_response/source=<source>/processor=<processor>/rolling.duckdb`;
  revision 7 is the default and only supported checkpoint schema. Schema,
  hashing, Polars-version, processor-config, and layout values do not add path
  levels. Compatibility/audit metadata remains inside the database. Because
  logical sharding uses `pl.Expr.hash`, a Polars-version change rebuilds the
  same stable database; DuckDB version is audit-only. Its journal records chunk
  fingerprints and is reconciled before every pending target.
  Exact-prefix gaps are filled from IH; only one state-ahead retry tail may be
  trimmed after aggregate publication failure. Changed/non-prefix,
  valid-but-incompatible, or forced state is rebuilt from authoritative IH
  rather than arbitrarily rewound. Corrupt, identity-invalid, or customer-dtype-
  drifted state fails closed unless explicitly forced. A DuckDB WAL is expected
  while the writer is active or recovering and must not be removed
  independently.
- The checkpoint is reconstructible from authoritative IH and has an
  independent, bounded retention lifecycle. `checkpoint.retention_days`
  defaults to `ceil((window_hours + partition_lag_hours) / 24)`; 168 hours with
  zero lag retains only the last seven calendar source days, with fewer stored
  chunk entries when dates are missing. The rolling writer prunes old history
  and journal rows after every chronological source-day commit. Every
  30 committed days, DuckDB `CHECKPOINT` runs on the same open connection to
  fold the WAL and partially reclaim or reuse deleted-row space; it does not
  guarantee complete compaction or immediate file shrinkage. The connection
  closes once at the source-run boundary. Backing the state up is optional when
  source replay is acceptable.
- Apply the same filesystem isolation, encryption-at-rest, deletion, and access
  controls as for other sensitive workspace-local data. Deleting checkpoint
  state changes ingestion time only; it never changes which data reports read.

## HTTP API Authentication

```sh
export VALUESTREAM_API_TOKEN=replace-me
uv run valuestream serve-api examples/demo --host 127.0.0.1 --port 8000
```

- Set a bearer token with `--token` or the `VALUESTREAM_API_TOKEN` environment
  variable; every endpoint except `GET /health` then requires
  `Authorization: Bearer <token>`.
- When no token is set the API is open — appropriate only for a trusted
  localhost deployment.
- `valuestream serve-api` **refuses a non-loopback bind** (anything other than
  `127.0.0.1`, `localhost`, or `::1`) unless a token is set.
- The API never mutates the catalog or aggregate store.

Remote HTTP MCP, OAuth/OIDC, and hosted multi-user auth are deferred; the
current surfaces are designed for trusted local or single-team use.

## Governed SQL Is Opt-In

SQL tools and endpoints are absent unless `--enable-sql` is passed to
`serve-mcp` or `serve-api` (and are opt-in inside Chat). When enabled, SQL is
tightly governed:

- Only a single read-only `SELECT` (or `WITH ... SELECT`) is accepted;
  comments, DDL/DML, multiple statements, and file/catalog functions such as
  `read_parquet` are rejected.
- Queries run only over allowlisted aggregate views
  (`meta/aggregate_views.duckdb`) and metric export tables
  (`meta/metric_export_*.duckdb`).
- Sketch state blob columns are masked from schemas and results; row counts
  are capped; long queries are interrupted.
- DuckDB external file access, automatic extension loading, and community
  extensions are disabled before user SQL executes.

## LLM and Chat Exposure

- Keep model API keys in environment variables referenced by
  `<workspace>/ai.yaml` (`api_key_env`), never in the file itself.
- Chat only sends catalog metadata and governed aggregate rows to the model,
  but those prompts leave the local app when using hosted model APIs. Do not
  use Chat With Data for sensitive raw samples.
- AI Configuration Studio requires confirmation of the current
  sample/provider/model sharing scope before sending a sample-backed prompt.
  Sample values are excluded by default and require per-field opt-in; changing
  the sample, model, provider, approved fields, or example selection
  invalidates the confirmation. The review identifies whether data uses the
  provider default or a configured custom endpoint. Approved schema names,
  types, null counts, and unique counts are disclosed even when examples are
  off; hidden field names are not sent. A sharing-scope change also clears prior
  Copilot context so echoed values cannot be forwarded into the new scope.
- LiteLLM calls log only privacy-safe operational metadata such as the call
  identifier, a redacted model identifier, duration, outcome, and safe token
  counts. Normal logs do not include prompt or response bodies, API keys,
  sample values, API-base values, or raw provider exception payloads. Provider
  failures are converted to safe reference errors before callers can display or
  log them. Invalid model-generated Chat plans are likewise converted to a safe
  status before UI or API error handling can log generated values. Studio
  sample-read and preprocessing failures log only a bounded error type, because
  parser exception text can echo an offending cell.
- Governed SQL logs a query hash, statement kind, length, cap, and result
  counts. It does not log SQL text, literal values, or workspace paths.

## Authoring Funnel Analytics

The optional Build rollout uses privacy-safe application log events rather
than a generic analytics payload. The event API accepts only enumerated
workflow, stage, event, and outcome values plus bounded duration/count values
and a materialization-required flag. An anonymous random journey ID lasts for
the browser session; it is not a user, workspace, sample, or catalog object ID.

There is deliberately no free-form metadata map. Sample and field values,
field/object names, workspace paths, prompts, responses, credentials,
endpoints, and provider exceptions therefore cannot be added at a call site.
See [Configuration authoring rollout](authoring-rollout.md) for the complete
event list and measurement procedure.

## Traceability

The catalog is validated and hashed; processor outputs carry provenance
columns (config hash, chunk, run, period, creation time); metadata databases
track runs, chunks, config versions, and lineage. API and MCP metric-query
responses include a provenance envelope with catalog/computation hashes and
contributing run/chunk IDs, so any reported number can be traced to its
inputs. See [Business functionality](../../concepts/business-functionality.md)
for the governance view.

## Related Docs

- [API & MCP reference](../../reference/api-and-mcp.md) — endpoints, tools,
  and error mapping.
- [Deployment](deployment.md) — hosting choices that this posture constrains.
- [Configuration authoring rollout](authoring-rollout.md) — feature flag,
  privacy-safe event contract, and rollout gates.
- [FAQ §F](../../reference/faq.md) — security and compliance questions.
