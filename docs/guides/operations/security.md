# Security

Value Stream's security posture in one page: what the surfaces expose, how
access is controlled, and what never leaves the aggregate layer. Intended for
operators deciding how to host the API/MCP surfaces and for reviewers
assessing exposure.

## The Aggregate-Only Contract

Every read surface — Reports, Chat With Data, CLI `query`, the Python SDK,
DuckDB export, MCP, and the HTTP API — reads through the same governed
aggregate query layer. None of them expose raw source rows, raw aggregate
parquet paths, or filesystem access. Raw event rows do not survive chunk
processing; the durable store contains only mergeable aggregate statistics
with provenance columns.

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
- Every LiteLLM call logs provider settings, prompts, responses, and timing.
  API key values are not logged, but prompts may contain catalog metadata and
  approved sample values from AI Configuration Studio.

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
- [FAQ §F](../../reference/faq.md) — security and compliance questions.
