# API and MCP Reference

The read-only HTTP API and the local stdio MCP server expose the same
governed tool layer over the aggregate query path. Neither surface mutates the
catalog or aggregate store, and neither exposes raw source rows. Design
rationale and provider setup live in
[Chat With Data MLP1](../design/chat-with-data-mlp1.md); the security posture
is summarized in [Security](../guides/operations/security.md).

```text
Streamlit Chat page      MCP client (stdio)          HTTP API client
        |                          |                          |
        v                          v                          v
LLM intent planner          stdio MCP tools           FastAPI endpoints
        |                          |                          |
        +----------- governed tool layer (one implementation) +
                         |
                         v
              query_metric / sql / freshness / manifest
```

Long-lived servers reload the catalog automatically when its YAML files change
on disk, so manifest and chart validation stay in sync with Config Builder
edits without a restart.

## Starting the Servers

```sh
uv sync --extra ai    # MCP dependency
uv sync --extra api   # FastAPI dependency

uv run valuestream serve-mcp WORKSPACE [--enable-sql]
uv run valuestream serve-api WORKSPACE --host 127.0.0.1 --port 8000 [--enable-sql]
```

CLI options are in the [CLI reference](cli.md#serve-api). Registration for
Claude Code:

```sh
claude mcp add valuestream -- uv run valuestream serve-mcp /absolute/path/to/workspace
```

## HTTP Endpoints

Interactive OpenAPI docs are served at `/docs`.

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness check (always open) |
| `GET /metrics` | Catalog manifest with per-metric dimensions, outputs, and chart kinds |
| `POST /metrics/{name}/query` | Run `query_metric` with filters, having, order_by, top_n, compare, quantiles |
| `POST /metrics/{name}/chart` | Validated chart spec plus rows |
| `GET /metrics/{name}/dimension-values` | Aggregate-backed dimension values |
| `GET /metrics/{name}/freshness` | Freshness metadata |
| `GET /sql/schema` | Governed DuckDB tables/views (only with `--enable-sql`) |
| `POST /sql` | One governed read-only SELECT (only with `--enable-sql`) |
| `POST /chat` | Plan and answer a natural-language question (requires a configured model) |

Error mapping: invalid requests return 400, missing aggregates 409, SQL
timeouts 504.

## Authentication

Set a bearer token with `--token` or `VALUESTREAM_API_TOKEN`; every endpoint
except `/health` then requires `Authorization: Bearer <token>`. With no token
set the API is open (trusted-localhost only), and `serve-api` refuses a
non-loopback bind.

## MCP Tools

| Tool | Purpose |
|---|---|
| `metric_list` | List metrics, dimensions, query time axes, and supported charts |
| `metric_query` | Query metric rows through `query_metric` (operator filters, having, order_by, top_n, compare, quantile suite) |
| `metric_chart_query` | Query metric rows and return an explicit validated chart spec |
| `dimension_values_tool` | Return aggregate-backed dimension values |
| `sql_schema` | List governed DuckDB tables/views and their non-masked columns (only with `--enable-sql`) |
| `sql_query` | Run one governed read-only SELECT over aggregate views and metric exports (only with `--enable-sql`) |
| `freshness_get` | Return metric freshness metadata |

## Query Criteria Semantics

Both `metric_query`-style tools and `POST /metrics/{name}/query` accept the
same intent fields:

- `filters` accept scalars, lists, or operator objects
  (`eq/ne/gt/gte/lt/lte/in/not_in/contains/starts_with/ends_with/is_null/not_null`)
  and apply to processor dimensions before aggregation.
- `having` applies the same operator objects to metric output columns after
  aggregation.
- `order_by` sorts the result; a `-` prefix means descending.
- `top_n` keeps the largest rows by `top_n_by` (a metric output column).
- `compare: "prior_period"` requires a time axis and adds `*_prev`, `*_delta`,
  and `*_pct_change` columns for each metric output.
- `quantiles: true` adds the Median/p25/p75/p90/p95 suite for digest metrics.
- Grain selection is deterministic inside Value Stream: clients supply query
  criteria (time axis, dimensions, date bounds), never a physical grain.
- Result row counts are capped before rendering or returning.

## Provenance Envelope

Metric-query responses include a provenance object: catalog and computation
hashes, selected physical grain, contributing pipeline run IDs and chunk IDs,
scanned aggregate-row count, and latest creation time. This is the same
envelope `query_metric_result` and the SDK's `to_result()` return.

## Governed SQL Rules

With `--enable-sql`, SQL runs over `meta/aggregate_views.duckdb` (config-hash
and successful-chunk filtered views) plus any `meta/metric_export_*.duckdb`
files:

- Single `SELECT` (or `WITH ... SELECT`) only; comments, DDL/DML, multiple
  statements, and file/catalog functions such as `read_parquet` are rejected.
- Sketch state blob columns are masked from schemas and results.
- Row counts are capped and long queries are interrupted.
- DuckDB external file access, automatic extension loading, and community
  extensions are disabled before user SQL executes.
