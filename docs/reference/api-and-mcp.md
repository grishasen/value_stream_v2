# API and MCP Reference

The read-only HTTP API and the local stdio MCP server expose the same
governed tool layer over the aggregate query path. Neither surface mutates the
catalog or aggregate store, and neither exposes raw source rows. LLM provider
setup for the in-app planner is in
[Chat with data](../guides/users/chat-with-data.md); the security posture
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

For desktop client setup and connection checks, see
[Connect Codex and Claude Desktop (Russian)](../guides/users/mcp-clients.md).

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

`POST /metrics/{name}/chart` also returns `warnings` describing any chart
parameter the planner substituted.

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
| `metric_list` | List metrics, dimensions, query time axes, and supported charts. Narrow with `search`, `processor`, or `dashboard`; `detail="full"` adds prose and configuration |
| `metric_query` | Query metric rows through `query_metric` (operator filters, having, order_by, top_n, compare, quantile suite) |
| `metric_chart_query` | Query metric rows and return an explicit validated chart spec, plus `requested` and `warnings` |
| `dashboard_list` | List authored dashboards, pages, page filters, and tiles with their chart specs |
| `tile_query` | Run one authored tile exactly as the report page runs it |
| `kpi_query` | Return a KPI card's value, comparison delta, period, and sparkline |
| `chat` | Plan and answer a natural-language question (requires a configured model) |
| `dimension_values_tool` | Return aggregate-backed dimension values |
| `workspace_status_tool` | Per-processor aggregate readiness, recent runs, and unfinished runs |
| `freshness_get` | Return metric freshness metadata |
| `sql_schema` | List governed DuckDB tables/views and their non-masked columns (only with `--enable-sql`) |
| `sql_query` | Run one governed read-only SELECT over aggregate views and metric exports (only with `--enable-sql`) |

The server also publishes two resources — `valuestream://catalog` (the
catalog YAML) and `valuestream://dashboards` (the dashboard manifest as
JSON) — and two prompts, `explore_workspace` and `starter_questions`.

## Reading a Dashboard Through the Tools

`dashboard_list` returns the ids that `tile_query` and `kpi_query` take.
A tile is not just a metric query: the tile tools apply its authored filters,
infer its grain and group-by from the chart, join a combo's secondary metric,
and remap a histogram's property to its distribution metric, so the numbers
match the report. The ad-hoc chart tool also supports extended kinds through
`chart_fields`, described below.

Authored tile filters define the metric's valid population, so they take
precedence over filters passed to the tool. Filters a tile cannot apply come
back in `ignored_filters` rather than being silently dropped, and sketch/state
blob columns are removed from tile rows and listed in `masked_columns`.

`applied_filters` contains the effective filters after authored overrides,
including authored filters the caller did not supply. `overridden_filters`
contains the caller's original values where they conflict with authored scope.
For example, authored `Channel: Web` overrides requested `Channel: Email`:
the applied value is `Web`, and `overridden_filters` reports `Email`.
Both tile and KPI tools use this contract. The report page and MCP use the
same KPI service for values, comparison windows, deltas, and sparklines.

## Errors and Substitutions

Tool execution failures return an MCP result with `isError: true`. The
diagnosis is included in both text content and `structuredContent`:

```json
{"error": {"kind": "aggregate_not_ready", "type": "AggregateNotReadyError",
           "message": "...", "remediation": "Run ingestion for this workspace..."}}
```

`kind` is one of `invalid_request`, `aggregate_not_ready`, `aggregate_missing`,
`sql_rejected`, `timeout`, or `dependency_missing`. When a metric or tile fails with an aggregate
error, `workspace_status_tool` reports which processors are `ready`, `stale`,
`unpublished`, or `missing`, using the same load a query performs.

`metric_chart_query` rejects a chart kind a metric cannot render, naming the
kinds that are available, rather than substituting one silently. When a valid
kind needs an axis or colour adjusted, it echoes what was asked in `requested`
and explains the change in `warnings`; compare `chart` with `requested` before
describing the result to a user.

## Chart Kinds

Each metric carries two lists. `chart_kinds` is what the LLM planner may
choose: kinds that render from a plain x/y/colour/facet spec.
`tool_chart_kinds` is the wider set an explicit caller may request, where the
kind's own inputs go in `chart_fields`:

| Kind | Required `chart_fields` |
|---|---|
| `funnel` | `stages` |
| `combo` | `secondary_metric` (optional `primary_mark`, `shared_y_axis`) |
| `interval` | `lower_output`, `upper_output` |
| `treemap`, `sankey` | `path` (two or more dimensions) |
| `boxplot`, `histogram` | `property` |
| `bar_polar` | `theta` |
| `pareto`, `waterfall`, `experiment_z_score`, `experiment_odds_ratio` | none beyond `x` and `y` |
| `gauge`, `gain_curve`, `lift_curve` | none |

`facet_row`, `line_dash`, `goal_line`, `x_axis_title`, and `y_axis_title` work
with any kind. A missing required field is named in the error. Fields a kind
does not use are dropped rather than echoed back, so the returned spec is
exactly what the chart factory will draw. A combo's secondary metric is
queried over the same dimensions and joined onto the rows, the same join the
dashboard tile path performs.

Supported row and column facets are preserved in the chart-factory tile;
facet dimensions and hierarchy paths are included in the aggregate grouping.
Rendering uses the original aggregate frame, including sketches where needed.
Only the response rows have binary sketch columns removed.

## Rendered Charts

`tile_query` and `metric_chart_query` take a `render` option, so a client can
see the chart rather than only its rows. The figure comes from the same chart
factory the report page uses.

| `render` | Result |
|---|---|
| `none` (default) | Rows only |
| `png` | The rendered image, attached as an image content block |
| `html` | A self-contained interactive page written to disk; the payload carries its `path` |
| `html_cdn` | The same page with Plotly loaded from its CDN — about 13 KB instead of 4 MB, but it needs a network |
| `spec` | The Plotly figure spec, for clients that draw Plotly themselves |

HTML is written rather than inlined: it is only useful in a browser, and the
markup would otherwise spend the response on something the caller cannot
render. Files go to `--render-dir`, defaulting to `valuestream-renders` under
the system temp directory. An explicitly supplied `--render-dir` is used as given.

Each export has a unique, exclusively created filename, so repeated calls and
different workspaces cannot overwrite previous charts. The returned path is
absolute. Files have a seven-day retention period, reported as `expires_at`;
subsequent exports remove expired files created by this renderer. Unrelated
files and symlinks are left untouched. Copy exports elsewhere to keep them.
PNG responses include an image block and structured query metadata.

Because the page is a file and only its path is returned, embedding Plotly's
JavaScript costs file size rather than response size, so `html` embeds it by
default: the page opens offline and keeps working after the CDN's current
version moves on. Use `html_cdn` when a small file matters more.

PNG needs the optional `viz` extra:

```sh
uv sync --extra viz    # installs kaleido
```

Without it, `render="png"` returns a `dependency_missing` error naming the
install command and pointing at `render="html"`. Chrome/Chromium must also
be installed; `BROWSER_PATH` can select its executable. Availability checks
verify the dependency and executable without launching a browser. Browser
startup failures return an actionable error with the HTML alternative.
Kaleido drives a headless Chrome, so the first render in a process pays a
browser start-up cost.

## Response Size

Two options keep responses small on a large catalog:

- `metric_list` defaults to `detail="compact"` and reports `matched_metrics`,
  `returned_metrics`, and `truncated`.
- `metric_query` defaults to `provenance="summary"`, which replaces the
  contributing chunk and run id lists with counts, and to
  `include_curves=false`, which omits the roc/pr point arrays. Set
  `provenance="full"` or `include_curves=true` when you need them. The HTTP
  API takes the same `include_curves` flag on `POST /metrics/{name}/query`.

`metric_list`, `metric_query`, `metric_chart_query`, and `tile_query` accept
`offset` (default 0) and `limit` (1–500). Follow `next_offset` until it is null.
Catalog pages are ordered by metric name. Query pages follow query ordering;
keep filters, ordering, and the underlying workspace unchanged between calls.
Pagination does not pin a snapshot across ingestion or catalog edits.

Row responses expose the total `row_count` before pagination, `returned_rows`,
`offset`, `next_offset`, and `truncated` (whether more rows remain after this
page). The chart tool renders its returned page; the authored tile tool renders
the full tile while limiting response rows. These query tools publish typed
output schemas, and enum/limit constraints appear in their input schemas.
Tool annotations identify read-only queries, additive rendering operations,
and the external LLM call made by `chat`.

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
- Returned rows are capped; authored tile rendering retains the full tile.

## Provenance Envelope

Metric-query responses include a provenance object: catalog and computation
hashes, selected physical grain, contributing pipeline run IDs and chunk IDs,
scanned aggregate-row count, and latest creation time. This is the same
envelope `query_metric_result` and the SDK's `to_result()` return.

Tile and KPI responses also include freshness and a `provenance` list, captured
from the actual queries rather than a second scan. Each entry includes its
metric, query options (including date bounds), catalog/computation hashes,
selected grain, and contributing chunk/run counts. A combo includes both
metrics; a KPI includes the current value, comparison, and sparkline queries
when applicable.

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
