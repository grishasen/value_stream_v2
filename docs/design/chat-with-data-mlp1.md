# Chat With Data MLP1

This document defines the first releasable LLM integration for Value Stream.
It intentionally keeps the LLM boundary narrow: the model plans a structured
aggregate query, Value Stream validates that plan against the catalog, and the
existing query layer executes it.

## Scope

MLP1 supports:

- Natural-language questions over configured metrics.
- Text answers, KPI cards, tables (with CSV download), and themed Plotly charts
  rendered through the same chart factory as dashboards.
- A metric-aware chart allowlist: line, bar, stacked_area, scatter, heatmap,
  donut, table, kpi_card, and (for model-quality metrics) roc_curve,
  precision_recall_curve, and calibration_curve. Each metric advertises the
  subset it supports.
- Operator filters, having thresholds on metric outputs, sort orders, top-N,
  period-over-period comparison columns, and quantile suites in the query
  intent.
- Clarifying questions when a request is ambiguous, and optional LLM
  narratives grounded in the returned aggregate rows.
- Governed read-only SQL over the aggregate DuckDB views and metric export
  tables. It is opt-in in Chat and must be enabled explicitly with
  `--enable-sql` for MCP or the HTTP API.
- LLM-backed planning through LiteLLM from the Streamlit Chat page.
- Starter questions, pinning an answer to a "Chat Pins" dashboard, and saving
  the session LLM settings to the workspace `ai.yaml`.
- Local Ollama, OpenAI API, and Anthropic API configuration through
  workspace-local `ai.yaml`.
- A small read-only stdio MCP server for Claude Code and other MCP clients, and
  a read-only HTTP API (`valuestream serve-api`) over the same tool layer.

MLP1 does not support:

- Generated Python execution.
- SQL beyond a single governed SELECT: DDL/DML, multiple statements, comments,
  and file/catalog functions are rejected; sketch state blobs are masked and
  row counts are capped.
- Raw source rows, raw aggregate parquet paths, or file-system access.
- Catalog writes from Chat With Data (pinning writes a dashboard tile only).
- Remote HTTP MCP, OAuth, or hosted multi-user auth. The HTTP API is read-only
  with optional single-token bearer auth intended for trusted local use.
- Arbitrary chart grammar. Charts are selected from a metric-aware allowlist and
  rendered by Value Stream.
- Deterministic natural-language fallback in the Streamlit Chat page. If the
  LLM planner is disabled or no model is configured, the chat input is disabled.

## Architecture

```text
Streamlit Chat page      Claude Code / MCP client      HTTP API client
        |                          |                          |
        v                          v                          v
LLM intent planner          stdio MCP tools           FastAPI endpoints
        |                          |                          |
        +----------- governed tool layer (one implementation) +
                         |
                         v
              query_metric / sql / freshness / manifest
                         |
                         v
               persisted aggregate store
```

Long-lived servers (MCP and the HTTP API) reload the catalog automatically when
its YAML files change on disk, so manifest and chart validation stay in sync
with edits made in the Config Builder without a restart.

The in-app planner produces JSON shaped like:

```json
{
  "metric": "CTR",
  "response": "chart",
  "group_by": ["CustomerType", "Channel"],
  "filters": {"Channel": {"op": "not_in", "values": ["Unknown"]}},
  "having": {"CTR": {"op": ">", "value": 0.05}},
  "order_by": ["-CTR"],
  "top_n": null,
  "top_n_by": null,
  "compare": null,
  "quantiles": false,
  "time_axis": "Day",
  "start": null,
  "end": null,
  "chart": {
    "kind": "line",
    "x": "Day",
    "y": "CTR",
    "color": "Channel",
    "facet_col": "CustomerType",
    "value_format": "percent"
  },
  "clarify": null,
  "limit": 100
}
```

Query criteria semantics:

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
- `response: "clarify"` plus a `clarify` question asks the user instead of
  guessing when the request is ambiguous.
- `response: "sql"` plus a `sql` SELECT is accepted only when governed SQL is
  enabled; the statement runs through the governed SQL tool described below.

The Chat page can also run a second grounded LLM pass ("Narrative answers")
that summarizes the returned rows in plain language. Only the governed
aggregate rows already shown to the user are sent to the model. Verbal
answers report the governed overall (summary-grain) value rather than an
unweighted average of grouped rows.

The Streamlit Chat page always calls the configured LiteLLM planner before it
executes `query_metric`. The sidebar no longer exposes manual metric, group-by,
or grain controls as a fallback path for natural-language prompts.

The planner prompt includes a compact catalog manifest:

- **Datasets** from `pipelines.yaml`: source id, description, reader kind,
  timestamp column, natural key, and transform-derived fields.
- **Processors** from `processors.yaml`: source dataset, processor kind,
  business dimensions, query time axes, state columns, outcome/stage/score config,
  and a short kind explanation.
- **Metrics** from `metrics.yaml`: metric id, description, kind explanation,
  source processor, source dataset, allowed dimensions/time axes, output columns,
  and kind-specific configuration such as formula expressions, sketch states,
  quantiles, variant roles, or funnel stages.

Datasets and processors are explanatory context only. The model must still
return a catalog metric id, not a dataset or processor id.

Optional chat-only guidance from `<workspace>/ai.yaml` is included in the same
planner prompt. The `chat_with_data.agent_prompt` text gives workspace-specific
business context, and `chat_with_data.dataset_descriptions` /
`chat_with_data.metric_descriptions` can override or clarify descriptions for
LLM planning without changing catalog semantics, aggregate storage, or metric
calculation. Governance rules still take precedence: the model cannot request
raw rows, SQL, Python, filesystem access, or non-allowlisted chart behavior.

Value Stream validates the intent before querying:

- `metric` must exist in `metrics.yaml`.
- `group_by`, `filters`, `chart.color`, and `chart.facet_col` must be
  processor group-by columns.
- The LLM does not choose aggregate grains. It may return query criteria such
  as `time_axis`, `chart.x`, dimensions, filters, and date bounds.
- Value Stream derives the logical query grain from the criteria and the query
  layer chooses the physical aggregate, falling back to a finer stored aggregate
  such as daily when a coarser requested bucket can be rolled up safely.
- Time columns are controlled by the derived query grain; the LLM cannot invent
  them.
- Result row count is capped before rendering or returning through MCP.

## Chart Behavior

Chat charts are rendered through the same `valuestream.charts.render_chart`
factory and dashboard theme the report builder uses, so a chat chart and a
saved dashboard tile of the same shape look identical. The supported chat
chart kinds are:

| Kind | Use |
|---|---|
| `line` | Daily/monthly trends |
| `bar` | Category comparisons |
| `stacked_area` | Composition over time |
| `scatter` | Two-measure relationships |
| `heatmap` | Two-dimension breakdowns (metric value as the color scale) |
| `donut` | Share-of-total across one dimension |
| `table` | Row display with CSV download |
| `kpi_card` | Single governed summary value rendered as a metric tile |
| `roc_curve` / `precision_recall_curve` / `calibration_curve` | Model-quality metrics only |

Each metric advertises its own `chart_kinds` in the catalog manifest, derived
from the processor's chart recipes intersected with the chat-renderable set
(the base line/bar/table/kpi_card kinds are always available). The planner must
pick `chart.kind` from that per-metric list; if it proposes a kind the metric
does not support, Value Stream logs it and falls back to a governed default
(line for time trends, otherwise bar).

`chart.value_format` (`percent`, `integer`, `number`, or `currency`) controls
axis, colorbar, and KPI number formatting. Every chart and table answer offers
a CSV download of the underlying governed aggregate rows.

Example request:

```text
Plot daily CTR by customer type and channel
```

Expected plan:

```text
criteria(metric="CTR", time_axis="Day", group_by=["CustomerType", "Channel"],
         chart={kind:"line", x:"Day", y:"CTR", color:"Channel",
                facet_col:"CustomerType", value_format:"percent"})
```

The UI derives the daily query grain, executes `query_metric`, maps the chart
intent to a tile spec, and renders it with `Day` on the x-axis, `CTR` on the
y-axis, `Channel` as color, and `CustomerType` as the facet.

Chart validation is role-aware:

- `chart.kind` must be one of the metric's advertised `chart_kinds`.
- Metric output columns are valid for `chart.y`, not `chart.x`.
- `chart.x` can be any query-result grouping column: a time column such as
  `Day`, or a business dimension such as `Issue`, `Channel`, or
  `CustomerType`.
- For `heatmap`, `chart.x` and `chart.color` are the two dimensions and
  `chart.y` is the metric value; for `donut`, `chart.x` is the category and
  `chart.y` the metric value.
- `roc_curve`, `precision_recall_curve`, and `calibration_curve` read fixed
  curve columns and take only an optional `chart.color` dimension.
- Time-trend charts should use `Day`/`Month`/`Quarter`/`Year` as `chart.x`.
- Time-trend charts grouped by a business dimension should use that dimension
  as `chart.color` so each group renders as a separate series.
- Business dimensions may be used for `group_by`, `chart.color`, and
  `chart.facet_col`.
- If a model proposes an invalid x-axis or a tile field that is not in the
  query result, Value Stream logs it and normalizes to a governed fallback (or,
  if the factory still cannot render, a plain Plotly line/bar) before rendering.

## Workspace AI Config

The Chat page and AI Configuration Studio both read `<workspace>/ai.yaml`.
Keep secrets in environment variables, not in the file.

Common shape:

```yaml
ai:
  llm:
    model: gpt-5.5
    api_key_env: OPENAI_API_KEY
    api_base: ""
    custom_provider: ""
    temperature: 0.1
    timeout_seconds: 120
chat_with_data:
  agent_prompt: |
    You are a data analysis agent helping business users analyze Pega CDH
    interaction history and product holdings aggregates.
  dataset_descriptions:
    ih: Interaction history aggregate dataset.
    holdings: Product holdings aggregate dataset.
  metric_descriptions:
    engagement: Use engagement metrics for CTR, response rate, and lift.
    conversion: Use conversion metrics for conversion rate and revenue.
```

`model` is a LiteLLM model string. `api_base` and `custom_provider` are useful
for local or proxy-backed providers. The UI also lets users override these
values for the current Streamlit session. Configuration Builder's Chat Review
step edits the chat-only prompt and description blocks.

## Run With Ollama

Install and start Ollama, then pull a model:

```sh
ollama pull llama3.1
ollama serve
```

Create `<workspace>/ai.yaml`:

```yaml
ai:
  llm:
    model: ollama/llama3.1
    api_base: http://localhost:11434
    custom_provider: ollama
    api_key_env: ""
    temperature: 0.1
    timeout_seconds: 120
```

Run Value Stream:

```sh
uv run valuestream validate examples/demo
uv run valuestream run examples/demo
uv run valuestream serve examples/demo --port 8501 --headless
```

Open Chat With Data, enable `LLM intent planner`, and ask:

```text
Plot daily CTR by customer type and channel
```

Ollama's chat API supports function tools and JSON/schema response formatting,
but MLP1 uses a provider-neutral JSON-planning prompt through LiteLLM.

## Run With OpenAI

Set an API key:

```sh
export OPENAI_API_KEY=...
```

Create `<workspace>/ai.yaml`:

```yaml
ai:
  llm:
    model: gpt-5.5
    api_key_env: OPENAI_API_KEY
    temperature: 0.1
    timeout_seconds: 120
```

Run the Streamlit app:

```sh
uv run valuestream serve examples/demo --port 8501 --headless
```

Open Chat With Data and enable `LLM intent planner`.

OpenAI supports strict function calling and structured outputs. MLP1 keeps the
implementation provider-neutral for now; a future version can switch the
planner from JSON prompting to provider-native strict tools without changing
the governed query layer.

## Run With Anthropic API

Set an API key:

```sh
export ANTHROPIC_API_KEY=...
```

Create `<workspace>/ai.yaml`:

```yaml
ai:
  llm:
    model: anthropic/claude-sonnet-4-6
    api_key_env: ANTHROPIC_API_KEY
    temperature: 0.1
    timeout_seconds: 120
```

Run the Streamlit app:

```sh
uv run valuestream serve examples/demo --port 8501 --headless
```

Anthropic's Messages API supports client tools, where Claude returns a tool
request and the application executes it locally. MLP1 uses LiteLLM JSON intent
planning in the Streamlit app and exposes local tools through MCP for Claude
Code.

## Run With Claude Code

Claude Code should use the MCP server, not the Streamlit Chat page.

Install the optional MCP dependency:

```sh
uv sync --extra ai
```

Register the local stdio server from a terminal where `uv` can resolve this
project:

```sh
claude mcp add valuestream -- uv run valuestream serve-mcp /absolute/path/to/workspace
```

For this repository's demo workspace:

```sh
claude mcp add valuestream -- uv run valuestream serve-mcp /Users/gregory/PycharmProjects/value_stream/examples/demo
```

Available MCP tools:

| Tool | Purpose |
|---|---|
| `metric_list` | List metrics, dimensions, query time axes, and supported charts |
| `metric_query` | Query metric rows through `query_metric` (operator filters, having, order_by, top_n, compare, quantile suite) |
| `metric_chart_query` | Query metric rows and return an explicit validated chart spec |
| `dimension_values_tool` | Return aggregate-backed dimension values |
| `sql_schema` | List governed DuckDB tables/views and their non-masked columns (only with `--enable-sql`) |
| `sql_query` | Run one governed read-only SELECT over aggregate views and metric exports (only with `--enable-sql`) |
| `freshness_get` | Return metric freshness metadata |

The governed SQL tools query `meta/aggregate_views.duckdb` (config-hash and
successful-chunk filtered views over aggregate parquet) plus any
`meta/metric_export_*.duckdb` files produced by `valuestream export-duckdb`.
Statements must be a single SELECT (or WITH ... SELECT); comments, DDL/DML,
multiple statements, and file/catalog functions such as `read_parquet` are
rejected. Sketch state blob columns are masked from schemas and results, row
counts are capped, and long queries are interrupted. Each connection is
allowlisted to the aggregate Parquet directories and metric-export databases;
DuckDB external file access, automatic extension loading, and community
extensions are disabled before user SQL executes.

Example Claude Code prompt:

```text
Using the valuestream MCP server, call metric_chart_query with chart_kind=line,
x=Day, y=CTR, color=Channel, facet_col=CustomerType, grain=daily, and
group_by=[CustomerType, Channel]. Summarize the top patterns and include the
freshness status.
```

For a dimension comparison, the model should make the dimension explicit:

```text
Using metric_chart_query, call chart_kind=bar, x=Issue, y=CTR, color=Channel,
grain=daily, and group_by=[Issue, Channel].
```

## Run The HTTP API

The read-only HTTP API exposes the same governed tool layer as MCP for
programmatic clients. Install the optional dependency and start it:

```sh
uv sync --extra api
uv run valuestream serve-api /absolute/path/to/workspace --host 127.0.0.1 --port 8000
# Add --enable-sql only when governed SQL endpoints are required.
```

Endpoints (interactive OpenAPI docs at `/docs`):

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

Set a bearer token with `--token` or the `VALUESTREAM_API_TOKEN` environment
variable; every endpoint except `/health` then requires
`Authorization: Bearer <token>`. When no token is set the API is open, which is
only appropriate for a trusted localhost deployment. The API never mutates the
catalog or aggregate store and never exposes raw source rows. Governed-layer
errors map to HTTP status codes: invalid requests return 400, missing
aggregates 409, and SQL timeouts 504.

`valuestream serve-api` refuses a non-loopback bind unless `--token` or
`VALUESTREAM_API_TOKEN` is set. Metric query responses include a provenance
object with the catalog/computation hashes, selected physical grain, pipeline
run IDs, chunk IDs, scanned aggregate-row count, and latest creation time.

## Operational Notes

- Run ingestion before asking questions; Chat queries persisted aggregates.
- Configure and enable the LLM intent planner before asking questions. A
  successful Streamlit Chat answer implies a LiteLLM call produced a validated
  intent first.
- If the model chooses an invalid dimension or time axis, the validator rejects
  the plan and shows the error. Grain selection remains deterministic inside
  Value Stream.
- For demos, use low-cardinality dimensions such as `Channel`, `CustomerType`,
  `PlacementType`, `Issue`, and `Group`.
- Keep `ai.yaml` in each workspace because metric names and provider defaults
  may differ by environment.
- Every LiteLLM call logs provider settings, the system prompt, the user prompt,
  the model response, failure details, and timing. API key values are not logged,
  but prompts may contain catalog metadata and approved sample values from AI
  Configuration Studio.
- Do not use Chat With Data for sensitive raw samples. It only exposes catalog
  metadata and aggregate query results, but prompts still leave the local app
  when using hosted model APIs.

## References

- OpenAI function calling and strict mode:
  <https://developers.openai.com/api/docs/guides/function-calling>
- Anthropic tool use overview:
  <https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview>
- Claude Code MCP:
  <https://code.claude.com/docs/en/mcp>
- Ollama chat API:
  <https://docs.ollama.com/api/chat>
