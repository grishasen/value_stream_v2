# Querying and Export

This guide covers metric access outside the UI: CLI queries, DuckDB export for
SQL BI tools, and the read-only API/MCP surfaces. All of these read through the
same governed aggregate query layer as the dashboards.

## Query From the CLI

Query one metric:

```sh
uv run valuestream query examples/demo VS_Engagement_Rate --by Channel --grain Day
```

Add filters:

```sh
uv run valuestream query examples/demo VS_Engagement_Rate --by Channel --grain Day --where Channel=Web
```

Include state columns for debugging:

```sh
uv run valuestream query examples/demo VS_Engagement_Rate --by Channel --grain Day --raw
```

Bound the date range with `--from` and `--to` (inclusive, `YYYY-MM-DD`). See
the [CLI reference](../../reference/cli.md#query) for all options.

## Export to DuckDB

Superset and other SQL BI tools can consume a materialized DuckDB export with
one table per metric at a selected grain:

```sh
uv run valuestream export-duckdb examples/demo --grain Summary
```

By default this writes `meta/metric_export_<grain>.duckdb` under the workspace.
Pass `--output path/to/file.duckdb` to choose the target file and
`--no-replace` to keep an existing file. Tables are named
`metric_<metric_id>_<grain>` and are produced through the same metric query
layer used by the UI, so formula and sketch-derived metrics export as ordinary
SQL-readable columns.

## Use the Read-Only API or MCP Server

Start local aggregate-safe tools:

```sh
uv run valuestream serve-mcp examples/demo
uv run valuestream serve-api examples/demo --host 127.0.0.1 --port 8000
```

Add `--enable-sql` only when governed SQL tools/endpoints are required. API
metric-query responses include catalog/computation hashes and contributing
run/chunk provenance. Set `--token` or `VALUESTREAM_API_TOKEN` for bearer
auth; the CLI refuses a non-loopback API bind without one.

Endpoints, MCP tools, and the governed SQL rules are documented in the
[API & MCP reference](../../reference/api-and-mcp.md); the security posture is
in [Security](../operations/security.md).
