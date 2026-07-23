# CLI Reference

The `valuestream` command line, transcribed from `src/valuestream/cli.py`.
Run any command with `-h`/`--help` for the live version; if this page and the
CLI ever disagree, the CLI is right and this page has a doc bug.

Global options:

| Option | Meaning |
|---|---|
| `-V`, `--version` | Print version and exit |
| `--logging-config PATH` | Logging config YAML file; uses the bundled config when omitted |

Conventions: `WORKSPACE` is a workspace directory containing `catalog/`.
Commands exit 0 on success and non-zero on failure, so they are safe to gate
scripts and cron jobs on.

## validate

```sh
uv run valuestream validate WORKSPACE
```

Loads the catalog from `WORKSPACE/catalog/`, validates each YAML against its
JSON Schema, type-checks every expression in transforms/processors/metrics,
and prints a structured pass/fail summary (errors and warnings with
locations). Exit code 0 on success, 1 on failure.

## probe

```sh
uv run valuestream probe WORKSPACE SOURCE_ID [--limit N]
```

Inspects a source after discovery and transforms: chunk and file counts,
calendar columns, the transformed schema, and sample rows.

| Option | Default | Meaning |
|---|---|---|
| `--limit N` | `10` | Sample rows to show |

## run

```sh
uv run valuestream run WORKSPACE [SOURCE_ID] [--force] [--parallel N]
```

Runs ingestion for one source, or all sources when `SOURCE_ID` is omitted.
Prints per-source and per-chunk results (ok/skipped/failed, rows kept, files
written, elapsed time).

| Option | Default | Meaning |
|---|---|---|
| `--force` | off | Process chunks even if the ledger says they are done; prior aggregate files are not deleted |
| `--parallel N` | `1` | Number of chunk worker processes |

For a guarded rebuild that removes old physical aggregates only after all
selected sources succeed, use **Data Load → Rebuild from scratch**. `--force`
alone is intentionally non-destructive.

## query

```sh
uv run valuestream query WORKSPACE METRIC_NAME [options]
```

Queries a metric from aggregate parquet and prints the resulting frame.

| Option | Default | Meaning |
|---|---|---|
| `--by COLUMN` | — | Column to group by; repeatable |
| `--where KEY=VALUE` | — | Filter by column; repeatable, comma-separate values for an in-list |
| `--grain GRAIN` | `daily` | Query grain; coarser buckets are rolled up from stored aggregates |
| `--from YYYY-MM-DD` | — | Inclusive start date |
| `--to YYYY-MM-DD` | — | Inclusive end date |
| `--raw` | off | Include underlying aggregate state columns |

## serve

```sh
uv run valuestream serve WORKSPACE [--port N] [--browser/--headless]
```

Starts the Streamlit dashboard UI for a workspace.

This review branch starts the UI with the dark instrument theme so reports
and configuration workflows can be evaluated against one deterministic palette.

| Option | Default | Meaning |
|---|---|---|
| `--port N` | `8501` | Streamlit server port |
| `--browser/--headless` | `--browser` | Open a browser, or run headless |

## serve-mcp

```sh
uv run valuestream serve-mcp WORKSPACE [--enable-sql]
```

Starts the read-only MCP server over stdio. Requires the `ai` extra
(`uv sync --extra ai`). `--enable-sql` exposes the governed aggregate SQL
tools, which are disabled by default. Tools are listed in the
[API & MCP reference](api-and-mcp.md).

## serve-api

```sh
uv run valuestream serve-api WORKSPACE [--host H] [--port N] [--token T] [--enable-sql]
```

Starts the read-only HTTP API. Requires the `api` extra
(`uv sync --extra api`).

| Option | Default | Meaning |
|---|---|---|
| `--host H` | `127.0.0.1` | Bind address; non-loopback binds require a token |
| `--port N` | `8000` | Port |
| `--token T` | `$VALUESTREAM_API_TOKEN` | Bearer token required on every request (except `/health`) |
| `--enable-sql` | off | Expose governed aggregate SQL endpoints |

## export-duckdb

```sh
uv run valuestream export-duckdb WORKSPACE [--grain G] [--output PATH] [--replace/--no-replace]
```

Exports one materialized DuckDB table per metric at a selected grain, through
the same metric query layer as the UI. Prints exported tables, rows, and any
skipped metrics with reasons.

| Option | Default | Meaning |
|---|---|---|
| `--grain G` | `summary` | Metric grain to export |
| `--output PATH` | `meta/metric_export_<grain>.duckdb` | Target DuckDB file |
| `--replace/--no-replace` | `--replace` | Replace an existing export file |

## generate-pega-dummy

```sh
uv run valuestream generate-pega-dummy --source SAMPLE --output-dir DIR --start-date YYYY-MM-DD (--days N | --end-date YYYY-MM-DD) [options]
```

Generates synthetic Pega-shaped interaction-history Parquet data, deriving the
schema from a real export sample.

| Option | Default | Meaning |
|---|---|---|
| `--source PATH` | required | Pega JSON/NDJSON export, or zip/gzip/tar.gz archive of JSON records |
| `--output-dir DIR` | required | One Parquet file per generated day is written here |
| `--start-date` | required | Inclusive start date |
| `--end-date` / `--days N` | one required | Inclusive end date, or number of days (mutually exclusive) |
| `--rows-per-day N` | `1000000` | Rows per generated day |
| `--batch-size N` | `100000` | Generation batch size |
| `--customer-count N` | `250000` | Distinct synthetic customers |
| `--positive-rate F` | `0.12` | Share of rows with `pyOutcome=Clicked` |
| `--seed N` | `13` | Random seed |
| `--file-prefix S` | `pega_interactions` | Output filename prefix |
| `--compression S` | `zstd` | Parquet compression |
| `--overwrite` | off | Replace existing generated files for the same days |

## migrate

```sh
uv run valuestream migrate --from LEGACY.toml --to WORKSPACE/catalog
```

Translates a legacy TOML config into catalog YAML and writes a
`migration_report.md` into the target directory. Prints generated files,
mapped fields, and gaps; review the report before sign-off
([migration guide](../guides/operations/migration.md)).

## backfill

```sh
uv run valuestream backfill --workspace WORKSPACE --from-legacy-db LEGACY.duckdb
```

Imports legacy DuckDB aggregate tables into the partitioned parquet aggregate
layout where they map to catalog targets. Prints per-table results and skipped
targets.

## vacuum

```sh
uv run valuestream vacuum WORKSPACE [--tmp/--no-tmp] [--dry-run]
```

Prunes superseded aggregate files and orphan reader temp directories, then
refreshes the aggregate DuckDB views.

| Option | Default | Meaning |
|---|---|---|
| `--tmp/--no-tmp` | `--tmp` | Also remove orphan reader temp directories |
| `--dry-run` | off | Report deletions without removing files |
