# Architecture Overview

This page summarizes how Value Stream is built. It is a bridge between the
product-facing concepts pages and the detailed architecture, domain,
algorithm, and processor references.

## System Shape

Value Stream is a Python application built around:

- Polars for file processing and dataframe transformations.
- PyArrow Parquet for aggregate storage.
- DuckDB for metadata and SQL-readable exports.
- Pydantic and JSON Schema for catalog validation.
- Streamlit and Plotly for the UI.
- Apache DataSketches-style sketches and digests for mergeable approximations.

The CLI entry point is `valuestream`, implemented in `src/valuestream/cli.py`.
The Streamlit app entry point is `src/valuestream/ui/app.py`.

## Workspace Layout

A workspace is the runtime boundary. A typical workspace contains:

```text
workspace/
  catalog/
    pipelines.yaml
    processors.yaml
    metrics.yaml
    dashboards.yaml
  data/
    ...
  aggregates/
    ...
  meta/
    chunks.duckdb
    pipeline_runs.duckdb
    config_versions.duckdb
    lineage.duckdb
    aggregate_views.duckdb
```

Only `catalog/` and source files are required up front. Aggregate and metadata
paths are created as the workspace runs.

## Catalog Files

| File | Defines |
|---|---|
| `pipelines.yaml` | Workspace name, sources, readers, schemas, transforms, defaults |
| `processors.yaml` | Processor definitions, source bindings, dimensions, time grains, state columns |
| `metrics.yaml` | User-facing metrics derived from processor state |
| `dashboards.yaml` | Dashboards, pages, tiles, chart kinds, and presentation settings |

The loader combines these files into a catalog, rejects duplicate YAML keys and
duplicate catalog IDs, validates references/dependency cycles and required
chart-role fields, and computes
canonical hashes. A full catalog hash supports audit; source and processor
computation hashes include upstream behavior and control reprocessing/stale
detection. Canonical payloads and file lineage are persisted in metadata.

## Ingestion Path

1. Discovery finds source files and groups them into chunks.
2. A durable pipeline-run row is inserted with `status=running`.
3. Readers load chunk files as Polars frames.
4. Transforms clean, normalize, enrich, filter, and deduplicate data.
5. Processors fan out over the transformed frame.
6. Each processor writes mergeable aggregate state and configured grains.
7. Complete lineage commits before the chunk's final `status=ok` marker.
8. The same run row becomes `ok`, `partial`, or `failed`, publishing only its
   committed chunks.

The ingestion engine is designed to be idempotent. A completed chunk is skipped
only when its source computation hash and current input-file fingerprint match.
Writes use immutable run-specific files plus atomic rename. Query visibility is
gated by successful chunk and run ledger rows, so a failed replacement leaves
the previous successful version visible.
If a process dies while the run is still `running`, the next source invocation
holds the same source lock, verifies committed chunk fingerprints, lineage,
files, and computation hashes, finalizes the interrupted run, and reuses only
the verified chunks.

## Processor and State Model

Processors define the durable aggregate state that metrics can query. Current
processor families include:

| Processor kind | Typical use |
|---|---|
| `binary_outcome` | Counts, positives, negatives, rates, experiments |
| `numeric_distribution` | Numeric summaries, quantiles, histograms |
| `score_distribution` | Model score curves, AUC, average precision, calibration |
| `entity_lifecycle` | CLV, RFM, recency, frequency, monetary value |
| `entity_set` | Approximate set operations and cohort comparisons |
| `funnel` | Stage counts and dropoff calculations |
| `snapshot` | Periodic aggregate state |

See [Processor Specs](../reference/processors.md) and [Algorithms](../reference/algorithms.md)
for implementation details.

## Metric Query Path

The query layer resolves a metric to its backing processor, chooses an aggregate
grain, reads persisted aggregate files, applies filters that match stored
dimensions, computes formula or sketch-derived outputs, and returns a Polars
frame.

`query_metric_result` and the SDK's `to_result()` add a provenance envelope:
selected physical grain, catalog and computation hashes, contributing run/chunk
IDs, aggregate scan count, and latest aggregate timestamp. API and MCP metric
queries return the same envelope.

All current read surfaces use this path:

- CLI `valuestream query`
- Streamlit Reports
- Chat With Data
- Python SDK helpers
- DuckDB metric export
- Local stdio MCP
- Read-only FastAPI HTTP API

## UI Architecture

The Streamlit app loads a `ValueStreamContext` for the active workspace. The
context contains the resolved workspace path, loaded catalog, validation result,
and shortened catalog hash. Page modules consume that context directly.

| Module | Responsibility |
|---|---|
| `ui/shell.py` | Page setup and navigation |
| `ui/pages/data_load.py` | Upload, discovery, and source runs |
| `ui/pages/reports.py` | Dashboard rendering, filters, presentation, inspect mode |
| `ui/pages/catalog.py` | Catalog inventory and validation visibility |
| `ui/pages/build.py` | Choice between sample-first and catalog-first authoring |
| `ui/pages/config_builder.py` | Catalog-first builder, chat review, settings, and YAML export |
| `ui/pages/ai_config_studio.py` | Sample-driven catalog drafting with optional model assistance |
| `ui/pages/chat.py` | Aggregate-aware chat over selected metrics |
| `ui/pages/ops.py` | Runs, source status, chunks, and operational controls |

Shared UI primitives in `ui/components.py` provide responsive metric grids,
compact summary-number formatting, validation cards, and searchable tables.
`ui/theme.py` supplies the light/dark surface tokens, accessible secondary-text
colors, responsive grid breakpoints, and the Plotly colorways used by reports.
The sidebar exposes workspace path and catalog revision through a details
popover so operational pages can prioritize business and run status.

## Extension Points

| Extension | Where to start |
|---|---|
| Add a reader | `src/valuestream/readers/` and `docs/reference/readers-and-formats.md` |
| Add a transform | `src/valuestream/transforms/` and expression validation if needed |
| Add a processor | `src/valuestream/processors/registry.py`, config model/schema, processor specs, tests |
| Add a metric kind | `src/valuestream/query/`, config model, tests, docs |
| Add a chart kind | `src/valuestream/charts/`, `docs/reference/chart-catalog.md`, Builder recipes |
| Add a UI workflow | `src/valuestream/ui/pages/`, shell navigation, user docs |
| Add a CLI command | `src/valuestream/cli.py`, README, user guide, runbook |

## Headless and Deferred Surfaces

`valuestream serve-api` exposes the read-only FastAPI surface and
`valuestream serve-mcp` exposes local stdio MCP. Both preserve the aggregate-only
contract and reuse the same query layer; SQL is absent unless `--enable-sql` is
passed. The API CLI requires a bearer token for non-loopback binds. Remote HTTP
MCP, OIDC, and multi-user service deployment remain deferred.

## Verification

Use these checks before merging behavior or documentation changes:

```sh
uv run valuestream validate examples/demo
uv run pytest -m "not bench and not slow"
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run mkdocs build --strict
```
