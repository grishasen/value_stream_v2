# Product Overview

This page describes Value Stream as a product: what it is, who it serves, what
it does today, and where its boundaries are.

## Product Description

Value Stream is an aggregate-first business intelligence platform for marketing,
decisioning, model-performance, and customer-lifecycle analytics. It ingests
batch files from operational systems, transforms them through a declarative
workspace catalog, writes mergeable aggregate statistics, and serves reports
from those persisted aggregates.

The core product choice is that raw event rows do not become the reporting
store. Every business number is derived from compact state such as counts,
sums, sketches, distribution digests, lifecycle summaries, funnel states, or
snapshot counts.

## Problems It Solves

| Problem | Product response |
|---|---|
| Large file exports are slow to query repeatedly | Process each chunk once, then query small aggregate files |
| Business dashboards depend on hand-coded metrics | Define metrics, processors, and dashboards in YAML |
| Analysts need filters and report pages without raw-row exposure | Persist approved group-by columns as aggregate dimensions |
| Operators need to know whether numbers are fresh | Track runs, chunks, config hashes, and freshness metadata |
| Migration from legacy dashboards is risky | Provide TOML migration and legacy DuckDB backfill tooling |
| External BI tools need SQL-readable output | Export metric tables into DuckDB from the same query layer |

## Primary Users

| User | Typical tasks |
|---|---|
| Marketing analyst | Review engagement, conversion, model quality, funnel, and customer segment reports |
| Decision scientist | Add formulas, compare variants, inspect score distributions, and validate experiments |
| Data engineer | Configure sources, transforms, processors, and scheduled ingestion |
| Workspace operator | Load data, monitor runs, inspect chunks, run vacuum, and export DuckDB tables |
| Product owner | Track dashboard coverage, migration readiness, and business value |
| Auditor or reviewer | Trace a reported number back to catalog config, run metadata, and chunks |

## Current Product Surfaces

- CLI command `valuestream` for validation, ingestion, querying, serving the UI,
  export, migration, backfill, dummy data generation, and cleanup.
- Streamlit application with Home, Reports, Chat With Data, a top-level Build
  choice, Configuration Builder, AI Configuration Studio, Catalog, Data Load,
  and Pipelines / Ops pages.
- Python SDK helpers for workspace and query access.
- DuckDB metric export for downstream SQL tools.
- Local read-only stdio MCP tools and a read-only FastAPI HTTP API over the
  governed aggregate query layer.
- MkDocs documentation site.

## Business Value

Value Stream gives teams a governed reporting layer that is reproducible,
configurable, and operationally observable. It is especially suited to
organizations that receive periodic operational exports and need business users
to ask repeatable metric questions without exposing raw event data to every
reporting workflow.

## In Scope

- Batch file ingestion from configured source folders.
- Pega CDH-style interaction history and product holding analytics.
- CSV, Parquet, Excel, and Pega Dataset Export reader patterns.
- Declarative transforms, processors, metrics, and dashboard tiles.
- Aggregate reports at configured time grains.
- Approximate distinct counts, distribution metrics, model-score curves,
  experiments, funnels, lifecycle metrics, set operations, and snapshots.
- Local or workspace-mounted execution on one host.

## Out of Scope or Deferred

| Area | Status |
|---|---|
| Raw event warehouse | Out of scope. Raw rows are not retained as the reporting store. |
| Streaming or CDC ingestion | Out of scope for the current architecture. |
| Distributed execution | Out of scope. The current target is one-node processing. |
| Arbitrary SQL over raw data | Out of scope by design. |
| Read-only HTTP API | In scope. Metric/chart/freshness/chat endpoints are implemented; SQL endpoints require `--enable-sql`. |
| Local stdio MCP | In scope. Aggregate-safe tools are implemented; SQL tools require `--enable-sql`. |
| Remote HTTP MCP and OIDC | Deferred. |

## Product Success Criteria

The product is working when:

- A workspace catalog validates cleanly.
- Source files are discovered as chunks and processed idempotently.
- Reports render from persisted aggregates without raw-row replay.
- Users can trace report values to metric definitions and run metadata.
- Operators can identify freshness, failed chunks, and stale configuration.
- New metrics or tiles can be authored through YAML or the Builder and then
  validated before use.
