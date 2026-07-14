# Business Functionality

This page explains what Value Stream does in business terms. It is intended for
analysts, product owners, data leads, and reviewers who need to understand the
capabilities without reading processor code.

## Business Objects

| Object | Business meaning |
|---|---|
| Workspace | One reporting environment, such as a demo, market, business line, or migration target |
| Source | A family of input files, such as Interaction History or Product Holdings |
| Transform | A repeatable rule that cleans or enriches source fields before aggregation |
| Processor | A business aggregation plan for a source, metric family, and set of dimensions |
| Metric | A user-facing measure calculated from processor state |
| Dashboard | A curated collection of report pages and tiles |
| Tile | One chart, KPI, table, or visualization bound to a metric |
| Run | One ingestion execution recorded for traceability |
| Chunk | The idempotent unit of input processing, commonly derived from file names |

## Capability Areas

| Area | What users can answer |
|---|---|
| Engagement | How many interactions occurred, how many were positive, and what was the engagement rate? |
| Conversion and revenue | Which channels, offers, or groups convert, and how much value do they generate? |
| Model quality | How are propensity, priority, rank, ROC AUC, average precision, and calibration behaving? |
| Response time and numeric distributions | What are medians, percentiles, and distribution shapes for numeric fields? |
| Experiment monitoring | Are test and control groups materially different? |
| Funnels | Where do users or outcomes drop between configured stages? |
| Customer lifecycle and CLV | What are recency, frequency, monetary value, lifetime value, and segment summaries? |
| Cohorts and sets | How do approximate entity sets overlap or change by cohort? |
| Snapshots | What is the aggregate state at periodic points in time? |
| Operations | Are source files loaded, current, complete, and traceable? |

## Core Workflow

1. A user or automated job places source files into the configured workspace.
2. Value Stream discovers files and groups them into chunks.
3. The ingestion engine applies catalog transforms and writes aggregate state.
4. The query layer resolves metrics from aggregate state rather than raw rows.
5. Reports, chat, CLI queries, SDK calls, and DuckDB exports use the same
   aggregate query path.
6. Operators inspect freshness, runs, chunks, config hashes, and validation
   status when results need review.

## Reports Workflow

The Reports page presents configured dashboards from `dashboards.yaml`.

- A dashboard contains pages.
- A page contains tiles.
- Each tile binds to one metric and one chart kind.
- Page filters are authored from persisted group-by columns, with inference for
  older catalogs and validated all-tile/compatible-tile coverage.
- Partial filters identify both their coverage and every unsupported tile.
- KPI-strip cards use explicit scalar, comparison/target, and sparkline settings;
  Reports never guesses a reducer for an ordinary chart.
- Presentation and Inspect modes let users switch between business viewing and
  data/query inspection.

This design keeps the report surface flexible while preserving the aggregate
contract: a page can only filter by dimensions that were persisted during
processing.

## Configuration Workflow

Business users and data engineers can update workspaces in three ways:

| Path | Best for |
|---|---|
| YAML edits | Precise, reviewable changes in source control |
| Configuration Builder | Catalog-first source, processor, metric, tile, chat, and settings authoring inside the UI |
| AI Configuration Studio | Drafting catalog YAML from an uploaded sample and LLM responses, then reviewing and applying it |

Every path should end with catalog validation before data is loaded or reports
are trusted.

## Governance and Traceability

Value Stream records enough context to explain where a number came from:

- The catalog is validated and hashed.
- Processor outputs include provenance columns such as config hash, chunk, run,
  period, and creation time.
- Metadata databases track runs, chunks, config versions, lineage, and aggregate
  views.
- Reports expose freshness, approximation/statistical method help, filter
  coverage, and on-demand query output for inspection.
- Operations pages show run status and chunk-level detail.

## Migration Functionality

The migration tooling helps teams move from legacy dashboard assets:

- `valuestream migrate` translates a legacy TOML configuration into Value
  Stream catalog YAML and writes a migration report.
- `valuestream backfill` imports existing legacy DuckDB aggregate tables into
  the partitioned aggregate layout.
- Side-by-side validation compares old and new metrics before sign-off.

Migration is not a blind import. The generated report should be reviewed for
mapped fields, gaps, and assumptions.

## Decision Rules for Business Changes

| Change | Business impact |
|---|---|
| Add a formula metric from existing state | Usually no raw replay required |
| Add a dashboard tile for an existing metric | Usually no raw replay required |
| Add a new processor | Future runs populate the new aggregate; backfill may be needed |
| Add a new group-by dimension | Raw source replay is usually required |
| Change outcome definitions | Raw source replay is usually required |
| Remove a dimension from reports | Usually safe if no tile depends on it |
| Change chart presentation settings | Affects rendering only, not stored aggregates |
