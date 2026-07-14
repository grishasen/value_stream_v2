# Aggregate-First BI Platform

**Replacement Design for the CDH Value Dashboard**

| | |
|---|---|
| Author | Generated for grigoriy.sen@pega.com |
| Date | 2026-05-08 |
| Current stack | Polars · DuckDB · Streamlit · Plotly |
| Current headless surfaces | Read-only FastAPI HTTP API · local stdio MCP |
| Deferred surfaces | Remote HTTP MCP · OIDC/multi-user deployment |
| Status | Target design with implemented Chat MLP1, local stdio MCP, and read-only HTTP API |
| Source review | Local repo, GitHub repo & wiki, local `wiki/` notes |

---

## 1. Executive Summary

The existing **CDH Value Dashboard** is a Streamlit application that ingests Pega Customer Decision Hub Interaction History (IH) and Product Holdings exports, computes a fixed catalogue of marketing/ML/CLV metrics, and renders Plotly dashboards. It already implements *part* of the desired architecture: a chunked, file-group-based pipeline that emits **mergeable sufficient statistics** (counts, sums, t-digests, HLL sketches) into per-metric DuckDB tables.

The replacement, codenamed **Value Stream**, takes the chunked-aggregate idea and elevates it to the centerpiece of the architecture, while keeping the same core engineering stack (**Polars + DuckDB + Streamlit + Plotly**). Streamlit Chat With Data, local stdio MCP, and a read-only FastAPI HTTP API share one governed aggregate tool layer; remote HTTP MCP and multi-user/OIDC deployment remain deferred. Value Stream is rethought from first principles around three invariants:

1. **No raw event row is ever persisted past chunk processing.** Only mergeable sufficient statistics are stored.
2. **Pipelines, processors, metrics, and dashboards are first-class declarative artifacts** authored in YAML and validated against a published schema — not Python `eval` strings.
3. **The same metric can be served from multiple physical aggregate tables at different grains**, and the query layer routes a request to the smallest aggregate that still answers it correctly.

This document specifies Value Stream in enough depth to start implementation: component layout, storage schemas, the YAML configuration DSL, the processor catalog, the query surface, the Chat/MCP/HTTP surfaces, the dashboard model, a migration strategy from the current app, and a phased delivery plan.

---

## 2. Background — What the current application does well, and where it strains

### 2.1 What it does well

The current code in `value_dashboard/` is a strong starting point.

The **chunked pipeline** in `pipeline/ih.py` and `pipeline/holdings.py` discovers files by glob, groups them by a regex extracted from filenames (typically a date), and processes each group as an independent unit. For each group it computes per-metric **partial aggregates** as Polars expressions, then `compact_data(...)` re-aggregates the concatenated partials to produce a final compacted dataset per metric. Intermediate and final tables are persisted in a per-variant DuckDB file (`db/pov_data_<variant>.duckdb`) via `PolarsDuckDBProxy`. File-level metadata (`save_file_meta`/`get_file_meta`) prevents re-processing the same file twice.

The **metric implementations** under `value_dashboard/metrics/` are textbook examples of mergeable state:

- `engagement.py` and `conversion.py` produce `(Count, Positives, Negatives)` (and `Revenue`, `Touchpoints` for conversion) per group-by tuple. The merge rule is plain summation.
- `experiment.py` does the same thing but always carries the experiment name and group so contingency tables can be reconstructed.
- `descriptive.py` carries `Count, Sum, Mean, Var, Min, Max` plus a binary t-digest per numeric column. Pooled variance is recomputed correctly during compaction.
- `ml.py` carries t-digests of `Propensity` and `FinalPropensity` separated by positive/negative outcome, plus `personalization`/`novelty` weighted means. ROC AUC, average precision, and calibration curves are reconstructed at query time from merged digests.
- `clv.py` carries `unique_holdings, lifetime_value, MinPurchasedDate, MaxPurchasedDate` (and `recurring_costs` for contractual CLV); `rfm_summary` derives RFM segments from those.

The **engine module** (`engine/processors.py`, `engine/query.py`, `engine/normalize.py`) is already a generalization toward processor-registry/metric-DSL territory: kinds (`binary_outcome`, `numeric_distribution`, `score_distribution`, `entity_lifecycle`, `entity_set`), state-merge rules (`sum, min, max, weighted_mean, tdigest_merge, hll_union`), and metric kinds (`formula, approx_distinct_count, curve_from_digests, calibration_from_digests, variant_compare, proportion_test, contingency_test, lifecycle_summary`).

The **wiki** captures the underlying mental model: which metric classes fit chunked pre-aggregation (binary rates, sums, means, pooled variance, sketch-able quantiles, HLL distincts) and which do not (sessionization, exact distinct, attribution, latest-state, sequence-sensitive).

### 2.2 Where it strains

| Strain | Symptom in current code | Implication for replacement |
|---|---|---|
| Two parallel configuration shapes | `normalize.py` translates legacy `[metrics.<family>]` TOML into a new `processors`/`metrics`/`reports` shape at runtime | The replacement should ship one DSL, validated, with a clean migration path |
| Code-name family detection by string prefix | `if metric.startswith("engagement")` everywhere (`pipeline/datatools.py`, `pipeline/ih.py`, `report_builder/recipes.py`) | Replace with explicit `kind` references and capability metadata |
| Filter / column expressions evaluated via `eval` of strings | `eval(global_ih_filter)`, `eval(add_columns)` in `pipeline/ih.py` and `pipeline/holdings.py` | Replace with a small typed expression DSL (parsed AST, no `eval`) |
| State-shape coupled to legacy column names | `Day/Month/Year/Quarter` synthesized by hand in IH loader, OUTCOME/INTERACTION_ID/RANK constants hard-coded | Make time grains and natural keys part of the source schema, not the engine |
| Reports tightly bound to one metric source | `params["metric"]` per report, no aggregate selection | Add aggregate routing so a report can be answered by the cheapest matching grain |
| Snapshot/state KPIs out of scope | Acknowledged in `wiki/chunked-bi-metrics.md` weak-fit table | First-class snapshot processor in the new design |
| Streamlit-coupled cache and session state | `@st.cache_data`, `st.session_state` reads inside `load_data` | Move caching/lineage to the storage layer; UI consumes a service |
| One DuckDB file per "variant" | `pov_data_<variant>.duckdb` is monolithic and large (the local checkout has files >700 MB) | Split by source × grain; keep variants as a top-level isolation namespace |
| `pandasai`-based "Chat with data" | LLM agent reads the aggregate tables directly | Governed LLM intent planner and MCP tools call the aggregate query layer without raw-row access |

These observations shape Value Stream's component boundaries below.

---

## 3. Goals and Non-Goals

### 3.1 Goals

1. **Aggregate-first.** Raw rows live only inside a single chunk's lazy pipeline; nothing raw is persisted, ever.
2. **Configurable.** A user with no Python skills can author a new metric and a new dashboard tile in YAML, with schema validation and useful errors.
3. **Multi-grain.** The same logical metric exists at multiple grains (daily, monthly, summary) and the query layer chooses the cheapest grain that still answers correctly.
4. **Mergeable state catalog.** Sum, min, max, weighted mean, pooled variance, t-digest, KLL, CPC, HLL, Theta sketches — and explicit rules for when each is correct.
5. **Snapshot support.** Periodic and accumulating snapshot tables for state KPIs (current pipeline, open tickets, subscription state).
6. **Programmable.** Current capabilities are reachable via CLI, Streamlit, Python SDK, DuckDB export, Chat With Data, local stdio MCP, and a read-only FastAPI HTTP API. Remote HTTP MCP and multi-user/OIDC deployment are deferred.
7. **Same stack.** Polars + DuckDB + Streamlit + Plotly stay; PyArrow Parquet is used for the long-term aggregate store and FastAPI supplies the read-only HTTP boundary.
8. **Provenance.** Every aggregate row carries refresh metadata (which chunk, which pipeline run, when, with what config hash).

### 3.2 Non-goals

- A streaming engine. Value Stream is batch / micro-batch; ingestion happens when new files appear or on a schedule.
- A general SQL warehouse. Value Stream is a metric platform; ad-hoc SQL is allowed against aggregate tables but not against raw events (which are not stored).
- A general distributed compute engine. Value Stream runs on one node, with optional shard-by-source parallelism; horizontal scale-out is a future concern.
- Replacing Pega CDH dataset semantics. Value Stream consumes IH/Holdings exports as-is; semantics live with Pega.

---

## 4. Core Design Principles

1. **Sufficient statistics over rows.** A processor's job is to produce a small, fixed-size record per (chunk, group-by tuple). The merge rule for that record is associative and commutative.
2. **One DSL, one validator.** YAML is the single source of truth for sources, processors, metrics, and dashboards. JSON Schema validates it; no ad-hoc Python expressions.
3. **Closed expression language.** Where dynamic expressions are unavoidable (filters, derived columns), they are parsed into a small typed AST and translated to Polars expressions — never `eval`-ed.
4. **Pluggable processor catalog.** Each processor is a Python class implementing a small interface (`schema`, `chunk_aggregate`, `merge`, `compact`, `derive`). Adding a processor never requires touching the engine.
5. **Aggregate routing.** A report names a metric, not a physical table. A planner picks the smallest aggregate that satisfies the report's group-by columns, time grain, and sketch needs.
6. **Idempotent chunks.** Re-running a chunk overwrites exactly that chunk's slice in the aggregate store; nothing else moves.
7. **Provenance per row.** Every aggregate row records `pipeline_run_id`, `chunk_id`, `config_hash`, `created_at`. This lets you answer "is this dashboard fresh?" without scanning files.

---

## 5. High-Level Architecture

```
                             +-------------------------------+
                             |   Configuration (YAML)        |
                             |   pipelines/ metrics/         |
                             |   processors/ dashboards      |
                             +---------------+---------------+
                                             |
                                             v
+------------+        +-------------+   +----+----+    +-----------------+
| File store | -----> |  Discovery  |-->| Chunk   |--->|   Aggregate     |
| (parquet,  |        |  & Grouping |   | Engine  |    |   Store         |
| pega zip,  |        +-------------+   | (Polars)|    | (Parquet+DuckDB)|
| csv, json) |                          +----+----+    +--------+--------+
+------------+                               |                   |
                                             |                   |
                                             v                   |
                                     +-------+-------+           |
                                     | Processor     |           |
                                     | Catalog       |           |
                                     +-------+-------+           |
                                             |                   |
                                             v                   v
                                  +----------+-------------------+----------+
                                  |              Query Layer                |
                                  |  (planner + executor + metric DSL)      |
                                  +----------+--------------------+---------+
                                             |                    |
                                             v                    v
                                  +---------+--------+   +-------+-------+
                                  |   Python SDK     |   | REST / remote |
                                  |   (in-process)   |   | MCP deferred  |
                                   +--+-----------+---+   +-------+-------+
                                      |           |               |
                          +-----------+--+   +----+------+ +------+--------+
                          | Streamlit UI |   | Notebook  | | LLM agents,   |
                          | (dashboards, |   | clients   | | external apps |
                          |  builder,    |   |           | |               |
                          |  ops console)|   +-----------+ +---------------+
                          +--------------+
```

**Layer responsibilities**

- **File store** — anywhere the data lands: local filesystem, S3, Pega DSS export folder, a partitioned parquet lake. Read-only to Value Stream.
- **Discovery & grouping** — finds files by glob, derives a `chunk_id` from a filename pattern, deduplicates against `chunks_meta`.
- **Chunk engine** — loads one chunk lazily (Polars), applies the source's transforms, fans out to all configured processors, writes per-processor partials.
- **Aggregate store** — Parquet files partitioned by `source × grain × period_key`, with DuckDB views over them and a small set of metadata tables.
- **Processor catalog** — pluggable classes that define how chunk partials look and merge.
- **Query layer** — planner picks the right physical aggregate, executor runs metric formulas / sketch queries / variant tests.
- **Surfaces** — Streamlit UI (interactive), Python SDK (in-process), DuckDB export, Chat With Data, local stdio MCP, and read-only FastAPI are current. Remote HTTP MCP and multi-user/OIDC deployment are deferred.

---

## 6. Data Model and Storage Layout

### 6.1 Storage hierarchy

The aggregate store is rooted at a configurable workspace path:

```
<workspace>/
├── catalog/
│   ├── pipelines.yaml            # source definitions
│   ├── processors.yaml           # processor configs (binding sources -> processors)
│   ├── metrics.yaml              # derived metric definitions
│   └── dashboards.yaml           # report/dashboard layouts
├── aggregates/
│   ├── <source_id>/
│   │   └── <processor_id>/
│   │       └── <grain>/
│   │           ├── period=YYYY-MM/   # hive-partitioned
│   │           │   └── part-<run>.parquet
│   │           └── _manifest.json
├── snapshots/
│   └── <snapshot_id>/
│       └── as_of=YYYY-MM-DD/
│           └── snapshot.parquet
├── meta/
│   ├── pipeline_runs.duckdb       # run-level metadata
│   ├── chunks.duckdb              # processed chunk ledger
│   ├── config_versions.duckdb     # config hashes & history
│   └── lineage.duckdb             # row-level provenance
└── duckdb/
    └── valuestream.duckdb             # views, dashboards, ad-hoc query target
```

This replaces the current single `db/pov_data_<variant>.duckdb` monolith. Variants survive as the top-level **workspace** identifier.

### 6.2 Why Parquet for aggregates and DuckDB for query

Parquet at rest gives:

- predictable on-disk layout (Hive partitioning on `period`),
- cheap deletes by partition (re-running a chunk just rewrites its partition),
- portability — the aggregate store is movable between environments,
- partial reads via column pruning and predicate pushdown.

DuckDB on top gives:

- a single SQL surface over any number of Parquet directories,
- `read_parquet('aggregates/.../**/*.parquet', hive_partitioning=true)`,
- table-valued functions for the metadata DBs,
- governed views over aggregate state, used by SQL tooling and the read-only HTTP/MCP surfaces when SQL is explicitly enabled.

### 6.3 Aggregate tables — common columns

Every aggregate table has, in addition to its group-by and state columns:

| Column | Type | Purpose |
|---|---|---|
| `pipeline_run_id` | `UUID` | the run that produced this row |
| `chunk_id` | `STRING` | the input file group (e.g. `2024-08-21`) |
| `period` | `STRING` | the time-grain partition value (e.g. `2024-08`) |
| `created_at` | `TIMESTAMP` | UTC, when the row was written |
| `config_hash` | `STRING` | sha256 of the materialized processor config |

These five columns enable:

- exact cache-busting when config changes,
- selective re-processing (drop and re-write a chunk),
- freshness reporting in the UI ("aggregates current as of …"),
- per-row lineage for audit.

### 6.4 Metadata tables (DuckDB)

```sql
-- meta/chunks.duckdb
CREATE TABLE chunks (
    source_id     VARCHAR NOT NULL,
    chunk_id      VARCHAR NOT NULL,
    files         JSON NOT NULL,          -- list of file paths in this chunk
    file_hash     VARCHAR NOT NULL,       -- sha256 of sorted (file, mtime, size)
    rows_in       BIGINT,                 -- input row count (for stats only)
    rows_kept     BIGINT,                 -- rows after filter/dedup
    started_at    TIMESTAMP,
    finished_at   TIMESTAMP,
    status        VARCHAR,                -- 'ok', 'failed', 'skipped'
    error         VARCHAR,
    pipeline_run_id UUID NOT NULL,
    PRIMARY KEY (source_id, chunk_id, pipeline_run_id)
);

-- meta/pipeline_runs.duckdb
CREATE TABLE pipeline_runs (
    id            UUID PRIMARY KEY,
    workspace     VARCHAR,
    source_id     VARCHAR,
    config_hash   VARCHAR,
    started_at    TIMESTAMP,
    finished_at   TIMESTAMP,
    status        VARCHAR,
    rows_in       BIGINT,
    rows_kept     BIGINT,
    chunks_total  INTEGER,
    chunks_ok     INTEGER,
    chunks_failed INTEGER
);

-- meta/config_versions.duckdb
CREATE TABLE config_versions (
    config_hash   VARCHAR PRIMARY KEY,
    yaml          VARCHAR,                -- canonicalized YAML body
    introduced_at TIMESTAMP
);
```

### 6.5 Mergeable state types

State columns in every aggregate table fall into one of these categories. Each is associative-commutative under its merge rule.

| Type | Storage column dtype | Merge rule | Used for |
|---|---|---|---|
| `count` | `BIGINT` | `SUM` | rows, positives, negatives, touchpoints |
| `value_sum` | `DOUBLE` | `SUM` | revenue, cost, quantities |
| `min` / `max` | matches data | `MIN` / `MAX` | first/last purchase, observed extremes |
| `pooled_mean` | `DOUBLE` (pair: `sum, count`) | `weighted_mean` | mean by group, ARPU, AOV |
| `pooled_variance` | `DOUBLE` (triple: `n, mean, m2`) | Welford-merge | variance, stddev, skew |
| `tdigest` | `BLOB` (datasketches-format) | `tdigest_merge` | quantiles, ROC AUC, calibration |
| `cpc` | `BLOB` (Apache DataSketches CPC) | `cpc_union` | default DAU/MAU/unique buyers/reach |
| `hll` | `BLOB` (Apache DataSketches HLL_8) | `hll_union` | backward-compatible / opt-in distinct counts |
| `theta` | `BLOB` (Theta sketch) | `theta_union/intersect/diff` | exact-set algebra (intersections, differences) — needed for retention/cohort |
| `kll` | `BLOB` (KLL sketch) | `kll_merge` | quantiles with stronger guarantees than t-digest |
| `topk` | `BLOB` (Frequent-Items sketch) | `topk_merge` | heavy hitters, frequent actions |

The state catalog is extensible — a processor declares which states it produces and by what type. The compaction engine looks up the merge rule from the type, not from the processor.

### 6.6 Multi-grain materialization

Each processor is materialized at one or more grains. Three are reserved and always available when a calendar column is declared:

- `daily` — one row per `(date, group_by_tuple)`
- `monthly` — one row per `(year-month, group_by_tuple)`
- `summary` — one row per `group_by_tuple` (no time)

A processor may also declare custom grains, e.g. `weekly_iso`, `quarterly`, `hourly` — each materialized as its own physical table.

Compaction across grains is automatic: monthly is compacted from daily, summary from monthly. Because state types are mergeable, this is a `groupby + merge` pass, not a re-aggregation from raw data.

---

## 7. Configuration DSL (YAML)

YAML replaces TOML. The full schema is published as JSON Schema and validated on load.

### 7.1 Top-level layout

```yaml
# pipelines.yaml
version: 1
workspace: bdt
defaults:
  time_zone: UTC
  calendar:
    grains: [daily, monthly, quarterly, yearly, summary]
    week_start: monday

sources:
  - id: ih
    description: Pega CDH Interaction History export
    reader:
      kind: pega_ds_export             # built-in reader
      file_pattern: "**/*.zip"
      group_by_filename: '\d{8}(?=\d{6}_)'   # chunk_id from filename
      streaming: true
    schema:
      timestamp_column: OutcomeTime
      natural_key: [InteractionID, ActionID, Rank, Outcome]
      drop_columns: [pyOutboundChannelInfo, ...]
    transforms:
      - kind: rename_capitalize
      - kind: parse_datetime
        columns: [OutcomeTime, DecisionTime]
        format: "%Y%m%dT%H%M%S%.3f %Z"
      - kind: derive_calendar
        from: OutcomeTime
        outputs: [Day, Month, Year, Quarter]
      - kind: derive_action_id
        parts: [Issue, Group, Name]
        sep: "/"
      - kind: filter
        expression:
          op: not_null
          column: Channel
      - kind: dedup
        keys: [InteractionID, ActionID, Rank, Outcome]
    defaults:
      ModelControlGroup: Test
      PlacementType: "N/A"
      ExperimentGroup: "N/A"
      FinalPropensity: 0.0
      Revenue: 0.0
```

The `transforms` block is a typed list — each item names a built-in transform with explicit parameters. Removed: arbitrary `eval`-strings.

### 7.2 Processor binding

```yaml
# processors.yaml
processors:

  - id: engagement
    source: ih
    kind: binary_outcome
    description: CTR, lift, and base counts per group-by tuple.
    group_by: [Channel, PlacementType, Issue, Group, CustomerType]
    time:
      column: OutcomeTime
      grains: [Day, Month, Summary]
    outcome:
      column: Outcome
      positive_values: [Clicked]
      negative_values: [Impression, Pending]
    dedup_keys: [InteractionID, ActionID, Rank]
    variant_column: ModelControlGroup
    variant_role_map:
      Test: Test
      Control: Control
    states:
      Count:               {type: count}
      Positives:           {type: count}
      Negatives:           {type: count}
      UniqueCustomers_cpc: {type: cpc, source_column: CustomerID, lg_k: 11}
      UniqueActions_cpc:   {type: cpc, source_column: ActionID,   lg_k: 11}

  - id: conversion
    source: ih
    kind: binary_outcome
    description: Conversion rate and revenue per group-by tuple.
    group_by: [Channel, PlacementType, Issue, Group, CustomerType]
    time:
      column: OutcomeTime
      grains: [Day, Month, Summary]
    outcome:
      column: Outcome
      positive_values: [Conversion]
      negative_values: [Impression]
    dedup_keys: [InteractionID, ActionID, Rank]
    touchpoint:
      customer_column: CustomerID
      event_column: ConversionEventID
      output_state: Touchpoints
    states:
      Count:        {type: count}
      Positives:    {type: count}
      Negatives:    {type: count}
      Revenue:      {type: value_sum, source_column: Revenue}
      Touchpoints:  {type: count}

  - id: experiment
    source: ih
    kind: binary_outcome
    group_by: [Year, Channel, PlacementType, CustomerType, ExperimentName, ExperimentGroup]
    time:
      column: OutcomeTime
      grains: [Month, Summary]
    outcome:
      column: Outcome
      positive_values: [Clicked]
      negative_values: [Impression, Pending]
    variant_column: ExperimentGroup
    filter:
      op: in
      column: ModelControlGroup
      values: [Test, Control]

  - id: descriptive
    source: ih
    kind: numeric_distribution
    group_by: [Channel, PlacementType, Issue, Group, CustomerType, Outcome]
    time:
      column: OutcomeTime
      grains: [Day, Month, Summary]
    properties: [Outcome, Propensity, FinalPropensity, Priority, ResponseTime]
    quantile_engine: tdigest        # or: kll, exact_quantiles
    states:
      "{prop}_Count":    {type: count,           per_property: true}
      "{prop}_Sum":      {type: value_sum,       per_property: true, numeric_only: true}
      "{prop}_Mean":     {type: pooled_mean,     per_property: true, numeric_only: true}
      "{prop}_Var":      {type: pooled_variance, per_property: true, numeric_only: true}
      "{prop}_Min":      {type: min,             per_property: true, numeric_only: true}
      "{prop}_Max":      {type: max,             per_property: true, numeric_only: true}
      "{prop}_tdigest":  {type: tdigest,         per_property: true, numeric_only: true}

  - id: model_ml_scores
    source: ih
    kind: score_distribution
    group_by: [Channel, PlacementType, Issue, Group, CustomerType]
    time:
      column: OutcomeTime
      grains: [Day, Month, Summary]
    score_properties: [Propensity, FinalPropensity]
    outcome:
      column: Outcome
      positive_values: [Clicked]
      negative_values: [Impression, Pending]
    states:
      Count:                          {type: count}
      personalization:                {type: pooled_mean, source_metric: personalization}
      novelty:                        {type: pooled_mean, source_metric: novelty}
      Propensity_tdigest_positives:       {type: tdigest, source_column: Propensity,      score_property: Propensity,      outcome: positive}
      Propensity_tdigest_negatives:       {type: tdigest, source_column: Propensity,      score_property: Propensity,      outcome: negative}
      FinalPropensity_tdigest_positives:  {type: tdigest, source_column: FinalPropensity, score_property: FinalPropensity, outcome: positive}
      FinalPropensity_tdigest_negatives:  {type: tdigest, source_column: FinalPropensity, score_property: FinalPropensity, outcome: negative}
      UniqueCustomers_cpc:            {type: cpc, source_column: CustomerID, lg_k: 11}

  - id: clv
    source: holdings
    kind: entity_lifecycle
    group_by: [ControlGroup]
    time:
      column: PurchasedDateTime
      grains: [Year, Summary]
    keys:
      customer_id: CustomerID
      order_id:    HoldingID
      monetary:    OneTimeCost
      purchase_date: PurchasedDateTime
    model: non_contractual              # or: contractual
    lifespan_years: 9
    rfm_segments: retail_banking
    states:
      unique_holdings:        {type: count}
      lifetime_value:         {type: value_sum}
      MinPurchasedDate:       {type: min}
      MaxPurchasedDate:       {type: max}
      UniquePurchasers_cpc:   {type: cpc, source_column: CustomerID, lg_k: 11}

  - id: action_funnel
    source: ih
    kind: funnel
    group_by: [Channel, PlacementType]
    time:
      column: OutcomeTime
      grains: [Day, Month]
    stages:
      - {name: Impression, when: {op: eq, column: Outcome, value: Impression}}
      - {name: Clicked,    when: {op: eq, column: Outcome, value: Clicked}}
      - {name: Conversion, when: {op: eq, column: Outcome, value: Conversion}}
    entity: CustomerID

  - id: subscription_state            # snapshot example (not in current app)
    source: subscriptions
    kind: snapshot
    snapshot_kind: periodic            # periodic | accumulating
    cadence: daily
    group_by: [Plan, Region]
    states:
      ActiveSubs:   {type: count}
      MRR:          {type: value_sum, source_column: monthly_recurring}
      ChurnedSubs:  {type: count}
```

### 7.3 Derived metric definitions

Derived metrics are not stored — they are computed at query time from a processor's state.

```yaml
# metrics.yaml
metrics:

  CTR:
    source: engagement
    kind: formula
    description: Positive outcomes divided by classified outcomes.
    display:
      label: Engagement rate
      unit: percent
      value_format: percent
      direction: higher_is_better
    expression: { op: safe_div, num: {col: Positives}, den: {op: add, args: [{col: Positives},{col: Negatives}]} }

  StdErr:
    source: engagement
    kind: formula
    depends_on: [CTR]
    expression:
      op: sqrt
      arg:
        op: safe_div
        num: { op: mul, args: [{col: CTR}, {op: sub, args: [{lit: 1.0},{col: CTR}]}] }
        den: { op: add, args: [{col: Positives},{col: Negatives}] }

  Lift:
    source: engagement
    kind: variant_compare
    variant_column: ModelControlGroup
    test_role: Test
    control_role: Control
    confidence_level: 0.95
    outputs: [TestCTR, ControlCTR, TestSampleSize, ControlSampleSize,
              AbsoluteRateDifference, AbsoluteRateDifference_CI_Low,
              AbsoluteRateDifference_CI_High, Lift, Lift_Z_Score,
              Lift_P_Val, StdErr]

  ConversionRate:
    source: conversion
    kind: formula
    expression: { op: safe_div, num: {col: Positives}, den: {op: add, args: [{col: Positives},{col: Negatives}]} }

  AvgTouchpoints:
    source: conversion
    kind: formula
    expression: { op: safe_div, num: {col: Touchpoints}, den: {col: Positives} }

  UniqueCustomers:
    source: engagement
    kind: approx_distinct_count
    state: UniqueCustomers_cpc

  ROC_AUC:
    source: model_ml_scores
    kind: curve_from_digests
    positive_state: Propensity_tdigest_positives
    negative_state: Propensity_tdigest_negatives
    output: roc_auc

  AvgPrecision:
    source: model_ml_scores
    kind: curve_from_digests
    positive_state: Propensity_tdigest_positives
    negative_state: Propensity_tdigest_negatives
    output: average_precision

  Calibration:
    source: model_ml_scores
    kind: calibration_from_digests
    positive_state: FinalPropensity_tdigest_positives
    negative_state: FinalPropensity_tdigest_negatives

  Median_ResponseTime:
    source: descriptive
    kind: tdigest_quantile
    state: ResponseTime_tdigest
    quantile: 0.5

  Experiment_Significance:
    source: experiment
    kind: contingency_test
    variant_column: ExperimentGroup
    tests: [chi2, g, z]
    outputs: [chi2_stat, chi2_p_val, g_stat, g_p_val, z_score, z_p_val,
              chi2_odds_ratio_stat, chi2_odds_ratio_ci_low, chi2_odds_ratio_ci_high]

  CLV_Summary:
    source: clv
    kind: lifecycle_summary
    rfm_segments_dict: ${ pipelines.clv.rfm_segments }
    outputs: [recency, frequency, monetary_value, tenure, lifetime_value,
              r_quartile, f_quartile, m_quartile, rfm_score, rfm_segment]
```

### 7.4 Reports and dashboards

Reports are bound to **metrics**, not to processors. The query planner picks the right processor and grain.

```yaml
# dashboards.yaml
theme:
  category_colors:
    Channel: {Web: "#2563EB", Mobile: "#14B8A6"}
dashboards:

  - id: marketing_overview
    title: Marketing Overview
    layout: tabs
    pages:
      - id: engagement
        title: Engagement
        time_filter:
          default: all_time
          presets: [last_30_days, last_90_days, year_to_date, custom, all_time]
        filters:
          - field: Channel
            label: Channel
            display: primary
            scope: all_tiles
            control: multiselect
        tiles:
          - id: ctr_kpi
            title: Engagement rate
            metric: CTR
            chart: kpi_card
            value: CTR
            placement: kpi_strip
            kpi:
              comparison: previous_period
              comparison_period: month
              sparkline_grain: daily
              sparkline_points: 30

          - id: daily_ctr
            title: Daily CTR by group
            metric: CTR
            chart: line
            x: day
            y: CTR
            color: group
            facets: {row: channel, col: placement}
            scale_mode: absolute

          - id: ctr_treemap
            title: CTR treemap
            metric: CTR
            chart: treemap
            path: [channel, placement, issue, group]
            color: CTR

          - id: ctr_gauge
            title: CTR vs target
            metric: CTR
            chart: gauge
            value: CTR
            references:
              Web/Leaderboard: 0.015
              Web/Skyscrapper: 0.004
              Mobile/Leaderboard: 0.03

      - id: ml
        title: Model quality
        tiles:
          - id: daily_roc_auc
            title: Daily ROC AUC by placement
            metric: ROC_AUC
            chart: line
            x: day
            y: ROC_AUC
            color: placement
            facets: {row: channel}

          - id: calibration
            title: Calibration curve
            metric: Calibration
            chart: calibration_curve
            facets: {row: channel}

      - id: experiments
        title: Experiments
        tiles:
          - id: experiment_significance
            title: Significance per experiment
            metric: Experiment_Significance
            chart: bar
            x: ExperimentName
            y: chi2_p_val
            facets: {row: channel}
```

Page filters may only reference dimensions already persisted by the backing
processors. `all_tiles` is rejected unless every tile supports the field;
`compatible_tiles` remains valid when at least one tile supports it, and the UI
shows both filter and tile coverage. KPI placement is explicit: report code does
not derive cards from arbitrary chart rows or guess a reducer. These properties
are presentation/query options over existing aggregates and do not trigger raw
reprocessing.

### 7.5 Expression AST

The closed expression DSL used in `filter`, derived columns, and metric formulas:

```yaml
# Atoms
{ col: <name> }
{ lit: <value> }

# Logical
{ op: and, args: [...] } | { op: or, args: [...] } | { op: not, arg: ... }

# Comparison
{ op: eq | ne | lt | le | gt | ge, column: <name>, value: <lit> }
{ op: in | not_in, column: <name>, values: [...] }
{ op: between, column: <name>, low: <lit>, high: <lit> }
{ op: not_null | is_null, column: <name> }
{ op: matches, column: <name>, pattern: <regex> }

# Arithmetic
{ op: add | sub | mul | div, args: [...] }
{ op: safe_div, num: <expr>, den: <expr> }
{ op: sqrt | log | exp | abs, arg: <expr> }
{ op: case, when: [{cond: <expr>, then: <expr>}, ...], else: <expr> }

# Date / time helpers
{ op: date_trunc, unit: day|month|quarter|year, arg: <expr> }
{ op: date_diff, unit: day|month|year, end: <expr>, start: <expr> }
{ op: now }
```

These translate cleanly into Polars expressions inside the engine; nothing is `eval`-ed.

---

## 8. Processor Catalog and Aggregate Schemas

### 8.1 Processor interface

```python
class Processor(Protocol):
    id: str
    kind: str
    source_id: str
    group_by: list[str]
    grains: list[str]
    states: dict[str, StateSpec]

    def schema(self, grain: str) -> pa.Schema: ...
    def chunk_aggregate(self, lf: pl.LazyFrame, ctx: ChunkCtx) -> pl.DataFrame: ...
    def merge(self, frames: Iterable[pl.DataFrame]) -> pl.DataFrame: ...
    def compact(self, df: pl.DataFrame, target_grain: str) -> pl.DataFrame: ...
    def derive(self, df: pl.DataFrame, params: dict) -> pl.DataFrame: ...
```

`merge` is generic — it iterates the processor's state catalog and applies the per-state-type rule (sum, weighted_mean, tdigest_merge, cpc_union, hll_union, …). Subclasses rarely override it.

### 8.2 Built-in processor kinds

#### binary_outcome

```text
Schema (daily grain):
+-----------------------+----------+
| <dim_columns>         | string   |
| Day                   | date     |
| Count                 | int64    |
| Positives             | int64    |
| Negatives             | int64    |
| Revenue               | double   |   (only if defined in states)
| Touchpoints           | int64    |   (only if touchpoint configured)
| <cardinality sketches>| bytes    |
| pipeline_run_id       | uuid     |
| chunk_id              | string   |
| period                | string   |
| created_at            | timestamp|
| config_hash           | string   |
+-----------------------+----------+
```

Chunk recipe (Polars):

```python
ih.filter(outcome_in(positive ∪ negative))
  .with_columns(outcome_binary)
  .filter(dedup_max_outcome over dedup_keys)
  .group_by(group_by + [time_grain])
  .agg([
      pl.len().alias("Count"),
      pl.sum("outcome_binary").alias("Positives"),
      *value_aggs,
      *touchpoint_aggs,
      *cardinality_sketch_build_aggs,
  ])
  .with_columns((pl.col("Count") - pl.col("Positives")).alias("Negatives"))
```

This is the same recipe the current app uses, but parameterized by config rather than hard-coded.

#### numeric_distribution

State per numeric column `<prop>`: `Count, Sum, Mean, Var, Min, Max, tdigest`.
Pooled-variance compaction follows Welford-merge:

```text
m2_total  = sum( (n_i - 1) * var_i + n_i * (mean_i - global_mean)^2 )
var_total = m2_total / (N - 1)
```

#### score_distribution

Specialization of `binary_outcome` plus per-outcome score-digest states. Curves (ROC, PR, calibration) are reconstructed at query time by deserializing+merging the digests.

#### entity_lifecycle (CLV)

Per `(customer, year, quarter, group_by_tuple)` row carries `unique_holdings, lifetime_value, MinPurchasedDate, MaxPurchasedDate, UniquePurchasers_cpc`. RFM segments and quartiles are derived in the query layer.

#### entity_set

Pure CPC/HLL/Theta sketch processor for unique-count / set-algebra metrics.

#### funnel

Stage flags are computed during chunk aggregation; per-stage counts are stored. Drop-off and stage rates are derived metrics.

#### snapshot

```text
snapshot_kind: periodic | accumulating
cadence: daily | weekly | monthly
state schema: any combination of count / value_sum / cpc / hll / theta
partitioning: as_of_date
```

A periodic snapshot writes one row per `(as_of_date, group_by_tuple)`.
Accumulating snapshots keep entity rows in immutable chunk partials; query
merging resolves conflicts by `MAX(as_of_date)` and then `created_at`.

### 8.3 Worked example — engagement aggregate at daily grain

```text
aggregates/ih/engagement/daily/period=2024-08/part-<run>.parquet
```

| Day | channel | placement | issue | group | customer_type | ModelControlGroup | Count | Positives | Negatives | UniqueCustomers_cpc | UniqueActions_cpc | pipeline_run_id | chunk_id | period | created_at | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2024-08-21 | Web | Leaderboard | Sales | Cards | Premium | Test | 12450 | 184 | 12266 | 0x… | 0x… | 8a32… | 2024-08-21 | 2024-08 | 2024-08-21T22:13:01Z | f1c2… |
| 2024-08-21 | Web | Leaderboard | Sales | Cards | Premium | Control | 1230 | 8 | 1222 | 0x… | 0x… | 8a32… | 2024-08-21 | 2024-08 | 2024-08-21T22:13:01Z | f1c2… |

For each chunk, monthly materialization consumes the base daily aggregate state
(never the raw rows), groups by `dim_tuple` without `Day`, sums counts, unions
CPC sketches, and writes an immutable run/chunk partial:

```text
aggregates/ih/engagement/monthly/period=2024-08/part-<run>.parquet
```

Cross-grain materialization is purely an aggregate-state operation — no raw
data is involved. Query-time merging combines the latest successful partial per
chunk; `valuestream vacuum` later removes superseded physical files.

---

## 9. Query Layer

### 9.1 Planner

```text
input: (metric, group_by, filters, time_axis/time_range, optional grain=auto)
1. resolve metric -> processor + state requirements + derive function
2. derive the requested logical grain from explicit criteria such as
   time_axis=Day/Month/Quarter/Year; default to summary when no time bucket is
   requested
3. enumerate physical aggregates of the processor (grain × period scope)
4. pick the smallest aggregate whose grain >= requested logical grain, or a
   finer aggregate that can be safely rolled up when the coarser aggregate is
   not materialized,
   AND whose group-by columns ⊇ requested group_by
   AND whose states cover the metric's needs
5. emit a DuckDB query:
     SELECT <dim_cols>, <state_cols>
     FROM read_parquet('aggregates/<source>/<processor>/<grain>/**/*.parquet',
                       hive_partitioning=true)
     WHERE <period predicate> AND <dim filter predicates>
6. run the metric's `derive` (formula / sketch query / variant compare / contingency test)
7. return polars.DataFrame
```

The planner is pure: same criteria → same resolved grain → same plan → same
SQL. Plans are cached by `(metric_id, dim_set, filters_hash, resolved_grain,
config_hash)`.

### 9.2 Python SDK

```python
import valuestream as ac

ws = ac.Workspace(path="~/valuestream/bdt")

# Run pipelines (idempotent)
ws.run_source("ih")
ws.run_source("holdings")

# Query one metric
df = ws.metric("CTR").by("day", "channel", "group") \
       .where(channel="Web", group__in=["Cards","Loans"]) \
       .between("2024-08-01", "2024-08-31") \
       .to_polars()

# Render a tile
ws.dashboard("marketing_overview").tile("daily_ctr").to_plotly()

# Render a whole dashboard (returns dict of tile_id -> plotly Figure)
figs = ws.dashboard("marketing_overview").to_plotly()
```

### 9.3 Read-only HTTP API (FastAPI)

The implemented API is deliberately smaller than the broader target sketch
below. `valuestream serve-api` currently exposes health, metric manifest/query,
validated chart query, dimension values, freshness, and optional Chat. Governed
SQL schema/query endpoints exist only with `--enable-sql`. Metric-query
responses include catalog/computation hashes, chosen physical grain,
contributing run/chunk IDs, aggregate scan count, and latest aggregate time.
The CLI refuses non-loopback binding without a bearer token.

Source-run triggers, config mutation, dashboard mutation, remote HTTP MCP, and
OIDC are still target architecture rather than current product behavior.

OpenAPI sketch:

```yaml
openapi: 3.1.0
info: { title: Value Stream API, version: 1.0 }
paths:

  /api/v1/workspace:
    get:    { summary: Get workspace metadata, freshness, version }

  /api/v1/sources:
    get:    { summary: List configured sources }
  /api/v1/sources/{source_id}/run:
    post:   { summary: Trigger a source pipeline run }
  /api/v1/sources/{source_id}/runs:
    get:    { summary: List recent pipeline runs for a source }
  /api/v1/sources/{source_id}/chunks:
    get:    { summary: List processed chunks for a source }

  /api/v1/processors:
    get:    { summary: List processors and their grains/group-by columns }

  /api/v1/metrics:
    get:    { summary: List defined metrics and their bound processors }
  /api/v1/metrics/{metric_id}/query:
    post:
      summary: Query a metric
      requestBody:
        application/json:
          schema:
            type: object
            properties:
              group_by:     { type: array, items: { type: string } }
              filters:      { type: object, additionalProperties: true }
              time_range:   { type: object, properties: { from: {type: string, format: date}, to: {type: string, format: date} } }
              grain:        { type: string, enum: [daily, monthly, quarterly, yearly, summary, auto] }
              format:       { type: string, enum: [json, arrow, csv], default: json }
      responses:
        200:
          description: rows + plan metadata
          content:
            application/json:
              schema:
                type: object
                properties:
                  rows:           { type: array, items: { type: object } }
                  plan:           { type: object }      # which physical aggregate was used
                  config_hash:    { type: string }
                  freshness:      { type: object }      # latest chunk per period

  /api/v1/dashboards:
    get:    { summary: List dashboards }
  /api/v1/dashboards/{id}:
    get:    { summary: Get dashboard layout (tiles, axes, refs) }
  /api/v1/dashboards/{id}/tiles/{tile_id}/data:
    get:    { summary: Resolve a tile to data + chart spec (Plotly JSON) }

  /api/v1/admin/config:
    get:    { summary: Get the active config + hash }
    put:    { summary: Replace the active config (validated by JSON Schema) }
```

The read-only endpoints support headless clients and notebooks. Streamlit
continues to use the in-process query layer directly for latency and
authentication simplicity.

### 9.4 SQL surface

DuckDB views materialized at startup:

```sql
CREATE OR REPLACE VIEW v_engagement_daily AS
  SELECT * FROM read_parquet(
    'aggregates/ih/engagement/daily/**/*.parquet',
    hive_partitioning=true);

CREATE OR REPLACE VIEW v_metric_ctr_daily AS
  SELECT
    Day, channel, placement, issue, "group", customer_type,
    ModelControlGroup,
    Count, Positives, Negatives,
    Positives::DOUBLE / NULLIF(Positives + Negatives, 0) AS CTR
  FROM v_engagement_daily;
```

Power users can write SQL against these views for ad-hoc questions — but the views never expose raw data.

---

## 10. Chunked Processing Engine

### 10.1 Run loop

```python
def run_source(source_id: str) -> RunReport:
    cfg = load_config()
    source = cfg.sources[source_id]
    run_id = uuid4()
    config_hash = hash_canonical(cfg)

    write_run(pipeline_runs, run_id, source_id, config_hash, status="running")

    files = discover_files(source.reader)
    chunks = group_files_by_pattern(files, source.reader)

    processors = [p for p in cfg.processors if p.source == source_id]

    for chunk_id, files in chunks.items():
        if already_processed(source_id, chunk_id, config_hash):
            continue
        with chunk_lifecycle(run_id, source_id, chunk_id):
            lf = read_files_lazy(files, source.reader)
            lf = apply_transforms(lf, source.transforms)

            for proc in processors:
                partial = proc.chunk_aggregate(lf, ctx=ChunkCtx(run_id, chunk_id))
                write_partial(proc, chunk_id, partial)

            write_chunk_meta(...)

    compact_grains(processors)             # daily -> monthly -> summary
    refresh_duckdb_views()

    finalize_run(run_id, status="ok")
```

### 10.2 Discovery and grouping

A chunk is the unit of idempotency. `chunk_id` is derived from the filename via the `group_by_filename` regex; if the regex fails the basename is used. Multiple files can belong to one chunk (e.g. partitioned exports).

Re-processing a chunk:

1. Pipeline writes a new partial parquet under the chunk's `period` partition with a new `pipeline_run_id`.
2. The next compaction step reads only **the latest run** per chunk (by `created_at`), so older partials are ignored.
3. A janitor job (`valuestream vacuum`) periodically deletes superseded partials.

The Data Load **Rebuild from scratch** operation is the coordinated retention
variant. It holds every selected source lock, force-processes all currently
discovered chunks, rejects empty or incomplete runs and catalog changes, then
performs a source-scoped vacuum that retains only the new successful run for
each selected source. Physical aggregate files from earlier or now-absent
chunks are removed only after those checks. DuckDB run, chunk, lineage, and
configuration-version metadata is not deleted.

### 10.3 Backpressure and streaming

Polars streaming engine is opt-in per source (`reader.streaming: true`). For very large chunks the engine collects the source LazyFrame once, aliases it through the Polars LRU cache, and fans out to processors so the IO is paid once. Memory pressure is monitored — if RSS exceeds a configured ceiling the engine spills the source DataFrame to a temporary parquet and re-scans for each processor.

### 10.4 Async fan-out

Processors are independent. The engine launches them with `asyncio.gather` (one task per processor) and Polars' background collect path. This mirrors the current `collect_ih_metrics_data` design but is generalized to N processors of arbitrary kind.

### 10.5 Failure semantics

- A processor failure isolates to that processor. Other processors finish; the failed one is recorded in `chunks` with `status='failed'` and an error message; the run is marked `status='partial'`.
- Source-level failures (e.g. unreadable file) abort the chunk, not the run; subsequent chunks are processed.
- Re-running the source picks up only failed/missing chunks unless `--force` is set.

---

## 11. Presentation Layer

### 11.1 Streamlit UI

```
Value Stream (sidebar)
├── Workspace  ← variants live here (Default, Demo, BDT, RBB, NBS …)
├── Pipelines
│   ├── Sources
│   ├── Recent runs
│   └── Chunks
├── Catalog
│   ├── Dimensions
│   ├── Processors
│   └── Metrics
├── Dashboards
│   └── (one entry per dashboards.yaml definition)
├── Builder         ← new dashboard / new metric authoring
├── Ops
│   ├── Config
│   ├── Health & freshness
│   └── Lineage
└── Chat With Data (LLM intent planner)
```

#### Dashboard rendering

A dashboard is a YAML-driven page. For each tile:

1. Resolve `metric` → processor + grain.
2. Resolve `chart` → Plotly figure factory (`line, bar, treemap, heatmap, scatter, bar_polar, gauge, funnel, boxplot, histogram, rfm_density, calibration_curve, exposure, corr, model`).
3. Pull data via the SDK with the tile's `group_by/filters/time_range`.
4. Pass into the figure factory.
5. Render with `st.plotly_chart(fig, use_container_width=True)`.

The figure factory registry is one Python module per chart type (mirrors the current `reports/*_plots.py` but each function takes a uniform `(df, tile_spec)` signature instead of bespoke configs).

#### Builder UI

Replaces the current `report_builder/` and most manual `config_generator/` editing with a single validation-first Builder workspace:

- **Health and review progress** — show source/processor/metric/report validation status before editing.
- **Sources** — edit reader kind, file pattern, grouping regex, root, streaming/hive flags, schema timestamp/natural-key/drop columns, source defaults, dataset filters, and calculated fields. Defaults run before filters; filter rules compile to the closed expression AST; calculated fields become `derive_column` transforms.
- **Dimensions** — review and update processor group-by fields from the approved source field catalog.
- **Processors** — edit source binding, kind, group-by, time grains, state columns, state source fields, and optional processor filters.
- **Metrics** — browse the shared KPI recipe library by business question/domain, inspect method accuracy and readiness, select processor-owned business fields plus any recipe-compatible sketch algorithm (or funnel stages/populations), optionally place a recommended report tile, or build a metric directly with the visual/AST editors. Internal state IDs remain technical detail. Before apply, the shared flow shows the exact generated YAML patch and, for a missing field/algorithm state, the source, fields, states, and current/proposed processor computation hashes. Configuration Builder installs and post-validates that patch inside one rollback boundary, then links to Data Load when a source run is required; it never starts ingestion implicitly.
- **Reports / Tiles** — search/filter a report library, create/duplicate/delete tiles, use visual chart-field mapping or raw YAML fallback, inspect chart recipe metadata, preview against aggregates, and save into `dashboards.yaml`.
- **Chat Review** — review the aggregate metrics, processors, and persisted group-by fields exposed to Chat With Data; confirm catalog validation and LLM settings readiness before relying on chat answers; edit chat-only `ai.yaml` guidance such as the generic agent prompt plus dataset/metric descriptions used by the LLM planner.
- **Settings** — edit `pipelines.yaml` workspace defaults such as workspace name, time zone, calendar grains, and week start, plus `dashboards.yaml` theme settings.
- **Save & Export** — show and download each catalog file independently because Value Stream stores YAML across `pipelines.yaml`, `processors.yaml`, `metrics.yaml`, and `dashboards.yaml`.

The implemented Builder presents these areas through a compact step selector
with Previous/Next actions and a phase progress bar. Validation and review
status are summarized in Workspace Health rather than duplicated above every
step. Shared report/home metric cards use breakpoint-aware grids, and both UI
and Plotly themes use explicit light/dark surface and contrast tokens.

AI Configuration Studio remains a separate guided workflow because it starts from a sample file rather than an existing catalog:

- **Sample** — upload/select a source sample and review runtime reader controls.
- **Required fields** — map subject, outcome time, decision time, and outcome fields.
- **Defaults, filters, calculations** — use the same row editors as the Builder; filters compile to AST, calculations compile to `derive_column` transforms.
- **Approve fields** — choose which working fields are eligible for draft metric/report generation and which sample values may be shared with a future LLM.
- **Draft** — generate reviewed YAML drafts. LLM responses can populate source, processor, metric, report, chat-readiness, workspace-default, and dashboard-theme settings; deterministic generation remains available as a baseline and fallback.
- **Save & Export** — download draft YAML or apply it through the same structured writers. Sources, processors, metrics, dashboards, and `ai.yaml` share one rollback boundary that includes post-write catalog validation. **Apply Draft & Run Source** is the separate explicit materialization action.

### 11.2 Plotly chart kinds (catalog parity with current app)

| Chart kind | Required tile fields | Notes |
|---|---|---|
| `line` | `x, y, color?, facets?` | drops to bar if `x` is categorical |
| `bar` | `x, y, color?, facets?` | |
| `bar_polar` | `r, theta, color` | |
| `treemap` | `path, color` | |
| `heatmap` | `x, y, color` | |
| `scatter` | `x, y, size?, color?, animation_frame?, animation_group?` | |
| `gauge` | `value, references?` | |
| `funnel` | `stages, color` | from `funnel` processor |
| `boxplot` | `x, y, color?` | from `numeric_distribution` quantile states |
| `histogram` | `property, color?, facets?` | reconstructed from t-digest bins |
| `calibration_curve` | `metric: Calibration` | from `calibration_from_digests` |
| `rfm_density` | `metric: CLV_Summary` | 2D / 3D density of R-F-M |
| `exposure` | `metric: CLV_Summary` | customer exposure curve |
| `corr` | `x, y` | Pearson/Spearman from sufficient stats |
| `model` | `metric: CLV_Summary` | fitted Beta-Geometric / Pareto-NBD overlay |

### 11.3 Chat with data MLP1

The first replacement for the `pandasai`-based page is a governed LLM intent
planner, not generated Python. The Streamlit Chat page can call a configured
LiteLLM model to translate natural language into a JSON intent:

- metric id,
- group-by dimensions,
- aggregate filters,
- optional time axis and time range,
- response type (`text`, `table`, or `chart`),
- optional chart spec (`line`, `bar`, `table`, `kpi_card`).

Value Stream validates that intent against the active catalog, derives the
logical query grain from the criteria, and then executes `query_metric`. The LLM
does not choose aggregate grains and never receives raw source rows, aggregate
parquet paths, or arbitrary SQL/Python execution rights. Chart generation is
deterministic: the LLM selects from an allowlist and the app renders Plotly from
the validated aggregate result.

Chat With Data reads optional prompt guidance from `ai.yaml` under
`chat_with_data`. `agent_prompt`, `dataset_descriptions`, and
`metric_descriptions` are included only in LLM planning prompts; they do not
change catalog metric definitions, persisted aggregates, or query execution.
Governed planner rules override chat guidance when there is any conflict.

The MLP1 MCP surface is local stdio only and exposes a small read-only tool
set for Claude Code and similar MCP clients:

- `metric_list`
- `metric_query`
- `dimension_values_tool`
- `freshness_get`

REST, remote HTTP MCP, OAuth/OIDC auth, dashboard tile tools, lineage drilldown,
and generated-code analysis remain later surfaces. See
`docs/design/chat-with-data-mlp1.md` for provider setup and runbooks.

---

## 12. Migration Strategy from the Current App

### 12.1 Mapping table

| Current concept | Value Stream concept |
|---|---|
| `[ih]` / `[holdings]` TOML section | `sources.<id>.reader` + `transforms` |
| `[ih.extensions.filter]` (eval string) | `transforms[*].kind: filter` with AST |
| `[ih.extensions.columns]` (eval list) | `transforms[*].kind: derive_column` with AST |
| `[ih.extensions.default_values]` | `sources.<id>.defaults` |
| `[metrics.engagement]` | `processors.engagement` (kind=binary_outcome) + `metrics.CTR/Lift/StdErr` |
| `[metrics.conversion]` | `processors.conversion` + `metrics.ConversionRate/AvgTouchpoints/Revenue` |
| `[metrics.experiment]` | `processors.experiment` + `metrics.Experiment_Significance` |
| `[metrics.descriptive]` | `processors.descriptive` (numeric_distribution) + per-property quantile/mean metrics |
| `[metrics.model_ml_scores]` | `processors.model_ml_scores` (score_distribution) + `metrics.ROC_AUC/AvgPrecision/Calibration` |
| `[metrics.clv]` | `processors.clv` (entity_lifecycle) + `metrics.CLV_Summary` |
| `[reports.<name>]` | `dashboards[].pages[].tiles[]` |
| `[variants]` | `workspace` identifier |
| `[chat_with_data]` | Governed Chat With Data planner plus local read-only MCP metric tools |
| `db/pov_data_<variant>.duckdb` | `aggregates/`, `meta/`, `duckdb/valuestream.duckdb` |
| `processed_files` table | `meta/chunks.duckdb` |

### 12.2 One-shot migration tool

`valuestream migrate --from value_dashboard/config/config.toml --to valuestream/catalog/`

The tool:

1. Reads the legacy TOML.
2. Runs the current app's `engine/normalize.py` logic to obtain `processors` + `metrics` + `reports` shapes.
3. Translates each into Value Stream YAML following the mapping table.
4. Translates `eval` strings into AST equivalents using a curated parser. Where a string cannot be safely translated, the tool emits a `# TODO migrate-by-hand` comment with the original text.
5. Emits a `migration_report.md` listing every legacy field, its target, and any gaps.

### 12.3 Aggregate-store backfill

`valuestream backfill --source ih --workspace bdt`

Reads the existing `pov_data_<variant>.duckdb` tables, re-keys the rows into the new partitioned parquet layout, and populates `meta/chunks.duckdb` with one synthetic `chunk_id` per (file, processed_at) pair from `processed_files`. The shape of the rows is preserved so no recomputation from raw is needed.

### 12.4 Run side-by-side

For one release, the legacy Streamlit app and the new one can read the same workspace folder. The legacy app keeps writing to `db/pov_data_<variant>.duckdb`; Value Stream reads from `aggregates/...`. Both can coexist while users get familiar with the new UI.

---

## 13. Observability and Health

| Surface | What it shows |
|---|---|
| `Pipelines / Recent runs` page | run id, source, started/finished, status, rows, chunks ok/failed |
| `Pipelines / Chunks` page | per-chunk ledger, with re-run button |
| `Ops / Health` page | data freshness per source × grain (latest `period`), config hash, schema-version drift |
| `Ops / Lineage` | for any tile, "this number came from these aggregate rows from these chunks" |
| Prometheus `/metrics` | Deferred service endpoint for `valuestream_chunk_seconds`, `valuestream_chunk_rows`, `valuestream_run_status`, `valuestream_aggregate_size_bytes` |
| Logs | structured JSON per chunk; correlation by `pipeline_run_id` |

---

## 14. Security and Privacy

1. **No raw event row is persisted** — by design, the only PII present at rest is what the configured states explicitly carry (e.g. min/max purchase dates and sketch blobs). Sketches are not a cryptographic anonymization boundary; identifiers should be tokenized or HMACed upstream where required.
2. **No `eval`** — all dynamic expressions are parsed AST. The migration tool flags any expression it can't translate.
3. **Workspace isolation** — variants are isolated workspaces, each with its own `catalog/`, `aggregates/`, `meta/`, `duckdb/` roots.
4. **Headless auth** — the read-only HTTP API supports a bearer token and requires one for non-loopback CLI binds. In-process SDK and local stdio MCP run under the host process identity; OIDC/SSO and remote HTTP MCP remain deferred.
5. **Config provenance** — every aggregate row is tied to a `config_hash` and YAML body in `config_versions`. A reviewer can reconstruct the exact config that produced any number on a dashboard.
6. **Cardinality-sketch caveat** — CPC is the generated distinct-count default and HLL remains supported. Theta also answers distinct count and should be selected when the same persisted state needs intersection/difference. Hashing inside a sketch is not a substitute for governed upstream tokenization.

---

## 15. Phased Implementation Plan

### Phase 0 — Foundations (1–2 weeks)

- Repo skeleton, package layout, `valuestream` CLI entry point.
- JSON Schema for `pipelines.yaml`, `processors.yaml`, `metrics.yaml`, and `dashboards.yaml`.
- AST module for the expression DSL + Polars translator + unit tests.
- `meta/` DuckDB schemas (chunks, runs, config_versions, lineage).

### Phase 1 — Aggregate-first IH pipeline (2–3 weeks)

- Source readers: `pega_ds_export`, `parquet`, `csv`, `xlsx` (parity with current).
- Transforms: `rename_capitalize`, `parse_datetime`, `derive_calendar`, `derive_action_id`, `filter`, `dedup`, `defaults`.
- Processor: `binary_outcome` (covers engagement, conversion, experiment).
- State catalog v1: `count, value_sum, min, max, cpc, hll`.
- Aggregate store layout, daily/monthly/summary compaction, DuckDB views.
- Chunks ledger and idempotent re-runs.
- Python SDK `metric.query` for engagement/conversion.

### Phase 2 — ML & descriptive (2–3 weeks)

- State catalog v2: `tdigest, pooled_mean, pooled_variance, kll`.
- Processor: `numeric_distribution` (descriptive parity).
- Processor: `score_distribution` (model_ml_scores parity).
- Derived metrics: `formula, tdigest_quantile, curve_from_digests, calibration_from_digests`.

### Phase 3 — CLV & funnels (2 weeks)

- Processor: `entity_lifecycle` (CLV parity).
- Processor: `funnel`.
- Derived metrics: `lifecycle_summary` (RFM), `funnel_dropoff`.

### Phase 4 — Streamlit UI (2–3 weeks)

- Dashboard renderer (chart factories + tile resolver).
- Pipelines, Catalog, Ops pages.
- Migration tool (TOML → YAML).

### Phase 5 — Builder UI and Chat MLP1

- Metric/dashboard authoring with live preview against the aggregate store.
- Chat With Data MLP1: LiteLLM JSON intent planner, deterministic chart
  rendering, and local stdio MCP tools over aggregate metrics.
- Implemented read-only HTTP API: health, metric list/query, chart query,
  dimension values, freshness, optional chat, and opt-in governed SQL. Remote
  HTTP MCP, dashboard mutation, source-run triggers, and OIDC service auth
  remain deferred.

### Phase 6 — Snapshots and set algebra (2 weeks)

- Processor: `snapshot` (periodic + accumulating).
- Processor: `entity_set` (Theta sketches for cohort/retention).
- Aggregate-aware planner improvements: pick the cheapest matching grain.

### Phase 7 — Hardening (ongoing)

- Prometheus metrics and structured logs.
- `vacuum` janitor.
- Performance benchmarking against the current app's largest variant (RBB / BDT, ~700 MB DuckDB today).
- Documentation site, examples gallery, workspace-owned recipe libraries, governance, and recipe packs. The built-in shared KPI recipe foundation is implemented.

Total estimated effort to feature-parity with the current app: **12–16 engineering weeks**. Each phase is independently shippable.

---

## 16. Risks and Open Questions

| Risk | Mitigation |
|---|---|
| Migration of `eval` strings is incomplete | Migration tool flags unsupported expressions for hand-conversion; conservative default is to refuse rather than misinterpret |
| Cardinality-sketch parameter mismatches across grains | Standardize sketch type and `lg_k` per state and forbid mixing incompatible sketches at compaction; planner errors out with a clear message |
| t-digest non-associativity edge cases at very small N | Build unit tests at small/large N to confirm AUC/AP reconstruction matches a brute-force check within tolerance |
| One-node ingestion ceiling | Phase 8 adds shard-by-source parallelism (multiple processes against the same workspace, file-level locks on chunk ledger) |
| Streamlit memory growth on long sessions | Move heavy data fetches behind the SDK with aggressive caching by `(metric_id, dim_set, filter_hash, config_hash)`; keep widgets lightweight |
| User dashboards depend on "weak-fit" metrics (sessionization, attribution) | Document explicitly; provide a `snapshot` and `entity_set` processor for the parts that are tractable; punt true sessionization to a separate roadmap item |
| Two configs floating around during migration | Side-by-side strategy in §12.4 plus a banner in the legacy UI pointing to Value Stream |

Open questions:

1. Should `dashboards.yaml` support computed columns at the tile level, or should everything be a metric? Leaning **metric**, to keep aggregate routing predictable.
2. Should `metrics.yaml` support metric inheritance (CTR variants)? Probably yes, with `extends:` keyword — defer to Phase 5.
3. Distinct-count default: CPC with `lg_k=11`. HLL with `lg_k=12` remains supported for backward compatibility and explicit performance trade-offs.
4. Do we need a tdigest-of-tdigests for very deep hierarchies, or is one-pass merging enough? Benchmarks in Phase 2.
5. Should the Builder UI persist YAML in git, or in DuckDB with import/export? Likely git (configs as code) but with a one-click export to share with non-developers.

---

## Appendix A — End-to-end YAML example (engagement only)

```yaml
# catalog/pipelines.yaml
version: 1
workspace: bdt
defaults:
  time_zone: UTC
  calendar:
    grains: [daily, monthly, summary]

sources:
  - id: ih
    reader:
      kind: pega_ds_export
      file_pattern: "**/*.zip"
      group_by_filename: '\d{8}(?=\d{6}_)'
      streaming: true
    schema:
      timestamp_column: OutcomeTime
      natural_key: [InteractionID, ActionID, Rank, Outcome]
    transforms:
      - {kind: rename_capitalize}
      - {kind: parse_datetime, columns: [OutcomeTime, DecisionTime], format: "%Y%m%dT%H%M%S%.3f %Z"}
      - {kind: derive_calendar, from: OutcomeTime, outputs: [Day, Month, Year, Quarter]}
      - {kind: derive_action_id, parts: [Issue, Group, Name], sep: "/"}
      - {kind: filter, expression: {op: not_null, column: Channel}}
      - {kind: dedup, keys: [InteractionID, ActionID, Rank, Outcome]}
    defaults:
      ModelControlGroup: Test
      PlacementType: "N/A"

# catalog/processors.yaml
processors:
  - id: engagement
    source: ih
    kind: binary_outcome
    group_by: [Channel, PlacementType, Issue, Group, CustomerType]
    time:
      column: OutcomeTime
      grains: [Day, Month, Summary]
    outcome:
      column: Outcome
      positive_values: [Clicked]
      negative_values: [Impression, Pending]
    dedup_keys: [InteractionID, ActionID, Rank]
    variant_column: ModelControlGroup
    states:
      Count:               {type: count}
      Positives:           {type: count}
      Negatives:           {type: count}
      UniqueCustomers_cpc: {type: cpc, source_column: CustomerID, lg_k: 11}

# catalog/metrics.yaml
metrics:
  CTR:
    source: engagement
    kind: formula
    expression: {op: safe_div, num: {col: Positives},
                 den: {op: add, args: [{col: Positives},{col: Negatives}]}}

  Lift:
    source: engagement
    kind: variant_compare
    variant_column: ModelControlGroup
    test_role: Test
    control_role: Control
    outputs: [TestCTR, ControlCTR, Lift, Lift_Z_Score, Lift_P_Val, StdErr]

  UniqueCustomers:
    source: engagement
    kind: approx_distinct_count
    state: UniqueCustomers_cpc

# catalog/dashboards.yaml
dashboards:
  - id: marketing_overview
    title: Marketing Overview
    pages:
      - id: engagement
        title: Engagement
        tiles:
          - id: daily_ctr
            title: Daily CTR by group
            metric: CTR
            chart: line
            x: day
            y: CTR
            color: group
            facets: {row: channel, col: placement}
          - id: ctr_treemap
            title: CTR treemap
            metric: CTR
            chart: treemap
            path: [channel, placement, issue, group]
            color: CTR
          - id: unique_customers_kpi
            title: Unique customers (last 30d)
            metric: UniqueCustomers
            chart: gauge
            value: UniqueCustomers
            time_range: {last: 30d}
```

---

## Appendix B — DuckDB DDL for engagement aggregate

```sql
-- daily grain (Parquet, but the same shape as a DuckDB table)
CREATE TABLE engagement_daily (
    Day                 DATE        NOT NULL,
    channel             VARCHAR,
    placement           VARCHAR,
    issue               VARCHAR,
    "group"             VARCHAR,
    customer_type       VARCHAR,
    ModelControlGroup   VARCHAR,
    Count               BIGINT      NOT NULL,
    Positives           BIGINT      NOT NULL,
    Negatives           BIGINT      NOT NULL,
    UniqueCustomers_cpc BLOB,
    pipeline_run_id     UUID        NOT NULL,
    chunk_id            VARCHAR     NOT NULL,
    period              VARCHAR     NOT NULL,        -- YYYY-MM
    created_at          TIMESTAMP   NOT NULL,
    config_hash         VARCHAR     NOT NULL,
    PRIMARY KEY (Day, channel, placement, issue, "group",
                 customer_type, ModelControlGroup,
                 pipeline_run_id, chunk_id)
);

CREATE INDEX engagement_daily_period_idx ON engagement_daily(period);
CREATE INDEX engagement_daily_dim_idx
    ON engagement_daily(channel, placement, issue, "group");

-- view used by the Streamlit UI
CREATE VIEW v_metric_ctr_daily AS
  SELECT
    Day, channel, placement, issue, "group", customer_type,
    ModelControlGroup,
    Positives::DOUBLE / NULLIF(Positives + Negatives, 0) AS CTR,
    Positives, Negatives, Count
  FROM (
    SELECT *,
           ROW_NUMBER() OVER (
             PARTITION BY Day, channel, placement, issue, "group",
                          customer_type, ModelControlGroup, chunk_id
             ORDER BY created_at DESC
           ) AS rn
    FROM engagement_daily
  )
  WHERE rn = 1;     -- pick the latest run per chunk
```

---

## Appendix C — HTTP metric query — request / response example

Request:

```http
POST /api/v1/metrics/CTR/query
Content-Type: application/json

{
  "group_by": ["Day", "Channel", "Group"],
  "filters": {
    "Channel": ["Web", "Mobile"],
    "Group":   ["Cards", "Loans"]
  },
  "time_range": {"from": "2024-08-01", "to": "2024-08-31"},
  "grain": "daily"
}
```

Response:

```json
{
  "rows": [
    {"day": "2024-08-21", "channel": "Web", "group": "Cards",
     "Positives": 184, "Negatives": 12266, "CTR": 0.01479},
    {"day": "2024-08-21", "channel": "Web", "group": "Loans",
     "Positives": 22,  "Negatives": 4011,  "CTR": 0.005456}
  ],
  "plan": {
    "metric": "CTR",
    "processor": "engagement",
    "physical_aggregate": "aggregates/ih/engagement/daily",
    "rows_scanned": 1842,
    "scan_ms": 38
  },
  "freshness": {
    "latest_chunk_id": "2024-08-31",
    "latest_chunk_finished_at": "2024-09-01T03:14:22Z",
    "stale_chunks": []
  },
  "config_hash": "f1c2…"
}
```

---

## Appendix D — Mapping current metrics to Value Stream (cheat sheet)

| Current metric (legacy) | Value Stream processor | Value Stream metric(s) |
|---|---|---|
| `engagement.CTR` | `engagement` (binary_outcome) | `CTR` (formula) |
| `engagement.Lift, Lift_Z_Score, Lift_P_Val` | `engagement` | `Lift` (variant_compare) |
| `conversion.ConversionRate` | `conversion` (binary_outcome) | `ConversionRate` (formula) |
| `conversion.Revenue` | `conversion` | exposed as state column directly |
| `conversion.Touchpoints / AvgTouchpoints` | `conversion` | `AvgTouchpoints` (formula) |
| `model_ml_scores.roc_auc` | `model_ml_scores` (score_distribution) | `ROC_AUC` (curve_from_digests) |
| `model_ml_scores.average_precision` | `model_ml_scores` | `AvgPrecision` (curve_from_digests) |
| `model_ml_scores.personalization, novelty` | `model_ml_scores` | exposed as state columns (pooled_mean) |
| `model_ml_scores.calibration` | `model_ml_scores` | `Calibration` (calibration_from_digests) |
| `descriptive.Count/Sum/Mean/Var/Min/Max` | `descriptive` (numeric_distribution) | exposed as state columns |
| `descriptive.Median/p25/p75/p90/p95` | `descriptive` | per-property `tdigest_quantile` metrics |
| `descriptive.Skew` | `descriptive` | `tdigest_quantile_skew` (formula on quartiles) |
| `experiment.z/g/chi2 stats + odds ratio` | `experiment` (binary_outcome) | `Experiment_Significance` (contingency_test) |
| `clv.recency/frequency/monetary/tenure/lifetime_value` | `clv` (entity_lifecycle) | `CLV_Summary` (lifecycle_summary) |
| `clv.rfm_segment/rfm_score` | `clv` | `CLV_Summary` (lifecycle_summary) |
| (none today) — `DAU / MAU / unique reach` | `entity_set` (CPC + Theta where set algebra is needed) | `UniqueCustomers`, `UniqueCustomers_30d`, `UniqueReach` |
| (none today) — current open subscriptions | `subscription_state` (snapshot, periodic) | `ActiveSubs`, `MRR`, `ChurnedSubs` |

---
