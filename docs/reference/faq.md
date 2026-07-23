# Value Stream — Q&A

A short, opinionated FAQ for engineers and stakeholders. Each entry is one question with one direct answer plus an example. Skim the headings; jump to the entries that match what you're trying to solve.

If a question feels load-bearing and isn't here, open a doc PR — these entries should be the most common questions, not an aspirational catalog.

---

## A. Architecture and storage

**A1. Why aggregates first? Why not just persist raw events and aggregate on read?**
Persisting raw events is expensive (storage, PII risk, retention legislation) and forces every dashboard to do the heavy lifting. Value Stream's premise is that 95% of business questions can be answered from small, mergeable summaries; the other 5% needs a different tool. Once you accept that, throwing away rows after the chunk pass is a feature, not a limitation.

**A2. Why DuckDB tables alongside Parquet — what does each one do?**
Parquet is the resting form of every aggregate (one directory per `source/processor/grain`, hive-partitioned by `period`). DuckDB does three jobs that Parquet alone can't: it serves a SQL surface to ad-hoc users (`read_parquet(...)`), it hosts the small metadata DBs (`chunks`, `pipeline_runs`, `config_versions`, `lineage`), and it provides workspace-level views named identically to logical metrics (so power users can `SELECT * FROM v_metric_ctr_daily` without learning the file layout). Inside the engine, frames are still Polars.

**A3. Why not keep Polars DataFrames in memory and skip persistence?**
Process-bound DataFrames die on Streamlit reload, can't be shared across the API and the UI, and force the system to re-aggregate from raw on every restart. The current app already discovered this and bolted on a DuckDB-backed cache; Value Stream makes the persistence explicit and uniform.

```python
# in-memory only — wrong
@st.cache_data
def load(): return run_pipeline(files)
df = load()  # gone on next deploy
```

```yaml
# Value Stream — right
# aggregates persist as Parquet partitions; the SDK reads them on demand
ws.metric("CTR").by("day", "channel").to_polars()
```

**A4. Are the aggregate tables dynamic?**
Append-only by run, with logical replacement: yes. Each chunk writes an immutable run-specific Parquet partial under its `period` partition; latest-successful-run-wins reads surface the freshest committed one. Schema-mutating without an explicit reprocess: no. The schema is determined by the Processor's computation contract; changing group-by columns or states is a behavior change, not a runtime mutation.

**A5. What happens when I change a Processor's config?**
The planner classifies the change against the existing store (concepts/domain-model.md §5):

- *Compatible widening* (add a derived metric or new processor) → no invalidation.
- *Compatible narrowing* (drop a group-by column, coarsen a grain) → re-compact directly from existing aggregates.
- *Incompatible* (add a group-by column that was not materialized, change a filter, change positive/negative outcomes, change CPC/HLL `lg_k`, switch sketch type) → re-run from chunks.

Aggregate `config_hash` is the processor computation hash: workspace defaults,
source reader/schema/transforms/defaults, and processor semantics. A source
computation hash additionally covers every processor bound to the source and
controls ingestion skip decisions. Presentation-only descriptions, metrics,
and dashboards do not trigger reprocessing. Old and new aggregate hashes
coexist until `valuestream vacuum`, or until a confirmed Data Load clean
rebuild completes successfully for that source scope. Clean rebuild retains
the audit databases under `meta/`.

**A6. How big does the aggregate store get compared to raw input?**
For Pega-shaped IH at typical cardinality, expect the daily aggregate to be ~0.1–1% of the raw row volume; with CPC/t-digest blobs included, plan for a low-single-digit percentage. Actual size depends on cardinality, grouping density, and sketch parameters, so measure it on representative data.

---

## B. Configuration

**B1. Is YAML actually editable by non-developers?**
Yes — the Builder UI generates YAML, validates it, and writes it back. Power users can edit YAML directly in git. The grammar (design/replacement-design.md §7, reference/expression-dsl.md) is small enough to learn in an afternoon.

**B2. How do I add a new metric?**
Edit `metrics.yaml` and run `valuestream validate`.

```yaml
metrics:
  Cost_per_Click:
    source: conversion        # any binary_outcome processor
    kind: formula
    expression:
      op: safe_div
      num: {col: Cost}        # add a `value_aggs` Cost state in the processor first
      den: {col: Positives}
```

**B3. How do I add a new dashboard tile?**
Edit `dashboards.yaml`:

```yaml
- id: cost_per_click
  title: Cost per click
  metric: Cost_per_Click
  chart: line
  x: Day
  y: Cost_per_Click
  color: Channel
```

Save, hit Refresh in Streamlit. No restart needed.

**B4. How do I add a new group-by column?**
Create or normalize the column in `pipelines.yaml` if needed:

```yaml
- kind: derive_column
  output: DeviceType
  expression: {col: RawDeviceType}
```

Then add it to the processors that should partition by it:

```yaml
processors:
  - id: engagement
    group_by: [Channel, PlacementType, DeviceType]
```

Adding a group-by column is an *incompatible* change (A5) — the engine re-runs the affected chunks under a new `config_hash`.

**B5. How do I roll back a bad config?**
Revert the YAML and re-run. The previous `config_hash`'s aggregates are still on disk (until you `vacuum`), so the dashboards immediately serve the old numbers. Any chunks that ran under the bad config are quarantined with `status='partial'` until vacuumed.

**B6. Can I share configs across workspaces?**
Yes — split optional column metadata, `dashboards.yaml`, and the common `metrics.yaml` into a shared module and import via Jinja2 includes:

```yaml
# bdt/catalog/dashboards.yaml
{% include "../shared/dashboards/marketing_overview.yaml" %}
```

Workspace-specific overrides live alongside the include.

---

## C. Ingestion and freshness

**C1. How fresh is the data on a tile?**
Every tile shows the latest `period` covered and the time since the last successful run. Future API responses should return the same freshness object (`plan + freshness + config_hash`).

**C2. How do I trigger an ingestion?**
Three ways:

```bash
# CLI (cron-friendly)
valuestream run --workspace bdt --source ih
```

```python
# SDK
ws.run_source("ih")
```

Both surfaces are idempotent: a chunk is skipped only when the source computation hash and current input-file fingerprint match a successful prior run. Pass `--force` to re-process. The read-only HTTP API does not trigger ingestion.

**C3. What happens when the same chunk is re-run?**
The new partial Parquet writes alongside the old one with a new `pipeline_run_id`. Complete file lineage commits first, then the successful chunk row acts as the chunk commit marker. It becomes visible only after the parent run is finalized as successful/partial. If replacement fails, readers keep the previous successful partial. `valuestream vacuum` deletes superseded or orphaned partials.

**C4. What if a file is corrupted mid-run?**
That chunk fails (`status='failed'` in `meta/chunks.duckdb`). The other chunks finish. The run status becomes `partial`. The operator fixes or removes the bad file and re-runs; only failed/missing chunks are processed.

**C4a. What if the process or machine stops mid-run?**
The durable run row remains `running`, so none of its chunks are published by
that row yet. Start the same source normally. After taking the source lock, the
engine verifies each committed chunk's current fingerprint, computation hashes,
lineage, physical files, and embedded provenance. Verified chunks are published
under a recovered `partial` run and reused; incomplete or changed chunks are
processed again. `--force` would intentionally disable that reuse.

**C5. Can multiple ingestions run at the same time?**
Per-source: no — a filesystem advisory lock at `meta/source_<id>.lock` enforces one run per source. Different sources can run in parallel. Inside one run, processors fan out concurrently.

**C6. Late-arriving files for a past chunk?**
Drop them into the source folder and re-run. The `chunk_id` extraction puts them into the right chunk; the engine notices the chunk's `file_hash` changed and re-processes it.

**C7. What about back-filling a new processor against old chunks?**
`valuestream backfill --workspace bdt --source ih --processor engagement_v2 --from 2024-01-01`. The chunks for that range are re-read and only the new processor runs against them. Existing processors' aggregates are untouched.

---

## D. Querying and metrics

**D1. How does the planner pick a physical aggregate?**
Three rules: (1) the grain must be ≥ the requested grain, (2) the group-by column set must include all requested report fields, (3) the states must cover the metric's needs. The smallest matching aggregate wins. If no aggregate matches, the planner errors with the specific gap (e.g., "no monthly aggregate covers `DeviceType`").

**D2. Why are the same numbers slightly different across grains?**
For exact metrics (count-based, sum-based, pooled mean/variance) — they're identical to floating-point precision. For sketch-based metrics (CPC/HLL/Theta distinct, t-digest quantiles) daily and monthly answers can differ within the sketch error bound. CPC exposes lower/upper bounds directly; legacy HLL `lg_k=12` has approximately ±1.6% RSE. The UI should surface the applicable bound and sketch parameters.

**D3. Can I get exact distinct counts?**
Not at the aggregate-store layer. Value Stream's contract is "no raw rows," so distincts use CPC by default, HLL by explicit configuration, or Theta when set algebra is required. A snapshot can preserve one aggregate row per entity when its grain permits an exact query, but a CPC/HLL/Theta state remains approximate.

**D4. How do I handle joins?**
Two patterns. (1) Fold the foreign key into the source schema upstream or with a transform so it becomes a group-by column. (2) Use snapshot processors with shared group-by columns, then join in the metric DSL via `join` (planned post-MVP). Value Stream deliberately makes ad-hoc joins inconvenient because they undermine pre-aggregation.

**D5. Can I query Value Stream with SQL?**
Yes. There are two SQL surfaces:

- `meta/aggregate_views.duckdb` contains views over the canonical aggregate
  Parquet files, one per source/processor/grain. These expose aggregate state,
  not raw data.
- `valuestream export-duckdb <workspace> --grain Summary` creates a materialized
  DuckDB file with one table per metric at the selected grain. Tables are named
  `metric_<metric_id>_<grain>` and are better suited for Superset or similar BI
  tools because formula and sketch-derived metrics have already been evaluated.
- `valuestream serve-api <workspace> --enable-sql` and
  `valuestream serve-mcp <workspace> --enable-sql` expose the same governed
  views only when SQL is explicitly enabled. DuckDB is locked to allowlisted
  aggregate/export paths; external file access and extension loading are
  disabled before user SQL runs.

**D6. How do I A/B test?**
Use a `binary_outcome` processor with `variant_column` set, a `variant_compare` metric for lift, and a `contingency_test` metric for significance:

```yaml
processors:
  - id: engagement
    variant_column: ModelControlGroup
    variant_role_map: {Test: Test, Control: Control}

metrics:
  Lift:
    source: engagement
    kind: variant_compare
    variant_column: ModelControlGroup
    test_role: Test
    control_role: Control
  Lift_Significance:
    source: engagement
    kind: contingency_test
    tests: [chi2, g, z]
```

A tile with `metric: Lift` gets `TestCTR, ControlCTR, Lift, Lift_Z_Score, Lift_P_Val, StdErr` per group. reference/algorithms.md §3 has the math.

**D7. How do I do a multi-variant experiment?**
Use the `experiment` flavor of `binary_outcome`: include `ExperimentName` and `ExperimentGroup` in `group_by`, set `variant_column = ExperimentGroup`, and add a `filter` restricting to in-experiment rows. The `Experiment_Significance` metric runs chi-square, G-test, and z-test on the contingency.

**D8. How do percentiles work for descriptive stats?**
Each numeric property gets a t-digest state (`<prop>_tdigest`). Percentiles are derived metrics:

```yaml
metrics:
  Median_ResponseTime:
    source: descriptive
    kind: tdigest_quantile
    state: ResponseTime_tdigest
    quantile: 0.5
  p95_ResponseTime:
    source: descriptive
    kind: tdigest_quantile
    state: ResponseTime_tdigest
    quantile: 0.95
```

**D9. How do I see ROC AUC over time?**
A `score_distribution` processor at `Day` grain stores `Propensity_tdigest_positives` and `Propensity_tdigest_negatives` per day for the selected score property. The `ROC_AUC` metric (kind `curve_from_digests`) reconstructs AUC at query time. A `line` chart with `x: Day, y: ROC_AUC, color: Channel` plots the trend.

For the actual model curves, use the same `curve_from_digests` metric with one
of the curve chart kinds:

```yaml
- id: roc_curve
  title: ROC Curve
  metric: ROC_AUC
  chart: roc_curve
  color: Channel
  value_format: percent
```

`precision_recall_curve` plots the average-precision curve, while `gain_curve`
and `lift_curve` derive gain/lift from the reconstructed `tpr`, `fpr`, and
`pos_fraction` arrays.

---

## E. Operations

**E1. How do I deploy?**
One process per workspace.

```bash
valuestream serve --workspace bdt
# starts Streamlit :8501 for the current product
# optional: local stdio MCP through valuestream serve-mcp
# optional: read-only FastAPI through valuestream serve-api
# deferred: remote HTTP MCP and OIDC/multi-user deployment
# pointed at /data/valuestream/bdt
```

A reverse proxy routes by hostname for multiple workspaces.

**E2. How do I do disaster recovery?**
The aggregate store + metadata is everything. Tar `<workspace>/`, ship to safe storage, untar to restore.

```bash
tar -C /data/valuestream -czf /backup/bdt-$(date +%F).tar.gz bdt/
```

**E3. How do I monitor?**
The Streamlit *Ops* page shows freshness and recent runs, and logs are structured JSON. Prometheus `/metrics` is reserved for the deferred service surface.

**E4. How big a machine do I need?**
Indicative numbers (Polars + DuckDB on a single 8-core / 32 GB host):

| Workspace size | Raw input / day | Chunk processing time |
|---|---|---|
| Small (Demo) | < 1 GB | seconds |
| Medium (NBS) | 5–20 GB | minutes |
| Large (BDT, RBB) | 50–200 GB | tens of minutes |

`streaming: true` reduces transient source-plan memory, but it does not impose a
fixed RSS bound and a materialized transformed chunk must still fit in memory.
Size the machine with the ingestion benchmark, reduce the chunk interval when
needed, and increase chunk-process parallelism only after measuring peak RSS.

**E5. Can I run Value Stream against a remote object store?**
Yes — the workspace path can be `s3://...` or `gs://...`. Polars and DuckDB read Parquet from object stores natively. The metadata DBs need to be local (DuckDB's WAL doesn't live well on S3) — typically mounted from a local volume.

**E6. How do I migrate from the current app?**
`valuestream migrate --from value_dashboard/config/<variant>.toml --to valuestream/<variant>/catalog/`. The migration tool translates legacy TOML, identifies expressions that need hand-conversion (and refuses to silently drop them), and emits a `migration_report.md`. Then `valuestream backfill` re-uses the existing DuckDB tables to populate the new aggregate store without recomputing from raw.

**E7. Can the legacy app and Value Stream coexist during migration?**
Yes — they read from disjoint locations (`db/pov_data_<variant>.duckdb` vs `<workspace>/aggregates/`). Run them side-by-side; pin a banner in the legacy UI pointing to Value Stream.

---

## F. Security and compliance

**F1. Is PII safe?**
Raw rows never leave a chunk's process. Only state columns are persisted: counts, sums, sketches. CPC/HLL/Theta store hashed sketch state, but sketches are not a cryptographic anonymization boundary; tokenize or HMAC identifiers upstream where required. There is no `eval`, so config injection isn't a vector. Per-row provenance lets an auditor reconstruct the exact config that produced any number.

**F2. How does authentication work?**
Current product: SDK and local stdio MCP run under the host process identity; UI auth is handled by Streamlit/basic auth or an upstream proxy; per-workspace filesystem permissions enforce hard isolation. The read-only HTTP API supports a single bearer token and the CLI refuses non-loopback binding without one. Remote HTTP MCP and OIDC/multi-user auth remain deferred.

**F3. How do I redact a customer?**
Drop the chunk that contained the customer's events and re-run the pipeline; the customer's contribution to all aggregates is removed because state types are associative. CPC/HLL/Theta states must be rebuilt from the remaining chunks; scalar states are recomputed through the same deterministic replay.

**F4. Can I disable a sketch type for compliance reasons?**
Yes — `processors.<id>.states` is explicit, so a workspace can omit any `cpc`, `hll`, or `theta` state. The corresponding metrics simply aren't available; downstream tiles that depend on them error out at validate time, not at runtime.

---

## G. Performance and limits

**G1. What's the chunk size sweet spot?**
For Pega IH, 1 chunk per day works well. Smaller (per hour) chunks pay extra fixed cost; larger (per month) chunks bloat memory. The grouping regex is the lever.

**G2. What's the largest chunk Value Stream can handle?**
There is no input-size-only limit: in-memory size depends on selected columns,
data types, transform expansion, group cardinality, processor states, and active
chunk workers. Streaming reduces transient scan/transform memory, while
`materialize_transforms: true` retains one transformed chunk through processor
fan-out. Measure the actual catalog with the ingestion benchmark; if peak RSS is
too high, split the grouping interval (for example, daily to hourly) and lower
chunk-process parallelism.

**G3. What's the largest aggregate store Value Stream can serve?**
DuckDB scans Parquet at multi-GB/s on local SSD. Aggregate stores up to a few hundred GB serve interactive dashboards comfortably; beyond that, increase the `monthly` grain's coverage and rely on summary aggregates for high-cardinality tiles.

**G4. Why is my dashboard slow?**
Three usual suspects:

1. The metric resolves to `daily` when `monthly` would do — make sure the
   processor publishes a `monthly` grain and use the page date selector to
   request a range that can use it.
2. The metric pulls a big sketch column (`<prop>_tdigest` ~32 KB) for many groups — fix by adding a more selective `filters` block.
3. A facet has too many categories — fix by reducing `facet_row` cardinality or pre-aggregating to `monthly`.

`query_metric_result`, SDK `to_result()`, and API/MCP metric queries identify the selected physical aggregate grain, catalog/computation hashes, contributing runs/chunks, scanned aggregate-row count, and latest aggregate timestamp.

**G5. How does Value Stream handle high-cardinality group-by columns?**
A processor with cardinality > 1M tuples per day at the `Day` grain is expensive. Recommendations: (1) bin the group-by column upstream (e.g., truncate timestamps), (2) drop the column from `Day` and keep it only at `Month`, (3) replace it with a CPC distinct-count state if the question is "how many."

---

## H. Roadmap and limits

**H1. What's not in v1?**
Listed explicitly in concepts/architecture.md §20: streaming/CDC, distributed compute, raw-event SQL, feature serving, full CDP. Anything outside the boundary is the upstream system's job.

**H2. Will Value Stream ever do streaming?**
Maybe — incremental view maintenance is on the long-term roadmap, gated on real demand for sub-minute freshness.

**H3. Will the engine support Spark / Ray / Dask backends?**
No — Value Stream is single-node by design. Horizontal scaling is "shard by source" via multiple processes against one workspace; the file-based store + advisory locks make this safe.

**H4. Is Value Stream open source?**
TBD — the design assumes an internal release first; nothing in the architecture prevents an open-source release later.

**H5. What if I need a metric that doesn't fit any built-in processor?**
Two options. (1) Plug in a custom processor — implement the `Processor` protocol and register the kind. (2) Pre-aggregate upstream and ingest the result as a `snapshot` source — i.e. delegate the difficult part to a system that has raw rows.

---

## I. Migration cookbook

**I1. Translating legacy `[metrics.engagement]` to Value Stream.**
Before:

```toml
[metrics.engagement]
group_by = ['Day','Month','Year','Quarter','Channel','PlacementType','PropensitySource','Issue','Group']
filter = """"""
scores = ['CTR','Lift','Lift_Z_Score','Lift_P_Val','Positives','Negatives','Count']
positive_model_response = ['Clicked']
negative_model_response = ['Impression','Pending']
```

After (minus sugar — full form in design/replacement-design.md):

```yaml
processors:
  - id: engagement
    source: ih
    kind: binary_outcome
    group_by: [Channel, PlacementType, PropensitySource, Issue, Group]
    time:
      column: OutcomeTime
      grains: [Day, Month, Summary]
    outcome:
      column: Outcome
      positive_values: [Clicked]
      negative_values: [Impression, Pending]
    variant_column: ModelControlGroup
    variant_role_map: {Test: Test, Control: Control}
    states:
      Count:     {type: count}
      Positives: {type: count}
      Negatives: {type: count}

metrics:
  CTR:  {source: engagement, kind: formula, expression: {op: safe_div, num: {col: Positives}, den: {op: add, args: [{col: Positives}, {col: Negatives}]}}}
  Lift: {source: engagement, kind: variant_compare}
```

**I2. Translating a `pl.col(...).is_in(...)` filter to AST.**
Before: `filter = "(pl.col(\"ModelControlGroup\").is_in([\"Test\",\"Control\"]))"`
After:

```yaml
filter:
  op: in
  column: ModelControlGroup
  values: [Test, Control]
```

**I3. Translating a derived column.**
Before: `columns = "[pl.when(pl.col('ConversionEventID') != '').then(pl.col('ConversionEventID')).otherwise(pl.col('Name')).alias('ConversionEventID')]"`
After:

```yaml
- kind: derive_column
  output: ConversionEventID
  expression:
    op: case
    when:
      - cond:  {op: ne, column: ConversionEventID, value: ""}
        then:  {col: ConversionEventID}
    else:      {col: Name}
```

---

## J. When the docs don't cover it

If you hit a gap:

1. Search this Q&A for the closest entry.
2. Read concepts/domain-model.md for vocabulary.
3. Read reference/processors.md for the relevant processor.
4. Read reference/algorithms.md for the math.
5. Read design/replacement-design.md for the broader DSL.
6. If still stuck, file a doc bug — the answer should be in here.
