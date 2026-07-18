# Value Stream — Domain Model and Glossary

This document fixes the vocabulary used everywhere else in Value Stream docs and code. Every term that has a specific meaning in the platform appears here exactly once, with a definition, key relationships, and a small example.

A reader who works through this document and the reference/expression-dsl.md grammar should be able to read every other Value Stream doc unambiguously.

---

## 1. Big picture

```
            Workspace
               |
               v
   +-------------------------------+
   |   Catalog (YAML)              |
   |   - Sources                   |
   |   - Dimensions                |
   |   - Processors                |
   |   - Metrics                   |
   |   - Dashboards                |
   +-------------------------------+
               |
               | drives
               v
   +-------------------------------+
   |   Ingestion                   |
   |   - File   -> Chunk          |
   |   - Chunk  -> Partial agg.    |
   |   - Partial-> Compacted agg.  |
   +-------------------------------+
               |
               | persisted in
               v
   +-------------------------------+
   |   Aggregate Store             |
   |   (Parquet, hive-partitioned) |
   +-------------------------------+
               |
               | read by
               v
   +-------------------------------+
   |   Query Layer                 |
   |   - Resolve metric -> processor + states
   |   - Plan -> physical aggregate
   |   - Execute -> Polars frame
   |   - Apply derive (DSL)
   +-------------------------------+
               |
               | rendered as
               v
   +-------------------------------+
   |   Surfaces                    |
   |   - Tile in Dashboard         |
   |   - REST response             |
   |   - SDK return value          |
   +-------------------------------+
```

## 2. Term graph

Lines mean "is composed of" or "refers to."

```
Workspace ─── Catalog ─── { Source, Dimension, Processor, Metric, Dashboard }
                           │
                           ▼
                     Source ── Reader ── ChunkPolicy
                          \─── Transform[]
                          \─── Defaults
                          \─── Schema (timestamp_col, natural_key, ...)
                           │
                           ▼
                     Processor ── Source (FK)
                                \── Dimension[] (FK)
                                \── Grain[]
                                \── State[]
                                \── (kind-specific) outcome / variant / properties / keys
                                │
                                ▼
                            Metric ── Processor (FK)
                                  \── kind ∈ { formula, approx_distinct_count,
                                              tdigest_quantile, variant_compare,
                                              proportion_test, contingency_test,
                                              curve_from_digests,
                                              calibration_from_digests,
                                              lifecycle_summary }
                                  \── Expression (AST)
                                  \── outputs[]
                                  │
                                  ▼
                              Tile ── Metric (FK)
                                  \── Chart (kind)
                                  \── x / y / color / facets / etc.
                                  │
                                  ▼
                          DashboardPage
                                  │
                                  ▼
                            Dashboard
```

## 3. Definitions (alphabetical)

### Aggregate
A row in a physical aggregate table, produced by a Processor. It contains group-by column values, state values, and provenance columns (`pipeline_run_id`, `chunk_id`, `period`, `created_at`, `config_hash`).

Aggregates are **never raw rows.** A row in `aggregates/ih/engagement/daily/period=2024-08/part-00.parquet` represents one `(Day, Channel, PlacementType, Issue, Group, CustomerType, ModelControlGroup)` tuple — never a single Pega interaction.

### Aggregate Store
The directory tree under `<workspace>/aggregates/` holding all physical aggregate Parquet files. The only place persisted business data lives.

### AST (Abstract Syntax Tree)
The internal representation of an expression in the closed expression DSL. AST nodes are JSON-shaped (e.g. `{op: safe_div, num: {col: P}, den: {col: N}}`). The evaluator turns an AST into a Polars expression. See reference/expression-dsl.md.

### Backfill
A one-shot operation that re-runs a Source's chunks within a time window, typically after a compatible-narrowing config change or to populate a new processor against historical chunks.

### Calendar
A built-in time-grain helper. The default calendar emits `Day, Month (YYYY-MM), Year (Int16), Quarter (YYYY_Q#)` from a timestamp column. Custom calendars can add `Week (ISO YYYY-Www)` or `FiscalQuarter`.

### Catalog
The set of YAML files in `<workspace>/catalog/` that define everything declarative about the workspace: `pipelines.yaml`, `processors.yaml`, `metrics.yaml`, and `dashboards.yaml`. Loaded once at startup, validated, and hashed into a single `config_hash`.

### Chart
A presentation-layer artifact bound to a Tile. Each chart has a `kind` (`line, stacked_area, bar, kpi_card, waterfall, pareto, treemap, heatmap, cohort_heatmap, scatter, combo, interval, donut, geo_map, table, calendar_heatmap, bar_polar, sankey, gauge, funnel, boxplot, histogram, calibration_curve, roc_curve, precision_recall_curve, gain_curve, lift_curve, rfm_density, exposure, corr, model`) and required Tile fields (`x, y, color, facets, …`). See reference/chart-catalog.md.

### Chunk
The unit of idempotent work in the ingestion engine. A chunk consists of:
- a `chunk_id` (typically derived from a filename pattern, e.g. `2024-08-21`),
- one or more file paths whose contents combine to form the chunk's input,
- a `pipeline_run_id` recording which run processed it,
- a durable ledger status (`ok | failed`).

A chunk is processed end-to-end (read → transform → fan out to all processors → write all partials) before the next chunk starts.
`skipped` is a result in the current run report when a prior committed chunk is
reused; it does not create another durable chunk-ledger row. For a successful
chunk, complete lineage commits before `status=ok`, making that row the durable
chunk commit marker.

### Chunk ledger
The DuckDB table at `meta/chunks.duckdb` recording every chunk processed by every run.
Only an `ok` row whose parent run is `ok` or `partial` is query-visible and
eligible for idempotent reuse.

### Compaction
The process of reducing finer-grained aggregates into coarser-grained aggregates by `groupby + state-merge`. Compaction is a pure file operation and never touches raw data. Standard chain: `Day → Month → Summary`, stored internally as `daily → monthly → summary`.

### Config hash
The sha256 of the canonicalized YAML for a Processor (or for the whole catalog, depending on context). Recorded on every aggregate row and returned by query surfaces where available. Used to:
- detect when an aggregate was produced under a different config,
- partition the aggregate store across coexisting configs,
- reproduce or audit results.

### Derived metric
A metric whose value is computed at query time from a Processor's persisted state. Examples: `CTR = Positives / (Positives + Negatives)`, `ROC_AUC = curve_from_digests(Propensity_tdigest_positives, Propensity_tdigest_negatives)`. Derived metrics are *not* stored; they live only in the metric DSL.

### Group-by column
A transformed source column that a Processor preserves as a business grouping/filtering axis. Processors declare these directly with `group_by: [Channel, PlacementType]`; if a user-facing name should differ from the incoming source field, create that column in the pipeline transforms and use the transformed column name here.

### Discovery
The phase of a pipeline run where the engine globs the Source's input folder, applies `group_by_filename` to assign each file to a `chunk_id`, and consults the chunk ledger to skip already-processed chunks.

### Expression
A node in the AST that evaluates to a column or a scalar at query time. Used in: `transforms[*].filter`, `transforms[*].derive_column`, `processors.<id>.filter`, and `metrics.<id>.expression` (when `kind: formula`).

### Freshness
The minimum gap between "the latest available chunk from upstream" and "the latest aggregate row served by the query layer," for a given `(source, processor, grain)`. Surfaced on every dashboard tile and in the `/api/v1/workspace` response.

### Grain
A time resolution at which a Processor materializes aggregates. User-facing configs use calendar names such as `Day`, `Month`, `Quarter`, `Year`, and `Summary`; the storage layer normalizes currently implemented grains to `daily`, `monthly`, and `summary`.

### CPC state
A binary state (Apache DataSketches CPC sketch) used by default to estimate
distinct counts. Generated states use `lg_k=11`; merge rule: union. CPC exposes
estimate bounds but does not support intersection or difference. See
reference/algorithms.md.

### HLL state
A binary state (Apache DataSketches HLL_4 or HLL_8 sketch) retained for
backward compatibility and opt-in distinct counts. Merge rule: union. See
reference/algorithms.md.

### Ingestion engine
The component that turns files into aggregates. Implemented as a sequence of phases: discovery → reader → transforms → processor fan-out → partial write → compaction → ledger update.

### KLL state
A binary state (Apache DataSketches KLL sketch) used for quantile estimation with strong error guarantees. Merge rule: KLL merge. Alternative to t-digest where guaranteed error matters.

### Lineage
The mapping from each aggregate row to the chunks and config that produced it. Stored in `meta/lineage.duckdb`. Surfaced in the Ops UI.

### Mergeable state
A state column whose values can be combined under a deterministic, associative-commutative rule across rows. The state catalog defines:
- `count` → SUM
- `value_sum` → SUM
- `min`, `max` → MIN/MAX
- `pooled_mean` → SUM-of-(value*count) / SUM-of-count
- `pooled_variance` → Welford-merge
- `tdigest`, `kll` → sketch merge
- `cpc`, `hll`, `theta` → sketch union (intersect/diff for theta only)
- `topk` → frequent-items merge

### Metric
A *derived* business measurement bound to a Processor. A metric has a `kind` and inputs from the Processor's state. Examples: `CTR (formula)`, `ROC_AUC (curve_from_digests)`, `Lift (variant_compare)`, `Median_ResponseTime (tdigest_quantile)`, `UniqueCustomers (approx_distinct_count)`, `Experiment_Significance (contingency_test)`, `CLV_Summary (lifecycle_summary)`.

Every metric may also carry `display` metadata: a friendly `label`, `unit`,
default `value_format`, and `direction` (`higher_is_better`,
`lower_is_better`, or `neutral`). Display metadata never changes the formula,
processor computation hash, or stored aggregate state.

A metric installed from a KPI Recipe may carry `recipe: {id, version}`
provenance. The installed metric remains authoritative and does not track
future recipe edits automatically.

### Partial
A Parquet file produced by one Processor processing one chunk. Lives at `aggregates/<src>/<proc>/daily/period=YYYY-MM/part-<pipeline_run_id>.parquet`. A "partial" is conceptually the smallest unit of write; compaction merges multiple partials into one.

### Period
The hive partition key used inside an aggregate directory. For `daily`, `monthly`, and the default `summary` materialization, `period = YYYY-MM`. `time.aggregation_levels` can coarsen summary storage to quarter or year. For optional grains the convention is the smallest "obviously coarser" string (`weekly_iso → YYYY-MM-Www`, `hourly → YYYY-MM-DD`).

### Pipeline run
One end-to-end execution of an ingestion against a Source. Has an id (UUID), a config hash, a status (`running | ok | failed | partial`), counts of total / ok / failed chunks, and start/end timestamps. Recorded in `meta/pipeline_runs.duckdb`.
The row is inserted as `running` after discovery and before the first chunk.
While it is running, `finished_at` is null and none of that run's new chunks are
visible. Normal completion transitions the same row to `ok`, `partial`, or
`failed`. If the process dies first, the next caller holding the source lock
verifies committed chunks and performs the terminal transition; it never
creates a second row for the interrupted run.
Run-level `rows_in` and `rows_kept` totals include only chunks whose durable
marker finishes as `ok`; failed, recovery-rejected, and skipped chunks do not
inflate the published-row totals shown by operational surfaces.

### Plan
The result of the Query Layer's planner: which physical aggregate to scan, what predicates to push down, what columns to read, what derive function to apply. Deterministic for a given input + config_hash.

### Processor
The most important concept in Value Stream. A Processor is a typed function `(Source, config) → Aggregate(s) at one or more Grains`. Built-in kinds:
- `binary_outcome` — counts and rates from positive/negative outcomes (engagement, conversion, experiments)
- `numeric_distribution` — moments, min/max, t-digest of numeric properties (descriptive)
- `score_distribution` — score-vs-outcome distributions, ROC/AP/calibration (model_ml_scores)
- `entity_lifecycle` — order/holding aggregates per entity for CLV/RFM
- `entity_set` — CPC/HLL/Theta distinct-count sketches, with Theta also supporting set algebra
- `funnel` — per-stage counts for funnel KPIs
- `snapshot` — periodic / accumulating snapshots for state KPIs

Processors implement the interface in reference/processors.md §1.

### Reader
A built-in component that turns a list of file paths into a Polars LazyFrame. Built-in readers:
- `parquet`
- `pega_ds_export` (zip/tar.gz/gz containing JSON or NDJSON)
- `csv` (with delimiter auto-detection)
- `xlsx`

See reference/readers-and-formats.md.

### KPI Recipe
A versioned, inert authoring artifact that connects a business question to
required Processor capabilities, a metric template, calculation/method
guidance, and recommended report presentation. It is not part of a workspace
Catalog and cannot execute. An explicit install maps its roles to persisted
aggregate states/stages and materializes ordinary Metric and optional Tile
YAML. See reference/kpi-recipes.md.

### Run
Shorthand for a pipeline run.

### Snapshot
A row representing the *state* of a business entity at a given `as_of_date`. Two flavors:
- `periodic`: written daily/weekly/monthly; one row per `(as_of, dim_tuple)`.
- `accumulating`: one row per business-entity, updated through milestones.

Snapshots live under `<workspace>/snapshots/<id>/as_of=YYYY-MM-DD/snapshot.parquet`.

### Source
A logical input to Value Stream. Each Source has an `id`, a `description`, a `reader`, a `transforms` list, a `defaults` map, and a `schema` block (timestamp column, natural key, columns to drop). Multiple Processors can be bound to the same Source.

Concrete Sources in the canonical workspace: `ih` (Pega CDH Interaction History), `holdings` (Product Holdings), and any number of optional Sources (`subscriptions`, `tickets`, …) for snapshot processors.

### State
A column on a Processor's aggregate output, with a declared **state type**. State types determine the merge rule. Examples: `Count {type: count}`, `Propensity_tdigest_positives {type: tdigest}`, `UniqueCustomers_cpc {type: cpc, lg_k: 11}`.

### State type
The catalog of mergeable kinds: `count, value_sum, min, max, pooled_mean, pooled_variance, tdigest, kll, cpc, hll, theta, topk`. See §3 of design/replacement-design.md and reference/algorithms.md.

### t-digest state
A binary state holding a serialized `datasketches.tdigest_double` (compression `k=500` by default). Used for quantiles, ROC/AP, calibration. Merge rule: deserialize-merge-reserialize.

### Theta state
A binary state holding a Theta sketch from Apache DataSketches. Used when set algebra (intersection / difference) is required, e.g. cohort retention, audience overlap.

### Tile
A single chart on a Dashboard page, bound to one metric and one chart kind. Required fields depend on the chart kind (e.g. `line` requires `x, y`; `treemap` requires `path, color`). Typed common fields include `description`, `placement`, `kpi`, `scale_mode`, and `value_format`; chart-specific extras remain permissive. `placement: kpi_strip` is valid only for an ungrouped `kpi_card`. A KPI query must resolve to one numeric scalar and can request a target, an equal-length previous period, and an aggregate time-series sparkline. `scale_mode` is presentation-only and supports `absolute`, `index_100`, and `percent_change` for line/stacked-area charts.

### Page filter
An interactive Reports-page control authored under `page.filters`, with a safe
inference fallback for older catalogs. Each filter declares its aggregate field,
label, primary/secondary placement, control type, and either `all_tiles` or
`compatible_tiles` scope. Validation proves the declared coverage against each
tile's processor. Partial filters remain usable, but the Reports UI identifies
unsupported tiles instead of silently skipping the filter. Adding a filter over
an already-persisted `group_by` column is presentation-compatible; adding a new
processor `group_by` column still requires raw replay.

### Time filter
A page-level list of supported date presets plus one default preset. Time
selection becomes query-layer `start`/`end` bounds and never bypasses aggregate
routing. `all_time` is the backward-compatible default.

### Time grain
The Processor-level storage contract declared under `time.grains`. Calendar columns like `Day`, `Month`, `Quarter`, and `Year` are generated by transforms and can be used in reports, but they are not ordinary business group-by columns.

### Transform
A typed step in the Source's `transforms` list. Each transform takes a Polars LazyFrame and returns a LazyFrame. Built-in transforms: `rename_capitalize, parse_datetime, derive_calendar, derive_action_id, filter, dedup, defaults, derive_column, cast, drop_columns`. Transforms operate **before** processors fan out, so they affect every Processor bound to the Source.

### Variant
Two distinct meanings; context disambiguates:
1. **A/B test variant** — the `Test` / `Control` arm of an experiment, defined by the Processor's `variant_column` and `variant_role_map`.
2. **Workspace variant** — a separate workspace for a different deployment (e.g. `BDT`, `RBB`, `NBS`, `Demo`). Each variant is its own filesystem root and its own catalog.

### Workspace
A self-contained Value Stream environment. Has its own:
- `catalog/` (YAML),
- `aggregates/` (Parquet),
- `snapshots/` (Parquet),
- `meta/` (DuckDB metadata),
- `duckdb/valuestream.duckdb` (views).

A workspace is the unit of deployment, security boundary, and backup.

---

## 4. Identity rules

| Identifier | Where | Format | Example |
|---|---|---|---|
| `workspace` | top-level YAML | `[a-z][a-z0-9_-]+` | `bdt` |
| `source.id` | `pipelines.yaml` | snake_case | `ih`, `holdings` |
| `group_by` column | `processors.yaml` | transformed column name | `Channel`, `CustomerType` |
| `processor.id` | `processors.yaml` | snake_case; recommended same as legacy family for migration | `engagement`, `model_ml_scores` |
| `metric key` | `metrics.yaml` | PascalCase or snake_case, free | `CTR`, `ROC_AUC`, `CLV_Summary` |
| `dashboard.id`, `page.id`, `tile.id` | `dashboards.yaml` | snake_case | `marketing_overview`, `engagement`, `daily_ctr` |
| `chunk_id` | derived from filename | string | `2024-08-21` |
| `pipeline_run_id` | UUIDv4 | hex | `8a32fa3d-…` |
| `config_hash` | sha256 of canonical YAML | 64-char hex | `f1c2…` |

Reserved IDs (do not use as user identifiers): `__all__` (planner placeholder), `_manifest` (filesystem reserved), `period`, `pipeline_run_id`, `chunk_id`, `created_at`, `config_hash`.

## 5. Compatibility classes for config changes

When the Catalog changes, the planner classifies the change against the existing aggregate store:

| Class | Examples | Behavior |
|---|---|---|
| **Compatible widening** | Add a new Processor; add a new Metric; add a new finer Grain | New aggregates are populated forward; optional backfill; nothing invalidated |
| **Compatible narrowing** | Remove a group-by column; coarsen a Grain; remove a State | Existing aggregates can be re-compacted directly; no raw replay needed |
| **Incompatible** | Add a group-by column; change a filter; change `positive_values`; change CPC/HLL `lg_k`; switch sketch type | Re-process from raw; old aggregates remain readable under their `config_hash` until vacuumed |
| **Catalog removal** | Remove a Source from the Builder | The confirmed cascade removes its Processors, transitive dependent Metrics, bound Tiles, unsupported page filters, and related Chat descriptions from configuration. Persisted aggregates and run audit history remain untouched by the catalog edit. |

Metric display metadata, dashboard/page/tile presentation, KPI comparison
settings, and filters over existing aggregate dimensions are compatible changes
and do not require reprocessing.

Adding a State to an existing Processor changes its computation hash. Queries
under the updated Catalog read only aggregate partials carrying that new hash;
older files remain available for the older Catalog version but are never mixed
with the new schema. Until ingestion publishes the new contract, report tiles
show that a first run or backfill is required. Optional backfill for a widening
therefore refers to historical coverage, not permission to derive a missing
state from old aggregate files.

### Authoring revision (session-local)

An authoring revision is a UI lifecycle object, not another catalog entity and
not a durable raw-data store. It consists of a canonical proposed catalog
object or multi-file bundle, its revision digest, one validation verdict for
that digest, and explicit review state. Editing changes the digest and clears
the prior verdict/review. Applying a reviewed revision produces ordinary YAML
catalog files and a new catalog hash inside the catalog transaction.

| Displayed state | Domain evidence | Permitted transition |
|---|---|---|
| Editing draft | Proposed canonical object differs from applied YAML | Validate or discard |
| Ready for review | Revision-keyed validation has zero blocking issues | Review dependency-closed changes |
| Reviewed | User accepted a valid dependency-consistent set for that revision | Apply transactionally |
| Applied | Catalog write and post-write validation succeeded | Classify materialization impact |
| Data refresh required | Source/processor computation contract changed | Explicit handoff to Data Load |
| Report ready | Existing aggregates satisfy the applied catalog | Open Reports through the query layer |

Uploaded Studio samples and their derived preview frames remain session-local
inputs to authoring. They do not become production source files or persisted
aggregates merely because a draft references their schema.

The migration tool (`valuestream migrate`) classifies each change automatically and reports the action required.

## 6. Examples

### 6.1 Source → Chunk → Aggregate

```
Files in folder:
  /data/ih/2024-08-21/interaction-XYZ-20240821000000_001.json.zip
  /data/ih/2024-08-21/interaction-XYZ-20240821000000_002.json.zip
  /data/ih/2024-08-22/interaction-XYZ-20240822000000_001.json.zip

group_by_filename: '\d{8}(?=\d{6}_)'

Chunks:
  20240821 -> [2 files]
  20240822 -> [1 file]

Run: 8a32fa3d-...

Per chunk, for processor 'engagement':
  -> aggregates/ih/engagement/daily/period=2024-08/part-8a32fa3d.parquet (rows for both days)

After compaction:
  -> aggregates/ih/engagement/monthly/period=2024-08/part-8a32fa3d.parquet
  -> aggregates/ih/engagement/summary/period=2024-08/part-8a32fa3d.parquet
```

### 6.2 Metric resolution

```
User asks: tile.metric = "CTR"

Resolver:
  CTR -> processor=engagement, kind=formula, expression=safe_div(Positives, Positives+Negatives)

Planner:
  tile fields = {grain: Day, x: Day, color: Channel}; tile.time_range = 2024-08
  -> aggregates/ih/engagement/daily, period=2024-08

Executor:
  read Positives, Negatives by (Day, Channel)
  -> Polars DataFrame

Derive (formula):
  CTR = safe_div(Positives, Positives + Negatives)

Return: rows + plan + freshness + config_hash
```

### 6.3 State catalog snippet

```
States produced by 'engagement' processor:

  Count               type=count            (sum-merge)
  Positives           type=count            (sum-merge)
  Negatives           type=count            (sum-merge)
  UniqueCustomers_cpc type=cpc, lg_k=11     (cpc-union merge)
  UniqueActions_cpc   type=cpc, lg_k=11     (cpc-union merge)
```

### 6.4 Group-by and Time Binding

```
processors.yaml:
  engagement:
    group_by: [Channel, PlacementType, ...]
    time:
      column: OutcomeTime
      grains: [Day, Month, Summary]

dashboards.yaml:
  daily_ctr:
    grain: Day
    x: Day
    color: Channel
```

Transforms define the user-facing schema. If a nicer business name is needed, create it in the pipeline with `rename_capitalize` or `derive_column`, then use that column everywhere: processors, reports, filters, and CLI queries.

---

## 7. Concept relationships (one big table)

| From | To | Relationship | Cardinality |
|---|---|---|---|
| Workspace | Catalog | has-one | 1 : 1 |
| Workspace | Aggregate store | has-one | 1 : 1 |
| Catalog | Source | contains | 1 : N |
| Catalog | Dimension | contains | 1 : N |
| Catalog | Processor | contains | 1 : N |
| Catalog | Metric | contains | 1 : N |
| Catalog | Dashboard | contains | 1 : N |
| KPI recipe library | KPI Recipe | contains | 1 : N |
| KPI Recipe | Metric | materializes on explicit install | 1 : 1 |
| KPI Recipe | Tile | recommends on explicit install | 1 : 0..1 |
| KPI Recipe | State | proposes when selected field/algorithm is not configured | 1 : 0..N |
| Source | Reader | uses | 1 : 1 |
| Source | Transform | applies | 1 : N (ordered) |
| Source | Processor | feeds | 1 : N |
| Processor | Dimension | groups by | N : N |
| Processor | State | produces | 1 : N |
| Processor | Grain | materializes at | 1 : N |
| Processor | Metric | sourced by | 1 : N |
| Metric | Tile | bound to | 1 : N |
| Tile | Dashboard | belongs to | N : 1 |
| Tile | Chart | uses | 1 : 1 |
| Run | Chunk | processes | 1 : N |
| Chunk | Aggregate (partial) | produces | 1 : N (one per processor) |
| Aggregate | Run | tagged with | N : 1 |

This table is the contract between docs and code: any data structure or API must respect these cardinalities.
