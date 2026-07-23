# Value Stream — High-Level Architecture

| | |
|---|---|
| Document | Architecture overview |
| Companion docs | design/replacement-design.md (detailed), concepts/domain-model.md, reference/processors.md, reference/algorithms.md, reference/readers-and-formats.md, reference/expression-dsl.md, reference/chart-catalog.md, reference/faq.md |
| Audience | Engineers, architects, and senior stakeholders |
| Current stack | Polars · DuckDB · Streamlit · Plotly · Apache DataSketches · PyArrow Parquet |
| Current headless surfaces | Read-only FastAPI HTTP API · local stdio MCP |
| Deferred surfaces | Remote HTTP MCP · OIDC/multi-user service deployment |

---

## 1. Mission and one-line description

Value Stream is a configuration-driven, aggregate-first business intelligence platform for marketing, ML, and customer-lifecycle metrics. It ingests batch exports from upstream operational systems (typically Pega CDH Interaction History and Product Holdings), reduces them to small, mergeable sufficient statistics during a single chunk pass, and serves business reports and dashboards from those persisted aggregates — never from raw rows.

Everything that can be expressed as a small, fixed-size summary per group-by tuple is in scope. Everything that requires raw event histories, exact identity sets across chunks, or per-entity ordered state is out of scope (or is approximated via sketches, or moved to a snapshot processor).

## 2. Design forces and quality attributes

| Quality attribute | What it means here | How the architecture satisfies it |
|---|---|---|
| Aggregate-only at rest | Raw event rows must not survive the chunk pass | Each `chunk_aggregate` produces a sufficient-statistics frame; raw rows are discarded after that |
| Configurable | Non-developers can add metrics, dashboards, and group-by columns | YAML DSL with JSON-Schema validation; closed expression AST replaces `eval`-strings |
| Deterministic | Same computation contract + same file fingerprints = same numbers | computation hash covers workspace defaults, source behavior, and processor semantics; merge ops are associative-commutative |
| Idempotent ingestion | Unchanged chunks are skipped; changed files are reprocessed | `(source, chunk, source-computation-hash, file_hash)` planning; latest-successful-run-wins reads |
| Observability first | Every number is traceable to a chunk and a config | `pipeline_run_id`, `chunk_id`, `period`, `created_at`, `config_hash` columns on every aggregate row |
| Multi-grain | Same metric available at `Day`, `Month`, and `Summary` grains | Per-(source, processor, grain) physical aggregate; planner picks cheapest grain |
| Operates on one node | Targets ~10–100 GB raw input per workspace | Polars + DuckDB; chunk-level concurrency; optional shard-by-source |
| Friendly defaults | First-time users get a working dashboard out of the box | Built-in processor presets, sample workspace, demo dataset, generated dashboards |
| Stack continuity | Reuse Polars, DuckDB, Streamlit, Plotly | Existing primitives keep their roles; Parquet is the rest format; FastAPI and local MCP reuse the governed query layer |

Non-goals: streaming/CDC, distributed compute, ad-hoc warehouse SQL on raw events, replacing Pega CDH semantics.

## 3. Stakeholders and use cases

- **Marketing analyst** opens *Marketing Overview* and asks "what was CTR yesterday on Web/Leaderboard for Cards/Loans?" The dashboard tile resolves to a `daily` engagement aggregate; the planner picks one physical Parquet partition; the answer comes back in milliseconds.
- **Data engineer** schedules a daily ingestion run via CLI/cron and monitors freshness in the *Ops* page.
- **Decision scientist** authors a new metric (`Cost_per_Impression = Cost / Impressions`) by editing `metrics.yaml`, validates with `valuestream validate`, and previews it in the Builder UI without touching Python.
- **Product owner** opens the *Experiments* page and reads the chi-square p-value and odds-ratio CI for the latest A/B test.
- **LLM agent** answers a question through Chat With Data or the local MCP server by calling aggregate metric tools such as `metric_query(metric="CTR", group_by=["Channel"], ...)`. Raw data is never exposed.
- **Auditor** asks "where did this number on the dashboard come from?" and gets a chunk list + config hash + YAML body in three clicks.

## 4. C4 — Context

```
                                +-------------------------------------+
                                |          Value Stream Workspace          |
                                |  (one logical environment, e.g.    |
                                |   "BDT", "RBB", "Demo")            |
                                +------------+------------------------+
                                             ^
                                             |
+--------------------+    files    +---------+----------+    config    +----------------+
| Upstream operational| ----------> | Value Stream Platform   | <-----------| Configuration  |
| systems (Pega CDH,  |             | (this design)      |             | YAML in git    |
| product holdings,   |             +---------+----------+             +----------------+
| subscriptions)      |                       ^
+--------------------+                        |
                                              | reads (UI / SDK / SQL export)
                              +---------------+----------------+
                              |                                |
                       +------+------+                  +------+------+
                       |  Streamlit  |                  |  Notebook,  |
                       |  UI         |                  |  SDK, SQL   |
                       +-------------+                  |  export     |
                                                        +------+------+
                                                               |
                                                 local MCP + read-only HTTP API;
                                                 remote HTTP MCP deferred
```

## 5. C4 — Containers (logical)

```
+-------------------------+   +-------------------------+   +-------------------------+
| 1. Configuration store  |   | 2. Aggregate store      |   | 3. Metadata store       |
| ----------------------- |   | ----------------------- |   | ----------------------- |
| YAML files in git or    |   | Parquet files,          |   | DuckDB databases:       |
| local catalog/ folder.  |   | hive-partitioned by     |   |   chunks, runs,         |
| JSON-Schema validated.  |   | period under            |   |   config_versions,      |
| Hashed -> config_hash.  |   | aggregates/<src>/<proc>/|   |   lineage.              |
+-----------+-------------+   |   <grain>/period=...    |   +-----------+-------------+
            |                 +-----------+-------------+               |
            |                             |                             |
            v                             v                             |
+-------------------------+   +-------------------------+                |
| 4. Ingestion engine     |-->| 5. Query layer          |<---------------+
| ----------------------- |   | ----------------------- |
| Discovery, grouping,    |   | Planner picks physical  |
| chunked Polars pipeline,|   | aggregate; executor     |
| processor fan-out,      |   | runs metric DSL formulas|
| compaction, ledger      |   | and sketch queries.     |
| writes.                 |   +-----------+-------------+
+-----+-------------------+               |
      ^                                   |
      |                                   v
      |                         +-------------------------+
      |                         | 6. Surfaces             |
      |                         | ----------------------- |
      |                         |  - Streamlit UI         |
      |                         |  - Python SDK           |
|                         |  - SQL via DuckDB views |
|                         |  - local MCP tools      |
|                         |  - read-only HTTP API   |
|                         |  - remote MCP deferred  |
      |                         +-------------------------+
      |
      |
+-----+-----+
| 7. CLI    |
| (valuestream) |
| run/     |
| migrate/ |
| vacuum/  |
| validate |
+----------+
```

Container responsibilities:

1. **Configuration store** — YAML files versioned in git: `pipelines.yaml`, `processors.yaml`, `metrics.yaml`, and `dashboards.yaml`. Loaded at startup, validated, hashed.
2. **Aggregate store** — file-system-rooted Parquet directories with hive partitioning. The only place persisted business data lives.
3. **Metadata store** — small DuckDB databases tracking chunks, runs, config versions, lineage. Single-writer per database file; readers can be many.
4. **Ingestion engine** — turns files into aggregates. Reads files lazily via Polars, applies transforms, fans out to processors, writes Parquet partials, runs compactions, updates the chunks ledger.
5. **Query layer** — turns metric requests into reads from the aggregate store. Plans, scans Parquet via DuckDB, materializes Polars frames, applies derived metric DSL, returns rows.
6. **Surfaces** — current read clients on top of the same query layer are Streamlit UI, Python SDK, DuckDB export, Chat With Data, local stdio MCP, and a read-only FastAPI HTTP API. Remote HTTP MCP and multi-user/OIDC deployment are deferred.
7. **CLI** — operator entry point: `run`, `validate`, `migrate`, `backfill`, `vacuum`, `serve`.

The Streamlit reports surface reads typed page-filter definitions from
`dashboards.yaml`, with a backward-compatible inference fallback over the
processors that back the page's tiles. Because Value Stream is aggregate-first,
only persisted processor `group_by` columns are eligible. Each filter declares
`all_tiles` or `compatible_tiles` coverage. The capability matrix is validated,
partial coverage is shown on the active chip, and every unsupported tile names
the active filters it did not apply; filters are never skipped silently.

Before scanning Parquet, the query executor uses file lineage to select only
partials whose `config_hash` matches the current Processor computation hash.
It never blends files from different Processor schemas or substitutes nulls
for newly configured aggregate states. If no current-hash partial has been
published, the query reports the metric as not ready and the Reports UI shows
**Backfill required** instead of exposing a storage-library exception. Legacy
imports without file-lineage rows retain the embedded-hash scan fallback.

Metric display metadata lives in `metrics.yaml`; page controls, KPI semantics,
and tile presentation live in `dashboards.yaml`. Typed presentation properties
cover labels, units, value formats, favorable direction, explicit KPI-strip
placement, previous-period/target comparison, sparklines, absolute/index/change
scales, semantic category colors, goal/reference lines, sorting, and conditional
colors. These settings are consumed by the query/presentation boundary and do
not alter persisted aggregate state or metric calculation semantics.

For external BI tools, `valuestream export-duckdb <workspace> --grain <grain>`
creates a materialized DuckDB file with one table per metric at the selected
grain. Each table is populated through the normal metric query layer using all
persisted `group_by` dimensions from the metric's processor, so SQL consumers
see ordinary metric-output columns rather than serialized sketch state. This is
an export artifact, not the canonical store; Parquet aggregates remain the
source of truth.

## 6. C4 — Components inside the ingestion engine

```
+--------------------------------------------------------------+
|                    Ingestion Engine                          |
|                                                              |
| +------------+   +-----------+   +-----------+   +---------+ |
| | Discovery  |-->| Reader    |-->| Transforms|-->|Processor| |
| | & grouping |   | (parquet, |   | pipeline  |   |fan-out  | |
| +------------+   |  pega zip,|   +-----------+   +----+----+ |
|                  |  csv,xlsx)|                        |      |
|                  +-----------+                        v      |
|                                              +--------+----+ |
|                                              | Per-processor| |
|                                              | chunk        | |
|                                              | aggregator   | |
|                                              +--------+----+ |
|                                                       |      |
|                                                       v      |
|                                              +--------+----+ |
|                                              | Partial      | |
|                                              | Parquet      | |
|                                              | writer       | |
|                                              +--------+----+ |
|                                                       |      |
|                                                       v      |
|                                              +--------+----+ |
|                                              | Compactor    | |
|                                              | (daily ->    | |
|                                              |  monthly ->  | |
|                                              |  summary)    | |
|                                              +--------+----+ |
|                                                       |      |
|                                                       v      |
|                                              +--------+----+ |
|                                              | Chunk ledger | |
|                                              | + run record | |
|                                              +--------------+ |
+--------------------------------------------------------------+
```

## 7. C4 — Components inside the query layer

```
+--------------------------------------------------------------+
|                       Query Layer                            |
|                                                              |
|   +-------------+   +---------+   +----------+   +--------+  |
|   | Resolver    |-->| Planner |-->| Executor |-->| Metric |  |
|   | (metric ->  |   | (pick   |   | (DuckDB  |   | DSL    |  |
|   | processor + |   | grain & |   | scan +   |   | (formula|  |
|   | states)     |   | filters)|   | Polars)  |   | / curve|  |
|   +-------------+   +---------+   +----------+   |  / test|  |
|                                                  |  / RFM)|  |
|                                                  +--------+  |
|                                                              |
|   Cross-cuts: cache (LRU keyed by                            |
|     metric_id, dim_set, filter_hash, grain, config_hash),    |
|     freshness reporter, lineage emitter.                     |
+--------------------------------------------------------------+
```

## 8. End-to-end data flow

For sources with `materialize_transforms: true`, the reader/transform portion
below is collected once per chunk (with the Polars streaming engine when
`reader.streaming: true`). Processor lazy plans then fan out together from that
shared eager frame with the in-memory engine. The shared transformed frame is
released after fan-out, and each processor/grain frame is released after its
immutable aggregate is written. This is an execution strategy only: no raw or
transformed row is persisted, and streaming/materialization settings do not
change the computation hash.

Order-sensitive bounded ML samples are the exception that proves the execution
rule: when a score processor needs personalization or novelty, ingestion adds a
temporary scan-order index before transforms and uses it inside those callbacks.
That keeps results invariant across streaming and in-memory scheduling without
persisting the index.

```
[Files in source folder]
      |
      v
[Discovery]  -- glob + group_by_filename pattern --> chunk_id
      |
      v
[Pipeline run row: status=running] -- durable publication barrier
      |
      v
[Reader]     -- pega_ds_export | parquet | csv | xlsx --> Polars LazyFrame
      |
      v
[Transforms] -- rename_capitalize, parse_datetime, derive_calendar,
                derive_action_id, filter (AST), dedup, defaults
      |
      v
[For each processor bound to this source:]
      |
      v
   [Processor.chunk_aggregate]
      |    -- group_by(group_by columns + time_grain)
      |    -- aggregate states (counts, sums, mins, maxes, sketches)
      v
   [Base chunk aggregate state]
      |
      v
[For each configured grain: processor.compact(base state)]
      |    -- remove finer calendar keys
      |    -- merge states by state-type rule, without rereading raw rows
      v
[Immutable run/chunk parquet partials written atomically]
      |    -- writer returns path/hash/rows/size/timestamp receipts
      |
      v
[Lineage transaction committed] -- receipts for every written aggregate path
      |
      v
[Chunk ledger row status=ok] -- last durable chunk commit marker
      |
      v
[Pipeline run finalized] -- ok / partial / failed; committed chunks become visible
      |
      v
=== ingestion done ===

=== query path begins ===

[Tile / SDK call: metric=CTR, grain=Day, fields=[Day,Channel,Group], time=2024-08]
      |
      v
[Resolver] -- CTR -> processor=engagement, states=[Positives,Negatives]
      |
      v
[Planner]  -- pick aggregates/ih/engagement/daily, period in [2024-08]
      |
      v
[Executor] -- DuckDB read_parquet(...) + compatible tile/page filters
      |
      v
[Polars frame]
      |
      v
[Metric DSL apply] -- CTR = Positives / (Positives + Negatives)
      |
      v
[Return rows + query provenance (stored grain, catalog/computation hashes,
 run IDs, chunk IDs, aggregate scan count, latest created_at)]
```

## 9. Storage layout

```
<workspace>/
├── catalog/                # versioned YAML configs
│   ├── pipelines.yaml
│   ├── processors.yaml
│   ├── metrics.yaml
│   └── dashboards.yaml
├── aggregates/             # the only place business data lives
│   └── <source_id>/
│       └── <processor_id>/
│           └── <grain>/
│               └── period=YYYY-MM/part-<run>-<chunk>.parquet
├── meta/                   # metadata DBs (DuckDB)
│   ├── chunks.duckdb
│   ├── pipeline_runs.duckdb
│   ├── config_versions.duckdb
│   ├── lineage.duckdb
│   └── aggregate_views.duckdb  # governed views over successful aggregates
```

The same workspace layout is used for every variant (BDT, RBB, NBS, Demo, …). Variants are **separate workspaces**; there is no commingling of variants inside one workspace.

## 10. Configuration model

Four YAML files form the catalog; each has a published JSON Schema in `schemas/`.

| File | What it defines |
|---|---|
| `pipelines.yaml` | Sources (where files come from) + readers + transforms + defaults |
| `processors.yaml` | Processors bound to sources, with `group_by`, `time.grains`, states, outcome rules |
| `metrics.yaml` | Derived metric definitions (formula, sketch query, variant compare, contingency test, …) |
| `dashboards.yaml` | Dashboards, pages, tiles; tiles bind to metrics, not processors |

The full DSL is specified in design/replacement-design.md §7 and concepts/domain-model.md §3. Expression semantics are in reference/expression-dsl.md.

Three related hashes serve different boundaries:

- the **catalog hash** identifies the complete authored catalog for audit;
- the **source computation hash** covers workspace defaults, the source
  reader/schema/transforms/defaults, and all processors bound to that source,
  and controls chunk skip/reprocess decisions;
- the **processor computation hash** covers workspace defaults, source
  behavior, and one processor, and is persisted as aggregate `config_hash`.

Presentation-only descriptions, dashboards, and metric prose do not invalidate
ingestion. Canonical payloads for all three identities are inserted into
`meta/config_versions.duckdb`; each emitted aggregate file is recorded in
`meta/lineage.duckdb` with its run, chunk, processor, grain, period, hash, row
count, and path.

The Parquet writer derives that lineage receipt from the in-memory partition
while writing it; before the chunk marker, the ledger confirms the file still
exists with the recorded size. The normal ingestion path does not reopen the
new file merely to recover metadata it already knows. Recovery still
deep-scans embedded provenance before publishing an interrupted run.

Data Load dispatches source, workspace, and clean-rebuild actions to daemon
threads in the Streamlit server process. A locked, process-local registry keyed
by resolved workspace and run scope supplies poll-friendly progress across
browser reloads and websocket reconnects. It is not a durable job queue:
restarting the server ends those threads, after which the normal ingestion
ledger recovery verifies and adopts completed chunk work.

### Configuration authoring surfaces

Value Stream exposes a top-level **Build** choice over two Streamlit authoring
paths backed by the same YAML catalog. **Start from a sample** enters AI
Configuration Studio; **Configure the current workspace** enters Configuration
Builder. The landing page chooses a workflow but does not create a third
authoring store.

- **Configuration Builder** is the catalog-first, validation-first editor for the active workspace. Its compact outline covers workspace health, sources, processors, dimensions, metrics, reports/tiles, chat review, settings, and **Export current workspace**. Each object editor compares a canonical session-local revision with the applied object. Internal navigation preserves a dirty revision until the user applies or discards it; simply visiting a step cannot make it dirty. A current object exposes exactly one **Apply to workspace** action, and apply never starts ingestion. Source steps edit reader runtime settings, schema keys, defaults, dataset filters, and calculated fields. Filters are authored either as rule rows (`field`, `operator`, `value`, `enabled`) or raw expression-AST YAML; calculated fields become `derive_column` transforms with typed AST expressions. Processor steps edit group-by dimensions, grains, states, and optional processor filters. Metric and report steps edit display metadata, page filters/time presets, KPI behavior, descriptions, and scales in addition to chart fields. Page-settings writes merge into the existing page and preserve its tiles, dashboard layout, and theme. Chat review shows which aggregate metrics will be available to Chat With Data and edits chat-only LLM prompt/description guidance in `ai.yaml`; settings edit workspace defaults plus dashboard theme.
- **Configuration Builder checkpointing** persists only privacy-filtered,
  JSON-safe draft registry state, the current step, a UTC timestamp, and the
  full base-catalog hash in `meta/config_builder_checkpoint.json`. Prompt,
  credential, provider, sample/upload, raw-provider, bytes, and DataFrame state
  is never written; a draft that would become incomplete is omitted. Restore
  remains subject to the object baseline and validation gates, catalog drift
  requires explicit reconciliation, and the file expires after seven days or
  is deleted when the recoverable registry becomes empty.
- **AI Configuration Studio** is the sample-first, optionally model-assisted draft workflow. Uploaded bytes are preview-only; the user separately reviews the generated production source plan. CSV, Parquet, JSON, and explicitly detected Pega/archive samples receive format-specific reader defaults, while unsupported archive shapes fail before a misleading source can be drafted. Required-field mappings select only fields present in the approved schema. Sample values are excluded from model prompts by default. Approved schema names, types, null counts, and unique counts remain part of the prompt, while hidden field names are excluded. Every sample-backed model action requires confirmation of the exact sample, provider, model, endpoint route, approved fields, and example-value scope; changing that scope invalidates the confirmation and its widget state, while ordinary step navigation preserves it. The always-open governed Copilot is the first configuration surface after that confirmation and before the active step's manual controls. Effective post-transform field names are authoritative for source filters and every downstream calculation, processor, metric, and report; observed physical sample columns seed catalog validation and evolve through the declared transform order. A draft whose active-source naming transform disagrees with the current Sample contract is retained for comparison but cannot call a provider or Apply; deterministic regeneration presents the complete source naming change and dependent consumers as one dependency-closed review bundle. User-initiated model work preflights the configured provider/model/credential capability, reports a safe corrective action, and caches a successful check for the session. Generation parses, merges, and validates a candidate before review; a bounded sequence of at most two repair passes may run inside the same named operation. An unrecoverable candidate is discarded while the last valid draft remains available, with deterministic generation as the validated fallback. Review uses semantic, dependency-closed bundles in the main canvas; removals require explicit selection and exact YAML stays collapsed. Governed Copilot remains available for read-only explanation while bundles are pending but blocks mutations that could overwrite them. Applying a reviewed revision validates and writes sources, processors, metrics, dashboards, and `ai.yaml` inside one rollback boundary, preserving theme, layout, page, tile, and chat-guidance properties.

Both workflows display one revision ledger: editing draft → ready for review →
reviewed → applied → data refresh required or report ready. Validation is keyed
by the canonical revision, so a field change invalidates the prior verdict and
review. Current-workspace and draft verdicts are never presented as if they
describe the same object.

Every Builder catalog mutation and its post-write validation share one
rollback boundary. A failed write or invalid resulting workspace restores all
affected catalog files before control returns to the UI.

Both metric steps also expose one shared, versioned KPI recipe library. The
packaged recipe YAML is an inert authoring artifact: it describes business
questions, aggregate capability requirements, metric templates, method
accuracy, and report recommendations. A deterministic resolver reports
`ready`, `mapping_required`, or `backfill_required`; only an explicit install
materializes ordinary metric/tile YAML. AI Studio installs into its draft,
while Configuration Builder installs into the active catalog. See the
[KPI recipe reference](../reference/kpi-recipes.md).

Recipe mapping is business-facing: sketch-backed metrics select a
processor-owned field and any algorithm declared compatible by the recipe;
paired digests select one score field, and funnels select stages/populations.
State IDs and sketch parameters are technical detail. A missing
field/algorithm state becomes a deterministic processor-state proposal. Before
installation, both Studios show the exact generated YAML patch plus the named
source, fields, states, and current/proposed processor computation hashes.
Configuration Builder applies and post-validates the multi-file patch inside a
rollback boundary. The changed processor requires the first ingestion run for
a new workspace or replay/backfill for existing aggregates; recipe
installation never starts that data operation or converts one stored sketch
family into another.

Both surfaces use structured YAML parsing and the closed expression AST.
Neither writes free-form Python or mutates the workspace until the user presses
an explicit apply action; every apply re-runs catalog validation. Apply then
classifies whether existing aggregates can open a report or whether the user
must continue to Data Load. The latter is a handoff, not an implicit run.

Privacy-safe authoring instrumentation records only allowlisted workflow,
stage, event, outcome, bounded duration/count, and materialization-required
flags under an anonymous session journey. It has no arbitrary metadata field,
so sample/field values, object identifiers, local paths, prompts, credentials,
and provider error text cannot be attached. The revised entry can be hidden
with `VALUESTREAM_AUTHORING_V2=0` during measured rollout. See the
[authoring rollout guide](../guides/operations/authoring-rollout.md).

## 11. Technology stack and rationale

| Layer | Technology | Why |
|---|---|---|
| Programming language | Python 3.11+ | Mature data tooling; team familiarity; rich sketch/Polars/Plotly ecosystem |
| Vectorized execution | Polars >= 1.x | Lazy frames, streaming engine, expressive groupby + custom map_groups |
| Persistent aggregates | Apache Parquet (PyArrow writer) | Columnar, hive-partitioned, portable, compact |
| SQL surface | DuckDB | Read-Parquet TVF, fast aggregation, easy views, single-file metadata DBs |
| BI export | DuckDB tables | Optional materialized metric tables for Superset/SQL tools |
| Distribution sketches | Apache DataSketches (Python `datasketches`) | t-digest, KLL, CPC, HLL, Theta, Frequent-Items |
| Statistics | SciPy | Chi-square / G-test / odds ratios / CIs |
| ML helpers | scikit-learn (FeatureHasher, cosine_similarity), polars_ds | Personalization, weighted mean |
| UI framework | Streamlit | Quickest path to interactive dashboards in Python |
| Plotting | Plotly | Interactive charts the existing user base already knows |
| Read-only API | FastAPI + Pydantic v2 | Typed metric/chart/freshness/chat endpoints with OpenAPI |
| Schema validation | jsonschema (Draft 2020-12) | Validate YAML config against a stable schema |
| Config templating | Jinja2 (optional, for variants) | Workspace-specific overrides |
| Packaging | `uv` + `pyproject.toml` | Already used in the current repo |
| Quality gates | Local `uv` commands | Lint, format, type-check, test, docs build, schema-validate sample configs |

Optional/future:
- Apache Arrow IPC for very large in-process transfers between SDK and notebook clients.
- ConnectorX / ADBC for upstream operational DB ingestion (today the upstream is files only).

## 12. Concurrency model

- One **pipeline run** at a time per source, enforced by a file-system advisory lock at `meta/source_<id>.lock`.
- A **clean rebuild** acquires all selected source locks in deterministic order and holds them through forced ingestion, coverage validation, scoped cleanup, and aggregate-view refresh. This prevents a concurrent run from publishing files that cleanup could mistake for superseded output.
- Inside a run, **chunks are processed sequentially by default** (predictable memory profile, simpler error semantics). The `--parallel <N>` flag runs chunks in a process pool of N workers: partial parquet part files are per-chunk so worker writes never collide, and all ledger writes stay in the parent process (the DuckDB metadata files are single-writer). Worker processes sidestep the GIL held by Python sketch building, so the initial load scales with cores.
- Inside a chunk, **processors fan out through one batched `pl.collect_all`**: every processor's `chunk_aggregate` plan collects in a single pass (sharing the scan via common-subplan elimination), with a logged sequential fallback if the batch fails.
- The **query layer is read-only**, fully concurrent — Streamlit, SDK, SQL export, MCP, and API reads can run while the engine is writing because Parquet writes go to immutable run-specific files. The run's durable `running` row is the outer publication barrier. Within it, atomic Parquet writes and complete lineage commit before the chunk's `ok` row, which is the chunk commit marker. A partial becomes visible only after that chunk row and a final `ok`/`partial` run row exist; until then readers retain the previous successful version.

## 13. Caching strategy

- **Reader cache** (per-chunk): if the same chunk is read by multiple processors, the source LazyFrame is collected once and aliased.
- **UI metric cache** (Streamlit `st.cache_data`, in `ui/data.py`): query results are memoized keyed by workspace, metric, group-by, filters, grain, date range, and a signature derived from the catalog, the processor config hash, the aggregate files' (count, mtime, size), and the ledger DBs' (mtime, size). Any ingestion run therefore invalidates the cache automatically. This cache lives in the Streamlit surface only.
- **Query layer** (`query/`): the executor itself is intentionally **stateless** — SDK, MCP, and CLI callers always read live aggregates. Predicate pushdown (config-hash filter and `period` partition pruning) keeps the per-call cost low, and DuckDB/Parquet plus the OS page cache absorb repeated reads. A process-level LRU at this layer is a possible future addition but is deliberately not present today, so headless callers never serve stale numbers.

There is **no cache below the storage layer** (no in-memory copy of the aggregate store) — Parquet + the OS page cache do that work.

## 14. Failure semantics

| Failure | Detection | Effect | Recovery |
|---|---|---|---|
| Reader can't open a file | I/O exception inside chunk loop | That chunk fails; run continues with next chunk | Operator fixes file; re-run the source; only failed chunks process |
| Processor exception | Exception inside `chunk_aggregate` | The chunk fails and none of that run's partials become query-visible; the previous successful chunk version remains visible | Operator fixes config or code; re-run the source |
| Partial Parquet write incomplete | Write done atomically (write-then-rename) | Incomplete file never visible to readers | None needed |
| Grain materialization fails | Exception while deriving a configured grain | The chunk remains unpublished at every grain; previous successful data remains visible | Fix the cause and re-run the source |
| Process is terminated before run finalization | A prior `running` row exists after the next caller acquires the source lock | The interrupted run remains invisible until its committed chunks are verified | The next normal source run verifies fingerprint, lineage, files, and computation hashes; valid chunks are published under a recovered `partial` run and reused, invalid chunks are reprocessed |
| Config validation fails on load | JSON-Schema error | Engine refuses to start; CLI prints actionable error | Operator fixes YAML |
| Clean rebuild safety check fails | Empty discovery, incomplete source run, missing published path, or catalog hash change | Scoped cleanup does not start; prior aggregate files and audit metadata remain | Fix discovery/config/run failure and retry the clean rebuild |
| Query layer error | Exception during plan/execute | UI/CLI surface returns a structured error; API returns a governed 4xx/5xx | Operator inspects logs |

A `chunk` is the unit of recovery; a `run` is the unit of reporting and the
outer publication barrier. Successful chunks within a partially-failed run are
kept, marked, and re-used on the next run. After acquiring the source lock, a
new caller treats every older `running` row for that source as interrupted: an
`ok` chunk is retained only when its current input fingerprint, file lineage,
physical provenance, and processor computation hashes all verify. The stale run
becomes `partial` when at least one chunk verifies and `failed` otherwise.
Files without a committed chunk marker remain invisible and are eligible for a
later vacuum.
Recovery fetches lineage once per stale run, indexes that run's physical paths
once, and deep-scans schema-compatible processor/grain files together; embedded
provenance verification remains mandatory. Run-level input/kept row totals sum
only the chunks whose final durable marker is `ok`.

## 15. Security and privacy

| Concern | Posture |
|---|---|
| Raw PII | Never persisted; only aggregate state lives at rest |
| Identity sketches | CPC is the distinct-count default; HLL remains supported; Theta can answer distinct count and is preferred when the same state also needs set algebra. Sketches are not a cryptographic anonymization boundary, so identifiers should be tokenized/HMACed upstream when required. |
| Code-injection via config | No `eval`; only the closed expression AST |
| HTTP API auth | Optional bearer token on loopback; CLI requires one for non-loopback binds; OIDC remains deferred |
| Multi-tenancy | One workspace per tenant/variant; filesystem permissions enforce isolation |
| Audit | Every aggregate row and every query response carries `config_hash` and lineage pointers |
| Logs | Structured JSON; sensitive fields scrubbed before logging |
| Secrets | Read from environment; never in YAML; `.env` files excluded from git |
| Governed SQL | Disabled by default for API/MCP; when enabled, only allowlisted aggregate/export paths are accessible and DuckDB external access/extension loading is locked down |

## 16. Observability

- **Logs** (structured JSON): every chunk start/end, processor timing, row counts, memory snapshots.
- **Metrics**: `valuestream_chunk_seconds`, `valuestream_chunk_rows_in/out`, `valuestream_run_status`, `valuestream_aggregate_size_bytes`, `valuestream_query_seconds`, `valuestream_query_rows_scanned`. Prometheus `/metrics` is reserved for the deferred service surface.
- **Tracing** (OpenTelemetry, optional): one span per run, child spans per chunk, child spans per processor, attributes for `config_hash`, `chunk_id`, and group-by columns.
- **Health**: the Streamlit Ops page and CLI validation expose operational health; the read-only API exposes `GET /health`.
- **Freshness and provenance**: Streamlit exposes freshness, while API/MCP metric queries return the selected physical grain, config hashes, contributing runs/chunks, scan count, and latest aggregate timestamp.

## 17. Deployment topology

Value Stream runs as one process per workspace by default:

```
+-------------------------------------------+
|  Host (VM / container / laptop)           |
|                                           |
|   valuestream serve --workspace bdt           |
|                                           |
|   ├── Streamlit on :8501                 |
|   ├── ingestion runner (CLI / cron)      |
|   ├── optional local stdio MCP           |
|   ├── optional read-only FastAPI         |
|   └── deferred: remote HTTP MCP/OIDC     |
|                                           |
|   $WORKSPACE_DIR -> /data/valuestream/bdt    |
+-------------------------------------------+
```

For multiple workspaces, run multiple processes — each has its own working directory, ports, and configuration. A reverse proxy (Traefik / nginx) routes by hostname.

For local development everything runs from a single `valuestream serve` command pointed at a local workspace directory.

## 18. Lifecycle of an aggregate row

```
        +-------------------+    file group    +---------------------+
        |  source folder    | ---------------> | discovery & grouping|
        +-------------------+                  +---------+-----------+
                                                          |
                                                          v
                                +-------------------+--------+
                                | reader -> Polars LazyFrame |
                                +---------+------------------+
                                          |
                                          v
                              +-----------+-----------+
                              | transforms (typed AST)|
                              +-----------+-----------+
                                          |
                                          v
                              +-----------+-----------+
                              | processor.chunk_agg() |
                              +-----------+-----------+
                                          |
                                          v
                              +-----------+-----------+
                              | partial parquet write |
                              +-----------+-----------+
                                          |
                                          v
                              +-----------+-----------+
                              | (later) compaction    |
                              | daily -> monthly ->   |
                              | summary               |
                              +-----------+-----------+
                                          |
                                          v
                              +-----------+-----------+
                              | served by query layer |
                              +-----------+-----------+
                                          |
                                  (config change?)
                                          |
                                  +-------+-------+
                                  |               |
                                  v               v
                       +----------+--+   +--------+----------+
                       | re-process  |   | leave on disk     |
                       | (new run,   |   | until vacuum,     |
                       |  new hash)  |   | served as legacy  |
                       +-------------+   +-------------------+
```

## 19. Cross-cutting concerns matrix

| Concern | Where handled |
|---|---|
| Schema validation | Config loader (jsonschema) at startup and on `validate` CLI |
| Time grains | Source transforms (`derive_calendar`); processor `time.grains`; planner |
| Sketch parameters | State spec in `processors.yaml` (`type, lg_k, k, …`) |
| Filtering | Two layers: source-wide transform filters; per-processor `filter` AST |
| Default values | `sources.<id>.defaults` map |
| Renaming / casing | Built-in `rename_capitalize` transform |
| Holiday/business calendars | Optional `calendar` block under `defaults`; future `derive_calendar` extension |
| Reproducibility | `config_hash` + chunk ledger |
| Dataset evolution | Workspace migrations table tracks YAML changes over time |
| Retention | `vacuum` CLI prunes superseded chunk partials and older `config_hash` aggregates; Data Load clean rebuild retains only the newly verified run files within its selected source scope |
| Backfill | `valuestream backfill --source X --from YYYY-MM-DD` re-runs all chunks in window |
| Disaster recovery | Aggregate store + metadata is everything; tar it up, ship it, untar |

## 20. Boundaries — what Value Stream is *not*

- Not an ETL platform — there is no transform graph, no joins of arbitrary sources, no schedule across sources.
- Not a warehouse — there is no `SELECT *` on raw events.
- Not a streaming platform — micro-batch is feasible, true streaming is out of scope.
- Not a feature store — the score-distribution processor exists for ML monitoring, not feature serving.
- Not a CDP — entity resolution beyond approximate distinct counting is not provided.

These boundaries keep the platform small enough to build, operate, and reason about. Anything outside the boundary is delegated to the upstream operational system or a downstream specialized tool.

---

**Reading order for a new engineer**

1. This document.
2. concepts/domain-model.md — concepts and their relationships.
3. design/replacement-design.md — full DSL, schemas, APIs, migration.
4. reference/processors.md — per-processor algorithms.
5. reference/algorithms.md — sketches and statistical tests.
6. reference/expression-dsl.md — formal grammar for the AST.
7. reference/readers-and-formats.md — file format specs.
8. reference/chart-catalog.md — Plotly chart bindings.
9. reference/faq.md — common questions; check this when stuck.
