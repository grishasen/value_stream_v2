# Value Stream — Implementation Plan

This is the detailed plan to build Value Stream from scratch using the architecture and specs in this `docs/` folder. It covers repo layout, tooling, dependencies, testing strategy, the phased delivery plan with exit criteria, and operational rollout.

Audience: implementation team. Estimated full-feature timeline: **12–16 engineering weeks** (one senior + one mid + one junior, full-time). Each phase is independently shippable.

Companion docs (must-reads before starting):

- concepts/architecture.md
- design/replacement-design.md
- concepts/domain-model.md
- reference/processors.md
- reference/algorithms.md
- reference/readers-and-formats.md
- reference/expression-dsl.md
- reference/chart-catalog.md
- reference/faq.md

---

## 1. Repository layout

```
valuestream/
├── README.md
├── pyproject.toml
├── uv.lock
├── ruff.toml
├── mypy.ini
├── .pre-commit-config.yaml
├── docs/                          # the docs you're reading
├── schemas/                       # JSON Schema for YAML config
│   ├── catalog.json
│   ├── pipelines.json
│   ├── processors.json
│   ├── metrics.json
│   ├── dashboards.json
│   ├── expr.json
│   └── processors/
│       ├── binary_outcome.json
│       ├── numeric_distribution.json
│       ├── score_distribution.json
│       ├── entity_lifecycle.json
│       ├── entity_set.json
│       ├── funnel.json
│       └── snapshot.json
├── src/valuestream/
│   ├── __init__.py
│   ├── cli.py                     # `valuestream` CLI entry point
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py              # YAML -> typed model + JSON-Schema validation
│   │   ├── canonical.py           # canonicalize for hashing
│   │   ├── model.py               # Pydantic v2 typed config model
│   │   └── migration.py           # legacy TOML -> Value Stream YAML
│   ├── expr/
│   │   ├── __init__.py
│   │   ├── ast.py                 # AST types
│   │   ├── parser.py              # JSON/YAML dict -> AST
│   │   ├── validator.py           # type / column-existence checks
│   │   └── translator.py          # AST -> Polars expression
│   ├── readers/
│   │   ├── __init__.py
│   │   ├── base.py                # Reader protocol
│   │   ├── parquet.py
│   │   ├── pega_ds_export.py
│   │   ├── csv.py
│   │   └── xlsx.py
│   ├── transforms/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── rename_capitalize.py
│   │   ├── parse_datetime.py
│   │   ├── derive_calendar.py
│   │   ├── derive_action_id.py
│   │   ├── derive_column.py
│   │   ├── filter.py
│   │   ├── dedup.py
│   │   ├── cast.py
│   │   ├── drop_columns.py
│   │   └── coalesce.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── base.py                # Processor protocol + generic merge
│   │   ├── binary_outcome.py
│   │   ├── numeric_distribution.py
│   │   ├── score_distribution.py
│   │   ├── entity_lifecycle.py
│   │   ├── entity_set.py
│   │   ├── funnel.py
│   │   └── snapshot.py
│   ├── states/
│   │   ├── __init__.py
│   │   ├── base.py                # StateType protocol
│   │   ├── count.py
│   │   ├── value_sum.py
│   │   ├── min_max.py
│   │   ├── pooled_mean.py
│   │   ├── pooled_variance.py
│   │   ├── tdigest.py
│   │   ├── kll.py
│   │   ├── hll.py
│   │   ├── theta.py
│   │   └── topk.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── stats.py               # z-test, chi2, g-test, odds ratio
│   │   ├── curves.py              # ROC AUC / AP / calibration from digests
│   │   ├── ml_helpers.py          # personalization, novelty
│   │   └── rfm.py                 # quartiles, segment dictionaries
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── runner.py              # the `run_source` loop
│   │   ├── discovery.py           # glob + chunk grouping
│   │   ├── compactor.py           # daily -> monthly -> summary
│   │   ├── ledger.py              # chunks / runs / config_versions
│   │   └── memory.py              # rss thresholds, spill helper
│   ├── store/
│   │   ├── __init__.py
│   │   ├── parquet.py             # write/read partials
│   │   ├── duckdb_views.py        # views & metadata DBs
│   │   └── lineage.py
│   ├── query/
│   │   ├── __init__.py
│   │   ├── resolver.py
│   │   ├── planner.py
│   │   ├── executor.py
│   │   ├── derive.py              # metric-DSL evaluators
│   │   └── cache.py               # LRU caches
│   ├── api/                       # deferred FastAPI app
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── workspace.py
│   │   │   ├── sources.py
│   │   │   ├── processors.py
│   │   │   ├── metrics.py
│   │   │   ├── dashboards.py
│   │   │   └── admin.py
│   │   └── schemas.py
│   ├── ui/                        # Streamlit app
│   │   ├── __init__.py
│   │   ├── app.py                 # entry point
│   │   ├── pages/
│   │   │   ├── home.py
│   │   │   ├── pipelines.py
│   │   │   ├── catalog.py
│   │   │   ├── dashboards.py
│   │   │   ├── builder.py
│   │   │   └── ops.py
│   │   └── components/
│   │       ├── tile.py
│   │       ├── filter_bar.py
│   │       └── ...
│   ├── charts/
│   │   ├── __init__.py
│   │   ├── recipes.py
│   │   ├── lttb.py                # downsampling
│   │   └── kinds/
│   │       ├── line.py
│   │       ├── bar.py
│   │       ├── treemap.py
│   │       ├── heatmap.py
│   │       ├── scatter.py
│   │       ├── bar_polar.py
│   │       ├── gauge.py
│   │       ├── funnel.py
│   │       ├── boxplot.py
│   │       ├── histogram.py
│   │       ├── calibration_curve.py
│   │       ├── rfm_density.py
│   │       ├── exposure.py
│   │       ├── corr.py
│   │       └── model.py
│   ├── mcp/                       # local stdio MCP server
│   │   ├── __init__.py
│   │   └── server.py
│   ├── sdk/                       # Python SDK
│   │   ├── __init__.py
│   │   ├── workspace.py
│   │   ├── metric.py
│   │   └── dashboard.py
│   └── utils/
│       ├── logger.py
│       ├── timer.py
│       ├── ids.py
│       ├── hashing.py
│       └── time.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   ├── fixtures/
│   └── benchmarks/
└── examples/
    ├── workspace_demo/
    ├── workspace_bdt/
    └── notebooks/
```

---

## 2. Tooling and standards

### 2.1 Languages and runtimes

- **Python**: 3.11+ (3.11 minimum, 3.12 supported).
- **No JavaScript** in v1 — Streamlit handles everything UI-side. Add a JS frontend only if the deferred API gains external clients beyond SDK/MCP.
- **No custom HTML/CSS in the Streamlit app** — render controls, layout,
  status, and actions with native Streamlit components. Do not use
  `unsafe_allow_html=True`, `st.html`, or `streamlit.components.v1.html` for
  in-app UI; downloadable artifacts such as exported Plotly HTML files are the
  only acceptable HTML output.

### 2.2 Package and environment management

- **uv** (`uv sync`, `uv run`, `uv build`) — already adopted by the existing repo.
- `pyproject.toml` is the single source of dependency truth.

### 2.3 Linting, formatting, typing

- **Ruff** (`ruff format` + `ruff check`) — replaces `black`/`isort`/`flake8`.
- **Mypy strict** for `valuestream.config`, `valuestream.expr`, `valuestream.processors`, `valuestream.engine`, `valuestream.store`, `valuestream.query`. Other modules use `mypy --check-untyped-defs`.
- **Pre-commit** hooks: ruff format, ruff check, mypy, schema validation against committed YAML examples.

### 2.4 Testing

- **pytest** with these markers:
  - `unit` — single-module tests.
  - `integration` — multi-module flows in a tmpdir workspace.
  - `e2e` — full pipeline + UI / API smoke tests.
  - `bench` — performance benchmarks (skipped in local default runs).
  - `slow` — anything > 5 s.
- **Hypothesis** for property tests on AST evaluators, state mergers, and pooled-variance correctness.
- **pytest-benchmark** for tracking ingestion throughput per phase.
- Coverage target: ≥ 90% for `valuestream.processors`, `valuestream.expr`, `valuestream.engine`; ≥ 80% elsewhere.

### 2.5 Local quality gates

The project intentionally uses local quality gates instead of checked-in GitHub Actions workflows. Run these before handing off a phase:

1. `uv sync --all-extras`
2. `uv run ruff check .`
3. `uv run ruff format --check .`
4. `uv run mypy src`
5. `uv run pytest -m "not bench and not slow"`
6. `uv run valuestream validate examples/workspace_demo`
7. `uv run mkdocs build --strict`

Benchmark and slow suites are run manually when the touched phase needs them.

### 2.6 Documentation

- Markdown lives in `docs/`. Render with **MkDocs** (`mkdocs.yml`) and verify locally with `uv run mkdocs build --strict`.
- Every public Python API has a docstring. `mkdocstrings` builds a reference page from them.
- ADRs (Architecture Decision Records) under `docs/adr/`, numbered.

### 2.7 Versioning and release

- **Semantic versioning**, starting at `0.1.0`.
- **Changelog** generated from Conventional Commits (`fix:`, `feat:`, `chore:` …).
- Pre-1.0: breaking changes allowed in minor versions; documented in changelog.
- 1.0 ships when all phases below are complete.

---

## 3. Dependencies

```toml
[project]
name = "valuestream"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "polars>=1.16,<2",
    "duckdb>=1.1,<2",
    "pyarrow>=18,<19",
    "datasketches>=5.0,<6",
    "scipy>=1.13,<2",
    "scikit-learn>=1.5,<2",
    "polars-ds>=0.7,<1",          # weighted_mean and friends
    "pydantic>=2.7,<3",
    "jsonschema>=4.22,<5",
    "pyyaml>=6.0,<7",
    "tomli>=2.0",
    "click>=8.1,<9",              # CLI
    "rich>=13.7,<14",             # CLI output
    "fastapi>=0.115,<1",
    "uvicorn[standard]>=0.32,<1",
    "httpx>=0.27,<1",             # SDK + tests
    "streamlit>=1.39,<2",
    "plotly>=6.8.0",
    "psutil>=6.0,<7",
    "structlog>=24.4,<25",
    "prometheus-client>=0.21,<1",
    "jinja2>=3.1,<4",
    "lifetimes>=0.11,<0.12",      # CLV BG/NBD model (optional via extra)
]

[project.optional-dependencies]
ai  = ["mcp>=1.0", "openai>=1.50"]
dev = [
    "pytest>=8.3", "pytest-benchmark>=4.0", "pytest-asyncio>=0.24",
    "hypothesis>=6.112", "ruff>=0.7", "mypy>=1.13",
    "mkdocs>=1.6", "mkdocs-material>=9.5", "mkdocstrings[python]>=0.27",
    "pre-commit>=4.0",
]
```

Pinned major versions; minor/patch movement allowed via `uv lock` updates.

---

## 4. Definition of Done (per feature)

Before any feature lands on `main`:

1. Code passes lint, format, type, and full test suite.
2. Public API has docstrings.
3. New behavior has unit tests (positive + edge case) and at least one integration test.
4. New YAML schema has a JSON-Schema entry under `schemas/` and a sample under `examples/`.
5. Behavior changes update the relevant doc in `docs/` (not just the changelog).
6. Performance-relevant features have a benchmark under `tests/benchmarks/`.
7. The smallest reasonable example workspace (`examples/workspace_demo/`) still produces the same dashboards visually (snapshot test).

---

## 5. Phased delivery plan

Each phase has a goal, the components it builds, exit criteria, and a demo that proves it.

### Phase 0 — Foundations (1–2 weeks)

**Goal**: skeleton repo, local quality gates, docs site, expression DSL, config loader, metadata schema.

Build:

- Repo skeleton matching §1.
- `pyproject.toml`, `uv.lock`, pre-commit, ruff, mypy.
- `valuestream.expr` AST types, parser, validator, Polars translator.
- `valuestream.config` loader with JSON-Schema validation; canonical hashing.
- `valuestream.utils.{logger, hashing, ids, time}`.
- `meta/{chunks, pipeline_runs, config_versions, lineage}.duckdb` schemas.
- `valuestream validate` CLI command.
- MkDocs site shipping.

Exit criteria:

- `valuestream validate examples/workspace_demo/catalog` passes against a hand-written demo catalog.
- The expression DSL has 100% coverage of operators in reference/expression-dsl.md, with a Hypothesis property test (round-trip canonical form).
- Local quality gates pass.
- Docs site renders.

Demo: validate a YAML workspace; show a structured error message when a column is misspelled.

---

### Phase 1 — Aggregate-first IH pipeline (2–3 weeks)

**Goal**: ingest Pega IH exports into `binary_outcome` aggregates with full provenance and idempotency.

Build:

- Readers: `pega_ds_export`, `parquet`.
- Transforms: `rename_capitalize`, `parse_datetime`, `derive_calendar`, `derive_action_id`, `derive_column`, `filter`, `dedup`, `coalesce`, `defaults`.
- States: `count`, `value_sum`, `min`, `max`, `hll`.
- Processor: `binary_outcome` (engagement / conversion / experiment).
- Engine: `discovery`, `runner` (chunk loop), `compactor` (daily → monthly → summary), `ledger`.
- Store: Parquet write + DuckDB views + `chunks` ledger.
- Query: resolver + planner + executor for `formula` and `approx_distinct_count` metric kinds.
- SDK: `Workspace.run_source(...)`, `Workspace.metric(...).by(...).where(...).between(...).to_polars()`.
- CLI: `valuestream run`, `valuestream vacuum`.

Exit criteria:

- A demo workspace ingests the bundled IH sample (existing `data/` zip) end to end.
- `metric("CTR")` returns identical numbers (within 1e-9) to the legacy app on the same input.
- Re-running with no new files is a no-op (skips all chunks).
- Re-running with one new file processes only that file's chunk(s).
- Aggregate rows carry all 5 provenance columns.
- CPC distinct-count estimates and reported bounds are validated against a 1 M-row fixture; explicitly configured HLL remains covered by backward-compatibility tests.
- Memory stays under 4 GB on the largest demo workspace.

Demo: ingest a 5 GB sample IH dataset, re-run idempotently, query CTR by channel/group; show chunk ledger.

---

### Phase 2 — ML and descriptive analytics (2–3 weeks)

**Goal**: numeric_distribution and score_distribution processors, t-digest/KLL state, ROC/AP/calibration curves.

Build:

- States: `pooled_mean`, `pooled_variance` (Welford merge), `tdigest`, `kll`.
- Algorithms: `algorithms.curves` (ROC AUC, average precision, calibration), `algorithms.ml_helpers` (personalization, novelty).
- Processors: `numeric_distribution`, `score_distribution`.
- Metric kinds: `tdigest_quantile`, `curve_from_digests`, `calibration_from_digests`.
- Charts: `boxplot`, `histogram`, `calibration_curve`.
- Migration: legacy `descriptive` and `model_ml_scores` translators in `valuestream.config.migration`.

Exit criteria:

- ROC AUC reconstructed from digests is within 1e-2 of `sklearn.metrics.roc_auc_score` on 100 random distributions in `tdigest_property_suite.json`.
- Pooled variance is within 1e-9 of brute-force variance on a 1 M-row fixture (Hypothesis property test).
- Personalization and novelty produce the same numbers as the legacy app on the bundled IH sample.
- Migration tool successfully translates `value_dashboard/config/config_template.toml` into Value Stream YAML; all metrics validate.

Demo: side-by-side dashboards from the legacy app and Value Stream on the same input, showing identical CTR/AUC/Median(Propensity) numbers.

---

### Phase 3 — CLV, funnels, and snapshots (2 weeks)

**Goal**: lifecycle and stateful processors.

Build:

- Processors: `entity_lifecycle` (CLV), `funnel`, `snapshot` (periodic + accumulating).
- States: `theta`, `topk`.
- Algorithms: `algorithms.rfm` with built-in segment dictionaries (`default`, `retail_banking`, `telco`, `e_commerce`).
- Metric kinds: `lifecycle_summary`, `set_op` (theta intersect/diff), `funnel_dropoff`.
- Charts: `funnel`, `rfm_density`, `exposure`, `corr`, `model`.

Exit criteria:

- RFM segmentation produces identical segment counts to the legacy app on the bundled holdings sample.
- Theta `intersect` and `a_not_b` operations produce correct cohort retention numbers within sketch error.
- A periodic snapshot with daily cadence retains the latest as_of_date per group-by tuple correctly.

Demo: CLV dashboard showing recency-frequency-monetary distribution; cohort retention rate over a 30-day window.

---

### Phase 4 — Streamlit UI (2–3 weeks)

**Goal**: a usable dashboard surface for end users.

Build:

- Streamlit app entry point and routing (`ui/app.py`).
- Pages: Home, Dashboards, Catalog, Pipelines, Ops.
- Tile renderer with chart factory dispatch (reference/chart-catalog.md).
- Filter bar, time-range picker, freshness banner.
- Theme support (per-workspace + per-tile overrides).

Exit criteria:

- Every chart kind in reference/chart-catalog.md renders with the bundled demo data.
- Tile downsampling triggers correctly on > 50 K-point line plots.
- Dashboard load time ≤ 1 s on the demo workspace, ≤ 5 s on a medium workspace.
- The UI never exposes raw row data.

Demo: full marketing/ML/CLV dashboard set running against the demo workspace; freshness banner accurate after a re-run.

---

### Phase 5 — Streamlit Builder UI plus Chat MLP1

**Goal**: give analysts a UI path for authoring catalog YAML without touching
Python, and provide a governed first Chat With Data release over aggregate
metrics.

Build now:

- Streamlit Builder page for new formula metrics and dashboard tiles.
- YAML preview before writing to `metrics.yaml` or `dashboards.yaml`.
- Draft validation after writes.
- Live tile preview against the aggregate store when data exists.
- Chat With Data MLP1: LiteLLM-backed JSON intent planning over catalog metrics,
  validated `query_metric` execution, deterministic text/table/chart rendering,
  and workspace-local `ai.yaml` provider defaults. The Streamlit Chat page does
  not answer natural-language prompts without a configured and enabled LLM
  planner.
- Local stdio MCP MLP1 for Claude Code: `metric_list`, `metric_query`,
  `dimension_values_tool`, and `freshness_get`.
- Read-only FastAPI app for metric manifest/query, validated chart queries,
  dimension values, freshness, optional chat, and opt-in governed SQL. Metric
  responses carry aggregate/config provenance; non-loopback CLI binds require
  a bearer token.

Later:

- OpenAPI spec generation; SDK uses the generated schema for typed responses.
- Remote HTTP MCP, OIDC/multi-user auth, dashboard tile tools, lineage tools, and generated-code
  analysis over query-result frames.

Exit criteria:

- The Builder UI can add or replace a formula metric and a dashboard tile.
- Generated YAML validates with `valuestream validate`.
- Tile preview works for metrics with existing aggregate data.
- Chat can answer "plot daily CTR by customer type and channel" only after the
  configured LLM planner produces a validated `CTR` intent with `time_axis=Day`;
  Value Stream derives the query grain and renders a deterministic Plotly chart.
- `valuestream serve-mcp <workspace>` exposes read-only metric tools when the
  optional `ai` dependency group is installed.
- `valuestream serve-api <workspace>` exposes the read-only HTTP boundary;
  governed SQL is absent unless `--enable-sql` is passed.

Later exit criteria:

- `pytest tests/api/` covers every endpoint with happy/error paths.
- A remote MCP client can answer "what was CTR last week on Web?" via
  authenticated registered tools.

Demo now: build a new metric and a new dashboard tile in the Builder UI; review
the generated YAML; validate the workspace; ask Chat With Data for a daily CTR
chart and inspect the generated aggregate query.

Demo later: remote LLM query via authenticated HTTP MCP.

---

### Phase 6 — Migration and parity sign-off (1–2 weeks)

**Goal**: replace the legacy app for one production workspace.

Build:

- `valuestream migrate --from <toml> --to <yaml>`: full translator, with structured `migration_report.md` listing every legacy field, its target, and gaps.
- `valuestream backfill --workspace <ws> --from-legacy-db <duckdb>`: re-key existing legacy DuckDB rows into the new partitioned Parquet layout.
- Side-by-side deployment story: documentation, banners, dashboards.

Exit criteria:

- Pick one variant (`Demo` first, then `BDT` or `RBB`): every legacy report number reproduced within tolerance (exact for additive metrics; sketch error bounds for HLL/t-digest).
- Migration tool runs in < 10 minutes per variant.
- Side-by-side validation runbook published.

Demo: legacy app and Value Stream on the same browser screen showing matching numbers for all dashboards.

---

### Phase 7 — Hardening (ongoing, ≥ 2 weeks)

**Goal**: production readiness.

Build:

- Prometheus metrics integration; Grafana dashboard JSON shipped under `examples/grafana/`.
- Structured JSON logs with correlation IDs.
- `valuestream vacuum` (legacy `config_hash`s, superseded partials, orphan temp dirs).
- Performance benchmarks committed under `tests/benchmarks/`; run them manually before release sign-off.
- Disaster-recovery runbook (backup/restore tar steps).
- Threat model document (`docs/SECURITY.md`).
- Operator's guide (`docs/OPERATIONS.md`).
- Release notes for `0.1.0` (1.0 once all phases are stable for a quarter).

Exit criteria:

- 90-day stability on a production workspace.
- p95 query latency ≤ 200 ms on monthly-grain dashboards for the largest workspace.
- p95 ingestion time per chunk ≤ 60 s on the canonical hardware profile.
- Successful disaster-recovery drill: restore a workspace from tarball, verify dashboards.
- Security review signed off.

Demo: Grafana dashboard showing live Value Stream metrics over a 30-day window.

---

## 6. Cross-cutting workstreams

### 6.1 Test fixtures

Build once at the start of Phase 1, reuse throughout:

- `tests/fixtures/ih_small.parquet` — 200 K rows, deterministic seed.
- `tests/fixtures/holdings_small.parquet` — 5 K rows, three customer cohorts.
- `tests/fixtures/expected_legacy.json` — numbers the legacy app produces on the small fixtures (recorded once; regression bound).
- `tests/fixtures/tdigest_property_suite.json` — 100 random distributions + sklearn AUC.
- `tests/fixtures/binary_outcome_property_suite.json` — 1 K random `(p, n)` pairs + expected stats.

### 6.2 Migration support

Run weekly during Phases 1–5: re-translate the largest legacy variant's TOML into Value Stream YAML, validate, and diff the produced metric numbers from a small backfill window. Catches DSL gaps early.

### 6.3 Documentation

The doc set in `docs/` is the spec. Every behavior change updates a doc in the same PR. Doc-only PRs are welcome and mergeable.

### 6.4 Observability

Logging and metrics scaffolding lands in Phase 0 even if the metrics are sparse — adding `structlog` later means rewriting log lines. The `valuestream_*` Prometheus metric names are reserved on day 1.

### 6.5 Security review

A standing review per phase. Phase 0 covers config/expr safety (no `eval`). Phase 1 covers PII (no raw rows on disk). Chat MLP1 covers tool scope, prompt/catalog exposure, and provider data-sharing warnings. The read-only API review covers bearer-token/non-loopback rules and SQL containment; the deferred remote-MCP phase covers OIDC and multi-user tool policy.

---

## 7. Concrete first-month plan (week-by-week)

### Week 1 — Foundations

- Day 1: repo skeleton, `pyproject.toml`, `uv sync`, ruff/mypy/pre-commit/local quality baseline.
- Day 2: `valuestream.expr` AST types, parser; first JSON-Schema for `expr.json`.
- Day 3: `valuestream.expr` translator (Polars) for atom + arithmetic + comparison.
- Day 4: `valuestream.expr` translator (date/time, case/when_then, type-rules).
- Day 5: `valuestream.config` loader + canonicalize + hashing; `valuestream validate` CLI.

### Week 2 — Foundations + start Phase 1

- Day 6: meta DuckDB schemas + `engine.ledger`.
- Day 7: `readers.parquet` + `readers.pega_ds_export`; chunk-grouping `engine.discovery`.
- Day 8: transforms `rename_capitalize`, `parse_datetime`, `derive_calendar`, `derive_action_id`.
- Day 9: transforms `filter`, `dedup`, `coalesce`, `defaults`.
- Day 10: tests for transforms and readers; first `pytest -m unit` green run.

### Week 3 — Phase 1 core

- Day 11: `processors.base` Processor protocol; generic `merge` over state catalog.
- Day 12: `processors.binary_outcome` chunk_aggregate (engagement-style).
- Day 13: `store.parquet` write_partial; `engine.compactor`.
- Day 14: `query.{resolver, planner, executor, derive}` for `formula` and `approx_distinct_count`.
- Day 15: `sdk.workspace`; integration test ingesting demo IH end-to-end.

### Week 4 — Phase 1 polish + start Phase 2

- Day 16: `valuestream.cli.run`, `valuestream.cli.vacuum`; idempotent re-run test.
- Day 17: HLL state + `approx_distinct_count` metric kind; HLL property tests.
- Day 18: `processors.binary_outcome` conversion + experiment flavors; migration translator stubs.
- Day 19: streamlit UI scaffold (just the home page + a single CTR tile).
- Day 20: end-of-phase demo to stakeholders; freeze, branch `0.1-phase1`.

After week 4, alternate between odd weeks (build the next phase) and even weeks (harden + write docs + add migration).

---

## 8. Risk register

| Risk | Phase | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| Pooled-variance numerical drift on tiny groups | 2 | Med | Med | Hypothesis property test; clamp `Count <= 1` to NULL |
| t-digest non-associativity at very small N | 2 | Low | Med | Property test with brute-force AUC; warn on `Count < 100` |
| HLL `lg_k` mismatch between processors | 1 | Low | High | Engine validates uniformity at write time; local schema/tests catch drift |
| Legacy `eval` strings that don't translate cleanly | 6 | High | Med | Migration tool refuses silently; emits manual-conversion list |
| Streamlit memory growth on long sessions | 4 | Med | Med | Aggressive query-cache invalidation; LTTB downsampling |
| Aggregate store grows faster than expected | 7 | Med | Med | Vacuum command; per-grain retention policies; per-state opt-in |
| Single-node ingestion ceiling | 7+ | Low | High | `--shard <hash>` multi-process mode; document the ceiling |
| Theta sketch precision insufficient for retention | 3 | Low | Med | Configurable `lg_k`; documented error bound |
| Plotly perf on big tiles | 4 | Med | Low | Downsampling cap; warn at render time |
| Documentation drift | all | High | Med | Doc-PR-with-feature rule; weekly doc review |

---

## 9. Definition of "ready to ship 1.0"

All of the following are simultaneously true:

1. Phases 0–7 complete; every exit criterion hit.
2. Two production workspaces (`Demo` and one of `BDT/RBB/NBS`) running on Value Stream for ≥ 60 days with no data-correctness incidents.
3. Migration tool successfully re-runs against every shipped legacy TOML config.
4. Performance: p95 query ≤ 200 ms (monthly grain, medium workspace); chunk ingestion p95 ≤ 60 s; aggregate-store growth ≤ 5% of raw input.
5. Documentation site deployed; every public API documented; doc-search works.
6. Security review signed; PII posture documented; threat model reviewed.
7. Disaster-recovery drill executed.
8. Changelog cleaned up; release notes published; pyproject pinned to non-pre-release versions of all major deps.
9. One external (or cross-team internal) team builds a workspace from documentation alone, without the legacy code as a reference.

The last point is the real test — it is the goal of the doc set.

---

## 10. After 1.0 — what's next

Not in the 1.0 plan, but in the roadmap:

- **Streaming / micro-batch.** Tail a kafka topic, accumulate into per-minute chunks, run the same compaction. Parameterized in the `reader` (`kind: kafka`, `cadence: 1m`).
- **Multi-node ingestion.** Shard chunks across worker processes; coordinator process owns the ledger.
- **Aggregate-aware optimizer.** Auto-suggest grains based on usage telemetry.
- **Power BI / Tableau connector.** ODBC over the DuckDB views.
- **Plugin marketplace.** Custom processors and chart kinds packaged as PyPI extras.
- **Workspace federation.** A query layer that spans multiple workspaces transparently for a parent-org rollup.

These are all additive to the 1.0 architecture; none requires rethinking the core.
