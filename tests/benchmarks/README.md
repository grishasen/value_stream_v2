# Ingestion benchmark contract

This directory contains the repeatable end-to-end ingestion benchmark used for
performance decisions. It deliberately does not ship raw benchmark data.
Instead, both suites point at the same source files from an operator-supplied
workspace and write each sample into a new temporary workspace.

## Suites

| Suite | Processor set | Purpose |
|---|---|---|
| `legacy_equivalent` | engagement, conversion, descriptive, model_ml_scores, experiment | Five-family current-engine workload used for directional legacy comparison |
| `full_current` | the five families above plus action_funnel and audience | Current seven-processor IH product contract |

`legacy_equivalent` is not a timing adapter for the old application. It keeps
the current engine and the selected current processor contracts. A strict
cross-application comparison still requires running the legacy application on
the exact file hashes recorded by this runner.

## Run

```sh
uv run python -m tests.benchmarks.run_ingestion \
  --workspace examples/fat \
  --source ih \
  --suite legacy_equivalent \
  --suite full_current \
  --warmups 1 \
  --repeats 3 \
  --parallel 1 \
  --output artifacts/benchmarks/baseline.json
```

The command hashes the full contents of every discovered input file once. It
then runs one warm-up and three measured samples for each suite. Every sample
gets clean `aggregates/` and `meta/` directories while reading the same physical
input files. It does not clear the operating-system page cache, so the result is
explicitly a warm-cache benchmark.

## Recorded contract

The result JSON records:

- contract version and exact input SHA-256 values;
- catalog and source computation hashes for each suite;
- git revision and dirty flag;
- Python, Polars, DuckDB, DataSketches, platform, CPU count, and Polars threads;
- streaming, transform-materialization, and process-parallel settings;
- wall time, CPU time, average cores, rows/s, CPU-s per million input rows;
- worker peak RSS, input/retained rows, chunk p50/p95, total chunk-pipeline
  time, and the sequential orchestration/metadata/view tail;
- aggregate file count/bytes, a provenance-normalized exact-state digest,
  bounded decoded probes for approximate states, and a raw representation
  digest for diagnostics.

Contract v3 does not require serialized sketch bytes to match. CPC, HLL,
Theta, t-digest, KLL, and Top-K payloads are algorithm state rather than a
canonical result representation; input order and randomized compaction can
change those bytes while estimates remain equivalent. The required
`outputs_equivalent` check combines an exact digest with decoded semantic
probes (up to 64 deterministic rows per processor/state) and the documented
per-algorithm tolerances. `representations_stable` is diagnostic only.
Before hashing, scalar float columns are normalized to 12 significant digits
(10 for pooled variance) so scheduler-level last-bit noise is not treated as a
business-result change; integer, string, date, key, and provenance fields stay
bit-exact. The exact digest uses a logical, length-prefixed scalar encoding
rather than Arrow IPC buffers, so equivalent StringView/chunk layouts hash
identically; `representation_digest` retains the physical IPC diagnostic.

| State | Semantic probe contract |
|---|---|
| CPC, HLL, Theta | two-sigma intervals overlap and estimates differ by no more than twice the wider half-interval (absolute floor 1) |
| t-digest | weight matches exactly; p01/p50/p99 differ by at most 3% (absolute floor `1e-6`) |
| KLL | item count matches exactly; p01/p50/p99 use the same 3%/`1e-6` probe tolerance |
| Top-K | total update weight matches exactly; frequent-item estimate/bound sums use 10% tolerance (absolute floor 1) |

The result JSON is written even when the correctness contract reports a
mismatch. The command prints a warning and preserves every completed timing
sample for diagnosis; do not accept the performance result until
`outputs_equivalent` is `true`.

Timing results are comparable only when the input hashes, computation hashes,
execution flags, and runtime environment match. A wall-time coefficient of
variation above 5% should be treated as an unstable measurement and repeated;
the runner reports the value but does not overwrite or bless a baseline
automatically.

Peak RSS is measured in a fresh worker process. The Phase-0 baseline contract
uses `parallel=1`; process-tree RSS sampling for later multi-process tuning is a
separate benchmark extension.

## CI smoke contract

The normal test suite runs a small, generated end-to-end benchmark twice in a
clean scratch workspace.  Its catalog uses the real binary, numeric, score,
funnel, and entity-set processor kinds and exercises t-digest, KLL, CPC, HLL,
Theta, and Top-K states.  CI checks successful publication and semantic output
equivalence, but deliberately imposes no timing or RSS threshold: shared runner
performance is too noisy to serve as a hardware baseline.  Operator benchmark
JSON remains under the ignored `artifacts/benchmarks/` directory and is attached
to a performance review instead of committed.

With `materialize_transforms: true`, Polars first collects the transformed source
plan once, alongside the raw input-row count and using the streaming engine when
configured. It then runs processor fan-out with the in-memory engine over that
shared DataFrame. Without materialization the processor plans remain branches
of the source lazy plan. The contract does not invent separate timings for
either optimizer-controlled graph: `chunk_pipeline_seconds_sum` is the honest
read-through-write chunk boundary, while `orchestration_wall_seconds` is the
remaining sequential planning, ledger, and view-refresh tail.

## AI Studio preview qualification

The bounded-preview gate is separate from the ingestion benchmark. It creates
a synthetic Parquet file with multiple row groups and profiles the canonical AI
Studio release fixture when that fixture is present:

```sh
uv run pytest -q tests/benchmarks/test_ai_studio_preview.py \
  --junitxml=artifacts/ai-studio-preview.xml
```

The JUnit properties record time, process peak RSS, rows, selected columns, the
logical minimum row groups needed for the row limit, and five synthetic or
three release-fixture cache-candidate cycles. See the AI Configuration Studio
guide for the 64 MiB buffered-input, 128 MiB archive-expansion, row, member, and
qualification thresholds. Peak RSS must satisfy both the 512 MiB absolute
release ceiling and the dynamic `2 * decoded frame + 128 MiB` ceiling; the
synthetic gate uses a stricter 256 MiB absolute ceiling. Five post-GC synthetic
cycles may retain no more than 64 MiB.

The file also records validation and transactional Apply time for a canonical
one-source, one-processor, four-metric, eight-tile draft. Each has a two-second
CI ceiling. `ai_studio_validation_apply_profile` is the first committed
authoring timing measurement; future changes exceeding it by more than 15%
need an approved baseline update rather than silently weakening the gate.
