# Benchmark ingestion performance

This guide produces a reproducible Value Stream ingestion baseline before a
performance change is accepted. Activity Monitor screenshots are useful for
diagnosis but are not benchmark evidence.

## Prerequisites

- Use one fixed source snapshot for every comparison.
- Stop unrelated CPU- or I/O-heavy jobs.
- Record results on the same machine and power/thermal configuration.
- Start with `parallel=1`; tune process parallelism only in a separate run.
- Ensure the source workspace validates and its reader root is available.

The runner reads the original source files but writes all aggregates and
metadata into temporary workspaces. The original workspace is not mutated.

## Capture the baseline

From the repository root, run:

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

Use the actual workspace path when benchmarking a production catalog. Generated
results belong under the ignored `artifacts/benchmarks/` directory; attach the
JSON to the performance review rather than committing machine-specific data.

The first suite selects the five processor families corresponding to the legacy
application. It measures those families on the current engine; it is not by
itself an old-versus-new application comparison. The second suite selects the
complete seven-processor IH contract. Both use the exact same physical input
files, whose content hashes are stored in the result.

## Validate the run

For each suite, check:

1. `outputs_equivalent` is `true`. Exact non-binary state must have one digest;
   approximate state is checked through decoded, deterministic probes and its
   algorithm tolerance.
2. `wall_seconds_cv` is at most `0.05`; otherwise repeat the session.
3. Every sample has zero failed chunks.
4. `fixture_id`, file SHA-256 values, catalog/source hashes, execution flags,
   and major runtime versions match before comparing two sessions.

The headline metrics are wall time, rows/s, CPU-s per million input rows, and
peak RSS. Average cores explains whether lower wall time came from additional
CPU saturation. Aggregate bytes/file count, the exact output digest, and
approximate semantic probes guard against "speedups" obtained by silently
doing less work.

The exact digest keeps integer, string, date, key, and stable provenance values
bit-exact. Scalar floats are normalized to 12 significant digits (10 for
pooled variance) to ignore last-bit reduction-order noise. Contract v3 hashes
those logical scalar values with a length-prefixed canonical encoding; Arrow
buffer and StringView layout differences remain confined to the diagnostic
`representation_digest`.

Do not use `representation_digest` as a pass/fail check. Serialized CPC, HLL,
Theta, t-digest, KLL, and Top-K payloads are not canonical: equivalent sketches
can have different bytes because their build order or randomized compaction
differs. `representations_stable=false` is therefore informational when
`outputs_equivalent=true`.

The runner always writes the JSON before reporting a correctness warning, so a
long measurement is retained for diagnosis. A warning does not bless the
timings: investigate `exact_outputs_deterministic` and
`approximate_comparison_issues` before accepting the result.

With `materialize_transforms: true`, execution has two optimizer-controlled
stages: one source-plan collection that materializes the transformed chunk once
(streaming when configured), then in-memory processor fan-out from the shared
DataFrame. Without materialization, processors remain branches of the source
lazy plan. In both cases the contract reports the honest chunk
read-through-write boundary and the remaining sequential
orchestration/metadata/view tail; it does not assign fictional per-node timings
inside either `collect_all` plan.

## Compare a later change

Re-run the same command after the change and retain both JSON files. Do not
update a correctness baseline merely because timings improved. Exact states
must remain equivalent and approximate states must remain inside their
documented error contracts.

The benchmark runner and full field contract are documented in the repository
file `tests/benchmarks/README.md`.

The repository's test workflow also runs a tiny two-repeat semantic smoke
benchmark across the real processor and sketch families. It protects the
contract from rotting, but does not compare wall time or RSS in shared CI.
Machine-specific benchmark JSON stays ignored and should be attached to the
performance review rather than committed as a portable baseline.
