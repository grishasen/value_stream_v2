# Operations Runbook

This runbook gives repeatable procedures for operating a Value Stream
workspace. It is intended for data engineers, operators, and support
engineers. Migration and backfill procedures live in
[Migration & backfill](migration.md); symptom lookup lives in
[Troubleshooting](troubleshooting.md).

## Standard Operating Loop

1. Validate the catalog.
2. Confirm source discovery.
3. Run the workspace or selected source.
4. Check run and chunk status.
5. Review report freshness.
6. Export or publish downstream artifacts if needed.
7. Vacuum superseded files after validation.

## Validate

```sh
uv run valuestream validate examples/demo
```

Expected outcome: exit code 0 and a catalog summary. If validation fails, fix
catalog issues before loading data.

## Probe Source Discovery

```sh
uv run valuestream probe examples/demo ih --limit 5
```

Use this to check:

- Files are discovered.
- Chunk count is reasonable.
- Expected calendar columns exist.
- Transformed schema matches processor references.
- Sample rows look sane.

## Run Ingestion

Run the full workspace:

```sh
uv run valuestream run examples/demo
```

Run one source:

```sh
uv run valuestream run examples/demo ih
```

Force reprocessing:

```sh
uv run valuestream run examples/demo --force
```

Use force after config changes when the existing ledger would otherwise skip
chunks that need to be rebuilt. Force is non-destructive: it publishes new
immutable partials and leaves older physical aggregate files for a later
vacuum.

A KPI recipe that adds processor state changes the source/processor computation
contract. The normal source run therefore reprocesses discovered chunks under
the new hash; force is not normally required for this case. The recipe preview
shows the hash transition and the post-install handoff names the source to run.

Every run is recorded as `running` before its first chunk. If the process or
machine stops before finalization, invoke the same normal command again; do not
use `--force`. After acquiring the source lock, the engine verifies committed
chunks from the interrupted run against current input fingerprints, computation
hashes, lineage, and physical files. Verified chunks are published under a
recovered `partial` run and skipped by the new run. An incomplete or changed
chunk remains invisible and is processed again.

Files created by older releases that have chunk/lineage metadata but no
`pipeline_runs` row cannot be adopted automatically because their source
computation hash was never persisted. Let the replacement run finish, inspect
reports, and use `valuestream vacuum <workspace> --dry-run` before removing
those orphan files.

## Clean Rebuild from Data Load

Use **Data Load → Rebuild from scratch** when old aggregate files should be
replaced as part of the rebuild rather than retained for a separate vacuum.
Choose one source or all sources and confirm the permanent deletion.

The operation follows this order:

1. Acquire every selected source lock.
2. Force-process every currently discovered chunk.
3. Require a complete successful run for every source and an unchanged catalog
   hash. A source with zero discovered chunks fails this safety check.
4. Keep only aggregate files written by those new runs and remove older files
   inside the selected source scope.
5. Refresh aggregate views and release the locks.

Cleanup never starts when a run or safety check fails. New files already
published by successful chunks remain immutable and queryable, while all old
files are preserved. The operation deletes aggregate Parquet and temporary
aggregate files only; run, chunk, lineage, and config-version audit databases
under `meta/` are retained.

## Monitor Runs

Use the Pipelines / Ops UI page for:

- Source status cards.
- Latest run state.
- Rows kept.
- Chunk completion.
- Recent run table.
- Chunk detail for selected runs.

Metadata is stored under `meta/` in DuckDB files. The UI is the preferred way
to inspect it because it preserves the app's projections and formatting.

## Serve the UI

```sh
uv run valuestream serve examples/demo --port 8501 --headless
```

If port 8501 is already used, choose another port:

```sh
uv run valuestream serve examples/demo --port 8502 --headless
```

## Export Metric Tables

```sh
uv run valuestream export-duckdb examples/demo --grain Summary
```

Default output:

```text
examples/demo/meta/metric_export_summary.duckdb
```

Use a custom output path:

```sh
uv run valuestream export-duckdb examples/demo --grain Day --output /tmp/value_stream_metrics.duckdb
```

## Vacuum

Preview cleanup:

```sh
uv run valuestream vacuum examples/demo --dry-run
```

Apply cleanup:

```sh
uv run valuestream vacuum examples/demo
```

Vacuum removes superseded aggregate files and orphan reader temp directories.
Run it after successful validation and report checks. Unlike the clean rebuild
workflow, standalone vacuum keeps the latest successful partial for each
chunk; it does not assert that all current input chunks were rebuilt together.
The CLI acquires every source lock before standalone cleanup, so it refuses to
race an active ingestion. The storage vacuum also protects final and temporary
aggregate files tagged with a `running` run id.

## Benchmark Ingestion

Use [Benchmark ingestion performance](performance-benchmarking.md) before and
after pipeline performance changes. The contract records exact input hashes,
execution settings, environment, throughput, CPU, RSS, output size, and a
normalized correctness digest.

## Related Procedures

- [Migration & backfill](migration.md) — legacy TOML translation, DuckDB
  backfill, and parity sign-off.
- [Troubleshooting](troubleshooting.md) — symptom table and escalation data.
- [Deployment](deployment.md) — hosting the UI, API, and scheduled ingestion.
- [Security](security.md) — tokens, governed SQL, and the aggregate-only
  contract.
