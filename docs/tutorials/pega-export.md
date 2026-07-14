# Pega Export Tutorial

This tutorial explains the Value Stream path for Pega-style export files. The
exact reader settings depend on the export format, but the workflow is the same:
configure the source, validate, probe, run, and inspect reports.

## Configure the Source

Pega Dataset Export-style workspaces usually define a source in
`catalog/pipelines.yaml` with:

- Reader kind `pega_ds_export` for zip-based Pega exports, or `parquet` when
  exports have already been converted.
- A `file_pattern` that matches the export files.
- A `group_by_filename` regex that groups files into idempotent chunks.
- Timestamp parsing transforms for fields such as `OutcomeTime` and
  `DecisionTime`.
- Calendar derivation for report grains such as Day, Month, Quarter, Year, and
  Summary.
- Outcome filters and defaults for fields that may be absent in some exports.

See `examples/demo/catalog/pipelines.yaml` for a runnable Parquet-based
Pega-style configuration example (this is the only example workspace checked
into the repository).

In AI Configuration Studio, use **Workspace sample** when the Pega archive is
already under the active workspace's `data/` folder. Select the archive, click
**Use Workspace Sample**, review the generated draft, and apply it only after
the Studio validation gate passes.

## Validate

```sh
uv run valuestream validate examples/demo
```

## Probe

```sh
uv run valuestream probe examples/demo ih --limit 5
```

Confirm that expected fields such as channel, issue, group, outcome, model
control group, propensity, and timestamp columns are present after transforms.

## Run

```sh
uv run valuestream run examples/demo ih
```

Use `--force` after changing source transforms, processor dimensions, or outcome
classification:

```sh
uv run valuestream run examples/demo ih --force
```

## Review Reports

Start the UI:

```sh
uv run valuestream serve examples/demo --port 8501 --headless
```

Use:

- Data Load to confirm discovered chunks and source run status.
- Reports to review engagement, propensity, funnel, experiment, and response
  time pages.
- Pipelines / Ops to inspect runs and chunks.
- Catalog to review generated sources, processors, metrics, and dashboards.

## Generate Synthetic Pega-Shaped Data

Use this when you need a realistic test dataset shaped like a Pega export but
cannot use production data. It derives the schema from a real export sample
and generates configurable volumes:

```sh
uv run valuestream generate-pega-dummy \
  --source path/to/source.json \
  --output-dir /tmp/value-stream-demo-data \
  --start-date 2026-01-01 \
  --days 7
```

`--source` accepts a Pega JSON/NDJSON export or a zip/gzip/tar.gz archive of
JSON records. See the [CLI reference](../reference/cli.md#generate-pega-dummy)
for volume, outcome-rate, and output options.

## Common Pega Checks

| Check | Why it matters |
|---|---|
| Timestamp parsing succeeds | Calendar grains and freshness depend on valid time columns |
| Outcome values are normalized | Engagement and conversion metrics depend on positive and negative definitions |
| Natural keys are stable | Deduplication depends on consistent identifiers |
| Group-by dimensions are approved | Report filters only work for persisted dimensions |
| Propensity fields are numeric | Model quality metrics require numeric score distributions |

## Related Docs

- [Running reports](../guides/users/running-reports.md)
- [Operations Runbook](../guides/operations/runbook.md)
- [Readers and Formats](../reference/readers-and-formats.md)
- [Processor Specs](../reference/processors.md)
