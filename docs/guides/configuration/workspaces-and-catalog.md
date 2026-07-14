# Workspaces and the Catalog

A workspace is the runtime boundary of Value Stream: one folder containing the
catalog (configuration), source data, and everything the engine derives from
them. This guide covers the layout, the catalog files, and the
validate-load-verify loop that every configuration change ends with.

## Prerequisites

- Python 3.11 or newer and `uv`.
- A workspace with a `catalog/` folder containing `pipelines.yaml`,
  `processors.yaml`, `metrics.yaml`, and `dashboards.yaml`.

Install dependencies:

```sh
uv sync --all-extras
```

## Workspace Layout

```text
workspace/
  catalog/
    pipelines.yaml
    processors.yaml
    metrics.yaml
    dashboards.yaml
  ai.yaml            # optional: LLM planner and chat settings
  data/              # source files (created/filled by you or Data Load)
  aggregates/        # created by the engine
  meta/              # created by the engine (DuckDB metadata)
```

Only `catalog/` and source files are required up front. Aggregate and metadata
paths are created as the workspace runs.

## Catalog Files

| File | Defines |
|---|---|
| `pipelines.yaml` | Workspace name, sources, readers, schemas, transforms, defaults |
| `processors.yaml` | Processor definitions, source bindings, dimensions, time grains, state columns |
| `metrics.yaml` | User-facing metrics derived from processor state |
| `dashboards.yaml` | Dashboards, pages, tiles, chart kinds, and presentation settings |

Every YAML file validates against a JSON Schema — see the
[catalog schemas reference](../../reference/catalog-schemas.md). Treat the
catalog as user-facing configuration under source control, not internal code.

## Validate

Always validate before loading data or trusting reports:

```sh
uv run valuestream validate examples/demo
```

Validation checks catalog shape, schema references, expression typing, metric
bindings, and dashboard tile bindings. Exit code 0 on success, 1 on failure.

## Load Data

Run every configured source:

```sh
uv run valuestream run examples/demo
```

Run one source:

```sh
uv run valuestream run examples/demo ih
```

Force a rebuild when the ledger would otherwise skip already processed chunks:

```sh
uv run valuestream run examples/demo --force
```

## Inspect a Source

Use `probe` to confirm discovery, transforms, schema, and sample rows:

```sh
uv run valuestream probe examples/demo ih --limit 5
```

This is useful before a full run, especially after changing reader patterns or
transforms.

## Which Changes Require Reprocessing

| Change | Business impact |
|---|---|
| Add a formula metric from existing state | Usually no raw replay required |
| Add a dashboard tile for an existing metric | Usually no raw replay required |
| Add a new processor | Future runs populate the new aggregate; backfill may be needed |
| Add a new group-by dimension | Raw source replay is usually required |
| Change outcome definitions | Raw source replay is usually required |
| Remove a dimension from reports | Usually safe if no tile depends on it |
| Change chart presentation settings | Affects rendering only, not stored aggregates |

## Three Ways to Edit the Catalog

| Path | Best for |
|---|---|
| YAML edits | Precise, reviewable changes in source control |
| [Configuration Builder](builder.md) | Catalog-first source, processor, metric, tile, chat, and settings authoring inside the UI |
| [AI Configuration Studio](ai-config-studio.md) | Drafting catalog YAML from an uploaded sample and LLM responses, then reviewing and applying it |

Every path ends the same way: save, validate, and re-run affected sources when
processor changes require new aggregates.
