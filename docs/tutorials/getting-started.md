# Getting Started

This tutorial takes you from a clean checkout to your first queried metric
using the checked-in demo workspace at `examples/demo`. It exercises the whole
aggregate-first loop:

1. validate the catalog YAML,
2. generate a small synthetic dataset,
3. discover and sample source chunks,
4. run ingestion,
5. query metric aggregates,
6. preview cleanup with `vacuum`.

## Prerequisites

- Python 3.11 or newer.
- [`uv`](https://docs.astral.sh/uv/) for dependency management.

Install dependencies:

```sh
uv sync --all-extras
```

## Validate

The demo workspace ships its catalog in git; the `data/`, `aggregates/`, and
`meta/` folders are intentionally gitignored and created on demand.

```sh
uv run valuestream validate examples/demo
```

Expected result: `ok — examples/demo validates clean.` plus a catalog summary
table (1 source, 4 processors, and the demo metrics and dashboards).

## Generate Demo Data

On a clean clone `examples/demo/data/` is empty. Generate a small synthetic
Pega-shaped dataset with the same helper the test suite uses:

```sh
uv run python -c "
import sys; sys.path.insert(0, 'tests')
from pathlib import Path
from conftest import generate_demo_interactions
files = generate_demo_interactions(Path('examples/demo/data'))
print(f'wrote {len(files)} parquet files')
"
```

This writes eight small Parquet files (`pega_interactions_*.parquet`) shaped
like a Pega Interaction History export. If you have a real Pega export sample
instead, see [Generate synthetic Pega-shaped data](pega-export.md#generate-synthetic-pega-shaped-data)
for the `generate-pega-dummy` command that scales to millions of rows.

## Probe

Confirm discovery, transforms, and schema before running ingestion:

```sh
uv run valuestream probe examples/demo ih --limit 5
```

Expected result: discovered chunks and files, calendar columns including `Day`
and `Month`, the transformed schema, and sample rows.

## Run

Run every source:

```sh
uv run valuestream run examples/demo
```

Run only the IH source:

```sh
uv run valuestream run examples/demo ih
```

The first run processes all chunks. A second run with no input changes skips
them — ingestion is idempotent. Use `--force` to reprocess anyway.

## Query

Engagement rate by channel:

```sh
uv run valuestream query examples/demo VS_Engagement_Rate --by Channel --grain Day
```

Filter with `--where`:

```sh
uv run valuestream query examples/demo VS_Engagement_Rate --by Channel --grain Day --where Channel=Web
```

By default, query output contains only grouping columns and the requested
metric. Use `--raw` when debugging aggregate state columns:

```sh
uv run valuestream query examples/demo VS_Engagement_Rate --by Channel --grain Day --raw
```

The demo processors persist the `Day` grain; coarser buckets such as `Month`
or `Summary` are rolled up safely by the query planner from the stored
aggregates.

## Vacuum

Dry-run cleanup:

```sh
uv run valuestream vacuum examples/demo --dry-run
```

`vacuum` removes superseded aggregate files and orphan reader temp
directories. It does not delete source data or catalog YAML.

## Where Next

- [UI tour](ui-tour.md) — serve the Streamlit app over the workspace you just built.
- [Distribution analytics](distribution-analytics.md) — quantiles, response times, and model-score metrics.
- [Lifecycle and funnel analytics](lifecycle-and-funnel-analytics.md) — funnels and experiment comparisons.
- [Pega export tutorial](pega-export.md) — the same flow against real Pega Dataset Export files.
- [Operations runbook](../guides/operations/runbook.md) — the repeatable operating loop.
