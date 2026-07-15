# Running Reports

This guide shows how to start the Value Stream UI and read dashboards
correctly. For first-time setup and data ingestion, start with the
[getting started tutorial](../../tutorials/getting-started.md) and the
[workspaces & catalog guide](../configuration/workspaces-and-catalog.md).

## Start the UI

```sh
uv run valuestream serve examples/demo --port 8501 --headless
```

Open the Streamlit URL printed by the command. The application contains these
pages:

| Page | Use it for |
|---|---|
| Home | Workspace summary, validation state, and quick navigation |
| Data Load | Upload or run source files, discover chunks, and refresh aggregates |
| Reports | View dashboard pages, filters, freshness, charts, and query inspection |
| Catalog | Browse sources, processors, metrics, dashboards, and validation issues |
| Configuration Builder | Edit catalog sections, preview tiles, review chat readiness, and update settings |
| AI Configuration Studio | Draft catalog YAML from sample data with optional model calls |
| Chat With Data | Ask aggregate-aware questions over a selected metric |
| Pipelines / Ops | Run the workspace, inspect recent runs, chunks, and source status |

## Use Data Load

1. Open Data Load.
2. Confirm the catalog status is OK.
3. Choose "Workspace folder" when files already exist under the configured
   source root, or "Upload files" to save files into that root.
4. Use "Run Source" for one source or "Run All Sources" for the workspace.
5. Review discovered chunks and the run result.

When Configuration Builder links here after installing a recipe, run the named
source to materialize its proposed states. A changed processor computation
contract normally makes the affected chunks eligible without **Force rebuild**;
the preview hash transition explains why reports remain at **Backfill required**
until that run publishes matching aggregates.

The "Force rebuild" toggle reprocesses chunks even when they were previously
completed under the current catalog hash. It is non-destructive: immutable
files from earlier successful runs remain until vacuumed.

Use "Rebuild from scratch" when the selected source or the whole workspace
must be recreated from the currently discovered inputs. The confirmation
dialog shows the scope and current aggregate footprint. Value Stream holds the
selected source locks, force-processes every discovered chunk, verifies that
all of them succeeded under an unchanged catalog, and only then removes older
aggregate Parquet files in that scope. If a source discovers no chunks, a run
is partial/failed, or the catalog changes during the operation, cleanup does
not start. Pipeline runs, chunk history, lineage, and configuration versions
under `meta/` remain available for audit.

## Use Reports

1. Open Reports.
2. Select a dashboard and page from the sidebar.
3. Use the compact toolbar for Presentation/Inspect mode, Advanced mode, and Filters.
4. Choose a date preset and up to three primary business filters; open More filters for secondary controls.
5. Read the highlighted calendar chip for the active date preset and exact date
   range (for example, `Last 90 days · Apr 16–Jul 14, 2026`). Click it to reset
   the time range to all time across every report page. Other active chips show
   filter coverage; a partial chip names how many charts support it, and
   unsupported charts name the filters they did not apply.
6. Review freshness, comparison-period labels, targets, and approximation badges before interpreting a result.
7. Use a tile's action menu for Inspect, Expand, and export actions. Chart tiles
   also offer View data. Table tiles are already native sortable dataframes and
   export their displayed rows directly to CSV.

Summary metric cards use compact display values such as `349K` and `120M`
when no explicit catalog `value_format` is configured. Help text and report
detail views remain the place for exact values.

## How Filters Behave

Reports only filter by dimensions persisted by the backing processor. Pages may
declare filters for all tiles or compatible tiles. Partial coverage remains
usable but is never silent: both the filter chip and unsupported tile disclose
it. KPI-strip cards are explicitly configured scalar queries; ordinary charts
are not promoted into KPI cards.

If a filter has no effect on one tile, the tile's backing processor does not
persist that dimension — see
[troubleshooting](../operations/troubleshooting.md).

## Related Guides

- [Chat with data](chat-with-data.md) — natural-language questions over metrics.
- [Querying & export](querying-and-export.md) — CLI queries, DuckDB export, API and MCP access.
- [Builder guide](../configuration/builder.md) — add or edit metrics and tiles.
