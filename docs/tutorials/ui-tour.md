# UI Tour

This tutorial serves the Streamlit application over the demo workspace and
walks through every page. Complete [Getting started](getting-started.md) first
so `examples/demo` validates and has ingested aggregates.

For task-oriented instructions, see [Running reports](../guides/users/running-reports.md).

## Start From the Demo Workspace

Validate the demo workspace:

```sh
uv run valuestream validate examples/demo
```

Run ingestion:

```sh
uv run valuestream run examples/demo
```

Serve the UI:

```sh
uv run valuestream serve examples/demo --port 8501 --headless
```

Open the Streamlit URL printed by the command.

## UI Pages

| Page | Purpose |
|---|---|
| Home | Workspace summary, validation state, and quick navigation |
| Data Load | Source discovery, file upload, source runs, and workspace runs |
| Reports | Dashboard pages, report filters, freshness, charts, and inspect mode |
| Catalog | Sources, processors, metrics, dashboards, and validation status |
| Configuration Builder | Catalog-first editing, chat review, settings, and YAML export |
| AI Configuration Studio | Sample-driven catalog drafting and model-assisted review |
| Chat With Data | Aggregate-aware questions over selected metrics |
| Pipelines / Ops | Runs, chunks, source health, and operational controls |

## Verification

After the UI starts:

1. Confirm the sidebar shows the expected workspace; open Workspace details to
   check the path and catalog revision.
2. Open Data Load and confirm source chunks are discovered.
3. Open Reports and select a dashboard page from the selector row; confirm the
   view, advanced, and filter actions appear on the second toolbar row.
4. Switch to Inspect mode for one tile and confirm data is returned.
5. Open Pipelines / Ops and confirm the latest run appears.

## Related Docs

- [Product overview](../concepts/product-overview.md)
- [Business functionality](../concepts/business-functionality.md)
- [Operations runbook](../guides/operations/runbook.md)
- [Chart catalog](../reference/chart-catalog.md)
