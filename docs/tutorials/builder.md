# Configuration Builder tutorial

Configuration Builder is the guided Streamlit workflow for reviewing and
changing catalog YAML without editing every catalog file by hand. In this
tutorial, you will inspect the workspace, apply a metric or report change, and
finish with the correct data or report handoff.

For the full application workflow, see the
[running reports guide](../guides/users/running-reports.md).

## Before you start

Validate the workspace:

```sh
uv run valuestream validate examples/demo
```

Start the UI:

```sh
uv run valuestream serve examples/demo --port 8501 --headless
```

Open **Configuration Builder** from the sidebar. The header shows **Step 1 of
9**, the current phase, and one task. Use **Back** and **Continue** to follow the
recommended order. **Jump to step** is available for direct navigation.

## 1. Check workspace health

Start on **Workspace Health**. Review the source, processor, metric, dashboard,
page, and tile counts, then expand validation details if the workspace needs
attention. Resolve validation errors before authoring dependent objects.

This step does not edit configuration, so its top-right primary action is
**Continue**.

## 2. Review the aggregate model

Continue through **Sources**, **Processors**, and **Dimensions**. These steps
define how source rows become persisted aggregates.

Make a harmless editor change and notice the draft status. The change remains
session-local and receives a new revision; it is not yet in the catalog. You
can move to another step and return without losing it. Choose **Discard draft**
to restore the persisted definition, or choose **Apply to workspace** to write
and validate it.

Applying configuration does not run the source. After a source, processor, or
dimension Apply, choose the separate **Run data** handoff when you are ready to
refresh aggregates.

Dimension recommendations are grouped as Recommended, Review, and Avoid.
Inspect Avoid candidates before selecting them: the Builder never preselects
them.

## 3. Create or edit a metric

Jump or continue to **Metrics**. Choose one path:

- **Create Metric** → **From Recipe Library** for a reviewed KPI definition.
- **Create Metric** → **From Scratch** for the direct editor.
- **Edit Existing Metric** for maintenance.

The scratch editor begins empty for a new metric. Choose a processor and metric
kind, then enter a display name. The Builder generates a stable technical ID;
existing metrics retain their IDs during guided edits.

For a recipe, read the business definition and accuracy notes, select a
compatible processor, and resolve any field, algorithm, population, or stage
mapping. Choose **Review changes** to inspect the exact processor, metric, and
optional report patches. Then use the single **Apply to workspace** action in
the step header.

After Apply, the Builder reloads the catalog and opens the saved metric for
editing. A formula or display-only metric based on existing aggregate state can
usually proceed directly to **Open report**. If the recipe added processor
state, use **Run data** to materialize it first.

| Change | Requires a data run? |
|---|---|
| Add a formula metric from existing state | No |
| Change a formula expression using existing state | No |
| Add an approximate-distinct metric from an existing CPC/HLL/Theta state | No |
| Add or change processor state | Yes |

Open **Report presentation** to set the friendly label, unit, default number
format, and whether higher or lower values are favorable. These fields affect
Reports only and do not require reprocessing.

For ROC, average-precision, and calibration metrics, choose the backing digest
property when the processor has positive and negative t-digest pairs. The
editor fills both state references while preserving manual selection for custom
state names.

## 4. Build a report tile

Continue to **Reports / Tiles**. Choose a dashboard and page, or create a new
target, then configure the tile. Existing tiles keep their IDs; new dashboard,
page, and tile IDs are generated from their display names.

Use **Page filters and time range** to set filter order, label, placement,
control type, coverage, date presets, and the default preset. An `all_tiles`
filter must be persisted by every tile processor; use `compatible_tiles` when
coverage is intentionally partial.

In **Chart Settings**, a `kpi_card` can use a previous-period comparison,
target, sparkline grain, and point count. Line and stacked-area tiles can use
absolute, index-100, or percent-change scales. Tile descriptions and value
format overrides are available for compatible charts. The Builder never
guesses an aggregation to turn another chart into a KPI.

Choose the one **Apply to workspace** action. The tile and page settings are
written and validated together. Then choose **Open report**. Browse the
searchable, human-readable **Report inventory** when you need to locate another
configured object; enable technical IDs only for exact catalog work.

## 5. Review chat and workspace settings

In **Chat Review**, review which aggregate metrics are available to Chat With
Data. Edit the chat-only agent prompt and dataset or metric descriptions to
clarify business language without changing aggregate behavior.

In **Settings**, review workspace defaults and report appearance. A setting
that changes aggregate defaults produces a **Run data** handoff after Apply;
presentation-only work can continue to the report.

## 6. Export and continue

Open **Export current workspace**. First use the outcome recommendation:

- **Run data** when the latest applied configuration requires refreshed
  aggregates.
- **Open report** when the current aggregates are ready for reporting.

Download sources, processors, metrics, or dashboards YAML directly. The
buttons appear before the collapsed YAML previews. Downloads include only
applied workspace configuration; discard or apply any remaining session draft
before exporting.

## Technical details and Raw YAML

The main workflow uses human labels. Open a collapsed **Technical details**
section when you need exact generated YAML or an expression tree. Reports /
Tiles also offers Raw YAML mode for uncommon chart settings. Raw YAML edits use
the same draft, Apply, transactional validation, and explicit data-run rules as
the visual editors.

## Related docs

- [Business functionality](../concepts/business-functionality.md)
- [Technical overview](../concepts/architecture-overview.md)
- [Expression DSL](../reference/expression-dsl.md)
- [Chart catalog](../reference/chart-catalog.md)
