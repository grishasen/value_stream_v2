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

Continue through **Sources**, **Dimensions**, and **Processors**. These steps
define how source rows become persisted aggregates.

On **Dimensions**, pick the workspace's common business dimensions once —
from a dimension pack, the profiler's recommendations, or the source fields
directly. The list is saved as `defaults.dimensions` in `pipelines.yaml`. Each
new processor starts from the common dimensions its source provides as its
Group By, and the Processor Coverage panel can extend existing processors with
their missing applicable dimensions in the same Apply. You can still extend or
trim any single processor's Group By on the **Processors** step.

Make a harmless editor change and notice the draft status. The change remains
session-local and receives a new revision; it is not yet in the catalog. You
can move to another step and return without losing it. Choose **Discard draft**
to restore the persisted definition, or choose **Apply to workspace** to write
and validate it.

Applying configuration does not run the source. After a source, processor, or
dimension Apply, choose the separate **Run data** handoff when you are ready to
refresh aggregates.

To add another source, choose **Create New Source** on **Sources**. In an empty
workspace, this editor opens automatically. Enter a unique Source ID, choose the
reader, and set its root and file pattern. The default template looks for
`**/*.parquet` below the workspace's `data` folder. If matching data is already
available, choose the optional **Load sample** step to populate later field
selectors from a bounded preview; this does not ingest data. Review the
generated YAML and choose **Apply to workspace**. The Builder rejects duplicate
Source IDs and never turns Create into an implicit edit. Use **AI Configuration
Studio** separately when you want a sample-first inferred configuration.

For a Pega outcome source, enter `OutcomeDate` or `OutcomeTime` as the Timestamp
Column. The generated YAML adds the standard datetime, calendar, action-ID, and
numeric-cast transforms for fields present in the loaded sample. Enable
**Materialize Transforms** only when one transformed chunk fits in memory and
several processors should reuse it.

To remove a processor, switch to **Edit Existing Processor**, select the exact
processor, and choose **Delete processor**. Review the dependent metric and
`dashboard/page/tile` paths, confirm the cascade, and apply it. The source and
other processors remain. Existing aggregate folders remain until a separate
`valuestream vacuum` operation removes eligible files.

On **Sources**, the Calculated Fields grid is a compact overview. Add a row,
keep **Enabled** selected, and choose either a guided calculation mode or **AST
YAML** / **Polars**. For a custom mode, select the row in the focused expression
editor below the grid. Author the expression in the multiline input and use the
copy-ready example and live validation beside it. The grid preview is read-only:
**Apply expression** commits the working text to the row, while **Cancel
changes** restores the row's applied expression. An unapplied working expression
survives ordinary reruns and is called out explicitly. When the Source has
other edits, **Apply to workspace** stays disabled until every working
expression is applied or cancelled.

Friendly errors appear next to the multiline input. For a conditional AST, each
`cond` is a complete expression and the fallback key is `else`, not
`otherwise`. Expand **Technical details** only when you need the underlying
parser path and error code.

Dimension recommendations are grouped as Recommended, Review, and Avoid.
Inspect Avoid candidates before selecting them: the Builder never preselects
them.

Dimension Packs show available, already-selected, and missing fields as compact
chips. One-Click Promotion shows its recommendation, group-by safety,
cardinality, and null percentage in a plain summary before you add the field.
The exact profile values remain in the collapsed **Technical details** section.

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

To remove a metric, use **Edit Existing Metric**, select it, and choose **Delete
metric**. Dependent metrics are shown as blockers rather than deleted
implicitly. When report tiles use the metric, review their exact paths and
explicitly select **Also delete these dependent report tiles** before the final
confirmation is enabled. Cancel leaves the catalog unchanged.

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
target, then configure the tile. Dashboard titles and `tabs`, `grid`, or
`stacked` layout are editable alongside page titles. Existing tiles keep their
IDs; new dashboard, page, and tile IDs are generated from their display names.

Use **Page filters and time range** to set filter order, label, placement,
control type, coverage, date presets, and the default preset. An `all_tiles`
filter must be persisted by every tile processor; use `compatible_tiles` when
coverage is intentionally partial.

In **Chart Settings**, a `kpi_card` can use a previous-period comparison,
target, sparkline grain, and point count. Line and stacked-area tiles can use
absolute, index-100, or percent-change scales. Tile descriptions and value
format overrides are available for compatible charts. The Builder never
guesses an aggregation to turn another chart into a KPI.

Chart selectors use the same friendly names and purpose descriptions as the
visual report library. The stored catalog kind, such as `bar_polar`, appears as
secondary technical detail and is unchanged when you save the tile.

Choose the single **Heatmap** chart for a matrix, cohort, descriptive, or
calendar view. Its intensity comes from the selected metric. X and optional Y
offer only configured time grains and processor dimensions; leave Y empty with
X set to Day for the calendar layout. Numeric-distribution heatmaps also offer
Property and Metric (statistic). Older specialized heatmap kinds are migrated
to `heatmap` when edited visually.

For a **Combo** chart, the selected tile metric supplies Y. Choose **Y2
Metric** from the compatible same-kind metrics backed by the same processor;
dimensions and raw aggregate columns are not offered for either value axis.

For a **Pareto** chart, choose only the X category. The selected metric
supplies the ranked bar values and cumulative share curve.

For a **Polar bar** chart, the selected metric supplies Radius. Choose Theta
and Color to control the categorical arrangement.

For a **Treemap** chart, the selected metric supplies Color. Choose Path for
the hierarchy; advanced mode can optionally set a separate size value.

For a **Donut** chart, the selected metric supplies the slice values. Choose
Names for the category labels; no separate Value field is needed.

For a **Sankey** chart, the selected metric supplies link values. Choose only
Source and Target for the flow endpoints.

Choose the one **Apply to workspace** action. The tile and page settings are
validated as a complete catalog before Apply, then written and validated
together. Visual mode preserves chart settings it does not own; use Raw YAML
when you need to edit those uncommon settings directly. Then choose **Open
report**. Browse the searchable, human-readable **Report inventory** when you
need to locate another configured object; enable technical IDs only for exact
catalog work.

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
