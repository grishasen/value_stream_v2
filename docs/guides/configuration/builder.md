# Configuration Builder

Use Configuration Builder for guided catalog edits inside the Streamlit UI.
For a first hands-on walkthrough, see the
[Builder tutorial](../../tutorials/builder.md); this page is the task
checklist.

## Guided Workflow

1. Use the Builder step selector or Previous/Next buttons to move through the workflow.
2. Check Workspace Health and resolve validation errors first.
3. Review source and dimension setup.
4. In Metrics, choose **Create Metric** and then either **From Recipe Library**
   or **From Scratch**. The library path is for reviewed business definitions;
   read the calculation/accuracy, select a compatible Processor, resolve
   business-field/algorithm or stage mappings, and optionally add its
   recommended tile. Internal aggregate state IDs appear only under technical
   details. Select **Review changes** to inspect the exact generated YAML patch
   and any source-run plan before installing.
5. After a metric is written, the Builder reloads the catalog, switches to
   **Edit Existing Metric**, and opens that metric so the saved definition is
   immediately visible. Use **Edit Existing Metric** directly for later
   maintenance.
6. Author page filters/time presets and KPI comparison, target, or sparkline behavior.
7. Review chat metric readiness and edit chat-only prompt/description guidance.
8. Update workspace defaults and dashboard theme settings.
9. Save or export YAML.
10. Validate the workspace.
11. Re-run affected sources when processor changes require new aggregates.

Every editable field has a help tooltip beside its label. The tooltip explains
the catalog meaning of the field and includes a concrete example when a value
shape is useful, such as `ih_ai_engagement` for a processor ID or
`Channel, Direction` for grouping dimensions. Editable table columns expose
the same help from their headers. These definitions are shared with AI
Configuration Studio and the KPI recipe library, so the same field keeps the
same meaning across workflows.

The Processor Editor groups short identity fields into three columns, gives
dimension selectors extra width beside descriptions, and places related
kind-specific fields in two- or three-column rows. Long multi-select values
therefore keep useful space without making every scalar field consume a full
row.

Source-field selectors combine the discovered source schema with fields
referenced by source transforms and processors, including an
`entities.subject` field. The resulting choices are listed alphabetically so
the same field is easy to find in every selector. Recipe-library selectors use
the same alphabetical ordering for recipes, processors, business fields,
binding choices, algorithms, populations, and report pages. Funnel stages keep
their configured order because that sequence defines the funnel.

The recipe readiness state tells you whether processor inputs are configured,
need an explicit mapping, or require a new aggregate state. Sketch recipes list
all processor `group_by` and configured business fields plus every compatible
algorithm. Selecting a missing field/algorithm combination adds the processor
state and metric configuration together. The preview names the affected
source, fields, states, and processor computation-hash transition. Installation
and post-write validation share one rollback boundary. After success, use the
Data Load link to run the affected source; the installer never starts that data
operation implicitly.
See the [KPI recipe reference](../../reference/kpi-recipes.md).

## Raw YAML Mode

Use Raw YAML mode inside the builder when you need full YAML control for a
metric, tile, or theme setting. It is useful for small changes such as title
edits, chart settings, or formula tweaks — the same rule applies: save,
validate, then rerun affected data when needed.

## Identifiers

The visual editors generate metric, dashboard, page, and tile IDs from display
names (a lower-case slug prefix plus a random suffix); existing items keep
their IDs during guided edits. Use Raw YAML mode when you need to rename or
override an identifier directly.

## Related Docs

- [Builder tutorial](../../tutorials/builder.md) — metric and tile change
  walkthrough, including which changes need raw replay.
- [Workspaces & catalog](workspaces-and-catalog.md) — the validate-load-verify
  loop this workflow ends with.
- [Expression DSL](../../reference/expression-dsl.md) — formula grammar.
- [Chart catalog](../../reference/chart-catalog.md) — chart kinds and their
  required tile fields.
