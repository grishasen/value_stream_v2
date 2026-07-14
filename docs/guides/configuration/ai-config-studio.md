# AI Configuration Studio

The AI Configuration Studio guides source onboarding through sample review,
field approval, defaults, filters, calculations, processors, metrics, reports,
chat readiness, settings, and export. LLM-generated drafts can pre-populate
most catalog settings; review them before applying the draft.

Every editable field and editable table column has a help tooltip. Tooltips
describe the underlying catalog property and show a concrete example when the
expected value shape is not obvious. AI Configuration Studio shares this help
catalog with Configuration Builder and the KPI recipe library; generated and
manually authored definitions therefore use the same terminology.

Processor Parameter Editor uses the same compact logical grid as
Configuration Builder: identity fields share one row, descriptions sit beside
dimensions, and outcome or distribution settings are grouped into related
columns.

## Start From a Sample

Start with either an uploaded CSV, Parquet, JSON, NDJSON, gzip, or zip sample,
or choose **Workspace sample** to reuse a supported file already stored under
`<workspace>/data`. The workspace option avoids uploading the same source file
again when Data Load or an operator has already placed it in the workspace.

## Review Before Applying

Use it as a drafting workflow. Review generated YAML before applying it, then
validate the catalog and run the workspace.

- Metric Review includes friendly labels, units, formats, and favorable
  direction.
- Metric Review includes the same KPI recipe library as Configuration Builder.
  Adding a recipe materializes a metric and optional report tile inside the
  session-local draft; it does not write the workspace immediately. Recipe
  inputs use business fields/algorithms, stages, and populations rather than
  internal aggregate-state IDs. All processor grouping/configuration fields
  and recipe-compatible algorithms remain selectable before the first load;
  a missing combination adds a processor-state proposal to the same draft and
  is marked as requiring the first run or a backfill.
  Before adding the recipe to the draft, **Review changes** shows the exact
  generated YAML patches and, when needed, the affected source, states, fields,
  and processor computation-hash transition.
- Reports Review includes page filters/time presets plus selected-tile
  description, scale, and KPI settings.
- Draft apply validates and preserves the full dashboard
  theme/layout/page/tile structure. Sources, processors, metrics, dashboards,
  and `ai.yaml` are written inside one rollback boundary; failed writes or
  post-write validation restore the prior workspace configuration.
- Use **Apply Draft & Run Source** when a reviewed recipe introduces processor
  state that must be materialized. Plain **Apply Draft To Workspace** changes
  configuration only.

## Identifiers

Generated metric IDs use the entered metric name, so the same metric kind can
be used multiple times without manually inventing IDs. Existing metric
selectors start from processor and kind, then show only the metric IDs that
match that processor/kind pair.

Generated report dashboards use display names for authoring and create
dashboard, page, and tile IDs automatically from those names. Raw dashboards
YAML is still available when you need to override an identifier directly.

## Related Docs

- [Workspaces & catalog](workspaces-and-catalog.md) — validate and re-run
  after applying a draft.
- [Pega export tutorial](../../tutorials/pega-export.md) — using Workspace
  sample with a Pega archive.
- [Chat with data](../users/chat-with-data.md) — the chat settings the Studio's
  chat-readiness step feeds.
- [KPI recipes](../../reference/kpi-recipes.md) — discovery, readiness,
  mapping, provenance, and backfill behavior.
