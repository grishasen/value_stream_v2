# Configuration Builder

Use Configuration Builder to make guided, reviewable changes to the active YAML
catalog. For a first hands-on walkthrough, see the
[Builder tutorial](../../tutorials/builder.md); this page is the operating
checklist.

## Guided workflow

The Builder presents one current task at a time. Its compact header shows
**Step _x_ of 9**, the phase, and the task. Use **Back** and **Continue** for the
guided order, or **Jump to step** when you already know where to work.

1. Start in **Workspace Health** and resolve catalog validation errors.
2. Review or edit **Sources**, **Processors**, and **Dimensions**.
3. In **Metrics**, create a metric from the recipe library or from scratch, or
   maintain an existing metric.
4. In **Reports / Tiles**, edit a tile and its page settings as one change.
5. Review the aggregate context available to Chat With Data in **Chat Review**.
6. Update shared defaults and report appearance in **Settings**.
7. Finish in **Export current workspace**, choose the next useful outcome, and
   download catalog files when needed.

Workspace Health is read-only, so its primary action is **Continue**. On an
editable step, the top-right action becomes **Apply to workspace** only when
the current draft is valid and differs from the persisted catalog. There is
exactly one active Apply action for the object being edited.

## Drafts, revisions, and Apply

Editor changes are session-local until you choose **Apply to workspace**. The
Builder compares canonical configuration content, not widget formatting, and
shows whether the draft is unchanged or has a new revision. A draft remains
available when you move to another Builder step. Return to the editor to
continue it, or choose **Discard draft** to restore the persisted definition.

Apply writes the current configuration transactionally and validates the
result. If either the write or validation fails, every affected catalog file is
restored to its exact prior contents. Applying configuration never ingests
source data and never materializes aggregates.

After a successful Apply, the Builder recommends one explicit next action:

- **Run data** for source, processor, dimension, workspace-setting, or recipe
  changes that require aggregate materialization.
- **Open report** for metric, report, tile, or chat-guidance changes that can use
  the current aggregates.

These handoffs preserve the Builder origin so the next page can report the
authoring outcome. An unresolved Run data requirement remains the recommended
outcome even if you apply a later report-only change. Starting a data run
remains a separate user action.

## Editors and technical detail

Editable tables start genuinely empty when no rows exist; they do not create a
placeholder configuration row. Add the first row deliberately. Exact YAML and
generated expression trees remain available in collapsed **Technical details**
sections, while the main editor leads with human labels and business meaning.

Section guidance explains the common path without surrounding every label with
helper copy. Ambiguous, high-risk, or syntax-sensitive controls keep targeted
tooltips and concrete examples, such as `ih_ai_engagement` for a processor ID
or `Channel, Direction` for grouping dimensions. Editable table columns expose
help from their headers where the meaning is not already evident. These focused
definitions are shared with AI Configuration Studio and the KPI recipe library.

The Processor editor groups short identity fields into compact rows and gives
dimension selectors more room. Source-field selectors combine discovered
schema fields with fields referenced by source transforms and processors,
including the `entities.subject` field. Human-readable labels lead selectors;
technical IDs remain available where they disambiguate an object.

Dimension recommendations are ranked as **Recommended**, **Review**, or
**Avoid**. Recommended and Review candidates may be selected for the draft by
default. Avoid candidates are never preselected; adding one must be an explicit
choice.

## Metrics

Choose **Create Metric**, then **From Recipe Library** for a reviewed business
definition or **From Scratch** for the direct editor. The recipe path asks you
to bind a compatible processor, business fields, algorithms, or ordered funnel
stages. Choose **Review changes** to inspect the exact processor, metric, and
optional report patches. The one top action then applies the reviewed recipe.

If a recipe adds aggregate state, the Apply outcome names the affected source
and offers **Run data**. The transaction does not start that run. After a
successful metric Apply, the Builder reloads the catalog and opens the saved
metric for maintenance. See the
[KPI recipe reference](../../reference/kpi-recipes.md).

## Reports and tiles

The Reports / Tiles Apply action writes the current tile and its page settings
together inside one rollback boundary. The visual and Raw YAML modes edit the
same draft.

The collapsed **Report inventory** is searchable and uses dashboard, page,
tile, metric, and chart labels designed for recognition. Enable technical IDs
only when exact catalog identity is needed. The visual report library groups
tiles by purpose and chart type and keeps large groups behind a compact
selector.

## Removing a source

Select a source on **Sources** and choose **Delete source** beside the selector.
The confirmation previews the complete catalog cascade: processors, metrics
including transitive `depends_on` metrics, report tiles, and page filters that
would otherwise have no remaining tile support. The deletion updates all
affected catalog files and related `ai.yaml` descriptions in one rollback
boundary, then validates the resulting workspace. Dashboard and page
containers are retained. Aggregate Parquet files and run history are not
deleted.

## Exporting

**Export current workspace** begins with the outcome handoff. Download buttons
for sources, processors, metrics, and dashboards come before collapsed YAML
previews so exporting does not require reading or copying embedded YAML. The
downloads contain the already-applied workspace; unapplied session drafts are
not included.

## Related docs

- [Builder tutorial](../../tutorials/builder.md) — complete guided walkthrough.
- [Workspaces & catalog](workspaces-and-catalog.md) — the validate-load-verify
  loop this workflow supports.
- [Expression DSL](../../reference/expression-dsl.md) — formula grammar.
- [Chart catalog](../../reference/chart-catalog.md) — chart kinds and required
  tile fields.
