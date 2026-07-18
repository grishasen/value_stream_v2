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

Builder mirrors the current step to the URL as a stable `builder_step` value.
Reloading a clean session therefore returns to the same step even when there is
no draft checkpoint. Unknown or obsolete values fall back to **Workspace
Health**; the URL contains no draft or catalog content.

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

When a source Apply changes its aggregate computation contract, the same
post-Apply screen immediately shows **Data refresh required**, names the
affected source, and offers **Run data**. You do not have to continue to Export
to discover that requirement, and the Apply action still never starts the run.

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
including the `entities.subject` field. Selectors that open an object for
editing lead with its stable technical ID, followed by a concise human label
and kind. Review, metric-binding, and report selectors remain human-first, with
the stable ID shown second when it helps disambiguate similar names.

New processor templates derive the Subject Entity Field from the selected
source's first natural-key field. If no natural key exists, the Builder may use
an identity-like field observed in the source sample; otherwise the field stays
explicitly empty and the editor asks you to select an existing source field.

Source schema discovery and dimension profiling share one bounded inspection.
The first uncached read shows an **Inspecting source** status; later controls
reuse the bounded transformed sample while the workspace, source definition,
and discovered file identity remain unchanged. If inspection fails, the inline
message names the source and path pattern. Correct the reader, path, transform,
or permissions issue, then choose **Retry source inspection** to invalidate only
that source inspection.

### Adding a source

Choose **Add source** on **Sources** to open AI Configuration Studio in its
deterministic, sample-first mode. The handoff stays on the active workspace and
keeps the current Builder authoring journey. The reviewed revision carries the
existing catalog forward and adds the generated source bundle; a duplicate
Source ID is rejected instead of silently editing the existing source.

Choose **Cancel and return to Builder** before Apply, or **Return to
Configuration Builder** from the revision receipt after Apply. Applying the
source definition still does not run data.

### Recovering unapplied drafts

While at least one recoverable draft exists, Configuration Builder atomically
writes `meta/config_builder_checkpoint.json`. The checkpoint contains the
current step, a UTC timestamp, the full base-catalog hash, and only JSON-safe,
non-secret draft/widget data. Chat guidance and any draft containing prompt,
provider, API credential, token, password, sample/upload, bytes, DataFrame, or
raw-provider state that cannot be removed without making the draft incomplete
are not checkpointed.

On the next browser session or Streamlit start, choose **Restore checkpoint**
or **Discard checkpoint**. Restore imports the safe registry only; each object
still has to match its current baseline and pass the normal validation gate
before **Apply to workspace** becomes available. If the catalog hash changed,
Builder shows **Reconciliation required** and never silently applies the older
draft.

Checkpoints expire after seven days. Expired or malformed files are removed on
the next Builder visit. Applying or discarding the last recoverable draft also
deletes the file. **Discard checkpoint** deletes it immediately; operators may
also delete `meta/config_builder_checkpoint.json` while Builder is not running.
Deleting a checkpoint never changes the applied YAML catalog.

Dimension recommendations are ranked as **Recommended**, **Review**, or
**Avoid**. Recommended and Review candidates may be selected for the draft by
default. Avoid candidates are never preselected; adding one must be an explicit
choice.

Dimension Packs present available, selected, and missing source fields as
responsive chips. One-Click Promotion leads with recommendation, group-by
safety, cardinality, null percentage, and the review reason. Exact profile and
pack values remain available as collapsed JSON technical detail.

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
selector. Its chart labels and purposes are shared with both chart selectors;
the persisted chart kind remains visible only as secondary technical detail and
continues to round-trip unchanged.

## Removing catalog definitions

Deletion always starts from the explicitly selected object and shows exact
`dashboard/page/tile` paths before it can change the workspace.

- On **Sources**, **Delete source** previews its processors, direct and
  transitive dependent metrics, report tiles, and page filters that would have
  no remaining tile support.
- On **Processors** in **Edit Existing Processor**, **Delete processor** keeps
  the source and every other processor while cascading the selected
  processor's direct and transitive dependent metrics, tiles, and unsupported
  page filters.
- On **Metrics** in **Edit Existing Metric**, **Delete metric** never selects a
  neighboring metric implicitly. Metrics with `depends_on` references block
  the deletion until those references are resolved. If report tiles use the
  metric, you must separately choose to cascade those exact tiles before the
  final confirmation becomes available.

Each confirmed deletion updates the affected catalog files and related
`ai.yaml` descriptions in one rollback boundary, then validates the resulting
workspace. Dashboard and page containers are retained. Aggregate Parquet files
and run history are not deleted by catalog CRUD; use the separate
`valuestream vacuum` lifecycle when persisted files are eligible for removal.

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
