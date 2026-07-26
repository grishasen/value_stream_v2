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
2. Review or edit **Sources**, **Dimensions**, and **Processors**. Dimensions
   comes before Processors because it defines the workspace's common business
   dimensions that new processors start from.
3. In **Metrics**, create a metric from the recipe library or from scratch, or
   maintain an existing metric. The edit workflow starts with a searchable
   metric selector; metric selection synchronizes processor and kind, while
   changing processor or kind selects a compatible metric.
4. In **Reports / Tiles**, edit a tile and its page settings as one change.
5. Review the aggregate context available to Chat With Data in **Chat Review**.
6. Update shared defaults and report appearance in **Settings**.
7. Finish in **Export current workspace**, choose the next useful outcome, and
   download catalog files when needed. The terminal header shows
   **Workspace current** as a completion status, not as a disabled action.

Workspace Health does not edit YAML, so its primary action is **Continue**.
The **Validation Sample** panel is available beside Catalog Health in both
Workspace Health and Export whenever sources exist. Load or reload a bounded
source sample there to validate transforms against observed physical fields;
those field names remain active while you move between Builder steps. On an
editable step, the top-right action becomes **Apply to workspace** only when the
current draft is valid and differs from the persisted catalog. There is exactly
one active Apply action for the object being edited.

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

Create drafts use the same recovery contract as maintenance drafts. For
example, a recovered **Create Metric** draft is offered before the source and
metric-kind selectors can replace its identity. When Apply is unavailable, the
editor states the exact missing or invalid input while **Continue** remains
available; the draft stays preserved until it is fixed, applied, or discarded.

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
Existing-source editors inspect on demand. A create-source editor does not read
files when it first opens; choose the optional **Load sample** step after setting
Reader, Root, File Pattern, and grouping when you want its field controls
populated from a representative match. The first uncached read shows an
**Inspecting source** status; later controls reuse the bounded result while the
workspace, reader definition, and discovered file identity remain unchanged.
If inspection fails, the inline message names the source and path pattern.
Correct the reader, path, or permissions issue, then choose **Retry source
inspection** to invalidate only that source inspection. In create mode, choose
**Load sample** again after replacing or changing the matching sample files.
The same observed physical fields feed Workspace Health validation. Returning
to Workspace Health shows **Reload sample**, so schema-aware validation can be
refreshed without reopening or recreating the source.

Catalog-only validation remains conservative because YAML does not enumerate
every raw column. Data Load therefore keeps the **Validation Sample** panel
available even when Catalog Health is blocked. Loading a matching sample
revalidates the catalog against its observed fields and unlocks the data-run
controls when the transforms are valid; genuinely missing fields and
transform-order mistakes remain blocking authoring errors. When ingestion
starts, structural catalog validation still runs, while physical source
expressions are evaluated by the reader and transform executor against the
actual chunks instead of the incomplete catalog-inferred schema.

### Adding a source

Choose **Create New Source** on **Sources**. In an empty workspace, the source
creator opens automatically. The initial template uses the Parquet reader,
`data` as its workspace-relative root, and `**/*.parquet` as its recursive file
pattern. Change the reader, root, pattern, grouping, schema, defaults, filters,
or calculated fields in the same editor used for existing sources.

After the file settings are correct, optionally choose **Load sample**. Builder
reads one bounded matching sample, keeps only its field names and types in the
editor session, and populates Natural Key, Drop Columns, defaults, filters, and
calculated-field selectors. The preview does not ingest data or write catalog
configuration. Changing reader, root, pattern, or grouping marks the preview
stale and asks you to reload it. Rename / Capitalize updates the displayed field
names without another file read.

Enter a unique, reference-safe Source ID. The Builder rejects a duplicate ID
instead of silently replacing an existing source. Review the generated YAML,
then choose **Apply to workspace**. Apply validates and writes the source in the
catalog transaction, returns the editor to **Edit Existing Source**, and does
not run data. Use **Data Load** separately after files are available.
Apply inspects the current draft ID and reader settings, then evolves the
observed physical columns through transforms in order. Calculated fields may
therefore reference effective names such as `Channel` in the same create
operation that enables `rename_capitalize` for a physical `pyChannel` column.

When the Timestamp Column is `OutcomeDate` or `OutcomeTime`, create mode adds
the standard Pega preprocessing transforms to the generated source: it parses
the outcome timestamp (and `DecisionTime` when present), derives calendar
fields, derives `ActionID` from `Issue`, `Group`, and `Name` when available,
and casts the available propensity, priority, and revenue fields to `Float64`.
Loaded sample fields constrain these generated references so an optional
standard field is not added when it is absent. Use **Materialize Transforms**
to collect each transformed chunk once in memory for reuse by all processors;
leave it off when the chunk may not fit comfortably in memory.

Source filters support both the Rules grid and an editable **Filter AST YAML**
area in Raw AST mode. Invalid YAML or expression DSL stays in the draft and
blocks Apply with an inline error; it is never replaced by a read-only `{}`.
Choosing **Discard draft**, or deleting the source being edited, clears the
source widgets before the next full render so a later create operation starts
from the default template rather than inheriting stale values.

Use **AI Configuration Studio** instead when a sample-first workflow should
infer format-specific defaults and propose a dependency-closed catalog bundle
for review.

When **Use Rename / Capitalize Transform** is enabled, the Sample preview changes
immediately to the transformed schema. Every later field control and validation
uses those transformed names: for example, select `Channel` in required-field
mapping, defaults, filters, calculations, field approval, processors, and
Copilot operations after a source `pyChannel` becomes `Channel`. Metric
dimension properties, dashboard page filters, and report tile fields follow
the same contract. A stale raw reference such as `pyChannel` is rejected while
the transform is enabled.
Turning the transform off restores the raw schema and remaps compatible editor
state back to the source names.

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

## Dimensions

The Dimensions step is processor-independent. It edits one workspace-level
list of common business dimensions, persisted as `defaults.dimensions` in
`pipelines.yaml`. Every processor created afterwards starts from the common
dimensions its source can actually provide (matched case-insensitively
against the source's fields) as its Group By, and each processor may extend
or trim that list in the Processors step.

The **Processor Coverage** panel shows how the shared list maps onto existing
processors and which applicable common dimensions each one is missing. An
explicit toggle extends every existing processor with its missing applicable
dimensions inside the same Apply transaction. A processor's variant column
counts as covered: the processor already persists it as a dimension, and
validation rejects repeating it in group-by. Applying only the common list
never requires a data run; extending processors changes their aggregate
computation contract, so the Apply outcome then names the affected sources
and offers **Run data**.

Dimension recommendations are ranked as **Recommended**, **Review**, or
**Avoid**. Recommended and Review candidates may be selected for the draft by
default. Avoid candidates are never preselected; adding one must be an explicit
choice.

Dimension Packs present available, selected, and missing source fields as
responsive chips. One-Click Promotion leads with recommendation, group-by
safety, cardinality, null percentage, and the review reason. Exact profile and
pack values remain available as collapsed JSON technical detail.

## Processors

Selecting a **Kind** while creating a processor auto-fills the Description
with that kind's definition; text you typed yourself is never overwritten. A
short guide under Group By explains what the kind is for and lists example
KPIs (for `entity_set`: unique counts, audience overlap, Top-N frequent
values).

**Auto Outputs** starts from the selected kind's engine-default states (binary
outcome: counts plus a unique-subject sketch; entity set: CPC and theta sketches
over the entity column; derived kinds show their computed states). Switching
kind reseeds those defaults while preserving rows whose state, type, source
column, or **Enabled** value you changed. In particular, disabling a default
row remains an intentional edit. Add further sketch states (Top-K, CPC, theta,
digests) with **Add state** or as grid rows when the kind supports them.
**Parameters YAML** round-trips state-specific settings such as `lg_k`, `k`,
`lg_max_map_size`, `weight`, `mean`, `score_property`, and `outcome`; **Type**
and **Source Column** remain separate fields and cannot be redefined there.
After a state is added, the popover clears before the next state is authored.
Sketch-size keys are checked against the selected type so, for example,
`lg_max_map_size` cannot leak from a Top-K state into a CPC or theta state.

For `numeric_distribution`, selecting a **Numeric Property** immediately adds
its count, pooled mean, pooled variance, minimum, maximum, and selected
t-digest/KLL rows to **Processor Sketches**. The operation is additive and
idempotent: existing and manually edited rows are not replaced. Count and mean
are included because pooled variance needs both merge dependencies.

Applying or updating a processor automatically adds one distribution metric for
every unconditioned t-digest or KLL state that does not already have one. The
metric omits `quantile`, so it reads the median as its primary value and exposes
the full quantile suite to distribution charts. A numeric pooled-variance state
also creates a standard-deviation formula metric using `sqrt(variance)`;
standard deviation is derived rather than persisted because variance is the
mergeable aggregate. Existing generated metrics are reused, explicit percentile
metrics remain separate, and outcome-specific positive/negative digests are
left as internal inputs to curve and calibration metrics. The processor and any
automatic metrics are written in the same validated transaction; the separate
data run materializes the new state.

Kind-specific settings cover every engine-read property: `entity_set` exposes
the sketched **Entity Column**, and `entity_lifecycle` exposes its customer,
order, monetary, and purchase-date key columns.

**Dedup Keys** is available only for `binary_outcome` and
`score_distribution`, the processor kinds that consume `dedup_keys` at runtime.
Rows with the same configured values are counted once within each load chunk;
leaving the control empty aggregates every row. Other processor kinds do not
persist a no-op dedup setting.

Renaming an existing processor is one transactional catalog change. It replaces
the old processor ID, retargets metrics whose `source` referenced that ID, and
moves the matching Chat With Data description in `ai.yaml` before validating
the resulting workspace.

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

The main editor exposes **Metric Label** directly below the immutable Metric
ID. This is the human-readable `display.label` used by report axes and
selectors; leave it blank to fall back to a label derived from the Metric ID.
Unit, number format, and favorable direction remain under **Report
presentation**.

Generated Metric IDs are always ASCII-safe, including when the display name
contains accents or other Unicode characters. If a manually entered ID is
invalid, the draft remains available for correction but does not prevent
continuing to Reports.

A digest state is the mergeable binary accumulator persisted by a processor;
it is not itself a public query result. Its automatically authored distribution
metric is the stable catalog name that tells the query layer to decode that
state and lets reports bind to it. Use **Create Metric** only for additional
named percentiles or other derived interpretations of the same digest.

## Reports and tiles

The Reports / Tiles Apply action writes the current tile and its page settings
together inside one rollback boundary. The visual and Raw YAML modes edit the
same draft. The dashboard title and layout (`tabs`, `grid`, or `stacked`) and
page title are editable in the same flow. Before Apply is enabled, the proposed
dashboard, page settings, and tile are validated as a complete catalog. Visual
mode round-trips settings outside its controls unchanged; Raw YAML remains the
escape hatch for editing them directly. A new Raw YAML draft starts from the
same valid report scaffold as Visual mode, and its session-local text survives
reruns caused by presentation and page-settings controls.
New-page draft state is keyed independently from its generated dashboard and
page IDs, so changing a new dashboard name cannot reset the entered Page Name.
The persisted dashboard and page titles are the exact trimmed values shown in
the editor.

Box plots bind only to a `distribution` metric. The selected metric already
identifies its processor digest and source property, so the tile exposes no
separate Property, Y, or Value control. Use X, Color, and optional facets only
to group the selected distribution.

The collapsed **Report inventory** is searchable and uses dashboard, page,
tile, metric, and chart labels designed for recognition. Enable technical IDs
only when exact catalog identity is needed. The visual report library groups
tiles by purpose and chart type and keeps large groups behind a compact
selector. Its chart labels and purposes are shared with both chart selectors;
the persisted chart kind remains visible only as secondary technical detail and
normally round-trips unchanged.

Heatmaps use one adaptive **Heatmap** choice. The selected metric supplies cell
intensity, so there is no Value or Color field. X and optional Y contain only
configured time grains and persisted processor dimensions. Leave Y empty with
a daily X for a calendar layout; set both axes for a matrix or cohort layout.
Intensity is selected only from the chosen metric's read-only output contract;
there are no separate Property or statistic controls. Retired heatmap aliases
are rejected by catalog-v2 validation.

Combo charts also use metric-owned values. The tile's selected **Metric**
supplies the primary Y series, so no Y field is shown. **Y2 Metric** offers
only other scalar metrics with the same metric kind and backing processor.
The report query derives both metrics independently and joins them on their
shared time and dimension keys.

Pareto charts follow the same metric-owned Y convention. Choose only the X
category; the selected metric supplies the ranked bar values and cumulative
share curve.

Waterfall charts also derive Y from the selected metric. Choose only the X
category. Bar colors are assigned by change semantics (increase, decrease, or
total), so Waterfall has no selectable Color dimension or facet controls.

Polar bar charts derive Radius from the selected metric. Choose only the
Theta category and optional Color grouping.

Treemap charts derive Color from the selected metric. Choose only the Path
hierarchy; advanced mode may optionally configure a separate size value.

Donut charts derive slice values from the selected metric. Choose only the
Names category and optional Color grouping.

Sankey charts derive link values from the selected metric. Choose an ordered
**Flow Path** containing at least two time/dimension fields. Every adjacent
pair becomes another link layer; legacy Source/Target tiles remain readable
and are migrated to Flow Path when edited visually.

The Metric editor shows **Outputs** as a read-only list computed from the
metric kind and its configuration. In report charts where the measure was
previously implicit—KPI, Gauge, Line, Bar, Stacked Area, Waterfall, Pareto,
Combo primary Y, Polar radius, Treemap value/color, Donut value, Sankey value,
and Heatmap intensity—the visual and advanced editors show the corresponding
role as a metric-output selector. It contains only that read-only output list.

Interval charts separate category and numeric roles. Choose a time or dimension
for X; Y and every Error field offer only outputs produced by the selected
metric. Variant comparison metrics therefore expose their estimates, confidence
bounds, standard error, lift, and related statistics without mixing dimensions
into numeric selectors.

The library always shows every purpose group and every chart kind of the
active group, whether or not the workspace uses it yet. Kinds with configured
reports list them as selectable pills; kinds without any offer **Create … 
report**, which opens a new tile draft with that chart preselected (and the
metric filter's metric, when one is chosen). Chart availability inside the
draft still follows the chosen metric's compatibility. A search hides chart
kinds without matching tiles, since searching targets existing reports.

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
workspace. Source, processor, metric, and tile cascades retain dashboard and
page containers. To remove a container, open **Report inventory → Manage
dashboards** and confirm the exact page or dashboard target separately. A page
confirmation cannot authorize another page or the whole dashboard; deleting
the last page also removes its dashboard. Aggregate Parquet files and run
history are not deleted by catalog CRUD; use the separate `valuestream vacuum`
lifecycle when persisted files are eligible for removal.

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
