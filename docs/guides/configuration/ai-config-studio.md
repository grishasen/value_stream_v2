# AI Configuration Studio

The AI Configuration Studio guides source onboarding through sample review,
field approval, defaults, filters, calculations, processors, metrics, reports,
chat readiness, settings, and export. LLM-generated drafts can pre-populate
most catalog settings; review them before applying the draft.

Steps are grouped into four phases — **Data** (sample through field
approval), **Draft** (the first generated draft), **Review** (processors,
metrics, and reports), and **Publish** (chat, settings, and export). Phase
markers distinguish complete (`✓`), attention required (`!`), and empty (`○`)
states. Publish is complete only after the current validated draft is applied
to the workspace; editing the draft afterward returns Publish to attention.
Selecting a phase jumps to its first step; the step selector then shows only
that phase's steps.

Every step shows the same **Save draft** action at the upper-right
of the existing status panel. It publishes the currently accepted, validated
session draft through the full rollback-protected write path without adding a
separate vertical block. Its tooltip explains when no draft exists, AI changes
are pending review, validation fails, or that exact draft is already saved.
Controls inside review steps use
**Update ... In Draft** wording: they accept the current panel into the
session draft but do not write the workspace. This keeps draft editing and
workspace persistence visibly separate.

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

## Business Requirements

The Sample and AI Draft steps include a free-form **Business Requirements**
field. Describe what you want to measure in plain language — for example
"weekly conversion by channel and average revenue per customer". The
requirements are sent to the model together with the approved schema when
generating the AI draft and when refreshing reports, so the generated
processors, metrics, and tiles target your goals instead of a generic starter
catalog. Requirements are kept when you switch samples; requirements the
approved schema cannot support are skipped rather than guessed.

## Review Data Sharing Before AI Runs

Field approval and sample-value sharing are separate choices. A new sample
starts with all discovered fields approved for schema use and **no sample
values selected for sharing**. Identifier-like names such as `CustomerID` or
`SubjectID` are marked **Likely ID** so they receive extra scrutiny before you
opt them into example sharing.

Before any model-backed draft, revision, repair, report refresh, Copilot, or
requirements-coverage action can run, confirm **Review data sent to AI**. The
checkpoint shows the configured provider and model, every approved schema
field, whether a custom endpoint is configured, and the fields whose sample
values will be included. Even with examples off, the approved schema includes
field names, types, null counts, and unique counts. Hidden field names are not
sent; matching names are also redacted from business requirements, change
requests, prior Copilot history, validation diagnostics, and draft identifiers.
Prompts can also contain the remaining business requirements and relevant
deterministic catalog or current draft settings. Provider storage and retention
follow the terms of the configured destination.

Confirmation is scoped to the current sample and sharing contract. Loading a
different sample or changing the provider, model, approved fields, or example
sharing invalidates it, clears prior consent controls, and requires another
review. It also clears prior Copilot conversation context so echoed sample
values cannot cross into a narrower scope or a different provider. The local
**Use Deterministic Draft** action never calls a model and remains available
without AI data-sharing confirmation.

## Revise With Free-Form Change Requests

The Processors, Metrics, and Reports Review steps include an **AI Revision**
panel. Enter a free-form change request — for example "add a KPI card with
total orders to the overview page" — and the model returns only the catalog
sections it needs to replace. The revision goes through the same pending
review as generated drafts: you select what to keep, validation runs, and
nothing updates the editable draft until you accept it.

## AI Copilot

The **AI Copilot** panel remains visible beside every step and runs as an
independent Streamlit fragment, so ordinary dialogue does not rerun the main
editor. It knows the current step, business requirements, approved schema,
and accepted draft. Ask a question or request a change in free form; the
copilot answers with a short reply and, when you asked for a change, governed
operations for source defaults, dataset filters, calculated fields, processors, metrics,
built-in KPI recipes, and report tiles. For example, "set the
ModelControlGroup default to Test" creates a `set_source_default` operation;
"calculate Margin from Revenue minus Cost" creates a validated
`derive_column` expression rather than free-form YAML.

On the **Calculations** step, the Copilot receives the complete closed
expression-DSL catalog, including every supported operator, its exact AST
shape, allowed date units and cast types, and nested examples. Concatenation is
an AST operation, not a function-call string: for example,
`{op: concat, args: [{col: Issue}, {col: Group}], sep: "/"}`. Non-string
inputs can be nested inside `op: cast` before concatenation. The prompt must
not report an operator as unavailable when it appears in this catalog; the
resulting `derive_column` still passes normal model and catalog validation
before it can enter patch review.

On the **Filters** step, dataset requests use `set_source_filter` or
`remove_source_filter`. They create a source `kind: filter` transform in
`pipelines.yaml`, before processor fan-out, and therefore affect every
processor bound to that source. A processor-level filter remains a separate
Processors-step concern. If a model attempts `set_processor` for a Filters-step
request, the governed loop rejects it and asks the model to correct the
operation before anything can enter patch review.

The operation loop is bounded to three model calls. It applies operations to
a temporary copy, validates that copy with the catalog validator, and sends
operation or validation errors back to the model for correction. Only a valid
result becomes pending review. Each structural change then appears as its own
Accept checkbox with before/after YAML. Reject preserves the accepted draft's
previous definition; it never deletes a changed object as a side effect.
While patches are pending, the copilot input is disabled so a later request
cannot overwrite unreviewed work. When a request is ambiguous, the copilot
asks a clarifying question with quick-reply options before executing tools.
Accepted source-default, source-filter, and calculated-field patches are also synchronized
back into the Defaults, Filters, and Calculations row editors on the next full rerun, so
the visual preprocessing controls and accepted `pipelines.yaml` draft remain
the same source of truth. Removing any of these definitions uses the same governed
operation and patch review path.
The conversation and draft reset when a different sample file is loaded,
including a file with the same column names; business requirements remain.
If the configured provider rejects a request for insufficient permissions,
the Studio identifies the selected model and points to **AI Settings** or the
provider's project/key permissions. The attempted prompt remains available in
**Last prompt**, and no draft operation is applied.

## Requirements Coverage

The Metrics Review, Reports Review, and Save & Export steps include a
**Requirements Coverage** panel. **Check Coverage** asks the model to split
the business requirements into individual requirements and judge each one
against the current draft: covered, partial, or missing, with the metric ids
and tile keys that cover it. Returned references are checked against the
draft; unknown metric or tile ids are removed, and an unsupported covered or
partial judgement is downgraded to missing. A warning appears when the draft
or requirements changed after the last check. Each uncovered requirement
gets an **Ask Copilot To Cover** shortcut that sends it to the copilot as a
change request.

## Review Before Applying

Use it as a drafting workflow. Review generated YAML or individual structural
patches before applying them, then validate the catalog and run the workspace.

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
- Use **Save Draft & Run Source** on Save & Export when a reviewed recipe
  introduces processor state that must be materialized. The top **Save draft
  to workspace** action changes configuration only and never starts ingestion.

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
