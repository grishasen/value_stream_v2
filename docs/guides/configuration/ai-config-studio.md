# AI Configuration Studio

AI Configuration Studio is a governed catalog-authoring workflow. It turns a
source sample and a business goal into YAML-backed sources, processors, metrics,
reports, and chat settings. The workspace catalog remains authoritative: model
output is only a proposal, and no proposal can write configuration or run data
without separate user actions.

The Studio uses one compact progress indicator, a **Jump to step** selector,
and persistent **Back** and **Continue** actions. The final step is **Apply**.
Validation, review, workspace apply, and data loading are distinct states;
object counts alone never imply that a revision is reviewed or applied.

## Start from a sample

The cold start is in the main canvas. Choose one of three paths:

- upload a CSV, Parquet, JSON, NDJSON, gzip, or ZIP sample for an in-memory
  preview;
- choose a supported file already under `<workspace>/data`;
- choose **Try deterministic demo**, which creates a small CSV under
  `<workspace>/data/studio` so preview and runtime use the same file.

An upload remains in memory until you explicitly choose **Stage sample in
workspace**. Staging places it under `data/studio`; simply previewing an upload
does not persist raw rows.

The **Source plan** reports the sample format, runtime reader, workspace root,
exact file pattern, and runtime readiness. CSV and Parquet map to their matching
runtime readers. JSON, NDJSON, gzip, and ZIP previews use the Pega DS runtime
reader only when the schema looks like a compatible interaction export.
Otherwise the Studio marks the source **Preview only** and asks you to confirm
Pega compatibility or convert it to CSV/Parquet. Pega grouping and timestamp
defaults are never applied to a generic CSV or Parquet file.

ZIP previews must contain at least one JSON or NDJSON member. An empty or
unrelated archive is rejected with an actionable message instead of appearing
to be a valid zero-row sample.

On **Required Fields**, mappings are strict selectors over the current schema.
An unknown column name cannot be typed into a source-field mapping. Defaults,
filters, and calculated-field editors start empty; rows appear only after an
explicit add action.

## Describe the outcome

The Sample and Draft steps include **Business Requirements**. Describe the
decision or measure in plain language, such as “weekly conversion by channel
and average revenue per customer.” Requirements survive a sample switch, but
the Studio never invents fields to satisfy a requirement the approved schema
cannot support.

## Review what can be sent to AI

Field approval and sample-value sharing are separate choices. A new sample
starts with discovered fields available for schema use and no values selected
for example sharing. Identifier-like fields receive a **Likely ID** warning.

Before a model-backed draft, revision, repair, report refresh, coverage check,
or Copilot request can run, confirm **Review data sent to AI**. The checkpoint
shows the model, provider, destination class, approved schema field count, and
fields whose values will be shared. Even with examples disabled, the approved
schema can include field names, types, null counts, and unique counts, plus the
business requirements and relevant catalog settings.

Confirmation is scoped to the sample, provider, model, endpoint, approved
fields, and example-sharing choices. Changing any part of that contract clears
the confirmation and prior Copilot context. Hidden field names are redacted
from dynamic prompt material, including derived identifiers, while approved
fields with overlapping names remain intact.

## Provider preflight and bounded generation

Every AI operation begins only after its button is clicked. The Studio then
preflights the exact provider, model, endpoint, and operation capability. A
successful preflight is cached for those session settings. Missing credentials,
model access failures, and provider errors are shown in safe product language;
raw provider payloads, credentials, prompts, sample values, and local paths are
not copied into routine UI errors or logs.

Draft-producing operations use the same bounded pipeline:

1. call the model;
2. parse catalog YAML;
3. merge complete returned sections onto the accepted base;
4. validate the full catalog;
5. if needed, make at most two internal repair calls and validate again.

Only a valid candidate can enter pending review. If all three attempts fail,
the candidate is discarded and the previously accepted revision remains
unchanged. The status panel names the preflight, generation, repair, and
validation stages. Retrying is another explicit operation; the Studio does not
show a cancel control it cannot honor.

The deterministic draft uses the same validation and explicit review boundary
but never calls a provider.

## Review complete change bundles

Pending changes occupy the full canvas. Checkboxes are off by default. Studio
groups related changes into dependency-closed bundles so a processor change
travels with changed metrics that use it and report tiles that use those
metrics. The selected combination is validated again before acceptance.

Invalid bundles are disabled. **Accept safe additions** excludes every removal.
To accept a removal, choose **Review individually** and explicitly select that
removal bundle; it starts rejected. This prevents a broad safe-changes action
from deleting configuration.
Before/after YAML remains available under **Technical details**, but human
labels, summaries, and consequences lead the review.

Accepting a valid bundle combination records the exact reviewed revision
signature. Any later manual edit creates a new revision and clears that review
status. The final step offers **Mark this revision reviewed** when a valid
manually edited revision still needs explicit business review.

## Ask Copilot

Copilot is progressively disclosed under **Ask AI about this step**. It knows
the active step, approved schema, business requirements, and current revision.
Structured operations are applied only to a temporary copy, validated, and
then sent to the same bundle-review boundary.

While a proposal is pending, Copilot stays available in read-only mode. You can
ask what a bundle changes or why it matters, but returned mutation operations
are ignored and the pending proposal cannot be overwritten. Ambiguous requests
can produce quick-reply questions before any operation is attempted.

On Filters, dataset requests modify the source filter before processor fan-out;
processor filters remain a separate Processors concern. On Calculations,
Copilot uses the closed expression AST catalog rather than executable code or
free-form function strings.

## Coverage and technical details

Requirements Coverage maps business requirements to existing measures and
reports. Returned metric and tile references are checked against the current
revision; unknown references are removed and unsupported “covered” claims are
downgraded. A coverage result becomes stale when the requirements or revision
changes.

Routine views lead with friendly names and key/value summaries. Internal IDs,
raw YAML, prompts, responses, and validation detail are available only in
collapsed **Technical details** sections. YAML downloads appear before raw YAML
inspection controls.

## Apply, load data, and open the outcome

The final step never creates an implicit deterministic draft. If no accepted
revision exists, it provides a direct **Go to Draft** action.

**Apply to workspace** is enabled only when all of the following are true:

- no proposal is pending;
- the exact accepted revision validates;
- that exact revision has been explicitly reviewed;
- it is not already applied.

If the accepted revision omits sources, processors, metrics, or dashboards
that exist in the current workspace catalog, applying it would remove them.
The apply bar discloses exactly which objects the replacement removes and
blocks Apply until you confirm the removal for that specific revision. The
revision receipt then reports the removal count.

Apply writes sources, processors, metrics, dashboards, and optional `ai.yaml`
inside the rollback-protected workspace transaction. It does not ingest data.
The resulting revision receipt shows the revision key, workspace status, source
count, and whether processor computation hashes indicate a data run is needed.

If aggregate computation changed, the primary **Run data** action routes to
`/data_load?from=ai_studio`. If no computation changed, **Open report** routes
to `/reports?from=ai_studio`. Data Load owns execution, progress, retry, and
diagnostics; Studio does not combine a catalog write with ingestion.

The authoring funnel records only allowlisted workflow stages, outcomes,
durations, counts, and whether a data run is required. It never records sample
values, field names, prompts, credentials, local paths, or catalog identifiers.

## Related docs

- [Workspaces & catalog](workspaces-and-catalog.md) — catalog ownership and
  validation.
- [Pega export tutorial](../../tutorials/pega-export.md) — loading a supported
  Pega interaction archive.
- [Chat with data](../users/chat-with-data.md) — using the generated chat
  settings.
- [KPI recipes](../../reference/kpi-recipes.md) — recipe readiness,
  provenance, and materialization impact.
