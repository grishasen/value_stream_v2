# AI Configuration Studio

AI Configuration Studio is a governed catalog-authoring workflow. It turns a
source sample and a business goal into YAML-backed sources, processors, metrics,
reports, and chat settings. The workspace catalog remains authoritative: model
output is only a proposal, and no proposal can write configuration or run data
without separate user actions.

The Studio uses a compact **Data / Draft / Review / Apply** phase rail. Every
phase names its state as **Complete**, **Attention**, or **Not started**, so the
rail never relies on color alone. The step selector is scoped to the active
phase; **Back** and **Continue** remain available below the current editor.
Legacy AI/deterministic step values migrate by step number, and phase or step
jumps change navigation only—they do not clear accepted drafts, reviewed
signatures, or committed editor state. The final step is **Apply**. Validation,
review, workspace apply, and data loading are distinct states; object counts
alone never imply that a revision is reviewed or applied.

## Start from a sample

The cold start is in the main canvas. Choose one of three paths:

- upload a CSV, Parquet, JSON, NDJSON, JSON/NDJSON gzip, or JSON/NDJSON ZIP
  sample for an in-memory preview;
- choose a supported file already under `<workspace>/data`;
- choose **Try deterministic demo**, which creates a small CSV under
  `<workspace>/data/studio` so preview and runtime use the same file.

An upload remains in memory until you explicitly choose **Stage sample in
workspace**. Staging places it under `data/studio`; simply previewing an upload
does not persist raw rows.

### Preview safety limits

The Studio rejects an upload larger than 64 MiB before materializing its byte
buffer. For a larger CSV or Parquet source, put the file under
`<workspace>/data` and choose **Use workspace sample**. Convert larger JSON,
NDJSON, gzip, or ZIP inputs to CSV or Parquet first; those formats require a
bounded in-memory parser.

**Preview Rows** accepts 100 through 100,000 rows. Format-specific behavior is:

| Preview source | Row bound | Column projection | Additional bound |
|---|---|---|---|
| Workspace Parquet | lazy `head` pushdown | optional lazy `select` pushdown | no whole-file byte copy |
| Workspace CSV | native `n_rows` | all source columns | no whole-file byte copy |
| Uploaded CSV or Parquet | native row bound | all source columns | 64 MiB upload |
| JSON or NDJSON | parser stops after the requested records | all parsed fields | 64 MiB buffered input |
| gzip JSON/NDJSON | parser stops after the requested records | all parsed fields | 64 MiB compressed input and 128 MiB expanded payload |
| ZIP JSON/NDJSON | stops reading members after the requested records | all parsed fields | 64 MiB archive, 128 MiB total supported-member expansion, at most 64 supported members |

ZIP and gzip data is never extracted to disk. The ZIP central directory is
checked before a member is opened, and the actual bytes read remain capped in
case archive metadata is incorrect. Encrypted ZIP members are rejected.

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

The upload picker, workspace-file discovery, preview dispatch, source-plan
labels, and unsupported-format validation use one capability registry. XLSX is
not advertised because Studio does not yet have a matching preview/runtime
contract; an unsupported extension is rejected before payload parsing.

The repeatable preview qualification gate creates a synthetic multi-row-group
Parquet file and, when present, profiles the canonical 280,045,584-byte release
fixture. It records elapsed time, peak RSS, returned rows and columns, the
logical minimum row groups needed, and repeated cold/warm file-cache candidate
cycles as JUnit properties:

```sh
uv run pytest -q tests/benchmarks/test_ai_studio_preview.py \
  --junitxml=artifacts/ai-studio-preview.xml
```

Both profiles require a 1,000-row, three-column preview to complete within five
seconds. The synthetic peak-RSS-growth ceiling is 256 MiB; the release-fixture
ceiling is 512 MiB. Both must also stay at or below twice the bounded decoded
frame size plus 128 MiB. Five synthetic preview/garbage-collection cycles must
retain no more than 64 MiB of additional process RSS. Cache labels describe
repeated filesystem-cache candidates, not an application-level preview cache.

The same gate validates and transactionally applies a canonical one-source,
one-processor, four-metric, eight-tile draft, with a two-second ceiling for
each operation. No older committed authoring latency baseline exists, so the
JUnit `ai_studio_validation_apply_profile` establishes the measured baseline;
a later regression above 15% requires an explicitly reviewed baseline update
even when it remains inside the two-second safety ceiling.

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

## Recover interrupted authoring

Studio keeps a bounded, workspace-local recovery checkpoint for unapplied,
committed authoring state. On the next session it loads the checkpoint once,
before sample initialization, and asks you to choose **Restore Studio
checkpoint** or **Discard Studio checkpoint**. It never restores silently.

The checkpoint can contain only the workspace sample's `data/...` relative path
and identity, source mappings, defaults, filters, calculations, approved
field-name sets, the accepted catalog draft and review signature, the active
step, and the full base-catalog hash. It never contains sample or upload bytes,
sample values, prompts, pending or raw provider payloads, Copilot history,
credentials or provider settings, or AI-sharing consent receipts. Checkpoints
are atomic, size bounded, and removed after seven days.

Uploaded samples are not retained. An upload-based session can recover only its
safe accepted draft/editor metadata, and Studio requires you to reselect and
review a sample before continuing. If the catalog or selected workspace sample
changed, Restore shows **Reconciliation required**, revalidates the restored
draft, clears its prior review, and blocks navigation to a ready-to-apply state
until the revision is reviewed again.

**Discard Studio checkpoint** deletes the recovery file. A successful Apply
also deletes it, and an unchanged draft loaded from the current catalog is
treated as a clean baseline rather than unfinished work. Invalid and expired
checkpoint files are deleted instead of being partially restored.

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
successful preflight is cached for those session settings. Connectivity checks
use an independent maximum five-second timeout; failures are cached briefly to
suppress duplicate rerun calls, while an explicit Retry bypasses that negative
cache. Missing credentials, model access failures, and provider errors are
shown in safe product language. Missing configuration is rejected locally
before a provider request; raw provider payloads, credentials, prompts, sample
values, and local paths are not copied into routine UI errors or logs.

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
but never calls a provider. Its first-run baseline contains one aggregate
processor, four metrics, and a stable three-page/six-tile report set for
Engagement, Volume, and Outcomes. New dashboard, page, tile, and metric IDs are
readable and deterministic; sibling collisions use numeric suffixes. Renaming
an existing artifact preserves its ID and references.

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

Processor, metric, and tile **Keep** selections reconcile against the exact
draft revision. Removed IDs disappear, new IDs are selected by default, and an
ID explicitly rejected by the user stays rejected if it reappears. Review
labels lead with the human title and parent context while retaining the stable
technical identity.

## Apply, load data, and open the outcome

The final step never creates an implicit deterministic draft. If no accepted
revision exists, it provides a direct **Go to Draft** action.

The final step begins with one canonical **Apply readiness** summary grouped by
Data, Processor, Metric, Report, Provider, and Runtime. Each group shows a
textual state, artifact count, and last accepted revision or session change.
Every finding names the safe object/path, current safe value, expected contract,
remediation, and a **Jump to fix** action. Runtime-only conditions—such as a
preview source whose runtime reader does not match the staged plan—are labeled
explicitly. The blocker and validation-warning totals are the same evidence
used by the Apply and Export controls.

**Apply to workspace** is enabled only when all of the following are true:

- no proposal is pending;
- the exact accepted revision validates;
- that exact revision has been explicitly reviewed;
- it is not already applied.

When Apply or YAML Export is disabled, its reason appears immediately beside
the control and points back to the readiness summary. Export requires an
accepted, non-pending revision with successful catalog validation; Apply also
requires exact-revision review, runtime source readiness, and any replacement
confirmation.

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

## Headless release qualification

The deterministic release journey exercises the Studio contract without a
browser, provider, or running server. It creates a compact production-shaped
Pega Parquet fixture in pytest's temporary workspace, builds and edits a draft
through the shared authoring operations, validates and applies it, then ingests
and queries count, click-through rate, unique reach, and Channel breakdowns. It
also verifies an idempotent rerun, a clean rebuild, provenance and computation
hashes, aggregate-only persistence, lineage, and the five deterministic Chat
starters. No checked-in example data or catalog is modified.

Run the fast release gate with:

```sh
uv run pytest -q tests/integration/test_ai_studio_release_journey.py \
  -m "e2e and not slow"
```

The separate slow/read-only qualification checks the canonical full fixture's
exact path, SHA-256, row count, and outcome totals, and confirms its size and
modification time are unchanged:

```sh
uv run pytest -q tests/integration/test_ai_studio_release_journey.py \
  -m "e2e and slow"
```

## Provider failure receipts

Provider failures are normalized before they reach logs or interactive
surfaces. A receipt carries a safe category, whether the operation is
retryable, and a correlation reference. The runtime does not copy raw provider
messages, prompts, response content, credentials, sample values, or local paths
into that receipt.

The categories are `configuration`, `authentication`, `authorization`,
`rate_limit`, `timeout`, `network`, `provider`, `response_validation`, and
`internal`. Rate-limit, timeout, network, provider, and response-validation
failures are retryable. Configuration, authentication, authorization, and
unexpected internal failures require correction or investigation first.

Classification uses only the exception type and bounded status or error codes.
For example, HTTP 401 is authentication, 403 is authorization, and 429 is rate
limiting. Keep the correlation reference when escalating a failure; it links
the UI receipt to privacy-safe operational logs without exposing the provider
payload.

## Related docs

- [Workspaces & catalog](workspaces-and-catalog.md) — catalog ownership and
  validation.
- [Pega export tutorial](../../tutorials/pega-export.md) — loading a supported
  Pega interaction archive.
- [Chat with data](../users/chat-with-data.md) — using the generated chat
  settings.
- [KPI recipes](../../reference/kpi-recipes.md) — recipe readiness,
  provenance, and materialization impact.
