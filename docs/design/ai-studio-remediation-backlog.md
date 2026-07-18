# Configuration Studios Remediation Backlog

This backlog turns the 2026-07-18 AI Configuration Studio and Configuration
Builder audits into one implementation-ready plan. Its target outcome is a
safe, deterministic, no-provider walkthrough in which an analyst can load the
`examples/test_ai_studio` parquet fixture, create calculated fields,
processors, metrics, and three report pages, apply a valid catalog, ingest the
full source, and use the resulting reports without editing raw YAML or
recovering from lost UI state. The same draft, identity, validation, CRUD, and
accessibility contracts apply when editing an existing workspace through
Configuration Builder.

The audit exercised all 14 Studio steps against
`examples/test_ai_studio/data/Month=08/Day=2024-08-31/934be6678a7948e7b10c1cca2f5299fa-0.parquet`.
The final repaired catalog processed 2,733,856 rows, rendered three pages and
eight tiles, skipped an idempotent rerun, and completed a clean rebuild. During
the original audit the Studio-produced catalog was not executable without
manual correction; the implementation recorded below removes that blocker.

This page is the source of truth for remediation of audit findings
`AI-STUDIO-001` through `AI-STUDIO-021` and Configuration Builder findings
`BUG-1` through `BUG-7` plus its 23 historical backlog items. The original
[Configuration Editor QA report](config-editor-qa-backlog.md) remains evidence,
not a second implementation backlog. New implementation work, dependency
changes, priority changes, and completion status for either Studio are recorded
only on this page; the historical report is not maintained as a parallel
delivery plan. The
[Reporting backlog](reporting-backlog.md) remains authoritative for the full
RPT feature scope, and the [KPI recipe backlog](kpi-recipe-backlog.md) remains
authoritative for the full KPR feature scope. Cross-references below identify
shared delivery rather than duplicate requirements.

## Implementation Status and Evidence

**Implementation status: complete (2026-07-18).** All findings in the audit
coverage tables below have been implemented in the shared catalog contracts,
both authoring surfaces, Reports, Chat, checkpoint/recovery services, and their
documentation. The historical Configuration Editor backlog is merged by
explicit finding-to-story mapping rather than duplicated.

| Evidence | Result |
|---|---|
| Unit and Streamlit AppTest regression suite | **Pass:** 1,079 tests |
| Isolated release journey and preview qualification | **Pass:** 5 tests |
| Canonical fixture identity | **Pass:** SHA-256 `ff54074bfffff95b38f8768a23671fdf9b9b443698f5bab275d82a858e4247c5`; 2,733,856 rows |
| Canonical aggregate queries | **Pass:** count 2,733,856; CTR `0.0178539030585371`; approximate reach `858055.1512611847`; Channel split verified |
| Runtime contracts | **Pass:** full ingest, provenance/config hashes, aggregate-only persistence, zero-row idempotent rerun, and clean rebuild |
| Preview budgets | **Pass:** canonical p95 0.00239 s; peak RSS increase 9.8 MiB; post-GC increase 9.3 MiB; validation 0.00383 s; Apply 0.01086 s |
| Static and documentation gates | **Pass:** Ruff, format check, `git diff --check`, and strict MkDocs build |
| Browser accessibility release proof | **Evidence pending:** AppTests cover the remediated UI contracts, but the final localhost browser run was denied by the in-app browser URL policy. Re-run the AIS-508 axe, keyboard, zoom, and target-width matrix in an approved browser environment before making the accessibility release claim. |

The temporary qualification server was stopped and port 8501 was verified no
longer listening after the test run. No release test mutated the canonical
example workspace.

## Guardrails

Every story must preserve the core platform invariants:

1. Value Stream remains aggregate-first; reports, previews of materialized
   results, and Chat query through the aggregate/query layer.
2. Raw rows do not persist beyond bounded sample handling and chunk processing.
3. Validated YAML remains the source of runtime behavior.
4. Ingestion remains deterministic and idempotent, and persisted aggregates
   retain provenance and configuration hashes.
5. Studio draft mutations remain session- or draft-local until a reviewed,
   atomic Apply transaction succeeds.
6. Configuration Builder and AI Configuration Studio share contracts and
   authoring components where they represent the same catalog property.

Provider credential provisioning, multi-user identity/OIDC, and a broad
Reports redesign are outside this backlog.

## Priority, Size, and Ownership

| Label | Meaning |
|---|---|
| P0 | Required before the Configuration Studios can be presented as safe, unattended end-to-end workflows. |
| P1 | Required in the next product release for a trustworthy supported workflow. |
| P2 | Scheduled usability or maintainability improvement after the supported flow is reliable. |
| P3 | Polish with a contained workaround and no correctness impact. |
| S | 1–2 engineering days, including focused tests and documentation. |
| M | 3–5 engineering days, including focused tests and documentation. |
| L | 6–10 engineering days; split before implementation when parallel seams are clear. |
| XL | 11–20 engineering days and must be decomposed during refinement. |

Suggested owners name the primary accountable group, not the only contributor:
**Studio UX**, **Catalog/Runtime**, **AI Runtime**, **Reports**, or **Quality**.

## Release Outcomes and Gates

| Outcome | Release gate |
|---|---|
| Intentional consent | No non-input element can change provider-sharing consent; consent is tied to the exact sharing contract. |
| Executable first apply | The canonical deterministic draft validates and ingests without a manual YAML correction or invented default. |
| Correct business semantics | Required fields and outcome classes are confirmed from observed schema/value evidence; all observed outcomes are covered or explicitly excluded. |
| Lossless authoring | Add, modify, delete, mode switch, navigation, rerun, and recovery tests preserve committed work and protect dirty work. |
| No-provider completion | A user creates a processor, four metrics, three calculated fields, three pages, and at least six tiles visually, with provider actions unavailable. |
| Bounded preview | Workspace Parquet preview never reads the whole 280 MB fixture into a Python byte buffer and stays below the performance budget in AIS-404. |
| Resilient AI | Missing or failing provider access returns actionable status within the configured preflight/timeout budget and cannot terminate the app or discard the draft. |
| Accessible release | The canonical browser journey has no critical/serious automated accessibility findings and is keyboard-completable at 1440 px and 1024 px widths. |
| Runtime proof | Full ingest, aggregate queries, reports, idempotent rerun, and clean rebuild pass against the canonical fixture. |

The implementation blockers AIS-001, AIS-002, AIS-003, AIS-004, AIS-005,
AIS-006, AIS-007, AIS-101A, AIS-101B, AIS-109, AIS-301, AIS-401, AIS-404,
AIS-506A, and AIS-509 are complete. The only outstanding release evidence is
the browser-only accessibility matrix identified above; it is not an open
implementation story.

## Delivery Map

| Increment | Product outcome | Stories |
|---|---|---|
| A | Safe consent and an executable, semantically correct catalog | AIS-001–AIS-008 |
| B | Editors and validation that never silently lose or misstate work | AIS-101A, AIS-101B, AIS-101C, AIS-102–AIS-110 |
| C | Complete visual creation without AI or raw YAML | AIS-201–AIS-207 |
| D | Fast, consistent AI failure handling and deterministic Chat fallback | AIS-301–AIS-304 |
| E | Bounded, observable sample preview | AIS-401, AIS-402, AIS-403A, AIS-403B, AIS-404 |
| F | Shorter workflow, report clarity, accessibility, and release qualification | AIS-501–AIS-503, AIS-504A, AIS-504B, AIS-505, AIS-506A, AIS-506B, AIS-507–AIS-511 |

Suffix stories such as AIS-101A, AIS-403B, AIS-504A, and AIS-506B are
independently deliverable stories. Their former unsuffixed IDs are retired and
must not be used for dependencies, completion status, or release evidence.

## Audit Coverage

| Audit finding | Remediation stories |
|---|---|
| AI-STUDIO-001 — Help mutates consent | AIS-001, AIS-002 |
| AI-STUDIO-002 — duplicate variant grouping | AIS-003, AIS-202, AIS-509 |
| AI-STUDIO-003 — Outcome maps to a time field | AIS-004, AIS-509 |
| AI-STUDIO-004 — timestamp failure is not actionable | AIS-005, AIS-104, AIS-503 |
| AI-STUDIO-005 — wrong outcome semantics | AIS-006, AIS-202 |
| AI-STUDIO-006 — invalid KPI recipe survives until Apply | AIS-007, AIS-203, AIS-509 |
| AI-STUDIO-007 — slow/generic AI failure and session loss | AIS-301–AIS-303, AIS-404, AIS-509 |
| AI-STUDIO-008 — eager full-file preview | AIS-401, AIS-402, AIS-403A, AIS-403B, AIS-404 |
| AI-STUDIO-009 — inconsistent toggle hit targets | AIS-103, AIS-508 |
| AI-STUDIO-010 — editor changes are discarded | AIS-101A–AIS-101C, AIS-102, AIS-504A |
| AI-STUDIO-011 — core creation requires AI/YAML | AIS-201–AIS-207, AIS-509 |
| AI-STUDIO-012 — contradictory validation status | AIS-104, AIS-503 |
| AI-STUDIO-013 — hard-coded workspace | AIS-106 |
| AI-STUDIO-014 — unstable “deterministic” IDs | AIS-106, AIS-507 |
| AI-STUDIO-015 — stale Keep selectors | AIS-105 |
| AI-STUDIO-016 — Chat has no local fallback | AIS-304 |
| AI-STUDIO-017 — unclear Select all behavior | AIS-505 |
| AI-STUDIO-018 — derived date fields cannot be mapped | AIS-107 |
| AI-STUDIO-019 — high-cardinality core dimensions | AIS-207 |
| AI-STUDIO-020 — XLSX contract mismatch | AIS-108 |
| AI-STUDIO-021 — stale report anchors | AIS-507 |

### Configuration Builder audit coverage

| Builder finding | Unified remediation |
|---|---|
| BUG-1 / A1 — calculated fields never dirty the draft | AIS-101A, AIS-509 |
| BUG-2 / A2 — untouched templates and phantom drafts | AIS-101B, AIS-509 |
| A4 — Continue disappears while dirty | AIS-101C, AIS-501 |
| BUG-3 / A3 — Tile Delete targets another selection | AIS-109 |
| BUG-4 / B1 — historical reports open as all `n/a` | AIS-506A, AIS-506B |
| BUG-5 / B2 — hashed IDs and technical toasts | AIS-106, AIS-110, AIS-203 |
| B3 — ambiguous source/processor selector identity | AIS-110 |
| C1–C3 — asymmetric Source/Processor/Metric CRUD | AIS-109, AIS-201–AIS-203 |
| D1/D2 — single-line expressions and disabled new rows | AIS-103, AIS-504A |
| D3 — raw validation internals | AIS-104, AIS-503 |
| D4/D5 — raw JSON and technical chart names | AIS-205, AIS-504B |
| D6 — reload loses step/draft | AIS-303, AIS-501 |
| BUG-6 / E1 — Home omits Build | AIS-511 |
| E2 — catalog/directory identity mismatch | AIS-106, AIS-110 |
| E3 — template invents `SubjectID` | AIS-004, AIS-202 |
| BUG-7 / E4 — duplicate widget state warning | AIS-103 |
| F1/F2 — repeated synchronous source reads without feedback | AIS-401, AIS-403A, AIS-403B |
| F3 — missing draft-lifecycle AppTests | AIS-101A–AIS-101C, AIS-109, AIS-509 |

## Increment A — Consent and Catalog Correctness

### AIS-001 — Isolate Help from provider-sharing consent

**Priority / size / owner:** P0 / S / Studio UX
**Audit:** AI-STUDIO-001
**Dependencies:** none

**User outcome:** Reading help can never authorize external data sharing.

**Acceptance criteria:**

- Help is a separate button/popover outside the checkbox label and input hit
  area; captions, warnings, expansion controls, and surrounding whitespace are
  also non-mutating.
- Consent changes only through explicit pointer or keyboard activation of the
  named confirmation input.
- All provider actions remain blocked when the current sharing-contract
  signature is unconfirmed.
- Focus order and visible focus make Help and Confirm distinguishable.

**Implementation and verification:** Refactor
`_render_ai_data_sharing_confirmation()` in
`src/valuestream/ui/pages/ai_config_studio.py`. Add a real-browser regression
that clicks Help and every adjacent target, then proves consent remains false;
add a keyboard confirmation case.

### AIS-002 — Record consent as an explicit, scoped event

**Priority / size / owner:** P0 / M / Studio UX + AI Runtime
**Audit:** AI-STUDIO-001
**Dependencies:** AIS-001

**User outcome:** Authorization is auditable and bound to exactly what will be
shared.

**Acceptance criteria:**

- `CONSENT_CONFIRMED` is recorded only on an explicit false-to-true input
  change, not merely because a widget renders with a true value.
- The event includes sharing-contract signature, provider/model identity,
  timestamp, and interaction source without raw values or secrets.
- Changing sample identity, provider/model, approved fields, or sample-value
  scope revokes consent immediately and requires reconfirmation.
- Back navigation and reruns do not manufacture a new consent event.

**Verification:** Extend consent/audit coverage in
`tests/unit/test_ai_studio_helpers.py` for render, Help, explicit confirm,
revoke, contract change, back navigation, rerun, and keyboard activation.

### AIS-003 — Enforce unique processor grouping keys

**Priority / size / owner:** P0 / M / Catalog/Runtime
**Audit:** AI-STUDIO-002
**Dependencies:** none

**User outcome:** A catalog reported as valid cannot fail because the variant
column is also a dimension.

**Acceptance criteria:**

- Cross-catalog validation reports a path-specific error when normalized
  `variant_column` is also present in `group_by`/legacy `dimensions`.
- Selecting a variant in the Studio processor editor removes it from the
  Dimensions selection or blocks Save with an explanation.
- `binary_outcome` defensively deduplicates runtime group keys, so legacy
  catalogs cannot raise a Polars duplicate-name exception.
- The variant remains persisted once, with the same query and provenance
  semantics as a valid current catalog.

**Implementation and verification:** Update
`src/valuestream/config/validate.py`,
`src/valuestream/processors/binary_outcome.py`, and the processor editor in
`src/valuestream/ui/pages/ai_config_studio.py`. Add negative loader coverage in
`tests/unit/test_config_loader.py`, runtime defense in
`tests/unit/test_binary_outcome_processor.py`, and an ingestion regression in
`tests/integration/test_phase1_pipeline.py`.

### AIS-004 — Replace substring field mapping with evidence-based role matching

**Priority / size / owner:** P0 / M / Studio UX + Catalog/Runtime
**Audit:** AI-STUDIO-003
**Dependencies:** AIS-401

**User outcome:** Subject, outcome, and time roles default to credible fields
and ambiguous matches require review.

**Acceptance criteria:**

- Candidate scoring considers exact normalized raw name, post-transform name,
  reviewed aliases, dtype, null rate, and cardinality.
- Time/date-like fields are excluded from Outcome candidates; categorical
  compatibility is required unless the user explicitly overrides it.
- `pyOutcome`/`Outcome` outranks `pxOutcomeTime` regardless of source column
  order.
- Close or incompatible scores leave the role unmapped, show the evidence, and
  block dependent continuation until confirmed.
- A manual choice remains stable across reruns unless the field disappears.

**Verification:** Extend `tests/unit/test_ai_studio_helpers.py` with exact,
alias, ambiguous, numeric, timestamp-exclusion, reordered-column, and manual
override cases using fixture-like schemas.

### AIS-005 — Infer timestamp formats and gate invalid preprocessing

**Priority / size / owner:** P0 / M / Studio UX + Catalog/Runtime
**Audit:** AI-STUDIO-004
**Dependencies:** AIS-401, AIS-104

**User outcome:** The user can repair a time parse at the failing field before
dependent work begins.

**Acceptance criteria:**

- A bounded detector identifies supported formats, including
  `%Y%m%dT%H%M%S%.3f %Z`, and shows the matching redacted examples and
  confidence.
- Pega-like Parquet uses the same inference contract as the other supported
  inputs instead of leaving the format empty.
- Failure identifies field, observed dtype, attempted format, representative
  redacted value, and at least one correction where one is known.
- Continue and direct phase jumps are blocked only when the destination
  depends on failed preprocessing; the blocking reason is beside navigation.
- Applying a correction reruns the bounded preview and clears the issue
  without restarting the journey.

**Verification:** Add source-plan and working-sample tests, an AppTest
navigation case, and a probe-style integration fixture with five timestamps in
the audited format.

### AIS-006 — Require complete, intent-aware outcome classification

**Priority / size / owner:** P0 / M / Studio UX + Catalog/Runtime
**Audit:** AI-STUDIO-005
**Dependencies:** AIS-004, AIS-401

**User outcome:** Generated metrics use observed business outcomes rather than
hard-coded or nonexistent values.

**Acceptance criteria:**

- The processor wizard shows each observed non-null outcome, sample count, and
  proposed Positive, Negative, or Explicitly excluded classification.
- For click-through intent, `Clicked` is positive and `Impression`,
  `NoConversion`, and `Conversion` are negative for the canonical fixture;
  `Pending` is never invented.
- Unclassified observed values block processor acceptance. Explicit exclusions
  require a visible justification and remain in YAML provenance/description.
- Case variants, numeric outcomes, nulls, and goals other than click-through
  use deterministic documented rules or require user confirmation.
- Separate click and conversion goals do not silently collapse into one
  processor meaning.

**Verification:** Add canonical four-value, case, numeric, null, exclusion, and
non-click goal cases to `tests/unit/test_ai_studio_helpers.py`, plus an
aggregate-result assertion in the golden journey.

### AIS-007 — Validate every generated draft mutation before commit

**Priority / size / owner:** P0 / M / Catalog/Runtime
**Audit:** AI-STUDIO-006
**Dependencies:** KPR-101, KPR-104; coordinates with RPT-004–RPT-006 and RPT-105

**User outcome:** A recipe, Copilot, deterministic generator, or compact editor
cannot add schema-invalid YAML that first fails at Apply.

**Acceptance criteria:**

- Each mutation builds and validates the complete candidate catalog before
  replacing the accepted draft revision.
- A failure leaves the prior revision byte-for-byte equivalent and displays
  the exact catalog path, invalid value, expected contract, and remediation.
- Recipe report KPI fields use typed models; `value_format` is canonicalized at
  tile level and is rejected under `kpi`.
- Raw YAML may remain open for advanced repair, but Update never labels invalid
  content accepted and cannot bypass final Apply validation.
- The same transaction/diff semantics are shared with Configuration Builder
  rather than reimplemented in the Studio.

**Implementation and verification:** Tighten
`src/valuestream/ai/copilot.py`, `src/valuestream/recipes/kpi.py`, and the
Studio draft mutation helpers. Strengthen `tests/unit/test_kpi_recipes.py`,
`tests/unit/test_ai_copilot.py`, and `tests/unit/test_ai_studio_helpers.py` with
malformed nested KPI, rollback, complete-dashboard, and round-trip cases.

### AIS-008 — Validate transforms against the observed source schema

**Priority / size / owner:** P1 / M / Catalog/Runtime
**Audit:** supporting audit observation: `Propensity: 0` was required only to
seed validation
**Dependencies:** AIS-401

**User outcome:** A valid calculated field does not require a fake missing-value
default merely to satisfy validation.

**Acceptance criteria:**

- Semantic validation accepts an optional authoritative, typed schema per
  source while preserving conservative CLI behavior when no observed schema is
  available.
- Studio validation supplies the bounded raw schema and distinguishes observed
  evidence from catalog declarations.
- Validation cache keys include sample identity and schema signature.
- `PropensityPct` validates from observed `Propensity` without adding a default;
  a genuinely absent input remains an error.

**Verification:** Add authoritative/provisional schema cases to
`tests/unit/test_config_loader.py` and sample/schema cache invalidation cases to
`tests/unit/test_ai_studio_helpers.py`.

### Increment A acceptance

- The canonical draft contains no duplicate grouping key, invalid KPI field,
  nonexistent outcome, or fake schema-seeding default.
- Static validation and the first full source run agree on executability.
- Consent cannot be granted through Help, rerendering, or a scope change.
- Mapping, outcome, timestamp, and schema errors are fixed before dependent
  work, not at Apply or ingestion.

## Increment B — Trustworthy Editor State and Validation

### AIS-101A — Propagate editor changes into draft state

**Priority / size / owner:** P0 / M / Studio UX
**Audit:** AI-STUDIO-010; Configuration Builder BUG-1/A1
**Dependencies:** none

**Acceptance criteria:**

- Defaults, filters, and calculated-field row editors recompute the owning
  source revision in the same rerun; fragment boundaries cannot leave the
  visible preview newer than the draft hash.
- Add, edit, enable/disable, and delete immediately produce the correct dirty
  state and exact generated YAML.
- Applying a calculated field writes its `derive_column` transform; applying a
  default or filter writes the same content shown in preview.
- Focused AppTests fail on the historical nested-fragment/equivalence-gate bug
  and pass after the fix.

### AIS-101B — Keep create/apply/discard baselines clean

**Priority / size / owner:** P0 / L / Studio UX
**Audit:** Configuration Builder BUG-2/A2
**Dependencies:** AIS-101A

**Acceptance criteria:**

- A newly initialized Source, Processor, Metric, Dashboard, Page, or Tile
  template is its clean baseline and is not registered as a draft until the
  canonical content changes.
- Discard restores that baseline once and stays clean; a successful Apply
  re-baselines against the applied object and purges empty registry entries.
- Continue advances on the first click even when the primary action label or
  handler changed during the prior rerun.
- The unapplied-draft count includes only non-empty diffs and remains correct
  after create, discard, Apply, navigation, and rerun.

### AIS-101C — Share Save, Cancel, Undo, and navigation guards

**Priority / size / owner:** P1 / L / Studio UX
**Audit:** AI-STUDIO-010; Configuration Builder A4
**Dependencies:** AIS-101A, AIS-101B

**Acceptance criteria:**

- Defaults, filters, calculations, field approval, processors, metrics,
  reports, page filters, and settings edit a local working copy.
- Dirty state is visible. Save validates and commits; Cancel restores the last
  committed copy; a just-completed deletion has Undo.
- Apply and Continue without applying remain simultaneously available when a
  valid draft exists, with Apply first in tab order.
- Navigation with dirty state offers Save, Discard, or Stay. Row selection,
  sorting, search, fragment reruns, and unrelated widgets never mutate values.
- Shared patterns use `builder_draft_status`, editor save bars, and validated
  draft actions in both Studios.

### AIS-102 — Make Rules and Raw AST switching lossless

**Priority / size / owner:** P1 / M / Studio UX + Catalog/Runtime
**Audit:** AI-STUDIO-010
**Dependencies:** AIS-101C

**Acceptance criteria:**

- Rules and Raw AST retain separate dirty buffers until explicit Save.
- Rules-to-AST compiles and previews the exact expression.
- AST-to-Rules converts only when representable; otherwise the user chooses
  Keep Raw, Replace with Rules, or Cancel after seeing the loss.
- A mode switch alone never commits or erases either representation.
- Round-trippable and non-round-trippable DSL expressions have unit and browser
  coverage.

### AIS-103 — Standardize toggle and checkbox interaction contracts

**Priority / size / owner:** P1 / M / Studio UX
**Audit:** AI-STUDIO-009
**Dependencies:** AIS-001

**Acceptance criteria:**

- The visible label, switch track, thumb, and keyboard input operate the same
  native control; duplicate Markdown labels and label-hidden controls are
  removed.
- State survives fragment and page reruns.
- Rename, Capitalize, calculation Enabled, Reports Advanced, confirmation, and
  consent toggles share accessible naming, focus, and hit-target behavior.
- Widgets are initialized through one seed-only state pattern; entering either
  Studio emits no Streamlit warning about passing a default while also setting
  the same widget key through Session State.
- Browser tests exercise pointer and Space/Enter activation on each contract
  class.

### AIS-104 — Create one canonical UI issue model

**Priority / size / owner:** P1 / L / Studio UX + Catalog/Runtime
**Audit:** AI-STUDIO-004, AI-STUDIO-012
**Dependencies:** none; provides the shared issue contract consumed by
AIS-003–AIS-008 and later authoring stories

**Acceptance criteria:**

- Every issue has stable ID, severity, catalog path, phase/editor owner,
  concise message, safe technical detail, remediation, and optional focus
  target.
- Catalog Health, revision status, phase badges, Apply gating, and ingestion
  preflight consume the same issue collection.
- Errors block the dependent action; warnings never count as blocking.
- Counts cannot disagree across header, revision, and Apply views.
- Fix actions focus the owning field or open the correct editor.

**Verification:** Add severity/count/gating contract tests and reproduce the
audited “OK / 0 warnings / 1 issue” and “2 blocking” contradictions as negative
fixtures.

### AIS-105 — Reconcile Keep selectors on every draft revision

**Priority / size / owner:** P2 / M / Studio UX
**Audit:** AI-STUDIO-015
**Dependencies:** AIS-101C, AIS-104

**Acceptance criteria:**

- Keep-list state reconciles whenever the accepted draft signature changes.
- New artifacts default to selected unless the user explicitly rejected that
  exact stable ID; removed IDs disappear from widget state.
- Labels lead with human title, type, and parent context; the technical ID is
  secondary.
- Recipe addition, raw YAML update, page switch, and step re-entry update the
  list immediately and deterministically.

### AIS-106 — Preserve active workspace and generate stable IDs

**Priority / size / owner:** P1 / M / Studio UX + Catalog/Runtime
**Audit:** AI-STUDIO-013, AI-STUDIO-014
**Dependencies:** AIS-007

**Acceptance criteria:**

- Deterministic draft and Copilot baseline creation receive the active
  workspace explicitly; `test_ai_studio` remains stable through revision,
  export, Apply, and reload.
- New IDs derive from artifact type, parent stable ID, and normalized semantic
  identity. Collisions use deterministic numeric suffixes, not random hex.
- Editing titles preserves existing IDs and references.
- Identical generation inputs produce identical IDs and a stable diff.
- Existing catalogs retain their IDs; migration is opt-in and reference-safe.

**Verification:** Replace the current unit expectation for random suffixes with
stable-generation, collision, edit, reload, and backward-compatibility cases.

### AIS-107 — Expose derived fields at the correct mapping stage

**Priority / size / owner:** P2 / S / Studio UX
**Audit:** AI-STUDIO-018
**Dependencies:** AIS-004, AIS-005

**Acceptance criteria:**

- A mapping selector explicitly distinguishes raw, renamed, and derived fields.
- Day, Month, Quarter, and Year are selectable only after their time transform
  is valid, or the UI explains why they are not valid for the requested role.
- Search results and validation use the same working-schema inventory.
- A mapping cannot reference a derived field before its dependency exists.

### AIS-108 — Unify reader and upload format capabilities

**Priority / size / owner:** P2 / S / Studio UX + Catalog/Runtime
**Audit:** AI-STUDIO-020
**Dependencies:** none; AIS-402 consumes this capability registry

**Acceptance criteria:**

- Reader selectors, file pickers, upload `accept` values, help text, and
  validation derive from one supported-format registry.
- XLSX is either accepted and previewed end to end or absent from every entry
  point until supported; the UI never advertises contradictory contracts.
- Each supported compression/container combination has a positive contract
  test and unsupported combinations fail before upload processing.

### AIS-109 — Make every destructive action target-explicit

**Priority / size / owner:** P0 / M / Studio UX
**Audit:** Configuration Builder BUG-3/A3 and CRUD concerns
**Dependencies:** none

**Acceptance criteria:**

- Delete derives its target from the editor state displayed beside the action,
  never a remote or stale library selection.
- Button, staged notice, and confirmation name the display title, stable ID,
  dashboard/page parent where relevant, and dependent artifacts.
- Processor and Metric deletion preview dependent metrics/tiles and aggregate
  cleanup consequences; cascade is explicit and transactionally validated.
- Stage, Apply, Discard, and Undo retain the current rollback contract.
- Focused AppTests cover a different library selection and editor selection,
  then prove only the named object can be staged.

### AIS-110 — Present artifact identity according to task context

**Priority / size / owner:** P1 / S / Studio UX
**Audit:** Configuration Builder BUG-5/B2, B3, E2
**Dependencies:** coordinates with AIS-106

**Acceptance criteria:**

- Editing selectors use `id — title or description · kind`, because the ID is
  the operative identity; review/Keep lists lead with human title and parent
  context, with stable ID second.
- Toasts and receipts use display names; technical detail retains the ID.
- From-scratch Metric creation proposes a complete readable slug, lets the user
  edit it before Apply, and adds deterministic numeric collision suffixes.
- The workspace shell shows catalog name plus directory basename.
- Both Studios use one shared formatter and have duplicate-name/collision tests.

### Increment B acceptance

- Every editable section supports add, modify, delete, Save, Cancel, and
  appropriate Undo without silent state loss.
- Validation severity, counts, wording, and gating are consistent everywhere.
- A draft round trip preserves workspace, IDs, filters, calculations, metrics,
  dashboard structure, and report properties.

## Increment C — Complete Guided Visual Authoring

The Configuration Builder already contains recipe/from-scratch metric and
dashboard/page/tile patterns. These stories extract shared, draft-targeted
components instead of creating a second catalog implementation.

### AIS-201 — Add a shared Create menu and draft mutation adapter

**Priority / size / owner:** P1 / L / Studio UX + Catalog/Runtime
**Audit:** AI-STUDIO-011
**Dependencies:** AIS-007, AIS-101C, AIS-104; coordinates with KPR-101 and RPT-004

**Acceptance criteria:**

- Source, Processor, Metric, Dashboard, Page, and Tile lists and empty states
  expose a primary Add action. Source creation can hand off to Start from a
  sample and return to the same Builder step.
- Shared authoring components target either a Studio draft transaction or a
  Configuration Builder catalog transaction through one typed adapter.
- No Studio create/edit/delete action writes live YAML before Apply.
- Each accepted mutation immediately updates inventory, dependencies, Keep
  selectors, validation, and the reviewed diff.
- Raw YAML is labeled Advanced and is never the sole no-provider route.

### AIS-202 — Add a visual Processor wizard

**Priority / size / owner:** P1 / L / Studio UX + Catalog/Runtime
**Audit:** AI-STUDIO-002, AI-STUDIO-005, AI-STUDIO-011
**Dependencies:** AIS-003, AIS-004, AIS-006, AIS-109, AIS-201

**Acceptance criteria:**

- The wizard covers source, kind, subject, outcome, positive/negative/excluded
  values, time, dimensions, variant, states, and description.
- Observed value counts and dimension cardinality are visible beside decisions.
- Variant/dimension overlap and uncovered outcomes are impossible to Save.
- Add, edit, duplicate, disable where supported, and delete show dependent
  metrics/tiles and support cancellation/undo.
- The generated processor validates against the complete draft before commit.

### AIS-203 — Add recipe and from-scratch Metric creation

**Priority / size / owner:** P1 / L / Studio UX + Catalog/Runtime
**Audit:** AI-STUDIO-006, AI-STUDIO-011
**Dependencies:** AIS-007, AIS-109, AIS-201; coordinates with KPR-101, KPR-103,
KPR-105, KPR-107, and RPT-105

**Acceptance criteria:**

- Users can browse a recipe or create supported metric kinds from scratch.
- The form exposes source processor, calculation/states, grouping support,
  output shape, format, description, and materialization readiness.
- Display metadata is written only to schema-valid locations.
- Near-duplicate metrics offer Reuse without replacing an explicit create
  choice.
- Optional Add to report supports metric-only, existing page, or create-new-page
  placement with a reviewed diff.
- Edit, duplicate, and delete use the same dependency preview; a referenced
  metric is blocked or removes its named tiles only after explicit cascade
  confirmation.

### AIS-204 — Add Dashboard and Page structure management

**Priority / size / owner:** P1 / L / Studio UX + Reports
**Audit:** AI-STUDIO-011, AI-STUDIO-014
**Dependencies:** AIS-106, AIS-109, AIS-201; coordinates with RPT-004–RPT-006

**Acceptance criteria:**

- Users can add, rename, duplicate, reorder, and delete dashboards and pages.
- The structure view leads with human titles and shows stable IDs second.
- Delete previews dependent tiles/filters, requires confirmation, and supports
  Undo while the mutation remains the latest draft change.
- Existing layout, theme, filter, time-filter, and unknown backward-compatible
  properties survive edits and round trips.
- Empty pages offer Add Tile and Configure Filters.

### AIS-205 — Add a schema-aware Tile composer with live preview

**Priority / size / owner:** P1 / L / Studio UX + Reports
**Audit:** AI-STUDIO-011
**Dependencies:** AIS-109, AIS-203, AIS-204; coordinates with RPT-105 and KPR-105

**Acceptance criteria:**

- The composer supports metric, chart, dimensions, time grain, placement,
  title, description, formatting, KPI comparison, and chart-specific settings.
- Options come from the chart catalog and selected metric/processor capability;
  invalid combinations are not offered or are explained inline.
- Preview uses the same aggregate resolution/query/chart path as Reports and
  shows provenance/freshness when materialized data exists.
- Users can add, edit, duplicate, move, and delete tiles without raw YAML.
- Full dashboard properties survive Apply and reload.

### AIS-206 — Add schema-aware page-filter authoring

**Priority / size / owner:** P1 / M / Studio UX + Reports
**Audit:** AI-STUDIO-011, AI-STUDIO-017
**Dependencies:** AIS-204, AIS-205; coordinates with RPT-203, RPT-206, RPT-207

**Acceptance criteria:**

- Filter fields come from persisted processor dimensions rather than free text.
- Each candidate shows cardinality and compatible tile coverage.
- `all_tiles` cannot be saved when any selected tile lacks the aggregate field;
  compatible/unsupported tiles are previewed before commit.
- Authoring and runtime Reports use one `FilterCapability` contract.
- Add, edit, reorder, and delete round-trip label, control type, display tier,
  scope, time presets, and default time selection.

### AIS-207 — Add cardinality and aggregate-cost guardrails

**Priority / size / owner:** P1 / M / Catalog/Runtime + Studio UX
**Audit:** AI-STUDIO-019
**Dependencies:** AIS-202, AIS-401

**Acceptance criteria:**

- “Add core dimensions” ranks candidates by semantic relevance and bounded
  cardinality rather than adding every plausible field.
- The Studio estimates relative group multiplication and flags configured
  high-cardinality thresholds before acceptance.
- `Treatment` with 738 sampled values is not silently added in the canonical
  fixture; the user may still opt in after reviewing the cost.
- The estimate is advisory and deterministic, records its sample basis, and
  does not claim exact storage size.

### Increment C acceptance

- With provider access disabled and raw YAML unopened, a new user can create,
  modify, and delete each core artifact type.
- The canonical journey visually creates at least one processor, four metrics,
  three pages, and six tiles, then produces a complete valid YAML diff.
- Every mutation is draft-local, atomic, lossless on round trip, and shared with
  Configuration Builder where the catalog contract is the same.

## Increment D — AI Resilience and Deterministic Fallback

### AIS-301 — Preflight every provider-backed entry point

**Priority / size / owner:** P0 / M / AI Runtime
**Audit:** AI-STUDIO-007
**Dependencies:** AIS-002

**Acceptance criteria:**

- Draft generation, report refresh, Copilot, and Chat call the same preflight
  contract before creating a provider request.
- Missing configuration or consent is detected locally and reported within two
  seconds without a provider call.
- A short independent connectivity check has a bounded timeout and briefly
  caches a negative result; Retry bypasses the negative cache intentionally.
- Disabled actions explain the missing setting and link to it.
- Preflight never changes the accepted draft or initializes provider/network
  work during Studio module import.

### AIS-302 — Standardize AI errors, timeout, cancellation, and isolation

**Priority / size / owner:** P1 / L / AI Runtime + Studio UX
**Audit:** AI-STUDIO-007
**Dependencies:** AIS-301, AIS-104

**Acceptance criteria:**

- Safe categories distinguish configuration, authentication, authorization,
  rate limit, timeout, network, provider, response validation, and internal
  failure; each category declares retryability and remediation.
- All AI surfaces use the same progress/error component and correlation
  reference without exposing prompt data, raw values, credentials, or provider
  internals.
- Requests show elapsed state, Cancel, and configured timeout; Cancel/timeout
  leaves the last accepted revision unchanged.
- Provider exceptions cannot terminate the Streamlit process. The health probe
  and deterministic Studio remain responsive after every failure class.

**Verification:** Extend `tests/unit/test_ai_studio_logging.py`, AI surface
AppTests, and a subprocess smoke test covering 401/403/429, timeout, network,
malformed response, and unexpected exception.

### AIS-303 — Checkpoint and restore non-secret drafts

**Priority / size / owner:** P1 / L / Studio UX + AI Runtime
**Audit:** AI-STUDIO-007
**Dependencies:** AIS-101C, AIS-302

**Acceptance criteria:**

- Each committed step checkpoints a workspace-local draft with timestamp and
  base-catalog hash.
- Secrets, sample values, raw provider payloads, and unapproved fields are
  excluded or redacted under the sharing contract.
- After refresh, provider exception, or Streamlit restart, the user can Restore
  or Discard. A changed base catalog triggers reconciliation rather than silent
  overwrite.
- Restored drafts revalidate before navigation or Apply.
- Retention and explicit deletion are documented and testable.

### AIS-304 — Add aggregate-only Chat fallback

**Priority / size / owner:** P2 / L / AI Runtime + Reports
**Audit:** AI-STUDIO-016
**Dependencies:** AIS-301; coordinates with the governed query contracts in
the reporting backlog

**Acceptance criteria:**

- Disabling or losing the LLM planner does not disable deterministic starter
  queries that can be resolved from known metrics, dimensions, time presets,
  and aggregate query templates.
- The UI labels deterministic capability and its limits; unsupported free-form
  questions invite enabling a configured planner rather than pretending to
  answer.
- All fallback results use the aggregate/query layer and expose the same
  provenance/freshness as normal Chat results.
- The canonical count, CTR, reach, Channel split, and available-date questions
  run without a provider.

### Increment D acceptance

- Missing provider configuration fails locally in under two seconds on every
  entry point.
- Timeout, cancellation, provider rejection, malformed response, and unexpected
  exception preserve the draft and leave the app responsive.
- The deterministic Studio and supported aggregate Chat questions remain usable
  with provider access disabled.

## Increment E — Bounded Preview Performance

### AIS-401 — Push Parquet row limits and projections into the scan

**Priority / size / owner:** P0 / L / Catalog/Runtime
**Audit:** AI-STUDIO-008
**Dependencies:** none

**Acceptance criteria:**

- Workspace Parquet preview reads from the validated path with
  `scan_parquet(...).select(...).head(limit).collect()` or an equivalent bounded
  operation; it never calls `Path.read_bytes()` or constructs a full-file
  `BytesIO` copy.
- Requested columns and row limit are pushed into the read where the format
  supports them.
- File identity uses validated path metadata and a safe incremental/content
  strategy without requiring a full memory copy.
- The returned frame contains no more than the requested rows/columns, and only
  that bounded frame is cached.
- Runtime ingestion behavior is unchanged and remains chunked/idempotent.

**Verification:** Monkeypatch tests prove no full byte read or unbounded Polars
read occurs; a multi-row-group fixture proves row-limit and projection
pushdown. Reuse the lazy reader patterns in `src/valuestream/readers/io.py`.

### AIS-402 — Apply bounded preview contracts to uploads and every format

**Priority / size / owner:** P1 / M / Catalog/Runtime
**Audit:** AI-STUDIO-008, AI-STUDIO-020
**Dependencies:** AIS-108, AIS-401

**Acceptance criteria:**

- Upload buffering is bounded by documented size limits and never duplicated
  unnecessarily; parsers stop after the requested sample where supported.
- CSV, JSON/NDJSON, Parquet, compression/container, and any enabled spreadsheet
  path declare whether row/column pushdown is supported.
- Unsupported or oversized inputs fail before expensive parsing with an
  actionable alternative.
- Format-specific preview and runtime-reader contracts share capability
  metadata and positive/negative tests.

### AIS-403A — Share source-inspection caching and invalidation

**Priority / size / owner:** P1 / M / Studio UX + Catalog/Runtime
**Audit:** AI-STUDIO-008; Configuration Builder F1
**Dependencies:** AIS-401

**Acceptance criteria:**

- AI Configuration Studio preview and Configuration Builder source-field
  discovery use one bounded source-inspection service instead of separate
  discover/read/schema paths.
- Cache keys include workspace identity, source ID, reader-config hash, sample
  identity, selected columns, row limit, and parsing settings as applicable.
- Unrelated widget and fragment reruns reuse the bounded result. One Builder
  Sources render performs at most one discovery/read/schema inspection for the
  same key rather than the historical two reads.
- Changing a key input invalidates the bounded sample, field options, rename
  mapping, and any schema-dependent validation evidence together.
- Only bounded frames and typed schema metadata are cached; raw files, upload
  buffers, and unapproved sample values are not persisted.

**Verification:** Add cache-hit, reader-config invalidation, and single-read
tests for both Studios, including the historical Builder render path.

### AIS-403B — Add source-read progress, cancellation, and recovery

**Priority / size / owner:** P1 / M / Studio UX + Catalog/Runtime
**Audit:** AI-STUDIO-008; Configuration Builder F2
**Dependencies:** AIS-402, AIS-403A

**Acceptance criteria:**

- Slow preview and source-inspection operations show elapsed progress and the
  source/path scope being inspected without exposing sensitive values.
- Cancel frees temporary resources and leaves the previous accepted draft and
  cached successful inspection unchanged.
- UI status distinguishes loading, cancelled, unsupported, failed, and ready;
  failures name the safe path or source that could not be inspected and give a
  remediation rather than silently returning an empty field list.
- Retry performs one fresh bounded inspection and does not create concurrent
  duplicate reads for the same source key.

### AIS-404 — Establish a repeatable preview performance gate

**Priority / size / owner:** P0 / M / Quality + Catalog/Runtime
**Audit:** AI-STUDIO-007, AI-STUDIO-008
**Dependencies:** AIS-401, AIS-402, AIS-403A, AIS-403B

**Acceptance criteria:**

- `tests/benchmarks/test_ai_studio_preview.py` uses a synthetic multi-row-group
  Parquet for CI and the canonical 280,045,584-byte fixture for the release
  qualification profile.
- Preview of 1,000 rows and selected columns has peak RSS increase no greater
  than 512 MiB and no greater than twice the bounded decoded sample plus
  128 MiB; exceeding either budget fails the qualification run.
- The canonical 1,000-row cold preview has p95 at or below five seconds across
  three runs on the documented reference runner. CI records time, peak RSS,
  rows, columns, row groups touched, and cache-hit behavior so hardware
  regressions are interpretable.
- Validation and Apply for the canonical 1-source, 1-processor, 4-metric,
  8-tile draft each complete within two seconds on the reference CI runner;
  existing stable benchmarks may not regress by more than 15% without an
  approved baseline update.
- Five consecutive preview/rerender cycles do not show monotonic retained-memory
  growth above 64 MiB after garbage collection.
- Failure testing distinguishes provider errors from preview OOM/resource
  pressure before assigning root cause to either subsystem.

### Increment E acceptance

- The canonical 280 MB Parquet can be previewed repeatedly without a whole-file
  byte copy, UI freeze, or process loss.
- Configuration Builder source-field discovery reuses the same bounded
  inspection and does not reread the source for each field-option consumer.
- Row and column bounds are enforceable by tests, and performance evidence is
  reported with release results.

## Increment F — Workflow, Reports, Accessibility, and Qualification

### AIS-501 — Add a sticky phase rail and navigation footer

**Priority / size / owner:** P1 / M / Studio UX
**Audit:** cross-cutting 14-step workflow concern
**Dependencies:** AIS-101C, AIS-104

**Acceptance criteria:**

- Data, Draft, Review, and Apply remain visible with Complete, Attention, and
  Not started text states; status does not rely on color alone.
- Back, Continue, Save state, and the current blocking reason remain reachable
  without scrolling through large editors.
- Jumping to a phase preserves committed work and invokes the dirty-state guard.
- At narrow width the rail becomes an accessible compact phase selector.

### AIS-502 — Consolidate 14 steps into four task workspaces

**Priority / size / owner:** P1 / XL / Studio UX
**Audit:** cross-cutting 14-step workflow concern
**Dependencies:** AIS-201–AIS-206, AIS-501

**Acceptance criteria:**

- Data contains Source & Mapping, Prepare Data, and Field Approval sections;
  Draft contains deterministic generation and optional AI enhancement; Review
  contains Processors, Metrics, Reports, Chat, and Settings; Apply contains the
  complete diff, validation, export, and transaction action.
- Existing session/deep-link step values migrate to the corresponding phase and
  subsection.
- The normal deterministic walkthrough takes no more than six top-level
  navigation actions, while advanced detail remains available progressively.
- Provider-sharing review appears at the point of first AI use. Later phases
  show compact Review/Revoke status instead of repeating the full panel.
- Before implementation, split this XL story by phase and preserve one shared
  navigation/state model.

### AIS-503 — Add an actionable Apply-readiness summary

**Priority / size / owner:** P1 / M / Studio UX
**Audit:** AI-STUDIO-004, AI-STUDIO-012 and error-placement UX concern
**Dependencies:** AIS-104, AIS-501

**Acceptance criteria:**

- Summary groups issues by Data, Processor, Metric, Report, Provider, and
  Runtime readiness.
- Each blocker shows affected object/field, current safe value, expected
  contract, remediation, and Jump to fix.
- Warning and blocker counts match all other surfaces exactly.
- Completed areas show artifact count and last committed change; runtime-only
  warnings are identified as such.
- Export and Apply explain why they are disabled beside the actions.

### AIS-504A — Add a focused calculation and expression editor

**Priority / size / owner:** P1 / M / Studio UX
**Audit:** AI-STUDIO-009, AI-STUDIO-010; Configuration Builder D1, D2
**Dependencies:** AIS-101C, AIS-103, AIS-104

**Acceptance criteria:**

- Calculation lists keep Enabled, human name, and mode visible without
  horizontal scrolling; selecting a row opens a structured detail form.
- AST YAML and Polars expressions use a multiline editor with live validation,
  mode-specific help, examples, and safe technical detail.
- A grid-added calculation defaults to Enabled even when Streamlit reports a
  missing/blank checkbox value; disabled rows remain visible as excluded.
- Add, duplicate, disable, delete, Save, Cancel, and Undo are explicit,
  keyboard reachable, and round-trip the exact transform shown in preview.
- Expression validation uses AIS-104 issue paths and plain remediation text
  rather than raw Pydantic error codes.

### AIS-504B — Replace dense catalog tables with responsive editors

**Priority / size / owner:** P1 / M / Studio UX
**Audit:** AI-STUDIO-009, AI-STUDIO-015; Configuration Builder D4, D5
**Dependencies:** AIS-101C, AIS-103, AIS-105, AIS-110

**Acceptance criteria:**

- Dense processor, metric, dimension, report, and review lists keep identity,
  state, type, and validation visible; selecting a row opens a focused detail
  form.
- Dimension packs and promotion previews use chips and key/value summaries
  rather than raw JSON blobs.
- Chart selectors use the shared chart-catalog display labels; technical kind
  IDs remain available as secondary detail.
- Keep selectors and review tables use the context-sensitive identity contract
  from AIS-110 rather than opaque IDs or description-only labels.
- Primary and destructive actions remain visible at 1440 px and 1024 px
  without horizontal scrolling, meet the target-size contract, and are not
  placed beside unrelated Help actions.

### AIS-505 — Define unambiguous report All/subset behavior

**Priority / size / owner:** P2 / M / Reports
**Audit:** AI-STUDIO-017
**Dependencies:** AIS-206; coordinates with RPT-203 and RPT-204

**Acceptance criteria:**

- All is represented by no concrete selected values and cannot coexist with a
  subset.
- Choosing Mobile while All is active transitions to Mobile in one interaction;
  Clear returns to All.
- Closed controls show All, the selected value, or “N selected,” and active
  chips match the actual query state and tile coverage.
- Pointer and keyboard browser tests cover All to Mobile, Mobile plus Web,
  removing one value, Clear, and page changes.

### AIS-506A — Open historical reports on available data

**Priority / size / owner:** P0 / M / Reports
**Audit:** Configuration Builder BUG-4/B1 and the canonical historical fixture
**Dependencies:** none; coordinates with RPT-203 and report freshness contracts

**Acceptance criteria:**

- Reports resolve the minimum and maximum available aggregate dates before
  choosing the initial range.
- If an authored relative default would be empty because the dataset ends in
  the past, initial load uses a visibly labelled Latest available period (or a
  disclosed anchor at `min(today, latest_data_date)`) and shows the data-through
  date; it does not mutate the authored catalog setting.
- A fresh open of the audited Executive overview shows computed KPI values
  rather than all `n/a` when aggregates exist.
- Explicit user-selected absolute/custom ranges are never clamped or silently
  changed, and a later manual relative-range selection remains visible as the
  user's choice.
- Builder tile preview and runtime Reports use the same date-resolution
  contract.

### AIS-506B — Make empty-report recovery and filter coverage explicit

**Priority / size / owner:** P2 / M / Reports
**Audit:** report UX observation from the canonical historical fixture
**Dependencies:** AIS-206, AIS-506A; coordinates with RPT-203

**Acceptance criteria:**

- Reports expose All time, Latest available period, authored rolling presets,
  and valid custom ranges with minimum/maximum aggregate dates.
- An empty user-selected range states the active range and filters and offers
  Show all available data without silently changing the selection.
- Empty, unsupported-filter, stale, query error, and not-yet-materialized states
  are visually and semantically distinct.
- Active filters disclose all-tiles or partial tile coverage.

### AIS-507 — Use stable page and tile IDs for anchors

**Priority / size / owner:** P3 / S / Reports
**Audit:** AI-STUDIO-021
**Dependencies:** AIS-106, AIS-204

**Acceptance criteria:**

- Page anchors use stable page IDs and tile anchors use stable tile IDs;
  duplicate titles remain unique.
- Changing Engagement to Audience to Volume updates or remounts headings and
  clears/replaces stale hashes.
- A deep link selects the correct dashboard/page, focuses its heading, and is
  announced correctly to assistive technology.

### AIS-508 — Complete keyboard, semantics, and responsive accessibility

**Priority / size / owner:** P1 / L / Studio UX + Quality
**Audit:** AI-STUDIO-001, AI-STUDIO-009 and cross-cutting UX observations
**Dependencies:** AIS-001, AIS-103, AIS-501–AIS-503, AIS-504A, AIS-504B;
coordinates with RPT-903 and KPR-904

**Acceptance criteria:**

- Every input, Help action, phase, editor command, dialog, status, and report
  heading has an accessible name, role, state, and visible focus.
- Text contrast is at least 4.5:1, large text and UI graphics are at least 3:1,
  and focus indicators are at least 3:1 under the adopted WCAG 2.2 AA checks.
- Error summaries announce once and focus/link to the failing field; success,
  progress, and cancellation updates do not cause repeated screen-reader noise.
- Pointer targets are at least 24 by 24 CSS pixels and do not overlap adjacent
  controls.
- The canonical journey is keyboard-completable at 1440 px and 1024 px; no
  primary action or current status is hidden by horizontal overflow.
- At 200% zoom and a 320 CSS-pixel viewport there is no loss of content or
  action. Tables may scroll, but Enabled, delete, and commit controls remain
  discoverable.
- Automated axe checks report zero critical or serious issues on every major
  phase and report page, followed by a documented manual keyboard/screen-reader
  smoke check.

### AIS-509 — Add the canonical real-browser and runtime release journey

**Priority / size / owner:** P0 / L / Quality
**Audit:** all findings; closes the missing browser E2E gap
**Dependencies:** AIS-001–AIS-008, AIS-101A–AIS-101C, AIS-109, AIS-110,
AIS-201–AIS-207, AIS-301–AIS-304, AIS-401, AIS-402, AIS-403A, AIS-403B,
AIS-404, AIS-501, AIS-503, AIS-504A, AIS-504B, AIS-505, AIS-506A, AIS-506B,
AIS-507, AIS-508; coordinates with RPT-901–RPT-903 and KPR-902–KPR-904

**Acceptance criteria:**

- From a clean `examples/test_ai_studio` catalog and provider-disabled session,
  the test selects the exact parquet, confirms mappings and timestamp parsing,
  creates/edits/deletes representative defaults, filters, and calculations,
  and retains `IsClicked`, `PropensityPct`, and `ResponseSeconds`.
- It visually creates one valid processor, four metrics, three report pages,
  and at least six tiles, including recipe and from-scratch paths, without raw
  YAML.
- It validates, reviews the diff, exports, applies, runs all 2,733,856 rows,
  verifies the four reference metric summaries/Channel splits, and renders all
  three pages.
- It proves immediate rerun keeps zero rows, clean rebuild replaces superseded
  aggregates with provenance intact, and aggregate-only Chat starters work.
- Negative branches cover consent Help, ambiguous mapping, invalid timestamp,
  duplicate grouping, invalid KPI placement, unsaved navigation, provider
  failure, cancellation, and recovery.
- A second journey uses a deterministic stub provider to cover successful
  Draft, Reports, and Copilot responses plus pending-review behavior without
  requiring an external service.
- The test uses isolated workspace copies, never mutates the checked-in example,
  and stops every server/process it starts while verifying the bound port is no
  longer listening.

### AIS-510 — Add operating metrics, documentation, and release evidence

**Priority / size / owner:** P1 / M / Quality + Product
**Audit:** cross-cutting
**Dependencies:** all applicable implemented stories, including AIS-511;
coordinates with RPT-904 and KPR-905

**Acceptance criteria:**

- Privacy-safe events measure phase completion/abandonment, validation category,
  recovery, preview performance, deterministic versus provider path, recipe
  reuse, and Apply outcome without raw/sample values or prompt content.
- `docs/guides/configuration/ai-config-studio.md`, processor/reader/chart
  references, Configuration Builder guidance, troubleshooting, and migration
  guidance change with behavior.
- The release record includes catalog validation, browser E2E, exact-fixture
  runtime, idempotency, rebuild, provider-error matrix, performance, visual, and
  accessibility evidence.
- Known limitations explicitly distinguish unavailable provider access from a
  deterministic product failure.

### AIS-511 — Align Home and navigation copy with Build

**Priority / size / owner:** P2 / S / Studio UX + Product
**Audit:** Configuration Builder BUG-6/E1
**Dependencies:** none

**Acceptance criteria:**

- Home Workspace Flow reflects the actual feature-flagged sidebar: Build names
  Configuration Builder and AI Configuration Studio, while Settings no longer
  claims ownership of authoring workflows.
- Copy remains correct with authoring v2 enabled or disabled, and links land on
  the corresponding available surface.
- Navigation labels, page titles, guidance, and documentation use
  “Configuration Studios” when a contract or backlog applies to both authoring
  surfaces; product-specific names remain where behavior is genuinely unique.
- Focused navigation tests cover both feature-flag states.

### Increment F acceptance

- The supported journey is shorter, keyboard-completable, and recoverable, with
  status and actions visible at both target widths.
- Report filters, date recovery, and anchors match visible/query state.
- The full canonical release matrix passes with zero open P0/P1 defects.

## Qualification Data and Test Matrix

### Compact pull-request fixture

Create `tests/fixtures/ai_studio/` with a committed 1,000–5,000-row Parquet and
golden catalog/aggregate JSON. It must contain `Outcome`, `pxOutcomeTime`,
`OutcomeTime`, `DecisionTime`, all four audited outcome values, null Propensity
values, Channel, Direction, Issue, Group, ModelControlGroup, a high-cardinality
Treatment, and the exact Pega-style timestamp strings. The goldens must prove
that persisted output contains only configured aggregate state, plus provenance
and configuration hashes, and never raw customer rows.

### Full release fixture manifest

| Property | Expected value |
|---|---|
| Relative path | `examples/test_ai_studio/data/Month=08/Day=2024-08-31/934be6678a7948e7b10c1cca2f5299fa-0.parquet` |
| SHA-256 | `ff54074bfffff95b38f8768a23671fdf9b9b443698f5bab275d82a858e4247c5` |
| Rows | 2,733,856 |
| Outcomes | Impression 1,951,721; NoConversion 704,751; Clicked 48,810; Conversion 28,574 |
| Click-through rate | `48,810 / 2,733,856 = 0.0178539030585371` |
| Approximate unique entities | 858,055, with a documented sketch/version tolerance |

Every test copies the example workspace into a temporary directory. No test may
mutate `examples/test_ai_studio` or depend on prior local aggregate state.

### Required automated scenarios

| Scenario | Required proof |
|---|---|
| Deterministic no-provider | Visual CRUD, Apply, ingest, idempotent rerun, rebuild, query, reports, filters, and local Chat fallback without AI or raw YAML. |
| Stubbed provider success | Explicit consent; valid Draft, Reports, and Copilot responses; reviewed revision; no direct mutation. |
| Provider failure/recovery | Missing configuration, 401/403, 429, timeout, network, 500, malformed/empty response; draft retained and server healthy after 20 consecutive failures. |
| Existing catalog modification | Add/modify/delete every object type, show cascade impact, exercise Cancel once, then confirm. |
| Editor persistence | Dirty defaults, rules, calculations, Keep selectors, navigation, refresh, reconnect, and restore. |
| Production-shaped fixture | No-manual-repair Apply; full ingest; exact count/CTR; idempotency; rebuild; aggregate-only persistence; three-page rendering. |
| Format matrix | Every advertised format previews, applies, and runs, or is rejected consistently before processing. |
| Responsive/accessibility | Major phases and all report pages in light/dark, desktop, 1024 px, 320 CSS px, and 200% zoom. |

The core regression automation is expected to require about 30 QA engineer-days.
Including browser/provider fixtures, accessibility integration, benchmark
infrastructure, and release soak, plan 35–38 QA engineer-days; two QA/SDET
engineers can parallelize that work alongside production fixes.

## Recommended Delivery Sequence

### Hotfix 1 — Consent containment

Ship AIS-001 and AIS-002 first. Until both are deployed, provider-backed actions
should remain disabled wherever the Help/input hit targets cannot be separated.

### Hotfix 2 — Draft trust and destructive-action containment

Deliver AIS-101A, AIS-101B, and AIS-109 with their focused AppTests. Do not wait
for the late full-browser journey: calculated-field changes must become
applicable, untouched templates must remain clean, and Delete must name and
target the object shown beside it before either Studio is described as
trustworthy.

### Release 1 — Safe and executable deterministic flow

Deliver AIS-401 before diagnosing further process loss, followed by AIS-403A so
both Studios reuse bounded inspection. Then deliver AIS-003, AIS-007, AIS-004,
AIS-005, AIS-006, AIS-008, AIS-104, AIS-301, AIS-506A, and the first executable
slice of AIS-509. Exit only when static validation and full ingest agree and a
historical workspace opens with populated KPIs.

### Release 2 — Editor trust and recoverability

Deliver AIS-101C, AIS-102–AIS-108, AIS-110, AIS-302, AIS-303, AIS-402,
AIS-403B, AIS-404, AIS-501, AIS-503, AIS-504A, and AIS-504B. AIS-511 may land
independently in this release or earlier. Exit only when reruns, navigation,
provider failures, source inspection, and restarts preserve committed work and
protect dirty work.

### Release 3 — No-YAML feature completion

Deliver AIS-201–AIS-207, update the golden journey, then complete AIS-502.
Exit only when all core artifact CRUD is achievable visually with provider
access disabled.

### Release 4 — Reports, fallback, and accessibility polish

Deliver AIS-304, AIS-505, AIS-506B, AIS-507, and AIS-508, then complete AIS-509,
AIS-510, and any outstanding AIS-511 copy alignment. Exit only when the full
release evidence matrix passes.

## Cross-Cutting Definition of Done

A backlog item is complete only when:

1. its acceptance criteria are proven by positive, negative, round-trip, and
   backward-compatibility tests at the lowest useful layer;
2. every audited UI regression also has AppTest or real-browser coverage, not
   helper-only coverage;
3. YAML model, schema, semantic validation, authoring UI, Apply transaction,
   and runtime behavior agree;
4. aggregate-first query behavior, raw-row disposal, deterministic/idempotent
   ingestion, provenance, and configuration hashes remain intact;
5. user-visible errors identify the affected field/object, safe current value,
   expected contract, and recovery action;
6. changes do not store secrets, prompts, unapproved sample values, or raw
   provider responses in logs, telemetry, checkpoints, or test artifacts;
7. both authoring studios support and round-trip any shared catalog property,
   and shared behavior is tracked under the exact suffixed story ID on this
   page rather than an obsolete unsuffixed or historical Config backlog ID;
8. behavior and migration documentation changes ship in the same pull request;
9. temporary processes, workspaces, and benchmark outputs are cleaned up; and
10. the relevant release gates on this page pass with evidence attached; and
11. implementation status and evidence links are updated only in this unified
    backlog, while the Configuration Editor QA report remains unchanged
    historical evidence apart from its superseded notice and mapping pointer.
