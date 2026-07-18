# Configuration authoring UX improvement plan

**Date:** 2026-07-17

**Status:** Milestones 0–3 complete; Milestone 4 is in manual accessibility verification; Milestone 5 is code-ready for measured rollout, with cohort evidence and legacy-path retirement open.
**Outcome:** A first-time user can choose the right authoring path, create a valid configuration without sharing sample values by default, review the proposal safely, apply it transactionally, and reach a clear report or data-processing next step.

This plan reviews the [independent UX audit](ux-review-config-builder-ai-studio.md) against the earlier [evidence-backed conversion audit](configuration-builder-ai-studio-conversion-audit-2026-07-17.md), the [architecture](../concepts/architecture.md), the [replacement design](replacement-design.md), the [implementation plan](implementation-plan.md), the [reporting backlog](reporting-backlog.md), and the [KPI recipe backlog](kpi-recipe-backlog.md).

## 1. Executive decision

The independent audit is directionally strong and corroborates the earlier test run. Treat the result as a configuration-authoring hardening program, not a replacement architecture.

The most serious problems are not visual polish:

1. Sample values are shared by default.
2. Invalid AI output can reach review and publish-adjacent screens.
3. Uploaded-file preview, generated source configuration, and later ingestion can describe different data.
4. Draft, validation, save, and run states are presented inconsistently.

The 47-patch review rail and 54,860-pixel export page are severe usability failures, but privacy, correctness, and persistence come first because they can expose data or create incorrect production behavior.

### Audit disposition

| Audit direction | Decision | Reason |
|---|---|---|
| Make sample-value sharing opt-in | Accept as critical | Both runs and the implementation confirm the unsafe default. |
| Pre-validate and repair AI output | Accept as critical | Invalid drafts and permission failures are proven happy-path failures. |
| Use one source of truth | Reframe | Require one revision-keyed displayed lifecycle; do not prescribe one internal store. |
| Use one save model | Reframe | Use three explicit verbs: **Update draft**, **Apply to workspace**, and **Run data**. Applying configuration must never imply ingestion. |
| Group patch review | Accept with dependency closure | Grouping only by object is insufficient when rejecting one change invalidates another. |
| Purge YAML and IDs | Reframe | YAML remains authoritative and inspectable; move it behind **Technical details** by default. |
| Lead with deterministic generation | Accept after coverage | Treat it as a validated safe path only after format and schema coverage prove that claim. |
| Add a browser unload guard and sticky custom controls | Do not adopt literally | The native Streamlit constraint rules out custom JavaScript and CSS. Preserve drafts across internal navigation and use native top and bottom actions. |
| Treat visual truncation as a screen-reader failure | Verify first | Visual truncation is proven; accessible-name behavior still needs a dedicated audit. |
| Palette, typography, favicon, and polish | Defer | Important for product quality, but not part of the trust-critical release gate. |

The run-specific numbers are evidence, not universal constants: one run produced 47 preselected patches and a 6-to-2 issue loop; the other used 19 fields and produced one invalid processor followed by a permission failure.

The source review found two additional release blockers:

- The shared AI call path logs full prompts and responses at INFO level. Opt-in sample values can therefore reach ordinary application logs even after the on-screen consent default is fixed.
- Several Builder mutations validate only after writing and then report the invalid workspace without rolling it back. AI draft apply already uses a safer transaction boundary; Builder should reuse it.

## 2. Product contract

These constraints govern every work item:

- Configuration Builder remains catalog-first; AI Configuration Studio remains sample-first. A shared **Build** landing page explains the choice without merging the tools.
- Both tools edit the same YAML catalog and reuse shared models, validators, writers, and transaction boundaries. YAML remains the source of behavior.
- **Update draft** changes session-local work only. **Apply to workspace** validates and writes YAML transactionally. **Run data** is a separate explicit materialization action.
- Reuse the existing catalog transaction and rollback behavior. Do not build a second transaction layer.
- No raw rows may persist beyond chunk processing. Ingestion remains deterministic, idempotent, provenance-carrying, and explicit.
- Uploaded bytes are preview-only until the user explicitly stages or selects a production source. Preview copy must never imply that the uploaded bytes will be processed later.
- Full prompts, model responses, sample values, credentials, and local paths must not appear in normal logs or analytics. Retain only operational metadata needed for support.
- Every visible validation verdict names the object and revision it describes: current workspace or proposed draft. One computed validation result is reused everywhere for that revision.
- Invalid generated output enters a recovery state, not an acceptance state. Applying or running an invalid draft is impossible.
- Patch selection operates on validated dependency bundles. Exact per-file and per-object YAML remains available under details.
- Preview and report behavior continues through the aggregate/query layer; authoring work must not create a raw-row bypass.
- Use native Streamlit components. Do not introduce custom HTML, CSS, or JavaScript to implement navigation, sticky controls, or unload interception.
- Preserve the product strengths: governed AI proposals, deterministic fallback, transparency counters, recipe explanations, purpose-based report creation, and inspectable YAML.

### Canonical displayed lifecycle

| State | Meaning | Allowed primary action |
|---|---|---|
| **Editing draft** | Session work differs from the applied workspace. | Update draft |
| **Ready for review** | The named draft revision is valid and has reviewable changes. | Review changes |
| **Reviewed** | The user accepted a dependency-consistent set for that revision. | Apply to workspace |
| **Applied** | Transactional YAML write and post-write validation succeeded. | Inspect impact |
| **Data refresh required** | The applied config changes require materialization or backfill. | Run data |
| **Report ready** | Existing or refreshed aggregates can satisfy the result. | Open report |

Changing any draft field creates a new revision and invalidates stale validation or review state. Current-workspace health and draft validation may appear together only when each is explicitly labelled.

## 3. Working backlog

Status values are intended to be updated in place: **Todo**, **In progress**, **Blocked**, or **Done**. Effort is relative: S, M, L, XL.

### Critical — trust and correctness release blockers

| ID | Status | Effort | Work | Acceptance |
|---|---|---:|---|---|
| CA-001 | Done | M | Establish a code-and-test baseline for related RPT and KPR stories; mark each Done, Partial, or Open before scheduling. Convert current guide/UI mismatches into regression tests. | The plan links to existing stories instead of duplicating them, and every claimed regression has a failing test before its fix. |
| CA-002 | Done | S | Stop INFO-level logging of full AI prompts and responses. Retain call ID, model, duration, token metadata, and outcome; redact diagnostics by default. | Sample values, prompts, responses, credentials, and local paths never appear in normal logs; a dedicated log-capture test proves it. |
| CA-003 | Done | S/M | Put every Builder catalog mutation behind the existing catalog transaction and post-write validity requirement. Reuse the AI apply rollback pattern rather than only reporting invalid state after a write. | Any failed write or post-write validation restores all affected catalog files byte-for-byte. |
| CA-004 | Done | S/M | Default sample examples to off, flag likely identifiers, and require an explicit data-sharing review showing provider, model, fields, and whether examples are included. | A fresh sample sends zero example values until opt-in; prompt-payload tests prove it; generation cannot bypass the consent checkpoint. |
| CA-005 | Done | M | Make source defaults format-aware. Separate **Preview sample** from the production source location. Detect or explicitly select Pega behavior; use neutral CSV and Parquet defaults otherwise. Either support archive members advertised by the UI or narrow the promise. | A generic CSV never creates a Pega ZIP source; every supported sample can be re-read by its generated source settings; unsupported archives fail with actionable copy. |
| CA-006 | Done | M | Add provider, model, credential, and operation capability checks on the user-initiated generate or repair path; cache successful checks for the session. Map provider exceptions to product-language errors with expandable diagnostics and make gating tooltips conditional. | Missing credentials or insufficient permissions identify the failed capability and a corrective action; raw LiteLLM errors never lead the screen; an enabled or running action never shows disabled-state guidance. |
| CA-007 | Done | L | Change generation to parse, merge, validate, and run a bounded internal repair before producing reviewable patches. If recovery fails, discard the invalid candidate and offer a validated deterministic draft or a retry. | No acceptance control renders for an invalid candidate; repair exhaustion preserves prior work; invalid drafts cannot reach apply or run actions. |
| CA-008 | Done | M | Introduce revision-keyed draft validation and derive phase, step, proposal, accepted-draft, and published-workspace status from it. Human completion requires explicit review, not inferred object counts. | The same object and revision has one issue count everywhere; distinct objects are labelled; no phase is complete before the required user confirmation. |
| CA-009 | Done | L | Make the existing step-local persistence model truthful before considering a global draft rewrite. Form changes remain session-local until an object-specific **Apply to workspace** action; dirty state compares canonical objects and survives internal navigation with Restore or Discard. Rename the final step **Export current workspace**. | Switching steps without editing cannot change save state; an unapplied proposal cannot disappear silently; exactly one active apply action exists for the current object; **Run data** remains separate. |

Do not begin with a cross-step global-draft rewrite. If later evidence shows that step-local proposals cannot satisfy the lifecycle, write an ADR before changing that persistence boundary and ship the migration behind a feature flag.

### High impact — activation, review, and successful handoff

| ID | Status | Effort | Work | Acceptance |
|---|---|---:|---|---|
| CA-101 | Done | M | Add a top-level **Build** entry that explains **Start from sample with AI** versus **Configure the current workspace manually**. Replace the blank Studio canvas with upload, workspace sample, and one-click deterministic demo choices. | A cold user can begin in the main canvas and reach a valid demo draft without opening the sidebar or configuring an LLM. |
| CA-102 | Done | L | Restore one truthful navigation rhythm: compact progress, one current task, Back, one primary Continue or Apply action, and a jump outline. Dead-end Review or Publish states route to the prerequisite action. | Every step has one obvious primary action; Builder guide and live Previous or Next behavior agree; completion reflects user actions. |
| CA-103 | Done | L | Move patch review into the main canvas. Build semantic, dependency-closed bundles with a business summary, consequences, and validity. Offer **Accept safe additions**, **Review individually**, and **Reject**; removals start rejected. Keep exact YAML in collapsed details. | Large proposals are reviewable without a sidebar checkbox wall; any accepted combination validates; a removal requires explicit selection. |
| CA-104 | Done | M | Replace free-text required-field mapping with searchable schema selectors showing type and a safe preview. Keep Copilot available for read-only explanation while patches are pending, while blocking mutating requests that could overwrite them. | A nonexistent field cannot be mapped; a user can ask what a pending patch does without changing the proposal. |
| CA-105 | Done | M | Add an outcome-first finish screen that explains what was applied, classifies materialization impact, and recommends either **Open report** or **Run data**. | A successful authoring flow ends at a report or an explicit processing handoff, never merely at YAML download. |
| CA-106 | Done | M | Apply progressive disclosure: collapse YAML, AST, generated transforms, paths, provider details, and IDs. Put validation and downloads above collapsed file previews. Replace the embedded repository README with task-scoped help. Add Report Inventory search and human labels. | No default screen renders a full catalog dump; **Export current workspace** actions are reachable near the top; expert details remain available. |
| CA-107 | Done | M/L | Make long AI operations recoverable with named stages, a hard timeout, preserved draft state, and retry. First run a technical spike for native cancellable background execution; add Cancel only if the operation can actually be stopped safely. | A timeout never loses work or leaves ambiguous state; the UI names the current stage and provides one recovery action. |

### Nice to have — coherence and polish

| ID | Status | Effort | Work | Acceptance |
|---|---|---:|---|---|
| CA-201 | Done | M | Show human names first; move generated IDs to details. Fix phantom table rows, unsafe default selections, clipped columns, wrapping, and Report Inventory filtering. | Empty editors read as empty; no **Avoid** recommendation is preselected; key table values remain inspectable. |
| CA-202 | Done | M | Replace narrow stat-card grids with compact key-value layouts. Reduce duplicate health summaries, card chrome, chip wrapping, and help-icon noise. | At desktop and narrow widths, status text is legible and one primary action dominates each screen. |
| CA-203 | In progress | M | Run a dedicated accessibility pass for keyboard order, focus, status announcements, accessible names, contrast, non-color diff semantics, and 200% zoom. | Keyboard-only and screen-reader checks pass the agreed matrix; no conclusion relies only on visual truncation. |
| CA-204 | Done | S/M | Normalize type hierarchy, casing, button roles, error severity, and business-language copy. Reproduce the reported browser-title and favicon issue before changing them because the shell already configures both. Load the configured fonts or remove the unsupported declarations. | Visual roles are consistent and provider, validator, and engine terminology is secondary to user language. |
| CA-205 | Done | S/M | Add privacy-safe funnel instrumentation for entry, sample chosen, consent confirmed, draft requested, valid proposal, review, apply, run, and report open. Measure time-to-valid and failure or abandonment by stage. | Events contain no raw sample values, field values, local paths, prompts, or credentials; numeric targets are set only after a baseline exists. |

## 4. Reuse before building

The following planned or implemented capabilities are dependencies, not new parallel systems:

- KPR-101 through KPR-108: exact diffs, query preview, shape validation, placement, atomic install, and undo.
- KPR-201 through KPR-207: replay and backfill impact classification.
- KPR-504 and KPR-507: AI install plans and privacy-safe recommendations.
- KPR-904 and KPR-906: responsive or accessibility work and outcome measurement.
- RPT-005, RPT-006, and RPT-901: AI schema, repair, and round-trip coverage.
- RPT-302 and RPT-307: human-facing labels in place of generated identifiers.
- RPT-903 and RPT-904: visual and accessibility release checks.

KPI recipe Increment A already supplies a shared browser, resolver, mapping, and instantiator. Reuse it instead of creating a second guided-metric system. The reporting backlog does not consistently distinguish Done, Partial, and Open work, so CA-001 must precede scheduling any RPT item.

## 5. Delivery sequence

### Milestone 0 — baseline and decisions

**Status: Done.** The story baseline records implementation ownership in the primary implementation map below while preserving Partial/Open status for the broader RPT/KPR stories.

- Complete CA-001.
- Land CA-002 and CA-003 as immediate trust hotfixes.
- Freeze the canonical lifecycle, action vocabulary, prompt consent contract, and preview-versus-source copy.
- Add deterministic fixtures for CSV, Parquet, Pega-style input, valid LLM output, invalid LLM output, and permission failure.

**Exit gate:** the current regressions are reproducible in tests and each work item has one owner.

### Milestone 1 — safe input and valid generation

**Status: Done.**

- Complete CA-004 through CA-008.
- Extend existing AI and Studio tests instead of starting a parallel test harness.
- Update the AI Studio guide and security guidance with the behavior changes.

**Exit gate:** no examples without opt-in, no invalid candidate in review, no raw provider error as primary copy, no generic CSV configured as Pega, and no conflicting verdicts.

### Milestone 2 — draft, review, and apply

**Status: Done.**

- Complete CA-009, CA-103, and the review portion of CA-104.
- Reuse the existing catalog transaction and post-write validation.
- Add dependency-closure and partial-rejection fixtures.

**Exit gate:** a proposal survives internal navigation, every accepted bundle validates, apply is atomic, and ingestion is still a separate action.

### Milestone 3 — activation and outcome

**Status: Done.**

- Complete CA-101, CA-102, CA-105, and CA-107.
- Update the Builder guide, AI Studio guide, UI tour, and relevant tutorial in the same changes.

**Exit gate:** a first-time user can choose a path, finish the deterministic demo, apply safely, and reach **Open report** or **Run data** without using YAML.

### Milestone 4 — disclosure, responsiveness, and accessibility

**Status: In progress.** Automated contrast, focus, reduced-motion, native-component, AppTest, and light-theme desktop/narrow reflow checks pass. Manual keyboard traversal, VoiceOver/NVDA, true browser 200% zoom, and dark-theme browser evidence remain open.

- Complete CA-106 and CA-201 through CA-204.
- Reconcile the current specification requiring help on every field with the audit evidence of help-icon overload: definitions remain accessible, while standalone visible help is reserved for ambiguous fields and section-level guidance.

**Exit gate:** the release matrix passes desktop, narrow width, 200% zoom, light and dark themes, keyboard, and screen-reader checks.

### Milestone 5 — measured rollout

**Status: In progress.** CA-205 implementation is complete. No representative baseline window, like-for-like cohort comparison, or documented legacy-navigation retirement has occurred.

- Complete CA-205 before broad exposure.
- Roll out behind a feature flag, compare funnel stages, inspect failure and abandonment, then retire the old semantics.

**Exit gate:** all critical items are Done, privacy and validity invariants are enforced by tests, funnel events contain no customer data, and a representative baseline/comparison is documented before the rollout flag and legacy grouping are retired.

## 6. Verification matrix

| Layer | Required coverage |
|---|---|
| Unit | Example sharing defaults; prompt payload minimization; format inference; revision invalidation; phase status; provider error mapping; dependency closure; validation classification. |
| Integration | Successful generation; invalid generation and bounded repair; permission failure; deterministic fallback; draft restore or discard; transactional apply and rollback; explicit data run; materialization-impact classification. |
| Streamlit AppTest | Cold empty state; Builder Back or Continue; consent checkpoint; single validation verdict; pending-patch explanation; blocked invalid apply; collapsed export; outcome handoff. |
| Browser journey | Demo to valid draft to review to apply to report or run; large patch set; partial rejection; timeout or retry; desktop and narrow viewport. |
| Accessibility | Keyboard order; focus after rerun; live status semantics; accessible names; non-color diff meaning; contrast; 200% zoom. |
| Architecture | YAML remains authoritative; no raw-row persistence; explicit ingestion; aggregate/query preview path; config hash and provenance behavior preserved. |
| Documentation | Updated architecture, ADR, guides, tutorial, security guidance, and UI tour; strict MkDocs build passes. |

Release-blocking assertions:

- No sample value enters an AI prompt without explicit opt-in.
- An invalid generated draft cannot be reviewed for acceptance, applied, or run.
- A generic CSV cannot silently become a Pega ZIP source.
- An unapplied proposal cannot disappear during internal navigation without a Restore or Discard decision.
- Every displayed verdict identifies the object and draft revision it describes.
- Applying configuration cannot start ingestion.
- Failed multi-file apply restores the prior valid workspace.

## 7. Primary implementation map

| Concern | Primary code and tests |
|---|---|
| Studio state, consent, generation, patch review, validation, apply, and run | src/valuestream/ui/pages/ai_config_studio.py; tests/unit/test_ai_studio_helpers.py; tests/unit/test_ai_copilot.py |
| Model settings, calls, and privacy-safe logging | src/valuestream/ai/settings.py; src/valuestream/ai/studio.py |
| Builder navigation, forms, save behavior, export, and inventory | src/valuestream/ui/pages/config_builder.py; src/valuestream/ui/builder.py; tests/unit/test_phase5_builder.py |
| Product navigation | src/valuestream/ui/shell.py; UI guardrail and component tests |
| Shared components and responsive behavior | src/valuestream/ui/components.py; tests/unit/test_ui_components.py; tests/unit/test_ui_guardrails.py |
| Product contract | docs/concepts/architecture.md; configuration guides, tutorial, security guide, and UI tour; an ADR under docs/concepts/adr only if the persistence boundary changes |

The first implementation release should combine CA-002, CA-003, and CA-004: stop leaking AI payloads to normal logs, make every Builder mutation rollback-safe, and remove opt-out sample sharing. In parallel, CA-001 should establish the regression baseline.
