# KPI Recipe Library and Workflow Backlog

This backlog turns metric algorithms into a governed, searchable business KPI
catalog. The target experience is: find a business question, understand the
calculation and confidence, see whether current aggregates support it, map
inputs, preview the generated YAML and result, then add it to the metric
catalog and one or more reports.

The library is an authoring layer, not a second runtime catalog. Installed
metrics and tiles are ordinary YAML; processors, aggregates, query planning,
and reports continue to obey the aggregate-first architecture.

The configuration-authoring program's Done/Partial/Open assessment of the KPR
stories it reuses is recorded in the
[configuration authoring story baseline](configuration-authoring-story-baseline-2026-07-18.md).
This backlog remains the source of truth for full KPR scope.

## Product Principles

1. **Business question first.** Search results lead with decision intent, not
   metric-kind or sketch names.
2. **Calculation is inspectable.** Every recipe shows its formula, aggregate
   inputs, accuracy class, algorithm, and caveats before installation.
3. **No hidden data work.** `ready`, `mapping_required`, and
   `backfill_required` are explicit; the author chooses before any mutation or
   ingestion run.
4. **One artifact, two studios.** Configuration Builder and AI Configuration
   Studio use the same recipe file, validation model, compatibility resolver,
   and instantiator.
5. **YAML remains authoritative.** A recipe only proposes artifacts. Runtime
   behavior begins after validated `processors.yaml`, `metrics.yaml`, and
   `dashboards.yaml` changes are applied.
6. **Versions are auditable.** Installed metrics retain recipe ID/version;
   upgrades are reviewable diffs, never implicit rewrites.

## Workflow State Model

| State | User outcome | System evidence |
|---|---|---|
| Discover | Find by domain, KPI, question, tag, or algorithm | Searchable recipe metadata |
| Understand | Read business meaning and method | Calculation, accuracy, caveat, examples |
| Assess | Know whether the workspace can support it | Processor capability/readiness result |
| Map | Resolve business roles to aggregate states/stages | Explicit input bindings |
| Preview | See YAML, sample aggregate result, and report form | Valid metric/tile draft plus query preview |
| Install | Add reviewed artifacts to catalog or AI draft | Recipe provenance and validation result |
| Materialize | Populate newly required aggregate state when approved | Change plan, run/backfill, lineage/config hash |
| Govern | Certify, deprecate, compare, or upgrade | Owner, version, review history, usage impact |

## Delivery Slices

| Increment | Outcome | Backlog range |
|---|---|---|
| A | Shared artifact foundation and both Studio entry points | KPR-001–KPR-008 |
| B | Trustworthy preview and transactional installation | KPR-101–KPR-108 |
| C | Missing-state and backfill planning | KPR-201–KPR-207 |
| D | Governance, workspace recipes, and lifecycle | KPR-301–KPR-310 |
| E | Business catalog and report packs | KPR-401–KPR-410 |
| F | AI recommendation and guided composition | KPR-501–KPR-508 |
| G | Release qualification and operating model | KPR-901–KPR-906 |

## Increment A — Shared Foundation

| ID | Status | Priority | Size | Backlog item |
|---|---|---:|---:|---|
| KPR-001 | Done | P0 | M | Define a strict, versioned KPI recipe model with business metadata, processor capabilities, input roles, metric template, method/accuracy, and report recommendation. |
| KPR-002 | Done | P0 | M | Ship a checked-in YAML library and JSON Schema with parity tests. Keep it outside the active workspace catalog until installation. |
| KPR-003 | Done | P0 | M | Implement deterministic readiness: compatible processor kind, required/absent state metadata, strict business roles, preferred algorithms, paired inputs, ambiguity, and missing-state detection. |
| KPR-004 | Done | P0 | M | Materialize and validate normal metric definitions through the typed metric model; attach recipe ID/version provenance. |
| KPR-005 | Done | P0 | S | Materialize and validate recommended report tiles, with deterministic collision-safe IDs. |
| KPR-006 | Done | P0 | M | Add the shared searchable browser, business/method detail, field/algorithm and stage/population mappings, hidden technical state bindings, and optional report placement to Configuration Builder. |
| KPR-007 | Done | P0 | M | Reuse the same browser and instantiator in AI Configuration Studio; mutations remain session-local until draft apply. |
| KPR-008 | Done | P0 | M | Seed reviewed recipes for engagement, audience/CPC-HLL distinct count, quantiles, ROC AUC, funnel, lifecycle, and Top-K, with focused tests and documentation. |

### Increment A Acceptance

- Both Studios render the same recipe count, versions, descriptions, and
  compatibility outcomes.
- Installing the same recipe with the same bindings yields the same metric
  definition in both surfaces.
- CPC is preferred for built-in unique-count mappings; existing HLL states are
  accepted without cross-family merging.
- Every processor-owned field and recipe-compatible algorithm can be selected
  before the first load; missing combinations become deterministic processor
  state definitions and are marked as requiring a first run/backfill.
- No recipe browse or install path reads raw event rows or starts ingestion.

## Increment B — Preview and Transactional Installation

| ID | Priority | Size | Backlog item | Dependencies |
|---|---:|---:|---|---|
| KPR-101 | P0 | M | **Partial:** show the exact generated processor, metric, and dashboard YAML patches, including recipe provenance and generated IDs, before installation. Full surrounding-file diff remains open. | A |
| KPR-102 | P0 | L | Add aggregate-query preview using the selected processor, bindings, grain, date window, and compatible dimensions. Route only through the query layer and show provenance/freshness. | KPR-101 |
| KPR-103 | P0 | M | Present result-shape and format validation: scalar KPI, multi-output table, empty result, non-numeric result, and unsupported grouping. | KPR-102 |
| KPR-104 | P0 | M | **Done:** recipe installs run inside `builder.catalog_transaction`; write failures and post-write catalog validation restore every catalog file. AI draft apply uses the same rollback semantics across all catalog files plus `ai.yaml`. Audit events remain open. | KPR-101 |
| KPR-105 | P0 | S | Add metric-only, existing-page, and create-new-page placement choices without inventing page semantics. | KPR-104 |
| KPR-106 | P1 | M | Suggest compatible alternative chart types and required mappings using the chart catalog; preserve the recipe default as the reviewed choice. | KPR-103 |
| KPR-107 | P1 | M | Detect near-duplicate installed metrics by normalized calculation/source/bindings and offer reuse instead of duplicate creation. | KPR-101 |
| KPR-108 | P1 | S | Add undo for the latest recipe transaction when no dependent catalog edit has occurred. | KPR-104 |

### Increment B Acceptance

- The user sees exact file-level changes before apply.
- Preview and installed report values use identical query semantics.
- A failed two-file install leaves both catalog files unchanged.
- Duplicate/reuse advice never replaces an explicit user choice.

## Increment C — Missing State and Backfill Workflow

| ID | Priority | Size | Backlog item | Dependencies |
|---|---:|---:|---|---|
| KPR-201 | P0 | M | **Partial:** the preview names processor fields/states, source, and current/proposed processor computation hashes, and the installed-metric handoff links to Data Load. Earliest usable date and date-window planning remain open. | A |
| KPR-202 | P0 | L | **Done (configuration):** add safe processor-state proposals for CPC, HLL compatibility, t-digest/KLL, Theta, and Top-K with source-column mapping and deterministic parameters. Detailed replay classification remains KPR-201/203. | A |
| KPR-203 | P0 | M | Classify the proposal using catalog compatibility rules: no replay, forward-only materialization, raw replay, or impossible without a new source field. | KPR-201 |
| KPR-204 | P0 | M | Show storage/time estimates and selected date range before run/backfill approval. | KPR-203 |
| KPR-205 | P0 | L | Apply approved processor change, validate, run/backfill, and install the waiting recipe only after aggregate success. Preserve run/config lineage. | KPR-202–KPR-204 |
| KPR-206 | P1 | M | Support resumable waiting recipes when a run fails or an operator completes backfill outside the Studio. | KPR-205 |
| KPR-207 | P1 | M | Add parameter guidance and incompatible-sketch checks, especially CPC/HLL `lg_k`, KLL `k`, t-digest compression, Theta nominal entries, and Top-K capacity. | KPR-202 |

### Increment C Acceptance

- No processor mutation or ingestion starts without explicit approval.
- Raw rows still disappear after chunk processing and are never stored by the
  recipe workflow.
- A metric may be configured before its first aggregate exists. Query/report
  readiness must distinguish that state until aggregates with the matching
  processor computation hash have been materialized.

## Increment D — Governance and Recipe Lifecycle

| ID | Priority | Size | Backlog item | Dependencies |
|---|---:|---:|---|---|
| KPR-301 | P0 | M | Define owner, steward, review date, source/reference links, applicability, and certification evidence. | A |
| KPR-302 | P0 | M | Enforce immutable published versions and semantic-version rules for calculation, inputs, display, or prose-only changes. | KPR-301 |
| KPR-303 | P0 | M | Show installed version, latest version, field-level diff, affected reports, and compatibility before upgrade. | KPR-302, KPR-101 |
| KPR-304 | P1 | M | Support deprecation, replacement recipe, end-of-support date, and non-destructive warnings for installed metrics. | KPR-302 |
| KPR-305 | P1 | L | Load workspace-owned recipe libraries from a governed directory with schema validation and deterministic precedence. | KPR-302 |
| KPR-306 | P1 | M | Add import/export bundles with recipe YAML, JSON Schema version, tests, examples, and signed provenance metadata. | KPR-305 |
| KPR-307 | P1 | M | Add role-based approval hooks for `reviewed` and `certified` transitions when multi-user identity exists. | KPR-301 |
| KPR-308 | P2 | M | Record install/upgrade/deprecate audit events without storing metric result data. | KPR-303 |
| KPR-309 | P2 | M | Show usage impact: installed metrics, tiles, dashboards, chat exposure, and API consumers. | KPR-303 |
| KPR-310 | P2 | S | Add recipe ownership and review-health operations views. | KPR-307–KPR-309 |

## Increment E — Business Catalog and Report Packs

| ID | Priority | Size | Backlog item | Dependencies |
|---|---:|---:|---|---|
| KPR-401 | P1 | M | Add glossary terms, KPI synonyms, decision owner, business process, leading/lagging indicator, and numerator/denominator semantics. | KPR-301 |
| KPR-402 | P1 | M | Add faceted discovery by domain, source type, processor capability, accuracy, maturity, owner, and readiness. | KPR-401 |
| KPR-403 | P1 | M | Add “available now” and “requires backfill” saved views per workspace. | KPR-402, C |
| KPR-404 | P1 | L | Define recipe packs that install several related metrics, page filters, and a report page as one reviewed unit. | B, KPR-401 |
| KPR-405 | P1 | M | Ship an Engagement & Reach pack: interactions, positive/negative outcomes, engagement rate, CPC reach, frequency, and period change. | KPR-404 |
| KPR-406 | P1 | M | Ship a Funnel Health pack: stage volumes, conversion/drop-off, bottleneck ranking, and trend. | KPR-404 |
| KPR-407 | P1 | M | Ship a Model Quality pack: ROC AUC, average precision, calibration, score quantiles, and experiment context. | KPR-404 |
| KPR-408 | P2 | M | Ship Distribution/Service, Lifecycle/RFM, Set/Cohort, and Category Concentration packs as aggregate capabilities mature. | KPR-404 |
| KPR-409 | P2 | M | Add localized titles/descriptions while preserving invariant IDs and calculations. | KPR-401 |
| KPR-410 | P2 | M | Add catalog export for documentation portals and read-only API consumers. | KPR-401, KPR-306 |

## Increment F — AI Recommendation and Composition

| ID | Priority | Size | Backlog item | Dependencies |
|---|---:|---:|---|---|
| KPR-501 | P1 | M | Give the AI planner the same read-only recipe inventory and readiness results; prohibit invented recipe IDs or unsupported inputs. | D, E |
| KPR-502 | P1 | M | Rank recommendations from approved fields, processor capabilities, existing metrics, and business goal; show deterministic evidence for every suggestion. | KPR-501 |
| KPR-503 | P1 | M | Ask for missing business intent only when it changes numerator, denominator, entity, stage, direction, target, or approximation choice. | KPR-502 |
| KPR-504 | P1 | M | Produce an editable install plan, not direct catalog mutation; reuse the shared preview/diff/apply path. | KPR-101, KPR-502 |
| KPR-505 | P1 | M | Let users request “build an executive engagement page” and compose only reviewed recipe-pack artifacts plus explicit mappings. | KPR-404, KPR-504 |
| KPR-506 | P2 | M | Explain why a recipe was not recommended: unavailable field, incompatible processor, ambiguous mapping, missing state, duplicate metric, or governance policy. | KPR-502 |
| KPR-507 | P2 | M | Use privacy-safe aggregate usage signals to improve ranking; never send raw rows or unrestricted sample values. | KPR-502 |
| KPR-508 | P2 | M | Add offline recommendation evaluation for relevance, validity, duplicate rate, and unsupported/backfill error rate. | KPR-502–KPR-507 |

## Increment G — Release Qualification

| ID | Priority | Size | Backlog item | Dependencies |
|---|---:|---:|---|---|
| KPR-901 | P0 | M | Add schema parity, recipe fixture, instantiation, invalid-template, paired-input, version, and compatibility tests. | Every increment |
| KPR-902 | P0 | M | Add both-Studio component tests proving identical artifacts and validation outcomes. | A, B |
| KPR-903 | P0 | M | Add end-to-end ready/mapping/backfill scenarios for demo, Interaction History, lifecycle, set, and snapshot workspaces. | B, C, E |
| KPR-904 | P0 | M | Add accessibility and narrow-layout checks for search, method detail, mapping forms, diff, and error states. | B |
| KPR-905 | P0 | S | Document operator runbooks, recipe author checklist, certification checklist, and upgrade/deprecation policy. | C, D |
| KPR-906 | P1 | M | Establish outcome measures: discovery success, install completion, reuse ratio, preview failures, backfill abandonment, and stale-version count. | D |

## Definition of Done

A recipe or workflow story is complete only when:

1. the reusable artifact validates against its typed model and checked-in JSON
   Schema;
2. Configuration Builder and AI Configuration Studio use the same core logic;
3. installed behavior is represented entirely by validated catalog YAML;
4. compatibility/readiness is deterministic and covered by reference fixtures;
5. aggregate/query paths remain the only source of metric preview and report
   data;
6. state additions preserve config hashes, lineage, idempotency, and raw-row
   disposal invariants;
7. business definition, calculation, method, accuracy, caveat, and governance
   metadata are visible before apply; and
8. current reference, guide, architecture, and migration/backfill documentation
   changes ship with the behavior.
