# Configuration authoring story baseline

**Date:** 2026-07-18
**Status:** Baseline of record for CA-001

This baseline prevents the configuration-authoring program from silently
claiming work already owned by the reporting (RPT) and KPI recipe (KPR)
backlogs. It classifies only the stories explicitly reused by the
[configuration authoring improvement plan](configuration-builder-ai-studio-improvement-plan-2026-07-17.md).
The source backlogs remain authoritative for their complete scope and delivery
order.

Status means:

- **Done** — the story's acceptance contract is implemented and covered.
- **Partial** — a useful subset exists, but at least one stated acceptance
  condition remains open.
- **Open** — the primary outcome is not implemented. Incidental helpers do not
  change this status.

## Reporting crosswalk

| Story | Baseline | Evidence and remaining boundary |
|---|---|---|
| RPT-005 | Done | The shared AI schema dictionary, generation and repair prompts, deterministic report generation, and invalid-scope validation cover the typed reporting properties in `src/valuestream/ai/studio.py` and Studio helper tests. |
| RPT-006 | Partial | Builder writer and AI apply/reload tests preserve explicit display, theme, layout, page-filter, time-filter, KPI, and tile properties. A single end-to-end hand-authored YAML → model → Builder edit → YAML fixture is still open. |
| RPT-901 | Partial | Model, schema, Builder, AI generation/repair/apply, query, chart, and Reports state tests exist. The story remains an umbrella release-qualification item and cannot be closed by the authoring program alone. |
| RPT-302 | Done | The central presentation path humanizes identifiers and consistently resolves labels, units, axes, legends, hover text, KPI cards, and tables. |
| RPT-307 | Partial | Bundled catalogs carry reviewed display metadata and human labels, but the broader visual migration and removal of every unsuitable legacy chart remains open. |
| RPT-903 | Open | The authoring accessibility pass does not substitute for the required Executive Summary desktop/narrow visual-accessibility suite. |
| RPT-904 | Partial | Architecture, domain, replacement design, chart/algorithm references, and authoring guides are maintained, but this umbrella story remains open until every reporting increment and its upgrade guidance ship. |

## KPI recipe crosswalk

| Story | Baseline | Evidence and remaining boundary |
|---|---|---|
| KPR-101 | Partial | The shared browser previews exact generated processor, metric, and tile YAML with provenance and generated IDs. A full surrounding-file diff remains open. |
| KPR-102 | Open | No aggregate-query recipe preview with date window, freshness, and provenance is shipped. YAML preview is not a substitute. |
| KPR-103 | Open | The recipe flow does not yet classify scalar, multi-output, empty, non-numeric, and unsupported-grouping result shapes from an aggregate preview. |
| KPR-104 | Partial | Builder and AI installs use transactional write plus post-write validation and rollback. Recipe-specific audit events remain open. |
| KPR-105 | Partial | Metric-only and existing-page placement are available. Creating a new page inside the recipe flow remains open. |
| KPR-106 | Open | The recipe default is preserved, but reviewed alternative-chart suggestions are not implemented. |
| KPR-107 | Open | Exact identifier collisions are blocked; semantic near-duplicate detection and reuse are not implemented. |
| KPR-108 | Open | There is no guarded undo for the latest recipe transaction. |
| KPR-201 | Partial | The preview names source fields/states and current/proposed computation hashes, and authoring hands off to Data Load. Earliest usable date and date-window planning remain open. |
| KPR-202 | Done | Safe processor-state proposals exist for the supported sketch families with deterministic parameters and source-field mapping. |
| KPR-203 | Open | The recipe workflow does not yet classify no replay, forward-only, raw replay, and impossible-without-source-field outcomes. |
| KPR-204 | Open | Storage/time estimates and a selected backfill date range are not presented for approval. |
| KPR-205 | Open | Authoring deliberately keeps apply and ingestion separate; the resumable waiting-recipe orchestration described by this story is not implemented. |
| KPR-206 | Open | A failed or externally completed backfill cannot resume a waiting recipe installation. |
| KPR-207 | Partial | Compatible sketch parameters and incompatible binding checks exist, but complete guidance for every supported parameter family remains open. |
| KPR-504 | Partial | Copilot and AI Studio produce a session-local, reviewable install patch for governed recipe IDs. Broader natural-language composition over the complete recommendation contract remains open. |
| KPR-507 | Open | The privacy-safe authoring funnel does not collect or rank recipe recommendations from aggregate usage signals. |
| KPR-904 | Partial | Native controls, visible focus, contrast checks, narrow layouts, and the authoring accessibility matrix cover the shared browser in context. Dedicated recipe search/mapping/diff/error screen-reader journeys remain open. |
| KPR-906 | Partial | The authoring funnel measures entry through report/run handoff without customer data. Recipe-specific discovery success, reuse ratio, preview failure, backfill abandonment, and stale-version measures remain open. |

## Regression ownership

The program converts each configuration-authoring mismatch into a test at the
lowest stable boundary instead of duplicating an RPT or KPR story:

| Contract | Regression coverage |
|---|---|
| Prompt and log privacy | `tests/unit/test_ai_studio_logging.py`, `tests/unit/test_ai_studio_helpers.py` |
| Format-aware preview/source separation | `tests/unit/test_ai_studio_helpers.py` |
| Provider preflight and bounded repair | `tests/unit/test_ai_studio_helpers.py`, `tests/unit/test_ai_copilot.py` |
| Revision-keyed validation and dependency-closed review | `tests/unit/test_ai_studio_helpers.py`, `tests/unit/test_ai_copilot.py` |
| Transactional Builder/Studio apply | `tests/unit/test_phase5_builder.py`, `tests/unit/test_ai_studio_helpers.py` |
| Truthful step-local draft lifecycle | `tests/unit/test_phase5_builder.py`, `tests/unit/test_ui_guardrails.py` |
| Build entry, outcome handoff, and rollout flag | `tests/unit/test_ui_navigation.py`, `tests/unit/test_authoring_instrumentation.py` |
| Native UI, hierarchy, disclosure, focus, and theme contrast | `tests/unit/test_ui_guardrails.py`, `tests/unit/test_ui_components.py` |

Any later change that expands one of these contracts must update the owning RPT
or KPR row as well as its regression coverage; this baseline is not permission
to fork the shared reporting or recipe architecture.
