# Value Stream Documentation

Value Stream is a configuration-driven, aggregate-first business intelligence
platform. It loads file-based exports (typically Pega CDH Interaction History
and Product Holdings), reduces raw rows to persisted mergeable aggregate
statistics during one chunk pass, and serves reports, dashboards, aggregate
chat, SDK queries, and SQL exports from those aggregates. Raw event rows never
survive the chunk pass.

## Find Your Path

| Reader | Start here | Then read | Reference when needed |
|---|---|---|---|
| Business user / analyst | [Getting started](tutorials/getting-started.md) | [Running reports](guides/users/running-reports.md), [Chat with data](guides/users/chat-with-data.md) | [FAQ](reference/faq.md), [Business functionality](concepts/business-functionality.md) |
| Product owner / stakeholder | [Product overview](concepts/product-overview.md) | [Business functionality](concepts/business-functionality.md) | [Reporting backlog](design/reporting-backlog.md), [KPI recipe backlog](design/kpi-recipe-backlog.md) |
| Workspace operator / data engineer | [Pega export tutorial](tutorials/pega-export.md) | [Operations runbook](guides/operations/runbook.md), [Migration & backfill](guides/operations/migration.md) | [CLI reference](reference/cli.md), [Readers & formats](reference/readers-and-formats.md) |
| Developer | [Architecture overview](concepts/architecture-overview.md) | [Architecture](concepts/architecture.md), [Domain model](concepts/domain-model.md), [Replacement design](design/replacement-design.md) | [Processors](reference/processors.md), [Algorithms](reference/algorithms.md), [Expression DSL](reference/expression-dsl.md) |
| Auditor / reviewer | [Business functionality](concepts/business-functionality.md) (governance) | [Troubleshooting](guides/operations/troubleshooting.md) (escalation data) | [FAQ](reference/faq.md), [Domain model](concepts/domain-model.md) |
| Documentation maintainer | [Documentation guide](meta/documentation-guide.md) | — | — |

## How These Docs Are Organized

The documentation follows the [Diátaxis](https://diataxis.fr/) model: each page
belongs to exactly one genre, and roles enter through the paths above rather
than through per-role copies of the same content.

| Section | Genre | Question it answers |
|---|---|---|
| [Tutorials](tutorials/getting-started.md) | Learning-oriented | "Can I try it end to end?" — runnable, start to finish |
| [Guides](guides/users/running-reports.md) | Task-oriented | "How do I accomplish X?" — users, configuration, operations |
| [Concepts](concepts/product-overview.md) | Understanding-oriented | "Why is it built like this?" — product, business, architecture, decisions |
| [Reference](reference/cli.md) | Information-oriented | "What exactly is X?" — CLI, DSL, processors, algorithms, charts, API, FAQ |
| [Design docs](design/replacement-design.md) | Engineering history | Master design, implementation plan, backlog, feature designs |

## Current Scope

| Area | Current status |
|---|---|
| Ingestion | File discovery, readers, transforms, chunk ledger, processor fan-out, and aggregate writes |
| Analytics | Binary outcomes, numeric distributions, score distributions, lifecycle, sets, funnels, and snapshots |
| Reports | Streamlit dashboards, report filters, chart rendering, inspection mode, and freshness metadata |
| Configuration | YAML catalog, validation, direct editors, shared KPI recipe library, deterministic builder, and AI-assisted draft flow |
| Operations | Validation, data load, run history, chunk detail, vacuum, DuckDB export, migration, and backfill |
| Headless access | Local read-only stdio MCP and read-only FastAPI HTTP API; governed SQL is opt-in |
| Deferred | Remote HTTP MCP and multi-user/OIDC service deployment |

## The Docs Are the Spec

These documents are the source of truth for Value Stream's behavior. Every
behavior-changing pull request updates the relevant page in the same commit,
and doc-only pull requests are first-class. The
[documentation guide](meta/documentation-guide.md) defines the structure,
conventions, completeness checklist, and maintenance rules.
