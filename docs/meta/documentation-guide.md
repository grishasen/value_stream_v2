# Documentation Guide

This page defines how the Value Stream documentation is structured and
maintained. It merges the former reading-order index and the wiki structure
plan. If you are looking for the documentation itself, start at the
[home page](../index.md).

## The Rule That Matters Most

**The docs are the spec.** Every behavior-changing PR updates the relevant
page in the same commit. Doc-only PRs are first-class; reviewers prioritize
them. When in doubt, ask: "Could a new engineer build this feature from the
doc alone?" If the answer is no, the doc is wrong, not the engineer.

## Structure

The docs follow the [Diátaxis](https://diataxis.fr/) model. Each page belongs
to exactly one genre; roles enter through the reading paths on the home page.

```text
docs/
  index.md          # single home page: role paths + genre map
  tutorials/        # learning-oriented, guaranteed runnable on a clean clone
  guides/           # task-oriented how-tos: users/, configuration/, operations/
  concepts/         # explanation: product, business, architecture, domain, adr/
  reference/        # information-oriented manuals: cli, dsl, processors, ...
  design/           # engineering design & history: replacement design, plans, mockups
  meta/             # this guide
```

Rules of placement: a page that teaches by doing is a tutorial; a page that
gets a task done is a guide; a page that explains why is a concept; a page you
look things up in is reference. Design history (plans, backlogs, feature
designs) is not user documentation and stays under `design/`.

## Audiences

| Audience | Primary need | Best section |
|---|---|---|
| Product owner | Understand scope, status, and business value | Concepts (product overview) |
| Marketing analyst or decision scientist | Interpret reports and metrics | Concepts (business functionality) + user guides |
| Data engineer or operator | Load data, validate workspaces, monitor runs | Operations guides + CLI reference |
| Application engineer | Extend processors, charts, query logic, or UI | Concepts (architecture) + reference + design |
| Auditor or reviewer | Trace numbers to config, data, and run history | Business functionality, troubleshooting, domain model |

## Reading Order for Engineers

Read the concepts pages first for orientation, then go deep in this order.

| # | Document | What it gives you | Time |
|---|---|---|---|
| 1 | [Architecture](../concepts/architecture.md) | Big-picture mental model: components, data flow, NFRs, deployment | 20 min |
| 2 | [Domain model](../concepts/domain-model.md) | Glossary and concept relationships — fix the vocabulary first | 15 min |
| 3 | [Replacement design](../design/replacement-design.md) | Master design: storage layout, YAML DSL, APIs, migration plan | 60 min |
| 4 | [Processors](../reference/processors.md) | Per-processor algorithms (chunk, merge, compact) and edge cases | 45 min |
| 5 | [Algorithms](../reference/algorithms.md) | Pooled variance, sketches, statistical tests, ML metrics, RFM | 30 min |
| 6 | [Expression DSL](../reference/expression-dsl.md) | Formal grammar and Polars translation of the AST | 20 min |
| 7 | [Readers & formats](../reference/readers-and-formats.md) | File-discovery rules, reader catalog, transform catalog | 25 min |
| 8 | [Chart catalog](../reference/chart-catalog.md) | Plotly chart kinds, required tile fields, render outlines | 15 min |
| 9 | [FAQ](../reference/faq.md) | Q&A by area: storage, config, ingestion, querying, ops, security | reference |
| 10 | [Implementation plan](../design/implementation-plan.md) | Repo layout, tooling, phased delivery | 30 min |
| 11 | [Reporting backlog](../design/reporting-backlog.md) | Current reporting-usability delivery backlog | 15 min |
| 12 | [KPI recipe backlog](../design/kpi-recipe-backlog.md) | Shared KPI catalog and workflow delivery backlog | 15 min |
| 13 | [Configuration Studios remediation backlog](../design/ai-studio-remediation-backlog.md) | Unified audit-driven correctness, UX, and release backlog for Configuration Builder and AI Configuration Studio | 20 min |

Visual sketches (sitemap, ingestion flow, dashboard, tile anatomy, Builder,
Pipelines) live in [`design/mockups/`](../design/mockups/index.html).

## Content Rules

- Start each page with the reader outcome: what the page helps the reader do.
- Keep pages task-scoped. If a section needs deep algorithms, link to the
  reference instead of copying it.
- Single-source commands: each command sequence lives in exactly one guide;
  other pages link to it (tutorials may repeat their own runnable steps).
- Tutorials must pass on a clean clone and use `examples/demo`; tutorials
  generate their own data. Additional catalog-only workspaces may be checked
  in as migration or configuration showcases, but must validate without data
  and document their source assumptions next to the example.
- Prefer tables for status, responsibilities, and decision criteria.
- Mark future or deferred capabilities explicitly; current behavior comes
  before roadmap content.
- Include verification steps after procedures that change data or artifacts.
- No phase numbers in filenames or titles — delivery history belongs in the
  implementation plan and the changelog.
- Every processor / metric / chart is referred to by its **kind** name (e.g.
  `binary_outcome`, `formula`, `line`) once introduced.
- All YAML examples must validate against the JSON Schemas under `schemas/`.
- `<workspace>/` is the workspace root; every relative path in the docs is
  implicitly under it.

## Navigation Rules

- `docs/index.md` is the only index. The README carries identity, quickstart,
  and a link here — nothing more.
- Every page is in the `mkdocs.yml` nav; a page that fits nowhere probably
  belongs under `design/` or should not be merged.
- Renames and moves get redirect mappings in `mkdocs.yml`
  (`mkdocs-redirects`) so existing deep links keep working.
- `mkdocs build --strict` is a CI gate; broken internal links fail the build.

## Doc-Completeness Checklist

If you're building Value Stream from scratch, you should be able to answer
all of these from the docs. If you can't, file a doc bug.

- [ ] What does Value Stream do, and what does it explicitly not do? → architecture §1, §20; FAQ §H
- [ ] What does the directory layout of a workspace look like? → architecture §9; replacement design §6.1
- [ ] What YAML files define the catalog and what does each contain? → architecture §10; replacement design §7; catalog schemas
- [ ] How is a chunk identified and processed? → architecture §6, §18; readers & formats §2; processors §1
- [ ] How does each processor compute its aggregates? → processors §3–9
- [ ] How are sketches built and merged? → algorithms §2.4–2.5, §6
- [ ] How are statistical tests computed? → algorithms §3, §8
- [ ] What's the formula for ROC AUC / AP / calibration from t-digests? → algorithms §4
- [ ] How does the expression DSL translate to Polars? → expression DSL §4
- [ ] How does the planner pick a physical aggregate? → architecture §7; replacement design §9.1; FAQ §D1
- [ ] How is the read-only API laid out and secured? → API & MCP reference; security guide
- [ ] What chart kinds exist and what fields do they need? → chart catalog §3
- [ ] What happens when config changes? → domain model §5; FAQ §A4–A5
- [ ] How do I migrate from the legacy app? → migration guide; replacement design §12; FAQ §I
- [ ] What's the build plan? → implementation plan §5–7
- [ ] What technology and version do I need? → implementation plan §3

## Maintenance Checklist

Update the docs when any of the following changes:

- A Streamlit page is added, renamed, or removed → user guides, UI tour.
- A CLI command, option, or example workspace path changes → CLI reference,
  affected tutorials/guides, README quickstart.
- A processor, metric kind, chart kind, reader, or transform is added →
  reference pages, relevant tutorial.
- The aggregate storage layout or metadata behavior changes → architecture,
  replacement design, ADR if the decision changes.
- Headless API/MCP scope or security posture changes → API & MCP reference,
  security guide.
- Migration or backfill assumptions change → migration guide.
- A significant design decision is made → new ADR under `concepts/adr/`.

## Documentation Backlog

- Public Python SDK reference generated from docstrings.
- Screenshot-based report guide after the UI stabilizes further.
- Workspace template guide for production onboarding.
- Security and access-control updates when remote MCP/OIDC land.
