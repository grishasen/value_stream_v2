# KPI Recipe Library

The KPI recipe library gives analysts reusable business definitions without
making hidden runtime decisions. A recipe describes a business question,
calculation, compatible aggregate inputs, presentation defaults, and method
caveats. Installing one materializes normal catalog YAML for review; the
packaged recipe is inert until that explicit action.

## Contract

The built-in library is stored in
`src/valuestream/recipes/kpis.yaml`, validates against
`schemas/kpi-recipes.json`, and is loaded through a typed Pydantic model. The
same browser and instantiation logic is used by Configuration Builder and AI
Configuration Studio.

A recipe contains:

| Field | Meaning |
|---|---|
| `id`, `version` | Stable recipe identity and immutable version number |
| `title`, `domain`, `summary` | Business-facing discovery metadata |
| `business_questions`, `tags` | Search and interpretation aids |
| `maturity` | `draft`, `reviewed`, or `certified` governance state |
| `processor_kinds` | Processor families that may satisfy the recipe |
| `inputs` | Required business roles, field/algorithm selection mode, accepted state types, metadata filters, pairing/exclusion rules, and preferences |
| `default_metric_id` | Proposed ID; the installer adds a stable numeric suffix on collision |
| `metric` | A normal metric definition with exact-value placeholders such as `${processor_id}` |
| `method` | Calculation, accuracy class, algorithm, and caveat |
| `report` | Recommended chart, placement, and optional KPI comparison defaults |

Template substitution is deliberately closed: a placeholder must occupy the
whole YAML scalar and must name a declared binding or built-in installer value.
Recipes cannot inject Python, SQL, or expression strings. Formula recipes
materialize the same closed expression AST used by hand-authored metrics.

## Readiness

The compatibility resolver evaluates a recipe against one Processor and
returns one of four states:

| State | Meaning | Install behavior |
|---|---|---|
| `ready` | Every required state/stage maps unambiguously | Installer preselects all bindings |
| `mapping_required` | Compatible candidates exist, but business intent is ambiguous | User chooses each unresolved input |
| `backfill_required` | A required aggregate state or stage is absent from the current processor contract | Configurable sketch inputs can propose a processor state; non-configurable inputs remain blocked |
| `incompatible` | Processor kind cannot execute the recipe | Processor is excluded from the selector |

Matching is deterministic. The resolver filters by source (`state` or
`stage`), state type, required/absent state metadata, and strict semantic roles,
then applies ordered name/algorithm preferences. A sole remaining candidate is
safe to map automatically; multiple business fields, algorithms, stages, or
populations require a user choice. Paired score digests require matching score
metadata and funnel endpoints must be different.

Readiness never examines raw event rows. It reads only processor configuration
and its effective aggregate-state contract.

For a `field_algorithm` input, the browser augments that static readiness with
safe configuration choices. It lists every processor-owned candidate field —
including every `group_by` field and configured identity/property field — and
every algorithm declared compatible by the recipe. A field/algorithm pair
that is not yet a processor state is shown as a proposal, not as unavailable.

## Install Workflow

Both authoring surfaces provide the same steps:

1. Search by KPI, business question, tag, calculation, or domain.
2. Read the business definition, accuracy class, algorithm, and caveat.
3. Select a compatible Processor and review readiness.
4. Select business fields and any recipe-compatible algorithm for sketch-backed
   metrics, stages or populations for funnels, and any other ambiguous roles.
   Proposed states are identified as requiring a first run or backfill.
5. Choose a unique metric ID.
6. Optionally add the recommended tile to an existing dashboard page.
7. Select **Review changes** and inspect the exact generated
   `processors.yaml`, `metrics.yaml`, and `dashboards.yaml` patches.
8. When a processor state is proposed, review the source, source fields,
   affected states, and current/proposed processor computation hashes.
9. Apply explicitly. If materialization is required, run the named source from
   Data Load (or use **Apply Draft & Run Source** in AI Configuration Studio).

Configuration Builder writes any proposed processor state first, followed by
the materialized metric and optional tile. All writes and post-write catalog
validation run inside one rollback boundary: a write or validation failure
restores every catalog file. After success, the Builder reloads the catalog,
switches to **Edit Existing Metric**, opens the new metric, and presents a
direct Data Load handoff when the processor contract changed.

AI Configuration Studio adds the same processor/metric/tile artifacts to its
session-local draft. The workspace remains unchanged until **Save & Export**
applies that draft. Applying a draft uses one rollback boundary for all four
catalog files plus `ai.yaml`, including post-write validation. Recipe
confirmation itself never starts ingestion; the separate **Apply Draft & Run
Source** action is the explicit materialization approval.

An installed metric carries recipe provenance as ordinary permissive metric
metadata:

```yaml
metrics:
  VS_Unique_Entities:
    source: ih_engagement
    kind: approx_distinct_count
    state: UniqueCustomers_cpc
    description: Approximate distinct entity count from a persisted mergeable sketch.
    display:
      label: Unique entities
      unit: entities
      value_format: integer
      direction: higher_is_better
    recipe:
      id: audience.unique_entities
      version: 1
```

If `Channel` and CPC were selected before that state existed, the same action
also adds the ordinary processor configuration:

```yaml
processors:
  - id: ih_engagement
    # existing processor fields remain unchanged
    states:
      Channel_cpc:
        type: cpc
        source_column: Channel
        lg_k: 11
```

After installation, `metrics.yaml` and `dashboards.yaml` remain the sole source
of runtime behavior. Editing or removing the packaged recipe does not silently
change an installed metric.

The authoring UI does not use state IDs as business choices. For
example, `UniqueSubjects_hll` is presented as field `SubjectID`, algorithm
`HLL`; `SubjectID_theta` is presented as field `SubjectID`, algorithm `Theta`.
The state ID and parameters remain available under **Technical aggregate
bindings**. If `SubjectID_cpc` is not configured, CPC still appears as a
recipe-compatible algorithm and is labelled as a proposed state. Exact
engagement roles are locked to the processor's
`Positives`/`Negatives` states, ROC AUC selects one score field and pairs its
positive/negative digests automatically, distribution quantiles exclude
outcome-conditioned digests, and funnel recipes expose stages/populations.
If two states have identical business field/algorithm metadata and no
distinguishing population, installation is blocked instead of exposing an
internal state-ID choice.

## Built-in Recipes

| Recipe ID | Business KPI | Required capability | Accuracy | Default report |
|---|---|---|---|---|
| `engagement.engagement_rate` | Engagement rate | Binary positive/negative counts | Exact | KPI card |
| `engagement.positive_outcomes` | Positive outcomes | Binary positive count | Exact | KPI card |
| `audience.unique_entities` | Distinct audience/reach | CPC, HLL, or Theta state; CPC preferred | Approximate | KPI card |
| `distribution.median` | Median numeric/score value | t-digest or KLL state | Approximate | KPI card |
| `distribution.p95` | High-tail numeric/score value | t-digest or KLL state | Approximate | KPI card |
| `model_quality.roc_auc` | Ranking discrimination | Matched positive/negative t-digests | Approximate | KPI card |
| `funnel.conversion_rate` | Funnel completion rate | Start/completion count states | Exact | KPI card |
| `funnel.dropoff_rate` | Funnel stage loss | Ordered funnel stages | Exact | KPI card |
| `lifecycle.summary` | Entity lifecycle measures | Entity lifecycle processor | Exact | Table |
| `category.top_items` | Frequent categories | Frequent-items/Top-K state | Approximate | Table |

The unique-entity recipe prefers CPC states created by current processor
defaults while accepting HLL and Theta states. Theta is useful when the same
persisted set also supports intersections or differences. The recipe does not
convert or merge different sketch families together.

## Versioning and Governance

- A recipe version is immutable once published. Calculation, input semantics,
  direction, or accuracy changes require a new version.
- Copy edits that do not alter interpretation may remain in the same version,
  but installed metrics are never rewritten automatically.
- `certified` means the calculation and business interpretation have named
  owners and reference tests; it does not make an approximate sketch exact.
- An installed metric records the recipe ID/version so future upgrade tooling
  can show a diff instead of silently migrating it.
- Workspace-owned recipes, approval owners, deprecation, upgrade assistance,
  and report packs are planned in the
  [KPI recipe backlog](../design/kpi-recipe-backlog.md).

## Aggregate and Backfill Rules

Installing a recipe that uses an already configured state does not change the
processor. When a selected field/algorithm pair is absent, the installer adds a
deterministic state definition to `processors.yaml` (or the AI draft) and binds
the metric to that state. Default parameters are CPC `lg_k=11`, HLL/Theta
`lg_k=12`, t-digest `k=500`, KLL `k=200`, and Top-K
`lg_max_map_size=10`; they remain inspectable in the technical binding.

This configuration is intentionally allowed before any data is loaded. The
preview names the changed processor states, source fields, source, and the
processor computation-hash transition. The next normal source run materializes
the new state for a fresh workspace and reprocesses discovered chunks whose
computation contract changed. Operators who need a narrower historical window
can use the normal backfill workflow. Until matching-hash aggregates exist,
reports continue to show **Backfill required**.

Adding the metric does not convert an HLL or Theta blob to CPC, reconstruct a
digest, read persisted raw rows, or start a run. Replay and computation-hash
behavior follow the compatibility rules in the
[domain model](../concepts/domain-model.md).

All recipe metrics execute through the normal query layer. Recipe browsing,
mapping, and report placement never persist or query raw event rows.
