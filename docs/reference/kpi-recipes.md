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
| `required_states` | Fixed state names and types required by metric implementations such as Test/Control tests |
| `parameters` | Optional bounded install-time numbers, including percentages displayed in business units |
| `inputs` | Required business roles, configured grouping dimensions, field/algorithm selection mode, accepted state types, metadata and filtered-state requirements, recipe-authored state templates, pairing/exclusion rules, and preferences |
| `default_metric_id` | Proposed ID; the installer adds a stable numeric suffix on collision |
| `metric` | A normal metric definition with exact-value placeholders such as `${processor_id}` |
| `method` | Calculation, accuracy class, algorithm, and caveat |
| `report` | Recommended chart, placement, optional KPI comparison defaults, and optional required `x` axis for non-scalar charts |

Template substitution is deliberately closed: a placeholder must occupy the
whole YAML scalar. Metric and report-axis placeholders must name a declared binding
or built-in installer value; state-template placeholders must name a declared,
bounded recipe parameter. Recipes cannot inject Python, SQL, or expression
strings. Formula recipes and recipe-authored filtered states materialize the
same closed expression AST used by hand-authored catalog objects.

## Readiness

The compatibility resolver evaluates a recipe against one Processor and
returns one of four states:

| State | Meaning | Install behavior |
|---|---|---|
| `ready` | Every required state/stage maps unambiguously | Installer preselects all bindings |
| `mapping_required` | Compatible candidates exist, but business intent is ambiguous | User chooses each unresolved input |
| `backfill_required` | A required aggregate state or stage is absent from the current processor contract | Configurable sketch inputs and closed recipe-authored state templates can propose a processor state; other non-configurable inputs remain blocked |
| `incompatible` | Processor kind cannot execute the recipe | Processor is excluded from the selector |

Matching is deterministic. Fixed `required_states` must be present with their
declared types. Dimension inputs select only configured `group_by` fields;
they never propose a new state. The resolver filters other inputs by source
(`state` or `stage`), state type, required/absent state metadata, required `where`
predicates or the exact parameter-resolved state template, and strict semantic
roles, then applies ordered name/algorithm preferences. A sole remaining
candidate is safe to map automatically; multiple business fields, algorithms,
stages, or populations require a user choice. Paired score digests require
matching score metadata and funnel endpoints must be different.

Readiness never examines raw event rows. It reads only processor configuration
and its effective aggregate-state contract.

For a `field_algorithm` input, the browser augments that static readiness with
safe configuration choices. It lists every processor-owned candidate field —
including every `group_by` field and configured identity/property field — and
every algorithm declared compatible by the recipe. A field/algorithm pair
that is not yet a processor state is shown as a proposal, not as unavailable.
The browser never invents a business-specific `where` predicate. It may
propose one only when the recipe contains the complete closed AST and every
substituted value is a declared, bounded parameter.

## Install Workflow

Both authoring surfaces provide the same steps:

1. Search by KPI, business question, tag, calculation, or domain.
2. Read the business definition, accuracy class, algorithm, and caveat.
3. Edit any bounded recipe parameters. Percentages are displayed as percentages
   even though the generated catalog expression stores their decimal value.
4. Select a compatible Processor and review readiness.
5. Select business fields and any recipe-compatible algorithm for sketch-backed
   metrics, stages or populations for funnels, and any other ambiguous roles.
   Proposed states are identified as requiring a first run or backfill.
6. Choose a unique metric ID.
7. Optionally add the recommended tile to an existing dashboard page.
8. Select **Review changes** and inspect the exact generated
   `processors.yaml`, `metrics.yaml`, and `dashboards.yaml` patches.
9. When a processor state is proposed, review the source, source fields,
   affected states, and current/proposed processor computation hashes.
10. Apply explicitly. If materialization is required, follow the named source
   handoff to Data Load and start the run there.

When the recipe's default metric ID is retained, installation also retains the
recipe-authored display label. Choosing a custom metric ID derives the initial
display label from that ID; the label can then be edited independently without
renaming the metric.

Configuration Builder writes any proposed processor state first, followed by
the materialized metric and optional tile. All writes and post-write catalog
validation run inside one rollback boundary: a write or validation failure
restores every catalog file. After success, the Builder reloads the catalog,
switches to **Edit Existing Metric**, opens the new metric, and presents a
direct Data Load handoff when the processor contract changed.

AI Configuration Studio adds the same processor/metric/tile artifacts to its
session-local draft. The workspace remains unchanged until **Apply to
workspace** writes the reviewed revision. Applying uses one rollback boundary
for all four catalog files plus `ai.yaml`, including post-write validation.
Recipe confirmation and apply never start ingestion; the outcome links to Data
Load when the processor contract requires materialization.

For recipe-created metrics, `display.label` is derived from the chosen Metric
ID. The recipe still supplies units, formatting, direction, and calculation
metadata, but it cannot replace the metric's identity with a generic recipe
name.

An installed metric carries recipe provenance in the strict typed metric
contract:

```yaml
metrics:
  VS_Unique_Entities:
    processor: ih_engagement
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

When a recipe has editable parameters, its installed metric also records the
resolved values under `recipe.parameters`. This makes the chosen definition
reviewable without making the packaged recipe part of runtime execution.

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

Choose a recipe by the business question it answers. The 36 recipes below are
grouped by business purpose; recipe IDs remain unchanged even when their
technical domain differs from the group. Examples use Pega Customer Decision
Hub (CDH) terminology. All numbers and
action scenarios below are illustrative, not measured industry results or benchmarks.

In CDH, a **customer** is identified by `CustomerID`, and a **unique action** by
`ActionID`, built from `Issue / Group / Name`. `Propensity` is the model’s raw
response probability; `FinalPropensity` includes exploration and other
adjustments. `Priority` is the arbitration ranking score and can include value,
context and levers, so it must not be read as a probability. `ResponseTime` is
`OutcomeTime − DecisionTime` in seconds. **CTR** means click-through rate.

- [Audience reach and activity mix](#audience-reach-and-activity-mix)
- [Engagement and positive responses](#engagement-and-positive-responses)
- [Business experiments: Test and Control](#business-experiments-test-and-control)
- [Customer journeys and lifecycle](#customer-journeys-and-lifecycle)
- [Marketing costs and revenue](#marketing-costs-and-revenue)
- [Products and revenue mix](#products-and-revenue-mix)
- [Value distributions and response time](#value-distributions-and-response-time)
- [Prediction quality](#prediction-quality)
- [Adaptive learning and exploration](#adaptive-learning-and-exploration)
- [Contact policy and repeated impressions](#contact-policy-and-repeated-impressions)

**Accuracy** describes how the result is calculated: **Exact** uses configured
counts and sums; **Approximate** uses compact statistical summaries or an
approximate contact-history interpretation; **Statistical** denotes an
inference test, confidence interval or heuristic diagnostic. Exact arithmetic still depends on the configured
population and outcome definitions.

### Audience Reach and Activity Mix

Understand how many different customers and actions appear in Interaction
History and which actions or treatments account for the most activity.

| Business KPI | Explanation | Recipe ID | Required capability | Accuracy | Default report |
|---|---|---|---|---|---|
| Unique customers or unique actions | Estimates distinct customers reached or distinct actions observed, depending on the selected field; repeat appearances count once. **Example:** 12,000 Interaction History events can represent about 4,500 unique customers (`CustomerID`) and 120 unique actions (`ActionID`). Install the recipe once for each field. | `audience.unique_entities` | CPC, HLL, or Theta state; CPC preferred | Approximate | KPI card |
| Top actions or treatments | Shows which action names or treatments occur most often in the selected Interaction History population. **Example:** A top-actions table estimates 6,000 events for a credit-card action and 3,000 for a savings action; these are event counts, not impression rates or unique-customer counts. | `category.top_items` | Frequent-items/Top-K state | Approximate | Table |

The unique-entity recipe prefers CPC states created by current processor
defaults while accepting HLL and Theta states. Theta is useful when the same
persisted set also supports intersections or differences. The recipe does not
convert or merge different sketch families together.

### Engagement and Positive Responses

Measure both the success rate and the volume of customer responses.

| Business KPI | Explanation | Recipe ID | Required capability | Accuracy | Default report |
|---|---|---|---|---|---|
| Engagement rate (CTR)/ Conversion rate | The share of classified action outcomes that are clicks. Count `Clicked` as positive and `Impression` or `Pending` as negative after deduplication. **Example:** 80 clicks plus 920 impression/pending outcomes give 80 / 1,000 = 8% CTR. | `engagement.engagement_rate` | Binary positive/negative counts | Exact | KPI card |
| Positive outcomes: clicks or conversions | Counts successful action outcomes under the selected processor: clicks for engagement, conversions for conversion reporting. **Example:** 250 `Clicked` outcomes produce 250 clicks; a separate metric with 40 `Conversion` outcomes produces 40 conversions. These are outcome counts, not unique customers. | `engagement.positive_outcomes` | Binary positive count | Exact | KPI card |

### Business Experiments: Test and Control

Measure the size of a business effect and the evidence behind it. For example,
`ExperimentGroup` identifies variants within `ExperimentName`; `Clicked` is a
positive response. Choose the relevant grouping when installing a recipe:
`ExperimentGroup`, `ModelControlGroup`, or `DefaultBannerControlGroup` when
that field is retained by the processor.

| Business KPI | Explanation | Recipe ID | Required capability | Accuracy | Default report |
|---|---|---|---|---|---|
| Experiment sample size | How many classified action outcomes each experiment arm contains. **Example:** Test has 10,000 click/impression/pending outcomes and Control has 8,000; inspect this balance before comparing CTR. | `experiments.sample_size` | Positive/negative counts and experiment grouping | Exact | Bar by arm |
| Experiment response rate | The click rate within each arm. **Example:** 400 clicks from 10,000 Test outcomes give 4% CTR; 300 from 10,000 Control outcomes give 3%. | `experiments.response_rate` | Positive/negative counts and experiment grouping | Exact | Bar by arm |
| Test/control effect and lift | Compares response rates, absolute difference, relative lift and a 95% confidence interval. **Example:** Test CTR of 4% against Control CTR of 3% is +1 percentage point and about +33% relative lift. | `experiments.test_control_comparison` | Exact `Positives`/`Negatives` counts and a grouping with Test/Control labels | Statistical | Table with effect and interval |
| Z-test: Test versus Control | Tests whether two response rates differ beyond the variation expected from sampling. **Example:** Compare 400 clicks from 10,000 Test outcomes with 300 from 10,000 Control outcomes; the table reports the z-score and two-sided p-value. | `experiments.z_test` | Exact `Positives`/`Negatives` counts and a grouping with Test/Control labels | Statistical | Statistical table |
| Chi-square test across experiment variants | Checks whether response rates differ across all variants. **Example:** Compare CTR of 4%, 3% and 2% for three action strategies; a small p-value suggests some rates differ, but does not identify the winning strategy. | `experiments.chi_square_test` | Exact `Positives`/`Negatives` counts and experiment grouping | Statistical | Chi-square and G-test table |

Test/Control comparison and Z-test use only the named `Test` and `Control`
arms. Chi-square and G-test include every retained variant, including `NBA`
when using `ModelControlGroup`. Filter or group by `ExperimentName` so separate
experiments are not pooled. Read the effect size, sample counts and confidence
interval alongside significance; a p-value is not the probability that Test
is better. Statistical tests assume suitable sample sizes and independent
observations. Repeated responses from the same customer can violate that
assumption, and a causal conclusion also requires a valid experimental design.

### Customer Journeys and Lifecycle

Compare impression, click and conversion volumes, and use product holdings
when available to understand the customer relationship.

| Business KPI | Explanation | Recipe ID | Required capability | Accuracy | Default report |
|---|---|---|---|---|---|
| Conversions | Counts successful conversion outcomes after the conversion processor’s deduplication. **Example:** 40 `Conversion` outcomes give 40 conversions, even if some belong to the same customer. | `conversion.conversions` | Conversion-positive count | Exact | KPI card |
| Conversion rate | The share of classified conversion opportunities that convert. **Example:** 40 `Conversion` and 160 `NoConversion` outcomes give a 20% conversion rate; impressions are not this denominator. | `conversion.conversion_rate` | Conversion and NoConversion counts | Exact | KPI card |
| Funnel completion rate | Compares action outcomes at two selected stages, such as impressions and conversions. **Example:** 50 conversion outcomes divided by 1,000 impression outcomes in the same Web reporting slice give a 5% impression-to-conversion rate. | `funnel.conversion_rate` | Start/completion count states | Exact | KPI card |
| Funnel drop-off rate | Shows the relative decrease in outcome counts between selected stages. **Example:** 100 click outcomes against 1,000 impression outcomes give 90% impression-to-click drop-off; 40 conversions against those 100 clicks give 60% click-to-conversion drop-off. | `funnel.dropoff_rate` | Ordered funnel stages | Exact | KPI card |
| Customer lifecycle summary | Summarizes a customer’s recorded product holdings, value and purchase timing to inform CDH retention or cross-sell decisions. **Example:** With a separate holdings source, a customer’s row could show three products held, €600 in recorded value and a last purchase 10 days before the observation end. | `lifecycle.summary` | Entity lifecycle processor with distinct holdings, monetary total, first-purchase and last-purchase states | Exact | Table |

For example, `action_funnel` compares `Impression`, `Clicked` and `Conversion` outcome
counts for Web and Mobile. These ratios describe stage volumes; they do not
track individual customers through an ordered path or attribute a conversion
to a particular impression. Its impression-to-click denominator also differs
from engagement CTR, which includes clicks and impression/pending outcomes.

The lifecycle recipe requires a separate product-holdings source and an
`entity_lifecycle` processor; Holdings represent products owned, not actions shown or clicked.
Its recorded monetary value is not a forecast of future customer value.

### Marketing Costs and Revenue

Choose the cost basis that matches your CDH implementation. A charge per
impression and a charge per interaction answer different accounting questions;
these are alternative totals, not amounts to add together. Use one currency
within each total.

| Business KPI | Explanation | Recipe ID | Required capability | Accuracy | Default report |
|---|---|---|---|---|---|
| Marketing cost — impressions | Adds `Cost` once for each billable action impression. **Example:** 1,000 impressions charged at €0.02 each cost €20, regardless of how many later generate a click. | `finance.marketing_cost_impressions` | Configured `ImpressionCost`, `CostedImpressions` and `Impressions` on an impression-deduplicated processor | Exact | KPI card |
| Cost per impression | Average cost for one billable impression. **Example:** €20 spent on 1,000 action impressions gives €0.02 per impression. | `finance.cost_per_impression` | Same impression-cost states and complete cost data | Exact | KPI card |
| Marketing cost — interactions | Adds `Cost` once per billed customer interaction, even if several actions are shown. **Example:** 100 interactions at €0.10 each cost €10; three actions within each interaction do not triple the cost. | `finance.marketing_cost_interactions` | Configured `InteractionCost`, `CostedInteractions` and `Interactions` on an interaction-deduplicated processor | Exact | KPI card |
| Cost per interaction | Average cost for one billable interaction. **Example:** €10 spent across 100 customer interactions gives €0.10 per interaction. | `finance.cost_per_interaction` | Same interaction-cost states and complete cost data | Exact | KPI card |
| Marketing cost data coverage | Shows how much of the billing population has a recorded `Cost`. **Example:** Cost is present on 900 of 1,000 impressions: coverage is 90%, and the total-cost recipe withholds a misleading partial total. | `finance.cost_coverage` | Recorded-cost and total-unit counts from one billing basis | Exact | KPI card |
| Conversion revenue | Adds the `Revenue` field for conversion outcomes. **Example:** Three conversions with recorded revenue of €20, €30 and €50 produce €100 revenue; impression and click events add nothing. | `finance.revenue` | A `Revenue` sum filtered to `Outcome = Conversion` | Exact | KPI card |
| Revenue per conversion | Average revenue earned per conversion. **Example:** €1,000 revenue from 40 conversions gives €25 per conversion. | `finance.revenue_per_conversion` | Conversion-filtered revenue and conversion count from the same population | Exact | KPI card |
| Revenue per conversion opportunity | Revenue per classified conversion opportunity, including those that did not convert. **Example:** €1,000 across 40 conversions and 160 non-conversions gives €5 per opportunity. | `finance.revenue_per_opportunity` | Conversion-filtered revenue and Conversion/NoConversion counts | Exact | KPI card |

The cost recipes require explicitly configured billing states; the installer
never substitutes a generic sum or customer count for a billing population.
For example, both cost processors may retain `Outcome = Impression` before deduplication.
The impression basis deduplicates by customer, interaction, action, placement
and rank; the interaction basis deduplicates by customer and interaction only.
The latter assumes the same interaction charge is repeated on each included
impression row. If your source records the charge on a separate interaction
event, bind a processor for that event population instead. Deduplication is
within each chunk: one billed identity must belong to one source chunk.

Totals and unit costs are available only
when every included billing unit has a cost; a recorded zero remains a valid
zero. Inspect **Marketing cost data coverage** when a total is unavailable.

### Products and Revenue Mix

Compare the products or product groups represented in conversion reporting.
For example,, `ProductGroup` is the product-group dimension. The recipe installer
also allows another product or action grouping already retained by the
processor, such as `Issue` or `Group`.

| Business KPI | Explanation | Recipe ID | Required capability | Accuracy | Default report |
|---|---|---|---|---|---|
| Revenue mix by product | Shows how each product group contributes to revenue. **Example:** €600 from cards and €400 from savings produce a €1,000 mix, with cards contributing 60%; the default bars show the currency amounts. | `products.revenue_mix` | Conversion-filtered revenue plus a configured product grouping | Exact | Bar by product |
| Conversion mix by product | Shows conversion volume by product group. **Example:** 30 card conversions and 20 savings conversions give a 60%/40% volume mix; this can differ from the revenue mix. | `products.conversion_mix` | Conversion-positive count plus a configured product grouping | Exact | Bar by product |
| Product revenue concentration | Ranks product groups by revenue and shows their cumulative contribution. **Example:** If the top two product groups generate €800 of €1,000, their cumulative revenue share is 80%. | `products.revenue_concentration` | Conversion-filtered revenue plus a configured product grouping | Exact | Pareto |
| Unique products represented | Estimates how many different product groups are represented in the selected conversion population. **Example:** 50 conversion opportunities spanning three `ProductGroup` values represent three product groups, not 50 products owned. | `products.unique_products` | CPC state over `ProductGroup` | Approximate | KPI card |

Mix and concentration describe the selected reporting population. Filtering
out a product changes the displayed total and cumulative shares. They do not
claim causal attribution from a particular action to a purchase, and product
coverage is distinct from the customer-holdings lifecycle recipe.

### Value Distributions and Response Time

Understand typical action scores, variation and unusually high values. For
response time, lower means faster recorded outcomes. Higher `Propensity` means
a higher predicted response probability; higher `Priority` means a higher
arbitration score, which does not by itself establish better model quality.

| Business KPI | Explanation | Recipe ID | Required capability | Accuracy | Default report |
|---|---|---|---|---|---|
| Median propensity, priority or response time | The middle value for the selected CDH field, less influenced by a few extremes than an average. **Example:** Median `Propensity` of 0.03 means roughly half of recorded action scores predict a response probability of 3% or less. | `distribution.median` | Unconditioned t-digest or KLL state | Approximate | KPI card |
| 95th percentile (P95) value | A high-value threshold that about 95% of observations do not exceed. **Example:** P95 `Priority` of 1.8 means roughly 5% of recorded action priorities exceed 1.8; this helps identify unusually high arbitration scores and is not a response probability. | `distribution.p95` | Unconditioned t-digest or KLL state | Approximate | KPI card |
| Propensity, priority or response-time distribution | Shows the middle value, the middle half of observations and the tails for a CDH score or duration. **Example:** Web and Mobile can both have median `Propensity` of 3%, while their middle halves span 2–4% and 1–8%, revealing a wider spread of action scores on Mobile. | `distribution.boxplot` | Unconditioned t-digest or KLL state | Approximate | Boxplot |
| Decision-to-outcome response time (P95) | The duration within which about 95% of recorded action outcomes arrive. **Example:** `ResponseTime`: P95 of 1,800 means roughly 95% of included outcomes were recorded within 30 minutes of the decision and 5% took longer. | `engagement.decision_to_outcome_latency_p95` | Unconditioned duration t-digest or KLL state | Approximate | KPI card with previous-period comparison |


### Prediction Quality

Check whether model scores put likely responders ahead of non-responders and
whether predicted response levels match observed results.

| Business KPI | Explanation | Recipe ID | Required capability | Accuracy | Default report |
|---|---|---|---|---|---|
| Propensity ranking quality (ROC AUC) | Measures how well `Propensity` ranks clicked actions above non-clicked actions; 0.5 is random ranking and 1.0 is perfect separation. **Example:** AUC near 0.8 means a randomly paired clicked action ranks above a non-clicked action about 80% of the time, counting ties as half. | `model_quality.roc_auc` | Matched positive/negative t-digests for the same score | Approximate | KPI card |
| Propensity calibration ratio | Compares observed CTR with average `Propensity`: 1 means they match, below 1 means overprediction, above 1 means underprediction. **Example:** In one propensity band and model-control arm, 40 clicks from 1,000 classified outcomes give 4% CTR; against 5% mean propensity, the ratio is 0.8. | `model_quality.score_calibration` | Positive count, score sum, and observation count for the same scored population | Exact | KPI card |

Read calibration within propensity bands as well as overall: an overall
ratio of 1 can hide overprediction in one band and underprediction in another.

### Adaptive Learning and Exploration

These three **draft diagnostics** describe how adaptive decisioning adjusts
model scores while learning. They support comparisons within a consistently
explored population; they do not establish that a policy caused better
customer outcomes.

| Business KPI | Explanation | Recipe ID | Required capability | Accuracy | Default report |
|---|---|---|---|---|---|
| Material upward exploration rate | The share of scored decisions where `FinalPropensity` exceeds positive raw `Propensity` by more than the chosen relative threshold. **Example:** At the default 10% threshold, an action moving from 0.02 to 0.023 qualifies; 150 qualifying decisions out of 1,000 Test-arm decisions give 15%. | `decisioning.material_upward_exploration_rate` | Purpose-built relatively filtered upward-revision and total counts on a numeric-distribution processor | Exact; draft diagnostic | KPI card |
| Implied evidence index | A rough learning-maturity indicator based on how far `FinalPropensity` moves from `Propensity`; larger values suggest narrower relative adjustments. **Example:** For the same action and treatment in the Test arm, an index rising from 9 to 19 suggests narrowing adjustments, not 19 clicks or conversions. | `model_quality.implied_evidence_index` | Pooled means of `p × (1 − p)` and squared score revisions | Statistical; draft diagnostic | KPI card |
| Relative exploration variance | Compares squared `FinalPropensity` adjustments with the variation implied by raw `Propensity`; smaller ratios indicate narrower relative adjustments. **Example:** For a new action’s treatment in the Test arm, a ratio falling from 0.20 to 0.05 as the treatment ages suggests exploration is narrowing; 0.05 is not a 5% uncertainty probability. | `model_quality.relative_exploration_variance` | Pooled means of squared score revisions and `p × (1 − p)` | Statistical; draft diagnostic | KPI card |

Upward exploration is an adaptive-decisioning policy KPI, not a
predictive-model-quality metric. It uses a dedicated numeric-distribution
processor so outcome filtering cannot change the population. **Material
upward exploration rate** counts a decision in the numerator only when the
raw score is positive and
`FinalPropensity - Propensity > Propensity × threshold`. The threshold defaults
to 10% and is editable during installation; equality at the boundary and
zero/negative raw scores are excluded from the numerator, while the
denominator remains all scored decisions in the configured population.

For the other two diagnostics, `p` is the raw response probability and the
score revision is the final score minus `p`. The implied evidence index is
`mean(p × (1 − p)) / mean(revision²) − 1`; it can be negative and is not a
response count or fitted posterior parameter. Relative exploration variance
is `mean(revision²) / mean(p × (1 − p))`, an unbounded ratio rather than an
uncertainty probability. Other score adjustments can affect both diagnostics.
Randomised control arms must be excluded from all three.

### Contact Policy and Repeated Impressions

Assess how the response to an action changes as the same customer sees it
again, to decide after how many impressions it should stop being shown. The
**engagement rate** is positive impressions divided by positive plus negative
impressions, as in the engagement recipes. An **impression** is any record
with one of the processor's configured positive or negative outcomes, at any
rank; it is not proof that a customer viewed it.

`ExposureBucket` counts the impressions of the same `CustomerID + ActionID`
inside the processor's `scope_by` fields (for example `Channel` and
`Placement`) within a trailing 168-hour window, including the current one; the
final bucket collects every later impression. `ScopeRank` is the action's rank
among the decision's shown actions inside the same scope, so a banner shown at
arbitration rank 6 in a one-slot placement is rank 1 there. `PriorPositive`
marks impressions whose customer already responded to the action inside the
window. A `Clicked` record takes precedence over an `Impression` or `Pending`
record for the same contact, so each contact counts once.

| Business KPI | Explanation | Recipe ID | Required capability | Accuracy | Default report |
|---|---|---|---|---|---|
| Engagement rate by number of impressions | The share of impressions with a positive outcome after the 1st, 2nd, 3rd, ... impression of the same action to the same customer. **Example:** In the Web Hero placement, 1,000 fifth impressions to customers who have not clicked yet produce 12 clicks: a 1.2% engagement rate, against 1.6% on first impressions, so the action has lost a quarter of its response by the fifth impression. | `contact_policy.engagement_rate_by_impressions` | Exact `Positives` and `Negatives` states | Approximate fixed-window interpretation | Line |
| Repeat impression share | The fraction of classified impressions at frequency 2 or higher. **Example:** If 800 of 1,000 impressions are in repeat buckets, the share is 80%. | `contact_policy.repeat_impression_share` | `Positives`, `Negatives`, and a terminal bucket above 1 | Approximate fixed-window interpretation | KPI card |
| Historical frequency-cap cost | For a chosen cap `k`, the fraction of historical impressions above `k`; the query also returns `PositiveShareAboveThreshold`, the fraction of positive outcomes above `k`. **Example:** If 200 of 1,000 impressions and 5 of 50 positive outcomes occurred above cap 3, the two shares are 20% and 10%. | `contact_policy.historical_cap_cost` | `Positives`, `Negatives`, and `k < max_frequency` | Approximate fixed-window interpretation | Table |

The **Contact policy** recipes target only the `frequency_response` processor
and require its exact `Positives` and `Negatives` states; they never propose a
generic count as a substitute. The engagement-rate line uses `ExposureBucket`,
whose value is an upstream fixed-window approximation; changing a report date
filter does not recompute customer contact history.

The two share recipes use `frequency_threshold_share`. For each requested
report group, the query first merges stored counts by frequency bucket, then
divides the counts above the threshold by all counts in the group. A zero
denominator yields zero. The repeat-share threshold is 1. The cap-cost recipe
accepts a whole-number `cap` parameter (default 3), which must be below the
processor's terminal `max_frequency` bucket. Its primary output is the
historical impression share; `PositiveShareAboveThreshold` is a companion
output with positive outcomes as both numerator and denominator. Queries may
group or filter by other stored dimensions, but cannot group or filter by the
frequency column because that would remove buckets from the denominator.
Both values describe observed data, not the outcomes a new cap would cause.

Customers who already responded are shown the same action again more often,
so an all-customer curve can rise with repeated impressions even when every
customer responds less each time. Split or filter by `PriorPositive` before
reading a decline as fatigue, and keep `Channel` and `Placement` apart, since
their rates can differ tenfold. The curve is observational, not a causal
estimate of wearout.

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
  and report packs are planned future work, not current behavior.

## Aggregate and Backfill Rules

Installing a recipe that uses an already configured state does not change the
processor. When a selected field/algorithm pair is absent, the installer adds a
deterministic state definition to `processors.yaml` (or the AI draft) and binds
the metric to that state. Default parameters are CPC `lg_k=11`, HLL/Theta
`lg_k=12`, t-digest `k=500`, KLL `k=200`, and Top-K
`lg_max_map_size=10`; they remain inspectable in the technical binding.

A parameter that changes aggregate-state semantics is part of that state
definition. The resolver reuses a state only when its normalized definition,
including the resolved predicate value, matches exactly. Choosing a different
material-upward threshold therefore proposes a distinct filtered count state,
changes the processor computation hash, and requires materialization rather
than silently reinterpreting existing counts.

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
