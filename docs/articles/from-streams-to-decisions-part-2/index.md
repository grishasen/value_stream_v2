# From Streams to Decisions, Part 2: Build the Metric Before the Dashboard

*How business KPIs become reusable recipes, mergeable aggregate states, and exact or approximate calculations—and how to know what a reported number actually means.*

In [Part 1](../from-streams-to-decisions-part-1/index.md), we followed events into compact aggregate state. Counts, sums, and sketches preserved the information needed for reporting, allowing source rows to leave the processing path after each chunk.

That answers a storage question. It leaves a business question open: **what information should we preserve?**

“Show conversion rate” sounds like a complete requirement. It is not. Conversion per impression, per interaction, per customer, and per eligible conversion opportunity are different measurements. They can all produce plausible percentages. A dashboard cannot resolve that ambiguity by choosing a better chart.

This article works through the missing translation: from a business question to a KPI definition, from that definition to a reusable recipe, and from the recipe to the states and calculations that produce an answer. The examples use [Value Stream](https://github.com/grishasen/value_stream_v2), but the reasoning applies to any system that serves metrics from aggregates.

All numerical examples below are illustrative.

## A KPI begins with a decision and a population

Suppose a team wants to compare customer engagement across channels. The intended decision might be whether a channel needs a different action strategy. Before writing a formula, the team needs to agree on a measurement contract.

| Question to settle | One concrete definition |
|---|---|
| What is the unit being counted? | One classified outcome per interaction, action, and rank, after configured deduplication |
| What counts as success? | `Clicked` |
| What counts as an eligible non-success? | `Impression` or `Pending` |
| Which records are included? | Records accepted by the source transforms and processor filter, with one of those outcomes |
| Which timestamp assigns the reporting period? | `OutcomeTime`, interpreted using the configured calendar and time zone |
| Which breakdowns must remain possible? | Day, channel, and customer segment |
| What does an empty denominator mean? | No eligible observations; the formula's numeric fallback must be understood alongside the sample count |

Under this definition, engagement rate is:

```text
positive outcomes / (positive outcomes + negative outcomes)
```

With 80 clicks and 920 impression/pending outcomes, the result is 8%. The denominator is 1,000 classified outcomes. It is not the number of unique customers, and it is not simply the number of impression rows in the original export.

A conversion processor could instead classify `Conversion` and `NoConversion`. Its rate would have the same algebra but a different business meaning. Forty conversions and 160 non-conversions give 20%; dividing the same 40 conversions by 1,000 impressions gives 4%. Neither denominator can be substituted silently.

This is the first principle of metric design:

> The formula describes the arithmetic. The population, identity rules, and time definition describe what the arithmetic means.

## Five concepts connect the question to the answer

Value Stream separates business intent, reusable definitions, ingestion, and querying.

| Concept | Responsibility | Engagement example |
|---|---|---|
| **Business KPI** | Defines the measurement that informs a decision | Share of eligible action outcomes that become clicks |
| **KPI recipe** | Packages a reusable definition, required inputs, method, and caveats | `engagement.engagement_rate` requires positive and negative outcome counts |
| **Processor** | Applies the ingestion rules and builds aggregate state | A `binary_outcome` processor classifies, deduplicates, and groups outcomes |
| **State** | Preserves the information needed for later merging and calculation | `Positives` and `Negatives`, stored per configured group and time period |
| **Metric** | Binds a calculation to a particular processor's states | `CTR`, a `formula` metric that divides merged positives by merged eligible outcomes |

The authoring direction is **business question → recipe → concrete processor/state bindings → metric**. Execution then follows **events → processor → persisted states → merge → metric calculation → report**.

The distinction between state and metric is especially useful. A state survives ingestion. A metric is derived when a query asks for it. The same count states can support outcome volume, engagement rate, experiment sample size, and statistical comparisons. A single distribution sketch can support a median, P95, and a boxplot.

The report chooses how to present a metric. Its chart type does not define the denominator or change the stored population.

## What a KPI recipe actually contains

A KPI recipe is a versioned authoring artifact: a reusable business definition with a validated metric template. It includes:

- A business title, questions it answers, and guidance on interpretation.
- Compatible processor kinds and required input roles, such as positive outcomes, conversion revenue, an entity field, or a numeric distribution.
- A calculation and an accuracy classification: exact, approximate, or statistical.
- Algorithm choices, caveats, and any bounded parameters the analyst may set.
- Suggested display units and a report tile.

For example, `audience.unique_entities` asks for an entity field and a compatible distinct-count sketch. Binding it to `CustomerID` creates a unique-customer metric; binding it to `ActionID` creates a unique-action metric. Reusing the recipe preserves the calculation pattern while making the chosen business entity explicit.

Installation resolves those roles against a processor's declared capabilities. If several fields or populations could satisfy a role, the analyst chooses. If a supported state is missing, the authoring flow can propose that state and identify the required source run. Some missing capabilities cannot be proposed automatically and remain blocked until the processor is configured appropriately.

The result is ordinary catalog configuration: processor states when needed, a metric, and optionally a report tile. The metric records the recipe ID, version, and resolved parameters. After installation, the workspace YAML defines behavior; changing a packaged recipe does not silently rewrite installed metrics. [The recipe reference](../../reference/kpi-recipes.md) describes the complete contract and installation flow.

This matters for ownership. A business definition should be reusable without becoming an invisible dependency that changes yesterday's dashboard when a library is updated.

It also sets a limit on automation. Compatibility checks can confirm that two inputs are counts or that two score digests describe the same score field. They cannot establish that the business chose the right eligible population. That definition still needs review.

## A worked example: building engagement rate

### 1. Preserve the inputs to the calculation

The following `processors.yaml` example assumes an existing source named `ih`. After its transforms, that source exposes the timestamp, identifiers, outcome, channel, and customer segment shown here.

```yaml
catalog_version: 2
processors:
  - id: engagement
    source: ih
    kind: binary_outcome
    time:
      property: OutcomeTime
      grain: daily
      calendar:
        timezone: UTC
    group_by: [Channel, CustomerSegment]
    dedup_keys: [InteractionID, ActionID, Rank]
    outcome:
      column: Outcome
      positive_values: [Clicked]
      negative_values: [Impression, Pending]
    states:
      Count: {type: count}
      Positives: {type: count, outcome: positive}
      Negatives: {type: count, outcome: negative}
      UniqueCustomers_cpc:
        type: cpc
        source_column: CustomerID
        lg_k: 11
```

For each chunk, the processor applies its filter, accepts the configured outcomes, and marks positive rows. When deduplication is configured, it prefers a positive outcome within each key and keeps one row. It then builds states by reporting day, channel, and customer segment.

An impression and a click for the same key within that chunk therefore contribute one positive outcome. Deduplication is local to the processor's chunk input: splitting the same business interaction across independent chunks does not create a global identity registry. Source preparation and chunk boundaries must support the intended counting rule. Idempotent ingestion prevents a completed chunk from being published twice; that is a separate concern from duplicates already present in the source.

The output contains counts and serialized sketch state, together with provenance and a computation hash. It contains no stored CTR percentage. The CPC state is optional for CTR; it is included here to support a second KPI from the same population.

### 2. Bind the recipe to an executable metric

The engagement recipe binds its positive and negative roles to `Positives` and `Negatives`. An installed metric, with its display label set to “Engagement rate,” can be represented as follows in `metrics.yaml`:

```yaml
catalog_version: 2
metrics:
  CTR:
    processor: engagement
    kind: formula
    description: Clicked outcomes divided by classified outcomes.
    expression:
      op: safe_div
      num: {col: Positives}
      den:
        op: add
        args:
          - {col: Positives}
          - {col: Negatives}
    display:
      label: Engagement rate
      unit: percent
      value_format: percent
      direction: higher_is_better
    recipe:
      id: engagement.engagement_rate
      version: 1

  UniqueCustomers:
    processor: engagement
    kind: approx_distinct_count
    state: UniqueCustomers_cpc
    recipe:
      id: audience.unique_entities
      version: 1
```

The nested expression is a small, validated expression tree: read two columns, add them, and divide. The [expression language](../../reference/expression-dsl.md) gives formulas a structured representation instead of hiding the calculation in a dashboard callback.

`safe_div` has a specific behavior in Value Stream: a zero denominator returns `0.0`. That avoids a division error, but a displayed zero alone cannot distinguish “no successes” from “no eligible observations.” Include the sample count when that distinction affects the decision. Null inputs and a query that returns no rows are separate cases; the zero-denominator rule does not turn every missing result into an observation.

The calculation returns a fraction such as `0.08`. Percent formatting displays it as `8%`; it does not change the formula or multiply the persisted counts.

### 3. Merge first, calculate second

Suppose two daily partials for the same channel and segment contain:

| Partial | Positives | Negatives | Eligible outcomes | Local rate |
|---|---:|---:|---:|---:|
| A | 80 | 920 | 1,000 | 8% |
| B | 20 | 80 | 100 | 20% |
| Combined | 100 | 1,000 | 1,100 | **9.09%** |

The combined result is `100 / 1,100`. Averaging the two percentages would give 14%, incorrectly giving the small partial the same weight as the large one.

For a monthly query, the query layer selects compatible aggregate partials, applies filters on retained dimensions, and merges them into the requested groups. It then evaluates the formula. A metric with declared formula dependencies derives those dependencies from the merged state as well.

This ordering is the core calculation rule:

> Merge the evidence at the requested scope, then calculate the answer at that scope.

It applies beyond rates. Monthly unique customers require a union of daily sketches followed by one estimate. Monthly P95 requires merging daily distribution sketches followed by one quantile query. Adding daily distinct estimates or averaging daily percentiles answers neither question correctly.

## Exact metrics need the right sufficient statistics

“Exact” means the aggregation introduces no deliberate sketch approximation. It still depends on input quality, counting rules, and the numeric representation; floating-point arithmetic can introduce rounding differences.

| Measurement | State to preserve | Calculation after merging |
|---|---|---|
| Outcome volume | Count | Add counts |
| Recorded revenue | Sum over the eligible revenue rows | Add sums |
| Engagement rate | Positive and negative counts | Divide total positives by total eligible outcomes |
| Mean response time | Valid-value count and sum, or count and mean | Total sum / total valid-value count |
| Response-time variance | Count, mean, and sample variance | Combine variation within groups and between group means |
| Lowest or highest value | Minimum or maximum | Minimum of minima or maximum of maxima |

A mean of means fails for the same reason that a mean of rates fails. If one group has 1,000 observations and another has 10, their means need weights of 1,000 and 10. Missing response times should not count toward the weight of a response-time mean; the count and value states must describe the same observations.

Variance needs one more piece of information. Imagine two groups with no internal variation, one containing only the value 10 and the other only 100. Both local variances are zero. The combined population clearly has variation. Value Stream's pooled variance calculation preserves the counts and means so it can include that difference between groups, using the configured sample-variance convention. [The algorithm reference](../../reference/algorithms.md#23-pooled-variance-welford-merge) gives the equations.

Money requires a similarly explicit population. The `finance.revenue_per_conversion` recipe divides conversion-filtered revenue by conversion count. If revenue is repeated on impression, click, and conversion records, an unfiltered sum counts the wrong business amount even though every addition is correct. Currency and missing-value handling must also be consistent.

## Approximation has several different meanings

An “approximate” badge is useful only if the reader understands what is being estimated. A distinct-count error, a percentile rank error, and a model diagnostic are different things.

### Distinct populations: CPC, HLL, and Theta

For unique customers, the processor feeds normalized, non-null identifiers into a sketch. Repeated appearances update a compact representation of the same population. Queries union compatible sketches and estimate the size of the union.

Value Stream uses CPC by default for newly generated distinct-count states, typically with `lg_k: 11`. HLL remains available, typically with `lg_k: 12`. The parameter is logarithmic: `lg_k: 12` corresponds to a nominal capacity parameter of 4,096. Increasing precision costs more state per aggregate group.

CPC and HLL support distinct counting and union. Theta also supports intersections and differences, making it appropriate when the business question includes “in both populations” or “in A but not B.” These capabilities must be chosen before ingestion; an HLL or CPC estimate cannot later be converted into a Theta set. Apache documents these tradeoffs in its [CPC overview](https://datasketches.apache.org/docs/CPC/CpcSketches.html).

If Monday has about 10,000 customers and Tuesday about 12,000, the two-day audience is not necessarily 22,000. Many customers may appear on both days. The stored sketches preserve enough information to estimate that overlap through union without retaining an enumerable customer list.

Precision should be interpreted using the selected algorithm's bounds, rather than a universal “±1%” label. A cardinality interval concerns uncertainty in estimating the recorded population. It does not account for missing customers in the source export or prove that the population represents all customers.

There is also a distinction between library capability and displayed output: the current `approx_distinct_count` metric returns a point estimate. The underlying sketch APIs provide bounds, but this metric does not automatically add lower- and upper-bound columns to the report. A recipe's accuracy description is not itself a displayed confidence interval.

### Percentiles: rank error is not value error

For P95 response time, the retained state is a distribution of durations. A `quantile` metric asks that state for the value at rank `0.95`; a `distribution` metric exposes a broader set of distribution outputs for reporting.

Value Stream supports t-digest and KLL. Typical configured capacities are `k: 500` for t-digest and `k: 200` for KLL, but the parameters belong to different algorithms and are not directly comparable accuracy settings.

**KLL summarizes ordered observations with a probabilistic rank-error contract.** Apache DataSketches reports approximately 1.33 percentage points of normalized rank error for quantile/rank queries at `k=200`, and about 1.65 points for histogram-bin mass queries, which involve two boundaries. These are library error estimates at its documented 99% confidence level. They are not percentages of the returned duration. See the [KLL accuracy table](https://datasketches.apache.org/docs/KLL/KLLAccuracyAndSize.html) and [Python API's error contract](https://apache.github.io/datasketches-python/main/quantiles/kll.html).

For intuition, a 1.33-point rank tolerance around P95 spans approximately ranks 93.67% to 96.33%. If the duration distribution rises steeply there, that interval can cover many seconds. A small rank error therefore need not imply a small error in the displayed response time.

**t-digest compresses numeric observations into weighted clusters and interpolates between them.** It can perform very well near the tails, but its accuracy depends on the data and processing. Apache explicitly describes it as empirical, without a mathematical error guarantee. A fixed `k` should not be advertised as a universal maximum P95 or P99 error. [Apache's t-digest overview](https://datasketches.apache.org/docs/tdigest/tdigest.html) explains the distinction.

The practical choice depends on the question: retain an explicit rank-error contract when that is needed, or measure t-digest accuracy on representative data when its empirical behavior fits the workload. Neither approach makes an average of daily P95 values a monthly P95.

### Frequent items and reconstructed model scores

The `category.top_items` recipe uses a frequent-items state to estimate common actions or treatments. Its `topk_items` metric returns items with estimated frequencies and bounds. When two items' bounds overlap, their apparent ordering need not be stable. The configured error mode also affects which candidates are returned; a requested limit of ten is a display limit, not proof of an exact top-ten ranking. [Apache's frequent-items reference](https://datasketches.apache.org/docs/Frequency/FrequentItemsOverview.html) describes these guarantees.

Model-quality KPIs introduce another layer of approximation. The `model_quality.roc_auc` recipe binds two t-digests for the same score field: one for positive outcomes and one for negative outcomes. The calculation estimates how many observations lie above a sequence of score thresholds, constructs a ROC curve, and integrates it. Error can come from both the compressed distributions and the finite threshold grid. Calibration similarly reconstructs score-bin behavior from digests. These are approximate reconstructions, not exact calculations over every original prediction/outcome pair. See [curves from digests](../../reference/algorithms.md#4-curves-from-t-digests-roc-ap-calibration).

### Statistical inference is a separate layer

An experiment may use entirely exact counts and still produce uncertain conclusions. Suppose Test has 400 positive outcomes from 10,000 opportunities and Control has 300 from 10,000:

```text
Test rate                = 4%
Control rate             = 3%
Absolute rate difference = 1 percentage point
Relative lift            = (4% - 3%) / 3% ≈ 33.3%
```

The observed rates and differences follow directly from the counts. The `experiments.test_control_comparison` recipe adds confidence intervals; `experiments.z_test` evaluates evidence against equal response rates. That inference depends on assumptions about sample size and independent observations. Repeated responses from the same customer may violate independence, and interpreting a difference causally requires a suitable experimental design. Exact aggregate counts do not supply those assumptions. [The experiment recipes](../../reference/kpi-recipes.md#business-experiments-test-and-control) document the interpretation.

Finally, some business approximations are **proxies**, even when their arithmetic is exact. Impression telemetry need not establish actual viewability. A funnel ratio of stage counts does not prove that the same people passed through those stages in order. Increasing sketch precision cannot fix either mismatch. Recipe caveats need to describe the measurement's meaning as well as its algorithm.

## Translating business KPIs into concrete calculations

The mapping becomes easier to review when every step is visible:

| Business question | KPI recipe | Persisted inputs | Metric kind and calculation |
|---|---|---|---|
| Are eligible interactions producing positive responses? | `engagement.engagement_rate` | Positive and negative outcome counts | `formula`: positives / (positives + negatives) |
| How many different customers did we observe? | `audience.unique_entities` bound to `CustomerID` | CPC, HLL, or Theta state over the eligible population | `approx_distinct_count`: estimate after union |
| How long do the slowest recorded outcomes take? | `engagement.decision_to_outcome_latency_p95` | Unconditioned t-digest or KLL over `ResponseTime` | `quantile`: query the merged state at 0.95 |
| How much revenue does a recorded conversion generate? | `finance.revenue_per_conversion` | Conversion-filtered revenue sum and positive conversion count | `formula`: total conversion revenue / total conversions |
| Does Test outperform Control? | `experiments.test_control_comparison` | Positive and negative counts with the experiment arm retained | `variant_compare`: rates, differences, lift, and intervals |
| What share of stage volume reaches completion? | `funnel.conversion_rate` | Selected start and completion count states | `formula`: completion count / start count; no inferred journey linkage |

Several KPIs reuse the same states. Others require new information. That is the dividing line between adding a calculation and changing ingestion.

### A parameter can change the state that must be collected

Consider `decisioning.material_upward_exploration_rate`, a draft recipe in the library. It measures the share of scored decisions whose upward score adjustment exceeds a chosen fraction of the original score.

Assuming the source derives `ExplorationDelta = FinalPropensity - Propensity`, its default numerator counts rows satisfying:

```text
Propensity > 0
and ExplorationDelta > 0.10 × Propensity
```

The denominator is the configured total decision count, `Explore_Count`. Zero or negative raw scores are excluded from the numerator, not automatically from that denominator. The eligible source population must also exclude randomized control or holdout arms, as the recipe's caveat requires.

The analyst sees a 10% threshold. Installation records the decimal parameter `0.10` and materializes the predicate in a filtered count state. The final metric simply divides that count by the observation count.

For an original propensity of `0.04`, an adjustment greater than `0.004` qualifies. Changing the threshold to 20% changes the qualifying adjustment to greater than `0.008`.

An existing count of rows above 10% cannot reveal how many exceeded 20%. Nor can a digest of the original propensity alone answer a condition involving both propensity and adjustment. A new parameter-resolved state needs source replay if it was not already collected. The exact count measures the chosen score-change rule; interpreting it as exploration also depends on how those scores were produced.

This is why recipes may propose aggregate states as well as formulas. A business parameter can change the evidence required, not just the number shown on a card.

## Dimensions, time, and precision are part of the contract

The engagement example preserves channel and customer segment. A later query can filter those dimensions or roll them up. It cannot split historical counts by device type if device type was never retained.

Likewise, a daily base grain supports suitable coarser calendar queries, but it cannot reconstruct an hourly pattern. An outcome-time metric and a decision-time metric assign delayed responses differently even if their formulas match. More dimensions and finer grains preserve more options while increasing the number of aggregate groups—and every sketch has a cost in each group.

There are two distinct places to apply a condition:

- **During ingestion:** source filters, processor filters, and state-specific predicates decide which observations contribute to state.
- **During querying:** filters over retained aggregate dimensions select already-built state for merging.

A report filter cannot retroactively change a deduplication rule, a discarded field, or the threshold inside a stored count. A customer sketch also does not make arbitrary customer-level filtering possible just because it was built from customer IDs.

Changing only a formula over available states can reuse those aggregates. Adding a missing state, changing outcome classification, or switching a sketch's algorithm or precision changes the processor's computation contract and requires ingestion under that contract. Historical results require replay from an available authoritative source. The new metric cannot recover information from raw rows that the aggregate store never retained.

Computation hashes keep incompatible states from being combined. Query provenance connects a result to the catalog, selected aggregate grain, and contributing runs and chunks. The recipe version explains which reusable definition was installed; the computation hash explains which ingestion definition produced its inputs. Both are useful, and neither replaces the other.

## How to check that a metric means what it claims

Validation needs to cover both business semantics and merge behavior.

1. **Use a small fixture with known answers.** Include positives, negatives, excluded outcomes, duplicates, null values, and an empty eligible population. Work out the expected numerator and denominator before running the calculation.
2. **Check rollups with unequal group sizes.** The 8% and 20% example should return 9.09% when combined, not 14%. Test the relevant day-to-month and segment-to-total paths.
3. **Check chunk boundaries.** Splitting data into chunks should preserve the intended result under the processor's identity contract. Include a duplicate crossing a boundary when the source can produce one; do not assume chunk-level deduplication solves it.
4. **Measure approximation in its own units.** Compare distinct estimates with known cardinalities and their bounds; check quantile rank error against sorted fixture data; characterize t-digest empirically. Do not demand identical sketch bytes across arbitrary build orders as a substitute for accuracy testing.
5. **Review what the reader sees.** Verify units, percent formatting, sample support, empty-result behavior, and caveats. An accurate calculation can still be misleading when labelled “customers” instead of “outcomes.”

These checks turn a formula into a measurement the team can explain and maintain.

## From metrics to reports

A trustworthy KPI has a traceable path: a decision motivates a business definition; a recipe packages that definition; processor states preserve the required evidence; and a metric merges and calculates at the requested scope.

Exact state, probabilistic sketches, and statistical inference each have a place. The useful question is not simply whether a result is approximate. It is **what was preserved, what was estimated, and whether that is sufficient for the decision**.

The next article will follow these metrics into reports: filters, time comparisons, KPI cards, tables, and charts that share the same calculation contract.

The guiding question from Part 1 now has a practical companion: what is the smallest durable state that preserves the decision—and can we explain every step from that state to the number on the screen?
