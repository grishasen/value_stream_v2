# Value Stream — Processor Specifications

This document specifies every built-in Processor in enough detail that an implementer can write the code from scratch without referring to the legacy application. Each spec covers:

- the YAML configuration shape,
- the input expectations on the Source schema,
- the output (state) schema at each grain,
- the chunk-aggregation algorithm (in pseudocode),
- the merge / compact algorithms,
- the derived metrics that bind to it,
- edge cases and invariants.

Companion docs:

- reference/algorithms.md — formulas for sketches, statistical tests, RFM, ML metrics.
- reference/expression-dsl.md — grammar for `filter` / `derive_column` / `expression` AST.
- reference/readers-and-formats.md — how Sources turn files into LazyFrames.

---

## 1. The Processor interface

Every processor implements the same Python protocol (sketched here in pseudocode for clarity). All references to "DataFrame" are Polars.

```python
class Processor(Protocol):

    # Static identity & configuration
    id: str
    kind: str              # one of binary_outcome | numeric_distribution | ...
    source_id: str
    group_by: list[str]    # transformed source columns preserved for reporting
    time.grain: str        # one physical base grain
    states: dict[str, StateSpec]
    config_hash: str

    # 1. Output schema announcement
    def schema(self, grain: str) -> pyarrow.Schema:
        """Returns the Parquet schema for one grain.
        Includes group-by columns, time-grain column,
        all state columns, and the 5 provenance columns."""

    # 2. Per-chunk aggregation
    def chunk_aggregate(
        self,
        lazy_frame: pl.LazyFrame,        # output of the source's transforms
        chunk_ctx: ChunkCtx,             # chunk_id, run_id, period, config_hash
    ) -> pl.DataFrame:                   # at the FINEST grain
        """Reduces the chunk's rows to one row per (group_by tuple, time_grain).
        Output schema must match self.schema(finest_grain)."""

    # 3. State merging across rows
    def merge(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Group by (group_by columns + time_grain) and apply the per-state-type
        merge rule. Used by compaction and by the query layer to fold
        multiple chunks' partials into a single answer."""

    # 4. Base-grain projection/merge
    def compact(
        self,
        frame: pl.DataFrame,             # chunk aggregate
        target_grain: str,               # configured base grain
    ) -> pl.DataFrame:
        """Project or merge the chunk aggregate to the one stored base grain."""

    # 5. Optional: derived helpers
    def derive(self, frame: pl.DataFrame, params: dict) -> pl.DataFrame:
        """Some processors expose derived helpers (e.g. RFM segmentation);
        most leave this to the metric DSL."""
```

The default `merge` and `compact` implementations are generic — they iterate
over `self.states` and dispatch to the per-state-type merge rule. Ingestion
calls `compact` only for `time.grain`; query-time rollups reuse the same merge
contract for coarser calendar grains. Subclasses override the merge only when
it requires extra context (e.g. pooled variance needs the
`_n_minus1_variance` and `_n_mean_diff_sq` temporaries documented in
reference/algorithms.md §2.3).

When the target uses the processor's finest physical level, `compact` first
checks whether the prepared target keys are unique. Unique rows are projected
directly instead of re-merging singleton sketches; duplicate keys and every
coarser target continue through the normal state merge rules. The projected
result is stamped with the current chunk provenance and processor config hash
in the same way as a merged result.

The 5 provenance columns are added by the engine wrapper, not by the processor itself.

---

## 2. State types — quick reference

| State type | Storage dtype | Build from | Merge rule | Used by |
|---|---|---|---|---|
| `count` | `INT64` | `pl.len()` or `pl.sum(<bool 0/1>)` | `SUM` | binary_outcome, frequency_response, numeric_distribution, score_distribution, entity_lifecycle, funnel, snapshot |
| `value_sum` | `FLOAT64` | `pl.sum(col)` | `SUM` | binary_outcome, frequency_response, numeric_distribution, snapshot |
| `min` | matches data | `pl.min(col)` | `MIN` | numeric_distribution, entity_lifecycle |
| `max` | matches data | `pl.max(col)` | `MAX` | numeric_distribution, entity_lifecycle |
| `pooled_mean` | `FLOAT64` (explicit `source_column` or recipe plus `weight`) | `pl.mean(col)` or named recipe | `weighted_mean(value, weight)` | numeric_distribution, score_distribution |
| `pooled_variance` | `FLOAT64` (explicit `source_column`, `mean`, and `weight`) | `pl.var(col)` | Welford-merge | numeric_distribution |
| `tdigest` | `BLOB` | `datasketches.tdigest_double(k=500)` | deserialize/merge/serialize | numeric_distribution, score_distribution |
| `kll` | `BLOB` | `datasketches.kll_floats_sketch(k=200)` | KLL merge | optional, alternative to tdigest |
| `cpc` | `BLOB` | `datasketches.cpc_sketch(lg_k=11)` | union | default distinct count for binary_outcome, score_distribution, entity_lifecycle, entity_set, funnel, snapshot |
| `hll` | `BLOB` | `datasketches.hll_sketch(lg_k=12, tgt_type=HLL_8)` | union | explicitly selected distinct-count alternative |
| `theta` | `BLOB` | `datasketches.theta_sketch(lg_k=12)` | union/intersect/diff | entity_set (cohort) |
| `topk` | `BLOB` | `datasketches.frequent_strings_sketch(lg_max_map_size=10)` | merge | optional |

reference/algorithms.md gives the full algorithmic detail.

`count` and `value_sum` states may carry a `where` expression on processors
that accept those states. The predicate is applied to that state only, after
the processor-wide filter and any processor deduplication. For a distinct
count, excluded rows are removed before distinctness is evaluated; they do not
create a synthetic null member.

Configuration Builder exposes these kind-specific state settings through a
`Parameters YAML` mapping beside the state name, type, and source column. The
mapping is preserved in `processors.yaml`; `type` and `source_column` remain
dedicated fields. Empty mappings use the defaults shown above. The Builder
accepts `lg_k` only for CPC/HLL/theta states, `k` only for t-digest/KLL states,
and `lg_max_map_size` only for Top-K states.

Whenever a structured authoring path writes a processor, each unconditioned
t-digest or KLL state receives a persisted `distribution` metric if no
distribution binding exists.
The state remains the mergeable binary storage contract; the metric is the
public query/report binding. Explicit `quantile` metrics can coexist with it.
States marked `outcome: positive` or `outcome: negative` are excluded because
they are paired implementation inputs for curve and calibration metrics.

---

## 3. binary_outcome processor

### 3.1 Purpose

Counts of positive / negative / total rows per group-by tuple, optionally with `value_sum` columns and touchpoint attribution. Used for engagement (CTR), conversion (rate, revenue), and experiment (z/chi2/G).

### 3.2 YAML

```yaml
processors:
  - id: engagement                          # snake_case
    source: ih
    kind: binary_outcome
    group_by: [Day, Week, Month, Quarter, Year, Channel, PlacementType, Issue, Group, CustomerType]
    time:
      property: OutcomeTime
      grain: daily

    outcome:
      column: Outcome
      positive_values: [Clicked]
      negative_values: [Impression, Pending]

    dedup_keys: [InteractionID, ActionID, Rank]   # optional; default omitted

    variant_column: ModelControlGroup             # optional, enables variant_compare
    variant_role_map:                              # optional
      Test: Test
      Control: Control

    touchpoint:                                    # optional, conversion-style
      customer_column: CustomerID
      event_column: ConversionEventID
      output_state: Touchpoints

    value_aggs:                                    # optional, conversion-style
      - {column: Revenue, alias: Revenue, agg: sum}

    filter:                                        # optional, processor-level filter (AST)
      op: in
      column: ModelControlGroup
      values: [Test, Control]

    states:
      Count:               {type: count}
      Positives:           {type: count, outcome: positive}
      Negatives:           {type: count, outcome: negative}
      Revenue:             {type: value_sum, source_column: Revenue}     # for conversion
      Touchpoints:         {type: count}                                  # for conversion
      UniqueCustomers_cpc: {type: cpc, source_column: CustomerID, lg_k: 11}
      UniqueActions_cpc:   {type: cpc, source_column: ActionID,   lg_k: 11}
      TopCampaigns:        {type: topk, source_column: Campaign, lg_max_map_size: 10}
```

Three concrete bindings of this kind exist in the canonical workspace:

- `engagement` — variant_column = ModelControlGroup, no touchpoint, no value_aggs.
- `conversion` — no variant_column, with touchpoint and `Revenue` value_agg.
- `experiment` — `group_by` includes `ExperimentName` and `ExperimentGroup`, variant_column = ExperimentGroup, filter restricts to `ModelControlGroup ∈ [Test, Control]`.

### 3.3 Source schema requirements

The Source must, after its transforms, expose:

- `<outcome.column>` (string) — typically `Outcome`.
- All columns referenced in `group_by`.
- `<dedup_keys>` columns (default `InteractionID, ActionID, Rank`) — needed only if dedup is desired.
- `<variant_column>` (if set).
- `<touchpoint.customer_column>` and `<touchpoint.event_column>` (if touchpoint is set).
- `<value_aggs[*].column>` (if value_aggs is set).
- A calendar derivation: `Day` (date), `Month` (string YYYY-MM), `Year` (Int16), `Quarter` (string YYYY_Qn). The default `derive_calendar` transform produces these.

### 3.4 Output schema (daily grain)

```
+--------------------+--------+
| Day                | DATE   |  -- finest time grain
| <dim columns>      | string |
| <variant_column>   | string |  -- only if variant_column set
| <extra dim cols>   | string |
| Count              | INT64  |
| Positives          | INT64  |
| Negatives          | INT64  |
| <value_aggs aliases>|FLOAT64|  -- only if value_aggs set
| Touchpoints        | INT64  |  -- only if touchpoint set
| <configured CPC/HLL/Theta/Top-K sketch states> | BLOB   |
| pipeline_run_id    | UUID   |
| chunk_id           | STRING |
| period             | STRING |  -- "YYYY-MM"
| created_at         | TIMESTAMP |
| config_hash        | STRING |
+--------------------+--------+
```

`monthly` drops `Day`; `summary` drops `Day` and any other calendar dim (`Month`, `Year`, `Quarter`).

### 3.5 chunk_aggregate algorithm

```text
INPUT lazy_frame F (rows after source transforms)

# Step 1. Apply processor-level filter, if any.
if config.filter is set:
    F = F.filter(eval_ast(config.filter))

# Step 2. Restrict to known outcomes.
F = F.filter(F[outcome.column] in (positive_values + negative_values))

# Step 3. Compute the binary outcome.
F = F.with_columns(
    Outcome_Binary = (F[outcome.column] in positive_values).cast(Int8))

# Step 4. Deduplicate per (interaction key) keeping the most "positive" row.
if dedup_keys is set:
    F = F.filter(
        F.Outcome_Binary == max(F.Outcome_Binary).over(dedup_keys)
    )

# Step 5. Optional touchpoint attribution.
if touchpoint is set:
    T = (F.filter(Outcome_Binary == 1)
            .group_by([touchpoint.customer_column, touchpoint.event_column])
            .agg(pl.len().alias(touchpoint.output_state)))
    F = F.join(T, on=[touchpoint.customer_column, touchpoint.event_column], how="left")
    F = F.with_columns(F[touchpoint.output_state].fill_null(0))

# Step 6. Group-by and aggregate.
group_keys = group_by + (variant_column or []) + finest_time_grain_columns
agg_exprs = [
    pl.len().alias("Count"),
    pl.sum("Outcome_Binary").alias("Positives"),
    *[ getattr(pl.col(va.column), va.agg)().alias(va.alias) for va in value_aggs ],
    *[ pl.sum(touchpoint.output_state).alias(touchpoint.output_state) ] if touchpoint set,
    *[ build_sketch(pl.col(s.source_column), s).alias(name)
       for name, s in states.items() if s.type in ("cpc", "hll", "theta", "topk") ],
]
G = F.group_by(group_keys).agg(agg_exprs)

# Step 7. Compute Negatives.
G = G.with_columns(Negatives = G.Count - G.Positives)

OUTPUT G
```

### 3.6 merge algorithm

```text
INPUT frame F at some grain
group_keys = group_by + (variant_column or []) + time_grain_columns

# Apply per-state-type rule.
agg_exprs = []
for name, spec in states.items():
    if spec.type in (count, value_sum):
        agg_exprs.append(pl.sum(name).alias(name))
    elif spec.type in (cpc, hll, theta, topk):
        agg_exprs.append(merge_sketch(pl.col(name), spec).alias(name))
    elif spec.type in (min, max):
        agg_exprs.append(getattr(pl, spec.type)(name).alias(name))
    # ...

# Re-derive Negatives after sum (or carry it; both work).
M = F.group_by(group_keys).agg(agg_exprs)
M = M.with_columns(Negatives = M.Count - M.Positives)
OUTPUT M
```

### 3.7 compact algorithm

```text
INPUT finer_grain frame F, target_grain G

if target_grain == 'monthly':   drop_columns = ['Day']
if target_grain == 'summary':   drop_columns = ['Day', 'Month', 'Year', 'Quarter']

F2 = F.drop(drop_columns)
OUTPUT merge(F2)
```

### 3.8 Derived metrics (canonical bindings)

| Metric | Kind | Inputs | Output |
|---|---|---|---|
| `CTR` (engagement, conversion) | `formula` | `Positives, Negatives` | scalar per row |
| `ConversionRate` | `formula` (alias of `CTR`) | same | same |
| `StdErr` | `formula` | `CTR, Positives + Negatives` | √(p(1-p)/n) |
| `AvgTouchpoints` | `formula` | `Touchpoints, Positives` | mean per conversion |
| `Revenue` | (state column) | `Revenue` | passthrough |
| `Lift` | `variant_compare` | `Positives, Negatives` per variant | TestCTR, ControlCTR, AbsoluteRateDifference + CI, Lift + CI, Lift_Z_Score, Lift_P_Val, StdErr |
| `Proportion_Significance` | `proportion_test` | `variant_column`, `test_role`, `control_role`, `Positives`, `Negatives` | Count, Positives, Negatives, z_score, z_p_val |
| `Experiment_Significance` | `contingency_test` | `Positives, Negatives` per variant_column | chi2/G/z stats + odds ratio + CIs |
| `UniqueCustomers` | `approx_distinct_count` | `UniqueCustomers_cpc` | scalar per row |

See reference/algorithms.md §3 (variant_compare), §4 (contingency_test), §6 (CPC/HLL/Theta distinct).

### 3.9 Edge cases and invariants

- `Count = Positives + Negatives` always after dedup (rows with outcome ∉ positive ∪ negative are filtered out).
- Empty groups are written as zeros — the engine never silently drops a group-by tuple it has seen.
- Variant comparison requires *both* variant roles present; if a group has only `Test`, `Lift_*` columns are NULL.
- `variant_compare` and `proportion_test` select exactly the configured
  `test_role` and `control_role`; unrelated variants are excluded rather than
  being folded into the test population. Their `variant_column` must be
  persisted in processor `group_by` (or the processor's explicit
  `variant_column`) and is validated before ingestion.
- Sketch parameters must be identical for the same state across all chunks;
  changing parameter values or sketch type requires replay from source chunks.
- `dedup_keys` only matters within a chunk; cross-chunk dedup is impossible without raw rows. If exact cross-chunk dedup is needed, switch the relevant state to `theta` (set algebra at chunk boundaries).

---

## 4. numeric_distribution processor

### 4.1 Purpose

Per-group descriptive statistics for numeric properties: count, sum, mean, variance, min, max, plus a t-digest for arbitrary quantiles, ROC-style histograms, etc. Replaces the legacy `descriptive` family.

### 4.2 YAML

```yaml
processors:
  - id: descriptive
    source: ih
    kind: numeric_distribution
    group_by: [Day, Week, Month, Quarter, Year, Channel, CustomerType]
    time:
      property: OutcomeTime
      grain: daily
    properties: [ResponseTime]
    quantile_engine: tdigest
    states:
      ResponseTime_Count: {type: count}
      ResponseTime_Sum: {type: value_sum, source_column: ResponseTime}
      ResponseTime_Mean: {type: pooled_mean, source_column: ResponseTime, weight: ResponseTime_Count}
      ResponseTime_Var: {type: pooled_variance, source_column: ResponseTime, mean: ResponseTime_Mean, weight: ResponseTime_Count}
      ResponseTime_Min: {type: min, source_column: ResponseTime}
      ResponseTime_Max: {type: max, source_column: ResponseTime}
      ResponseTime_tdigest: {type: tdigest, source_column: ResponseTime, k: 500}
```

`properties` identifies the approved numeric inputs and `states` is the exact
persisted output contract. The engine does not generate, merge, or rename
states implicitly. In Configuration Builder, selecting a numeric property is
an authoring shortcut that appends the merge-safe `<property>_Count`,
`<property>_Mean`, `<property>_Var`, `<property>_Min`, `<property>_Max`, and
`<property>_<quantile_engine>` definitions to the editable state grid. Existing
or manually edited rows are preserved. Standard deviation is not persisted as
a separate state: applying the processor creates a formula metric that evaluates
`sqrt(<property>_Var)`. Sketch construction always uses the bulk implementation.

### 4.3 Source schema requirements

Each property in `properties` must exist in the Source's row schema. Numeric properties produce numeric states; string/categorical properties produce only `Count`. Group-by and time-grain columns must exist.

### 4.4 Output schema (daily grain, abbreviated)

```
| Day | <dim columns> |
| Outcome_Count       | INT64 |
| Propensity_Count    | INT64 |
| Propensity_Sum      | FLOAT64 |
| Propensity_Mean     | FLOAT64 |
| Propensity_Var      | FLOAT64 |
| Propensity_Min      | FLOAT64 |
| Propensity_Max      | FLOAT64 |
| Propensity_tdigest  | BLOB |
| ... (one set per numeric property) ...
| <provenance>
```

### 4.5 chunk_aggregate algorithm

```text
INPUT lazy_frame F

if config.filter:
    F = F.filter(eval_ast(config.filter))

properties = config.properties
schema = F.schema()
existing_props = [p for p in properties if p in schema]
numeric_props  = [p for p in existing_props if schema[p] is numeric]
group_keys = group_by + finest_time_grain_columns

agg_exprs = []
for p in existing_props:
    agg_exprs.append(pl.col(p).count().alias(f"{p}_Count"))
for p in numeric_props:
    agg_exprs += [
        pl.col(p).sum().alias(f"{p}_Sum"),
        pl.col(p).mean().alias(f"{p}_Mean"),
        pl.col(p).var().alias(f"{p}_Var"),
        pl.col(p).min().alias(f"{p}_Min"),
        pl.col(p).max().alias(f"{p}_Max"),
    ]

# t-digest state via map_groups (one struct field per numeric property,
# then unnest into top-level columns to avoid nested schema growth).
if config.quantile_engine == "tdigest":
    agg_exprs.append(map_groups_build_tdigests(numeric_props, k=500)
                       .alias("__tdigests"))

G = F.group_by(group_keys).agg(agg_exprs)
if "__tdigests" in G.columns:
    G = G.unnest("__tdigests")

OUTPUT G
```

`map_groups_build_tdigests(props, k)` builds one t-digest per numeric property by feeding the property's values into a `datasketches.tdigest_double(k)`. See reference/algorithms.md §2.4 for the exact procedure.

### 4.6 merge algorithm — pooled variance

The non-trivial part of merging is variance. `merge` follows the Welford pooled formula. For each numeric property `p`:

```text
let n_i  = group i's <p>_Count
let m_i  = group i's <p>_Mean
let v_i  = group i's <p>_Var

global_n    = sum(n_i)
global_mean = sum(n_i * m_i) / global_n        # weighted_mean
ssw         = sum((n_i - 1) * v_i)             # within-group SS
ssb         = sum(n_i * (m_i - global_mean)^2) # between-group SS
global_var  = (ssw + ssb) / (global_n - 1)     # pooled variance
```

The full implementation precomputes two helpers (`{p}_n_minus1_variance = (Count - 1) * Var`, `{p}_n_mean_diff_sq = Count * (Mean - GroupMean)^2`) per row before the group-by, then sums them, and finally divides by `global_n - 1`. This avoids a self-join and keeps the operation associative across multiple merge passes.

The merge rule for the t-digest column is "deserialize, merge, reserialize" via `datasketches`.

### 4.7 compact algorithm

Same shape as binary_outcome's: drop the appropriate calendar columns, then `merge`.

### 4.8 Derived metrics

| Metric | Kind | Inputs | Output |
|---|---|---|---|
| `Mean(p)`, `Var(p)`, `StdDev(p)` | (state passthrough or formula) | `<p>_Mean`, `<p>_Var` | scalar |
| `Distribution(p)` | `distribution` | `<p>_tdigest` or `<p>_kll` | quantile suite |
| `Median(p)`, `p25(p)`, `p75(p)`, `p90(p)`, `p95(p)`, `p99(p)` | `quantile` | digest state plus required `quantile` | scalar |
| `Skew(p)` | `formula` | `p25, p50, p75` | Bowley skew = `(p75 + p25 − 2·p50) / (p75 − p25)` |

`distribution` exposes the complete quantile suite to boxplots and related
charts. `quantile` always requires an explicit value between 0 and 1.
Configuration Builder automatically creates `StdDev(p)` as a formula metric
when it applies a numeric processor with a pooled-variance state.

### 4.9 Edge cases

- A property may exist in some chunks and not in others. The engine fills missing property states with sentinel values (`Count=0`, sketches empty) and never errors on a missing property.
- `Count <= 1` per group makes variance undefined; the engine emits NULL.
- Strings as properties must NOT be passed to numeric aggregations; the engine filters them automatically based on schema.

---

## 5. score_distribution processor

### 5.1 Purpose

ML model evaluation per group-by tuple: ranks the model's score by outcome,
stores per-outcome score t-digests, supports unconditioned t-digest/KLL states
and CPC/HLL/Theta/Top-K states over configured source fields, and computes
`personalization` and `novelty`. Replaces the legacy `model_ml_scores` family.

### 5.2 YAML

```yaml
processors:
  - id: model_ml_scores
    source: ih
    kind: score_distribution
    group_by: [Day, Week, Month, Quarter, Year, Channel, CustomerType]
    time:
      property: OutcomeTime
      grain: daily

    score_properties:
      - {column: Propensity, role: primary}
      - {column: FinalPropensity, role: calibrated}

    outcome:
      column: Outcome
      positive_values: [Clicked]
      negative_values: [Impression, Pending]

    dedup_keys: [InteractionID, ActionID, Rank]

    states:
      Count:                         {type: count}
      personalization:               {type: pooled_mean, weight: Count, recipe: personalization}
      novelty:                       {type: pooled_mean, weight: Count, recipe: novelty}
      Propensity_tdigest_positives:       {type: tdigest, source_column: Propensity,      score_property: Propensity,      outcome: positive, k: 500}
      Propensity_tdigest_negatives:       {type: tdigest, source_column: Propensity,      score_property: Propensity,      outcome: negative, k: 500}
      FinalPropensity_tdigest_positives:  {type: tdigest, source_column: FinalPropensity, score_property: FinalPropensity, outcome: positive, k: 500}
      FinalPropensity_tdigest_negatives:  {type: tdigest, source_column: FinalPropensity, score_property: FinalPropensity, outcome: negative, k: 500}
      UniqueCustomers_cpc:           {type: cpc, source_column: CustomerID, lg_k: 11}
      Priority_kll:                  {type: kll, source_column: Priority, k: 200}
      Category_topk:                 {type: topk, source_column: Category, lg_max_map_size: 10}
```

The ingestion runner assigns a hidden source-order index before transforms when
`personalization` or `novelty` is present. Their bounded samples are restored to
that source order inside the group callback, so Polars streaming scheduling and
the `bulk` sketch plan cannot silently select different rows. This stabilization
carries a score-processor algorithm revision in its computation hash. The
adaptive native Polars reductions for large personalization/novelty groups carry
a separate revision because their deterministic floating reduction can differ
from the scalar path in insignificant trailing digits. Together these revisions
require a replay for affected sources with older score-distribution aggregates;
unrelated sources retain their existing computation hashes. The hidden index is
discarded with the raw chunk and is never persisted.

### 5.3 chunk_aggregate algorithm

```text
F = lazy_frame
F = F.filter(F[outcome.column] in (pos ∪ neg))
F = F.with_columns(Outcome_Boolean = (F[outcome.column] in pos))
F = F.filter(any(Outcome_Boolean) over group_keys)             # drop groups with zero positives
F = F.filter(F.Outcome_Boolean == max(F.Outcome_Boolean).over(dedup_keys))
G = F.group_by(group_keys).agg(
    pl.len().alias("Count"),
    map_groups(personalization, [CustomerID, Name]).alias("personalization"),
    map_groups(novelty, [CustomerID, InteractionID, Name]).alias("novelty"),
    build_tdigest(F[states.Propensity_tdigest_positives.source_column] where Outcome_Boolean is true).alias("Propensity_tdigest_positives"),
    build_tdigest(F[states.Propensity_tdigest_negatives.source_column] where Outcome_Boolean is false).alias("Propensity_tdigest_negatives"),
    build_tdigest(F[states.FinalPropensity_tdigest_positives.source_column] where Outcome_Boolean is true).alias("FinalPropensity_tdigest_positives"),
    build_tdigest(F[states.FinalPropensity_tdigest_negatives.source_column] where Outcome_Boolean is false).alias("FinalPropensity_tdigest_negatives"),
    build_cpc(F[CustomerID]).alias("UniqueCustomers_cpc"),
)
```

`personalization` and `novelty` formulas live in reference/algorithms.md §5; their inputs are `(CustomerID, ActionName)` and `(CustomerID, InteractionID, ActionName)` respectively.

Each t-digest or KLL state selects its transformed input with the required
`source_column`. `score_properties` declares each score column and its role;
it does not generate states. Only states with
`outcome: positive` or `outcome: negative` apply an outcome-side filter.
Unconditioned states include all retained outcome rows. Explicit
CPC/HLL/Theta/Top-K states also require `source_column`.
Curve metrics then select stored positive and negative t-digest states with
`positive_state` and `negative_state` in `metrics.yaml`.

### 5.4 merge algorithm

- `Count` → SUM.
- `personalization`, `novelty` → weighted mean by `Count`.
- t-digest/KLL states → deserialize/merge/reserialize.
- CPC/HLL/Theta/Top-K states → their state-specific union/merge.

### 5.5 Derived metrics

| Metric | Kind | Inputs | Output |
|---|---|---|---|
| `ROC_AUC` | `curve_from_digests` | `<ScoreProperty>_tdigest_positives, <ScoreProperty>_tdigest_negatives` | scalar |
| `AvgPrecision` | `curve_from_digests` | same | scalar |
| `Calibration` | `calibration_from_digests` | same property-backed positive/negative digest pair | struct: bins, predicted, observed |
| `Personalization`, `Novelty` | (state passthrough) | `personalization`, `novelty` | scalar |
| `UniqueCustomers` | `approx_distinct_count` | `UniqueCustomers_cpc` | scalar |

reference/algorithms.md §4 describes the curve reconstruction.

### 5.6 Edge cases

- Empty positives or negatives: `ROC_AUC = 0`, `AP = 0`, calibration arrays default to `[0.0]`.
- Highly imbalanced groups: t-digest with `k=500` is well-calibrated for `n ≥ 100` per group; the engine emits a warning when a chunk produces a group with `Count < 100`.
- `Count < 50_000` uses the full group for `personalization` and `novelty`;
  `50_000 ≤ Count < 100_000` uses the second-half slice; `Count ≥ 100_000`
  uses a 50,000-row middle slice. These fixed implementation thresholds are
  specified in `reference/algorithms.md §5`.

---

## 6. entity_lifecycle processor (CLV)

### 6.1 Purpose

Per-customer lifetime aggregates from a transaction-like Source (Product Holdings). Used to derive RFM segments, CLV inputs, and downstream BG/NBD or Pareto/NBD models.

### 6.2 YAML

```yaml
processors:
  - id: clv
    source: holdings
    kind: entity_lifecycle
    group_by: [ControlGroup]
    time:
      property: PurchasedDateTime
      grain: summary
      calendar:
        timezone: UTC
        week_start: monday
        fiscal_year_start_month: 1

    keys:
      customer_id:    CustomerID
      order_id:       HoldingID
      monetary:       OneTimeCost
      purchase_date:  PurchasedDateTime

    lifespan_years: 9

    states:
      unique_holdings:      {type: count, source_column: HoldingID, distinct: true}
      lifetime_value:       {type: value_sum, source_column: OneTimeCost}
      MinPurchasedDate:     {type: min,       source_column: PurchasedDateTime}
      MaxPurchasedDate:     {type: max,       source_column: PurchasedDateTime}
      UniquePurchasers_cpc: {type: cpc, source_column: CustomerID, lg_k: 11}
```

### 6.3 chunk_aggregate algorithm

```text
F = lazy_frame
F = F.filter(F[purchase_date_col] > now() - relativedelta(years=lifespan_years))
F = F.with_columns(F[monetary_col].cast(Float64))

# Calendar derivation specific to lifecycle (per row).
F = F.with_columns(
    Day      = F[purchase_date_col].dt.date(),
    Month    = F[purchase_date_col].dt.strftime("%Y-%m"),
    Year     = F[purchase_date_col].dt.year().cast(String),
    Quarter  = concat(Year, "_Q", F[purchase_date_col].dt.quarter().cast(String)),
)

group_keys = group_by + [customer_id_col, "Year", "Quarter"]   # entity-level keys

agg = [
    build_state(state_id, typed_state_spec)
    for state_id, typed_state_spec in processor.states
]

G = F.group_by(group_keys).agg(agg)

OUTPUT G
```

### 6.4 merge / compact

- `unique_holdings` → SUM (per-(entity, year, quarter) granularity allows lossless sum across chunks).
- `lifetime_value` → SUM.
- `MinPurchasedDate`, `MaxPurchasedDate` → MIN/MAX.
- `UniquePurchasers_cpc` → union.
- Compaction to summary drops `Year, Quarter` and re-merges.

### 6.5 Derived metric — `lifecycle_summary` (RFM)

The metric explicitly binds the four required state roles; the names below are
examples, not reserved identifiers:

```yaml
CLV_Summary:
  processor: clv
  kind: lifecycle_summary
  entity_column: CustomerID
  holdings_state: unique_holdings
  monetary_state: lifetime_value
  first_purchase_state: MinPurchasedDate
  last_purchase_state: MaxPurchasedDate
```

```text
INPUT compacted lifecycle frame F

observation_end = max(F.MaxPurchasedDate)
group_keys      = group_by + [customer_id]    # entity-level

S = F.group_by(group_keys).agg([
    pl.n_unique(customer_id).alias("customers_count"),
    pl.sum(unique_holdings).alias("unique_holdings"),
    pl.sum(lifetime_value).alias("lifetime_value"),
    pl.min(MinPurchasedDate).alias("MinPurchasedDate"),
    pl.max(MaxPurchasedDate).alias("MaxPurchasedDate"),
])

S = S.with_columns(
    frequency      = unique_holdings - 1,
    recency_raw    = (MaxPurchasedDate - MinPurchasedDate).days,
    tenure         = (observation_end - MinPurchasedDate).days,
    monetary_value = lifetime_value / unique_holdings,
)
S = S.with_columns(
    recency        = tenure - recency_raw,                     # so larger = more recent
    monetary_value = if_else(frequency == 0, 0.0, monetary_value),
)

# Quartile labeling.
labels    = ["1","2","3","4"]
labels_r  = ["4","3","2","1"]    # reversed for recency

S = S.with_columns(
    f_quartile = qcut(frequency,      4, labels  ),
    m_quartile = qcut(monetary_value, 4, labels  ),
    r_quartile = qcut(recency,        4, labels_r),
)

S = S.with_columns(
    rfm_seg     = concat(r_quartile, f_quartile, m_quartile),
    rfm_segment = lookup_segment(rfm_seg, rfm_segments_dict, default="Unknown"),
    rfm_score   = mean(r_quartile.to_decimal, f_quartile.to_decimal, m_quartile.to_decimal),
)

OUTPUT S
```

`rfm_segments_dict` maps RFM codes (e.g. `"344"`) to segment names. Built-in presets: `retail_banking`, `telco`, `e_commerce`, `default`. See reference/algorithms.md §7 for the full code-to-segment tables.

### 6.6 Edge cases

- `unique_holdings` of 0 or 1 implies `frequency = 0` and forces `monetary_value = 0`.
- `qcut` with all-equal values: enable `allow_duplicates=true` to fall back to fewer buckets without error.
- Customers with a single purchase get `recency = 0` (so `tenure - recency_raw = tenure`).

---

## 7. entity_set processor

### 7.1 Purpose

Pure approximate-set processor for unique-count style metrics that are not easily attached to another processor: DAU, MAU, unique reach, audience overlap, retention cohorts. Optional component (no equivalent in legacy app, but called out in `wiki/chunked-bi-metrics.md` and `wiki/industry-patterns-for-bi-metrics.md`).

### 7.2 YAML

```yaml
processors:
  - id: unique_users
    source: ih
    kind: entity_set
    group_by: [Day, Week, Month, Quarter, Year, Channel, PlacementType]
    time:
      property: OutcomeTime
      grain: daily
    states:
      ActiveUsers_cpc:    {type: cpc,   source_column: CustomerID, lg_k: 11}
      ActiveUsers_theta:  {type: theta, source_column: CustomerID, lg_k: 12}
```

### 7.3 chunk_aggregate algorithm

```text
G = F.group_by(group_by + finest_time_grain_columns).agg(
    build_cpc(F[source_column]).alias("ActiveUsers_cpc"),
    build_theta(F[source_column]).alias("ActiveUsers_theta"),
)
```

### 7.4 merge — set algebra

- CPC and HLL merge with `union`.
- Theta merges with `union`; intersect/diff are exposed via the metric DSL (`set_op` kind, see reference/algorithms.md §6).

### 7.5 Derived metrics

| Metric | Kind | Inputs | Output |
|---|---|---|---|
| `ActiveUsers` | `approx_distinct_count` | `ActiveUsers_cpc`, `ActiveUsers_hll`, or `ActiveUsers_theta` | scalar |
| `RetainedUsers_30d` | `set_op` | `ActiveUsers_theta(window_t-30, t-1)`, `ActiveUsers_theta(window_t-1)` | `count(intersection)` |
| `NewUsers_today` | `set_op` | `ActiveUsers_theta(today)`, `ActiveUsers_theta(history)` | `count(diff)` |

Relative `time_window` operands are evaluated from the daily aggregate at
`grain: summary`. Their anchor is the query `end` date when supplied, otherwise
the latest available `Day`. `last: Nd` is inclusive of the anchor; `between:
[offset_a, offset_b]` applies inclusive day/week offsets such as `[-30d,
-1d]`. Windowed set queries require a configured daily grain so retention is
computed from persisted sketches rather than raw events.

An operand without `time_window` means all aggregate history through the
anchor. When it is combined with a windowed operand, the planner removes the
query's lower scan bound so the all-time operand is not shortened by a report
date preset.

Reports therefore read exact daily freshness for windowed set metrics and
anchor relative presets to the latest common covered day across the page. A
partial month ending on September 18 remains anchored to September 18; its
monthly period label must not advance the window anchor to September 30.

The planner is responsible for finding the right `period` partitions and for assembling theta operands.

---

## 8. funnel processor

### 8.1 Purpose

Per-stage counts plus implied drop-off rates. Stage assignment is configured as a list of named conditions (AST). One row per `(group_by tuple, time_grain)` carries `<stage>_Count` for every stage.

### 8.2 YAML

```yaml
processors:
  - id: action_funnel
    source: ih
    kind: funnel
    group_by: [Day, Week, Month, Quarter, Year, Channel, PlacementType]
    time:
      property: OutcomeTime
      grain: daily
    stages:
      - {name: Impression, when: {op: eq, column: Outcome, value: Impression}}
      - {name: Clicked,    when: {op: eq, column: Outcome, value: Clicked}}
      - {name: Conversion, when: {op: eq, column: Outcome, value: Conversion}}
    entity: CustomerID                # optional; if set, also produces "<stage>_Customers_cpc"
    states:                           # optional unscoped recipe states
      Region_cpc:   {type: cpc, source_column: Region, lg_k: 11}
      Category_topk: {type: topk, source_column: Category, lg_max_map_size: 10}
```

### 8.3 chunk_aggregate

```text
F = lazy_frame
G = F.group_by(group_by + finest_time_grain_columns).agg([
    pl.sum(when(stage1.when).then(1).otherwise(0)).alias("Impression_Count"),
    pl.sum(when(stage2.when).then(1).otherwise(0)).alias("Clicked_Count"),
    pl.sum(when(stage3.when).then(1).otherwise(0)).alias("Conversion_Count"),
    *[ build_cpc(when(s.when).then(F[entity]).otherwise(NULL)).alias(f"{s.name}_Customers_cpc")
       for s in stages ] if entity is set,
    *[ build_configured_sketch(state.source_column).alias(state.name)
       for state in unscoped_sketch_states ]
])
```

Stage customer sketches remain conditioned by the stage expression. Explicit
states that do not use a generated stage-state name are unscoped and can build
CPC, HLL, Theta, or Top-K over any configured source field, even when the
funnel has no `entity` default.

### 8.4 Derived metrics

| Metric | Kind | Inputs | Output |
|---|---|---|---|
| `<stageA>_to_<stageB>_rate` | `formula` | `<B>_Count, <A>_Count` | `B / A` |
| `<stageA>_dropoff` | `formula` | `<A>_Count, <B>_Count` | `(A − B) / A` |

---

## 9. snapshot processor

### 9.1 Purpose

State KPIs that don't fit additive rollups — current open subscriptions, MRR today, current backlog, current open tickets. Two flavors:

- `periodic` — one snapshot per cadence (daily / weekly / monthly), each independent.
- `accumulating` — one row per business entity, mutated through milestones (created → first_response → resolved).

### 9.2 YAML — periodic

```yaml
processors:
  - id: subscription_state
    source: subscriptions
    kind: snapshot
    snapshot_kind: periodic
    cadence: daily
    group_by: [Day, Week, Month, Quarter, Year, Plan, Region]
    time:
      property: as_of_date
      grain: daily
    states:
      ActiveSubs:  {type: count}
      MRR:         {type: value_sum, source_column: monthly_recurring}
      ChurnedSubs: {type: count, where: {op: eq, column: status, value: churned}}
```

### 9.3 YAML — accumulating

```yaml
processors:
  - id: ticket_lifecycle
    source: tickets
    kind: snapshot
    snapshot_kind: accumulating
    entity: ticket_id
    group_by: [Team, Severity]
    time:
      property: created_at
      grain: summary
    milestones:
      - {name: created_at,        property: created_at}
      - {name: first_response_at, property: first_response_at}
      - {name: resolved_at,       property: resolved_at}
    states:
      OpenTickets:      {type: count, where: {op: is_null, column: resolved_at}}
      MeanResolveHours: {type: pooled_mean, source_metric: resolve_hours, weight: ResolvedTickets}
      ResolvedTickets:  {type: count, where: {op: not_null, column: resolved_at}}
```

### 9.4 chunk_aggregate (periodic)

```text
F = lazy_frame.with_columns(as_of_date = today())   # or chunk's effective date
G = F.group_by(group_by + ['as_of_date']).agg(...)
```

Periodic snapshots never add `entity` to their grouping key. If an `entity`
field is present for sketch source defaults, it remains an input column only;
the persisted aggregate still has one row per `(as_of_date, group_by_tuple)`.

Snapshot rows retain `as_of_date`; physical storage still uses the common
`period=YYYY-MM` hive partition derived from that date. The query layer keeps
the latest `as_of_date` (bounded by the query range when supplied).

### 9.5 chunk_aggregate (accumulating)

```text
For each entity in the chunk:
    - upsert the entity's row with the latest non-null milestone columns
    - state aggregates are recomputed from the merged row

The "merge" rule is therefore "MAX(as_of_date) wins per entity"
inside the snapshot.parquet, with a deterministic tiebreaker.
```

Accumulating snapshots first keep the latest row per entity within a chunk.
Across chunks, immutable partials coexist and the query merge keeps the latest
`as_of_date` per entity with `created_at` as the deterministic tiebreaker. This
preserves atomic publication and history; vacuum removes superseded files.

---

## 10. frequency_response processor

### 10.1 Purpose

`frequency_response` measures how response changes with the number of impressions
of the same action in a fixed trailing time window. It also records the raw
propensity of the selected rank-2 action from an explicitly configured
comparison group, so the selected rank-1 action curve can be compared with the
opportunity cost of occupying the placement.

This is a sequence-aware processor. Its published contract remains
aggregate-first: only daily count and sum states are queryable. Ingestion can
either rescan bounded source history or retain a minimal, rebuildable sharded
checkpoint that is never exposed to reports.

### 10.2 YAML

```yaml
processors:
  - id: frequency_response
    source: ih
    kind: frequency_response
    time:
      property: DecisionTime
      grain: daily
      calendar:
        timezone: UTC
    columns:
      customer: CustomerID
      interaction: InteractionID
      action: ActionID
      placement: Placement
      rank: Rank
      outcome: Outcome
      propensity: Propensity
      priority: Priority               # optional diagnostic
    alternative_group_by: [Placement]
    positive_values: [Clicked]
    exposure_values: [Impression, Clicked]
    candidate_values: [Pending, Impression, Clicked]
    window_hours: 168
    partition_lag_hours: 24          # default 0; dependency padding only
    max_frequency: 7
    frequency_column: ExposureBucket
    checkpoint:
      mode: persistent_sharded       # default: source_scan
      shards: 64                     # default 64; routing, not anonymization
      retention_days: 9              # optional; >= active history + current
    group_by:
      - Day
      - Channel
      - Placement
      - ActionID
      - ExposureBucket
    states:
      Contacts:                    {type: count}
      Clicks:                      {type: count, source_column: ClickedContact}
      ComparableContacts:         {type: count, source_column: ComparableContact}
      ComparableClicks:           {type: count, source_column: ComparableClick}
      RunnerAvailable:            {type: count, source_column: RunnerAvailable}
      RunnerPropensitySum:        {type: value_sum, source_column: RunnerPropensity}
      PriorityComparableContacts: {type: count, source_column: PriorityComparableContact}
      FocalPriorityComparableSum: {type: value_sum, source_column: FocalPriorityComparable}
      RunnerPriorityComparableSum: {type: value_sum, source_column: RunnerPriorityComparable}
```

The processor accepts only `count` and `value_sum` states. `Day` and the
configured number-of-impressions column (`frequency_column` in YAML) are
derived by the processor; the other group-by
columns must be present after source transforms. The transformed source must
not already contain the configured number-of-impressions column because the processor owns
that derived name. Raw bindings and state inputs may not use the reserved
`__valuestream_` prefix. `rank` must have an integer dtype, while `propensity`
and an optional `priority` must be numeric; configure strict source casts when
the raw export uses text fields. These checks run on the target schema before
it is combined with history, so relaxed union coercion cannot mask a bad target
day.

`alternative_group_by` is a list of zero or more **physical transformed source
columns** appended to the processor's implicit customer + interaction key. It
is evaluated before aggregation and is separate from the published `group_by`.
The default is `[Placement]`, so the default comparison group is `CustomerID +
InteractionID + Placement` for the bindings above. Multiple fields are allowed,
for example `[Placement, Channel, Issue]`.

The customer and interaction bindings are always present and must not be
repeated in `alternative_group_by`. Customer preserves customer/checkpoint-shard
isolation; interaction keeps ranked candidates inside one decision. Canonical
examples are:

- `[]` selects across placements within the same customer and interaction.
- `[Placement]` selects within the same customer, interaction, and placement
  (the default).
- `[Placement, Channel]` further requires the same channel.
- `[Placement, Channel, Issue]` requires every listed field to match.

The processor never crosses `InteractionID` when selecting a ranked candidate.
Null values in an additional comparison field form one group: a null value on
the selected rank-1 row matches a null value on a candidate row.

A processor-level `filter` runs on transformed source rows before contacts,
number-of-impressions buckets, or virtual state columns are derived. It may
reference only raw/transformed source fields—not `Day`, the configured
number-of-impressions column,
`ClickedContact`, `RunnerPropensity`, or another processor-created field.
State-level `where` expressions run after enrichment and may use the documented
virtual fields.

### 10.3 Contact and number-of-impressions semantics

For one target day, the engine plans the target chunk plus the bounded set of
preceding calendar-day chunks needed to cover
`window_hours + partition_lag_hours`. The latter field is an input-planning
allowance for sources whose partition timestamp can trail the configured
decision timestamp; it does not change the semantic exposure window.

With `checkpoint.mode: source_scan`, the processor receives the transformed
target/history rows in an ephemeral frame. With `persistent_sharded`, each raw
chunk is transformed, filtered, classified, and projected through a partitioned
streaming sink into customer-hash shards without collecting the complete
prepared day. That complete target payload is the source for a second, narrow
history payload in the same atomic generation, so building the history role
does not reread raw IH. Candidate rows remain uncollapsed until current and
history shards are combined, preserving cross-partition duplicate precedence.
Target processing reads the complete corresponding target shard and only the
narrow history shard from each earlier day.
The two modes implement the same following rules:

1. A contact is identified by customer, interaction, action, placement, and
   rank. Repeated outcome rows for that contact are collapsed; a positive
   outcome wins over a non-positive exposure.
2. A selected rank-1 action contact is rank 1, has an exposure outcome, and belongs
   to the target chunk. Historical rows influence the number of impressions but are
   never emitted again.
3. The number of impressions is counted for the same customer, action, and placement
   in the strict interval `(decision_time - window_hours, decision_time]`.
   `max_frequency` is a terminal bucket: with a value of 7, the seventh and
   every later impression proxy are stored as `7` and may be labelled `7+` by
   a report.
4. The response flag comes from `positive_values`. `Clicked` may therefore win
   over an `Impression` row for the same contact.

The fixed window makes the x-axis reproducible. It is not a calendar-week
bucket and it does not reset at midnight or on Monday.

### 10.4 Selected rank-2 action semantics

The opportunity comparison is resolved within the implicit customer +
interaction keys plus the physical fields configured in
`alternative_group_by`. The processor selects exact rank 2 in that group when
it exists; otherwise it selects the smallest recorded rank greater than 1 in
the same group. “Selected rank-2
action” is therefore the business role: the underlying recorded rank can be
greater than 2 when rank 2 is absent. If no such row exists, the selected
rank-1 action contact remains in the all-contact response curve but is excluded
from comparable curves.

`RunnerPropensity` is the selected rank-2 action's raw `Propensity`, interpreted
as its expected response probability. `Priority` must not be substituted for this
field: priority drives arbitration ranking, but it can also include context,
business value, levers, and other multipliers, so it is neither a probability
nor a CTR. When configured, selected rank-1 action and selected rank-2 action
priority sums are retained as a separate arbitration diagnostic over rows where
both values exist.

Comparison scope is declarative and always decision-local. Keeping the default
physical `Placement` field answers the same-placement opportunity question and
may reduce comparable coverage. Removing `Placement` allows cross-placement
selection within the same interaction. In every case the fallback chooses the
smallest available rank
greater than 1 inside the complete configured group rather than silently
crossing one of its boundaries.

### 10.5 Merge and derived KPIs

All stored states merge by addition. Canonical formulas include:

```text
selected rank-1 action CTR = Clicks / Contacts
comparable selected rank-1 action CTR = ComparableClicks / ComparableContacts
selected rank-2 action expected CTR = RunnerPropensitySum / ComparableContacts
selected rank-2 action coverage = ComparableContacts / Contacts
response opportunity    = (ComparableClicks - RunnerPropensitySum)
                          / ComparableContacts
priority opportunity gap = (FocalPriorityComparableSum
                            - RunnerPriorityComparableSum)
                           / PriorityComparableContacts
```

The comparable selected rank-1 action and selected rank-2 action curves use the
same denominator and therefore belong on the same response-rate axis. Selected
rank-1 action CTR over all contacts should remain visible as a separate curve or
diagnostic because missing selected rank-2 actions change its population.

### 10.6 Dependency, idempotency, and limitations

Sources bound to this processor must discover ISO `YYYY-MM-DD` chunk IDs. The
target chunk remains the output and idempotency unit, while its input
fingerprint includes every bounded history file. Changing a historical file
therefore invalidates each later target whose window depends on it. Ordinary
processors bound to the same source still receive only the target chunk.

`partition_lag_hours` defaults to `0`. Set it to a conservative upper bound
when chunk dates are based on a later event time (for example, outcome date
versus decision time). The planner reads
`ceil((window_hours + partition_lag_hours) / 24)` preceding calendar days, then
the processor still enforces `(decision_time - window_hours, decision_time]`
by timestamp. Exactness requires UTC-aligned daily chunk IDs, or an equivalent
partition convention whose displacement from decision time stays within the
configured allowance. Extra dependency days only increase source I/O.

Every transformed history chunk is validated independently for the decision
time, customer, interaction, action, placement, rank, outcome, and any
processor-filter fields before it is combined or checkpointed. This prevents a
relaxed multi-day schema union from turning a missing history key into null and
silently undercounting the number of impressions.

`checkpoint.mode` has these exact meanings:

- `source_scan` (the compatibility default) retains no processor state and
  rereads the bounded source closure for each target.
- `persistent_sharded` persists a complete target-candidate role containing the
  fields required to repeat exact cross-partition normalization,
  number-of-impressions bucketing, deduplication, grouping/state calculation,
  and configured alternative-group selected rank-2 action resolution. In the
  same generation it persists a history role filtered to exposed rank-1 rows
  and projected to customer, interaction, action, placement, decision time,
  contact classification, and deterministic local order. The history role is
  derived from staged target shards and deliberately omits propensity,
  priority, report/alternative groups, outcome, and all rank>1 candidates.
  `shards`
  is an integer from `1` through `4096` and defaults to
  `64`; larger values lower per-shard memory while increasing file count.

All fields under `checkpoint` are execution/storage settings. They remain in
the full catalog hash for audit but are excluded from processor and source
computation hashes:

- `mode` selects the exact execution path;
- `shards` selects a separately hashed physical checkpoint layout;
- `retention_days` selects vacuum policy and is not part of artifact identity.

Changing these fields does not schedule aggregate replay. Unchanged targets
remain skipped. After a shard change, vacuum may remove the old layout and the
new layout is built lazily from authoritative IH for the bounded closure of the
next new or invalidated target. `--force` is still an explicit request to
rebuild both aggregates and state. Window, partition lag, columns, filters,
outcome/contact rules, groups, and states remain result semantics and therefore
do change computation hashes.

`retention_days` optionally sets the number of daily checkpoint partitions
kept per processor. Its default is
`ceil((window_hours + partition_lag_hours) / 24) + 1`, and an explicit value
cannot be smaller. Retention runs after each terminal source run and during
workspace vacuum. Replaying an older correction rebuilds any evicted IH
partition for that bounded replay and may evict it again afterward.

Persistent generations are source-fingerprint-addressed by the source,
processor semantic computation identity, independent checkpoint-layout
identity, chunk id, and raw-file fingerprint; their manifests also version the
checkpoint schema, customer dtype, shard-hash algorithm/seeds, and Polars
runtime used for native hashing. Before workers start, the parent validates the
manifest plus the exact payload roles pending work can open: `target` for a
current chunk, `history` for a historical dependency, or both when the chunk is
needed in both roles. Each selected role's declared files, sizes, row counts,
schemas, and SHA-256 digests are checked once; a history-only validation never
reads the complete target payload. A newly written shard supplies its manifest
metadata from that first inspection; publication revalidates the JSON against
those entries rather than hashing and scanning the file again. Customer
dtype must be stable across the bounded closure; normalize it in source
transforms if exports drift. They live in the processor-state namespace,
outside aggregates and ledger publication.
Queries, reports, DuckDB views, API/MCP, and SQL export cannot read them. A
missing, obsolete, or vacuumed generation is rebuilt from authoritative IH. A
corrupt generation is rejected rather than used silently; a forced run safely
replaces it from IH. A preparation failure fails only target chunks whose
closures require that generation. Checkpoint retention therefore changes
storage/rebuild cost rather than results.

Customer hashing determines which exact shard to open; it does not anonymize
the original customer key retained for exact comparison. Treat the checkpoint
as sensitive source-derived data, apply workspace access/encryption and
retention controls, and tokenize or HMAC identifiers upstream where required.
Sampling or sketches are not a transparent replacement because they cannot
preserve exact event order and selected rank-2 action joins.

A late positive outcome that arrives in a later chunk is not allowed to rewrite
an already materialized earlier-day contact; recompute the affected source
history when corrected source files replace the original export. Its changed raw
fingerprint creates a new checkpoint generation in persistent mode and
invalidates the same bounded target set in both modes.

An impression is only an exposure proxy unless the source explicitly guarantees
viewability. The processor does not infer dismisses, irritation, or a dismiss
rate from ordinary IH outcomes. Such a curve requires an explicit dismiss
event in the source and a separately defined state contract.

## 11. Putting it together — example workspace

Catalog excerpt (pruned for clarity; full example in design/replacement-design.md Appendix A):

```yaml
processors:
  - {id: engagement,       source: ih,         kind: binary_outcome,        ...}
  - {id: conversion,       source: ih,         kind: binary_outcome,        ...}
  - {id: experiment,       source: ih,         kind: binary_outcome,        ...}
  - {id: descriptive,      source: ih,         kind: numeric_distribution,  ...}
  - {id: model_ml_scores,  source: ih,         kind: score_distribution,    ...}
  - {id: action_funnel,    source: ih,         kind: funnel,                ...}
  - {id: clv,              source: holdings,   kind: entity_lifecycle,      ...}
  - {id: unique_users,     source: ih,         kind: entity_set,            ...}
  - {id: subscription_state, source: subscriptions, kind: snapshot,         ...}
```

A workspace can have any subset of these. The `ih` Source is shared across most of them; running `valuestream run --source ih` reads each chunk once and fans out to all 6 IH-bound processors in parallel.

---

## 12. Implementation checklist

For each processor, the implementer must deliver:

1. A YAML schema fragment (JSON Schema) under `schemas/processors/<kind>.json`.
2. A Python class implementing the `Processor` protocol.
3. Unit tests:
   - `test_chunk_aggregate_basic` — small synthetic frame, exact expected aggregate.
   - `test_merge_associativity` — `merge(A, merge(B, C)) == merge(merge(A, B), C)` for all state types.
   - `test_compact_idempotent` — `compact(compact(F)) == compact(F)` (already at the target grain).
   - `test_pooled_variance_correctness` — pooled var matches a brute-force computation on the un-grouped data, within `1e-9`.
   - `test_tdigest_curve_correctness` — ROC AUC reconstructed from digests is within `1e-2` of `sklearn.metrics.roc_auc_score` on the raw scores.
   - `test_cpc_distinct_correctness` — CPC distinct-count estimates and bounds cover deterministic fixtures for `n ∈ {1e2, 1e4, 1e6}`.
   - `test_hll_state_contract` — an explicitly configured HLL state builds, merges, and queries correctly.
4. A markdown reference page in `docs/processors/<id>.md` describing the canonical YAML, expected Source schema, and example output.
