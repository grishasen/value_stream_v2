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
    time.grains: list[str] # Day, Month, Summary, ...
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

    # 4. Cross-grain compaction
    def compact(
        self,
        frame: pl.DataFrame,             # finer grain
        target_grain: str,
    ) -> pl.DataFrame:
        """Reduce a finer-grained aggregate to a coarser one
        by dropping the fine time-grain column and re-merging."""

    # 5. Optional: derived helpers
    def derive(self, frame: pl.DataFrame, params: dict) -> pl.DataFrame:
        """Some processors expose derived helpers (e.g. RFM segmentation);
        most leave this to the metric DSL."""
```

The default `merge` and `compact` implementations are generic — they iterate over `self.states` and dispatch to the per-state-type merge rule. Subclasses override them only when the merge requires extra context (e.g. pooled variance needs the `_n_minus1_variance` and `_n_mean_diff_sq` temporaries documented in reference/algorithms.md §2.3).

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
| `count` | `INT64` | `pl.len()` or `pl.sum(<bool 0/1>)` | `SUM` | binary_outcome, numeric_distribution, score_distribution, entity_lifecycle, funnel, snapshot |
| `value_sum` | `FLOAT64` | `pl.sum(col)` | `SUM` | binary_outcome, numeric_distribution, snapshot |
| `min` | matches data | `pl.min(col)` | `MIN` | numeric_distribution, entity_lifecycle |
| `max` | matches data | `pl.max(col)` | `MAX` | numeric_distribution, entity_lifecycle |
| `pooled_mean` | `FLOAT64` (paired with `Count`) | `pl.mean(col)` | `weighted_mean(value, count)` | numeric_distribution, score_distribution |
| `pooled_variance` | `FLOAT64` (paired with `Count, Mean`) | `pl.var(col)` | Welford-merge | numeric_distribution |
| `tdigest` | `BLOB` | `datasketches.tdigest_double(k=500)` | deserialize/merge/serialize | numeric_distribution, score_distribution |
| `kll` | `BLOB` | `datasketches.kll_floats_sketch(k=200)` | KLL merge | optional, alternative to tdigest |
| `cpc` | `BLOB` | `datasketches.cpc_sketch(lg_k=11)` | union | default distinct count for binary_outcome, score_distribution, entity_lifecycle, entity_set, funnel, snapshot |
| `hll` | `BLOB` | `datasketches.hll_sketch(lg_k=12, tgt_type=HLL_8)` | union | backward-compatible / opt-in distinct counts |
| `theta` | `BLOB` | `datasketches.theta_sketch(lg_k=12)` | union/intersect/diff | entity_set (cohort) |
| `topk` | `BLOB` | `datasketches.frequent_strings_sketch(lg_max_map_size=10)` | merge | optional |

reference/algorithms.md gives the full algorithmic detail.

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
    group_by: [Channel, PlacementType, Issue, Group, CustomerType]
    time:
      column: OutcomeTime
      grains: [Day, Month, Summary]

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
      Positives:           {type: count}
      Negatives:           {type: count}
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
| `Lift` | `variant_compare` | `Positives, Negatives` per variant | TestCTR, ControlCTR, Lift, Lift_Z_Score, Lift_P_Val, StdErr |
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
    group_by: [Channel, PlacementType, Issue, Group, CustomerType, Outcome]
    time:
      column: OutcomeTime
      grains: [Day, Month, Summary]
    properties: [Outcome, Propensity, FinalPropensity, Priority, ResponseTime]
    quantile_engine: tdigest                # tdigest | kll | exact
    sketch_build_mode: bulk                 # bulk (default) | legacy; execution-only
    states:                                  # template, expanded per property
      "{prop}_Count":    {type: count,           per_property: true}
      "{prop}_Sum":      {type: value_sum,       per_property: true, numeric_only: true}
      "{prop}_Mean":     {type: pooled_mean,     per_property: true, numeric_only: true, weight: "{prop}_Count"}
      "{prop}_Var":      {type: pooled_variance, per_property: true, numeric_only: true, count: "{prop}_Count", mean: "{prop}_Mean"}
      "{prop}_Min":      {type: min,             per_property: true, numeric_only: true}
      "{prop}_Max":      {type: max,             per_property: true, numeric_only: true}
      "{prop}_tdigest":  {type: tdigest,         per_property: true, numeric_only: true, k: 500}
```

The engine expands `{prop}` for each template entry in `properties`. A state
whose name/metadata does not contain `{prop}` is an ordinary explicit state and
is built once. This lets recipes add, for example, `ResponseTime_kll` alongside
the default `ResponseTime_tdigest`, or a CPC/HLL/Theta/Top-K sketch over another
configured source field. For non-numeric properties (e.g. `Outcome` is a
string), only `Count` is generated unless an explicit compatible sketch state
is configured.

`sketch_build_mode` controls only how t-digest/KLL states cross the Python
callback boundary. `bulk` is the default: it combines the processor's
quantile-sketch construction for each group and reduces repeated Python
callbacks while retaining the same state and merge contracts. `legacy` keeps
the per-field callback plan as an explicit comparison and rollback escape
hatch. The mode does not change state definitions, merge semantics, or
source/processor computation hashes; the catalog hash still records the
authored runtime choice.

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
| `Median(p)` | `tdigest_quantile` | `<p>_tdigest` | scalar |
| `p25(p)`, `p75(p)`, `p90(p)`, `p95(p)`, `p99(p)` | `tdigest_quantile` | `<p>_tdigest` | scalar |
| `Skew(p)` | `formula` | `p25, p50, p75` | Bowley skew = `(p75 + p25 − 2·p50) / (p75 − p25)` |

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
    sketch_build_mode: bulk                 # bulk (default) | legacy; execution-only
    group_by: [Channel, PlacementType, Issue, Group, CustomerType]
    time:
      column: OutcomeTime
      grains: [Day, Month, Summary]

    score_properties: [Propensity, FinalPropensity]

    outcome:
      column: Outcome
      positive_values: [Clicked]
      negative_values: [Impression, Pending]

    dedup_keys: [InteractionID, ActionID, Rank]

    states:
      Count:                         {type: count}
      personalization:               {type: pooled_mean, weight: Count, source_metric: personalization}
      novelty:                       {type: pooled_mean, weight: Count, source_metric: novelty}
      Propensity_tdigest_positives:       {type: tdigest, source_column: Propensity,      score_property: Propensity,      outcome: positive, k: 500}
      Propensity_tdigest_negatives:       {type: tdigest, source_column: Propensity,      score_property: Propensity,      outcome: negative, k: 500}
      FinalPropensity_tdigest_positives:  {type: tdigest, source_column: FinalPropensity, score_property: FinalPropensity, outcome: positive, k: 500}
      FinalPropensity_tdigest_negatives:  {type: tdigest, source_column: FinalPropensity, score_property: FinalPropensity, outcome: negative, k: 500}
      UniqueCustomers_cpc:           {type: cpc, source_column: CustomerID, lg_k: 11}
      Priority_kll:                  {type: kll, source_column: Priority, k: 200}
      Category_topk:                 {type: topk, source_column: Category, lg_max_map_size: 10}
```

`sketch_build_mode` has the same execution-only semantics and computation-hash
exclusion as for `numeric_distribution`. `bulk` is the default for new and
omitted configurations; set `legacy` explicitly to compare the former
per-field callback plan or roll back the optimization without changing the
logical aggregate contract.

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

Each t-digest state selects its transformed input with `source_column`. For
implicit score states, `score_properties` generates
`<ScoreProperty>_tdigest_positives` and `<ScoreProperty>_tdigest_negatives`
for every selected score property. Older `score: primary` and
`score: calibrated` forms remain supported for existing catalogs and resolve
through legacy `score_columns` when present.
An explicit generic state named `<ScoreProperty>_tdigest` with no
`source_column` infers `<ScoreProperty>` from the state name and includes all
configured positive and negative outcome rows. Only states with
`outcome: positive` or `outcome: negative` apply an outcome-side filter.
`<ScoreProperty>_kll` follows the same unconditioned rule. Explicit
CPC/HLL/Theta/Top-K states use `source_column` and are built over all retained
outcome rows.
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
      column: PurchasedDateTime
      grains: [Year, Summary]

    keys:
      customer_id:    CustomerID
      order_id:       HoldingID
      monetary:       OneTimeCost
      purchase_date:  PurchasedDateTime

    model: non_contractual          # non_contractual | contractual

    # only for model=contractual
    recurring_period_column: RecurringPeriod
    recurring_cost_column:   RecurringCost

    lifespan_years: 9
    rfm_segments: retail_banking    # named preset; or inline dict

    states:
      unique_holdings:      {type: count,     source_aggregation: n_unique_order}
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
    pl.col(order_id_col).n_unique().alias("unique_holdings"),
    pl.sum(monetary_col).alias("lifetime_value"),
    pl.min(purchase_date_col).alias("MinPurchasedDate"),
    pl.max(purchase_date_col).alias("MaxPurchasedDate"),
    build_cpc(pl.col(customer_id_col)).alias("UniquePurchasers_cpc"),
]

if model == "contractual":
    agg += [(pl.col(recurring_cost_col) * pl.col(recurring_period_column)).sum().alias("recurring_costs")]

G = F.group_by(group_keys).agg(agg)

if model == "contractual":
    G = G.with_columns(lifetime_value = G.lifetime_value + G.recurring_costs).drop("recurring_costs")

OUTPUT G
```

### 6.4 merge / compact

- `unique_holdings` → SUM (per-(entity, year, quarter) granularity allows lossless sum across chunks).
- `lifetime_value` → SUM.
- `MinPurchasedDate`, `MaxPurchasedDate` → MIN/MAX.
- `UniquePurchasers_cpc` → union.
- Compaction to summary drops `Year, Quarter` and re-merges.

### 6.5 Derived metric — `lifecycle_summary` (RFM)

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
    group_by: [Channel, PlacementType]
    time:
      column: OutcomeTime
      grains: [Day, Month, Summary]
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
    group_by: [Channel, PlacementType]
    time:
      column: OutcomeTime
      grains: [Day, Month]
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
    group_by: [Plan, Region]
    time:
      column: as_of_date
      grains: [Day, Month, Summary]
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
      grains: [Summary]
    milestones:
      - {name: created_at,        column: created_at}
      - {name: first_response_at, column: first_response_at}
      - {name: resolved_at,       column: resolved_at}
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

## 10. Putting it together — example workspace

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

## 11. Implementation checklist

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
   - `test_hll_backward_compatibility` — explicitly configured HLL states still build, merge, and query correctly.
4. A markdown reference page in `docs/processors/<id>.md` describing the canonical YAML, expected Source schema, and example output.
5. (Optional) A migration mapping in `migration.py` if a legacy family corresponds to this processor.
