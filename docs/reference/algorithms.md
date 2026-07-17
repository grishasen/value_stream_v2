# Value Stream — Algorithms Reference

This document is the math contract. Every formula, sketch parameter, and statistical test that Value Stream relies on is specified here in enough detail that an implementer can write it from scratch and a reviewer can verify correctness from the doc alone.

Companion docs:

- reference/processors.md — which algorithms each Processor uses.
- reference/expression-dsl.md — the AST grammar for filters and formulas.

Conventions used below:

- All log functions are natural log (`ln`) unless explicitly stated `log2`.
- "RSE" = relative standard error.
- All sketch types refer to the Apache DataSketches Python bindings (`datasketches` package), with explicit parameters.

---

## 1. Mergeable state algebra

A state is **mergeable** iff there exists a binary operator `⊕` such that:

```text
merge({a, b}) = a ⊕ b
merge({a, b, c}) = a ⊕ b ⊕ c = (a ⊕ b) ⊕ c    (associative)
                                = a ⊕ (b ⊕ c)
                                = (b ⊕ a) ⊕ c   (commutative across rows)
```

All Value Stream state types satisfy this. The state catalog and merge rules:

| State type | Merge `⊕` |
|---|---|
| `count` | `a + b` |
| `value_sum` | `a + b` |
| `min` | `min(a, b)` |
| `max` | `max(a, b)` |
| `pooled_mean` | weighted mean (§2.2) |
| `pooled_variance` | Welford-merge (§2.3) |
| `tdigest` | digest-merge (§2.4) |
| `kll` | KLL-merge (§2.5) |
| `cpc` | union (§6.1) |
| `hll` | union (§6.2) |
| `theta` | union/intersect/diff (§6.3) |
| `topk` | frequent-items merge (§6.4) |

---

## 2. Numeric distribution algorithms

### 2.1 Counts and sums

Trivial. `Count = Σ count_i`, `Sum = Σ sum_i`.

### 2.2 Weighted (pooled) mean

For groups with counts `n_i` and means `m_i`:

```text
global_mean = Σ (n_i · m_i) / Σ n_i
```

Implementation: store `Mean` and `Count` per row; in merge, compute `weighted_mean(Mean, Count)` (Polars-DS `weighted_mean(value_col, weight_col)` does this in one expression).

### 2.3 Pooled variance — Welford merge

Given groups with `n_i`, `m_i`, `v_i` (sample variance, ddof=1):

Step 1 — global mean:
```text
N = Σ n_i
GroupMean = Σ (n_i · m_i) / N
```

Step 2 — sum of squares within and between:
```text
SS_within   = Σ ( (n_i − 1) · v_i )
SS_between  = Σ ( n_i · (m_i − GroupMean)² )
```

Step 3 — pooled variance:
```text
Var = (SS_within + SS_between) / (N − 1)
```

This is **exact** (assuming each `v_i` was computed with `ddof=1`) and associative under repeated merging.

Reference implementation pattern (in pseudocode close to Polars):

```text
# Inputs: a frame F with columns Count, Mean, Var per group.
# Step 1: compute GroupMean per merge group.
GM = F.group_by(group_keys).agg(weighted_mean(Mean, Count).alias("GroupMean"))
F  = F.join(GM, on=group_keys)
# Step 2: compute the two helper columns per row.
F  = F.with_columns(
    n_minus1_variance = (Count - 1) * Var,
    n_mean_diff_sq    = Count * (Mean - GroupMean) ** 2,
)
# Step 3: re-aggregate.
M = F.group_by(group_keys).agg(
    Count.sum(),
    weighted_mean(Mean, Count).alias("Mean"),
    n_minus1_variance.sum().alias("ssw"),
    n_mean_diff_sq.sum().alias("ssb"),
)
M = M.with_columns(Var = (ssw + ssb) / (Count - 1))
```

Numerical caveat: when `Count = 1`, `Var = 0` produces `NaN` after division. Guard with `pl.when(Count > 1).then(...).otherwise(NULL)`.

### 2.4 t-digest

**Library**: `datasketches.tdigest_double` with compression parameter `k`.

**Default**: `k = 500`. This gives a digest of ~32 KB serialized; quantile error ≤ 0.3% at the median, ≤ 0.05% at p99.

**Build (per group)**:
```text
sketch = tdigest_double(k=500)
if len(values) >= 32:
    values_array = writable_contiguous_float64(values)
    sketch.update(values_array)        # native bulk update
else:
    for value in values:
        sketch.update(value)           # avoid array setup for tiny groups
return sketch.serialize()              # bytes
```

Nulls are removed before construction. The array path removes one Python-to-C
call per value for non-trivial groups; the scalar path remains for tiny groups,
where allocating and normalizing an array can cost more than it saves. The
default `sketch_build_mode: bulk` is a separate optimization: it bundles all
t-digest and KLL fields for one group into a single Polars Python callback
before calling these native builders. `sketch_build_mode: legacy` retains the
former per-field callback plan as an explicit comparison and rollback path;
both modes emit merge-compatible sketch states.

**Merge**:
```text
merged = tdigest_double.deserialize(blobs[0])
for b in blobs[1:]:
    merged.merge(tdigest_double.deserialize(b))
return merged.serialize()
```

**Edge case — empty input**:
If a group has zero values for a property, the engine writes a serialized empty
digest so the column dtype stays `Binary` and downstream merges remain valid.
Quantile queries against an empty group return `0.0`.

**Quantile query**:
```text
sk = tdigest_double.deserialize(blob)
return sk.get_quantile(q)
```

**Histogram from digest** (used by descriptive_histogram chart):
```text
sk = tdigest_double.deserialize(blob)
edges = linspace(value_range[0], value_range[1], bins + 1)
masses = [ sk.get_cdf([right])[0] - sk.get_cdf([left])[0]
           for left, right in zip(edges[:-1], edges[1:]) ]
return edges, masses
```

### 2.5 KLL sketch (alternative for stronger error guarantees)

**Library**: `datasketches.kll_floats_sketch(k=200)`.
**Default**: `k = 200` (≈ 1.65% normalized rank error two-sided at 0.5).
KLL is mergeable, has stronger formal error guarantees than t-digest, and serializes to ~1.5 KB.

Use KLL where SLA-bound percentile errors matter; use t-digest where compactness and per-quantile speed matter.

Build / merge / quantile mirror t-digest with `k=200`; the native bulk array is
contiguous writable `float32`, and groups below the same 32-value threshold use
scalar updates.

---

## 3. Variant comparison and proportions

Variant comparison is a derived metric of `binary_outcome` processors with `variant_column` set. Inputs per group: `Positives_T, Negatives_T, Count_T, Positives_C, Negatives_C, Count_C` (T = test, C = control).

### 3.1 Click-through rate per variant

```text
CTR = Positives / (Positives + Negatives)
```

### 3.2 Lift

```text
Lift = (TestCTR - ControlCTR) / ControlCTR
```

NaN/Inf-safe: replace `Inf` with `0.0`, NaN with `0.0` (no control = no lift to display).

The same metric also returns `TestSampleSize`, `ControlSampleSize`, and the
absolute rate effect:

```text
AbsoluteRateDifference = TestCTR - ControlCTR
```

### 3.3 Confidence interval for the absolute rate difference

`variant_compare.confidence_level` defaults to `0.95`. For each arm, compute a
two-sided Wilson score interval at that confidence level. If the test interval
is `[L_T, U_T]` and the control interval is `[L_C, U_C]`, the deterministic
Newcombe-Wilson difference interval is:

```text
AbsoluteRateDifference_CI_Low  = L_T - U_C
AbsoluteRateDifference_CI_High = U_T - L_C
```

This interval uses only persisted `Positives` and `Negatives`; it never needs
raw assignments. The implementation reports the historical `Lift`, z-score,
p-value, `CTR`, and `StdErr` outputs unchanged for compatibility.

### 3.4 Standard error of a proportion

For one variant with `n = Positives + Negatives`:
```text
StdErr = sqrt( CTR · (1 − CTR) / n )
```

### 3.5 Two-proportion z-test (proportion z-test)

Given `c_T, c_C` successes, `n_T, n_C` total per variant:

```text
if any of c_T, c_C, n_T, n_C is 0:
    return { z_score: 0, z_p_val: 0 }

p_pool = (c_T + c_C) / (n_T + n_C)
z = (c_T/n_T − c_C/n_C) / sqrt( p_pool · (1 − p_pool) · (1/n_T + 1/n_C) )
p_one_tail = norm.cdf(−|z|)
p_two_tail = 2 · p_one_tail
return { z_score: z, z_p_val: p_two_tail }     # default two-tailed
```

Notation: `n` here is "successes + failures" — i.e. `Positives + Negatives` (rows we count for the rate). This matches how the engine stores `Count` after dedup (since outcomes outside positive ∪ negative are filtered out).

---

## 4. Curves from t-digests (ROC, AP, calibration)

These derive metrics for the `score_distribution` processor.

### 4.1 ROC AUC and average precision from t-digests

Inputs per group:
- `tdigest_pos` — digest of scores where outcome was positive,
- `tdigest_neg` — digest of scores where outcome was negative.

Procedure:

```text
thresholds = linspace(0, 1, 101)        # 101 evenly spaced thresholds

pos_sk = tdigest.deserialize(tdigest_pos)
neg_sk = tdigest.deserialize(tdigest_neg)

if pos_sk.is_empty() or neg_sk.is_empty():
    return { roc_auc: 0, average_precision: 0,
             tpr: [0], fpr: [0], precision: [0], recall: [0],
             pos_fraction: 0 }

P = pos_sk.get_total_weight()
N = neg_sk.get_total_weight()

cdf_p = array(pos_sk.get_cdf(thresholds))
cdf_n = array(neg_sk.get_cdf(thresholds))

tpr = 1.0 - cdf_p
fpr = 1.0 - cdf_n

# Sort by FPR ascending for trapezoidal integration.
idx = argsort(fpr)
fpr_s = fpr[idx]; tpr_s = tpr[idx]
roc_auc = trapz(tpr_s, fpr_s)

# Precision–recall reconstruction.
recall    = tpr[::-1]            # threshold descending
fpr_desc  = fpr[::-1]
TP        = P * recall
FP        = N * fpr_desc
precision = TP / (TP + FP + 1e-10)

# Make precision monotone non-increasing in recall (interp-AP).
precision = max_accumulate(precision[::-1])[::-1]

# Anchor the PR curve at recall=0 / precision=1.
if recall[0] != 0.0:
    recall    = [0.0] ++ recall
    precision = [1.0] ++ precision

dr = recall[1:] - recall[:-1]
average_precision = sum(dr * precision[1:])

pos_fraction = P / (P + N)
return { roc_auc, average_precision, tpr_s, fpr_s, precision, recall, pos_fraction }
```

Why 101 thresholds? Empirically a sweet spot — closer than that yields no meaningful gain on `k=500` digests, but doubles compute. The number is parameterizable in the YAML (`metrics.<name>.curve_resolution`).

### 4.2 Gain and lift curves from reconstructed ROC arrays

The chart layer derives gain and lift from the same `fpr`, `tpr`, and
`pos_fraction` arrays returned by `curve_from_digests`:

```text
sample_fraction = pos_fraction * tpr + (1 - pos_fraction) * fpr
gain = tpr
lift = gain / sample_fraction       # 0 when sample_fraction is effectively 0
```

The gain chart compares `gain` to the random baseline `y = x`; the lift chart
compares `lift` to the random baseline `y = 1`.

### 4.3 Calibration from t-digests

Inputs: `tdigest_pos`, `tdigest_neg` (typically of `FinalPropensity`, the calibrated score).

Procedure (mirrors the canonical Pega calibration view):

```text
# Bins: denser at low scores where most of the mass lives.
edges = linspace(0.0, 0.1, 10) ++ linspace(0.1, 1.0, 17)    # 26 edges, 25 bins (deduped)

P = pos_sk.get_total_weight()
N = neg_sk.get_total_weight()

cdf_p = pos_sk.get_cdf(edges)
cdf_n = neg_sk.get_cdf(edges)

delta_p = diff(cdf_p)            # mass per bin (positive)
delta_n = diff(cdf_n)            # mass per bin (negative)

pos_in_bin   = P * delta_p
neg_in_bin   = N * delta_n
total_in_bin = pos_in_bin + neg_in_bin
positives_rate = pos_in_bin / total_in_bin

# Mean predicted propensity per bin: weighted mean of bin-internal quantiles
# from positive and negative digests.
for each bin i (left=edges[i], right=edges[i+1]):
    if cdf_b_pos == cdf_a_pos and cdf_b_neg == cdf_a_neg:
        mean_propensity = (left + right) / 2
    else:
        # Sample 10 quantiles inside the bin from each digest.
        q_pos = linspace(cdf_a_pos, cdf_b_pos, 10, endpoint=False)
        pos_vals = [pos_sk.get_quantile(q) for q in q_pos]
        pos_bin_mean = mean(pos_vals)
        # ...same for neg...
        if total_in_bin[i] > 0:
            mean_propensity = (pos_bin_mean*pos_in_bin[i] + neg_bin_mean*neg_in_bin[i]) / total_in_bin[i]
        else:
            mean_propensity = (left + right) / 2
    output:
       calibration_bin[i]   = (left + right) / 2
       calibration_proba[i] = mean_propensity
       calibration_rate[i]  = positives_rate[i]   if total_in_bin[i] > 0 else 0
```

The calibration plot then draws `y = calibration_rate vs. x = calibration_proba`; a perfectly calibrated model lies on `y = x`.

---

## 5. ML helpers — personalization and novelty

### 5.1 Personalization (cosine-similarity proxy)

Definition (after Statisticianinstilettos's `recmetrics`):

```text
1 − mean( cosine_sim(rec_i, rec_j) over all pairs i ≠ j )
```

Direct user-pair computation is `O(N²)`. Value Stream instead accumulates sparse
per-customer action-count vectors and computes the same result in `O(N)` input
work plus the observed customer/action pairs:

```text
# Inputs: per-customer list of action names within the group.
# Step 1: count each exact action name per customer (a sparse matrix R).
R[customer, action] = count(customer, action)

# Step 2: L2-normalize rows.
R_norm[customer, action] = R[customer, action] / ||R[customer]||₂

# Step 3: total similarity sum = || sum(R_norm, axis=0) ||²
s = R_norm.sum(axis=0)                # one value per observed action
total_sim = sum(s_i²)

# Step 4: subtract diagonals (each customer with themselves contributes 1.0).
off_diag_sum = total_sim − N

# Step 5: average off-diagonal similarity.
avg_sim = off_diag_sum / (N · (N − 1))    # for N > 1, else 0

return 1 − avg_sim
```

Returns 0 if `N <= 1`.

For small groups and repeatedly updated low-cardinality customer populations,
the implementation uses Python `Counter` objects because their setup cost is
lower. For Polars Series with at least 256 rows and sufficiently high customer
cardinality, grouped counts, window norms, and action sums execute as native
Polars expressions. Both paths implement the formula above; the native result
is rounded to 12 significant decimal digits so parallel reduction order cannot
change persisted bytes across repeated runs. The native-reduction algorithm
revision is part of the score processor computation hash.

**Subsampling rules** (preserve representativeness while bounding compute):
- `N < 50_000` → use full sample.
- `50_000 ≤ N < 100_000` → use the second half of rows.
- `N ≥ 100_000` → use a 50_000-row middle slice.

(The legacy app uses `5_000`-step thresholds; the new defaults of `50_000` reflect the larger workspaces seen in production.)

### 5.2 Novelty (information-theoretic)

For a group with action counts `c_a`:

```text
unique_users   = n_distinct(CustomerID)
total_self_info = Σ_a [ c_a · −(log2(c_a / unique_users) + 1e-10) ]
max_rec_length  = max over interactions of len(actions in that interaction)
novelty = total_self_info / (unique_users · max_rec_length)
```

Returns 0 if `unique_users == 0` or `max_rec_length == 0`. Subsampling rules same as personalization.

For Polars Series groups of at least 256 rows, distinct users, action
frequencies, and interaction lengths are reduced with native Polars expressions;
smaller or non-Series inputs retain the scalar compatibility path.

Both metrics are stored as `pooled_mean` states with `Count` as the weight, so cross-chunk merging is just a weighted mean.

---

## 6. Approximate-set algorithms

### 6.1 CPC (default distinct-count sketch)

**Library**: `datasketches.cpc_sketch(lg_k)` and `datasketches.cpc_union(lg_k)`.

**Default**: `lg_k = 11`. CPC is the default for newly generated unique-count
states because it provides compact serialized state at accuracy comparable to
the previous HLL default. Existing HLL states remain supported and readable.

**Build (per group)**:
```text
sketch = cpc_sketch(lg_k=11)
for v in column.drop_nulls():
    sketch.update(str(v))     # always feed a stable string representation
return sketch.serialize()     # bytes
```

For string, categorical/enum, integer, Boolean, date, and decimal source
columns, the group plan performs the exact Python-string normalization with
native Polars expressions before entering the sketch callback. This removes a
Python `str(...)` call per distinct value. Dtypes whose Polars spelling can
differ from Python (`Float`, datetime/time/duration, binary, and nested values)
retain the scalar compatibility path.

**Union (merge)**:
```text
union = cpc_union(lg_k=11)
for blob in blobs:
    union.update(cpc_sketch.deserialize(blob))
return union.get_result().serialize()
```

**Estimate and bounds**:
```text
sk = cpc_sketch.deserialize(blob)
estimate = sk.get_estimate()
lower = sk.get_lower_bound(kappa=2)
upper = sk.get_upper_bound(kappa=2)
```

CPC supports union, but not intersection or difference. Use Theta when a KPI
requires set algebra. Every blob in a state must use the same `lg_k`; changing
`lg_k` or switching among CPC, HLL, and Theta requires replay from raw chunks.

### 6.2 HLL (HyperLogLog++)

**Library**: `datasketches.hll_sketch(lg_k, tgt_type)`.

**Defaults**:
- `lg_k = 12` → 4096 buckets, ~4 KB sketch, ±1.6% RSE at 1σ.
- `tgt_type = HLL_8` (one byte per bucket; HLL_4 halves the size with the same RSE but higher CPU on merges).

**Build (per group)**:
```text
sketch = hll_sketch(lg_k=12, tgt_type=HLL_8)
for v in column.drop_nulls():
    sketch.update(str(v))     # always feed a stable string representation
return sketch.serialize_compact()    # bytes
```

**Union (merge)**:
```text
union = hll_union(lg_k=12)
for blob in blobs:
    union.update(hll_sketch.deserialize(blob))
return union.get_result(tgt_type=HLL_8).serialize_compact()
```

**Estimate**:
```text
sk = hll_sketch.deserialize(blob)
return sk.get_estimate()
```

**Invariant**: every blob participating in a union must have the same `lg_k`. The engine validates this at write time (raises a helpful error otherwise).

HLL remains a supported opt-in state type for existing catalogs and workloads
that prioritize HLL serialization/deserialization speed over compact CPC state.
It is no longer the generated default for unique counts.

### 6.3 Theta sketch

**Library**: `datasketches.update_theta_sketch(lg_k=12)`; for unions/intersections use `theta_union`, `theta_intersection`, `theta_a_not_b`.

**Use case**: when we need set algebra, not just unique counts.

**Cardinality query**: a compact Theta sketch exposes `get_estimate()` and is
therefore also a valid `approx_distinct_count` input. CPC remains the generated
default for count-only KPIs; choose Theta when the persisted state will be
reused for union, intersection, or difference.

**Build / Union**: analogous to the cardinality-sketch procedures above.

**Cohort retention example** (in metric DSL):
```yaml
RetainedUsers_30d:
  source: unique_users
  kind: set_op
  op: intersection
  operands:
    - {state: ActiveUsers_theta, time_window: {last: 1d}}
    - {state: ActiveUsers_theta, time_window: {between: [-30d, -1d]}}
  output: count
```

The planner reads two theta blobs, runs `theta_intersection`, and reports `get_estimate()`.
The implemented planner resolves relative windows against the explicit query
end date or, when absent, the latest available daily aggregate date. `last` and
`between` bounds are inclusive and accept day/week durations. Windowed set
operations currently return summary-grain results and require a daily physical
aggregate.

### 6.4 Frequent-items (top-K)

**Library**: `datasketches.frequent_strings_sketch(lg_max_map_size=10)`.

Optional state for "top campaigns by count" style metrics. Merge: built-in. Output: list of `(item, estimate, lower_bound, upper_bound)` tuples.

For the same safe string-normalizable dtypes used by cardinality sketches,
Polars normalizes item strings before the callback, removing one Python
`str(...)` call per row. The callback deliberately preserves the original item
order and performs the same unweighted sketch updates as the compatibility
path: frequent-items sketches are order-sensitive once cardinality exceeds
their map capacity. Unsafe dtypes keep the original Python conversion path.

---

## 7. RFM segmentation (entity_lifecycle derive)

### 7.1 Computing R, F, M from lifecycle aggregates

Per (entity = customer_id, optional group-by columns):

```text
unique_holdings   = number of distinct orders by this customer
lifetime_value    = sum of monetary values
MinPurchasedDate  = earliest purchase
MaxPurchasedDate  = latest purchase

frequency      = unique_holdings − 1            # repeat purchases
tenure         = (observation_period_end − MinPurchasedDate).days
recency_raw    = (MaxPurchasedDate − MinPurchasedDate).days
recency        = tenure − recency_raw            # so larger = "more recently active"
monetary_value = lifetime_value / unique_holdings
                 (set to 0 when frequency == 0)
```

`observation_period_end` is `max(MaxPurchasedDate)` across the entire summary frame (after compaction).

### 7.2 Quartile labeling

```text
F: f_quartile in {1,2,3,4}             # higher = more frequent
M: m_quartile in {1,2,3,4}             # higher = more monetary
R: r_quartile in {1,2,3,4}             # higher = MORE RECENT
                                         (use reversed labels in qcut)
```

Polars helper: `pl.col(x).qcut(4, labels=labels, allow_duplicates=true)`.

### 7.3 Segment code

```text
rfm_seg = concat(r_quartile, f_quartile, m_quartile)    # e.g. "344"
```

### 7.4 Segment dictionaries

A segment dictionary maps `rfm_seg` → segment name. Defaults:

```yaml
default:
  Premium Customer:  ["334","443","444","344","434","433","343","333"]
  Repeat Customer:   ["244","234","232","332","143","233","243","242"]
  Top Spender:       ["424","414","144","314","324","124","224","423","413","133","323","313","134"]
  At Risk Customer:  ["422","223","212","122","222","132","322","312","412","123","214"]
  Inactive Customer: ["411","111","113","114","112","211","311"]

retail_banking:
  # tuned for retail-banking patterns; same shape, different mapping
  Premium Customer:  ["334","443","444","344","434","433","343","333"]
  Repeat Customer:   ["244","234","232","332","143","233","243","242"]
  ...
```

The engine ships `default`, `retail_banking`, `telco`, `e_commerce`. A workspace can override with a literal dict.

### 7.5 RFM score

```text
rfm_score = round( mean(r_quartile, f_quartile, m_quartile) , 2 )
```

with each quartile cast to decimal.

---

## 8. Statistical tests for experiment processors

These derive metrics for `binary_outcome` processors with `variant_column` set or the experiment columns included in `group_by`.

### 8.1 Two-sample z-test on proportions

Defined in §3.5. Used by `proportion_test` metric kind on a 2-row contingency.

### 8.2 Pearson chi-square test of homogeneity

Inputs: a 2D contingency table (variants × {Positives, Negatives}).

```text
g, p, dof, expected = scipy.stats.chi2_contingency(table, correction=False)
return { chi2_stat: g, chi2_dof: dof, chi2_p_val: p }
```

For a `2 × 2` table, also compute the sample odds ratio and 95% CI:

```text
res = scipy.stats.contingency.odds_ratio(table, kind="sample")
return {
    chi2_odds_ratio_stat:  res.statistic,
    chi2_odds_ratio_ci_low,
    chi2_odds_ratio_ci_high  # = res.confidence_interval(confidence_level=0.95)
}
```

### 8.3 G-test (log-likelihood ratio)

```text
g, p, dof, expected = scipy.stats.chi2_contingency(table, lambda_="log-likelihood", correction=False)
return { g_stat: g, g_dof: dof, g_p_val: p, g_odds_ratio_*: ... }
```

### 8.4 Single-degree-of-freedom G-test (variant proportion)

For a 2×2 contingency `[[c_T, n_T − c_T], [c_C, n_C − c_C]]`:

```text
expected_T_succ = n_T * (c_T + c_C) / (n_T + n_C)
expected_T_fail = n_T * ((n_T − c_T) + (n_C − c_C)) / (n_T + n_C)
expected_C_succ = n_C * (c_T + c_C) / (n_T + n_C)
expected_C_fail = n_C * ((n_T − c_T) + (n_C − c_C)) / (n_T + n_C)

G = 2 · (
    c_T · ln(c_T / expected_T_succ) +
    (n_T − c_T) · ln((n_T − c_T) / expected_T_fail) +
    c_C · ln(c_C / expected_C_succ) +
    (n_C − c_C) · ln((n_C − c_C) / expected_C_fail)
)
p = 1 − chi2.cdf(G, df=1)
return { g_test_stat: G, g_p_val: p }
```

### 8.5 Defaults

- All tests are two-sided unless stated otherwise.
- No Yates correction.
- Sample odds ratio (not unconditional MLE) is reported; CI uses Fisher's noncentral hypergeometric (`scipy`'s `kind="sample"`).

---

## 9. Calendar derivations

### 9.1 Default calendar (`derive_calendar` transform)

From a timestamp column `ts`:

| Output column | Polars expression | Example |
|---|---|---|
| `Day` | `ts.dt.date()` | `2024-08-21` |
| `Month` | `ts.dt.strftime("%Y-%m")` | `"2024-08"` |
| `Year` | `ts.dt.year().cast(Int16)` | `2024` |
| `Quarter` | `ts.dt.year().cast(Utf8) + "_Q" + ts.dt.quarter().cast(Utf8)` | `"2024_Q3"` |

### 9.2 ISO week (optional)

| `Week` | `ts.dt.strftime("%G-W%V")` | `"2024-W34"` |

### 9.3 Period partition values

| Grain | Period |
|---|---|
| `daily` | `Day.dt.strftime("%Y-%m")` (the file partitions monthly) |
| `monthly` | `Month` |
| `quarterly` | `Quarter` |
| `yearly` | `Year` (as string) |
| `summary` | default `Month`; configurable to `Quarter` or `Year` |

### 9.4 Time zone

All timestamps are normalized to UTC at ingestion. Display-time-zone conversion is a presentation concern, applied at query time.

---

## 10. Numeric robustness checklist

Implementers should add tests for these edge cases per algorithm:

| Algorithm | Edge case | Expected behavior |
|---|---|---|
| Pooled variance | `Count == 1` | NaN/NULL, not crash |
| t-digest curve | empty pos or neg | `roc_auc=0, ap=0`, arrays `[0]` |
| CPC/HLL/Theta distinct | empty input | `0` |
| Theta intersect | disjoint operands | `0` |
| Personalization | `N < 2` | `0.0` |
| Novelty | `unique_users == 0` | `0.0` |
| Lift | `ControlCTR == 0` | `Lift = 0`, `Lift_P_Val` from z-test |
| Z-test proportions | any zero count | `{z_score: 0, z_p_val: 0}` |
| Skew (Bowley) | `p75 == p25` | NULL |
| qcut | all values equal | `allow_duplicates=true`, fewer buckets |

---

## 11. Test fixtures (recommended)

For reproducibility tests, ship these fixtures in `tests/fixtures/`:

1. `imdb_small.parquet` (2 MB) — synthetic Pega-shaped IH with 200 K rows, three channels, three placements, five issues, deterministic seed; expected `CTR, ROC AUC, ConversionRate` printed to `expected.json`.
2. `holdings_small.parquet` (200 KB) — synthetic holdings with three customer cohorts, one year of data; expected RFM segment counts in `expected_rfm.json`.
3. `binary_outcome_property_suite.json` — 1 K randomly generated `(positives, negatives)` matrices across three variants with computed expected `lift / z-test / chi2 / g`. Used as parametric tests.
4. `tdigest_property_suite.json` — 100 random distributions (uniform, exponential, mixture-of-Gaussians) with their `roc_auc` from `sklearn.metrics.roc_auc_score`. Used as parametric tests for `curve_from_digests`.

---

## 12. Numerical-equivalence rules

Whenever Value Stream claims a number is "exact":
- Counts and sums are exact.
- Pooled variance is exact (assuming `ddof=1` per group).
- Min/max are exact.
- Lift, CTR, conversion rate, revenue are exact.

Whenever Value Stream claims a number is "approximate":
- Quantiles via t-digest: error ≤ 0.3% at the median, ≤ 0.05% at p99 with `k=500`.
- Quantiles via KLL: error ≤ 1.65% normalized rank with `k=200`.
- Distinct counts via CPC: report the sketch's lower/upper bounds; generated states use `lg_k=11`.
- Distinct counts via legacy/opt-in HLL: ±1.6% RSE with `lg_k=12`.
- Set operations via Theta: as above.
- ROC AUC reconstructed from digests: typically within 1e-2 of the exact value at `n ≥ 1000` per group.

These bounds drive parameter defaults and inform the warnings the engine emits when a group is too small for a sketch to be reliable (e.g. `Count < 100` for t-digest curves emits a "low-support" warning that the UI surfaces on the tile).
