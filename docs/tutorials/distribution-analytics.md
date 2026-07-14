# Distribution Analytics

This tutorial adds descriptive numeric analytics and model-score analytics on
top of [Getting started](getting-started.md). Complete that tutorial first so
`examples/demo` has ingested aggregates.

The demo workspace configures two distribution processors:

- `ih_response_time` (`numeric_distribution`) — t-digest of decision-to-outcome
  response times.
- `ih_propensity_scores` (`score_distribution`) — t-digests of propensity,
  final propensity, priority, and rank, split by positive/negative outcome for
  model-quality curves.

## Query Numeric Distributions

95th-percentile response time by channel:

```sh
uv run valuestream query examples/demo VS_ResponseTime_P95 --by Channel --grain Day
```

## Query Score Distributions

Median and P90 final propensity:

```sh
uv run valuestream query examples/demo VS_FinalPropensity_Median --by Channel --grain Day
uv run valuestream query examples/demo VS_FinalPropensity_P90 --by Channel --grain Day
```

Median priority:

```sh
uv run valuestream query examples/demo VS_Priority_Median --by Channel --grain Day
```

## Query Model-Quality Metrics

ROC AUC reconstructed from the positive/negative propensity t-digests:

```sh
uv run valuestream query examples/demo ih_propensity_scores_roc_auc --by Channel --grain Day
```

## How to Read These Numbers

Quantile metrics are computed from mergeable t-digest sketches, not raw rows,
so they are approximate with bounded error; ROC AUC, average precision, and
calibration are reconstructed from the score digests. Reports mark these
values with approximation badges.

- [Algorithms](../reference/algorithms.md) — sketch formulas, error behavior,
  and curve reconstruction.
- [Processors](../reference/processors.md) — `numeric_distribution` and
  `score_distribution` state layouts.
