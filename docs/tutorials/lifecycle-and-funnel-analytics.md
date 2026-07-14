# Funnel, Experiment, and Lifecycle Analytics

This tutorial covers the stateful analytics families using the demo workspace.
Complete [Getting started](getting-started.md) first so `examples/demo` has
ingested aggregates.

## Funnels

The demo workspace configures `ih_outcome_funnel` (`funnel`) with the stages
Impression → Clicked → Conversion. Query the stage dropoffs:

```sh
uv run valuestream query examples/demo VS_Impression_to_Click_Dropoff --by Channel --grain Day
uv run valuestream query examples/demo VS_Click_to_Conversion_Dropoff --by Channel --grain Day
```

## Experiment Monitoring

The engagement processor persists the `ModelControlGroup` dimension, so test
and control variants can be compared statistically:

```sh
uv run valuestream query examples/demo VS_ModelControl_Engagement_Compare --by Channel --grain Day
uv run valuestream query examples/demo VS_ModelControl_Engagement_Z_Test --by Channel --grain Day
```

`variant_compare` returns per-variant rates, lift, and standard error;
`contingency_test` returns the experiment p-value. The formulas are in
[Algorithms](../reference/algorithms.md).

## Lifecycle, Sets, and Snapshots

Three further stateful processor kinds are available but not configured in the
demo catalog:

| Kind | Use |
|---|---|
| `entity_lifecycle` | CLV, RFM segments, recency/frequency/monetary summaries |
| `entity_set` | Approximate set overlap and cohort comparisons (theta sketches) |
| `snapshot` | Periodic aggregate state, e.g. active holdings per period |

To try them, add a processor of the relevant kind to your workspace's
`catalog/processors.yaml`, bind metrics to its state, validate, and re-run the
source. See [Processors](../reference/processors.md) for each kind's
configuration and state layout, and the
[configuration guide](../guides/configuration/workspaces-and-catalog.md) for
the edit-validate-rerun workflow.
