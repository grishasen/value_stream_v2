# ADR 0005 — Unified Metric-Owned Heatmaps

**Status:** Accepted (2026-07-24)

## Context

The report catalog exposed ordinary, cohort, calendar, and descriptive
heatmaps as separate chart kinds. Their editors also exposed `color` or
`value`, even though a report tile is already bound to a metric. This allowed
authors to select an unrelated aggregate state as intensity or as an axis and
made one visual family appear to be four unrelated charts.

## Decision

New report authoring exposes one adaptive `heatmap` kind. The selected metric
owns cell intensity. `x` and optional `y` can reference only configured time
grains or persisted processor dimensions:

- `x` and `y` render a matrix, including cohort layouts;
- daily `x` without `y` renders a calendar;
- another `x` without `y` renders a one-row intensity strip;
- numeric-distribution heatmaps may select `property` and `score`.

The legacy `calendar_heatmap`, `cohort_heatmap`, and
`descriptive_heatmap` kinds remain loadable and renderable. Visual or Advanced
editing normalizes them to `heatmap`; Raw YAML remains available for explicit
legacy maintenance.

## Consequences

- Authors cannot redirect intensity to a dimension or unrelated state.
- Heatmap axes stay within the aggregate query contract.
- The Builder and Reports Advanced mode have one heatmap choice and one field
  model.
- Existing catalogs remain readable without a forced bulk migration.
- Runtime compatibility aliases must remain covered until a future catalog
  version explicitly removes them.
