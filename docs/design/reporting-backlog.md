# Reporting Usability Backlog

This backlog implements the selected reporting improvements:

1. correct KPI semantics;
2. simpler, transparent report filters;
3. more readable analytical presentation; and
4. decision-oriented metrics supported by the existing aggregate state.

The work remains aggregate-first. Reports, Configuration Builder, and AI
Configuration Studio must read and write the same YAML behavior, and every
runtime query must continue through the aggregate/query layer.

## Scope and compatibility

- New catalog fields are optional and backward-compatible at load time.
- Bundled catalogs are migrated in place; existing user catalogs keep safe
  fallbacks and receive validation guidance where an automatic migration would
  require guessing business semantics.
- `dashboards.yaml` remains the source of report, page-filter, KPI, and visual
  behavior. `metrics.yaml` remains the source of metric calculation and display
  metadata.
- Configuration Builder and AI Configuration Studio are part of every catalog
  property story, not follow-up work.
- Runtime customization remains session-local unless it is explicitly applied
  through an authoring surface and validated as YAML.

## Proposed catalog additions

The detailed field shapes must be finalized in the specification story before
implementation. The intended contract is:

```yaml
# metrics.yaml
metrics:
  VS_Conversion_Rate:
    source: ih_outcome_funnel
    kind: formula
    description: Conversions divided by impressions.
    display:
      label: Conversion rate
      unit: percent
      value_format: percent
      direction: higher_is_better

# dashboards.yaml
theme:
  category_colors:
    Channel:
      Web: "#2563EB"
      Mobile: "#14B8A6"

dashboards:
  - id: ih_value_stream_overview
    title: Interaction History Value Stream Overview
    layout: tabs
    pages:
      - id: executive_summary
        title: Executive Summary
        time_filter:
          default: last_30_days
          presets: [last_30_days, last_90_days, year_to_date, all_time]
        filters:
          - field: Channel
            label: Channel
            display: primary
            scope: all_tiles
          - field: CustomerSegment
            label: Customer segment
            display: secondary
            scope: compatible_tiles
        tiles:
          - id: total_interactions
            title: Interactions
            metric: VS_Interactions
            chart: kpi_card
            value: VS_Interactions
            placement: kpi_strip
            kpi:
              comparison: previous_period
              comparison_period: month
              sparkline_grain: daily
              target: 500000
            scale_mode: absolute
```

## Delivery order

| Increment | Outcome | Included backlog |
|---|---|---|
| A | Shared catalog contract and lossless authoring | RPT-001–RPT-006 |
| B | Trustworthy KPI strip and migrated reports | RPT-101–RPT-106 |
| C | Compact filters with explicit tile coverage | RPT-201–RPT-207 |
| D | Consistent labels, units, colors, and chart alternatives | RPT-301–RPT-307 |
| E | Decision metrics from existing aggregates | RPT-401–RPT-408 |
| F | End-to-end release qualification | RPT-901–RPT-904 |

## Increment A — Catalog and authoring foundation

| ID | Priority | Size | Backlog item | Dependencies |
|---|---:|---:|---|---|
| RPT-001 | P0 | M | Specify `MetricDisplaySpec`, `PageFilterSpec`, `TimeFilterSpec`, `KpiSpec`, tile `placement`/`scale_mode`, and theme `category_colors` in `concepts/domain-model.md`, `docs/design/replacement-design.md`, and `reference/chart-catalog.md`. Define defaults, invalid combinations, and whether each field affects query or presentation behavior. | — |
| RPT-002 | P0 | M | Add the typed Pydantic properties and cross-catalog validation. Regenerate `schemas/metrics.json`, `schemas/dashboards.json`, and `schemas/catalog.json`. | RPT-001 |
| RPT-003 | P0 | M | Add shared builder helpers for metric display metadata, page settings, and KPI/presentation tile settings. Ensure YAML serialization omits defaults but round-trips every explicit property. | RPT-002 |
| RPT-004 | P0 | M | Add an in-place page-settings writer for Configuration Builder and a full-dashboard-section writer for AI Studio. The current AI apply path reconstructs dashboards tile by tile and would otherwise lose theme, layout, filters, time-filter, and other page properties. Writes must remain atomic and validate before success is reported. | RPT-002 |
| RPT-005 | P0 | M | Update the AI catalog schema dictionary, generation prompts, repair prompts, response examples, and deterministic report generator so the model can emit only valid new properties. Explicitly prohibit inferred KPI reducers and unsupported filter scopes. | RPT-001, RPT-002 |
| RPT-006 | P0 | M | Add authoring round-trip tests: hand-authored YAML → model → Configuration Builder edit → YAML, and AI draft → review → apply → reload. Assert that no dashboard-, page-, tile-, theme-, or metric-display property is dropped. | RPT-003–RPT-005 |

### Increment A acceptance

- All new properties validate through the same catalog model and JSON schemas.
- Configuration Builder and AI Studio can both create, edit, preview, apply,
  export, and reload the properties without raw-YAML-only workarounds.
- Applying an AI draft preserves dashboard theme, layout, page settings, and
  tile settings exactly.
- Invalid AI output is rejected during draft validation before workspace
  mutation.

## Increment B — Correct KPI semantics

| ID | Priority | Size | Backlog item | Dependencies |
|---|---:|---:|---|---|
| RPT-101 | P0 | M | Replace derived KPI discovery in Reports with explicit `kpi_card` tiles using `placement: kpi_strip`. Remove last-value/mean/sum fallback summarization. | RPT-002 |
| RPT-102 | P0 | L | Implement a cached KPI query bundle through the query layer: scalar summary value, equal-length previous-period summary, and optional evenly spaced sparkline series. Define complete-period behavior when no explicit date range is selected. | RPT-101 |
| RPT-103 | P0 | M | Validate that a KPI value query returns exactly one row and the configured numeric value column. Reject grouped KPIs and ambiguous reducers rather than guessing. | RPT-101 |
| RPT-104 | P0 | M | Render responsive native metric cards with labeled comparison period, target variance, method help, and optional sparkline. Keep each independent card query cached and fragment-scoped. | RPT-102, RPT-103 |
| RPT-105 | P0 | M | Add Configuration Builder tile controls for KPI placement, comparison, comparison period, target, sparkline grain, value format, and preview. Update AI Studio prompt generation, deterministic generation, review inventory, and compact tile editing for the same fields. | RPT-003–RPT-005 |
| RPT-106 | P0 | M | Migrate bundled Executive Summary and experiment KPI tiles in place. Remove the unsafe duplicate-axis combo and any catalog `summary_aggregation` usage. Add a validation warning for external catalogs that previously depended on implicit KPI derivation. | RPT-101–RPT-105 |

### Increment B acceptance

- A KPI value equals a direct `grain="summary"`, ungrouped metric query under
  identical filters and dates.
- HLL and t-digest metrics are merged by their processor/query implementation;
  report code never averages or sums their rendered rows.
- Every comparison names its current and reference periods.
- Bundled dashboards contain no implicitly derived KPI cards.

## Increment C — Simpler, transparent filters

| ID | Priority | Size | Backlog item | Dependencies |
|---|---:|---:|---|---|
| RPT-201 | P0 | M | Replace the page filter union helper with a filter-capability matrix containing field, supporting tile IDs, unsupported tile IDs, and effective scope. Preserve an inference fallback for catalogs without explicit page filters. | RPT-002 |
| RPT-202 | P0 | M | Add validation for `all_tiles` and `compatible_tiles`. `all_tiles` is invalid unless every tile processor persists the field; `compatible_tiles` must have at least one supported tile. | RPT-201 |
| RPT-203 | P0 | L | Redesign report controls: segmented date presets, no more than three primary business filters, secondary/high-cardinality filters under “More filters,” active filter badges, and one clear-all action. | RPT-201 |
| RPT-204 | P0 | M | Eliminate silent filter skipping. Show partial coverage on the active filter and an explicit “not applied” state on unsupported tiles. Keep tile queries aggregate-only. | RPT-201–RPT-203 |
| RPT-205 | P1 | M | Cache aggregate-backed option queries by workspace, field, catalog/computation signature, and eligible processors. Add a high-cardinality control that does not load an unbounded option list. | RPT-201 |
| RPT-206 | P0 | M | Add Configuration Builder page controls for filter order, label, primary/secondary display, control type, scope, time presets, and default time selection. Add preview coverage before apply. | RPT-003, RPT-004, RPT-202 |
| RPT-207 | P0 | M | Update AI Studio schema/prompt rules, deterministic page-filter selection, report review, full-dashboard apply, and repair feedback for invalid filter coverage. Migrate bundled page filters in place. | RPT-004, RPT-005, RPT-202 |

### Increment C acceptance

- The Executive Summary initially shows at most four primary controls including
  time.
- No active filter is silently ignored.
- Filter coverage remains correct after changing chart type, metric, or tile
  processor in either authoring studio.
- Filter option queries read only aggregate dimensions and invalidate when
  catalog or aggregate signatures change.

## Increment D — Analytical readability

| ID | Priority | Size | Backlog item | Dependencies |
|---|---:|---:|---|---|
| RPT-301 | P1 | M | Add a central presentation resolver that combines metric display metadata, tile overrides, and dashboard theme without changing metric calculation semantics. | RPT-002 |
| RPT-302 | P1 | L | Apply friendly labels and units consistently to axis titles, legend titles, hover text, KPI cards, and table columns across all chart recipes. Fall back to humanized identifiers for older catalogs. | RPT-301 |
| RPT-303 | P1 | M | Apply semantic category colors centrally and stably across charts, row order, filters, and reruns. Do not override explicit conditional or experiment-role colors. | RPT-301 |
| RPT-304 | P1 | L | Implement presentation-only `absolute`, `index_100`, and `percent_change` scale modes over aggregate result rows. Partition normalization by color and facet dimensions and handle zero/empty baselines explicitly. | RPT-301 |
| RPT-305 | P1 | M | Add visible metric help and a “View as table” alternative to every report tile. Use metric description when the tile has no description and preserve query provenance in Inspect. | RPT-301 |
| RPT-306 | P1 | M | Add Configuration Builder editors for metric label/unit/format/direction, tile description/scale mode, and theme category colors. Add equivalent AI Studio metric editing, prompt rules, deterministic defaults, report review columns, and repair validation. | RPT-003–RPT-005, RPT-301 |
| RPT-307 | P1 | M | Migrate all bundled reports in place: replace generated metric IDs in visible text, fix titles, declare units/formats, define shared colors, remove unsuitable donuts/dual axes, and split materially different scales into separate or faceted views. | RPT-302–RPT-306 |

### Increment D acceptance

- The same category has the same color on every page regardless of result order.
- Executive-facing axes and tooltips contain no generated metric identifiers.
- Counts, percentages, durations, and statistical values use consistent units
  and formats.
- Every chart has a meaningful description and a readable table alternative.

## Increment E — Decision-oriented metrics

| ID | Priority | Size | Backlog item | Dependencies |
|---|---:|---:|---|---|
| RPT-401 | P1 | S | Add click-through, impression-to-conversion, and click-to-conversion formula metrics from existing `ih_outcome_funnel` aggregate states. Add descriptions and display metadata. | RPT-002 |
| RPT-402 | P1 | M | Expose existing prior-period query support through tile query configuration and use it for period change in trends and KPIs. Reuse the KPI target contract for target attainment and variance. | RPT-102, RPT-301 |
| RPT-403 | P1 | L | Extend `variant_compare` with clearly named test/control rates, absolute rate difference, relative lift, sample sizes, and a documented 95% confidence interval for the absolute difference. Preserve current output compatibility. | RPT-001 |
| RPT-404 | P1 | M | Add an interval report for effect estimate plus confidence interval and update experiment KPI cards with sample size, effect direction, and p-value context. | RPT-403 |
| RPT-405 | P1 | M | Add derived metric-quality metadata for HLL and t-digest metrics, including method and configured approximation guidance. Display it on cards/tiles and retain full query provenance in Inspect. | RPT-301 |
| RPT-406 | P2 | L | Specify and implement distribution drift from persisted score-distribution digests using current versus configured reference periods. Keep the algorithm deterministic and query-layer-only. | RPT-001, RPT-405 |
| RPT-407 | P2 | L | Specify and implement deterministic anomaly scoring over aggregate time series with configured grain, window, minimum history, method, and threshold. Return score, expected range, and flag. | RPT-001, RPT-402 |
| RPT-408 | P1/P2 | L | For every new or extended metric capability, update Configuration Builder metric-kind options/forms/output discovery and AI Studio metric dictionaries, generation/repair prompts, deterministic generation, review, and apply tests. Add the new reports to bundled catalogs. | RPT-401–RPT-407 |

### Increment E acceptance

- Funnel rates match formulas over persisted funnel counts.
- Experiment confidence intervals match documented reference cases.
- Period, target, approximation, drift, and anomaly calculations never access
  raw event rows.
- Both authoring studios can create and round-trip every newly introduced
  metric or report property.

## Increment F — Release qualification

| ID | Priority | Size | Backlog item | Dependencies |
|---|---:|---:|---|---|
| RPT-901 | P0 | M | Expand config/schema tests, builder helper tests, Configuration Builder widget tests, AI prompt/parse/repair/apply tests, query tests, chart tests, and Reports state tests. | All applicable stories |
| RPT-902 | P0 | M | Add end-to-end catalog round-trip fixtures for `examples/demo`, `examples/ih_test`, and `examples/workspace_bol`. Validate and render every dashboard page after in-place migration. | RPT-106, RPT-207, RPT-307, RPT-408 |
| RPT-903 | P0 | M | Add visual and accessibility smoke coverage for Executive Summary at desktop and narrow widths, including KPI wrapping, filter overflow, descriptions, table alternatives, and color consistency. | RPT-104, RPT-203, RPT-305 |
| RPT-904 | P0 | S | Update architecture, domain, chart, algorithm, Builder tutorial, AI Studio workflow, and user-guide documentation in the same changes as behavior. Record upgrade guidance for optional properties and removed implicit KPI summaries. | All applicable stories |

## Definition of done for each backlog item

An item is complete only when:

1. runtime behavior remains aggregate-first and uses the query layer;
2. the YAML model, schema, and validation rules agree;
3. Configuration Builder and AI Studio support any property the item adds or
   changes;
4. existing bundled reports are updated in place when their behavior or visual
   contract changes;
5. positive, invalid, backward-compatibility, and round-trip tests pass; and
6. the relevant specification and user documentation are updated.

