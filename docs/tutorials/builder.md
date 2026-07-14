# Configuration Builder Tutorial

The Configuration Builder is the Streamlit workflow for reviewing, editing, and
generating catalog YAML. It helps users create or update sources, processors,
metrics, report tiles, and exports without editing every file by hand.

For the full application workflow, see the [running reports guide](../guides/users/running-reports.md).

## Before You Start

Validate the workspace:

```sh
uv run valuestream validate examples/demo
```

Start the UI:

```sh
uv run valuestream serve examples/demo --port 8501 --headless
```

Open Configuration Builder from the sidebar.

The Builder uses one compact step selector with Previous and Next actions. The
progress bar shows the current step and its Define, Model, Report, Review, or
Export phase; validation summaries remain in Workspace Health instead of being
repeated above every editor.

## Recommended Flow

1. Select Workspace Health and resolve validation errors first.
2. Review the source and dimension setup.
3. Add or edit a metric.
4. Preview the generated YAML.
5. Add or edit a report tile.
6. Preview the tile output when aggregate data exists.
7. Review Chat With Data metric readiness and chat-only prompt guidance.
8. Update workspace defaults and dashboard theme settings.
9. Save or export the catalog YAML.
10. Run validation again.
11. Re-run affected sources when processor changes require new aggregate state.

## Raw YAML Fallback

Use Raw YAML mode when you need full control over a metric, tile, or dashboard
theme. It is useful for small changes such as title edits, chart settings, or
formula tweaks, but the same validation rule applies: save, validate, then rerun
affected data when needed.

## Chat Guidance

The Chat Review step edits chat-only settings in `ai.yaml`. Use the agent prompt
for workspace business context, and use dataset/metric description rows to
clarify terminology for the LLM planner without changing catalog metric
definitions or aggregate behavior.

## Metric Changes

| Change | Usually requires raw replay? |
|---|---|
| Add formula metric from existing state | No |
| Change formula expression only | No, if it uses existing state |
| Add approximate distinct metric from an existing CPC/HLL/Theta state | No |
| Add metric requiring a missing state | Yes, add state to processor and reprocess |

The Metric Workflow separates creation from maintenance. Choose **Create
Metric**, then choose **From Recipe Library** for a reviewed KPI definition or
**From Scratch** for the direct editor. The scratch path starts blank: choose a
processor, choose a kind, then enter a display name. The editor generates the
metric ID from a lower-case 20-character slug prefix plus a random 8-byte
suffix.

After either creation path writes a metric, the Builder performs a full catalog
reload, switches to **Edit Existing Metric**, and opens the saved metric by its
processor, kind, and ID. This confirms that it is present in
`catalog/metrics.yaml`. Recipe creation first requires **Review changes**, which
shows the exact generated processor, metric, and report YAML patches. A write
or post-write validation failure rolls the entire recipe transaction back.
When a new processor state was added, the success message names the affected
source and links to Data Load so the state can be materialized. Existing
metrics keep their IDs during guided edits. Use Raw YAML mode when you need to
rename or override a metric identifier directly.

Open **Report presentation** to set the friendly label, unit, default number
format, and whether higher or lower values are favorable. These fields affect
Reports only; adding or changing them does not require reprocessing.

For ROC, average precision, and calibration metrics, choose the backing digest
property when the processor has positive/negative t-digest pairs. The editor
then fills the positive and negative digest states for that property, while
still allowing manual state selection for custom state names.

## Tile Changes

Tile changes affect report rendering. They do not change aggregate storage
unless the tile requires a metric or dimension that is not already available.

In the Tile Editor, choose the target dashboard from existing dashboard names
and choose the page from the selected dashboard's page names, or choose the
new-dashboard/new-page option and enter a name. The visual editor generates
dashboard, page, and new tile IDs from the displayed names using a lower-case
20-character slug prefix plus a random 8-byte suffix. Existing opened tiles keep
their existing IDs so edits replace the selected tile. Use Raw YAML mode only
when you need to inspect or override the generated identifier.

Use **Page filters and time range** to author filter order, label,
primary/secondary placement, control type, coverage, available date presets,
and the default date preset. The editor updates the selected page in place and
preserves its tiles, dashboard layout, and theme. An `all_tiles` filter must be
persisted by every tile processor; use `compatible_tiles` when coverage is
intentionally partial.

In **Chart Settings**, a `kpi_card` can be placed in the KPI strip and given an
explicit previous-period comparison, target, sparkline grain, and point count.
Line and stacked-area tiles can use absolute, index-100, or percent-change
scales. Tile descriptions and value-format overrides are available for every
compatible chart. A KPI is never inferred from another chart or reduced with a
guessed mean/sum/last-value rule.

AI Configuration Studio uses the same generated identifier rule and catalog
contract when it creates or refreshes starter reports. Metric Review exposes
the same display metadata; Report Settings exposes page controls and selected
tile KPI/scale settings. Applying a draft validates and writes the complete
workspace configuration transactionally, so theme, layout, filters, time
presets, tile properties, and AI guidance survive the round trip together. Raw
dashboards YAML remains the explicit
fallback for less common chart settings.

## Related Docs

- [Business Functionality](../concepts/business-functionality.md)
- [Technical Overview](../concepts/architecture-overview.md)
- [Expression DSL](../reference/expression-dsl.md)
- [Chart Catalog](../reference/chart-catalog.md)
