# Value Stream — Chart Catalog

This doc specifies the presentation surface: every chart kind a Tile can use,
the Tile fields it requires, and how the chart consumes a metric's output. Most
charts render through Plotly. Report `table` tiles use Streamlit's native
dataframe, while the chart factory retains a Plotly table adapter for headless
and compatibility callers.

Companion docs:

- concepts/domain-model.md (Tile / Dashboard).
- reference/processors.md / reference/algorithms.md (what data shape each chart receives).

---

## 1. Tile model

Dashboard pages may author aggregate-backed controls before their tiles:

```yaml
filters:
  - field: Channel
    label: Channel
    display: primary            # primary | secondary
    scope: all_tiles            # all_tiles | compatible_tiles
    control: multiselect        # multiselect | selectbox | text
time_filter:
  default: all_time
  presets: [last_30_days, last_90_days, year_to_date, custom, all_time]
```

The validator checks filter coverage against every tile processor. Older pages
without these blocks retain an inference fallback.

For `multiselect` and `selectbox`, options are aggregate-derived suggestions,
not an exhaustive allowlist. The report runtime checks compatible `content`
tiles in authored order and takes up to 500 sorted distinct values from the
first aggregate result that contains them; it does not union results from every
tile. An empty or not-yet-ready result falls through to the next compatible
content tile, and compatible `kpi_strip` tiles are considered only if no content
tile yields values. Both controls accept a custom typed value even when the
suggestion list is empty. A `text` control remains a free-form comma-separated
input and does not perform option lookup.

```yaml
tiles:
  - id: <snake_case>
    title: <string>
    metric: <metric id>          # required
    chart: <chart kind>          # required (key in this catalog)
    # chart-specific fields below ...
    filters:                     # optional, AND-ed with dashboard filters
      channel: ["Web", "Mobile"]
    description: <string>        # optional
    placement: content           # content | kpi_strip
    scale_mode: absolute         # absolute | index_100 | percent_change
```

Tile-level filters are passed straight to the metric query as `filters`.

The Configuration Builder and catalog validator use the same chart-field
contract. A tile is rejected when it uses an unsupported key, a field with the
wrong shape, a chart that is incompatible with the metric's processor or metric
kind, or a field that the selected metric cannot expose. Raw YAML supports the
same contract as Visual mode; it is an editing escape hatch, not a way to bypass
validation.

The chart factory receives `(rows: pl.DataFrame, tile: dict, plan: PlanInfo)` and returns a `plotly.graph_objects.Figure`.

---

## 2. Common Tile fields

| Field | Type | Used by | Meaning |
|---|---|---|---|
| `x` | str | most | column name on x axis |
| `y` | str | scatter, corr, descriptive_line, experiment charts | explicit second dimension or specialist output |
| `secondary_metric` | str | combo | compatible scalar metric ID with the same kind and backing processor as `metric` |
| `x_axis_title` / `y_axis_title` | str | cartesian charts | explicit primary-axis labels |
| `y2_axis_title` | str | combo | explicit secondary-axis label; otherwise the secondary metric label is used |
| `color` | str | most | column name to map to color; treemap color comes from `metric` |
| `size` | str | scatter | column name for marker size |
| `path` | list[str] | treemap | hierarchy of group-by columns |
| `description` | str | any | plain-language context; falls back to metric description |
| `placement` | str | kpi_card | `content` or explicit `kpi_strip` placement |
| `kpi` | dict | kpi_card | comparison period, target, sparkline grain/points |
| `scale_mode` | str | line, stacked_area | `absolute`, `index_100`, or `percent_change` |
| `theta` | str | bar_polar | angle column |
| `facet_row` | str | most | column for facet rows |
| `facet_col` | str | most | column for facet columns |
| `animation_frame` | str | scatter | column for play axis |
| `animation_group` | str | scatter | column to group across frames |
| `log_x` / `log_y` | bool | line, scatter | log axis toggle |
| `reference` | number | gauge | scalar reference level for all gauges |
| `references` | dict | gauge | reference levels per (row's join-key) |
| `stages` | list[str] | funnel | ordered stage names |
| `property` | str | histogram, descriptive_* | numeric property column |
| `score` | str | descriptive_line | which score (Mean/Median/p95/…) to plot |
| `showlegend` | bool | any | Plotly `showlegend` |
| `value_format` | str | most | display format: `percent`, `integer`, `number`, or `currency` |
| `goal_line` | number / dict | cartesian charts | horizontal reference line, e.g. `{value: 0.12, label: Target}` |
| `show_trend_delta` | bool | line, bar | show latest-vs-first delta annotation |
| `sort_by` | str | bar | column used to sort bars; defaults to `y` |
| `sort_direction` | str | bar | `ascending` or `descending` |
| `top_n` | int | bar | keep the top N rows after sorting |
| `barmode` | str | bar | `group`, `stack`, `relative`, or `percent` / `stacked_percent` |
| `conditional_formatting` | list[dict] | bar, scatter | per-row colors when no explicit `color` field is set |
| `metric_output` | str | metric-owned measure charts | selected column from the metric's read-only output contract |
| `lower_output` / `upper_output` | str | interval | lower/upper confidence-bound outputs from the selected metric |

Every metric exposes its computed output contract in the Metric editor as a
read-only list. The report editor uses that same list for `metric_output`;
dimensions, time fields, and processor states are never offered in that
selector. Scalar metrics have one selectable output, normally their Metric ID.

Interval confidence-bound columns are coerced to nullable numeric values before
the renderer calculates positive and negative error lengths. A comparison group
with no eligible rows can therefore render null bounds without raising a
storage-type error. In both report editors, `metric_output`, `lower_output`,
and `upper_output` can select only outputs produced by the tile's selected
metric; processor dimensions and time fields are excluded.

---

## 3. Chart catalog

For each chart: required Tile fields, expected metric output shape, and an outline of the Plotly construction.

### 3.0 `kpi_card`

Required chart fields: none. The displayed value is always derived from the
selected metric's primary output; `value` and `y` are not configurable card
roles. Existing overrides are ignored at runtime and removed when a card is
saved through the visual or advanced editor. Converting a KPI card to another
chart also removes KPI-only `placement` and `kpi` settings. An ungrouped card
becomes part of the responsive KPI strip only when `placement: kpi_strip` is
explicit. The `kpi` mapping supports:

- `comparison`: `none` or `previous_period`;
- `comparison_period`: `day`, `week`, `month`, `quarter`, or `year`;
- `sparkline_grain`: `daily`, `weekly`, `monthly`, or omitted;
- `sparkline_points`: 2–366;
- `target`: optional numeric target.

The current scalar and comparison scalar are separate `grain: summary` queries
with identical aggregate filters. When the page has no selected dates, the
latest available aggregate period anchors the current calendar period. The
reference window has the same number of days. A card that returns zero/multiple
rows or a non-numeric value displays `n/a`; report code never invents a reducer.

### 3.1 `line`

Required: `x`. Optional: `color, facet_row, facet_col, log_x, log_y`.
The Y value is always the selected metric's primary output; neither `y` nor
`value` is a configurable role. Existing overrides are ignored and removed on
save.

Metric shape: a Polars frame with `x`, the metric output, and the group-by
columns referenced by `color/facets`.

Construction:

```python
fig = px.line(
    df.to_pandas(),
    x=tile["x"], y=tile["metric"],
    color=tile.get("color"),
    facet_row=tile.get("facet_row"),
    facet_col=tile.get("facet_col"),
    log_x=tile.get("log_x", False),
    log_y=tile.get("log_y", False),
    title=tile.get("title"),
)
fig.update_layout(showlegend=tile.get("showlegend", True))
return fig
```

If `x` is categorical (a non-temporal string column) the factory falls back to `px.bar` with `barmode="group"`.

`stacked_area` uses the same metric-owned Y contract and requires `x` plus
`color`; the selected metric supplies the stacked values.

### 3.2 `bar`

Same metric-owned Y contract as `line`, but always renders as bars. Required:
`x`. Optional: `color`, facets, and `barmode ∈ {group, stack, relative}`.

For 100% stacked bars, set `barmode: percent` or `barmode: stacked_percent`.
The chart factory renders a stacked bar chart with Plotly `barnorm="percent"`.
Bars can also be sorted and capped:

```yaml
chart: bar
x: Channel
sort_by: CTR
sort_direction: desc
top_n: 10
```

For simple conditional color rules, omit the `color` field and provide rules:

```yaml
conditional_formatting:
  - column: CTR
    operator: ">="
    value: 0.12
    color: "#2e7d32"
  - column: CTR
    operator: "<"
    value: 0.12
    color: "#c62828"
```

### 3.3 `treemap`

Required: `path: list[str]`. Color is always the selected metric's primary
output; `color` is not a configurable role.

Metric shape: aggregated to `path[0], path[1], …`. Typically the metric's
`summary` grain.

```python
fig = px.treemap(
    df.to_pandas(),
    path=[px.Constant("All")] + tile["path"],
    color=metric_value_column,
    color_continuous_scale="Viridis",
    title=tile["title"],
)
```

### 3.4 `heatmap`

Required: `x`. Optional: `y`.

The selected metric owns the cell intensity. `value` and `color` are not
selectable roles and are removed when the tile is saved through an editor.
Both axes are constrained to configured calendar grains or the metric
processor's persisted `group_by` dimensions, so aggregate states such as
`Count` cannot be selected as axes.

The one chart adapts to the configured axes:

- `x` plus `y` renders an `x` × `y` matrix. This covers ordinary comparison
  and cohort layouts.
- daily `x` without `y` renders the calendar layout.
- another one-dimensional `x` without `y` renders a one-row intensity strip.
- Intensity always comes from the explicitly selected metric output. To show a
  descriptive statistic such as `ResponseTime_p95`, select the corresponding
  metric and output instead of binding a second property or score.

```python
value = selected_metric_output
matrix = df.pivot(index=tile["y"], columns=tile["x"], values=value)
fig = px.imshow(
    matrix,
    color_continuous_scale="Viridis",
    title=tile["title"],
    aspect="auto",
)
```

The retired calendar, cohort, and descriptive heatmap aliases are rejected.
Use `chart: heatmap` with `x`, optional `y`, and the selected metric.

### 3.5 `scatter`

Required: `x, y`. Optional: `color, size, animation_frame, animation_group, facet_row, facet_col, log_x, log_y`.

```python
fig = px.scatter(
    df.to_pandas(),
    x=tile["x"], y=tile["y"],
    color=tile.get("color"),
    size=tile.get("size"),
    animation_frame=tile.get("animation_frame"),
    animation_group=tile.get("animation_group"),
    facet_row=tile.get("facet_row"),
    facet_col=tile.get("facet_col"),
    log_x=tile.get("log_x", False),
    log_y=tile.get("log_y", False),
)
```

### 3.6 `bar_polar`

Required: `theta, color`. Radius is always the selected metric's primary
output; `r` is not a configurable role.

```python
fig = px.bar_polar(
    df.to_pandas(),
    r=metric_value_column, theta=tile["theta"],
    color=tile["color"],
    title=tile["title"],
)
```

### 3.7 `gauge`

Required chart fields: none. The gauge value is always the selected metric's
primary output; `value` and `y` are not configurable roles. Existing overrides
are ignored at runtime and removed when the gauge is saved through an editor.

Optional: `reference` (scalar reference for every gauge) or `references` (dict of `key -> reference_value`). The Tile uses the row whose `(facet_row, facet_col)` value matches a key in `references` for the threshold rings.

```python
fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=row[tile["metric"]],
    delta={"reference": references.get(key, 0)},
    gauge={
        "axis": {"range": [None, max(row[tile["metric"]] * 1.5, references_max)]},
        "threshold": {"line": {"color": "red"}, "value": references.get(key, 0)},
    },
    title={"text": title},
))
```

A grid of gauges is rendered when there are multiple `(facet_row, facet_col)` keys.

### 3.8 `funnel`

Required: `stages: list[str], color`.

The chart accepts either configured Funnel-processor stage states such as
`<stage>_Count`, or numeric-distribution count rows. For categorical count
rows, set optional `x` to the stage dimension. Facet fields remain optional.

```python
fig = px.funnel(
    df.to_pandas(),
    x="Count", y="stage",
    color=tile["color"],
    title=tile["title"],
)
```

### 3.9 `boxplot`

Required: a `distribution` metric. Optional: `x, color, facet_row, facet_col`.
The metric binds one unconditioned t-digest or KLL state and therefore owns the
numeric property. There is no second Property, Y, or Value field. To change
the measured property, select its automatically persisted distribution metric.
Without `x`, the renderer produces one overall distribution box; set `x` to
compare the distribution across time or another persisted dimension.

Scalar processor outputs such as `<prop>_Count` are not offered. Boxplots are reconstructed from
`<prop>_p25, <prop>_Median, <prop>_p75, <prop>_Min, <prop>_Max` (or t-digest
quantiles) already exposed by that metric.

```python
fig = go.Figure()
for group, sub in df.partition_by(tile.get("color"), as_dict=True).items():
    fig.add_trace(go.Box(
        x=sub[tile["x"]], q1=sub[f"{prop}_p25"], median=sub[f"{prop}_Median"],
        q3=sub[f"{prop}_p75"], lowerfence=sub[f"{prop}_Min"], upperfence=sub[f"{prop}_Max"],
        name=str(group),
    ))
```

### 3.10 `histogram`

Required: `property`.

For numeric_distribution metrics, the histogram is reconstructed from the t-digest:

```python
edges, masses = digest_to_histogram(row["<property>_tdigest"], bins=30)
fig = go.Figure(go.Bar(x=(edges[:-1]+edges[1:])/2, y=masses, width=edges[1:]-edges[:-1]))
```

For CLV (entity_lifecycle metrics), the histogram is over `frequency`, `monetary_value`, etc.; the metric DSL emits already-binned data.

### 3.11 `calibration_curve`

Used with metric `Calibration` (calibration_from_digests). Metric output: per row `calibration_bin`, `calibration_proba`, `calibration_rate`.

```python
fig = go.Figure([
    go.Scatter(x=row["calibration_proba"], y=row["calibration_rate"],
               mode="lines+markers", name="Observed"),
    go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Ideal", line=dict(dash="dash")),
])
fig.update_xaxes(title="Predicted propensity", range=[0,1])
fig.update_yaxes(title="Observed rate", range=[0,1])
```

### 3.12 ML score curves

Used with `curve_from_digests` metrics. The query layer returns scalar
`roc_auc` / `average_precision` plus list-valued curve columns: `fpr`, `tpr`,
`precision`, `recall`, and `pos_fraction`.

Chart kinds:

- `roc_curve`: x = `fpr`, y = `tpr`, with an `y = x` random-model reference.
- `precision_recall_curve`: x = `recall`, y = `precision`.
- `gain_curve`: x = `sample_fraction`, y = `tpr`; `sample_fraction = pos_fraction * tpr + (1 - pos_fraction) * fpr`.
- `lift_curve`: x = `sample_fraction`, y = `gain / sample_fraction`, with an `y = 1` baseline.

```yaml
- id: roc_curve
  title: ROC Curve
  metric: MIL_ROC_AUC
  chart: roc_curve
  color: placement_type
  value_format: percent
```

```python
fig = px.line(df.explode(["fpr", "tpr"]).to_pandas(), x="fpr", y="tpr", color="placement_type")
fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
```

### 3.13 `rfm_density`

Used with metric `CLV_Summary`. Renders a 2D or 3D density plot over `(recency, frequency, monetary_value)`.

```python
# 2D variant
fig = px.density_heatmap(df.to_pandas(), x="recency", y="frequency", marginal_x="histogram", marginal_y="histogram")
# 3D variant
fig = px.density_contour(df.to_pandas(), x="recency", y="frequency", z="monetary_value")
```

### 3.14 `exposure`

Customer Exposure curve from `CLV_Summary`. The x-axis is cumulative customer count (sorted by `lifetime_value` desc); the y-axis is cumulative `lifetime_value` share.

```python
sorted_df = df.sort("lifetime_value", descending=True)
sorted_df = sorted_df.with_columns(
    cum_customers = pl.col("customers_count").cum_sum() / pl.col("customers_count").sum(),
    cum_lv        = pl.col("lifetime_value").cum_sum() / pl.col("lifetime_value").sum(),
)
fig = px.line(sorted_df.to_pandas(), x="cum_customers", y="cum_lv")
```

### 3.15 `corr`

Correlation chart for two CLV variables (typically `frequency` vs `monetary_value`).

```python
fig = px.scatter(df.to_pandas(), x=tile["x"], y=tile["y"], trendline="ols")
```

### 3.16 `model`

User-facing name: **Purchase frequency projection**. This
`entity_lifecycle` chart extrapolates each customer's observed purchase rate
over a configurable horizon (30 days by default):

```python
predicted_purchases = (frequency / max(tenure, 1)) * horizon
fig = px.scatter(df.to_pandas(), x="frequency", y="predicted_purchases", color="rfm_segment")
```

This is a simple run-rate projection. It does not fit a BG/NBD, Pareto/NBD, or
machine-learning model and must not be interpreted as a model-quality report.
The rendered axes explicitly identify historical frequency and the configured
projection horizon.

### 3.17 `descriptive_line`

Specialization for `numeric_distribution` metrics. Required Tile fields are
`x, property, score` (for example, `score: Mean`).

The chart factory pulls `<property>_<score>` from the metric output; see
reference/processors.md §4 for the column naming convention. Score names are
case-insensitive (`P95` and `p95` both resolve to the canonical `_p95` column).
Use `histogram` for a digest-backed distribution and the unified `heatmap` with
the corresponding metric output for a descriptive matrix.

### 3.18 `experiment_z_score` / `experiment_odds_ratio`

Specializations for `experiment` metrics. The Tile picks which experiment statistic to plot:

| Variant | Required | y axis | x axis |
|---|---|---|---|
| `experiment_z_score` | `y` | `ExperimentName` | `z_score` |
| `experiment_odds_ratio` | `x, y` | `ExperimentName` | one of `g_odds_ratio_stat`, `chi2_odds_ratio_stat` |

These are horizontal charts: `value_format` applies to the numeric x axis and
hover value only. The categorical experiment-name y axis is never numerically
formatted.

### 3.19 `clv_treemap`

Treemap of RFM segments by lifetime value.

Required: none (uses `rfm_segment, lifetime_value` from `CLV_Summary`).

```python
fig = px.treemap(df.to_pandas(), path=[px.Constant("All"), "rfm_segment"], values="lifetime_value")
```

### 3.20 Marketing dashboard chart set

These chart kinds cover common marketing reporting needs. Charts use Plotly
primitives except for the native Streamlit table on the Reports surface:

| Chart kind | Required fields | Primary use |
|---|---|---|
| `kpi_card` | none; value comes from `metric` | Executive scorecard metric with optional delta reference |
| `stacked_area` | `x, color`; Y comes from `metric` | Channel/campaign mix over time |
| `waterfall` | `x`; Y comes from `metric` | Single-trace contribution or change decomposition; no color grouping or facets |
| `pareto` | `x`; Y comes from `metric` | Top campaigns/offers plus cumulative share |
| `heatmap` | `x` and optional `y`; intensity comes from `metric` | Matrix, cohort, descriptive, or daily calendar intensity |
| `sankey` | ordered `path` of 2+ dimensions; value comes from `metric` | Multi-step journey/path flow between stages or channels |
| `combo` | `x, secondary_metric`; Y comes from `metric` | Bar + line dual-axis comparison of two compatible scalar metrics; use `y2_axis_title` to override the secondary-axis label |
| `interval` | `x, metric_output` plus optional `lower_output`/`upper_output` | Lift/estimate with uncertainty interval |
| `donut` | `names`; values come from `metric` | Simple share-of-total for small category sets |
| `geo_map` | `locations, value` or `lat, lon, value` | Country/region/city performance |
| `table` | optional `columns` | Native sortable and scrollable report table with optional conditional formatting; a selected `topk_items` list is expanded into rank, item, estimate, and lower/upper-bound columns |

```yaml
- id: campaign_pareto
  title: Revenue Pareto
  metric: Revenue
  chart: pareto
  x: Campaign
  top_n: 12

- id: customer_journey
  title: Customer journey
  metric: Clicks
  chart: sankey
  path: [Channel, CustomerType, Placement]
```

For Sankey, every adjacent pair in `path` becomes a link layer. Nodes are
stage-scoped, so the same label may appear independently at different steps.
Legacy `source` and `target` fields remain readable as a two-step path, but
Builder writes the canonical `path` form.

---

## 4. Recipe metadata

The Builder UI uses the following metadata table to filter chart kinds by the metric's underlying processor and to default required fields:

| Chart kind | Allowed processor kinds | Default x | Default y |
|---|---|---|---|
| `line` | binary_outcome, score_distribution, conversion, snapshot | first time-grain dim | selected metric's primary output |
| `stacked_area` | binary_outcome, score_distribution, snapshot | first time-grain dim | selected metric's primary output |
| `bar` | binary_outcome, snapshot | first non-time dim | selected metric's primary output |
| `kpi_card` | aggregate metrics | — | selected metric's primary output |
| `waterfall` | aggregate metrics | first non-time dim | selected metric's primary output |
| `pareto` | aggregate metrics | first non-time dim | selected metric's primary output |
| `treemap` | binary_outcome, score_distribution | path: processor dimensions | color: selected metric's primary output |
| `heatmap` | aggregate metrics | first time grain, else first dimension | first compatible dimension when available; intensity is the selected metric |
| `scatter` | binary_outcome, score_distribution | metric.outputs[0] | metric.outputs[1] |
| `combo` | aggregate metrics | first time-grain dim | selected metric + first compatible same-kind metric on the same processor |
| `interval` | aggregate metrics | first dim | metric.outputs[0] |
| `donut` | aggregate metrics | first dim | selected metric's primary output |
| `geo_map` | binary_outcome, score_distribution, snapshot | location dim | metric.outputs[0] |
| `table` | aggregate metrics | — | — |
| `bar_polar` | binary_outcome | theta: first dimension | radius: selected metric's primary output |
| `sankey` | aggregate metrics | ordered processor dimension path | selected metric's primary output |
| `gauge` | binary_outcome, snapshot | — | selected metric's primary output |
| `funnel` | funnel, numeric_distribution | optional first dimension | stage counts |
| `boxplot` | numeric_distribution, score_distribution | optional first time-grain dim | selected `distribution` metric (no `property` or `y`) |
| `histogram` | numeric_distribution, entity_lifecycle | — | — |
| `calibration_curve` | score_distribution (Calibration metric) | — | — |
| `roc_curve` | score_distribution (`curve_from_digests`) | — | — |
| `precision_recall_curve` | score_distribution (`curve_from_digests`) | — | — |
| `gain_curve` | score_distribution (`curve_from_digests`) | — | — |
| `lift_curve` | score_distribution (`curve_from_digests`) | — | — |
| `rfm_density` | entity_lifecycle (CLV_Summary) | recency | frequency |
| `exposure` | entity_lifecycle | — | — |
| `corr` | entity_lifecycle | frequency | monetary_value |
| `model` | entity_lifecycle | — | — |
| `descriptive_*` | numeric_distribution | varies | varies |
| `experiment_*` | binary_outcome with experiment | varies | varies |

This table is also encoded in `valuestream.charts.recipes` and queried by the Builder UI.

---

## 5. Theming

A workspace-level theme block in `dashboards.yaml`:

```yaml
theme:
  template: "valuestream"           # app-matching default; any Plotly template name
  paper_bgcolor: "#162438"           # defaults to the app chart-card surface
  plot_bgcolor: "#162438"            # defaults to the app chart-card surface
  margins: { l: 32, r: 16, t: 48, b: 32 }
  legend: { orientation: "h", y: -0.2 }
  colorway: ["#4B73F0", "#22C7F3", "#45D6A5", "#F2C14E"]
  color_continuous_scale: "Viridis"
  qualitative_palette: "Set2"
  category_colors_dark:
    Channel: { Web: "#4B73F0", Mobile: "#22C7F3" }
  category_colors_light:
    Channel: { Web: "#0072B2", Mobile: "#009E73" }
```

The application initializes a built-in `valuestream` Plotly template before
dashboard rendering. Its qualitative sequence starts with the app's royal blue,
cyan, verified green, attention amber, danger coral, and violet tokens. Its
typography uses the self-hosted Inter variable webfont at 13px for chart text
and medium-weight axis ticks, 14px SemiBold axis titles, 12px legends and hover
labels, and Inter Display SemiBold at 16px for chart titles. Axis ticks have 6px
of label spacing, axis titles have a 12px standoff, and automatic margins keep
long labels visible. Legend titles use Inter SemiBold at 13px. The font files
are served by Streamlit from the packaged `ui/static/fonts` directory, with
native system sans-serif fallbacks when a webfont cannot load. Workspaces
normally omit `theme.font`; defining it remains supported as an intentional
per-workspace override.

The template's
`paper_bgcolor` and `plot_bgcolor` match the chart-card surface (`#ffffff` in
light mode and `#162438` in dark mode), so Plotly figures blend into the
Streamlit report cards instead of rendering a separate panel. The Configuration
Builder's chart library uses the same sequence and foreground tokens. The
default dashboard theme passes these values explicitly, so a workspace or tile
that overrides only `template` still keeps the app-matched presentation.
`category_colors_dark` and `category_colors_light` keep important business
categories stable within each theme without putting low-contrast dark-theme
accents onto the white chart surface. The legacy theme-neutral
`category_colors` mapping remains supported as a fallback.

The chart factory applies the theme via `fig.update_layout(template=...)` etc. Per-tile overrides are allowed:

```yaml
tiles:
  - id: hot_metric
    chart: line
    theme:
      base: light
      template: "valuestream_light"
```

In Configuration Builder Visual mode, **Chart Theme** offers **Follow
application**, **Light Plotly**, **Dark Plotly**, and **Custom YAML**. Selecting
Light or Dark stores the corresponding `base` and `valuestream_*` template on
the tile. The Report library's **Preview theme** control switches the chart
gallery without changing saved configuration.

## 6. Presentation settings

Presentation settings are optional Tile fields. They are intentionally limited
to common BI refinements so report authors can improve clarity without adding
new chart kinds.

```yaml
tiles:
  - id: daily_ctr
    title: Daily CTR
    metric: CTR
    chart: line
    x: Day
    y: CTR
    color: Channel
    value_format: percent
    show_trend_delta: true
    goal_line:
      value: 0.12
      label: Target CTR
      color: "#475569"
```

Supported values:

- `value_format`: `percent`, `integer`, `number`, or `currency`.
- `goal_line`: a number or mapping with `value`, optional `label`, `color`,
  `dash`, and `axis`. The default axis is `y`.
- `show_trend_delta`: adds an annotation comparing the first and last plotted
  value in the chosen metric column.
- `sort_by`, `sort_direction`, `top_n`: currently apply to bar charts.
- `conditional_formatting`: list of `{column, operator, value, color}` rules.
  Operators are `>`, `>=`, `<`, `<=`, `==`, and `!=`. The first matching rule
  wins. Conditional colors apply to bar and scatter charts when the tile does
  not already use a `color` dimension.
- `scale_mode`: `index_100` divides each series by its first non-zero value and
  multiplies by 100; `percent_change` subtracts one after the same partitioned
  normalization. Partitions are defined by color and facet dimensions. A zero
  or empty baseline yields null display values rather than infinity.
- `labels`, metric `display.label`, and metric `display.unit` resolve axis,
  legend, hover, KPI, and table labels centrally. Tile overrides win.
- Report table tiles render with native `st.dataframe`, use the available tile
  width, size automatically up to ten visible rows, and provide native sorting,
  column resizing, scrolling, localized numeric formatting, and row highlighting
  from `conditional_formatting`. A single selected `topk_items` output renders as
  one ranked row per frequent item, with its estimate and lower/upper bounds,
  rather than as one comma-separated text cell. Their action menu exports the
  displayed rows as CSV; image and Plotly HTML exports apply only to charts.
- The headless `render_chart` compatibility path still returns a content-sized
  Plotly table figure for `chart: table` callers outside the Reports surface.
- `theme.category_colors` maps dimension values to stable colors across result
  order, filters, and reruns. Conditional formatting and experiment-role colors
  retain precedence.

---

## 7. Streaming-specific concerns

For Tiles whose metric covers a long time range, the chart factory may receive a frame too large to serialize to Plotly cleanly. The factory enforces a soft cap (`max_points = 50_000`) and:

- downsamples line plots via LTTB (largest-triangle-three-buckets) if `len(rows) > max_points`,
- raises a `RowLimitExceeded` warning and renders the cap if downsampling is not supported (`scatter` with `animation_frame`).

The downsampling implementation is in `valuestream.charts.lttb`.

---

## 8. Accessibility

- Color choices respect color-blind-safe defaults (`Set2`/`Viridis`).
- All charts include a `title` and `description` (used as `hovertemplate` and screen-reader summary).
- Number formatting respects locale; defaults to en-US with thousands separators.

---

## 9. Adding a new chart kind

1. Add a section to this doc with required fields and a Plotly outline.
2. Add the kind to `valuestream.charts.recipes`.
3. Implement `valuestream.charts.kinds.<kind>.render(rows, tile) -> Figure`.
4. Add a unit test with a synthetic frame and a saved PNG snapshot for visual regression (Plotly figure JSON also accepted).
5. Update reference/chart-catalog.md (§4 table) so the Builder UI offers the new kind.

The chart layer is intentionally decoupled from the storage and query layers: a tile is just `(metric, chart, fields)` and the render path is pure given those plus the metric's output.
