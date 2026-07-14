# Value Stream — Chart Catalog

This doc specifies the presentation surface: every chart kind a Tile can use, the Tile fields it requires, the Plotly API calls used to render it, and how the chart consumes a metric's output. Implementers build the chart-factory registry from this doc directly.

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

```yaml
tiles:
  - id: <snake_case>
    title: <string>
    metric: <metric id>          # required
    chart: <chart kind>          # required (key in this catalog)
    # chart-specific fields below ...
    time_range: { last: 30d }    # optional, overrides dashboard default
    filters:                     # optional, AND-ed with dashboard filters
      channel: ["Web", "Mobile"]
    description: <string>        # optional
    placement: content           # content | kpi_strip
    scale_mode: absolute         # absolute | index_100 | percent_change
```

Tile-level filters are passed straight to the metric query as `filters`.

The chart factory receives `(rows: pl.DataFrame, tile: dict, plan: PlanInfo)` and returns a `plotly.graph_objects.Figure`.

---

## 2. Common Tile fields

| Field | Type | Used by | Meaning |
|---|---|---|---|
| `x` | str | most | column name on x axis |
| `y` | str / list[str] | most | column name(s) on y axis |
| `color` | str | most | column name to map to color |
| `size` | str | scatter | column name for marker size |
| `path` | list[str] | treemap | hierarchy of group-by columns |
| `value` | str | gauge | scalar column to display |
| `description` | str | any | plain-language context; falls back to metric description |
| `placement` | str | kpi_card | `content` or explicit `kpi_strip` placement |
| `kpi` | dict | kpi_card | comparison period, target, sparkline grain/points |
| `scale_mode` | str | line, stacked_area | `absolute`, `index_100`, or `percent_change` |
| `r` | str | bar_polar | radius column |
| `theta` | str | bar_polar | angle column |
| `facet_row` | str | most | column for facet rows |
| `facet_col` | str | most | column for facet columns |
| `facets` | dict | most | shorthand: `{row: x, col: y}` |
| `animation_frame` | str | scatter | column for play axis |
| `animation_group` | str | scatter | column to group across frames |
| `log_x` / `log_y` | bool | line, scatter | log axis toggle |
| `reference` | number | gauge | scalar reference level for all gauges |
| `references` | dict | gauge | reference levels per (row's join-key) |
| `stages` | list[str] | funnel | ordered stage names |
| `property` | str | descriptive_* | numeric property column |
| `score` | str | descriptive_line | which score (Mean/Median/p95/…) to plot |
| `showlegend` | bool | any | Plotly `showlegend` |
| `value_format` | str | most | display format: `percent`, `integer`, `number`, or `currency` |
| `goal_line` | number / dict | cartesian charts | horizontal reference line, e.g. `{value: 0.12, label: Target}` |
| `show_trend_delta` | bool | line, bar | show latest-vs-first delta annotation |
| `sort_by` | str | bar | column used to sort bars; defaults to `y` |
| `sort_direction` | str | bar | `asc` or `desc` |
| `top_n` | int | bar | keep the top N rows after sorting |
| `barmode` | str | bar | `group`, `stack`, `relative`, or `percent` / `stacked_percent` |
| `conditional_formatting` | list[dict] | bar, scatter | per-row colors when no explicit `color` field is set |
| `error_y_lower` / `error_y_upper` | str | interval | absolute lower/upper confidence-bound columns |

---

## 3. Chart catalog

For each chart: required Tile fields, expected metric output shape, and an outline of the Plotly construction.

### 3.0 `kpi_card`

Required: `value`. An ungrouped card becomes part of the responsive KPI strip
only when `placement: kpi_strip` is explicit. The `kpi` mapping supports:

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

Required: `x, y`. Optional: `color, facet_row, facet_col, log_x, log_y`.

Metric shape: a Polars frame with `x, y` and the group-by columns referenced by `color/facets`.

Construction:

```python
fig = px.line(
    df.to_pandas(),
    x=tile["x"], y=tile["y"],
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

### 3.2 `bar`

Same as `line` but always renders as bars. Optional: `barmode ∈ {group, stack, relative}`.

For 100% stacked bars, set `barmode: percent` or `barmode: stacked_percent`.
The chart factory renders a stacked bar chart with Plotly `barnorm="percent"`.
Bars can also be sorted and capped:

```yaml
chart: bar
x: Channel
y: CTR
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

Required: `path: list[str], color: str`.

Metric shape: aggregated to `path[0], path[1], …, color`. Typically the metric's `summary` grain.

```python
fig = px.treemap(
    df.to_pandas(),
    path=[px.Constant("All")] + tile["path"],
    color=tile["color"],
    color_continuous_scale="Viridis",
    title=tile["title"],
)
```

### 3.4 `heatmap`

Required: `x, y, color`. The frame is pivoted: `x` × `y` matrix with `color` as the value.

```python
matrix = df.pivot(index=tile["y"], columns=tile["x"], values=tile["color"])
fig = px.imshow(
    matrix,
    color_continuous_scale="Viridis",
    title=tile["title"],
    aspect="auto",
)
```

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

Required: `r, theta, color`.

```python
fig = px.bar_polar(
    df.to_pandas(),
    r=tile["r"], theta=tile["theta"],
    color=tile["color"],
    title=tile["title"],
)
```

### 3.7 `gauge`

Required: `value` (a scalar column from the metric).

Optional: `reference` (scalar reference for every gauge) or `references` (dict of `key -> reference_value`). The Tile uses the row whose `(facet_row, facet_col)` value matches a key in `references` for the threshold rings.

```python
fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=row[tile["value"]],
    delta={"reference": references.get(key, 0)},
    gauge={
        "axis": {"range": [None, max(row[tile["value"]] * 1.5, references_max)]},
        "threshold": {"line": {"color": "red"}, "value": references.get(key, 0)},
    },
    title={"text": title},
))
```

A grid of gauges is rendered when there are multiple `(facet_row, facet_col)` keys.

### 3.8 `funnel`

Required: `stages: list[str], color`.

Metric shape: one row per `(stage_name, color, …)` with `Count`.

```python
fig = px.funnel(
    df.to_pandas(),
    x="Count", y="stage",
    color=tile["color"],
    title=tile["title"],
)
```

### 3.9 `boxplot`

Required: `x, y`. Optional: `color, facet_row, facet_col`.

Boxplots are reconstructed from `<prop>_p25, <prop>_Median, <prop>_p75, <prop>_Min, <prop>_Max` (or t-digest quantiles) — the metric output already exposes those.

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

CLV prediction overlay using `lifetimes` (or equivalent) BG/NBD or Pareto/NBD models fit at query time.

```python
import lifetimes
bgnbd = lifetimes.BetaGeoFitter()
bgnbd.fit(df["frequency"], df["recency"], df["tenure"])
df = df.with_columns(predicted_purchases = bgnbd.conditional_expected_number_of_purchases_up_to_time(
    horizon, df["frequency"], df["recency"], df["tenure"]))
fig = px.scatter(df.to_pandas(), x="frequency", y="predicted_purchases", color="rfm_segment")
```

### 3.17 `descriptive_line` / `descriptive_boxplot` / `descriptive_histogram` / `descriptive_heatmap` / `descriptive_funnel`

Specializations for `numeric_distribution` metrics. Required Tile fields differ:

| Variant | Required |
|---|---|
| `descriptive_line` | `x, property, score` (e.g. `score: Mean`) |
| `descriptive_boxplot` | `x, property` |
| `descriptive_histogram` | `property` |
| `descriptive_heatmap` | `x, y, property, score` |
| `descriptive_funnel` | `x, color, stages` |

The chart factory pulls `<property>_<score>` from the metric output; see reference/processors.md §4 for the column naming convention.

### 3.18 `experiment_z_score` / `experiment_odds_ratio`

Specializations for `experiment` metrics. The Tile picks which experiment statistic to plot:

| Variant | Required | y axis | x axis |
|---|---|---|---|
| `experiment_z_score` | `y` | `ExperimentName` | `z_score` |
| `experiment_odds_ratio` | `x, y` | `ExperimentName` | one of `g_odds_ratio_stat`, `chi2_odds_ratio_stat` |

### 3.19 `clv_treemap`

Treemap of RFM segments by lifetime value.

Required: none (uses `rfm_segment, lifetime_value` from `CLV_Summary`).

```python
fig = px.treemap(df.to_pandas(), path=[px.Constant("All"), "rfm_segment"], values="lifetime_value")
```

### 3.20 Marketing dashboard chart set

These chart kinds cover common marketing reporting needs and are implemented
with Plotly primitives:

| Chart kind | Required fields | Primary use |
|---|---|---|
| `kpi_card` | `value` | Executive scorecard metric with optional delta reference |
| `stacked_area` | `x, y, color` | Channel/campaign mix over time |
| `waterfall` | `x, y` | Contribution or change decomposition |
| `pareto` | `x, y` | Top campaigns/offers plus cumulative share |
| `cohort_heatmap` | `x, y, color` | Cohort or retention matrix |
| `sankey` | `source, target, value` | Journey/path flow between stages or channels |
| `combo` | `x, y, y2` | Bar + line dual-axis comparisons, e.g. spend vs revenue |
| `interval` | `x, y` plus optional `error_y` | Lift/estimate with uncertainty interval |
| `donut` | `names, values` | Simple share-of-total for small category sets |
| `geo_map` | `locations, value` or `lat, lon, value` | Country/region/city performance |
| `table` | optional `columns` | Ranked operational table with optional conditional formatting |
| `calendar_heatmap` | `date, value` | Daily seasonality and campaign activity |

```yaml
- id: campaign_pareto
  title: Revenue Pareto
  metric: Revenue
  chart: pareto
  x: Campaign
  y: Revenue
  top_n: 12
```

---

## 4. Recipe metadata

The Builder UI uses the following metadata table to filter chart kinds by the metric's underlying processor and to default required fields:

| Chart kind | Allowed processor kinds | Default x | Default y |
|---|---|---|---|
| `line` | binary_outcome, score_distribution, conversion, snapshot | first time-grain dim | metric.outputs[0] |
| `stacked_area` | binary_outcome, score_distribution, snapshot | first time-grain dim | metric.outputs[0] |
| `bar` | binary_outcome, snapshot | first non-time dim | metric.outputs[0] |
| `kpi_card` | aggregate metrics | — | metric.outputs[0] |
| `waterfall` | aggregate metrics | first non-time dim | metric.outputs[0] |
| `pareto` | aggregate metrics | first non-time dim | metric.outputs[0] |
| `treemap` | binary_outcome, score_distribution | — | — |
| `heatmap` | binary_outcome, score_distribution | first dim | second dim |
| `cohort_heatmap` | binary_outcome, snapshot | time/cohort dim | metric.outputs[0] |
| `scatter` | binary_outcome, score_distribution | metric.outputs[0] | metric.outputs[1] |
| `combo` | aggregate metrics | first time-grain dim | metric.outputs[0] + metric.outputs[1] |
| `interval` | aggregate metrics | first dim | metric.outputs[0] |
| `donut` | aggregate metrics | first dim | metric.outputs[0] |
| `geo_map` | binary_outcome, score_distribution, snapshot | location dim | metric.outputs[0] |
| `table` | aggregate metrics | — | — |
| `calendar_heatmap` | binary_outcome, score_distribution, snapshot | date dim | metric.outputs[0] |
| `bar_polar` | binary_outcome | — | — |
| `sankey` | aggregate metrics | source/target dims | metric.outputs[0] |
| `gauge` | binary_outcome, snapshot | — | — |
| `funnel` | funnel | — | — |
| `boxplot` | numeric_distribution | first time-grain dim | property |
| `histogram` | numeric_distribution | — | — |
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
  paper_bgcolor: "#f5f3ee"          # defaults to the app background
  plot_bgcolor: "#f5f3ee"           # defaults to the app background
  font: { family: "Inter", size: 14 }
  margins: { l: 32, r: 16, t: 48, b: 32 }
  legend: { orientation: "h", y: -0.2 }
  color_continuous_scale: "Viridis"
  qualitative_palette: "Set2"
  category_colors:
    Channel:
      Web: "#2563EB"
      Mobile: "#14B8A6"
```

The application initializes a built-in `valuestream` Plotly template before
dashboard rendering. Its `paper_bgcolor` and `plot_bgcolor` match the app
background token (`#f5f3ee` in light mode, `#020203` in dark mode) so Plotly
figures blend into the Streamlit shell instead of rendering white panels. The
default dashboard theme also passes those colors explicitly, so a workspace or
tile that overrides only `template` still keeps the app-matched background.

The chart factory applies the theme via `fig.update_layout(template=...)` etc. Per-tile overrides are allowed:

```yaml
tiles:
  - id: hot_metric
    chart: line
    theme:
      template: "plotly_dark"
```

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
