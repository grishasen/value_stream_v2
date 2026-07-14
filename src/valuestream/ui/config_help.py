"""Shared help text for configuration-editor fields.

Tooltips are deliberately keyed by catalog meaning instead of visible label.
The same field can therefore keep one definition and example in Config Builder,
AI Configuration Studio, and the KPI recipe workflow.
"""

from __future__ import annotations


def _tip(description: str, example: str | None = None) -> str:
    if not example:
        return description
    return f"{description}\n\n**Example:** `{example}`"


FIELD_HELP: dict[str, str] = {
    # Editor navigation and catalog selection.
    "editor.config_section": _tip(
        "Switch between the guided builder, README, and report inventory."
    ),
    "editor.builder_step": _tip("Open one stage of the catalog configuration workflow."),
    "editor.studio_step": _tip("Open one stage of the AI-assisted configuration workflow."),
    "editor.studio_phase": _tip(
        "Group of related studio steps; markers show complete, attention required, or empty.",
        "✓ Review",
    ),
    "editor.draft_step": _tip("Open one section of the current catalog draft."),
    # Sources and transforms.
    "source.selector": _tip("Choose the source definition to edit."),
    "source.id": _tip(
        "Stable YAML identifier used by processors to reference this source.",
        "ih_ai",
    ),
    "source.description": _tip(
        "Plain-language summary of the dataset and its business purpose.",
        "Interaction History outcomes exported daily.",
    ),
    "source.reader": _tip(
        "Reader implementation used to discover and decode matching files.",
        "pega_ds_export",
    ),
    "source.file_pattern": _tip(
        "Glob matched below the source root. Keep it narrow enough to exclude unrelated files.",
        "Data-Decision-*.zip",
    ),
    "source.group_pattern": _tip(
        "Optional regular expression that extracts a logical file group from each filename.",
        "Data-(?P<group>[^-]+)-",
    ),
    "source.root": _tip(
        "Directory resolved relative to the workspace unless an absolute path is supplied.",
        "data/interaction_history",
    ),
    "source.streaming": _tip(
        "Process the source in chunks so raw rows do not accumulate in memory."
    ),
    "source.hive_partitioning": _tip(
        "Read key=value path segments as columns for partitioned Parquet datasets.",
        "year=2026/month=07/part.parquet",
    ),
    "source.rename_capitalize": _tip(
        "Apply the Pega-aware rename/capitalize transform before defaults, filters, and calculations.",
        "pyName → Name",
    ),
    "source.timestamp_column": _tip(
        "Event timestamp used for calendar grains, incremental windows, and freshness.",
        "OutcomeTime",
    ),
    "source.timestamp_format": _tip(
        "Optional strptime format used when timestamp strings are not ISO-compatible.",
        "%Y-%m-%d %H:%M:%S",
    ),
    "source.natural_key": _tip(
        "Columns that together identify one source event for deterministic deduplication.",
        "InteractionID, OutcomeTime",
    ),
    "source.drop_columns": _tip(
        "Columns removed before processors run; use this for unused or sensitive inputs.",
        "CustomerName, RawPayload",
    ),
    "source.filter_mode": _tip(
        "Use Rules for common predicates or Raw AST for nested expression DSL logic."
    ),
    "source.filter_ast": _tip(
        "Expression DSL YAML evaluated before source rows reach processors.",
        'op: eq\nleft: {col: Channel}\nright: {lit: "Web"}',
    ),
    "default.field": _tip(
        "Source column that receives a fallback when its value is null.", "Channel"
    ),
    "default.value": _tip(
        "Fallback scalar value written before filtering and calculations.", "Unknown"
    ),
    "row.enabled": _tip("Include this row when generating catalog configuration."),
    "filter.field": _tip("Column evaluated by this filter rule.", "Channel"),
    "filter.operator": _tip("Comparison applied between the field and configured value.", "in"),
    "filter.value": _tip(
        "Comparison value. Comma-separated values are accepted by list operators.",
        "Web, Mobile",
    ),
    "calculation.name": _tip("Name of the derived column created by this transform.", "Margin"),
    "calculation.mode": _tip(
        "Calculation template or expression language used for the new column.", "Subtract"
    ),
    "calculation.left": _tip(
        "Left operand column or expression, depending on the selected mode.", "Revenue"
    ),
    "calculation.right_kind": _tip(
        "Interpret the right operand as a field reference or literal value.", "Field"
    ),
    "calculation.right": _tip("Right operand used by binary calculation modes.", "Cost"),
    "calculation.expression": _tip(
        "Full AST YAML or Polars expression for modes that need a custom expression.",
        'pl.col("Revenue") - pl.col("Cost")',
    ),
    "mapping.subject": _tip(
        "Source field mapped to the canonical SubjectID entity identifier.",
        "CustomerID",
    ),
    "mapping.outcome": _tip(
        "Source field mapped to the canonical Outcome category.",
        "pyOutcome",
    ),
    "mapping.outcome_time": _tip(
        "Source field mapped to the canonical OutcomeTime event timestamp.",
        "EventTime",
    ),
    "mapping.decision_time": _tip(
        "Optional source field mapped to the canonical DecisionTime timestamp.",
        "DecisionTime",
    ),
    "mapping.day": _tip("Optional existing day bucket field; otherwise it is derived.", "Day"),
    "mapping.month": _tip(
        "Optional existing month bucket field; otherwise it is derived.", "Month"
    ),
    "mapping.quarter": _tip(
        "Optional existing quarter bucket field; otherwise it is derived.", "Quarter"
    ),
    "mapping.year": _tip("Optional existing year bucket field; otherwise it is derived.", "Year"),
    # Processors and aggregate states.
    "processor.mode": _tip("Choose whether to create a processor or edit an existing definition."),
    "processor.selector": _tip("Choose the processor definition to edit."),
    "processor.id": _tip(
        "Stable YAML identifier referenced by metrics and report recipes.",
        "ih_ai_engagement",
    ),
    "processor.source": _tip("Source whose transformed chunks feed this processor.", "ih_ai"),
    "processor.kind": _tip(
        "Aggregation family that determines required fields, states, and query behavior.",
        "binary_outcome",
    ),
    "processor.description": _tip(
        "Plain-language definition of the population and aggregation purpose.",
        "Daily customer engagement outcomes by channel.",
    ),
    "processor.group_by": _tip(
        "Dimensions persisted in aggregate rows and available to filters and report grouping.",
        "Channel, Direction, Issue",
    ),
    "processor.time_column": _tip(
        "Timestamp column used to assign each input row to the configured grains.",
        "OutcomeTime",
    ),
    "processor.grains": _tip(
        "Time grains materialized for this processor. Summary is an all-time aggregate.",
        "Day, Month, Summary",
    ),
    "processor.filter_mode": _tip(
        "Use Rules for common predicates or Raw AST for nested processor-specific logic."
    ),
    "processor.filter_ast": _tip(
        "Expression DSL YAML applied only to this processor after source transforms.",
        'op: ne\nleft: {col: Outcome}\nright: {lit: "Pending"}',
    ),
    "processor.subject_field": _tip(
        "Field that identifies the business entity counted or followed by the processor.",
        "SubjectID",
    ),
    "processor.outcome_column": _tip(
        "Categorical field mapped to positive and negative business outcomes.",
        "Outcome",
    ),
    "processor.positive_values": _tip(
        "Comma-separated source values classified as positive outcomes.",
        "Clicked, Conversion",
    ),
    "processor.negative_values": _tip(
        "Comma-separated source values classified as negative outcomes.",
        "Impression, Pending",
    ),
    "processor.variant_column": _tip(
        "Field that distinguishes experimental roles or treatment variants.",
        "ControlGroup",
    ),
    "processor.score_properties": _tip(
        "Numeric model-score fields summarized into positive and negative digest states.",
        "Propensity, FinalPropensity",
    ),
    "processor.numeric_properties": _tip(
        "Numeric fields summarized with count, moments, extrema, and quantile sketches.",
        "Revenue, Margin",
    ),
    "processor.quantile_engine": _tip(
        "Mergeable sketch used for approximate percentiles. t-digest is strong near tails; KLL has rank guarantees.",
        "tdigest",
    ),
    "processor.stages": _tip(
        "Comma-separated funnel stage names in business order; each stage also needs a when expression.",
        "Presented, Clicked, Converted",
    ),
    "processor.snapshot_kind": _tip(
        "Periodic snapshots describe each interval; accumulating snapshots track progress toward completion.",
        "periodic",
    ),
    "processor.cadence": _tip(
        "Optional expected snapshot interval used for interpretation.", "monthly"
    ),
    "state.name": _tip(
        "Unique aggregate-state name exposed to metric calculations.", "UniqueSubjects_cpc"
    ),
    "state.type": _tip("Mergeable aggregation algorithm used to build this state.", "cpc"),
    "state.source_column": _tip(
        "Input field consumed by the state; count states may leave this blank.",
        "SubjectID",
    ),
    "state.derived_from": _tip(
        "Processor setting that generated this state; shown for provenance.", "score_properties"
    ),
    # Dimensions and exploration.
    "dimension.processor": _tip("Processor whose persisted grouping dimensions you want to edit."),
    "dimension.group_by": _tip(
        "Complete set of dimensions retained in this processor's aggregates.",
        "Channel, Direction, Issue",
    ),
    "dimension.profile_filter": _tip(
        "Limit field profiling to all, recommended, or currently selected dimensions."
    ),
    "dimension.pack": _tip(
        "Reusable business-oriented group of dimensions to add together.", "Engagement context"
    ),
    "dimension.promote_field": _tip(
        "Source field to add to the processor's permanent aggregate grain.", "Treatment"
    ),
    "dimension.exploration_dimensions": _tip(
        "Temporary grouping fields materialized for a bounded exploration window.",
        "Issue, Group",
    ),
    "dimension.window_days": _tip(
        "How many recent source days the temporary exploration reads.", "30"
    ),
    "dimension.ttl_days": _tip(
        "Days before the exploration definition is considered expired.", "14"
    ),
    "dimension.topk_enabled": _tip("Add a Top-K sketch for approximate frequent-value analysis."),
    "dimension.topk_field": _tip(
        "Field whose frequent values the Top-K state tracks.", "Treatment"
    ),
    "dimension.cpc_enabled": _tip("Add a CPC sketch for mergeable approximate unique counts."),
    "dimension.theta_enabled": _tip(
        "Add a Theta sketch when set union, intersection, or difference is needed."
    ),
    "dimension.entity_field": _tip(
        "Entity identifier supplied to the selected cardinality sketches.", "SubjectID"
    ),
    "dimension.sketch_group_by": _tip(
        "Dimensions at which sketch states are persisted and can later be compared.",
        "Channel, Direction",
    ),
    "dimension.exploration_selector": _tip(
        "Choose a temporary exploration definition to promote into the permanent catalog."
    ),
    # Metrics.
    "metric.action": _tip("Choose whether to create a new metric or edit an existing one."),
    "metric.create_from": _tip(
        "Start from a governed KPI recipe or configure every field manually."
    ),
    "metric.processor": _tip(
        "Processor whose aggregate states supply this metric.", "ih_ai_engagement"
    ),
    "metric.kind": _tip(
        "Calculation algorithm used to turn processor states into query results.", "formula"
    ),
    "metric.selector": _tip("Choose the existing metric definition to edit."),
    "metric.id": _tip(
        "Stable YAML key referenced by tiles, chat, APIs, and dependent metrics.",
        "VS_Click_Through_Rate",
    ),
    "metric.description": _tip(
        "Business definition describing what the metric measures and how it should be interpreted.",
        "Positive outcomes divided by all eligible outcomes.",
    ),
    "metric.depends_on": _tip(
        "Comma-separated metric IDs that must be evaluated before this metric.",
        "VS_Positive_Outcomes, VS_Total_Outcomes",
    ),
    "metric.display_label": _tip(
        "Friendly label used in axes, tables, cards, and tooltips.", "Click-through rate"
    ),
    "metric.unit": _tip("Optional business unit displayed with values.", "%"),
    "metric.value_format": _tip(
        "Default presentation format inherited by report tiles.", "percent"
    ),
    "metric.direction": _tip(
        "Whether higher or lower values should be interpreted as favorable.", "higher_is_better"
    ),
    "metric.formula_mode": _tip("Use the guided ratio form or edit the complete expression AST."),
    "metric.expression": _tip(
        "Expression DSL YAML evaluated over scalar aggregate states.", "col: Count"
    ),
    "metric.numerator": _tip(
        "Scalar state used as the ratio numerator or passthrough value.", "Positives"
    ),
    "metric.denominator": _tip(
        "Optional scalar state used as the safe-divide denominator.", "Count"
    ),
    "metric.state": _tip("Processor state consumed by this metric.", "UniqueSubjects_cpc"),
    "metric.output": _tip(
        "Result column or statistic produced by the metric algorithm.", "roc_auc"
    ),
    "metric.topk_limit": _tip("Maximum number of frequent items returned.", "10"),
    "metric.topk_error_type": _tip(
        "Error guarantee used when reading the Top-K sketch.", "NO_FALSE_POSITIVES"
    ),
    "metric.quantile": _tip("Requested percentile as a fraction from 0 to 1.", "0.95"),
    "metric.digest_property": _tip(
        "Matched positive/negative digest pair for one score property.", "Propensity"
    ),
    "metric.positive_digest": _tip(
        "Digest state built from positive outcomes.", "Propensity_Positive_tdigest"
    ),
    "metric.negative_digest": _tip(
        "Digest state built from negative outcomes.", "Propensity_Negative_tdigest"
    ),
    "metric.variant_column": _tip(
        "Aggregate dimension that identifies test and control roles.", "ControlGroup"
    ),
    "metric.test_role": _tip(
        "Value in the variant column interpreted as the test population.", "Test"
    ),
    "metric.control_role": _tip(
        "Value in the variant column interpreted as the baseline population.", "Control"
    ),
    "metric.confidence_level": _tip(
        "Confidence level used for intervals and significance calculations.", "0.95"
    ),
    "metric.outputs": _tip(
        "Optional comma-separated result columns to expose.", "Lift, Lift_P_Val"
    ),
    "metric.tests": _tip("Statistical tests to calculate for the contingency table.", "chi2, g"),
    "metric.lifecycle_outputs": _tip(
        "Lifecycle/RFM result columns exposed by this metric.",
        "frequency, monetary_value, rfm_segment",
    ),
    "metric.set_operation": _tip(
        "Theta set operation applied across the selected states.", "intersection"
    ),
    "metric.theta_states": _tip(
        "Theta states used as ordered set operands.", "Customers_A_theta, Customers_B_theta"
    ),
    "metric.from_stage": _tip(
        "Earlier funnel stage used as the drop-off denominator.", "Presented"
    ),
    "metric.to_stage": _tip("Later funnel stage compared with the starting stage.", "Clicked"),
    "metric.funnel_output": _tip("Return the drop-off as a rate or an absolute count.", "rate"),
    # Reports and tiles.
    "report.editing_mode": _tip("Use the guided editor or edit the complete tile YAML definition."),
    "report.library_search": _tip(
        "Filter the report library by dashboard, page, or tile title.", "engagement"
    ),
    "report.metric_filter": _tip("Show only tiles that use the selected metric."),
    "report.chart_filter": _tip("Show only tiles rendered with the selected chart kind."),
    "report.open_tile": _tip("Choose an existing tile to edit or start a new tile draft."),
    "report.metric": _tip("Catalog metric queried by this tile.", "VS_Click_Through_Rate"),
    "report.chart": _tip("Chart recipe used to render the metric result.", "kpi_card"),
    "report.dashboard": _tip("Dashboard that owns the report page and tile.", "ih_overview"),
    "report.dashboard_id": _tip("Stable ID for a new dashboard.", "ih_overview"),
    "report.page": _tip(
        "Page within the selected dashboard that owns this tile.", "executive_summary"
    ),
    "report.page_id": _tip("Stable ID for a new dashboard page.", "executive_summary"),
    "report.tile_title": _tip(
        "User-facing title displayed above the chart or KPI.", "Click-through rate"
    ),
    "report.tile_yaml": _tip(
        "Complete tile YAML for settings not represented by guided controls.",
        "chart: kpi_card\nmetric: VS_Click_Through_Rate",
    ),
    "report.field": _tip("Metric result column assigned to this chart role.", "Month"),
    "report.stages": _tip(
        "Comma-separated funnel stage columns rendered in business order.",
        "Presented, Clicked, Converted",
    ),
    "report.description": _tip(
        "Plain-language context displayed with the report tile.", "Monthly engagement efficiency."
    ),
    "report.value_format": _tip(
        "Tile-specific numeric format overriding the metric default.", "percent"
    ),
    "report.scale": _tip(
        "Transform time-series values to absolute, index-100, or percentage change.", "index_100"
    ),
    "report.show_trend_delta": _tip("Show the change from the comparison period when supported."),
    "report.goal_enabled": _tip("Draw a configured goal line on compatible charts."),
    "report.goal_value": _tip("Numeric position of the goal line.", "0.12"),
    "report.goal_label": _tip("Legend label for the goal line.", "Target CTR"),
    "report.goal_color": _tip("Hex color used to render the goal line.", "#2e7d32"),
    "report.bar_mode": _tip("How multiple bar series are arranged.", "stack"),
    "report.sort_by": _tip("Result column used to order bar categories.", "Count"),
    "report.sort_direction": _tip("Ascending or descending category order.", "desc"),
    "report.top_n": _tip(
        "Keep only the first N categories after sorting; zero disables the limit.", "10"
    ),
    "report.conditional_formatting": _tip(
        "YAML list of value rules used by supported table and KPI renderers.",
        'column: CTR, operator: ">=", value: 0.12, color: "#2e7d32"',
    ),
    "report.placement": _tip(
        "Render the KPI in normal page content or the page-level KPI strip.", "kpi_strip"
    ),
    "report.comparison": _tip("Comparison used to calculate the KPI delta.", "previous_period"),
    "report.comparison_period": _tip(
        "Calendar period shifted to obtain the comparison value.", "month"
    ),
    "report.sparkline_grain": _tip("Time grain used for the compact KPI history chart.", "daily"),
    "report.sparkline_points": _tip("Maximum observations shown in the sparkline.", "30"),
    "report.target_enabled": _tip("Attach a numeric business target to the KPI."),
    "report.target_value": _tip("Business target shown alongside the KPI.", "0.12"),
    "report.reference_enabled": _tip("Add a reference marker to a gauge chart."),
    "report.reference_value": _tip("Numeric value used for the gauge reference marker.", "0.10"),
    "report.filter_field": _tip("Metric result field exposed as a page filter.", "Channel"),
    "report.filter_label": _tip("User-facing label for the page filter.", "Channel"),
    "report.filter_placement": _tip(
        "Place the control in the primary filter row or secondary controls.",
        "primary",
    ),
    "report.filter_scope": _tip(
        "Apply the filter to every tile or only tiles exposing the selected field.",
        "compatible_tiles",
    ),
    "report.filter_control": _tip("Filter widget type selected for this field.", "multiselect"),
    "report.available_ranges": _tip(
        "Named time presets offered by the report page.", "Last 30 days, Last 12 months"
    ),
    "report.default_range": _tip(
        "Time preset selected when the report first opens.", "Last 30 days"
    ),
    # Workspace, theme, and chat configuration.
    "workspace.name": _tip("Workspace identifier stored in pipelines.yaml.", "customer_engagement"),
    "workspace.time_zone": _tip(
        "IANA time zone used for calendar boundaries and display.", "Europe/Berlin"
    ),
    "workspace.calendar_grains": _tip(
        "Default grains generated for processors and reports.", "Day, Month, Quarter, Year"
    ),
    "workspace.week_start": _tip("First weekday used for weekly calendar buckets.", "monday"),
    "workspace.theme_yaml": _tip(
        "Dashboard theme mapping merged into chart presentation settings.",
        'colorway: ["#1f77b4", "#ff7f0e"]',
    ),
    "chat.agent_prompt": _tip(
        "Business terminology and interpretation guidance sent to the aggregate-only chat planner.",
        "Treat SubjectID as a customer identifier.",
    ),
    "chat.description_type": _tip(
        "Catalog object type; read-only provenance for this row.", "Metric"
    ),
    "chat.description_key": _tip(
        "Source, processor, or metric ID described by this row.", "VS_Click_Through_Rate"
    ),
    "chat.description": _tip(
        "Business description used by the LLM planner when matching questions.",
        "Share of eligible outcomes that were positive.",
    ),
    # AI Studio runtime configuration and review fields.
    "ai.source_sample": _tip(
        "Upload a representative source file used only for schema discovery and preview."
    ),
    "ai.preview_rows": _tip("Maximum sampled rows used for schema inference and preview.", "5000"),
    "ai.workspace_sample": _tip("Choose a sample already stored in the workspace data directory."),
    "ai.model": _tip("LiteLLM model identifier, usually provider/model.", "openai/gpt-5"),
    "ai.api_base": _tip(
        "Optional endpoint for a proxy or local OpenAI-compatible server.", "http://localhost:11434"
    ),
    "ai.custom_provider": _tip(
        "Optional LiteLLM provider override when the model prefix is ambiguous.", "openai"
    ),
    "ai.api_key": _tip(
        "API credential used for this session; environment variables are preferred.",
        "OPENAI_API_KEY",
    ),
    "ai.temperature_override": _tip("Send an explicit temperature only to models that support it."),
    "ai.temperature": _tip("Sampling temperature used when an override is enabled.", "0.2"),
    "ai.reasoning_effort": _tip("Optional provider-supported reasoning budget.", "high"),
    "ai.verbosity": _tip("Optional provider-supported response detail level.", "low"),
    "ai.timeout": _tip("Maximum seconds allowed for one model request.", "120"),
    "ai.raw_yaml": _tip("Complete draft YAML for advanced review or manual correction."),
    "ai.user_goals": _tip(
        "Free-form business requirements that guide AI catalog generation.",
        "Weekly conversion by channel and average revenue per customer",
    ),
    "ai.refine_instruction": _tip(
        "Free-form change request the AI applies to the current draft.",
        "Add a KPI card with total orders to the overview page",
    ),
    "ai.copilot_message": _tip(
        "Free-form question or change request for the step-aware copilot.",
        "Add weekly revenue by channel",
    ),
    "ai.patch_accept": _tip(
        "Apply this structural change when the pending review is accepted; clear it to keep the previous accepted definition."
    ),
    "ai.field_approve": _tip(
        "Allow this field to be used in generated processors, metrics, and reports."
    ),
    "ai.field_send_values": _tip(
        "Allow representative values from this field to be included in AI prompts."
    ),
    "ai.field_name": _tip(
        "Post-processed field name available to catalog generation.", "SubjectID"
    ),
    "ai.field_type": _tip("Data type inferred from the working sample.", "String"),
    "ai.field_unique_count": _tip("Exact distinct count within the local working sample.", "42"),
    "ai.field_mode": _tip("Most frequent non-null value in the local sample.", "Web"),
    "ai.field_values": _tip(
        "Representative local values; sent only when sharing is enabled.", "Web, Mobile"
    ),
    "ai.field_tags": _tip(
        "Inferred semantic roles used to guide catalog generation.", "entity_id, categorical"
    ),
    "ai.field_search": _tip("Filter the field approval table by field name.", "Outcome"),
    "ai.keep_processors": _tip("Processor definitions retained in the reviewed draft."),
    "ai.keep_metrics": _tip("Metric definitions retained in the reviewed draft."),
    "ai.keep_tiles": _tip("Report tiles retained in the reviewed draft."),
    # KPI recipe library workflow.
    "recipe.search": _tip(
        "Filter recipes by name, domain, tag, or description.", "unique customers"
    ),
    "recipe.domain": _tip("Limit the catalog to one business KPI domain.", "Audience"),
    "recipe.selector": _tip("Choose the documented KPI recipe to configure."),
    "recipe.processor": _tip(
        "Processor whose fields and states will supply the recipe.", "ih_ai_engagement"
    ),
    "recipe.binding": _tip(
        "Choose the compatible processor field or aggregate input used for this recipe."
    ),
    "recipe.algorithm": _tip("Sketch or aggregation algorithm used for this binding.", "cpc"),
    "recipe.add_tile": _tip("Also install the recipe's recommended report tile."),
    "recipe.report_page": _tip("Dashboard page that receives the recommended tile."),
    "recipe.population": _tip(
        "Aggregate population counted for the selected field and algorithm.",
        "All eligible outcomes",
    ),
}


def field_help(key: str) -> str:
    """Return the required tooltip for a semantic configuration field."""
    try:
        return FIELD_HELP[key]
    except KeyError as exc:  # Fail fast during development instead of silently omitting help.
        raise KeyError(f"Unknown configuration help key: {key}") from exc


__all__ = ["FIELD_HELP", "field_help"]
