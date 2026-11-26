---
name: valuestream-catalog-authoring
description: Create a Value Stream analytics catalog or dashboard from a CSV or Parquet sample, including Interaction History rates, ROC AUC and audience metrics. Use when someone asks to create a catalog, метрики or дашборд through Value Stream.
---

Use the connected Value Stream MCP tools for sample inspection and catalog edits. A data/ path belongs to the server workspace. Repository files and shell commands are unnecessary for this workflow.

1. Profile the sample. Ask for example values only for fields whose values you need to interpret. Reuse business rules supplied by the user. Clarify outcome classes or timestamp format when they are ambiguous.
2. Create a draft using all fields needed by the request. The baseline contains one rows KPI. Its validation alone does not fulfill a dashboard request.
3. Call catalog_draft_apply_template with draft_id, expected_revision and template. Choose daily_rows for counts by day and channel, or interaction_metrics for CTR, conversion rate, ROC AUC, distinct visitors and actions. The latter includes daily_rows. Supply bindings, outcome rules and dashboard title when they differ from the defaults. The standard IH template counts Clicked and Conversion as positive and divides by classified rows. Use the current revision for each edit.
4. For custom objects, look up a single kind or operation with catalog_authoring_help and submit the edit. Follow a returned help_call on an error. If the same failure persists after correcting the request, report the tool, field and issue.
5. Compare the returned metric names and tile IDs with the user's request. The template validates the sample and returns a preview call with expected_metrics and expected_tiles (dashboard/page/tile). Check these against the request and run that call. For manual edits, supply the expected names and IDs yourself. Missing items need another edit. Show the resulting metric definitions and dashboard changes to the user. Apply the displayed revision with its review token only after confirmation. Ingestion is a separate action.
