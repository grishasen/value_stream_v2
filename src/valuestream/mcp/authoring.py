"""Opt-in MCP adapters for the shared sample-backed catalog authoring layer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from valuestream.ai.catalog_authoring import CatalogAuthoring
from valuestream.ai.catalog_templates import (
    PEGA_TIMESTAMP_FORMAT,
    OutcomeRules,
    TemplateBindings,
    TemplateName,
)
from valuestream.mcp.authoring_help import HelpTopic, authoring_help, help_call


def register_authoring_tools(
    mcp: Any, workspace: Path, annotations: Any, guarded: Callable[..., dict[str, Any]]
) -> None:
    """Register the same authoring workflow for stdio and HTTP."""

    authoring = CatalogAuthoring(workspace)
    read_only = annotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    draft_write = annotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
    apply_write = annotations(
        readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False
    )

    @mcp.tool(annotations=read_only)
    def catalog_authoring_help(
        topic: HelpTopic = "workflow",
        name: Annotated[
            str,
            Field(description="Exact kind, operation, operator or recipe ID. Omit to list names."),
        ] = "",
        definition: Annotated[
            str, Field(description="For a schema $ref, the name following #/$defs/.")
        ] = "",
    ) -> dict[str, Any]:
        """Look up the live catalog contract before guessing or reading repository files.

        Start with workflow for the editing sequence. For example request topic=metric
        and name=curve_from_digests for ROC AUC, or topic=chart and name=line.
        Examples use sample field names; bind them to the approved fields of your draft.
        """
        return guarded("catalog_authoring_help", lambda: authoring_help(topic, name, definition))

    @mcp.tool(annotations=read_only)
    def sample_profile(
        sample_path: Annotated[
            str, Field(description="CSV or Parquet path under this workspace's data/ directory")
        ],
        limit: Annotated[
            int, Field(ge=1, le=100_000, description="Maximum sample rows to inspect")
        ] = 1000,
        approved_fields: Annotated[
            list[str] | None,
            Field(description="Columns whose schema may be shared; defaults to all"),
        ] = None,
        example_fields: Annotated[
            list[str] | None,
            Field(
                description="Approved columns to share up to eight example values from; none by default"
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Profile a bounded sample without returning raw rows or persisting it.

        This is an authoring-only exception to aggregate queries. By default only
        schema/counts are shared; example values require explicit example_fields.
        """
        return guarded(
            "sample_profile",
            lambda: authoring.profile(
                sample_path,
                limit=limit,
                approved_fields=approved_fields,
                example_fields=example_fields,
            ),
        )

    @mcp.tool(annotations=draft_write)
    def catalog_draft_create(
        sample_path: Annotated[str, Field(description="CSV/Parquet file inside workspace data/")],
        source_id: Annotated[
            str, Field(min_length=1, max_length=100, description="New source identifier")
        ] = "sample",
        user_goals: Annotated[
            str,
            Field(
                max_length=20_000,
                description="Business requirements for the client model's catalog proposal",
            ),
        ] = "",
        limit: Annotated[int, Field(ge=1, le=100_000, description="Maximum sample rows")] = 1000,
        approved_fields: Annotated[
            list[str] | None,
            Field(description="Sample columns approved for authoring; defaults to all"),
        ] = None,
        *,
        include_studio_prompt: Annotated[
            bool, Field(description="Request the full legacy Studio generation prompt.")
        ] = False,
    ) -> dict[str, Any]:
        """Add a new sampled source to a draft. To change an existing source use catalog_draft_open.

        Use catalog_authoring_help to resolve schema and recipe questions directly
        through MCP. Keep the sampled source id and reader; edit transforms and schema.
        Approved fields are fixed for this draft. For additional fields create a new draft.
        Drafts expire after one hour or server restart.
        """
        return guarded(
            "catalog_draft_create",
            lambda: authoring.create(
                sample_path,
                source_id=source_id,
                user_goals=user_goals,
                include_studio_prompt=include_studio_prompt,
                limit=limit,
                approved_fields=approved_fields,
            ),
        )

    @mcp.tool(annotations=draft_write)
    def catalog_draft_open(
        source_id: Annotated[str, Field(description="Existing catalog source ID")],
        metric: Annotated[
            str | None,
            Field(
                description="Optional metric to inspect. Returns its processor, shared metrics and tiles, then a before/after sample review."
            ),
        ] = None,
        sample_path: Annotated[
            str | None,
            Field(
                description="Matching CSV/Parquet under data/. Inferred for a literal reader filename; required for a wildcard."
            ),
        ] = None,
        approved_fields: Annotated[
            list[str] | None,
            Field(
                description="Physical fields approved for this edit, including fields newly needed for grouping; defaults to all."
            ),
        ] = None,
        limit: Annotated[int, Field(ge=1, le=100_000)] = 1000,
    ) -> dict[str, Any]:
        """Open the published source for an in-place edit in a fresh draft, including after reconnect.

        Use for requests to change a metric or add a chart to an existing dashboard.
        Pass metric for focused dependency inspection and count-ratio sample previews.
        Edit with catalog_draft_apply_operations, preview, then obtain approval to apply.
        """
        return guarded(
            "catalog_draft_open",
            lambda: authoring.open_existing(
                source_id,
                metric=metric,
                sample_path=sample_path,
                approved_fields=approved_fields,
                limit=limit,
            ),
        )

    @mcp.tool(annotations=draft_write)
    def catalog_draft_update(
        draft_id: Annotated[str, Field(description="Draft identifier from catalog_draft_create")],
        sections: Annotated[
            dict[str, Any],
            Field(
                description="Complete replacement sections: pipelines, processors, metrics, dashboards, optional chat_with_data"
            ),
        ],
        expected_revision: Annotated[
            str, Field(description="Current draft revision; protects concurrent updates")
        ],
    ) -> dict[str, Any]:
        """Replace selected draft sections and invalidate the previous review token.

        Call catalog_authoring_help for the exact kind and a JSON example.
        Include a business description for each new metric. For an existing metric,
        omitting description keeps its text. An explicit string edits or clears it.
        Preserve unsampled sources. Structural errors return an MCP error without
        modifying the draft; semantic issues remain editable via validation.
        """
        return guarded(
            "catalog_draft_update", lambda: authoring.update(draft_id, sections, expected_revision)
        )

    @mcp.tool(annotations=draft_write)
    def catalog_draft_apply_operations(
        draft_id: str,
        expected_revision: str,
        operations: Annotated[list[dict[str, Any]], Field(min_length=1, max_length=100)],
    ) -> dict[str, Any]:
        """Edit the in-memory draft with one atomic batch of Studio operations.

        Call catalog_authoring_help(topic="operation") for names, then request the
        selected name for its argument shape. Recipes are under topic="recipe".
        Operations run in order. A failed batch preserves the revision and review.
        A successful batch requires validation and fresh review before catalog_draft_apply.
        """
        return guarded(
            "catalog_draft_apply_operations",
            lambda: authoring.apply_operations(draft_id, operations, expected_revision),
        )

    @mcp.tool(annotations=draft_write)
    def catalog_draft_apply_template(
        draft_id: str,
        expected_revision: str,
        template: TemplateName,
        *,
        bindings: TemplateBindings | None = None,
        outcomes: OutcomeRules | None = None,
        timestamp_format: Annotated[
            str | None,
            Field(description="Confirmed timestamp format. Use null for temporal columns."),
        ] = PEGA_TIMESTAMP_FORMAT,
        dashboard_title: str | None = None,
    ) -> dict[str, Any]:
        """Build and validate a complete dashboard in the draft in one call.

        daily_rows counts rows by day and channel. interaction_metrics includes daily_rows,
        five KPIs and their daily trends (CTR, ConversionRate, RocAuc, UniqueVisitors,
        UniqueActions) plus channel volume. Default field names match Pega IH. Rate
        denominators include rows with classified outcomes. Supply confirmed classes and
        field bindings when they differ. Compare outputs with the request before preview.
        """
        return guarded(
            "catalog_draft_apply_template",
            lambda: authoring.apply_template(
                draft_id,
                expected_revision,
                template,
                bindings=bindings or TemplateBindings(),
                outcomes=outcomes or OutcomeRules(),
                timestamp_format=timestamp_format,
                dashboard_title=dashboard_title,
            ),
        )

    def with_issue_help(result: dict[str, Any]) -> dict[str, Any]:
        if not result["ok"]:
            result["help_call"] = help_call(" ".join(result["issues"]))
        return result

    @mcp.tool(annotations=read_only)
    def catalog_draft_validate(
        draft_id: Annotated[str, Field(description="Draft identifier")],
    ) -> dict[str, Any]:
        """Validate catalog dependencies, approved sample fields, and bounded execution.

        Returns ok and actionable issues. Does not write catalog or aggregate files.
        """
        return guarded(
            "catalog_draft_validate", lambda: with_issue_help(authoring.validate(draft_id))
        )

    @mcp.tool(annotations=draft_write)
    def catalog_draft_preview(
        draft_id: Annotated[str, Field(description="Draft identifier")],
        expected_metrics: Annotated[
            list[str] | None, Field(description="Required metric names for the sampled source.")
        ] = None,
        expected_tiles: Annotated[
            list[str] | None,
            Field(description="Required dashboard/page/tile IDs for the sampled source."),
        ] = None,
        *,
        compact: Annotated[
            bool,
            Field(
                description="Show changed object IDs and focused metric review, omitting full catalog and YAML diff."
            ),
        ] = False,
    ) -> dict[str, Any]:
        """Return the draft review and exact-revision review token.

        Present changes and validation issues to the user before apply. A token is
        issued only for a valid draft. Changes to the draft, workspace config, or
        sample invalidate review. No catalog is written by preview.
        """
        return guarded(
            "catalog_draft_preview",
            lambda: with_issue_help(
                authoring.preview(
                    draft_id,
                    expected_metrics=expected_metrics,
                    expected_tiles=expected_tiles,
                    compact=compact,
                )
            ),
        )

    @mcp.tool(annotations=apply_write)
    def catalog_draft_apply(
        draft_id: Annotated[str, Field(description="Draft identifier")],
        review_token: Annotated[
            str,
            Field(description="Token issued for the exact valid revision by catalog_draft_preview"),
        ],
        confirm_removals: Annotated[
            bool,
            Field(
                description="True only after the user explicitly accepts the removals listed in preview"
            ),
        ] = False,
    ) -> dict[str, Any]:
        """Apply a reviewed catalog in a rollback-protected workspace transaction.

        Call only when the user authorized applying the displayed changes. Stale
        samples/configurations and unreviewed revisions are rejected. Retrying an
        already successful apply returns its receipt. Reports requires_data_run;
        ingestion is a separate CLI/UI action and is never started by this tool.
        """
        return guarded(
            "catalog_draft_apply",
            lambda: authoring.apply(draft_id, review_token, confirm_removals=confirm_removals),
        )

    @mcp.tool(annotations=draft_write)
    def catalog_draft_discard(
        draft_id: Annotated[str, Field(description="Draft identifier to release")],
    ) -> dict[str, Any]:
        """Discard an in-memory draft and release its configuration; no files are changed."""
        return guarded("catalog_draft_discard", lambda: authoring.discard(draft_id))
