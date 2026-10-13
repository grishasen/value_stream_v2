"""Opt-in MCP adapters for the shared sample-backed catalog authoring layer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from valuestream.ai.catalog_authoring import CatalogAuthoring


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
    ) -> dict[str, Any]:
        """Create an in-memory baseline catalog and Studio generation prompt from a sample.

        Works without an existing catalog. Returns a valid rows-count baseline,
        schema, and the shared Studio prompt. Use the client model to propose richer
        analytics, update the draft, validate, preview changes, and apply separately.
        No server-side LLM credentials are required. Drafts expire after one hour
        or restart; no raw rows, examples, or prompts are retained by the server.
        """
        return guarded(
            "catalog_draft_create",
            lambda: authoring.create(
                sample_path,
                source_id=source_id,
                user_goals=user_goals,
                limit=limit,
                approved_fields=approved_fields,
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

        Preserve unsampled sources. Structural errors return an MCP error without
        modifying the draft; semantic issues remain editable via validation.
        """
        return guarded(
            "catalog_draft_update", lambda: authoring.update(draft_id, sections, expected_revision)
        )

    @mcp.tool(annotations=read_only)
    def catalog_draft_validate(
        draft_id: Annotated[str, Field(description="Draft identifier")],
    ) -> dict[str, Any]:
        """Validate catalog dependencies, approved sample fields, and bounded execution.

        Returns ok and actionable issues. Does not write catalog or aggregate files.
        """
        return guarded("catalog_draft_validate", lambda: authoring.validate(draft_id))

    @mcp.tool(annotations=draft_write)
    def catalog_draft_preview(
        draft_id: Annotated[str, Field(description="Draft identifier")],
    ) -> dict[str, Any]:
        """Return the full draft, object changes, YAML diff, and exact-revision review token.

        Present changes and validation issues to the user before apply. A token is
        issued only for a valid draft. Changes to the draft, workspace config, or
        sample invalidate review. No catalog is written by preview.
        """
        return guarded("catalog_draft_preview", lambda: authoring.preview(draft_id))

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
