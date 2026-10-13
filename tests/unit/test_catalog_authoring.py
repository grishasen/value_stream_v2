"""Shared catalog authoring boundaries and MCP opt-in contract."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from click.testing import CliRunner

from valuestream.ai import catalog_authoring as authoring
from valuestream.ai.copilot import _processor_state_source_fields
from valuestream.cli import main
from valuestream.config import model
from valuestream.config.loader import load
from valuestream.mcp.server import create_server


@pytest.fixture
def service(tmp_path: Path) -> authoring.CatalogAuthoring:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sample.csv").write_text(
        "Region,Amount,Secret\nA,10,hidden\nB,20,private\nA,30,hidden\n"
    )
    return authoring.CatalogAuthoring(tmp_path)


@pytest.mark.parametrize("kind", ["csv", "parquet"])
def test_bounded_profile_shares_schema_and_only_selected_examples(
    service: authoring.CatalogAuthoring,
    kind: str,
) -> None:
    if kind == "parquet":
        pl.read_csv(service.workspace / "data/sample.csv").write_parquet(
            service.workspace / "data/sample.parquet"
        )
    profile = service.profile(f"data/sample.{kind}", limit=2, approved_fields=["Region", "Amount"])
    assert profile["sample_rows"] == 2
    assert [row["column"] for row in profile["schema"]] == ["Region", "Amount"]
    assert all("examples" not in row for row in profile["schema"])
    shared = service.profile(
        f"data/sample.{kind}", approved_fields=["Region"], example_fields=["Region"]
    )
    assert shared["schema"][0]["examples"] == ["A", "B"]
    with pytest.raises(ValueError, match="subset"):
        service.profile(
            f"data/sample.{kind}", approved_fields=["Region"], example_fields=["Secret"]
        )
    assert not (service.workspace / "catalog").exists()


def test_empty_workspace_starts_only_with_authoring_and_without_writes(tmp_path: Path) -> None:
    workspace = tmp_path / "absent"
    with pytest.raises(ValueError, match="catalog directory"):
        create_server(workspace)
    server = create_server(workspace, enable_authoring=True)
    tools = {tool.name for tool in server._tool_manager.list_tools()}
    assert {
        "sample_profile",
        "catalog_draft_create",
        "catalog_draft_update",
        "catalog_draft_validate",
        "catalog_draft_preview",
        "catalog_draft_apply",
        "catalog_draft_discard",
    } <= tools
    assert not workspace.exists()
    apply = server._tool_manager.get_tool("catalog_draft_apply")
    assert apply.annotations.read_only_hint is False
    assert apply.annotations.destructive_hint is True


def test_partial_catalog_is_not_silently_bootstrapped(tmp_path: Path) -> None:
    (tmp_path / "catalog").mkdir()
    (tmp_path / "catalog/pipelines.yaml").write_text(
        "catalog_version: 2\nworkspace: broken\nsources: []\n"
    )
    with pytest.raises(ValueError, match="missing catalog file"):
        create_server(tmp_path, enable_authoring=True)


def test_revision_review_apply_and_idempotent_receipt(service: authoring.CatalogAuthoring) -> None:
    created = service.create("data/sample.csv")
    draft_id = created["draft_id"]
    assert "Processor kind dictionary" in created["generation_prompt"]
    assert service.validate(draft_id)["ok"]
    with pytest.raises(ValueError, match="Preview"):
        service.apply(draft_id, "unreviewed")
    reviewed = service.preview(draft_id)
    assert reviewed["changes"]["sources"]["added"] == ["sample"]
    assert reviewed["changes"]["metrics"]["added"] == ["sample_Rows"]
    assert reviewed["changes"]["yaml_diff"]
    receipt = service.apply(draft_id, reviewed["review_token"])
    assert receipt["requires_data_run"]
    assert receipt["ingestion_started"] is False
    assert service.apply(draft_id, reviewed["review_token"]) == receipt
    assert "sample_Rows" in load(service.workspace).metrics.metrics
    assert not (service.workspace / "aggregates").exists()
    assert not (service.workspace / "meta").exists()
    assert all(
        not isinstance(value, pl.DataFrame) for value in vars(service.drafts[draft_id]).values()
    )


def test_updates_clear_review_and_detect_concurrent_revision(
    service: authoring.CatalogAuthoring,
) -> None:
    created = service.create("data/sample.csv")
    reviewed = service.preview(created["draft_id"])
    metrics = copy.deepcopy(created["catalog"]["metrics"])
    metrics["metrics"]["sample_Rows"]["description"] = "Reviewed row count"
    changed = service.update(created["draft_id"], {"metrics": metrics}, created["revision"])
    assert changed["revision"] != created["revision"]
    with pytest.raises(ValueError, match="revision changed"):
        service.update(created["draft_id"], {"metrics": metrics}, created["revision"])
    with pytest.raises(ValueError, match="Preview"):
        service.apply(created["draft_id"], reviewed["review_token"])


def test_noop_update_also_advances_revision(service: authoring.CatalogAuthoring) -> None:
    created = service.create("data/sample.csv")
    updated = service.update(
        created["draft_id"], {"metrics": created["catalog"]["metrics"]}, created["revision"]
    )
    assert updated["revision"] != created["revision"]


def test_structurally_invalid_update_leaves_draft_intact(
    service: authoring.CatalogAuthoring,
) -> None:
    created = service.create("data/sample.csv")
    with pytest.raises(ValueError, match="validation"):
        service.update(created["draft_id"], {"metrics": {"bogus": "field"}}, created["revision"])
    preview = service.preview(created["draft_id"])
    assert preview["revision"] == created["revision"]
    assert preview["catalog"] == created["catalog"]


def test_sample_dtype_failure_has_no_values_in_diagnostics(
    service: authoring.CatalogAuthoring,
) -> None:
    created = service.create("data/sample.csv")
    pipelines = copy.deepcopy(created["catalog"]["pipelines"])
    pipelines["sources"][0]["transforms"].append(
        {
            "kind": "cast",
            "columns": {"Secret": "Int64"},
        }
    )
    service.update(created["draft_id"], {"pipelines": pipelines}, created["revision"])
    result = service.validate(created["draft_id"])
    assert not result["ok"]
    assert "hidden" not in str(result["issues"])
    assert "private" not in str(result["issues"])
    assert "Sample execution failed" in result["issues"][0]


def test_reader_changes_cannot_bypass_sample_execution(service: authoring.CatalogAuthoring) -> None:
    created = service.create("data/sample.csv")
    pipelines = copy.deepcopy(created["catalog"]["pipelines"])
    pipelines["sources"][0]["reader"]["separator"] = ";"
    service.update(created["draft_id"], {"pipelines": pipelines}, created["revision"])
    result = service.validate(created["draft_id"])
    assert not result["ok"]
    assert "reader configuration unchanged" in result["issues"][0]


def test_renamed_fields_validate_without_approving_hidden_fields(
    service: authoring.CatalogAuthoring,
) -> None:
    (service.workspace / "data/raw.csv").write_text("pyRegion,pySecret\nA,hidden\nB,private\n")
    created = service.create("data/raw.csv", approved_fields=["pyRegion"])
    pipelines = copy.deepcopy(created["catalog"]["pipelines"])
    pipelines["sources"][0]["transforms"].insert(0, {"kind": "rename_capitalize"})
    processors = copy.deepcopy(created["catalog"]["processors"])
    processors["processors"][0]["group_by"].append("Region")
    changed = service.update(
        created["draft_id"], {"pipelines": pipelines, "processors": processors}, created["revision"]
    )
    assert service.validate(created["draft_id"])["ok"]
    processors["processors"][0]["group_by"].append("Secret")
    service.update(created["draft_id"], {"processors": processors}, changed["revision"])
    invalid = service.validate(created["draft_id"])
    assert not invalid["ok"]
    assert any("Secret" in issue for issue in invalid["issues"])


def test_score_recipe_states_require_runtime_inputs_not_output_names() -> None:
    processor = model.ScoreDistributionProcessor.model_validate(
        {
            "id": "scores",
            "source": "ih",
            "kind": "score_distribution",
            "time": {"property": "", "grain": "summary"},
            "score_properties": [{"column": "Score", "role": "primary"}],
            "outcome": {
                "column": "Outcome",
                "positive_values": ["Clicked"],
                "negative_values": ["Impression"],
            },
            "states": {
                "Count": {"type": "count"},
                "personalization": {
                    "type": "pooled_mean",
                    "recipe": "personalization",
                    "weight": "Count",
                },
                "novelty": {"type": "pooled_mean", "recipe": "novelty", "weight": "Count"},
            },
        }
    )
    fields = _processor_state_source_fields(processor)
    assert fields == {"CustomerID", "Name", "InteractionID"}


def test_adding_source_preserves_existing_catalog_and_guards_inactive_changes(
    service: authoring.CatalogAuthoring,
) -> None:
    first = service.create("data/sample.csv", source_id="first")
    review = service.preview(first["draft_id"])
    service.apply(first["draft_id"], review["review_token"])
    second = service.create("data/sample.csv", source_id="second")
    assert service.validate(second["draft_id"])["ok"]
    processors = copy.deepcopy(second["catalog"]["processors"])
    processors["processors"][0]["description"] = "Unreviewed inactive source change"
    service.update(second["draft_id"], {"processors": processors}, second["revision"])
    result = service.validate(second["draft_id"])
    assert not result["ok"]
    assert any("outside the active" in issue for issue in result["issues"])


def test_mid_write_failure_restores_existing_bytes(
    service: authoring.CatalogAuthoring, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = service.create("data/sample.csv", source_id="first")
    review = service.preview(first["draft_id"])
    service.apply(first["draft_id"], review["review_token"])
    second = service.create("data/sample.csv", source_id="second")
    review = service.preview(second["draft_id"])
    paths = [service.workspace / "catalog" / name for name in authoring.CATALOG_FILENAMES]
    originals = {path: path.read_bytes() for path in paths}
    writer = authoring.write_yaml

    def fail_metrics(path: Path, payload: dict[str, Any]) -> None:
        if path.name == "metrics.yaml":
            raise OSError("injected disk write failure")
        writer(path, payload)

    monkeypatch.setattr(authoring, "write_yaml", fail_metrics)
    with pytest.raises(OSError, match="injected"):
        service.apply(second["draft_id"], review["review_token"])
    assert {path: path.read_bytes() for path in paths} == originals
    assert len(load(service.workspace).pipelines.sources) == 1


def test_invalid_proposal_is_editable_but_never_gets_an_apply_token(
    service: authoring.CatalogAuthoring,
) -> None:
    created = service.create("data/sample.csv", approved_fields=["Region"])
    processors = copy.deepcopy(created["catalog"]["processors"])
    processors["processors"][0]["group_by"].append("Secret")
    changed = service.update(created["draft_id"], {"processors": processors}, created["revision"])
    assert changed["catalog"]["processors"] == processors
    validation = service.validate(created["draft_id"])
    assert not validation["ok"]
    assert any("Secret" in issue for issue in validation["issues"])
    assert service.preview(created["draft_id"])["review_token"] is None
    assert not (service.workspace / "catalog").exists()


@pytest.mark.parametrize("changed", ["sample", "catalog", "ai"])
def test_stale_inputs_prevent_apply(service: authoring.CatalogAuthoring, changed: str) -> None:
    created = service.create("data/sample.csv")
    reviewed = service.preview(created["draft_id"])
    if changed == "sample":
        (service.workspace / "data/sample.csv").write_text("Region\nChanged\n")
    elif changed == "catalog":
        (service.workspace / "catalog").mkdir()
        (service.workspace / "catalog/metrics.yaml").write_text("external change")
    else:
        (service.workspace / "ai.yaml").write_text("ai: {}\n")
    with pytest.raises(ValueError, match="changed"):
        service.apply(created["draft_id"], reviewed["review_token"])
    assert not (service.workspace / "catalog/pipelines.yaml").exists()


def test_apply_rolls_back_every_catalog_and_ai_file(
    service: authoring.CatalogAuthoring, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = service.create("data/sample.csv")
    draft = copy.deepcopy(created["catalog"])
    draft["chat_with_data"] = {"agent_prompt": "Use counts"}
    ai_path = service.workspace / "ai.yaml"
    ai_path.write_text("ai:\n  llm:\n    model: unchanged\n")
    original = ai_path.read_bytes()

    def fail_written(*args: Any, **kwargs: Any) -> None:
        raise ValueError("injected post-write failure")

    with pytest.raises(ValueError, match="injected"):
        authoring.apply_catalog_draft(service.workspace, draft, validate_written=fail_written)
    assert ai_path.read_bytes() == original
    assert not any(
        (service.workspace / "catalog" / filename).exists()
        for filename in authoring.CATALOG_FILENAMES
    )


def test_sample_and_configuration_paths_cannot_escape_workspace(
    service: authoring.CatalogAuthoring, tmp_path: Path
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.csv"
    outside.write_text("Region\nA\n")
    (tmp_path / "data/link.csv").symlink_to(outside)
    for name in (str(outside), "../outside.csv", "data/link.csv"):
        with pytest.raises(ValueError, match="inside"):
            service.profile(name)
    created = service.create("data/sample.csv")
    reviewed = service.preview(created["draft_id"])
    destination = tmp_path.parent / f"{tmp_path.name}-catalog"
    destination.mkdir()
    (tmp_path / "catalog").symlink_to(destination)
    with pytest.raises(ValueError, match="inside"):
        service.apply(created["draft_id"], reviewed["review_token"])
    assert not list(destination.iterdir())


def test_drafts_are_bounded_expire_and_can_be_discarded(
    service: authoring.CatalogAuthoring, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(authoring, "MAX_DRAFTS", 1)
    created = service.create("data/sample.csv")
    with pytest.raises(ValueError, match="Too many"):
        service.create("data/sample.csv")
    assert service.discard(created["draft_id"])["discarded"]
    expired = service.create("data/sample.csv")
    service.drafts[expired["draft_id"]].created_at -= authoring.DRAFT_TTL_SECONDS + 1
    with pytest.raises(ValueError, match="expired"):
        service.validate(expired["draft_id"])


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_cli_passes_authoring_flag_for_missing_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transport: str
) -> None:
    captured: dict[str, Any] = {}

    def run(path: str, **options: Any) -> None:
        captured.update(path=path, **options)

    monkeypatch.setattr("valuestream.cli.run_mcp_stdio", run)
    monkeypatch.setattr("valuestream.mcp.http.run_http", run)
    result = CliRunner().invoke(
        main,
        ["serve-mcp", str(tmp_path / "missing"), "--transport", transport, "--enable-authoring"],
    )
    assert result.exit_code == 0, result.output
    assert captured["enable_authoring"] is True
