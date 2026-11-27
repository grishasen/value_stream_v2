"""Real MCP clients bootstrap a catalog over both transports and query aggregates."""

from __future__ import annotations

import asyncio
import hashlib
import os
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import httpx2
import polars as pl
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import ReadResourceRequestParams, Request, RequestParams, TextResourceContents
from pydantic import TypeAdapter

from valuestream.config.loader import load
from valuestream.engine.runner import run_workspace


@asynccontextmanager
async def _client(
    transport: str, workspace: Path, log_path: Path, *, discover: bool = False
) -> AsyncIterator[ClientSession]:
    args = [
        "-c",
        "from valuestream.cli import main; main()",
        "serve-mcp",
        str(workspace),
        "--enable-authoring",
    ]
    env = {
        **os.environ,
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "VALUESTREAM_MCP_TOKEN": "authoring-test",
    }
    with log_path.open("w") as log:
        if transport == "stdio":
            parameters = StdioServerParameters(command=sys.executable, args=args, env=env)
            async with (
                stdio_client(parameters, errlog=log) as (read, write),
                ClientSession(read, write, read_timeout_seconds=20) as client,
            ):
                initialized = await client.discover() if discover else await client.initialize()
                assert "catalog_authoring_help" in (initialized.instructions or "")
                yield client
            return
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        process = subprocess.Popen(
            [sys.executable, *args, "--transport", "http", "--port", str(port)],
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(f"MCP startup failed:\n{log_path.read_text()}")
                with socket.socket() as probe:
                    if probe.connect_ex(("127.0.0.1", port)) == 0:
                        break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("MCP startup timed out")
            async with (
                httpx2.AsyncClient(headers={"Authorization": "Bearer authoring-test"}) as http,
                streamable_http_client(f"http://127.0.0.1:{port}/mcp", http_client=http) as (
                    read,
                    write,
                ),
                ClientSession(read, write, read_timeout_seconds=20) as client,
            ):
                initialized = await client.discover() if discover else await client.initialize()
                assert "catalog_authoring_help" in (initialized.instructions or "")
                yield client
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            with socket.socket() as probe:
                assert probe.connect_ex(("127.0.0.1", port)) != 0


async def _call(client: ClientSession, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await client.call_tool(name, arguments)
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return cast(dict[str, Any], result.structured_content)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize("kind", ["csv", "parquet"])
async def test_sample_to_catalog_to_aggregate_query(
    tmp_path: Path, transport: str, kind: str
) -> None:
    workspace = tmp_path / "new-workspace"
    async with asyncio.timeout(50), _client(transport, workspace, tmp_path / "mcp.log") as client:
        listing = await client.list_tools()
        assert "catalog_draft_create" in {tool.name for tool in listing.tools}
        assert not workspace.exists()  # Startup itself has no side effects.
        before = await client.call_tool("metric_list", {})
        assert before.is_error
        (workspace / "data").mkdir(parents=True)
        sample_relative = f"data/events.{kind}"
        sample = workspace / sample_relative
        frame = pl.DataFrame({"Region": ["A", "B", "A"], "Amount": [10, 20, 30]})
        if kind == "csv":
            frame.write_csv(sample)
        else:
            frame.write_parquet(sample)
        original_sample = sample.read_bytes()
        profile = await _call(
            client, "sample_profile", {"sample_path": sample_relative, "limit": 2}
        )
        assert profile["sample_rows"] == 2
        assert "examples" not in profile["schema"][0]
        created = await _call(
            client,
            "catalog_draft_create",
            {"sample_path": sample_relative, "user_goals": "Show record volume by region"},
        )
        draft_id = created["draft_id"]
        assert not (workspace / "catalog").exists()
        # The client model can replace complete sections. Here the deterministic
        # test adds a sampled dimension and a dashboard chart without an LLM.
        processors = created["catalog"]["processors"]
        processors["processors"][0]["group_by"].append("Region")
        dashboards = created["catalog"]["dashboards"]
        dashboards["dashboards"][0]["pages"][0]["tiles"].append(
            {
                "id": "by_region",
                "title": "Rows by region",
                "chart": "bar",
                "metric": "sample_Rows",
                "x": "Region",
            }
        )
        await _call(
            client,
            "catalog_draft_update",
            {
                "draft_id": draft_id,
                "expected_revision": created["revision"],
                "sections": {"processors": processors, "dashboards": dashboards},
            },
        )
        validated = await _call(client, "catalog_draft_validate", {"draft_id": draft_id})
        assert validated["ok"], validated["issues"]
        reviewed = await _call(client, "catalog_draft_preview", {"draft_id": draft_id})
        assert reviewed["changes"]["sources"]["added"] == ["sample"]
        assert reviewed["changes"]["tiles"]["added"]
        receipt = await _call(
            client,
            "catalog_draft_apply",
            {"draft_id": draft_id, "review_token": reviewed["review_token"]},
        )
        assert receipt["applied"]
        assert receipt["requires_data_run"]
        assert receipt["ingestion_started"] is False
        assert not (workspace / "aggregates").exists()
        assert sample.read_bytes() == original_sample
        assert len(load(workspace).pipelines.sources) == 1
        manifest = await _call(client, "metric_list", {})
        assert manifest["matched_metrics"] == 1
        assert run_workspace(workspace).status == "ok"
        assert run_workspace(workspace).results[0].chunks_skipped == 1
        query = await _call(
            client, "metric_query", {"metric": "sample_Rows", "group_by": ["Region"]}
        )
        assert sorted(query["rows"], key=lambda row: row["Region"]) == [
            {"Region": "A", "sample_Rows": 2},
            {"Region": "B", "sample_Rows": 1},
        ]
        assert query["provenance"]["catalog_hash"] == receipt["config_hash"]
        assert query["provenance"]["computation_hash"]
        retry = await _call(
            client,
            "catalog_draft_apply",
            {"draft_id": draft_id, "review_token": reviewed["review_token"]},
        )
        assert retry == receipt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_client_builds_ih_dashboard_from_mcp_help(tmp_path: Path) -> None:
    workspace = tmp_path / "ih"
    data = workspace / "data"
    data.mkdir(parents=True)
    pl.DataFrame(
        {
            "pxOutcomeTime": ["20261001T120000.000 GMT"] * 4,
            "pyChannel": ["Web"] * 4,
            "pyOutcome": ["Clicked", "Conversion", "Impression", "Pending"],
            "pySubjectID": ["C1", "C2", "C1", "C1"],
            "pyName": ["A", "A", "A", "B"],
            "pyPropensity": [0.9, 0.8, 0.2, 0.1],
        }
    ).write_parquet(data / "ih.parquet")
    async with _client("stdio", workspace, tmp_path / "mcp.log") as client:
        created = await _call(
            client,
            "catalog_draft_create",
            {
                "source_id": "ih",
                "sample_path": "data/ih.parquet",
            },
        )
        draft = {"draft_id": created["draft_id"]}
        help_request = created["help_call"]
        workflow = await _call(client, help_request["tool"], help_request["arguments"])
        assert "daily_rows" in workflow["daily_count"]
        daily = (
            await _call(
                client,
                "catalog_authoring_help",
                {
                    "topic": "example",
                    "name": "daily_rows",
                },
            )
        )["contract"]
        pipelines = created["catalog"]["pipelines"]
        pipelines["sources"][0]["transforms"].extend(daily["append_transforms"])
        updated = await _call(
            client,
            "catalog_draft_update",
            {
                **draft,
                "expected_revision": created["revision"],
                "sections": {"pipelines": pipelines},
            },
        )
        metrics = (
            await _call(
                client,
                "catalog_authoring_help",
                {
                    "topic": "example",
                    "name": "interaction_metrics",
                },
            )
        )["contract"]
        operations = daily["operations"] + metrics["operations"]
        arguments = {**draft, "expected_revision": updated["revision"]}
        rejected = await client.call_tool(
            "catalog_draft_apply_operations",
            {
                **arguments,
                "operations": [operations[0], {"op": "set_metric"}],
            },
        )
        assert rejected.is_error
        recovery = rejected.structured_content["error"]["help_call"]
        shape = await _call(client, recovery["tool"], recovery["arguments"])
        assert shape["contract"]["shape"]["op"] == "set_metric"

        changed = await _call(
            client,
            "catalog_draft_apply_operations",
            {
                **arguments,
                "operations": operations,
            },
        )
        assert changed["stage"] == "edited"
        next_call = changed["next_call"]
        checked = await _call(client, next_call["tool"], next_call["arguments"])
        assert checked["ok"], checked["issues"]
        reviewed = await _call(client, "catalog_draft_preview", draft)
        expected = {
            "ih_Rows": 4,
            "CTR": 0.5,
            "ConversionRate": 0.25,
            "RocAuc": 1.0,
            "UniqueVisitors": 2,
            "UniqueActions": 2,
        }
        assert set(reviewed["changes"]["metrics"]["added"]) == expected.keys()
        receipt = await _call(
            client, "catalog_draft_apply", {**draft, "review_token": reviewed["review_token"]}
        )
        assert receipt["requires_data_run"]
        assert run_workspace(workspace).status == "ok"
        for name, value in expected.items():
            response = await _call(client, "metric_query", {"metric": name})
            assert response["rows"][0][name] == pytest.approx(value, abs=0.02)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", "http"])
async def test_skill_discovery_and_bound_template(tmp_path: Path, transport: str) -> None:
    workspace = tmp_path / "workspace"
    data = workspace / "data"
    data.mkdir(parents=True)
    pl.DataFrame(
        {
            "at": ["2026-10-01"] * 4,
            "via": ["App"] * 4,
            "label": ["visit", "sale", "lost", "lost"],
            "prediction": [0.8, 0.9, 0.2, 0.1],
            "person": ["A", "B", "A", "B"],
            "offer": ["offer1", "offer2", "offer3", "offer1"],
        }
    ).write_parquet(data / "events.parquet")
    async with _client(transport, workspace, tmp_path / "mcp.log", discover=True) as client:
        discovery = client.discover_result
        assert discovery is not None
        assert "io.modelcontextprotocol/skills" in (discovery.capabilities.extensions or {})
        result_type = TypeAdapter(dict[str, Any])
        listing = await client.send_request(
            Request(method="skills/list", params=RequestParams()), result_type
        )
        (entry,) = listing["skills"]
        name = entry["frontmatter"]["name"]
        assert name == "valuestream-catalog-authoring"
        lookup = await client.send_request(
            Request[ReadResourceRequestParams, str](
                method="skills/get", params=ReadResourceRequestParams(uri=entry["uri"])
            ),
            result_type,
        )
        assert lookup["skill"] == entry
        resource = await client.read_resource(entry["uri"])
        text = resource.contents[0]
        assert isinstance(text, TextResourceContents)
        content = text.text.encode()
        local = Path(__file__).resolve().parents[2] / ".opencode" / "skills" / name / "SKILL.md"
        assert local.read_bytes() == content
        assert entry["resources"] == [
            {
                "uri": entry["uri"],
                "size": len(content),
                "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
            }
        ]
        draft = await _call(
            client,
            "catalog_draft_create",
            {"sample_path": "data/events.parquet", "source_id": "events"},
        )
        handle = {"draft_id": draft["draft_id"]}
        expected_metrics = ["CTR", "ConversionRate", "RocAuc", "UniqueVisitors", "UniqueActions"]
        expected_tiles = [f"events_overview/overview/{metric}_day" for metric in expected_metrics]
        expectations = {"expected_metrics": expected_metrics, "expected_tiles": expected_tiles}
        baseline = await _call(client, "catalog_draft_preview", {**handle, **expectations})
        assert not baseline["ok"]
        assert baseline["review_token"] is None
        arguments = {
            **handle,
            "expected_revision": draft["revision"],
            "template": "interaction_metrics",
            "timestamp_format": "%Y-%m-%d",
            "dashboard_title": "October engagement",
            "bindings": {
                "timestamp": "at",
                "channel": "via",
                "outcome": "label",
                "visitor": "person",
                "action": "offer",
                "score": "prediction",
            },
            "outcomes": {
                "positive": ["visit", "sale"],
                "conversion": ["sale"],
                "negative": ["lost"],
            },
        }
        bad = await client.call_tool("catalog_draft_apply_template", {**arguments, "bindings": {}})
        assert bad.is_error
        invalid_time = await client.call_tool(
            "catalog_draft_apply_template", {**arguments, "timestamp_format": "%d/%m/%Y"}
        )
        assert invalid_time.is_error
        changed = await _call(client, "catalog_draft_apply_template", arguments)
        assert set(changed["changes"]["metrics"]["added"]) == set(expected_metrics)
        assert set(expected_tiles) <= set(changed["next_call"]["arguments"]["expected_tiles"])
        assert set(expected_metrics) <= set(changed["outputs"]["metrics"])
        assert set(expected_tiles) <= set(changed["outputs"]["tiles"])
        invalidated = await _call(
            client,
            "catalog_draft_preview",
            {**handle, "expected_tiles": ["events_overview/overview/absent"]},
        )
        assert invalidated["review_token"] is None
        assert not invalidated["ok"]
        reviewed = await _call(
            client, changed["next_call"]["tool"], changed["next_call"]["arguments"]
        )
        assert reviewed["ok"], reviewed["issues"]
        assert reviewed["catalog"]["dashboards"]["dashboards"][0]["title"] == "October engagement"
        assert not (workspace / "catalog").exists()
        await _call(
            client, "catalog_draft_apply", {**handle, "review_token": reviewed["review_token"]}
        )
        assert run_workspace(workspace).status == "ok"
        for name, value in zip(expected_metrics, [0.5, 0.25, 1, 2, 3], strict=True):
            query = await _call(client, "metric_query", {"metric": name})
            assert query["rows"][0][name] == pytest.approx(value, abs=0.02)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", "http"])
async def test_reopen_published_ctr_in_fresh_session(tmp_path: Path, transport: str) -> None:
    workspace = tmp_path / "workspace"
    data = workspace / "data"
    data.mkdir(parents=True)
    frame = pl.DataFrame(
        {
            "pxOutcomeTime": ["20261001T120000.000 GMT"] * 8,
            "pyChannel": ["Web", "Web", "App", "App"] * 2,
            "pyIssue": ["Sales", "Retention"] * 4,
            "pyOutcome": [
                "Clicked",
                "Conversion",
                "Impression",
                "Impression",
                "Pending",
                "Clicked",
                "Clicked",
                "Pending",
            ],
            "pySubjectID": ["A", "B", "C", "A"] * 2,
            "pyName": ["a", "b"] * 4,
            "pyPropensity": [0.8, 0.9, 0.3, 0.1, 0.4, 0.9, 0.6, 0.5],
        }
    )
    frame.write_parquet(data / "ih.parquet")
    await _exercise_ctr_edit(transport, workspace, "data/ih.parquet", frame)


async def _exercise_ctr_edit(
    transport: str, workspace: Path, sample_path: str, frame: pl.DataFrame
) -> None:
    source = "interactions"
    published_fields = [col for col in frame.columns if col != "pyIssue"]
    async with _client(transport, workspace, workspace.parent / "create.log") as client:
        draft = await _call(
            client,
            "catalog_draft_create",
            {"sample_path": sample_path, "source_id": source, "approved_fields": published_fields},
        )
        built = await _call(
            client,
            "catalog_draft_apply_template",
            {
                "draft_id": draft["draft_id"],
                "expected_revision": draft["revision"],
                "template": "interaction_metrics",
            },
        )
        preview = await _call(client, "catalog_draft_preview", built["next_call"]["arguments"])
        assert preview["counts"]["Tiles"] == 13
        await _call(
            client,
            "catalog_draft_apply",
            {"draft_id": draft["draft_id"], "review_token": preview["review_token"]},
        )
    before = load(workspace)
    async with _client(transport, workspace, workspace.parent / "edit.log") as client:
        args = {"source_id": source, "sample_path": sample_path, "approved_fields": frame.columns}
        duplicate = await client.call_tool("catalog_draft_create", args)
        assert duplicate.is_error
        recovery = duplicate.structured_content["error"]["next_call"]
        assert recovery["tool"] == "catalog_draft_open"
        assert recovery["arguments"]["approved_fields"] == frame.columns
        opened = await _call(
            client,
            "catalog_draft_open",
            {"source_id": source, "metric": "CTR", "approved_fields": frame.columns},
        )
        assert opened["stage"] == "opened"
        assert "catalog" not in opened
        old = opened["metric_context"]
        assert set(old["processor_metrics"]) == {"CTR", "UniqueVisitors", "UniqueActions"}
        handle = {"draft_id": opened["draft_id"]}
        proc = {
            **old["processor"],
            "id": "interactions_ctr",
            "group_by": ["Day", "pyChannel", "pyIssue"],
            "outcome": {
                "column": "pyOutcome",
                "positive_values": ["Clicked"],
                "negative_values": ["Impression"],
            },
            "states": {
                "Clicks": {"type": "count", "outcome": "positive"},
                "Impressions": {"type": "count", "outcome": "negative"},
            },
        }
        metric = {
            **old["metric"],
            "processor": proc["id"],
            "description": "Clicked divided by Impression.",
            "expression": {
                "op": "safe_div",
                "num": {"col": "Clicks"},
                "den": {"col": "Impressions"},
            },
        }
        tile = {
            "id": "CTR_by_channel_issue",
            "title": "CTR by channel and issue",
            "metric": "CTR",
            "chart": "bar",
            "x": "pyChannel",
            "color": "pyIssue",
        }
        edits = [
            {"op": "set_processor", "processor": proc},
            {"op": "set_metric", "name": "CTR", "metric": metric},
            {
                "op": "set_tile",
                "dashboard": "interactions_overview",
                "page": "overview",
                "tile": tile,
            },
        ]
        changed = await _call(
            client,
            "catalog_draft_apply_operations",
            {**handle, "expected_revision": opened["revision"], "operations": edits},
        )
        call = changed["next_call"]
        review = await _call(
            client,
            call["tool"],
            {
                **call["arguments"],
                "expected_metrics": ["CTR"],
                "expected_tiles": ["interactions_overview/overview/CTR_by_channel_issue"],
            },
        )
        assert review["ok"], review["issues"]
        assert "catalog" not in review
        assert "yaml_diff" not in review["changes"]
        assert review["counts"] == {
            **preview["counts"],
            "Processors": preview["counts"]["Processors"] + 1,
            "Tiles": 14,
        }
        focused = review["metric_review"]
        assert focused["before"] == old
        assert focused["after"]["metric"]["expression"] == metric["expression"]
        assert {t["id"] for t in focused["after"]["tiles"]} == {
            "CTR_day",
            "CTR_total",
            "CTR_by_channel_issue",
        }
        sample = frame.head(1000)
        clicks = sample["pyOutcome"].eq("Clicked").sum()
        impressions = sample["pyOutcome"].eq("Impression").sum()
        assert focused["sample"] == {
            "status": "ok",
            "sample_rows": sample.height,
            "numerator": {"state": "Clicks", "count": clicks},
            "denominator": {"state": "Impressions", "count": impressions},
            "value": pytest.approx(clicks / impressions if impressions else 0),
        }
        assert load(workspace) == before
        receipt = await _call(
            client, "catalog_draft_apply", {**handle, "review_token": review["review_token"]}
        )
        assert receipt["requires_data_run"]
        assert not receipt["ingestion_started"]
        after = load(workspace)
        assert after.pipelines == before.pipelines
        assert [
            p for p in after.processors.processors if p.id != proc["id"]
        ] == before.processors.processors
        assert {k: v for k, v in after.metrics.metrics.items() if k != "CTR"} == {
            k: v for k, v in before.metrics.metrics.items() if k != "CTR"
        }
        assert (
            after.dashboards.dashboards[0].pages[0].tiles[:-1]
            == before.dashboards.dashboards[0].pages[0].tiles
        )
        assert run_workspace(workspace).status == "ok"
        result = await _call(
            client, "metric_query", {"metric": "CTR", "group_by": ["pyChannel", "pyIssue"]}
        )
        for row in result["rows"]:
            labels = frame.filter(
                (pl.col("pyChannel") == row["pyChannel"]) & (pl.col("pyIssue") == row["pyIssue"])
            )["pyOutcome"]
            den = labels.eq("Impression").sum()
            assert row["CTR"] == pytest.approx(labels.eq("Clicked").sum() / den if den else 0)
        assert result["rows"]
