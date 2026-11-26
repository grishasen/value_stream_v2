"""Real MCP clients bootstrap a catalog over both transports and query aggregates."""

from __future__ import annotations

import asyncio
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

from valuestream.config.loader import load
from valuestream.engine.runner import run_workspace


@asynccontextmanager
async def _client(transport: str, workspace: Path, log_path: Path) -> AsyncIterator[ClientSession]:
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
                initialized = await client.initialize()
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
                initialized = await client.initialize()
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
