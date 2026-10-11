"""Exercise the HTTP CLI with the actual MCP client and a TCP listener."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextResourceContents

from valuestream.mcp.server import create_server

TOKEN = "integration-test-secret"


@pytest.fixture(params=[False, True], ids=["no-sql", "sql"])
def http_server(
    request: pytest.FixtureRequest, demo_workspace: Path, tmp_path: Path
) -> Iterator[tuple[str, bool]]:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    args = [
        sys.executable,
        "-c",
        "from valuestream.cli import main; main()",
        "serve-mcp",
        str(demo_workspace),
        "--transport",
        "http",
        "--port",
        str(port),
    ]
    if request.param:
        args.append("--enable-sql")
    env = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True", "VALUESTREAM_MCP_TOKEN": TOKEN}
    log_path = tmp_path / "http-mcp.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(args, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(f"HTTP MCP exited during startup:\n{log_path.read_text()}")
                with socket.socket() as probe:
                    if probe.connect_ex(("127.0.0.1", port)) == 0:
                        break
                time.sleep(0.05)
            else:
                pytest.fail(f"HTTP MCP did not start:\n{log_path.read_text()}")
            yield f"http://127.0.0.1:{port}/mcp", bool(request.param)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            # Verify that teardown released the listening port, even after a failed test.
            with socket.socket() as probe:
                assert probe.connect_ex(("127.0.0.1", port)) != 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_http_protocol_matches_shared_tools(
    http_server: tuple[str, bool], demo_workspace: Path
) -> None:
    url, enable_sql = http_server
    reference = create_server(demo_workspace, enable_sql=enable_sql)
    async with (
        asyncio.timeout(30),
        httpx2.AsyncClient(headers={"Authorization": f"Bearer {TOKEN}"}) as http,
        streamable_http_client(url, http_client=http) as (read, write),
        ClientSession(read, write, read_timeout_seconds=20) as client,
    ):
        await client.initialize()
        listing = await client.list_tools()
        assert "chat" not in {tool.name for tool in listing.tools}
        removed = await client.call_tool("chat", {"question": "Show CTR by channel"})
        assert removed.is_error
        assert {tool.name for tool in listing.tools} == {
            tool.name for tool in await reference.list_tools()
        }
        assert ("sql_query" in {tool.name for tool in listing.tools}) is enable_sql
        assert len((await client.list_resources()).resources) == 2
        assert len((await client.list_prompts()).prompts) == 2
        catalog = await client.read_resource("valuestream://catalog")
        assert isinstance(catalog.contents[0], TextResourceContents)
        assert "catalog_version" in catalog.contents[0].text

        arguments = {"metric": "VS_Interactions", "group_by": ["Channel"]}
        result = (await client.call_tool("metric_query", arguments)).model_dump(by_alias=True)
        expected = (await reference.call_tool("metric_query", arguments)).model_dump(by_alias=True)
        assert result == expected
        assert result["structuredContent"]["rows"]
        assert result["structuredContent"]["provenance"]["processor_id"] == "ih_engagement"

        error = (await client.call_tool("metric_query", {"metric": "missing"})).model_dump(
            by_alias=True
        )
        assert error["isError"] is True
        assert "metric_list" in error["structuredContent"]["error"]["remediation"]
        chart = (
            await client.call_tool(
                "metric_chart_query",
                {
                    "metric": "VS_ResponseTime_Distribution",
                    "chart_kind": "histogram",
                    "x": "Outcome",
                    "y": "VS_ResponseTime_Distribution",
                    "group_by": ["Outcome"],
                    "chart_fields": {"property": "ResponseTime"},
                    "render": "spec",
                },
            )
        ).model_dump(by_alias=True)
        assert chart["isError"] is False
        assert chart["structuredContent"]["render"]["spec"]["data"]
        assert "ResponseTime_tdigest" not in chart["structuredContent"]["columns"]
        trend = await client.call_tool(
            "metric_chart_query",
            {
                "metric": "VS_Engagement_Rate",
                "chart_kind": "line",
                "x": "Day",
                "y": "VS_Engagement_Rate",
                "group_by": ["Day", "Channel"],
                "color": "Channel",
                "render": "spec",
            },
        )
        assert not trend.is_error
        assert trend.structured_content["grain"] == "daily"
        assert "Day" in trend.structured_content["columns"]
        assert trend.structured_content["render"]["spec"]["data"]
        if enable_sql:
            schema = await client.call_tool("sql_schema", {})
            assert not schema.is_error
            assert schema.structured_content["tables"]
            sql = await client.call_tool("sql_query", {"sql": "SELECT 1 AS value"})
            assert not sql.is_error
            assert sql.structured_content["rows"] == [{"value": 1}]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_http_clients_are_independent_and_every_request_is_authenticated(
    http_server: tuple[str, bool],
) -> None:
    url, _enable_sql = http_server

    async def query() -> dict[str, Any]:
        async with (
            httpx2.AsyncClient(headers={"Authorization": f"Bearer {TOKEN}"}) as http,
            streamable_http_client(url, http_client=http) as (read, write),
            ClientSession(read, write, read_timeout_seconds=20) as client,
        ):
            await client.initialize()
            return (await client.call_tool("metric_list", {"limit": 1})).model_dump(by_alias=True)

    async with asyncio.timeout(30):
        first, second = await asyncio.gather(query(), query())
        assert first == second
        assert first["structuredContent"]["returned_metrics"] == 1
        async with httpx2.AsyncClient() as http:
            for method in ("GET", "POST", "DELETE"):
                denied = await http.request(method, url, headers={"Authorization": "Bearer wrong"})
                assert denied.status_code == 401
