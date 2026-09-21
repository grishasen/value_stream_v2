"""Exercise the actual stdio protocol, including SDK result/schema conversion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stdio_discovery_errors_and_stateful_rendering(
    demo_workspace: Path,
    tmp_path: Path,
) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from valuestream.cli import main; main()", "serve-mcp", str(demo_workspace)],
        env={"LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    # Both contexts own and close their child processes/streams even on failure.
    with (tmp_path / "mcp-stderr.log").open("w") as stderr:
        async with stdio_client(params, errlog=stderr) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=30) as client:
                await client.initialize()
                listing = await client.list_tools()
                assert "sql_query" not in {tool.name for tool in listing.tools}
                assert len((await client.list_resources()).resources) == 2
                assert len((await client.list_prompts()).prompts) == 2
                error = (await client.call_tool("metric_query", {"metric": "missing"})).model_dump(
                    by_alias=True
                )
                assert error["isError"] is True
                assert "metric_list" in error["structuredContent"]["error"]["remediation"]
                invalid = (await client.call_tool("metric_list", {"offset": -1})).model_dump(
                    by_alias=True
                )
                assert invalid["isError"] is True
                page = (await client.call_tool("metric_list", {"limit": 1})).model_dump(
                    by_alias=True
                )
                assert page["isError"] is False
                assert page["structuredContent"]["next_offset"] == 1
                histogram = (
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
                assert histogram["isError"] is False
                payload = histogram["structuredContent"]
                assert payload["render"]["spec"]["data"]
                assert "ResponseTime_tdigest" not in payload["columns"]
                assert json.loads(histogram["content"][0]["text"]) == payload
