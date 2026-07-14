"""Integration tests for the read-only HTTP API over the demo workspace."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from valuestream.api import create_app


@pytest.fixture(scope="module")
def client(demo_workspace: Path) -> TestClient:
    return TestClient(create_app(demo_workspace))


@pytest.fixture(scope="module")
def sql_client(demo_workspace: Path) -> TestClient:
    return TestClient(create_app(demo_workspace, enable_sql=True))


@pytest.mark.integration
def test_health_is_open_without_token(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.integration
def test_metrics_manifest_lists_metrics_and_chart_kinds(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    names = {metric["name"] for metric in body["metrics"]}
    assert "VS_Engagement_Rate" in names
    engagement = next(m for m in body["metrics"] if m["name"] == "VS_Engagement_Rate")
    assert "kpi_card" in engagement["chart_kinds"]


@pytest.mark.integration
def test_metric_query_supports_operator_filters_and_top_n(client: TestClient) -> None:
    response = client.post(
        "/metrics/VS_Interactions/query",
        json={
            "group_by": ["Channel"],
            "grain": "summary",
            "order_by": ["-VS_Interactions"],
            "top_n": 2,
            "top_n_by": "VS_Interactions",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["rows"]) == 2
    values = [row["VS_Interactions"] for row in body["rows"]]
    assert values == sorted(values, reverse=True)
    assert body["provenance"]["processor_id"] == "ih_engagement"
    assert body["provenance"]["stored_grain"] == "daily"
    assert body["provenance"]["chunk_ids"]


@pytest.mark.integration
def test_metric_query_reports_bad_request_for_unknown_dimension(client: TestClient) -> None:
    response = client.post(
        "/metrics/VS_Interactions/query",
        json={"group_by": ["CustomerID"], "grain": "summary"},
    )
    assert response.status_code == 400
    assert "CustomerID" in response.json()["detail"]


@pytest.mark.integration
def test_metric_chart_returns_validated_chart_spec(client: TestClient) -> None:
    response = client.post(
        "/metrics/VS_Engagement_Rate/chart",
        json={
            "chart_kind": "bar",
            "x": "Channel",
            "y": "VS_Engagement_Rate",
            "group_by": ["Channel"],
            "grain": "summary",
            "value_format": "percent",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chart"]["kind"] == "bar"
    assert body["chart"]["value_format"] == "percent"
    assert body["rows"]


@pytest.mark.integration
def test_sql_endpoints_are_disabled_by_default(client: TestClient) -> None:
    assert client.get("/sql/schema").status_code == 404
    assert client.post("/sql", json={"sql": "SELECT 1"}).status_code == 404


@pytest.mark.integration
def test_governed_sql_endpoint_masks_and_caps(sql_client: TestClient) -> None:
    schema = sql_client.get("/sql/schema")
    assert schema.status_code == 200
    tables = {table["name"] for table in schema.json()["tables"]}
    assert any("ih_engagement" in name for name in tables)

    view = next(name for name in sorted(tables) if "ih_engagement" in name)
    response = sql_client.post(
        "/sql",
        json={"sql": f"SELECT Channel, SUM(Count) AS total FROM {view} GROUP BY Channel"},
    )
    assert response.status_code == 200
    assert response.json()["rows"]

    rejected = sql_client.post("/sql", json={"sql": "DROP TABLE x"})
    assert rejected.status_code == 400


@pytest.mark.integration
def test_governed_sql_endpoint_cannot_read_source_files(
    sql_client: TestClient,
    demo_workspace: Path,
) -> None:
    raw_source = next((demo_workspace / "data").glob("*.parquet"))
    response = sql_client.post("/sql", json={"sql": f"SELECT * FROM '{raw_source}'"})

    assert response.status_code == 400
    assert "file system operations are disabled" in response.json()["detail"]


@pytest.mark.integration
def test_chat_endpoint_requires_configured_model(demo_workspace: Path) -> None:
    # The demo ai.yaml points at a local Ollama model that is not running in CI;
    # disabling chat lets the rest of the API serve without an LLM.
    client = TestClient(create_app(demo_workspace, enable_chat=False))
    assert client.post("/chat", json={"question": "hi"}).status_code == 404


@pytest.mark.integration
def test_bearer_token_guards_endpoints(demo_workspace: Path) -> None:
    client = TestClient(create_app(demo_workspace, api_token="secret"))

    assert client.get("/health").status_code == 200  # health is always open
    assert client.get("/metrics").status_code == 401
    ok = client.get("/metrics", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
