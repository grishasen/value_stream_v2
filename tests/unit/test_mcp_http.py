"""HTTP authentication and transport boundaries without starting a listener."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from valuestream.mcp.http import create_http_app, run_http

DEMO_WS = Path(__file__).resolve().parents[2] / "examples" / "demo"


@pytest.mark.unit
def test_http_uses_one_worker_even_with_web_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(app: Any, **options: Any) -> None:
        captured.update(options)

    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    monkeypatch.setattr("uvicorn.run", fake_run)
    run_http(DEMO_WS, token="secret", port=9000)
    assert captured == {"host": "127.0.0.1", "port": 9000, "workers": 1}


@pytest.mark.unit
@pytest.mark.parametrize("token", ["", " ", "secret\nvalue"])
def test_http_requires_valid_token(token: str) -> None:
    with pytest.raises(ValueError, match="requires a non-empty bearer token"):
        create_http_app(DEMO_WS, token=token)


@pytest.mark.unit
def test_http_requires_token_from_argument_or_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VALUESTREAM_MCP_TOKEN", raising=False)
    with pytest.raises(ValueError, match="VALUESTREAM_MCP_TOKEN"):
        create_http_app(DEMO_WS)


@pytest.mark.unit
@pytest.mark.parametrize("host", ["0.0.0.0", "::"])
def test_wildcard_bind_requires_public_host_allowlist(host: str) -> None:
    with pytest.raises(ValueError, match="--allowed-host"):
        create_http_app(DEMO_WS, token="secret", host=host)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["GET", "POST", "DELETE", "OPTIONS"])
@pytest.mark.parametrize("authorization", [None, "Bearer wrong", "Basic secret"])
def test_auth_precedes_protocol_handling(method: str, authorization: str | None) -> None:
    app = create_http_app(DEMO_WS, token="secret")
    headers = {} if authorization is None else {"Authorization": authorization}
    with TestClient(app, base_url="http://localhost") as client:
        response = client.request(method, "/mcp", headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Bearer realm="valuestream-mcp"'
    assert response.json() == {"detail": "invalid or missing bearer token"}


@pytest.mark.unit
def test_environment_token_and_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VALUESTREAM_MCP_TOKEN", "environment-secret")
    for token, expected in [(None, "environment-secret"), ("explicit-secret", "explicit-secret")]:
        app = create_http_app(DEMO_WS, token=token)
        with TestClient(app, base_url="http://localhost") as client:
            # An authenticated empty POST reaches the SDK's content-type check.
            assert (
                client.post("/mcp", headers={"Authorization": f"Bearer {expected}"}).status_code
                == 400
            )
            assert client.post("/mcp", headers={"Authorization": "Bearer wrong"}).status_code == 401


@pytest.mark.unit
def test_duplicate_authorization_headers_are_rejected() -> None:
    with TestClient(
        create_http_app(DEMO_WS, token="secret"), base_url="http://localhost"
    ) as client:
        response = client.post(
            "/mcp", headers=[("Authorization", "Bearer secret"), ("Authorization", "Bearer secret")]
        )
    assert response.status_code == 401


@pytest.mark.unit
def test_bearer_scheme_is_case_insensitive_but_token_is_not() -> None:
    with TestClient(
        create_http_app(DEMO_WS, token="Secret"), base_url="http://localhost"
    ) as client:
        assert client.post("/mcp", headers={"Authorization": "bearer Secret"}).status_code == 400
        assert client.post("/mcp", headers={"Authorization": "Bearer secret"}).status_code == 401


@pytest.mark.unit
@pytest.mark.parametrize(
    ("headers", "expected"),
    [({"Host": "untrusted.example"}, 421), ({"Origin": "https://untrusted.example"}, 403)],
)
def test_http_rejects_untrusted_host_and_origin(headers: dict[str, str], expected: int) -> None:
    with TestClient(
        create_http_app(DEMO_WS, token="secret"), base_url="http://localhost"
    ) as client:
        response = client.post(
            "/mcp", headers={"Authorization": "Bearer secret", **headers}, json={}
        )
    assert response.status_code == expected


@pytest.mark.unit
def test_public_host_and_origin_can_be_explicitly_allowed() -> None:
    app = create_http_app(
        DEMO_WS,
        token="secret",
        host="0.0.0.0",
        allowed_hosts=("metrics.example.com",),
        allowed_origins=("https://client.example.com",),
    )
    with TestClient(app, base_url="https://metrics.example.com") as client:
        response = client.post(
            "/mcp",
            headers={"Authorization": "Bearer secret", "Origin": "https://client.example.com"},
            json={},
        )
    # Invalid JSON-RPC reaches protocol validation after passing both allowlists.
    assert response.status_code == 400
