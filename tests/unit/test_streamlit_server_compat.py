"""Compatibility contracts for Streamlit's Starlette server integration."""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from streamlit.web.server.starlette.starlette_app import create_streamlit_middleware


@pytest.mark.unit
def test_streamlit_gzip_middleware_handles_gzip_request() -> None:
    """Exercise HTTP compression with the middleware stack used by the Streamlit server."""
    payload = "healthy" * 2048

    async def health(_request: Request) -> PlainTextResponse:
        return PlainTextResponse(payload)

    app = Starlette(routes=[Route("/healthz", health)], middleware=create_streamlit_middleware())
    with TestClient(app) as client:
        response = client.get("/healthz", headers={"Accept-Encoding": "gzip"})

    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip"
    assert response.text == payload
