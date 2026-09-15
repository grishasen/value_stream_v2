"""Compatibility contracts for Streamlit's Starlette server integration."""

from __future__ import annotations

import asyncio
import gzip

import pytest
from starlette.types import Message, Receive, Scope, Send
from streamlit.web.server.starlette.starlette_gzip_middleware import (
    MediaAwareGZipMiddleware,
)


@pytest.mark.unit
def test_streamlit_gzip_middleware_handles_gzip_request() -> None:
    """Exercise the ASGI path that broke when Starlette 1.4 changed its responder API."""
    payload = b"healthy" * 128
    messages: list[Message] = []

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/plain"),
                    (b"content-length", str(len(payload)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
        "scheme": "http",
        "method": "GET",
        "root_path": "",
        "path": "/healthz",
        "raw_path": b"/healthz",
        "query_string": b"",
        "headers": [(b"accept-encoding", b"gzip")],
        "state": {},
    }

    middleware = MediaAwareGZipMiddleware(app, minimum_size=1)
    asyncio.run(middleware(scope, receive, send))

    response_headers = dict(messages[0]["headers"])
    assert response_headers[b"content-encoding"] == b"gzip"
    assert gzip.decompress(messages[1]["body"]) == payload
