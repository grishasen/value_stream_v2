"""Bearer-protected Streamable HTTP transport for one workspace."""

from __future__ import annotations

import hmac
import os
from importlib import import_module
from pathlib import Path

from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from valuestream.mcp.server import create_server

TOKEN_ENV = "VALUESTREAM_MCP_TOKEN"
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]")


class BearerTokenMiddleware:
    """Authenticate every HTTP request without buffering MCP response streams."""

    def __init__(self, app: ASGIApp, *, token: str) -> None:
        self.app = app
        self.expected = token.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = Headers(scope=scope)
            authorization = headers.getlist("authorization")
            scheme, _, credential = (authorization[0] if len(authorization) == 1 else "").partition(
                " "
            )
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                credential.encode(), self.expected
            ):
                response = JSONResponse(
                    {"detail": "invalid or missing bearer token"},
                    status_code=401,
                    headers={"WWW-Authenticate": 'Bearer realm="valuestream-mcp"'},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_http_app(
    workspace_path: str | Path,
    *,
    token: str | None = None,
    host: str = "127.0.0.1",
    allowed_hosts: tuple[str, ...] = (),
    allowed_origins: tuple[str, ...] = (),
    enable_sql: bool = False,
    render_dir: str | Path | None = None,
) -> Starlette:
    """Build /mcp with mandatory token auth and explicit Host/Origin checks.

    The SDK owns protocol handling and lifespan. Stateless HTTP avoids shared
    client sessions; all clients read the same fixed workspace and catalog.
    """

    effective_token = token if token is not None else os.environ.get(TOKEN_ENV, "")
    if not effective_token or any(char.isspace() for char in effective_token):
        raise ValueError(f"HTTP MCP requires a non-empty bearer token (--token or {TOKEN_ENV})")
    if host in {"0.0.0.0", "::"} and not allowed_hosts:
        raise ValueError("a wildcard HTTP MCP bind requires at least one --allowed-host")

    bind_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    hosts = list(_LOOPBACK_HOSTS)
    if host not in {"0.0.0.0", "::"}:
        hosts.append(bind_host)
    security_cls = import_module("mcp.server.transport_security").TransportSecuritySettings
    security = security_cls(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(
            dict.fromkeys([*hosts, *(f"{item}:*" for item in hosts), *allowed_hosts])
        ),
        allowed_origins=list(allowed_origins),
    )
    server = create_server(workspace_path, enable_sql=enable_sql, render_dir=render_dir)
    # These options are the MCP SDK 2.x API pinned by the project's ai extra.
    app: Starlette = server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        host=host,
        transport_security=security,
    )
    app.add_middleware(BearerTokenMiddleware, token=effective_token)
    return app


def run_http(
    workspace_path: str | Path,
    *,
    token: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8001,
    allowed_hosts: tuple[str, ...] = (),
    allowed_origins: tuple[str, ...] = (),
    enable_sql: bool = False,
    render_dir: str | Path | None = None,
) -> None:
    """Serve the SDK ASGI app in one Uvicorn worker."""

    app = create_http_app(
        workspace_path,
        token=token,
        host=host,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
        enable_sql=enable_sql,
        render_dir=render_dir,
    )
    import_module("uvicorn").run(app, host=host, port=port, workers=1)
