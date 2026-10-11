# ADR 0009 — Single-Workspace HTTP MCP

**Status:** Accepted (2026-10-09)

## Context

The MCP surface used stdio, requiring the client and workspace to share a
host. Remote clients need the same governed tools without a local checkout or
access to source files. The existing REST API is not an MCP endpoint.

## Decision

- Retain stdio as the default and add Streamable HTTP at `/mcp` using the
  pinned Python MCP SDK, served by one Uvicorn worker.
- Build tools, resources, and prompts once through `create_server`; both
  transports use that factory and the existing aggregate query layer.
- Fix the workspace at process startup. Requests cannot select another
  workspace, and all holders of the server token have the same permissions.
- Require a configured bearer token on every HTTP request, even on loopback.
  Use `VALUESTREAM_MCP_TOKEN` or `--token`, constant-time token comparison,
  and an ASGI middleware that preserves streaming and SDK lifespan handling.
- Use SDK stateless HTTP with JSON responses. Do not maintain application
  sessions or introduce OAuth discovery, accounts, or per-user permissions.
- Keep SDK Host/Origin validation enabled. Allow loopback/concrete bind hosts
  and explicit public hosts; reject Origin headers unless explicitly allowed.
  Wildcard binds require an explicit allowed host. Terminate remote HTTPS
  at a reverse proxy, normally with a loopback backend.
- MCP has no built-in `chat` or server-side LLM calls. The client interprets
  natural-language questions and selects deterministic analytics tools.
- Keep SQL opt-in and preserve provenance, output bounds, catalog reload,
  aggregate-only reads, and all existing tool contracts. HTML render paths
  remain server-local; remote clients receive PNG blocks or Plotly specs.

## Consequences

- A remote MCP client needs a URL and support for a configured bearer header,
  rather than a local command. No MCP message handling is reimplemented.
- One compromised token grants the complete configured workspace surface,
  including catalog resources and any explicitly enabled SQL. Token rotation
  requires replacing the secret and restarting the service and clients.
- HTTP MCP is a separate application from REST; REST authentication does not
  implicitly protect it. HTTP MCP has no unauthenticated health endpoint.
- Multiple workspaces require separate processes, endpoints, and tokens.
  Multi-user/OIDC permissions and remote HTML hosting remain deferred.
- Integration tests cover the actual HTTP listener/client, stdio regression,
  authentication, tool parity, concurrent clients, and server/port cleanup.
