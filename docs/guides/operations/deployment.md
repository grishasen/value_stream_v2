# Deployment

Value Stream currently targets **single-host deployment**: one machine (or
container) runs ingestion, the Streamlit UI, and optionally the read-only API
and MCP surfaces against workspace directories on local or mounted storage.
Distributed execution, multi-user auth, and remote HTTP MCP are out of scope
or deferred — see [Product overview](../../concepts/product-overview.md).

## Host Requirements

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).
- Disk for workspace directories (source files, aggregates, metadata). The
  aggregate store is compact relative to raw data, but source files dominate
  until they are archived.
- Install with all extras (`ai` for MCP, `api` for FastAPI):

```sh
uv sync --all-extras
```

## Workspace Placement

Place each workspace on storage the host can read and write, e.g.
`/data/valuestream/<workspace>`. Keep `catalog/` under source control and
deploy catalog changes through the normal validate-then-run loop. The
`data/`, `aggregates/`, and `meta/` folders are runtime state.

## Scheduled Ingestion

`valuestream run` is cron-friendly: it exits non-zero on failure, skips
already-processed chunks, and is safe to re-run. A minimal cron entry:

```text
15 * * * * cd /opt/value_stream && uv run valuestream run /data/valuestream/prod >> /var/log/valuestream/run.log 2>&1
```

Schedule `vacuum` less frequently (e.g. daily) and only after report checks:

```text
30 3 * * * cd /opt/value_stream && uv run valuestream vacuum /data/valuestream/prod >> /var/log/valuestream/vacuum.log 2>&1
```

## Serving the UI

```sh
uv run valuestream serve /data/valuestream/prod --port 8501 --headless
```

Streamlit itself provides no authentication. For anything beyond a trusted
network, put the UI behind a reverse proxy that terminates TLS and enforces
access (e.g. nginx with SSO), and treat the host as single-tenant.

## Serving the API and MCP

```sh
export VALUESTREAM_API_TOKEN=replace-me
uv run valuestream serve-api /data/valuestream/prod --host 127.0.0.1 --port 8000
uv run valuestream serve-mcp /data/valuestream/prod
```

- Prefer a loopback bind with a reverse proxy in front; a non-loopback bind
  requires a bearer token (see [Security](security.md)).
- Add `--enable-sql` only when governed SQL is genuinely needed.
- MCP is stdio-based and intended for local clients (e.g. Claude Code) on the
  same host.

Long-lived API/MCP servers reload the catalog automatically when its YAML
files change on disk, so catalog deploys do not require a restart.

## Process Supervision

Run `serve`/`serve-api` under a supervisor (systemd, supervisor, or a
container orchestrator) with restart-on-failure. A minimal systemd unit:

```ini
[Unit]
Description=Value Stream UI
After=network.target

[Service]
WorkingDirectory=/opt/value_stream
ExecStart=/usr/local/bin/uv run valuestream serve /data/valuestream/prod --port 8501 --headless
Restart=on-failure
User=valuestream

[Install]
WantedBy=multi-user.target
```

## Publishing the Docs Site

```sh
uv run mkdocs build --strict   # gate
uv run mkdocs serve            # local preview
```

CI deploys the MkDocs site via the `docs.yml` GitHub Actions workflow on
pushes to `main`.

## Deferred

Remote HTTP MCP, OIDC/multi-user deployment, and hosted service operation are
deferred. When they land, this page gains the corresponding sections; until
then, treat every deployment as single-tenant and network-restricted.
