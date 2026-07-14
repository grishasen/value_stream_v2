# Value Stream

Value Stream is a configuration-driven, aggregate-first business intelligence platform. It ingests file-based exports (typically Pega CDH Interaction History and Product Holdings), reduces them to small mergeable sufficient statistics during one chunk pass, and serves business reports and dashboards from those persisted aggregates. Raw event rows never survive the chunk pass.

## Documentation

The docs are the spec and live in [`docs/`](docs/index.md), organized by
tutorials, guides, concepts, and reference, with role-based reading paths on
the home page. Serve them locally with `uv run mkdocs serve`.

- [Documentation home](docs/index.md) — start here; pick your reading path
- [Getting started tutorial](docs/tutorials/getting-started.md) — clean clone to first queried metric
- [Architecture](docs/concepts/architecture.md) and [domain model](docs/concepts/domain-model.md)
- [Replacement design](docs/design/replacement-design.md) — master design (storage, DSL, APIs)
- [CLI reference](docs/reference/cli.md)

## Quickstart

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```sh
uv sync --all-extras
uv run valuestream --help
uv run valuestream validate examples/demo
```

`examples/demo` ships its catalog only; the
[getting started tutorial](docs/tutorials/getting-started.md) shows how to
generate demo data, run ingestion, query metrics, and serve the UI:

```sh
uv run valuestream run examples/demo
uv run valuestream query examples/demo VS_Engagement_Rate --by Channel --grain Day
uv run valuestream serve examples/demo --port 8501 --headless
```

## Tests and Quality

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -m "not bench and not slow"
uv run mkdocs build --strict
```

## License

Proprietary. See repository owner.
