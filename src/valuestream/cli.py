"""``valuestream`` CLI entry point."""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from valuestream import __version__
from valuestream.config import model as config_model
from valuestream.config.loader import CatalogLoadError, load
from valuestream.config.migration import migrate_toml
from valuestream.config.validate import CatalogIssue, CatalogValidationResult, validate_catalog
from valuestream.engine import PipelineRunResult, ledger, probe_source, run_source, run_workspace
from valuestream.generators import PegaDummyGenerationConfig, generate_pega_dummy_data
from valuestream.mcp.server import run_stdio as run_mcp_stdio
from valuestream.query import query_metric
from valuestream.store.backfill import backfill_from_legacy_db
from valuestream.store.duckdb_export import export_metric_tables_to_duckdb, metric_export_db_path
from valuestream.store.duckdb_views import refresh_aggregate_views
from valuestream.store.vacuum import vacuum_workspace
from valuestream.utils import logger as log_utils


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="valuestream")
@click.option(
    "--logging-config",
    "--logging_config",
    type=click.Path(exists=True, dir_okay=False),
    help="Logging config YAML file. Uses the bundled config when omitted.",
)
def main(logging_config: str | None) -> None:
    """Value Stream — aggregate-first business intelligence platform."""
    log_utils.configure(config_path=logging_config)


@main.command()
@click.argument("workspace_dir", type=click.Path(exists=True, file_okay=False))
def validate(workspace_dir: str) -> None:
    """Validate a workspace catalog.

    Loads the catalog from ``<workspace_dir>/catalog/``, validates each
    YAML against its JSON Schema, type-checks every expression in
    transforms / processors / metrics, and prints structured success or
    errors. Exit code 0 on success, 1 on failure.
    """
    console = Console()

    try:
        catalog = load(workspace_dir)
    except CatalogLoadError as exc:
        print(traceback.format_exc())
        console.print(f"[bold red]error[/]: {exc}")
        if exc.file is not None:
            console.print(f"[dim]file: {exc.file}[/]")
        sys.exit(1)

    result = validate_catalog(catalog)
    _render_result(console, workspace_dir, catalog, result)
    sys.exit(0 if result.ok else 1)


def _render_result(
    console: Console,
    workspace_dir: str,
    catalog: object,  # ``model.Catalog`` — kept loose to avoid an extra import here
    result: CatalogValidationResult,
) -> None:
    """Render a structured pass/fail summary."""
    if result.ok and not result.issues:
        console.print(f"[bold green]ok[/] — {workspace_dir} validates clean.")
        _render_summary(console, catalog)
        return

    if result.ok:
        # Warnings only.
        console.print(
            f"[bold yellow]ok with warnings[/] — {workspace_dir} validates "
            f"({len(result.issues)} warning(s))."
        )
    else:
        errors = [i for i in result.issues if i.severity == "error"]
        warnings = [i for i in result.issues if i.severity == "warning"]
        console.print(
            f"[bold red]failed[/] — {workspace_dir}: "
            f"{len(errors)} error(s), {len(warnings)} warning(s)."
        )

    _render_issues(console, result.issues)


def _render_summary(console: Console, catalog: object) -> None:
    if not isinstance(catalog, config_model.Catalog):
        return
    table = Table(title="catalog summary", show_header=True, header_style="bold")
    table.add_column("kind")
    table.add_column("count", justify="right")
    table.add_row("sources", str(len(catalog.pipelines.sources)))
    table.add_row("processors", str(len(catalog.processors.processors)))
    table.add_row("metrics", str(len(catalog.metrics.metrics)))
    table.add_row("dashboards", str(len(catalog.dashboards.dashboards)))
    tile_count = sum(len(page.tiles) for d in catalog.dashboards.dashboards for page in d.pages)
    table.add_row("tiles", str(tile_count))
    console.print(table)


def _render_issues(console: Console, issues: list[CatalogIssue]) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("severity", width=8)
    table.add_column("location")
    table.add_column("message")
    for it in issues:
        color = "red" if it.severity == "error" else "yellow"
        table.add_row(f"[{color}]{it.severity}[/]", it.location, it.message)
    console.print(table)


@main.command()
@click.argument("workspace_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("source_id", required=False)
@click.option("--force", is_flag=True, help="Process chunks even if the ledger says they are done.")
@click.option(
    "--parallel",
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of chunk worker processes.",
)
def run(workspace_dir: str, source_id: str | None, *, force: bool, parallel: int) -> None:
    """Run Phase 1 ingestion for one source, or all sources when omitted."""
    console = Console()
    try:
        if source_id is None:
            workspace_result = run_workspace(workspace_dir, force=force, parallel=parallel)
            console.print(
                f"[bold green]{workspace_result.status}[/] workspace run: "
                f"{workspace_result.sources_ok} ok, {workspace_result.sources_partial} partial, "
                f"{workspace_result.sources_failed} failed across "
                f"{workspace_result.sources_total} source(s)."
            )
            summary = Table(show_header=True, header_style="bold")
            summary.add_column("source")
            summary.add_column("status")
            summary.add_column("ok", justify="right")
            summary.add_column("skipped", justify="right")
            summary.add_column("failed", justify="right")
            for source_result in workspace_result.results:
                summary.add_row(
                    source_result.source_id,
                    source_result.status,
                    str(source_result.chunks_ok),
                    str(source_result.chunks_skipped),
                    str(source_result.chunks_failed),
                )
            console.print(summary)
            return
        result = run_source(workspace_dir, source_id, force=force, parallel=parallel)
    except Exception as exc:
        print(traceback.format_exc())
        raise click.ClickException(str(exc)) from exc

    _render_run_details(console, result)


def _render_run_details(console: Console, result: PipelineRunResult) -> None:
    console.print(
        f"[bold green]{result.status}[/] run {result.run_id} for source "
        f"{result.source_id}: {result.chunks_ok} ok, {result.chunks_skipped} skipped, "
        f"{result.chunks_failed} failed."
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("chunk")
    table.add_column("status")
    table.add_column("rows", justify="right")
    table.add_column("written", justify="right")
    table.add_column("time", justify="right")
    for chunk in result.chunks:
        table.add_row(
            chunk.chunk_id,
            chunk.status,
            str(chunk.rows_kept),
            str(len(chunk.written)),
            _format_elapsed_ms(chunk.elapsed_ms),
        )
    console.print(table)


@main.command()
@click.argument("workspace_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("metric_name")
@click.option("--by", "group_by", multiple=True, help="Column to group by.")
@click.option(
    "--where",
    "where_clauses",
    multiple=True,
    metavar="KEY=VALUE",
    help="Filter by column. Repeat or comma-separate values.",
)
@click.option("--grain", default="daily", show_default=True)
@click.option("--from", "start", help="Inclusive start date, YYYY-MM-DD.")
@click.option("--to", "end", help="Inclusive end date, YYYY-MM-DD.")
@click.option("--raw", is_flag=True, help="Include underlying aggregate state columns.")
def query(
    workspace_dir: str,
    metric_name: str,
    group_by: tuple[str, ...],
    where_clauses: tuple[str, ...],
    grain: str,
    start: str | None,
    end: str | None,
    raw: bool,
) -> None:
    """Query a Phase 1 metric from aggregate parquet."""
    console = Console()
    try:
        frame = query_metric(
            workspace_dir,
            metric_name,
            group_by=list(group_by),
            filters=_parse_where_clauses(where_clauses),
            grain=grain,
            start=start,
            end=end,
            include_state_columns=raw,
        )
    except Exception as exc:
        print(traceback.format_exc())
        raise click.ClickException(str(exc)) from exc
    console.print(frame)


def _parse_where_clauses(clauses: tuple[str, ...]) -> dict[str, str | list[str]]:
    filters: dict[str, str | list[str]] = {}
    for clause in clauses:
        key, sep, raw_value = clause.partition("=")
        key = key.strip()
        if not sep or not key:
            raise click.ClickException("--where must use KEY=VALUE")
        values = [value.strip() for value in raw_value.split(",") if value.strip()]
        if not values:
            raise click.ClickException("--where must include at least one value")
        filters[key] = values[0] if len(values) == 1 else values
    return filters


@main.command()
@click.argument("workspace_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("source_id")
@click.option("--limit", default=10, show_default=True, type=int, help="Sample rows to show.")
def probe(workspace_dir: str, source_id: str, *, limit: int) -> None:
    """Inspect a source after discovery and transforms."""
    console = Console()
    try:
        result = probe_source(workspace_dir, source_id, limit=limit)
    except Exception as exc:
        print(traceback.format_exc())
        raise click.ClickException(str(exc)) from exc

    console.print(
        f"[bold green]ok[/] — source {source_id}: {result.chunk_count} chunk(s), "
        f"{result.file_count} file(s)."
    )
    if result.calendar_columns:
        console.print("calendar columns: " + ", ".join(result.calendar_columns))
    schema = Table(title="schema", show_header=True, header_style="bold")
    schema.add_column("column")
    schema.add_column("dtype")
    for name, dtype in result.schema:
        schema.add_row(name, dtype)
    console.print(schema)
    if result.sample.height:
        console.print(result.sample)


@main.command()
@click.argument("workspace_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--tmp/--no-tmp", "include_tmp", default=True, show_default=True)
@click.option("--dry-run", is_flag=True, help="Report deletions without removing files.")
def vacuum(workspace_dir: str, *, include_tmp: bool, dry_run: bool) -> None:
    """Prune superseded aggregate files and orphan reader temp dirs."""
    console = Console()
    try:
        catalog = load(workspace_dir)
        with ExitStack() as locks:
            for source in sorted(catalog.pipelines.sources, key=lambda item: item.id):
                locks.enter_context(ledger.source_run_lock(workspace_dir, source.id))
            result = vacuum_workspace(
                workspace_dir,
                catalog,
                include_tmp=include_tmp,
                dry_run=dry_run,
            )
            if not dry_run:
                refresh_aggregate_views(workspace_dir, catalog)
    except Exception as exc:
        print(traceback.format_exc())
        raise click.ClickException(str(exc)) from exc
    verb = "would delete" if dry_run else "deleted"
    console.print(
        f"[bold green]ok[/] — {verb} {result.files_deleted} file(s), "
        f"{result.dirs_deleted} dir(s), {result.bytes_deleted} byte(s)."
    )


@main.command("export-duckdb")
@click.argument("workspace_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--grain", default="summary", show_default=True, help="Metric grain to export.")
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False),
    help="DuckDB file to create. Defaults to meta/metric_export_<grain>.duckdb.",
)
@click.option(
    "--replace/--no-replace",
    default=True,
    show_default=True,
    help="Replace the existing export database file before writing.",
)
def export_duckdb(
    workspace_dir: str, grain: str, output_path: str | None, *, replace: bool
) -> None:
    """Export one materialized DuckDB table per metric at a selected grain."""
    console = Console()
    try:
        catalog = load(workspace_dir)
        target = output_path or str(metric_export_db_path(workspace_dir, grain))
        result = export_metric_tables_to_duckdb(
            workspace_dir,
            catalog,
            grain=grain,
            output_path=target,
            replace=replace,
        )
    except Exception as exc:
        print(traceback.format_exc())
        raise click.ClickException(str(exc)) from exc

    console.print(
        f"[bold green]ok[/] — exported {len(result.tables)} metric table(s), "
        f"{result.rows} row(s), grain `{result.grain}` to {result.path}."
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("table")
    table.add_column("source / processor")
    table.add_column("rows", justify="right")
    for item in result.tables:
        table.add_row(
            item.metric_name,
            item.table_name,
            f"{item.source_id}/{item.processor_id}",
            str(item.rows),
        )
    if result.tables:
        console.print(table)
    if result.skipped:
        skipped = Table(title="skipped metrics", show_header=True, header_style="bold")
        skipped.add_column("metric")
        skipped.add_column("reason")
        for skipped_metric in result.skipped:
            skipped.add_row(skipped_metric.metric_name, skipped_metric.reason)
        console.print(skipped)


@main.command()
@click.argument("workspace_dir", type=click.Path(file_okay=False))
@click.option("--port", default=8501, show_default=True, type=int)
@click.option("--browser/--headless", default=True, show_default=True)
def serve(workspace_dir: str, *, port: int, browser: bool) -> None:
    """Start the Phase 4 Streamlit dashboard UI for a workspace."""
    app_path = Path(__file__).parent / "ui" / "app.py"
    args = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        str(not browser).lower(),
        "--",
        "--workspace",
        workspace_dir,
    ]
    env = dict(os.environ)
    # Keep Arrow off its bundled mimalloc: it segfaults in per-thread heap
    # init on macOS arm64 under Streamlit's thread-per-rerun model. app.py
    # sets the same default for direct `streamlit run` launches.
    env.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
    raise SystemExit(subprocess.run(args, check=False, env=env).returncode)


@main.command("serve-mcp")
@click.argument("workspace_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--enable-sql",
    is_flag=True,
    help="Expose governed aggregate SQL tools. Disabled by default.",
)
def serve_mcp(workspace_dir: str, *, enable_sql: bool) -> None:
    """Start the read-only MCP server over stdio for a workspace."""

    try:
        run_mcp_stdio(workspace_dir, enable_sql=enable_sql)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("serve-api")
@click.argument("workspace_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option(
    "--token",
    default=None,
    help="Bearer token required on every request. Defaults to $VALUESTREAM_API_TOKEN.",
)
@click.option(
    "--enable-sql",
    is_flag=True,
    help="Expose governed aggregate SQL endpoints. Disabled by default.",
)
def serve_api(
    workspace_dir: str,
    *,
    host: str,
    port: int,
    token: str | None,
    enable_sql: bool,
) -> None:
    """Start the read-only HTTP API server for a workspace."""

    try:
        import uvicorn  # noqa: PLC0415 — optional `api` extra imported lazily

        from valuestream.api import create_app  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise click.ClickException(
            "The API server requires the optional `api` dependencies. "
            "Install them with `uv sync --extra api` or `uv sync --all-extras`."
        ) from exc

    effective_token = token or os.environ.get("VALUESTREAM_API_TOKEN", "")
    if host not in {"127.0.0.1", "localhost", "::1"} and not effective_token:
        raise click.ClickException(
            "a bearer token is required when serve-api binds to a non-loopback host"
        )

    try:
        app = create_app(workspace_dir, api_token=effective_token, enable_sql=enable_sql)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    uvicorn.run(app, host=host, port=port)


@main.command("generate-pega-dummy")
@click.option(
    "--source",
    "source_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Pega JSON/NDJSON export, or zip/gzip/tar.gz archive containing JSON records.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory where one Parquet file per generated day will be written.",
)
@click.option("--start-date", required=True, help="Inclusive start date, YYYY-MM-DD.")
@click.option("--end-date", help="Inclusive end date, YYYY-MM-DD. Mutually exclusive with --days.")
@click.option(
    "--days", type=int, help="Number of days to generate. Mutually exclusive with --end-date."
)
@click.option("--rows-per-day", default=1_000_000, show_default=True, type=int)
@click.option("--batch-size", default=100_000, show_default=True, type=int)
@click.option("--customer-count", default=250_000, show_default=True, type=int)
@click.option(
    "--positive-rate",
    default=0.12,
    show_default=True,
    type=float,
    help="Share of generated rows with pyOutcome=Clicked.",
)
@click.option("--seed", default=13, show_default=True, type=int)
@click.option("--file-prefix", default="pega_interactions", show_default=True)
@click.option("--compression", default="zstd", show_default=True)
@click.option(
    "--overwrite", is_flag=True, help="Replace existing generated files for the same days."
)
def generate_pega_dummy(
    source_path: str,
    output_dir: str,
    start_date: str,
    end_date: str | None,
    days: int | None,
    rows_per_day: int,
    batch_size: int,
    customer_count: int,
    positive_rate: float,
    seed: int,
    file_prefix: str,
    compression: str,
    *,
    overwrite: bool,
) -> None:
    """Generate synthetic Pega-shaped interaction-history Parquet data."""
    console = Console()
    try:
        start = _parse_iso_date(start_date, "start-date")
        end = _resolve_generation_end_date(start, end_date, days)
        report = generate_pega_dummy_data(
            PegaDummyGenerationConfig(
                source_path=Path(source_path),
                output_dir=Path(output_dir),
                start_date=start,
                end_date=end,
                rows_per_day=rows_per_day,
                batch_size=batch_size,
                seed=seed,
                customer_count=customer_count,
                positive_rate=positive_rate,
                file_prefix=file_prefix,
                compression=compression,
                overwrite=overwrite,
            )
        )
    except Exception as exc:
        print(traceback.format_exc())
        raise click.ClickException(str(exc)) from exc

    console.print(
        f"[bold green]ok[/] — generated {report.rows} row(s) across "
        f"{len(report.files)} Parquet file(s) in {report.output_dir}."
    )
    console.print(
        f"range: {report.start_date.isoformat()} to {report.end_date.isoformat()}; "
        f"rows/day: {report.rows_per_day}; columns: {len(report.columns)}"
    )


def _parse_iso_date(value: str, option_name: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise click.ClickException(f"--{option_name} must use YYYY-MM-DD") from exc


def _resolve_generation_end_date(start: dt.date, end_date: str | None, days: int | None) -> dt.date:
    if end_date and days is not None:
        raise click.ClickException("Use either --end-date or --days, not both")
    if not end_date and days is None:
        raise click.ClickException("Provide --end-date or --days")
    if days is not None:
        if days <= 0:
            raise click.ClickException("--days must be positive")
        return start + dt.timedelta(days=days - 1)
    if end_date is None:
        raise click.ClickException("Provide --end-date or --days")
    return _parse_iso_date(end_date, "end-date")


@main.command()
@click.option(
    "--from",
    "source_toml",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Legacy TOML config to translate.",
)
@click.option(
    "--to",
    "target_catalog",
    required=True,
    type=click.Path(file_okay=False),
    help="Destination catalog directory, usually <workspace>/catalog.",
)
def migrate(source_toml: str, target_catalog: str) -> None:
    """Translate a legacy TOML config into Phase 6 catalog YAML."""
    console = Console()
    try:
        report = migrate_toml(source_toml, target_catalog)
    except Exception as exc:
        click.echo(traceback.format_exc(), err=True)
        _log_caught_exception(
            "migration_failed",
            exc,
            command="migrate",
            source_toml=source_toml,
            target_catalog=target_catalog,
        )
        raise click.ClickException(str(exc)) from exc

    color = "green" if report.ok else "yellow"
    status = "ok" if report.ok else "needs review"
    console.print(
        f"[bold {color}]{status}[/] — generated {len(report.generated_files)} file(s), "
        f"mapped {len(report.mappings)} field(s), found {len(report.gaps)} gap(s)."
    )
    console.print(f"report: {report.target / 'migration_report.md'}")


@main.command()
@click.option(
    "--workspace",
    "workspace_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Value Stream workspace directory.",
)
@click.option(
    "--from-legacy-db",
    "legacy_db",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Legacy DuckDB database containing aggregate tables.",
)
def backfill(workspace_dir: str, legacy_db: str) -> None:
    """Import legacy DuckDB aggregate tables into partitioned parquet."""
    console = Console()
    try:
        result = backfill_from_legacy_db(workspace_dir, legacy_db)
    except Exception as exc:
        click.echo(traceback.format_exc(), err=True)
        _log_caught_exception(
            "backfill_failed",
            exc,
            command="backfill",
            workspace_dir=workspace_dir,
            legacy_db=legacy_db,
        )
        raise click.ClickException(str(exc)) from exc

    console.print(
        f"[bold green]ok[/] — backfilled {len(result.tables)} table(s), "
        f"{result.rows} row(s); skipped {len(result.skipped)} catalog target(s)."
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("legacy table")
    table.add_column("target")
    table.add_column("rows", justify="right")
    table.add_column("files", justify="right")
    for item in result.tables:
        table.add_row(
            item.table,
            f"{item.source_id}/{item.processor_id}/{item.grain}",
            str(item.rows),
            str(len(item.written)),
        )
    if result.tables:
        console.print(table)


def _log_caught_exception(event: str, exc: Exception, **context: str) -> None:
    log_utils.configure()
    context_str = ", ".join(f"{key}={value}" for key, value in context.items())
    click.echo(f"{event}: {context_str}; error={exc}", err=True)
    log_utils.get_logger(__name__).error(f"{event}: {context_str}; error={exc}", exc_info=True)


def _format_elapsed_ms(elapsed_ms: float) -> str:
    return "-" if elapsed_ms <= 0 else f"{elapsed_ms:.03f}ms"


if __name__ == "__main__":
    main()
