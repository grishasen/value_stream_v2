"""FastAPI application exposing the governed aggregate tool layer over HTTP.

Every endpoint is a thin wrapper over the exact same governed functions the
MCP server and Streamlit chat use: ``query_metric``, ``chart_intent_from_parameters``
/ ``execute_chat_intent``, ``dimension_values``, ``metric_freshness``, and the
governed ``run_sql_query`` / ``list_sql_tables``. There is one tool layer with
two transports (MCP and HTTP); this module adds no query capability of its own.

The service is read-only: no endpoint mutates the catalog or the aggregate
store. Raw source rows are never reachable — only catalog metadata, aggregate
query results, and governed SQL over the aggregate DuckDB views.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from valuestream.ai.chat import (
    catalog_chat_manifest,
    chart_intent_from_parameters,
    dimension_values,
    execute_chat_intent,
    narrate_chat_result,
    plan_chat_intent,
)
from valuestream.ai.settings import (
    configured_api_key,
    load_chat_with_data_config,
    load_llm_settings_config,
)
from valuestream.ai.sql_tool import list_sql_tables, run_sql_query, sql_schema_summary
from valuestream.ai.studio import AICallSettings
from valuestream.config.watch import CatalogCache
from valuestream.query import query_metric_result
from valuestream.ui.freshness import freshness_label, metric_freshness
from valuestream.utils.logger import get_logger

logger = get_logger(__name__)

API_TOKEN_ENV = "VALUESTREAM_API_TOKEN"


class MetricQueryRequest(BaseModel):
    group_by: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    grain: str = "summary"
    start: str | None = None
    end: str | None = None
    having: dict[str, Any] = Field(default_factory=dict)
    order_by: list[str] = Field(default_factory=list)
    top_n: int | None = None
    top_n_by: str | None = None
    compare: str | None = None
    include_quantile_suite: bool = False
    limit: int = 100


class MetricChartRequest(BaseModel):
    chart_kind: str
    x: str
    y: str
    group_by: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    grain: str = "summary"
    start: str | None = None
    end: str | None = None
    color: str | None = None
    facet_col: str | None = None
    having: dict[str, Any] = Field(default_factory=dict)
    order_by: list[str] = Field(default_factory=list)
    top_n: int | None = None
    top_n_by: str | None = None
    compare: str | None = None
    value_format: str | None = None
    limit: int = 100


class SqlRequest(BaseModel):
    sql: str
    limit: int = 200


class ChatRequest(BaseModel):
    question: str
    allow_sql: bool = False
    narrate: bool = False


def create_app(  # noqa: PLR0915
    workspace_path: str | Path,
    *,
    api_token: str | None = None,
    enable_chat: bool = True,
    enable_sql: bool = False,
) -> FastAPI:
    """Build the read-only Value Stream HTTP API for one workspace.

    ``api_token`` (or the ``VALUESTREAM_API_TOKEN`` environment variable) gates
    every endpoint behind a bearer token; when neither is set the API is open,
    which is only appropriate for a trusted localhost deployment.
    """

    workspace = Path(workspace_path).resolve()
    catalog_cache = CatalogCache(workspace)
    catalog_cache.get()  # fail fast on an invalid catalog
    token = api_token if api_token is not None else os.environ.get(API_TOKEN_ENV, "")

    app = FastAPI(
        title="Value Stream API",
        version="1",
        summary="Read-only governed access to aggregate metrics.",
    )

    def require_token(authorization: str | None = Header(default=None)) -> None:
        if not token:
            return
        expected = f"Bearer {token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    guard = [Depends(require_token)]

    def _chat_config() -> dict[str, Any]:
        _, config = load_chat_with_data_config(workspace)
        return config

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "workspace": workspace.name}

    @app.get("/metrics", dependencies=guard)
    def metrics() -> dict[str, Any]:
        return catalog_chat_manifest(catalog_cache.get(), chat_config=_chat_config())

    @app.post("/metrics/{metric}/query", dependencies=guard)
    def metric_query_endpoint(metric: str, request: MetricQueryRequest) -> dict[str, Any]:
        with _translate_errors():
            result = query_metric_result(
                workspace,
                metric,
                group_by=request.group_by,
                filters=request.filters,
                grain=request.grain,
                start=request.start,
                end=request.end,
                having=request.having,
                order_by=request.order_by,
                top_n=request.top_n,
                top_n_by=request.top_n_by,
                compare=request.compare,
                include_quantile_suite=request.include_quantile_suite,
                include_curve_columns=True,
            )
            frame = result.rows
        clipped = frame.head(max(1, min(int(request.limit), 500)))
        return {
            "metric": metric,
            "grain": request.grain,
            "row_count": frame.height,
            "columns": clipped.columns,
            "rows": clipped.to_dicts(),
            "provenance": result.provenance.to_dict(),
        }

    @app.post("/metrics/{metric}/chart", dependencies=guard)
    def metric_chart_endpoint(metric: str, request: MetricChartRequest) -> dict[str, Any]:
        catalog = catalog_cache.get()
        with _translate_errors():
            intent = chart_intent_from_parameters(
                catalog,
                metric=metric,
                chart_kind=request.chart_kind,
                x=request.x,
                y=request.y,
                group_by=request.group_by,
                filters=request.filters,
                grain=request.grain,
                start=request.start,
                end=request.end,
                color=request.color,
                facet_col=request.facet_col,
                having=request.having,
                order_by=request.order_by,
                top_n=request.top_n,
                top_n_by=request.top_n_by,
                compare=request.compare,
                value_format=request.value_format,
                limit=request.limit,
            )
            result = execute_chat_intent(workspace, catalog, intent)
        chart = result.intent.chart
        return {
            "metric": result.intent.metric,
            "grain": result.intent.grain,
            "row_count": result.rows.height,
            "columns": result.rows.columns,
            "chart": {
                "kind": chart.kind if chart else request.chart_kind,
                "x": chart.x if chart else request.x,
                "y": chart.y if chart else request.y,
                "color": chart.color if chart else request.color,
                "facet_col": chart.facet_col if chart else request.facet_col,
                "value_format": chart.value_format if chart else request.value_format,
            },
            "query": result.query_summary,
            "freshness": result.freshness,
            "rows": result.rows.to_dicts(),
        }

    @app.get("/metrics/{metric}/dimension-values", dependencies=guard)
    def dimension_values_endpoint(
        metric: str,
        column: str,
        grain: str = "summary",
        limit: int = 50,
    ) -> dict[str, Any]:
        with _translate_errors():
            values = dimension_values(
                workspace,
                catalog_cache.get(),
                metric,
                column,
                grain=grain,
                limit=limit,
            )
        return {"metric": metric, "column": column, "values": values}

    @app.get("/metrics/{metric}/freshness", dependencies=guard)
    def freshness_endpoint(metric: str, grain: str = "summary") -> dict[str, Any]:
        with _translate_errors():
            fresh = metric_freshness(workspace, catalog_cache.get(), metric, grain=grain)
        return {
            "metric": metric,
            "grain": grain,
            "latest_period": fresh.latest_period,
            "status": fresh.status,
            "label": freshness_label(fresh),
        }

    if enable_sql:

        @app.get("/sql/schema", dependencies=guard)
        def sql_schema_endpoint() -> dict[str, Any]:
            tables = list_sql_tables(workspace, catalog_cache.get())
            return {
                "tables": [
                    {
                        "name": table.name,
                        "kind": table.kind,
                        "columns": [{"name": name, "type": dtype} for name, dtype in table.columns],
                    }
                    for table in tables
                ],
                "schema_text": sql_schema_summary(workspace, catalog_cache.get()),
            }

        @app.post("/sql", dependencies=guard)
        def sql_endpoint(request: SqlRequest) -> dict[str, Any]:
            with _translate_errors():
                result = run_sql_query(
                    workspace, request.sql, catalog=catalog_cache.get(), limit=request.limit
                )
            return {
                "sql": result.sql,
                "row_count": result.row_count,
                "truncated": result.truncated,
                "masked_columns": result.masked_columns,
                "columns": result.rows.columns,
                "rows": result.rows.to_dicts(),
            }

    if enable_chat:

        @app.post("/chat", dependencies=guard)
        def chat_endpoint(request: ChatRequest) -> dict[str, Any]:
            settings = _settings_from_workspace(workspace)
            if settings is None:
                raise HTTPException(
                    status_code=503,
                    detail="no LLM model configured; set ai.llm.model in the workspace ai.yaml",
                )
            catalog = catalog_cache.get()
            if request.allow_sql and not enable_sql:
                raise HTTPException(
                    status_code=403,
                    detail="governed SQL is disabled for this API server",
                )
            sql_schema = sql_schema_summary(workspace, catalog) if request.allow_sql else None
            with _translate_errors():
                intent, raw = plan_chat_intent(
                    settings,
                    catalog,
                    request.question,
                    chat_config=_chat_config(),
                    sql_schema=sql_schema,
                )
                payload = _answer_for_intent(
                    workspace,
                    catalog,
                    intent,
                    settings,
                    request.narrate,
                    enable_sql=enable_sql,
                )
            payload["model_raw"] = raw
            return payload

    return app


def _answer_for_intent(
    workspace: Path,
    catalog: Any,
    intent: Any,
    settings: AICallSettings,
    narrate: bool,
    *,
    enable_sql: bool,
) -> dict[str, Any]:
    if intent.response == "clarify":
        return {"response": "clarify", "clarify": intent.clarify}
    if intent.response == "sql":
        if not enable_sql:
            raise HTTPException(status_code=403, detail="governed SQL is disabled")
        sql_result = run_sql_query(workspace, intent.sql or "", catalog=catalog)
        return {
            "response": "sql",
            "sql": sql_result.sql,
            "row_count": sql_result.row_count,
            "truncated": sql_result.truncated,
            "columns": sql_result.rows.columns,
            "rows": sql_result.rows.to_dicts(),
        }
    result = execute_chat_intent(workspace, catalog, intent)
    payload: dict[str, Any] = {
        "response": intent.response,
        "metric": intent.metric,
        "grain": intent.grain,
        "query": result.query_summary,
        "freshness": result.freshness,
        "row_count": result.rows.height,
        "columns": result.rows.columns,
        "rows": result.rows.to_dicts(),
    }
    if narrate and not result.rows.is_empty():
        try:
            payload["narrative"] = narrate_chat_result(settings, result)
        except Exception:
            logger.exception("API chat narrative failed")
    return payload


def _settings_from_workspace(workspace: Path) -> AICallSettings | None:
    _, config = load_llm_settings_config(workspace)
    model = str(config.get("model") or "").strip()
    if not model:
        return None
    temperature = config.get("temperature")
    return AICallSettings(
        model=model,
        api_key=configured_api_key(config),
        api_base=str(config.get("api_base") or ""),
        custom_llm_provider=str(
            config.get("custom_provider") or config.get("custom_llm_provider") or ""
        ),
        temperature=float(temperature) if temperature is not None else None,
        reasoning_effort=str(config.get("reasoning_effort") or ""),
        verbosity=str(config.get("verbosity") or ""),
        timeout_seconds=int(config.get("timeout_seconds") or 90),
    )


@contextmanager
def _translate_errors() -> Iterator[None]:
    """Map governed-layer exceptions to HTTP status codes."""
    try:
        yield
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except duckdb.Error as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc


__all__ = ["API_TOKEN_ENV", "create_app"]
