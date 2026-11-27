"""Focused, sample-backed review of one metric in an authoring draft."""

from __future__ import annotations

import copy
from typing import Any

import polars as pl

from valuestream.config import model
from valuestream.expr.ast import Col, OpSafeDiv
from valuestream.expr.translator import translate


class OpenSourceError(ValueError):
    def __init__(
        self, message: str, source_id: str, *, kind: str = "existing_source", **arguments: Any
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.next_call = {
            "tool": "catalog_draft_open",
            "arguments": {"source_id": source_id, **arguments},
        }


def _owned_objects(payload: dict[str, Any]) -> dict[tuple[str, str], tuple[str, Any]]:
    sources = payload["pipelines"]["sources"]
    processors = {p["id"]: p["source"] for p in payload["processors"]["processors"]}
    metrics = {
        key: processors.get(m["processor"], "") for key, m in payload["metrics"]["metrics"].items()
    }
    return {
        **{("source", s["id"]): (s["id"], s) for s in sources},
        **{("processor", p["id"]): (p["source"], p) for p in payload["processors"]["processors"]},
        **{("metric", key): (metrics[key], m) for key, m in payload["metrics"]["metrics"].items()},
        **{
            ("tile", f"{d['id']}/{p['id']}/{t['id']}"): (metrics.get(t["metric"], ""), t)
            for d in payload["dashboards"]["dashboards"]
            for p in d["pages"]
            for t in p["tiles"]
        },
    }


def scope_error(
    base: dict[str, Any], candidate: dict[str, Any], source_id: str, issues: list[str]
) -> ValueError:
    before, after = _owned_objects(base), _owned_objects(candidate)
    details = []
    owners = set()
    for key in sorted(before.keys() | after.keys()):
        if before.get(key) == after.get(key):
            continue
        foreign = {
            entry[0]
            for entry in (before.get(key), after.get(key))
            if entry and entry[0] and entry[0] != source_id
        }
        if foreign:
            owners.update(foreign)
            details.append(f"{key[0]} {key[1]!r} belongs to {', '.join(sorted(foreign))}")
    message = f"Active source: {source_id!r}. " + ". ".join(details or issues)
    if len(owners) == 1:
        return OpenSourceError(message, owners.pop(), kind="source_scope_mismatch")
    return ValueError(message + ". Open the owning source with catalog_draft_open.")


def metric_context(payload: dict[str, Any], name: str, source_id: str) -> dict[str, Any] | None:
    metrics = payload["metrics"]["metrics"]
    metric = metrics.get(name)
    if metric is None:
        return None
    processor = next(
        (p for p in payload["processors"]["processors"] if p["id"] == metric["processor"]), None
    )
    if processor is None or processor["source"] != source_id:
        return None
    return copy.deepcopy(
        {
            "name": name,
            "metric": metric,
            "processor": processor,
            "processor_metrics": sorted(
                key for key, value in metrics.items() if value["processor"] == processor["id"]
            ),
            "dependent_metrics": sorted(
                key for key, value in metrics.items() if name in value.get("depends_on", [])
            ),
            "tiles": [
                {"path": f"{dashboard['id']}/{page['id']}/{tile['id']}", **tile}
                for dashboard in payload["dashboards"]["dashboards"]
                for page in dashboard["pages"]
                for tile in page["tiles"]
                if name in (tile["metric"], tile.get("secondary_metric"))
            ],
        }
    )


def sample_rate(
    metric: model.Metric, processor: model.Processor, partials: pl.DataFrame, rows: int
) -> dict[str, Any]:
    """Show operands only for ratios of additive counts. Never return sample records."""
    if (
        not isinstance(processor, model.BinaryOutcomeProcessor)
        or not isinstance(metric, model.FormulaMetric)
        or not isinstance(metric.expression, OpSafeDiv)
    ):
        return {"status": "unsupported", "sample_rows": rows}
    expr = metric.expression
    if not isinstance(expr.num, Col) or not isinstance(expr.den, Col):
        return {"status": "unsupported", "sample_rows": rows}
    numerator, denominator = expr.num.col, expr.den.col
    specs = model.effective_processor_states(processor)
    names = list(dict.fromkeys((numerator, denominator)))
    if metric.depends_on or any(
        not isinstance(specs.get(name), model.CountState) for name in names
    ):
        return {"status": "unsupported", "sample_rows": rows}
    if partials.is_empty():
        return {"status": "no_groups", "sample_rows": rows, "value": None}
    totals = partials.select(pl.col(names).sum())
    return {
        "status": "ok",
        "sample_rows": rows,
        "numerator": {"state": numerator, "count": totals[numerator].item()},
        "denominator": {"state": denominator, "count": totals[denominator].item()},
        "value": totals.select(translate(expr)).item(),
    }
