"""Stable contract helpers shared by the ingestion benchmark runner."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from typing import Any

CONTRACT_VERSION = 3
SUITE_PROCESSORS: dict[str, tuple[str, ...]] = {
    "legacy_equivalent": (
        "engagement",
        "conversion",
        "descriptive",
        "model_ml_scores",
        "experiment",
    ),
    "full_current": (
        "engagement",
        "conversion",
        "descriptive",
        "model_ml_scores",
        "experiment",
        "action_funnel",
        "audience",
    ),
}


def summarize_samples(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return stable summary statistics without imposing performance thresholds."""

    rows = list(samples)
    if not rows:
        raise ValueError("at least one benchmark sample is required")
    wall = [float(row["wall_seconds"]) for row in rows]
    cpu = [float(row["cpu_seconds"]) for row in rows]
    throughput = [float(row["rows_per_second"]) for row in rows]
    cpu_per_million = [float(row["cpu_seconds_per_million_rows"]) for row in rows]
    rss = [int(row["peak_rss_bytes"]) for row in rows]
    exact_digests = {str(_exact_digest(row)) for row in rows}
    representation_values = [row.get("representation_digest") for row in rows]
    representation_digests = {str(value) for value in representation_values if value is not None}
    representations_recorded = all(value is not None for value in representation_values)
    approximate_issues = _approximate_comparison_issues(rows)
    exact_outputs_deterministic = len(exact_digests) == 1
    approximate_outputs_equivalent = not approximate_issues
    outputs_equivalent = exact_outputs_deterministic and approximate_outputs_equivalent
    return {
        "samples": len(rows),
        "wall_seconds_mean": statistics.fmean(wall),
        "wall_seconds_median": statistics.median(wall),
        "wall_seconds_cv": _coefficient_of_variation(wall),
        "cpu_seconds_mean": statistics.fmean(cpu),
        "rows_per_second_mean": statistics.fmean(throughput),
        "cpu_seconds_per_million_rows_mean": statistics.fmean(cpu_per_million),
        "peak_rss_bytes_max": max(rss),
        "output_digest": (next(iter(exact_digests)) if exact_outputs_deterministic else None),
        "exact_output_digest": (next(iter(exact_digests)) if exact_outputs_deterministic else None),
        "exact_outputs_deterministic": exact_outputs_deterministic,
        "representation_digest": (
            next(iter(representation_digests))
            if representations_recorded and len(representation_digests) == 1
            else None
        ),
        "representations_stable": (
            len(representation_digests) == 1 if representations_recorded else None
        ),
        "approximate_outputs_equivalent": approximate_outputs_equivalent,
        "approximate_comparison_issues": approximate_issues,
        "outputs_equivalent": outputs_equivalent,
        # Compatibility name from contract v1. In v2+ it means semantic output
        # equivalence, not equality of opaque serialized sketch bytes.
        "outputs_deterministic": outputs_equivalent,
    }


def _coefficient_of_variation(values: list[float]) -> float:
    mean = statistics.fmean(values)
    if len(values) < 2 or mean == 0:
        return 0.0
    return statistics.stdev(values) / mean


def _approximate_comparison_issues(rows: list[Mapping[str, Any]]) -> list[str]:
    issues: list[str] = []
    baseline = _mapping(rows[0].get("approximate_state_probes", {}))
    for sample_index, row in enumerate(rows, start=1):
        unknown = list(row.get("unclassified_binary_states", []))
        if unknown:
            _issue(
                issues,
                f"sample {sample_index} has unclassified binary state(s): "
                + ", ".join(str(value) for value in unknown),
            )
        for state_name, raw_contract in _mapping(row.get("approximate_state_probes", {})).items():
            contract = _mapping(raw_contract)
            if int(contract.get("decode_errors", 0)):
                _issue(
                    issues,
                    f"sample {sample_index} {state_name} contains undecodable payloads",
                )
    for sample_index, row in enumerate(rows[1:], start=2):
        candidate = _mapping(row.get("approximate_state_probes", {}))
        if set(candidate) != set(baseline):
            missing = sorted(set(baseline).difference(candidate))
            extra = sorted(set(candidate).difference(baseline))
            _issue(
                issues,
                f"sample {sample_index} approximate state set differs "
                f"(missing={missing}, extra={extra})",
            )
            continue
        for state_name in sorted(baseline):
            _compare_approximate_state(
                issues,
                state_name,
                _mapping(baseline[state_name]),
                _mapping(candidate[state_name]),
                sample_index=sample_index,
            )
    return issues


def _compare_approximate_state(
    issues: list[str],
    state_name: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    sample_index: int,
) -> None:
    state_type = str(baseline.get("type"))
    if candidate.get("type") != state_type:
        _issue(
            issues,
            f"sample {sample_index} {state_name} type differs: "
            f"{candidate.get('type')!r} != {state_type!r}",
        )
        return
    for field in (
        "rows",
        "payloads",
        "nulls",
        "present_partitions",
        "decode_errors",
    ):
        if candidate.get(field) != baseline.get(field):
            _issue(
                issues,
                f"sample {sample_index} {state_name} {field} differs: "
                f"{candidate.get(field)!r} != {baseline.get(field)!r}",
            )
    baseline_samples = list(baseline.get("samples", []))
    candidate_samples = list(candidate.get("samples", []))
    if len(candidate_samples) != len(baseline_samples):
        _issue(
            issues,
            f"sample {sample_index} {state_name} semantic sample count differs: "
            f"{len(candidate_samples)} != {len(baseline_samples)}",
        )
        return
    for baseline_sample, candidate_sample in zip(baseline_samples, candidate_samples, strict=True):
        left = _mapping(baseline_sample)
        right = _mapping(candidate_sample)
        row_id = str(left.get("row"))
        if right.get("row") != row_id:
            _issue(
                issues,
                f"sample {sample_index} {state_name} probe row differs: "
                f"{right.get('row')!r} != {row_id!r}",
            )
            continue
        if not _probes_equivalent(
            state_type,
            _mapping(left.get("probes", {})),
            _mapping(right.get("probes", {})),
        ):
            _issue(
                issues,
                f"sample {sample_index} {state_name} semantic probes differ at {row_id}",
            )


def _probes_equivalent(  # noqa: PLR0911
    state_type: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    if set(candidate) != set(baseline):
        return False
    if state_type in {"cpc", "hll", "theta"}:
        left_interval = (_number(baseline["lower"]), _number(baseline["upper"]))
        right_interval = (_number(candidate["lower"]), _number(candidate["upper"]))
        if left_interval[0] > right_interval[1] or right_interval[0] > left_interval[1]:
            return False
        left_half_width = (left_interval[1] - left_interval[0]) / 2
        right_half_width = (right_interval[1] - right_interval[0]) / 2
        allowed = max(1.0, 2 * left_half_width, 2 * right_half_width)
        return abs(_number(baseline["estimate"]) - _number(candidate["estimate"])) <= allowed
    if state_type == "tdigest":
        if not math.isclose(
            _number(baseline["weight"]),
            _number(candidate["weight"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
        return all(
            math.isclose(
                _number(baseline[name]),
                _number(candidate[name]),
                rel_tol=0.03,
                abs_tol=1e-6,
            )
            for name in ("q01", "q50", "q99")
        )
    if state_type == "kll":
        if baseline["count"] != candidate["count"]:
            return False
        return all(
            math.isclose(
                _number(baseline[name]),
                _number(candidate[name]),
                rel_tol=0.03,
                abs_tol=1e-6,
            )
            for name in ("q01", "q50", "q99")
        )
    if state_type == "topk":
        if baseline["weight"] != candidate["weight"]:
            return False
        return all(
            math.isclose(
                _number(baseline[name]),
                _number(candidate[name]),
                rel_tol=0.10,
                abs_tol=1.0,
            )
            for name in ("estimate_sum", "lower_sum", "upper_sum")
        )
    return all(
        math.isclose(_number(value), _number(candidate[name]), rel_tol=1e-12, abs_tol=1e-12)
        for name, value in baseline.items()
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _exact_digest(row: Mapping[str, Any]) -> Any:
    if "exact_output_digest" in row:
        return row["exact_output_digest"]
    return row["output_digest"]


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result


def _issue(issues: list[str], message: str) -> None:
    if len(issues) < 50:
        issues.append(message)


__all__ = ["CONTRACT_VERSION", "SUITE_PROCESSORS", "summarize_samples"]
