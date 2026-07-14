"""Curve reconstruction from mergeable score digests."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import fsum

from valuestream.states import tdigest


@dataclass(frozen=True)
class CurveResult:
    """Scalar and curve arrays reconstructed from positive/negative digests."""

    roc_auc: float
    average_precision: float
    tpr: tuple[float, ...]
    fpr: tuple[float, ...]
    precision: tuple[float, ...]
    recall: tuple[float, ...]
    pos_fraction: float


@dataclass(frozen=True)
class CalibrationResult:
    """Calibration arrays reconstructed from positive/negative score digests."""

    bin: tuple[float, ...]
    predicted: tuple[float, ...]
    observed: tuple[float, ...]


def curve_from_digests(
    positive: bytes | bytearray | memoryview | None,
    negative: bytes | bytearray | memoryview | None,
    *,
    resolution: int = 101,
) -> CurveResult:
    """Approximate ROC AUC and average precision from score t-digests."""
    pos_weight = tdigest.weight(positive)
    neg_weight = tdigest.weight(negative)
    if pos_weight <= 0 or neg_weight <= 0:
        return CurveResult(0.0, 0.0, (0.0,), (0.0,), (0.0,), (0.0,), 0.0)

    thresholds = _linspace(0.0, 1.0, resolution)
    tpr = tuple(1.0 - tdigest.cdf(positive, threshold) for threshold in thresholds)
    fpr = tuple(1.0 - tdigest.cdf(negative, threshold) for threshold in thresholds)
    roc_pairs = sorted(zip(fpr, tpr, strict=True), key=lambda pair: pair[0])
    fpr_sorted = tuple(pair[0] for pair in roc_pairs)
    tpr_sorted = tuple(pair[1] for pair in roc_pairs)
    roc_auc = _step_auc(tpr_sorted, fpr_sorted)

    thresholds_desc = tuple(reversed(thresholds))
    recall = tuple(1.0 - tdigest.cdf(positive, threshold) for threshold in thresholds_desc)
    fpr_desc = tuple(1.0 - tdigest.cdf(negative, threshold) for threshold in thresholds_desc)
    precision_raw: list[float] = []
    for rec, fpr_value in zip(recall, fpr_desc, strict=True):
        tp = pos_weight * rec
        fp = neg_weight * fpr_value
        precision_raw.append(tp / (tp + fp + 1e-10))
    precision = _monotone_precision(tuple(precision_raw))
    if recall and recall[0] != 0.0:
        recall = (0.0, *recall)
        precision = (1.0, *precision)
    average_precision = fsum(
        (recall[i] - recall[i - 1]) * precision[i] for i in range(1, len(recall))
    )
    return CurveResult(
        roc_auc=float(max(0.0, min(1.0, roc_auc))),
        average_precision=float(max(0.0, min(1.0, average_precision))),
        tpr=tpr_sorted,
        fpr=fpr_sorted,
        precision=precision,
        recall=recall,
        pos_fraction=float(pos_weight / (pos_weight + neg_weight)),
    )


def calibration_from_digests(
    positive: bytes | bytearray | memoryview | None,
    negative: bytes | bytearray | memoryview | None,
) -> CalibrationResult:
    """Approximate calibration bins from positive/negative calibrated-score digests."""
    pos_weight = tdigest.weight(positive)
    neg_weight = tdigest.weight(negative)
    if pos_weight <= 0 or neg_weight <= 0:
        return CalibrationResult((0.0,), (0.0,), (0.0,))

    edges = tuple(dict.fromkeys([*_linspace(0.0, 0.1, 10), *_linspace(0.1, 1.0, 17)]))
    bins: list[float] = []
    predicted: list[float] = []
    observed: list[float] = []
    for left, right in pairwise(edges):
        cdf_pos_left = tdigest.cdf(positive, left)
        cdf_pos_right = tdigest.cdf(positive, right)
        cdf_neg_left = tdigest.cdf(negative, left)
        cdf_neg_right = tdigest.cdf(negative, right)
        pos_in_bin = pos_weight * max(0.0, cdf_pos_right - cdf_pos_left)
        neg_in_bin = neg_weight * max(0.0, cdf_neg_right - cdf_neg_left)
        total = pos_in_bin + neg_in_bin
        bins.append((left + right) / 2)
        predicted.append(_bin_predicted(positive, negative, left, right, pos_in_bin, neg_in_bin))
        observed.append(pos_in_bin / total if total > 0 else 0.0)
    return CalibrationResult(tuple(bins), tuple(predicted), tuple(observed))


def _bin_predicted(
    positive: bytes | bytearray | memoryview | None,
    negative: bytes | bytearray | memoryview | None,
    left: float,
    right: float,
    pos_in_bin: float,
    neg_in_bin: float,
) -> float:
    total = pos_in_bin + neg_in_bin
    if total <= 0:
        return (left + right) / 2
    midpoint = (left + right) / 2
    pos_mean = _quantile_bin_mean(positive, left, right)
    neg_mean = _quantile_bin_mean(negative, left, right)
    pos_value = pos_mean if pos_mean is not None else midpoint
    neg_value = neg_mean if neg_mean is not None else midpoint
    return (pos_value * pos_in_bin + neg_value * neg_in_bin) / total


def _quantile_bin_mean(
    payload: bytes | bytearray | memoryview | None,
    left: float,
    right: float,
) -> float | None:
    mass_left = tdigest.cdf(payload, left)
    mass_right = tdigest.cdf(payload, right)
    if mass_right <= mass_left:
        return None
    qs = _linspace(mass_left, mass_right, 10, endpoint=False)
    return fsum(tdigest.quantile(payload, q) for q in qs) / len(qs)


def _linspace(start: float, end: float, count: int, *, endpoint: bool = True) -> tuple[float, ...]:
    if count <= 1:
        return (start,)
    denom = count - 1 if endpoint else count
    return tuple(start + (end - start) * i / denom for i in range(count))


def _step_auc(y_values: tuple[float, ...], x_values: tuple[float, ...]) -> float:
    return fsum((x_values[i] - x_values[i - 1]) * y_values[i] for i in range(1, len(x_values)))


def _monotone_precision(values: tuple[float, ...]) -> tuple[float, ...]:
    out = list(values)
    best = 0.0
    for idx in range(len(out) - 1, -1, -1):
        best = max(best, out[idx])
        out[idx] = best
    return tuple(out)


__all__ = ["CalibrationResult", "CurveResult", "calibration_from_digests", "curve_from_digests"]
