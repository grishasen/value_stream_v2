"""Unit coverage for Phase 2 state helpers and curve algorithms."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from valuestream.algorithms.curves import calibration_from_digests, curve_from_digests
from valuestream.states import kll, tdigest
from valuestream.states._numeric import bulk_numeric_array


def test_tdigest_build_merge_and_quantile() -> None:
    left = tdigest.build([0.1, 0.2, 0.3])
    right = tdigest.build([0.7, 0.8, 0.9])

    merged = tdigest.merge([left, right])

    assert tdigest.weight(merged) == 6
    assert tdigest.quantile(merged, 0.5) == pytest.approx(0.5, abs=0.25)


def test_kll_build_merge_and_quantile() -> None:
    left = kll.build([1.0, 2.0])
    right = kll.build([3.0, 4.0])

    merged = kll.merge([left, right])

    assert kll.count(merged) == 4
    assert kll.quantile(merged, 0.5) == pytest.approx(2.5, abs=1.0)


def test_tiny_numeric_groups_keep_scalar_null_and_nan_semantics() -> None:
    values = pl.Series("value", [1.0, None, float("nan")])

    assert bulk_numeric_array(values, dtype=np.float64) is None
    assert tdigest.weight(tdigest.build(values)) == 1
    assert kll.count(kll.build(values)) == 1


def test_bulk_numeric_array_drops_nulls_but_preserves_genuine_nan() -> None:
    values = pl.Series(
        "value",
        [float(value) for value in range(32)] + [None, float("nan")],
    )

    array = bulk_numeric_array(values, dtype=np.float64)

    assert array is not None
    assert array.shape == (33,)
    assert np.isnan(array).sum() == 1
    assert tdigest.weight(tdigest.build(values)) == 32
    assert kll.count(kll.build(values)) == 32


def test_curve_and_calibration_from_separated_digests() -> None:
    negatives = tdigest.build([0.01, 0.02, 0.05, 0.10, 0.20])
    positives = tdigest.build([0.70, 0.80, 0.90, 0.95, 0.99])

    curve = curve_from_digests(positives, negatives)
    calibration = calibration_from_digests(positives, negatives)

    assert curve.roc_auc > 0.95
    assert curve.average_precision > 0.90
    assert curve.pos_fraction == pytest.approx(0.5)
    assert len(calibration.bin) == len(calibration.predicted) == len(calibration.observed)


def test_empty_curves_default_to_zero() -> None:
    curve = curve_from_digests(tdigest.build([]), tdigest.build([0.1]))
    calibration = calibration_from_digests(tdigest.build([]), tdigest.build([0.1]))

    assert curve.roc_auc == 0
    assert curve.average_precision == 0
    assert calibration.observed == (0.0,)
