"""Repeated queries over unchanged aggregates must return identical numbers.

Three FAT dashboard tiles drifted run to run against byte-identical aggregate
data. Two causes were code-level and are covered here: sketch merges fold
payloads in a non-associative order, and the contingency readout reads its
first two variant rows as test and control.
"""

from __future__ import annotations

import random

import polars as pl
import pytest

from valuestream.config import model
from valuestream.processors.processors_helper import merge_state_frame
from valuestream.query import executor
from valuestream.states import tdigest


def _digest_frame(seed: int, parts: int = 12, rows: int = 400) -> pl.DataFrame:
    """One aggregate-shaped frame: several partial digests per group."""
    rng = random.Random(seed)
    return pl.DataFrame(
        {
            "Model": ["a" if index % 2 else "b" for index in range(parts)],
            "Value_tdigest": [
                tdigest.build([rng.lognormvariate(-5, 1.2) for _ in range(rows)], k=500)
                for _ in range(parts)
            ],
        }
    )


_DIGEST_SPECS: dict[str, model.StateSpec] = {
    "Value_tdigest": model.TDigestState(type="tdigest", source_column="Value", k=500),
}

_QUANTILES = (0.25, 0.5, 0.75, 0.9, 0.95)


@pytest.mark.unit
def test_digest_merge_is_identical_across_repeated_calls() -> None:
    frame = _digest_frame(seed=7)

    merged = [merge_state_frame(frame, _DIGEST_SPECS, ["Model"]).sort("Model") for _ in range(5)]

    first = merged[0]["Value_tdigest"].to_list()
    for other in merged[1:]:
        assert other["Value_tdigest"].to_list() == first


@pytest.mark.unit
def test_digest_merge_ignores_aggregate_row_order() -> None:
    """Row order follows parquet part order, so it must not reach the estimate."""
    frame = _digest_frame(seed=11)
    shuffles = [
        frame,
        frame.reverse(),
        frame.sample(fraction=1.0, shuffle=True, seed=3),
        frame.sample(fraction=1.0, shuffle=True, seed=99),
    ]

    merged = [
        merge_state_frame(shuffled, _DIGEST_SPECS, ["Model"]).sort("Model") for shuffled in shuffles
    ]

    baseline = merged[0]
    for other in merged[1:]:
        assert other["Value_tdigest"].to_list() == baseline["Value_tdigest"].to_list()
        for payload, expected in zip(
            other["Value_tdigest"].to_list(),
            baseline["Value_tdigest"].to_list(),
            strict=True,
        ):
            for quantile in _QUANTILES:
                assert tdigest.quantile(payload, quantile) == tdigest.quantile(expected, quantile)


@pytest.mark.unit
def test_contingency_readout_does_not_depend_on_variant_row_order() -> None:
    """Odds ratio inverts and z flips sign when test and control swap places."""
    metric = model.ContingencyTestMetric(
        processor="experiment",
        kind="contingency_test",
        variant_column="Variant",
        tests=["chi2", "g", "z"],
    )
    rows = pl.DataFrame(
        {
            "Variant": ["control", "treatment"],
            "Positives": [120, 160],
            "Negatives": [880, 840],
        }
    )

    forward = executor._derive_contingency_test(rows, metric, [])
    reversed_rows = executor._derive_contingency_test(rows.reverse(), metric, [])

    for column in ("chi2_odds_ratio_stat", "chi2_odds_ratio_ci_low", "z_score", "chi2_stat"):
        assert forward[column].to_list() == pytest.approx(reversed_rows[column].to_list())
    # Direction is pinned to the variant label, not to arrival order.
    assert forward["z_score"].item() < 0
