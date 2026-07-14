"""Statistical helpers for aggregate experiment metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable

from scipy.stats import chi2_contingency, norm  # type: ignore[import-untyped]
from scipy.stats.contingency import odds_ratio  # type: ignore[import-untyped]

_TEST_OUTPUTS = (
    "chi2_stat",
    "chi2_dof",
    "chi2_p_val",
    "chi2_odds_ratio_stat",
    "chi2_odds_ratio_ci_low",
    "chi2_odds_ratio_ci_high",
    "g_stat",
    "g_dof",
    "g_p_val",
    "g_odds_ratio_stat",
    "g_odds_ratio_ci_low",
    "g_odds_ratio_ci_high",
    "z_score",
    "z_p_val",
)


def variant_comparison(
    *,
    test_positives: int,
    test_negatives: int,
    control_positives: int,
    control_negatives: int,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Return rates, effect, Newcombe-Wilson interval, and z-test outputs."""

    test_total = test_positives + test_negatives
    control_total = control_positives + control_negatives
    positives = test_positives + control_positives
    negatives = test_negatives + control_negatives
    total = positives + negatives
    test_ctr = _safe_div(test_positives, test_total)
    control_ctr = _safe_div(control_positives, control_total)
    ctr = _safe_div(positives, total)
    lift = _safe_div(test_ctr - control_ctr, control_ctr)
    absolute_difference = test_ctr - control_ctr
    test_interval = _wilson_interval(test_positives, test_total, confidence_level)
    control_interval = _wilson_interval(control_positives, control_total, confidence_level)
    std_err = math.sqrt(ctr * (1.0 - ctr) / total) if total else 0.0
    z_test = proportions_ztest(
        test_positives=test_positives,
        test_total=test_total,
        control_positives=control_positives,
        control_total=control_total,
    )
    return {
        "TestCTR": test_ctr,
        "ControlCTR": control_ctr,
        "TestSampleSize": float(test_total),
        "ControlSampleSize": float(control_total),
        "AbsoluteRateDifference": absolute_difference,
        "AbsoluteRateDifference_CI_Low": test_interval[0] - control_interval[1],
        "AbsoluteRateDifference_CI_High": test_interval[1] - control_interval[0],
        "Lift": lift,
        "Lift_Z_Score": z_test["z_score"],
        "Lift_P_Val": z_test["z_p_val"],
        "StdErr": std_err,
        "CTR": ctr,
        "Count": float(total),
        "Positives": float(positives),
        "Negatives": float(negatives),
    }


def _wilson_interval(successes: int, total: int, confidence_level: float) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for one binomial proportion."""

    if total <= 0:
        return 0.0, 0.0
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z_squared / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def proportions_ztest(
    *,
    test_positives: int,
    test_total: int,
    control_positives: int,
    control_total: int,
) -> dict[str, float]:
    """Return the two-sided pooled two-proportion z-test."""

    if not all((test_positives, test_total, control_positives, control_total)):
        return {"z_score": 0.0, "z_p_val": 0.0}
    pooled = (test_positives + control_positives) / (test_total + control_total)
    variance = pooled * (1.0 - pooled) * (1.0 / test_total + 1.0 / control_total)
    if variance <= 0:
        return {"z_score": 0.0, "z_p_val": 0.0}
    score = (test_positives / test_total - control_positives / control_total) / math.sqrt(variance)
    return {"z_score": score, "z_p_val": float(2.0 * norm.cdf(-abs(score)))}


def contingency_tests(rows: Iterable[tuple[int, int]]) -> dict[str, float]:
    """Return Pearson chi-square, G-test, z-test, and odds-ratio outputs."""

    table = [[int(positives), int(negatives)] for positives, negatives in rows]
    valid = [row for row in table if row[0] > 0 and row[1] > 0]
    out = dict.fromkeys(_TEST_OUTPUTS, 0.0)
    if len(valid) < 2:
        return out

    chi2_stat, chi2_p_val, chi2_dof, _ = chi2_contingency(valid, correction=False)
    g_stat, g_p_val, g_dof, _ = chi2_contingency(
        valid,
        correction=False,
        lambda_="log-likelihood",
    )
    out.update(
        {
            "chi2_stat": float(chi2_stat),
            "chi2_dof": float(chi2_dof),
            "chi2_p_val": float(chi2_p_val),
            "g_stat": float(g_stat),
            "g_dof": float(g_dof),
            "g_p_val": float(g_p_val),
        }
    )
    if len(valid) == 2:
        ratio = odds_ratio(valid, kind="sample")
        interval = ratio.confidence_interval(confidence_level=0.95)
        odds_outputs = {
            "odds_ratio_stat": float(ratio.statistic),
            "odds_ratio_ci_low": float(interval.low),
            "odds_ratio_ci_high": float(interval.high),
        }
        out.update({f"chi2_{name}": value for name, value in odds_outputs.items()})
        out.update({f"g_{name}": value for name, value in odds_outputs.items()})
        out.update(
            proportions_ztest(
                test_positives=valid[0][0],
                test_total=sum(valid[0]),
                control_positives=valid[1][0],
                control_total=sum(valid[1]),
            )
        )
    return out


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


__all__ = ["contingency_tests", "proportions_ztest", "variant_comparison"]
