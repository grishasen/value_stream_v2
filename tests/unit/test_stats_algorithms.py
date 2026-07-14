"""Statistical helper tests."""

from __future__ import annotations

import pytest

from valuestream.algorithms.stats import contingency_tests, proportions_ztest, variant_comparison


@pytest.mark.unit
def test_variant_comparison_returns_lift_and_two_proportion_z_test() -> None:
    result = variant_comparison(
        test_positives=30,
        test_negatives=70,
        control_positives=20,
        control_negatives=80,
    )

    assert result["TestCTR"] == pytest.approx(0.3)
    assert result["ControlCTR"] == pytest.approx(0.2)
    assert result["TestSampleSize"] == 100
    assert result["ControlSampleSize"] == 100
    assert result["AbsoluteRateDifference"] == pytest.approx(0.1)
    assert result["AbsoluteRateDifference_CI_Low"] == pytest.approx(-0.0698803)
    assert result["AbsoluteRateDifference_CI_High"] == pytest.approx(0.2624816)
    assert result["Lift"] == pytest.approx(0.5)
    assert result["CTR"] == pytest.approx(0.25)
    assert result["Lift_Z_Score"] == pytest.approx(1.632993, rel=1e-5)
    assert result["Lift_P_Val"] == pytest.approx(0.10247, rel=1e-4)


@pytest.mark.unit
def test_contingency_tests_return_chi2_g_z_and_odds_ratio_outputs() -> None:
    result = contingency_tests([(30, 70), (20, 80)])

    assert result["chi2_stat"] == pytest.approx(2.666667, rel=1e-5)
    assert result["g_stat"] == pytest.approx(2.680713, rel=1e-5)
    assert result["z_score"] == pytest.approx(1.632993, rel=1e-5)
    assert result["chi2_odds_ratio_stat"] == pytest.approx(1.714286, rel=1e-5)
    assert result["g_odds_ratio_ci_low"] < result["g_odds_ratio_stat"]
    assert result["g_odds_ratio_ci_high"] > result["g_odds_ratio_stat"]


@pytest.mark.unit
def test_proportions_ztest_handles_empty_success_counts() -> None:
    assert proportions_ztest(
        test_positives=0,
        test_total=100,
        control_positives=20,
        control_total=100,
    ) == {"z_score": 0.0, "z_p_val": 0.0}
