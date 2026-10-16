"""Tests for the statistical evaluation layer (Wilson score, Newcombe, BH, power)."""

from __future__ import annotations

import math
import pytest

from agentxray.statistics import (
    adjust_p_values_bh,
    compare_proportions,
    estimate_two_proportion_power,
    proportion_difference_interval,
    wilson_score_interval,
)


def test_wilson_score_interval_boundaries():
    """Verify Wilson interval handles edge proportions without collapsing to zero width."""
    # Zero successes: Wald interval would be [0, 0], Wilson must have positive upper bound
    low, high = wilson_score_interval(successes=0, trials=10, confidence=0.95)
    assert low == 0.0
    assert 0.20 < high < 0.35

    # All successes: Wald interval would be [1, 1], Wilson must have lower bound < 1
    low, high = wilson_score_interval(successes=10, trials=10, confidence=0.95)
    assert 0.65 < low < 0.80
    assert high == 1.0

    # Intermediate proportion: 5/10 = 0.50
    low, high = wilson_score_interval(successes=5, trials=10, confidence=0.95)
    assert low < 0.50 < high
    assert round((low + high) / 2.0, 2) == 0.50


def test_wilson_score_interval_zero_trials():
    """Zero trials should return full uninformative interval [0.0, 1.0]."""
    assert wilson_score_interval(0, 0) == (0.0, 1.0)


def test_wilson_score_interval_invalid_inputs():
    """Test validation errors for invalid trials, successes, or confidence."""
    with pytest.raises(ValueError, match="trials must be non-negative"):
        wilson_score_interval(0, -1)
    with pytest.raises(ValueError, match="successes must be non-negative"):
        wilson_score_interval(-1, 10)
    with pytest.raises(ValueError, match="cannot exceed trials"):
        wilson_score_interval(11, 10)
    with pytest.raises(ValueError, match="confidence must be in"):
        wilson_score_interval(5, 10, confidence=1.5)


def test_proportion_difference_interval_newcombe():
    """Verify Newcombe hybrid score interval for risk difference."""
    # Substantial difference: 9/10 vs 1/10
    low, high = proportion_difference_interval(k1=9, n1=10, k2=1, n2=10, confidence=0.95)
    assert low > 0.30
    assert high <= 1.0
    # Zero difference: 5/10 vs 5/10
    low, high = proportion_difference_interval(k1=5, n1=10, k2=5, n2=10, confidence=0.95)
    assert low < 0.0 < high


def test_compare_proportions_significant_difference():
    """Test two-sample proportion comparison with significant difference."""
    res = compare_proportions(k1=18, n1=20, k2=2, n2=20, alpha=0.05)
    assert res.p1 == 0.90
    assert res.p2 == 0.10
    assert res.difference == 0.80
    assert res.difference_ci[0] > 0.50
    assert res.statistically_significant is True
    assert res.p_value < 0.001
    assert res.p_value_one_sided < 0.001


def test_compare_proportions_null_difference():
    """Test two-sample proportion comparison with no difference."""
    res = compare_proportions(k1=10, n1=20, k2=10, n2=20, alpha=0.05)
    assert res.p1 == 0.50
    assert res.p2 == 0.50
    assert res.difference == 0.0
    assert res.difference_ci[0] < 0.0 < res.difference_ci[1]
    assert res.statistically_significant is False
    assert res.p_value > 0.90


def test_adjust_p_values_bh_monotonicity_and_bounds():
    """Verify Benjamini-Hochberg procedure preserves ranking and valid bounds."""
    raw_p = [0.001, 0.01, 0.04, 0.05, 0.50]
    q_vals = adjust_p_values_bh(raw_p)

    assert len(q_vals) == len(raw_p)
    # Each q-value must be >= raw p-value
    for p, q in zip(raw_p, q_vals):
        assert q >= p - 1e-9
        assert 0.0 <= q <= 1.0

    # Monotonicity: sorted raw p-values must yield non-decreasing q-values
    for i in range(len(q_vals) - 1):
        assert q_vals[i] <= q_vals[i + 1]


def test_adjust_p_values_bh_empty_and_single():
    """Test edge cases with empty or single p-value."""
    assert adjust_p_values_bh([]) == []
    assert adjust_p_values_bh([0.03]) == [0.03]
    assert adjust_p_values_bh([1.5]) == [1.0]


def test_estimate_two_proportion_power_properties():
    """Test statistical power estimation properties."""
    # When p1 == p2, power should equal alpha (type I error rate)
    power_null = estimate_two_proportion_power(p1=0.20, p2=0.20, n_per_group=30, alpha=0.05)
    assert math.isclose(power_null, 0.05, abs_tol=0.01)

    # Power should increase with larger effect size
    power_small = estimate_two_proportion_power(p1=0.20, p2=0.30, n_per_group=30)
    power_large = estimate_two_proportion_power(p1=0.10, p2=0.85, n_per_group=30)
    assert power_large > power_small
    assert power_large > 0.99

    # Power should increase with larger sample size
    power_n10 = estimate_two_proportion_power(p1=0.10, p2=0.40, n_per_group=10)
    power_n100 = estimate_two_proportion_power(p1=0.10, p2=0.40, n_per_group=100)
    assert power_n100 > power_n10

    # One-sided power should be greater than two-sided power for p1 > p2
    power_one_sided = estimate_two_proportion_power(p1=0.35, p2=0.10, n_per_group=20, alternative="greater")
    power_two_sided = estimate_two_proportion_power(p1=0.35, p2=0.10, n_per_group=20, alternative="two-sided")
    assert power_one_sided > power_two_sided


def test_fisher_exact_test_known_tables():
    """Verify exact hypergeometric p-values against standard reference tables."""
    from agentxray.statistics import fisher_exact_test

    # Classical Lady Tasting Tea problem: 4 successes in 4 trials vs 0 in 4 trials
    # P(X >= 4) = 1 / binom(8, 4) = 1 / 70 ≈ 0.01429
    p_greater = fisher_exact_test(4, 4, 0, 4, alternative="greater")
    assert math.isclose(p_greater, 1.0 / 70.0, abs_tol=1e-4)

    # Two-sided p-value includes symmetric tail (0 successes in group 1): 2 / 70 ≈ 0.02857
    p_two = fisher_exact_test(4, 4, 0, 4, alternative="two-sided")
    assert math.isclose(p_two, 2.0 / 70.0, abs_tol=1e-4)

    # Identical proportions (null): 5/10 vs 5/10
    assert fisher_exact_test(5, 10, 5, 10, alternative="two-sided") == 1.0

    # Extreme boundaries: zero failures
    assert fisher_exact_test(0, 10, 0, 10, alternative="two-sided") == 1.0
    assert fisher_exact_test(10, 10, 10, 10, alternative="two-sided") == 1.0


def test_compare_proportions_prespecified_alternative():
    """Verify compare_proportions adheres to prespecified alternative hypothesis."""
    # Weak signal at n=20: 7/20 (0.35) vs 2/20 (0.10)
    # Two-sided test: p ≈ 0.0583 >= 0.05 -> not significant
    res_two = compare_proportions(k1=7, n1=20, k2=2, n2=20, alpha=0.05, alternative="two-sided")
    assert res_two.p_value >= 0.05
    assert res_two.statistically_significant is False

    # One-sided test (greater): p ≈ 0.0292 < 0.05 -> significant
    res_greater = compare_proportions(k1=7, n1=20, k2=2, n2=20, alpha=0.05, alternative="greater")
    assert res_greater.p_value_one_sided < 0.05
    assert res_greater.statistically_significant is True
    assert res_greater.fisher_exact_p_value is not None
