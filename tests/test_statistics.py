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


def test_regularized_incomplete_beta_properties_and_boundaries():
    """Verify regularized incomplete beta function boundaries, symmetry, and known polynomials."""
    from agentxray.statistics import regularized_incomplete_beta

    # Boundary conditions
    assert regularized_incomplete_beta(2.0, 3.0, 0.0) == 0.0
    assert regularized_incomplete_beta(2.0, 3.0, 1.0) == 1.0

    # Input validations
    with pytest.raises(ValueError, match="must be in"):
        regularized_incomplete_beta(1.0, 1.0, -0.1)
    with pytest.raises(ValueError, match="must be in"):
        regularized_incomplete_beta(1.0, 1.0, 1.1)
    with pytest.raises(ValueError, match="must be positive"):
        regularized_incomplete_beta(0.0, 1.0, 0.5)

    # Analytical case I_x(1, 1) = x
    for x_val in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert math.isclose(regularized_incomplete_beta(1.0, 1.0, x_val), x_val, abs_tol=1e-12)

    # Analytical case I_x(1, 2) = 2x - x^2
    for x_val in (0.2, 0.4, 0.6, 0.8):
        exp = 2.0 * x_val - x_val * x_val
        assert math.isclose(regularized_incomplete_beta(1.0, 2.0, x_val), exp, abs_tol=1e-12)

    # Symmetry: I_x(a, b) + I_{1-x}(b, a) == 1.0
    a, b, x = 3.5, 5.2, 0.42
    val1 = regularized_incomplete_beta(a, b, x)
    val2 = regularized_incomplete_beta(b, a, 1.0 - x)
    assert math.isclose(val1 + val2, 1.0, abs_tol=1e-12)


def test_student_t_distribution_p_value_known_quantiles():
    """Verify Student's t distribution p-value against standard analytical quantiles."""
    from agentxray.statistics import student_t_distribution_p_value

    # Zero statistic: p = 1.0 for two-sided, 0.5 for one-sided
    assert student_t_distribution_p_value(0.0, df=10, alternative="two-sided") == 1.0
    assert student_t_distribution_p_value(0.0, df=10, alternative="greater") == 0.5
    assert student_t_distribution_p_value(0.0, df=10, alternative="less") == 0.5

    # Standard critical value at df=10, alpha=0.05 (two-sided) is t = 2.22813885
    p_10 = student_t_distribution_p_value(2.22813885, df=10, alternative="two-sided")
    assert math.isclose(p_10, 0.05, abs_tol=1e-6)

    # One-sided tests
    p_gt = student_t_distribution_p_value(2.22813885, df=10, alternative="greater")
    assert math.isclose(p_gt, 0.025, abs_tol=1e-6)
    p_lt = student_t_distribution_p_value(-2.22813885, df=10, alternative="less")
    assert math.isclose(p_lt, 0.025, abs_tol=1e-6)

    # Extreme df=1 (Cauchy): t=12.7062047 gives p=0.05
    p_cauchy = student_t_distribution_p_value(12.7062047, df=1, alternative="two-sided")
    assert math.isclose(p_cauchy, 0.05, abs_tol=1e-6)

    # Benchmark dataset df=399, t=-3.7649: exact p is 0.000192 (0.00019166...)
    p_bm = student_t_distribution_p_value(-3.7649, df=399, alternative="two-sided")
    assert math.isclose(p_bm, 0.00019166, abs_tol=1e-6)

    # Invalid df or alternative
    with pytest.raises(ValueError, match="Degrees of freedom must be >= 1"):
        student_t_distribution_p_value(1.0, df=0)
    with pytest.raises(ValueError, match="alternative must be"):
        student_t_distribution_p_value(1.0, df=5, alternative="invalid")


def test_compare_paired_differences_exact_student_t_and_wilcoxon_precision():
    """Verify compare_paired_differences uses exact Student t and preserves extreme p-values."""
    from agentxray.statistics import compare_paired_differences

    # Strong asymmetric difference across 400 pairs
    x = [10.0] * 350 + [50.0] * 50
    y = [20.0] * 350 + [10.0] * 50
    summary = compare_paired_differences(x, y)

    assert summary.n_pairs == 400
    assert summary.mean_difference == -3.75
    assert summary.paired_t_statistic < 0.0
    # Exact Student's t p-value must be positive float and less than 0.001
    assert 0.0 < summary.paired_t_p_value < 0.001
    # Wilcoxon p-value must be positive float and not truncated to 0.0
    assert 0.0 < summary.wilcoxon_p_value < 0.001

