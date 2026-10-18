"""Unit and benchmark tests for exact Fisher power analysis and joint sample-size solvers."""

from fractions import Fraction
import math
import pytest

from agentxray.power import (
    FisherPowerResult,
    JointPowerResult,
    PowerSolverStatus,
    exact_fisher_p_value_unrounded,
    exact_fisher_power,
    solve_fisher_sample_size,
    solve_joint_minimality_power,
)


def test_exact_fisher_p_value_unrounded_vs_rational():
    """Verify unrounded Fisher p-values match exact rational fraction computation."""
    # Representative 2x2 tables across balanced n=10, 20, 50, including near alpha=0.05
    test_tables = [
        (9, 10, 1, 10),
        (8, 10, 2, 10),
        (7, 10, 3, 10),
        (6, 10, 2, 10),
        (16, 20, 4, 20),
        (15, 20, 7, 20),
        (40, 50, 20, 50),
        (35, 50, 25, 50),
    ]

    for k1, n1, k2, n2 in test_tables:
        p_float = exact_fisher_p_value_unrounded(k1, n1, k2, n2, alternative="greater")

        # Compute exact rational fraction
        K = k1 + k2
        N = n1 + n2
        max_x = min(n1, K)
        denom = math.comb(N, K)
        num = sum(math.comb(n1, x) * math.comb(n2, K - x) for x in range(k1, max_x + 1))
        p_frac = float(Fraction(num, denom))

        assert math.isclose(p_float, p_frac, abs_tol=1e-12)


def test_stage_1_benchmark_case_1_large_effect():
    """Verify Case 1 (p1=0.9, p2=0.1, alpha=0.05, target=0.80):
    power near 0.736099 at n=5, 0.889130 at n=6, 0.863607 at n=7; required n*=6.
    """
    pw5 = exact_fisher_power(5, 0.9, 0.1, alpha=0.05)
    pw6 = exact_fisher_power(6, 0.9, 0.1, alpha=0.05)
    pw7 = exact_fisher_power(7, 0.9, 0.1, alpha=0.05)

    assert math.isclose(pw5, 0.736099, abs_tol=1e-5)
    assert math.isclose(pw6, 0.889130, abs_tol=1e-5)
    assert math.isclose(pw7, 0.863607, abs_tol=1e-5)

    res = solve_fisher_sample_size(p1=0.9, p2=0.1, target_power=0.80, alpha=0.05, n_max=120)
    assert res.status == PowerSolverStatus.VALIDATED
    assert res.n_required == 6
    assert math.isclose(res.power_achieved, 0.889130, abs_tol=1e-5)


def test_stage_1_benchmark_case_2_moderate_effect():
    """Verify Case 2 (p1=0.8, p2=0.2, alpha=0.05, target=0.80):
    power near 0.720787 at n=9, 0.805390 at n=10; required n*=10.
    """
    pw9 = exact_fisher_power(9, 0.8, 0.2, alpha=0.05)
    pw10 = exact_fisher_power(10, 0.8, 0.2, alpha=0.05)

    assert math.isclose(pw9, 0.720787, abs_tol=1e-5)
    assert math.isclose(pw10, 0.805390, abs_tol=1e-5)

    res = solve_fisher_sample_size(p1=0.8, p2=0.2, target_power=0.80, alpha=0.05, n_max=120)
    assert res.status == PowerSolverStatus.VALIDATED
    assert res.n_required == 10
    assert math.isclose(res.power_achieved, 0.805390, abs_tol=1e-5)


def test_stage_1_benchmark_sawtooth_dip_and_monotonic_ceiling():
    """Verify that for target power 0.90 with p1=0.8, p2=0.2, the solver detects
    the dip at n=13 (about 0.883247) despite n=12 reaching 0.910900, and returns n*=14.
    """
    pw12 = exact_fisher_power(12, 0.8, 0.2, alpha=0.05)
    pw13 = exact_fisher_power(13, 0.8, 0.2, alpha=0.05)
    pw14 = exact_fisher_power(14, 0.8, 0.2, alpha=0.05)

    assert math.isclose(pw12, 0.910900, abs_tol=1e-5)
    assert math.isclose(pw13, 0.883247, abs_tol=1e-5)
    assert pw13 < 0.90  # Dip below target!
    assert math.isclose(pw14, 0.910619, abs_tol=1e-5)
    assert pw14 >= 0.90

    res = solve_fisher_sample_size(p1=0.8, p2=0.2, target_power=0.90, alpha=0.05, n_max=120)
    assert res.status == PowerSolverStatus.VALIDATED
    assert res.n_required == 14  # Must be 14, NOT 12!
    assert math.isclose(res.power_achieved, 0.910619, abs_tol=1e-5)


def test_stage_1_range_exhaustion():
    """Verify range exhaustion for weak effect (p1=0.55, p2=0.50) with n_max=120."""
    res = solve_fisher_sample_size(p1=0.55, p2=0.50, target_power=0.80, alpha=0.05, n_max=120)
    assert res.status == PowerSolverStatus.RANGE_EXHAUSTED
    assert res.n_required is None
    assert res.power_achieved < 0.80
    assert "Search range exhausted" in res.diagnostic_message


def test_stage_1_input_validation():
    """Verify strict validation of input domains and edge cases."""
    with pytest.raises(ValueError, match="strictly greater than ablation rate"):
        solve_fisher_sample_size(p1=0.5, p2=0.5)

    with pytest.raises(ValueError, match="strictly greater than ablation rate"):
        solve_fisher_sample_size(p1=0.3, p2=0.7)

    with pytest.raises(ValueError, match="target_power must be in"):
        solve_fisher_sample_size(p1=0.9, p2=0.1, target_power=1.5)

    with pytest.raises(ValueError, match="alpha must be in"):
        solve_fisher_sample_size(p1=0.9, p2=0.1, alpha=-0.05)

    with pytest.raises(ValueError, match="n_max .* must be >= n_min"):
        solve_fisher_sample_size(p1=0.9, p2=0.1, n_min=50, n_max=10)


def test_stage_2_joint_power_k_1():
    """Verify joint power solver with k=1 matches single-component solver."""
    res = solve_joint_minimality_power(
        p1=0.9,
        p2_rates=0.1,
        k=1,
        joint_target_power=0.80,
        alpha=0.05,
    )
    assert res.status == PowerSolverStatus.VALIDATED
    assert res.n_required == 6
    assert res.k_components == 1
    assert math.isclose(res.component_target_power, 0.80)
    assert math.isclose(res.component_beta_target, 0.20)


def test_stage_2_joint_power_k_2_union_bound():
    """Verify joint power solver for k=2 allocates beta_j = 0.20 / 2 = 0.10 (component power 0.90).
    For p1=0.9, p2=0.1: required n* = 8.
    For p1=0.8, p2=0.2: required n* = 14.
    """
    res1 = solve_joint_minimality_power(
        p1=0.9,
        p2_rates=0.1,
        k=2,
        joint_target_power=0.80,
        alpha=0.05,
    )
    assert res1.status == PowerSolverStatus.VALIDATED
    assert res1.n_required == 8
    assert res1.k_components == 2
    assert math.isclose(res1.component_target_power, 0.90)
    assert math.isclose(res1.component_beta_target, 0.10)

    res2 = solve_joint_minimality_power(
        p1=0.8,
        p2_rates=0.2,
        k=2,
        joint_target_power=0.80,
        alpha=0.05,
    )
    assert res2.status == PowerSolverStatus.VALIDATED
    assert res2.n_required == 14
    assert res2.k_components == 2
    assert math.isclose(res2.component_target_power, 0.90)


def test_stage_2_joint_power_heterogeneous_rates():
    """Verify joint power solver with heterogeneous per-component ablation rates."""
    # Component 0 has p2=0.10 (requires n=8 for target 0.90)
    # Component 1 has p2=0.20 (requires n=10 for target 0.90 with p1=0.90)
    res = solve_joint_minimality_power(
        p1=0.9,
        p2_rates=[0.10, 0.20],
        joint_target_power=0.80,
        alpha=0.05,
    )
    assert res.status == PowerSolverStatus.VALIDATED
    assert res.k_components == 2
    assert res.component_results[0].n_required == 8
    assert res.component_results[1].n_required == 10
    assert res.n_required == 10  # max(8, 10)


def test_stage_2_joint_power_invalid_inputs():
    """Verify validation for joint power inputs."""
    with pytest.raises(ValueError, match="k must be positive integer"):
        solve_joint_minimality_power(p1=0.9, p2_rates=0.1, k=0)

    with pytest.raises(ValueError, match="p2_rates sequence cannot be empty"):
        solve_joint_minimality_power(p1=0.9, p2_rates=[])

    with pytest.raises(ValueError, match="does not match length of p2_rates"):
        solve_joint_minimality_power(p1=0.9, p2_rates=[0.1, 0.2], k=3)
