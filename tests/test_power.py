"""Unit and benchmark tests for exact Fisher power analysis and joint sample-size solvers."""

from fractions import Fraction
import math
import pytest

from agentxray.power import (
    FisherPowerResult,
    JointPowerResult,
    McNemarPowerResult,
    PowerSolverStatus,
    exact_fisher_p_value_unrounded,
    exact_fisher_power,
    exact_mcnemar_power,
    solve_fisher_sample_size,
    solve_joint_minimality_power,
    solve_mcnemar_sample_size,
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


# ---------------------------------------------------------------------------
# Exact McNemar Power & Sample-Size Tests
# ---------------------------------------------------------------------------

def test_exact_mcnemar_power_small_cases_rational():
    """Verify exact McNemar power matches hand-calculated rational fraction enumeration on small cases.

    For N=5, psi=0.5, theta=3.0, alpha=0.05 (alternative='greater'):
    - Rejection requires n_d=5 and b=5.
    - P(n_d=5) = (0.5)^5 = 1/32.
    - P(b=5 | n_d=5) = (0.75)^5 = 243/1024.
    - Exact Power = (0.375)^5 = 243 / 32768 = 0.007415771484375.

    For N=6:
    - n_d=5 requires b=5: P(n_d=5, b=5) = 6 * (0.5)^6 * (0.75)^5 = 1458 / 65536.
    - n_d=6 requires b=6: P(n_d=6, b=6) = (0.5)^6 * (0.75)^6 = 729 / 262144.
    - Exact Power = 1458/65536 + 729/262144 = 6561 / 262144 = 0.025028228759765625.

    For N <= 4:
    - Since 0.5^4 = 0.0625 > 0.05, no rejection boundary exists; power must be strictly 0.0.
    """
    for n in range(1, 5):
        assert exact_mcnemar_power(n=n, psi=0.5, theta=3.0, alpha=0.05) == 0.0

    pw5 = exact_mcnemar_power(n=5, psi=0.5, theta=3.0, alpha=0.05)
    expected_frac_5 = float(Fraction(243, 32768))
    assert math.isclose(pw5, expected_frac_5, abs_tol=1e-12)
    assert math.isclose(pw5, 0.007415771484375, abs_tol=1e-12)

    pw6 = exact_mcnemar_power(n=6, psi=0.5, theta=3.0, alpha=0.05)
    expected_frac_6 = float(Fraction(6561, 262144))
    assert math.isclose(pw6, expected_frac_6, abs_tol=1e-12)
    assert math.isclose(pw6, 0.025028228759765625, abs_tol=1e-12)


def test_exact_mcnemar_power_null_calibration():
    """Verify that under the null hypothesis (theta = 1.0), power is bounded by alpha.

    Exact discrete tests are strictly conservative: rejection rate under H0 <= alpha.
    """
    for n in (10, 20, 50, 100):
        pw = exact_mcnemar_power(n=n, psi=0.5, theta=1.0, alpha=0.05, alternative="greater")
        assert 0.0 <= pw <= 0.05, f"Type I error rate {pw} exceeded alpha=0.05 at N={n}"


def test_exact_mcnemar_power_monotonicity_with_n():
    """Verify that exact McNemar power is monotonically non-decreasing as sample size N increases."""
    powers = [exact_mcnemar_power(n=n, psi=0.4, theta=3.0, alpha=0.05) for n in range(5, 51)]
    for i in range(len(powers) - 1):
        assert powers[i] <= powers[i + 1] + 1e-12, (
            f"Power decreased from N={i + 5} ({powers[i]}) to N={i + 6} ({powers[i + 1]})"
        )


def test_exact_mcnemar_power_boundary_cases():
    """Verify edge cases: psi=0.0, theta=0.0, and psi=1.0."""
    # When discordance is 0, no discordant pairs can occur -> power is 0.0
    assert exact_mcnemar_power(n=20, psi=0.0, theta=3.0) == 0.0

    # When odds ratio is 0 for alternative='greater' -> power is 0.0
    assert exact_mcnemar_power(n=20, psi=0.5, theta=0.0, alternative="greater") == 0.0

    # When psi=1.0, exactly N discordant pairs are guaranteed (n_d = N)
    pw_full = exact_mcnemar_power(n=10, psi=1.0, theta=3.0, alpha=0.05)
    # Conditional rejection probability for n_d=10 under p_disc=0.75
    # For n_d=10: b >= 9 gives p_val <= 0.05
    # P(B >= 9 | n_d=10, p=0.75) = comb(10, 9)*0.75^9*0.25^1 + comb(10, 10)*0.75^10
    expected_full = 10 * (0.75 ** 9) * 0.25 + (0.75 ** 10)
    assert math.isclose(pw_full, expected_full, abs_tol=1e-10)


def test_exact_mcnemar_power_input_validation():
    """Verify parameter validation rejects nonsensical inputs with clear ValueErrors."""
    with pytest.raises(ValueError, match="Sample size n must be positive integer"):
        exact_mcnemar_power(n=0, psi=0.5, theta=2.0)

    with pytest.raises(ValueError, match="Discordance probability psi must be in"):
        exact_mcnemar_power(n=10, psi=-0.1, theta=2.0)

    with pytest.raises(ValueError, match="Discordance probability psi must be in"):
        exact_mcnemar_power(n=10, psi=1.5, theta=2.0)

    with pytest.raises(ValueError, match="Odds ratio theta must be non-negative"):
        exact_mcnemar_power(n=10, psi=0.5, theta=-1.0)

    with pytest.raises(ValueError, match="Significance level alpha must be in"):
        exact_mcnemar_power(n=10, psi=0.5, theta=2.0, alpha=0.0)

    with pytest.raises(ValueError, match="alternative must be"):
        exact_mcnemar_power(n=10, psi=0.5, theta=2.0, alternative="invalid_alt")


def test_solve_mcnemar_sample_size_minimality():
    """Verify solve_mcnemar_sample_size returns the minimal qualifying sample size N*.

    Guarantees:
    - Power(N*) >= target_power
    - For all n in [n_min, N* - 1], Power(n) < target_power
    """
    res = solve_mcnemar_sample_size(
        psi=0.4,
        theta=3.0,
        target_power=0.80,
        alpha=0.05,
        n_min=1,
        n_max=150,
    )
    assert res.status == PowerSolverStatus.VALIDATED
    assert res.n_required is not None
    n_star = res.n_required

    # Must achieve target power at N*
    assert res.powers_by_n[n_star] >= 0.80
    assert math.isclose(res.power_achieved, res.powers_by_n[n_star], abs_tol=1e-6)

    # Every smaller sample size must fail target power
    for smaller_n in range(res.n_min, n_star):
        assert res.powers_by_n[smaller_n] < 0.80, f"Sample size {smaller_n} already achieved target power!"

    # Specific validated sample size
    assert n_star == 67


def test_solve_mcnemar_sample_size_range_exhausted():
    """Verify solve_mcnemar_sample_size reports range_exhausted when target power is unreachable."""
    res = solve_mcnemar_sample_size(
        psi=0.2,
        theta=2.0,
        target_power=0.95,
        alpha=0.05,
        n_min=1,
        n_max=50,
    )
    assert res.status == PowerSolverStatus.RANGE_EXHAUSTED
    assert res.n_required is None
    assert res.power_achieved < 0.95
    assert "Search range exhausted" in res.diagnostic_message


def test_solve_mcnemar_sample_size_input_validation():
    """Verify solve_mcnemar_sample_size input validation."""
    with pytest.raises(ValueError, match="psi must be in"):
        solve_mcnemar_sample_size(psi=0.0, theta=3.0)

    with pytest.raises(ValueError, match="theta .* must be strictly greater than 1.0"):
        solve_mcnemar_sample_size(psi=0.4, theta=1.0, alternative="greater")

    with pytest.raises(ValueError, match="target_power must be in"):
        solve_mcnemar_sample_size(psi=0.4, theta=3.0, target_power=1.0)

    with pytest.raises(ValueError, match="n_max .* must be >= n_min"):
        solve_mcnemar_sample_size(psi=0.4, theta=3.0, n_min=50, n_max=10)

