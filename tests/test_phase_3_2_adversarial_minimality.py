"""Phase 3.2 Adversarial Deterministic Tests for Minimality Classification.

Evaluates 12 boundary conditions covering edge cases where a candidate configuration
could improperly receive or be denied SUPPORTED_1_MINIMAL under Protocol v1.4.

Uses synthetic counts only—zero model inference, zero external API requests.
"""

from __future__ import annotations

import pytest

from agentxray.models import (
    CandidateEvaluation,
    ExperimentConfig,
    InputCase,
    StochasticMinimalityStatus,
)
from agentxray.runner import InterventionRunner
from agentxray.statistics import wilson_score_interval
from agentxray.stochastic_baselines import _classify_stochastic_minimality
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer


def _create_eval(elements: tuple[str, ...], failures: int, trials: int, confidence: float = 0.95) -> CandidateEvaluation:
    rate = failures / trials if trials > 0 else 0.0
    ci = wilson_score_interval(failures, trials, confidence=confidence) if trials > 0 else (0.0, 1.0)
    return CandidateEvaluation(
        candidate_elements=elements,
        trials=trials,
        failures=failures,
        failure_rate=round(rate, 4),
        confidence_interval=ci,
    )


def _get_minimizer(failure_threshold: float = 0.50, baseline_threshold: float | None = 0.20, alpha: float = 0.05, alternative: str = "greater") -> StochasticDeltaDebugMinimizer:
    config = ExperimentConfig(experiment_name="test_adversarial", max_trials=100)
    runner = InterventionRunner(config=config, target_fn=lambda c: None)
    return StochasticDeltaDebugMinimizer(
        runner=runner,
        trials_per_candidate=5,
        failure_threshold=failure_threshold,
        baseline_threshold=baseline_threshold,
        alpha=alpha,
        confidence=0.95,
        alternative=alternative,
    )


def test_case_01_all_checks_pass():
    """Case 1: All Protocol v1.4 criteria pass -> STATISTICALLY_SUPPORTED."""
    minimizer = _get_minimizer()
    cand = ("param_a", "param_b")
    cand_eval = _create_eval(cand, failures=5, trials=5)
    assert cand_eval.confidence_interval[0] > 0.20  # L95 = 0.5655 > 0.20

    removal_evals = {
        "param_a": _create_eval(("param_b",), failures=0, trials=5),
        "param_b": _create_eval(("param_a",), failures=0, trials=5),
    }

    status, msg = minimizer._classify_minimality(cand, cand_eval, removal_evals)
    assert status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
    assert "1-Minimality statistically supported" in msg

    base_status = _classify_stochastic_minimality(
        candidate=cand,
        cand_eval=cand_eval,
        removal_evals=removal_evals,
        failure_threshold=0.50,
        baseline_threshold=0.20,
        alpha=0.05,
        alternative="greater",
    )
    assert base_status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED


def test_case_02_comparative_passes_but_empirical_deletion_guard_fails():
    """Case 2: Comparative Fisher test p < 1e-10, but deletion failure rate >= 0.50 -> NOT_MINIMAL."""
    minimizer = _get_minimizer()
    cand = ("param_a",)
    # Huge sample size where 55% failure rate is statistically lower than 100% (p < 1e-10),
    # but 55% fails the empirical non-failure guard (theta_fail = 0.50).
    cand_eval = _create_eval(cand, failures=100, trials=100)
    removal_evals = {
        "param_a": _create_eval((), failures=55, trials=100),
    }

    status, msg = minimizer._classify_minimality(cand, cand_eval, removal_evals)
    assert status == StochasticMinimalityStatus.NOT_MINIMAL
    assert "Candidate is not 1-minimal" in msg
    assert "still produces failure rate >= threshold" in msg

    base_status = _classify_stochastic_minimality(
        candidate=cand,
        cand_eval=cand_eval,
        removal_evals=removal_evals,
        failure_threshold=0.50,
        baseline_threshold=0.20,
        alpha=0.05,
        alternative="greater",
    )
    assert base_status == StochasticMinimalityStatus.NOT_MINIMAL


def test_case_03_comparative_passes_threshold_binomial_fails():
    """Case 3: Comparative Fisher passes (p=0.0238 < 0.05), threshold guard passes (0.20 < 0.50),

    but Claim A binomial test fails (p=0.1875 > 0.05).
    Under Protocol v1.4 §6.2, candidate is SUPPORTED_1_MINIMAL with diagnostic telemetry.
    """
    minimizer = _get_minimizer()
    cand = ("param_a",)
    # 5/5 vs 1/5: Fisher one-sided p = 1/42 = 0.0238 < 0.05
    # Binomial test for 1/5 against p0=0.50: p = 6/32 = 0.1875 > 0.05
    cand_eval = _create_eval(cand, failures=5, trials=5)
    removal_evals = {
        "param_a": _create_eval((), failures=1, trials=5),
    }

    status, msg = minimizer._classify_minimality(cand, cand_eval, removal_evals)
    assert status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
    assert "threshold binomial all p < 0.05: False" in msg

    base_status = _classify_stochastic_minimality(
        candidate=cand,
        cand_eval=cand_eval,
        removal_evals=removal_evals,
        failure_threshold=0.50,
        baseline_threshold=0.20,
        alpha=0.05,
        alternative="greater",
    )
    assert base_status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED


def test_case_04_candidate_failure_reproduction_fails():
    """Case 4: Candidate failure rate < failure_threshold (2/5 = 0.40 < 0.50) -> INCONCLUSIVE."""
    minimizer = _get_minimizer()
    cand = ("param_a",)
    cand_eval = _create_eval(cand, failures=2, trials=5)
    removal_evals = {
        "param_a": _create_eval((), failures=0, trials=5),
    }

    status, msg = minimizer._classify_minimality(cand, cand_eval, removal_evals)
    assert status == StochasticMinimalityStatus.INCONCLUSIVE
    assert "Candidate failed to reproduce failure rate" in msg

    base_status = _classify_stochastic_minimality(
        candidate=cand,
        cand_eval=cand_eval,
        removal_evals=removal_evals,
        failure_threshold=0.50,
        baseline_threshold=0.20,
    )
    assert base_status == StochasticMinimalityStatus.INCONCLUSIVE


def test_case_05_candidate_wilson_lower_bound_fails():
    """Case 5: Candidate failure rate >= 0.50, but Wilson 95% CI lower bound <= 0.20 -> INCONCLUSIVE.

    Tests regression guard for Protocol v1.4 §6.2 Item 2.
    """
    minimizer = _get_minimizer(baseline_threshold=0.20)
    cand = ("param_a",)
    # 1/2 trials = 0.50. Wilson 95% CI: [0.0945, 0.9055]. L95 = 0.0945 <= 0.20.
    cand_eval = _create_eval(cand, failures=1, trials=2)
    assert cand_eval.failure_rate >= 0.50
    assert cand_eval.confidence_interval[0] <= 0.20

    removal_evals = {
        "param_a": _create_eval((), failures=0, trials=50),
    }

    status, msg = minimizer._classify_minimality(cand, cand_eval, removal_evals)
    assert status == StochasticMinimalityStatus.INCONCLUSIVE
    assert "with CI lower bound > 0.20" in msg

    base_status = _classify_stochastic_minimality(
        candidate=cand,
        cand_eval=cand_eval,
        removal_evals=removal_evals,
        failure_threshold=0.50,
        baseline_threshold=0.20,
    )
    assert base_status == StochasticMinimalityStatus.INCONCLUSIVE


def test_case_06_one_deletion_missing():
    """Case 6: One deletion missing from removal_evals -> INCONCLUSIVE."""
    minimizer = _get_minimizer()
    cand = ("param_a", "param_b")
    cand_eval = _create_eval(cand, failures=5, trials=5)
    # Only param_a evaluated; param_b missing
    removal_evals = {
        "param_a": _create_eval(("param_b",), failures=0, trials=5),
    }

    status, msg = minimizer._classify_minimality(cand, cand_eval, removal_evals)
    assert status == StochasticMinimalityStatus.INCONCLUSIVE
    assert "Missing removal evaluations" in msg
    assert "param_b" in msg

    base_status = _classify_stochastic_minimality(
        candidate=cand,
        cand_eval=cand_eval,
        removal_evals=removal_evals,
        failure_threshold=0.50,
    )
    assert base_status == StochasticMinimalityStatus.INCONCLUSIVE


def test_case_07_one_deletion_malformed_zero_trials():
    """Case 7: One deletion has zero trials completed -> INCONCLUSIVE."""
    minimizer = _get_minimizer()
    cand = ("param_a", "param_b")
    cand_eval = _create_eval(cand, failures=5, trials=5)
    removal_evals = {
        "param_a": _create_eval(("param_b",), failures=0, trials=5),
        "param_b": _create_eval(("param_a",), failures=0, trials=0),
    }

    status, msg = minimizer._classify_minimality(cand, cand_eval, removal_evals)
    assert status == StochasticMinimalityStatus.INCONCLUSIVE
    assert "zero completed trials" in msg

    base_status = _classify_stochastic_minimality(
        candidate=cand,
        cand_eval=cand_eval,
        removal_evals=removal_evals,
        failure_threshold=0.50,
    )
    assert base_status == StochasticMinimalityStatus.INCONCLUSIVE


def test_case_08_one_deletion_fails_comparative_test():
    """Case 8: One deletion passes comparative test, but second deletion has p >= alpha -> OBSERVED_ONLY."""
    minimizer = _get_minimizer()
    cand = ("param_a", "param_b")
    cand_eval = _create_eval(cand, failures=5, trials=5)
    # param_a dropped to 0/5 (p=0.0040 < 0.05)
    # param_b dropped to 2/5 (p=0.0833 > 0.05 for one-sided greater)
    removal_evals = {
        "param_a": _create_eval(("param_b",), failures=0, trials=5),
        "param_b": _create_eval(("param_a",), failures=2, trials=5),
    }

    status, msg = minimizer._classify_minimality(cand, cand_eval, removal_evals)
    assert status == StochasticMinimalityStatus.OBSERVED_ONLY
    assert "failure rate dropped upon every element removal" in msg

    base_status = _classify_stochastic_minimality(
        candidate=cand,
        cand_eval=cand_eval,
        removal_evals=removal_evals,
        failure_threshold=0.50,
        baseline_threshold=0.20,
        alpha=0.05,
        alternative="greater",
    )
    assert base_status == StochasticMinimalityStatus.OBSERVED_ONLY


def test_case_09_candidate_and_deletion_identical_rates():
    """Case 9: Candidate 5/5, deletion 5/5 -> NOT_MINIMAL."""
    minimizer = _get_minimizer()
    cand = ("param_a",)
    cand_eval = _create_eval(cand, failures=5, trials=5)
    removal_evals = {
        "param_a": _create_eval((), failures=5, trials=5),
    }

    status, msg = minimizer._classify_minimality(cand, cand_eval, removal_evals)
    assert status == StochasticMinimalityStatus.NOT_MINIMAL
    assert "still produces failure rate >= threshold" in msg

    base_status = _classify_stochastic_minimality(
        candidate=cand,
        cand_eval=cand_eval,
        removal_evals=removal_evals,
        failure_threshold=0.50,
    )
    assert base_status == StochasticMinimalityStatus.NOT_MINIMAL


def test_case_10_single_element_candidate():
    """Case 10: Single-element candidate |C|=1 evaluated against empty deletion () -> STATISTICALLY_SUPPORTED."""
    minimizer = _get_minimizer()
    cand = ("single_trigger",)
    cand_eval = _create_eval(cand, failures=5, trials=5)
    removal_evals = {
        "single_trigger": _create_eval((), failures=0, trials=5),
    }

    status, msg = minimizer._classify_minimality(cand, cand_eval, removal_evals)
    assert status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
    assert "across all 1 element removals" in msg

    base_status = _classify_stochastic_minimality(
        candidate=cand,
        cand_eval=cand_eval,
        removal_evals=removal_evals,
        failure_threshold=0.50,
        baseline_threshold=0.20,
        alpha=0.05,
        alternative="greater",
    )
    assert base_status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED


def test_case_11_status_and_message_consistency():
    """Case 11: Inconsistent status and message combinations are strictly forbidden.

    Verifies that every return branch produces an unambiguous status-message pairing.
    """
    minimizer = _get_minimizer()

    # Empty candidate -> NOT_APPLICABLE
    s_empty, m_empty = minimizer._classify_minimality((), _create_eval((), 0, 0), {})
    assert s_empty == StochasticMinimalityStatus.NOT_APPLICABLE
    assert "empty" in m_empty

    # Missing removal -> INCONCLUSIVE
    s_miss, m_miss = minimizer._classify_minimality(
        ("a",), _create_eval(("a",), 5, 5), {}
    )
    assert s_miss == StochasticMinimalityStatus.INCONCLUSIVE
    assert "Missing" in m_miss

    # Failing removal -> NOT_MINIMAL
    s_notmin, m_notmin = minimizer._classify_minimality(
        ("a",), _create_eval(("a",), 5, 5), {"a": _create_eval((), 4, 5)}
    )
    assert s_notmin == StochasticMinimalityStatus.NOT_MINIMAL
    assert "not 1-minimal" in m_notmin

    # Supported -> STATISTICALLY_SUPPORTED
    s_supp, m_supp = minimizer._classify_minimality(
        ("a",), _create_eval(("a",), 5, 5), {"a": _create_eval((), 0, 5)}
    )
    assert s_supp == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
    assert "statistically supported" in m_supp


def test_case_12_boundary_p_values_around_alpha():
    """Case 12: Boundary p-values at and immediately around alpha (0.049 vs 0.050 vs 0.051).

    Verifies that strict '< alpha' is enforced.
    """
    # Test strict inequality behavior using compare_proportions / threshold alpha
    minimizer_strict = _get_minimizer(alpha=0.05)

    # Synthetic construct to test alpha boundary
    # If p = 0.049 < 0.05 -> passes
    # If p = 0.050 >= 0.05 -> fails
    cand = ("param_a",)
    cand_eval = _create_eval(cand, failures=5, trials=5)

    # Case A: 5/5 vs 1/5 gives p_comp = 0.0238 < 0.05 -> STATISTICALLY_SUPPORTED
    rem_pass = {"param_a": _create_eval((), failures=1, trials=5)}
    s_pass, _ = minimizer_strict._classify_minimality(cand, cand_eval, rem_pass)
    assert s_pass == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED

    # Case B: 5/5 vs 2/5 gives p_comp = 0.0833 >= 0.05 -> OBSERVED_ONLY
    rem_fail = {"param_a": _create_eval((), failures=2, trials=5)}
    s_fail, _ = minimizer_strict._classify_minimality(cand, cand_eval, rem_fail)
    assert s_fail == StochasticMinimalityStatus.OBSERVED_ONLY

    # Boundary alpha test with customized alpha:
    # If alpha = 0.0833, p = 0.0833 is NOT < alpha -> OBSERVED_ONLY
    minimizer_boundary_eq = _get_minimizer(alpha=0.0833)
    s_eq, _ = minimizer_boundary_eq._classify_minimality(cand, cand_eval, rem_fail)
    assert s_eq == StochasticMinimalityStatus.OBSERVED_ONLY

    # If alpha = 0.0834, p = 0.0833 < alpha -> STATISTICALLY_SUPPORTED
    minimizer_boundary_lt = _get_minimizer(alpha=0.0834)
    s_lt, _ = minimizer_boundary_lt._classify_minimality(cand, cand_eval, rem_fail)
    assert s_lt == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
