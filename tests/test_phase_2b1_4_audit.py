"""Deterministic Tests for Phase 2B.1.4: Final Hypothesis Consistency & Execution Preflight Audit."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import pytest

from agentxray.models import CandidateEvaluation, StochasticMinimalityStatus
from agentxray.statistics import exact_binomial_test
from agentxray.stochastic_baselines import _classify_stochastic_minimality
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.model_pilot.validate_confirmatory_protocol import ConfirmatoryProtocolValidator


# ---------------------------------------------------------------------------
# 1. Exact Binomial Test Mathematical Verification
# ---------------------------------------------------------------------------

def test_exact_binomial_test_known_values() -> None:
    """Verify exact binomial test against analytical PMF sums for n=5, p0=0.50."""
    # k=0, n=5: (0.5)^5 = 1/32 = 0.03125
    p_k0 = exact_binomial_test(0, 5, p0=0.50, alternative="less")
    assert pytest.approx(p_k0, abs=1e-5) == 0.03125

    # k=1, n=5: (1 + 5)/32 = 6/32 = 0.1875
    p_k1 = exact_binomial_test(1, 5, p0=0.50, alternative="less")
    assert pytest.approx(p_k1, abs=1e-5) == 0.1875

    # k=2, n=5: (1 + 5 + 10)/32 = 16/32 = 0.50
    p_k2 = exact_binomial_test(2, 5, p0=0.50, alternative="less")
    assert pytest.approx(p_k2, abs=1e-5) == 0.50

    # k=5, n=5: 32/32 = 1.0
    p_k5 = exact_binomial_test(5, 5, p0=0.50, alternative="less")
    assert pytest.approx(p_k5, abs=1e-5) == 1.0

    # Alternative = "greater"
    # k=5, n=5: 1/32 = 0.03125
    p_k5_greater = exact_binomial_test(5, 5, p0=0.50, alternative="greater")
    assert pytest.approx(p_k5_greater, abs=1e-5) == 0.03125

    # Alternative = "two-sided"
    # k=0, n=5: 2 * 0.03125 = 0.0625
    p_k0_two = exact_binomial_test(0, 5, p0=0.50, alternative="two-sided")
    assert pytest.approx(p_k0_two, abs=1e-5) == 0.0625


# ---------------------------------------------------------------------------
# 2. Critical Audit A: Claim A vs Claim B Mathematical Counterexamples
# ---------------------------------------------------------------------------

def test_counterexample_1_claim_b_significant_claim_a_not() -> None:
    """Candidate 5/5, Deletion 1/5:
    - Claim B (Comparative Fisher one-sided): p = 0.0238 < 0.05 (STATISTICALLY SIGNIFICANT)
    - Claim A (Threshold Binomial): p = 0.1875 > 0.05 (NOT SIGNIFICANT)
    """
    minimizer = StochasticDeltaDebugMinimizer(
        runner=None,  # type: ignore
        trials_per_candidate=5,
        failure_threshold=0.50,
        alpha=0.05,
        alternative="greater",
    )
    cand = ("e1",)
    cand_eval = CandidateEvaluation(
        candidate_elements=cand,
        trials=5,
        failures=5,
        failure_rate=1.0,
        confidence_interval=(0.57, 1.0),
    )
    removals = {
        "e1": CandidateEvaluation(
            candidate_elements=(),
            trials=5,
            failures=1,
            failure_rate=0.20,
            confidence_interval=(0.04, 0.62),
        ),
    }

    status, msg = minimizer._classify_minimality(cand, cand_eval, removals)

    # In Protocol v1.4, Claim B is the primary standard with threshold guard passed:
    assert status == StochasticMinimalityStatus.SUPPORTED_1_MINIMAL
    assert "Intersection-Union Test" in msg
    assert "comparative Fisher all p < 0.05" in msg
    # Verify that telemetry distinguishes Claim B (< 0.05) from Claim A (>= 0.05)
    assert "p_comp=0.0238" in msg  # Claim B
    assert "p_thresh=0.1875" in msg  # Claim A


def test_counterexample_2_claim_a_significant_claim_b_not() -> None:
    """Candidate 3/5, Deletion 0/5:
    - Claim A (Threshold Binomial): p = 0.03125 < 0.05 (STATISTICALLY SIGNIFICANT)
    - Claim B (Comparative Fisher one-sided): p = 0.0833 > 0.05 (NOT SIGNIFICANT)
    Candidate failure rate is 3/5 (60%). Although 0/5 is significantly below 0.50,
    the candidate does not reproduce failure strongly enough to distinguish 3/5 from 0/5.
    """
    minimizer = StochasticDeltaDebugMinimizer(
        runner=None,  # type: ignore
        trials_per_candidate=5,
        failure_threshold=0.50,
        alpha=0.05,
        alternative="greater",
    )
    cand = ("e1",)
    cand_eval = CandidateEvaluation(
        candidate_elements=cand,
        trials=5,
        failures=3,
        failure_rate=0.60,
        confidence_interval=(0.23, 0.88),
    )
    removals = {
        "e1": CandidateEvaluation(
            candidate_elements=(),
            trials=5,
            failures=0,
            failure_rate=0.0,
            confidence_interval=(0.0, 0.43),
        ),
    }

    status, msg = minimizer._classify_minimality(cand, cand_eval, removals)

    # Must be OBSERVED_1_MINIMAL because Claim B (comparative Fisher test) p=0.0833 >= 0.05!
    assert status == StochasticMinimalityStatus.OBSERVED_1_MINIMAL
    assert "p_comp=0.0833" in msg
    assert "p_thresh=0.0312" in msg


def test_counterexample_3_claim_b_significant_but_deletion_above_threshold() -> None:
    """Candidate 50/50, Deletion 30/50:
    - Candidate failure rate: 1.0 (50/50)
    - Deletion failure rate: 0.60 (30/50)
    - Comparative Fisher p < 1e-6 (massive comparative drop)
    - BUT deletion failure rate 0.60 >= failure_threshold (0.50)
    Without the empirical threshold guard, this would incorrectly be called 1-minimal.
    With the threshold guard, it MUST be classified as NOT_MINIMAL.
    """
    minimizer = StochasticDeltaDebugMinimizer(
        runner=None,  # type: ignore
        trials_per_candidate=50,
        failure_threshold=0.50,
        alpha=0.05,
        alternative="greater",
    )
    cand = ("e1", "e2")
    cand_eval = CandidateEvaluation(
        candidate_elements=cand,
        trials=50,
        failures=50,
        failure_rate=1.0,
        confidence_interval=(0.93, 1.0),
    )
    removals = {
        "e1": CandidateEvaluation(
            candidate_elements=("e2",),
            trials=50,
            failures=30,
            failure_rate=0.60,
            confidence_interval=(0.46, 0.72),
        ),
        "e2": CandidateEvaluation(
            candidate_elements=("e1",),
            trials=50,
            failures=0,
            failure_rate=0.0,
            confidence_interval=(0.0, 0.07),
        ),
    }

    status, msg = minimizer._classify_minimality(cand, cand_eval, removals)

    assert status == StochasticMinimalityStatus.NOT_MINIMAL
    assert "still produces failure rate >= threshold" in msg


def test_baseline_stochastic_minimality_classification_claim_b_and_a() -> None:
    """Verify stochastic baseline minimality classifier mirrors the minimizer implementation."""
    cand = ("e1",)
    cand_eval = CandidateEvaluation(
        candidate_elements=cand,
        trials=5,
        failures=5,
        failure_rate=1.0,
        confidence_interval=(0.57, 1.0),
    )
    removals = {
        "e1": CandidateEvaluation(
            candidate_elements=(),
            trials=5,
            failures=1,
            failure_rate=0.20,
            confidence_interval=(0.04, 0.62),
        ),
    }

    status = _classify_stochastic_minimality(
        cand, cand_eval, removals, failure_threshold=0.50, alpha=0.05, alternative="greater"
    )
    assert status == StochasticMinimalityStatus.SUPPORTED_1_MINIMAL


# ---------------------------------------------------------------------------
# 3. Critical Audit B: Exact Wilcoxon Signed-Rank Null Distribution for n=6
# ---------------------------------------------------------------------------

def test_wilcoxon_exact_permutation_distribution_n6() -> None:
    """Exhaustively verify the exact Wilcoxon signed-rank permutation null distribution for n=6.
    
    Ranks: [1, 2, 3, 4, 5, 6]. Total rank sum = 21.
    Total sign permutations: 2^6 = 64.
    """
    ranks = [1, 2, 3, 4, 5, 6]
    n = len(ranks)
    total_perms = 2**n
    assert total_perms == 64

    # Calculate W+ for all 64 sign assignments
    w_plus_distribution: dict[int, int] = {}
    w_min_distribution: dict[int, int] = {}

    for signs in itertools.product([-1, 1], repeat=n):
        w_plus = sum(r for r, s in zip(ranks, signs) if s > 0)
        w_minus = sum(r for r, s in zip(ranks, signs) if s < 0)
        w_stat = min(w_plus, w_minus)

        w_plus_distribution[w_plus] = w_plus_distribution.get(w_plus, 0) + 1
        w_min_distribution[w_stat] = w_min_distribution.get(w_stat, 0) + 1

    # W = min(W+, W-) = 0 occurs only when W+ = 0 (all negative) or W+ = 21 (all positive)
    assert w_min_distribution[0] == 2
    two_sided_p_w0 = 2 / 64
    assert pytest.approx(two_sided_p_w0) == 0.03125

    # W = 1 occurs when sign of rank 1 is flipped (W+ = 1 or W+ = 20)
    assert w_min_distribution[1] == 2
    # Two-sided p-value for W <= 1:
    p_w_le_1 = (w_min_distribution[0] + w_min_distribution[1]) / 64
    assert p_w_le_1 == 4 / 64
    assert pytest.approx(p_w_le_1) == 0.06250

    # CRITICAL AUDIT CONCLUSION:
    # 0.03125 is the ONLY two-sided p-value <= 0.05 for n=6.
    # Therefore, rejecting H0 at alpha=0.05 strictly requires W=0 (unanimous win on all 6 mechanisms).
    assert two_sided_p_w0 < 0.05
    assert p_w_le_1 > 0.05


def test_wilcoxon_exact_permutation_distribution_n5_impossible_to_reject() -> None:
    """Demonstrate that with n=5 (e.g. if 1 mechanism ties out of 6), alpha=0.05 cannot be rejected."""
    n = 5
    total_perms = 2**n  # 32
    # Even for W = 0 (unanimous positive or negative), two-sided p-value is:
    min_two_sided_p = 2 / total_perms
    assert pytest.approx(min_two_sided_p) == 0.06250
    # 0.0625 > 0.05 -> mathematically impossible to achieve statistical significance at 0.05!
    assert min_two_sided_p > 0.05


# ---------------------------------------------------------------------------
# 4. Protocol v1.4 and Configuration v1.4 Structural & Invariant Tests
# ---------------------------------------------------------------------------

def test_protocol_v1_4_validator_passes() -> None:
    """The frozen v1.4 protocol and config must pass 100% of validator checks."""
    config_path = "benchmarks/configs/phase_2b1_exploratory_config_v1_4.json"
    protocol_path = "docs/research/phase_2b1_exploratory_protocol_v1_4.md"
    validator = ConfirmatoryProtocolValidator(config_path=config_path, protocol_path=protocol_path)
    summary = validator.validate()

    assert summary["validation_status"] == "VALIDATION_PASSED"
    assert summary["all_checks_passed"] is True
    assert summary["total_discrepancies"] == 0
    assert summary["total_checks_passed"] >= 35


def test_protocol_v1_4_mutation_tamper_detection(tmp_path: Path) -> None:
    """Mutations to v1.4 config or protocol must trigger validator discrepancies."""
    base_cfg_path = Path("benchmarks/configs/phase_2b1_exploratory_config_v1_4.json")
    base_proto_path = Path("docs/research/phase_2b1_exploratory_protocol_v1_4.md")

    # Mutation 1: Change primary test back to unclustered paired test
    mutant_cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    mutant_cfg["statistical_analysis_plan"]["primary_test"] = "wilcoxon_signed_rank_paired"
    m_cfg_file = tmp_path / "mutant_config_1.json"
    m_cfg_file.write_text(json.dumps(mutant_cfg), encoding="utf-8")

    validator = ConfirmatoryProtocolValidator(config_path=m_cfg_file, protocol_path=base_proto_path)
    summary = validator.validate()
    assert summary["validation_status"] == "VALIDATION_FAILED"
    assert any("v1_4_primary_test" in d for d in summary["discrepancies"])

    # Mutation 2: Change pooled_test_status to confirmatory
    mutant_cfg2 = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    mutant_cfg2["statistical_analysis_plan"]["pooled_test_status"] = "CONFIRMATORY_PRIMARY"
    m_cfg_file2 = tmp_path / "mutant_config_2.json"
    m_cfg_file2.write_text(json.dumps(mutant_cfg2), encoding="utf-8")

    validator2 = ConfirmatoryProtocolValidator(config_path=m_cfg_file2, protocol_path=base_proto_path)
    summary2 = validator2.validate()
    assert summary2["validation_status"] == "VALIDATION_FAILED"
    assert any("v1_4_pooled_test_status" in d for d in summary2["discrepancies"])

    # Mutation 3: Change cluster_unit
    mutant_cfg3 = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    mutant_cfg3["statistical_analysis_plan"]["cluster_unit"] = "task_instance"
    m_cfg_file3 = tmp_path / "mutant_config_3.json"
    m_cfg_file3.write_text(json.dumps(mutant_cfg3), encoding="utf-8")

    validator3 = ConfirmatoryProtocolValidator(config_path=m_cfg_file3, protocol_path=base_proto_path)
    summary3 = validator3.validate()
    assert summary3["validation_status"] == "VALIDATION_FAILED"
    assert any("v1_4_cluster_unit" in d for d in summary3["discrepancies"])
