"""Tests for Phase 4.5E prospective power validation, missingness sensitivity, and status precedence."""

import json
import pytest
from pydantic import ValidationError

from agentxray.models import (
    CandidateEvaluation,
    ExperimentConfig,
    InputCase,
    StochasticDiscoveredCondition,
    StochasticMinimalityStatus,
    TargetOutcome,
    TargetResult,
)
from agentxray.power import (
    PowerSolverStatus,
    exact_fisher_p_value_unrounded,
    solve_joint_minimality_power,
)
from agentxray.reporting import (
    format_model_pilot_json_report,
    format_model_pilot_markdown_report,
)
from agentxray.runner import InterventionRunner
from agentxray.verification import (
    VerificationEvidenceTier,
    VerificationExecutionStatus,
    VerificationResult,
    evaluate_missingness_sensitivity,
    is_confirmatory_power_established,
    verify_candidate_stochastically,
)


def test_model_invariants_confirmatory_power():
    """Verify that confirmatory_power_established=True requires validated status, confirmatory tier, and adequate sample size."""
    # 1. Valid configuration succeeds
    res_valid = VerificationResult(
        candidate=("a",),
        minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
        verification_status=VerificationExecutionStatus.COMPLETE,
        evidence_tier=VerificationEvidenceTier.CONFIRMATORY,
        sample_size_rationale="prospective_power_validated",
        sample_size_validation_status="validated",
        planned_sample_size_adequate=True,
        confirmatory_power_established=True,
    )
    assert res_valid.confirmatory_power_established is True
    assert res_valid.planned_sample_size_adequate is True

    # 2. confirmatory_power_established=True fails if planned_sample_size_adequate is False
    with pytest.raises(ValueError, match="confirmatory_power_established cannot be True without"):
        VerificationResult(
            candidate=("a",),
            minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
            verification_status=VerificationExecutionStatus.COMPLETE,
            evidence_tier=VerificationEvidenceTier.CONFIRMATORY,
            sample_size_rationale="test",
            sample_size_validation_status="validated",
            planned_sample_size_adequate=False,
            confirmatory_power_established=True,
        )

    # 3. confirmatory_power_established=True fails if evidence_tier is not confirmatory
    with pytest.raises(ValueError, match="cannot coexist with evidence_tier"):
        VerificationResult(
            candidate=("a",),
            minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
            verification_status=VerificationExecutionStatus.COMPLETE,
            evidence_tier=VerificationEvidenceTier.EXPLORATORY,
            sample_size_rationale="test",
            sample_size_validation_status="validated",
            planned_sample_size_adequate=True,
            confirmatory_power_established=True,
        )

    # 4. confirmatory_power_established=True fails if validation status is not validated
    with pytest.raises(ValueError, match="cannot coexist with sample_size_validation_status"):
        VerificationResult(
            candidate=("a",),
            minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
            verification_status=VerificationExecutionStatus.COMPLETE,
            evidence_tier=VerificationEvidenceTier.CONFIRMATORY,
            sample_size_rationale="test",
            sample_size_validation_status="range_exhausted",
            planned_sample_size_adequate=True,
            confirmatory_power_established=True,
        )


def test_is_confirmatory_power_established_helper():
    """Verify is_confirmatory_power_established helper semantics."""
    assert is_confirmatory_power_established("confirmatory", validation_status="validated", planned_sample_size_adequate=True) is True
    assert is_confirmatory_power_established("confirmatory", validation_status="validated", planned_sample_size_adequate=False) is False
    assert is_confirmatory_power_established("confirmatory", validation_status="unresolved", planned_sample_size_adequate=True) is False
    assert is_confirmatory_power_established("exploratory", validation_status="validated", planned_sample_size_adequate=True) is False


def test_confirmatory_verification_with_adequate_prospective_power():
    """Scenario 1: Full confirmatory execution with adequate prospective power and positive result (STATISTICALLY_SUPPORTED)."""
    initial_input = InputCase(case_id="case_conf", elements=("p_a", "p_b"))
    candidate = ("p_a", "p_b")

    def target_fn(case: InputCase) -> TargetResult:
        elems = set(case.elements)
        # Fails only when BOTH p_a and p_b are present
        is_fail = ("p_a" in elems and "p_b" in elems)
        return TargetResult(
            case_id=case.case_id,
            outcome=TargetOutcome.FAILURE if is_fail else TargetOutcome.SUCCESS,
            observable_signals={},
        )

    runner = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_conf", max_trials=100),
        target_fn=target_fn,
    )

    # p1=0.9, p2=0.1, k=2 requires n*=8 per arm for joint power >= 0.80 (component target 0.90)
    res = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=8,
        failure_threshold=0.50,
        alternative="greater",
        evidence_tier="confirmatory",
        trials_explicitly_set=True,
        prospective_p1=0.90,
        prospective_p2=0.10,
        prospective_target_power=0.80,
        prospective_alpha=0.05,
    )

    assert res.evidence_tier == VerificationEvidenceTier.CONFIRMATORY
    assert res.sample_size_validation_status == "validated"
    assert res.prospective_power_validated is True
    assert res.planned_sample_size_adequate is True
    assert res.confirmatory_power_established is True
    assert res.execution_protocol_adherent is True
    assert res.missingness_sensitivity_status == "not_applicable"
    assert res.minimality_status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
    assert "prospective confirmatory Intersection-Union Test" in res.status_message


def test_adequately_powered_non_rejection_observed_drop():
    """Scenario 2: Adequately powered non-rejection where observed failure rates dropped but did not reach statistical significance."""
    initial_input = InputCase(case_id="case_drop", elements=("p_a",))
    candidate = ("p_a",)

    # Candidate has 6/8 failures (75%); ablation has 3/8 failures (37.5% < 0.50 threshold).
    # Fisher exact test (6, 8, 3, 8) p = 0.1573 >= 0.05. Drop observed, but non-rejection!
    call_counts = {"cand": 0, "ablat": 0}

    def target_fn(case: InputCase) -> TargetResult:
        if "p_a" in case.elements:
            call_counts["cand"] += 1
            is_fail = call_counts["cand"] <= 6
        else:
            call_counts["ablat"] += 1
            is_fail = call_counts["ablat"] <= 3
        return TargetResult(
            case_id=case.case_id,
            outcome=TargetOutcome.FAILURE if is_fail else TargetOutcome.SUCCESS,
            observable_signals={},
        )

    runner = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_drop", max_trials=100),
        target_fn=target_fn,
    )

    res = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=8,
        failure_threshold=0.50,
        alternative="greater",
        evidence_tier="confirmatory",
        trials_explicitly_set=True,
        prospective_p1=0.90,
        prospective_p2=0.10,
        prospective_target_power=0.80,
    )

    assert res.confirmatory_power_established is True
    assert res.planned_sample_size_adequate is True
    assert res.minimality_status == StochasticMinimalityStatus.OBSERVED_ONLY
    assert "adequately powered non-rejection; does not prove dispensability" in res.status_message


def test_adequately_powered_non_rejection_no_drop():
    """Scenario 3: Adequately powered non-rejection where observed failure rate did not drop (NOT_MINIMAL)."""
    initial_input = InputCase(case_id="case_nodrop", elements=("p_a",))
    candidate = ("p_a",)

    # Candidate has 6/8 failures (75%); ablation has 7/8 failures (87.5%).
    call_counts = {"cand": 0, "ablat": 0}

    def target_fn(case: InputCase) -> TargetResult:
        if "p_a" in case.elements:
            call_counts["cand"] += 1
            is_fail = call_counts["cand"] <= 6
        else:
            call_counts["ablat"] += 1
            is_fail = call_counts["ablat"] <= 7
        return TargetResult(
            case_id=case.case_id,
            outcome=TargetOutcome.FAILURE if is_fail else TargetOutcome.SUCCESS,
            observable_signals={},
        )

    runner = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_nodrop", max_trials=100),
        target_fn=target_fn,
    )

    res = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=8,
        failure_threshold=0.50,
        alternative="greater",
        evidence_tier="confirmatory",
        trials_explicitly_set=True,
        prospective_p1=0.90,
        prospective_p2=0.10,
        prospective_target_power=0.80,
    )

    assert res.confirmatory_power_established is True
    assert res.minimality_status == StochasticMinimalityStatus.NOT_MINIMAL
    assert "observed-data classification, not an equivalence or non-inferiority proof" in res.status_message


def test_underpowered_confirmatory_execution():
    """Scenario 4: Confirmatory execution where planned sample size is smaller than required (n=6 < n*=8)."""
    initial_input = InputCase(case_id="case_under", elements=("p_a", "p_b"))
    candidate = ("p_a", "p_b")

    def target_fn(case: InputCase) -> TargetResult:
        is_fail = len(case.elements) == 2
        return TargetResult(
            case_id=case.case_id,
            outcome=TargetOutcome.FAILURE if is_fail else TargetOutcome.SUCCESS,
            observable_signals={},
        )

    runner = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_under", max_trials=100),
        target_fn=target_fn,
    )

    # Required n*=8 for (0.9, 0.1, k=2), but caller executes only 6 trials
    res = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=6,
        failure_threshold=0.50,
        alternative="greater",
        evidence_tier="confirmatory",
        trials_explicitly_set=True,
        prospective_p1=0.90,
        prospective_p2=0.10,
        prospective_target_power=0.80,
    )

    assert res.evidence_tier == VerificationEvidenceTier.CONFIRMATORY
    assert res.sample_size_validation_status == "validated"
    assert res.prospective_power_validated is True
    assert res.planned_sample_size_adequate is False
    assert res.confirmatory_power_established is False
    assert res.minimality_status == StochasticMinimalityStatus.OBSERVED_ONLY
    assert "confirmatory power was not established (exploratory significance" in res.status_message


def test_range_exhaustion_in_verification():
    """Scenario 5: Prospective power solver exhausts supported domain [1, 120] for weak effect."""
    initial_input = InputCase(case_id="case_exhaust", elements=("p_a",))
    candidate = ("p_a",)

    def target_fn(case: InputCase) -> TargetResult:
        return TargetResult(
            case_id=case.case_id,
            outcome=TargetOutcome.FAILURE,
            observable_signals={},
        )

    runner = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_exhaust", max_trials=100),
        target_fn=target_fn,
    )

    res = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=10,
        failure_threshold=0.50,
        evidence_tier="confirmatory",
        trials_explicitly_set=True,
        prospective_p1=0.55,
        prospective_p2=0.50,
        prospective_target_power=0.80,
    )

    assert res.sample_size_validation_status == "range_exhausted"
    assert res.confirmatory_power_established is False
    assert res.planned_sample_size_adequate is False
    assert res.minimality_status != StochasticMinimalityStatus.STATISTICALLY_SUPPORTED


def test_missingness_sensitivity_robust_caps_status():
    """Scenario 6A: Missingness present, but nominal and worst-case sensitivity both reject.
    Protocol adherence failed -> status is capped at OBSERVED_ONLY (cannot be STATISTICALLY_SUPPORTED).
    """
    initial_input = InputCase(case_id="case_miss_rob", elements=("p_a",))
    candidate = ("p_a",)

    def target_fn(case: InputCase) -> TargetResult:
        is_fail = ("p_a" in case.elements)
        return TargetResult(
            case_id=case.case_id,
            outcome=TargetOutcome.FAILURE if is_fail else TargetOutcome.SUCCESS,
            observable_signals={},
        )

    runner = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_miss_rob", max_trials=100),
        target_fn=target_fn,
    )

    # 10 completed trials (10/10 vs 0/10) with 1 missing trial on candidate arm.
    # Imputed: candidate 10/11, ablation 0/10. Fisher exact p = 0.000003 < 0.05. Robust!
    res = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=10,
        failure_threshold=0.50,
        evidence_tier="confirmatory",
        trials_explicitly_set=True,
        prospective_p1=0.90,
        prospective_p2=0.10,
        candidate_missing_trials=1,
    )

    assert res.execution_protocol_adherent is False
    assert res.missingness_sensitivity_status == "robust"
    assert res.minimality_status == StochasticMinimalityStatus.OBSERVED_ONLY
    assert "Qualified 1-minimality" in res.status_message
    assert "confirmatory status is withheld due to protocol attrition" in res.status_message


def test_missingness_sensitivity_fragile_yields_inconclusive():
    """Scenario 6B: Nominal test passes, but significance collapses under worst-case sensitivity imputation (INCONCLUSIVE)."""
    initial_input = InputCase(case_id="case_miss_frag", elements=("p_a",))
    candidate = ("p_a",)

    call_counts = {"cand": 0, "ablat": 0}

    # Candidate: 6/6 failures. Ablation: 2/6 failures.
    # Nominal Fisher exact (6/6 vs 2/6) p = 0.0303 < 0.05 (passes nominally).
    # If ablation had 2 missing trials imputed as Failure (Y=1):
    # Imputed ablation: (2 + 2) / (6 + 2) = 4/8 failures.
    # Imputed candidate: 6/6 failures.
    # Fisher exact (6/6 vs 4/8) p = 0.0515 >= 0.05 (fails sensitivity!).
    def target_fn(case: InputCase) -> TargetResult:
        if "p_a" in case.elements:
            call_counts["cand"] += 1
            is_fail = True
        else:
            call_counts["ablat"] += 1
            is_fail = (call_counts["ablat"] <= 2)
        return TargetResult(
            case_id=case.case_id,
            outcome=TargetOutcome.FAILURE if is_fail else TargetOutcome.SUCCESS,
            observable_signals={},
        )

    runner = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_miss_frag", max_trials=100),
        target_fn=target_fn,
    )

    res = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=6,
        failure_threshold=0.50,
        evidence_tier="confirmatory",
        trials_explicitly_set=True,
        prospective_p1=0.90,
        prospective_p2=0.10,
        ablation_missing_trials={"p_a": 2},
    )

    assert res.execution_protocol_adherent is False
    assert res.missingness_sensitivity_status == "fragile"
    assert res.minimality_status == StochasticMinimalityStatus.INCONCLUSIVE
    assert "collapses under adversarial missingness sensitivity analysis" in res.status_message


def test_strict_significance_boundary_p_equals_alpha():
    """Verify that strict inequality p < alpha is enforced when p == alpha exactly.

    For candidate failures 3/3 (100%) and ablation failures 0/3 (0%) under alternative='greater',
    the hypergeometric tail probability is comb(3,3)*comb(3,0) / comb(6,3) = 1/20 = 0.05.
    At alpha=0.05, p < alpha evaluates to False.
    The result must NOT be classified as STATISTICALLY_SUPPORTED; it must yield OBSERVED_ONLY.
    """
    initial_input = InputCase(case_id="case_p_alpha", elements=("p_a",))
    candidate = ("p_a",)

    def target_fn(case: InputCase) -> TargetResult:
        is_fail = ("p_a" in case.elements)
        return TargetResult(
            case_id=case.case_id,
            outcome=TargetOutcome.FAILURE if is_fail else TargetOutcome.SUCCESS,
            observable_signals={},
        )

    runner = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_p_alpha", max_trials=50),
        target_fn=target_fn,
    )

    res = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=3,
        failure_threshold=0.50,
        alpha=0.05,
        alternative="greater",
    )

    assert res.minimality_status == StochasticMinimalityStatus.OBSERVED_ONLY
    assert res.minimality_status != StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
    assert "did not achieve statistical significance at alpha=0.05" in res.status_message
    assert "p_comp=0.0500" in res.status_message


def test_equal_failure_rates_classified_not_minimal():
    """Verify that equal candidate and ablation failure rates yield NOT_MINIMAL with the observed-data disclaimer.

    Candidate confirmation requires failure rate >= failure_threshold (e.g. 4/8 = 0.50 with threshold=0.50).
    When the single-element ablation yields an identical failure rate (4/8 = 0.50), removing the element
    does not reduce failure rate below threshold, and is classified as NOT_MINIMAL based on observed sample rates.
    """
    initial_input = InputCase(case_id="case_equal_rate", elements=("p_a",))
    candidate = ("p_a",)

    call_counts = {"cand": 0, "ablat": 0}

    def target_fn(case: InputCase) -> TargetResult:
        if "p_a" in case.elements:
            call_counts["cand"] += 1
            is_fail = (call_counts["cand"] <= 4)
        else:
            call_counts["ablat"] += 1
            is_fail = (call_counts["ablat"] <= 4)
        return TargetResult(
            case_id=case.case_id,
            outcome=TargetOutcome.FAILURE if is_fail else TargetOutcome.SUCCESS,
            observable_signals={},
        )

    runner = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_equal_rate", max_trials=50),
        target_fn=target_fn,
    )

    res = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=8,
        failure_threshold=0.50,
        alternative="greater",
    )

    assert res.minimality_status == StochasticMinimalityStatus.NOT_MINIMAL
    assert "Candidate is not 1-minimal" in res.status_message
    assert "observed-data classification, not an equivalence or non-inferiority proof" in res.status_message
