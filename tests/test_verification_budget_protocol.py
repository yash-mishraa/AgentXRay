"""Regression tests for the Phase 4.5D verification-budget fairness protocol."""

from __future__ import annotations

from pathlib import Path
import pytest

from agentxray.agent_harness.cost_tracker import (
    CostBudgetExceededError,
    CostTracker,
    ExperimentBudgetConfig,
    IsolatedAlgorithmCostTracker,
    IsolatedPhaseCostTracker,
)
from agentxray.models import (
    CandidateEvaluation,
    ExperimentConfig,
    InputCase,
    SearchStatus,
    StochasticMinimalityStatus,
    TargetOutcome,
    TargetResult,
)
from agentxray.providers.base import UsageMetadata
from agentxray.runner import BudgetExhaustedError, InterventionRunner, TimeoutExceededError
from agentxray.statistics import compare_proportions, exact_binomial_test, fisher_exact_test
from agentxray.stochastic_baselines import (
    run_stochastic_greedy_ablation,
    run_stochastic_random_search,
)
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from agentxray.verification import (
    VerificationExecutionStatus,
    VerificationResult,
    verify_candidate_stochastically,
)
from benchmarks.model_pilot.runner import run_model_pilot_benchmark
from benchmarks.synthetic.stochastic_tasks import create_stochastic_non_monotonic_task


# ---------------------------------------------------------------------------
# Test 1: Independent Discovery and Verification Request Quotas
# ---------------------------------------------------------------------------
def test_independent_discovery_and_verification_request_quotas():
    """Verify that discovery and verification receive independent request quotas and cannot starve each other."""
    budget_cfg = ExperimentBudgetConfig(
        max_model_requests=50,
        max_discovery_requests=5,
        max_verification_requests=4,
        max_spend_usd=1.0,
    )
    global_tracker = CostTracker(budget_cfg)

    tracker_disc = IsolatedPhaseCostTracker(
        algorithm_config=budget_cfg,
        phase="discovery",
        global_tracker=global_tracker,
        enforce_global_requests=True,
    )
    tracker_verif = IsolatedPhaseCostTracker(
        algorithm_config=budget_cfg,
        phase="verification",
        global_tracker=global_tracker,
        enforce_global_requests=True,
    )

    # Exhaust discovery quota (5 requests)
    for _ in range(4):
        tracker_disc.check_budget_before_request()
        tracker_disc.record_usage(UsageMetadata(prompt_tokens=10, completion_tokens=10, total_tokens=20))
    assert tracker_disc.total_requests == 4

    # 5th request hits exact boundary
    tracker_disc.check_budget_before_request()
    with pytest.raises(CostBudgetExceededError, match="Discovery model request budget"):
        tracker_disc.record_usage(UsageMetadata(prompt_tokens=10, completion_tokens=10, total_tokens=20))
    assert tracker_disc.total_requests == 5
    assert tracker_disc.is_request_cap_reached is True

    # 6th request fails discovery pre-flight
    with pytest.raises(CostBudgetExceededError, match="Discovery model request budget"):
        tracker_disc.check_budget_before_request()

    # Verification tracker is independent: 0 requests used, can execute its full 4 requests
    assert tracker_verif.total_requests == 0
    assert tracker_verif.is_request_cap_reached is False
    for _ in range(3):
        tracker_verif.check_budget_before_request()
        tracker_verif.record_usage(UsageMetadata(prompt_tokens=10, completion_tokens=10, total_tokens=20))
    assert tracker_verif.total_requests == 3

    # 4th request reaches boundary
    tracker_verif.check_budget_before_request()
    with pytest.raises(CostBudgetExceededError, match="Verification model request budget"):
        tracker_verif.record_usage(UsageMetadata(prompt_tokens=10, completion_tokens=10, total_tokens=20))
    assert tracker_verif.total_requests == 4
    assert tracker_verif.is_request_cap_reached is True

    # 5th verification request fails pre-flight
    with pytest.raises(CostBudgetExceededError, match="Verification model request budget"):
        tracker_verif.check_budget_before_request()

    # Global tracker accumulated all 9 requests
    assert global_tracker.total_requests == 9


# ---------------------------------------------------------------------------
# Test 2: Global Safety Caps Overriding Phase Allowances
# ---------------------------------------------------------------------------
def test_global_safety_caps_override_phase_allowances():
    """Verify that an overarching global cap halts execution even when phase allowance remains."""
    # Global cap is 8 requests, while each phase allows up to 10
    global_cfg = ExperimentBudgetConfig(max_model_requests=8, max_spend_usd=1.0)
    global_tracker = CostTracker(global_cfg)

    algo_cfg = ExperimentBudgetConfig(
        max_model_requests=20,
        max_discovery_requests=10,
        max_verification_requests=10,
        max_spend_usd=1.0,
    )

    tracker_disc = IsolatedPhaseCostTracker(
        algorithm_config=algo_cfg,
        phase="discovery",
        global_tracker=global_tracker,
        enforce_global_requests=True,
    )
    tracker_verif = IsolatedPhaseCostTracker(
        algorithm_config=algo_cfg,
        phase="verification",
        global_tracker=global_tracker,
        enforce_global_requests=True,
    )

    # Discovery consumes 6 requests
    for _ in range(6):
        tracker_disc.record_usage(UsageMetadata(prompt_tokens=10, completion_tokens=10, total_tokens=20))
    assert tracker_disc.total_requests == 6
    assert global_tracker.total_requests == 6

    # Verification begins: local allowance is 10, but global limit only has 2 requests remaining
    tracker_verif.record_usage(UsageMetadata(prompt_tokens=10, completion_tokens=10, total_tokens=20))
    assert tracker_verif.total_requests == 1
    assert global_tracker.total_requests == 7

    # 2nd verification request hits global safety limit (8 requests total)
    with pytest.raises(CostBudgetExceededError, match="Model request budget exhausted"):
        tracker_verif.record_usage(UsageMetadata(prompt_tokens=10, completion_tokens=10, total_tokens=20))
    assert tracker_verif.total_requests == 2
    assert global_tracker.total_requests == 8

    # 3rd verification request hits global safety ceiling pre-flight
    with pytest.raises(CostBudgetExceededError, match="Global benchmark model request budget exhausted"):
        tracker_verif.check_budget_before_request()


# ---------------------------------------------------------------------------
# Test 3: Actual Multi-Turn Request Accounting, Retries, and Failures
# ---------------------------------------------------------------------------
def test_actual_multiturn_request_accounting_retries_and_failed_requests():
    """Verify that requests, retries, and failed attempts are recorded from actual calls without multipliers."""
    cfg = ExperimentBudgetConfig(max_model_requests=20, max_discovery_requests=15, max_spend_usd=1.0)
    tracker = IsolatedPhaseCostTracker(algorithm_config=cfg, phase="discovery")

    # Turn 1: successful request with 1 transient retry
    tracker.record_usage(UsageMetadata(prompt_tokens=50, completion_tokens=25, total_tokens=75), success=True, retries=1)
    assert tracker.total_requests == 2
    assert tracker.successful_requests == 1
    assert tracker.retries == 1

    # Turn 2: failed request attempt with 2 retries
    tracker.record_failed_request(retries=2)
    assert tracker.total_requests == 5
    assert tracker.failed_requests == 3
    assert tracker.retries == 3

    # Turn 3: second successful turn
    tracker.record_usage(UsageMetadata(prompt_tokens=50, completion_tokens=25, total_tokens=75), success=True, retries=0)
    assert tracker.total_requests == 6
    assert tracker.successful_requests == 2
    assert tracker.failed_requests == 3
    assert tracker.retries == 3


# ---------------------------------------------------------------------------
# Test 4: Exact Request-Cap Boundaries
# ---------------------------------------------------------------------------
def test_exact_request_cap_boundaries_in_phase_tracker():
    """Verify that exact-boundary properties distinguish total_requests == cap from total_requests > cap."""
    cfg = ExperimentBudgetConfig(max_model_requests=10, max_discovery_requests=3)
    tracker = IsolatedPhaseCostTracker(cfg, phase="discovery")

    tracker.record_usage(UsageMetadata(prompt_tokens=10, completion_tokens=10, total_tokens=20))
    assert tracker.is_request_cap_reached is False
    assert tracker.is_request_cap_exceeded is False

    tracker.record_usage(UsageMetadata(prompt_tokens=10, completion_tokens=10, total_tokens=20))
    assert tracker.is_request_cap_reached is False
    assert tracker.is_request_cap_exceeded is False

    # 3rd request reaches cap exactly
    with pytest.raises(CostBudgetExceededError):
        tracker.record_usage(UsageMetadata(prompt_tokens=10, completion_tokens=10, total_tokens=20))
    assert tracker.total_requests == 3
    assert tracker.is_request_cap_reached is True
    assert tracker.is_request_cap_exceeded is False  # not strictly > cap!

    # 4th request strictly exceeds cap
    with pytest.raises(CostBudgetExceededError):
        tracker.record_failed_request()
    assert tracker.total_requests == 4
    assert tracker.is_request_cap_reached is True
    assert tracker.is_request_cap_exceeded is True


# ---------------------------------------------------------------------------
# Test 5: Candidate Immutability During Verification
# ---------------------------------------------------------------------------
def test_candidate_immutability_during_verification():
    """Verify that verification never shrinks or mutates the candidate condition."""
    initial_input = InputCase(case_id="c_immut", elements=("param_a", "param_b", "distractor_c"))
    candidate = ("param_a", "param_b", "distractor_c")

    # Target that always fails
    def target_fn(case: InputCase) -> TargetResult:
        return TargetResult(case_id=case.case_id, outcome=TargetOutcome.FAILURE, observable_signals={})

    runner = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_immut", max_trials=50),
        target_fn=target_fn,
    )

    res = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=3,
        failure_threshold=0.50,
    )

    # Since all ablations fail, the candidate is NOT_MINIMAL
    assert res.minimality_status == StochasticMinimalityStatus.NOT_MINIMAL
    # Crucially, the candidate tuple must remain exactly the 3 elements originally passed
    assert res.candidate == candidate
    assert set(res.candidate) == {"param_a", "param_b", "distractor_c"}


# ---------------------------------------------------------------------------
# Test 6: Fresh Candidate and Ablation Evaluations
# ---------------------------------------------------------------------------
def test_fresh_candidate_and_ablation_evaluations():
    """Verify that verification evaluates both the candidate and every ablation on fresh trials."""
    initial_input = InputCase(case_id="c_fresh", elements=("param_a", "param_b"))
    candidate = ("param_a", "param_b")

    evaluated_interventions: list[str] = []

    def target_fn(case: InputCase) -> TargetResult:
        elems = set(case.elements)
        is_fail = ("param_a" in elems and "param_b" in elems)
        return TargetResult(
            case_id=case.case_id,
            outcome=TargetOutcome.FAILURE if is_fail else TargetOutcome.SUCCESS,
            observable_signals={},
        )

    runner = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_fresh", max_trials=50),
        target_fn=target_fn,
    )

    res = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=5,
        failure_threshold=0.50,
        alternative="greater",
    )

    # Candidate confirmation: 5 trials
    assert res.candidate_evaluation is not None
    assert res.candidate_evaluation.trials == 5
    assert res.candidate_evaluation.failures == 5

    # Ablations: 2 elements * 5 trials = 10 trials
    assert len(res.removal_evaluations) == 2
    assert "param_a" in res.removal_evaluations
    assert "param_b" in res.removal_evaluations
    assert res.removal_evaluations["param_a"].trials == 5
    assert res.removal_evaluations["param_a"].failures == 0
    assert res.removal_evaluations["param_b"].trials == 5
    assert res.removal_evaluations["param_b"].failures == 0

    assert res.target_calls_used == 15
    assert res.minimality_status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED


# ---------------------------------------------------------------------------
# Test 7: DD Refinement Semantics Unchanged During Discovery
# ---------------------------------------------------------------------------
def test_dd_refinement_semantics_unchanged_during_discovery():
    """Verify that DD preserves its existing refinement behavior and isolates non-monotonic triggers."""
    target, initial_input, gt = create_stochastic_non_monotonic_task(target_seed=42)
    config = ExperimentConfig(experiment_name="test_dd_refine", max_trials=300)
    runner = InterventionRunner(config=config, target_fn=target.execute)

    minimizer = StochasticDeltaDebugMinimizer(
        runner=runner,
        trials_per_candidate=20,
        failure_threshold=0.30,
    )
    result = minimizer.minimize(initial_input)

    assert result.status == SearchStatus.COMPLETE
    assert result.is_failure is True
    # DD must refine through the non-monotonic suppressor to find feature_flag_a
    assert "feature_flag_a" in result.subset_elements


# ---------------------------------------------------------------------------
# Test 8: Partial Verification Classification
# ---------------------------------------------------------------------------
def test_partial_verification_classification():
    """Verify partial verification classifies as NOT_MINIMAL if counterexample observed, else INCONCLUSIVE."""
    initial_input = InputCase(case_id="c_part", elements=("a", "b", "c"))
    candidate = ("a", "b", "c")

    # Case A: candidate confirmation (3 trials), ablation 'a' succeeds (3 trials), but 'a' still fails!
    # Even if budget halts after 'a', we already proved non-minimality!
    def target_always_fail(case: InputCase) -> TargetResult:
        return TargetResult(case_id=case.case_id, outcome=TargetOutcome.FAILURE, observable_signals={})

    runner_cap6 = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_part_a", max_trials=6),
        target_fn=target_always_fail,
    )
    res_a = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner_cap6,
        trials_per_candidate=3,
        failure_threshold=0.50,
    )
    assert res_a.minimality_status == StochasticMinimalityStatus.NOT_MINIMAL
    assert res_a.verification_status == VerificationExecutionStatus.BUDGET_EXHAUSTED
    assert res_a.is_complete is False

    # Case B: candidate confirmation (3 trials), ablation 'a' drops failure to 0 (3 trials).
    # Then budget halts before 'b' and 'c' can be tested. Since no counterexample observed, outcome is INCONCLUSIVE.
    def target_drop_on_a(case: InputCase) -> TargetResult:
        is_fail = ("a" in set(case.elements))
        return TargetResult(
            case_id=case.case_id,
            outcome=TargetOutcome.FAILURE if is_fail else TargetOutcome.SUCCESS,
            observable_signals={},
        )

    runner_cap6_b = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_part_b", max_trials=6),
        target_fn=target_drop_on_a,
    )
    res_b = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner_cap6_b,
        trials_per_candidate=3,
        failure_threshold=0.50,
    )
    assert res_b.minimality_status == StochasticMinimalityStatus.INCONCLUSIVE
    assert res_b.verification_status == VerificationExecutionStatus.BUDGET_EXHAUSTED
    assert res_b.is_complete is False


# ---------------------------------------------------------------------------
# Test 9: Timeouts and Budget Exhaustion in Verification
# ---------------------------------------------------------------------------
def test_verification_timeouts_and_budget_exhaustion():
    """Verify that timeout during candidate confirmation immediately returns INCONCLUSIVE with TIMED_OUT."""
    initial_input = InputCase(case_id="c_to", elements=("x", "y"))
    candidate = ("x", "y")

    def target_slow(case: InputCase) -> TargetResult:
        return TargetResult(case_id=case.case_id, outcome=TargetOutcome.FAILURE, observable_signals={})

    # Limit timeout to 0.0001 seconds
    runner = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_to", max_trials=50, timeout_seconds=0.0001),
        target_fn=target_slow,
    )
    import time
    time.sleep(0.001)

    res = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=5,
    )
    assert res.minimality_status == StochasticMinimalityStatus.INCONCLUSIVE
    assert res.verification_status == VerificationExecutionStatus.TIMED_OUT


# ---------------------------------------------------------------------------
# Test 10: Random Search Without Trial Reservation
# ---------------------------------------------------------------------------
def test_random_search_no_trial_request_confusion_with_reservation_flag():
    """Verify that when reserve_verification_budget=False, Random Search does not break early."""
    elements = tuple(f"elem_{i}" for i in range(10))
    initial_input = InputCase(case_id="c_rnd", elements=elements)

    call_count = 0
    def target_fn(case: InputCase) -> TargetResult:
        nonlocal call_count
        call_count += 1
        if len(case.elements) == 10:
            return TargetResult(case_id=case.case_id, outcome=TargetOutcome.FAILURE, observable_signals={})
        return TargetResult(case_id=case.case_id, outcome=TargetOutcome.SUCCESS, observable_signals={})

    # 1. With reserve_verification_budget=True, it halts when remaining_budget <= len(best_candidate) * trials
    # len(initial) = 10, trials = 2 -> needed = 20. With max_trials = 25, it stops early after 2 samples!
    runner_reserved = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_res_true", max_trials=25),
        target_fn=target_fn,
    )
    comp_reserved = run_stochastic_random_search(
        initial_input=initial_input,
        runner=runner_reserved,
        trials_per_candidate=2,
        reserve_verification_budget=True,
    )
    # Search phase halted early to reserve verification budget
    assert comp_reserved.search_target_calls < 15
    assert comp_reserved.verification_target_calls > 0

    # 2. With reserve_verification_budget=False, discovery utilizes the full trial budget
    call_count = 0
    runner_full = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_res_false", max_trials=25),
        target_fn=target_fn,
    )
    comp_full = run_stochastic_random_search(
        initial_input=initial_input,
        runner=runner_full,
        trials_per_candidate=2,
        reserve_verification_budget=False,
    )
    # Search phase did not halt early for verification reservation
    assert comp_full.search_target_calls >= 24
    assert comp_full.verification_target_calls == 0


# ---------------------------------------------------------------------------
# Test 11: Reporting of Phase-Specific and Total Costs
# ---------------------------------------------------------------------------
def test_reporting_of_phase_specific_and_total_costs(tmp_path: Path):
    """Verify that run_model_pilot_benchmark reports discovery and verification costs separately."""
    report = run_model_pilot_benchmark(
        provider_type="fake",
        task_filter="conflicting_prompt",
        trials_per_candidate=2,
        max_trials=20,
        max_model_requests=100,
        output_dir=tmp_path,
        enable_independent_verification=True,
        verification_trials=2,
    )

    assert len(report.evaluations) == 1
    eval_res = report.evaluations[0]

    # Verify DD condition recorded search and verification requests
    cond = eval_res.minimizer_condition
    assert cond.search_model_requests > 0
    assert cond.verification_model_requests > 0
    assert cond.total_model_requests == cond.search_model_requests + cond.verification_model_requests
    assert cond.search_target_calls > 0
    assert cond.verification_target_calls > 0
    assert cond.target_calls_used == cond.search_target_calls + cond.verification_target_calls
    assert cond.verification_status in ("complete", "partial", "budget_exhausted")


# ---------------------------------------------------------------------------
# Test 12: Statistical Alternative Direction and Contingency Table Orientation
# ---------------------------------------------------------------------------
def test_statistical_alternative_direction_and_contingency_table_orientation():
    """Verify contingency table orientation and directional hypotheses in statistics functions."""
    # 1. Compare proportions: k1=failures in candidate, k2=failures in ablation
    # When candidate fails 5/5 and ablation fails 0/5, alternative="greater" tests H1: p1 > p2
    res_greater = compare_proportions(k1=5, n1=5, k2=0, n2=5, alpha=0.05, alternative="greater")
    assert res_greater.statistically_significant is True
    assert res_greater.difference == 1.0
    assert res_greater.fisher_exact_p_value is not None
    assert res_greater.fisher_exact_p_value < 0.05

    # Reverse: candidate fails 0/5 and ablation fails 5/5 -> p1 < p2
    # alternative="greater" must NOT reject H0
    res_not_greater = compare_proportions(k1=0, n1=5, k2=5, n2=5, alpha=0.05, alternative="greater")
    assert res_not_greater.statistically_significant is False
    assert res_not_greater.fisher_exact_p_value == 1.0

    # 2. Exact binomial test: tests H1: p < p0 (ablation drops below failure threshold)
    # k=0 out of n=5 against p0=0.50
    p_thresh = exact_binomial_test(k=0, n=5, p0=0.50, alternative="less")
    assert p_thresh == (0.5 ** 5)  # 0.03125 < 0.05


# ---------------------------------------------------------------------------
# Test 13: Retry Headroom Bounding in ControlledAgent
# ---------------------------------------------------------------------------
def test_retry_headroom_bounding_in_agent_run_session():
    """Verify that ControlledAgent bounds retries using tracker's remaining_request_headroom."""
    from agentxray.agent_harness.agent import ControlledAgent
    from agentxray.agent_harness.tools import MockToolEnvironment
    from agentxray.providers.base import ProviderError
    from agentxray.providers.fake import FakeProvider

    # Budget config with max_discovery_requests = 1
    budget_cfg = ExperimentBudgetConfig(
        max_model_requests=10,
        max_discovery_requests=1,
    )
    phase_tracker = IsolatedPhaseCostTracker(algorithm_config=budget_cfg, phase="discovery")
    provider = FakeProvider(model_name="fake-model", max_retries_limit=3, simulate_error=ProviderError)
    agent = ControlledAgent(
        provider=provider,
        mock_env=MockToolEnvironment(),
        cost_tracker=phase_tracker,
    )

    # With max_discovery_requests=1, remaining_request_headroom is 1.
    # Therefore allowed_retries must be min(provider.max_retries, max(0, 1 - 1)) = 0!
    assert phase_tracker.remaining_request_headroom == 1

    # First session fails, provider receives max_retries=0
    trace = agent.run_session(system_prompt="sys", user_prompt="test")
    assert trace.finish_reason == "error"
    assert provider.last_max_retries == 0
    assert phase_tracker.total_requests == 1
    assert phase_tracker.remaining_request_headroom == 0
    assert phase_tracker.retries == 0

    # Second session fails immediately at pre-flight check
    trace2 = agent.run_session(system_prompt="sys", user_prompt="test")
    assert trace2.finish_reason == "budget_exhausted"
    assert "Discovery model request budget exhausted" in (trace2.error_message or "")

    with pytest.raises(CostBudgetExceededError, match="Discovery model request budget exhausted"):
        phase_tracker.check_budget_before_request()


# ---------------------------------------------------------------------------
# Test 14: Verification Evidence Tier: Exploratory Default & Confirmatory Guard
# ---------------------------------------------------------------------------
def test_verification_evidence_tier_exploratory_default_and_confirmatory_rejection():
    """Verify that verification defaults to exploratory tier, and confirmatory requires explicit sample size."""
    from agentxray.verification import VerificationEvidenceTier

    initial_input = InputCase(case_id="c_tier", elements=("x", "y"))
    candidate = ("x", "y")

    def target_fn(case: InputCase) -> TargetResult:
        return TargetResult(case_id=case.case_id, outcome=TargetOutcome.FAILURE, observable_signals={})

    runner = InterventionRunner(
        config=ExperimentConfig(experiment_name="test_tier", max_trials=500),
        target_fn=target_fn,
    )

    # 1. Default call: exploratory tier and exploratory_default rationale
    res_exploratory = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=5,
        trials_explicitly_set=False,
    )
    assert res_exploratory.evidence_tier == VerificationEvidenceTier.EXPLORATORY
    assert res_exploratory.sample_size_rationale == "exploratory_default"
    assert res_exploratory.confirmatory_power_established is False

    # 2. Confirmatory tier with default 5 trials (without trials_explicitly_set) must raise ValueError
    with pytest.raises(ValueError, match="Confirmatory verification requires an explicitly configured sample size"):
        verify_candidate_stochastically(
            candidate=candidate,
            initial_input=initial_input,
            runner=runner,
            trials_per_candidate=5,
            evidence_tier="confirmatory",
            trials_explicitly_set=False,
        )

    # 3. Explicitly passing 5 trials with confirmatory tier must NOT by itself establish confirmatory power
    res_conf_5_explicit = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=5,
        evidence_tier="confirmatory",
        trials_explicitly_set=True,
    )
    assert res_conf_5_explicit.evidence_tier == VerificationEvidenceTier.CONFIRMATORY
    assert res_conf_5_explicit.sample_size_rationale == "unresolved_user_specified"
    assert res_conf_5_explicit.sample_size_validation_status == "not_implemented"
    assert res_conf_5_explicit.confirmatory_power_established is False

    # 4. Confirmatory tier with explicit sample size and user-supplied rationale preserves rationale context
    # but does NOT establish power without formal validation (sample_size_validation_status remains 'unresolved')
    res_confirmatory = verify_candidate_stochastically(
        candidate=candidate,
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=10,
        evidence_tier="confirmatory",
        sample_size_rationale="power_calculation_80pct",
        trials_explicitly_set=True,
    )
    assert res_confirmatory.evidence_tier == VerificationEvidenceTier.CONFIRMATORY
    assert res_confirmatory.sample_size_rationale == "power_calculation_80pct"
    assert res_confirmatory.sample_size_validation_status == "unresolved"
    assert res_confirmatory.confirmatory_power_established is False

    # 5. Automated power calculation is not implemented; affirmative power claims are strictly rejected
    assert res_confirmatory.confirmatory_power_established is False
    from agentxray.verification import VerificationExecutionStatus
    with pytest.raises(ValueError, match="confirmatory_power_established cannot be True"):
        VerificationResult(
            candidate=candidate,
            minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
            verification_status=VerificationExecutionStatus.COMPLETE,
            evidence_tier=VerificationEvidenceTier.CONFIRMATORY,
            sample_size_rationale="power_calculation_80pct",
            sample_size_validation_status="validated",
            confirmatory_power_established=True,
        )

    # 6. Separation of hypothesis testing from evidentiary tier:
    # Mathematical minimality classification is identical regardless of evidence tier
    assert (
        res_exploratory.minimality_status
        == res_conf_5_explicit.minimality_status
        == res_confirmatory.minimality_status
    )


# ---------------------------------------------------------------------------
# Test 15: Aggregate Metrics Outcome Accounting & Diagnostic Overlap
# ---------------------------------------------------------------------------
def test_aggregate_metrics_outcome_accounting():
    """Verify compute_stochastic_aggregate_metrics Intention-to-Treat outcome partitions."""
    from agentxray.models import ModelPilotEvaluationResult, StochasticDiscoveredCondition
    from agentxray.reporting import compute_stochastic_aggregate_metrics

    dummy_eval = CandidateEvaluation(
        candidate_elements=("a",),
        trials=10,
        failures=10,
        failure_rate=1.0,
        confidence_interval=(0.7, 1.0),
    )

    # Task 1: Supported minimality (success)
    eval_1 = ModelPilotEvaluationResult(
        task_name="task_1",
        description="desc1",
        initial_elements=("a", "b"),
        ground_truth_trigger=("a",),
        matches_ground_truth=True,
        precision=1.0,
        recall=1.0,
        minimizer_condition=StochasticDiscoveredCondition(
            subset_elements=("a",),
            status=SearchStatus.COMPLETE,
            is_failure=True,
            candidate_evaluation=dummy_eval,
            minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
            verification_status="complete",
            target_calls_used=10,
        ),
    )

    # Task 2: Inconclusive minimality (failure in primary ITT success)
    eval_2 = ModelPilotEvaluationResult(
        task_name="task_2",
        description="desc2",
        initial_elements=("a", "b"),
        ground_truth_trigger=("a",),
        matches_ground_truth=True,
        precision=1.0,
        recall=1.0,
        minimizer_condition=StochasticDiscoveredCondition(
            subset_elements=("a",),
            status=SearchStatus.COMPLETE,
            is_failure=True,
            candidate_evaluation=dummy_eval,
            minimality_status=StochasticMinimalityStatus.INCONCLUSIVE,
            verification_status="complete",
            target_calls_used=10,
        ),
    )

    # Task 3: Candidate confirmation failure (inconclusive)
    eval_3 = ModelPilotEvaluationResult(
        task_name="task_3",
        description="desc3",
        initial_elements=("a", "b"),
        ground_truth_trigger=("a",),
        matches_ground_truth=False,
        precision=0.0,
        recall=0.0,
        minimizer_condition=StochasticDiscoveredCondition(
            subset_elements=("a", "b"),
            status=SearchStatus.COMPLETE,
            is_failure=False,
            candidate_evaluation=dummy_eval,
            minimality_status=StochasticMinimalityStatus.INCONCLUSIVE,
            verification_status="complete",
            target_calls_used=5,
            status_message="Candidate failed to reproduce failure rate >= 0.50 on fresh verification trials",
        ),
    )

    # Task 4: Not minimal
    eval_4 = ModelPilotEvaluationResult(
        task_name="task_4",
        description="desc4",
        initial_elements=("a", "b"),
        ground_truth_trigger=("a",),
        matches_ground_truth=False,
        precision=0.5,
        recall=1.0,
        minimizer_condition=StochasticDiscoveredCondition(
            subset_elements=("a", "b"),
            status=SearchStatus.COMPLETE,
            is_failure=True,
            candidate_evaluation=dummy_eval,
            minimality_status=StochasticMinimalityStatus.NOT_MINIMAL,
            verification_status="complete",
            target_calls_used=15,
        ),
    )

    # Task 5: Interrupted in verification
    eval_5 = ModelPilotEvaluationResult(
        task_name="task_5",
        description="desc5",
        initial_elements=("a", "b"),
        ground_truth_trigger=("a",),
        matches_ground_truth=False,
        precision=0.5,
        recall=1.0,
        minimizer_condition=StochasticDiscoveredCondition(
            subset_elements=("a", "b"),
            status=SearchStatus.BUDGET_EXHAUSTED,
            is_failure=True,
            candidate_evaluation=dummy_eval,
            minimality_status=StochasticMinimalityStatus.INCONCLUSIVE,
            verification_status="budget_exhausted",
            target_calls_used=8,
        ),
    )

    evals = [eval_1, eval_2, eval_3, eval_4, eval_5]
    metrics = compute_stochastic_aggregate_metrics(evals)

    # Total evaluated (ITT denominator) is 5
    assert metrics["tasks_evaluated"] == 5
    # Primary success numerator only counts statistically supported minimality: 1 / 5
    assert metrics["statistically_supported_minimality"] == 1
    # Inconclusive count is 3 (eval_2, eval_3, eval_5)
    assert metrics["inconclusive_tasks"] == 3
    # Candidate confirmation failure count is 1 (eval_3)
    assert metrics["candidate_confirmation_failures"] == 1
    # Not minimal count is 1 (eval_4)
    assert metrics["not_minimal_tasks"] == 1
    # Interrupted count is 1 (eval_5)
    assert metrics["interrupted_tasks"] == 1

    # Overlap verification: inconclusive_tasks (3) contains eval_3 (candidate confirmation failure)
    # and eval_5 (interrupted). Summing diagnostic counters yields 7 > 5 (tasks_evaluated),
    # demonstrating they form overlapping diagnostic subsets, not a disjoint partition.
    sum_diagnostics = (
        metrics["statistically_supported_minimality"]
        + metrics["inconclusive_tasks"]
        + metrics["not_minimal_tasks"]
        + metrics["candidate_confirmation_failures"]
        + metrics["interrupted_tasks"]
    )
    assert sum_diagnostics == 7
    assert sum_diagnostics > metrics["tasks_evaluated"]
    assert metrics["tasks_evaluated"] == 5


# ---------------------------------------------------------------------------
# Test 16: Reconciled Spend and Requests in Model Pilot
# ---------------------------------------------------------------------------
def test_reconciled_spend_and_requests_in_model_pilot(tmp_path: Path):
    """Verify that algo_breakdown spend and requests reconcile with global cost tracker."""
    report = run_model_pilot_benchmark(
        provider_type="fake",
        task_filter="conflicting_prompt",
        trials_per_candidate=2,
        max_trials=20,
        max_model_requests=100,
        output_dir=tmp_path,
        enable_independent_verification=True,
        verification_trials=2,
        verification_tier="exploratory",
    )

    bs = report.budget_summary
    per_algo = bs["per_algorithm_usage"]
    assert len(per_algo) > 0

    sum_requests = sum(row["model_requests"] for row in per_algo)
    assert sum_requests == bs["total_requests"]

    sum_spend = sum(row["estimated_spend_usd"] for row in per_algo)
    assert sum_spend == pytest.approx(bs["estimated_spend_usd"], abs=1e-5)

    for row in per_algo:
        assert "discovery_spend_usd" in row
        assert "verification_spend_usd" in row
        assert row["estimated_spend_usd"] == round(row["discovery_spend_usd"] + row["verification_spend_usd"], 7)
        assert row["model_requests"] == row["discovery_requests"] + row["verification_requests"]
        assert row["evidence_tier"] == "exploratory"
        assert row["sample_size_rationale"] == "unresolved_user_specified"


# ---------------------------------------------------------------------------
# Test 17: CLI Verification Options and Validation
# ---------------------------------------------------------------------------
def test_cli_verification_options_and_validation():
    """Verify CLI parser accepts verification arguments and enforces validation rules."""
    import subprocess
    import sys

    # Valid invocation with exploratory default
    cmd_ok = [
        sys.executable,
        "-m",
        "agentxray",
        "--mode",
        "demo",
        "--enable-independent-verification",
        "--max-discovery-requests",
        "10",
        "--max-verification-requests",
        "10",
        "--verification-trials",
        "5",
        "--verification-tier",
        "exploratory",
    ]
    res_ok = subprocess.run(cmd_ok, capture_output=True, text=True)
    assert res_ok.returncode == 0

    # Confirmatory without explicit verification trials must fail validation
    cmd_confirmatory_fail = [
        sys.executable,
        "-m",
        "agentxray",
        "--mode",
        "demo",
        "--verification-tier",
        "confirmatory",
    ]
    res_cf = subprocess.run(cmd_confirmatory_fail, capture_output=True, text=True)
    assert res_cf.returncode != 0
    assert "Confirmatory verification requires an explicitly configured sample size" in res_cf.stderr

    # Non-positive requests must fail validation
    cmd_neg_disc = [
        sys.executable,
        "-m",
        "agentxray",
        "--mode",
        "demo",
        "--max-discovery-requests",
        "-5",
    ]
    res_neg = subprocess.run(cmd_neg_disc, capture_output=True, text=True)
    assert res_neg.returncode != 0
    assert "--max-discovery-requests must be a positive integer" in res_neg.stderr


# ---------------------------------------------------------------------------
# Test 18: Markdown Report Transparency
# ---------------------------------------------------------------------------
def test_markdown_report_transparency(tmp_path: Path):
    """Verify format_model_pilot_markdown_report outputs breakdown table and verification transparency."""
    from agentxray.reporting import format_model_pilot_markdown_report

    report = run_model_pilot_benchmark(
        provider_type="fake",
        task_filter="conflicting_prompt",
        trials_per_candidate=2,
        max_trials=20,
        max_model_requests=100,
        output_dir=tmp_path,
        enable_independent_verification=True,
        verification_trials=2,
        verification_tier="exploratory",
    )
    md = format_model_pilot_markdown_report(report)

    assert "### Phase & Algorithm Usage Breakdown" in md
    assert "| Task | Algorithm | Discovery Reqs | Verif Reqs | Total Reqs | Discovery Spend | Verif Spend | Total Spend | Verif Status | Evidence Tier (Requested) |" in md
    assert "| Pilot Task | Method | Discovered Trigger | Minimality Status | Target Calls | Distractors Left | Matches Ground Truth |" in md
    assert "- **Verification Status**:" in md
    assert "- **Evidence Tier**:" in md
    assert "- **Sample Size Rationale**:" in md


# ---------------------------------------------------------------------------
# Test 19: ControlledAgent Retry Headroom Bounding and Custom Provider Fallback
# ---------------------------------------------------------------------------
def test_controlled_agent_retry_capping_and_provider_fallbacks():
    """Verify ControlledAgent bounds retries by remaining headroom and provider limits, with custom provider fallback."""
    from agentxray.agent_harness.agent import ControlledAgent
    from agentxray.agent_harness.cost_tracker import CostTracker, ExperimentBudgetConfig
    from agentxray.agent_harness.tools import MockToolEnvironment
    from agentxray.providers.base import ModelProvider, ModelResponse, UsageMetadata
    from agentxray.providers.fake import FakeProvider

    class MockInspectableProvider(ModelProvider):
        supports_retry_limits = True
        is_trusted_for_hard_budgets = True

        def __init__(self, max_retries_limit: int = 3, supports_max_retries: bool = True):
            self._max_retries = max_retries_limit
            self.supports_max_retries = supports_max_retries
            if not supports_max_retries:
                self.supports_retry_limits = False
                self.is_trusted_for_hard_budgets = False
            self.last_max_retries_passed: int | None = None
            self.generate_call_count = 0

        @property
        def provider_name(self) -> str:
            return "mock_inspectable"

        @property
        def model_name(self) -> str:
            return "mock-v1"

        def generate(self, messages, tools=(), temperature=0.0, max_tokens=512, max_retries=None) -> ModelResponse:
            self.generate_call_count += 1
            if not self.supports_max_retries and max_retries is not None:
                raise TypeError("generate() got an unexpected keyword argument 'max_retries'")
            self.last_max_retries_passed = max_retries
            return ModelResponse(
                model_identifier="mock-v1",
                content="Mock answer",
                tool_calls=(),
                usage=UsageMetadata(prompt_tokens=20, completion_tokens=10, retries=0),
            )

    env = MockToolEnvironment()

    # 1. Headroom = 1: allowed_retries MUST be capped at 0
    tracker_1 = CostTracker(config=ExperimentBudgetConfig(model_name="fake-model", max_model_requests=1))
    provider_1 = FakeProvider(model_name="fake-model", max_retries_limit=3)
    agent_1 = ControlledAgent(provider=provider_1, mock_env=env, cost_tracker=tracker_1)
    res_1 = agent_1.run_session(system_prompt="", user_prompt="Hello")
    assert res_1.finish_reason == "stop"
    assert provider_1.last_max_retries == 0

    # 2. Headroom = 3, provider limit = 1: allowed_retries MUST be min(1, 3 - 1) = 1
    tracker_2 = CostTracker(config=ExperimentBudgetConfig(model_name="fake-model", max_model_requests=3))
    provider_2 = FakeProvider(model_name="fake-model", max_retries_limit=1)
    agent_2 = ControlledAgent(provider=provider_2, mock_env=env, cost_tracker=tracker_2)
    res_2 = agent_2.run_session(system_prompt="", user_prompt="Hello")
    assert res_2.finish_reason == "stop"
    assert provider_2.last_max_retries == 1

    # 3. Headroom = 2, provider limit = 5: allowed_retries MUST be min(5, 2 - 1) = 1
    tracker_3 = CostTracker(config=ExperimentBudgetConfig(model_name="fake-model", max_model_requests=2))
    provider_3 = FakeProvider(model_name="fake-model", max_retries_limit=5)
    agent_3 = ControlledAgent(provider=provider_3, mock_env=env, cost_tracker=tracker_3)
    res_3 = agent_3.run_session(system_prompt="", user_prompt="Hello")
    assert res_3.finish_reason == "stop"
    assert provider_3.last_max_retries == 1

    # 4. Headroom = 0 (exhausted): MUST abort with budget_exhausted before dispatch
    tracker_4 = CostTracker(config=ExperimentBudgetConfig(model_name="fake-model", max_model_requests=1))
    try:
        tracker_4.record_failed_request(retries=0)  # Consumes 1 request -> headroom is now 0
    except Exception:
        pass
    assert tracker_4.remaining_request_headroom == 0
    provider_4 = FakeProvider(model_name="fake-model", max_retries_limit=3)
    agent_4 = ControlledAgent(provider=provider_4, mock_env=env, cost_tracker=tracker_4)
    res_4 = agent_4.run_session(system_prompt="", user_prompt="Hello")
    assert res_4.finish_reason == "budget_exhausted"
    assert provider_4.call_count == 0  # Zero provider dispatches

    # 5. Incompatible custom provider without max_retries support: fails safely upon dispatch
    tracker_5 = CostTracker(config=ExperimentBudgetConfig(model_name="fake-model", max_model_requests=5))
    provider_5 = MockInspectableProvider(max_retries_limit=3, supports_max_retries=False)
    agent_5 = ControlledAgent(provider=provider_5, mock_env=env, cost_tracker=tracker_5)
    res_5 = agent_5.run_session(system_prompt="", user_prompt="Hello")
    assert res_5.finish_reason == "error"
    assert "Unsupported provider" in (res_5.error_message or "")
    assert provider_5.generate_call_count == 0  # Pre-dispatch rejection prevents unconstrained call
    assert tracker_5.total_requests == 0
    assert tracker_5.remaining_request_headroom == 5

    # 6. Provider lacking max_retries parameter in signature is rejected before dispatch
    class MockNoRetrySigProvider(ModelProvider):
        def __init__(self):
            self.generate_call_count = 0

        @property
        def provider_name(self) -> str:
            return "mock_no_retry_sig"

        @property
        def model_name(self) -> str:
            return "mock-sig"

        def generate(self, messages, tools=(), temperature=0.0, max_tokens=512) -> ModelResponse:
            self.generate_call_count += 1
            return ModelResponse(model_identifier="mock-sig", content="hi", tool_calls=(), usage=UsageMetadata())

    tracker_6 = CostTracker(config=ExperimentBudgetConfig(model_name="fake-model", max_model_requests=5))
    provider_6 = MockNoRetrySigProvider()
    agent_6 = ControlledAgent(provider=provider_6, mock_env=env, cost_tracker=tracker_6)
    res_6 = agent_6.run_session(system_prompt="", user_prompt="Hello")
    assert res_6.finish_reason == "error"
    assert "Unsupported provider" in (res_6.error_message or "")
    assert provider_6.generate_call_count == 0  # Rejection happens before dispatch
    assert tracker_6.total_requests == 0
    assert tracker_6.remaining_request_headroom == 5


# ---------------------------------------------------------------------------
# Test 20: Unresolved Confirmatory Sample Size Disclosure in Reports
# ---------------------------------------------------------------------------
def test_unresolved_confirmatory_disclosure_in_reports(tmp_path: Path):
    """Verify that explicitly configured confirmatory runs with unresolved rationale are warned in Markdown and JSON."""
    import json
    from agentxray.reporting import format_model_pilot_json_report, format_model_pilot_markdown_report

    report = run_model_pilot_benchmark(
        provider_type="fake",
        task_filter="conflicting_prompt",
        trials_per_candidate=5,
        max_trials=20,
        max_model_requests=100,
        output_dir=tmp_path,
        enable_independent_verification=True,
        verification_trials=5,
        verification_tier="confirmatory",
        sample_size_rationale=None,  # Unresolved rationale
    )

    # 1. Markdown disclosures
    md = format_model_pilot_markdown_report(report)
    assert "> [!WARNING]" in md
    assert "Unresolved Confirmatory Sample Size" in md
    assert "Confirmatory statistical power has NOT been established" in md
    assert "- **Evidentiary Validity Disclosure**: Confirmatory power has not been established" in md
    assert "- **Requested Evidence Tier**: `confirmatory`" in md
    assert "- **Statistical Minimality Outcome**:" in md
    assert "- **Rationale Validation Status**: `not_implemented`" in md
    assert "- **Confirmatory Power Established**: `NO (Unresolved / Unvalidated)`" in md

    # 2. JSON disclosures
    raw_json = format_model_pilot_json_report(report)
    data = json.loads(raw_json)
    for eval_item in data["evaluations"]:
        cond = eval_item["minimizer_condition"]
        assert cond["evidence_tier"] == "confirmatory"
        assert cond["sample_size_rationale"] == "unresolved_user_specified"
        assert cond["sample_size_validation_status"] == "not_implemented"
        assert cond["confirmatory_power_established"] is False


# ---------------------------------------------------------------------------
# Test 21: Free-Text Rationale and Evidence Disentanglement
# ---------------------------------------------------------------------------
def test_free_text_rationale_disentangled_from_validated_power(tmp_path: Path):
    """Verify that user-supplied free-text rationale preserves context but leaves power unestablished."""
    import json
    from agentxray.reporting import format_model_pilot_json_report, format_model_pilot_markdown_report

    report = run_model_pilot_benchmark(
        provider_type="fake",
        task_filter="conflicting_prompt",
        trials_per_candidate=10,
        max_trials=20,
        max_model_requests=100,
        output_dir=tmp_path,
        enable_independent_verification=True,
        verification_trials=10,
        verification_tier="confirmatory",
        sample_size_rationale="user_calculated_power_90pct_alpha_01",
    )

    md = format_model_pilot_markdown_report(report)
    assert "- **Requested Evidence Tier**: `confirmatory`" in md
    assert "- **Sample Size Rationale Context**: `user_calculated_power_90pct_alpha_01`" in md
    assert "- **Rationale Validation Status**: `unresolved`" in md
    assert "- **Confirmatory Power Established**: `NO (Unresolved / Unvalidated)`" in md

    raw_json = format_model_pilot_json_report(report)
    data = json.loads(raw_json)
    for eval_item in data["evaluations"]:
        cond = eval_item["minimizer_condition"]
        assert cond["evidence_tier"] == "confirmatory"
        assert cond["sample_size_rationale"] == "user_calculated_power_90pct_alpha_01"
        assert cond["sample_size_validation_status"] == "unresolved"
        assert cond["confirmatory_power_established"] is False

    bs = data["budget_summary"]
    for row in bs["per_algorithm_usage"]:
        assert row["sample_size_rationale"] == "user_calculated_power_90pct_alpha_01"
        assert row["sample_size_validation_status"] == "unresolved"
        assert row["confirmatory_power_established"] is False


# ---------------------------------------------------------------------------
# Test 22: Task 3 Adversarial and Strict Provider Hard-Budget Enforcement
# ---------------------------------------------------------------------------
def test_task2_adversarial_and_untrusted_provider_hard_budget_enforcement():
    """Verify pre-dispatch rejection of untrusted/unsupported custom providers under hard budgets."""
    from unittest.mock import MagicMock
    from agentxray.agent_harness.agent import ControlledAgent, check_provider_retry_support
    from agentxray.agent_harness.cost_tracker import CostTracker, ExperimentBudgetConfig
    from agentxray.agent_harness.tools import MockToolEnvironment
    from agentxray.providers.base import ModelProvider, ModelResponse, UsageMetadata
    from agentxray.providers.fake import FakeProvider
    from agentxray.providers.ollama_adapter import OllamaProvider
    from agentxray.providers.openai_adapter import OpenAIProvider

    env = MockToolEnvironment()

    # 1. Custom ModelProvider subclass with is_trusted_for_hard_budgets=True
    class TrustedClaimCustomProvider(ModelProvider):
        supports_retry_limits = True
        is_trusted_for_hard_budgets = True

        def __init__(self):
            self.call_count = 0

        @property
        def provider_name(self) -> str:
            return "custom_trusted"

        @property
        def model_name(self) -> str:
            return "custom-v1"

        def generate(self, messages, tools=(), temperature=0.0, max_tokens=512, max_retries=None) -> ModelResponse:
            self.call_count += 1
            return ModelResponse(model_identifier="custom-v1", content="ok", tool_calls=(), usage=UsageMetadata())

    # 2. Custom provider spoofing provider_name="openai"
    class SpoofedOpenAIProvider(ModelProvider):
        supports_retry_limits = True
        is_trusted_for_hard_budgets = True

        def __init__(self):
            self.call_count = 0

        @property
        def provider_name(self) -> str:
            return "openai"

        @property
        def model_name(self) -> str:
            return "gpt-4o-mini"

        def generate(self, messages, tools=(), temperature=0.0, max_tokens=512, max_retries=None) -> ModelResponse:
            self.call_count += 1
            return ModelResponse(model_identifier="gpt-4o-mini", content="ok", tool_calls=(), usage=UsageMetadata())

    # 3. Custom provider spoofing provider_name="ollama"
    class SpoofedOllamaProvider(ModelProvider):
        supports_retry_limits = True
        is_trusted_for_hard_budgets = True

        def __init__(self):
            self.call_count = 0

        @property
        def provider_name(self) -> str:
            return "ollama"

        @property
        def model_name(self) -> str:
            return "llama3"

        def generate(self, messages, tools=(), temperature=0.0, max_tokens=512, max_retries=None) -> ModelResponse:
            self.call_count += 1
            return ModelResponse(model_identifier="llama3", content="ok", tool_calls=(), usage=UsageMetadata())

    # 4. Subclasses of trusted providers (FakeProvider, OpenAIProvider, OllamaProvider)
    class CustomFakeSubclass(FakeProvider):
        def __init__(self):
            super().__init__(model_name="fake-subclass")
            self.subclass_call_count = 0

        def generate(self, *args, **kwargs):
            self.subclass_call_count += 1
            return super().generate(*args, **kwargs)

    class CustomOpenAISubclass(OpenAIProvider):
        def __init__(self):
            super().__init__(model_name="gpt-4o-mini", api_key="sk-test")
            self.subclass_call_count = 0

        def generate(self, *args, **kwargs):
            self.subclass_call_count += 1
            return super().generate(*args, **kwargs)

    class CustomOllamaSubclass(OllamaProvider):
        def __init__(self):
            super().__init__(model_name="llama3")
            self.subclass_call_count = 0

        def generate(self, *args, **kwargs):
            self.subclass_call_count += 1
            return super().generate(*args, **kwargs)

    # 5. Wrapper or proxy delegating to a trusted provider
    class ProxyProvider:
        def __init__(self, target):
            self._target = target
            self.proxy_call_count = 0

        @property
        def provider_name(self) -> str:
            return self._target.provider_name

        @property
        def model_name(self) -> str:
            return self._target.model_name

        def generate(self, *args, **kwargs):
            self.proxy_call_count += 1
            return self._target.generate(*args, **kwargs)

    # 6. MagicMock(spec=ModelProvider)
    mock_provider = MagicMock(spec=ModelProvider)
    mock_provider.provider_name = "mock_provider"
    mock_provider.model_name = "mock-model"

    # 7. Custom provider that accepts max_retries, ignores it, and simulates hidden attempts
    class AdversarialHiddenRetryProvider(ModelProvider):
        supports_retry_limits = True
        is_trusted_for_hard_budgets = True

        def __init__(self):
            self.generate_call_count = 0
            self.hidden_call_counter = 0

        @property
        def provider_name(self) -> str:
            return "adversarial_hidden"

        @property
        def model_name(self) -> str:
            return "adv-model"

        def generate(self, messages, tools=(), temperature=0.0, max_tokens=512, max_retries=None) -> ModelResponse:
            self.generate_call_count += 1
            for _ in range(10):
                self.hidden_call_counter += 1
            return ModelResponse(model_identifier="adv-model", content="done", tool_calls=(), usage=UsageMetadata())

    adversarial_providers = [
        ("custom_trusted_subclass", TrustedClaimCustomProvider()),
        ("spoofed_openai", SpoofedOpenAIProvider()),
        ("spoofed_ollama", SpoofedOllamaProvider()),
        ("subclass_fake", CustomFakeSubclass()),
        ("subclass_openai", CustomOpenAISubclass()),
        ("subclass_ollama", CustomOllamaSubclass()),
        ("proxy_wrapper", ProxyProvider(FakeProvider())),
        ("magic_mock", mock_provider),
        ("adversarial_hidden_retries", AdversarialHiddenRetryProvider()),
    ]

    for name, prov in adversarial_providers:
        # Check pre-dispatch check directly
        ok, reason = check_provider_retry_support(prov, strict_budget=True)
        assert not ok, f"Expected {name} to fail check_provider_retry_support under strict_budget=True"
        assert "Unsupported provider" in reason
        assert "is not an approved built-in provider class" in reason

        # Check ControlledAgent execution pre-dispatch rejection
        tracker = CostTracker(config=ExperimentBudgetConfig(model_name="fake-model", max_model_requests=5))
        agent = ControlledAgent(provider=prov, mock_env=env, cost_tracker=tracker)
        res = agent.run_session(system_prompt="", user_prompt="hi")

        assert res.finish_reason == "error", f"Expected error finish_reason for {name}, got {res.finish_reason}"
        assert "Unsupported provider" in (res.error_message or "")
        assert "is not an approved built-in provider class" in (res.error_message or "")

        # Verify generate() was NEVER called
        if hasattr(prov, "call_count"):
            assert prov.call_count == 0, f"generate() was called {prov.call_count} times for {name}"
        if hasattr(prov, "subclass_call_count"):
            assert prov.subclass_call_count == 0, f"generate() was called for {name}"
        if hasattr(prov, "proxy_call_count"):
            assert prov.proxy_call_count == 0, f"generate() was called for {name}"
        if hasattr(prov, "hidden_call_counter"):
            assert prov.hidden_call_counter == 0, f"hidden calls executed for {name}"
            assert prov.generate_call_count == 0
        if isinstance(prov, MagicMock):
            assert prov.generate.call_count == 0

        # Verify zero requests were charged to tracker
        assert tracker.total_requests == 0, f"Tracker charged {tracker.total_requests} requests for {name}"
        assert tracker.remaining_request_headroom == 5


# ---------------------------------------------------------------------------
# Test 23: Task 3 TypeError Discrimination (Call-Site vs Internal Bug)
# ---------------------------------------------------------------------------
def test_task3_type_error_discrimination_call_site_vs_internal():
    """Verify that ControlledAgent distinguishes invocation argument-binding TypeError from internal TypeError."""
    from agentxray.agent_harness.agent import ControlledAgent
    from agentxray.agent_harness.cost_tracker import CostTracker, ExperimentBudgetConfig
    from agentxray.agent_harness.tools import MockToolEnvironment
    from agentxray.providers.fake import FakeProvider

    env = MockToolEnvironment()
    tracker = CostTracker(config=ExperimentBudgetConfig(model_name="fake-model", max_model_requests=5))

    # Internal TypeError raised deep inside generate() on a trusted built-in provider
    def buggy_handler(messages, tools):
        # Internal type error: adding incompatible types
        _ = "string" + 12345
        return None

    p_buggy = FakeProvider(custom_handler=buggy_handler)
    agent_buggy = ControlledAgent(provider=p_buggy, mock_env=env, cost_tracker=tracker)
    res_buggy = agent_buggy.run_session(system_prompt="", user_prompt="hi")
    assert res_buggy.finish_reason == "error"
    # MUST be classified as internal TypeError, NOT as max_retries rejection!
    assert "encountered internal TypeError during generation" in (res_buggy.error_message or "")
    assert "rejected retry limit" not in (res_buggy.error_message or "")


# ---------------------------------------------------------------------------
# Test 24: Task 4 Model Evidentiary Invariants on Instantiation and JSON Deserialization
# ---------------------------------------------------------------------------
def test_task4_model_evidentiary_invariants_and_deserialization():
    """Verify Pydantic invariant validators reject contradictory or affirmative power claims."""
    import json
    from pydantic import ValidationError
    from agentxray.models import (
        CandidateEvaluation,
        SearchStatus,
        StochasticBaselineComparison,
        StochasticDiscoveredCondition,
        StochasticMinimalityStatus,
    )
    from agentxray.verification import VerificationEvidenceTier, VerificationExecutionStatus, VerificationResult

    cand_eval = CandidateEvaluation(
        candidate_elements=("a",),
        trials=5,
        failures=5,
        failure_rate=1.0,
        confidence_interval=(0.5, 1.0),
    )

    # 1. VerificationResult rejects confirmatory_power_established=True
    with pytest.raises(ValidationError, match="confirmatory_power_established cannot be True"):
        VerificationResult(
            candidate=("a",),
            minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
            verification_status=VerificationExecutionStatus.COMPLETE,
            evidence_tier=VerificationEvidenceTier.CONFIRMATORY,
            sample_size_rationale="test",
            sample_size_validation_status="validated",
            confirmatory_power_established=True,
        )

    # 2. StochasticDiscoveredCondition rejects confirmatory_power_established=True
    with pytest.raises(ValidationError, match="confirmatory_power_established cannot be True"):
        StochasticDiscoveredCondition(
            subset_elements=("a",),
            status=SearchStatus.COMPLETE,
            is_failure=True,
            minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
            candidate_evaluation=cand_eval,
            target_calls_used=5,
            evidence_tier="confirmatory",
            sample_size_rationale="test",
            sample_size_validation_status="validated",
            confirmatory_power_established=True,
        )

    # 3. StochasticBaselineComparison rejects confirmatory_power_established=True
    with pytest.raises(ValidationError, match="confirmatory_power_established cannot be True"):
        StochasticBaselineComparison(
            baseline_name="test_baseline",
            discovered_elements=("a",),
            failure_reproduced=True,
            minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
            target_calls=5,
            candidate_evaluation=cand_eval,
            evidence_tier="confirmatory",
            sample_size_rationale="test",
            sample_size_validation_status="validated",
            confirmatory_power_established=True,
        )

    # 4. JSON deserialization with confirmatory_power_established=True is rejected
    valid_cond = StochasticDiscoveredCondition(
        subset_elements=("a",),
        status=SearchStatus.COMPLETE,
        is_failure=True,
        minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
        candidate_evaluation=cand_eval,
        target_calls_used=5,
        evidence_tier="exploratory",
        sample_size_rationale="exploratory_default",
        sample_size_validation_status="not_implemented",
        confirmatory_power_established=False,
    )
    raw_dict = json.loads(valid_cond.model_dump_json())
    raw_dict["confirmatory_power_established"] = True
    raw_dict["evidence_tier"] = "confirmatory"
    raw_dict["sample_size_validation_status"] = "validated"

    with pytest.raises(ValidationError, match="confirmatory_power_established cannot be True"):
        StochasticDiscoveredCondition.model_validate_json(json.dumps(raw_dict))

    # 5. Inconsistent state rejection: unverified condition with sample_size_validation_status="validated"
    with pytest.raises(ValidationError, match="cannot be 'validated' when verification has not been performed"):
        StochasticDiscoveredCondition(
            subset_elements=("a",),
            status=SearchStatus.COMPLETE,
            is_failure=True,
            minimality_status=StochasticMinimalityStatus.INCONCLUSIVE,
            candidate_evaluation=cand_eval,
            target_calls_used=5,
            evidence_tier=None,
            verification_status=None,
            sample_size_validation_status="validated",
        )

