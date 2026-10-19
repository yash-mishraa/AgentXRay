"""Focused regression tests for Null-Shock Outcome Semantics and Scoring Correction (Phase 4.6K.13).

Verifies the 10 requirements:
1. External-shock failure persists when the empty candidate set is actually evaluated.
2. Verified persistent external failure yields an empty causal diagnosis only when the task contract supports it.
3. NOT_MINIMAL without a verified empty-set result does not automatically become an empty causal diagnosis.
4. A candidate cannot be selected merely because the minimizer cannot reduce below one element.
5. The clean null task retains its existing correct-empty behavior.
6. A false candidate attribution on either null task is penalized.
7. An unverified empty prediction is not credited as exact recovery.
8. Active-task exact recovery and false-attribution scoring remain unchanged.
9. Infrastructure failure remains distinguishable from a diagnostic outcome.
10. Historical synthetic-control regression tests remain unchanged in meaning.
"""

from __future__ import annotations

import pytest

from agentxray.agent_harness.tools import CALCULATOR_TOOL_DEF
from agentxray.models import (
    CandidateEvaluation,
    ExperimentConfig,
    InputCase,
    SearchStatus,
    StochasticMinimalityStatus,
    TargetOutcome,
    TargetResult,
)
from agentxray.runner import InterventionRunner
from agentxray.stochastic_baselines import run_stochastic_greedy_ablation
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.multi_mechanism.runner import (
    evaluate_mechanism_three_way,
    run_multi_mechanism_benchmark,
    score_diagnostic_outcome,
)
from benchmarks.multi_mechanism.tasks import (
    MECHANISM_REGISTRY,
    get_mechanism_task,
    get_trace_evaluated_task,
    list_mechanism_tasks,
)


class TestNullShockSemanticsPhase46K13:
    """Verifies all 10 requirements of Phase 4.6K.13."""

    def test_case_1_external_shock_persists_on_empty_set(self) -> None:
        """1. External-shock failure persists when the empty candidate set is actually evaluated."""
        for task_id in ("trace_null_02_unmodeled_external_shock", "mech_null_ext"):
            spec = (
                get_trace_evaluated_task(task_id)
                if task_id.startswith("trace_")
                else get_mechanism_task(task_id)
            )
            target = spec.create_target()
            initial_input = spec.create_initial_input()

            # Full set of candidates fails
            res_full = target.execute(initial_input)
            assert res_full.outcome == TargetOutcome.FAILURE
            assert "500" in res_full.output_message

            # Empty candidate set () MUST also be evaluated and reproduce failure
            empty_input = initial_input.subset(())
            assert empty_input.elements == ()
            res_empty = target.execute(empty_input)
            assert res_empty.outcome == TargetOutcome.FAILURE
            assert "500" in res_empty.output_message

    def test_case_2_verified_persistent_failure_yields_empty_diagnosis_when_contract_supports(self) -> None:
        """2. Verified persistent external failure yields an empty causal diagnosis only when task contract supports it."""
        spec = get_trace_evaluated_task("trace_null_02_unmodeled_external_shock")
        initial_input = spec.create_initial_input()

        # A: Active DD minimizer on external shock
        dd_target = spec.create_target()
        dd_runner = InterventionRunner(
            config=ExperimentConfig(experiment_name="test_c2_dd", max_trials=15),
            target_fn=dd_target.execute,
        )
        minimizer = StochasticDeltaDebugMinimizer(
            runner=dd_runner,
            trials_per_candidate=2,
            failure_threshold=0.30,
        )
        cond = minimizer.minimize(initial_input)
        assert cond.subset_elements == ()
        assert cond.minimality_status == StochasticMinimalityStatus.VERIFIED_EMPTY
        assert "empty-set failure rate" in cond.status_message.lower()

        # B: Greedy ablation on external shock
        greedy_target = spec.create_target()
        greedy_runner = InterventionRunner(
            config=ExperimentConfig(experiment_name="test_c2_greedy", max_trials=15),
            target_fn=greedy_target.execute,
        )
        greedy_res = run_stochastic_greedy_ablation(
            initial_input=initial_input,
            runner=greedy_runner,
            trials_per_candidate=2,
            failure_threshold=0.30,
        )
        assert greedy_res.discovered_elements == ()
        assert greedy_res.minimality_status == StochasticMinimalityStatus.VERIFIED_EMPTY

        # Scored on null contract: S_true == () -> exact recovery R=1, false attribution Y=0
        score_null = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="active_dd",
            candidate_set=cond.subset_elements,
            is_valid_diagnosis=True,
            ground_truth_trigger=(),
            is_null_case=True,
            verification_status=cond.minimality_status.value,
        )
        assert score_null.false_attribution_indicator == 0
        assert score_null.exact_recovery_indicator == 1
        assert score_null.verification_status == "verified_empty"

        # On an active task contract (e.g. ground truth has planted trigger):
        # A verified empty prediction yields Y=0, but exact_recovery R=0 (contract does NOT support empty trigger)
        score_active_contract = score_diagnostic_outcome(
            mechanism_id="active_task_hypothetical",
            method_name="active_dd",
            candidate_set=(),
            is_valid_diagnosis=True,
            ground_truth_trigger=("planted_trigger_element",),
            is_null_case=False,
            verification_status="verified_empty",
        )
        assert score_active_contract.false_attribution_indicator == 0
        assert score_active_contract.exact_recovery_indicator == 0

    def test_case_3_not_minimal_without_verified_empty_does_not_become_empty_diagnosis(self) -> None:
        """3. NOT_MINIMAL without a verified empty-set result does not automatically become an empty causal diagnosis."""
        # Scenario: Minimizer classifies candidate as NOT_MINIMAL (e.g. non-monotonic trial), but () was not evaluated/verified
        # The runner must mark it as invalid candidate diagnosis, NOT convert it to ()
        spec = get_trace_evaluated_task("trace_null_02_unmodeled_external_shock")

        # Simulate a condition that ended in NOT_MINIMAL with candidate elements retained
        mock_eval = CandidateEvaluation(
            candidate_elements=("elem_0", "elem_1"),
            trials=5,
            failures=5,
            failure_rate=1.0,
            confidence_interval=(0.56, 1.0),
        )

        # Direct scoring of a NOT_MINIMAL diagnosis:
        # If a method attempts to output () with verification_status="not_minimal":
        score_not_min_empty = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="flawed_method",
            candidate_set=(),
            is_valid_diagnosis=True,
            ground_truth_trigger=(),
            is_null_case=True,
            verification_status="not_minimal",
        )
        # Cannot receive credit as exact recovery because it is not verified empty
        assert score_not_min_empty.false_attribution_indicator == 0
        assert score_not_min_empty.exact_recovery_indicator == 0

        # And if runner receives cond.minimality_status == NOT_MINIMAL, is_valid_diagnosis is False
        score_not_min_invalid = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="active_dd",
            candidate_set=None,
            is_valid_diagnosis=False,
            ground_truth_trigger=(),
            is_null_case=True,
            verification_status="not_minimal",
            termination_reason="not_minimal",
        )
        assert score_not_min_invalid.is_valid_diagnosis is False
        assert score_not_min_invalid.predicted_candidate_set is None
        assert score_not_min_invalid.false_attribution_indicator is None
        assert score_not_min_invalid.exact_recovery_indicator == 0

    def test_case_4_candidate_not_selected_merely_due_to_minimizer_element_floor(self) -> None:
        """4. A candidate cannot be selected merely because the minimizer cannot reduce below one element."""
        # Set up a target with 1 candidate element where mock calculator fails unconditionally
        def unconditional_fail_target(input_case: InputCase) -> TargetResult:
            return TargetResult(
                outcome=TargetOutcome.FAILURE,
                output_message="HTTP 500 Unconditional Failure",
            )

        single_cand_input = InputCase(
            case_id="floor_test",
            elements=("lone_candidate_element",),
        )

        # Active DD: must evaluate () when current_candidate has length 1, and reduce to ()
        dd_runner = InterventionRunner(
            config=ExperimentConfig(experiment_name="floor_dd", max_trials=10),
            target_fn=unconditional_fail_target,
        )
        minimizer = StochasticDeltaDebugMinimizer(
            runner=dd_runner,
            trials_per_candidate=2,
            failure_threshold=0.30,
        )
        dd_cond = minimizer.minimize(single_cand_input)
        assert dd_cond.subset_elements == ()
        assert dd_cond.minimality_status == StochasticMinimalityStatus.VERIFIED_EMPTY

        # Greedy Ablation: must evaluate () when current_elements has length 1, and reduce to ()
        greedy_runner = InterventionRunner(
            config=ExperimentConfig(experiment_name="floor_greedy", max_trials=10),
            target_fn=unconditional_fail_target,
        )
        greedy_res = run_stochastic_greedy_ablation(
            initial_input=single_cand_input,
            runner=greedy_runner,
            trials_per_candidate=2,
            failure_threshold=0.30,
        )
        assert greedy_res.discovered_elements == ()
        assert greedy_res.minimality_status == StochasticMinimalityStatus.VERIFIED_EMPTY

    def test_case_5_clean_null_task_retains_existing_correct_empty_behavior(self) -> None:
        """5. The clean null task retains its existing correct-empty behavior."""
        for clean_id in ("trace_null_01_clean_execution", "mech_null_clean"):
            spec = (
                get_trace_evaluated_task(clean_id)
                if clean_id.startswith("trace_")
                else get_mechanism_task(clean_id)
            )
            initial_input = spec.create_initial_input()

            # Baseline execution succeeds cleanly
            target = spec.create_target()
            res = target.execute(initial_input)
            assert res.outcome == TargetOutcome.SUCCESS

            # Active DD Minimizer returns () with NO_TRIGGER_FOUND
            dd_target = spec.create_target()
            dd_runner = InterventionRunner(
                config=ExperimentConfig(experiment_name="c5_dd", max_trials=10),
                target_fn=dd_target.execute,
            )
            dd_cond = StochasticDeltaDebugMinimizer(runner=dd_runner, trials_per_candidate=2).minimize(initial_input)
            assert dd_cond.subset_elements == ()
            assert dd_cond.status == SearchStatus.NO_TRIGGER_FOUND
            assert dd_cond.is_failure is False

            dd_score = score_diagnostic_outcome(
                mechanism_id=spec.mechanism_id,
                method_name="active_dd",
                candidate_set=dd_cond.subset_elements,
                is_valid_diagnosis=True,
                ground_truth_trigger=(),
                is_null_case=True,
                verification_status=dd_cond.minimality_status.value,
            )
            assert dd_score.false_attribution_indicator == 0
            assert dd_score.exact_recovery_indicator == 1

            # Greedy ablation returns () with failure_reproduced=False
            greedy_target = spec.create_target()
            greedy_runner = InterventionRunner(
                config=ExperimentConfig(experiment_name="c5_greedy", max_trials=10),
                target_fn=greedy_target.execute,
            )
            greedy_res = run_stochastic_greedy_ablation(initial_input, runner=greedy_runner, trials_per_candidate=2)
            assert greedy_res.discovered_elements == ()
            assert greedy_res.failure_reproduced is False

            greedy_score = score_diagnostic_outcome(
                mechanism_id=spec.mechanism_id,
                method_name="greedy_ablation",
                candidate_set=greedy_res.discovered_elements,
                is_valid_diagnosis=True,
                ground_truth_trigger=(),
                is_null_case=True,
                verification_status=greedy_res.minimality_status.value,
            )
            assert greedy_score.false_attribution_indicator == 0
            assert greedy_score.exact_recovery_indicator == 1

    def test_case_6_false_candidate_attribution_on_null_tasks_penalized(self) -> None:
        """6. A false candidate attribution on either null task is penalized."""
        for null_id in (
            "trace_null_01_clean_execution",
            "trace_null_02_unmodeled_external_shock",
            "mech_null_clean",
            "mech_null_ext",
        ):
            spec = (
                get_trace_evaluated_task(null_id)
                if null_id.startswith("trace_")
                else get_mechanism_task(null_id)
            )
            distractor = spec.candidate_elements[0]

            score = score_diagnostic_outcome(
                mechanism_id=spec.mechanism_id,
                method_name="passive_judge",
                candidate_set=(distractor,),
                is_valid_diagnosis=True,
                ground_truth_trigger=(),
                is_null_case=True,
            )
            assert score.false_attribution_indicator == 1
            assert score.exact_recovery_indicator == 0

    def test_case_7_unverified_empty_prediction_not_credited_as_exact_recovery(self) -> None:
        """7. An unverified empty prediction is not credited as exact recovery."""
        for unverified_status in ("unverified", "not_verified", "inconclusive", "not_minimal"):
            score = score_diagnostic_outcome(
                mechanism_id="null_task_test",
                method_name="unverified_method",
                candidate_set=(),
                is_valid_diagnosis=True,
                ground_truth_trigger=(),
                is_null_case=True,
                verification_status=unverified_status,
            )
            # Must not be penalized for false attribution (it blamed nothing)
            assert score.false_attribution_indicator == 0
            # BUT must NOT receive unearned credit for exact recovery
            assert score.exact_recovery_indicator == 0

    def test_case_8_active_task_scoring_remains_unchanged(self) -> None:
        """8. Active-task exact recovery and false-attribution scoring remain unchanged."""
        spec = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        trigger = spec.planted_trigger_elements[0]
        distractor = spec.candidate_elements[1]

        # A: Exact recovery of true trigger
        score_exact = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="active_dd",
            candidate_set=(trigger,),
            is_valid_diagnosis=True,
            ground_truth_trigger=spec.planted_trigger_elements,
            is_null_case=False,
            verification_status="statistically_supported",
        )
        assert score_exact.false_attribution_indicator == 0
        assert score_exact.exact_recovery_indicator == 1

        # B: Spurious distractor inclusion
        score_spurious = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="active_dd",
            candidate_set=(trigger, distractor),
            is_valid_diagnosis=True,
            ground_truth_trigger=spec.planted_trigger_elements,
            is_null_case=False,
        )
        assert score_spurious.false_attribution_indicator == 1
        assert score_spurious.exact_recovery_indicator == 0

        # C: Valid empty prediction on active task (missed cause, no spurious blame)
        score_empty_active = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="active_dd",
            candidate_set=(),
            is_valid_diagnosis=True,
            ground_truth_trigger=spec.planted_trigger_elements,
            is_null_case=False,
        )
        assert score_empty_active.false_attribution_indicator == 0
        assert score_empty_active.exact_recovery_indicator == 0

    def test_case_9_infrastructure_failure_distinguishable_from_diagnostic_outcome(self) -> None:
        """9. Infrastructure failure remains distinguishable from a diagnostic outcome."""
        score_infra = score_diagnostic_outcome(
            mechanism_id="trace_env_01_service_http_500",
            method_name="active_dd",
            candidate_set=None,
            is_valid_diagnosis=False,
            ground_truth_trigger=("some_trigger",),
            is_null_case=False,
            verification_status="infra_failure",
            termination_reason="infra_failure",
        )
        assert score_infra.is_valid_diagnosis is False
        assert score_infra.predicted_candidate_set is None
        assert score_infra.false_attribution_indicator is None
        assert score_infra.exact_recovery_indicator is None
        assert score_infra.termination_reason == "infra_failure"
        assert score_infra.verification_status == "infra_failure"

    def test_case_10_historical_synthetic_control_regression_tests_unchanged_meaning(self) -> None:
        """10. Historical synthetic-control regression tests remain unchanged in meaning."""
        assert len(MECHANISM_REGISTRY) == 18
        specs = list_mechanism_tasks()
        assert len(specs) == 18

        # Run synthetic control tier through the runner
        report = run_multi_mechanism_benchmark(
            tier="synthetic_control",
            trials_per_candidate=1,
            max_trials=10,
            seed=42,
        )
        assert report.n_total == 18
        assert report.n_infra == 0
        assert report.n_evaluable == 18
        assert len(report.evaluations) == 18

        # In particular, check that mech_null_ext now achieves exact recovery without false attribution
        ext_eval = next(e for e in report.evaluations if e.mechanism_id == "mech_null_ext")
        assert ext_eval.is_null_case is True
        assert ext_eval.active_dd_outcome.is_valid_diagnosis is True
        assert ext_eval.active_dd_outcome.predicted_candidate_set == ()
        assert ext_eval.active_dd_outcome.false_attribution_indicator == 0
        assert ext_eval.active_dd_outcome.exact_recovery_indicator == 1
        assert ext_eval.active_dd_outcome.verification_status == "verified_empty"

        assert ext_eval.greedy_ablation_outcome.is_valid_diagnosis is True
        assert ext_eval.greedy_ablation_outcome.predicted_candidate_set == ()
        assert ext_eval.greedy_ablation_outcome.false_attribution_indicator == 0
        assert ext_eval.greedy_ablation_outcome.exact_recovery_indicator == 1
        assert ext_eval.greedy_ablation_outcome.verification_status == "verified_empty"
