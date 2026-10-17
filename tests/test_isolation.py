"""Architectural isolation tests.

Verifies that the probing algorithms and runner have zero import dependencies
or leakage from ground_truth.py.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any
import pytest

import agentxray.baselines
import agentxray.minimizer
import agentxray.models
import agentxray.runner
from agentxray.baselines import run_greedy_ablation, run_seeded_random_search
from agentxray.minimizer import DeltaDebugMinimizer
from agentxray.models import (
    ExperimentConfig,
    InputCase,
    TargetOutcome,
    TargetResult,
)
from agentxray.runner import InterventionRunner
from agentxray.stochastic_baselines import (
    run_stochastic_greedy_ablation,
    run_stochastic_random_search,
)
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.synthetic.ground_truth import GroundTruthSpec, evaluate_ground_truth
from benchmarks.synthetic.stochastic_ground_truth import (
    StochasticGroundTruthSpec,
    evaluate_stochastic_ground_truth,
)
from benchmarks.synthetic.stochastic_target import StochasticSyntheticTarget
from benchmarks.synthetic.target import SyntheticTarget


class TestArchitecturalIsolation:
    """Verifies that ground truth does not leak into the algorithm's decision path."""

    def test_minimizer_module_does_not_import_ground_truth(self) -> None:
        minimizer_source = inspect.getsource(agentxray.minimizer)
        assert "ground_truth" not in minimizer_source
        assert "GroundTruthSpec" not in minimizer_source
        assert "evaluate_ground_truth" not in minimizer_source

    def test_runner_module_does_not_import_ground_truth(self) -> None:
        runner_source = inspect.getsource(agentxray.runner)
        assert "ground_truth" not in runner_source
        assert "GroundTruthSpec" not in runner_source
        assert "evaluate_ground_truth" not in runner_source

    def test_baselines_module_does_not_import_ground_truth(self) -> None:
        baselines_source = inspect.getsource(agentxray.baselines)
        assert "benchmarks.synthetic.ground_truth" not in baselines_source
        assert "GroundTruthSpec" not in baselines_source
        assert "evaluate_ground_truth" not in baselines_source

    def test_stochastic_minimizer_does_not_import_ground_truth(self) -> None:
        import agentxray.stochastic_minimizer
        source = inspect.getsource(agentxray.stochastic_minimizer)
        assert "ground_truth" not in source
        assert "StochasticGroundTruthSpec" not in source
        assert "evaluate_stochastic_ground_truth" not in source

    def test_stochastic_baselines_does_not_import_ground_truth(self) -> None:
        import agentxray.stochastic_baselines
        source = inspect.getsource(agentxray.stochastic_baselines)
        assert "benchmarks.synthetic" not in source
        assert "StochasticGroundTruthSpec" not in source
        assert "evaluate_stochastic_ground_truth" not in source

    def test_target_exposes_no_public_rule_attributes(self) -> None:
        def sample_rule(elements: set[str]) -> tuple[bool, str, str | None]:
            return (False, "ok", None)

        target = SyntheticTarget(rule_fn=sample_rule, name="IsolationTarget")
        public_methods = [m for m in dir(target) if not m.startswith("_")]

        assert "execute" in public_methods
        assert "call_count" in public_methods
        assert "reset_call_count" in public_methods
        assert "name" in public_methods

        # Ensure no rule inspectability
        assert "rule_fn" not in public_methods
        assert "ground_truth" not in public_methods
        assert "trigger" not in public_methods

    def test_stochastic_target_exposes_no_public_rule_attributes(self) -> None:
        from benchmarks.synthetic.stochastic_target import StochasticSyntheticTarget

        def sample_prob_rule(elements: set[str]) -> float:
            return 0.50

        target = StochasticSyntheticTarget(rule_fn=sample_prob_rule, name="ProbIsolationTarget")
        public_methods = [m for m in dir(target) if not m.startswith("_")]

        assert "execute" in public_methods
        assert "call_count" in public_methods
        assert "reset" in public_methods
        assert "name" in public_methods

        # Ensure no rule or PRNG inspectability
        assert "rule_fn" not in public_methods
        assert "ground_truth" not in public_methods
        assert "trigger" not in public_methods
        assert "rng" not in public_methods


class AccessTrackingGuard:
    """Runtime guard and access monitor wrapping TargetResult.

    In strict mode, any attempt by a caller to read `metadata` or `output_message`
    immediately raises an AssertionError. It also records access counts to verify
    which attributes were queried during search algorithm execution.
    """

    def __init__(self, strict: bool = True) -> None:
        self.metadata_access_count = 0
        self.output_message_access_count = 0
        self.outcome_access_count = 0
        self.strict = strict

    def create_result(
        self,
        outcome: TargetOutcome,
        message: str = "diagnostic_message",
        metadata: dict[str, Any] | None = None,
        latency_ms: float = 1.0,
        error_code: str | None = None,
    ) -> TargetResult:
        tracker = self

        class GuardedTargetResult(TargetResult):
            def __getattribute__(self, name: str) -> Any:
                if name == "metadata":
                    tracker.metadata_access_count += 1
                    if tracker.strict:
                        raise AssertionError(
                            "Illegal isolation breach: search algorithm accessed TargetResult.metadata"
                        )
                elif name == "output_message":
                    tracker.output_message_access_count += 1
                    if tracker.strict:
                        raise AssertionError(
                            "Illegal isolation breach: search algorithm accessed TargetResult.output_message"
                        )
                elif name == "outcome":
                    tracker.outcome_access_count += 1
                return super().__getattribute__(name)

        return GuardedTargetResult(
            outcome=outcome,
            output_message=message,
            error_code=error_code,
            latency_ms=latency_ms,
            metadata=metadata if metadata is not None else {"forbidden_latent_key": "covert_channel_value"},
        )


class TestRuntimeIsolation:
    """Verifies at runtime that search methods never inspect metadata or output messages,
    stochastic targets do not disclose latent probabilities, and post-hoc evaluation operates separately.
    """

    def test_guard_validity_raises_on_metadata_or_output_message_inspection(self) -> None:
        """Verify the test double guard actively catches illegal access to TargetResult fields."""
        guard = AccessTrackingGuard(strict=True)
        res = guard.create_result(
            outcome=TargetOutcome.FAILURE,
            message="diagnostic error message",
            metadata={"secret_probability": 0.99},
        )

        # Legal access: outcome
        assert res.outcome == TargetOutcome.FAILURE
        assert guard.outcome_access_count == 1

        # Illegal access: metadata raises AssertionError
        with pytest.raises(AssertionError, match="TargetResult.metadata"):
            _ = res.metadata["secret_probability"]

        # Illegal access: output_message raises AssertionError
        with pytest.raises(AssertionError, match="TargetResult.output_message"):
            _ = "error" in res.output_message

    def test_guard_catches_simulated_candidate_ranking_leakage(self) -> None:
        """Verify guard fails if an algorithm attempts to rank candidates using latent metadata."""
        guard = AccessTrackingGuard(strict=True)

        def leaking_evaluator(result: TargetResult) -> float:
            return float(result.metadata.get("secret_probability", 0.0))

        res = guard.create_result(
            outcome=TargetOutcome.FAILURE,
            metadata={"secret_probability": 0.95},
        )

        with pytest.raises(AssertionError, match="TargetResult.metadata"):
            leaking_evaluator(res)

    def test_guard_catches_simulated_output_message_leakage(self) -> None:
        """Verify guard fails if an algorithm inspects output message text for stopping heuristics."""
        guard = AccessTrackingGuard(strict=True)

        def leaking_message_checker(result: TargetResult) -> bool:
            return "failed" in result.output_message

        res = guard.create_result(
            outcome=TargetOutcome.FAILURE,
            message="diagnostic error occurred",
        )

        with pytest.raises(AssertionError, match="TargetResult.output_message"):
            leaking_message_checker(res)

    def test_delta_debugging_runtime_isolation(self) -> None:
        """Verify DeltaDebugMinimizer.minimize makes zero metadata and output_message accesses."""
        guard = AccessTrackingGuard(strict=True)

        def target_fn(input_case: InputCase) -> TargetResult:
            is_failure = "trigger_a" in input_case.elements
            return guard.create_result(
                outcome=TargetOutcome.FAILURE if is_failure else TargetOutcome.SUCCESS,
                message="Failure occurred" if is_failure else "Success",
                metadata={"planted_rule": "trigger_a"},
            )

        config = ExperimentConfig(experiment_name="dd_isolation", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target_fn)
        minimizer = DeltaDebugMinimizer(runner=runner)

        initial_input = InputCase(
            case_id="case_dd",
            elements=("trigger_a", "distractor_1", "distractor_2", "distractor_3"),
            description="Test case for DD isolation",
        )

        discovered = minimizer.minimize(initial_input)

        assert discovered.subset_elements == ("trigger_a",)
        assert discovered.is_failure is True
        assert guard.metadata_access_count == 0
        assert guard.output_message_access_count == 0
        assert guard.outcome_access_count > 0

    def test_greedy_ablation_runtime_isolation(self) -> None:
        """Verify run_greedy_ablation makes zero metadata and output_message accesses."""
        guard = AccessTrackingGuard(strict=True)

        def target_fn(input_case: InputCase) -> TargetResult:
            is_failure = "trigger_a" in input_case.elements
            return guard.create_result(
                outcome=TargetOutcome.FAILURE if is_failure else TargetOutcome.SUCCESS,
                message="Failure occurred" if is_failure else "Success",
                metadata={"planted_rule": "trigger_a"},
            )

        config = ExperimentConfig(experiment_name="greedy_isolation", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target_fn)

        initial_input = InputCase(
            case_id="case_greedy",
            elements=("trigger_a", "distractor_1", "distractor_2", "distractor_3"),
            description="Test case for Greedy isolation",
        )

        comparison = run_greedy_ablation(initial_input=initial_input, runner=runner)

        assert comparison.discovered_elements == ("trigger_a",)
        assert comparison.failure_reproduced is True
        assert guard.metadata_access_count == 0
        assert guard.output_message_access_count == 0
        assert guard.outcome_access_count > 0

    def test_seeded_random_search_runtime_isolation(self) -> None:
        """Verify run_seeded_random_search makes zero metadata and output_message accesses."""
        guard = AccessTrackingGuard(strict=True)

        def target_fn(input_case: InputCase) -> TargetResult:
            is_failure = "trigger_a" in input_case.elements
            return guard.create_result(
                outcome=TargetOutcome.FAILURE if is_failure else TargetOutcome.SUCCESS,
                message="Failure occurred" if is_failure else "Success",
                metadata={"planted_rule": "trigger_a"},
            )

        config = ExperimentConfig(experiment_name="random_isolation", max_trials=100)
        runner = InterventionRunner(config=config, target_fn=target_fn)

        initial_input = InputCase(
            case_id="case_random",
            elements=("trigger_a", "distractor_1", "distractor_2"),
            description="Test case for Random Search isolation",
        )

        comparison = run_seeded_random_search(initial_input=initial_input, runner=runner, seed=42)

        assert comparison.discovered_elements == ("trigger_a",)
        assert comparison.failure_reproduced is True
        assert guard.metadata_access_count == 0
        assert guard.output_message_access_count == 0
        assert guard.outcome_access_count > 0

    def test_stochastic_delta_debugging_runtime_isolation(self) -> None:
        """Verify StochasticDeltaDebugMinimizer.minimize makes zero metadata and output_message accesses."""
        guard = AccessTrackingGuard(strict=True)

        def target_fn(input_case: InputCase) -> TargetResult:
            is_failure = "trigger_a" in input_case.elements
            return guard.create_result(
                outcome=TargetOutcome.FAILURE if is_failure else TargetOutcome.SUCCESS,
                message="Stochastic execution failed" if is_failure else "Stochastic execution succeeded",
                metadata={"latent_p": 0.95 if is_failure else 0.05},
            )

        config = ExperimentConfig(experiment_name="stochastic_dd_isolation", max_trials=100)
        runner = InterventionRunner(config=config, target_fn=target_fn)
        minimizer = StochasticDeltaDebugMinimizer(
            runner=runner,
            trials_per_candidate=5,
            failure_threshold=0.50,
        )

        initial_input = InputCase(
            case_id="case_stoch_dd",
            elements=("trigger_a", "distractor_1", "distractor_2"),
            description="Test case for Stochastic DD isolation",
        )

        discovered = minimizer.minimize(initial_input)

        assert discovered.subset_elements == ("trigger_a",)
        assert discovered.is_failure is True
        assert guard.metadata_access_count == 0
        assert guard.output_message_access_count == 0
        assert guard.outcome_access_count > 0

    def test_stochastic_greedy_ablation_runtime_isolation(self) -> None:
        """Verify run_stochastic_greedy_ablation makes zero metadata and output_message accesses."""
        guard = AccessTrackingGuard(strict=True)

        def target_fn(input_case: InputCase) -> TargetResult:
            is_failure = "trigger_a" in input_case.elements
            return guard.create_result(
                outcome=TargetOutcome.FAILURE if is_failure else TargetOutcome.SUCCESS,
                message="Stochastic execution failed" if is_failure else "Stochastic execution succeeded",
                metadata={"latent_p": 0.95 if is_failure else 0.05},
            )

        config = ExperimentConfig(experiment_name="stochastic_greedy_isolation", max_trials=100)
        runner = InterventionRunner(config=config, target_fn=target_fn)

        initial_input = InputCase(
            case_id="case_stoch_greedy",
            elements=("trigger_a", "distractor_1", "distractor_2"),
            description="Test case for Stochastic Greedy isolation",
        )

        comparison = run_stochastic_greedy_ablation(
            initial_input=initial_input,
            runner=runner,
            trials_per_candidate=5,
            failure_threshold=0.50,
        )

        assert comparison.discovered_elements == ("trigger_a",)
        assert comparison.failure_reproduced is True
        assert guard.metadata_access_count == 0
        assert guard.output_message_access_count == 0
        assert guard.outcome_access_count > 0

    def test_stochastic_random_search_runtime_isolation(self) -> None:
        """Verify run_stochastic_random_search makes zero metadata and output_message accesses."""
        guard = AccessTrackingGuard(strict=True)

        def target_fn(input_case: InputCase) -> TargetResult:
            is_failure = "trigger_a" in input_case.elements
            return guard.create_result(
                outcome=TargetOutcome.FAILURE if is_failure else TargetOutcome.SUCCESS,
                message="Stochastic execution failed" if is_failure else "Stochastic execution succeeded",
                metadata={"latent_p": 0.95 if is_failure else 0.05},
            )

        config = ExperimentConfig(experiment_name="stochastic_random_isolation", max_trials=100)
        runner = InterventionRunner(config=config, target_fn=target_fn)

        initial_input = InputCase(
            case_id="case_stoch_random",
            elements=("trigger_a", "distractor_1", "distractor_2"),
            description="Test case for Stochastic Random Search isolation",
        )

        comparison = run_stochastic_random_search(
            initial_input=initial_input,
            runner=runner,
            seed=42,
            trials_per_candidate=5,
            failure_threshold=0.50,
        )

        assert comparison.discovered_elements == ("trigger_a",)
        assert comparison.failure_reproduced is True
        assert guard.metadata_access_count == 0
        assert guard.output_message_access_count == 0
        assert guard.outcome_access_count > 0

    def test_stochastic_target_observable_results_do_not_disclose_latent_probability(self) -> None:
        """Verify StochasticSyntheticTarget.execute() observable fields do not disclose latent probabilities."""
        def sample_rule(elements: set[str]) -> float:
            return 0.85 if "trigger" in elements else 0.15

        target = StochasticSyntheticTarget(rule_fn=sample_rule, name="TelemetryCheckTarget", target_seed=42)

        res_fail = target.execute(InputCase(case_id="c1", elements=("trigger", "d1")))
        res_succ = target.execute(InputCase(case_id="c2", elements=("d1",)))

        for res in (res_fail, res_succ):
            # Metadata must NOT contain underlying_probability or any float matching the latent parameter
            assert "underlying_probability" not in res.metadata
            assert 0.85 not in res.metadata.values()
            assert 0.15 not in res.metadata.values()
            # Message must NOT format P= or the probability value
            assert "P=" not in res.output_message
            assert "0.85" not in res.output_message
            assert "0.15" not in res.output_message
            # Observable status messages
            assert res.output_message in ("Stochastic execution failed", "Stochastic execution succeeded")
            # Operational non-sensitive metadata is preserved
            assert res.metadata["target_name"] == "TelemetryCheckTarget"
            assert "element_count" in res.metadata
            assert "replicate_index" in res.metadata
            assert "crn_mode" in res.metadata

        # Internal probability calculation is preserved and accurate
        assert target._rule_fn({"trigger", "d1"}) == 0.85
        assert target._rule_fn({"d1"}) == 0.15

    def test_post_hoc_ground_truth_evaluation_operates_without_telemetry(self) -> None:
        """Confirm ground-truth evaluation works strictly post-hoc and needs zero target telemetry."""
        # 1. Deterministic post-hoc evaluation
        det_gt = GroundTruthSpec(
            task_id="det_task",
            name="Deterministic Spec",
            description="Fails on trigger_a",
            trigger_elements=("trigger_a",),
        )
        guard_det = AccessTrackingGuard(strict=True)

        def det_target(input_case: InputCase) -> TargetResult:
            is_fail = "trigger_a" in input_case.elements
            return guard_det.create_result(
                outcome=TargetOutcome.FAILURE if is_fail else TargetOutcome.SUCCESS,
                message="fail" if is_fail else "ok",
                metadata={"rule": "trigger_a"},
            )

        runner_det = InterventionRunner(ExperimentConfig(experiment_name="det_eval"), det_target)
        discovered_det = DeltaDebugMinimizer(runner_det).minimize(
            InputCase(case_id="c", elements=("trigger_a", "d1", "d2"))
        )

        matches_gt, prec, rec, excess, missing = evaluate_ground_truth(det_gt, discovered_det)
        assert matches_gt is True
        assert prec == 1.0
        assert rec == 1.0
        assert excess == ()
        assert missing == ()

        # 2. Stochastic post-hoc evaluation
        stoch_gt = StochasticGroundTruthSpec(
            task_id="stoch_task",
            name="Stochastic Spec",
            description="Fails on trigger_s",
            trigger_elements=("trigger_s",),
            base_probability=0.05,
            trigger_probability=0.95,
        )
        guard_stoch = AccessTrackingGuard(strict=True)

        def stoch_target(input_case: InputCase) -> TargetResult:
            is_fail = "trigger_s" in input_case.elements
            return guard_stoch.create_result(
                outcome=TargetOutcome.FAILURE if is_fail else TargetOutcome.SUCCESS,
                message="fail" if is_fail else "ok",
                metadata={"prob": 0.95 if is_fail else 0.05},
            )

        runner_stoch = InterventionRunner(ExperimentConfig(experiment_name="stoch_eval"), stoch_target)
        discovered_stoch = StochasticDeltaDebugMinimizer(
            runner=runner_stoch, trials_per_candidate=5, failure_threshold=0.5
        ).minimize(InputCase(case_id="cs", elements=("trigger_s", "d1")))

        s_matches, s_prec, s_rec, s_excess, s_missing = evaluate_stochastic_ground_truth(
            stoch_gt, discovered_stoch
        )
        assert s_matches is True
        assert s_prec == 1.0
        assert s_rec == 1.0
        assert s_excess == ()
        assert s_missing == ()
