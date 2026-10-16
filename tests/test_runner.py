"""Tests for InterventionRunner in agentxray.runner."""

from __future__ import annotations

import time
import pytest

from agentxray.models import ExperimentConfig, InputCase, TargetOutcome, TargetResult
from agentxray.runner import BudgetExhaustedError, InterventionRunner, TimeoutExceededError


def mock_target_always_fail(case: InputCase) -> TargetResult:
    return TargetResult(outcome=TargetOutcome.FAILURE, output_message="failed")


def mock_target_always_succeed(case: InputCase) -> TargetResult:
    return TargetResult(outcome=TargetOutcome.SUCCESS, output_message="ok")


class TestInterventionRunner:
    """Tests for runner execution, recording, and budget enforcement."""

    def test_trial_recording_and_provenance(self) -> None:
        config = ExperimentConfig(experiment_name="run_test", max_trials=10)
        runner = InterventionRunner(config=config, target_fn=mock_target_always_fail)

        case1 = InputCase(elements=["a", "b"])
        case2 = InputCase(elements=["a"])

        res1 = runner.execute_trial(case1, intervention_type="baseline")
        res2 = runner.execute_trial(case2, intervention_type="ablation", parent_trial_id=runner.trials[0].trial_id)

        assert runner.trial_count == 2
        assert len(runner.trials) == 2
        assert runner.trials[0].trial_index == 0
        assert runner.trials[0].intervention_type == "baseline"
        assert runner.trials[0].parent_trial_id is None
        assert runner.trials[0].input_case == case1
        assert runner.trials[0].result == res1

        assert runner.trials[1].trial_index == 1
        assert runner.trials[1].intervention_type == "ablation"
        assert runner.trials[1].parent_trial_id == runner.trials[0].trial_id
        assert runner.trials[1].input_case == case2
        assert runner.trials[1].result == res2

    def test_budget_exhaustion_enforced(self) -> None:
        config = ExperimentConfig(experiment_name="budget_test", max_trials=3)
        runner = InterventionRunner(config=config, target_fn=mock_target_always_succeed)
        case = InputCase(elements=["a", "b"])

        # Execute 3 allowed trials
        runner.execute_trial(case, intervention_type="t1")
        runner.execute_trial(case, intervention_type="t2")
        runner.execute_trial(case, intervention_type="t3")

        assert runner.is_budget_exhausted() is True

        # 4th trial must raise BudgetExhaustedError
        with pytest.raises(BudgetExhaustedError):
            runner.execute_trial(case, intervention_type="t4")

    def test_timeout_enforced(self) -> None:
        config = ExperimentConfig(experiment_name="timeout_test", max_trials=100, timeout_seconds=0.05)
        runner = InterventionRunner(config=config, target_fn=mock_target_always_succeed)
        case = InputCase(elements=["x", "y"])

        # Run 1 trial
        runner.execute_trial(case, intervention_type="t1")

        # Sleep to exceed timeout
        time.sleep(0.06)
        assert runner.is_timed_out() is True

        with pytest.raises(TimeoutExceededError):
            runner.execute_trial(case, intervention_type="t2")

    def test_deterministic_reproducibility(self) -> None:
        """Verify that identical targets produce identical trial histories."""
        def stateful_mock(case: InputCase) -> TargetResult:
            is_fail = "fail_elem" in case.elements
            return TargetResult(
                outcome=TargetOutcome.FAILURE if is_fail else TargetOutcome.SUCCESS,
                output_message="det_output",
            )

        config = ExperimentConfig(experiment_name="det_test", max_trials=10)

        runner1 = InterventionRunner(config=config, target_fn=stateful_mock)
        runner2 = InterventionRunner(config=config, target_fn=stateful_mock)

        cases = [
            InputCase(elements=["fail_elem", "other"]),
            InputCase(elements=["other"]),
            InputCase(elements=["fail_elem"]),
        ]

        for c in cases:
            runner1.execute_trial(c, intervention_type="step")
            runner2.execute_trial(c, intervention_type="step")

        assert [t.result.outcome for t in runner1.trials] == [t.result.outcome for t in runner2.trials]
        assert [t.input_case.elements for t in runner1.trials] == [t.input_case.elements for t in runner2.trials]
