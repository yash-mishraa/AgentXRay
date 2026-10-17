"""Tests for DeltaDebugMinimizer in agentxray.minimizer."""

from __future__ import annotations

import pytest

from agentxray.minimizer import DeltaDebugMinimizer
from agentxray.models import ExperimentConfig, InputCase, SearchStatus, TargetOutcome, TargetResult
from agentxray.runner import InterventionRunner
from benchmarks.synthetic.tasks import (
    create_conjunctive_task,
    create_distractor_task,
    create_null_task,
    create_single_factor_task,
)


class TestDeltaDebugMinimizer:
    """Core algorithmic tests for delta debugging on synthetic benchmark tasks."""

    def test_single_factor_discovery(self) -> None:
        target, initial_input, ground_truth = create_single_factor_task()
        config = ExperimentConfig(experiment_name="test_single_factor", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target.execute)
        minimizer = DeltaDebugMinimizer(runner=runner)

        result = minimizer.minimize(initial_input)

        assert result.status == SearchStatus.COMPLETE
        assert result.is_failure is True
        assert result.subset_elements == ("param_mode_debug",)
        assert result.is_one_minimal is True
        assert result.is_globally_minimal is True
        assert result.verified_by_check is True
        assert result.target_calls_used > 0
        assert set(result.subset_elements) == ground_truth.trigger_set

    def test_conjunctive_two_factor_discovery(self) -> None:
        target, initial_input, ground_truth = create_conjunctive_task()
        config = ExperimentConfig(experiment_name="test_conjunctive", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target.execute)
        minimizer = DeltaDebugMinimizer(runner=runner)

        result = minimizer.minimize(initial_input)

        assert result.status == SearchStatus.COMPLETE
        assert result.is_failure is True
        assert set(result.subset_elements) == {"schema_type_strict", "flag_allow_null"}
        assert result.is_one_minimal is True
        # For multi-element trigger, delta debugging does not claim global minimality without exhaustive search
        assert result.is_globally_minimal is False
        assert result.verified_by_check is True
        assert set(result.subset_elements) == ground_truth.trigger_set

    def test_distractor_removal(self) -> None:
        """Verify that all 4 distractor elements are stripped, leaving only the causal element."""
        target, initial_input, ground_truth = create_distractor_task()
        config = ExperimentConfig(experiment_name="test_distractors", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target.execute)
        minimizer = DeltaDebugMinimizer(runner=runner)

        result = minimizer.minimize(initial_input)

        assert result.status == SearchStatus.COMPLETE
        assert result.is_failure is True
        assert result.subset_elements == ("token_auth_bearer",)
        assert result.is_one_minimal is True
        # Ensure none of the distractor_* elements are in the result
        for elem in result.subset_elements:
            assert not elem.startswith("distractor_")
        assert set(result.subset_elements) == ground_truth.trigger_set

    def test_null_non_failing_task(self) -> None:
        """Verify that a target which never fails is reported as no trigger found."""
        target, initial_input, ground_truth = create_null_task()
        config = ExperimentConfig(experiment_name="test_null", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target.execute)
        minimizer = DeltaDebugMinimizer(runner=runner)

        result = minimizer.minimize(initial_input)

        assert result.status == SearchStatus.NO_TRIGGER_FOUND
        assert result.is_failure is False
        assert result.subset_elements == ()
        assert result.is_one_minimal is False
        assert result.verified_by_check is False

    def test_budget_exhaustion_during_minimization(self) -> None:
        """Verify that when budget is insufficient, incomplete status is reported."""
        target, initial_input, ground_truth = create_distractor_task()
        # Set budget to only 2 calls (baseline + 1 trial)
        config = ExperimentConfig(experiment_name="test_exhaustion", max_trials=2)
        runner = InterventionRunner(config=config, target_fn=target.execute)
        minimizer = DeltaDebugMinimizer(runner=runner)

        result = minimizer.minimize(initial_input)

        assert result.status == SearchStatus.BUDGET_EXHAUSTED
        assert result.verified_by_check is False
        assert result.is_one_minimal is False
        assert runner.trial_count == 2

    def test_non_monotonic_conservative_handling(self) -> None:
        """Verify that target exhibiting non-monotonic behavior is handled conservatively."""
        # Fails on {A, B, C}, fails on {A, B}, fails on {A}.
        # But passes on {B, C}.
        def non_monotonic_rule(elements: set[str]) -> tuple[bool, str, str | None]:
            if "A" in elements:
                return (True, "Fails because A is present", "ERR_A")
            return (False, "Passes without A", None)

        from benchmarks.synthetic.target import SyntheticTarget

        target = SyntheticTarget(rule_fn=non_monotonic_rule, name="NonMonotonicTarget")
        initial_input = InputCase(
            case_id="non_monotonic_case",
            elements=["A", "B", "C", "D", "E"],
        )

        config = ExperimentConfig(experiment_name="non_monotonic_test", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target.execute)
        minimizer = DeltaDebugMinimizer(runner=runner)

        result = minimizer.minimize(initial_input)

        assert result.status == SearchStatus.COMPLETE
        assert result.subset_elements == ("A",)
        assert result.is_one_minimal is True
        assert result.verified_by_check is True

    def test_unconditional_failure_on_empty_set(self) -> None:
        """Verify that when target fails unconditionally (even on empty set), 1-minimality is False."""
        from benchmarks.synthetic.target import SyntheticTarget

        def always_fail_rule(elements: set[str]) -> tuple[bool, str, str | None]:
            return (True, "Unconditional failure", "ERR_ALWAYS")

        target = SyntheticTarget(rule_fn=always_fail_rule, name="AlwaysFailingTarget")
        initial_input = InputCase(
            case_id="always_fail",
            elements=["x", "y", "z"],
        )
        config = ExperimentConfig(experiment_name="test_always_fail", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target.execute)
        minimizer = DeltaDebugMinimizer(runner=runner)

        result = minimizer.minimize(initial_input)

        assert result.status == SearchStatus.COMPLETE
        assert result.subset_elements == ()
        assert result.is_one_minimal is False
        assert result.is_globally_minimal is False
        assert result.verified_by_check is False
        assert "fails unconditionally" in result.status_message.lower()
