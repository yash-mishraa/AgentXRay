"""Tests for Greedy Ablation and Seeded Random Search baselines."""

from __future__ import annotations

import pytest

from agentxray.baselines import run_greedy_ablation, run_seeded_random_search
from agentxray.models import ExperimentConfig, InputCase
from agentxray.runner import InterventionRunner
from benchmarks.synthetic.tasks import (
    create_conjunctive_task,
    create_distractor_task,
    create_null_task,
    create_single_factor_task,
)


class TestGreedyAblationBaseline:
    """Tests for Greedy Single-Element Ablation."""

    def test_greedy_ablation_single_factor(self) -> None:
        target, initial_input, ground_truth = create_single_factor_task()
        config = ExperimentConfig(experiment_name="test_greedy_single", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target.execute)

        comp = run_greedy_ablation(
            initial_input=initial_input,
            runner=runner,
            true_trigger=ground_truth.trigger_elements,
        )

        assert comp.failure_reproduced is True
        assert comp.minimality_check_passed is True
        assert comp.discovered_elements == ("param_mode_debug",)
        assert comp.irrelevant_elements_remaining == 0
        assert comp.matches_ground_truth is True
        assert comp.budget_exhausted is False

    def test_greedy_ablation_conjunctive(self) -> None:
        target, initial_input, ground_truth = create_conjunctive_task()
        config = ExperimentConfig(experiment_name="test_greedy_conj", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target.execute)

        comp = run_greedy_ablation(
            initial_input=initial_input,
            runner=runner,
            true_trigger=ground_truth.trigger_elements,
        )

        assert comp.failure_reproduced is True
        assert comp.minimality_check_passed is True
        assert set(comp.discovered_elements) == {"schema_type_strict", "flag_allow_null"}
        assert comp.irrelevant_elements_remaining == 0
        assert comp.matches_ground_truth is True

    def test_greedy_ablation_null_task(self) -> None:
        target, initial_input, ground_truth = create_null_task()
        config = ExperimentConfig(experiment_name="test_greedy_null", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target.execute)

        comp = run_greedy_ablation(
            initial_input=initial_input,
            runner=runner,
            true_trigger=ground_truth.trigger_elements,
        )

        assert comp.failure_reproduced is False
        assert comp.discovered_elements == ()
        assert comp.matches_ground_truth is True


class TestSeededRandomSearchBaseline:
    """Tests for Seeded Random Candidate Search."""

    def test_seeded_random_reproducibility(self) -> None:
        """Verify that identical seeds produce identical trial sequences and results."""
        target1, initial_input1, ground_truth1 = create_distractor_task()
        target2, initial_input2, ground_truth2 = create_distractor_task()

        config = ExperimentConfig(experiment_name="test_random_seed", max_trials=30, seed=42)

        runner1 = InterventionRunner(config=config, target_fn=target1.execute)
        runner2 = InterventionRunner(config=config, target_fn=target2.execute)

        comp1 = run_seeded_random_search(
            initial_input=initial_input1,
            runner=runner1,
            true_trigger=ground_truth1.trigger_elements,
            seed=42,
        )
        comp2 = run_seeded_random_search(
            initial_input=initial_input2,
            runner=runner2,
            true_trigger=ground_truth2.trigger_elements,
            seed=42,
        )

        assert comp1.discovered_elements == comp2.discovered_elements
        assert comp1.target_calls == comp2.target_calls
        assert comp1.matches_ground_truth == comp2.matches_ground_truth

    def test_budget_exhaustion_in_random_search(self) -> None:
        """Verify that random search halts when max_trials is exhausted."""
        target, initial_input, ground_truth = create_distractor_task()
        config = ExperimentConfig(experiment_name="test_random_exhaust", max_trials=5, seed=42)
        runner = InterventionRunner(config=config, target_fn=target.execute)

        comp = run_seeded_random_search(
            initial_input=initial_input,
            runner=runner,
            true_trigger=ground_truth.trigger_elements,
            seed=42,
        )

        assert runner.trial_count <= 5
        assert comp.budget_exhausted is True

    def test_seeded_random_without_ground_truth(self) -> None:
        """Verify seeded random search runs without ground truth passed."""
        target, initial_input, _ = create_single_factor_task()
        config = ExperimentConfig(experiment_name="test_no_gt_random", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target.execute)
        comp = run_seeded_random_search(initial_input=initial_input, runner=runner, seed=42)
        assert comp.failure_reproduced is True
        assert comp.discovered_elements == ("param_mode_debug",)

    def test_greedy_ablation_without_ground_truth(self) -> None:
        """Verify greedy ablation runs without ground truth passed."""
        target, initial_input, _ = create_single_factor_task()
        config = ExperimentConfig(experiment_name="test_no_gt_greedy", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target.execute)
        comp = run_greedy_ablation(initial_input=initial_input, runner=runner)
        assert comp.failure_reproduced is True
        assert comp.discovered_elements == ("param_mode_debug",)

