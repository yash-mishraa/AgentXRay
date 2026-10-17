"""Tests for Greedy Ablation and Seeded Random Search baselines."""

from __future__ import annotations

import pytest

from agentxray.baselines import run_greedy_ablation, run_seeded_random_search
from agentxray.models import ExperimentConfig, InputCase
from agentxray.runner import InterventionRunner
from benchmarks.synthetic.ground_truth import GroundTruthSpec, evaluate_ground_truth
from benchmarks.synthetic.target import SyntheticTarget
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
        )
        assert comp.failure_reproduced is True
        assert comp.minimality_check_passed is True
        assert comp.discovered_elements == ("param_mode_debug",)
        assert comp.matches_ground_truth is False  # isolated before post-hoc evaluation

        # Post-hoc ground truth evaluation
        matches, _, _, excess, _ = evaluate_ground_truth(ground_truth, comp)
        comp = comp.model_copy(
            update={
                "matches_ground_truth": matches,
                "irrelevant_elements_remaining": len(excess),
            }
        )
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
        )
        assert comp.failure_reproduced is True
        assert comp.minimality_check_passed is True
        assert set(comp.discovered_elements) == {"schema_type_strict", "flag_allow_null"}

        # Post-hoc evaluation
        matches, _, _, excess, _ = evaluate_ground_truth(ground_truth, comp)
        comp = comp.model_copy(
            update={
                "matches_ground_truth": matches,
                "irrelevant_elements_remaining": len(excess),
            }
        )
        assert comp.irrelevant_elements_remaining == 0
        assert comp.matches_ground_truth is True

    def test_greedy_ablation_null_task(self) -> None:
        target, initial_input, ground_truth = create_null_task()
        config = ExperimentConfig(experiment_name="test_greedy_null", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target.execute)

        comp = run_greedy_ablation(
            initial_input=initial_input,
            runner=runner,
        )
        assert comp.failure_reproduced is False
        assert comp.discovered_elements == ()

        matches, _, _, excess, _ = evaluate_ground_truth(ground_truth, comp)
        comp = comp.model_copy(
            update={
                "matches_ground_truth": matches,
                "irrelevant_elements_remaining": len(excess),
            }
        )
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
            seed=42,
        )
        comp2 = run_seeded_random_search(
            initial_input=initial_input2,
            runner=runner2,
            seed=42,
        )

        assert comp1.discovered_elements == comp2.discovered_elements
        assert comp1.target_calls == comp2.target_calls
        assert comp1.minimality_check_passed == comp2.minimality_check_passed

        # Post-hoc evaluation
        m1, _, _, e1, _ = evaluate_ground_truth(ground_truth1, comp1)
        m2, _, _, e2, _ = evaluate_ground_truth(ground_truth2, comp2)
        comp1 = comp1.model_copy(update={"matches_ground_truth": m1, "irrelevant_elements_remaining": len(e1)})
        comp2 = comp2.model_copy(update={"matches_ground_truth": m2, "irrelevant_elements_remaining": len(e2)})
        assert comp1.matches_ground_truth == comp2.matches_ground_truth

    def test_budget_exhaustion_in_random_search(self) -> None:
        """Verify that random search halts when max_trials is exhausted."""
        target, initial_input, ground_truth = create_conjunctive_task()
        config = ExperimentConfig(experiment_name="test_random_exhaust", max_trials=3, seed=42)
        runner = InterventionRunner(config=config, target_fn=target.execute)

        comp = run_seeded_random_search(
            initial_input=initial_input,
            runner=runner,
            seed=42,
        )

        assert runner.trial_count <= 3
        assert comp.budget_exhausted is True

    def test_seeded_random_without_ground_truth(self) -> None:
        """Verify seeded random search runs without ground truth passed."""
        target, initial_input, _ = create_single_factor_task()
        config = ExperimentConfig(experiment_name="test_no_gt_random", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target.execute)
        comp = run_seeded_random_search(initial_input=initial_input, runner=runner, seed=42)
        assert comp.failure_reproduced is True
        assert comp.discovered_elements == ("param_mode_debug",)
        assert comp.matches_ground_truth is False
        assert comp.irrelevant_elements_remaining == 0

    def test_greedy_ablation_without_ground_truth(self) -> None:
        """Verify greedy ablation runs without ground truth passed."""
        target, initial_input, _ = create_single_factor_task()
        config = ExperimentConfig(experiment_name="test_no_gt_greedy", max_trials=50)
        runner = InterventionRunner(config=config, target_fn=target.execute)
        comp = run_greedy_ablation(initial_input=initial_input, runner=runner)
        assert comp.failure_reproduced is True
        assert comp.discovered_elements == ("param_mode_debug",)
        assert comp.matches_ground_truth is False
        assert comp.irrelevant_elements_remaining == 0

    def test_random_search_early_stopping_on_singleton(self) -> None:
        """Verify random search stops early once a failing singleton is found, conserving budget."""
        target, initial_input, ground_truth = create_single_factor_task()
        # Large budget: 100 trials allowed
        config = ExperimentConfig(experiment_name="test_early_stop", max_trials=100)
        runner = InterventionRunner(config=config, target_fn=target.execute)

        comp = run_seeded_random_search(
            initial_input=initial_input,
            runner=runner,
            seed=42,
        )

        assert comp.failure_reproduced is True
        assert len(comp.discovered_elements) == 1
        assert comp.minimality_check_passed is True
        # Early stopping should terminate trial count far below max_trials (100)
        assert comp.target_calls < 20
        assert comp.budget_exhausted is False

    def test_missing_vs_null_ground_truth(self) -> None:
        """Verify missing ground truth never matches merely because both sets are empty."""
        target, initial_input, ground_truth = create_null_task()
        config = ExperimentConfig(experiment_name="test_null_vs_missing", max_trials=30)

        runner = InterventionRunner(config=config, target_fn=target.execute)
        comp = run_greedy_ablation(initial_input=initial_input, runner=runner)
        assert comp.failure_reproduced is False
        assert comp.discovered_elements == ()

        # 1. Genuine null task with is_null_case=True -> matches
        null_spec = GroundTruthSpec(
            task_id="null_1",
            name="Null Task",
            description="Null control",
            trigger_elements=(),
            is_null_case=True,
        )
        matches, _, _, _, _ = evaluate_ground_truth(null_spec, comp)
        assert matches is True

        # 2. Non-failing task with missing ground truth (None) -> does NOT match
        matches_none, _, _, _, _ = evaluate_ground_truth(None, comp)
        assert matches_none is False

        # 3. Non-failing task with empty true_trigger but is_null_case=False -> does NOT match
        fake_empty_spec = GroundTruthSpec(
            task_id="fake_1",
            name="Fake Spec",
            description="Empty without null flag",
            trigger_elements=(),
            is_null_case=False,
        )
        matches_fake, _, _, _, _ = evaluate_ground_truth(fake_empty_spec, comp)
        assert matches_fake is False

    def test_singleton_minimality_empty_set_succeeds(self) -> None:
        """Verify that when the empty set succeeds, singleton triggers pass 1-minimality verification."""
        target, initial_input, _ = create_single_factor_task()
        config = ExperimentConfig(experiment_name="test_single_min_pass", max_trials=50)

        runner_greedy = InterventionRunner(config=config, target_fn=target.execute)
        comp_greedy = run_greedy_ablation(initial_input=initial_input, runner=runner_greedy)
        assert comp_greedy.failure_reproduced is True
        assert comp_greedy.discovered_elements == ("param_mode_debug",)
        assert comp_greedy.minimality_check_passed is True

        runner_rand = InterventionRunner(config=config, target_fn=target.execute)
        comp_rand = run_seeded_random_search(initial_input=initial_input, runner=runner_rand, seed=42)
        assert comp_rand.failure_reproduced is True
        assert comp_rand.discovered_elements == ("param_mode_debug",)
        assert comp_rand.minimality_check_passed is True

    def test_singleton_minimality_empty_set_fails(self) -> None:
        """Verify that when the empty set reproduces failure (unconditional failure), 1-minimality check fails."""
        def unconditional_failure_rule(elements: set[str]) -> tuple[bool, str, str | None]:
            # Always fails, even when elements is empty set!
            return (True, "Unconditional failure regardless of elements", "ERR_ALWAYS")

        target = SyntheticTarget(rule_fn=unconditional_failure_rule, name="AlwaysFailingTarget")
        initial_input = InputCase(
            case_id="always_fail_case",
            elements=["feature_a", "feature_b", "feature_c"],
        )
        config = ExperimentConfig(experiment_name="test_single_min_fail", max_trials=50)

        # 1. Greedy Ablation: reduces to singleton, but in Step 3 empty set fails -> minimality_check_passed=False
        runner_greedy = InterventionRunner(config=config, target_fn=target.execute)
        comp_greedy = run_greedy_ablation(initial_input=initial_input, runner=runner_greedy)
        assert comp_greedy.failure_reproduced is True
        assert comp_greedy.minimality_check_passed is False

        # 2. Seeded Random Search: finds singleton, but in Step 3 empty set fails -> minimality_check_passed=False
        runner_rand = InterventionRunner(config=config, target_fn=target.execute)
        comp_rand = run_seeded_random_search(initial_input=initial_input, runner=runner_rand, seed=42)
        assert comp_rand.failure_reproduced is True
        assert comp_rand.minimality_check_passed is False
