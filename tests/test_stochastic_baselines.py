"""Tests for stochastic baseline algorithms and fair budget enforcement."""

from __future__ import annotations

import pytest

from agentxray.models import ExperimentConfig, StochasticMinimalityStatus
from agentxray.runner import InterventionRunner
from agentxray.stochastic_baselines import (
    run_stochastic_greedy_ablation,
    run_stochastic_random_search,
)
from benchmarks.synthetic.stochastic_tasks import (
    create_stochastic_conjunctive_task,
    create_stochastic_single_factor_task,
)


def test_stochastic_greedy_ablation_success():
    """Verify stochastic greedy ablation successfully isolates trigger."""
    target, initial_input, gt = create_stochastic_single_factor_task(target_seed=42)
    config = ExperimentConfig(experiment_name="test_stoch_greedy", max_trials=200)
    runner = InterventionRunner(config=config, target_fn=target.execute)

    comp = run_stochastic_greedy_ablation(
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=20,
        failure_threshold=0.30,
        true_trigger=gt.trigger_elements,
    )

    assert comp.failure_reproduced is True
    assert set(comp.discovered_elements) == {"param_mode_debug"}
    assert comp.matches_ground_truth is True
    assert comp.target_calls <= config.max_trials


def test_stochastic_greedy_ablation_budget_cap():
    """Verify stochastic greedy ablation honors trial budget cap."""
    target, initial_input, gt = create_stochastic_conjunctive_task(target_seed=42)
    config = ExperimentConfig(experiment_name="test_greedy_cap", max_trials=50)
    runner = InterventionRunner(config=config, target_fn=target.execute)

    comp = run_stochastic_greedy_ablation(
        initial_input=initial_input,
        runner=runner,
        trials_per_candidate=20,
        failure_threshold=0.30,
        true_trigger=gt.trigger_elements,
    )

    assert comp.target_calls <= 50
    assert comp.budget_exhausted is True
    assert comp.minimality_status == StochasticMinimalityStatus.INCONCLUSIVE


def test_stochastic_random_search_reproducible_and_capped():
    """Verify seeded random search is repeatable and strictly capped at max_trials."""
    target, initial_input, gt = create_stochastic_single_factor_task(target_seed=42)
    config = ExperimentConfig(experiment_name="test_random_cap", max_trials=100)

    # Run 1
    target.reset(new_seed=42)
    runner1 = InterventionRunner(config=config, target_fn=target.execute)
    comp1 = run_stochastic_random_search(
        initial_input=initial_input,
        runner=runner1,
        trials_per_candidate=20,
        failure_threshold=0.30,
        true_trigger=gt.trigger_elements,
        seed=999,
    )

    # Run 2 with same seed
    target.reset(new_seed=42)
    runner2 = InterventionRunner(config=config, target_fn=target.execute)
    comp2 = run_stochastic_random_search(
        initial_input=initial_input,
        runner=runner2,
        trials_per_candidate=20,
        failure_threshold=0.30,
        true_trigger=gt.trigger_elements,
        seed=999,
    )

    assert comp1.target_calls <= 100
    assert comp2.target_calls <= 100
    assert comp1.discovered_elements == comp2.discovered_elements
    assert comp1.target_calls == comp2.target_calls


def test_stochastic_baselines_null_case():
    """Verify stochastic baselines correctly report NOT_APPLICABLE and no failure on null case."""
    from benchmarks.synthetic.stochastic_tasks import create_stochastic_null_task

    target, initial_input, gt = create_stochastic_null_task(target_seed=42)
    config = ExperimentConfig(experiment_name="test_baselines_null", max_trials=100)

    # Greedy Ablation on Null Case
    target.reset(new_seed=42)
    runner_g = InterventionRunner(config=config, target_fn=target.execute)
    comp_g = run_stochastic_greedy_ablation(
        initial_input=initial_input,
        runner=runner_g,
        trials_per_candidate=20,
        failure_threshold=0.30,
        true_trigger=gt.trigger_elements,
    )
    assert comp_g.failure_reproduced is False
    assert comp_g.discovered_elements == ()
    assert comp_g.minimality_status == StochasticMinimalityStatus.NOT_APPLICABLE
    assert comp_g.matches_ground_truth is True

    # Random Search on Null Case
    target.reset(new_seed=42)
    runner_r = InterventionRunner(config=config, target_fn=target.execute)
    comp_r = run_stochastic_random_search(
        initial_input=initial_input,
        runner=runner_r,
        trials_per_candidate=20,
        failure_threshold=0.30,
        true_trigger=gt.trigger_elements,
        seed=42,
    )
    assert comp_r.failure_reproduced is False
    assert comp_r.discovered_elements == ()
    assert comp_r.minimality_status == StochasticMinimalityStatus.NOT_APPLICABLE
    assert comp_r.matches_ground_truth is True

