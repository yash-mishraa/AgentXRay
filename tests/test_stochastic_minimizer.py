"""Tests for the StochasticDeltaDebugMinimizer across all benchmark scenarios."""

from __future__ import annotations

import pytest

from agentxray.models import (
    ExperimentConfig,
    SearchStatus,
    StochasticMinimalityStatus,
)
from agentxray.runner import InterventionRunner
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.synthetic.stochastic_tasks import (
    create_stochastic_conjunctive_task,
    create_stochastic_distractor_task,
    create_stochastic_non_monotonic_task,
    create_stochastic_null_task,
    create_stochastic_single_factor_task,
    create_stochastic_weak_signal_task,
)


def test_stochastic_minimizer_single_factor():
    """Verify stochastic minimizer isolates single-factor trigger and establishes minimality."""
    target, initial_input, gt = create_stochastic_single_factor_task(target_seed=42)
    config = ExperimentConfig(experiment_name="test_stoch_sf", max_trials=200)
    runner = InterventionRunner(config=config, target_fn=target.execute)

    minimizer = StochasticDeltaDebugMinimizer(
        runner=runner,
        trials_per_candidate=20,
        failure_threshold=0.30,
    )
    result = minimizer.minimize(initial_input)

    assert result.status == SearchStatus.COMPLETE
    assert result.is_failure is True
    assert set(result.subset_elements) == {"param_mode_debug"}
    assert result.minimality_status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
    assert result.candidate_evaluation.failure_rate >= 0.70


def test_stochastic_minimizer_conjunctive():
    """Verify stochastic minimizer isolates two-factor conjunctive trigger with removal CIs."""
    target, initial_input, gt = create_stochastic_conjunctive_task(target_seed=42)
    config = ExperimentConfig(experiment_name="test_stoch_conj", max_trials=300)
    runner = InterventionRunner(config=config, target_fn=target.execute)

    minimizer = StochasticDeltaDebugMinimizer(
        runner=runner,
        trials_per_candidate=20,
        failure_threshold=0.30,
    )
    result = minimizer.minimize(initial_input)

    assert result.status == SearchStatus.COMPLETE
    assert result.is_failure is True
    assert set(result.subset_elements) == {"schema_type_strict", "flag_allow_null"}
    assert result.minimality_status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED

    # Verify removal evaluations exist for both conjunctive factors
    assert "schema_type_strict" in result.removal_evaluations
    assert "flag_allow_null" in result.removal_evaluations
    assert result.removal_evaluations["schema_type_strict"].failure_rate < 0.20
    assert result.removal_evaluations["flag_allow_null"].failure_rate < 0.20


def test_stochastic_minimizer_distractor_heavy():
    """Verify stochastic minimizer strips 4 distractors in probabilistic environment."""
    target, initial_input, gt = create_stochastic_distractor_task(target_seed=42)
    config = ExperimentConfig(experiment_name="test_stoch_dist", max_trials=300)
    runner = InterventionRunner(config=config, target_fn=target.execute)

    minimizer = StochasticDeltaDebugMinimizer(
        runner=runner,
        trials_per_candidate=20,
        failure_threshold=0.30,
    )
    result = minimizer.minimize(initial_input)

    assert result.status == SearchStatus.COMPLETE
    assert set(result.subset_elements) == {"token_auth_bearer"}
    assert result.minimality_status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED


def test_stochastic_minimizer_null_case():
    """Verify stochastic minimizer detects null case without false positive trigger."""
    target, initial_input, gt = create_stochastic_null_task(target_seed=42)
    config = ExperimentConfig(experiment_name="test_stoch_null", max_trials=100)
    runner = InterventionRunner(config=config, target_fn=target.execute)

    minimizer = StochasticDeltaDebugMinimizer(
        runner=runner,
        trials_per_candidate=20,
        failure_threshold=0.30,
    )
    result = minimizer.minimize(initial_input)

    assert result.status == SearchStatus.NO_TRIGGER_FOUND
    assert result.is_failure is False
    assert result.subset_elements == ()
    assert result.minimality_status == StochasticMinimalityStatus.NOT_APPLICABLE


def test_stochastic_minimizer_non_monotonic():
    """Verify stochastic minimizer handles non-monotonic suppression and restoration."""
    target, initial_input, gt = create_stochastic_non_monotonic_task(target_seed=42)
    config = ExperimentConfig(experiment_name="test_stoch_nonmono", max_trials=300)
    runner = InterventionRunner(config=config, target_fn=target.execute)

    minimizer = StochasticDeltaDebugMinimizer(
        runner=runner,
        trials_per_candidate=20,
        failure_threshold=0.30,
    )
    result = minimizer.minimize(initial_input)

    assert result.status == SearchStatus.COMPLETE
    assert result.is_failure is True
    # Can find feature_flag_a (the globally minimal 1-element trigger)
    assert "feature_flag_a" in result.subset_elements


def test_stochastic_minimizer_weak_signal():
    """Verify stochastic minimizer detects modest effect size in weak-signal scenario.

    With n=20 per condition on seed 44, the effect is observed (7/20 vs 3/20) but two-sided p >= 0.05,
    so minimality status is accurately classified as OBSERVED_ONLY, not statistically supported.
    """
    target, initial_input, gt = create_stochastic_weak_signal_task(target_seed=44)
    config = ExperimentConfig(experiment_name="test_stoch_weak", max_trials=300)
    runner = InterventionRunner(config=config, target_fn=target.execute)

    minimizer = StochasticDeltaDebugMinimizer(
        runner=runner,
        trials_per_candidate=20,
        failure_threshold=0.25,
    )
    result = minimizer.minimize(initial_input)

    assert result.status == SearchStatus.COMPLETE
    assert result.is_failure is True
    assert set(result.subset_elements) == {"weak_trigger_prompt"}
    assert result.minimality_status == StochasticMinimalityStatus.OBSERVED_ONLY


def test_stochastic_minimizer_weak_signal_adequate_sample_size():
    """Verify that with adequate sample size (n=100), weak signal reaches statistical support."""
    target, initial_input, gt = create_stochastic_weak_signal_task(target_seed=42)
    config = ExperimentConfig(experiment_name="test_stoch_weak_powered", max_trials=500)
    runner = InterventionRunner(config=config, target_fn=target.execute)

    minimizer = StochasticDeltaDebugMinimizer(
        runner=runner,
        trials_per_candidate=100,
        failure_threshold=0.25,
    )
    result = minimizer.minimize(initial_input)

    assert result.status == SearchStatus.COMPLETE
    assert set(result.subset_elements) == {"weak_trigger_prompt"}
    assert result.minimality_status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED


def test_stochastic_minimizer_budget_exhaustion():
    """Verify minimizer terminates gracefully and reports INCONCLUSIVE on budget limit."""
    target, initial_input, gt = create_stochastic_conjunctive_task(target_seed=42)
    # Give budget insufficient to complete minimization (e.g. only 25 trials for 20-trial subsets)
    config = ExperimentConfig(experiment_name="test_budget", max_trials=25)
    runner = InterventionRunner(config=config, target_fn=target.execute)

    minimizer = StochasticDeltaDebugMinimizer(
        runner=runner,
        trials_per_candidate=20,
        failure_threshold=0.30,
    )
    result = minimizer.minimize(initial_input)

    assert result.status == SearchStatus.BUDGET_EXHAUSTED
    assert result.minimality_status == StochasticMinimalityStatus.INCONCLUSIVE
    assert runner.is_budget_exhausted() is True
