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
    )

    assert comp.failure_reproduced is True
    assert set(comp.discovered_elements) == {"param_mode_debug"}
    assert set(comp.discovered_elements) == set(gt.trigger_elements)
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
        seed=999,
    )

    assert comp1.target_calls <= 100
    assert comp2.target_calls <= 100
    assert comp1.discovered_elements == comp2.discovered_elements
    assert comp1.target_calls == comp2.target_calls


def test_stochastic_baselines_null_case():
    """Verify stochastic baselines correctly report NOT_APPLICABLE and no failure on null case."""
    from agentxray.models import SearchStatus, StochasticDiscoveredCondition
    from benchmarks.synthetic.stochastic_ground_truth import evaluate_stochastic_ground_truth
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
    )
    assert comp_g.failure_reproduced is False
    assert comp_g.discovered_elements == ()
    assert comp_g.minimality_status == StochasticMinimalityStatus.NOT_APPLICABLE

    dummy_g = StochasticDiscoveredCondition(
        subset_elements=comp_g.discovered_elements,
        status=SearchStatus.COMPLETE,
        is_failure=comp_g.failure_reproduced,
        minimality_status=comp_g.minimality_status,
        candidate_evaluation=comp_g.candidate_evaluation,
        target_calls_used=comp_g.target_calls,
    )
    matches_g, _, _, _, _ = evaluate_stochastic_ground_truth(gt, dummy_g)
    assert matches_g is True

    # Random Search on Null Case
    target.reset(new_seed=42)
    runner_r = InterventionRunner(config=config, target_fn=target.execute)
    comp_r = run_stochastic_random_search(
        initial_input=initial_input,
        runner=runner_r,
        trials_per_candidate=20,
        failure_threshold=0.30,
        seed=42,
    )
    assert comp_r.failure_reproduced is False
    assert comp_r.discovered_elements == ()
    assert comp_r.minimality_status == StochasticMinimalityStatus.NOT_APPLICABLE

    dummy_r = StochasticDiscoveredCondition(
        subset_elements=comp_r.discovered_elements,
        status=SearchStatus.COMPLETE,
        is_failure=comp_r.failure_reproduced,
        minimality_status=comp_r.minimality_status,
        candidate_evaluation=comp_r.candidate_evaluation,
        target_calls_used=comp_r.target_calls,
    )
    matches_r, _, _, _, _ = evaluate_stochastic_ground_truth(gt, dummy_r)
    assert matches_r is True


def test_stochastic_baselines_strict_search_isolation_and_post_hoc_evaluation():
    """Verify stochastic baselines maintain strict search isolation and support post-hoc evaluation."""
    from benchmarks.synthetic.stochastic_ground_truth import (
        StochasticGroundTruthSpec,
        evaluate_stochastic_ground_truth,
    )

    target, initial_input, gt = create_stochastic_single_factor_task(target_seed=42)
    config = ExperimentConfig(experiment_name="test_gt_match", max_trials=200)

    # 1. Greedy Ablation: returns default False/0 (no ground-truth leakage)
    runner_g = InterventionRunner(config=config, target_fn=target.execute)
    comp_g = run_stochastic_greedy_ablation(
        initial_input=initial_input,
        runner=runner_g,
        trials_per_candidate=20,
        failure_threshold=0.30,
    )
    assert comp_g.failure_reproduced is True
    assert comp_g.matches_ground_truth is False
    assert comp_g.irrelevant_elements_remaining == 0

    # Post-hoc evaluation attaches correct metrics
    g_match, prec, rec, excess, missing = evaluate_stochastic_ground_truth(gt, comp_g)
    assert g_match is True
    assert prec == 1.0
    assert rec == 1.0
    assert excess == ()
    assert missing == ()

    comp_g_updated = comp_g.model_copy(update={
        "matches_ground_truth": g_match,
        "irrelevant_elements_remaining": len(excess),
    })
    assert comp_g_updated.matches_ground_truth is True
    assert comp_g_updated.irrelevant_elements_remaining == 0
    # Search trace and discovered condition remain unmodified
    assert comp_g_updated.discovered_elements == comp_g.discovered_elements
    assert comp_g_updated.target_calls == comp_g.target_calls

    # 2. Post-hoc evaluation with mismatched ground-truth
    mismatched_gt = StochasticGroundTruthSpec(
        task_id="mismatch",
        name="Mismatch",
        description="",
        trigger_elements=("wrong_element",),
        base_probability=0.0,
        trigger_probability=0.9,
    )
    m_match, _, _, m_excess, _ = evaluate_stochastic_ground_truth(mismatched_gt, comp_g)
    assert m_match is False
    assert len(m_excess) == 1

    # 3. Post-hoc evaluation with alternative trigger match
    alt_gt = StochasticGroundTruthSpec(
        task_id="alt",
        name="Alt",
        description="",
        trigger_elements=("primary_mismatch",),
        base_probability=0.0,
        trigger_probability=0.9,
        alternative_valid_triggers=(gt.trigger_elements,),
    )
    alt_match, _, _, alt_excess, _ = evaluate_stochastic_ground_truth(alt_gt, comp_g)
    assert alt_match is True
    assert len(alt_excess) == 0

    # 4. Seeded Random Search: returns default False/0
    target.reset(new_seed=42)
    runner_r = InterventionRunner(config=config, target_fn=target.execute)
    comp_r = run_stochastic_random_search(
        initial_input=initial_input,
        runner=runner_r,
        trials_per_candidate=20,
        failure_threshold=0.30,
        seed=1,
    )
    assert comp_r.matches_ground_truth is False
    assert comp_r.irrelevant_elements_remaining == 0

    r_match, _, _, r_excess, _ = evaluate_stochastic_ground_truth(gt, comp_r)
    assert r_match is True
    assert len(r_excess) == 0


def test_stochastic_baselines_search_trace_invariance():
    """Verify stochastic baseline search traces are completely invariant to post-hoc evaluation."""
    from benchmarks.synthetic.stochastic_ground_truth import (
        StochasticGroundTruthSpec,
        evaluate_stochastic_ground_truth,
    )

    target, initial_input, gt = create_stochastic_single_factor_task(target_seed=42)
    config = ExperimentConfig(experiment_name="test_invariance", max_trials=100)

    # Run Greedy Ablation
    runner_g = InterventionRunner(config=config, target_fn=target.execute)
    comp_g = run_stochastic_greedy_ablation(
        initial_input=initial_input,
        runner=runner_g,
        trials_per_candidate=20,
        failure_threshold=0.30,
    )
    trace_before = [(t.intervention_type, t.input_case.elements, t.result.outcome) for t in runner_g.trials]

    # Evaluate multiple different ground-truth hypotheses post-hoc
    gt1 = gt
    gt2 = StochasticGroundTruthSpec(
        task_id="alt", name="Alt", description="", trigger_elements=("param_mode_debug",),
        base_probability=0.0, trigger_probability=0.8
    )
    gt3 = StochasticGroundTruthSpec(
        task_id="wrong", name="Wrong", description="", trigger_elements=("distractor_0",),
        base_probability=0.0, trigger_probability=0.8
    )

    m1, _, _, _, _ = evaluate_stochastic_ground_truth(gt1, comp_g)
    m2, _, _, _, _ = evaluate_stochastic_ground_truth(gt2, comp_g)
    m3, _, _, _, _ = evaluate_stochastic_ground_truth(gt3, comp_g)

    assert m1 is True
    assert m2 is True
    assert m3 is False

    trace_after = [(t.intervention_type, t.input_case.elements, t.result.outcome) for t in runner_g.trials]
    assert trace_before == trace_after
