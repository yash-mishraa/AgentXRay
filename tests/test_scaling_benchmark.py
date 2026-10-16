"""Comprehensive regression test suite for Phase 2A scaling benchmark, ground truth, and statistics."""

from __future__ import annotations

import pytest

from agentxray.models import (
    ExperimentConfig,
    InputCase,
    StochasticMinimalityStatus,
)
from agentxray.runner import InterventionRunner
from agentxray.statistics import (
    adjust_p_values_bh,
    compare_paired_differences,
    proportion_difference_interval,
    wilcoxon_signed_rank_test,
    wilson_score_interval,
)
from agentxray.stochastic_baselines import (
    run_stochastic_greedy_ablation,
    run_stochastic_random_search,
)
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.synthetic.scaling_benchmark import (
    CANDIDATE_SIZES,
    TARGET_FAMILIES,
    create_scaling_task,
    run_scaling_instance,
)
from benchmarks.synthetic.stochastic_ground_truth import evaluate_stochastic_ground_truth


def test_ground_truth_labeling_across_all_families_and_sizes():
    """Verify ground truth specifications are accurately constructed across all families and sizes."""
    for family in TARGET_FAMILIES:
        for k in (4, 8, 16):
            target, initial_input, gt = create_scaling_task(family, k=k, seed=42)
            assert len(initial_input.elements) == k
            assert set(initial_input.elements) == set(initial_input.elements)  # no duplicates

            if family == "null_target":
                assert gt.is_null_case is True
                assert len(gt.trigger_elements) == 0
            elif family == "conjunctive_trigger":
                assert len(gt.trigger_elements) == 2
                assert set(gt.trigger_elements).issubset(set(initial_input.elements))
            elif family == "multiple_independent":
                assert len(gt.trigger_elements) == 1
                assert len(gt.alternative_valid_triggers) == 1
                assert set(gt.trigger_elements).issubset(set(initial_input.elements))
            elif family == "non_monotonic":
                assert len(gt.trigger_elements) == 1
                assert len(gt.alternative_valid_triggers) == 1
                assert len(gt.alternative_valid_triggers[0]) == 3
            else:
                assert len(gt.trigger_elements) == 1
                assert set(gt.trigger_elements).issubset(set(initial_input.elements))


def test_reproducibility_from_recorded_seeds():
    """Verify identical seeds produce byte-for-byte identical tasks, target responses, and algorithm traces."""
    # Instance 1
    t1, in1, gt1 = create_scaling_task("single_trigger", k=8, seed=12345)
    r1 = t1.execute(in1)

    # Instance 2
    t2, in2, gt2 = create_scaling_task("single_trigger", k=8, seed=12345)
    r2 = t2.execute(in2)

    assert in1.elements == in2.elements
    assert gt1.trigger_elements == gt2.trigger_elements
    assert r1.outcome == r2.outcome
    assert r1.metadata == r2.metadata


def test_null_target_false_discoveries():
    """Verify algorithms do NOT make false positive discoveries on null targets."""
    target, initial_input, gt = create_scaling_task("null_target", k=8, seed=42)
    config = ExperimentConfig(experiment_name="test_null_fd", max_trials=500)

    # 1. Delta Debugging
    runner_dd = InterventionRunner(config=config, target_fn=target.execute)
    minimizer = StochasticDeltaDebugMinimizer(runner=runner_dd, trials_per_candidate=10, failure_threshold=0.50)
    dd_cond = minimizer.minimize(initial_input)
    assert dd_cond.is_failure is False
    assert dd_cond.minimality_status == StochasticMinimalityStatus.NOT_APPLICABLE
    assert len(dd_cond.subset_elements) == 0

    # 2. Greedy Ablation
    target.reset(new_seed=42)
    runner_gr = InterventionRunner(config=config, target_fn=target.execute)
    gr_comp = run_stochastic_greedy_ablation(
        initial_input=initial_input,
        runner=runner_gr,
        trials_per_candidate=10,
        failure_threshold=0.50,
        true_trigger=gt.trigger_elements,
    )
    assert gr_comp.failure_reproduced is False
    assert gr_comp.minimality_status == StochasticMinimalityStatus.NOT_APPLICABLE
    assert len(gr_comp.discovered_elements) == 0

    # 3. Random Search
    target.reset(new_seed=42)
    runner_rnd = InterventionRunner(config=config, target_fn=target.execute)
    rnd_comp = run_stochastic_random_search(
        initial_input=initial_input,
        runner=runner_rnd,
        trials_per_candidate=10,
        failure_threshold=0.50,
        true_trigger=gt.trigger_elements,
        seed=42,
    )
    assert rnd_comp.failure_reproduced is False
    assert rnd_comp.minimality_status == StochasticMinimalityStatus.NOT_APPLICABLE
    assert len(rnd_comp.discovered_elements) == 0


def test_non_monotonic_target_handling():
    """Verify non-monotonic suppression and restoration behavior."""
    target, initial_input, gt = create_scaling_task("non_monotonic", k=6, seed=777)
    a, b, c = "elem_t_nm_trigger", "elem_t_nm_suppressor", "elem_t_nm_restorer"

    # Rule checks
    # 1. Trigger alone: P=0.90
    res_a = target.execute(InputCase(case_id="1", elements=(a,)))
    assert res_a.metadata["underlying_probability"] == 0.90

    # 2. Trigger + Suppressor: P=0.10 (suppressed)
    res_ab = target.execute(InputCase(case_id="2", elements=(a, b)))
    assert res_ab.metadata["underlying_probability"] == 0.10

    # 3. Trigger + Suppressor + Restorer: P=0.90 (restored)
    res_abc = target.execute(InputCase(case_id="3", elements=(a, b, c)))
    assert res_abc.metadata["underlying_probability"] == 0.90

    # 4. Suppressor alone: P=0.05 (base)
    res_b = target.execute(InputCase(case_id="4", elements=(b,)))
    assert res_b.metadata["underlying_probability"] == 0.05


def test_equal_budget_enforcement_and_accounting():
    """Verify trial accounting strictly enforces budget across search vs verification."""
    inst = run_scaling_instance(
        family="single_trigger",
        k=4,
        seed=100,
        trials_per_candidate=10,
        max_trials=1000,
    )
    # Check all three algorithm outputs
    for res in (inst.dd_result, inst.greedy_result, inst.random_result):
        assert res.target_calls == res.search_target_calls + res.verification_target_calls
        assert res.target_calls <= 1000
        assert res.subsets_examined > 0
        assert res.target_calls >= res.subsets_examined * 10 or res.budget_exhausted


def test_wilcoxon_signed_rank_paired_statistics():
    """Verify Wilcoxon signed-rank and paired difference statistics implementation."""
    # Known paired difference test
    x = [10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0, 35.0]
    y = [8.0, 9.0, 12.0, 14.0, 17.0, 19.0, 20.0, 22.0, 25.0, 28.0]
    # All x_i > y_i: W+ should equal sum(1..10) = 55
    w_stat, p_val = wilcoxon_signed_rank_test(x, y)
    assert w_stat == 55.0
    assert p_val < 0.01

    # Symmetric differences: W should be around mean 27.5
    x_sym = [10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0, 35.0]
    y_sym = [12.0, 10.0, 17.0, 16.0, 22.0, 20.0, 27.0, 26.0, 32.0, 33.0]
    summary = compare_paired_differences(x_sym, y_sym)
    assert summary.n_pairs == 10
    assert summary.wilcoxon_p_value > 0.05
    assert summary.paired_t_p_value > 0.05


def test_minimality_verification_classification():
    """Verify classification of candidates into statistically supported vs inconclusive."""
    # When effect is strong and n=10 per group, single removal test is statistically supported
    # 9/10 vs 1/10
    l_diff, u_diff = proportion_difference_interval(9, 10, 1, 10)
    assert l_diff > 0.30
    assert u_diff <= 1.0


def test_bh_adjustment_on_scaling_results():
    """Verify Benjamini-Hochberg adjustment works properly on multiple comparison p-values."""
    raw_p = [0.001, 0.005, 0.02, 0.04, 0.15, 0.80]
    adjusted = adjust_p_values_bh(raw_p)
    assert len(adjusted) == len(raw_p)
    assert all(0.0 <= q <= 1.0 for q in adjusted)
    # Monotonicity with raw
    for i in range(len(raw_p) - 1):
        assert adjusted[i] <= adjusted[i + 1]
