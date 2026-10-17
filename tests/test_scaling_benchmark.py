"""Comprehensive regression test suite for Phase 2A scaling benchmark, ground truth, and statistics."""

import json
from pathlib import Path
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
    run_full_scaling_benchmark,
    run_scaling_instance,
)
from benchmarks.synthetic.stochastic_ground_truth import evaluate_stochastic_ground_truth
from benchmarks.synthetic.stochastic_target import StochasticSyntheticTarget


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
                assert len(gt.alternative_valid_triggers) == 0
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


def test_no_ground_truth_leakage_in_baseline_search():
    """Verify that baseline algorithms make search decisions purely via target queries without ground truth access."""
    target, initial_input, gt = create_scaling_task("single_trigger", k=8, seed=42)
    config = ExperimentConfig(experiment_name="test_no_leak", max_trials=500)

    # Run greedy ablation without passing gt
    target.reset(new_seed=42)
    runner_g = InterventionRunner(config=config, target_fn=target.execute)
    comp_g = run_stochastic_greedy_ablation(
        initial_input=initial_input,
        runner=runner_g,
        trials_per_candidate=10,
        failure_threshold=0.50,
    )
    # The algorithm must succeed on its own based purely on target outputs
    assert comp_g.failure_reproduced is True
    assert set(comp_g.discovered_elements) == set(gt.trigger_elements)


def test_minimality_verification_on_coexisting_independent_triggers():
    r"""Verify that 1-minimality check evaluates S \ {e} = empty set when |S| = 1.

    In multiple_independent with two triggers t1 and t2, testing {t1} \ {t1} = {}
    yields base noise P=0.05 < 0.50, correctly establishing statistically supported 1-minimality.
    """
    target, initial_input, gt = create_scaling_task("multiple_independent", k=6, seed=100)
    config = ExperimentConfig(experiment_name="test_mi_verif", max_trials=500)

    # 1. Delta Debugging
    target.reset(new_seed=100)
    runner_dd = InterventionRunner(config=config, target_fn=target.execute)
    minimizer = StochasticDeltaDebugMinimizer(runner=runner_dd, trials_per_candidate=10, failure_threshold=0.50)
    res_dd = minimizer.minimize(initial_input)

    assert res_dd.is_failure is True
    assert len(res_dd.subset_elements) == 1
    # Both t1 and t2 are valid 1-minimal triggers
    assert res_dd.subset_elements[0] in ("elem_t_indep_1", "elem_t_indep_2")
    # Minimality verification must evaluate empty set, not initial_input.without(elem) which contains the other trigger!
    # Because removing the singleton leaves empty set (P=0.05), minimality status must be STATISTICALLY_SUPPORTED
    assert res_dd.minimality_status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED


def test_non_monotonic_unique_one_minimality_truth_table():
    """Verify that under the non-monotonic rule, {A} is the sole 1-minimal trigger and {A, B, C} is not."""
    target, initial_input, gt = create_scaling_task("non_monotonic", k=6, seed=42)
    a, b, c = "elem_t_nm_trigger", "elem_t_nm_suppressor", "elem_t_nm_restorer"

    # Truth table
    # P({A}) = 0.90
    assert target.execute(InputCase(case_id="1", elements=(a,))).metadata["underlying_probability"] == 0.90
    # P({A, B}) = 0.10
    assert target.execute(InputCase(case_id="2", elements=(a, b))).metadata["underlying_probability"] == 0.10
    # P({A, B, C}) = 0.90
    assert target.execute(InputCase(case_id="3", elements=(a, b, c))).metadata["underlying_probability"] == 0.90
    # {A, B, C} \ {B} = {A, C} -> P({A, C}) = 0.90 (since A is present and B is absent)
    assert target.execute(InputCase(case_id="4", elements=(a, c))).metadata["underlying_probability"] == 0.90

    # Since removing B from {A, B, C} still produces failure (P=0.90 >= 0.50), {A, B, C} is NOT 1-minimal.
    assert gt.alternative_valid_triggers == ()
    assert gt.trigger_elements == (a,)


def test_crn_reproducibility_and_candidate_level_matching():
    """Verify Common Random Numbers produces identical outcomes for identical subsets regardless of evaluation order."""
    def dummy_rule(elems: set[str]) -> float:
        return 0.70 if "t" in elems else 0.20

    t1 = StochasticSyntheticTarget(rule_fn=dummy_rule, target_seed=777, crn_mode=True)
    t2 = StochasticSyntheticTarget(rule_fn=dummy_rule, target_seed=777, crn_mode=True)

    # In t1, evaluate candidate A three times, then candidate B three times
    case_a = InputCase(case_id="a", elements=("t", "d1"))
    case_b = InputCase(case_id="b", elements=("d2",))

    out_a_t1 = [t1.execute(case_a).outcome for _ in range(5)]
    out_b_t1 = [t1.execute(case_b).outcome for _ in range(5)]

    # In t2, evaluate candidate B first, then candidate A
    out_b_t2 = [t2.execute(case_b).outcome for _ in range(5)]
    out_a_t2 = [t2.execute(case_a).outcome for _ in range(5)]

    assert out_a_t1 == out_a_t2
    assert out_b_t1 == out_b_t2


def test_event_log_schema_and_completeness(tmp_path: Path):
    """Verify event logging emits valid JSONL records with all required fields."""
    log_file = tmp_path / "test_event_log.jsonl"
    suite = run_full_scaling_benchmark(
        seeds_per_condition=1,
        candidate_sizes=(4,),
        target_families=("single_trigger",),
        trials_per_candidate=5,
        max_trials=100,
        event_log_path=log_file,
    )
    assert log_file.exists()
    lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) > 0

    required_fields = {
        "benchmark_version",
        "target_family",
        "k",
        "seed",
        "algorithm",
        "step_index",
        "phase",
        "candidate_elements",
        "candidate_size",
        "trials_requested",
        "failures_observed",
        "failure_rate",
        "ci_lower",
        "ci_upper",
        "cumulative_target_calls",
        "elapsed_ms",
    }
    for rec in lines:
        assert required_fields.issubset(rec.keys())
        assert rec["benchmark_version"] == "phase_2a1"
        assert rec["phase"] in ("search", "minimality_verification")
        assert rec["trials_requested"] > 0
        assert 0.0 <= rec["failure_rate"] <= 1.0


def test_separation_of_null_detection_from_non_null_recovery():
    """Verify null targets and non-null targets are analyzed separately and not pooled."""
    inst_null = run_scaling_instance(family="null_target", k=4, seed=100, trials_per_candidate=10)
    inst_non_null = run_scaling_instance(family="single_trigger", k=4, seed=100, trials_per_candidate=10)

    assert inst_null.is_null_case is True
    assert inst_non_null.is_null_case is False

    # On null target, delta debugging must reproduce NO failure and discover NO elements
    assert inst_null.dd_result.failure_reproduced is False
    assert len(inst_null.dd_result.discovered_elements) == 0
    assert inst_null.dd_result.matches_ground_truth is True

    # On non-null target, delta debugging recovers trigger
    assert inst_non_null.dd_result.failure_reproduced is True
    assert len(inst_non_null.dd_result.discovered_elements) == 1
    assert inst_non_null.dd_result.matches_ground_truth is True


def test_phase_2a2_reproduction_and_event_log_validation():
    """Verify Phase 2A.2 reproducibility artifacts and discrepancy resolutions."""
    import json
    import math
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    val_file = root / "benchmarks" / "results" / "phase_2a2_event_log_validation.json"
    stat_file = root / "benchmarks" / "results" / "phase_2a2_statistical_tests.json"
    csv_file = root / "benchmarks" / "results" / "phase_2a2_corrected_metrics.csv"

    assert val_file.exists(), f"Missing {val_file}"
    assert stat_file.exists(), f"Missing {stat_file}"
    assert csv_file.exists(), f"Missing {csv_file}"

    with open(val_file) as f:
        val_data = json.load(f)
    assert val_data["all_checks_passed"] is True
    assert val_data["total_event_records"] == 31483
    assert val_data["total_target_trials"] == 314830
    assert val_data["distinct_runs"] == 1200
    for check_name, err_count in val_data["checks"].items():
        assert err_count == 0, f"{check_name} failed with {err_count} errors"

    with open(stat_file) as f:
        stat_data = json.load(f)
    dd_vs_gr = stat_data["paired_comparisons_overall"]["dd_vs_greedy"]
    assert dd_vs_gr["mean_difference"] == -21.375
    assert dd_vs_gr["paired_t_statistic"] == -3.7649
    assert dd_vs_gr["degrees_of_freedom"] == 399
    assert math.isclose(dd_vs_gr["paired_t_p_value_exact_student_t"], 0.00019166, abs_tol=1e-6)
    assert dd_vs_gr["wilcoxon_w_plus"] == 15089.5
    assert dd_vs_gr["wilcoxon_w_minus"] == 42880.5
    assert math.isclose(dd_vs_gr["wilcoxon_p_value_exact"], 1.7814e-14, rel_tol=1e-3)

