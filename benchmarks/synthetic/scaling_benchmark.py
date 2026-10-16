"""Phase 2A: Algorithmic Scaling Benchmark Engine.

Implements controlled synthetic benchmark generation, multi-instance evaluation,
paired statistical analysis, and reproducible serialization for:
- Candidate-space sizes: K in {4, 8, 12, 16, 20}
- Target families:
  1. single_trigger
  2. conjunctive_trigger
  3. multiple_independent
  4. distractor_heavy
  5. null_target
  6. non_monotonic
  7. stochastic_target
  8. weak_signal
- Algorithms:
  - Stochastic Delta Debugging
  - Stochastic Greedy Ablation
  - Stochastic Seeded Random Search
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from agentxray.models import (
    ExperimentConfig,
    InputCase,
    StochasticDiscoveredCondition,
    StochasticMinimalityStatus,
)
from agentxray.runner import InterventionRunner
from agentxray.statistics import (
    adjust_p_values_bh,
    compare_paired_differences,
    wilcoxon_signed_rank_test,
)
from agentxray.stochastic_baselines import (
    run_stochastic_greedy_ablation,
    run_stochastic_random_search,
)
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.synthetic.stochastic_ground_truth import (
    StochasticGroundTruthSpec,
    evaluate_stochastic_ground_truth,
)
from benchmarks.synthetic.stochastic_target import StochasticSyntheticTarget

# Supported target families
TARGET_FAMILIES = (
    "single_trigger",
    "conjunctive_trigger",
    "multiple_independent",
    "distractor_heavy",
    "null_target",
    "non_monotonic",
    "stochastic_target",
    "weak_signal",
)

# Standard candidate sizes
CANDIDATE_SIZES = (4, 8, 12, 16, 20)


def create_scaling_task(
    target_family: str,
    k: int,
    seed: int = 42,
) -> tuple[StochasticSyntheticTarget, InputCase, StochasticGroundTruthSpec]:
    """Generate an isolated, reproducible synthetic scaling task.

    Parameters
    ----------
    target_family : str
        One of the 8 target families in TARGET_FAMILIES.
    k : int
        Candidate space size (number of input elements, >= 4).
    seed : int
        Random seed for element permutation and target noise.

    Returns
    -------
    tuple[StochasticSyntheticTarget, InputCase, StochasticGroundTruthSpec]
        Isolated target oracle, input case, and ground truth specification.
    """
    if k < 4:
        raise ValueError(f"Candidate space size K must be >= 4, got {k}")
    if target_family not in TARGET_FAMILIES:
        raise ValueError(f"Unknown target family: {target_family!r}. Must be one of {TARGET_FAMILIES}")

    rng = random.Random(seed)

    if target_family == "single_trigger":
        trigger = "elem_t_single"
        distractors = [f"elem_d_{i:02d}" for i in range(1, k)]
        all_elems = [trigger] + distractors
        rng.shuffle(all_elems)

        def rule_fn(elements: set[str]) -> float:
            return 0.90 if trigger in elements else 0.05

        gt = StochasticGroundTruthSpec(
            task_id=f"single_k{k}_s{seed}",
            name=f"Single Trigger (K={k})",
            description=f"Single trigger '{trigger}' with P=0.90 vs base P=0.05 across K={k} elements.",
            trigger_elements=(trigger,),
            base_probability=0.05,
            trigger_probability=0.90,
        )

    elif target_family == "conjunctive_trigger":
        t1, t2 = "elem_t_conj_a", "elem_t_conj_b"
        distractors = [f"elem_d_{i:02d}" for i in range(1, k - 1)]
        all_elems = [t1, t2] + distractors
        rng.shuffle(all_elems)

        def rule_fn(elements: set[str]) -> float:
            return 0.90 if (t1 in elements and t2 in elements) else 0.05

        gt = StochasticGroundTruthSpec(
            task_id=f"conjunctive_k{k}_s{seed}",
            name=f"Conjunctive Trigger (K={k})",
            description=f"Two-factor conjunctive trigger ('{t1}', '{t2}') with P=0.90 vs base P=0.05 across K={k} elements.",
            trigger_elements=(t1, t2),
            base_probability=0.05,
            trigger_probability=0.90,
        )

    elif target_family == "multiple_independent":
        t1, t2 = "elem_t_indep_1", "elem_t_indep_2"
        distractors = [f"elem_d_{i:02d}" for i in range(1, k - 1)]
        all_elems = [t1, t2] + distractors
        rng.shuffle(all_elems)

        def rule_fn(elements: set[str]) -> float:
            return 0.90 if (t1 in elements or t2 in elements) else 0.05

        gt = StochasticGroundTruthSpec(
            task_id=f"multiple_indep_k{k}_s{seed}",
            name=f"Multiple Independent Triggers (K={k})",
            description=f"Multiple independent triggers ('{t1}' or '{t2}') each with P=0.90 vs base P=0.05 across K={k} elements.",
            trigger_elements=(t1,),
            alternative_valid_triggers=((t2,),),
            base_probability=0.05,
            trigger_probability=0.90,
        )

    elif target_family == "distractor_heavy":
        trigger = "elem_t_heavy"
        distractors = [f"elem_d_distractor_{i:02d}" for i in range(1, k)]
        all_elems = [trigger] + distractors
        rng.shuffle(all_elems)

        def rule_fn(elements: set[str]) -> float:
            return 0.90 if trigger in elements else 0.05

        gt = StochasticGroundTruthSpec(
            task_id=f"distractor_heavy_k{k}_s{seed}",
            name=f"Distractor Heavy (K={k})",
            description=f"1 trigger '{trigger}' among {k - 1} distractors (P=0.90 vs 0.05).",
            trigger_elements=(trigger,),
            base_probability=0.05,
            trigger_probability=0.90,
        )

    elif target_family == "null_target":
        distractors = [f"elem_d_null_{i:02d}" for i in range(1, k + 1)]
        all_elems = list(distractors)
        rng.shuffle(all_elems)

        def rule_fn(elements: set[str]) -> float:
            return 0.05

        gt = StochasticGroundTruthSpec(
            task_id=f"null_k{k}_s{seed}",
            name=f"Null Target (K={k})",
            description=f"Null case with unconditional background failure P=0.05 across K={k} elements; no causal factor.",
            trigger_elements=(),
            base_probability=0.05,
            trigger_probability=0.05,
            is_null_case=True,
        )

    elif target_family == "non_monotonic":
        a = "elem_t_nm_trigger"
        b = "elem_t_nm_suppressor"
        c = "elem_t_nm_restorer"
        distractors = [f"elem_d_{i:02d}" for i in range(1, k - 2)]
        all_elems = [a, b, c] + distractors
        rng.shuffle(all_elems)

        def rule_fn(elements: set[str]) -> float:
            has_a = a in elements
            has_b = b in elements
            has_c = c in elements
            if has_a and has_b and has_c:
                return 0.90
            if has_a and has_b:
                return 0.10
            if has_a:
                return 0.90
            return 0.05

        gt = StochasticGroundTruthSpec(
            task_id=f"non_monotonic_k{k}_s{seed}",
            name=f"Non-Monotonic Target (K={k})",
            description=f"Non-monotonic suppression/restoration: '{a}' (0.90), suppressed by '{b}' (0.10), restored by '{c}' (0.90).",
            trigger_elements=(a,),
            alternative_valid_triggers=((a, b, c),),
            base_probability=0.05,
            trigger_probability=0.90,
        )

    elif target_family == "stochastic_target":
        trigger = "elem_t_stoch"
        distractors = [f"elem_d_{i:02d}" for i in range(1, k)]
        all_elems = [trigger] + distractors
        rng.shuffle(all_elems)

        def rule_fn(elements: set[str]) -> float:
            return 0.75 if trigger in elements else 0.15

        gt = StochasticGroundTruthSpec(
            task_id=f"stochastic_target_k{k}_s{seed}",
            name=f"Stochastic Target (K={k})",
            description=f"Probabilistic trigger with moderate noise (P=0.75 vs base P=0.15) across K={k} elements.",
            trigger_elements=(trigger,),
            base_probability=0.15,
            trigger_probability=0.75,
        )

    elif target_family == "weak_signal":
        trigger = "elem_t_weak"
        distractors = [f"elem_d_{i:02d}" for i in range(1, k)]
        all_elems = [trigger] + distractors
        rng.shuffle(all_elems)

        def rule_fn(elements: set[str]) -> float:
            return 0.40 if trigger in elements else 0.15

        gt = StochasticGroundTruthSpec(
            task_id=f"weak_signal_k{k}_s{seed}",
            name=f"Weak Signal Target (K={k})",
            description=f"Weak signal with delta=0.25 (P=0.40 vs base P=0.15) across K={k} elements.",
            trigger_elements=(trigger,),
            base_probability=0.15,
            trigger_probability=0.40,
        )

    target = StochasticSyntheticTarget(
        rule_fn=rule_fn,
        name=f"Synthetic_{target_family}_K{k}_s{seed}",
        target_seed=seed,
    )
    initial_input = InputCase(
        case_id=f"case_{target_family}_k{k}_s{seed}",
        elements=tuple(all_elems),
        description=f"Scaling task {target_family} with K={k} elements (seed={seed})",
    )

    return (target, initial_input, gt)


class AlgorithmRunResult(BaseModel):
    """Execution metrics for a single algorithm run on a benchmark instance."""

    model_config = ConfigDict(frozen=True)

    algorithm_name: str
    discovered_elements: tuple[str, ...]
    target_calls: int
    search_target_calls: int
    verification_target_calls: int
    subsets_examined: int
    failure_reproduced: bool
    minimality_status: str
    is_verified_one_minimal: bool
    matches_ground_truth: bool
    precision: float
    recall: float
    budget_exhausted: bool
    runtime_ms: float


class ScalingInstanceResult(BaseModel):
    """Comprehensive paired benchmark instance results across all three algorithms."""

    model_config = ConfigDict(frozen=True)

    family: str
    k: int
    seed: int
    initial_elements: tuple[str, ...]
    ground_truth_trigger: tuple[str, ...]
    alternative_valid_triggers: tuple[tuple[str, ...], ...]
    is_null_case: bool
    dd_result: AlgorithmRunResult
    greedy_result: AlgorithmRunResult
    random_result: AlgorithmRunResult


def run_scaling_instance(
    family: str,
    k: int,
    seed: int,
    trials_per_candidate: int = 10,
    max_trials: int = 1000,
    timeout_seconds: float = 60.0,
    random_search_seed: int = 999,
) -> ScalingInstanceResult:
    """Execute all three algorithms on an identical, paired benchmark instance."""
    # Determine failure threshold: 0.25 for weak_signal, 0.50 for others
    threshold = 0.25 if family == "weak_signal" else 0.50

    target, initial_input, gt = create_scaling_task(target_family=family, k=k, seed=seed)

    # 1. Delta Debugging
    target.reset(new_seed=seed)
    config_dd = ExperimentConfig(
        experiment_name=f"dd_{family}_k{k}_s{seed}",
        max_trials=max_trials,
        timeout_seconds=timeout_seconds,
    )
    runner_dd = InterventionRunner(config=config_dd, target_fn=target.execute)
    minimizer = StochasticDeltaDebugMinimizer(
        runner=runner_dd,
        trials_per_candidate=trials_per_candidate,
        failure_threshold=threshold,
    )
    start_time = time.perf_counter()
    dd_cond = minimizer.minimize(initial_input)
    dd_runtime = (time.perf_counter() - start_time) * 1000.0

    dd_matches, dd_prec, dd_rec, _, _ = evaluate_stochastic_ground_truth(gt, dd_cond)
    dd_res = AlgorithmRunResult(
        algorithm_name="stochastic_delta_debugging",
        discovered_elements=dd_cond.subset_elements,
        target_calls=dd_cond.target_calls_used,
        search_target_calls=dd_cond.search_target_calls,
        verification_target_calls=dd_cond.verification_target_calls,
        subsets_examined=dd_cond.subsets_examined,
        failure_reproduced=dd_cond.is_failure,
        minimality_status=dd_cond.minimality_status.value,
        is_verified_one_minimal=(dd_cond.minimality_status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED),
        matches_ground_truth=dd_matches,
        precision=dd_prec,
        recall=dd_rec,
        budget_exhausted=(dd_cond.status.value == "budget_exhausted"),
        runtime_ms=round(dd_runtime, 2),
    )

    # 2. Greedy Ablation
    target.reset(new_seed=seed)
    config_greedy = ExperimentConfig(
        experiment_name=f"greedy_{family}_k{k}_s{seed}",
        max_trials=max_trials,
        timeout_seconds=timeout_seconds,
    )
    runner_greedy = InterventionRunner(config=config_greedy, target_fn=target.execute)
    start_time = time.perf_counter()
    greedy_comp = run_stochastic_greedy_ablation(
        initial_input=initial_input,
        runner=runner_greedy,
        trials_per_candidate=trials_per_candidate,
        failure_threshold=threshold,
        true_trigger=gt.trigger_elements,
        alternative_valid_triggers=gt.alternative_valid_triggers,
    )
    greedy_runtime = (time.perf_counter() - start_time) * 1000.0

    # Wrap greedy condition to evaluate ground truth
    dummy_greedy_cond = StochasticDiscoveredCondition(
        subset_elements=greedy_comp.discovered_elements,
        status=dd_cond.status,
        is_failure=greedy_comp.failure_reproduced,
        minimality_status=greedy_comp.minimality_status,
        candidate_evaluation=greedy_comp.candidate_evaluation,
        target_calls_used=greedy_comp.target_calls,
        search_target_calls=greedy_comp.search_target_calls,
        verification_target_calls=greedy_comp.verification_target_calls,
        subsets_examined=greedy_comp.subsets_examined,
    )
    gr_matches, gr_prec, gr_rec, _, _ = evaluate_stochastic_ground_truth(gt, dummy_greedy_cond)
    greedy_res = AlgorithmRunResult(
        algorithm_name="stochastic_greedy_ablation",
        discovered_elements=greedy_comp.discovered_elements,
        target_calls=greedy_comp.target_calls,
        search_target_calls=greedy_comp.search_target_calls,
        verification_target_calls=greedy_comp.verification_target_calls,
        subsets_examined=greedy_comp.subsets_examined,
        failure_reproduced=greedy_comp.failure_reproduced,
        minimality_status=greedy_comp.minimality_status.value,
        is_verified_one_minimal=(greedy_comp.minimality_status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED),
        matches_ground_truth=gr_matches,
        precision=gr_prec,
        recall=gr_rec,
        budget_exhausted=greedy_comp.budget_exhausted,
        runtime_ms=round(greedy_runtime, 2),
    )

    # 3. Seeded Random Search
    target.reset(new_seed=seed)
    config_random = ExperimentConfig(
        experiment_name=f"random_{family}_k{k}_s{seed}",
        max_trials=max_trials,
        timeout_seconds=timeout_seconds,
    )
    runner_random = InterventionRunner(config=config_random, target_fn=target.execute)
    start_time = time.perf_counter()
    random_comp = run_stochastic_random_search(
        initial_input=initial_input,
        runner=runner_random,
        trials_per_candidate=trials_per_candidate,
        failure_threshold=threshold,
        true_trigger=gt.trigger_elements,
        alternative_valid_triggers=gt.alternative_valid_triggers,
        seed=random_search_seed + seed,
    )
    random_runtime = (time.perf_counter() - start_time) * 1000.0

    dummy_random_cond = StochasticDiscoveredCondition(
        subset_elements=random_comp.discovered_elements,
        status=dd_cond.status,
        is_failure=random_comp.failure_reproduced,
        minimality_status=random_comp.minimality_status,
        candidate_evaluation=random_comp.candidate_evaluation,
        target_calls_used=random_comp.target_calls,
        search_target_calls=random_comp.search_target_calls,
        verification_target_calls=random_comp.verification_target_calls,
        subsets_examined=random_comp.subsets_examined,
    )
    rnd_matches, rnd_prec, rnd_rec, _, _ = evaluate_stochastic_ground_truth(gt, dummy_random_cond)
    random_res = AlgorithmRunResult(
        algorithm_name="stochastic_seeded_random",
        discovered_elements=random_comp.discovered_elements,
        target_calls=random_comp.target_calls,
        search_target_calls=random_comp.search_target_calls,
        verification_target_calls=random_comp.verification_target_calls,
        subsets_examined=random_comp.subsets_examined,
        failure_reproduced=random_comp.failure_reproduced,
        minimality_status=random_comp.minimality_status.value,
        is_verified_one_minimal=(random_comp.minimality_status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED),
        matches_ground_truth=rnd_matches,
        precision=rnd_prec,
        recall=rnd_rec,
        budget_exhausted=random_comp.budget_exhausted,
        runtime_ms=round(random_runtime, 2),
    )

    return ScalingInstanceResult(
        family=family,
        k=k,
        seed=seed,
        initial_elements=initial_input.elements,
        ground_truth_trigger=gt.trigger_elements,
        alternative_valid_triggers=gt.alternative_valid_triggers,
        is_null_case=gt.is_null_case,
        dd_result=dd_res,
        greedy_result=greedy_res,
        random_result=random_res,
    )


class ScalingBenchmarkSuite(BaseModel):
    """Complete collection of scaling benchmark results and paired statistical summaries."""

    model_config = ConfigDict(frozen=True)

    timestamp: str
    total_instances: int
    candidate_sizes: tuple[int, ...]
    target_families: tuple[str, ...]
    trials_per_candidate: int
    max_trials: int
    instances: tuple[ScalingInstanceResult, ...]


def run_full_scaling_benchmark(
    seeds_per_condition: int = 10,
    candidate_sizes: Sequence[int] = CANDIDATE_SIZES,
    target_families: Sequence[str] = TARGET_FAMILIES,
    trials_per_candidate: int = 10,
    max_trials: int = 1000,
) -> ScalingBenchmarkSuite:
    """Execute complete multi-instance scaling benchmark sweep."""
    all_instances: list[ScalingInstanceResult] = []

    for k in candidate_sizes:
        for family in target_families:
            for s in range(seeds_per_condition):
                inst_res = run_scaling_instance(
                    family=family,
                    k=k,
                    seed=100 + s,
                    trials_per_candidate=trials_per_candidate,
                    max_trials=max_trials,
                )
                all_instances.append(inst_res)

    return ScalingBenchmarkSuite(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_instances=len(all_instances),
        candidate_sizes=tuple(candidate_sizes),
        target_families=tuple(target_families),
        trials_per_candidate=trials_per_candidate,
        max_trials=max_trials,
        instances=tuple(all_instances),
    )


def export_benchmark_results(
    suite: ScalingBenchmarkSuite,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Serialize benchmark suite to structured JSON and tabular CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase_2a_scaling_results.json"
    csv_path = output_dir / "phase_2a_scaling_summary.csv"

    # 1. JSON Export
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(suite.model_dump_json(indent=2))

    # 2. CSV Export (flat tabular per-instance per-algorithm rows)
    fieldnames = [
        "family",
        "k",
        "seed",
        "is_null_case",
        "algorithm",
        "target_calls",
        "search_target_calls",
        "verification_target_calls",
        "subsets_examined",
        "failure_reproduced",
        "minimality_status",
        "is_verified_one_minimal",
        "matches_ground_truth",
        "precision",
        "recall",
        "budget_exhausted",
        "runtime_ms",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for inst in suite.instances:
            for algo_res in (inst.dd_result, inst.greedy_result, inst.random_result):
                writer.writerow({
                    "family": inst.family,
                    "k": inst.k,
                    "seed": inst.seed,
                    "is_null_case": inst.is_null_case,
                    "algorithm": algo_res.algorithm_name,
                    "target_calls": algo_res.target_calls,
                    "search_target_calls": algo_res.search_target_calls,
                    "verification_target_calls": algo_res.verification_target_calls,
                    "subsets_examined": algo_res.subsets_examined,
                    "failure_reproduced": algo_res.failure_reproduced,
                    "minimality_status": algo_res.minimality_status,
                    "is_verified_one_minimal": algo_res.is_verified_one_minimal,
                    "matches_ground_truth": algo_res.matches_ground_truth,
                    "precision": algo_res.precision,
                    "recall": algo_res.recall,
                    "budget_exhausted": algo_res.budget_exhausted,
                    "runtime_ms": algo_res.runtime_ms,
                })

    return (json_path, csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2A Scaling Benchmark Suite")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per condition")
    parser.add_argument("--trials", type=int, default=10, help="Trials per candidate")
    parser.add_argument("--max-trials", type=int, default=1000, help="Max trial budget per run")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory")

    args = parser.parse_args()
    print(f"Running Scaling Benchmark: {len(CANDIDATE_SIZES)} sizes x {len(TARGET_FAMILIES)} families x {args.seeds} seeds...")
    start = time.perf_counter()
    suite = run_full_scaling_benchmark(
        seeds_per_condition=args.seeds,
        trials_per_candidate=args.trials,
        max_trials=args.max_trials,
    )
    elapsed = time.perf_counter() - start
    out_dir = Path(args.output_dir)
    j_p, c_p = export_benchmark_results(suite, out_dir)
    print(f"Completed {suite.total_instances} benchmark instances in {elapsed:.2f}s!")
    print(f"Results written to:\n  JSON: {j_p}\n  CSV:  {c_p}")
