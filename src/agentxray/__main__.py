"""CLI Entry point for AgentXRay.

Executes deterministic and stochastic synthetic benchmark suites across all tasks
and baselines, evaluates results against isolated ground truth, and outputs
machine-readable JSON and human-readable Markdown reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is available on sys.path for benchmark discovery
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agentxray.baselines import run_greedy_ablation, run_seeded_random_search
from agentxray.minimizer import DeltaDebugMinimizer
from agentxray.models import (
    AccessMode,
    EvaluationResult,
    ExperimentConfig,
    ExperimentReport,
    StochasticEvaluationResult,
    StochasticExperimentReport,
    StochasticMinimalityStatus,
)
from agentxray.reporting import (
    format_markdown_report,
    format_stochastic_markdown_report,
    write_reports,
    write_stochastic_reports,
)
from agentxray.runner import InterventionRunner
from agentxray.stochastic_baselines import (
    run_stochastic_greedy_ablation,
    run_stochastic_random_search,
)
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.synthetic.ground_truth import evaluate_ground_truth
from benchmarks.synthetic.stochastic_ground_truth import evaluate_stochastic_ground_truth
from benchmarks.synthetic.stochastic_tasks import get_all_stochastic_benchmark_tasks
from benchmarks.synthetic.tasks import get_all_benchmark_tasks


def run_synthetic_benchmark(
    max_trials: int = 100,
    timeout_seconds: float = 30.0,
    seed: int = 42,
    output_dir: Path | str = ".",
) -> ExperimentReport:
    """Run all deterministic benchmark tasks across Delta Debugging, Greedy Ablation, and Seeded Random Search."""
    config = ExperimentConfig(
        experiment_name="Phase1A_Deterministic_Benchmark",
        max_trials=max_trials,
        timeout_seconds=timeout_seconds,
        seed=seed,
        mode=AccessMode.CONTROLLED_TESTBED,
        metadata={"phase": "1A", "type": "synthetic_deterministic_validation"},
    )

    tasks = get_all_benchmark_tasks()
    evaluations: list[EvaluationResult] = []
    total_calls = 0

    for target, initial_input, ground_truth in tasks:
        # 1. Run Delta Debugging Minimizer
        target.reset_call_count()
        runner_dd = InterventionRunner(config=config, target_fn=target.execute)
        minimizer = DeltaDebugMinimizer(runner=runner_dd)
        discovered_dd = minimizer.minimize(initial_input)
        total_calls += runner_dd.trial_count

        # Evaluate Delta Debugging against ground truth
        matches_gt, prec, rec, excess, missing = evaluate_ground_truth(ground_truth, discovered_dd)

        # 2. Run Greedy Ablation Baseline
        target.reset_call_count()
        runner_greedy = InterventionRunner(config=config, target_fn=target.execute)
        greedy_comp = run_greedy_ablation(
            initial_input=initial_input,
            runner=runner_greedy,
            true_trigger=ground_truth.trigger_elements,
        )
        total_calls += runner_greedy.trial_count

        # 3. Run Seeded Random Search Baseline
        target.reset_call_count()
        runner_random = InterventionRunner(config=config, target_fn=target.execute)
        random_comp = run_seeded_random_search(
            initial_input=initial_input,
            runner=runner_random,
            true_trigger=ground_truth.trigger_elements,
            seed=seed,
        )
        total_calls += runner_random.trial_count

        eval_res = EvaluationResult(
            task_name=ground_truth.name,
            description=ground_truth.description,
            initial_elements=initial_input.elements,
            ground_truth_trigger=ground_truth.trigger_elements,
            minimizer_condition=discovered_dd,
            baseline_comparisons=(greedy_comp, random_comp),
            matches_ground_truth=matches_gt,
            precision=prec,
            recall=rec,
            excess_elements=excess,
            missing_elements=missing,
        )
        evaluations.append(eval_res)

    all_matched = all(e.matches_ground_truth for e in evaluations)
    summary_text = (
        f"Executed {len(evaluations)} synthetic tasks across Delta Debugging, Greedy Ablation, and Seeded Random Search. "
        f"Total target evaluations: {total_calls}. "
        f"Delta Debugging recovered ground truth on {sum(1 for e in evaluations if e.matches_ground_truth)}/{len(evaluations)} tasks "
        f"({'100% success' if all_matched else 'partial success'})."
    )

    report = ExperimentReport(
        experiment_name=config.experiment_name,
        config=config,
        total_target_calls=total_calls,
        evaluations=tuple(evaluations),
        summary=summary_text,
    )

    out_path = Path(output_dir)
    json_path = out_path / "report.json"
    md_path = out_path / "report.md"
    write_reports(report, json_path=json_path, md_path=md_path)

    return report


def run_stochastic_benchmark(
    max_trials: int = 300,
    trials_per_candidate: int = 20,
    failure_threshold: float = 0.30,
    alpha: float = 0.05,
    confidence: float = 0.95,
    timeout_seconds: float = 60.0,
    seed: int = 42,
    target_seed: int = 42,
    output_dir: Path | str = ".",
) -> StochasticExperimentReport:
    """Run all 6 stochastic benchmark tasks across Minimizer and Baselines with uncertainty intervals."""
    config = ExperimentConfig(
        experiment_name="Phase1B_Stochastic_Benchmark",
        max_trials=max_trials,
        timeout_seconds=timeout_seconds,
        seed=seed,
        mode=AccessMode.CONTROLLED_TESTBED,
        metadata={
            "phase": "1B",
            "type": "synthetic_stochastic_validation",
            "trials_per_candidate": trials_per_candidate,
            "failure_threshold": failure_threshold,
            "alpha": alpha,
            "confidence": confidence,
            "target_seed": target_seed,
        },
    )

    tasks = get_all_stochastic_benchmark_tasks(target_seed=target_seed)
    evaluations: list[StochasticEvaluationResult] = []
    total_calls = 0

    for target, initial_input, ground_truth in tasks:
        # 1. Run Stochastic Delta Debugging Minimizer
        target.reset(new_seed=target_seed)
        runner_dd = InterventionRunner(config=config, target_fn=target.execute)
        minimizer = StochasticDeltaDebugMinimizer(
            runner=runner_dd,
            trials_per_candidate=trials_per_candidate,
            failure_threshold=failure_threshold,
            alpha=alpha,
            confidence=confidence,
        )
        discovered_dd = minimizer.minimize(initial_input)
        total_calls += runner_dd.trial_count

        matches_gt, prec, rec, excess, missing = evaluate_stochastic_ground_truth(ground_truth, discovered_dd)

        # 2. Run Stochastic Greedy Ablation Baseline
        target.reset(new_seed=target_seed)
        runner_greedy = InterventionRunner(config=config, target_fn=target.execute)
        greedy_comp = run_stochastic_greedy_ablation(
            initial_input=initial_input,
            runner=runner_greedy,
            trials_per_candidate=trials_per_candidate,
            failure_threshold=failure_threshold,
            alpha=alpha,
            confidence=confidence,
            true_trigger=ground_truth.trigger_elements,
            alternative_valid_triggers=ground_truth.alternative_valid_triggers,
        )
        total_calls += runner_greedy.trial_count

        # 3. Run Stochastic Seeded Random Search Baseline
        target.reset(new_seed=target_seed)
        runner_random = InterventionRunner(config=config, target_fn=target.execute)
        random_comp = run_stochastic_random_search(
            initial_input=initial_input,
            runner=runner_random,
            trials_per_candidate=trials_per_candidate,
            failure_threshold=failure_threshold,
            alpha=alpha,
            confidence=confidence,
            true_trigger=ground_truth.trigger_elements,
            alternative_valid_triggers=ground_truth.alternative_valid_triggers,
            seed=seed,
        )
        total_calls += runner_random.trial_count

        eval_res = StochasticEvaluationResult(
            task_name=ground_truth.name,
            description=ground_truth.description,
            initial_elements=initial_input.elements,
            ground_truth_trigger=ground_truth.trigger_elements,
            base_probability=ground_truth.base_probability,
            trigger_probability=ground_truth.trigger_probability,
            minimizer_condition=discovered_dd,
            baseline_comparisons=(greedy_comp, random_comp),
            matches_ground_truth=matches_gt,
            precision=prec,
            recall=rec,
            excess_elements=excess,
            missing_elements=missing,
        )
        evaluations.append(eval_res)

    matched_count = sum(1 for e in evaluations if e.matches_ground_truth)
    stat_supp_count = sum(
        1 for e in evaluations
        if e.minimizer_condition.minimality_status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
    )
    summary_text = (
        f"Executed {len(evaluations)} stochastic synthetic tasks across Delta Debugging, "
        f"Greedy Ablation, and Seeded Random Search. Total target invocations: {total_calls}. "
        f"Delta Debugging matched ground truth on {matched_count}/{len(evaluations)} tasks. "
        f"Statistically supported 1-minimality established on {stat_supp_count}/{len(evaluations)} tasks."
    )

    report = StochasticExperimentReport(
        experiment_name=config.experiment_name,
        config=config,
        trials_per_candidate=trials_per_candidate,
        total_target_calls=total_calls,
        evaluations=tuple(evaluations),
        aggregate_metrics={
            "tasks_evaluated": len(evaluations),
            "ground_truth_matches": matched_count,
            "statistically_supported_minimality": stat_supp_count,
            "mean_precision": round(sum(e.precision for e in evaluations) / len(evaluations), 4),
            "mean_recall": round(sum(e.recall for e in evaluations) / len(evaluations), 4),
        },
        summary=summary_text,
    )

    out_path = Path(output_dir)
    json_path = out_path / "stochastic_report.json"
    md_path = out_path / "stochastic_report.md"
    write_stochastic_reports(report, json_path=json_path, md_path=md_path)

    return report


def main() -> None:
    """CLI entry point supporting deterministic and stochastic benchmark execution."""
    parser = argparse.ArgumentParser(
        description="AgentXRay: Controlled Black-Box Probing and Behavioral Minimization Framework."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["deterministic", "stochastic", "all"],
        default="deterministic",
        help="Benchmark execution mode (default: deterministic).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Maximum trial budget per algorithm execution (default: 100 for deterministic, 300 for stochastic).",
    )
    parser.add_argument(
        "--trials-per-candidate",
        type=int,
        default=20,
        help="Number of trials per candidate subset evaluation in stochastic mode (default: 20).",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.30,
        help="Failure proportion threshold to deem candidate failing in stochastic mode (default: 0.30).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Statistical significance threshold for FDR adjustments and hypothesis tests (default: 0.05).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for Wilson and Newcombe uncertainty intervals (default: 0.95).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Wall-clock timeout in seconds (default: 60.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for algorithm sampling (default: 42).",
    )
    parser.add_argument(
        "--target-seed",
        type=int,
        default=42,
        help="Random seed for synthetic stochastic target outcomes (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save output reports (default: current directory).",
    )

    args = parser.parse_args()

    print("=" * 75)
    print(" AgentXRay: Probing & Behavioral Minimization Execution")
    print("=" * 75)
    print(f"Mode: {args.mode.upper()}")
    print(f"Algorithm Seed: {args.seed} | Target Seed: {args.target_seed}")
    print(f"Output Directory: {args.output_dir}")
    print("-" * 75)

    if args.mode in ("deterministic", "all"):
        det_max_trials = args.max_trials if args.max_trials is not None else 100
        print(f"\n[Running Deterministic Prototype (max_trials={det_max_trials})]")
        det_report = run_synthetic_benchmark(
            max_trials=det_max_trials,
            timeout_seconds=args.timeout,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print(format_markdown_report(det_report))

    if args.mode in ("stochastic", "all"):
        stoch_max_trials = args.max_trials if args.max_trials is not None else 300
        print(f"\n[Running Stochastic Benchmark (max_trials={stoch_max_trials}, trials_per_candidate={args.trials_per_candidate})]")
        stoch_report = run_stochastic_benchmark(
            max_trials=stoch_max_trials,
            trials_per_candidate=args.trials_per_candidate,
            failure_threshold=args.failure_threshold,
            alpha=args.alpha,
            confidence=args.confidence,
            timeout_seconds=args.timeout,
            seed=args.seed,
            target_seed=args.target_seed,
            output_dir=args.output_dir,
        )
        print(format_stochastic_markdown_report(stoch_report))

    print("\n" + "=" * 75)
    print(" Execution finished successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
