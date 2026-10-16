"""CLI Entry point for AgentXRay.

Executes the deterministic synthetic benchmark suite across all tasks and baselines,
evaluates results against ground truth, and outputs JSON and Markdown reports.
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
)
from agentxray.reporting import format_markdown_report, write_reports
from agentxray.runner import InterventionRunner
from benchmarks.synthetic.ground_truth import evaluate_ground_truth
from benchmarks.synthetic.tasks import get_all_benchmark_tasks


def run_synthetic_benchmark(
    max_trials: int = 100,
    timeout_seconds: float = 30.0,
    seed: int = 42,
    output_dir: Path | str = ".",
) -> ExperimentReport:
    """Run all benchmark tasks across Delta Debugging, Greedy Ablation, and Seeded Random Search."""
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


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AgentXRay Phase 1A: Deterministic Probing and Delta-Debugging Prototype."
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=100,
        help="Maximum trial budget per algorithm execution (default: 100).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Wall-clock timeout in seconds (default: 30.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable baseline sampling (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save report.json and report.md (default: current directory).",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" AgentXRay Phase 1A: Deterministic Prototype Execution")
    print("=" * 70)
    print(f"Max Trials per Method: {args.max_trials}")
    print(f"Timeout: {args.timeout}s | Seed: {args.seed}")
    print(f"Output Directory: {args.output_dir}")
    print("-" * 70)

    report = run_synthetic_benchmark(
        max_trials=args.max_trials,
        timeout_seconds=args.timeout,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print("\n" + format_markdown_report(report))
    print(f"Reports successfully written to: {Path(args.output_dir) / 'report.json'} and {Path(args.output_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
