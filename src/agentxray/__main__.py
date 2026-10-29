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

from agentxray.audit import (
    format_audit_markdown_report,
    run_multiseed_audit,
    write_audit_reports,
)
from agentxray.baselines import run_greedy_ablation, run_seeded_random_search
from agentxray.minimizer import DeltaDebugMinimizer
from agentxray.models import (
    AccessMode,
    EvaluationResult,
    ExperimentConfig,
    ExperimentReport,
    StochasticEvaluationResult,
    StochasticExperimentReport,
)
from agentxray.reporting import (
    compute_stochastic_aggregate_metrics,
    format_cohort_summary_sentences,
    format_markdown_report,
    format_model_pilot_markdown_report,
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
from benchmarks.model_pilot import (
    dry_run_model_pilot,
    run_model_pilot_benchmark,
)
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
        )
        total_calls += runner_greedy.trial_count
        g_matches, _, _, g_excess, _ = evaluate_ground_truth(ground_truth, greedy_comp)
        greedy_comp = greedy_comp.model_copy(
            update={
                "matches_ground_truth": g_matches,
                "irrelevant_elements_remaining": len(g_excess),
            }
        )

        # 3. Run Seeded Random Search Baseline
        target.reset_call_count()
        runner_random = InterventionRunner(config=config, target_fn=target.execute)
        random_comp = run_seeded_random_search(
            initial_input=initial_input,
            runner=runner_random,
            seed=seed,
        )
        total_calls += runner_random.trial_count
        r_matches, _, _, r_excess, _ = evaluate_ground_truth(ground_truth, random_comp)
        random_comp = random_comp.model_copy(
            update={
                "matches_ground_truth": r_matches,
                "irrelevant_elements_remaining": len(r_excess),
            }
        )

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
    """Run all 6 stochastic benchmark tasks across Minimizer and Baselines with uncertainty intervals.

    The emitted report's aggregate_metrics dictionary contains:
    - tasks_evaluated: Total count of evaluated stochastic tasks.
    - ground_truth_matches: Tasks matching ground truth exactly.
    - statistically_supported_minimality: Tasks achieving STATISTICALLY_SUPPORTED minimality.
    - active_tasks: Count of active (non-null) failure tasks.
    - active_mean_precision: Mean precision across active tasks.
    - active_mean_recall: Mean recall across active tasks.
    - null_tasks: Count of null control tasks.
    - null_detection_rate: Proportion of null controls matching ground truth (clean non-failure).
    - null_false_positive_rate: Overall control false-alarm rate (1.0 - null_detection_rate).
    - null_element_false_positives: Count of null controls with non-empty discovered elements.
    - null_spurious_failures: Count of null controls with empty discovered sets that spuriously reproduced failure.
    - mean_precision: Legacy mean precision across all evaluations where precision is not None.
    - mean_recall: Legacy mean recall across all evaluations where recall is not None.
    - mean_precision_evaluations: Denominator count for mean_precision.
    - mean_recall_evaluations: Denominator count for mean_recall.
    """
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
        )
        total_calls += runner_greedy.trial_count
        g_matches, _, _, g_excess, _ = evaluate_stochastic_ground_truth(ground_truth, greedy_comp)
        greedy_comp = greedy_comp.model_copy(update={
            "matches_ground_truth": g_matches,
            "irrelevant_elements_remaining": len(g_excess),
        })

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
            seed=seed,
        )
        total_calls += runner_random.trial_count
        r_matches, _, _, r_excess, _ = evaluate_stochastic_ground_truth(ground_truth, random_comp)
        random_comp = random_comp.model_copy(update={
            "matches_ground_truth": r_matches,
            "irrelevant_elements_remaining": len(r_excess),
        })

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

    metrics = compute_stochastic_aggregate_metrics(evaluations)

    summary_parts = [
        f"Executed {len(evaluations)} stochastic synthetic tasks across Delta Debugging, "
        f"Greedy Ablation, and Seeded Random Search. Total target invocations: {total_calls}. "
        f"Delta Debugging matched ground truth on {metrics['ground_truth_matches']}/{len(evaluations)} tasks. "
        f"Statistically supported 1-minimality established on {metrics['statistically_supported_minimality']}/{len(evaluations)} tasks."
    ]
    summary_parts.extend(format_cohort_summary_sentences(metrics))
    summary_text = " ".join(summary_parts)

    report = StochasticExperimentReport(
        experiment_name=config.experiment_name,
        config=config,
        trials_per_candidate=trials_per_candidate,
        total_target_calls=total_calls,
        evaluations=tuple(evaluations),
        aggregate_metrics=metrics,
        summary=summary_text,
    )

    out_path = Path(output_dir)
    json_path = out_path / "stochastic_report.json"
    md_path = out_path / "stochastic_report.md"
    write_stochastic_reports(report, json_path=json_path, md_path=md_path)

    return report


def _run_legacy_benchmark_mode(args: argparse.Namespace) -> int:
    """Execute research benchmark suites (--mode deterministic|stochastic|audit|model)."""
    # Validate verification parameters
    if getattr(args, "max_discovery_requests", None) is not None and args.max_discovery_requests <= 0:
        print("Error: --max-discovery-requests must be a positive integer.", file=sys.stderr)
        return 2
    if getattr(args, "max_verification_requests", None) is not None and args.max_verification_requests <= 0:
        print("Error: --max-verification-requests must be a positive integer.", file=sys.stderr)
        return 2
    if getattr(args, "verification_trials", None) is not None and args.verification_trials <= 0:
        print("Error: --verification-trials must be a positive integer.", file=sys.stderr)
        return 2
    if getattr(args, "verification_tier", "exploratory").lower() == "confirmatory":
        if getattr(args, "verification_trials", None) is None:
            print(
                "Error: Confirmatory verification requires an explicitly configured sample size (--verification-trials). "
                "Defaulting to exploratory 5 trials is not permitted for confirmatory claims.",
                file=sys.stderr,
            )
            return 2

    mode = args.mode or "deterministic"
    print("=" * 75)
    print(" AgentXRay: Probing & Behavioral Minimization Execution")
    print("=" * 75)
    print(f"Mode: {mode.upper()}")
    print(f"Algorithm Seed: {args.seed} | Target Seed: {args.target_seed}")
    print(f"Output Directory: {args.output_dir}")
    print("-" * 75)

    if mode in ("deterministic", "all"):
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
        stoch_tpc = args.trials_per_candidate if args.trials_per_candidate is not None else 20
        print(f"\n[Running Stochastic Benchmark (max_trials={stoch_max_trials}, trials_per_candidate={stoch_tpc})]")
        stoch_report = run_stochastic_benchmark(
            max_trials=stoch_max_trials,
            trials_per_candidate=stoch_tpc,
            failure_threshold=args.failure_threshold,
            alpha=args.alpha,
            confidence=args.confidence,
            timeout_seconds=args.timeout,
            seed=args.seed,
            target_seed=args.target_seed,
            output_dir=args.output_dir,
        )
        print(format_stochastic_markdown_report(stoch_report))

    if args.mode in ("audit", "all"):
        seeds = [int(s.strip()) for s in args.audit_seeds.split(",") if s.strip()]
        audit_max_trials = args.max_trials if args.max_trials is not None else 300
        audit_tpc = args.trials_per_candidate if args.trials_per_candidate is not None else 20
        print(f"\n[Running Multi-Seed Stochastic Audit (seeds={seeds}, max_trials={audit_max_trials}, trials_per_candidate={audit_tpc})]")
        audit_report = run_multiseed_audit(
            seeds=seeds,
            max_trials=audit_max_trials,
            trials_per_candidate=audit_tpc,
            failure_threshold=args.failure_threshold,
            alpha=args.alpha,
            confidence=args.confidence,
            timeout_seconds=args.timeout,
        )
        out_path = Path(args.output_dir)
        write_audit_reports(
            audit_report,
            json_path=out_path / "audit_report.json",
            md_path=out_path / "audit_report.md",
        )
        print(format_audit_markdown_report(audit_report))

    if args.mode in ("model", "all"):
        if args.live and args.max_spend > 1.00:
            print(
                f"\n[ERROR] Safety safeguard violation: --max-spend is ${args.max_spend:.2f}, "
                "but initial real-model validation pilots strictly mandate --max-spend <= $1.00 USD.",
                file=sys.stderr,
            )
            sys.exit(1)

        pilot_tpc = args.trials_per_candidate if args.trials_per_candidate is not None else 5
        pilot_max_trials = args.max_trials if args.max_trials is not None else 50
        if args.model:
            resolved_model = args.model
        elif args.provider == "openai":
            resolved_model = "gpt-4o-mini"
        elif args.provider == "ollama":
            resolved_model = "llama3.2:3b"
        else:
            resolved_model = "fake-model-pilot"

        effective_max_requests = (
            min(args.max_model_requests, 120) if args.provider == "ollama" else args.max_model_requests
        )

        if args.dry_run:
            print(f"\n[Executing Pre-Flight Dry Run for Model Pilot ({args.provider} / {resolved_model})]")
            dry_info = dry_run_model_pilot(
                provider_type=args.provider,
                model_name=resolved_model,
                live=args.live,
                task_filter=args.task,
                trials_per_candidate=pilot_tpc,
                max_trials=pilot_max_trials,
                max_model_requests=effective_max_requests,
                max_spend_usd=args.max_spend,
                seed=args.seed,
            )
            print("\nPre-Flight Configuration & Cost Validation:")
            print(f"- Provider Type: `{dry_info['provider_type']}` | Model: `{dry_info['model_name']}`")
            print(f"- Random Seed: `{dry_info['seed']}`")
            print(f"- Live API Calls Authorized (--live): `{dry_info['live_requested']}`")
            print(f"- API Key Present in Environment: `{dry_info['api_key_configured']}`")
            print("- Endpoint Connectivity: NOT CHECKED (offline dry-run makes zero network requests)")
            print(f"- Selected Task Filter: `{args.task}`")
            print(f"- Planned Pilot Tasks: `{len(dry_info['tasks'])}`")
            for t in dry_info["tasks"]:
                print(f"  * {t['name']} ({t['initial_elements_count']} candidate elements) - {t['type']}")
            cost = dry_info["cost_estimation"]
            print("- Resource Safeguards & Upfront Spend Estimation:")
            print(f"  * Estimated Requests: {cost['estimated_requests']} (Max cap: {cost['max_allowed_requests']})")
            print(f"  * Estimated Total Tokens: {cost['estimated_total_tokens']:,}")
            print(f"  * Estimated Cost (USD): ${cost['estimated_spend_usd']:.4f} (Spend limit: ${cost['max_allowed_spend_usd']:.4f})")
            print(f"- Authorization Status: **{dry_info['status']}**")
            if args.provider == "ollama":
                print("\n> [!NOTE]")
                print(f"> Local Ollama inference target: `{resolved_model}` at `http://127.0.0.1:11434`.")
                print("> Commercial API spend: $0.0000 USD (runs entirely on local hardware).")
                print(f"> Hard Cap Enforced: {effective_max_requests} model requests maximum.")
                print(f"> Prerequisite: Ensure the local Ollama daemon is running with model pulled (`ollama run {resolved_model}`).")
            elif not dry_info["can_execute_live"] and args.provider == "openai":
                print("\n> [!CAUTION]")
                print("> Live OpenAI API calls require 'OPENAI_API_KEY' set in environment AND the '--live' CLI flag.")
                print(f"> Planned maximum spend: ${cost['estimated_spend_usd']:.4f} USD (Hard cap: ${args.max_spend:.2f} USD).")
                print(f"> Command to run live experiment: `OPENAI_API_KEY='sk-...' agentxray --mode model --provider openai --task {args.task} --live --max-spend {args.max_spend:.2f}`")
        else:
            print(f"\n[Running Model-Backed Agent Pilot (provider={args.provider}, model={resolved_model}, task={args.task}, live={args.live})]")
            try:
                pilot_report = run_model_pilot_benchmark(
                    provider_type=args.provider,
                    model_name=resolved_model,
                    live=args.live,
                    task_filter=args.task,
                    max_trials=pilot_max_trials,
                    trials_per_candidate=pilot_tpc,
                    max_model_requests=effective_max_requests,
                    max_spend_usd=args.max_spend,
                    failure_threshold=args.failure_threshold,
                    alpha=args.alpha,
                    confidence=args.confidence,
                    timeout_seconds=args.timeout,
                    output_dir=args.output_dir,
                    seed=args.seed,
                    max_discovery_requests=args.max_discovery_requests,
                    max_verification_requests=args.max_verification_requests,
                    enable_independent_verification=args.enable_independent_verification,
                    verification_trials=args.verification_trials,
                    verification_tier=args.verification_tier,
                    sample_size_rationale=args.sample_size_rationale,
                )
                print(format_model_pilot_markdown_report(pilot_report))
            except RuntimeError as exc:
                print(f"\n[ABORTED] {exc}", file=sys.stderr)
                sys.exit(1)

    print("\n" + "=" * 75)
    print(" Execution finished successfully.")
    print("=" * 75)
    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point supporting developer commands (demo, diagnose, version) and legacy benchmark suites."""
    from agentxray.cli import cli_main

    exit_code = cli_main(argv)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
