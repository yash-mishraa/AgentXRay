"""Experiment execution orchestrator and pre-flight dry-run for model-backed pilot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agentxray.agent_harness.cost_tracker import CostTracker, ExperimentBudgetConfig
from agentxray.models import (
    AccessMode,
    ExecutionTier,
    ExperimentConfig,
    ModelPilotEvaluationResult,
    ModelPilotExperimentReport,
    StochasticMinimalityStatus,
)
from agentxray.providers.base import ModelProvider
from agentxray.providers.fake import FakeProvider
from agentxray.providers.openai_adapter import OpenAIProvider
from agentxray.reporting import write_model_pilot_reports
from agentxray.runner import InterventionRunner
from agentxray.stochastic_baselines import (
    run_stochastic_greedy_ablation,
    run_stochastic_random_search,
)
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.model_pilot.ground_truth import evaluate_model_pilot_ground_truth
from benchmarks.model_pilot.tasks import get_model_pilot_tasks


def dry_run_model_pilot(
    provider_type: str = "fake",
    model_name: str = "fake-model-pilot",
    live: bool = False,
    task_filter: str | None = "all",
    trials_per_candidate: int = 5,
    max_trials: int = 50,
    max_model_requests: int = 150,
    max_spend_usd: float = 1.00,
) -> dict[str, Any]:
    """Inspect and validate experiment parameters and estimate costs without contacting any provider."""
    # Check credentials
    api_key_set = bool(os.environ.get("OPENAI_API_KEY", "").strip())

    budget_cfg = ExperimentBudgetConfig(
        max_model_requests=max_model_requests,
        max_spend_usd=max_spend_usd,
        model_name=model_name,
    )
    tracker = CostTracker(budget_cfg)

    all_tasks_meta = [
        {
            "task_id": "pilot_baseline_calc",
            "name": "Model Pilot Baseline (Control)",
            "initial_elements_count": 2,
            "type": "Control condition (clean tool calling)",
        },
        {
            "task_id": "pilot_conflicting_prompt",
            "name": "Model Pilot Conflicting Prompt Constraint",
            "initial_elements_count": 3,
            "type": "Prompt constraint intervention",
        },
        {
            "task_id": "pilot_distractor_override",
            "name": "Model Pilot Distractor-Heavy Field Override",
            "initial_elements_count": 4,
            "type": "Distractor + misleading schema intervention",
        },
    ]

    filter_key = (task_filter or "all").strip().lower()
    if filter_key in ("baseline", "pilot_baseline_calc", "control"):
        tasks_info = [all_tasks_meta[0]]
    elif filter_key in ("conflicting_prompt", "pilot_conflicting_prompt", "prompt"):
        tasks_info = [all_tasks_meta[1]]
    elif filter_key in ("distractor_override", "pilot_distractor_override", "distractor"):
        tasks_info = [all_tasks_meta[2]]
    else:
        tasks_info = all_tasks_meta

    # Pre-flight cost estimation based on selected tasks
    planned_target_calls = min(max_trials, len(tasks_info) * 15)  # estimate ~15 calls per task
    cost_est = tracker.estimate_preflight_cost(
        planned_trials=planned_target_calls,
        turns_per_trial=2,
        est_prompt_tokens_per_turn=250,
        est_completion_tokens_per_turn=50,
    )

    can_execute_live = (provider_type == "openai" and live and api_key_set)

    # Validate spend limit
    spend_cap_valid = (max_spend_usd <= 1.00) if live else True
    status = "VALID" if ((provider_type == "fake" or can_execute_live) and spend_cap_valid) else (
        "INVALID_SPEND_CAP_EXCEEDS_PILOT_LIMIT" if not spend_cap_valid else "REQUIRES_AUTH_OR_OPT_IN"
    )

    return {
        "provider_type": provider_type,
        "model_name": model_name,
        "live_requested": live,
        "api_key_configured": api_key_set,
        "connectivity_verified": False,
        "network_transport_contacted": False,
        "can_execute_live": can_execute_live,
        "task_filter": task_filter,
        "tasks": tasks_info,
        "trials_per_candidate": trials_per_candidate,
        "max_trials_per_algorithm": max_trials,
        "cost_estimation": cost_est,
        "status": status,
    }


def run_model_pilot_benchmark(
    provider_type: str = "fake",
    model_name: str | None = None,
    live: bool = False,
    task_filter: str | None = "all",
    max_trials: int = 50,
    trials_per_candidate: int = 5,
    max_model_requests: int = 150,
    max_spend_usd: float = 1.00,
    failure_threshold: float = 0.50,
    alpha: float = 0.05,
    confidence: float = 0.95,
    timeout_seconds: float = 60.0,
    output_dir: Path | str = ".",
) -> ModelPilotExperimentReport:
    """Execute the model-backed agent pilot across the selected pilot task(s).

    Parameters
    ----------
    provider_type : str
        'fake' (default, hermetic, zero-cost) or 'openai' (live API).
    model_name : str | None
        Target model identifier.
    live : bool
        Explicit opt-in required to make external API calls.
    task_filter : str | None
        'all' (default), 'conflicting_prompt', 'distractor_override', or 'baseline'.
    max_trials : int
        Maximum target evaluations per algorithm run.
    trials_per_candidate : int
        Trials executed per candidate subset.
    max_model_requests : int
        Hard safeguard cap on model requests.
    max_spend_usd : float
        Hard safeguard cap on estimated expenditure in USD (must be <= 1.00 for initial live pilot).
    failure_threshold : float
        Proportion threshold to classify candidate as failing.
    alpha : float
        Significance level for hypothesis tests and FDR adjustments.
    confidence : float
        Confidence level for Wilson and Newcombe uncertainty intervals.
    timeout_seconds : float
        Wall-clock timeout in seconds.
    output_dir : Path | str
        Directory to write output JSON and Markdown reports.

    Returns
    -------
    ModelPilotExperimentReport
        Complete validated experimental report.
    """
    if live and max_spend_usd > 1.00:
        raise ValueError(
            f"Safety safeguard violation: max_spend_usd is ${max_spend_usd:.2f}, "
            "but initial real-model validation pilots strictly mandate max_spend_usd <= $1.00 USD."
        )

    # 1. Resolve Provider and Execution Tier
    if provider_type == "openai":
        if not live:
            raise RuntimeError(
                "Live API calls require the explicit '--live' opt-in flag. "
                "Use '--dry-run' to inspect configuration or '--provider fake' for offline testing."
            )
        resolved_model = model_name or "gpt-4o-mini"
        provider = OpenAIProvider(model_name=resolved_model, timeout_seconds=timeout_seconds)
        provider.validate_configuration()
        tier = ExecutionTier.LIVE_MODEL
    elif provider_type == "fake":
        resolved_model = model_name or "fake-model-pilot"
        provider = FakeProvider(model_name=resolved_model)
        tier = ExecutionTier.FAKE_PROVIDER
    else:
        raise ValueError(f"Unknown provider_type: {provider_type!r}. Supported: 'fake', 'openai'.")

    # 2. Configure Cost Tracker and Budgets
    budget_cfg = ExperimentBudgetConfig(
        max_model_requests=max_model_requests,
        max_spend_usd=max_spend_usd,
        model_name=resolved_model,
        request_timeout_seconds=timeout_seconds,
    )
    cost_tracker = CostTracker(budget_cfg)

    config = ExperimentConfig(
        experiment_name=f"ModelPilot_{tier.value}_{resolved_model}",
        max_trials=max_trials,
        timeout_seconds=timeout_seconds,
        mode=AccessMode.CONTROLLED_TESTBED,
        metadata={
            "tier": tier.value,
            "provider": provider.provider_name,
            "model": resolved_model,
            "task_filter": task_filter,
            "trials_per_candidate": trials_per_candidate,
            "failure_threshold": failure_threshold,
        },
    )

    # 3. Load Pilot Benchmark Tasks
    tasks = get_model_pilot_tasks(provider=provider, cost_tracker=cost_tracker, task_filter=task_filter)
    evaluations: list[ModelPilotEvaluationResult] = []
    total_calls = 0

    for target, initial_input, ground_truth in tasks:
        # 1. Run Stochastic Delta Debugging Minimizer
        target.reset_call_count()
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

        matches_gt, prec, rec, excess, missing = evaluate_model_pilot_ground_truth(ground_truth, discovered_dd)

        # 2. Run Stochastic Greedy Ablation Baseline
        target.reset_call_count()
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
        target.reset_call_count()
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
            seed=42,
        )
        total_calls += runner_random.trial_count

        eval_res = ModelPilotEvaluationResult(
            task_name=ground_truth.name,
            description=ground_truth.description,
            initial_elements=initial_input.elements,
            ground_truth_trigger=ground_truth.trigger_elements,
            is_null_case=ground_truth.is_null_case,
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
        f"Executed {len(evaluations)} model-backed pilot tasks under execution tier [{tier.value.upper()}] "
        f"using provider '{provider.provider_name}' ({resolved_model}). "
        f"Total target invocations: {total_calls}. "
        f"Delta Debugging recovered ground-truth trigger on {matched_count}/{len(evaluations)} tasks. "
        f"Statistically supported minimality established on {stat_supp_count}/{len(evaluations)} tasks."
    )

    valid_precisions = [e.precision for e in evaluations if e.precision is not None]
    valid_recalls = [e.recall for e in evaluations if e.recall is not None]

    report = ModelPilotExperimentReport(
        experiment_name=config.experiment_name,
        execution_tier=tier,
        provider_name=provider.provider_name,
        model_name=resolved_model,
        config=config,
        budget_summary=cost_tracker.get_summary(),
        trials_per_candidate=trials_per_candidate,
        total_target_calls=total_calls,
        evaluations=tuple(evaluations),
        aggregate_metrics={
            "tasks_evaluated": len(evaluations),
            "ground_truth_matches": matched_count,
            "statistically_supported_minimality": stat_supp_count,
            "mean_precision": round(sum(valid_precisions) / len(valid_precisions), 4) if valid_precisions else None,
            "mean_recall": round(sum(valid_recalls) / len(valid_recalls), 4) if valid_recalls else None,
        },
        summary=summary_text,
    )

    out_p = Path(output_dir)
    json_path = out_p / "model_pilot_report.json"
    md_path = out_p / "model_pilot_report.md"
    write_model_pilot_reports(report, json_path=json_path, md_path=md_path)

    return report
