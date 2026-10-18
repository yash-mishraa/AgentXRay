"""Experiment execution orchestrator and pre-flight dry-run for model-backed pilot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agentxray.agent_harness.cost_tracker import (
    CostTracker,
    ExperimentBudgetConfig,
    IsolatedAlgorithmCostTracker,
    IsolatedPhaseCostTracker,
)
from agentxray.verification import verify_candidate_stochastically
from agentxray.models import (
    AccessMode,
    ExecutionTier,
    ExperimentConfig,
    ModelPilotEvaluationResult,
    ModelPilotExperimentReport,
)
from agentxray.providers.base import ModelProvider
from agentxray.providers.fake import FakeProvider
from agentxray.providers.ollama_adapter import OllamaProvider
from agentxray.providers.openai_adapter import OpenAIProvider
from agentxray.reporting import (
    compute_stochastic_aggregate_metrics,
    format_cohort_summary_sentences,
    write_model_pilot_reports,
)
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
    seed: int = 42,
) -> dict[str, Any]:
    """Inspect and validate experiment parameters and estimate costs without contacting any provider."""
    # Check credentials
    api_key_set = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if not api_key_set:
        env_path = Path(".env")
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:].strip()
                if line.startswith("OPENAI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip("'\"")
                    if val:
                        api_key_set = True
                        break

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
        {
            "task_id": "pilot_calibrated_currency",
            "name": "Model Pilot Calibrated Currency Syntax",
            "initial_elements_count": 4,
            "type": "Calibrated neural tool syntax intervention",
        },
    ]

    filter_key = (task_filter or "all").strip().lower()
    if filter_key in ("baseline", "pilot_baseline_calc", "control"):
        tasks_info = [all_tasks_meta[0]]
    elif filter_key in ("conflicting_prompt", "pilot_conflicting_prompt", "prompt"):
        tasks_info = [all_tasks_meta[1]]
    elif filter_key in ("distractor_override", "pilot_distractor_override", "distractor"):
        tasks_info = [all_tasks_meta[2]]
    elif filter_key in ("calibrated", "calibrated_currency", "currency", "currency_injection", "pilot_calibrated_currency"):
        tasks_info = [all_tasks_meta[3]]
    else:
        tasks_info = all_tasks_meta[:3]

    # Pre-flight cost estimation based on selected tasks and multi-algorithm benchmark suite
    # (Stochastic Delta Debugging + Greedy Ablation + Seeded Random Search)
    est_calls_per_task = min(max_trials, 15) + min(max_trials, 15) + min(max_trials, 20)
    nominal_calls = len(tasks_info) * est_calls_per_task
    planned_target_calls = min(nominal_calls, max_model_requests // 2)
    cost_est = tracker.estimate_preflight_cost(
        planned_trials=planned_target_calls,
        turns_per_trial=2,
        est_prompt_tokens_per_turn=250,
        est_completion_tokens_per_turn=50,
    )

    can_execute_live = (provider_type == "openai" and live and api_key_set) or (provider_type == "ollama")

    # Validate spend limit
    spend_cap_valid = (max_spend_usd <= 1.00) if (live and provider_type == "openai") else True
    status = "VALID" if ((provider_type in ("fake", "ollama") or can_execute_live) and spend_cap_valid) else (
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
        "seed": seed,
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
    seed: int = 42,
    max_discovery_requests: int | None = None,
    max_verification_requests: int | None = None,
    enable_independent_verification: bool = False,
    verification_trials: int | None = None,
    verification_tier: str = "exploratory",
    sample_size_rationale: str | None = None,
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
        Significance level for 1-minimality verification hypothesis tests.
    confidence : float
        Confidence level for Wilson and Newcombe uncertainty intervals.
    timeout_seconds : float
        Wall-clock timeout in seconds.
    output_dir : Path | str
        Directory to write output JSON and Markdown reports.
    seed : int
        RNG seed for random baselines and local model generation reproducibility.
    max_discovery_requests : int | None
        Independent request quota for discovery phase.
    max_verification_requests : int | None
        Independent request quota for verification phase.
    enable_independent_verification : bool
        Whether to run post-discovery independent verification audit.
    verification_trials : int | None
        Number of trials per condition in verification phase.
    verification_tier : str
        'exploratory' (default) or 'confirmatory'.
    sample_size_rationale : str | None
        Documentation / rationale status for verification sample size.

    Returns
    -------
    ModelPilotExperimentReport
        Complete validated experimental report. The aggregate_metrics dictionary contains:
        - tasks_evaluated: Total count of evaluated pilot tasks.
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
        - budget_exhausted_tasks: Count of evaluated tasks where the minimizer exhausted the trial budget.
        - timed_out_tasks: Count of evaluated tasks where the minimizer exceeded execution timeout.
        - mean_precision: Legacy mean precision across all evaluations where precision is not None.
        - mean_recall: Legacy mean recall across all evaluations where recall is not None.
        - mean_precision_evaluations: Denominator count for mean_precision.
        - mean_recall_evaluations: Denominator count for mean_recall.
    """
    if live and max_spend_usd > 1.00:
        raise ValueError(
            f"Safety safeguard violation: max_spend_usd is ${max_spend_usd:.2f}, "
            "but initial real-model validation pilots strictly mandate max_spend_usd <= $1.00 USD."
        )

    if enable_independent_verification:
        tier_norm = verification_tier.strip().lower()
        if tier_norm == "confirmatory":
            if verification_trials is None:
                raise ValueError(
                    "Confirmatory verification requires an explicitly configured sample size (verification_trials), "
                    "but none was provided. Defaulting to exploratory 5 trials is not permitted for confirmatory claims."
                )

    # 1. Resolve Provider and Execution Tier
    if provider_type == "openai":
        if not live:
            raise RuntimeError(
                "Live API calls require the explicit '--live' opt-in flag. "
                "Use '--dry-run' to inspect configuration or '--provider fake' for offline testing."
            )
        resolved_model = model_name or "gpt-4o-mini"
        provider = OpenAIProvider(model_name=resolved_model, timeout_seconds=timeout_seconds, seed=seed)
        provider.validate_configuration()
        tier = ExecutionTier.LIVE_MODEL
    elif provider_type == "ollama":
        resolved_model = model_name or "llama3.2:3b"
        provider = OllamaProvider(model_name=resolved_model, timeout_seconds=timeout_seconds, seed=seed)
        provider.validate_configuration()
        tier = ExecutionTier.LOCAL_MODEL
    elif provider_type == "fake":
        resolved_model = model_name or "fake-model-pilot"
        provider = FakeProvider(model_name=resolved_model)
        tier = ExecutionTier.FAKE_PROVIDER
    else:
        raise ValueError(f"Unknown provider_type: {provider_type!r}. Supported: 'fake', 'openai', 'ollama'.")

    if tier == ExecutionTier.LOCAL_MODEL and max_model_requests > 120:
        raise ValueError(
            f"Safety safeguard violation: max_model_requests is {max_model_requests}, "
            "but Phase 1G.2 local model pilots strictly mandate max_model_requests <= 120."
        )

    # 2. Configure Cost Tracker and Budgets
    algo_budget_cfg = ExperimentBudgetConfig(
        max_model_requests=max_model_requests,
        max_spend_usd=max_spend_usd,
        model_name=resolved_model,
        request_timeout_seconds=timeout_seconds,
        max_discovery_requests=max_discovery_requests,
        max_verification_requests=max_verification_requests,
    )

    config = ExperimentConfig(
        experiment_name=f"ModelPilot_{tier.value}_{resolved_model}",
        max_trials=max_trials,
        timeout_seconds=timeout_seconds,
        seed=seed,
        mode=AccessMode.CONTROLLED_TESTBED,
        metadata={
            "tier": tier.value,
            "provider": provider.provider_name,
            "model": resolved_model,
            "task_filter": task_filter,
            "trials_per_candidate": trials_per_candidate,
            "failure_threshold": failure_threshold,
            "seed": seed,
            "max_model_requests_per_algorithm": max_model_requests,
        },
    )

    # 3. Load Pilot Benchmark Tasks
    tasks = get_model_pilot_tasks(provider=provider, task_filter=task_filter)

    # Centralized global tracker accumulating resource consumption across all tasks and algorithms
    total_planned_runs = max(1, len(tasks) * 3)
    global_budget_cfg = ExperimentBudgetConfig(
        max_model_requests=max_model_requests * total_planned_runs,
        max_spend_usd=max_spend_usd,
        model_name=resolved_model,
        request_timeout_seconds=timeout_seconds,
    )
    global_cost_tracker = CostTracker(global_budget_cfg)

    evaluations: list[ModelPilotEvaluationResult] = []
    total_calls = 0
    algo_breakdown: list[dict[str, Any]] = []

    for target, initial_input, ground_truth in tasks:
        # 1. Run Stochastic Delta Debugging Minimizer
        target.reset_call_count()
        tracker_dd = IsolatedPhaseCostTracker(
            algorithm_config=algo_budget_cfg,
            phase="discovery",
            global_tracker=global_cost_tracker,
            enforce_global_requests=True,
        )
        target.agent.cost_tracker = tracker_dd

        runner_dd = InterventionRunner(config=config, target_fn=target.execute)
        minimizer = StochasticDeltaDebugMinimizer(
            runner=runner_dd,
            trials_per_candidate=trials_per_candidate,
            failure_threshold=failure_threshold,
            alpha=alpha,
            confidence=confidence,
        )
        discovered_dd = minimizer.minimize(initial_input)
        dd_disc_calls = runner_dd.trial_count
        dd_disc_reqs = tracker_dd.total_requests
        dd_verif_calls = 0
        dd_verif_reqs = 0
        dd_verif_status = "skipped"

        if enable_independent_verification:
            target.reset_call_count()
            tracker_dd_verif = IsolatedPhaseCostTracker(
                algorithm_config=algo_budget_cfg,
                phase="verification",
                global_tracker=global_cost_tracker,
                enforce_global_requests=True,
            )
            target.agent.cost_tracker = tracker_dd_verif
            runner_dd_verif = InterventionRunner(
                config=ExperimentConfig(
                    experiment_name=f"{config.experiment_name}_dd_verif",
                    max_trials=max_trials,
                    timeout_seconds=timeout_seconds,
                    seed=seed,
                ),
                target_fn=target.execute,
            )
            v_res = verify_candidate_stochastically(
                candidate=discovered_dd.subset_elements,
                initial_input=initial_input,
                runner=runner_dd_verif,
                trials_per_candidate=verification_trials or trials_per_candidate,
                failure_threshold=failure_threshold,
                alpha=alpha,
                confidence=confidence,
                cost_tracker=tracker_dd_verif,
                evidence_tier=verification_tier,
                sample_size_rationale=sample_size_rationale,
                trials_explicitly_set=(verification_trials is not None),
            )
            dd_verif_calls = runner_dd_verif.trial_count
            dd_verif_reqs = tracker_dd_verif.total_requests
            dd_verif_status = v_res.verification_status.value
            discovered_dd = discovered_dd.model_copy(update={
                "minimality_status": v_res.minimality_status,
                "verification_status": v_res.verification_status.value,
                "evidence_tier": v_res.evidence_tier.value,
                "sample_size_rationale": v_res.sample_size_rationale,
                "sample_size_validation_status": v_res.sample_size_validation_status,
                "confirmatory_power_established": v_res.confirmatory_power_established,
                "search_target_calls": dd_disc_calls,
                "verification_target_calls": dd_verif_calls,
                "target_calls_used": dd_disc_calls + dd_verif_calls,
                "search_model_requests": dd_disc_reqs,
                "verification_model_requests": dd_verif_reqs,
                "total_model_requests": dd_disc_reqs + dd_verif_reqs,
                "removal_evaluations": v_res.removal_evaluations if v_res.removal_evaluations else discovered_dd.removal_evaluations,
            })
        else:
            discovered_dd = discovered_dd.model_copy(update={
                "search_model_requests": dd_disc_reqs,
                "total_model_requests": dd_disc_reqs,
                "sample_size_validation_status": "not_applicable",
            })

        dd_disc_spend = tracker_dd.estimated_spend_usd
        dd_verif_spend = tracker_dd_verif.estimated_spend_usd if (enable_independent_verification and tracker_dd_verif is not None) else 0.0
        dd_total_spend = round(dd_disc_spend + dd_verif_spend, 7)

        total_calls += (dd_disc_calls + dd_verif_calls)
        algo_breakdown.append({
            "task_name": ground_truth.name,
            "algorithm": "delta_debugging",
            "model_requests": dd_disc_reqs + dd_verif_reqs,
            "target_calls": dd_disc_calls + dd_verif_calls,
            "discovery_requests": dd_disc_reqs,
            "verification_requests": dd_verif_reqs,
            "discovery_target_calls": dd_disc_calls,
            "verification_target_calls": dd_verif_calls,
            "discovery_spend_usd": dd_disc_spend,
            "verification_spend_usd": dd_verif_spend,
            "estimated_spend_usd": dd_total_spend,
            "verification_status": dd_verif_status,
            "evidence_tier": v_res.evidence_tier.value if enable_independent_verification else None,
            "sample_size_rationale": v_res.sample_size_rationale if enable_independent_verification else None,
            "sample_size_validation_status": v_res.sample_size_validation_status if enable_independent_verification else "not_applicable",
            "confirmatory_power_established": v_res.confirmatory_power_established if enable_independent_verification else False,
        })

        matches_gt, prec, rec, excess, missing = evaluate_model_pilot_ground_truth(ground_truth, discovered_dd)

        # 2. Run Stochastic Greedy Ablation Baseline
        target.reset_call_count()
        tracker_greedy = IsolatedPhaseCostTracker(
            algorithm_config=algo_budget_cfg,
            phase="discovery",
            global_tracker=global_cost_tracker,
            enforce_global_requests=True,
        )
        target.agent.cost_tracker = tracker_greedy

        runner_greedy = InterventionRunner(config=config, target_fn=target.execute)
        greedy_comp = run_stochastic_greedy_ablation(
            initial_input=initial_input,
            runner=runner_greedy,
            trials_per_candidate=trials_per_candidate,
            failure_threshold=failure_threshold,
            alpha=alpha,
            confidence=confidence,
        )
        greedy_disc_calls = runner_greedy.trial_count
        greedy_disc_reqs = tracker_greedy.total_requests
        greedy_verif_calls = 0
        greedy_verif_reqs = 0
        greedy_verif_status = "skipped"
        tracker_greedy_verif: IsolatedPhaseCostTracker | None = None

        if enable_independent_verification:
            target.reset_call_count()
            tracker_greedy_verif = IsolatedPhaseCostTracker(
                algorithm_config=algo_budget_cfg,
                phase="verification",
                global_tracker=global_cost_tracker,
                enforce_global_requests=True,
            )
            target.agent.cost_tracker = tracker_greedy_verif
            runner_greedy_verif = InterventionRunner(
                config=ExperimentConfig(
                    experiment_name=f"{config.experiment_name}_greedy_verif",
                    max_trials=max_trials,
                    timeout_seconds=timeout_seconds,
                    seed=seed,
                ),
                target_fn=target.execute,
            )
            v_res_g = verify_candidate_stochastically(
                candidate=greedy_comp.discovered_elements,
                initial_input=initial_input,
                runner=runner_greedy_verif,
                trials_per_candidate=verification_trials or trials_per_candidate,
                failure_threshold=failure_threshold,
                alpha=alpha,
                confidence=confidence,
                cost_tracker=tracker_greedy_verif,
                evidence_tier=verification_tier,
                sample_size_rationale=sample_size_rationale,
                trials_explicitly_set=(verification_trials is not None),
            )
            greedy_verif_calls = runner_greedy_verif.trial_count
            greedy_verif_reqs = tracker_greedy_verif.total_requests
            greedy_verif_status = v_res_g.verification_status.value
            greedy_comp = greedy_comp.model_copy(update={
                "minimality_status": v_res_g.minimality_status,
                "verification_status": v_res_g.verification_status.value,
                "evidence_tier": v_res_g.evidence_tier.value,
                "sample_size_rationale": v_res_g.sample_size_rationale,
                "sample_size_validation_status": v_res_g.sample_size_validation_status,
                "confirmatory_power_established": v_res_g.confirmatory_power_established,
                "search_target_calls": greedy_disc_calls,
                "verification_target_calls": greedy_verif_calls,
                "target_calls": greedy_disc_calls + greedy_verif_calls,
                "search_model_requests": greedy_disc_reqs,
                "verification_model_requests": greedy_verif_reqs,
                "total_model_requests": greedy_disc_reqs + greedy_verif_reqs,
            })
        else:
            greedy_comp = greedy_comp.model_copy(update={
                "search_model_requests": greedy_disc_reqs,
                "total_model_requests": greedy_disc_reqs,
                "sample_size_validation_status": "not_applicable",
            })

        greedy_disc_spend = tracker_greedy.estimated_spend_usd
        greedy_verif_spend = tracker_greedy_verif.estimated_spend_usd if (enable_independent_verification and tracker_greedy_verif is not None) else 0.0
        greedy_total_spend = round(greedy_disc_spend + greedy_verif_spend, 7)

        total_calls += (greedy_disc_calls + greedy_verif_calls)
        algo_breakdown.append({
            "task_name": ground_truth.name,
            "algorithm": "stochastic_greedy_ablation",
            "model_requests": greedy_disc_reqs + greedy_verif_reqs,
            "target_calls": greedy_disc_calls + greedy_verif_calls,
            "discovery_requests": greedy_disc_reqs,
            "verification_requests": greedy_verif_reqs,
            "discovery_target_calls": greedy_disc_calls,
            "verification_target_calls": greedy_verif_calls,
            "discovery_spend_usd": greedy_disc_spend,
            "verification_spend_usd": greedy_verif_spend,
            "estimated_spend_usd": greedy_total_spend,
            "verification_status": greedy_verif_status,
            "evidence_tier": v_res_g.evidence_tier.value if enable_independent_verification else None,
            "sample_size_rationale": v_res_g.sample_size_rationale if enable_independent_verification else None,
            "sample_size_validation_status": v_res_g.sample_size_validation_status if enable_independent_verification else "not_applicable",
            "confirmatory_power_established": v_res_g.confirmatory_power_established if enable_independent_verification else False,
        })
        g_matches, _, _, g_excess, _ = evaluate_model_pilot_ground_truth(ground_truth, greedy_comp)
        greedy_comp = greedy_comp.model_copy(update={
            "matches_ground_truth": g_matches,
            "irrelevant_elements_remaining": len(g_excess),
        })

        # 3. Run Stochastic Seeded Random Search Baseline
        target.reset_call_count()
        tracker_random = IsolatedPhaseCostTracker(
            algorithm_config=algo_budget_cfg,
            phase="discovery",
            global_tracker=global_cost_tracker,
            enforce_global_requests=True,
        )
        target.agent.cost_tracker = tracker_random

        runner_random = InterventionRunner(config=config, target_fn=target.execute)
        random_comp = run_stochastic_random_search(
            initial_input=initial_input,
            runner=runner_random,
            trials_per_candidate=trials_per_candidate,
            failure_threshold=failure_threshold,
            alpha=alpha,
            confidence=confidence,
            seed=seed,
            reserve_verification_budget=(not enable_independent_verification),
        )
        random_disc_calls = runner_random.trial_count
        random_disc_reqs = tracker_random.total_requests
        random_verif_calls = 0
        random_verif_reqs = 0
        random_verif_status = "skipped"
        tracker_random_verif: IsolatedPhaseCostTracker | None = None

        if enable_independent_verification:
            target.reset_call_count()
            tracker_random_verif = IsolatedPhaseCostTracker(
                algorithm_config=algo_budget_cfg,
                phase="verification",
                global_tracker=global_cost_tracker,
                enforce_global_requests=True,
            )
            target.agent.cost_tracker = tracker_random_verif
            runner_random_verif = InterventionRunner(
                config=ExperimentConfig(
                    experiment_name=f"{config.experiment_name}_random_verif",
                    max_trials=max_trials,
                    timeout_seconds=timeout_seconds,
                    seed=seed,
                ),
                target_fn=target.execute,
            )
            v_res_r = verify_candidate_stochastically(
                candidate=random_comp.discovered_elements,
                initial_input=initial_input,
                runner=runner_random_verif,
                trials_per_candidate=verification_trials or trials_per_candidate,
                failure_threshold=failure_threshold,
                alpha=alpha,
                confidence=confidence,
                cost_tracker=tracker_random_verif,
                evidence_tier=verification_tier,
                sample_size_rationale=sample_size_rationale,
                trials_explicitly_set=(verification_trials is not None),
            )
            random_verif_calls = runner_random_verif.trial_count
            random_verif_reqs = tracker_random_verif.total_requests
            random_verif_status = v_res_r.verification_status.value
            random_comp = random_comp.model_copy(update={
                "minimality_status": v_res_r.minimality_status,
                "verification_status": v_res_r.verification_status.value,
                "evidence_tier": v_res_r.evidence_tier.value,
                "sample_size_rationale": v_res_r.sample_size_rationale,
                "sample_size_validation_status": v_res_r.sample_size_validation_status,
                "confirmatory_power_established": v_res_r.confirmatory_power_established,
                "search_target_calls": random_disc_calls,
                "verification_target_calls": random_verif_calls,
                "target_calls": random_disc_calls + random_verif_calls,
                "search_model_requests": random_disc_reqs,
                "verification_model_requests": random_verif_reqs,
                "total_model_requests": random_disc_reqs + random_verif_reqs,
            })
        else:
            random_comp = random_comp.model_copy(update={
                "search_model_requests": random_disc_reqs,
                "total_model_requests": random_disc_reqs,
                "sample_size_validation_status": "not_applicable",
            })

        random_disc_spend = tracker_random.estimated_spend_usd
        random_verif_spend = tracker_random_verif.estimated_spend_usd if (enable_independent_verification and tracker_random_verif is not None) else 0.0
        random_total_spend = round(random_disc_spend + random_verif_spend, 7)

        total_calls += (random_disc_calls + random_verif_calls)
        algo_breakdown.append({
            "task_name": ground_truth.name,
            "algorithm": "stochastic_seeded_random",
            "model_requests": random_disc_reqs + random_verif_reqs,
            "target_calls": random_disc_calls + random_verif_calls,
            "discovery_requests": random_disc_reqs,
            "verification_requests": random_verif_reqs,
            "discovery_target_calls": random_disc_calls,
            "verification_target_calls": random_verif_calls,
            "discovery_spend_usd": random_disc_spend,
            "verification_spend_usd": random_verif_spend,
            "estimated_spend_usd": random_total_spend,
            "verification_status": random_verif_status,
            "evidence_tier": v_res_r.evidence_tier.value if enable_independent_verification else None,
            "sample_size_rationale": v_res_r.sample_size_rationale if enable_independent_verification else None,
            "sample_size_validation_status": v_res_r.sample_size_validation_status if enable_independent_verification else "not_applicable",
            "confirmatory_power_established": v_res_r.confirmatory_power_established if enable_independent_verification else False,
        })
        r_matches, _, _, r_excess, _ = evaluate_model_pilot_ground_truth(ground_truth, random_comp)
        random_comp = random_comp.model_copy(update={
            "matches_ground_truth": r_matches,
            "irrelevant_elements_remaining": len(r_excess),
        })

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

    metrics = compute_stochastic_aggregate_metrics(evaluations)

    summary_parts = [
        f"Executed {len(evaluations)} model-backed pilot tasks under execution tier [{tier.value.upper()}] "
        f"using provider '{provider.provider_name}' ({resolved_model}). "
        f"Total target invocations: {total_calls}. "
        f"Delta Debugging recovered ground-truth trigger on {metrics['ground_truth_matches']}/{len(evaluations)} tasks. "
        f"Statistically supported minimality established on {metrics['statistically_supported_minimality']}/{len(evaluations)} tasks."
    ]
    summary_parts.extend(format_cohort_summary_sentences(metrics))
    summary_text = " ".join(summary_parts)

    budget_summary = global_cost_tracker.get_summary()
    budget_summary["max_requests_per_algorithm"] = max_model_requests
    budget_summary["per_algorithm_usage"] = algo_breakdown

    report = ModelPilotExperimentReport(
        experiment_name=config.experiment_name,
        execution_tier=tier,
        provider_name=provider.provider_name,
        model_name=resolved_model,
        config=config,
        budget_summary=budget_summary,
        trials_per_candidate=trials_per_candidate,
        total_target_calls=total_calls,
        evaluations=tuple(evaluations),
        aggregate_metrics=metrics,
        summary=summary_text,
    )

    if output_dir is not None:
        out_p = Path(output_dir)
        json_path = out_p / "model_pilot_report.json"
        md_path = out_p / "model_pilot_report.md"
        write_model_pilot_reports(report, json_path=json_path, md_path=md_path)

    return report
