"""Execution runner and three-way comparative benchmark orchestrator for Phase 4.6."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from typing import Any, Callable, Sequence

from agentxray.agent_harness.agent import AgentExecutionTrace
from agentxray.agent_harness.cost_tracker import CostTracker, ExperimentBudgetConfig
from agentxray.llm_judge import LLMJudgeBaseline
from agentxray.models import (
    DiagnosticOutcome,
    ExperimentConfig,
    ExploratoryBenchmarkReport,
    InputCase,
    MechanismEvaluationResult,
    SearchStatus,
    StochasticMinimalityStatus,
)
from agentxray.providers.base import (
    Message,
    ModelProvider,
    ModelResponse,
    ProviderError,
    ToolDefinition,
    UsageMetadata,
)
from agentxray.providers.fake import FakeProvider
from agentxray.runner import InterventionRunner
from agentxray.statistics import (
    compute_contrast_sensitivity_bounds,
    compute_discovery_level_fdr,
    compute_mean_mechanism_fdp,
    mcnemar_exact_test,
)
from agentxray.stochastic_baselines import run_stochastic_greedy_ablation
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.multi_mechanism.tasks import (
    MECHANISM_REGISTRY,
    MechanismTaskSpec,
    MechanismTargetAdapter,
    list_mechanism_tasks,
)

logger = logging.getLogger(__name__)


def score_diagnostic_outcome(
    mechanism_id: str,
    method_name: str,
    candidate_set: tuple[str, ...] | None,
    is_valid_diagnosis: bool,
    ground_truth_trigger: tuple[str, ...],
    is_null_case: bool,
    verification_status: str = "not_applicable",
    termination_reason: str = "normal_completion",
    target_calls_used: int = 0,
    model_requests: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    diagnostics: str = "",
) -> DiagnosticOutcome:
    """Score any method's candidate prediction against ground truth under Phase 4.6 protocol rules."""
    if not is_valid_diagnosis:
        false_attribution = None
        exact_recovery = None if termination_reason == "infra_failure" else 0
    else:
        assert candidate_set is not None
        if is_null_case:
            # Null task (S_true == ())
            if len(candidate_set) == 0:
                false_attribution = 0
                exact_recovery = 1
            else:
                false_attribution = 1
                exact_recovery = 0
        else:
            # Active task (S_true != ())
            true_set = set(ground_truth_trigger)
            pred_set = set(candidate_set)
            spurious = pred_set - true_set

            # False attribution occurs iff any candidate element is not in true_set
            # (Note: valid empty prediction on active task has spurious=set(), so Y=0)
            false_attribution = 1 if len(spurious) > 0 else 0
            exact_recovery = 1 if pred_set == true_set else 0

    return DiagnosticOutcome(
        mechanism_id=mechanism_id,
        method_name=method_name,
        predicted_candidate_set=candidate_set,
        is_valid_diagnosis=is_valid_diagnosis,
        false_attribution_indicator=false_attribution,
        exact_recovery_indicator=exact_recovery,
        verification_status=verification_status,
        termination_reason=termination_reason,
        target_calls_used=target_calls_used,
        model_requests=model_requests,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
        diagnostics=diagnostics,
    )


def run_with_infra_guard(
    fn: Callable[[], DiagnosticOutcome],
    mechanism_id: str,
    method_name: str,
    seed: int | None = None,
    deviation_log: list[dict[str, Any]] | None = None,
) -> DiagnosticOutcome:
    """Execute a diagnostic method with single identical-seed rerun upon infrastructure failure."""
    if seed is not None:
        random.seed(seed)
    try:
        return fn()
    except (ProviderError, OSError, ConnectionError, TimeoutError) as first_err:
        entry = {
            "mechanism_id": mechanism_id,
            "method_name": method_name,
            "event": "INFRA_FAILURE_FIRST_ATTEMPT",
            "error": str(first_err),
        }
        if deviation_log is not None:
            deviation_log.append(entry)
        logger.warning(f"Infrastructure failure on {mechanism_id} ({method_name}): {first_err}. Attempting single rerun.")

        # Single identical-seed rerun
        if seed is not None:
            random.seed(seed)
        try:
            return fn()
        except Exception as second_err:
            rerun_entry = {
                "mechanism_id": mechanism_id,
                "method_name": method_name,
                "event": "INFRA_FAILURE_UNRESOLVED",
                "error": str(second_err),
            }
            if deviation_log is not None:
                deviation_log.append(rerun_entry)
            logger.error(f"Unresolved infrastructure failure on {mechanism_id} ({method_name}): {second_err}.")

            return DiagnosticOutcome(
                mechanism_id=mechanism_id,
                method_name=method_name,
                predicted_candidate_set=None,
                is_valid_diagnosis=False,
                false_attribution_indicator=None,
                exact_recovery_indicator=None,
                verification_status="not_applicable",
                termination_reason="infra_failure",
                diagnostics=f"Unresolved infrastructure failure: {second_err}",
            )


class ScriptedJudgeProvider(ModelProvider):
    """Deterministic scripted judge provider for offline pipeline orchestration validation only.

    CRITICAL SCIENTIFIC INTEGRITY GUARANTEE:
    This provider is strictly for testing benchmark pipeline orchestration, missingness accounting,
    and statistical arithmetic offline without incurring inference costs.
    It does NOT measure or represent real LLM diagnostic capability.
    """

    def __init__(
        self,
        decision_map: dict[str, Sequence[str]] | None = None,
        default_prediction: Sequence[str] = (),
        model_name: str = "fake-scripted-judge",
    ) -> None:
        self._decision_map = decision_map or {}
        self._default_prediction = tuple(default_prediction)
        self._model_name = model_name
        self.request_count = 0

    @property
    def supports_retry_limits(self) -> bool:
        return True

    @property
    def is_trusted_for_hard_budgets(self) -> bool:
        return True

    @property
    def call_count(self) -> int:
        return self.request_count

    @property
    def provider_name(self) -> str:
        return "scripted_judge"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] = (),
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int | None = None,
    ) -> ModelResponse:
        self.request_count += 1
        prompt_len = sum(len(m.content) for m in messages) // 4 + 20
        user_msg = messages[-1].content if messages else ""

        pred = list(self._default_prediction)
        for key, p in self._decision_map.items():
            if key in user_msg:
                pred = list(p)
                break

        payload = {
            "culprit_element_ids": pred,
            "rationale": "Scripted offline methodological pipeline validation response.",
        }
        content = json.dumps(payload)
        comp_len = len(content) // 4 + 5

        return ModelResponse(
            content=content,
            tool_calls=(),
            usage=UsageMetadata(
                prompt_tokens=prompt_len,
                completion_tokens=comp_len,
                total_tokens=prompt_len + comp_len,
                estimated_cost_usd=round((prompt_len * 0.15 + comp_len * 0.60) / 1_000_000, 7),
            ),
            finish_reason="stop",
            model_identifier=self._model_name,
            latency_ms=1.0,
        )


def derive_mechanism_seed(mechanism_id: str, root_seed: int = 42) -> int:
    """Derive a reproducible 32-bit mechanism-specific seed from root seed and mechanism ID.

    Uses the first 4 bytes of SHA-256 (interpreted as big-endian integer) combined with
    the root seed multiplier (root_seed * 10007), masked to 32 bits (0xFFFFFFFF).
    This guarantees:
    1. Perfect stability across processes and runs (unlike Python's salted hash()).
    2. Distinct seeds across all benchmark mechanisms for any fixed root seed.
    3. Distinct seeds across different root seeds.
    4. Pairwise seed sharing between methods evaluating the same mechanism and root seed.
    """
    digest = hashlib.sha256(mechanism_id.encode("utf-8")).digest()
    mech_hash = int.from_bytes(digest[:4], "big")
    return (root_seed * 10007 + mech_hash) & 0xFFFFFFFF


def evaluate_mechanism_three_way(
    spec: MechanismTaskSpec,
    provider: ModelProvider | None = None,
    cost_tracker: CostTracker | None = None,
    trials_per_candidate: int = 5,
    max_trials: int = 50,
    seed: int = 42,
    deviation_log: list[dict[str, Any]] | None = None,
) -> MechanismEvaluationResult:
    """Evaluate Active Delta Debugging, Greedy Ablation, and LLM Judge on a single mechanism."""
    prov = provider or FakeProvider(model_name="fake-multi-mech")
    gt = spec.get_ground_truth()

    # Deterministic mechanism-specific seed derived from root seed and mechanism_id
    mech_seed = derive_mechanism_seed(spec.mechanism_id, root_seed=seed)

    # 1. Instantiate Target and Initial Input with seeded candidate permutation
    target = spec.create_target(provider=prov, cost_tracker=cost_tracker)
    initial_input = spec.create_initial_input(seed=mech_seed)

    # 2. Run initial unablated execution (Trial 1) to generate baseline trace
    base_res = target.execute(initial_input)
    init_trace = target.last_trace or AgentExecutionTrace(
        messages=(),
        final_response=base_res.output_message,
        finish_reason="stop",
        error_message=base_res.output_message if not base_res.is_success else None,
    )
    init_requests = target.call_count
    init_tokens = init_trace.total_tokens
    init_cost = init_trace.estimated_cost_usd

    # ----------------------------------------------------
    # Method A: Active Stochastic Delta Debugging
    # ----------------------------------------------------
    def _run_active_dd() -> DiagnosticOutcome:
        dd_target = spec.create_target(provider=prov, cost_tracker=cost_tracker)
        cfg = ExperimentConfig(experiment_name=f"{spec.mechanism_id}_active_dd", max_trials=max_trials)
        runner = InterventionRunner(config=cfg, target_fn=dd_target.execute)
        minimizer = StochasticDeltaDebugMinimizer(
            runner=runner,
            trials_per_candidate=trials_per_candidate,
            failure_threshold=0.30,
        )
        cond = minimizer.minimize(initial_input)

        # Task-semantic validity determination:
        if cond.status == SearchStatus.INFRA_FAILURE:
            is_valid = False
            cand_set = None
            term_reason = "infra_failure"
        elif cond.status in (SearchStatus.BUDGET_EXHAUSTED, SearchStatus.TIMEOUT) and len(cond.subset_elements) == 0:
            is_valid = False
            cand_set = None
            term_reason = "budget_exhausted" if cond.status == SearchStatus.BUDGET_EXHAUSTED else "timeout"
        else:
            is_valid = True
            cand_set = tuple(cond.subset_elements)
            term_reason = "normal_completion"
            if cond.status == SearchStatus.BUDGET_EXHAUSTED:
                term_reason = "budget_exhausted"
            elif cond.status == SearchStatus.TIMEOUT:
                term_reason = "timeout"

        return score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="active_dd",
            candidate_set=cand_set,
            is_valid_diagnosis=is_valid,
            ground_truth_trigger=spec.planted_trigger_elements,
            is_null_case=spec.is_null_case,
            verification_status=cond.minimality_status.value,
            termination_reason=term_reason,
            target_calls_used=cond.target_calls_used,
            model_requests=cond.total_model_requests or cond.target_calls_used,
            diagnostics=cond.status_message,
        )

    dd_outcome = run_with_infra_guard(
        _run_active_dd,
        mechanism_id=spec.mechanism_id,
        method_name="active_dd",
        seed=mech_seed,
        deviation_log=deviation_log,
    )

    # ----------------------------------------------------
    # Method B: Active Stochastic Greedy Ablation
    # ----------------------------------------------------
    def _run_greedy() -> DiagnosticOutcome:
        greedy_target = spec.create_target(provider=prov, cost_tracker=cost_tracker)
        cfg = ExperimentConfig(experiment_name=f"{spec.mechanism_id}_greedy", max_trials=max_trials)
        runner = InterventionRunner(config=cfg, target_fn=greedy_target.execute)
        res = run_stochastic_greedy_ablation(
            initial_input=initial_input,
            runner=runner,
            trials_per_candidate=trials_per_candidate,
            failure_threshold=0.30,
        )
        if res.budget_exhausted and len(res.discovered_elements) == 0:
            is_valid = False
            cand_set = None
            term_reason = "budget_exhausted"
        else:
            is_valid = True
            cand_set = tuple(res.discovered_elements)
            term_reason = "budget_exhausted" if res.budget_exhausted else "normal_completion"

        return score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="greedy_ablation",
            candidate_set=cand_set,
            is_valid_diagnosis=is_valid,
            ground_truth_trigger=spec.planted_trigger_elements,
            is_null_case=spec.is_null_case,
            verification_status=res.minimality_status.value,
            termination_reason=term_reason,
            target_calls_used=res.target_calls,
            model_requests=res.total_model_requests or res.target_calls,
        )

    greedy_outcome = run_with_infra_guard(
        _run_greedy,
        mechanism_id=spec.mechanism_id,
        method_name="greedy_ablation",
        seed=mech_seed,
        deviation_log=deviation_log,
    )

    # ----------------------------------------------------
    # Method C: Passive LLM Judge
    # ----------------------------------------------------
    def _run_judge() -> DiagnosticOutcome:
        judge = LLMJudgeBaseline(provider=prov, cost_tracker=cost_tracker, max_retries=2)
        judge_res = judge.diagnose(initial_input, init_trace)

        return score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="llm_judge",
            candidate_set=judge_res.predicted_candidate_set,
            is_valid_diagnosis=judge_res.is_valid_diagnosis,
            ground_truth_trigger=spec.planted_trigger_elements,
            is_null_case=spec.is_null_case,
            verification_status="not_applicable",
            termination_reason=judge_res.termination_reason,
            target_calls_used=0,
            model_requests=judge_res.attempts_used,
            prompt_tokens=judge_res.prompt_tokens,
            completion_tokens=judge_res.completion_tokens,
            total_tokens=judge_res.total_tokens,
            estimated_cost_usd=judge_res.estimated_cost_usd,
            diagnostics=judge_res.rationale,
        )

    judge_outcome = run_with_infra_guard(
        _run_judge,
        mechanism_id=spec.mechanism_id,
        method_name="llm_judge",
        seed=mech_seed,
        deviation_log=deviation_log,
    )

    return MechanismEvaluationResult(
        mechanism_id=spec.mechanism_id,
        mechanism_name=spec.name,
        category=spec.category,
        is_null_case=spec.is_null_case,
        is_conjunctive=spec.is_conjunctive,
        initial_elements=initial_input.elements,
        ground_truth_trigger=spec.planted_trigger_elements,
        active_dd_outcome=dd_outcome,
        greedy_ablation_outcome=greedy_outcome,
        llm_judge_outcome=judge_outcome,
        initial_trace_requests=init_requests,
        initial_trace_tokens=init_tokens,
        initial_trace_cost_usd=init_cost,
    )


def run_multi_mechanism_benchmark(
    provider: ModelProvider | None = None,
    cost_tracker: CostTracker | None = None,
    mechanisms: Sequence[MechanismTaskSpec] | None = None,
    trials_per_candidate: int = 5,
    max_trials: int = 50,
    seed: int = 42,
    experiment_name: str = "Phase_4_6_Exploratory_Benchmark",
) -> ExploratoryBenchmarkReport:
    """Run full Phase 4.6 exploratory benchmark across all mechanisms."""
    specs = list(mechanisms) if mechanisms is not None else list_mechanism_tasks()
    prov = provider or FakeProvider(model_name="fake-multi-mech")
    deviation_log: list[dict[str, Any]] = []

    evaluations: list[MechanismEvaluationResult] = []
    for spec in specs:
        res = evaluate_mechanism_three_way(
            spec=spec,
            provider=prov,
            cost_tracker=cost_tracker,
            trials_per_candidate=trials_per_candidate,
            max_trials=max_trials,
            seed=seed,
            deviation_log=deviation_log,
        )
        evaluations.append(res)

    n_total = len(evaluations)

    # Count infrastructure failures (asymmetric or mutual)
    infra_failed_ids = {
        ev.mechanism_id
        for ev in evaluations
        if ev.active_dd_outcome.termination_reason == "infra_failure"
        or ev.llm_judge_outcome.termination_reason == "infra_failure"
    }
    n_infra = len(infra_failed_ids)
    n_evaluable = n_total - n_infra

    # Filter to retained analysis cohort
    evaluable_evs = [ev for ev in evaluations if ev.mechanism_id not in infra_failed_ids]

    # Extract false attribution indicators for paired contrast
    y_judge = [ev.llm_judge_outcome.false_attribution_indicator for ev in evaluable_evs]
    y_activedd = [ev.active_dd_outcome.false_attribution_indicator for ev in evaluable_evs]

    # Complete-case pairs
    complete_pairs = [(yj, ya) for yj, ya in zip(y_judge, y_activedd) if yj is not None and ya is not None]
    n_complete = len(complete_pairs)

    # Discordant counts for McNemar test:
    # b: Judge false attribution (1), ActiveDD correct (0)
    # c: Judge correct (0), ActiveDD false attribution (1)
    b = sum(1 for yj, ya in complete_pairs if yj == 1 and ya == 0)
    c = sum(1 for yj, ya in complete_pairs if yj == 0 and ya == 1)

    mcnemar_res = mcnemar_exact_test(b=b, c=c, alternative="greater", alpha=0.05)
    sens_res = compute_contrast_sensitivity_bounds(y_judge=y_judge, y_activedd=y_activedd)

    # Secondary metrics:
    pred_dd = [ev.active_dd_outcome.predicted_candidate_set for ev in evaluable_evs]
    pred_greedy = [ev.greedy_ablation_outcome.predicted_candidate_set for ev in evaluable_evs]
    pred_judge = [ev.llm_judge_outcome.predicted_candidate_set for ev in evaluable_evs]
    gts = [ev.ground_truth_trigger for ev in evaluable_evs]

    fdr_dd = compute_discovery_level_fdr(pred_dd, gts)
    fdr_greedy = compute_discovery_level_fdr(pred_greedy, gts)
    fdr_judge = compute_discovery_level_fdr(pred_judge, gts)

    fdp_dd = compute_mean_mechanism_fdp(pred_dd, gts)
    fdp_greedy = compute_mean_mechanism_fdp(pred_greedy, gts)
    fdp_judge = compute_mean_mechanism_fdp(pred_judge, gts)

    rec_dd = sum(1 for ev in evaluable_evs if ev.active_dd_outcome.exact_recovery_indicator == 1) / max(1, n_evaluable)
    rec_greedy = sum(1 for ev in evaluable_evs if ev.greedy_ablation_outcome.exact_recovery_indicator == 1) / max(1, n_evaluable)
    rec_judge = sum(1 for ev in evaluable_evs if ev.llm_judge_outcome.exact_recovery_indicator == 1) / max(1, n_evaluable)

    req_dd = [ev.active_dd_outcome.model_requests for ev in evaluable_evs]
    req_greedy = [ev.greedy_ablation_outcome.model_requests for ev in evaluable_evs]
    req_judge = [ev.llm_judge_outcome.model_requests for ev in evaluable_evs]

    secondary = {
        "pooled_fdr": {"active_dd": fdr_dd, "greedy_ablation": fdr_greedy, "llm_judge": fdr_judge},
        "mean_fdp": {"active_dd": fdp_dd, "greedy_ablation": fdp_greedy, "llm_judge": fdp_judge},
        "exact_recovery_rate": {
            "active_dd": round(rec_dd, 4),
            "greedy_ablation": round(rec_greedy, 4),
            "llm_judge": round(rec_judge, 4),
        },
        "mean_requests": {
            "active_dd": round(sum(req_dd) / max(1, len(req_dd)), 2),
            "greedy_ablation": round(sum(req_greedy) / max(1, len(req_greedy)), 2),
            "llm_judge": round(sum(req_judge) / max(1, len(req_judge)), 2),
        },
        "protocol_deviations": deviation_log,
    }

    summary_text = (
        f"Evaluated {n_total} mechanisms across 3 methods. Retained {n_evaluable} mechanisms (excluded {n_infra} infra failures). "
        f"Complete-case risk difference Delta_Risk = {sens_res.complete_case_risk_difference}. "
        f"Exact directional McNemar test p-value = {mcnemar_res.p_value:.5f} (b={b}, c={c}). "
        f"Cohort sensitivity bounds: [{sens_res.lower_bound}, {sens_res.upper_bound}]."
    )

    return ExploratoryBenchmarkReport(
        experiment_name=experiment_name,
        n_total=n_total,
        n_infra=n_infra,
        n_evaluable=n_evaluable,
        n_complete=n_complete,
        n_miss_judge=sens_res.n_miss_judge,
        n_miss_activedd=sens_res.n_miss_activedd,
        complete_case_risk_difference=sens_res.complete_case_risk_difference,
        mcnemar_b=b,
        mcnemar_c=c,
        mcnemar_n_discordant=mcnemar_res.n_discordant,
        mcnemar_odds_ratio=mcnemar_res.odds_ratio,
        mcnemar_p_value=mcnemar_res.p_value,
        mcnemar_statistically_significant=mcnemar_res.statistically_significant,
        sensitivity_upper_bound=sens_res.upper_bound,
        sensitivity_lower_bound=sens_res.lower_bound,
        evaluations=tuple(evaluations),
        secondary_metrics=secondary,
        summary=summary_text,
    )
