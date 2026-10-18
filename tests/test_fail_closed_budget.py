"""Regression tests for fail-closed budget exhaustion and accurate request telemetry (Phase 4.6H.7).

Verifies:
1. MechanismTargetAdapter halts fail-closed on budget exhaustion without evaluating predicates or returning results.
2. StochasticDeltaDebugMinimizer cleanly halts without counting blocked trials when target budget is exhausted.
3. Active DD accurately tracks model requests and reports invalid diagnosis on pre-candidate exhaustion.
4. Greedy ablation accurately tracks model requests and reports invalid diagnosis on pre-candidate exhaustion.
5. Multi-turn agents accurately report distinct target_calls_used and model_requests without fallback substitution.
6. Global budget cap enforcement halts dispatches across multiple methods without exceeding the cap.
7. Null controls remain valid and provider infrastructure errors raise RuntimeError triggering infra guards.
8. Benchmark reporting includes mean_target_calls alongside mean_requests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentxray.agent_harness.agent import AgentExecutionTrace, ControlledAgent
from agentxray.agent_harness.cost_tracker import (
    CostBudgetExceededError,
    CostTracker,
    ExperimentBudgetConfig,
)
from agentxray.agent_harness.evaluator import AgentOutcomeEvaluator
from agentxray.models import (
    ExperimentConfig,
    InputCase,
    SearchStatus,
    TargetOutcome,
    TargetResult,
)
from agentxray.providers.base import (
    Message,
    ModelProvider,
    ModelResponse,
    ProviderTimeoutError,
    ToolCall,
    ToolDefinition,
    UsageMetadata,
)
from agentxray.providers.fake import FakeProvider
from agentxray.runner import InterventionRunner
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.multi_mechanism.runner import (
    evaluate_mechanism_three_way,
    run_multi_mechanism_benchmark,
)
from benchmarks.multi_mechanism.tasks import (
    MechanismTaskSpec,
    MechanismTargetAdapter,
    get_mechanism_task,
)


class TestMechanismTargetFailClosed:
    """Verifies Fix 1: Target adapter fails closed immediately on budget exhaustion."""

    def test_target_adapter_raises_on_budget_exhaustion(self) -> None:
        cfg = ExperimentBudgetConfig(
            max_model_requests=1,
            max_spend_usd=1.0,
            model_name="fake-test",
        )
        # Single-turn handler: gives immediate direct response without requiring tool call turn
        def direct_handler(messages: list[Message], tools: list[ToolDefinition]) -> ModelResponse:
            return ModelResponse(
                content="Direct calculation result 400.",
                tool_calls=(),
                usage=UsageMetadata(prompt_tokens=50, completion_tokens=10, total_tokens=60),
                finish_reason="stop",
                model_identifier="fake-test",
            )

        prov = FakeProvider(model_name="fake-test", custom_handler=direct_handler)
        tracker = CostTracker(cfg, provider=prov)
        spec = get_mechanism_task("mech_s01_type_mismatch")
        target = spec.create_target(provider=prov, cost_tracker=tracker)

        # Wrap predicate with a mock to verify it is NOT called when budget is exhausted
        orig_pred = target._failure_predicate
        mock_pred = MagicMock(side_effect=orig_pred)
        target._failure_predicate = mock_pred

        input_case = spec.create_initial_input(seed=42)

        # Call 1: Consumes 1 request (reaches max_model_requests=1)
        res1 = target.execute(input_case)
        assert isinstance(res1, TargetResult)
        assert target.call_count == 1
        assert mock_pred.call_count == 1
        assert tracker.total_requests == 1

        # Call 2: Budget is now exhausted; agent halts with finish_reason='budget_exhausted'
        with pytest.raises(CostBudgetExceededError) as exc_info:
            target.execute(input_case)

        assert "budget exhausted" in str(exc_info.value).lower()
        # Predicate MUST NOT have been called on Call 2
        assert mock_pred.call_count == 1
        # Call count MUST NOT have been incremented
        assert target.call_count == 1
        # Last trace MUST reflect budget exhaustion
        assert target.last_trace is not None
        assert target.last_trace.finish_reason == "budget_exhausted"


class TestMinimizerBudgetExhaustion:
    """Verifies that StochasticDeltaDebugMinimizer handles CostBudgetExceededError cleanly."""

    def test_minimizer_halts_without_counting_blocked_trials(self) -> None:
        cfg = ExperimentBudgetConfig(
            max_model_requests=1,
            max_spend_usd=1.0,
            model_name="fake-test",
        )
        prov = FakeProvider(model_name="fake-test")
        tracker = CostTracker(cfg, provider=prov)
        # Pre-exhaust the tracker
        tracker.total_requests = 1

        spec = get_mechanism_task("mech_s01_type_mismatch")
        target = spec.create_target(provider=prov, cost_tracker=tracker)
        exp_cfg = ExperimentConfig(experiment_name="test_exhausted_dd", max_trials=10)
        runner = InterventionRunner(config=exp_cfg, target_fn=target.execute)
        minimizer = StochasticDeltaDebugMinimizer(runner=runner, trials_per_candidate=2)

        initial_input = spec.create_initial_input(seed=42)
        cond = minimizer.minimize(initial_input)

        assert cond.status == SearchStatus.BUDGET_EXHAUSTED
        assert cond.target_calls_used == 0
        assert cond.candidate_evaluation.trials == 0
        assert cond.is_failure is False


class TestBaselinePartialUsageAccounting:
    """Verifies Task 1 & 2: Partial-usage telemetry preservation on mid-session baseline exhaustion."""

    def test_baseline_partial_usage_preserved_on_mid_session_exhaustion(self) -> None:
        """Verify that when Turn 1 succeeds and Turn 2 is blocked by budget, Turn 1's usage is preserved."""
        turn_1_usage = UsageMetadata(prompt_tokens=100, completion_tokens=25, total_tokens=125, estimated_cost_usd=0.00003)
        turn_2_usage = UsageMetadata(prompt_tokens=150, completion_tokens=30, total_tokens=180, estimated_cost_usd=0.00004)

        def multiturn_handler(messages: list[Message], tools: list[ToolDefinition]) -> ModelResponse:
            has_tool_result = any(m.role == "tool" for m in messages)
            if not has_tool_result:
                return ModelResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            call_id="call_01",
                            name="calculator",
                            arguments={"operation": "add", "a": 150, "b": 250},
                        ),
                    ),
                    usage=turn_1_usage,
                    finish_reason="tool_calls",
                    model_identifier="fake-multiturn",
                )
            else:
                return ModelResponse(
                    content="The total sum calculated is 400.",
                    tool_calls=(),
                    usage=turn_2_usage,
                    finish_reason="stop",
                    model_identifier="fake-multiturn",
                )

        prov = FakeProvider(model_name="fake-multiturn", custom_handler=multiturn_handler)
        # Cap is set to exactly 1 request: Turn 1 succeeds, Turn 2 is blocked before dispatch
        cfg = ExperimentBudgetConfig(
            max_model_requests=1,
            max_spend_usd=1.0,
            model_name="fake-multiturn",
        )
        tracker = CostTracker(cfg, provider=prov)
        spec = get_mechanism_task("mech_null_clean")

        res = evaluate_mechanism_three_way(
            spec=spec,
            provider=prov,
            cost_tracker=tracker,
            trials_per_candidate=2,
            max_trials=10,
            seed=42,
        )

        # Baseline execution was halted mid-session after Turn 1 succeeded
        # Telemetry MUST preserve Turn 1 usage rather than zeroing it
        assert res.initial_trace_requests == 1
        assert res.initial_trace_tokens == 125
        assert res.initial_trace_cost_usd == pytest.approx(0.00003, rel=1e-5)
        assert tracker.total_requests == 1

        # All subsequent methods had 0 requests available -> must fail closed cleanly
        assert res.active_dd_outcome.is_valid_diagnosis is False
        assert res.active_dd_outcome.predicted_candidate_set is None
        assert res.active_dd_outcome.termination_reason == "budget_exhausted"
        assert res.active_dd_outcome.model_requests == 0
        assert res.active_dd_outcome.target_calls_used == 0

        assert res.greedy_ablation_outcome.is_valid_diagnosis is False
        assert res.greedy_ablation_outcome.predicted_candidate_set is None
        assert res.greedy_ablation_outcome.termination_reason == "budget_exhausted"
        assert res.greedy_ablation_outcome.model_requests == 0
        assert res.greedy_ablation_outcome.target_calls_used == 0

        assert res.llm_judge_outcome.is_valid_diagnosis is False
        assert res.llm_judge_outcome.termination_reason == "budget_exhausted"
        assert res.llm_judge_outcome.model_requests == 0

    def test_baseline_usage_zero_when_pre_exhausted_before_first_turn(self) -> None:
        """Verify that when budget is exhausted before Turn 1, baseline usage correctly reports zero."""
        prov = FakeProvider(model_name="fake-test")
        cfg = ExperimentBudgetConfig(
            max_model_requests=1,
            max_spend_usd=1.0,
            model_name="fake-test",
        )
        tracker = CostTracker(cfg, provider=prov)
        # Pre-exhaust the tracker before baseline execution starts
        tracker.total_requests = 1

        spec = get_mechanism_task("mech_s01_type_mismatch")
        res = evaluate_mechanism_three_way(
            spec=spec,
            provider=prov,
            cost_tracker=tracker,
            trials_per_candidate=2,
            max_trials=10,
            seed=42,
        )

        # 0 requests could be dispatched during baseline
        assert res.initial_trace_requests == 0
        assert res.initial_trace_tokens == 0
        assert res.initial_trace_cost_usd == 0.0
        assert tracker.total_requests == 1  # Blocked request not counted

        assert res.active_dd_outcome.is_valid_diagnosis is False
        assert res.active_dd_outcome.termination_reason == "budget_exhausted"
        assert res.greedy_ablation_outcome.is_valid_diagnosis is False
        assert res.greedy_ablation_outcome.termination_reason == "budget_exhausted"


class TestAccurateRequestTelemetry:
    """Verifies Fix 2: Accurate telemetry and removal of target-call fallback."""

    def test_active_dd_accurate_telemetry_and_invalid_diagnosis_on_pre_exhaustion(self) -> None:
        cfg = ExperimentBudgetConfig(
            max_model_requests=2,
            max_spend_usd=1.0,
            model_name="fake-test",
        )
        prov = FakeProvider(model_name="fake-test")
        tracker = CostTracker(cfg, provider=prov)
        spec = get_mechanism_task("mech_s01_type_mismatch")

        # Initial baseline execution uses 2 model requests (tool call + answer)
        res = evaluate_mechanism_three_way(
            spec=spec,
            provider=prov,
            cost_tracker=tracker,
            trials_per_candidate=2,
            max_trials=10,
            seed=42,
        )

        assert res.initial_trace_requests == 2
        # Active DD had 0 requests available -> must be invalid diagnosis with budget_exhausted
        assert res.active_dd_outcome.is_valid_diagnosis is False
        assert res.active_dd_outcome.predicted_candidate_set is None
        assert res.active_dd_outcome.termination_reason == "budget_exhausted"
        assert res.active_dd_outcome.target_calls_used == 0
        assert res.active_dd_outcome.model_requests == 0

        # Greedy ablation also had 0 requests available -> must be invalid diagnosis
        assert res.greedy_ablation_outcome.is_valid_diagnosis is False
        assert res.greedy_ablation_outcome.predicted_candidate_set is None
        assert res.greedy_ablation_outcome.termination_reason == "budget_exhausted"
        assert res.greedy_ablation_outcome.target_calls_used == 0
        assert res.greedy_ablation_outcome.model_requests == 0

    def test_multiturn_agent_reports_distinct_model_requests_and_target_calls(self) -> None:
        """Verifies that multi-turn dispatches report actual model requests, not target calls."""

        # Multi-turn handler: Turn 1 issues a tool call, Turn 2 gives final answer
        def multiturn_handler(messages: list[Message], tools: list[ToolDefinition]) -> ModelResponse:
            has_tool_result = any(m.role == "tool" for m in messages)
            if not has_tool_result:
                # Turn 1: request tool execution
                return ModelResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            call_id="call_01",
                            name="calculator",
                            arguments={"operation": "add", "a": 150, "b": 250},
                        ),
                    ),
                    usage=UsageMetadata(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                    finish_reason="tool_calls",
                    model_identifier="fake-multiturn",
                )
            else:
                # Turn 2: provide final response with expected substring "400"
                return ModelResponse(
                    content="The total sum calculated is 400.",
                    tool_calls=(),
                    usage=UsageMetadata(prompt_tokens=150, completion_tokens=30, total_tokens=180),
                    finish_reason="stop",
                    model_identifier="fake-multiturn",
                )

        prov = FakeProvider(model_name="fake-multiturn", custom_handler=multiturn_handler)
        cfg = ExperimentBudgetConfig(
            max_model_requests=50,
            max_spend_usd=1.0,
            model_name="fake-multiturn",
        )
        tracker = CostTracker(cfg, provider=prov)
        spec = get_mechanism_task("mech_null_clean")

        res = evaluate_mechanism_three_way(
            spec=spec,
            provider=prov,
            cost_tracker=tracker,
            trials_per_candidate=2,
            max_trials=10,
            seed=42,
        )

        # In null control, baseline evaluation in Active DD runs 2 trials
        # Each trial consumes 2 model requests (tool call + final answer)
        # Therefore: target_calls_used == 2, but model_requests == 4
        assert res.active_dd_outcome.target_calls_used == 2
        assert res.active_dd_outcome.model_requests == 4
        assert res.active_dd_outcome.model_requests != res.active_dd_outcome.target_calls_used

        # Initial trace also consumed 2 model requests
        assert res.initial_trace_requests == 2


class TestGlobalCapEnforcementAcrossMethods:
    """Verifies that global request cap is strictly respected across all methods."""

    def test_global_cap_strictly_enforced(self) -> None:
        cfg = ExperimentBudgetConfig(
            max_model_requests=5,
            max_spend_usd=1.0,
            model_name="fake-test",
        )
        prov = FakeProvider(model_name="fake-test")
        tracker = CostTracker(cfg, provider=prov)
        spec = get_mechanism_task("mech_s01_type_mismatch")

        res = evaluate_mechanism_three_way(
            spec=spec,
            provider=prov,
            cost_tracker=tracker,
            trials_per_candidate=2,
            max_trials=20,
            seed=42,
        )

        # Global tracker must NEVER exceed the configured cap of 5
        assert tracker.total_requests <= 5
        assert res.initial_trace_requests <= 5


class TestNullControlsAndSecondaryMetrics:
    """Verifies null control behavior and presence of mean_target_calls."""

    def test_null_controls_remain_valid(self) -> None:
        spec = get_mechanism_task("mech_null_clean")
        prov = FakeProvider(model_name="fake-test")
        cfg = ExperimentBudgetConfig(max_model_requests=50, max_spend_usd=1.0, model_name="fake-test")
        tracker = CostTracker(cfg, provider=prov)

        res = evaluate_mechanism_three_way(
            spec=spec,
            provider=prov,
            cost_tracker=tracker,
            trials_per_candidate=2,
            max_trials=10,
            seed=42,
        )

        assert res.active_dd_outcome.is_valid_diagnosis is True
        assert res.active_dd_outcome.predicted_candidate_set == ()
        assert res.active_dd_outcome.false_attribution_indicator == 0
        assert res.active_dd_outcome.exact_recovery_indicator == 1

        assert res.greedy_ablation_outcome.is_valid_diagnosis is True
        assert res.greedy_ablation_outcome.predicted_candidate_set == ()
        assert res.greedy_ablation_outcome.false_attribution_indicator == 0
        assert res.greedy_ablation_outcome.exact_recovery_indicator == 1

    def test_benchmark_report_contains_mean_target_calls(self) -> None:
        specs = [
            get_mechanism_task("mech_null_clean"),
            get_mechanism_task("mech_null_bloat"),
        ]
        prov = FakeProvider(model_name="fake-test")
        cfg = ExperimentBudgetConfig(max_model_requests=50, max_spend_usd=1.0, model_name="fake-test")
        tracker = CostTracker(cfg, provider=prov)

        report = run_multi_mechanism_benchmark(
            provider=prov,
            cost_tracker=tracker,
            mechanisms=specs,
            trials_per_candidate=2,
            max_trials=10,
            seed=42,
        )

        assert "mean_target_calls" in report.secondary_metrics
        assert "mean_requests" in report.secondary_metrics
        assert "active_dd" in report.secondary_metrics["mean_target_calls"]
        assert "greedy_ablation" in report.secondary_metrics["mean_target_calls"]
        assert "llm_judge" in report.secondary_metrics["mean_target_calls"]
        assert report.secondary_metrics["mean_target_calls"]["llm_judge"] == 0.0

    def test_infrastructure_error_not_reclassified_as_budget_exhaustion(self) -> None:
        """Verify that provider infrastructure failures result in infra_failure, not budget_exhausted."""
        prov = FakeProvider(model_name="fake-test", simulate_error=ProviderTimeoutError)
        cfg = ExperimentBudgetConfig(max_model_requests=50, max_spend_usd=1.0, model_name="fake-test")
        tracker = CostTracker(cfg, provider=prov)
        spec = get_mechanism_task("mech_null_clean")

        res = evaluate_mechanism_three_way(
            spec=spec,
            provider=prov,
            cost_tracker=tracker,
            trials_per_candidate=2,
            max_trials=10,
            seed=42,
        )

        assert res.llm_judge_outcome.termination_reason == "infra_failure"
        assert res.llm_judge_outcome.is_valid_diagnosis is False
        assert res.llm_judge_outcome.false_attribution_indicator is None
        assert res.llm_judge_outcome.exact_recovery_indicator is None
