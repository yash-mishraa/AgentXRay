"""Tests for the first trace-evaluated prompt mechanism (Phase 4.6K.3).

Verifies:
1. Candidate materialization: candidate is present in actual prompt passed to provider when enabled, absent when ablated.
2. Observed failure: trace violating evaluator's criterion yields TargetOutcome.FAILURE.
3. Observed success: trace satisfying evaluator's criterion yields TargetOutcome.SUCCESS.
4. Counterfactual ablation path: full case fails, ablated case succeeds, and active DD isolates minimal trigger.
5. Predicate independence: contradictory predicate does not alter trace-evaluated outcomes.
6. Trace integrity: genuine trace records are preserved without synthetic predicate text injection.
7. Budget behavior: budget exhaustion remains fail-closed in trace-evaluated mode.
8. Infrastructure behavior: provider infrastructure errors raise RuntimeError without misclassification.
9. Synthetic regression: existing 18 synthetic mechanisms and registry semantics remain untouched.
"""

from __future__ import annotations

from typing import Sequence
from unittest.mock import MagicMock

import pytest

from agentxray.agent_harness.agent import AgentExecutionTrace, ControlledAgent
from agentxray.agent_harness.cost_tracker import (
    CostBudgetExceededError,
    CostTracker,
    ExperimentBudgetConfig,
)
from agentxray.agent_harness.evaluator import ToolCallingTaskEvaluator
from agentxray.agent_harness.tools import CALCULATOR_TOOL_DEF, MockToolEnvironment
from agentxray.models import (
    ExperimentConfig,
    InputCase,
    SearchStatus,
    StochasticMinimalityStatus,
    TargetOutcome,
    TargetResult,
)
from agentxray.providers.base import (
    Message,
    ModelResponse,
    ProviderTimeoutError,
    ToolCall,
    ToolDefinition,
    UsageMetadata,
)
from agentxray.providers.fake import FakeProvider
from agentxray.runner import InterventionRunner
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.multi_mechanism.tasks import (
    MECHANISM_REGISTRY,
    MechanismTargetAdapter,
    TargetMode,
    TraceEvaluatedPromptTaskSpec,
    create_trace_evaluated_prompt_task,
    get_mechanism_task,
    list_mechanism_tasks,
)


def make_prompt_sensitive_scripted_handler(trigger_text: str):
    """Create a scripted FakeProvider handler that branches on observable prompt content.

    SCIENTIFIC INTEGRITY DISCLAIMER:
    This scripted handler is strictly for testing harness orchestration, prompt materialization,
    and trace-evaluated outcome determination offline without inference cost. It simulates an
    agent that complies with prompt instructions, but does NOT measure real LLM reasoning.
    """
    def handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
        system_text = messages[0].content if messages and messages[0].role == "system" else ""
        has_tool_reply = any(m.role == "tool" for m in messages)

        # If the prohibitive constraint is present in the prompt, agent complies by avoiding tools
        if trigger_text in system_text:
            return ModelResponse(
                content="Estimated receipts sum without calculator: approximately $400.",
                tool_calls=(),
                usage=UsageMetadata(prompt_tokens=50, completion_tokens=12, total_tokens=62),
                finish_reason="stop",
                model_identifier="fake-scripted-agent",
            )

        # Counterfactual path: constraint is ablated. Agent proceeds with tool invocation.
        if not has_tool_reply:
            return ModelResponse(
                content="Invoking calculator tool...",
                tool_calls=(
                    ToolCall(call_id="call_calc", name="calculator", arguments={"expression": "150 + 250"}),
                ),
                usage=UsageMetadata(prompt_tokens=45, completion_tokens=15, total_tokens=60),
                finish_reason="tool_calls",
                model_identifier="fake-scripted-agent",
            )
        else:
            return ModelResponse(
                content="Calculation complete. Final verified total is 400.",
                tool_calls=(),
                usage=UsageMetadata(prompt_tokens=65, completion_tokens=12, total_tokens=77),
                finish_reason="stop",
                model_identifier="fake-scripted-agent",
            )

    return handler


class TestCandidateMaterialization:
    """Verifies candidate presence in InputCase directly alters prompt instructions passed to provider."""

    def test_candidate_in_prompt_when_enabled_and_absent_when_ablated(self) -> None:
        task = create_trace_evaluated_prompt_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()

        captured_messages: list[Sequence[Message]] = []

        def spy_handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            captured_messages.append(tuple(messages))
            return ModelResponse(
                content="Spy response.",
                tool_calls=(),
                usage=UsageMetadata(prompt_tokens=40, completion_tokens=5, total_tokens=45),
                finish_reason="stop",
                model_identifier="fake-spy",
            )

        prov = FakeProvider(model_name="fake-spy", custom_handler=spy_handler)
        target = task.create_target(provider=prov)

        # 1. Full input execution: candidate must be in system prompt
        target.execute(initial_input)
        assert len(captured_messages) == 1
        system_content_full = captured_messages[0][0].content
        assert trigger in system_content_full
        for d in task.candidate_elements:
            assert d in system_content_full

        # 2. Ablated input execution: candidate must be absent from system prompt
        ablated_input = initial_input.without(trigger)
        target.execute(ablated_input)
        assert len(captured_messages) == 2
        system_content_ablated = captured_messages[1][0].content
        assert trigger not in system_content_ablated
        # Distractors must remain present
        for d in task.candidate_elements:
            if d != trigger:
                assert d in system_content_ablated


class TestObservedOutcomes:
    """Verifies target outcome is strictly determined by ToolCallingTaskEvaluator on the trace."""

    def test_observed_failure_when_trace_violates_evaluator(self) -> None:
        task = create_trace_evaluated_prompt_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()

        handler = make_prompt_sensitive_scripted_handler(trigger)
        prov = FakeProvider(model_name="fake-agent", custom_handler=handler)
        target = task.create_target(provider=prov)

        res = target.execute(initial_input)
        assert res.outcome == TargetOutcome.FAILURE
        assert "Agent failed to invoke required tool 'calculator'" in res.output_message
        assert res.error_code == "AGENT_FAILURE"
        assert res.metadata.get("target_mode") == "trace_evaluated"
        assert target.last_trace is not None
        assert len(target.last_trace.tool_executions) == 0

    def test_observed_success_when_trace_satisfies_evaluator(self) -> None:
        task = create_trace_evaluated_prompt_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()
        ablated_input = initial_input.without(trigger)

        handler = make_prompt_sensitive_scripted_handler(trigger)
        prov = FakeProvider(model_name="fake-agent", custom_handler=handler)
        target = task.create_target(provider=prov)

        res = target.execute(ablated_input)
        assert res.outcome == TargetOutcome.SUCCESS
        assert "Task executed successfully" in res.output_message
        assert res.error_code is None
        assert res.metadata.get("target_mode") == "trace_evaluated"
        assert target.last_trace is not None
        assert len(target.last_trace.tool_executions) == 1
        assert target.last_trace.tool_executions[0].tool_name == "calculator"
        assert "400" in target.last_trace.final_response


class TestCounterfactualAblationAndMinimization:
    """Verifies counterfactual ablation path and Active Delta Debugging convergence on trace-evaluated target."""

    def test_counterfactual_ablation_branches_evaluator_outcome(self) -> None:
        task = create_trace_evaluated_prompt_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()

        handler = make_prompt_sensitive_scripted_handler(trigger)
        prov = FakeProvider(model_name="fake-agent", custom_handler=handler)
        target = task.create_target(provider=prov)

        # Full case fails
        assert target.execute(initial_input).outcome == TargetOutcome.FAILURE

        # Distractors-only case passes
        distractors = tuple(e for e in task.candidate_elements if e != trigger)
        assert target.execute(InputCase(case_id="clean", elements=distractors)).outcome == TargetOutcome.SUCCESS

        # Subset with trigger + 1 distractor fails
        sub_with_trigger = (trigger, distractors[0])
        assert target.execute(InputCase(case_id="sub_fail", elements=sub_with_trigger)).outcome == TargetOutcome.FAILURE

        # Subset of single distractor passes
        sub_distractor_only = (distractors[0],)
        assert target.execute(InputCase(case_id="sub_pass", elements=sub_distractor_only)).outcome == TargetOutcome.SUCCESS

    def test_active_delta_debugging_isolates_causal_trigger(self) -> None:
        task = create_trace_evaluated_prompt_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()

        handler = make_prompt_sensitive_scripted_handler(trigger)
        prov = FakeProvider(model_name="fake-agent", custom_handler=handler)
        target = task.create_target(provider=prov)

        cfg = ExperimentConfig(experiment_name="trace_prompt_dd_test", max_trials=20)
        runner = InterventionRunner(config=cfg, target_fn=target.execute)
        minimizer = StochasticDeltaDebugMinimizer(runner=runner, trials_per_candidate=1, failure_threshold=0.5)

        cond = minimizer.minimize(initial_input)
        assert cond.status == SearchStatus.COMPLETE
        assert cond.is_failure is True
        assert cond.minimality_status in (
            StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
            StochasticMinimalityStatus.OBSERVED_ONLY,
        )
        # Active DD must isolate exactly the planted trigger
        assert cond.subset_elements == (trigger,)


class TestPredicateIndependenceAndTraceIntegrity:
    """Verifies that contradictory failure predicates are ignored and trace integrity is preserved."""

    def test_contradictory_predicate_does_not_affect_outcomes(self) -> None:
        task = create_trace_evaluated_prompt_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()
        ablated_input = initial_input.without(trigger)

        handler = make_prompt_sensitive_scripted_handler(trigger)
        prov = FakeProvider(model_name="fake-agent", custom_handler=handler)

        # Contradictory predicate: claims success when trigger present, failure when absent
        mock_pred = MagicMock(side_effect=lambda ic: (False, "PREDICATE_FALSE_PASS") if trigger in ic.elements else (True, "PREDICATE_FALSE_FAIL"))

        target = task.create_target(provider=prov, failure_predicate=mock_pred)

        # Full configuration: evaluator fails -> target MUST return FAILURE despite predicate claiming pass
        res_full = target.execute(initial_input)
        assert res_full.outcome == TargetOutcome.FAILURE
        assert mock_pred.call_count == 0  # Predicate must not be called

        # Ablated configuration: evaluator passes -> target MUST return SUCCESS despite predicate claiming fail
        res_ablated = target.execute(ablated_input)
        assert res_ablated.outcome == TargetOutcome.SUCCESS
        assert mock_pred.call_count == 0

    def test_trace_error_message_integrity(self) -> None:
        task = create_trace_evaluated_prompt_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()

        handler = make_prompt_sensitive_scripted_handler(trigger)
        prov = FakeProvider(model_name="fake-agent", custom_handler=handler)
        leak_pred = MagicMock(return_value=(True, "SYNTHETIC_LEAK_STRING"))

        target = task.create_target(provider=prov, failure_predicate=leak_pred)
        res = target.execute(initial_input)

        assert target.last_trace is not None
        assert target.last_trace.error_message is None
        assert "SYNTHETIC_LEAK_STRING" not in target.last_trace.final_response
        assert "SYNTHETIC_LEAK_STRING" not in res.output_message


class TestFailClosedBudgetAndInfraErrors:
    """Verifies fail-closed budget and infrastructure error preservation on trace-evaluated prompt task."""

    def test_budget_exhaustion_halts_fail_closed(self) -> None:
        task = create_trace_evaluated_prompt_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()
        ablated_input = initial_input.without(trigger)

        handler = make_prompt_sensitive_scripted_handler(trigger)
        prov = FakeProvider(model_name="fake-agent", custom_handler=handler)
        tracker = CostTracker(ExperimentBudgetConfig(max_model_requests=2, max_spend_usd=1.0, model_name="fake-agent"), provider=prov)

        target = task.create_target(provider=prov, cost_tracker=tracker)

        # Ablated call consumes 2 turns/requests (turn 1 tool call + turn 2 answer)
        res1 = target.execute(ablated_input)
        assert res1.outcome == TargetOutcome.SUCCESS
        assert tracker.total_requests == 2

        # Subsequent call must halt fail-closed
        with pytest.raises(CostBudgetExceededError) as exc_info:
            target.execute(ablated_input)

        assert "budget exhausted" in str(exc_info.value).lower()

    def test_provider_infrastructure_error_raises_runtime_error(self) -> None:
        task = create_trace_evaluated_prompt_task()
        initial_input = task.create_initial_input()

        prov = FakeProvider(model_name="fake-crash", simulate_error=ProviderTimeoutError)
        target = task.create_target(provider=prov)

        with pytest.raises(RuntimeError, match="Controlled agent execution aborted due to provider infrastructure error"):
            target.execute(initial_input)


class TestSyntheticBenchmarkIsolation:
    """Verifies existing 18 synthetic mechanisms and registry semantics remain untouched."""

    def test_existing_18_mechanisms_unaffected(self) -> None:
        assert len(MECHANISM_REGISTRY) == 18
        assert len(list_mechanism_tasks()) == 18
        assert "trace_prompt_01_tool_prohibition" not in MECHANISM_REGISTRY

        # Confirm all 18 registered mechanisms default to SYNTHETIC_CONTROL
        for spec in list_mechanism_tasks():
            t = spec.create_target()
            assert t.target_mode == TargetMode.SYNTHETIC_CONTROL
            assert t._failure_predicate is not None
