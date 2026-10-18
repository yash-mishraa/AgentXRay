"""Tests for target execution modes and evaluator isolation (Phase 4.6K.2).

Verifies:
1. Default mode preserves current synthetic behavior.
2. Explicit synthetic-control mode preserves predicate-driven outcomes.
3. Trace-evaluated mode uses the evaluator's result even when a supplied predicate disagrees.
4. Trace-evaluated mode does not call the predicate.
5. Synthetic predicate error text is not injected into trace-evaluated traces.
6. Genuine trace error information is preserved.
7. Budget exhaustion remains fail-closed and does not become a false success.
8. MechanismTaskSpec propagates target_mode properly.
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
from agentxray.agent_harness.evaluator import ToolCallingTaskEvaluator
from agentxray.agent_harness.tools import (
    CALCULATOR_TOOL_DEF,
    DATABASE_LOOKUP_TOOL_DEF,
    MockToolEnvironment,
)
from agentxray.models import InputCase, TargetOutcome, TargetResult
from agentxray.providers.base import (
    Message,
    ModelResponse,
    ProviderTimeoutError,
    ToolCall,
    ToolDefinition,
    UsageMetadata,
)
from agentxray.providers.fake import FakeProvider
from benchmarks.multi_mechanism.tasks import (
    MechanismTargetAdapter,
    TargetMode,
    get_mechanism_task,
)


def _make_passing_direct_handler():
    """Handler returning direct final response with '400' without tool calls."""
    def handler(messages: list[Message], tools: list[ToolDefinition]) -> ModelResponse:
        return ModelResponse(
            content="Calculated receipts total: 400.",
            tool_calls=(),
            usage=UsageMetadata(prompt_tokens=40, completion_tokens=10, total_tokens=50),
            finish_reason="stop",
            model_identifier="fake-eval-test",
        )
    return handler


def _make_passing_tool_handler():
    """Handler that invokes calculator tool with '150 + 250' on turn 1, returns '400' on turn 2."""
    def handler(messages: list[Message], tools: list[ToolDefinition]) -> ModelResponse:
        # Check if there is already a tool message
        has_tool_reply = any(m.role == "tool" for m in messages)
        if not has_tool_reply:
            return ModelResponse(
                content="Calling calculator...",
                tool_calls=(
                    ToolCall(call_id="call_calc_1", name="calculator", arguments={"expression": "150 + 250"}),
                ),
                usage=UsageMetadata(prompt_tokens=40, completion_tokens=15, total_tokens=55),
                finish_reason="tool_calls",
                model_identifier="fake-eval-test",
            )
        return ModelResponse(
            content="Final calculated result is 400.",
            tool_calls=(),
            usage=UsageMetadata(prompt_tokens=60, completion_tokens=10, total_tokens=70),
            finish_reason="stop",
            model_identifier="fake-eval-test",
        )
    return handler


def _make_failing_tool_handler():
    """Handler that invokes calculator tool with invalid syntax '150 + $250'."""
    def handler(messages: list[Message], tools: list[ToolDefinition]) -> ModelResponse:
        has_tool_reply = any(m.role == "tool" for m in messages)
        if not has_tool_reply:
            return ModelResponse(
                content="Calling calculator with raw string...",
                tool_calls=(
                    ToolCall(call_id="call_calc_err", name="calculator", arguments={"expression": "150 + $250"}),
                ),
                usage=UsageMetadata(prompt_tokens=40, completion_tokens=15, total_tokens=55),
                finish_reason="tool_calls",
                model_identifier="fake-eval-test",
            )
        return ModelResponse(
            content="Calculation produced error.",
            tool_calls=(),
            usage=UsageMetadata(prompt_tokens=60, completion_tokens=10, total_tokens=70),
            finish_reason="stop",
            model_identifier="fake-eval-test",
        )
    return handler


def _make_omitted_answer_handler():
    """Handler that runs calculator but omits '400' in final response."""
    def handler(messages: list[Message], tools: list[ToolDefinition]) -> ModelResponse:
        has_tool_reply = any(m.role == "tool" for m in messages)
        if not has_tool_reply:
            return ModelResponse(
                content="Calling calculator...",
                tool_calls=(
                    ToolCall(call_id="call_calc_1", name="calculator", arguments={"expression": "150 + 250"}),
                ),
                usage=UsageMetadata(prompt_tokens=40, completion_tokens=15, total_tokens=55),
                finish_reason="tool_calls",
                model_identifier="fake-eval-test",
            )
        return ModelResponse(
            content="Calculation complete. The sum is four hundred dollars.",
            tool_calls=(),
            usage=UsageMetadata(prompt_tokens=60, completion_tokens=10, total_tokens=70),
            finish_reason="stop",
            model_identifier="fake-eval-test",
        )
    return handler


class TestTargetModeContract:
    """Verifies target mode enum and constructor contracts."""

    def test_default_mode_is_synthetic_control(self) -> None:
        prov = FakeProvider(model_name="fake-test", custom_handler=_make_passing_tool_handler())
        tracker = CostTracker(ExperimentBudgetConfig(max_model_requests=50, max_spend_usd=1.0, model_name="fake-test"), provider=prov)
        agent = ControlledAgent(provider=prov, mock_env=MockToolEnvironment(), cost_tracker=tracker)
        evaluator = ToolCallingTaskEvaluator(required_tool="calculator", expected_substring="400")

        target = MechanismTargetAdapter(
            mechanism_id="mech_test",
            agent=agent,
            evaluator=evaluator,
            user_prompt="Calculate sum",
        )
        assert target.target_mode == TargetMode.SYNTHETIC_CONTROL
        assert target.target_mode.value == "synthetic_control"

    def test_explicit_modes_and_string_coercion(self) -> None:
        prov = FakeProvider(model_name="fake-test", custom_handler=_make_passing_tool_handler())
        tracker = CostTracker(ExperimentBudgetConfig(max_model_requests=50, max_spend_usd=1.0, model_name="fake-test"), provider=prov)
        agent = ControlledAgent(provider=prov, mock_env=MockToolEnvironment(), cost_tracker=tracker)
        evaluator = ToolCallingTaskEvaluator(required_tool="calculator", expected_substring="400")

        # Explicit enum
        t_trace = MechanismTargetAdapter(
            mechanism_id="mech_test",
            agent=agent,
            evaluator=evaluator,
            user_prompt="Calculate sum",
            target_mode=TargetMode.TRACE_EVALUATED,
        )
        assert t_trace.target_mode == TargetMode.TRACE_EVALUATED

        # String coercion
        t_str = MechanismTargetAdapter(
            mechanism_id="mech_test",
            agent=agent,
            evaluator=evaluator,
            user_prompt="Calculate sum",
            target_mode="trace_evaluated",
        )
        assert t_str.target_mode == TargetMode.TRACE_EVALUATED

    def test_mechanism_task_spec_create_target_mode_propagation(self) -> None:
        spec = get_mechanism_task("mech_s01_type_mismatch")
        prov = FakeProvider(model_name="fake-test", custom_handler=_make_passing_tool_handler())
        tracker = CostTracker(ExperimentBudgetConfig(max_model_requests=50, max_spend_usd=1.0, model_name="fake-test"), provider=prov)

        # Default create_target
        t_def = spec.create_target(provider=prov, cost_tracker=tracker)
        assert t_def.target_mode == TargetMode.SYNTHETIC_CONTROL

        # Explicit trace_evaluated
        t_trace = spec.create_target(provider=prov, cost_tracker=tracker, target_mode=TargetMode.TRACE_EVALUATED)
        assert t_trace.target_mode == TargetMode.TRACE_EVALUATED


class TestSyntheticControlMode:
    """Verifies synthetic-control mode preserves predicate-driven outcomes."""

    def test_synthetic_control_predicate_determines_outcome(self) -> None:
        prov = FakeProvider(model_name="fake-test", custom_handler=_make_passing_tool_handler())
        tracker = CostTracker(ExperimentBudgetConfig(max_model_requests=50, max_spend_usd=1.0, model_name="fake-test"), provider=prov)
        agent = ControlledAgent(provider=prov, mock_env=MockToolEnvironment(), cost_tracker=tracker)
        evaluator = ToolCallingTaskEvaluator(required_tool="calculator", expected_substring="400")

        # Predicate returning failure
        failing_pred = MagicMock(return_value=(True, "Synthetic type error planted"))
        target = MechanismTargetAdapter(
            mechanism_id="mech_test",
            agent=agent,
            evaluator=evaluator,
            user_prompt="Calculate sum",
            failure_predicate=failing_pred,
            target_mode=TargetMode.SYNTHETIC_CONTROL,
        )

        res = target.execute(InputCase(case_id="c1", elements=("el1",)))
        assert res.outcome == TargetOutcome.FAILURE
        assert "Synthetic type error planted" in res.output_message
        assert res.metadata.get("target_mode") == "synthetic_control"
        assert res.metadata.get("error") == "Synthetic type error planted"
        assert target.last_trace is not None
        assert target.last_trace.error_message == "Synthetic type error planted"
        assert failing_pred.call_count == 1

        # Predicate returning success
        passing_pred = MagicMock(return_value=(False, "Passing condition"))
        target._failure_predicate = passing_pred
        res_pass = target.execute(InputCase(case_id="c2", elements=()))
        assert res_pass.outcome == TargetOutcome.SUCCESS
        assert "Passing condition" in res_pass.output_message
        assert res_pass.metadata.get("target_mode") == "synthetic_control"
        assert passing_pred.call_count == 1


class TestTraceEvaluatedMode:
    """Verifies trace-evaluated mode isolates the evaluator and bypasses synthetic predicates."""

    def test_trace_evaluated_uses_evaluator_when_predicate_says_fail(self) -> None:
        """Agent produces passing trace (calculator invoked, '400' in output), predicate says fail."""
        prov = FakeProvider(model_name="fake-test", custom_handler=_make_passing_tool_handler())
        tracker = CostTracker(ExperimentBudgetConfig(max_model_requests=50, max_spend_usd=1.0, model_name="fake-test"), provider=prov)
        agent = ControlledAgent(provider=prov, mock_env=MockToolEnvironment(), cost_tracker=tracker)
        evaluator = ToolCallingTaskEvaluator(required_tool="calculator", expected_substring="400")

        contradicting_pred = MagicMock(return_value=(True, "PLANTED_SYNTHETIC_FAILURE_RULE"))
        target = MechanismTargetAdapter(
            mechanism_id="mech_test",
            agent=agent,
            evaluator=evaluator,
            user_prompt="Calculate sum",
            tools=[CALCULATOR_TOOL_DEF],
            failure_predicate=contradicting_pred,
            target_mode=TargetMode.TRACE_EVALUATED,
        )

        res = target.execute(InputCase(case_id="c1", elements=("planted_trigger",)))
        # Target must succeed because the agent's observable trace satisfied the evaluator
        assert res.outcome == TargetOutcome.SUCCESS
        assert contradicting_pred.call_count == 0  # Predicate MUST NOT be called
        assert res.metadata.get("target_mode") == "trace_evaluated"

    def test_trace_evaluated_uses_evaluator_when_predicate_says_success(self) -> None:
        """Agent produces failing trace (missing '400'), predicate says success."""
        prov = FakeProvider(model_name="fake-test", custom_handler=_make_omitted_answer_handler())
        tracker = CostTracker(ExperimentBudgetConfig(max_model_requests=50, max_spend_usd=1.0, model_name="fake-test"), provider=prov)
        agent = ControlledAgent(provider=prov, mock_env=MockToolEnvironment(), cost_tracker=tracker)
        evaluator = ToolCallingTaskEvaluator(required_tool="calculator", expected_substring="400")

        contradicting_pred = MagicMock(return_value=(False, "PLANTED_SYNTHETIC_PASSING_RULE"))
        target = MechanismTargetAdapter(
            mechanism_id="mech_test",
            agent=agent,
            evaluator=evaluator,
            user_prompt="Calculate sum",
            tools=[CALCULATOR_TOOL_DEF],
            failure_predicate=contradicting_pred,
            target_mode=TargetMode.TRACE_EVALUATED,
        )

        res = target.execute(InputCase(case_id="c1", elements=()))
        # Target must fail because trace omitted '400'
        assert res.outcome == TargetOutcome.FAILURE
        assert contradicting_pred.call_count == 0  # Predicate MUST NOT be called
        assert "expected content '400'" in res.output_message
        assert res.error_code == "AGENT_FAILURE"
        assert res.metadata.get("target_mode") == "trace_evaluated"

    def test_synthetic_predicate_error_text_not_injected_in_trace(self) -> None:
        """Trace-evaluated traces must preserve genuine model and runtime text without synthetic corruption."""
        prov = FakeProvider(model_name="fake-test", custom_handler=_make_passing_tool_handler())
        tracker = CostTracker(ExperimentBudgetConfig(max_model_requests=50, max_spend_usd=1.0, model_name="fake-test"), provider=prov)
        agent = ControlledAgent(provider=prov, mock_env=MockToolEnvironment(), cost_tracker=tracker)
        evaluator = ToolCallingTaskEvaluator(required_tool="calculator", expected_substring="400")

        leak_pred = MagicMock(return_value=(True, "LEAKED_SYNTHETIC_DIAGNOSTIC_STRING"))
        target = MechanismTargetAdapter(
            mechanism_id="mech_test",
            agent=agent,
            evaluator=evaluator,
            user_prompt="Calculate sum",
            tools=[CALCULATOR_TOOL_DEF],
            failure_predicate=leak_pred,
            target_mode=TargetMode.TRACE_EVALUATED,
        )

        res = target.execute(InputCase(case_id="c1", elements=("trigger",)))
        assert target.last_trace is not None
        assert target.last_trace.error_message is None
        assert "LEAKED_SYNTHETIC_DIAGNOSTIC_STRING" not in target.last_trace.final_response
        assert "LEAKED_SYNTHETIC_DIAGNOSTIC_STRING" not in res.output_message

    def test_genuine_trace_error_preserved_on_tool_failure(self) -> None:
        """When tool execution fails at runtime, genuine tool error output is preserved and evaluated."""
        prov = FakeProvider(model_name="fake-test", custom_handler=_make_failing_tool_handler())
        tracker = CostTracker(ExperimentBudgetConfig(max_model_requests=50, max_spend_usd=1.0, model_name="fake-test"), provider=prov)
        agent = ControlledAgent(provider=prov, mock_env=MockToolEnvironment(), cost_tracker=tracker)
        evaluator = ToolCallingTaskEvaluator(required_tool="calculator", expected_substring="400")

        target = MechanismTargetAdapter(
            mechanism_id="mech_test",
            agent=agent,
            evaluator=evaluator,
            user_prompt="Calculate sum",
            tools=[CALCULATOR_TOOL_DEF],
            target_mode=TargetMode.TRACE_EVALUATED,
        )

        res = target.execute(InputCase(case_id="c1", elements=("el1",)))
        assert res.outcome == TargetOutcome.FAILURE
        assert "Required tool 'calculator' returned an error" in res.output_message
        assert target.last_trace is not None
        assert len(target.last_trace.tool_executions) == 1
        assert target.last_trace.tool_executions[0].output.startswith("Error:")
        assert "invalid syntax" in target.last_trace.tool_executions[0].output

    def test_provider_infrastructure_error_aborts_without_falsifying_outcome(self) -> None:
        """Provider infrastructure crash raises RuntimeError and is not classified as task failure or success."""
        prov = FakeProvider(model_name="fake-test", simulate_error=ProviderTimeoutError)
        tracker = CostTracker(ExperimentBudgetConfig(max_model_requests=50, max_spend_usd=1.0, model_name="fake-test"), provider=prov)
        agent = ControlledAgent(provider=prov, mock_env=MockToolEnvironment(), cost_tracker=tracker)
        evaluator = ToolCallingTaskEvaluator(required_tool="calculator", expected_substring="400")

        target = MechanismTargetAdapter(
            mechanism_id="mech_test",
            agent=agent,
            evaluator=evaluator,
            user_prompt="Calculate sum",
            tools=[CALCULATOR_TOOL_DEF],
            target_mode=TargetMode.TRACE_EVALUATED,
        )

        with pytest.raises(RuntimeError, match="Controlled agent execution aborted due to provider infrastructure error"):
            target.execute(InputCase(case_id="c1", elements=("el1",)))


class TestBudgetFailClosedBothModes:
    """Verifies that budget exhaustion halts fail-closed without false successes in both modes."""

    def test_budget_exhaustion_in_trace_evaluated_mode(self) -> None:
        cfg = ExperimentBudgetConfig(max_model_requests=1, max_spend_usd=1.0, model_name="fake-test")
        prov = FakeProvider(model_name="fake-test", custom_handler=_make_passing_direct_handler())
        tracker = CostTracker(cfg, provider=prov)
        agent = ControlledAgent(provider=prov, mock_env=MockToolEnvironment(), cost_tracker=tracker)
        evaluator = ToolCallingTaskEvaluator(expected_substring="400")

        target = MechanismTargetAdapter(
            mechanism_id="mech_test",
            agent=agent,
            evaluator=evaluator,
            user_prompt="Calculate sum",
            target_mode=TargetMode.TRACE_EVALUATED,
        )

        # Call 1: Consumes 1 request (hits max_model_requests=1)
        res1 = target.execute(InputCase(case_id="c1", elements=()))
        assert res1.outcome == TargetOutcome.SUCCESS

        # Call 2: Budget exhausted; must raise CostBudgetExceededError fail-closed
        with pytest.raises(CostBudgetExceededError) as exc_info:
            target.execute(InputCase(case_id="c2", elements=()))

        assert "budget exhausted" in str(exc_info.value).lower()

    def test_budget_exhaustion_in_synthetic_control_mode(self) -> None:
        cfg = ExperimentBudgetConfig(max_model_requests=1, max_spend_usd=1.0, model_name="fake-test")
        prov = FakeProvider(model_name="fake-test", custom_handler=_make_passing_direct_handler())
        tracker = CostTracker(cfg, provider=prov)
        agent = ControlledAgent(provider=prov, mock_env=MockToolEnvironment(), cost_tracker=tracker)
        evaluator = ToolCallingTaskEvaluator(expected_substring="400")
        mock_pred = MagicMock(return_value=(False, "Passing condition"))

        target = MechanismTargetAdapter(
            mechanism_id="mech_test",
            agent=agent,
            evaluator=evaluator,
            user_prompt="Calculate sum",
            failure_predicate=mock_pred,
            target_mode=TargetMode.SYNTHETIC_CONTROL,
        )

        # Call 1: Consumes 1 request
        res1 = target.execute(InputCase(case_id="c1", elements=()))
        assert res1.outcome == TargetOutcome.SUCCESS
        assert mock_pred.call_count == 1

        # Call 2: Budget exhausted; must raise CostBudgetExceededError without calling predicate
        with pytest.raises(CostBudgetExceededError) as exc_info:
            target.execute(InputCase(case_id="c2", elements=()))

        assert "budget exhausted" in str(exc_info.value).lower()
        assert mock_pred.call_count == 1  # Not called on Call 2
