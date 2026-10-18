"""Tests for the isolated trace-evaluated schema fault mechanism (Phase 4.6K.5).

Verifies:
1. Schema materialization: candidate alters tool schema (renaming 'expression' -> 'expr') when enabled,
   and restores canonical schema when ablated.
2. Prompt/schema separation: schema candidate strings never leak into system or user prompt messages.
3. Scripted provider fidelity: provider complies with whatever parameter key is declared in the schema.
4. Observable execution failure: faulty schema causes backend MockToolEnvironment to reject call with
   "Missing required argument(s): expression", failing ToolCallingTaskEvaluator.
5. Observable recovery: ablated schema executes successfully with output "400", passing evaluator.
6. TargetMode.TRACE_EVALUATED outcome: outcome determined strictly by evaluator; predicate ignored.
7. Schema immutability: global CALCULATOR_TOOL_DEF, target.tools, and mock_env._tools are never mutated.
8. State non-leakage: repeated alternating executions do not accumulate schema mutations or state.
9. Active DD minimization: StochasticDeltaDebugMinimizer isolates the single causal schema trigger.
10. Budget and infrastructure robustness: fail-closed budget exhaustion and RuntimeError on provider error.
11. Isolation: MECHANISM_REGISTRY remains exactly 18 synthetic mechanisms; prompt slice unchanged.
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
from agentxray.agent_harness.tools import CALCULATOR_TOOL_DEF, DATABASE_LOOKUP_TOOL_DEF, MockToolEnvironment
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
    TraceEvaluatedSchemaTaskSpec,
    create_trace_evaluated_prompt_task,
    create_trace_evaluated_schema_task,
    get_mechanism_task,
    list_mechanism_tasks,
)


def make_schema_sensitive_scripted_handler():
    """Create a scripted FakeProvider handler that inspects the tool schemas presented to it.

    SCIENTIFIC INTEGRITY DISCLAIMER:
    This scripted handler is strictly for testing harness orchestration, dynamic schema
    materialization, and trace-evaluated outcome determination offline without inference cost.
    It simulates an agent that adheres to the tool schema provided to it at runtime,
    but does NOT measure real LLM reasoning.
    """
    def handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
        has_tool_reply = any(m.role == "tool" for m in messages)

        if not has_tool_reply:
            # Find calculator tool definition from presented tools
            calc_tool = next((t for t in tools if t.name == "calculator"), None)
            calc_props = calc_tool.parameters.get("properties", {}) if calc_tool else {}

            if "expr" in calc_props:
                # Agent adheres to the schema presented: uses 'expr'
                args = {"expr": "150 + 250"}
            else:
                # Canonical schema: uses 'expression'
                args = {"expression": "150 + 250"}

            return ModelResponse(
                content="Calling calculator according to schema...",
                tool_calls=(
                    ToolCall(call_id="call_calc", name="calculator", arguments=args),
                ),
                usage=UsageMetadata(prompt_tokens=45, completion_tokens=15, total_tokens=60),
                finish_reason="tool_calls",
                model_identifier="fake-schema-sensitive-agent",
            )
        else:
            # Inspect tool execution response
            tool_msg = next((m for m in messages if m.role == "tool"), None)
            tool_content = tool_msg.content if tool_msg else ""

            if tool_content.startswith("Error:"):
                # Tool failed
                return ModelResponse(
                    content=f"Calculation failed with error from calculator: {tool_content}",
                    tool_calls=(),
                    usage=UsageMetadata(prompt_tokens=70, completion_tokens=20, total_tokens=90),
                    finish_reason="stop",
                    model_identifier="fake-schema-sensitive-agent",
                )
            else:
                # Tool succeeded
                return ModelResponse(
                    content=f"Calculation succeeded. Verified receipt total is {tool_content.strip()}.",
                    tool_calls=(),
                    usage=UsageMetadata(prompt_tokens=65, completion_tokens=15, total_tokens=80),
                    finish_reason="stop",
                    model_identifier="fake-schema-sensitive-agent",
                )

    return handler


class TestSchemaCandidateMaterialization:
    """Verifies that candidate presence alters tool schemas without leaking into prompt messages."""

    def test_schema_materialization_faulty(self) -> None:
        """Full case presents calculator tool with 'expr' parameter and required: ['expr'], not 'expression'."""
        task = create_trace_evaluated_schema_task()
        initial_input = task.create_initial_input()

        captured_tools: list[Sequence[ToolDefinition]] = []

        def spy_handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            captured_tools.append(tuple(tools))
            return ModelResponse(
                content="Done.",
                tool_calls=(),
                usage=UsageMetadata(prompt_tokens=20, completion_tokens=5, total_tokens=25),
                finish_reason="stop",
                model_identifier="fake-spy",
            )

        prov = FakeProvider(model_name="fake-spy", custom_handler=spy_handler)
        target = task.create_target(provider=prov)

        target.execute(initial_input)
        assert len(captured_tools) == 1
        tools = captured_tools[0]
        calc = next((t for t in tools if t.name == "calculator"), None)
        assert calc is not None
        assert "expr" in calc.parameters["properties"]
        assert "expression" not in calc.parameters["properties"]
        assert calc.parameters["required"] == ["expr"]

    def test_schema_materialization_ablated(self) -> None:
        """Ablated case presents calculator tool with canonical 'expression' parameter and required: ['expression']."""
        task = create_trace_evaluated_schema_task()
        trigger = task.planted_trigger_elements[0]
        ablated_input = task.create_initial_input().without(trigger)

        captured_tools: list[Sequence[ToolDefinition]] = []

        def spy_handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            captured_tools.append(tuple(tools))
            return ModelResponse(
                content="Done.",
                tool_calls=(),
                usage=UsageMetadata(prompt_tokens=20, completion_tokens=5, total_tokens=25),
                finish_reason="stop",
                model_identifier="fake-spy",
            )

        prov = FakeProvider(model_name="fake-spy", custom_handler=spy_handler)
        target = task.create_target(provider=prov)

        target.execute(ablated_input)
        assert len(captured_tools) == 1
        tools = captured_tools[0]
        calc = next((t for t in tools if t.name == "calculator"), None)
        assert calc is not None
        assert "expression" in calc.parameters["properties"]
        assert "expr" not in calc.parameters["properties"]
        assert calc.parameters["required"] == ["expression"]

    def test_prompt_cleanliness(self) -> None:
        """Schema candidates must never appear in prompt messages (system or user)."""
        task = create_trace_evaluated_schema_task()
        initial_input = task.create_initial_input()

        captured_messages: list[Sequence[Message]] = []

        def spy_handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            captured_messages.append(tuple(messages))
            return ModelResponse(
                content="Done.",
                tool_calls=(),
                usage=UsageMetadata(prompt_tokens=20, completion_tokens=5, total_tokens=25),
                finish_reason="stop",
                model_identifier="fake-spy",
            )

        prov = FakeProvider(model_name="fake-spy", custom_handler=spy_handler)
        target = task.create_target(provider=prov)

        target.execute(initial_input)
        assert len(captured_messages) == 1
        messages = captured_messages[0]

        # Check all candidate elements (trigger and distractors)
        for element in task.candidate_elements:
            for msg in messages:
                assert element not in msg.content, f"Candidate element leaked into prompt message: {element!r}"

        # Check prompt purity
        sys_msg = next((m for m in messages if m.role == "system"), None)
        user_msg = next((m for m in messages if m.role == "user"), None)
        assert sys_msg is not None and sys_msg.content == "You are an automated accounting bot."
        assert user_msg is not None and user_msg.content == "Calculate the sum of receipts $150 and $250 using calculator."

    def test_scripted_provider_adheres_to_schema(self) -> None:
        """Scripted provider emits 'expr' when given faulty schema, and 'expression' when given canonical schema."""
        handler = make_schema_sensitive_scripted_handler()

        faulty_calc = ToolDefinition(
            name="calculator",
            description="calculator",
            parameters={"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]},
        )
        canonical_calc = CALCULATOR_TOOL_DEF

        msgs = (
            Message(role="system", content="sys"),
            Message(role="user", content="user"),
        )

        resp_faulty = handler(msgs, (faulty_calc,))
        assert len(resp_faulty.tool_calls) == 1
        assert resp_faulty.tool_calls[0].arguments == {"expr": "150 + 250"}

        resp_canonical = handler(msgs, (canonical_calc,))
        assert len(resp_canonical.tool_calls) == 1
        assert resp_canonical.tool_calls[0].arguments == {"expression": "150 + 250"}


class TestObservableExecutionAndEvaluator:
    """Verifies that backend tool execution failures and recoveries determine target outcome."""

    def test_faulty_schema_produces_observable_backend_error(self) -> None:
        """Backend rejects 'expr' with missing required argument error; evaluator marks failure."""
        task = create_trace_evaluated_schema_task()
        initial_input = task.create_initial_input()

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_schema_sensitive_scripted_handler())
        target = task.create_target(provider=prov)

        result = target.execute(initial_input)

        assert result.outcome == TargetOutcome.FAILURE
        assert target.last_trace is not None
        assert len(target.last_trace.tool_executions) == 1
        texec = target.last_trace.tool_executions[0]
        assert texec.tool_name == "calculator"
        assert texec.arguments == {"expr": "150 + 250"}
        assert "Missing required argument(s): expression" in texec.output
        assert "Required tool 'calculator' returned an error" in result.output_message

    def test_ablated_schema_produces_clean_pass(self) -> None:
        """Tool call succeeds with '400' on ablated schema; evaluator marks success."""
        task = create_trace_evaluated_schema_task()
        trigger = task.planted_trigger_elements[0]
        ablated_input = task.create_initial_input().without(trigger)

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_schema_sensitive_scripted_handler())
        target = task.create_target(provider=prov)

        result = target.execute(ablated_input)

        assert result.outcome == TargetOutcome.SUCCESS
        assert target.last_trace is not None
        assert len(target.last_trace.tool_executions) == 1
        texec = target.last_trace.tool_executions[0]
        assert texec.tool_name == "calculator"
        assert texec.arguments == {"expression": "150 + 250"}
        assert texec.output == "400"
        assert "400" in target.last_trace.final_response

    def test_target_outcome_determined_by_evaluator_not_predicate(self) -> None:
        """Target outcome is determined strictly by evaluator in TargetMode.TRACE_EVALUATED."""
        task = create_trace_evaluated_schema_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()
        ablated_input = initial_input.without(trigger)

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_schema_sensitive_scripted_handler())
        target = task.create_target(provider=prov)

        res_faulty = target.execute(initial_input)
        assert res_faulty.outcome == TargetOutcome.FAILURE
        assert res_faulty.error_code == "AGENT_FAILURE"

        res_ablated = target.execute(ablated_input)
        assert res_ablated.outcome == TargetOutcome.SUCCESS
        assert res_ablated.error_code is None

    def test_contradictory_predicate_ignored(self) -> None:
        """Contradictory predicate returning forced success does not change TargetOutcome.FAILURE."""
        task = create_trace_evaluated_schema_task()
        initial_input = task.create_initial_input()

        # Contradictory predicate that claims success
        mock_pred = MagicMock(side_effect=lambda ic: (False, "Contradictory forced success predicate"))

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_schema_sensitive_scripted_handler())
        target = task.create_target(provider=prov, failure_predicate=mock_pred)

        result = target.execute(initial_input)
        # Outcome must still be FAILURE because the trace evaluator detected the backend error
        assert result.outcome == TargetOutcome.FAILURE
        assert "Required tool 'calculator' returned an error" in result.output_message
        assert mock_pred.call_count == 0  # Predicate must not be invoked at all


class TestSchemaImmutabilityAndIsolation:
    """Verifies that schema alteration never mutates shared definitions or target state."""

    def test_canonical_definitions_unmutated(self) -> None:
        """Global CALCULATOR_TOOL_DEF, target.tools, and backend environment remain unchanged."""
        task = create_trace_evaluated_schema_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()
        ablated_input = initial_input.without(trigger)

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_schema_sensitive_scripted_handler())
        target = task.create_target(provider=prov)

        # Pre-execution checks
        assert "expression" in CALCULATOR_TOOL_DEF.parameters["properties"]
        assert CALCULATOR_TOOL_DEF.parameters["required"] == ["expression"]
        assert "expression" in target.tools[0].parameters["properties"]
        assert target.tools[0].parameters["required"] == ["expression"]

        # Run faulty execution
        target.execute(initial_input)

        # Post-faulty check
        assert "expression" in CALCULATOR_TOOL_DEF.parameters["properties"]
        assert "expr" not in CALCULATOR_TOOL_DEF.parameters["properties"]
        assert CALCULATOR_TOOL_DEF.parameters["required"] == ["expression"]
        assert "expression" in target.tools[0].parameters["properties"]
        assert "expr" not in target.tools[0].parameters["properties"]
        assert target.tools[0].parameters["required"] == ["expression"]
        assert target.agent.mock_env._tools["calculator"][0].parameters["required"] == ["expression"]

        # Run ablated execution
        target.execute(ablated_input)

        # Post-ablated check
        assert "expression" in CALCULATOR_TOOL_DEF.parameters["properties"]
        assert "expression" in target.tools[0].parameters["properties"]
        assert target.agent.mock_env._tools["calculator"][0].parameters["required"] == ["expression"]

    def test_reproducibility_and_non_leaking_state(self) -> None:
        """Repeated and reordered executions do not accumulate schema mutations."""
        task = create_trace_evaluated_schema_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()
        ablated_input = initial_input.without(trigger)

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_schema_sensitive_scripted_handler())
        target = task.create_target(provider=prov)

        results = [
            target.execute(initial_input).outcome,
            target.execute(ablated_input).outcome,
            target.execute(initial_input).outcome,
            target.execute(ablated_input).outcome,
        ]

        assert results == [
            TargetOutcome.FAILURE,
            TargetOutcome.SUCCESS,
            TargetOutcome.FAILURE,
            TargetOutcome.SUCCESS,
        ]
        assert target.call_count == 4


class TestActiveMinimizationOnSchemaFault:
    """Verifies that StochasticDeltaDebugMinimizer isolates the single schema fault trigger."""

    def test_active_dd_minimization_schema_fault(self) -> None:
        """Active DD minimizer isolates the planted schema trigger via trace evaluation."""
        task = create_trace_evaluated_schema_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_schema_sensitive_scripted_handler())
        target = task.create_target(provider=prov)

        cfg = ExperimentConfig(experiment_name="trace_schema_dd_test", max_trials=20)
        runner = InterventionRunner(config=cfg, target_fn=target.execute)
        minimizer = StochasticDeltaDebugMinimizer(runner=runner, trials_per_candidate=1, failure_threshold=0.5)

        cond = minimizer.minimize(initial_input)
        assert cond.status == SearchStatus.COMPLETE
        assert cond.is_failure is True
        assert cond.minimality_status in (
            StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
            StochasticMinimalityStatus.OBSERVED_ONLY,
        )
        assert cond.subset_elements == (trigger,)


class TestFailClosedBudgetAndInfra:
    """Verifies budget exhaustion remains fail-closed and provider infrastructure errors raise RuntimeError."""

    def test_budget_exhaustion_fail_closed(self) -> None:
        """Target halts with CostBudgetExceededError when request budget is exhausted."""
        task = create_trace_evaluated_schema_task()
        initial_input = task.create_initial_input()

        tracker = CostTracker(
            ExperimentBudgetConfig(max_model_requests=1, max_spend_usd=1.00, model_name="fake-schema")
        )
        prov = FakeProvider(model_name="fake-schema", custom_handler=make_schema_sensitive_scripted_handler())
        target = task.create_target(provider=prov, cost_tracker=tracker)

        with pytest.raises(CostBudgetExceededError):
            target.execute(initial_input)

    def test_infrastructure_error_raises_runtime_error(self) -> None:
        """Provider infrastructure error raises RuntimeError without misclassifying as agent failure."""
        task = create_trace_evaluated_schema_task()
        initial_input = task.create_initial_input()

        def timeout_handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            raise ProviderTimeoutError("Inference request timed out after 30s")

        prov = FakeProvider(model_name="fake-timeout", custom_handler=timeout_handler)
        target = task.create_target(provider=prov)

        with pytest.raises(RuntimeError, match="provider infrastructure error"):
            target.execute(initial_input)


class TestBenchmarkIsolationAndRegression:
    """Verifies registry integrity and that Phase 4.6K.3 prompt task remains fully functional."""

    def test_trace_evaluated_prompt_regression(self) -> None:
        """Phase 4.6K.3 trace-evaluated prompt task specification remains valid."""
        prompt_task = create_trace_evaluated_prompt_task()
        assert isinstance(prompt_task, TraceEvaluatedPromptTaskSpec)
        assert prompt_task.mechanism_id == "trace_prompt_01_tool_prohibition"
        gt = prompt_task.get_ground_truth()
        assert len(gt.trigger_elements) == 1

    def test_synthetic_mechanisms_unaffected(self) -> None:
        """MECHANISM_REGISTRY contains exactly the 18 original synthetic control mechanisms."""
        assert len(MECHANISM_REGISTRY) == 18
        assert "trace_schema_01_param_key_mismatch" not in MECHANISM_REGISTRY
        assert "trace_prompt_01_tool_prohibition" not in MECHANISM_REGISTRY

        all_tasks = list_mechanism_tasks()
        assert len(all_tasks) == 18
        categories = {t.category for t in all_tasks}
        assert {"prompt", "schema", "environment", "conjunctive", "null_clean", "null_bloat", "null_ext"}.issubset(categories)
