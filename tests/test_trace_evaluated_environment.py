"""Tests for the isolated trace-evaluated environment fault mechanism (Phase 4.6K.7).

Verifies:
1. Environment fault materialization: candidate injects mock calculator service HTTP 500 error
   inside the mock environment handler only, and restores canonical safe calculator when ablated.
2. Prompt/schema invariance: prompt messages and model-facing tool schemas remain strictly
   identical and canonical between faulted and ablated trials.
3. Provider consistency: the scripted provider emits the identical valid tool call in both configurations.
4. Observable trace failure: the HTTP 500 error appears in ToolExecutionRecord.output and the tool message,
   causing ToolCallingTaskEvaluator to fail the trace with FAIL_TOOL_EXEC.
5. Observable recovery: ablated execution produces tool output "400", passing evaluator with CLEAN_PASS.
6. Evaluator-driven outcome: TargetMode.TRACE_EVALUATED determines outcome strictly from evaluator;
   contradictory failure predicates are bypassed.
7. Environment immutability: global CALCULATOR_TOOL_DEF, target.tools, and the agent's default mock_env
   are never mutated.
8. State non-leakage: repeated alternating executions do not accumulate state or leak handlers across trials.
9. Distractor benignity: benign operational distractors do not activate the service fault.
10. Active DD minimization: StochasticDeltaDebugMinimizer isolates the single causal environment trigger.
11. Robustness: budget exhaustion halts fail-closed; provider infrastructure errors raise RuntimeError.
12. Isolation: MECHANISM_REGISTRY remains exactly 18 synthetic mechanisms; prompt and schema tasks unchanged.
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
from agentxray.agent_harness.tools import CALCULATOR_TOOL_DEF, DATABASE_LOOKUP_TOOL_DEF, MockToolEnvironment, safe_calculator
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
    TraceEvaluatedEnvironmentTaskSpec,
    TraceEvaluatedPromptTaskSpec,
    TraceEvaluatedSchemaTaskSpec,
    create_trace_evaluated_environment_task,
    create_trace_evaluated_prompt_task,
    create_trace_evaluated_schema_task,
    get_mechanism_task,
    list_mechanism_tasks,
)


def make_env_sensitive_scripted_handler():
    """Create a scripted FakeProvider handler for environment fault experiments.

    SCIENTIFIC INTEGRITY DISCLAIMER:
    This scripted handler is strictly for testing harness orchestration, per-trial environment
    injection, and trace-evaluated outcome determination offline without inference cost.
    It simulates an agent that issues legitimate tool calls and reads environment responses,
    but does NOT measure real LLM reasoning.
    """
    def handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
        has_tool_reply = any(m.role == "tool" for m in messages)

        if not has_tool_reply:
            # Turn 1: Emit valid calculator tool call
            return ModelResponse(
                content="Calling calculator for receipts sum...",
                tool_calls=(
                    ToolCall(call_id="call_calc", name="calculator", arguments={"expression": "150 + 250"}),
                ),
                usage=UsageMetadata(prompt_tokens=45, completion_tokens=15, total_tokens=60),
                finish_reason="tool_calls",
                model_identifier="fake-env-agent",
            )
        else:
            # Turn 2: Inspect tool execution response
            tool_msg = next((m for m in messages if m.role == "tool"), None)
            tool_content = tool_msg.content if tool_msg else ""

            if tool_content.startswith("Error:"):
                return ModelResponse(
                    content=f"Calculation failed due to tool server error: {tool_content}",
                    tool_calls=(),
                    usage=UsageMetadata(prompt_tokens=70, completion_tokens=20, total_tokens=90),
                    finish_reason="stop",
                    model_identifier="fake-env-agent",
                )
            else:
                return ModelResponse(
                    content=f"Calculation complete. Final verified total is {tool_content.strip()}.",
                    tool_calls=(),
                    usage=UsageMetadata(prompt_tokens=65, completion_tokens=15, total_tokens=80),
                    finish_reason="stop",
                    model_identifier="fake-env-agent",
                )

    return handler


class TestEnvironmentCandidateMaterialization:
    """Verifies that the candidate activates an HTTP 500 fault inside the mock environment while prompts and schemas remain canonical."""

    def test_trigger_present_materializes_http_500_fault(self) -> None:
        """Trigger present causes mock environment to return HTTP 500 error; target returns FAILURE."""
        task = create_trace_evaluated_environment_task()
        initial_input = task.create_initial_input()

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_env_sensitive_scripted_handler())
        target = task.create_target(provider=prov)

        result = target.execute(initial_input)

        assert result.outcome == TargetOutcome.FAILURE
        assert result.error_code == "AGENT_FAILURE"
        assert target.last_trace is not None
        assert len(target.last_trace.tool_executions) == 1
        texec = target.last_trace.tool_executions[0]
        assert texec.tool_name == "calculator"
        assert texec.output.startswith("Error: 500 Internal Server Error")
        assert "500 Internal Server Error" in result.output_message

    def test_trigger_ablated_uses_canonical_handler_and_returns_400(self) -> None:
        """Trigger ablated uses canonical safe_calculator handler and returns 400; target returns SUCCESS."""
        task = create_trace_evaluated_environment_task()
        trigger = task.planted_trigger_elements[0]
        ablated_input = task.create_initial_input().without(trigger)

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_env_sensitive_scripted_handler())
        target = task.create_target(provider=prov)

        result = target.execute(ablated_input)

        assert result.outcome == TargetOutcome.SUCCESS
        assert result.error_code is None
        assert target.last_trace is not None
        assert len(target.last_trace.tool_executions) == 1
        texec = target.last_trace.tool_executions[0]
        assert texec.tool_name == "calculator"
        assert texec.output == "400"
        assert "400" in target.last_trace.final_response

    def test_faulted_and_healthy_runs_use_identical_prompts_and_schemas(self) -> None:
        """Prompts and model-facing tool schemas are strictly identical between faulted and ablated executions."""
        task = create_trace_evaluated_environment_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()
        ablated_input = initial_input.without(trigger)

        captured_calls: list[tuple[Sequence[Message], Sequence[ToolDefinition]]] = []

        def spy_handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            captured_calls.append((tuple(messages), tuple(tools)))
            return ModelResponse(
                content="Spy stop.",
                tool_calls=(),
                usage=UsageMetadata(prompt_tokens=20, completion_tokens=5, total_tokens=25),
                finish_reason="stop",
                model_identifier="fake-spy",
            )

        prov = FakeProvider(model_name="fake-spy", custom_handler=spy_handler)
        target = task.create_target(provider=prov)

        target.execute(initial_input)
        target.execute(ablated_input)

        assert len(captured_calls) == 2
        msgs_faulted, tools_faulted = captured_calls[0]
        msgs_ablated, tools_ablated = captured_calls[1]

        # 1. Prompts are identical in content and roles
        assert len(msgs_faulted) == len(msgs_ablated) == 2
        assert msgs_faulted[0].role == msgs_ablated[0].role == "system"
        assert msgs_faulted[0].content == msgs_ablated[0].content == "You are an automated accounting bot."
        assert msgs_faulted[1].role == msgs_ablated[1].role == "user"
        assert msgs_faulted[1].content == msgs_ablated[1].content == "Calculate the sum of receipts $150 and $250 using calculator."

        # Candidate strings must never appear in prompt messages
        for cand in task.candidate_elements:
            assert cand not in msgs_faulted[0].content
            assert cand not in msgs_faulted[1].content

        # 2. Tool schemas are identical and canonical
        assert len(tools_faulted) == len(tools_ablated) == 2
        calc_faulted = next(t for t in tools_faulted if t.name == "calculator")
        calc_ablated = next(t for t in tools_ablated if t.name == "calculator")
        assert calc_faulted.parameters == calc_ablated.parameters == CALCULATOR_TOOL_DEF.parameters
        assert "expression" in calc_faulted.parameters["properties"]
        assert calc_faulted.parameters["required"] == ["expression"]

    def test_scripted_provider_issues_same_valid_calculator_call(self) -> None:
        """Scripted provider emits the identical valid calculator call in both faulted and healthy configurations."""
        task = create_trace_evaluated_environment_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()
        ablated_input = initial_input.without(trigger)

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_env_sensitive_scripted_handler())
        target = task.create_target(provider=prov)

        target.execute(initial_input)
        faulted_trace = target.last_trace
        target.execute(ablated_input)
        ablated_trace = target.last_trace

        assert faulted_trace is not None and ablated_trace is not None
        assert len(faulted_trace.tool_executions) == len(ablated_trace.tool_executions) == 1
        assert faulted_trace.tool_executions[0].tool_name == ablated_trace.tool_executions[0].tool_name == "calculator"
        assert faulted_trace.tool_executions[0].arguments == ablated_trace.tool_executions[0].arguments == {"expression": "150 + 250"}

    def test_fault_appears_in_tool_execution_record_and_tool_message(self) -> None:
        """The HTTP 500 fault appears in ToolExecutionRecord.output and the conversation tool message."""
        task = create_trace_evaluated_environment_task()
        initial_input = task.create_initial_input()

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_env_sensitive_scripted_handler())
        target = task.create_target(provider=prov)

        target.execute(initial_input)
        trace = target.last_trace

        assert trace is not None
        assert len(trace.tool_executions) == 1
        texec = trace.tool_executions[0]
        assert "500 Internal Server Error" in texec.output

        tool_msgs = [m for m in trace.messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert "500 Internal Server Error" in tool_msgs[0].content


class TestObservableExecutionAndEvaluator:
    """Verifies that ToolCallingTaskEvaluator correctly classifies environment outcomes and ignores predicates."""

    def test_trace_evaluator_distinguishes_failure_from_clean_success(self) -> None:
        """Evaluator classifies faulted trace as FAIL_TOOL_EXEC and healthy trace as CLEAN_PASS."""
        task = create_trace_evaluated_environment_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()
        ablated_input = initial_input.without(trigger)

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_env_sensitive_scripted_handler())
        target = task.create_target(provider=prov)

        target.execute(initial_input)
        faulted_trace = target.last_trace
        target.execute(ablated_input)
        ablated_trace = target.last_trace

        evaluator = target.evaluator
        assert isinstance(evaluator, ToolCallingTaskEvaluator)

        is_fail_ok, fail_diag, fail_class = evaluator.evaluate_detailed(faulted_trace)
        assert is_fail_ok is False
        assert fail_class == "FAIL_TOOL_EXEC"
        assert "500 Internal Server Error" in fail_diag

        is_pass_ok, pass_diag, pass_class = evaluator.evaluate_detailed(ablated_trace)
        assert is_pass_ok is True
        assert pass_class == "CLEAN_PASS"
        assert "successfully" in pass_diag

    def test_contradictory_predicate_ignored(self) -> None:
        """Contradictory predicate claiming forced pass is bypassed in TRACE_EVALUATED mode."""
        task = create_trace_evaluated_environment_task()
        initial_input = task.create_initial_input()

        mock_pred = MagicMock(side_effect=lambda ic: (False, "PREDICATE_FORCED_SUCCESS"))
        prov = FakeProvider(model_name="fake-agent", custom_handler=make_env_sensitive_scripted_handler())
        target = task.create_target(provider=prov, failure_predicate=mock_pred)

        result = target.execute(initial_input)

        # Trace evaluator dictates failure
        assert result.outcome == TargetOutcome.FAILURE
        assert "500 Internal Server Error" in result.output_message
        assert mock_pred.call_count == 0  # Predicate must not be invoked

    def test_benign_distractors_do_not_activate_fault(self) -> None:
        """Candidate subset containing only benign distractors produces clean success."""
        task = create_trace_evaluated_environment_task()
        trigger = task.planted_trigger_elements[0]
        distractors = tuple(e for e in task.candidate_elements if e != trigger)
        distractor_only_case = InputCase(case_id="distractors_only", elements=distractors)

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_env_sensitive_scripted_handler())
        target = task.create_target(provider=prov)

        result = target.execute(distractor_only_case)

        assert result.outcome == TargetOutcome.SUCCESS
        assert target.last_trace is not None
        assert target.last_trace.tool_executions[0].output == "400"


class TestEnvironmentImmutabilityAndIsolation:
    """Verifies that dynamic environment construction never mutates singletons or leaks across trials."""

    def test_canonical_definitions_unmutated(self) -> None:
        """Global CALCULATOR_TOOL_DEF, target.tools, and agent's default mock_env remain unchanged."""
        task = create_trace_evaluated_environment_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()
        ablated_input = initial_input.without(trigger)

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_env_sensitive_scripted_handler())
        target = task.create_target(provider=prov)

        # Baseline checks
        assert "expression" in CALCULATOR_TOOL_DEF.parameters["properties"]
        assert CALCULATOR_TOOL_DEF.parameters["required"] == ["expression"]
        assert target.tools[0].parameters["required"] == ["expression"]
        assert target.agent.mock_env._tools["calculator"][1] is safe_calculator

        # Execute faulted trial
        target.execute(initial_input)

        # Post-fault checks: default environment on agent must still hold safe_calculator
        assert target.agent.mock_env._tools["calculator"][1] is safe_calculator
        assert CALCULATOR_TOOL_DEF.parameters["required"] == ["expression"]
        assert target.tools[0].parameters["required"] == ["expression"]

        # Execute ablated trial
        target.execute(ablated_input)

        # Post-ablated checks
        assert target.agent.mock_env._tools["calculator"][1] is safe_calculator
        assert CALCULATOR_TOOL_DEF.parameters["required"] == ["expression"]

    def test_reproducibility_and_non_leaking_state(self) -> None:
        """Alternating faulted -> healthy -> faulted -> healthy runs produce identical deterministic outcomes."""
        task = create_trace_evaluated_environment_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()
        ablated_input = initial_input.without(trigger)

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_env_sensitive_scripted_handler())
        target = task.create_target(provider=prov)

        outcomes = [
            target.execute(initial_input).outcome,
            target.execute(ablated_input).outcome,
            target.execute(initial_input).outcome,
            target.execute(ablated_input).outcome,
        ]

        assert outcomes == [
            TargetOutcome.FAILURE,
            TargetOutcome.SUCCESS,
            TargetOutcome.FAILURE,
            TargetOutcome.SUCCESS,
        ]
        assert target.call_count == 4


class TestActiveMinimizationOnEnvironmentFault:
    """Verifies that StochasticDeltaDebugMinimizer isolates the single environment trigger."""

    def test_active_dd_minimization_env_fault(self) -> None:
        """Active DD minimizer isolates the planted environment trigger via trace evaluation."""
        task = create_trace_evaluated_environment_task()
        trigger = task.planted_trigger_elements[0]
        initial_input = task.create_initial_input()

        prov = FakeProvider(model_name="fake-agent", custom_handler=make_env_sensitive_scripted_handler())
        target = task.create_target(provider=prov)

        cfg = ExperimentConfig(experiment_name="trace_env_dd_test", max_trials=20)
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
    """Verifies budget exhaustion fails closed and provider errors remain distinguishable from tool 500 errors."""

    def test_budget_exhaustion_fail_closed(self) -> None:
        """Request budget breach raises CostBudgetExceededError fail-closed."""
        task = create_trace_evaluated_environment_task()
        initial_input = task.create_initial_input()

        tracker = CostTracker(
            ExperimentBudgetConfig(max_model_requests=1, max_spend_usd=1.00, model_name="fake-env")
        )
        prov = FakeProvider(model_name="fake-env", custom_handler=make_env_sensitive_scripted_handler())
        target = task.create_target(provider=prov, cost_tracker=tracker)

        with pytest.raises(CostBudgetExceededError):
            target.execute(initial_input)

    def test_infrastructure_error_raises_runtime_error(self) -> None:
        """Provider infrastructure error raises RuntimeError, distinct from tool execution 500 errors."""
        task = create_trace_evaluated_environment_task()
        initial_input = task.create_initial_input()

        def timeout_handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            raise ProviderTimeoutError("Inference request timed out after 30s")

        prov = FakeProvider(model_name="fake-timeout", custom_handler=timeout_handler)
        target = task.create_target(provider=prov)

        with pytest.raises(RuntimeError, match="provider infrastructure error"):
            target.execute(initial_input)


class TestBenchmarkIsolationAndRegression:
    """Verifies that prompt, schema, and synthetic benchmarks remain unaffected and valid."""

    def test_trace_evaluated_prompt_and_schema_regression(self) -> None:
        """Phase 4.6K.3 prompt task and Phase 4.6K.5 schema task remain valid."""
        prompt_task = create_trace_evaluated_prompt_task()
        assert isinstance(prompt_task, TraceEvaluatedPromptTaskSpec)
        assert prompt_task.mechanism_id == "trace_prompt_01_tool_prohibition"

        schema_task = create_trace_evaluated_schema_task()
        assert isinstance(schema_task, TraceEvaluatedSchemaTaskSpec)
        assert schema_task.mechanism_id == "trace_schema_01_param_key_mismatch"

    def test_synthetic_mechanisms_unaffected(self) -> None:
        """MECHANISM_REGISTRY contains exactly the 18 historical synthetic mechanisms."""
        assert len(MECHANISM_REGISTRY) == 18
        assert "trace_env_01_service_http_500" not in MECHANISM_REGISTRY
        assert "trace_prompt_01_tool_prohibition" not in MECHANISM_REGISTRY
        assert "trace_schema_01_param_key_mismatch" not in MECHANISM_REGISTRY

        all_tasks = list_mechanism_tasks()
        assert len(all_tasks) == 18
        categories = {t.category for t in all_tasks}
        assert {"prompt", "schema", "environment", "conjunctive", "null_clean", "null_bloat", "null_ext"}.issubset(categories)
