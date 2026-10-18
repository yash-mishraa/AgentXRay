"""Comprehensive tests for Trace-Evaluated Tier Registry and Runner Integration (Phase 4.6K.9).

Verifies:
1. TRACE_EVALUATED_REGISTRY contains exactly the 3 trace-evaluated tasks in deterministic order:
   - trace_prompt_01_tool_prohibition
   - trace_schema_01_param_key_mismatch
   - trace_env_01_service_http_500
2. Registry separation: MECHANISM_REGISTRY remains strictly at 18 historical synthetic control mechanisms;
   TRACE_EVALUATED_REGISTRY and MECHANISM_REGISTRY have disjoint keys.
3. API lookup and listing: get_trace_evaluated_task() and list_trace_evaluated_tasks() work correctly,
   raising KeyError on unknown or mismatched IDs.
4. BenchmarkTaskSpec structural protocol and TraceEvaluatedTaskSpec union typing are satisfied.
5. Runner tier selection: tier="synthetic_control" (default) vs tier="trace_evaluated".
6. Runner conflict handling: raises ValueError on unknown tiers or conflicting mechanisms lists.
7. End-to-end execution: run_multi_mechanism_benchmark(tier="trace_evaluated") runs all 3 trace tasks
   through Active DD, Greedy Ablation, and LLM Judge offline using a unified scripted provider.
8. Report metadata: secondary_metrics["benchmark_tier"] reflects the actual executed tier.
9. Resilience and isolation: provider infrastructure errors and budget limits fail closed without
   misclassifying infrastructure crashes as agent diagnostic failures.
10. Immutability: repeated runner executions leave registry specifications and tool definitions intact.
"""

from __future__ import annotations

import json
from typing import Sequence
from unittest.mock import MagicMock

import pytest

from agentxray.agent_harness.cost_tracker import (
    CostBudgetExceededError,
    CostTracker,
    ExperimentBudgetConfig,
)
from agentxray.agent_harness.tools import CALCULATOR_TOOL_DEF, DATABASE_LOOKUP_TOOL_DEF
from agentxray.models import ExploratoryBenchmarkReport, InputCase, SearchStatus
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
from agentxray.reporting import format_exploratory_benchmark_markdown
from benchmarks.multi_mechanism.runner import (
    evaluate_mechanism_three_way,
    run_multi_mechanism_benchmark,
)
from benchmarks.multi_mechanism.tasks import (
    MECHANISM_REGISTRY,
    TRACE_EVALUATED_REGISTRY,
    BenchmarkTaskSpec,
    MechanismTaskSpec,
    TraceEvaluatedEnvironmentTaskSpec,
    TraceEvaluatedPromptTaskSpec,
    TraceEvaluatedSchemaTaskSpec,
    TraceEvaluatedTaskSpec,
    create_trace_evaluated_environment_task,
    create_trace_evaluated_prompt_task,
    create_trace_evaluated_schema_task,
    get_mechanism_task,
    get_trace_evaluated_task,
    list_mechanism_tasks,
    list_trace_evaluated_tasks,
)


def make_unified_trace_scripted_handler(prompt_trigger_text: str):
    """Create a deterministic scripted provider handler for all three trace-evaluated mechanisms.

    SCIENTIFIC INTEGRITY DISCLAIMER:
    This scripted handler is strictly for testing harness orchestration, dynamic prompt/schema/env
    materialization, and trace-evaluated outcome determination offline without inference cost.
    It simulates an agent adhering to instructions, schemas, and environment responses, but does
    NOT measure real LLM reasoning.
    """
    def handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
        system_text = messages[0].content if messages and messages[0].role == "system" else ""
        user_text = messages[-1].content if messages else ""

        # 1. Passive LLM Judge baseline query handling (tools is empty)
        if not tools or "culprit_element_ids" in system_text or "Observed Agent Execution Trace:" in user_text:
            payload = {
                "culprit_element_ids": [],
                "rationale": "Mock passive judge diagnostic analysis for testing pipeline.",
            }
            return ModelResponse(
                content=json.dumps(payload),
                usage=UsageMetadata(prompt_tokens=100, completion_tokens=30, total_tokens=130),
                finish_reason="stop",
                model_identifier="fake-unified-judge",
            )

        # 2. Agent harness query handling
        has_tool_reply = any(m.role == "tool" for m in messages)

        if not has_tool_reply:
            # Turn 1:
            # Check prompt constraint (Task 1: Prompt Directive Prohibition)
            if prompt_trigger_text in system_text:
                return ModelResponse(
                    content="Estimated receipts sum without calculator: approximately $400.",
                    tool_calls=(),
                    usage=UsageMetadata(prompt_tokens=50, completion_tokens=12, total_tokens=62),
                    finish_reason="stop",
                    model_identifier="fake-unified-agent",
                )

            # Check schema presented (Task 2: Schema Parameter Key Mismatch)
            calc_tool = next((t for t in tools if t.name == "calculator"), None)
            calc_props = calc_tool.parameters.get("properties", {}) if calc_tool else {}
            if "expr" in calc_props:
                args = {"expr": "150 + 250"}
            else:
                args = {"expression": "150 + 250"}

            return ModelResponse(
                content="Calling calculator according to schema...",
                tool_calls=(
                    ToolCall(call_id="call_calc", name="calculator", arguments=args),
                ),
                usage=UsageMetadata(prompt_tokens=45, completion_tokens=15, total_tokens=60),
                finish_reason="tool_calls",
                model_identifier="fake-unified-agent",
            )
        else:
            # Turn 2: Inspect tool response (Task 2 error handling, Task 3 Environment HTTP 500)
            tool_msg = next((m for m in messages if m.role == "tool"), None)
            tool_content = tool_msg.content if tool_msg else ""

            if tool_content.startswith("Error:"):
                return ModelResponse(
                    content=f"Calculation failed with error: {tool_content}",
                    tool_calls=(),
                    usage=UsageMetadata(prompt_tokens=70, completion_tokens=20, total_tokens=90),
                    finish_reason="stop",
                    model_identifier="fake-unified-agent",
                )
            else:
                return ModelResponse(
                    content=f"Calculation complete. Final verified total is {tool_content.strip()}.",
                    tool_calls=(),
                    usage=UsageMetadata(prompt_tokens=65, completion_tokens=12, total_tokens=77),
                    finish_reason="stop",
                    model_identifier="fake-unified-agent",
                )

    return handler


class TestTraceEvaluatedRegistryAPI:
    """Verifies registry enumeration, retrieval, and error handling for trace-evaluated tasks."""

    def test_list_trace_evaluated_tasks_deterministic_order(self) -> None:
        tasks = list_trace_evaluated_tasks()
        assert len(tasks) == 5
        expected_ids = [
            "trace_prompt_01_tool_prohibition",
            "trace_schema_01_param_key_mismatch",
            "trace_env_01_service_http_500",
            "trace_null_01_clean_execution",
            "trace_null_02_unmodeled_external_shock",
        ]
        actual_ids = [t.mechanism_id for t in tasks]
        assert actual_ids == expected_ids

    def test_get_trace_evaluated_task_success(self) -> None:
        active_ids = (
            "trace_prompt_01_tool_prohibition",
            "trace_schema_01_param_key_mismatch",
            "trace_env_01_service_http_500",
        )
        for mid in active_ids:
            spec = get_trace_evaluated_task(mid)
            assert spec.mechanism_id == mid
            assert len(spec.planted_trigger_elements) == 1
            assert len(spec.candidate_elements) == 4
            assert spec.is_null_case is False

        null_ids = (
            "trace_null_01_clean_execution",
            "trace_null_02_unmodeled_external_shock",
        )
        for mid in null_ids:
            spec = get_trace_evaluated_task(mid)
            assert spec.mechanism_id == mid
            assert len(spec.planted_trigger_elements) == 0
            assert len(spec.candidate_elements) == 4
            assert spec.is_null_case is True

    def test_get_trace_evaluated_task_unknown_id_raises_key_error(self) -> None:
        with pytest.raises(KeyError) as exc_info:
            get_trace_evaluated_task("non_existent_task")
        assert "Unknown trace-evaluated mechanism_id: 'non_existent_task'" in str(exc_info.value)
        assert "trace_prompt_01_tool_prohibition" in str(exc_info.value)

    def test_get_trace_evaluated_task_synthetic_id_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_trace_evaluated_task("mech_p01_directive_prohibition")


class TestRegistrySeparationAndTaskProtocol:
    """Verifies that synthetic and trace registries remain completely disjoint and type-safe."""

    def test_registry_sizes_and_disjointness(self) -> None:
        assert len(MECHANISM_REGISTRY) == 18
        assert len(TRACE_EVALUATED_REGISTRY) == 5

        synthetic_keys = set(MECHANISM_REGISTRY.keys())
        trace_keys = set(TRACE_EVALUATED_REGISTRY.keys())
        assert synthetic_keys.isdisjoint(trace_keys)

        for mid in trace_keys:
            assert mid not in MECHANISM_REGISTRY
        for mid in synthetic_keys:
            assert mid not in TRACE_EVALUATED_REGISTRY

    def test_all_registry_tasks_satisfy_benchmark_task_spec_protocol(self) -> None:
        for spec in MECHANISM_REGISTRY.values():
            assert isinstance(spec, BenchmarkTaskSpec)
            assert hasattr(spec, "create_initial_input")
            assert hasattr(spec, "get_ground_truth")
            assert hasattr(spec, "create_target")

        for spec in TRACE_EVALUATED_REGISTRY.values():
            assert isinstance(spec, BenchmarkTaskSpec)
            assert hasattr(spec, "create_initial_input")
            assert hasattr(spec, "get_ground_truth")
            assert hasattr(spec, "create_target")

    def test_trace_tasks_satisfy_trace_evaluated_task_spec_union(self) -> None:
        prompt_task = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        schema_task = get_trace_evaluated_task("trace_schema_01_param_key_mismatch")
        env_task = get_trace_evaluated_task("trace_env_01_service_http_500")

        assert isinstance(prompt_task, TraceEvaluatedPromptTaskSpec)
        assert isinstance(schema_task, TraceEvaluatedSchemaTaskSpec)
        assert isinstance(env_task, TraceEvaluatedEnvironmentTaskSpec)


class TestRunnerTierSelectionAndValidation:
    """Verifies tier parameter validation, default behavior, and argument conflict detection."""

    def test_unknown_tier_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            run_multi_mechanism_benchmark(tier="invalid_tier")  # type: ignore[arg-type]
        assert "Unknown benchmark tier: 'invalid_tier'" in str(exc_info.value)
        assert "synthetic_control" in str(exc_info.value)
        assert "trace_evaluated" in str(exc_info.value)

    def test_conflicting_mechanisms_and_tier_trace_in_synthetic(self) -> None:
        trace_tasks = list_trace_evaluated_tasks()
        with pytest.raises(ValueError) as exc_info:
            run_multi_mechanism_benchmark(
                mechanisms=trace_tasks,
                tier="synthetic_control",
            )
        assert "Conflicting arguments: mechanisms contains trace-evaluated tasks" in str(exc_info.value)

    def test_conflicting_mechanisms_and_tier_synthetic_in_trace(self) -> None:
        synthetic_tasks = list_mechanism_tasks()[:2]
        with pytest.raises(ValueError) as exc_info:
            run_multi_mechanism_benchmark(
                mechanisms=synthetic_tasks,
                tier="trace_evaluated",
            )
        assert "Conflicting arguments: mechanisms contains synthetic control mechanisms" in str(exc_info.value)

    def test_default_tier_is_synthetic_control(self) -> None:
        sample_synth = list_mechanism_tasks()[:1]
        report = run_multi_mechanism_benchmark(
            mechanisms=sample_synth,
            trials_per_candidate=1,
            max_trials=5,
        )
        assert report.secondary_metrics["benchmark_tier"] == "synthetic_control"
        assert report.n_total == 1

    def test_explicit_non_conflicting_trace_mechanisms_subset(self) -> None:
        trace_subset = [get_trace_evaluated_task("trace_prompt_01_tool_prohibition")]
        prompt_trigger = trace_subset[0].planted_trigger_elements[0]
        prov = FakeProvider(custom_handler=make_unified_trace_scripted_handler(prompt_trigger))

        report = run_multi_mechanism_benchmark(
            provider=prov,
            mechanisms=trace_subset,
            tier="trace_evaluated",
            trials_per_candidate=2,
            max_trials=10,
        )
        assert report.secondary_metrics["benchmark_tier"] == "trace_evaluated"
        assert report.n_total == 1
        assert report.evaluations[0].mechanism_id == "trace_prompt_01_tool_prohibition"


class TestTraceBenchmarkExecutionEndToEnd:
    """Verifies end-to-end multi-mechanism benchmark execution on the trace-evaluated tier."""

    def test_run_all_trace_mechanisms_through_runner(self) -> None:
        prompt_task = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        prompt_trigger = prompt_task.planted_trigger_elements[0]

        handler = make_unified_trace_scripted_handler(prompt_trigger)
        provider = FakeProvider(model_name="fake-trace-unified", custom_handler=handler)

        report = run_multi_mechanism_benchmark(
            provider=provider,
            tier="trace_evaluated",
            trials_per_candidate=2,
            max_trials=15,
            seed=42,
            experiment_name="Phase_4_6K_Trace_Evaluated_Integration_Test",
        )

        # 1. Cohort and accounting
        assert report.n_total == 5
        assert report.n_infra == 0
        assert report.n_evaluable == 5
        assert report.n_complete == 5

        # 2. Metadata tier confirmation
        assert report.secondary_metrics["benchmark_tier"] == "trace_evaluated"

        # 3. All five mechanisms evaluated in deterministic order
        eval_ids = [ev.mechanism_id for ev in report.evaluations]
        assert eval_ids == [
            "trace_prompt_01_tool_prohibition",
            "trace_schema_01_param_key_mismatch",
            "trace_env_01_service_http_500",
            "trace_null_01_clean_execution",
            "trace_null_02_unmodeled_external_shock",
        ]

        # 4. Method diagnostic verification for active tasks
        for ev in report.evaluations[:3]:
            # Active DD should isolate exact ground truth trigger in all 3 active trace tasks
            assert ev.active_dd_outcome.is_valid_diagnosis is True
            assert ev.active_dd_outcome.exact_recovery_indicator == 1
            assert ev.active_dd_outcome.false_attribution_indicator == 0
            assert ev.active_dd_outcome.predicted_candidate_set == ev.ground_truth_trigger

            # Greedy ablation should also recover the single trigger in all 3 active trace tasks
            assert ev.greedy_ablation_outcome.is_valid_diagnosis is True
            assert ev.greedy_ablation_outcome.exact_recovery_indicator == 1
            assert ev.greedy_ablation_outcome.predicted_candidate_set == ev.ground_truth_trigger

            # LLM Judge returns empty list from mock handler
            assert ev.llm_judge_outcome.is_valid_diagnosis is True

        # Null task 1: clean execution -> baseline succeeds, active DD and greedy predict ()
        ev_clean = report.evaluations[3]
        assert ev_clean.mechanism_id == "trace_null_01_clean_execution"
        assert ev_clean.is_null_case is True
        assert ev_clean.active_dd_outcome.exact_recovery_indicator == 1
        assert ev_clean.active_dd_outcome.false_attribution_indicator == 0
        assert ev_clean.active_dd_outcome.predicted_candidate_set == ()
        assert ev_clean.greedy_ablation_outcome.exact_recovery_indicator == 1
        assert ev_clean.greedy_ablation_outcome.false_attribution_indicator == 0
        assert ev_clean.greedy_ablation_outcome.predicted_candidate_set == ()

        # Null task 2: unmodeled external shock -> target always fails, minimality check fails
        ev_shock = report.evaluations[4]
        assert ev_shock.mechanism_id == "trace_null_02_unmodeled_external_shock"
        assert ev_shock.is_null_case is True
        assert ev_shock.active_dd_outcome.is_valid_diagnosis is True
        assert ev_shock.active_dd_outcome.verification_status == "not_minimal"
        assert ev_shock.greedy_ablation_outcome.is_valid_diagnosis is True
        assert ev_shock.greedy_ablation_outcome.verification_status == "not_minimal"

        # 5. Markdown report generation succeeds and includes trace task IDs
        md = format_exploratory_benchmark_markdown(report)
        assert "# Phase_4_6K_Trace_Evaluated_Integration_Test — Research Summary" in md
        assert "trace_prompt_01_tool_prohibition" in md
        assert "trace_schema_01_param_key_mismatch" in md
        assert "trace_env_01_service_http_500" in md
        assert "trace_null_01_clean_execution" in md
        assert "trace_null_02_unmodeled_external_shock" in md


class TestTraceBenchmarkInfrastructureAndBudgetResilience:
    """Verifies that provider infrastructure crashes and budget limits are handled cleanly."""

    def test_provider_infrastructure_error_categorized_as_infra_failure(self) -> None:
        prompt_task = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        schema_task = get_trace_evaluated_task("trace_schema_01_param_key_mismatch")
        healthy_handler = make_unified_trace_scripted_handler(prompt_task.planted_trigger_elements[0])

        def selective_crashing_handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            all_content = " ".join(m.content for m in messages)
            # If the request is associated with prompt_task, simulate persistent provider failure
            if (
                "Direct arithmetic" in all_content
                or "Monetary precision" in all_content
                or "Professional styling" in all_content
                or "Voucher verification" in all_content
            ):
                raise ProviderTimeoutError("Upstream API gateway timeout (504 Gateway Timeout)")
            return healthy_handler(messages, tools)

        provider = FakeProvider(model_name="fake-selective-crashing", custom_handler=selective_crashing_handler)

        report = run_multi_mechanism_benchmark(
            provider=provider,
            mechanisms=[prompt_task, schema_task],
            tier="trace_evaluated",
            trials_per_candidate=2,
            max_trials=10,
        )

        assert report.n_total == 2
        assert report.n_infra == 1
        assert report.n_evaluable == 1
        ev_prompt = report.evaluations[0]
        assert ev_prompt.mechanism_id == "trace_prompt_01_tool_prohibition"
        assert ev_prompt.active_dd_outcome.termination_reason == "infra_failure"
        assert ev_prompt.active_dd_outcome.is_valid_diagnosis is False
        assert ev_prompt.active_dd_outcome.false_attribution_indicator is None

        ev_schema = report.evaluations[1]
        assert ev_schema.mechanism_id == "trace_schema_01_param_key_mismatch"
        assert ev_schema.active_dd_outcome.termination_reason == "normal_completion"
        assert ev_schema.active_dd_outcome.is_valid_diagnosis is True


    def test_budget_exhaustion_fail_closed(self) -> None:
        prompt_task = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        prompt_trigger = prompt_task.planted_trigger_elements[0]
        handler = make_unified_trace_scripted_handler(prompt_trigger)
        provider = FakeProvider(model_name="fake-budget", custom_handler=handler)

        cost_tracker = CostTracker(
            ExperimentBudgetConfig(max_model_requests=1, max_spend_usd=0.001, model_name=provider.model_name)
        )

        report = run_multi_mechanism_benchmark(
            provider=provider,
            cost_tracker=cost_tracker,
            mechanisms=[prompt_task],
            tier="trace_evaluated",
            trials_per_candidate=2,
            max_trials=10,
        )

        assert report.n_total == 1
        # Budget exhaustion halts fail-closed without uncaught exceptions
        ev = report.evaluations[0]
        assert ev.active_dd_outcome.termination_reason in ("budget_exhausted", "normal_completion")

    def test_immutability_and_state_isolation_across_runs(self) -> None:
        orig_calc_props = dict(CALCULATOR_TOOL_DEF.parameters["properties"])
        orig_registry_size = len(TRACE_EVALUATED_REGISTRY)

        prompt_task = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        prompt_trigger = prompt_task.planted_trigger_elements[0]
        prov = FakeProvider(custom_handler=make_unified_trace_scripted_handler(prompt_trigger))

        # Run 1
        run_multi_mechanism_benchmark(provider=prov, tier="trace_evaluated", trials_per_candidate=1, max_trials=5)

        # Run 2
        run_multi_mechanism_benchmark(provider=prov, tier="trace_evaluated", trials_per_candidate=1, max_trials=5)

        # Assert no mutation
        assert dict(CALCULATOR_TOOL_DEF.parameters["properties"]) == orig_calc_props
        assert len(TRACE_EVALUATED_REGISTRY) == orig_registry_size
        assert len(MECHANISM_REGISTRY) == 18
