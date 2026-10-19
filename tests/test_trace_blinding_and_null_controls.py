"""Targeted tests for Candidate Blinding, Leakage Remediation, and Trace Null Controls (Phase 4.6K.11).

Verifies:
1. Candidate Blinding in LLMJudgeBaseline:
   - Judge prompts display neutral candidate IDs (elem_0, elem_1, ...) instead of mechanism prefixes.
   - Neutral IDs map bidirectionally and deterministically to canonical elements.
   - Resolved predictions faithfully match ground truth for scoring.
   - Backward-compatible alias mapping (prefixes, full strings) is preserved in id_to_full.
2. Candidate-Only Diagnostic Input Mode:
   - build_judge_prompt(initial_input, execution_trace=None) outputs candidate elements without trace blocks.
   - Neutral IDs in candidate-only mode exactly match those in candidate-plus-trace mode.
   - diagnose_candidate_only() executes passive diagnosis without trace information.
3. Lexical Leakage Remediation:
   - Active trace mechanisms (prompt, schema, env) exhibit zero or near-zero lexical overlap with observable symptoms.
   - No active trace task exceeds Jaccard (>= 0.20) or LCS (>= 15 chars) warning thresholds.
   - audit_all_mechanisms(tier="trace_evaluated") reports 0/5 flagged mechanisms.
4. Trace Null Control 1: Clean Execution (trace_null_01_clean_execution):
   - is_null_case is True, planted_trigger_elements is empty ().
   - Target execution succeeds cleanly on initial unablated input.
   - Active DD and Greedy Ablation identify baseline pass and return empty candidate set ().
   - Empty diagnosis scores false_attribution=0 and exact_recovery=1.
   - Spurious attribution of distractors scores false_attribution=1 and exact_recovery=0.
5. Trace Null Control 2: Unmodeled External Shock (trace_null_02_unmodeled_external_shock):
   - is_null_case is True, planted_trigger_elements is empty ().
   - Mock calculator always fails with HTTP 500 regardless of candidate ablation, including empty input.
   - Minimizers verify that dropping elements does not cure failure, establishing verified empty causal set.
   - Spurious attribution of candidate elements to the external shock scores false_attribution=1.
6. Registry Coherence & Separation:
   - TRACE_EVALUATED_REGISTRY contains exactly 5 tasks in deterministic order.
   - MECHANISM_REGISTRY remains strictly at 18 historical synthetic mechanisms.
   - Both registries are completely disjoint and satisfy BenchmarkTaskSpec.
"""

from __future__ import annotations

import json
from typing import Sequence

import pytest

from agentxray.agent_harness.agent import AgentExecutionTrace
from agentxray.agent_harness.cost_tracker import CostTracker, ExperimentBudgetConfig
from agentxray.agent_harness.tools import CALCULATOR_TOOL_DEF
from agentxray.llm_judge import LLMJudgeBaseline, build_judge_prompt
from agentxray.models import (
    CandidateEvaluation,
    ExperimentConfig,
    InputCase,
    SearchStatus,
    StochasticMinimalityStatus,
    TargetOutcome,
)
from agentxray.providers.base import Message, ModelProvider, ModelResponse, ToolCall, ToolDefinition, UsageMetadata
from agentxray.providers.fake import FakeProvider
from agentxray.runner import InterventionRunner
from agentxray.stochastic_baselines import run_stochastic_greedy_ablation
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.multi_mechanism.leakage_audit import (
    audit_all_mechanisms,
    audit_mechanism_leakage,
)
from benchmarks.multi_mechanism.runner import score_diagnostic_outcome
from benchmarks.multi_mechanism.tasks import (
    MECHANISM_REGISTRY,
    TRACE_EVALUATED_REGISTRY,
    BenchmarkTaskSpec,
    TraceEvaluatedEnvironmentTaskSpec,
    TraceEvaluatedPromptTaskSpec,
    TraceEvaluatedSchemaTaskSpec,
    create_trace_evaluated_clean_null_task,
    create_trace_evaluated_environment_task,
    create_trace_evaluated_external_shock_null_task,
    create_trace_evaluated_prompt_task,
    create_trace_evaluated_schema_task,
    get_mechanism_task,
    get_trace_evaluated_task,
    list_mechanism_tasks,
    list_trace_evaluated_tasks,
)


class TestCandidateBlindingAndResolverFidelity:
    """Verifies that passive judge inputs are blinded to neutral IDs while preserving resolution fidelity."""

    def test_judge_prompt_blinds_candidate_ids(self) -> None:
        spec = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        initial_input = spec.create_initial_input()

        trace = AgentExecutionTrace(
            messages=(),
            final_response="Calculated total.",
            finish_reason="stop",
            error_message="Agent failed to invoke required tool 'calculator'.",
        )

        messages, id_to_full, display_map = build_judge_prompt(initial_input=initial_input, execution_trace=trace)
        prompt_str = f"{messages[0].content}\n{messages[1].content}"

        # 1. Blinded neutral element IDs must be present in display_map and user prompt
        for idx in range(len(initial_input.elements)):
            assert f"elem_{idx}" in display_map
            assert f'- id: "elem_{idx}"' in messages[1].content

        # 2. Raw prefix identifiers must NOT be used as keys in the prompt list
        assert "trace_prompt_01:" not in prompt_str
        assert "Direct arithmetic:" in prompt_str  # Description text present

    def test_judge_diagnosis_resolves_blinded_element_ids_to_canonical(self) -> None:
        spec = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        initial_input = spec.create_initial_input()

        trace = AgentExecutionTrace(
            messages=(),
            final_response="Response without tool.",
            finish_reason="stop",
            error_message=spec.observable_failure_symptom,
        )

        # Trigger is at index 0 in candidate_elements
        trigger_text = spec.planted_trigger_elements[0]

        # Scripted judge returns neutral ID elem_0
        response_payload = {
            "culprit_element_ids": ["elem_0"],
            "rationale": "Element elem_0 directed the agent not to use arithmetic tools.",
        }

        def mock_judge_handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            return ModelResponse(
                content=json.dumps(response_payload),
                usage=UsageMetadata(prompt_tokens=80, completion_tokens=25, total_tokens=105),
                finish_reason="stop",
                model_identifier="fake-blinded-judge",
            )

        prov = FakeProvider(custom_handler=mock_judge_handler)
        judge = LLMJudgeBaseline(provider=prov)
        result = judge.diagnose(initial_input, trace)

        assert result.is_valid_diagnosis is True
        assert result.predicted_candidate_set == (trigger_text,)

    def test_backward_compatible_prefix_and_full_string_resolution(self) -> None:
        spec = get_mechanism_task("mech_p01_directive_prohibition")
        initial_input = spec.create_initial_input()
        trigger_elem = spec.planted_trigger_elements[0]

        trace = AgentExecutionTrace(
            messages=(),
            final_response="Calculated total without tool.",
            finish_reason="stop",
            error_message=spec.observable_failure_symptom,
        )

        # 1. Prediction using prefix alias resolves correctly
        trigger_prefix = trigger_elem.split(":")[0].strip()
        prov_prefix = FakeProvider(
            custom_handler=lambda msgs, tools: ModelResponse(
                content=json.dumps({"culprit_element_ids": [trigger_prefix]}),
                usage=UsageMetadata(prompt_tokens=50, completion_tokens=10, total_tokens=60),
                finish_reason="stop",
                model_identifier="fake-test-judge",
            )
        )
        res_prefix = LLMJudgeBaseline(provider=prov_prefix).diagnose(initial_input, trace)
        assert res_prefix.predicted_candidate_set == (trigger_elem,)

        # 2. Prediction using full string resolves correctly
        prov_full = FakeProvider(
            custom_handler=lambda msgs, tools: ModelResponse(
                content=json.dumps({"culprit_element_ids": [trigger_elem]}),
                usage=UsageMetadata(prompt_tokens=50, completion_tokens=10, total_tokens=60),
                finish_reason="stop",
                model_identifier="fake-test-judge",
            )
        )
        res_full = LLMJudgeBaseline(provider=prov_full).diagnose(initial_input, trace)
        assert res_full.predicted_candidate_set == (trigger_elem,)


class TestCandidateOnlyDiagnosticMode:
    """Verifies the trace-free candidate-only diagnostic prompt and execution path."""

    def test_build_judge_prompt_candidate_only_mode(self) -> None:
        spec = get_trace_evaluated_task("trace_schema_01_param_key_mismatch")
        initial_input = spec.create_initial_input()

        # Build prompt with execution_trace=None
        messages, id_to_full, display_map = build_judge_prompt(initial_input=initial_input, execution_trace=None)
        prompt_str = f"{messages[0].content}\n{messages[1].content}"

        # 1. Candidate elements are present with neutral IDs
        for idx in range(len(initial_input.elements)):
            assert f"elem_{idx}" in display_map
            assert f'- id: "elem_{idx}"' in messages[1].content

        # 2. Trace section is completely omitted
        assert "Observed Agent Execution Trace:" not in prompt_str
        assert "Agent Final Response:" not in prompt_str
        assert "Error Message:" not in prompt_str
        assert "Tool Calls Executed:" not in prompt_str

        # 3. Diagnostic instruction specifies candidate-only prior identification
        assert "candidate input element specifications" in prompt_str or "inherently faulty" in prompt_str

    def test_candidate_only_ids_match_trace_evaluated_ids(self) -> None:
        spec = get_trace_evaluated_task("trace_env_01_service_http_500")
        initial_input = spec.create_initial_input()

        trace = AgentExecutionTrace(
            messages=(),
            final_response="",
            finish_reason="stop",
            error_message="Error: 500",
        )

        msgs_trace, _, display_trace = build_judge_prompt(initial_input=initial_input, execution_trace=trace)
        msgs_notrace, _, display_notrace = build_judge_prompt(initial_input=initial_input, execution_trace=None)

        # Neutral IDs and descriptions must match identically
        assert display_trace == display_notrace
        for cid, desc in display_trace.items():
            line = f'- id: "{cid}" | description: "{desc}"'
            assert line in msgs_trace[1].content
            assert line in msgs_notrace[1].content

    def test_diagnose_candidate_only_execution(self) -> None:
        spec = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        initial_input = spec.create_initial_input()

        captured_prompts: list[str] = []

        def recording_handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            prompt_content = " ".join(m.content for m in messages)
            captured_prompts.append(prompt_content)
            return ModelResponse(
                content=json.dumps({"culprit_element_ids": ["elem_0"], "rationale": "A priori heuristic."}),
                usage=UsageMetadata(prompt_tokens=60, completion_tokens=15, total_tokens=75),
                finish_reason="stop",
                model_identifier="fake-candidate-only-judge",
            )

        judge = LLMJudgeBaseline(provider=FakeProvider(custom_handler=recording_handler))
        result = judge.diagnose_candidate_only(initial_input)

        assert result.is_valid_diagnosis is True
        assert result.predicted_candidate_set == (spec.planted_trigger_elements[0],)
        assert len(captured_prompts) == 1
        assert "Observed Agent Execution Trace:" not in captured_prompts[0]


class TestLexicalLeakageRemediation:
    """Verifies that candidate descriptions no longer leak observable symptoms via keyword overlap."""

    def test_trace_prompt_task_leakage_remediation(self) -> None:
        spec = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        report = audit_mechanism_leakage(spec)

        # Jaccard overlap between trigger and symptom must be 0.000
        assert report.trigger_jaccard_overlap == pytest.approx(0.000, abs=1e-3)
        # LCS must be less than 8 characters (and strictly < 15 threshold)
        assert report.trigger_lcs_length < 8
        assert report.is_flagged is False
        assert report.warning_message is None

    def test_trace_schema_task_leakage_remediation(self) -> None:
        spec = get_trace_evaluated_task("trace_schema_01_param_key_mismatch")
        report = audit_mechanism_leakage(spec)

        # Parameter names 'expression' and 'expr' eliminated from trigger description
        assert "expression" not in spec.planted_trigger_elements[0].lower()
        assert "expr" not in spec.planted_trigger_elements[0].lower()
        # Jaccard overlap < 0.10, LCS < 15
        assert report.trigger_jaccard_overlap < 0.10
        assert report.trigger_lcs_length < 15
        assert report.is_flagged is False

    def test_trace_env_task_leakage_remediation(self) -> None:
        spec = get_trace_evaluated_task("trace_env_01_service_http_500")
        report = audit_mechanism_leakage(spec)

        # Telltale codes eliminated from trigger description
        assert "500" not in spec.planted_trigger_elements[0]
        assert "internal server error" not in spec.planted_trigger_elements[0].lower()
        # Jaccard overlap < 0.10, LCS < 15
        assert report.trigger_jaccard_overlap < 0.10
        assert report.trigger_lcs_length < 15
        assert report.is_flagged is False

    def test_audit_all_mechanisms_trace_evaluated_tier_zero_flags(self) -> None:
        summary = audit_all_mechanisms(tier="trace_evaluated")
        assert summary.total_mechanisms == 5
        assert summary.flagged_mechanisms == 0
        assert summary.mean_trigger_jaccard < 0.05
        for r in summary.reports:
            assert r.is_flagged is False


class TestTraceNullControlCleanExecution:
    """Verifies that trace_null_01_clean_execution passes and evaluates false attribution."""

    def test_clean_null_task_specification(self) -> None:
        spec = get_trace_evaluated_task("trace_null_01_clean_execution")
        assert spec.is_null_case is True
        assert spec.planted_trigger_elements == ()
        assert len(spec.candidate_elements) == 4
        assert spec.observable_failure_symptom == ""

    def test_clean_null_target_succeeds_on_initial_input(self) -> None:
        spec = get_trace_evaluated_task("trace_null_01_clean_execution")
        target = spec.create_target()
        initial_input = spec.create_initial_input()

        res = target.execute(initial_input)
        assert res.outcome == TargetOutcome.SUCCESS
        assert target.last_trace is not None
        assert "400" in target.last_trace.final_response

    def test_active_dd_and_greedy_return_empty_set_on_clean_null_task(self) -> None:
        spec = get_trace_evaluated_task("trace_null_01_clean_execution")
        initial_input = spec.create_initial_input()

        # 1. Active DD Minimizer
        dd_target = spec.create_target()
        dd_runner = InterventionRunner(
            config=ExperimentConfig(experiment_name="test_clean_null_dd", max_trials=10),
            target_fn=dd_target.execute,
        )
        minimizer = StochasticDeltaDebugMinimizer(
            runner=dd_runner,
            trials_per_candidate=2,
            failure_threshold=0.30,
        )
        cond = minimizer.minimize(initial_input)

        # Baseline failure rate is 0.0 < 0.30 -> NO_TRIGGER_FOUND, subset is ()
        assert cond.subset_elements == ()
        assert cond.status == SearchStatus.NO_TRIGGER_FOUND
        assert cond.is_failure is False

        # Score Active DD outcome
        dd_score = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="active_dd",
            candidate_set=cond.subset_elements,
            is_valid_diagnosis=True,
            ground_truth_trigger=(),
            is_null_case=True,
        )
        assert dd_score.false_attribution_indicator == 0
        assert dd_score.exact_recovery_indicator == 1

        # 2. Greedy Ablation
        greedy_target = spec.create_target()
        greedy_runner = InterventionRunner(
            config=ExperimentConfig(experiment_name="test_clean_null_greedy", max_trials=10),
            target_fn=greedy_target.execute,
        )
        greedy_res = run_stochastic_greedy_ablation(
            initial_input=initial_input,
            runner=greedy_runner,
            trials_per_candidate=2,
            failure_threshold=0.30,
        )

        assert greedy_res.discovered_elements == ()
        assert greedy_res.failure_reproduced is False

        greedy_score = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="greedy_ablation",
            candidate_set=greedy_res.discovered_elements,
            is_valid_diagnosis=True,
            ground_truth_trigger=(),
            is_null_case=True,
        )
        assert greedy_score.false_attribution_indicator == 0
        assert greedy_score.exact_recovery_indicator == 1

    def test_spurious_attribution_on_clean_null_task_penalized(self) -> None:
        spec = get_trace_evaluated_task("trace_null_01_clean_execution")
        distractor = spec.candidate_elements[0]

        score = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="passive_judge",
            candidate_set=(distractor,),
            is_valid_diagnosis=True,
            ground_truth_trigger=(),
            is_null_case=True,
        )
        assert score.false_attribution_indicator == 1
        assert score.exact_recovery_indicator == 0


class TestTraceNullControlExternalShock:
    """Verifies that trace_null_02_unmodeled_external_shock fails unconditionally and penalizes false attribution."""

    def test_external_shock_target_fails_unconditionally(self) -> None:
        spec = get_trace_evaluated_task("trace_null_02_unmodeled_external_shock")
        target = spec.create_target()
        initial_input = spec.create_initial_input()

        # 1. Full candidate set fails
        res_full = target.execute(initial_input)
        assert res_full.outcome == TargetOutcome.FAILURE
        assert "500 Internal Server Error" in res_full.output_message

        # 2. Empty candidate set ALSO fails (shock is uncaused by candidates)
        empty_input = initial_input.subset(())
        res_empty = target.execute(empty_input)
        assert res_empty.outcome == TargetOutcome.FAILURE
        assert "500 Internal Server Error" in res_empty.output_message

    def test_minimizers_verify_empty_on_external_shock(self) -> None:
        spec = get_trace_evaluated_task("trace_null_02_unmodeled_external_shock")
        initial_input = spec.create_initial_input()

        # Active DD minimizer
        target = spec.create_target()
        runner = InterventionRunner(
            config=ExperimentConfig(experiment_name="test_shock_dd", max_trials=15),
            target_fn=target.execute,
        )
        minimizer = StochasticDeltaDebugMinimizer(
            runner=runner,
            trials_per_candidate=1,
            failure_threshold=0.30,
        )
        cond = minimizer.minimize(initial_input)

        # Verification must classify condition as VERIFIED_EMPTY because empty subset still fails
        assert cond.minimality_status == StochasticMinimalityStatus.VERIFIED_EMPTY
        assert cond.subset_elements == ()
        assert "empty-set failure rate" in cond.status_message.lower()

    def test_spurious_attribution_on_external_shock_penalized(self) -> None:
        spec = get_trace_evaluated_task("trace_null_02_unmodeled_external_shock")
        distractor = spec.candidate_elements[0]

        # Valid empty diagnosis on external shock: Y=0, R=1
        score_empty = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="active_dd",
            candidate_set=(),
            is_valid_diagnosis=True,
            ground_truth_trigger=(),
            is_null_case=True,
        )
        assert score_empty.false_attribution_indicator == 0
        assert score_empty.exact_recovery_indicator == 1

        # Spurious blame on benign candidate: Y=1, R=0
        score_spurious = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="passive_judge",
            candidate_set=(distractor,),
            is_valid_diagnosis=True,
            ground_truth_trigger=(),
            is_null_case=True,
        )
        assert score_spurious.false_attribution_indicator == 1
        assert score_spurious.exact_recovery_indicator == 0


class TestRegistryIntegrityAndDisjointness:
    """Verifies registry separation and protocol conformance."""

    def test_registry_sizes_and_task_types(self) -> None:
        assert len(MECHANISM_REGISTRY) == 18
        assert len(TRACE_EVALUATED_REGISTRY) == 5

        # Disjoint keys
        assert set(MECHANISM_REGISTRY.keys()).isdisjoint(set(TRACE_EVALUATED_REGISTRY.keys()))

        # Trace registry order
        expected_order = (
            "trace_prompt_01_tool_prohibition",
            "trace_schema_01_param_key_mismatch",
            "trace_env_01_service_http_500",
            "trace_null_01_clean_execution",
            "trace_null_02_unmodeled_external_shock",
        )
        assert tuple(TRACE_EVALUATED_REGISTRY.keys()) == expected_order

    def test_benchmark_task_spec_protocol_conformance(self) -> None:
        for task in list_trace_evaluated_tasks():
            assert isinstance(task, BenchmarkTaskSpec)
            assert hasattr(task, "mechanism_id")
            assert hasattr(task, "candidate_elements")
            assert hasattr(task, "planted_trigger_elements")
            assert hasattr(task, "create_initial_input")
            assert hasattr(task, "create_target")
            assert hasattr(task, "get_ground_truth")
