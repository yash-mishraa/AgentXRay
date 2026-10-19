"""Deterministic offline tests for candidate-only judge integration and metric reporting (Phase 4.6K.15).

Verifies the 11 requirements:
1. Candidate-only and trace-informed judges receive identical candidate information and neutral IDs.
2. Candidate-only prompts contain no trace or trace-derived summary.
3. Both judge modes resolve returned candidate IDs to the correct canonical candidates.
4. The four diagnostic arms are represented separately in trace-evaluated results.
5. Existing three-arm outputs remain compatible wherever intended.
6. A candidate-only invalid response does not corrupt other diagnostic arms.
7. Empty, invalid, and inconclusive predictions retain their proper scoring semantics.
8. Metric denominators and missing-outcome counts are correct.
9. The five-task trace registry and 18-mechanism synthetic registry remain unchanged in count.
10. Existing statistical test behavior remains unchanged in this phase.
11. All relevant regression tests continue to pass.
"""

from __future__ import annotations

import json
from typing import Sequence
import pytest

from agentxray.agent_harness.agent import AgentExecutionTrace
from agentxray.llm_judge import LLMJudgeBaseline, build_judge_prompt
from agentxray.models import (
    DiagnosticOutcome,
    ExploratoryBenchmarkReport,
    InputCase,
    MechanismEvaluationResult,
)
from agentxray.providers.base import (
    Message,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    UsageMetadata,
)
from agentxray.providers.fake import FakeProvider
from agentxray.reporting import format_exploratory_benchmark_markdown
from agentxray.statistics import mcnemar_exact_test
from benchmarks.multi_mechanism.runner import (
    ScriptedJudgeProvider,
    evaluate_mechanism_three_way,
    run_multi_mechanism_benchmark,
    score_diagnostic_outcome,
)
from benchmarks.multi_mechanism.tasks import (
    MECHANISM_REGISTRY,
    TRACE_EVALUATED_REGISTRY,
    get_mechanism_task,
    get_trace_evaluated_task,
    list_mechanism_tasks,
    list_trace_evaluated_tasks,
)


def make_four_arm_test_handler(
    prompt_trigger_text: str = "",
    trace_judge_pred: Sequence[str] = (),
    cand_judge_pred: Sequence[str] = (),
    malformed_cand_only: bool = False,
):
    """Create a deterministic FakeProvider handler for both ControlledAgent and both Judge modes."""
    def handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
        system_text = messages[0].content if messages and messages[0].role == "system" else ""
        user_text = messages[-1].content if messages else ""

        # 1. Candidate-Only Judge query
        is_candidate_only = "inherently faulty" in system_text or any("based only on the candidate specifications" in m.content for m in messages)
        if is_candidate_only:
            if malformed_cand_only:
                return ModelResponse(
                    content="MALFORMED_NON_JSON_OUTPUT",
                    usage=UsageMetadata(prompt_tokens=80, completion_tokens=10, total_tokens=90),
                    finish_reason="stop",
                    model_identifier="fake-judge",
                )
            payload = {
                "culprit_element_ids": list(cand_judge_pred),
                "rationale": "Candidate-only scripted prediction.",
            }
            return ModelResponse(
                content=json.dumps(payload),
                usage=UsageMetadata(prompt_tokens=80, completion_tokens=25, total_tokens=105),
                finish_reason="stop",
                model_identifier="fake-judge",
            )

        # 2. Trace-Informed Judge query
        if not tools or "Observed Agent Execution Trace:" in user_text or "culprit_element_ids" in system_text:
            payload = {
                "culprit_element_ids": list(trace_judge_pred),
                "rationale": "Trace-informed scripted prediction.",
            }
            return ModelResponse(
                content=json.dumps(payload),
                usage=UsageMetadata(prompt_tokens=120, completion_tokens=30, total_tokens=150),
                finish_reason="stop",
                model_identifier="fake-judge",
            )

        # 3. Agent harness execution query
        has_tool_reply = any(m.role == "tool" for m in messages)
        if not has_tool_reply:
            if prompt_trigger_text and prompt_trigger_text in system_text:
                return ModelResponse(
                    content="Estimated receipts sum without calculator: approximately $400.",
                    tool_calls=(),
                    usage=UsageMetadata(prompt_tokens=50, completion_tokens=12, total_tokens=62),
                    finish_reason="stop",
                    model_identifier="fake-agent",
                )

            calc_tool = next((t for t in tools if t.name == "calculator"), None)
            calc_props = calc_tool.parameters.get("properties", {}) if calc_tool else {}
            args = {"expr": "150 + 250"} if "expr" in calc_props else {"expression": "150 + 250"}
            return ModelResponse(
                content="Calling calculator according to schema...",
                tool_calls=(
                    ToolCall(call_id="call_calc", name="calculator", arguments=args),
                ),
                usage=UsageMetadata(prompt_tokens=45, completion_tokens=15, total_tokens=60),
                finish_reason="tool_calls",
                model_identifier="fake-agent",
            )
        else:
            tool_msg = next((m for m in messages if m.role == "tool"), None)
            tool_content = tool_msg.content if tool_msg else ""
            if tool_content.startswith("Error:"):
                return ModelResponse(
                    content=f"Calculation failed with error: {tool_content}",
                    tool_calls=(),
                    usage=UsageMetadata(prompt_tokens=70, completion_tokens=20, total_tokens=90),
                    finish_reason="stop",
                    model_identifier="fake-agent",
                )
            return ModelResponse(
                content="The total sum calculated is 400.",
                tool_calls=(),
                usage=UsageMetadata(prompt_tokens=60, completion_tokens=15, total_tokens=75),
                finish_reason="stop",
                model_identifier="fake-agent",
            )

    return handler


class TestCandidateOnlyJudgeArmIntegration:
    """Offline validation of the candidate-only judge arm and metric reporting."""

    def test_requirement_1_identical_candidates_and_neutral_ids(self) -> None:
        """1. Candidate-only and trace-informed judges receive identical candidate information and neutral IDs."""
        for task_id in ("trace_prompt_01_tool_prohibition", "trace_schema_01_param_key_mismatch"):
            spec = get_trace_evaluated_task(task_id)
            initial_input = spec.create_initial_input(seed=42)
            mock_trace = AgentExecutionTrace(
                messages=(
                    Message(role="system", content="System instruction"),
                    Message(role="user", content="Calculate arithmetic"),
                ),
                final_response="Output response",
                error_message="Simulated error message",
            )

            # Build both prompts from the exact same initial_input
            msgs_trace, id_to_full_trace, disp_trace = build_judge_prompt(initial_input, mock_trace)
            msgs_cand, id_to_full_cand, disp_cand = build_judge_prompt(initial_input, execution_trace=None)

            # ID mappings and display texts must be strictly identical
            assert id_to_full_trace == id_to_full_cand
            assert disp_trace == disp_cand
            assert list(disp_trace.keys()) == ["elem_0", "elem_1", "elem_2", "elem_3"]

    def test_requirement_2_candidate_only_prompt_omits_trace_and_summary(self) -> None:
        """2. Candidate-only prompts contain no trace or trace-derived summary."""
        spec = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        initial_input = spec.create_initial_input(seed=42)
        mock_trace = AgentExecutionTrace(
            messages=(
                Message(role="system", content="System instruction"),
                Message(role="user", content="User problem"),
                Message(role="assistant", content="Calling tool calculator"),
            ),
            final_response="Task failed with HTTP 500",
            error_message="HTTP 500 Internal Server Error in calculator",
        )

        msgs_cand, _, _ = build_judge_prompt(initial_input, execution_trace=None)
        combined_cand_prompt = " ".join(m.content for m in msgs_cand)

        # Must not contain trace components
        assert "Observed Agent Execution Trace" not in combined_cand_prompt
        assert "HTTP 500" not in combined_cand_prompt
        assert "Calling tool calculator" not in combined_cand_prompt
        assert "Task failed with HTTP 500" not in combined_cand_prompt
        assert spec.observable_failure_symptom not in combined_cand_prompt

        # Must indicate candidate-only rule instructions
        assert "inherently faulty" in combined_cand_prompt or "candidate specifications" in combined_cand_prompt

    def test_requirement_3_both_judge_modes_resolve_neutral_ids_to_canonical_candidates(self) -> None:
        """3. Both judge modes resolve returned candidate IDs to the correct canonical candidates."""
        spec = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        initial_input = spec.create_initial_input(seed=42)
        mock_trace = AgentExecutionTrace(
            messages=(Message(role="user", content="Test"),),
            final_response="Failed",
        )

        # FakeProvider returning neutral ID elem_0
        handler = make_four_arm_test_handler(
            trace_judge_pred=["elem_0"],
            cand_judge_pred=["elem_0"],
        )
        provider = FakeProvider(model_name="fake-judge-test", custom_handler=handler)
        judge = LLMJudgeBaseline(provider=provider, max_retries=1)

        res_trace = judge.diagnose(initial_input, mock_trace)
        res_cand = judge.diagnose_candidate_only(initial_input)

        # Both must resolve "elem_0" to the canonical candidate element string
        expected_element = initial_input.elements[0]
        assert res_trace.is_valid_diagnosis is True
        assert res_trace.predicted_candidate_set == (expected_element,)
        assert res_cand.is_valid_diagnosis is True
        assert res_cand.predicted_candidate_set == (expected_element,)

        # Also test empty list resolution
        empty_handler = make_four_arm_test_handler(
            trace_judge_pred=[],
            cand_judge_pred=[],
        )
        empty_prov = FakeProvider(model_name="fake-empty-judge", custom_handler=empty_handler)
        empty_judge = LLMJudgeBaseline(provider=empty_prov, max_retries=1)
        res_trace_empty = empty_judge.diagnose(initial_input, mock_trace)
        res_cand_empty = empty_judge.diagnose_candidate_only(initial_input)
        assert res_trace_empty.predicted_candidate_set == ()
        assert res_cand_empty.predicted_candidate_set == ()

    def test_requirement_4_four_diagnostic_arms_represented_separately(self) -> None:
        """4. The four diagnostic arms are represented separately in trace-evaluated results."""
        prompt_task = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        trigger = prompt_task.planted_trigger_elements[0]

        # Handler where trace judge predicts elem_0 and candidate-only predicts empty
        handler = make_four_arm_test_handler(
            prompt_trigger_text=trigger,
            trace_judge_pred=["elem_0"],
            cand_judge_pred=[],
        )
        prov = FakeProvider(model_name="fake-trace-unified", custom_handler=handler)

        res = evaluate_mechanism_three_way(
            spec=prompt_task,
            provider=prov,
            trials_per_candidate=2,
            max_trials=10,
            seed=42,
        )

        assert res.active_dd_outcome.method_name == "active_dd"
        assert res.greedy_ablation_outcome.method_name == "greedy_ablation"
        assert res.llm_judge_outcome.method_name == "llm_judge"
        assert res.candidate_only_judge_outcome is not None
        assert res.candidate_only_judge_outcome.method_name == "candidate_only_judge"

        # Check separate predictions
        assert res.candidate_only_judge_outcome.predicted_candidate_set == ()
        assert res.llm_judge_outcome.predicted_candidate_set != ()

    def test_requirement_5_existing_three_arm_outputs_remain_compatible(self) -> None:
        """5. Existing three-arm outputs remain compatible wherever intended."""
        dd = DiagnosticOutcome(
            mechanism_id="test_m",
            method_name="active_dd",
            predicted_candidate_set=("elem1",),
            is_valid_diagnosis=True,
            false_attribution_indicator=0,
            exact_recovery_indicator=1,
        )
        gr = DiagnosticOutcome(
            mechanism_id="test_m",
            method_name="greedy_ablation",
            predicted_candidate_set=("elem1",),
            is_valid_diagnosis=True,
            false_attribution_indicator=0,
            exact_recovery_indicator=1,
        )
        jd = DiagnosticOutcome(
            mechanism_id="test_m",
            method_name="llm_judge",
            predicted_candidate_set=None,
            is_valid_diagnosis=False,
            false_attribution_indicator=None,
            exact_recovery_indicator=0,
        )

        # MechanismEvaluationResult without candidate_only_judge_outcome defaults to None
        res3 = MechanismEvaluationResult(
            mechanism_id="test_m",
            mechanism_name="Test Mechanism",
            category="prompt",
            initial_elements=("elem1", "elem2"),
            ground_truth_trigger=("elem1",),
            active_dd_outcome=dd,
            greedy_ablation_outcome=gr,
            llm_judge_outcome=jd,
        )
        assert res3.candidate_only_judge_outcome is None

        # Report markdown formatting with 3 arms works without error
        report3 = ExploratoryBenchmarkReport(
            experiment_name="Three_Arm_Test",
            evaluations=(res3,),
            secondary_metrics={"exact_recovery_rate": {"active_dd": 1.0, "greedy_ablation": 1.0, "llm_judge": 0.0}},
        )
        md3 = format_exploratory_benchmark_markdown(report3)
        assert "Active DD ($Y, R$)" in md3
        assert "Candidate-Only Judge" not in md3

    def test_requirement_6_candidate_only_invalid_response_does_not_corrupt_other_arms(self) -> None:
        """6. A candidate-only invalid response does not corrupt other diagnostic arms."""
        spec = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        trigger = spec.planted_trigger_elements[0]

        handler = make_four_arm_test_handler(
            prompt_trigger_text=trigger,
            trace_judge_pred=["elem_0"],
            malformed_cand_only=True,
        )
        prov = FakeProvider(model_name="fake-selective-malformed", custom_handler=handler)

        res = evaluate_mechanism_three_way(
            spec=spec,
            provider=prov,
            trials_per_candidate=2,
            max_trials=10,
            seed=42,
        )

        # Candidate-only judge must be invalid
        cand_out = res.candidate_only_judge_outcome
        assert cand_out is not None
        assert cand_out.is_valid_diagnosis is False
        assert cand_out.predicted_candidate_set is None
        assert cand_out.termination_reason == "malformed_output"
        assert cand_out.false_attribution_indicator is None
        assert cand_out.exact_recovery_indicator == 0

        # Other three arms must NOT be corrupted
        assert res.active_dd_outcome.is_valid_diagnosis is True
        assert res.active_dd_outcome.exact_recovery_indicator == 1
        assert res.greedy_ablation_outcome.is_valid_diagnosis is True
        assert res.greedy_ablation_outcome.exact_recovery_indicator == 1
        assert res.llm_judge_outcome.is_valid_diagnosis is True
        assert res.llm_judge_outcome.predicted_candidate_set is not None

    def test_requirement_7_empty_invalid_inconclusive_scoring_semantics(self) -> None:
        """7. Empty, invalid, and inconclusive predictions retain their proper scoring semantics."""
        # A: Null task + verified empty set -> Y=0, R=1
        score_null_verified = score_diagnostic_outcome(
            mechanism_id="null_task",
            method_name="active_dd",
            candidate_set=(),
            is_valid_diagnosis=True,
            ground_truth_trigger=(),
            is_null_case=True,
            verification_status="verified_empty",
        )
        assert score_null_verified.false_attribution_indicator == 0
        assert score_null_verified.exact_recovery_indicator == 1

        # B: Null task + unverified empty set -> Y=0, R=0
        score_null_unverified = score_diagnostic_outcome(
            mechanism_id="null_task",
            method_name="candidate_only_judge",
            candidate_set=(),
            is_valid_diagnosis=True,
            ground_truth_trigger=(),
            is_null_case=True,
            verification_status="unverified",
        )
        assert score_null_unverified.false_attribution_indicator == 0
        assert score_null_unverified.exact_recovery_indicator == 0

        # C: Null task + spurious candidate blamed -> Y=1, R=0
        score_null_spurious = score_diagnostic_outcome(
            mechanism_id="null_task",
            method_name="candidate_only_judge",
            candidate_set=("innocent_distractor",),
            is_valid_diagnosis=True,
            ground_truth_trigger=(),
            is_null_case=True,
        )
        assert score_null_spurious.false_attribution_indicator == 1
        assert score_null_spurious.exact_recovery_indicator == 0

        # D: Active task + empty prediction -> Y=0, R=0
        score_active_empty = score_diagnostic_outcome(
            mechanism_id="active_task",
            method_name="candidate_only_judge",
            candidate_set=(),
            is_valid_diagnosis=True,
            ground_truth_trigger=("planted_trigger",),
            is_null_case=False,
        )
        assert score_active_empty.false_attribution_indicator == 0
        assert score_active_empty.exact_recovery_indicator == 0

        # E: Invalid diagnosis (malformed output) -> Y=None, R=0
        score_invalid = score_diagnostic_outcome(
            mechanism_id="active_task",
            method_name="candidate_only_judge",
            candidate_set=None,
            is_valid_diagnosis=False,
            ground_truth_trigger=("planted_trigger",),
            is_null_case=False,
            termination_reason="malformed_output",
        )
        assert score_invalid.false_attribution_indicator is None
        assert score_invalid.exact_recovery_indicator == 0

        # F: Infrastructure failure -> Y=None, R=None
        score_infra = score_diagnostic_outcome(
            mechanism_id="active_task",
            method_name="candidate_only_judge",
            candidate_set=None,
            is_valid_diagnosis=False,
            ground_truth_trigger=("planted_trigger",),
            is_null_case=False,
            termination_reason="infra_failure",
        )
        assert score_infra.false_attribution_indicator is None
        assert score_infra.exact_recovery_indicator is None

    def test_requirement_8_metric_denominators_and_missing_counts(self) -> None:
        """8. Metric denominators and missing-outcome counts are correct."""
        tasks = list_trace_evaluated_tasks()
        prompt_task = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        trigger = prompt_task.planted_trigger_elements[0]

        handler = make_four_arm_test_handler(
            prompt_trigger_text=trigger,
            trace_judge_pred=[],
            cand_judge_pred=[],
        )
        prov = FakeProvider(model_name="fake-metrics-test", custom_handler=handler)

        report = run_multi_mechanism_benchmark(
            provider=prov,
            tier="trace_evaluated",
            trials_per_candidate=2,
            max_trials=10,
            seed=42,
        )

        assert report.secondary_metrics is not None
        arm_sums = report.secondary_metrics["arm_summaries"]

        # Check Active DD summary
        dd_sum = arm_sums["active_dd"]
        assert dd_sum["total_observations"] == 5
        assert dd_sum["infrastructure_failures"] == 0
        assert dd_sum["evaluable_observations"] == 5
        assert dd_sum["null_tasks_total"] == 2
        assert dd_sum["active_tasks_total"] == 3
        assert dd_sum["active_exact_recovery_rate"] == 1.0
        assert dd_sum["null_false_attribution_rate"] == 0.0

        # Check Candidate-Only Judge summary
        cand_sum = arm_sums["candidate_only_judge"]
        assert cand_sum["total_observations"] == 5
        assert cand_sum["evaluable_observations"] == 5
        assert cand_sum["empty_predictions_total"] == 5
        assert cand_sum["null_false_attribution_rate"] == 0.0
        # Active recovery rate is 0.0 because it predicted empty on active tasks
        assert cand_sum["active_exact_recovery_rate"] == 0.0

        # Check Paired Comparisons
        paired = report.secondary_metrics["paired_comparisons"]
        assert "active_dd_vs_llm_judge" in paired
        assert "active_dd_vs_candidate_only_judge" in paired
        assert "llm_judge_vs_candidate_only_judge" in paired

        comp = paired["active_dd_vs_candidate_only_judge"]
        assert comp["n_complete_pairs"] == 5
        assert comp["n_infra_excluded"] == 0
        assert comp["discordant_b"] + comp["discordant_c"] == comp["n_discordant"]

    def test_requirement_9_registry_task_counts_unchanged(self) -> None:
        """9. The five-task trace registry and 18-mechanism synthetic registry remain unchanged in count."""
        assert len(MECHANISM_REGISTRY) == 18
        assert len(TRACE_EVALUATED_REGISTRY) == 5
        assert len(list_mechanism_tasks()) == 18
        assert len(list_trace_evaluated_tasks()) == 5

    def test_requirement_10_mcnemar_statistical_test_behavior_unchanged(self) -> None:
        """10. Existing statistical test behavior remains unchanged in this phase."""
        # Greater alternative (directional, H1: b > c)
        res_dir = mcnemar_exact_test(b=7, c=0, alternative="greater", alpha=0.05)
        assert res_dir.alternative == "greater"
        assert res_dir.n_discordant == 7
        assert res_dir.p_value == pytest.approx((0.5) ** 7, rel=1e-5)
        assert res_dir.statistically_significant is True

        # Inverted direction
        res_inv = mcnemar_exact_test(b=0, c=7, alternative="greater", alpha=0.05)
        assert res_inv.statistically_significant is False

        # Two-sided alternative remains available as specified in statistics.py
        res_two = mcnemar_exact_test(b=7, c=0, alternative="two-sided", alpha=0.05)
        assert res_two.alternative == "two-sided"
        assert res_two.p_value == pytest.approx(2.0 * (0.5) ** 7, rel=1e-5)

    def test_requirement_11_full_report_markdown_includes_all_arms(self) -> None:
        """11. Generated markdown report includes candidate-only judge column and arm accounting."""
        prompt_task = get_trace_evaluated_task("trace_prompt_01_tool_prohibition")
        trigger = prompt_task.planted_trigger_elements[0]

        handler = make_four_arm_test_handler(
            prompt_trigger_text=trigger,
            trace_judge_pred=[],
            cand_judge_pred=[],
        )
        prov = FakeProvider(model_name="fake-md-test", custom_handler=handler)

        report = run_multi_mechanism_benchmark(
            provider=prov,
            tier="trace_evaluated",
            trials_per_candidate=2,
            max_trials=10,
            seed=42,
        )

        md = format_exploratory_benchmark_markdown(report)
        assert "Candidate-Only Judge ($Y, R$)" in md
        assert "Arm-Specific Diagnostic & Error Accounting" in md
        assert "Paired Method Comparisons" in md
        assert "active_dd_vs_candidate_only_judge" in md
        assert "llm_judge_vs_candidate_only_judge" in md
