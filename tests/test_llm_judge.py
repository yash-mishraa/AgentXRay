"""Comprehensive offline tests for LLM Judge baseline, multi-mechanism benchmark, and statistical reporting."""

from __future__ import annotations

import json
from typing import Sequence
import pytest

from agentxray.agent_harness.agent import AgentExecutionTrace
from agentxray.agent_harness.cost_tracker import CostTracker, ExperimentBudgetConfig
from agentxray.llm_judge import LLMJudgeBaseline, run_llm_judge_baseline
from agentxray.models import (
    DiagnosticOutcome,
    ExperimentConfig,
    InputCase,
    LLMJudgeComparison,
    LLMJudgeResult,
    TargetOutcome,
)
from agentxray.providers.base import Message, ModelResponse, ProviderError, ToolDefinition, UsageMetadata
from agentxray.providers.fake import FakeProvider
from agentxray.reporting import format_exploratory_benchmark_markdown
from agentxray.statistics import (
    compute_contrast_sensitivity_bounds,
    compute_discovery_level_fdr,
    compute_mean_mechanism_fdp,
    mcnemar_exact_test,
)
from benchmarks.multi_mechanism.runner import (
    evaluate_mechanism_three_way,
    run_multi_mechanism_benchmark,
    run_with_infra_guard,
    score_diagnostic_outcome,
)
from benchmarks.multi_mechanism.tasks import (
    MECHANISM_REGISTRY,
    MechanismTaskSpec,
    get_mechanism_task,
    list_mechanism_tasks,
)


def _make_trace(error: str = "Execution failed", finish_reason: str = "error") -> AgentExecutionTrace:
    return AgentExecutionTrace(
        messages=(
            Message(role="user", content="Execute financial audit task"),
            Message(role="assistant", content="Initiating tool call..."),
        ),
        final_response="",
        finish_reason=finish_reason,
        error_message=error,
        turns_used=2,
    )


def _make_response(
    content: str,
    role: str = "assistant",
    finish_reason: str = "stop",
    usage: UsageMetadata | None = None,
    model_identifier: str = "fake-judge-model",
) -> ModelResponse:
    return ModelResponse(
        content=content,
        role=role,
        finish_reason=finish_reason,
        usage=usage or UsageMetadata(),
        model_identifier=model_identifier,
    )


class TestStructuredPredictionAndParsing:
    """Verifies JSON parsing, markdown code-fence extraction, and candidate normalization."""

    def test_valid_raw_json_response(self) -> None:
        def handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            payload = {
                "predicted_candidate_set": ["param_mode_debug"],
                "confidence": 0.85,
                "rationale": "The debug mode caused excessive output leading to parsing failure.",
            }
            return _make_response(
                content=json.dumps(payload),
                usage=UsageMetadata(prompt_tokens=100, completion_tokens=40, total_tokens=140),
            )

        provider = FakeProvider(custom_handler=handler)
        judge = LLMJudgeBaseline(provider=provider)
        inp = InputCase(case_id="c1", elements=("param_mode_debug", "irrelevant_param"))
        trace = _make_trace()

        res = judge.diagnose(inp, trace)
        assert res.is_valid_diagnosis is True
        assert res.predicted_candidate_set == ("param_mode_debug",)
        assert res.attempts_used == 1
        assert "excessive output" in res.rationale
        assert res.termination_reason == "normal_completion"

    def test_valid_markdown_code_fence_extraction(self) -> None:
        def handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            payload = {
                "predicted_candidate_set": ["strict_schema_v2"],
                "confidence": 0.92,
                "rationale": "Schema validation rejected optional field.",
            }
            wrapped = f"Here is my diagnostic evaluation:\n```json\n{json.dumps(payload, indent=2)}\n```\nHope this helps!"
            return _make_response(
                content=wrapped,
                usage=UsageMetadata(prompt_tokens=120, completion_tokens=50, total_tokens=170),
            )

        provider = FakeProvider(custom_handler=handler)
        judge = LLMJudgeBaseline(provider=provider)
        inp = InputCase(case_id="c2", elements=("strict_schema_v2", "distractor_timeout"))
        trace = _make_trace()

        res = judge.diagnose(inp, trace)
        assert res.is_valid_diagnosis is True
        assert res.predicted_candidate_set == ("strict_schema_v2",)
        assert res.attempts_used == 1

    def test_valid_empty_prediction(self) -> None:
        def handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            payload = {
                "predicted_candidate_set": [],
                "confidence": 0.99,
                "rationale": "No candidate input elements caused this failure; error is external.",
            }
            return _make_response(
                content=json.dumps(payload),
                usage=UsageMetadata(prompt_tokens=80, completion_tokens=30, total_tokens=110),
            )

        provider = FakeProvider(custom_handler=handler)
        judge = LLMJudgeBaseline(provider=provider)
        inp = InputCase(case_id="c3", elements=("elem1", "elem2"))
        trace = _make_trace("External 500 error")

        res = judge.diagnose(inp, trace)
        assert res.is_valid_diagnosis is True
        assert res.predicted_candidate_set == ()

    def test_candidate_normalization_and_whitespace(self) -> None:
        def handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            payload = {
                "predicted_candidate_set": ["  PARAM_MODE_DEBUG  "],
                "confidence": 0.70,
                "rationale": "Casing and whitespace mismatch.",
            }
            return _make_response(
                content=json.dumps(payload),
                usage=UsageMetadata(prompt_tokens=90, completion_tokens=30, total_tokens=120),
            )

        provider = FakeProvider(custom_handler=handler)
        judge = LLMJudgeBaseline(provider=provider)
        inp = InputCase(case_id="c4", elements=("param_mode_debug", "other"))
        trace = _make_trace()

        res = judge.diagnose(inp, trace)
        assert res.is_valid_diagnosis is True
        assert res.predicted_candidate_set == ("param_mode_debug",)


class TestRetryAndFormatEnforcement:
    """Verifies bounded retry loop, format error prompt feedback, and retry exhaustion."""

    def test_malformed_json_retry_then_success(self) -> None:
        attempts = 0
        received_messages: list[Sequence[Message]] = []

        def handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            nonlocal attempts
            attempts += 1
            received_messages.append(messages)
            if attempts == 1:
                # Malformed output
                return _make_response(
                    content="I believe param_mode_debug is the cause of the failure.",
                    usage=UsageMetadata(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                )
            # Correct JSON on retry
            payload = {
                "predicted_candidate_set": ["param_mode_debug"],
                "confidence": 0.9,
                "rationale": "Recovered valid format.",
            }
            return _make_response(
                content=json.dumps(payload),
                usage=UsageMetadata(prompt_tokens=140, completion_tokens=30, total_tokens=170),
            )

        provider = FakeProvider(custom_handler=handler)
        judge = LLMJudgeBaseline(provider=provider, max_retries=2)
        inp = InputCase(case_id="c_retry", elements=("param_mode_debug", "distractor"))
        trace = _make_trace()

        res = judge.diagnose(inp, trace)
        assert res.is_valid_diagnosis is True
        assert res.predicted_candidate_set == ("param_mode_debug",)
        assert res.attempts_used == 2
        assert len(received_messages) == 2
        # Verify retry prompt format
        retry_msg = received_messages[1][-1].content
        assert "Could not parse response as valid JSON" in retry_msg
        assert "culprit_element_ids" in retry_msg

    def test_malformed_json_exhaustion_after_three_attempts(self) -> None:
        call_count = 0

        def handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            return _make_response(
                content=f"Attempt {call_count}: Still unparseable text.",
                usage=UsageMetadata(prompt_tokens=100, completion_tokens=15, total_tokens=115),
            )

        provider = FakeProvider(custom_handler=handler)
        judge = LLMJudgeBaseline(provider=provider, max_retries=2)  # 1 initial + 2 retries = 3 total
        inp = InputCase(case_id="c_fail", elements=("param_mode_debug",))
        trace = _make_trace()

        res = judge.diagnose(inp, trace)
        assert res.is_valid_diagnosis is False
        assert res.predicted_candidate_set is None
        assert res.attempts_used == 3
        assert res.termination_reason == "malformed_output"
        assert len(res.raw_responses) == 3

    def test_unrecognized_candidate_retry_then_success(self) -> None:
        attempts = 0
        retry_prompt_text = ""

        def handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            nonlocal attempts, retry_prompt_text
            attempts += 1
            if attempts == 1:
                return _make_response(
                    content=json.dumps({"predicted_candidate_set": ["hallucinated_element"], "confidence": 0.8}),
                    usage=UsageMetadata(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                )
            retry_prompt_text = messages[-1].content
            return _make_response(
                content=json.dumps({"predicted_candidate_set": ["valid_element"], "confidence": 0.9}),
                usage=UsageMetadata(prompt_tokens=130, completion_tokens=20, total_tokens=150),
            )

        provider = FakeProvider(custom_handler=handler)
        judge = LLMJudgeBaseline(provider=provider, max_retries=2)
        inp = InputCase(case_id="c_unrec", elements=("valid_element", "other_element"))
        trace = _make_trace()

        res = judge.diagnose(inp, trace)
        assert res.is_valid_diagnosis is True
        assert res.predicted_candidate_set == ("valid_element",)
        assert res.attempts_used == 2
        assert "not found in the valid candidate list: hallucinated_element" in retry_prompt_text

    def test_unrecognized_candidate_exhaustion(self) -> None:
        def handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            return _make_response(
                content=json.dumps({"predicted_candidate_set": ["phantom_element"]}),
                usage=UsageMetadata(prompt_tokens=100, completion_tokens=20, total_tokens=120),
            )

        provider = FakeProvider(custom_handler=handler)
        judge = LLMJudgeBaseline(provider=provider, max_retries=2)
        inp = InputCase(case_id="c_unrec_exhaust", elements=("real_element",))
        trace = _make_trace()

        res = judge.diagnose(inp, trace)
        assert res.is_valid_diagnosis is False
        assert res.predicted_candidate_set is None
        assert res.attempts_used == 3
        assert res.termination_reason == "unrecognized_candidate"


class TestBudgetEnforcement:
    """Verifies strict dollar and token budget enforcement for the LLM Judge."""

    def test_pre_exhausted_budget_terminates_immediately(self) -> None:
        config = ExperimentBudgetConfig(max_spend_usd=0.01)
        tracker = CostTracker(config=config)
        # Pre-exhaust the spend budget
        try:
            tracker.record_usage(UsageMetadata(prompt_tokens=150000, completion_tokens=50000, total_tokens=200000))
        except Exception:
            pass

        call_made = False

        def handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            nonlocal call_made
            call_made = True
            return _make_response(content="{}")

        provider = FakeProvider(custom_handler=handler)
        judge = LLMJudgeBaseline(provider=provider, cost_tracker=tracker)
        inp = InputCase(case_id="c_budget", elements=("elem1",))
        trace = _make_trace()

        res = judge.diagnose(inp, trace)
        assert res.is_valid_diagnosis is False
        assert res.termination_reason == "budget_exhausted"
        assert res.attempts_used == 0
        assert call_made is False

    def test_budget_exhaustion_during_retry(self) -> None:
        config = ExperimentBudgetConfig(max_spend_usd=0.00005, max_model_requests=5)
        tracker = CostTracker(config=config)

        def handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            # Emit malformed response consuming enough tokens to breach 0.00005 USD
            return _make_response(
                content="Not JSON",
                usage=UsageMetadata(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
            )

        provider = FakeProvider(custom_handler=handler)
        judge = LLMJudgeBaseline(provider=provider, cost_tracker=tracker, max_retries=2)
        inp = InputCase(case_id="c_budget_retry", elements=("elem1",))
        trace = _make_trace()

        res = judge.diagnose(inp, trace)
        assert res.is_valid_diagnosis is False
        assert res.termination_reason == "budget_exhausted"
        assert res.attempts_used == 1  # Halted after attempt 1 because budget exceeded


class TestConjunctiveTriggerBehavior:
    """Verifies conjunctive trigger mechanics on mech_conj_01_schema_routing_deadlock."""

    def test_conjunctive_target_failure_logic(self) -> None:
        spec = get_mechanism_task("mech_conj_01_schema_routing_deadlock")
        assert spec.is_conjunctive is True
        t1, t2 = spec.planted_trigger_elements
        distractors = [c for c in spec.candidate_elements if c not in spec.planted_trigger_elements]
        d1 = distractors[0]

        target = spec.create_target()

        # Both elements present -> failure
        res_both = target.execute(InputCase(case_id="t1", elements=(t1, t2)))
        assert res_both.outcome == TargetOutcome.FAILURE

        # Both elements + distractor -> failure
        res_all = target.execute(InputCase(case_id="t2", elements=(t1, t2, d1)))
        assert res_all.outcome == TargetOutcome.FAILURE

        # Only one element present -> success
        res_t1_only = target.execute(InputCase(case_id="t3", elements=(t1, d1)))
        assert res_t1_only.outcome == TargetOutcome.SUCCESS

        res_t2_only = target.execute(InputCase(case_id="t4", elements=(t2, d1)))
        assert res_t2_only.outcome == TargetOutcome.SUCCESS

        # Neither trigger present -> success
        res_none = target.execute(InputCase(case_id="t5", elements=(d1,)))
        assert res_none.outcome == TargetOutcome.SUCCESS

    def test_conjunctive_diagnostic_scoring_rules(self) -> None:
        spec = get_mechanism_task("mech_conj_01_schema_routing_deadlock")
        t1, t2 = spec.planted_trigger_elements
        distractors = [c for c in spec.candidate_elements if c not in spec.planted_trigger_elements]
        d1 = distractors[0]
        trigger = (t1, t2)

        # 1. Partial subset: t1 only (subset of true trigger)
        # S_pred is strict subset of S_true -> Recovery failure (0), but NO spurious attribution (0)
        out_partial = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="test_method",
            candidate_set=(t1,),
            is_valid_diagnosis=True,
            ground_truth_trigger=trigger,
            is_null_case=False,
        )
        assert out_partial.exact_recovery_indicator == 0
        assert out_partial.false_attribution_indicator == 0

        # 2. Oversized subset: both triggers + distractor
        # S_pred contains non-causal element -> Spurious attribution (1), Recovery failure (0)
        out_oversized = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="test_method",
            candidate_set=(t1, t2, d1),
            is_valid_diagnosis=True,
            ground_truth_trigger=trigger,
            is_null_case=False,
        )
        assert out_oversized.exact_recovery_indicator == 0
        assert out_oversized.false_attribution_indicator == 1

        # 3. Exact recovery: both triggers exactly
        out_exact = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="test_method",
            candidate_set=(t1, t2),
            is_valid_diagnosis=True,
            ground_truth_trigger=trigger,
            is_null_case=False,
        )
        assert out_exact.exact_recovery_indicator == 1
        assert out_exact.false_attribution_indicator == 0

        # 4. Spurious only: distractor alone
        out_spurious = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="test_method",
            candidate_set=(d1,),
            is_valid_diagnosis=True,
            ground_truth_trigger=trigger,
            is_null_case=False,
        )
        assert out_spurious.exact_recovery_indicator == 0
        assert out_spurious.false_attribution_indicator == 1


class TestNullControlMechanics:
    """Verifies separation between Condition A (clean/bloat pass) and Condition B (external shock 500)."""

    def test_condition_a_clean_pass_null_control(self) -> None:
        spec = get_mechanism_task("mech_null_clean")
        assert spec.is_null_case is True
        target = spec.create_target()
        res = target.execute(spec.create_initial_input())
        assert res.outcome == TargetOutcome.SUCCESS

        # Valid empty prediction: Y=0, R_exact=1
        out_empty = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="llm_judge",
            candidate_set=(),
            is_valid_diagnosis=True,
            ground_truth_trigger=(),
            is_null_case=True,
        )
        assert out_empty.false_attribution_indicator == 0
        assert out_empty.exact_recovery_indicator == 1

        # Non-empty prediction: Y=1, R_exact=0
        out_spurious = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="llm_judge",
            candidate_set=("candidate_calc_1",),
            is_valid_diagnosis=True,
            ground_truth_trigger=(),
            is_null_case=True,
        )
        assert out_spurious.false_attribution_indicator == 1
        assert out_spurious.exact_recovery_indicator == 0

    def test_condition_b_external_shock_server_error_null_control(self) -> None:
        spec = get_mechanism_task("mech_null_ext")
        assert spec.is_null_case is True
        target = spec.create_target()
        res = target.execute(spec.create_initial_input())
        # Target fails unconditionally due to mock 500
        assert res.outcome == TargetOutcome.FAILURE
        assert "HTTP 500" in res.output_message

        # Valid empty prediction on external shock: correctly attributes 0 candidate elements
        out_empty = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="llm_judge",
            candidate_set=(),
            is_valid_diagnosis=True,
            ground_truth_trigger=(),
            is_null_case=True,
        )
        assert out_empty.false_attribution_indicator == 0
        assert out_empty.exact_recovery_indicator == 1

        # Post-hoc attribution of agent prompt/parameters to external error: Y=1, R_exact=0
        out_hallucinated = score_diagnostic_outcome(
            mechanism_id=spec.mechanism_id,
            method_name="llm_judge",
            candidate_set=("distractor_client_tag: client_internal",),
            is_valid_diagnosis=True,
            ground_truth_trigger=(),
            is_null_case=True,
        )
        assert out_hallucinated.false_attribution_indicator == 1
        assert out_hallucinated.exact_recovery_indicator == 0


class TestStatisticalInferenceAndSensitivity:
    """Verifies directional McNemar test, Manski contrast sensitivity bounds, and FDR/FDP metrics."""

    def test_directional_mcnemar_exact_test(self) -> None:
        # Case 1: Significant difference favoring ActiveDD (Judge worse: b=7, c=0)
        res_sig = mcnemar_exact_test(b=7, c=0, alternative="greater", alpha=0.05)
        assert res_sig.n_discordant == 7
        assert res_sig.p_value == pytest.approx(0.5**7, rel=1e-4)
        assert res_sig.p_value < 0.05
        assert res_sig.statistically_significant is True
        assert res_sig.odds_ratio == float("inf")

        # Case 2: Inverted difference favoring Judge (b=0, c=7)
        res_inv = mcnemar_exact_test(b=0, c=7, alternative="greater", alpha=0.05)
        assert res_inv.p_value == 1.0
        assert res_inv.statistically_significant is False
        assert res_inv.odds_ratio == 0.0

        # Case 3: Zero discordant pairs (b=0, c=0)
        res_zero = mcnemar_exact_test(b=0, c=0, alternative="greater")
        assert res_zero.p_value == 1.0
        assert res_zero.odds_ratio is None
        assert res_zero.statistically_significant is False

        # Case 4: Balanced discordant pairs (b=4, c=4)
        res_bal = mcnemar_exact_test(b=4, c=4, alternative="greater")
        assert res_bal.odds_ratio == 1.0
        assert res_bal.statistically_significant is False

    def test_manski_contrast_sensitivity_bounds(self) -> None:
        # 10 mechanisms in retained cohort
        # 6 complete pairs: (Judge=1, ActiveDD=0) x 4, (Judge=0, ActiveDD=0) x 2
        # 2 partial: Judge=None, ActiveDD=0
        # 1 partial: Judge=1, ActiveDD=None
        # 1 mutual missing: Judge=None, ActiveDD=None
        y_judge = [1, 1, 1, 1, 0, 0, None, None, 1, None]
        y_dd =    [0, 0, 0, 0, 0, 0, 0,    0,    None, None]

        bounds = compute_contrast_sensitivity_bounds(y_judge=y_judge, y_activedd=y_dd)
        assert bounds.n_evaluable == 10
        assert bounds.n_complete == 6
        assert bounds.n_miss_judge == 3
        assert bounds.n_miss_activedd == 2

        # Complete-case: 4/6 - 0/6 = 0.6667
        assert bounds.complete_case_risk_difference == pytest.approx(4 / 6, rel=1e-3)

        # Upper bound: impute missing Judge as 1, missing ActiveDD as 0
        # Observed Judge sum: 1+1+1+1+0+0+1 = 5. Imputed 3 missing as 1 -> Total Judge = 8
        # Observed ActiveDD sum: 0. Imputed 2 missing as 0 -> Total ActiveDD = 0
        # Upper = (8 - 0) / 10 = 0.80
        assert bounds.upper_bound == pytest.approx(0.80, rel=1e-3)

        # Lower bound: impute missing Judge as 0, missing ActiveDD as 1
        # Observed Judge sum: 5. Imputed 3 missing as 0 -> Total Judge = 5
        # Observed ActiveDD sum: 0. Imputed 2 missing as 1 -> Total ActiveDD = 2
        # Lower = (5 - 2) / 10 = 0.30
        assert bounds.lower_bound == pytest.approx(0.30, rel=1e-3)

        assert bounds.lower_bound <= bounds.upper_bound

    def test_discovery_level_fdr_and_fdp_metrics(self) -> None:
        # Predictions across 3 mechanisms:
        # Mech 1: GT = ("a", "b"), Pred = ("a", "b", "c") -> 1 spurious / 3 discoveries, FDP = 1/3
        # Mech 2: GT = ("x",), Pred = ("x",) -> 0 spurious / 1 discovery, FDP = 0
        # Mech 3: GT = ("m",), Pred = () -> 0 spurious / 0 discoveries, FDP = 0
        preds = [("a", "b", "c"), ("x",), ()]
        gts = [("a", "b"), ("x",), ("m",)]

        fdr = compute_discovery_level_fdr(preds, gts)
        # Total spurious = 1 (c), Total discoveries = 3 + 1 + 0 = 4. FDR = 1/4 = 0.25
        assert fdr == 0.25

        fdp = compute_mean_mechanism_fdp(preds, gts)
        # FDP_1 = 1/3 = 0.3333, FDP_2 = 0.0, FDP_3 = 0.0. Mean = (1/3 + 0 + 0) / 3 = 0.1111
        assert fdp == pytest.approx(1 / 9, rel=1e-3)


class TestInfrastructureRerunGuard:
    """Verifies single identical-seed rerun upon infrastructure failure and error logging."""

    def test_single_rerun_recovers_successfully(self) -> None:
        calls = 0
        deviation_log: list[dict] = []

        def flaky_fn() -> DiagnosticOutcome:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ProviderError("503 Service Unavailable")
            return DiagnosticOutcome(
                mechanism_id="mech_test",
                method_name="test_method",
                predicted_candidate_set=("elem1",),
                is_valid_diagnosis=True,
                false_attribution_indicator=0,
                exact_recovery_indicator=1,
            )

        res = run_with_infra_guard(flaky_fn, mechanism_id="mech_test", method_name="test_method", deviation_log=deviation_log)
        assert calls == 2
        assert res.is_valid_diagnosis is True
        assert len(deviation_log) == 1
        assert deviation_log[0]["event"] == "INFRA_FAILURE_FIRST_ATTEMPT"
        assert "503" in deviation_log[0]["error"]

    def test_unresolved_failure_marks_outcome_invalid(self) -> None:
        calls = 0
        deviation_log: list[dict] = []

        def persistent_error_fn() -> DiagnosticOutcome:
            nonlocal calls
            calls += 1
            raise ProviderError("500 Internal Server Error")

        res = run_with_infra_guard(persistent_error_fn, mechanism_id="mech_test_err", method_name="test_method", deviation_log=deviation_log)
        assert calls == 2
        assert res.is_valid_diagnosis is False
        assert res.termination_reason == "infra_failure"
        assert res.false_attribution_indicator is None
        assert res.exact_recovery_indicator is None
        assert len(deviation_log) == 2
        assert deviation_log[1]["event"] == "INFRA_FAILURE_UNRESOLVED"


class TestBenchmarkExecutionAndReporting:
    """Verifies end-to-end multi-mechanism benchmark execution and markdown report generation."""

    def test_benchmark_registry_contains_18_prespecified_mechanisms(self) -> None:
        tasks = list_mechanism_tasks()
        assert len(tasks) == 18
        categories = {t.category for t in tasks}
        assert "prompt" in categories
        assert "conjunctive" in categories
        assert "schema" in categories
        assert "environment" in categories
        assert "null_clean" in categories
        assert "null_bloat" in categories
        assert "null_ext" in categories

    def test_run_multi_mechanism_benchmark_offline(self) -> None:
        # Run across 3 representative mechanisms (1 prompt, 1 conjunctive, 1 null)
        m_p01 = get_mechanism_task("mech_p01_directive_prohibition")
        m_conj = get_mechanism_task("mech_conj_01_schema_routing_deadlock")
        m_null = get_mechanism_task("mech_null_clean")
        test_tasks = [m_p01, m_conj, m_null]

        def smart_judge_handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            user_msg = messages[-1].content
            # Passive judge heuristic: predicts candidate matching mechanism
            if "Clean" in user_msg or "clean" in user_msg:
                pred = []
            elif "Deadlock" in user_msg or "deadlock" in user_msg:
                pred = [m_conj.planted_trigger_elements[0]]
            else:
                pred = [m_p01.planted_trigger_elements[0]]
            payload = {
                "predicted_candidate_set": pred,
                "confidence": 0.8,
                "rationale": "Mock diagnostic analysis.",
            }
            return _make_response(
                content=json.dumps(payload),
                usage=UsageMetadata(prompt_tokens=100, completion_tokens=30, total_tokens=130),
            )

        provider = FakeProvider(custom_handler=smart_judge_handler)
        report = run_multi_mechanism_benchmark(
            provider=provider,
            mechanisms=test_tasks,
            trials_per_candidate=2,
            max_trials=10,
        )

        assert report.n_total == 3
        assert report.n_infra == 0
        assert report.n_evaluable == 3
        assert report.mcnemar_n_discordant >= 0
        assert len(report.evaluations) == 3

        # Generate markdown report
        md = format_exploratory_benchmark_markdown(report)
        assert f"# {report.experiment_name} — Research Summary" in md
        assert "Analysis Cohort and Attrition Accounting" in md
        assert "Contrast-Specific Non-Parametric Sensitivity Bounds" in md
        assert "Per-Mechanism Three-Way Diagnostic Results" in md
        assert "Scientific Limitations" in md
        assert "mech_p01_directive_prohibition" in md
        assert "mech_conj_01_schema_routing_deadlock" in md
        assert "mech_null_clean" in md

    def test_full_18_mechanism_benchmark_execution(self) -> None:
        """Verify that all 18 mechanisms execute hermetically through the full pipeline."""
        def uniform_judge_handler(messages: Sequence[Message], tools: Sequence[ToolDefinition]) -> ModelResponse:
            user_msg = messages[-1].content
            # If null case, emit empty list; else emit empty list or valid format
            payload = {
                "culprit_element_ids": [],
                "rationale": "Offline hermetic evaluation test.",
            }
            return _make_response(content=json.dumps(payload))

        provider = FakeProvider(custom_handler=uniform_judge_handler)
        report = run_multi_mechanism_benchmark(
            provider=provider,
            trials_per_candidate=2,
            max_trials=10,
        )

        assert report.n_total == 18
        assert report.n_infra == 0
        assert report.n_evaluable == 18
        assert len(report.evaluations) == 18

        md = format_exploratory_benchmark_markdown(report)
        assert len(md) > 1000
        # Check all 18 mechanism IDs appear in report
        for mech in list_mechanism_tasks():
            assert mech.mechanism_id in md
