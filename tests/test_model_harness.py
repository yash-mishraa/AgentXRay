"""Tests for the controlled model-backed agent harness, mock tools, cost tracking, and pilot runner."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import pytest

from agentxray.agent_harness.agent import ControlledAgent
from agentxray.agent_harness.cost_tracker import (
    CostBudgetExceededError,
    CostTracker,
    ExperimentBudgetConfig,
)
from agentxray.agent_harness.evaluator import ToolCallingTaskEvaluator
from agentxray.agent_harness.target import ModelBackedAgentTarget
from agentxray.agent_harness.tools import (
    CALCULATOR_TOOL_DEF,
    DATABASE_LOOKUP_TOOL_DEF,
    MockToolEnvironment,
    mock_database_lookup,
    mock_unit_converter,
    safe_calculator,
)
from agentxray.models import (
    ExecutionTier,
    ExperimentConfig,
    InputCase,
    SearchStatus,
    StochasticMinimalityStatus,
    TargetOutcome,
)
from agentxray.providers.fake import FakeProvider
from agentxray.runner import InterventionRunner
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.model_pilot.runner import (
    dry_run_model_pilot,
    run_model_pilot_benchmark,
)
from benchmarks.model_pilot.tasks import (
    create_model_pilot_baseline_task,
    create_model_pilot_conflicting_prompt_task,
    create_model_pilot_distractor_task,
)


def test_safe_calculator_ast_isolation():
    """Verify safe calculator evaluates standard arithmetic and rejects arbitrary code or shell injection."""
    # Valid arithmetic expressions
    assert safe_calculator("150 + 250") == "400"
    assert safe_calculator("42 * 10 - 20") == "400"
    assert safe_calculator("100 / 4") == "25"
    assert safe_calculator("(10 + 20) * 3") == "90"
    assert safe_calculator("-50 + 100") == "50"

    # Division by zero
    assert "Division by zero" in safe_calculator("100 / 0")

    # Malicious injection attempts must be safely rejected
    assert "Error:" in safe_calculator("__import__('os').system('ls')")
    assert "Error:" in safe_calculator("open('/etc/passwd').read()")
    assert "Error:" in safe_calculator("lambda x: x")
    assert "Error:" in safe_calculator("")


def test_mock_database_lookup():
    """Verify sandboxed database lookup retrieves in-memory records and handles missing entities."""
    assert mock_database_lookup("user_42", "balance") == "450.0"
    assert mock_database_lookup("user_42", "status") == "active"
    assert "Error: Entity 'unknown_user' not found" in mock_database_lookup("unknown_user", "balance")
    assert "Error: Field 'non_existent_field' not found" in mock_database_lookup("user_42", "non_existent_field")


def test_mock_unit_converter():
    """Verify in-memory unit converter arithmetic."""
    # Length
    assert mock_unit_converter(1.0, "km", "m") == "1000.0"
    assert mock_unit_converter(1000.0, "m", "km") == "1.0"
    # Mass
    assert mock_unit_converter(1.0, "kg", "g") == "1000.0"
    # Incompatible
    assert "Error:" in mock_unit_converter(10.0, "km", "kg")


def test_mock_tool_environment_dispatch():
    """Verify MockToolEnvironment dispatches calls safely without external side effects."""
    env = MockToolEnvironment()
    defs = env.get_definitions()
    assert len(defs) == 3

    # Successful call
    out = env.execute("calculator", {"expression": "200 + 300"})
    assert out == "500"

    # Unknown tool
    err_out = env.execute("non_existent_tool", {})
    assert "not recognized" in err_out

    # Invalid arguments
    type_err = env.execute("calculator", {"wrong_arg": "123"})
    assert "Invalid arguments" in type_err


def test_cost_tracker_safeguards():
    """Verify CostTracker enforces request limits and spend budgets strictly."""
    cfg = ExperimentBudgetConfig(
        max_model_requests=3,
        max_spend_usd=0.001,
        model_name="gpt-4o-mini",
    )
    tracker = CostTracker(cfg)

    from agentxray.providers.base import UsageMetadata

    # Request 1: OK
    tracker.record_usage(UsageMetadata(prompt_tokens=100, completion_tokens=50, total_tokens=150))
    assert tracker.total_requests == 1

    # Request 2: OK
    tracker.record_usage(UsageMetadata(prompt_tokens=100, completion_tokens=50, total_tokens=150))
    assert tracker.total_requests == 2

    # Request 3: Hits request cap -> raises CostBudgetExceededError
    with pytest.raises(CostBudgetExceededError, match="Model request budget exhausted"):
        tracker.record_usage(UsageMetadata(prompt_tokens=100, completion_tokens=50, total_tokens=150))


def test_cost_tracker_spend_limit():
    """Verify CostTracker enforces spend budget when request count is not exceeded."""
    cfg = ExperimentBudgetConfig(
        max_model_requests=100,
        max_spend_usd=0.0001,  # very small spend limit
        model_name="gpt-4o",  # more expensive model ($2.50/$10 per 1M)
    )
    tracker = CostTracker(cfg)
    from agentxray.providers.base import UsageMetadata

    # Large request breaches spend limit: 10,000 prompt tokens * $2.50/1M = $0.025 > $0.0001
    with pytest.raises(CostBudgetExceededError, match="Estimated spend budget exhausted"):
        tracker.record_usage(UsageMetadata(prompt_tokens=10_000, completion_tokens=5_000, total_tokens=15_000))


def test_evaluator_tool_and_answer():
    """Verify ToolCallingTaskEvaluator applies observable criteria without looking at ground truth."""
    evaluator = ToolCallingTaskEvaluator(
        required_tool="calculator",
        expected_substring="400",
        forbidden_substrings=("cannot use external tools",),
    )

    from agentxray.agent_harness.agent import AgentExecutionTrace, ToolExecutionRecord

    # 1. Success trace
    trace_ok = AgentExecutionTrace(
        final_response="The calculated sum is $400.",
        tool_executions=(
            ToolExecutionRecord(
                tool_call_id="call_1",
                tool_name="calculator",
                arguments={"expression": "150 + 250"},
                output="400",
            ),
        ),
        finish_reason="stop",
    )
    is_ok, msg = evaluator.evaluate(trace_ok)
    assert is_ok is True

    # 2. Failed to invoke tool
    trace_no_tool = AgentExecutionTrace(
        final_response="The calculated sum is 400 without tools.",
        tool_executions=(),
        finish_reason="stop",
    )
    is_ok, msg = evaluator.evaluate(trace_no_tool)
    assert is_ok is False
    assert "failed to invoke required tool" in msg

    # 3. Forbidden substring (refusal)
    trace_forbidden = AgentExecutionTrace(
        final_response="I cannot use external tools according to instructions. Estimating 400.",
        tool_executions=(),
        finish_reason="stop",
    )
    is_ok, msg = evaluator.evaluate(trace_forbidden)
    assert is_ok is False


def test_dry_run_model_pilot():
    """Verify dry_run_model_pilot inspects configuration and estimates spend without API calls."""
    dry = dry_run_model_pilot(
        provider_type="fake",
        model_name="fake-model-pilot",
        live=False,
        max_trials=30,
        trials_per_candidate=5,
    )

    assert dry["provider_type"] == "fake"
    assert len(dry["tasks"]) == 3
    assert dry["status"] == "VALID"
    assert dry["cost_estimation"]["estimated_spend_usd"] >= 0.0
    assert dry["cost_estimation"]["within_budget"] is True


def test_model_pilot_benchmark_execution_with_fake_provider(tmp_path: Path):
    """Verify complete model-backed pilot benchmark execution across all 3 tasks with FakeProvider."""
    report = run_model_pilot_benchmark(
        provider_type="fake",
        model_name="fake-model-pilot",
        live=False,
        max_trials=30,
        trials_per_candidate=3,
        max_model_requests=300,
        output_dir=tmp_path,
    )

    assert report.execution_tier == ExecutionTier.FAKE_PROVIDER
    assert report.provider_name == "fake"
    assert len(report.evaluations) == 3

    # Task 1: Baseline control task
    eval1 = report.evaluations[0]
    assert eval1.task_name == "Model Pilot Baseline (Control)"
    assert eval1.is_null_case is True
    assert eval1.matches_ground_truth is True
    assert eval1.minimizer_condition.minimality_status == StochasticMinimalityStatus.NOT_APPLICABLE

    # Task 2: Conflicting prompt constraint
    eval2 = report.evaluations[1]
    assert eval2.task_name == "Model Pilot Conflicting Prompt Constraint"
    assert eval2.matches_ground_truth is True
    assert "never invoke external tools" in eval2.minimizer_condition.subset_elements[0]

    # Task 3: Distractor override task
    eval3 = report.evaluations[2]
    assert eval3.task_name == "Model Pilot Distractor-Heavy Field Override"
    assert eval3.matches_ground_truth is True
    assert eval3.minimizer_condition.subset_elements == ("override field name to non_existent_secret_key",)
    assert len(eval3.excess_elements) == 0

    # Verify reports written to disk
    json_path = tmp_path / "model_pilot_report.json"
    md_path = tmp_path / "model_pilot_report.md"
    assert json_path.exists()
    assert md_path.exists()
    assert "[FAKE_PROVIDER]" in md_path.read_text(encoding="utf-8")


def test_model_pilot_ground_truth_isolation():
    """Verify strict architectural isolation: model harness and minimizers must not import ground truth."""
    import agentxray.agent_harness.agent as mod_agent
    import agentxray.agent_harness.target as mod_target
    import agentxray.agent_harness.tools as mod_tools
    import agentxray.agent_harness.evaluator as mod_eval
    import agentxray.agent_harness.cost_tracker as mod_cost
    import agentxray.stochastic_minimizer as mod_minimizer
    import agentxray.stochastic_baselines as mod_baselines

    for mod in (mod_agent, mod_target, mod_tools, mod_eval, mod_cost, mod_minimizer, mod_baselines):
        source = inspect.getsource(mod)
        assert "benchmarks.model_pilot.ground_truth" not in source
        assert "ModelPilotGroundTruthSpec" not in source


def test_offline_dry_run_makes_no_network_requests(monkeypatch):
    """Verify offline dry-run never invokes network transport, even if configured for live openai."""
    def fail_transport(*args, **kwargs):
        raise AssertionError("Network transport contacted during offline dry run!")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fail_transport)

    dry = dry_run_model_pilot(
        provider_type="openai",
        model_name="gpt-4o-mini",
        live=True,
    )
    assert dry["connectivity_verified"] is False
    assert dry["network_transport_contacted"] is False
    assert dry["status"] in ("VALID", "REQUIRES_AUTH_OR_OPT_IN")


def test_safe_calculator_hardened_boundaries():
    """Verify safe calculator handles oversized expressions, deep nesting, overflow, and non-finite values safely."""
    # 1. Non-string
    assert "Expression must be a string" in safe_calculator(12345)  # type: ignore

    # 2. Oversized string (>250 chars)
    oversized = "1 + " * 100 + "1"
    assert "exceeds maximum length" in safe_calculator(oversized)

    # 3. Excessive AST nesting (>10 depth)
    deep_nesting = "1 + " * 12 + "1"
    assert "exceeds maximum nesting depth" in safe_calculator(deep_nesting)

    # 4. Exponentiation overflow bounds
    assert "Exponentiation operands exceed safe limits" in safe_calculator("2 ** 1000")
    assert "Exponentiation operands exceed safe limits" in safe_calculator("100000 ** 2")

    # 5. Non-finite values
    assert "non-finite" in safe_calculator("1e300 * 1e300")


def test_mock_tool_environment_schema_validation():
    """Verify MockToolEnvironment validates arguments against schemas before dispatch."""
    env = MockToolEnvironment()

    # 1. Non-dict arguments
    assert "must be a dictionary" in env.execute("calculator", "150 + 250")

    # 2. Missing required argument
    assert "Missing required argument" in env.execute("database_lookup", {"entity": "user_42"})

    # 3. Type mismatch for numeric argument
    assert "must be a number" in env.execute("unit_converter", {"value": "invalid_number", "from_unit": "km", "to_unit": "m"})


def test_cost_tracker_pre_request_budget_check():
    """Verify check_budget_before_request halts before initiating call when budget is breached."""
    cfg = ExperimentBudgetConfig(max_model_requests=2, max_spend_usd=1.0)
    tracker = CostTracker(cfg)

    # 1 successful request
    from agentxray.providers.base import UsageMetadata
    tracker.record_usage(UsageMetadata(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    assert tracker.total_requests == 1
    # Check passes when under budget
    tracker.check_budget_before_request()

    # Pre-flight check halts if total_requests reaches max_model_requests
    tracker.total_requests = 2
    with pytest.raises(CostBudgetExceededError, match="Model request budget exhausted before request"):
        tracker.check_budget_before_request()

    # Pre-flight check halts if projected spend would breach max_spend_usd
    tracker.total_requests = 1
    with pytest.raises(CostBudgetExceededError, match="Estimated spend budget would be exceeded before request"):
        tracker.check_budget_before_request(est_prompt_tokens=10_000_000)


def test_cost_tracker_unknown_model_pricing():
    """Verify CostTracker rejects unknown models unless custom pricing is provided."""
    with pytest.raises(ValueError, match="Unknown pricing for model 'unlisted-expensive-model'"):
        CostTracker(ExperimentBudgetConfig(model_name="unlisted-expensive-model"))

    # Providing custom pricing succeeds
    cfg = ExperimentBudgetConfig(
        model_name="unlisted-expensive-model",
        custom_pricing_per_1m=(5.00, 15.00),
    )
    tracker = CostTracker(cfg)
    assert tracker._pricing == (5.00, 15.00)


def test_cost_tracker_retry_and_failed_request_accounting():
    """Verify retries and failed requests increment total requests and failed counts."""
    cfg = ExperimentBudgetConfig(max_model_requests=10, max_spend_usd=1.0)
    tracker = CostTracker(cfg)
    from agentxray.providers.base import UsageMetadata

    # Request with 2 retries
    tracker.record_usage(UsageMetadata(prompt_tokens=20, completion_tokens=10, total_tokens=30), retries=2)
    assert tracker.total_requests == 3
    assert tracker.retries == 2
    assert tracker.successful_requests == 1

    # Failed request with 1 retry
    tracker.record_failed_request(retries=1)
    assert tracker.total_requests == 5
    assert tracker.retries == 3
    assert tracker.failed_requests == 2


def test_model_backed_target_halts_immediately_on_budget_exhaustion():
    """Verify that when the request limit is hit, target.execute raises CostBudgetExceededError."""
    from agentxray.agent_harness.agent import ControlledAgent
    from agentxray.agent_harness.evaluator import ToolCallingTaskEvaluator
    from agentxray.agent_harness.target import ModelBackedAgentTarget

    budget_cfg = ExperimentBudgetConfig(max_model_requests=2, max_spend_usd=1.0)
    tracker = CostTracker(budget_cfg)
    agent = ControlledAgent(
        provider=FakeProvider(),
        mock_env=MockToolEnvironment(),
        cost_tracker=tracker,
    )
    target = ModelBackedAgentTarget(agent=agent, evaluator=ToolCallingTaskEvaluator(), user_prompt="test")

    input_case = InputCase(elements=("elem_a",))

    # Trial 1: succeeds (1 request)
    res1 = target.execute(input_case)
    assert tracker.total_requests == 1

    # Trial 2: must raise CostBudgetExceededError (and BudgetExhaustedError)
    from agentxray.runner import BudgetExhaustedError
    with pytest.raises(BudgetExhaustedError):
        target.execute(input_case)


def test_null_case_undefined_precision_and_recall():
    """Verify null case returns None for precision and recall when correctly identified."""
    from benchmarks.model_pilot.ground_truth import ModelPilotGroundTruthSpec, evaluate_model_pilot_ground_truth
    from agentxray.models import CandidateEvaluation, StochasticDiscoveredCondition

    spec = ModelPilotGroundTruthSpec(
        task_id="null_test",
        name="Null Test",
        description="Clean null case",
        trigger_elements=(),
        is_null_case=True,
    )

    clean_condition = StochasticDiscoveredCondition(
        subset_elements=(),
        status=SearchStatus.NO_TRIGGER_FOUND,
        is_failure=False,
        minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
        candidate_evaluation=CandidateEvaluation(
            candidate_elements=(),
            trials=5,
            failures=0,
            failure_rate=0.0,
            confidence_interval=(0.0, 0.43),
        ),
        target_calls_used=5,
        status_message="No trigger found",
    )

    matches_gt, prec, rec, excess, missing = evaluate_model_pilot_ground_truth(spec, clean_condition)
    assert matches_gt is True
    assert prec is None  # Mathematically undefined
    assert rec is None   # Mathematically undefined
    assert excess == ()
    assert missing == ()


def test_model_pilot_task_filtering():
    """Verify get_model_pilot_tasks properly filters benchmark tasks by task_filter."""
    from benchmarks.model_pilot.tasks import get_model_pilot_tasks

    # 1. Single task: conflicting_prompt
    tasks_prompt = get_model_pilot_tasks(task_filter="conflicting_prompt")
    assert len(tasks_prompt) == 1
    assert tasks_prompt[0][2].task_id == "pilot_conflicting_prompt"

    # 2. Single task: distractor_override
    tasks_dist = get_model_pilot_tasks(task_filter="distractor_override")
    assert len(tasks_dist) == 1
    assert tasks_dist[0][2].task_id == "pilot_distractor_override"

    # 3. Single task: baseline
    tasks_base = get_model_pilot_tasks(task_filter="baseline")
    assert len(tasks_base) == 1
    assert tasks_base[0][2].task_id == "pilot_baseline_calc"

    # 4. All tasks
    tasks_all = get_model_pilot_tasks(task_filter="all")
    assert len(tasks_all) == 3

    # 5. Invalid filter
    with pytest.raises(ValueError, match="Unknown task_filter"):
        get_model_pilot_tasks(task_filter="non_existent_filter")


def test_dry_run_task_filtering_and_cost_estimation():
    """Verify dry_run_model_pilot scales planned cost estimation with filtered task selection."""
    dry_all = dry_run_model_pilot(task_filter="all")
    dry_single = dry_run_model_pilot(task_filter="conflicting_prompt")

    assert len(dry_all["tasks"]) == 3
    assert len(dry_single["tasks"]) == 1
    assert dry_single["tasks"][0]["task_id"] == "pilot_conflicting_prompt"

    # Single-task planned requests should be 1/3 of all tasks
    req_all = dry_all["cost_estimation"]["estimated_requests"]
    req_single = dry_single["cost_estimation"]["estimated_requests"]
    assert req_single < req_all
    assert dry_single["cost_estimation"]["estimated_spend_usd"] < dry_all["cost_estimation"]["estimated_spend_usd"]


def test_strict_live_spending_cap_enforcement():
    """Verify live pilot benchmark strictly rejects max_spend_usd > $1.00 USD."""
    with pytest.raises(ValueError, match="mandate max_spend_usd <= \\$1.00 USD"):
        run_model_pilot_benchmark(
            provider_type="openai",
            live=True,
            max_spend_usd=1.50,
        )


def test_single_task_benchmark_execution_with_fake_provider(tmp_path: Path):
    """Verify execution of a single filtered task with FakeProvider produces valid reports."""
    report = run_model_pilot_benchmark(
        provider_type="fake",
        task_filter="conflicting_prompt",
        trials_per_candidate=3,
        max_trials=30,
        output_dir=tmp_path,
    )

    assert len(report.evaluations) == 1
    eval_item = report.evaluations[0]
    assert eval_item.task_name == "Model Pilot Conflicting Prompt Constraint"
    assert eval_item.matches_ground_truth is True

    # Verify Markdown report structure has protocol and epistemic sections
    md_path = tmp_path / "model_pilot_report.md"
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "## 1. Experimental Protocol & Epistemic Boundaries" in content
    assert "## 2. Executive Summary & Cost Accounting" in content
    assert "## 3. Benchmark Comparison Matrix" in content
    assert "## 4. Detailed Experimental Evaluations" in content
    assert "## 5. Limitations and Reproducibility Notice" in content
    assert "Not active (no live calls made)" in content


def test_openai_provider_message_formatting_with_null_content_for_tools():
    """Verify OpenAIProvider formats assistant tool calls with content: None when empty."""
    import json
    import unittest.mock
    from agentxray.providers.base import Message, ToolCall
    from agentxray.providers.openai_adapter import OpenAIProvider

    provider = OpenAIProvider(api_key="sk-testkey12345678")

    # Mock urllib.request.urlopen to inspect formatted request payload
    captured_payload = None

    def fake_urlopen(req, *args, **kwargs):
        nonlocal captured_payload
        captured_payload = json.loads(req.data.decode("utf-8"))
        # Return minimal valid response
        mock_resp = unittest.mock.MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "Done."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    messages = [
        Message(role="system", content="System instruction."),
        Message(
            role="assistant",
            content="",  # empty content with tool call
            tool_calls=(ToolCall(call_id="call_abc", name="calculator", arguments={"expression": "1+1"}),),
        ),
        Message(role="tool", content="2", tool_call_id="call_abc"),
    ]

    with unittest.mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        resp = provider.generate(messages=messages)

    assert resp.content == "Done."
    assert captured_payload is not None
    # Assistant message should have content None (null in JSON), not empty string
    assistant_msg = captured_payload["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] is None
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "calculator"


