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
    IsolatedAlgorithmCostTracker,
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

    # Trial 2: succeeds (consumes 2nd allowed request, bringing total to limit 2)
    res2 = target.execute(input_case)
    assert tracker.total_requests == 2

    # Trial 3: must raise CostBudgetExceededError (and BudgetExhaustedError) pre-flight
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


def test_openai_provider_and_dry_run_load_key_from_env_file(tmp_path: Path, monkeypatch):
    """Verify OpenAIProvider and dry_run_model_pilot read credentials from local .env if unset in os.environ."""
    from agentxray.providers.openai_adapter import OpenAIProvider

    # Clear OPENAI_API_KEY from os.environ
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    # Without .env, should be missing
    dry_missing = dry_run_model_pilot(provider_type="openai")
    assert dry_missing["api_key_configured"] is False

    # Create local .env file
    env_file = tmp_path / ".env"
    env_file.write_text('OPENAI_API_KEY="sk-from-env-file-12345678"\n', encoding="utf-8")

    # Now dry run should detect it
    dry_detected = dry_run_model_pilot(provider_type="openai")
    assert dry_detected["api_key_configured"] is True

    # OpenAIProvider should also resolve it
    prov = OpenAIProvider()
    assert prov._api_key == "sk-from-env-file-12345678"


def test_target_aborts_on_provider_infrastructure_error():
    """Verify ModelBackedAgentTarget aborts immediately on provider crashes without treating them as agent failures."""
    from agentxray.agent_harness.agent import AgentExecutionTrace, ControlledAgent
    from agentxray.agent_harness.cost_tracker import CostTracker, ExperimentBudgetConfig
    from agentxray.agent_harness.evaluator import ToolCallingTaskEvaluator
    from agentxray.agent_harness.target import ModelBackedAgentTarget
    from agentxray.models import InputCase
    from agentxray.providers.fake import FakeProvider

    prov = FakeProvider(model_name="fake-crash", simulate_error=RuntimeError)
    tracker = CostTracker(ExperimentBudgetConfig(max_model_requests=10, max_spend_usd=1.0, model_name="fake"))
    agent = ControlledAgent(provider=prov, mock_env=None, cost_tracker=tracker, max_turns=2)
    evaluator = ToolCallingTaskEvaluator(required_tool="calculator")
    target = ModelBackedAgentTarget(agent=agent, evaluator=evaluator, user_prompt="test")

    with pytest.raises(RuntimeError, match="Controlled agent execution aborted due to provider infrastructure error"):
        target.execute(InputCase(case_id="c1", elements=("el1",)))


def test_dry_run_makes_zero_network_calls():
    """Verify dry_run_model_pilot makes strictly zero network transport calls even when configured for live openai."""
    from unittest.mock import patch
    with patch("urllib.request.urlopen") as mock_urlopen:
        result = dry_run_model_pilot(
            provider_type="openai",
            live=True,
            task_filter="conflicting_prompt",
            max_model_requests=100,
            max_spend_usd=1.00,
        )
        assert mock_urlopen.call_count == 0
        assert result["network_transport_contacted"] is False


def test_agent_retries_bounded_by_budget_headroom():
    """Verify ControlledAgent bounds provider retries by remaining request headroom so limit is never breached."""
    from agentxray.agent_harness.agent import ControlledAgent
    from agentxray.agent_harness.cost_tracker import CostTracker, ExperimentBudgetConfig
    from agentxray.providers.fake import FakeProvider

    provider = FakeProvider(model_name="gpt-4o-mini", max_retries_limit=3)

    tracker = CostTracker(ExperimentBudgetConfig(max_model_requests=2, max_spend_usd=1.00, model_name="gpt-4o-mini"))
    # Consume 1 request so only 1 request remains in budget
    tracker.record_failed_request(retries=0)
    assert tracker.total_requests == 1

    agent = ControlledAgent(provider=provider, mock_env=None, cost_tracker=tracker, max_turns=1)
    trace = agent.run_session(system_prompt="sys", user_prompt="user")

    # generate must have been called with max_retries=0 (since headroom was 2 - 1 = 1, so 0 retries allowed)
    assert provider.request_count == 1
    assert provider.last_max_retries == 0
    assert trace.finish_reason == "stop"
    assert trace.final_response != ""
    assert tracker.total_requests == 2  # exactly at limit, not exceeded

    # Any subsequent request is blocked pre-flight
    trace2 = agent.run_session(system_prompt="sys", user_prompt="user")
    assert trace2.finish_reason == "budget_exhausted"
    assert trace2.final_response == ""


def test_model_pilot_benchmark_seed_propagation(tmp_path: Path):
    """Verify run_model_pilot_benchmark records seed in config/metadata and seeds random search baseline."""
    report_a1 = run_model_pilot_benchmark(
        provider_type="fake",
        task_filter="distractor_override",
        max_trials=25,
        trials_per_candidate=2,
        seed=101,
        output_dir=tmp_path / "run_a1",
    )
    report_a2 = run_model_pilot_benchmark(
        provider_type="fake",
        task_filter="distractor_override",
        max_trials=25,
        trials_per_candidate=2,
        seed=101,
        output_dir=tmp_path / "run_a2",
    )
    report_b = run_model_pilot_benchmark(
        provider_type="fake",
        task_filter="distractor_override",
        max_trials=25,
        trials_per_candidate=2,
        seed=999,
        output_dir=tmp_path / "run_b",
    )

    # 1. Config and metadata check
    assert report_a1.config.seed == 101
    assert report_a1.config.metadata["seed"] == 101
    assert report_b.config.seed == 999
    assert report_b.config.metadata["seed"] == 999

    # 2. Baseline seeded random search determinism: identical seeds yield identical sampling
    rand_a1 = report_a1.evaluations[0].baseline_comparisons[1]
    rand_a2 = report_a2.evaluations[0].baseline_comparisons[1]
    rand_b = report_b.evaluations[0].baseline_comparisons[1]

    assert rand_a1.baseline_name == "stochastic_seeded_random"
    assert rand_a1.target_calls == rand_a2.target_calls
    assert rand_a1.subsets_examined == rand_a2.subsets_examined
    assert rand_a1.discovered_elements == rand_a2.discovered_elements

    # 3. Explicit check that configured non-default seed reaches run_stochastic_random_search
    from unittest.mock import patch
    import benchmarks.model_pilot.runner as runner_mod

    captured_random_search_seeds: list[int] = []
    orig_random_search = runner_mod.run_stochastic_random_search

    def spy_random_search(*args, **kwargs):
        if "seed" in kwargs:
            captured_random_search_seeds.append(kwargs["seed"])
        return orig_random_search(*args, **kwargs)

    with patch.object(runner_mod, "run_stochastic_random_search", side_effect=spy_random_search):
        run_model_pilot_benchmark(
            provider_type="fake",
            task_filter="distractor_override",
            max_trials=10,
            trials_per_candidate=1,
            seed=777,
            output_dir=None,
        )

    assert 777 in captured_random_search_seeds


def test_model_pilot_benchmark_ollama_receives_seed(monkeypatch: pytest.MonkeyPatch):
    """Verify run_model_pilot_benchmark passes seed to OllamaProvider constructor."""
    from agentxray.providers.ollama_adapter import OllamaProvider

    class MockInitStop(Exception):
        pass

    captured_init_kwargs: dict = {}

    def mock_init(self, *args, **kwargs):
        captured_init_kwargs.update(kwargs)
        raise MockInitStop("Provider instantiated with seed")

    monkeypatch.setattr(OllamaProvider, "__init__", mock_init)

    with pytest.raises(MockInitStop):
        run_model_pilot_benchmark(
            provider_type="ollama",
            model_name="llama3.2:3b",
            task_filter="calibrated_currency",
            max_trials=10,
            max_model_requests=50,
            seed=4321,
            output_dir=None,
        )

    assert captured_init_kwargs.get("seed") == 4321


def test_model_pilot_benchmark_openai_receives_seed(monkeypatch: pytest.MonkeyPatch):
    """Verify run_model_pilot_benchmark passes seed to OpenAIProvider constructor."""
    from agentxray.providers.openai_adapter import OpenAIProvider

    class MockInitStop(Exception):
        pass

    captured_init_kwargs: dict = {}

    def mock_init(self, *args, **kwargs):
        captured_init_kwargs.update(kwargs)
        raise MockInitStop("OpenAIProvider instantiated with seed")

    monkeypatch.setattr(OpenAIProvider, "__init__", mock_init)

    with pytest.raises(MockInitStop):
        run_model_pilot_benchmark(
            provider_type="openai",
            model_name="gpt-4o-mini",
            live=True,
            task_filter="baseline",
            max_trials=10,
            max_model_requests=50,
            seed=9876,
            output_dir=None,
        )

    assert captured_init_kwargs.get("seed") == 9876


def test_dry_run_model_pilot_seed():
    """Verify dry_run_model_pilot accepts and returns the specified seed."""
    res_default = dry_run_model_pilot()
    assert res_default["seed"] == 42

    res_custom = dry_run_model_pilot(seed=789)
    assert res_custom["seed"] == 789


def test_cli_model_pilot_seed_forwarding(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """Verify CLI forwards seed to dry-run banner and benchmark runner."""
    from agentxray.__main__ import main
    import agentxray.__main__ as main_mod

    # 1. Dry run CLI includes seed in stdout
    monkeypatch.setattr("sys.argv", ["agentxray", "--mode", "model", "--dry-run", "--seed", "31415"])
    main()
    out = capsys.readouterr().out
    assert "Random Seed: `31415`" in out

    # 2. Execution CLI forwards seed to run_model_pilot_benchmark
    captured_args = {}

    def mock_run_pilot(**kwargs):
        captured_args.update(kwargs)
        raise RuntimeError("Stop after capturing kwargs")

    monkeypatch.setattr(main_mod, "run_model_pilot_benchmark", mock_run_pilot)
    monkeypatch.setattr("sys.argv", ["agentxray", "--mode", "model", "--provider", "fake", "--seed", "9876"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    assert captured_args.get("seed") == 9876


def test_model_pilot_per_algorithm_budget_isolation():
    """Verify that each algorithm receives its intended request allowance independently."""
    # Under conflicting_prompt, DD needs ~17-20 calls. With max_model_requests=10, DD will breach.
    # But Greedy and Random must start with fresh allowances and execute their own trials (>0).
    report = run_model_pilot_benchmark(
        provider_type="fake",
        task_filter="conflicting_prompt",
        max_model_requests=10,
        max_trials=20,
    )
    assert len(report.evaluations) == 1
    eval_res = report.evaluations[0]

    # Algorithm 1: DD was capped by its own budget
    assert eval_res.minimizer_condition.target_calls_used > 0
    assert eval_res.minimizer_condition.status == SearchStatus.BUDGET_EXHAUSTED

    # Algorithm 2: Greedy started fresh and made non-zero calls
    greedy_res = eval_res.baseline_comparisons[0]
    assert greedy_res.target_calls > 0

    # Algorithm 3: Seeded Random started fresh and made non-zero calls
    random_res = eval_res.baseline_comparisons[1]
    assert random_res.target_calls > 0


def test_first_algorithm_exhaustion_does_not_starve_subsequent_algorithms():
    """Verify that earlier algorithms consuming their full budget cannot starve later algorithms."""
    report = run_model_pilot_benchmark(
        provider_type="fake",
        task_filter="conflicting_prompt",
        max_model_requests=8,
        max_trials=15,
    )
    eval_res = report.evaluations[0]

    # DD exhausts at 8 requests
    assert eval_res.minimizer_condition.status == SearchStatus.BUDGET_EXHAUSTED

    # In the old flawed implementation, Greedy and Random were starved with target_calls=0.
    # In the isolated implementation, both receive independent quotas and execute trials.
    assert eval_res.baseline_comparisons[0].target_calls > 0
    assert eval_res.baseline_comparisons[1].target_calls > 0


def test_budget_exhaustion_accurate_attribution():
    """Verify that budget exhaustion is not falsely attributed to algorithms that operated within their allowance."""
    # On baseline task, each algorithm requires 5 target calls (10 model requests with 2 turns/call).
    # With max_model_requests=12, each algorithm requires 10 <= 12 requests.
    # Under a shared budget, 10 + 10 = 20 > 12, so Greedy and Random would falsely record budget_exhausted=True.
    # Under isolated budgets, every algorithm operates within its allowance and budget_exhausted is False.
    report = run_model_pilot_benchmark(
        provider_type="fake",
        task_filter="baseline",
        max_model_requests=12,
        max_trials=10,
    )
    eval_res = report.evaluations[0]

    assert eval_res.minimizer_condition.status == SearchStatus.NO_TRIGGER_FOUND
    assert eval_res.baseline_comparisons[0].budget_exhausted is False
    assert eval_res.baseline_comparisons[1].budget_exhausted is False


def test_cumulative_budget_summary_accurate_and_no_double_counting():
    """Verify that global budget summary aggregates all algorithm requests without duplication."""
    report = run_model_pilot_benchmark(
        provider_type="fake",
        task_filter="baseline",
        max_model_requests=25,
        max_trials=10,
    )
    bs = report.budget_summary
    assert bs["max_requests_per_algorithm"] == 25
    assert "per_algorithm_usage" in bs

    sum_algo_reqs = sum(item["model_requests"] for item in bs["per_algorithm_usage"])
    assert bs["total_requests"] == sum_algo_reqs
    sum_algo_calls = sum(item["target_calls"] for item in bs["per_algorithm_usage"])
    assert report.total_target_calls == sum_algo_calls
    # Each target trial executes a 2-turn agent session (calculator tool call + final response)
    assert bs["total_requests"] == report.total_target_calls * 2
    assert bs["successful_requests"] == bs["total_requests"]
    assert bs["failed_requests"] == 0


def test_global_spend_safety_cap_halts_execution():
    """Verify that cumulative global spend limit is enforced across isolated algorithm trackers."""
    from agentxray.providers.base import UsageMetadata

    global_cfg = ExperimentBudgetConfig(
        max_model_requests=100,
        max_spend_usd=0.001,
        model_name="fake-model",
        custom_pricing_per_1m=(100.0, 100.0),
    )
    global_tracker = CostTracker(global_cfg)

    algo_cfg = ExperimentBudgetConfig(
        max_model_requests=50,
        max_spend_usd=1.0,
        model_name="fake-model",
        custom_pricing_per_1m=(100.0, 100.0),
    )

    t1 = IsolatedAlgorithmCostTracker(algo_cfg, global_tracker=global_tracker, enforce_global_spend=True)
    t2 = IsolatedAlgorithmCostTracker(algo_cfg, global_tracker=global_tracker, enforce_global_spend=True)

    # 4 prompt tokens + 4 completion tokens at $100/1M = $0.0008 USD (< $0.0010 USD)
    usage = UsageMetadata(prompt_tokens=4, completion_tokens=4, total_tokens=8)
    t1.check_budget_before_request(est_prompt_tokens=4)
    t1.record_usage(usage)
    assert t1.total_requests == 1
    assert global_tracker.total_requests == 1
    assert global_tracker.estimated_spend_usd == 0.0008

    # Now t2 runs and attempts a call with est_prompt_tokens=4 ($0.0004 USD), which would push total spend to $0.0012 >= $0.0010
    with pytest.raises(CostBudgetExceededError, match="Global estimated spend budget"):
        t2.check_budget_before_request(est_prompt_tokens=4)


def test_provider_failure_recorded_in_both_local_and_global_trackers():
    """Verify failed request attempts and retries are mirrored to global tracker and maintain counter integrity."""
    global_cfg = ExperimentBudgetConfig(
        max_model_requests=10,
        max_spend_usd=5.0,
        model_name="fake-model",
    )
    global_tracker = CostTracker(global_cfg)

    algo_cfg = ExperimentBudgetConfig(
        max_model_requests=5,
        max_spend_usd=5.0,
        model_name="fake-model",
    )
    tracker = IsolatedAlgorithmCostTracker(
        algo_cfg,
        global_tracker=global_tracker,
        enforce_global_requests=True,
    )

    # 1. Non-exhausting failure with retries
    tracker.record_failed_request(retries=2)
    assert tracker.total_requests == 3
    assert tracker.failed_requests == 3
    assert tracker.retries == 2
    assert global_tracker.total_requests == 3
    assert global_tracker.failed_requests == 3
    assert global_tracker.retries == 2

    # 2. Local budget exhaustion on failure
    with pytest.raises(CostBudgetExceededError, match="Model request budget exhausted"):
        tracker.record_failed_request(retries=1)

    # Counters must be mutated despite exception (exception-safety and counter consistency)
    assert tracker.total_requests == 5
    assert tracker.failed_requests == 5
    assert tracker.retries == 3
    assert global_tracker.total_requests == 5
    assert global_tracker.failed_requests == 5
    assert global_tracker.retries == 3


def test_multiturn_agent_request_headroom_and_retry_bounding():
    """Verify agent bounds retries by remaining headroom and blocks requests pre-flight once budget is reached."""
    cfg = ExperimentBudgetConfig(
        max_model_requests=2,
        max_spend_usd=5.0,
        model_name="fake-model",
    )
    tracker = CostTracker(cfg)
    provider = FakeProvider(model_name="fake-model")
    env = MockToolEnvironment()
    agent = ControlledAgent(
        provider=provider,
        mock_env=env,
        cost_tracker=tracker,
        max_turns=3,
    )

    # Headroom retry bounding check: with 2 requests remaining, allowed retries is 1
    remaining_requests = tracker.config.max_model_requests - tracker.total_requests
    assert max(0, remaining_requests - 1) == 1

    # Prompt triggers calculator tool call on turn 1 and final answer on turn 2
    trace = agent.run_session(
        system_prompt="You are an automated accounting agent. Always use available tools for calculations.",
        user_prompt="Calculate the total expense reimbursement for receipts of $150 and $250 using the calculator tool.",
        tools=[CALCULATOR_TOOL_DEF],
    )

    # Turn 1 executed tool; turn 2 completed answer and hit the request cap of 2
    assert len(trace.tool_executions) == 1
    assert trace.tool_executions[0].tool_name == "calculator"
    assert tracker.total_requests == 2
    assert trace.finish_reason == "stop"
    assert "400" in trace.final_response

    # With remaining headroom = 0, retry bounding produces 0
    remaining_now = tracker.config.max_model_requests - tracker.total_requests
    assert max(0, remaining_now - 1) == 0

    # Any subsequent request is blocked pre-flight without sending network calls
    with pytest.raises(CostBudgetExceededError, match="Model request budget exhausted before request"):
        tracker.check_budget_before_request()

    # Agent running a new session aborts on turn 1 pre-flight without incrementing requests
    trace2 = agent.run_session(
        system_prompt="",
        user_prompt="Follow-up question",
        tools=[],
    )
    assert trace2.finish_reason == "budget_exhausted"
    assert "budget exhausted before request" in trace2.error_message
    assert tracker.total_requests == 2


def test_suite_wide_request_limit_enforced_across_algorithms():
    """Verify that cumulative suite-wide request limit is enforced across multiple algorithms."""
    from agentxray.providers.base import UsageMetadata

    # Suite cap: 6 total requests across all algorithms
    global_cfg = ExperimentBudgetConfig(
        max_model_requests=6,
        max_spend_usd=5.0,
        model_name="fake-model",
    )
    global_tracker = CostTracker(global_cfg)

    # Each algorithm has local allowance of 5 requests
    algo_cfg = ExperimentBudgetConfig(
        max_model_requests=5,
        max_spend_usd=5.0,
        model_name="fake-model",
    )

    t1 = IsolatedAlgorithmCostTracker(
        algo_cfg,
        global_tracker=global_tracker,
        enforce_global_requests=True,
    )
    t2 = IsolatedAlgorithmCostTracker(
        algo_cfg,
        global_tracker=global_tracker,
        enforce_global_requests=True,
    )

    usage = UsageMetadata(prompt_tokens=10, completion_tokens=10, total_tokens=20)

    # Algorithm 1 executes 4 requests (within local limit 5 and global limit 6)
    for _ in range(4):
        t1.check_budget_before_request()
        t1.record_usage(usage)

    assert t1.total_requests == 4
    assert global_tracker.total_requests == 4

    # Algorithm 2 starts fresh locally. First request succeeds (global reaches 5).
    t2.check_budget_before_request()
    t2.record_usage(usage)
    assert t2.total_requests == 1
    assert global_tracker.total_requests == 5

    # 2nd request for algorithm 2 raises CostBudgetExceededError on record_usage because global limit (6) is hit
    with pytest.raises(CostBudgetExceededError, match="Model request budget exhausted"):
        t2.check_budget_before_request()
        t2.record_usage(usage)

    assert t2.total_requests == 2
    assert global_tracker.total_requests == 6

    # Subsequent request is blocked pre-flight by check_budget_before_request
    with pytest.raises(CostBudgetExceededError, match="Global benchmark model request budget exhausted before request"):
        t2.check_budget_before_request()


def test_local_usage_recording_exhaustion_updates_global_tracker_before_raising():
    """Verify global tracker records usage even when local tracker raises CostBudgetExceededError on exhaustion."""
    from agentxray.providers.base import UsageMetadata

    global_cfg = ExperimentBudgetConfig(
        max_model_requests=10,
        max_spend_usd=5.0,
        model_name="fake-model",
    )
    global_tracker = CostTracker(global_cfg)

    algo_cfg = ExperimentBudgetConfig(
        max_model_requests=2,
        max_spend_usd=5.0,
        model_name="fake-model",
    )
    tracker = IsolatedAlgorithmCostTracker(
        algo_cfg,
        global_tracker=global_tracker,
        enforce_global_requests=True,
    )

    usage = UsageMetadata(prompt_tokens=50, completion_tokens=25, total_tokens=75)

    # Request 1: succeeds locally and globally
    tracker.check_budget_before_request(est_prompt_tokens=50)
    tracker.record_usage(usage)
    assert tracker.total_requests == 1
    assert tracker.successful_requests == 1
    assert tracker.total_tokens == 75
    assert global_tracker.total_requests == 1
    assert global_tracker.successful_requests == 1
    assert global_tracker.total_tokens == 75

    # Request 2: reaches local limit (2 >= 2) -> record_usage raises CostBudgetExceededError
    tracker.check_budget_before_request(est_prompt_tokens=50)
    with pytest.raises(CostBudgetExceededError, match="Model request budget exhausted: executed 2 requests"):
        tracker.record_usage(usage)

    # Counter mutation & exception safety: global tracker MUST record request 2 and its tokens
    assert tracker.total_requests == 2
    assert tracker.successful_requests == 2
    assert tracker.total_tokens == 150
    assert global_tracker.total_requests == 2
    assert global_tracker.successful_requests == 2
    assert global_tracker.total_tokens == 150

    # Subsequent request is blocked pre-flight before sending any provider request
    with pytest.raises(CostBudgetExceededError, match="Model request budget exhausted before request"):
        tracker.check_budget_before_request(est_prompt_tokens=50)

    # Counters remain unchanged after pre-flight block
    assert tracker.total_requests == 2
    assert global_tracker.total_requests == 2


def test_global_usage_recording_exhaustion_halts_local_execution_and_updates_counters():
    """Verify local tracker halts when global request limit is hit, with consistent counter mutation."""
    from agentxray.providers.base import UsageMetadata

    global_cfg = ExperimentBudgetConfig(
        max_model_requests=2,
        max_spend_usd=5.0,
        model_name="fake-model",
    )
    global_tracker = CostTracker(global_cfg)

    algo_cfg = ExperimentBudgetConfig(
        max_model_requests=10,
        max_spend_usd=5.0,
        model_name="fake-model",
    )
    tracker = IsolatedAlgorithmCostTracker(
        algo_cfg,
        global_tracker=global_tracker,
        enforce_global_requests=True,
    )

    usage = UsageMetadata(prompt_tokens=40, completion_tokens=20, total_tokens=60)

    # Request 1: OK
    tracker.check_budget_before_request(est_prompt_tokens=40)
    tracker.record_usage(usage)
    assert tracker.total_requests == 1
    assert global_tracker.total_requests == 1

    # Request 2: reaches global limit (2 >= 2) -> record_usage raises CostBudgetExceededError from global tracker
    tracker.check_budget_before_request(est_prompt_tokens=40)
    with pytest.raises(CostBudgetExceededError, match="Model request budget exhausted"):
        tracker.record_usage(usage)

    # Both counters are updated to 2
    assert tracker.total_requests == 2
    assert tracker.total_tokens == 120
    assert global_tracker.total_requests == 2
    assert global_tracker.total_tokens == 120

    # Subsequent request is blocked pre-flight by global check
    with pytest.raises(CostBudgetExceededError, match="Global benchmark model request budget exhausted before request"):
        tracker.check_budget_before_request(est_prompt_tokens=40)

    # No phantom requests
    assert tracker.total_requests == 2
    assert global_tracker.total_requests == 2


def test_spend_post_response_enforcement_and_subsequent_request_prevention():
    """Verify spend is enforced post-response when completion tokens exceed limit, blocking subsequent requests."""
    from agentxray.providers.base import UsageMetadata

    # $100 per 1M input, $100 per 1M output ($0.0001 per 1 token)
    cfg = ExperimentBudgetConfig(
        max_model_requests=10,
        max_spend_usd=0.0010,
        model_name="fake-model",
        custom_pricing_per_1m=(100.0, 100.0),
    )
    tracker = CostTracker(cfg)

    # Pre-request check: 4 prompt tokens ($0.0004 USD < $0.0010 USD) passes
    tracker.check_budget_before_request(est_prompt_tokens=4)

    # Response returns 4 prompt tokens + 10 completion tokens = 14 tokens ($0.0014 USD > $0.0010 USD)
    usage = UsageMetadata(prompt_tokens=4, completion_tokens=10, total_tokens=14)

    # Post-response enforcement: record_usage records the incurred spend and raises CostBudgetExceededError
    with pytest.raises(CostBudgetExceededError, match="Estimated spend budget exhausted"):
        tracker.record_usage(usage)

    assert tracker.total_requests == 1
    assert tracker.estimated_spend_usd == 0.0014

    # Any subsequent request with projected tokens is blocked pre-flight before sending
    with pytest.raises(CostBudgetExceededError, match="Estimated spend budget would be exceeded before request"):
        tracker.check_budget_before_request(est_prompt_tokens=4)

    # Even a request with 0 estimated prompt tokens is blocked because current spend >= max_spend_usd
    with pytest.raises(CostBudgetExceededError, match="Estimated spend budget exhausted before request"):
        tracker.check_budget_before_request(est_prompt_tokens=0)

    assert tracker.total_requests == 1  # No phantom requests attempted


def test_failed_request_global_accounting_and_subsequent_call_prevention():
    """Demonstrate local exhaustion cannot prevent global accounting and global exhaustion halts further calls."""
    global_cfg = ExperimentBudgetConfig(
        max_model_requests=5,
        max_spend_usd=10.0,
        model_name="fake-model",
    )
    global_tracker = CostTracker(global_cfg)

    # Local tracker 1 has quota 2
    algo1_cfg = ExperimentBudgetConfig(
        max_model_requests=2,
        max_spend_usd=10.0,
        model_name="fake-model",
    )
    t1 = IsolatedAlgorithmCostTracker(algo1_cfg, global_tracker=global_tracker, enforce_global_requests=True)

    # 1. t1 experiences a failure with 1 retry (2 requests attempted). Hits local cap (2 >= 2).
    with pytest.raises(CostBudgetExceededError, match="Model request budget exhausted: executed 2 requests"):
        t1.record_failed_request(retries=1)

    # Local exhaustion DID NOT prevent global accounting:
    assert t1.total_requests == 2
    assert t1.failed_requests == 2
    assert t1.retries == 1
    assert global_tracker.total_requests == 2
    assert global_tracker.failed_requests == 2
    assert global_tracker.retries == 1

    # Local tracker 1 is blocked pre-flight
    with pytest.raises(CostBudgetExceededError, match="Model request budget exhausted before request"):
        t1.check_budget_before_request()

    # 2. Local tracker 2 starts for next algorithm with quota 5, attached to the same global tracker
    algo2_cfg = ExperimentBudgetConfig(
        max_model_requests=5,
        max_spend_usd=10.0,
        model_name="fake-model",
    )
    t2 = IsolatedAlgorithmCostTracker(algo2_cfg, global_tracker=global_tracker, enforce_global_requests=True)

    # t2 pre-flight check passes initially (local 0 < 5, global 2 < 5)
    t2.check_budget_before_request()

    # t2 experiences a failed request with 2 retries (3 requests attempted).
    # Global tracker reaches 2 + 3 = 5 >= 5!
    with pytest.raises(CostBudgetExceededError, match="Model request budget exhausted: executed 5 requests"):
        t2.record_failed_request(retries=2)

    assert t2.total_requests == 3
    assert t2.failed_requests == 3
    assert t2.retries == 2
    assert global_tracker.total_requests == 5
    assert global_tracker.failed_requests == 5
    assert global_tracker.retries == 3

    # Now global exhaustion prevents ALL subsequent calls across all trackers pre-flight
    with pytest.raises(CostBudgetExceededError, match="Global benchmark model request budget exhausted before request"):
        t2.check_budget_before_request()

    # Counter remains at 5: no subsequent provider calls attempted
    assert global_tracker.total_requests == 5
    assert t2.total_requests == 3


def test_exact_boundary_final_answer_vs_tool_call_exhaustion():
    """Verify exact boundary behavior when total_requests reaches max_model_requests.

    1. Case A (final answer): when response has no tool calls and brings total_requests == max_model_requests,
       ControlledAgent preserves and returns its content with finish_reason='stop'. No extra request is issued.
    2. Case B (tool call): when response requests a tool call but remaining budget is 0,
       ControlledAgent reports budget_exhausted appropriately and blocks further turns.
    3. Safety Rule: in both cases, no further provider request can start after limit is reached.
    """
    from agentxray.providers.base import ModelResponse, UsageMetadata

    # Case A: Model gives a complete final answer on the single allowed request
    cfg_a = ExperimentBudgetConfig(max_model_requests=1, max_spend_usd=5.0, model_name="fake-model")
    tracker_a = CostTracker(cfg_a)
    provider_a = FakeProvider(custom_handler=lambda msgs, tools: ModelResponse(
        content="The answer is 42.",
        tool_calls=(),
        finish_reason="stop",
        model_identifier="fake-model",
        usage=UsageMetadata(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    ))
    agent_a = ControlledAgent(
        provider=provider_a,
        mock_env=MockToolEnvironment(),
        cost_tracker=tracker_a,
        max_turns=3,
    )
    trace_a = agent_a.run_session(system_prompt="", user_prompt="What is 40 + 2?")

    # Preserved final answer:
    assert trace_a.finish_reason == "stop"
    assert trace_a.final_response == "The answer is 42."
    assert trace_a.error_message is None
    assert tracker_a.total_requests == 1
    assert provider_a.request_count == 1  # Exactly 1 request issued

    # Safety rule: subsequent request is blocked pre-flight without sending extra provider request
    with pytest.raises(CostBudgetExceededError, match="budget exhausted before request"):
        tracker_a.check_budget_before_request()

    trace_a2 = agent_a.run_session(system_prompt="", user_prompt="Follow-up question")
    assert trace_a2.finish_reason == "budget_exhausted"
    assert trace_a2.final_response == ""
    assert provider_a.request_count == 1  # Still 1: no extra provider request was attempted!

    # Case B: Model requests tool call, but budget is 1 (so remaining budget for Turn 2 is zero)
    cfg_b = ExperimentBudgetConfig(max_model_requests=1, max_spend_usd=5.0, model_name="fake-model")
    tracker_b = CostTracker(cfg_b)
    provider_b = FakeProvider(model_name="fake-model")
    agent_b = ControlledAgent(
        provider=provider_b,
        mock_env=MockToolEnvironment(),
        cost_tracker=tracker_b,
        max_turns=3,
    )
    trace_b = agent_b.run_session(
        system_prompt="Always use tools.",
        user_prompt="Calculate 150 + 250 using calculator",
        tools=[CALCULATOR_TOOL_DEF],
    )

    # In Case B, exhaustion reporting is legitimate because Turn 2 cannot proceed
    assert trace_b.finish_reason == "budget_exhausted"
    assert trace_b.final_response == ""
    assert tracker_b.total_requests == 1
    assert provider_b.request_count == 1  # Exactly 1 request issued, no subsequent turns

    # Safety rule: subsequent request blocked pre-flight
    with pytest.raises(CostBudgetExceededError, match="budget exhausted before request"):
        tracker_b.check_budget_before_request()


def test_spend_cap_breach_on_final_text_response_halts_with_budget_exhausted():
    """Verify that a final text response exceeding max_spend_usd is NOT suppressed as successful.

    Even if tool_calls is empty, if estimated spend meets or exceeds the spend cap,
    ControlledAgent halts with finish_reason='budget_exhausted' and empty final_response.
    """
    from agentxray.providers.base import ModelResponse, UsageMetadata

    # $100 per 1M tokens = $0.0001 per token.
    # Cap is $0.0030 USD.
    cfg = ExperimentBudgetConfig(
        max_model_requests=5,
        max_spend_usd=0.0030,
        custom_pricing_per_1m=(100.0, 100.0),
        model_name="fake-model",
    )
    tracker = CostTracker(cfg)
    # Pre-flight for "Generate response" (17 chars // 4 + 20 = 24 tokens) projects $0.0024 < $0.0030 USD (passes).
    # Actual response returns 24 prompt tokens + 20 completion tokens = 44 tokens ($0.0044 USD >= $0.0030 USD).
    provider = FakeProvider(
        custom_handler=lambda msgs, tools: ModelResponse(
            content="Expensive final answer",
            tool_calls=(),
            finish_reason="stop",
            model_identifier="fake-model",
            usage=UsageMetadata(prompt_tokens=24, completion_tokens=20, total_tokens=44),
        )
    )
    agent = ControlledAgent(
        provider=provider,
        mock_env=MockToolEnvironment(),
        cost_tracker=tracker,
        max_turns=3,
    )
    trace = agent.run_session(system_prompt="", user_prompt="Generate response")

    assert trace.finish_reason == "budget_exhausted"
    assert trace.final_response == ""
    assert "Estimated spend budget exhausted" in trace.error_message
    assert tracker.total_requests == 1
    assert provider.request_count == 1  # Request completed before spend breach was detected post-response
    assert tracker.estimated_spend_usd == 0.0044
    assert tracker.is_spend_exceeded is True

    # Pre-flight guard blocks further requests
    with pytest.raises(CostBudgetExceededError, match="Estimated spend budget exhausted before request"):
        tracker.check_budget_before_request(est_prompt_tokens=0)


def test_global_spend_cap_breach_on_final_text_response_halts_with_budget_exhausted():
    """Verify that an enforced global spend breach on a final text response halts with budget_exhausted.

    Even if local spend and request budget are well within limits, if global spend cap is
    breached by the response, ControlledAgent reports budget_exhausted.
    """
    from agentxray.providers.base import ModelResponse, UsageMetadata

    # $100 per 1M tokens = $0.0001 per token.
    # Global cap is $0.0050 USD.
    global_cfg = ExperimentBudgetConfig(
        max_model_requests=10,
        max_spend_usd=0.0050,
        custom_pricing_per_1m=(100.0, 100.0),
        model_name="fake-model",
    )
    global_tracker = CostTracker(global_cfg)

    algo_cfg = ExperimentBudgetConfig(
        max_model_requests=5,
        max_spend_usd=1.0,
        custom_pricing_per_1m=(100.0, 100.0),
        model_name="fake-model",
    )

    # Prior algorithm consumed $0.0020 USD (20 tokens)
    prior_usage = UsageMetadata(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    global_tracker.record_usage(prior_usage)
    assert global_tracker.estimated_spend_usd == 0.0020

    algo2_tracker = IsolatedAlgorithmCostTracker(
        algo_cfg,
        global_tracker=global_tracker,
        enforce_global_spend=True,
    )

    # Pre-flight for "Question" (8 chars // 4 + 20 = 22 tokens) projects $0.0022 USD.
    # Global spend $0.0020 + projected $0.0022 = $0.0042 < $0.0050 USD (passes pre-flight).
    # Next algorithm response: 20 prompt + 20 completion = 40 tokens ($0.0040 USD).
    # Global spend reaches $0.0020 + $0.0040 = $0.0060 >= $0.0050 USD.
    provider = FakeProvider(
        custom_handler=lambda msgs, tools: ModelResponse(
            content="Answer breaching global spend",
            tool_calls=(),
            finish_reason="stop",
            model_identifier="fake-model",
            usage=UsageMetadata(prompt_tokens=20, completion_tokens=20, total_tokens=40),
        )
    )
    agent = ControlledAgent(
        provider=provider,
        mock_env=MockToolEnvironment(),
        cost_tracker=algo2_tracker,
        max_turns=3,
    )
    trace = agent.run_session(system_prompt="", user_prompt="Question")

    assert trace.finish_reason == "budget_exhausted"
    assert trace.final_response == ""
    assert "spend budget exhausted" in trace.error_message
    assert algo2_tracker.total_requests == 1
    assert provider.request_count == 1  # Request completed before global spend breach was detected post-response
    assert global_tracker.total_requests == 2
    assert global_tracker.estimated_spend_usd == 0.0060
    assert algo2_tracker.is_spend_exceeded is True

    # Pre-flight guard blocks further requests
    with pytest.raises(CostBudgetExceededError, match="Global estimated spend budget"):
        algo2_tracker.check_budget_before_request(est_prompt_tokens=0)


def test_request_count_overflow_via_retries_halts_with_budget_exhausted():
    """Verify that request count strictly exceeding limit via retries halts with budget_exhausted.

    If max_model_requests is 1, but the provider call required 1 retry (2 attempted requests),
    total_requests strictly exceeds the limit (2 > 1). Even though the final answer succeeded
    without tool calls, ControlledAgent must not treat it as a valid under-budget completion.
    """
    from agentxray.providers.base import ModelResponse, UsageMetadata

    cfg = ExperimentBudgetConfig(
        max_model_requests=1,
        max_spend_usd=5.0,
        model_name="fake-model",
    )
    tracker = CostTracker(cfg)
    provider = FakeProvider(
        custom_handler=lambda msgs, tools: ModelResponse(
            content="Recovered answer after 1 retry",
            tool_calls=(),
            finish_reason="stop",
            model_identifier="fake-model",
            usage=UsageMetadata(prompt_tokens=10, completion_tokens=5, total_tokens=15, retries=1),
        )
    )
    agent = ControlledAgent(
        provider=provider,
        mock_env=MockToolEnvironment(),
        cost_tracker=tracker,
        max_turns=3,
    )
    trace = agent.run_session(system_prompt="", user_prompt="Compute something")

    assert trace.finish_reason == "budget_exhausted"
    assert trace.final_response == ""
    assert "Model request budget exhausted: executed 2 requests (limit: 1)" in trace.error_message
    assert tracker.total_requests == 2
    assert tracker.retries == 1
    assert tracker.is_request_cap_exceeded is True

    # Pre-flight guard blocks subsequent requests
    with pytest.raises(CostBudgetExceededError, match="Model request budget exhausted before request"):
        tracker.check_budget_before_request()


def test_enforced_global_request_overflow_halts_with_budget_exhausted():
    """Verify that enforced global request count strictly exceeding limit halts with budget_exhausted.

    When enforce_global_requests=True and global total_requests strictly exceeds global limit
    (e.g., via retries), ControlledAgent halts with finish_reason='budget_exhausted'.
    """
    from agentxray.providers.base import ModelResponse, UsageMetadata

    global_cfg = ExperimentBudgetConfig(
        max_model_requests=2,
        max_spend_usd=5.0,
        model_name="fake-model",
    )
    global_tracker = CostTracker(global_cfg)

    algo_cfg = ExperimentBudgetConfig(
        max_model_requests=5,
        max_spend_usd=5.0,
        model_name="fake-model",
    )

    # Algorithm 1 executes 1 request
    t1 = IsolatedAlgorithmCostTracker(algo_cfg, global_tracker=global_tracker, enforce_global_requests=True)
    t1.record_usage(UsageMetadata(prompt_tokens=10, completion_tokens=10, total_tokens=20))
    assert global_tracker.total_requests == 1

    # Algorithm 2 executes a call with 1 retry (2 attempted requests)
    # Global requests become 1 + 2 = 3 > 2 (strictly exceeded)
    t2 = IsolatedAlgorithmCostTracker(algo_cfg, global_tracker=global_tracker, enforce_global_requests=True)
    provider = FakeProvider(
        custom_handler=lambda msgs, tools: ModelResponse(
            content="Answer after retry breaching global request cap",
            tool_calls=(),
            finish_reason="stop",
            model_identifier="fake-model",
            usage=UsageMetadata(prompt_tokens=10, completion_tokens=5, total_tokens=15, retries=1),
        )
    )
    agent = ControlledAgent(
        provider=provider,
        mock_env=MockToolEnvironment(),
        cost_tracker=t2,
        max_turns=3,
    )
    trace = agent.run_session(system_prompt="", user_prompt="Compute something")

    assert trace.finish_reason == "budget_exhausted"
    assert trace.final_response == ""
    assert "Model request budget exhausted: executed 3 requests (limit: 2)" in trace.error_message
    assert t2.total_requests == 2
    assert global_tracker.total_requests == 3
    assert t2.is_request_cap_exceeded is True

    # Pre-flight guard blocks further requests
    with pytest.raises(CostBudgetExceededError, match="Global benchmark model request budget exhausted before request"):
        t2.check_budget_before_request()
