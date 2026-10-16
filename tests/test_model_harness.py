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

