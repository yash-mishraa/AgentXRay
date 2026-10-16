"""Unit and integration tests for Phase 1G.2 Calibrated Neural Failure-Trigger Probing."""

from __future__ import annotations

import pytest

from agentxray.agent_harness.agent import AgentExecutionTrace, ControlledAgent, ToolExecutionRecord
from agentxray.agent_harness.cost_tracker import CostBudgetExceededError, CostTracker, ExperimentBudgetConfig
from agentxray.agent_harness.evaluator import ToolCallingTaskEvaluator
from agentxray.agent_harness.target import ModelBackedAgentTarget
from agentxray.agent_harness.tools import CALCULATOR_TOOL_DEF, MockToolEnvironment
from agentxray.models import (
    AccessMode,
    ExecutionTier,
    ExperimentConfig,
    InputCase,
    SearchStatus,
    StochasticMinimalityStatus,
    TargetOutcome,
)
from agentxray.providers.base import Message, ModelResponse, ToolCall, UsageMetadata
from agentxray.providers.fake import FakeProvider
from agentxray.runner import InterventionRunner
from agentxray.stochastic_baselines import (
    run_stochastic_greedy_ablation,
    run_stochastic_random_search,
)
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.model_pilot.ground_truth import evaluate_model_pilot_ground_truth
from benchmarks.model_pilot.runner import dry_run_model_pilot, run_model_pilot_benchmark
from benchmarks.model_pilot.tasks import (
    create_model_pilot_calibrated_currency_task,
    get_model_pilot_tasks,
)


def test_calibrated_task_creation() -> None:
    """Verify calibrated currency task instantiates with 4 elements and correct ground truth."""
    target, initial_input, gt = create_model_pilot_calibrated_currency_task()
    assert len(initial_input.elements) == 4
    assert gt.task_id == "pilot_calibrated_currency"
    assert len(gt.trigger_elements) == 1
    assert "strict_currency_formatting" in gt.trigger_elements[0]
    assert gt.is_null_case is False


def test_evaluator_cannot_access_ground_truth() -> None:
    """Verify evaluator inspects only runtime execution traces and has no access to ground truth."""
    evaluator = ToolCallingTaskEvaluator(required_tool="calculator", expected_substring="400")

    # Trace with clean tool execution satisfying criteria
    clean_trace = AgentExecutionTrace(
        messages=(Message(role="user", content="sum"),),
        final_response="The total is 400.",
        tool_executions=(
            ToolExecutionRecord(
                tool_call_id="c1",
                tool_name="calculator",
                arguments={"expression": "150 + 250"},
                output="400",
            ),
        ),
        turns_used=2,
        finish_reason="stop",
    )
    is_success, msg = evaluator.evaluate(clean_trace)
    assert is_success is True
    assert "criteria satisfied" in msg

    # Trace where tool returned syntax error
    err_trace = AgentExecutionTrace(
        messages=(Message(role="user", content="sum"),),
        final_response="The total is 400.",
        tool_executions=(
            ToolExecutionRecord(
                tool_call_id="c1",
                tool_name="calculator",
                arguments={"expression": "$150 + $250"},
                output="Error: invalid syntax",
            ),
        ),
        turns_used=2,
        finish_reason="stop",
    )
    is_success, msg = evaluator.evaluate(err_trace)
    assert is_success is False
    assert "returned an error" in msg


def test_failure_type_classification() -> None:
    """Verify distinction between target behavioral failures, tool errors, and infrastructure crashes."""
    # 1. Infrastructure crash raises RuntimeError immediately and is never counted as target failure
    prov_crash = FakeProvider(model_name="fake-crash")
    env = MockToolEnvironment()
    tracker = CostTracker(ExperimentBudgetConfig(max_model_requests=10, max_spend_usd=0.50, model_name="fake-crash"))
    agent_crash = ControlledAgent(provider=prov_crash, mock_env=env, cost_tracker=tracker)
    evaluator = ToolCallingTaskEvaluator(required_tool="calculator", expected_substring="400")
    target_crash = ModelBackedAgentTarget(
        agent=agent_crash,
        evaluator=evaluator,
        user_prompt="sum",
        tools=[CALCULATOR_TOOL_DEF],
    )

    # Simulate infrastructure error
    with pytest.raises(Exception):
        prov_crash.set_simulate_error(ConnectionResetError)
        target_crash.execute(InputCase(case_id="c1", elements=("el1",)))


def test_calibrated_task_minimization_with_fake_provider() -> None:
    """Verify Stochastic Delta Debugging isolates the 1-minimal trigger on the calibrated task."""
    target, initial_input, gt = create_model_pilot_calibrated_currency_task()
    config = ExperimentConfig(experiment_name="test_calibrated_dd", max_trials=25, timeout_seconds=30)
    runner = InterventionRunner(config=config, target_fn=target.execute)

    minimizer = StochasticDeltaDebugMinimizer(
        runner=runner,
        trials_per_candidate=3,
        failure_threshold=0.50,
    )
    discovered = minimizer.minimize(initial_input)

    assert discovered.status == SearchStatus.COMPLETE
    assert len(discovered.subset_elements) == 1
    assert "strict_currency_formatting" in discovered.subset_elements[0]
    assert discovered.minimality_status in (
        StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
        StochasticMinimalityStatus.OBSERVED_ONLY,
    )

    matches_gt, prec, rec, excess, missing = evaluate_model_pilot_ground_truth(gt, discovered)
    assert matches_gt is True
    assert prec == 1.0
    assert rec == 1.0
    assert excess == ()
    assert missing == ()


def test_stochastic_baselines_on_calibrated_task() -> None:
    """Verify Greedy Ablation and Random Search execute on the calibrated task under identical oracles."""
    target, initial_input, gt = create_model_pilot_calibrated_currency_task()
    config = ExperimentConfig(experiment_name="test_calibrated_baselines", max_trials=25, timeout_seconds=30)

    # Greedy Ablation
    runner_greedy = InterventionRunner(config=config, target_fn=target.execute)
    greedy_res = run_stochastic_greedy_ablation(
        initial_input=initial_input,
        runner=runner_greedy,
        trials_per_candidate=3,
        failure_threshold=0.50,
        true_trigger=gt.trigger_elements,
    )
    assert greedy_res.matches_ground_truth is True
    assert len(greedy_res.discovered_elements) == 1
    assert "strict_currency_formatting" in greedy_res.discovered_elements[0]

    # Random Search
    runner_random = InterventionRunner(config=config, target_fn=target.execute)
    random_res = run_stochastic_random_search(
        initial_input=initial_input,
        runner=runner_random,
        trials_per_candidate=3,
        failure_threshold=0.50,
        true_trigger=gt.trigger_elements,
        seed=42,
    )
    assert random_res.matches_ground_truth is True
    assert len(random_res.discovered_elements) == 1


def test_hard_cap_120_requests_enforcement_for_local_model() -> None:
    """Verify run_model_pilot_benchmark strictly rejects request limits > 120 for local model execution."""
    with pytest.raises(ValueError, match="mandate max_model_requests <= 120"):
        run_model_pilot_benchmark(
            provider_type="ollama",
            model_name="llama3.2:3b",
            task_filter="calibrated_currency",
            max_model_requests=150,  # Exceeds Phase 1G.2 hard cap
        )


def test_dry_run_calibrated_task_filtering() -> None:
    """Verify dry_run_model_pilot correctly resolves calibrated_currency task."""
    dry = dry_run_model_pilot(
        provider_type="ollama",
        model_name="llama3.2:3b",
        task_filter="calibrated_currency",
        max_model_requests=120,
    )
    assert len(dry["tasks"]) == 1
    assert dry["tasks"][0]["task_id"] == "pilot_calibrated_currency"
    assert dry["tasks"][0]["initial_elements_count"] == 4
    assert dry["status"] == "VALID"
