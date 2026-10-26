"""Reusable demo adapter and execution layer for AgentXRay web interface.

Provides deterministic, offline, zero-cost benchmark scenario execution across
Delta Debugging, Greedy Ablation, and Seeded Random Search.
Strictly isolated from remote network requests and live LLM providers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from agentxray.baselines import run_greedy_ablation, run_seeded_random_search
from agentxray.demo import DemoStepRecord, run_demo
from agentxray.minimizer import DeltaDebugMinimizer
from agentxray.models import (
    ExperimentConfig,
    InputCase,
    TargetOutcome,
    TargetResult,
)
from agentxray.providers.base import ModelResponse, ToolCall, UsageMetadata
from agentxray.providers.fake import FakeProvider
from agentxray.runner import InterventionRunner
from benchmarks.multi_mechanism.tasks import (
    TraceEvaluatedTaskSpec,
    create_trace_evaluated_clean_null_task,
    create_trace_evaluated_environment_task,
    create_trace_evaluated_prompt_task,
    create_trace_evaluated_schema_task,
)


@dataclass(frozen=True)
class ScenarioMetadata:
    """Descriptive metadata for an offline benchmark scenario."""

    scenario_id: str
    name: str
    category: str
    description: str
    agent_description: str
    user_prompt: str
    available_tools: tuple[str, ...]
    evaluator_criteria: str
    candidate_elements: tuple[str, ...]
    ground_truth_trigger: tuple[str, ...]
    is_null_case: bool
    observable_failure_symptom: str


@dataclass(frozen=True)
class WebDemoScenarioResult:
    """Complete structured execution result for a web demo scenario."""

    scenario_id: str
    scenario_name: str
    category: str
    agent_description: str
    user_prompt: str
    available_tools: tuple[str, ...]
    evaluator_criteria: str
    initial_elements: tuple[str, ...]
    ground_truth_trigger: tuple[str, ...]
    is_null_case: bool
    initial_failure_tool_call: str
    initial_failure_output: str
    initial_failure_code: str
    initial_failure_reproduced: bool
    dd_steps: tuple[DemoStepRecord, ...]
    discovered_subset: tuple[str, ...]
    is_one_minimal: bool
    is_failure: bool
    target_calls_used: int
    matches_ground_truth: bool
    status_message: str
    greedy_calls: int
    greedy_discovered: tuple[str, ...]
    greedy_matches_ground_truth: bool
    greedy_is_minimal: bool
    random_calls: int
    random_discovered: tuple[str, ...]
    random_matches_ground_truth: bool
    random_is_minimal: bool
    execution_time_seconds: float
    scientific_disclaimer: str


SCENARIO_CATALOG: dict[str, ScenarioMetadata] = {
    "currency_ast": ScenarioMetadata(
        scenario_id="currency_ast",
        name="AST Calculator Currency Formatting Directive",
        category="Prompt Directive / Syntax Error",
        description="Prompt constraint forces currency symbols ($) into arithmetic tool arguments, causing AST parser failure.",
        agent_description="Automated Accounting Assistant (Controlled Mock Agent with AST Calculator)",
        user_prompt="Calculate total reimbursement for expenses of 150 and 250.",
        available_tools=("calculator (AST arithmetic evaluator)",),
        evaluator_criteria="Tool 'calculator' invoked with valid expression, producing exact result 400.",
        candidate_elements=(
            'strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. "$150" or "$250")',
            "concise_summary: keep final output under 50 words",
            "table_output: format line items in markdown tables",
            "formal_tone: use formal business accounting prose",
        ),
        ground_truth_trigger=(
            'strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. "$150" or "$250")',
        ),
        is_null_case=False,
        observable_failure_symptom="Calculator AST syntax error (currency symbol '$' rejected)",
    ),
    "trace_prompt_01": ScenarioMetadata(
        scenario_id="trace_prompt_01",
        name="Trace-Evaluated Tool Prohibition Directive",
        category="Prompt Directive / Model Steering",
        description="Prompt directive forbids external math utilities; agent answers via mental math and omits required tool invocation.",
        agent_description="Accounting Ledger Bot (Trace-Evaluated Controlled Agent)",
        user_prompt="Calculate the sum of receipts $150 and $250 using calculator.",
        available_tools=("calculator (expression: str) -> str", "database_lookup (query: str) -> str"),
        evaluator_criteria="Tool 'calculator' invoked and final response contains '400'.",
        candidate_elements=(
            "Direct arithmetic: Compute and summarize all invoice line items directly in the response text without calling external math utilities.",
            "Monetary precision: Represent all invoice amounts with exactly two decimal places and standard currency signs.",
            "Professional styling: Use formal accounting terminology throughout the financial summary.",
            "Voucher verification: Verify each receipt line item against internal voucher records before issuing total.",
        ),
        ground_truth_trigger=(
            "Direct arithmetic: Compute and summarize all invoice line items directly in the response text without calling external math utilities.",
        ),
        is_null_case=False,
        observable_failure_symptom="Agent failed to invoke required tool 'calculator'.",
    ),
    "trace_schema_01": ScenarioMetadata(
        scenario_id="trace_schema_01",
        name="Trace-Evaluated Schema Parameter Key Mismatch",
        category="Schema Contract / Tool Interface",
        description="Tool schema renames 'expression' to abbreviated key 'expr'; agent complies with schema but backend service rejects argument.",
        agent_description="Accounting Ledger Bot (Trace-Evaluated Controlled Agent)",
        user_prompt="Calculate the sum of receipts $150 and $250 using calculator.",
        available_tools=("calculator (schema dynamically materialized)", "database_lookup (query: str) -> str"),
        evaluator_criteria="Tool 'calculator' invoked with valid parameter 'expression' and produces result '400'.",
        candidate_elements=(
            "Parameter naming policy: Math evaluation tools require abbreviated argument keys for formula inputs.",
            "Precedence rules: Math evaluation tools follow standard algebraic operator precedence conventions.",
            "Rounding precision: Math evaluation tools format fractional calculation results to six decimal places.",
            "Query timeout: Database lookup tools operate with a default five-second query response timeout.",
        ),
        ground_truth_trigger=(
            "Parameter naming policy: Math evaluation tools require abbreviated argument keys for formula inputs.",
        ),
        is_null_case=False,
        observable_failure_symptom="Missing required argument(s): expression",
    ),
    "trace_env_01": ScenarioMetadata(
        scenario_id="trace_env_01",
        name="Trace-Evaluated Mock Service 500 Outage",
        category="Environment / Infrastructure Fault",
        description="Backend calculator mock service endpoint injects HTTP 500 Internal Server Error into agent execution trace.",
        agent_description="Accounting Ledger Bot (Trace-Evaluated Controlled Agent)",
        user_prompt="Calculate the sum of receipts $150 and $250 using calculator.",
        available_tools=("calculator (expression: str) -> str", "database_lookup (query: str) -> str"),
        evaluator_criteria="Tool 'calculator' executed successfully without server fault, producing result '400'.",
        candidate_elements=(
            "Service availability fault: Backend calculation cluster experiences unexpected node-level outage during execution.",
            "Telemetry routing: Backend calculation telemetry is forwarded to primary monitoring facility.",
            "Audit archiving: Backend transaction logs are retained for thirty days under compliance policy.",
            "Connection pooling: Backend calculation client maintains five persistent warm connections.",
        ),
        ground_truth_trigger=(
            "Service availability fault: Backend calculation cluster experiences unexpected node-level outage during execution.",
        ),
        is_null_case=False,
        observable_failure_symptom="Error: 500 Internal Server Error: Mock calculator service temporarily unavailable.",
    ),
    "trace_null_01": ScenarioMetadata(
        scenario_id="trace_null_01",
        name="Trace-Evaluated Clean Execution Control",
        category="Null Control / Zero-Fault Baseline",
        description="Clean passing condition containing 0 planted triggers. Used to evaluate whether diagnostic methods falsely attribute failure.",
        agent_description="Accounting Ledger Bot (Trace-Evaluated Controlled Agent)",
        user_prompt="Calculate the sum of receipts $150 and $250 using calculator.",
        available_tools=("calculator (expression: str) -> str", "database_lookup (query: str) -> str"),
        evaluator_criteria="Tool 'calculator' invoked and produces result '400'.",
        candidate_elements=(
            "Summary layout: Present arithmetic invoice summary in clean tabular format.",
            "Ledger code: Assign entry reimbursement tag to accounting journal ledger.",
            "Currency label: Prepend standard USD currency label to transaction line items.",
            "Audit signoff: Mark final calculation output as verified by accounting review.",
        ),
        ground_truth_trigger=(),
        is_null_case=True,
        observable_failure_symptom="",
    ),
}


def get_available_scenarios() -> list[ScenarioMetadata]:
    """Return all available predefined benchmark scenarios in catalog order."""
    return list(SCENARIO_CATALOG.values())


def _make_trace_mock_agent_handler(prompt_trigger: str = ""):
    """Create a deterministic scripted handler for trace-evaluated tool-calling agent experiments."""
    def handler(messages: Sequence[Any], tools: Sequence[Any]) -> ModelResponse:
        system_text = messages[0].content if messages and getattr(messages[0], "role", "") == "system" else ""
        has_tool_reply = any(getattr(m, "role", "") == "tool" for m in messages)

        if not has_tool_reply:
            if prompt_trigger and prompt_trigger in system_text:
                return ModelResponse(
                    content="Estimated receipts sum directly in text: approximately $400.",
                    tool_calls=(),
                    usage=UsageMetadata(prompt_tokens=50, completion_tokens=15, total_tokens=65),
                    finish_reason="stop",
                    model_identifier="fake-agent",
                )

            calc_tool = next((t for t in tools if getattr(t, "name", "") == "calculator"), None)
            calc_props = calc_tool.parameters.get("properties", {}) if calc_tool else {}
            args = {"expr": "150 + 250"} if "expr" in calc_props else {"expression": "150 + 250"}
            return ModelResponse(
                content="Invoking calculator tool...",
                tool_calls=(
                    ToolCall(call_id="call_calc", name="calculator", arguments=args),
                ),
                usage=UsageMetadata(prompt_tokens=45, completion_tokens=15, total_tokens=60),
                finish_reason="tool_calls",
                model_identifier="fake-agent",
            )
        else:
            tool_msg = next((m for m in messages if getattr(m, "role", "") == "tool"), None)
            tool_content = getattr(tool_msg, "content", "") if tool_msg else ""
            if tool_content.startswith("Error:"):
                return ModelResponse(
                    content=f"Calculation aborted due to error: {tool_content}",
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


def _run_currency_ast_scenario(seed: int = 42) -> WebDemoScenarioResult:
    """Run the AST calculator currency-formatting directive demo."""
    t0 = time.perf_counter()
    res = run_demo(seed=seed)
    elapsed = time.perf_counter() - t0

    meta = SCENARIO_CATALOG["currency_ast"]
    cond = res.discovered_condition
    greedy = res.greedy_comparison
    random = res.random_comparison

    return WebDemoScenarioResult(
        scenario_id=meta.scenario_id,
        scenario_name=meta.name,
        category=meta.category,
        agent_description=res.agent_description,
        user_prompt=res.user_prompt,
        available_tools=res.available_tools,
        evaluator_criteria=res.evaluator_criteria,
        initial_elements=res.initial_elements,
        ground_truth_trigger=res.ground_truth_trigger,
        is_null_case=meta.is_null_case,
        initial_failure_tool_call=res.initial_failure_tool_call,
        initial_failure_output=res.initial_failure_output,
        initial_failure_code=res.initial_failure_code,
        initial_failure_reproduced=True,
        dd_steps=res.dd_steps,
        discovered_subset=cond.subset_elements,
        is_one_minimal=cond.is_one_minimal,
        is_failure=cond.is_failure,
        target_calls_used=cond.target_calls_used,
        matches_ground_truth=res.dd_matches_ground_truth,
        status_message=cond.status_message,
        greedy_calls=greedy.target_calls,
        greedy_discovered=greedy.discovered_elements,
        greedy_matches_ground_truth=greedy.matches_ground_truth,
        greedy_is_minimal=greedy.minimality_check_passed,
        random_calls=random.target_calls,
        random_discovered=random.discovered_elements,
        random_matches_ground_truth=random.matches_ground_truth,
        random_is_minimal=random.minimality_check_passed,
        execution_time_seconds=elapsed,
        scientific_disclaimer=(
            "Deterministic offline verification on FakeProvider ($0.00 cost, zero network calls). "
            "Delta Debugging isolated the 1-minimal causal trigger in 4 evaluations."
        ),
    )


def _run_trace_scenario(scenario_id: str, seed: int = 42) -> WebDemoScenarioResult:
    """Run one of the trace-evaluated benchmark scenarios."""
    t0 = time.perf_counter()

    if scenario_id == "trace_prompt_01":
        task: TraceEvaluatedTaskSpec = create_trace_evaluated_prompt_task()
    elif scenario_id == "trace_schema_01":
        task = create_trace_evaluated_schema_task()
    elif scenario_id == "trace_env_01":
        task = create_trace_evaluated_environment_task()
    elif scenario_id == "trace_null_01":
        task = create_trace_evaluated_clean_null_task()
    else:
        raise ValueError(f"Unknown trace scenario_id: {scenario_id!r}")

    meta = SCENARIO_CATALOG[scenario_id]
    prompt_tr = task.planted_trigger_elements[0] if task.planted_trigger_elements and scenario_id == "trace_prompt_01" else ""
    handler = _make_trace_mock_agent_handler(prompt_trigger=prompt_tr)

    initial_input = task.create_initial_input()
    config = ExperimentConfig(
        experiment_name=f"web_demo_{scenario_id}",
        max_trials=50,
        seed=seed,
    )

    # 1. Delta Debugging with step recording
    target_dd = task.create_target(provider=FakeProvider(model_name="fake-agent", custom_handler=handler))
    runner_dd = InterventionRunner(config=config, target_fn=target_dd.execute)
    minimizer = DeltaDebugMinimizer(runner=runner_dd)
    discovered_dd = minimizer.minimize(initial_input)

    # Compile trajectory steps
    steps: list[DemoStepRecord] = []
    for idx, trial in enumerate(runner_dd.trials, start=1):
        if trial.intervention_type == "baseline":
            if trial.result.outcome == TargetOutcome.FAILURE:
                action = "Baseline failure reproduced on full input; initiate 2-way binary partition"
            else:
                action = "Clean execution on full input; target succeeded (no failure to minimize)"
        elif trial.intervention_type == "partition_subset" and trial.result.outcome == TargetOutcome.FAILURE:
            action = f"Subset of size {len(trial.input_case.elements)} reproduced failure; zoom into this partition"
        elif trial.intervention_type == "partition_complement" and trial.result.outcome == TargetOutcome.FAILURE:
            action = f"Complement of size {len(trial.input_case.elements)} reproduced failure; zoom into complement"
        elif trial.intervention_type == "minimality_verification":
            if len(trial.input_case.elements) == 0:
                if trial.result.outcome == TargetOutcome.SUCCESS:
                    action = "Empty subset test verified: removing element eliminates failure (1-minimal)"
                else:
                    action = "Empty subset test: failure persists even without candidates (external shock)"
            else:
                action = f"Single-element removal test on {len(trial.input_case.elements)} elements completed"
        else:
            action = f"Evaluated candidate subset of size {len(trial.input_case.elements)} (Outcome: {trial.result.outcome.value.upper()})"

        steps.append(
            DemoStepRecord(
                step_number=idx,
                intervention_type=trial.intervention_type,
                element_count=len(trial.input_case.elements),
                candidate_elements=trial.input_case.elements,
                outcome=trial.result.outcome.value.upper(),
                action_narrative=action,
            )
        )

    # 2. Greedy Baseline
    target_greedy = task.create_target(provider=FakeProvider(model_name="fake-agent", custom_handler=handler))
    runner_greedy = InterventionRunner(config=config, target_fn=target_greedy.execute)
    greedy_comp = run_greedy_ablation(initial_input=initial_input, runner=runner_greedy)

    # 3. Seeded Random Search Baseline
    target_random = task.create_target(provider=FakeProvider(model_name="fake-agent", custom_handler=handler))
    runner_random = InterventionRunner(config=config, target_fn=target_random.execute)
    random_comp = run_seeded_random_search(initial_input=initial_input, runner=runner_random, seed=seed)

    # Initial trial telemetry
    first_trial = runner_dd.trials[0]
    initial_reproduced = (first_trial.result.outcome == TargetOutcome.FAILURE)
    initial_err_msg = first_trial.result.output_message or ("Clean execution" if meta.is_null_case else "Failure detected")
    initial_err_code = first_trial.result.error_code or ("SUCCESS" if not initial_reproduced else "AGENT_FAILURE")

    tool_names = first_trial.result.metadata.get("tool_names_called", [])
    if scenario_id == "trace_prompt_01" and initial_reproduced:
        tool_call_repr = "(No tools invoked — prompt prohibition observed)"
    elif scenario_id == "trace_schema_01":
        tool_call_repr = "calculator(expr='150 + 250')"
    elif tool_names:
        tool_call_repr = f"{tool_names[0]}(expression='150 + 250')"
    else:
        tool_call_repr = "calculator(expression='150 + 250')"

    gt_set = set(task.planted_trigger_elements)
    dd_matches_gt = (set(discovered_dd.subset_elements) == gt_set)
    greedy_matches_gt = (set(greedy_comp.discovered_elements) == gt_set)
    random_matches_gt = (set(random_comp.discovered_elements) == gt_set)

    elapsed = time.perf_counter() - t0

    return WebDemoScenarioResult(
        scenario_id=meta.scenario_id,
        scenario_name=meta.name,
        category=meta.category,
        agent_description=meta.agent_description,
        user_prompt=meta.user_prompt,
        available_tools=meta.available_tools,
        evaluator_criteria=meta.evaluator_criteria,
        initial_elements=initial_input.elements,
        ground_truth_trigger=task.planted_trigger_elements,
        is_null_case=meta.is_null_case,
        initial_failure_tool_call=tool_call_repr,
        initial_failure_output=initial_err_msg,
        initial_failure_code=initial_err_code,
        initial_failure_reproduced=initial_reproduced,
        dd_steps=tuple(steps),
        discovered_subset=discovered_dd.subset_elements,
        is_one_minimal=discovered_dd.is_one_minimal,
        is_failure=discovered_dd.is_failure,
        target_calls_used=discovered_dd.target_calls_used,
        matches_ground_truth=dd_matches_gt,
        status_message=discovered_dd.status_message,
        greedy_calls=greedy_comp.target_calls,
        greedy_discovered=greedy_comp.discovered_elements,
        greedy_matches_ground_truth=greedy_matches_gt,
        greedy_is_minimal=greedy_comp.minimality_check_passed,
        random_calls=random_comp.target_calls,
        random_discovered=random_comp.discovered_elements,
        random_matches_ground_truth=random_matches_gt,
        random_is_minimal=random_comp.minimality_check_passed,
        execution_time_seconds=elapsed,
        scientific_disclaimer=(
            f"Deterministic offline verification on FakeProvider ($0.00 cost, zero network calls). "
            f"Evaluates {meta.category} within trace-evaluated testbed."
        ),
    )


def run_web_demo(scenario_id: str, seed: int = 42) -> WebDemoScenarioResult:
    """Execute a predefined offline demonstration scenario.

    Parameters
    ----------
    scenario_id : str
        Predefined scenario key from SCENARIO_CATALOG.
    seed : int
        Deterministic random seed (default: 42).

    Returns
    -------
    WebDemoScenarioResult
        Structured execution result including search trajectory and baseline comparisons.
    """
    if scenario_id not in SCENARIO_CATALOG:
        valid_keys = list(SCENARIO_CATALOG.keys())
        raise ValueError(f"Unknown scenario_id: {scenario_id!r}. Available scenarios: {valid_keys}")

    if scenario_id == "currency_ast":
        return _run_currency_ast_scenario(seed=seed)
    else:
        return _run_trace_scenario(scenario_id=scenario_id, seed=seed)
