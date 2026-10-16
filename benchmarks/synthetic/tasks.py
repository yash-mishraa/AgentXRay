"""Benchmark task definitions with isolated synthetic targets and ground-truth specs."""

from __future__ import annotations

from agentxray.models import InputCase
from benchmarks.synthetic.ground_truth import GroundTruthSpec
from benchmarks.synthetic.target import SyntheticTarget


def create_single_factor_task() -> tuple[SyntheticTarget, InputCase, GroundTruthSpec]:
    """Task 1: Single-factor failure condition.

    Fails if and only if 'param_mode_debug' is present.
    """
    initial_elements = (
        "param_mode_debug",
        "config_timeout_30",
        "user_prompt_summarize",
        "tool_calculator",
    )

    def rule_fn(elements: set[str]) -> tuple[bool, str, str | None]:
        if "param_mode_debug" in elements:
            return (True, "Crash: Invalid debug mode configuration", "ERR_DEBUG_MODE")
        return (False, "Execution succeeded normally", None)

    target = SyntheticTarget(rule_fn=rule_fn, name="SingleFactorTarget")
    initial_input = InputCase(
        case_id="task_single_factor",
        elements=initial_elements,
        description="Single-factor failure task",
    )
    ground_truth = GroundTruthSpec(
        task_id="task_single_factor",
        name="Single-Factor Trigger",
        description="Fails whenever param_mode_debug is present in the input.",
        trigger_elements=("param_mode_debug",),
    )
    return (target, initial_input, ground_truth)


def create_conjunctive_task() -> tuple[SyntheticTarget, InputCase, GroundTruthSpec]:
    """Task 2: Two-factor conjunctive failure condition.

    Fails if and only if BOTH 'schema_type_strict' AND 'flag_allow_null' are present.
    """
    initial_elements = (
        "schema_type_strict",
        "flag_allow_null",
        "doc_max_tokens_100",
        "role_system_assistant",
        "query_search",
    )

    def rule_fn(elements: set[str]) -> tuple[bool, str, str | None]:
        if "schema_type_strict" in elements and "flag_allow_null" in elements:
            return (True, "TypeConflict: strict schema rejects null values", "ERR_TYPE_CONFLICT")
        return (False, "Execution succeeded normally", None)

    target = SyntheticTarget(rule_fn=rule_fn, name="ConjunctiveTarget")
    initial_input = InputCase(
        case_id="task_conjunctive",
        elements=initial_elements,
        description="Two-factor conjunctive failure task",
    )
    ground_truth = GroundTruthSpec(
        task_id="task_conjunctive",
        name="Two-Factor Conjunctive Trigger",
        description="Fails when both schema_type_strict AND flag_allow_null are present simultaneously.",
        trigger_elements=("schema_type_strict", "flag_allow_null"),
    )
    return (target, initial_input, ground_truth)


def create_distractor_task() -> tuple[SyntheticTarget, InputCase, GroundTruthSpec]:
    """Task 3: Distractor-heavy input with a single causal factor.

    Contains 4 irrelevant distractors and 1 causal factor 'token_auth_bearer'.
    """
    initial_elements = (
        "token_auth_bearer",
        "distractor_header_user_agent",
        "distractor_retry_count_3",
        "distractor_cache_ttl_60",
        "distractor_log_level_info",
    )

    def rule_fn(elements: set[str]) -> tuple[bool, str, str | None]:
        if "token_auth_bearer" in elements:
            return (True, "AuthError: Bearer token format malformed in payload", "ERR_AUTH_BEARER")
        return (False, "Execution succeeded normally", None)

    target = SyntheticTarget(rule_fn=rule_fn, name="DistractorHeavyTarget")
    initial_input = InputCase(
        case_id="task_distractor_heavy",
        elements=initial_elements,
        description="Distractor-heavy failure task",
    )
    ground_truth = GroundTruthSpec(
        task_id="task_distractor_heavy",
        name="Distractor-Heavy Trigger",
        description="Fails exclusively due to token_auth_bearer; all distractor_* elements are non-causal.",
        trigger_elements=("token_auth_bearer",),
    )
    return (target, initial_input, ground_truth)


def create_null_task() -> tuple[SyntheticTarget, InputCase, GroundTruthSpec]:
    """Task 4: Non-failing null case.

    Never fails. The engine must report no trigger found within the search space.
    """
    initial_elements = (
        "input_param_alpha",
        "input_param_beta",
        "input_param_gamma",
        "input_param_delta",
    )

    def rule_fn(elements: set[str]) -> tuple[bool, str, str | None]:
        return (False, "Execution succeeded without errors", None)

    target = SyntheticTarget(rule_fn=rule_fn, name="NullTarget")
    initial_input = InputCase(
        case_id="task_null_no_failure",
        elements=initial_elements,
        description="Non-failing null task",
    )
    ground_truth = GroundTruthSpec(
        task_id="task_null_no_failure",
        name="Non-Failing Null Case",
        description="Target executes successfully under all configurations; no failure trigger exists.",
        trigger_elements=(),
        is_null_case=True,
    )
    return (target, initial_input, ground_truth)


def get_all_benchmark_tasks() -> list[tuple[SyntheticTarget, InputCase, GroundTruthSpec]]:
    """Return all four standard benchmark tasks."""
    return [
        create_single_factor_task(),
        create_conjunctive_task(),
        create_distractor_task(),
        create_null_task(),
    ]
