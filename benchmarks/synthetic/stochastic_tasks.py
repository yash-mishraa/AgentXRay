"""Stochastic benchmark task definitions with isolated synthetic targets and ground-truth specs."""

from __future__ import annotations

from agentxray.models import InputCase
from benchmarks.synthetic.stochastic_ground_truth import StochasticGroundTruthSpec
from benchmarks.synthetic.stochastic_target import StochasticSyntheticTarget


def create_stochastic_single_factor_task(
    target_seed: int = 42,
    p_trigger: float = 0.85,
    p_base: float = 0.05,
) -> tuple[StochasticSyntheticTarget, InputCase, StochasticGroundTruthSpec]:
    """Task 1: Single-factor probabilistic failure condition.

    Elevates failure probability from p_base (0.05) to p_trigger (0.85) when
    'param_mode_debug' is present.
    """
    initial_elements = (
        "param_mode_debug",
        "config_timeout_30",
        "user_prompt_summarize",
        "tool_calculator",
    )

    def rule_fn(elements: set[str]) -> float:
        if "param_mode_debug" in elements:
            return p_trigger
        return p_base

    target = StochasticSyntheticTarget(
        rule_fn=rule_fn,
        name="StochasticSingleFactorTarget",
        target_seed=target_seed,
    )
    initial_input = InputCase(
        case_id="stochastic_single_factor",
        elements=initial_elements,
        description="Probabilistic single-factor failure task",
    )
    ground_truth = StochasticGroundTruthSpec(
        task_id="stochastic_single_factor",
        name="Stochastic Single-Factor Trigger",
        description=f"Induces failure with P={p_trigger:.2f} when param_mode_debug is present (base P={p_base:.2f}).",
        trigger_elements=("param_mode_debug",),
        base_probability=p_base,
        trigger_probability=p_trigger,
    )
    return (target, initial_input, ground_truth)


def create_stochastic_conjunctive_task(
    target_seed: int = 42,
    p_trigger: float = 0.85,
    p_base: float = 0.05,
) -> tuple[StochasticSyntheticTarget, InputCase, StochasticGroundTruthSpec]:
    """Task 2: Two-factor conjunctive probabilistic failure condition.

    Elevates failure probability from p_base (0.05) to p_trigger (0.85) if and only if
    BOTH 'schema_type_strict' AND 'flag_allow_null' are present simultaneously.
    """
    initial_elements = (
        "schema_type_strict",
        "flag_allow_null",
        "doc_max_tokens_100",
        "role_system_assistant",
        "query_search",
    )

    def rule_fn(elements: set[str]) -> float:
        if "schema_type_strict" in elements and "flag_allow_null" in elements:
            return p_trigger
        return p_base

    target = StochasticSyntheticTarget(
        rule_fn=rule_fn,
        name="StochasticConjunctiveTarget",
        target_seed=target_seed,
    )
    initial_input = InputCase(
        case_id="stochastic_conjunctive",
        elements=initial_elements,
        description="Probabilistic two-factor conjunctive failure task",
    )
    ground_truth = StochasticGroundTruthSpec(
        task_id="stochastic_conjunctive",
        name="Stochastic Conjunctive Trigger",
        description=f"Induces failure with P={p_trigger:.2f} when both schema_type_strict AND flag_allow_null are present (base P={p_base:.2f}).",
        trigger_elements=("schema_type_strict", "flag_allow_null"),
        base_probability=p_base,
        trigger_probability=p_trigger,
    )
    return (target, initial_input, ground_truth)


def create_stochastic_distractor_task(
    target_seed: int = 42,
    p_trigger: float = 0.85,
    p_base: float = 0.05,
) -> tuple[StochasticSyntheticTarget, InputCase, StochasticGroundTruthSpec]:
    """Task 3: Distractor-heavy input with a single probabilistic causal factor.

    Contains 4 irrelevant distractors and 1 causal factor 'token_auth_bearer'.
    """
    initial_elements = (
        "token_auth_bearer",
        "distractor_header_user_agent",
        "distractor_retry_count_3",
        "distractor_cache_ttl_60",
        "distractor_log_level_info",
    )

    def rule_fn(elements: set[str]) -> float:
        if "token_auth_bearer" in elements:
            return p_trigger
        return p_base

    target = StochasticSyntheticTarget(
        rule_fn=rule_fn,
        name="StochasticDistractorHeavyTarget",
        target_seed=target_seed,
    )
    initial_input = InputCase(
        case_id="stochastic_distractor_heavy",
        elements=initial_elements,
        description="Probabilistic distractor-heavy failure task",
    )
    ground_truth = StochasticGroundTruthSpec(
        task_id="stochastic_distractor_heavy",
        name="Stochastic Distractor-Heavy Trigger",
        description=f"Fails with P={p_trigger:.2f} exclusively when token_auth_bearer is present (base P={p_base:.2f}); 4 distractors are non-causal.",
        trigger_elements=("token_auth_bearer",),
        base_probability=p_base,
        trigger_probability=p_trigger,
    )
    return (target, initial_input, ground_truth)


def create_stochastic_null_task(
    target_seed: int = 42,
    p_base: float = 0.05,
) -> tuple[StochasticSyntheticTarget, InputCase, StochasticGroundTruthSpec]:
    """Task 4: Non-failing null case with background noise only.

    No candidate element changes the failure probability. It remains unconditionally at p_base.
    """
    initial_elements = (
        "input_param_alpha",
        "input_param_beta",
        "input_param_gamma",
        "input_param_delta",
    )

    def rule_fn(elements: set[str]) -> float:
        return p_base

    target = StochasticSyntheticTarget(
        rule_fn=rule_fn,
        name="StochasticNullTarget",
        target_seed=target_seed,
    )
    initial_input = InputCase(
        case_id="stochastic_null",
        elements=initial_elements,
        description="Probabilistic non-failing null task",
    )
    ground_truth = StochasticGroundTruthSpec(
        task_id="stochastic_null",
        name="Stochastic Null Case",
        description=f"Target has unconditional background failure rate P={p_base:.2f}; no candidate element alters failure probability.",
        trigger_elements=(),
        base_probability=p_base,
        trigger_probability=p_base,
        is_null_case=True,
    )
    return (target, initial_input, ground_truth)


def create_stochastic_non_monotonic_task(
    target_seed: int = 42,
    p_trigger: float = 0.85,
    p_suppressed: float = 0.10,
    p_base: float = 0.05,
) -> tuple[StochasticSyntheticTarget, InputCase, StochasticGroundTruthSpec]:
    """Task 5: Non-monotonic failure condition.

    - feature_flag_a alone: P(fail) = 0.85
    - feature_flag_a + suppressor_b: P(fail) = 0.10 (suppression)
    - feature_flag_a + suppressor_b + restorer_c: P(fail) = 0.85 (restoration)
    - otherwise: P(fail) = 0.05 (base)
    """
    initial_elements = (
        "feature_flag_a",
        "suppressor_b",
        "restorer_c",
        "distractor_env_staging",
    )

    def rule_fn(elements: set[str]) -> float:
        has_a = "feature_flag_a" in elements
        has_b = "suppressor_b" in elements
        has_c = "restorer_c" in elements

        if has_a and has_b and has_c:
            return p_trigger
        if has_a and has_b:
            return p_suppressed
        if has_a:
            return p_trigger
        return p_base

    target = StochasticSyntheticTarget(
        rule_fn=rule_fn,
        name="StochasticNonMonotonicTarget",
        target_seed=target_seed,
    )
    initial_input = InputCase(
        case_id="stochastic_non_monotonic",
        elements=initial_elements,
        description="Probabilistic non-monotonic failure task with suppression and restoration",
    )
    ground_truth = StochasticGroundTruthSpec(
        task_id="stochastic_non_monotonic",
        name="Stochastic Non-Monotonic Trigger",
        description="Fails when feature_flag_a is present (P=0.85), suppressed by suppressor_b (P=0.10), restored by restorer_c (P=0.85).",
        trigger_elements=("feature_flag_a",),
        base_probability=p_base,
        trigger_probability=p_trigger,
        alternative_valid_triggers=(("feature_flag_a", "suppressor_b", "restorer_c"),),
    )
    return (target, initial_input, ground_truth)


def create_stochastic_weak_signal_task(
    target_seed: int = 42,
    p_trigger: float = 0.35,
    p_base: float = 0.10,
) -> tuple[StochasticSyntheticTarget, InputCase, StochasticGroundTruthSpec]:
    """Task 6: Weak-signal failure condition.

    Elevates failure probability modestly from p_base (0.10) to p_trigger (0.35).
    Evaluates statistical sample size adequacy and power limits.
    """
    initial_elements = (
        "weak_trigger_prompt",
        "distractor_context_1",
        "distractor_context_2",
        "distractor_context_3",
    )

    def rule_fn(elements: set[str]) -> float:
        if "weak_trigger_prompt" in elements:
            return p_trigger
        return p_base

    target = StochasticSyntheticTarget(
        rule_fn=rule_fn,
        name="StochasticWeakSignalTarget",
        target_seed=target_seed,
    )
    initial_input = InputCase(
        case_id="stochastic_weak_signal",
        elements=initial_elements,
        description="Probabilistic weak-signal failure task with modest effect size",
    )
    ground_truth = StochasticGroundTruthSpec(
        task_id="stochastic_weak_signal",
        name="Stochastic Weak-Signal Trigger",
        description=f"Weak trigger elevating failure rate from base P={p_base:.2f} to P={p_trigger:.2f} (delta = {p_trigger - p_base:.2f}).",
        trigger_elements=("weak_trigger_prompt",),
        base_probability=p_base,
        trigger_probability=p_trigger,
    )
    return (target, initial_input, ground_truth)


def get_all_stochastic_benchmark_tasks(
    target_seed: int = 42,
) -> list[tuple[StochasticSyntheticTarget, InputCase, StochasticGroundTruthSpec]]:
    """Return all six standard stochastic benchmark tasks."""
    return [
        create_stochastic_single_factor_task(target_seed=target_seed),
        create_stochastic_conjunctive_task(target_seed=target_seed),
        create_stochastic_distractor_task(target_seed=target_seed),
        create_stochastic_null_task(target_seed=target_seed),
        create_stochastic_non_monotonic_task(target_seed=target_seed),
        create_stochastic_weak_signal_task(target_seed=target_seed),
    ]
