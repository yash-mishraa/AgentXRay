"""Public Python SDK Facade for AgentXRay.

Provides a clean, typed, ergonomic interface for diagnosing and minimizing
failure-inducing input subsets in tool-calling AI agents without requiring
knowledge of research internals.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Callable, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from agentxray.minimizer import DeltaDebugMinimizer
from agentxray.models import (
    ExperimentConfig,
    InputCase,
    ProbeTrial,
    SearchStatus,
    TargetOutcome,
    TargetResult,
)
from agentxray.runner import BudgetExhaustedError, InterventionRunner, TimeoutExceededError


class DiagnosticStatus(str, Enum):
    """Machine-readable outcome status for a diagnostic run."""

    VERIFIED_CULPRIT = "VERIFIED_CULPRIT"  # Minimal failure-inducing subset isolated and verified
    NO_FAILURE = "NO_FAILURE"              # Full candidate set passed; no failure to minimize
    INCONCLUSIVE = "INCONCLUSIVE"          # Evaluator returned inconclusive or search halted
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"  # Evaluation budget ran out before minimality verification
    TIMEOUT = "TIMEOUT"                    # Timeout exceeded during diagnosis
    ERROR = "ERROR"                        # Evaluator or infrastructure raised an unexpected exception


# Evaluator return types supported
EvaluatorReturn = Union[
    bool,
    TargetOutcome,
    TargetResult,
    tuple[bool, str],
    None,
]


class DiagnosticResult(BaseModel):
    """Authoritative structured result from an AgentXRay diagnostic run.

    Provides high-level ergonomics (`summary`, `is_verified`, `isolated_candidates`)
    alongside full telemetry and candidate mapping for developer inspection.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=False)

    status: DiagnosticStatus
    summary: str
    is_verified: bool
    is_one_minimal: bool
    isolated_candidates: tuple[str, ...] = Field(default_factory=tuple)
    isolated_candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    candidate_map: dict[str, str] = Field(default_factory=dict)
    initial_candidates_count: int
    evaluations_used: int
    execution_time_seconds: float
    algorithm: str = "ddmin"
    seed: int = 42
    matches_ground_truth: bool | None = None
    error_message: str | None = None
    trials: tuple[ProbeTrial, ...] = Field(default_factory=tuple)


def diagnose(
    candidates: Sequence[str] | None = None,
    evaluator: Callable[[tuple[str, ...]], EvaluatorReturn] | None = None,
    *,
    elements: Sequence[str] | None = None,
    max_evaluations: int = 50,
    timeout_seconds: float = 60.0,
    seed: int = 42,
    ground_truth: Sequence[str] | None = None,
    algorithm: str = "ddmin",
) -> DiagnosticResult:
    """Diagnose and minimize the causal subset of candidate directives inducing agent failure.

    Args:
        candidates: Sequence of string directives, prompt elements, or configuration rules.
            Must be non-empty and contain unique strings. (Alias: `elements`).
        evaluator: Callable that receives an active subset of candidates as `tuple[str, ...]`.
            Must return one of:
            - `bool`: `True` if the agent fails on this subset, `False` if it passes cleanly.
            - `TargetOutcome`: `TargetOutcome.FAILURE`, `TargetOutcome.SUCCESS`, or `TargetOutcome.INCONCLUSIVE`.
            - `TargetResult`: full structured target outcome.
            - `tuple[bool, str]`: `(is_failure, diagnostic_message)`.
        elements: Alias for `candidates` (keyword-only).
        max_evaluations: Maximum number of target evaluations allowed (default: 50).
        timeout_seconds: Wall-clock timeout in seconds (default: 60.0).
        seed: Random seed for deterministic reproducibility (default: 42).
        ground_truth: Optional expected minimal failure subset for validation/testing.
            When omitted, `matches_ground_truth` remains `None`.
        algorithm: Minimization algorithm. Currently supports `"ddmin"` (default).

    Returns:
        DiagnosticResult: Typed, authoritative diagnostic outcome.

    Raises:
        ValueError: If candidates is empty, contains duplicates, or arguments are invalid.
        TypeError: If candidates are not strings or evaluator is not callable.
    """
    # 1. Resolve candidates vs elements
    candidate_seq = candidates if candidates is not None else elements
    if candidate_seq is None:
        raise ValueError("Either 'candidates' or 'elements' must be provided.")

    if not isinstance(candidate_seq, (list, tuple)):
        try:
            candidate_seq = list(candidate_seq)
        except Exception as exc:
            raise TypeError(
                f"'candidates' must be an iterable of strings, got {type(candidate_seq).__name__}"
            ) from exc

    if len(candidate_seq) == 0:
        raise ValueError("Candidate pool cannot be empty. Provide at least one candidate directive to diagnose.")

    validated_candidates: list[str] = []
    seen: set[str] = set()
    for idx, item in enumerate(candidate_seq):
        if not isinstance(item, str):
            raise TypeError(f"Candidate element at index {idx} must be a string, got {type(item).__name__}.")
        if item in seen:
            raise ValueError(f"Candidate elements must be unique. Found duplicate element: {item!r}")
        seen.add(item)
        validated_candidates.append(item)

    initial_candidates = tuple(validated_candidates)

    # 2. Validate evaluator
    if evaluator is None or not callable(evaluator):
        raise TypeError(f"'evaluator' must be callable, got {type(evaluator).__name__}.")

    # 3. Validate algorithm
    if algorithm.lower() != "ddmin":
        raise ValueError(f"Unsupported algorithm {algorithm!r}. Currently supported algorithms: 'ddmin'.")

    # 4. Validate budget parameters
    if max_evaluations <= 0:
        raise ValueError(f"'max_evaluations' must be a positive integer, got {max_evaluations}.")
    if timeout_seconds <= 0.0:
        raise ValueError(f"'timeout_seconds' must be positive, got {timeout_seconds}.")

    # 5. Validate ground_truth if provided
    gt_tuple: tuple[str, ...] | None = None
    if ground_truth is not None:
        if not isinstance(ground_truth, (list, tuple)):
            ground_truth = list(ground_truth)
        for gt_item in ground_truth:
            if not isinstance(gt_item, str):
                raise TypeError(f"Ground truth element must be a string, got {type(gt_item).__name__}.")
            if gt_item not in seen:
                raise ValueError(f"Ground truth element {gt_item!r} is not in the candidate pool.")
        gt_tuple = tuple(ground_truth)

    # 6. Build candidate index maps
    candidate_map = {f"elem_{idx}": elem for idx, elem in enumerate(initial_candidates)}
    reverse_map = {elem: f"elem_{idx}" for idx, elem in enumerate(initial_candidates)}

    # 7. Setup experiment runner and adapter
    config = ExperimentConfig(
        experiment_name="sdk_diagnosis",
        max_trials=max_evaluations,
        timeout_seconds=timeout_seconds,
        seed=seed,
    )

    t0 = time.perf_counter()
    evaluator_error: str | None = None
    evaluator_inconclusive: bool = False

    def target_fn(input_case: InputCase) -> TargetResult:
        nonlocal evaluator_error, evaluator_inconclusive
        try:
            raw_out = evaluator(input_case.elements)
        except Exception as exc:
            evaluator_error = f"{type(exc).__name__}: {exc}"
            return TargetResult(
                outcome=TargetOutcome.SUCCESS,
                output_message=evaluator_error,
                error_code="EVALUATOR_EXCEPTION",
            )

        if isinstance(raw_out, TargetResult):
            return raw_out
        elif isinstance(raw_out, TargetOutcome):
            return TargetResult(outcome=raw_out)
        elif isinstance(raw_out, bool):
            outcome = TargetOutcome.FAILURE if raw_out else TargetOutcome.SUCCESS
            msg = "Failure reproduced" if raw_out else "Clean pass"
            return TargetResult(outcome=outcome, output_message=msg)
        elif isinstance(raw_out, tuple) and len(raw_out) == 2 and isinstance(raw_out[0], bool):
            outcome = TargetOutcome.FAILURE if raw_out[0] else TargetOutcome.SUCCESS
            return TargetResult(outcome=outcome, output_message=str(raw_out[1]))
        elif raw_out is None:
            evaluator_inconclusive = True
            return TargetResult(
                outcome=TargetOutcome.SUCCESS,
                output_message="Evaluator returned inconclusive (None)",
                error_code="INCONCLUSIVE",
            )
        else:
            evaluator_error = f"Unsupported evaluator return type: {type(raw_out).__name__}"
            return TargetResult(
                outcome=TargetOutcome.SUCCESS,
                output_message=evaluator_error,
                error_code="INVALID_RETURN_TYPE",
            )

    runner = InterventionRunner(config=config, target_fn=target_fn)
    initial_input = InputCase(elements=initial_candidates)
    minimizer = DeltaDebugMinimizer(runner=runner)

    discovered = minimizer.minimize(initial_input)
    elapsed = time.perf_counter() - t0

    # 8. Resolve final diagnosis status and plain-English summary
    if evaluator_error is not None:
        status = DiagnosticStatus.ERROR
        summary = f"Diagnosis aborted due to an evaluator error: {evaluator_error}"
        is_verified = False
        isolated: tuple[str, ...] = ()
        isolated_ids: tuple[str, ...] = ()
    elif evaluator_inconclusive:
        status = DiagnosticStatus.INCONCLUSIVE
        summary = "Evaluator returned an inconclusive result. Could not establish definitive failure."
        is_verified = False
        isolated = ()
        isolated_ids = ()
    elif discovered.status == SearchStatus.NO_TRIGGER_FOUND:
        status = DiagnosticStatus.NO_FAILURE
        summary = f"Baseline execution passed cleanly across {runner.trial_count} evaluation(s). No failure condition to minimize."
        is_verified = False
        isolated = ()
        isolated_ids = ()
    elif discovered.status == SearchStatus.BUDGET_EXHAUSTED:
        status = DiagnosticStatus.BUDGET_EXHAUSTED
        summary = f"Evaluation budget of {max_evaluations} trials was exhausted before 1-minimality could be verified."
        is_verified = False
        isolated = discovered.subset_elements
        isolated_ids = tuple(reverse_map[e] for e in isolated if e in reverse_map)
    elif discovered.status == SearchStatus.TIMEOUT:
        status = DiagnosticStatus.TIMEOUT
        summary = f"Evaluation timed out after {elapsed:.2f} seconds."
        is_verified = False
        isolated = discovered.subset_elements
        isolated_ids = tuple(reverse_map[e] for e in isolated if e in reverse_map)
    else:
        # SearchStatus.COMPLETE
        isolated = discovered.subset_elements
        isolated_ids = tuple(reverse_map[e] for e in isolated if e in reverse_map)
        is_verified = discovered.is_one_minimal
        if is_verified:
            status = DiagnosticStatus.VERIFIED_CULPRIT
            id_str = ", ".join(isolated_ids)
            summary = (
                f"Isolated {len(isolated)} verified 1-minimal culprit directive(s) in {runner.trial_count} evaluations: "
                f"[{id_str}]. Removing {id_str} eliminated the failure."
            )
        else:
            status = DiagnosticStatus.INCONCLUSIVE
            summary = f"Isolated candidate subset [{', '.join(isolated_ids)}], but 1-minimality could not be verified: {discovered.status_message}"

    matches_gt: bool | None = None
    if gt_tuple is not None:
        matches_gt = (set(isolated) == set(gt_tuple))

    return DiagnosticResult(
        status=status,
        summary=summary,
        is_verified=is_verified,
        is_one_minimal=is_verified,
        isolated_candidates=isolated,
        isolated_candidate_ids=isolated_ids,
        candidate_map=candidate_map,
        initial_candidates_count=len(initial_candidates),
        evaluations_used=runner.trial_count,
        execution_time_seconds=elapsed,
        algorithm=algorithm,
        seed=seed,
        matches_ground_truth=matches_gt,
        error_message=evaluator_error,
        trials=tuple(runner.trials),
    )
