"""Controlled intervention runner with budget and timeout enforcement."""

from __future__ import annotations

import time
from typing import Callable
from uuid import uuid4

from agentxray.models import ExperimentConfig, InputCase, ProbeTrial, TargetResult


class RunnerException(Exception):
    """Base exception for intervention runner."""


class BudgetExhaustedError(RunnerException):
    """Raised when the maximum trial budget has been reached."""


class TimeoutExceededError(RunnerException):
    """Raised when execution duration exceeds the configured wall-clock timeout."""


class InterventionRunner:
    """Manages controlled execution of target trials under strict budget constraints.

    Ensures every target invocation is recorded as an immutable ProbeTrial with
    unique identifiers, intervention categorization, and provenance.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        target_fn: Callable[[InputCase], TargetResult],
        on_candidate_evaluated: Callable[..., None] | None = None,
    ) -> None:
        self.config = config
        self._target_fn = target_fn
        self.on_candidate_evaluated = on_candidate_evaluated
        self._trials: list[ProbeTrial] = []
        self._start_time = time.perf_counter()

    @property
    def trials(self) -> list[ProbeTrial]:
        """List of all recorded probe trials in execution order."""
        return list(self._trials)

    @property
    def trial_count(self) -> int:
        """Total number of executed trials."""
        return len(self._trials)

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed wall-clock time since runner initialization."""
        return time.perf_counter() - self._start_time

    def is_budget_exhausted(self) -> bool:
        """Check if trial count cap has been reached."""
        return len(self._trials) >= self.config.max_trials

    def is_timed_out(self) -> bool:
        """Check if wall-clock timeout has been exceeded."""
        return self.elapsed_seconds >= self.config.timeout_seconds

    def execute_trial(
        self,
        input_case: InputCase,
        intervention_type: str,
        parent_trial_id: str | None = None,
    ) -> TargetResult:
        """Execute a single controlled trial against the target.

        Parameters
        ----------
        input_case : InputCase
            The candidate input configuration to evaluate.
        intervention_type : str
            Category of intervention (e.g., 'baseline', 'partition', 'ablation').
        parent_trial_id : str | None
            UUID of the parent trial from which this intervention branched.

        Returns
        -------
        TargetResult
            Observable response from the target.

        Raises
        ------
        BudgetExhaustedError
            If trial limit is exceeded.
        TimeoutExceededError
            If wall-clock timeout is exceeded.
        """
        if self.is_timed_out():
            raise TimeoutExceededError(
                f"Experiment {self.config.experiment_name!r} timed out after {self.elapsed_seconds:.2f}s "
                f"(limit: {self.config.timeout_seconds}s)"
            )

        if self.is_budget_exhausted():
            raise BudgetExhaustedError(
                f"Experiment {self.config.experiment_name!r} reached maximum trial budget of {self.config.max_trials}"
            )

        trial_index = len(self._trials)
        result = self._target_fn(input_case)

        trial = ProbeTrial(
            trial_id=str(uuid4()),
            experiment_name=self.config.experiment_name,
            trial_index=trial_index,
            parent_trial_id=parent_trial_id,
            intervention_type=intervention_type,
            input_case=input_case,
            result=result,
        )
        self._trials.append(trial)
        return result
