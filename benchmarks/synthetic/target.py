"""Synthetic black-box target implementation.

IMPORTANT: This is a synthetic deterministic testbed designed exclusively
to verify the search mechanics of delta debugging and baseline algorithms.
It does NOT simulate a neural Large Language Model and does NOT prove
real-world reverse-engineering capability against commercial AI systems.
"""

from __future__ import annotations

import time
from typing import Callable

from agentxray.models import InputCase, TargetOutcome, TargetResult

RuleCallable = Callable[[set[str]], tuple[bool, str, str | None]]


class SyntheticTarget:
    """An opaque deterministic target with a planted internal rule.

    The probing engine interacts strictly via the public `execute()` method.
    The internal rule callable is private and must not be accessed or inspected
    by the probing or minimization algorithms.
    """

    def __init__(self, rule_fn: RuleCallable, name: str = "SyntheticTarget") -> None:
        self.name = name
        self._rule_fn = rule_fn
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Total number of times execute() has been called."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the execution counter."""
        self._call_count = 0

    def execute(self, input_case: InputCase) -> TargetResult:
        """Execute the input case against the opaque target logic.

        Parameters
        ----------
        input_case : InputCase
            The candidate input configuration to evaluate.

        Returns
        -------
        TargetResult
            Observable failure or success outcome with message and latency.
        """
        start = time.perf_counter()
        self._call_count += 1
        elements_set = set(input_case.elements)

        is_failure, message, error_code = self._rule_fn(elements_set)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        outcome = TargetOutcome.FAILURE if is_failure else TargetOutcome.SUCCESS

        return TargetResult(
            outcome=outcome,
            output_message=message,
            error_code=error_code,
            latency_ms=round(elapsed_ms, 3),
            metadata={"target_name": self.name, "element_count": len(input_case.elements)},
        )
