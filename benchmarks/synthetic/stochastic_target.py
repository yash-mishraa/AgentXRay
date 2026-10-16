"""Stochastic synthetic black-box target implementation.

Provides probabilistic execution targets for evaluating delta-debugging
minimization and baseline algorithms under binomial outcome uncertainty.

IMPORTANT: This is a synthetic stochastic testbed designed exclusively to
evaluate statistical sample size, power, and uncertainty bounds. It does NOT
simulate a real Large Language Model or API-backed agent system.
"""

from __future__ import annotations

import random
import time
from typing import Callable

from agentxray.models import InputCase, TargetOutcome, TargetResult

StochasticRuleCallable = Callable[[set[str]], float]


class StochasticSyntheticTarget:
    """An opaque probabilistic target governed by an internal probability function.

    The probing engine interacts strictly through the public `execute()` method.
    The internal probability function and seed are isolated from the search engine.
    """

    def __init__(
        self,
        rule_fn: StochasticRuleCallable,
        name: str = "StochasticSyntheticTarget",
        target_seed: int = 42,
    ) -> None:
        self.name = name
        self._rule_fn = rule_fn
        self._target_seed = target_seed
        self._rng = random.Random(target_seed)
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Total number of times execute() has been called."""
        return self._call_count

    def reset(self, new_seed: int | None = None) -> None:
        """Reset execution counter and reseed the internal PRNG."""
        self._call_count = 0
        if new_seed is not None:
            self._target_seed = new_seed
        self._rng = random.Random(self._target_seed)

    def execute(self, input_case: InputCase) -> TargetResult:
        """Execute a single probabilistic trial for the given input case.

        Parameters
        ----------
        input_case : InputCase
            The candidate input configuration to evaluate.

        Returns
        -------
        TargetResult
            Observable failure or success outcome sampled from the underlying probability.
        """
        start = time.perf_counter()
        self._call_count += 1
        elements_set = set(input_case.elements)

        # Evaluate underlying failure probability P(failure | elements)
        failure_prob = self._rule_fn(elements_set)
        if not (0.0 <= failure_prob <= 1.0):
            raise ValueError(f"Rule function returned invalid probability: {failure_prob}")

        # Independent Bernoulli trial: success if u < failure_prob
        u = self._rng.random()
        is_failure = u < failure_prob

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        outcome = TargetOutcome.FAILURE if is_failure else TargetOutcome.SUCCESS

        msg = (
            f"Stochastic execution failed (P={failure_prob:.2f})"
            if is_failure
            else f"Stochastic execution succeeded (P={failure_prob:.2f})"
        )
        err = "ERR_STOCHASTIC_FAILURE" if is_failure else None

        return TargetResult(
            outcome=outcome,
            output_message=msg,
            error_code=err,
            latency_ms=round(elapsed_ms, 3),
            metadata={
                "target_name": self.name,
                "element_count": len(input_case.elements),
                "underlying_probability": failure_prob,
            },
        )
