"""Stochastic synthetic black-box target implementation.

Provides probabilistic execution targets for evaluating delta-debugging
minimization and baseline algorithms under binomial outcome uncertainty.

IMPORTANT: This is a synthetic stochastic testbed designed exclusively to
evaluate statistical sample size, power, and uncertainty bounds. It does NOT
simulate a real Large Language Model or API-backed agent system.
"""

from __future__ import annotations

import hashlib
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
        crn_mode: bool = False,
    ) -> None:
        self.name = name
        self._rule_fn = rule_fn
        self._target_seed = target_seed
        self._crn_mode = crn_mode
        self._rng = random.Random(target_seed)
        self._call_count = 0
        self._subset_replicate_counts: dict[tuple[str, ...], int] = {}

    @property
    def call_count(self) -> int:
        """Total number of times execute() has been called."""
        return self._call_count

    @property
    def crn_mode(self) -> bool:
        """Whether Common Random Numbers mode is enabled."""
        return self._crn_mode

    def reset(self, new_seed: int | None = None, crn_mode: bool | None = None) -> None:
        """Reset execution counter and reseed the internal PRNG."""
        self._call_count = 0
        self._subset_replicate_counts.clear()
        if new_seed is not None:
            self._target_seed = new_seed
        if crn_mode is not None:
            self._crn_mode = crn_mode
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
        canonical_subset = tuple(sorted(input_case.elements))

        # Evaluate underlying failure probability P(failure | elements)
        failure_prob = self._rule_fn(elements_set)
        if not (0.0 <= failure_prob <= 1.0):
            raise ValueError(f"Rule function returned invalid probability: {failure_prob}")

        if self._crn_mode:
            rep_idx = self._subset_replicate_counts.get(canonical_subset, 0)
            self._subset_replicate_counts[canonical_subset] = rep_idx + 1
            key = f"{self._target_seed}:{','.join(canonical_subset)}:{rep_idx}".encode("utf-8")
            h = hashlib.sha256(key).digest()
            int_val = int.from_bytes(h[:8], "big") >> 11
            u = int_val / (1 << 53)
        else:
            rep_idx = self._call_count
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
                "replicate_index": rep_idx,
                "crn_mode": self._crn_mode,
            },
        )
