"""1-Minimal Delta-Debugging Algorithm.

Implements an adaptation of Zeller's ddmin algorithm for discrete input configurations,
with explicit 1-minimality verification and conservative non-monotonic handling.
"""

from __future__ import annotations

from typing import Sequence

from agentxray.models import (
    DiscoveredCondition,
    InputCase,
    SearchStatus,
    TargetOutcome,
)
from agentxray.runner import BudgetExhaustedError, InterventionRunner, TimeoutExceededError


def split_into_chunks(elements: tuple[str, ...], n: int) -> list[tuple[str, ...]]:
    """Divide elements into n pairwise disjoint chunks of roughly equal size."""
    n = max(1, min(n, len(elements)))
    chunk_size = len(elements) // n
    remainder = len(elements) % n

    chunks: list[tuple[str, ...]] = []
    start = 0
    for i in range(n):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(elements[start:end])
        start = end
    return [c for c in chunks if c]


class DeltaDebugMinimizer:
    """Discovers a 1-minimal failure-inducing subset using delta debugging."""

    def __init__(self, runner: InterventionRunner) -> None:
        self.runner = runner

    def minimize(self, initial_input: InputCase) -> DiscoveredCondition:
        """Find a 1-minimal subset of initial_input that reproduces target failure.

        Parameters
        ----------
        initial_input : InputCase
            The full input configuration suspected of inducing failure.

        Returns
        -------
        DiscoveredCondition
            The isolated condition, verification status, and call counts.
        """
        # Step 1: Verify baseline failure
        try:
            baseline_result = self.runner.execute_trial(
                input_case=initial_input,
                intervention_type="baseline",
            )
        except (BudgetExhaustedError, TimeoutExceededError) as exc:
            return DiscoveredCondition(
                subset_elements=initial_input.elements,
                status=SearchStatus.BUDGET_EXHAUSTED if isinstance(exc, BudgetExhaustedError) else SearchStatus.TIMEOUT,
                is_failure=False,
                is_one_minimal=False,
                is_globally_minimal=False,
                verified_by_check=False,
                target_calls_used=self.runner.trial_count,
                status_message=f"Search halted during baseline check: {exc}",
            )

        if baseline_result.outcome != TargetOutcome.FAILURE:
            return DiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.NO_TRIGGER_FOUND,
                is_failure=False,
                is_one_minimal=False,
                is_globally_minimal=False,
                verified_by_check=False,
                target_calls_used=self.runner.trial_count,
                status_message="Initial input did not produce a failure on the target; no trigger to minimize.",
            )

        current_candidate: tuple[str, ...] = initial_input.elements
        n = 2

        # Step 2: Hierarchical partitioning loop (ddmin)
        try:
            while len(current_candidate) >= 2:
                chunks = split_into_chunks(current_candidate, n)
                reduced = False

                # 2A: Test individual partition subsets
                for chunk in chunks:
                    if len(chunk) < len(current_candidate):
                        sub_case = initial_input.subset(chunk)
                        res = self.runner.execute_trial(
                            input_case=sub_case,
                            intervention_type="partition_subset",
                        )
                        if res.outcome == TargetOutcome.FAILURE:
                            current_candidate = chunk
                            n = 2
                            reduced = True
                            break

                if reduced:
                    continue

                # 2B: Test partition complements
                for chunk in chunks:
                    complement = tuple(e for e in current_candidate if e not in set(chunk))
                    if complement and len(complement) < len(current_candidate):
                        comp_case = initial_input.subset(complement)
                        res = self.runner.execute_trial(
                            input_case=comp_case,
                            intervention_type="partition_complement",
                        )
                        if res.outcome == TargetOutcome.FAILURE:
                            current_candidate = complement
                            n = max(n - 1, 2)
                            reduced = True
                            break

                if reduced:
                    continue

                # 2C: Increase granularity if no reduction occurred
                if n < len(current_candidate):
                    n = min(n * 2, len(current_candidate))
                else:
                    # Granularity cannot be increased further
                    break

            # Step 3: Explicit 1-minimality verification and refinement
            # Test removing each remaining element one by one.
            # If removing any element STILL preserves failure, reduce to that smaller subset!
            verified_minimal = False
            while not verified_minimal:
                if len(current_candidate) == 0:
                    # Target fails unconditionally even on empty input
                    verified_minimal = False
                    break

                unnecessary_element_found = False
                for elem in current_candidate:
                    candidate_without_elem = tuple(e for e in current_candidate if e != elem)
                    sub_case = initial_input.subset(candidate_without_elem)
                    res = self.runner.execute_trial(
                        input_case=sub_case,
                        intervention_type="minimality_verification",
                    )
                    if res.outcome == TargetOutcome.FAILURE:
                        # Non-monotonic or further reducible: dropping elem still fails!
                        current_candidate = candidate_without_elem
                        unnecessary_element_found = True
                        break  # restart 1-minimality loop on the smaller candidate

                if not unnecessary_element_found:
                    verified_minimal = True

            # If size is 1 and verified minimal, it is globally minimal among non-empty subsets
            is_global = (len(current_candidate) == 1 and verified_minimal)

            return DiscoveredCondition(
                subset_elements=current_candidate,
                status=SearchStatus.COMPLETE,
                is_failure=True,
                is_one_minimal=verified_minimal,
                is_globally_minimal=is_global,
                verified_by_check=verified_minimal,
                target_calls_used=self.runner.trial_count,
                status_message=(
                    "Successfully discovered and verified 1-minimal failure-inducing subset."
                    if verified_minimal
                    else "Target fails unconditionally even on empty input; condition is not 1-minimal."
                ),
            )

        except (BudgetExhaustedError, TimeoutExceededError) as exc:
            return DiscoveredCondition(
                subset_elements=current_candidate,
                status=SearchStatus.BUDGET_EXHAUSTED if isinstance(exc, BudgetExhaustedError) else SearchStatus.TIMEOUT,
                is_failure=True,
                is_one_minimal=False,
                is_globally_minimal=False,
                verified_by_check=False,
                target_calls_used=self.runner.trial_count,
                status_message=f"Search interrupted before full 1-minimality verification: {exc}",
            )
