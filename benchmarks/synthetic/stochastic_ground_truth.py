"""Ground-truth specifications and post-hoc evaluation logic for stochastic benchmarks.

CRITICAL ARCHITECTURAL BOUNDARY:
This module contains ground-truth failure probability rules and post-hoc evaluation functions.
These specifications are accessed ONLY by the benchmark evaluator AFTER the probing
and delta-debugging algorithms have completed their search.
The probing engine must NEVER import or reference this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agentxray.models import StochasticBaselineComparison, StochasticDiscoveredCondition


@dataclass(frozen=True)
class StochasticGroundTruthSpec:
    """Specification of planted ground-truth failure rules for a stochastic task."""

    task_id: str
    name: str
    description: str
    trigger_elements: tuple[str, ...]
    base_probability: float
    trigger_probability: float
    is_null_case: bool = False
    alternative_valid_triggers: tuple[tuple[str, ...], ...] = ()

    @property
    def trigger_set(self) -> set[str]:
        return set(self.trigger_elements)


def evaluate_stochastic_ground_truth(
    spec: StochasticGroundTruthSpec,
    discovered: StochasticDiscoveredCondition | StochasticBaselineComparison,
) -> tuple[bool, float, float, tuple[str, ...], tuple[str, ...]]:
    """Compare a stochastic discovered condition against the planted ground truth.

    Returns
    -------
    tuple:
        matches_ground_truth (bool): Exact set equality with primary or alternative valid triggers.
        precision (float): True positives / (True positives + False positives).
        recall (float): True positives / (True positives + False negatives).
        excess_elements (tuple[str, ...]): Non-causal elements falsely retained.
        missing_elements (tuple[str, ...]): True causal elements missed.
    """
    if hasattr(discovered, "subset_elements"):
        discovered_elements = discovered.subset_elements
        is_failure = discovered.is_failure
    elif hasattr(discovered, "discovered_elements"):
        discovered_elements = discovered.discovered_elements
        is_failure = discovered.failure_reproduced
    else:
        discovered_elements = tuple(discovered)
        is_failure = True

    discovered_set = set(discovered_elements)
    true_set = spec.trigger_set

    # Null case handling
    if spec.is_null_case:
        if not is_failure and not discovered_set:
            return (True, 1.0, 1.0, (), ())
        return (False, 0.0, 0.0, tuple(sorted(discovered_set)), ())

    if not is_failure or not discovered_set:
        return (False, 0.0, 0.0, (), tuple(sorted(true_set)))

    # Check primary trigger match
    if discovered_set == true_set:
        return (True, 1.0, 1.0, (), ())

    # Check alternative valid triggers (e.g. for non-monotonic tasks with multiple minimal triggers)
    for alt in spec.alternative_valid_triggers:
        alt_set = set(alt)
        if discovered_set == alt_set:
            return (True, 1.0, 1.0, (), ())

    # Compute precision and recall against primary trigger set
    true_positives = len(discovered_set & true_set)
    false_positives = len(discovered_set - true_set)
    false_negatives = len(true_set - discovered_set)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    excess = tuple(sorted(discovered_set - true_set))
    missing = tuple(sorted(true_set - discovered_set))

    return (False, round(precision, 4), round(recall, 4), excess, missing)
