"""Ground-truth specifications and post-hoc evaluation logic.

CRITICAL ARCHITECTURAL BOUNDARY:
This module contains ground-truth failure rule specifications and evaluation functions.
These specifications are accessed ONLY by the benchmark evaluator AFTER the probing
and delta-debugging algorithms have completed their search.
The probing engine must NEVER import or reference this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agentxray.models import BaselineComparison, DiscoveredCondition


@dataclass(frozen=True)
class GroundTruthSpec:
    """Specification of the planted ground truth for a benchmark task."""

    task_id: str
    name: str
    description: str
    trigger_elements: tuple[str, ...]
    is_null_case: bool = False
    alternative_valid_triggers: tuple[tuple[str, ...], ...] = ()

    @property
    def trigger_set(self) -> set[str]:
        return set(self.trigger_elements)


def evaluate_ground_truth(
    spec: GroundTruthSpec | None,
    discovered: DiscoveredCondition | BaselineComparison,
) -> tuple[bool, float, float, tuple[str, ...], tuple[str, ...]]:
    """Compare a discovered condition or baseline comparison against planted ground truth.

    Returns
    -------
    tuple:
        matches_ground_truth (bool): Exact set equality with primary or alternative valid triggers.
        precision (float): True positives / (True positives + False positives).
        recall (float): True positives / (True positives + False negatives).
        excess_elements (tuple[str, ...]): Distractors falsely reported as part of trigger.
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

    # Missing ground truth handling (never matches)
    if spec is None:
        return (False, 0.0, 0.0, tuple(sorted(discovered_set)), ())

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

    # Check alternative valid triggers
    for alt in spec.alternative_valid_triggers:
        if discovered_set == set(alt):
            return (True, 1.0, 1.0, (), ())

    true_positives = len(discovered_set & true_set)
    false_positives = len(discovered_set - true_set)
    false_negatives = len(true_set - discovered_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    excess = tuple(sorted(discovered_set - true_set))
    missing = tuple(sorted(true_set - discovered_set))
    matches = (discovered_set == true_set)

    return (matches, round(precision, 4), round(recall, 4), excess, missing)
