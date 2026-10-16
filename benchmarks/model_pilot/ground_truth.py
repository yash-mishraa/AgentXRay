"""Ground-truth specifications and post-hoc evaluation for model-backed pilot tasks.

CRITICAL ARCHITECTURAL BOUNDARY:
This module contains ground-truth causal rules for the model pilot.
These specifications are accessed ONLY by post-hoc benchmark evaluation AFTER the
probing and minimization algorithms have completed their search.
The probing engine, minimizers, runners, and baselines must NEVER import or inspect this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agentxray.models import StochasticDiscoveredCondition


@dataclass(frozen=True)
class ModelPilotGroundTruthSpec:
    """Planted causal intervention rule for model-backed pilot tasks."""

    task_id: str
    name: str
    description: str
    trigger_elements: tuple[str, ...]
    is_null_case: bool = False
    alternative_valid_triggers: tuple[tuple[str, ...], ...] = ()

    @property
    def trigger_set(self) -> set[str]:
        return set(self.trigger_elements)


def evaluate_model_pilot_ground_truth(
    spec: ModelPilotGroundTruthSpec,
    discovered: StochasticDiscoveredCondition,
) -> tuple[bool, float, float, tuple[str, ...], tuple[str, ...]]:
    """Compare a discovered condition from model-backed probing against planted ground truth.

    Returns
    -------
    tuple:
        matches_ground_truth (bool): Exact match with primary or alternative valid triggers.
        precision (float): True positives / (True positives + False positives).
        recall (float): True positives / (True positives + False negatives).
        excess_elements (tuple[str, ...]): Non-causal elements falsely retained.
        missing_elements (tuple[str, ...]): True causal elements missed.
    """
    discovered_set = set(discovered.subset_elements)
    true_set = spec.trigger_set

    # Null case handling
    if spec.is_null_case:
        if not discovered.is_failure and not discovered_set:
            return (True, 1.0, 1.0, (), ())
        return (False, 0.0, 0.0, tuple(sorted(discovered_set)), ())

    if not discovered.is_failure or not discovered_set:
        return (False, 0.0, 0.0, (), tuple(sorted(true_set)))

    # Primary trigger match
    if discovered_set == true_set:
        return (True, 1.0, 1.0, (), ())

    # Check alternative valid triggers
    for alt in spec.alternative_valid_triggers:
        if discovered_set == set(alt):
            return (True, 1.0, 1.0, (), ())

    tp = len(discovered_set & true_set)
    fp = len(discovered_set - true_set)
    fn = len(true_set - discovered_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    excess = tuple(sorted(discovered_set - true_set))
    missing = tuple(sorted(true_set - discovered_set))

    return (False, precision, recall, excess, missing)
