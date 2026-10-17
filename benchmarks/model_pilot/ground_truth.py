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

from agentxray.models import (
    BaselineComparison,
    DiscoveredCondition,
    StochasticBaselineComparison,
    StochasticDiscoveredCondition,
)


@dataclass(frozen=True)
class ModelPilotGroundTruthSpec:
    """Independently specified reference condition or intervention hypothesis.

    EPISTEMIC BOUNDARY:
    In synthetic benchmarks with FakeProvider, ground-truth triggers are programmed
    deterministically. In real-model experiments, the reference trigger represents an
    independently specified intervention hypothesis. It is considered an empirical
    ground truth only if the experimental setup genuinely controls it and independently
    verifies its effect against a baseline control condition.
    """

    task_id: str
    name: str
    description: str
    trigger_elements: tuple[str, ...]
    is_null_case: bool = False
    alternative_valid_triggers: tuple[tuple[str, ...], ...] = ()
    reference_type: str = "planted_reference_hypothesis"

    @property
    def trigger_set(self) -> set[str]:
        return set(self.trigger_elements)


def evaluate_model_pilot_ground_truth(
    spec: ModelPilotGroundTruthSpec,
    discovered: (
        StochasticDiscoveredCondition
        | StochasticBaselineComparison
        | DiscoveredCondition
        | BaselineComparison
        | Sequence[str]
        | set[str]
    ),
) -> tuple[bool, float | None, float | None, tuple[str, ...], tuple[str, ...]]:
    """Compare a discovered condition from model-backed probing against planted ground truth.

    Returns
    -------
    tuple:
        matches_ground_truth (bool): Exact match with primary or alternative valid triggers.
        precision (float | None): True positives / (True positives + False positives), or None for null cases.
        recall (float | None): True positives / (True positives + False negatives), or None for null cases.
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

    # Null case handling
    if spec.is_null_case:
        if not discovered_set:
            # Both predicted set and true trigger set are empty -> precision and recall are undefined (None).
            # Outcome correctness / exact match is True iff no failure occurred.
            return (not is_failure, None, None, (), ())
        # Non-empty false discovery on null task:
        # Precision is 0/|D| = 0.0, while recall is undefined (None) as true trigger is empty.
        return (False, 0.0, None, tuple(sorted(discovered_set)), ())

    true_set = spec.trigger_set

    if not is_failure or not discovered_set:
        return (False, 0.0, 0.0, (), tuple(sorted(true_set)))

    candidates = (true_set,) + tuple(set(alt) for alt in spec.alternative_valid_triggers)

    # 1. Exact match check (primary or any alternative valid trigger)
    for cand in candidates:
        if discovered_set == cand:
            return (True, 1.0, 1.0, (), ())

    # 2. Non-exact match: evaluate against primary and all alternative valid triggers.
    # Select the reference trigger that maximizes F1 score with deterministic tie-breaking:
    #   1. Highest F1 score
    #   2. Highest Recall (more causal elements recovered)
    #   3. Smallest reference trigger size (|T|, parsimony)
    #   4. Candidate order (primary trigger index 0 first, then alternatives in order of appearance)
    best_cand: set[str] | None = None
    best_metrics: tuple[float, float, int, int] | None = None
    best_prec = 0.0
    best_rec = 0.0

    for idx, cand in enumerate(candidates):
        tp = len(discovered_set & cand)
        fp = len(discovered_set - cand)
        fn = len(cand - discovered_set)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        sort_key = (f1, rec, -len(cand), -idx)
        if best_metrics is None or sort_key > best_metrics:
            best_metrics = sort_key
            best_cand = cand
            best_prec = prec
            best_rec = rec

    assert best_cand is not None

    excess = tuple(sorted(discovered_set - best_cand))
    missing = tuple(sorted(best_cand - discovered_set))

    return (False, best_prec, best_rec, excess, missing)
