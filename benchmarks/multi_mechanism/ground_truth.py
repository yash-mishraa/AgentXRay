"""Ground-truth causal specifications for Phase 4.6 multi-mechanism benchmark.

CRITICAL EPISTEMIC INTEGRITY GUARANTEE:
This module contains planted ground-truth causal rules for post-hoc evaluation.
Neither the minimizers, baseline algorithms, nor the LLM judge may import or inspect
any definitions in this file during search or diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MultiMechanismGroundTruthSpec:
    """Isolated ground-truth causal specification for post-hoc benchmark scoring."""

    mechanism_id: str
    name: str
    category: str
    is_null_case: bool
    is_conjunctive: bool
    trigger_elements: tuple[str, ...]
    distractor_elements: tuple[str, ...]
    description: str = ""

    def evaluate_candidate(self, candidate: Sequence[str]) -> tuple[bool, float, float]:
        """Evaluate a candidate subset against ground truth.

        Returns
        -------
        tuple[bool, float, float]
            (exact_match, precision, recall)
        """
        cand_set = set(candidate)
        true_set = set(self.trigger_elements)

        if self.is_null_case:
            # Null task: correct iff candidate is empty
            exact = len(cand_set) == 0
            prec = 1.0 if exact else 0.0
            rec = 1.0 if exact else 0.0
            return (exact, prec, rec)

        exact = cand_set == true_set
        if not cand_set:
            return (False, 0.0, 0.0)

        tp = len(cand_set & true_set)
        prec = tp / len(cand_set) if cand_set else 0.0
        rec = tp / len(true_set) if true_set else 0.0
        return (exact, round(prec, 4), round(rec, 4))
