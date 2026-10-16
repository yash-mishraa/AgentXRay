"""Stochastic baseline comparison algorithms: Greedy Ablation and Seeded Random Search."""

from __future__ import annotations

import random
from typing import Sequence

from agentxray.models import (
    CandidateEvaluation,
    InputCase,
    StochasticBaselineComparison,
    StochasticMinimalityStatus,
)
from agentxray.runner import BudgetExhaustedError, InterventionRunner, TimeoutExceededError
from agentxray.statistics import (
    adjust_p_values_bh,
    compare_proportions,
)
from agentxray.stochastic_minimizer import evaluate_candidate_stochastically


def _classify_stochastic_minimality(
    candidate: tuple[str, ...],
    cand_eval: CandidateEvaluation,
    removal_evals: dict[str, CandidateEvaluation],
    failure_threshold: float = 0.30,
    alpha: float = 0.05,
    confidence: float = 0.95,
    alternative: str = "two-sided",
) -> StochasticMinimalityStatus:
    """Helper to classify stochastic minimality for baseline results."""
    if not candidate:
        return StochasticMinimalityStatus.NOT_APPLICABLE

    missing = set(candidate) - set(removal_evals.keys())
    if missing:
        return StochasticMinimalityStatus.INCONCLUSIVE

    # Check if removing any element still met the failure threshold
    if any(rem.failure_rate >= failure_threshold for rem in removal_evals.values()):
        return StochasticMinimalityStatus.NOT_MINIMAL

    p_values: list[float] = []
    observed_drops: list[bool] = []

    for elem in candidate:
        rem_eval = removal_evals[elem]
        comp = compare_proportions(
            k1=cand_eval.failures,
            n1=cand_eval.trials,
            k2=rem_eval.failures,
            n2=rem_eval.trials,
            alpha=alpha,
            confidence=confidence,
            alternative=alternative,
        )
        p_values.append(comp.p_value if alternative == "two-sided" else comp.p_value_one_sided)
        observed_drops.append(cand_eval.failure_rate > rem_eval.failure_rate)

    adjusted = adjust_p_values_bh(p_values)
    if all(q < alpha for q in adjusted):
        return StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
    if all(observed_drops):
        return StochasticMinimalityStatus.OBSERVED_ONLY
    return StochasticMinimalityStatus.NOT_MINIMAL


def run_stochastic_greedy_ablation(
    initial_input: InputCase,
    runner: InterventionRunner,
    trials_per_candidate: int = 20,
    failure_threshold: float = 0.30,
    alpha: float = 0.05,
    confidence: float = 0.95,
    alternative: str = "two-sided",
    true_trigger: tuple[str, ...] = (),
    alternative_valid_triggers: tuple[tuple[str, ...], ...] = (),
) -> StochasticBaselineComparison:
    """Stochastic greedy single-element ablation baseline.

    Sequentially attempts to remove each element from the candidate set one-by-one.
    If the subset without element 'e' still reproduces failure rate >= failure_threshold,
    'e' is permanently dropped.
    """
    true_set = set(true_trigger)
    alt_sets = [set(alt) for alt in alternative_valid_triggers]

    # Step 1: Baseline check
    try:
        base_eval = evaluate_candidate_stochastically(
            runner=runner,
            input_case=initial_input,
            trials=trials_per_candidate,
            intervention_type="stochastic_greedy_baseline",
            confidence=confidence,
        )
    except (BudgetExhaustedError, TimeoutExceededError):
        return StochasticBaselineComparison(
            baseline_name="stochastic_greedy_ablation",
            discovered_elements=initial_input.elements,
            failure_reproduced=False,
            minimality_status=StochasticMinimalityStatus.INCONCLUSIVE,
            target_calls=runner.trial_count,
            candidate_evaluation=CandidateEvaluation(
                candidate_elements=initial_input.elements,
                trials=0,
                failures=0,
                failure_rate=0.0,
                confidence_interval=(0.0, 1.0),
            ),
            irrelevant_elements_remaining=len(set(initial_input.elements) - true_set),
            matches_ground_truth=False,
            budget_exhausted=True,
        )

    if base_eval.failure_rate < failure_threshold:
        return StochasticBaselineComparison(
            baseline_name="stochastic_greedy_ablation",
            discovered_elements=(),
            failure_reproduced=False,
            minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
            target_calls=runner.trial_count,
            candidate_evaluation=base_eval,
            irrelevant_elements_remaining=0,
            matches_ground_truth=(len(true_set) == 0),
            budget_exhausted=False,
        )

    current_elements = list(initial_input.elements)
    current_eval = base_eval

    try:
        # Step 2: Sequential removal pass
        i = 0
        while i < len(current_elements):
            if len(current_elements) <= 1:
                break
            candidate = tuple(e for idx, e in enumerate(current_elements) if idx != i)
            sub_case = initial_input.subset(candidate)
            step_eval = evaluate_candidate_stochastically(
                runner=runner,
                input_case=sub_case,
                trials=trials_per_candidate,
                intervention_type="stochastic_greedy_step",
                confidence=confidence,
            )

            if step_eval.failure_rate >= failure_threshold:
                current_elements.pop(i)
                current_eval = step_eval
            else:
                i += 1

        discovered = tuple(current_elements)

        # Step 3: Minimality evaluation of discovered subset
        removal_evals: dict[str, CandidateEvaluation] = {}
        for elem in discovered:
            if len(discovered) > 1:
                without_elem = tuple(e for e in discovered if e != elem)
                sub_case = initial_input.subset(without_elem)
            else:
                sub_case = initial_input.without(elem)

            rem_eval = evaluate_candidate_stochastically(
                runner=runner,
                input_case=sub_case,
                trials=trials_per_candidate,
                intervention_type="stochastic_greedy_minimality_check",
                confidence=confidence,
            )
            removal_evals[elem] = rem_eval

        minimality_status = _classify_stochastic_minimality(
            candidate=discovered,
            cand_eval=current_eval,
            removal_evals=removal_evals,
            failure_threshold=failure_threshold,
            alpha=alpha,
            confidence=confidence,
            alternative=alternative,
        )

        discovered_set = set(discovered)
        matches_gt = (discovered_set == true_set) or any(discovered_set == alt for alt in alt_sets)

        return StochasticBaselineComparison(
            baseline_name="stochastic_greedy_ablation",
            discovered_elements=discovered,
            failure_reproduced=True,
            minimality_status=minimality_status,
            target_calls=runner.trial_count,
            candidate_evaluation=current_eval,
            irrelevant_elements_remaining=len(discovered_set - true_set),
            matches_ground_truth=matches_gt,
            budget_exhausted=runner.is_budget_exhausted(),
        )

    except (BudgetExhaustedError, TimeoutExceededError):
        discovered = tuple(current_elements)
        discovered_set = set(discovered)
        matches_gt = (discovered_set == true_set) or any(discovered_set == alt for alt in alt_sets)
        return StochasticBaselineComparison(
            baseline_name="stochastic_greedy_ablation",
            discovered_elements=discovered,
            failure_reproduced=True,
            minimality_status=StochasticMinimalityStatus.INCONCLUSIVE,
            target_calls=runner.trial_count,
            candidate_evaluation=current_eval,
            irrelevant_elements_remaining=len(discovered_set - true_set),
            matches_ground_truth=matches_gt,
            budget_exhausted=True,
        )


def run_stochastic_random_search(
    initial_input: InputCase,
    runner: InterventionRunner,
    trials_per_candidate: int = 20,
    failure_threshold: float = 0.30,
    alpha: float = 0.05,
    confidence: float = 0.95,
    alternative: str = "two-sided",
    true_trigger: tuple[str, ...] = (),
    alternative_valid_triggers: tuple[tuple[str, ...], ...] = (),
    seed: int = 42,
) -> StochasticBaselineComparison:
    """Stochastic seeded random candidate search baseline.

    Randomly samples non-empty proper subsets under the configured trial budget,
    tracking the smallest candidate that achieves failure rate >= failure_threshold.
    """
    true_set = set(true_trigger)
    alt_sets = [set(alt) for alt in alternative_valid_triggers]
    rng = random.Random(seed)

    # Step 1: Baseline check
    try:
        base_eval = evaluate_candidate_stochastically(
            runner=runner,
            input_case=initial_input,
            trials=trials_per_candidate,
            intervention_type="stochastic_random_baseline",
            confidence=confidence,
        )
    except (BudgetExhaustedError, TimeoutExceededError):
        return StochasticBaselineComparison(
            baseline_name="stochastic_seeded_random",
            discovered_elements=initial_input.elements,
            failure_reproduced=False,
            minimality_status=StochasticMinimalityStatus.INCONCLUSIVE,
            target_calls=runner.trial_count,
            candidate_evaluation=CandidateEvaluation(
                candidate_elements=initial_input.elements,
                trials=0,
                failures=0,
                failure_rate=0.0,
                confidence_interval=(0.0, 1.0),
            ),
            irrelevant_elements_remaining=len(set(initial_input.elements) - true_set),
            matches_ground_truth=False,
            budget_exhausted=True,
        )

    if base_eval.failure_rate < failure_threshold:
        return StochasticBaselineComparison(
            baseline_name="stochastic_seeded_random",
            discovered_elements=(),
            failure_reproduced=False,
            minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
            target_calls=runner.trial_count,
            candidate_evaluation=base_eval,
            irrelevant_elements_remaining=0,
            matches_ground_truth=(len(true_set) == 0),
            budget_exhausted=False,
        )

    all_elements = list(initial_input.elements)
    best_candidate: tuple[str, ...] = tuple(all_elements)
    best_eval = base_eval
    budget_exhausted = False

    try:
        while not runner.is_budget_exhausted() and not runner.is_timed_out():
            if len(all_elements) <= 1:
                break
            k = rng.randint(1, len(all_elements) - 1)
            sample = tuple(sorted(rng.sample(all_elements, k)))

            sub_case = initial_input.subset(sample)
            cand_eval = evaluate_candidate_stochastically(
                runner=runner,
                input_case=sub_case,
                trials=trials_per_candidate,
                intervention_type="stochastic_random_sample",
                confidence=confidence,
            )

            if cand_eval.failure_rate >= failure_threshold:
                if len(sample) < len(best_candidate):
                    best_candidate = sample
                    best_eval = cand_eval

    except (BudgetExhaustedError, TimeoutExceededError):
        budget_exhausted = True

    # Attempt minimality check on best_candidate
    removal_evals: dict[str, CandidateEvaluation] = {}
    minimality_status = StochasticMinimalityStatus.INCONCLUSIVE

    if not budget_exhausted:
        try:
            for elem in best_candidate:
                if len(best_candidate) > 1:
                    without_elem = tuple(e for e in best_candidate if e != elem)
                    sub_case = initial_input.subset(without_elem)
                else:
                    sub_case = initial_input.without(elem)

                rem_eval = evaluate_candidate_stochastically(
                    runner=runner,
                    input_case=sub_case,
                    trials=trials_per_candidate,
                    intervention_type="stochastic_random_minimality_check",
                    confidence=confidence,
                )
                removal_evals[elem] = rem_eval

            minimality_status = _classify_stochastic_minimality(
                candidate=best_candidate,
                cand_eval=best_eval,
                removal_evals=removal_evals,
                failure_threshold=failure_threshold,
                alpha=alpha,
                confidence=confidence,
                alternative=alternative,
            )
        except (BudgetExhaustedError, TimeoutExceededError):
            budget_exhausted = True
            minimality_status = StochasticMinimalityStatus.INCONCLUSIVE

    best_set = set(best_candidate)
    matches_gt = (best_set == true_set) or any(best_set == alt for alt in alt_sets)

    return StochasticBaselineComparison(
        baseline_name="stochastic_seeded_random",
        discovered_elements=best_candidate,
        failure_reproduced=True,
        minimality_status=minimality_status,
        target_calls=runner.trial_count,
        candidate_evaluation=best_eval,
        irrelevant_elements_remaining=len(best_set - true_set),
        matches_ground_truth=matches_gt,
        budget_exhausted=budget_exhausted or runner.is_budget_exhausted(),
    )
