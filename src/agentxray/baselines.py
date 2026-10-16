"""Baseline comparison algorithms: Greedy Ablation and Seeded Random Search."""

from __future__ import annotations

import random
from typing import Sequence

from agentxray.models import BaselineComparison, InputCase, TargetOutcome
from agentxray.runner import BudgetExhaustedError, InterventionRunner, TimeoutExceededError


def run_greedy_ablation(
    initial_input: InputCase,
    runner: InterventionRunner,
    true_trigger: tuple[str, ...] = (),
) -> BaselineComparison:
    """Greedy single-element ablation baseline.

    Sequentially attempts to remove each element from the candidate set one-by-one.
    If the subset without element 'e' still fails, 'e' is permanently dropped.
    """
    true_set = set(true_trigger)
    budget_exhausted = False

    # Step 1: Baseline check
    try:
        base_res = runner.execute_trial(initial_input, intervention_type="greedy_baseline")
    except (BudgetExhaustedError, TimeoutExceededError):
        return BaselineComparison(
            baseline_name="greedy_ablation",
            discovered_elements=initial_input.elements,
            failure_reproduced=False,
            minimality_check_passed=False,
            target_calls=runner.trial_count,
            irrelevant_elements_remaining=len(set(initial_input.elements) - true_set),
            matches_ground_truth=False,
            budget_exhausted=True,
        )

    if base_res.outcome != TargetOutcome.FAILURE:
        return BaselineComparison(
            baseline_name="greedy_ablation",
            discovered_elements=(),
            failure_reproduced=False,
            minimality_check_passed=False,
            target_calls=runner.trial_count,
            irrelevant_elements_remaining=0,
            matches_ground_truth=(len(true_set) == 0),
            budget_exhausted=False,
        )

    current_elements = list(initial_input.elements)

    try:
        # Step 2: Sequential removal pass
        i = 0
        while i < len(current_elements):
            if len(current_elements) <= 1:
                break
            candidate = tuple(e for idx, e in enumerate(current_elements) if idx != i)
            sub_case = initial_input.subset(candidate)
            res = runner.execute_trial(sub_case, intervention_type="greedy_ablation_step")

            if res.outcome == TargetOutcome.FAILURE:
                # Element at index i was unnecessary; remove it and don't advance i
                current_elements.pop(i)
            else:
                # Element at index i was necessary to trigger failure; keep it and advance i
                i += 1

        discovered = tuple(current_elements)

        # Step 3: Minimality check on discovered subset
        minimality_passed = True
        if len(discovered) > 1:
            for elem in discovered:
                without_elem = tuple(e for e in discovered if e != elem)
                sub_case = initial_input.subset(without_elem)
                res = runner.execute_trial(sub_case, intervention_type="greedy_minimality_check")
                if res.outcome == TargetOutcome.FAILURE:
                    minimality_passed = False
                    break

        discovered_set = set(discovered)
        return BaselineComparison(
            baseline_name="greedy_ablation",
            discovered_elements=discovered,
            failure_reproduced=True,
            minimality_check_passed=minimality_passed,
            target_calls=runner.trial_count,
            irrelevant_elements_remaining=len(discovered_set - true_set),
            matches_ground_truth=(discovered_set == true_set),
            budget_exhausted=runner.is_budget_exhausted(),
        )

    except (BudgetExhaustedError, TimeoutExceededError):
        discovered = tuple(current_elements)
        discovered_set = set(discovered)
        return BaselineComparison(
            baseline_name="greedy_ablation",
            discovered_elements=discovered,
            failure_reproduced=True,
            minimality_check_passed=False,
            target_calls=runner.trial_count,
            irrelevant_elements_remaining=len(discovered_set - true_set),
            matches_ground_truth=(discovered_set == true_set),
            budget_exhausted=True,
        )


def run_seeded_random_search(
    initial_input: InputCase,
    runner: InterventionRunner,
    true_trigger: tuple[str, ...] = (),
    seed: int = 42,
) -> BaselineComparison:
    """Seeded random candidate search baseline.

    Randomly samples non-empty proper subsets under the configured trial budget,
    tracking the smallest candidate that reproduces the target failure.
    """
    true_set = set(true_trigger)
    rng = random.Random(seed)

    # Step 1: Baseline check
    try:
        base_res = runner.execute_trial(initial_input, intervention_type="random_baseline")
    except (BudgetExhaustedError, TimeoutExceededError):
        return BaselineComparison(
            baseline_name="seeded_random",
            discovered_elements=initial_input.elements,
            failure_reproduced=False,
            minimality_check_passed=False,
            target_calls=runner.trial_count,
            irrelevant_elements_remaining=len(set(initial_input.elements) - true_set),
            matches_ground_truth=False,
            budget_exhausted=True,
        )

    if base_res.outcome != TargetOutcome.FAILURE:
        return BaselineComparison(
            baseline_name="seeded_random",
            discovered_elements=(),
            failure_reproduced=False,
            minimality_check_passed=False,
            target_calls=runner.trial_count,
            irrelevant_elements_remaining=0,
            matches_ground_truth=(len(true_set) == 0),
            budget_exhausted=False,
        )

    all_elements = list(initial_input.elements)
    best_failing: tuple[str, ...] = tuple(all_elements)
    budget_exhausted = False

    try:
        # Sample random sub-configurations until budget runs out
        while not runner.is_budget_exhausted() and not runner.is_timed_out():
            if len(all_elements) <= 1:
                break
            # Pick a random subset size between 1 and len - 1
            k = rng.randint(1, len(all_elements) - 1)
            sample = tuple(sorted(rng.sample(all_elements, k)))

            sub_case = initial_input.subset(sample)
            res = runner.execute_trial(sub_case, intervention_type="random_sample")

            if res.outcome == TargetOutcome.FAILURE:
                if len(sample) < len(best_failing):
                    best_failing = sample

    except (BudgetExhaustedError, TimeoutExceededError):
        budget_exhausted = True

    # Check 1-minimality of best found subset
    minimality_passed = False
    if len(best_failing) == 1:
        minimality_passed = True
    elif not budget_exhausted:
        try:
            passed = True
            for elem in best_failing:
                without_elem = tuple(e for e in best_failing if e != elem)
                sub_case = initial_input.subset(without_elem)
                res = runner.execute_trial(sub_case, intervention_type="random_minimality_check")
                if res.outcome == TargetOutcome.FAILURE:
                    passed = False
                    break
            minimality_passed = passed
        except (BudgetExhaustedError, TimeoutExceededError):
            budget_exhausted = True
            minimality_passed = False

    best_set = set(best_failing)
    return BaselineComparison(
        baseline_name="seeded_random",
        discovered_elements=best_failing,
        failure_reproduced=True,
        minimality_check_passed=minimality_passed,
        target_calls=runner.trial_count,
        irrelevant_elements_remaining=len(best_set - true_set),
        matches_ground_truth=(best_set == true_set),
        budget_exhausted=budget_exhausted or runner.is_budget_exhausted(),
    )
