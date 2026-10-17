"""Stochastic 1-Minimal Delta-Debugging Algorithm.

Extends delta debugging to probabilistic black-box targets, supporting:
- Multi-trial candidate subset evaluation.
- Wilson score confidence intervals for binomial failure proportions.
- Statistical hypothesis testing for single-element removals against ablated controls.
- Benjamini-Hochberg FDR correction.
- Explicit classification into STATISTICALLY_SUPPORTED, OBSERVED_ONLY,
  INCONCLUSIVE, NOT_MINIMAL, or NOT_APPLICABLE.
"""

from __future__ import annotations

from typing import Sequence

from agentxray.models import (
    CandidateEvaluation,
    InputCase,
    SearchStatus,
    StochasticDiscoveredCondition,
    StochasticMinimalityStatus,
    TargetOutcome,
)
from agentxray.runner import BudgetExhaustedError, InterventionRunner, TimeoutExceededError
from agentxray.statistics import (
    adjust_p_values_bh,
    compare_proportions,
    wilson_score_interval,
)


def evaluate_candidate_stochastically(
    runner: InterventionRunner,
    input_case: InputCase,
    trials: int,
    intervention_type: str,
    confidence: float = 0.95,
) -> CandidateEvaluation:
    """Evaluate a candidate input configuration across multiple stochastic trials.

    Parameters
    ----------
    runner : InterventionRunner
        Active trial runner enforcing budgets.
    input_case : InputCase
        Candidate subset configuration to execute.
    trials : int
        Number of trials to execute.
    intervention_type : str
        Label categorizing the intervention.
    confidence : float
        Confidence level for the Wilson score interval.

    Returns
    -------
    CandidateEvaluation
        Observed trials, failure counts, point estimate, and confidence interval.
    """
    failures = 0
    executed = 0

    for _ in range(trials):
        res = runner.execute_trial(input_case, intervention_type=intervention_type)
        executed += 1
        if res.outcome == TargetOutcome.FAILURE:
            failures += 1

    rate = failures / executed if executed > 0 else 0.0
    ci = wilson_score_interval(failures, executed, confidence=confidence)

    eval_result = CandidateEvaluation(
        candidate_elements=input_case.elements,
        trials=executed,
        failures=failures,
        failure_rate=round(rate, 4),
        confidence_interval=ci,
    )
    if hasattr(runner, "on_candidate_evaluated") and runner.on_candidate_evaluated is not None:
        runner.on_candidate_evaluated(eval_result, intervention_type)

    return eval_result


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


class StochasticDeltaDebugMinimizer:
    """Discovers a 1-minimal failure condition in a probabilistic environment."""

    def __init__(
        self,
        runner: InterventionRunner,
        trials_per_candidate: int = 20,
        failure_threshold: float = 0.30,
        alpha: float = 0.05,
        confidence: float = 0.95,
        alternative: str = "two-sided",
    ) -> None:
        self.runner = runner
        self.trials_per_candidate = trials_per_candidate
        self.failure_threshold = failure_threshold
        self.alpha = alpha
        self.confidence = confidence
        self.alternative = alternative

    def minimize(self, initial_input: InputCase) -> StochasticDiscoveredCondition:
        """Execute stochastic delta-debugging minimization on initial_input.

        Parameters
        ----------
        initial_input : InputCase
            The full input configuration suspected of inducing failure.

        Returns
        -------
        StochasticDiscoveredCondition
            Isolated condition, sample statistics, minimality classification, and budget usage.
        """
        # Step 1: Baseline evaluation of initial input
        try:
            base_eval = evaluate_candidate_stochastically(
                runner=self.runner,
                input_case=initial_input,
                trials=self.trials_per_candidate,
                intervention_type="stochastic_baseline",
                confidence=self.confidence,
            )
        except (BudgetExhaustedError, TimeoutExceededError) as exc:
            return StochasticDiscoveredCondition(
                subset_elements=initial_input.elements,
                status=SearchStatus.BUDGET_EXHAUSTED if isinstance(exc, BudgetExhaustedError) else SearchStatus.TIMEOUT,
                is_failure=False,
                minimality_status=StochasticMinimalityStatus.INCONCLUSIVE,
                candidate_evaluation=CandidateEvaluation(
                    candidate_elements=initial_input.elements,
                    trials=0,
                    failures=0,
                    failure_rate=0.0,
                    confidence_interval=(0.0, 1.0),
                ),
                target_calls_used=self.runner.trial_count,
                search_target_calls=self.runner.trial_count,
                verification_target_calls=0,
                subsets_examined=len({t.input_case.elements for t in self.runner.trials}),
                status_message=f"Search halted during baseline evaluation: {exc}",
            )

        if base_eval.failure_rate < self.failure_threshold:
            return StochasticDiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.NO_TRIGGER_FOUND,
                is_failure=False,
                minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
                candidate_evaluation=base_eval,
                target_calls_used=self.runner.trial_count,
                search_target_calls=self.runner.trial_count,
                verification_target_calls=0,
                subsets_examined=len({t.input_case.elements for t in self.runner.trials}),
                status_message=(
                    f"Initial input failure rate {base_eval.failure_rate:.2f} did not meet "
                    f"threshold {self.failure_threshold:.2f}; no failure trigger detected."
                ),
            )

        current_candidate: tuple[str, ...] = initial_input.elements
        current_eval = base_eval
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
                        chunk_eval = evaluate_candidate_stochastically(
                            runner=self.runner,
                            input_case=sub_case,
                            trials=self.trials_per_candidate,
                            intervention_type="stochastic_partition_subset",
                            confidence=self.confidence,
                        )
                        if chunk_eval.failure_rate >= self.failure_threshold:
                            current_candidate = chunk
                            current_eval = chunk_eval
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
                        comp_eval = evaluate_candidate_stochastically(
                            runner=self.runner,
                            input_case=comp_case,
                            trials=self.trials_per_candidate,
                            intervention_type="stochastic_partition_complement",
                            confidence=self.confidence,
                        )
                        if comp_eval.failure_rate >= self.failure_threshold:
                            current_candidate = complement
                            current_eval = comp_eval
                            n = max(n - 1, 2)
                            reduced = True
                            break

                if reduced:
                    continue

                # 2C: Increase granularity if no reduction occurred
                if n < len(current_candidate):
                    n = min(n * 2, len(current_candidate))
                else:
                    break

            # Step 3: Explicit 1-minimality verification and refinement
            # Continuously check if removing any single element still meets the failure threshold.
            # For |current_candidate| > 1: ablate element within candidate: S \ {e}
            # For |current_candidate| == 1: ablate element from initial_input: initial_input \ {e}
            verified_minimal = False
            removal_evals: dict[str, CandidateEvaluation] = {}

            while not verified_minimal:
                unnecessary_element_found = False

                if len(current_candidate) == 0:
                    verified_minimal = True
                    break

                for elem in current_candidate:
                    candidate_without_elem = tuple(e for e in current_candidate if e != elem)
                    sub_case = initial_input.subset(candidate_without_elem)

                    eval_without = evaluate_candidate_stochastically(
                        runner=self.runner,
                        input_case=sub_case,
                        trials=self.trials_per_candidate,
                        intervention_type="stochastic_minimality_check",
                        confidence=self.confidence,
                    )
                    removal_evals[elem] = eval_without

                    if eval_without.failure_rate >= self.failure_threshold:
                        if len(current_candidate) > 1:
                            # Non-monotonic or further reducible: dropping elem still triggers failure!
                            current_candidate = candidate_without_elem
                            current_eval = eval_without
                            unnecessary_element_found = True
                            removal_evals.clear()
                            break

                if not unnecessary_element_found:
                    verified_minimal = True

            # Step 4: Statistical classification of 1-minimality
            # Compare current_candidate against each single-element removal
            minimality_status, status_msg = self._classify_minimality(
                current_candidate=current_candidate,
                current_eval=current_eval,
                removal_evals=removal_evals,
            )

            search_calls = sum(1 for t in self.runner.trials if t.intervention_type != "stochastic_minimality_check")
            verif_calls = sum(1 for t in self.runner.trials if t.intervention_type == "stochastic_minimality_check")
            subsets_cnt = len({t.input_case.elements for t in self.runner.trials})

            return StochasticDiscoveredCondition(
                subset_elements=current_candidate,
                status=SearchStatus.COMPLETE,
                is_failure=True,
                minimality_status=minimality_status,
                candidate_evaluation=current_eval,
                removal_evaluations=removal_evals,
                target_calls_used=self.runner.trial_count,
                search_target_calls=search_calls,
                verification_target_calls=verif_calls,
                subsets_examined=subsets_cnt,
                status_message=status_msg,
            )

        except (BudgetExhaustedError, TimeoutExceededError) as exc:
            search_calls = sum(1 for t in self.runner.trials if t.intervention_type != "stochastic_minimality_check")
            verif_calls = sum(1 for t in self.runner.trials if t.intervention_type == "stochastic_minimality_check")
            subsets_cnt = len({t.input_case.elements for t in self.runner.trials})
            return StochasticDiscoveredCondition(
                subset_elements=current_candidate,
                status=SearchStatus.BUDGET_EXHAUSTED if isinstance(exc, BudgetExhaustedError) else SearchStatus.TIMEOUT,
                is_failure=(current_eval.failure_rate >= self.failure_threshold),
                minimality_status=StochasticMinimalityStatus.INCONCLUSIVE,
                candidate_evaluation=current_eval,
                removal_evaluations=removal_evals if "removal_evals" in locals() else {},
                target_calls_used=self.runner.trial_count,
                search_target_calls=search_calls,
                verification_target_calls=verif_calls,
                subsets_examined=subsets_cnt,
                status_message=f"Search interrupted before verification: {exc}",
            )

    def _classify_minimality(
        self,
        current_candidate: tuple[str, ...],
        current_eval: CandidateEvaluation,
        removal_evals: dict[str, CandidateEvaluation],
    ) -> tuple[StochasticMinimalityStatus, str]:
        """Classify 1-minimality of current_candidate against removal evaluations."""
        if not current_candidate:
            return (
                StochasticMinimalityStatus.NOT_APPLICABLE,
                "Candidate set is empty; cannot establish minimality.",
            )

        # If any element was not evaluated in removal_evals, it's inconclusive
        missing_checks = set(current_candidate) - set(removal_evals.keys())
        if missing_checks:
            return (
                StochasticMinimalityStatus.INCONCLUSIVE,
                f"Missing removal evaluations for elements: {missing_checks}",
            )

        # Check if removing any element still met the failure threshold
        still_failing = [
            elem for elem, rem in removal_evals.items()
            if rem.failure_rate >= self.failure_threshold
        ]
        if still_failing:
            msg = (
                f"Candidate is not 1-minimal: removal of {still_failing} still produces failure rate "
                f">= threshold {self.failure_threshold:.2f}."
            )
            return (StochasticMinimalityStatus.NOT_MINIMAL, msg)

        p1_k = current_eval.failures
        p1_n = current_eval.trials

        p_values: list[float] = []
        observed_drops: list[bool] = []
        diff_summaries: list[str] = []

        for elem in current_candidate:
            rem_eval = removal_evals[elem]
            comp = compare_proportions(
                k1=p1_k,
                n1=p1_n,
                k2=rem_eval.failures,
                n2=rem_eval.trials,
                alpha=self.alpha,
                confidence=self.confidence,
                alternative=self.alternative,
            )
            p_values.append(comp.p_value if self.alternative == "two-sided" else comp.p_value_one_sided)
            observed_drops.append(current_eval.failure_rate > rem_eval.failure_rate)
            diff_summaries.append(
                f"without '{elem}': diff={comp.difference:+.2f} 95% CI [{comp.difference_ci[0]:.2f}, {comp.difference_ci[1]:.2f}], p={comp.p_value:.4f}"
            )

        # Benjamini-Hochberg FDR correction across all removal tests
        adjusted_p_values = adjust_p_values_bh(p_values)
        all_statistically_significant = all(q < self.alpha for q in adjusted_p_values)
        all_observed_drops = all(observed_drops)

        if all_statistically_significant:
            msg = (
                f"1-Minimality statistically supported across all {len(current_candidate)} element removals "
                f"(all BH-adjusted q-values < {self.alpha}; comparisons: {'; '.join(diff_summaries)})."
            )
            return (StochasticMinimalityStatus.STATISTICALLY_SUPPORTED, msg)

        if all_observed_drops:
            msg = (
                f"Observed 1-minimality: failure rate dropped upon every element removal, "
                f"but one or more tests did not achieve statistical significance at alpha={self.alpha} "
                f"(adjusted q-values: {[round(q, 4) for q in adjusted_p_values]}; comparisons: {'; '.join(diff_summaries)})."
            )
            return (StochasticMinimalityStatus.OBSERVED_ONLY, msg)

        msg = "Removal of at least one element did not decrease the observed failure rate."
        return (StochasticMinimalityStatus.NOT_MINIMAL, msg)
