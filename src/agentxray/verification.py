"""Standardized, non-mutating 1-minimality verification protocol for candidate conditions."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentxray.agent_harness.cost_tracker import CostBudgetExceededError, CostTracker
from agentxray.models import (
    CandidateEvaluation,
    InputCase,
    StochasticMinimalityStatus,
)
from agentxray.runner import BudgetExhaustedError, InterventionRunner, TimeoutExceededError
from agentxray.statistics import compare_proportions, exact_binomial_test
from agentxray.stochastic_minimizer import evaluate_candidate_stochastically


class VerificationExecutionStatus(str, Enum):
    """Operational status of the verification stage execution."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"
    NOT_APPLICABLE = "not_applicable"


class VerificationEvidenceTier(str, Enum):
    """Evidence tier classification for candidate verification."""

    EXPLORATORY = "exploratory"
    CONFIRMATORY = "confirmatory"


class VerificationResult(BaseModel):
    """Complete validated record of post-discovery candidate verification."""

    model_config = ConfigDict(frozen=True)

    candidate: tuple[str, ...] = Field(default=(), description="Candidate subset audited.")
    minimality_status: StochasticMinimalityStatus = Field(..., description="Statistical 1-minimality classification.")
    verification_status: VerificationExecutionStatus = Field(..., description="Operational status of verification execution.")
    evidence_tier: VerificationEvidenceTier = Field(
        default=VerificationEvidenceTier.EXPLORATORY,
        description="Evidence tier ('exploratory' or 'confirmatory').",
    )
    sample_size_rationale: str = Field(
        default="exploratory_default",
        description="User-supplied context or default rationale for verification sample size.",
    )
    sample_size_validation_status: str = Field(
        default="not_implemented",
        description="Validation status of sample size / statistical power ('not_implemented', 'unresolved', 'validated').",
    )
    confirmatory_power_established: bool = Field(
        default=False,
        description="Whether confirmatory statistical power has been formally established with a validated power calculation.",
    )
    candidate_evaluation: CandidateEvaluation | None = Field(default=None, description="Fresh evaluation of candidate C.")
    removal_evaluations: dict[str, CandidateEvaluation] = Field(default_factory=dict, description="Fresh evaluations of C \\ {e}.")
    target_calls_used: int = Field(default=0, ge=0, description="Target calls consumed in verification.")
    model_requests_used: int = Field(default=0, ge=0, description="Model provider requests consumed in verification.")
    status_message: str = Field(default="", description="Detailed outcome explanation.")
    is_complete: bool = Field(default=True, description="Whether all planned verification trials finished without interruption.")

    @model_validator(mode="after")
    def validate_evidentiary_consistency(self) -> VerificationResult:
        tier_val = (
            self.evidence_tier.value.lower()
            if isinstance(self.evidence_tier, VerificationEvidenceTier)
            else str(self.evidence_tier or "").lower()
        )
        val_status = str(self.sample_size_validation_status or "").lower()

        # Invariant 1: confirmatory_power_established=True cannot coexist with non-confirmatory tier
        if self.confirmatory_power_established and tier_val != "confirmatory":
            raise ValueError(
                f"Invalid evidentiary state: confirmatory_power_established=True cannot coexist "
                f"with evidence_tier={self.evidence_tier!r}."
            )

        # Invariant 2: confirmatory_power_established=True cannot coexist with unresolved or unimplemented validation status
        if self.confirmatory_power_established and val_status in ("unresolved", "not_implemented", "not_applicable", ""):
            raise ValueError(
                f"Invalid evidentiary state: confirmatory_power_established=True cannot coexist "
                f"with sample_size_validation_status={self.sample_size_validation_status!r}."
            )

        # Invariant 3: Because automated prospective power validation is not implemented in AgentXRay,
        # affirmative claims of confirmatory power are strictly rejected.
        if self.confirmatory_power_established:
            raise ValueError(
                "Invalid evidentiary claim: confirmatory_power_established cannot be True because "
                "automated prospective power analysis is not implemented in AgentXRay."
            )

        return self


def is_confirmatory_power_established(
    tier: VerificationEvidenceTier | str,
    rationale: str | None = None,
) -> bool:
    """Return False. Automated prospective power validation is not implemented in AgentXRay.

    Prospective power calculations require formal validation against effect size, base rate,
    alpha, and sample size. Until an automated validation workflow exists, this strictly returns False.
    """
    return False


def resolve_evidence_tier(
    evidence_tier: str | VerificationEvidenceTier = "exploratory",
    trials_per_candidate: int = 5,
    trials_explicitly_set: bool = False,
    sample_size_rationale: str | None = None,
) -> tuple[VerificationEvidenceTier, str, str, bool]:
    """Resolve evidence tier, user-supplied rationale context, validation status, and power establishment.

    Enforces that default exploratory trial count (5) cannot be promoted
    to confirmatory without explicit trial configuration.
    User-supplied rationale context is preserved, but validation status is
    'unresolved' (if user provided context) or 'not_implemented', and
    confirmatory_power_established is strictly False.
    """
    if isinstance(evidence_tier, VerificationEvidenceTier):
        tier_str = evidence_tier.value.lower()
    else:
        tier_str = str(evidence_tier).strip().lower()

    if tier_str == "confirmatory":
        if not trials_explicitly_set and trials_per_candidate == 5:
            raise ValueError(
                "Confirmatory verification requires an explicitly configured sample size (trials_per_candidate), "
                "but defaulted to exploratory 5 trials. Do not claim confirmatory evidence on default exploratory runs."
            )
        resolved_tier = VerificationEvidenceTier.CONFIRMATORY
        if sample_size_rationale and sample_size_rationale.strip():
            resolved_rationale = sample_size_rationale.strip()
            validation_status = "unresolved"
        else:
            resolved_rationale = "unresolved_user_specified"
            validation_status = "not_implemented"
        power_established = False
    elif tier_str == "exploratory":
        resolved_tier = VerificationEvidenceTier.EXPLORATORY
        if not trials_explicitly_set and trials_per_candidate == 5:
            resolved_rationale = "exploratory_default"
            validation_status = "not_implemented"
        else:
            resolved_rationale = (
                sample_size_rationale.strip()
                if sample_size_rationale and sample_size_rationale.strip()
                else "unresolved_user_specified"
            )
            validation_status = "unresolved"
        power_established = False
    else:
        raise ValueError(f"Unknown verification evidence_tier: {evidence_tier!r}. Supported: 'exploratory', 'confirmatory'.")

    return resolved_tier, resolved_rationale, validation_status, power_established


def verify_candidate_stochastically(
    candidate: tuple[str, ...] | list[str],
    initial_input: InputCase,
    runner: InterventionRunner,
    trials_per_candidate: int = 5,
    failure_threshold: float = 0.50,
    baseline_threshold: float | None = None,
    alpha: float = 0.05,
    confidence: float = 0.95,
    alternative: str = "greater",
    cost_tracker: CostTracker | None = None,
    evidence_tier: str = "exploratory",
    sample_size_rationale: str | None = None,
    trials_explicitly_set: bool = False,
) -> VerificationResult:
    """Execute a standardized, non-mutating 1-minimality verification audit on candidate.

    Parameters
    ----------
    candidate : tuple[str, ...] | list[str]
        The discovered candidate condition to audit.
    initial_input : InputCase
        The initial input case defining the universe of elements.
    runner : InterventionRunner
        Active trial runner enforcing verification trial budgets and timeouts.
    trials_per_candidate : int
        Number of trials to execute per condition.
    failure_threshold : float
        Proportion threshold defining target failure.
    baseline_threshold : float | None
        Optional upper bound on baseline noise for candidate validation.
    alpha : float
        Significance level for Berger's Intersection-Union Test.
    confidence : float
        Confidence level for uncertainty intervals.
    alternative : str
        Directional hypothesis for proportions comparison: 'greater' (H1: P(fail|C) > P(fail|C\\{e}))
        or 'two-sided'.
    cost_tracker : CostTracker | None
        Optional phase-specific cost tracker to monitor provider requests.
    evidence_tier : str
        Evidence tier requested: 'exploratory' or 'confirmatory'.
    sample_size_rationale : str | None
        Contextual rationale for the sample size.
    trials_explicitly_set : bool
        Whether trials_per_candidate was explicitly specified by caller.

    Returns
    -------
    VerificationResult
        Immutable audit record containing statistical classification and usage metrics.
    """
    cand_tuple = tuple(candidate)
    start_calls = runner.trial_count
    start_reqs = cost_tracker.total_requests if cost_tracker is not None else 0

    resolved_tier, resolved_rationale, validation_status, power_established = resolve_evidence_tier(
        evidence_tier=evidence_tier,
        trials_per_candidate=trials_per_candidate,
        trials_explicitly_set=trials_explicitly_set,
        sample_size_rationale=sample_size_rationale,
    )

    # 1. Null / Empty Candidate
    if not cand_tuple:
        return VerificationResult(
            candidate=(),
            minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
            verification_status=VerificationExecutionStatus.NOT_APPLICABLE,
            evidence_tier=resolved_tier,
            sample_size_rationale=resolved_rationale,
            sample_size_validation_status=validation_status,
            confirmatory_power_established=power_established,
            target_calls_used=0,
            model_requests_used=0,
            status_message="Candidate set is empty; 1-minimality verification not applicable.",
            is_complete=True,
        )

    # 2. Fresh Candidate Confirmation
    sub_case = initial_input.subset(cand_tuple)
    try:
        cand_eval = evaluate_candidate_stochastically(
            runner=runner,
            input_case=sub_case,
            trials=trials_per_candidate,
            intervention_type="verification_candidate_confirmation",
            confidence=confidence,
        )
    except (BudgetExhaustedError, CostBudgetExceededError) as exc:
        return VerificationResult(
            candidate=cand_tuple,
            minimality_status=StochasticMinimalityStatus.INCONCLUSIVE,
            verification_status=VerificationExecutionStatus.BUDGET_EXHAUSTED,
            evidence_tier=resolved_tier,
            sample_size_rationale=resolved_rationale,
            sample_size_validation_status=validation_status,
            confirmatory_power_established=power_established,
            target_calls_used=runner.trial_count - start_calls,
            model_requests_used=(cost_tracker.total_requests - start_reqs) if cost_tracker else 0,
            status_message=f"Verification budget exhausted during candidate confirmation: {exc}",
            is_complete=False,
        )
    except TimeoutExceededError as exc:
        return VerificationResult(
            candidate=cand_tuple,
            minimality_status=StochasticMinimalityStatus.INCONCLUSIVE,
            verification_status=VerificationExecutionStatus.TIMED_OUT,
            evidence_tier=resolved_tier,
            sample_size_rationale=resolved_rationale,
            sample_size_validation_status=validation_status,
            confirmatory_power_established=power_established,
            target_calls_used=runner.trial_count - start_calls,
            model_requests_used=(cost_tracker.total_requests - start_reqs) if cost_tracker else 0,
            status_message=f"Verification timed out during candidate confirmation: {exc}",
            is_complete=False,
        )

    # Check whether candidate reproduced target failure on fresh trials
    if (
        cand_eval.trials == 0
        or cand_eval.failure_rate < failure_threshold
        or (baseline_threshold is not None and cand_eval.confidence_interval[0] <= baseline_threshold)
    ):
        ci_msg = (
            f" with CI lower bound > {baseline_threshold:.2f}"
            if baseline_threshold is not None
            else ""
        )
        return VerificationResult(
            candidate=cand_tuple,
            candidate_evaluation=cand_eval,
            minimality_status=StochasticMinimalityStatus.INCONCLUSIVE,
            verification_status=VerificationExecutionStatus.COMPLETE,
            evidence_tier=resolved_tier,
            sample_size_rationale=resolved_rationale,
            sample_size_validation_status=validation_status,
            confirmatory_power_established=power_established,
            target_calls_used=runner.trial_count - start_calls,
            model_requests_used=(cost_tracker.total_requests - start_reqs) if cost_tracker else 0,
            status_message=(
                f"Candidate failed to reproduce failure rate >= {failure_threshold:.2f}{ci_msg} on fresh "
                f"verification trials (observed: {cand_eval.failure_rate:.2f}, trials: {cand_eval.trials})."
            ),
            is_complete=True,
        )

    # 3. Fresh Single-Element Removal Evaluations
    removal_evals: dict[str, CandidateEvaluation] = {}
    interrupted_status: VerificationExecutionStatus | None = None
    interrupted_msg: str = ""

    for elem in cand_tuple:
        if len(cand_tuple) == 1:
            without_elem = ()
        else:
            without_elem = tuple(e for e in cand_tuple if e != elem)
        rem_sub_case = initial_input.subset(without_elem)

        try:
            rem_eval = evaluate_candidate_stochastically(
                runner=runner,
                input_case=rem_sub_case,
                trials=trials_per_candidate,
                intervention_type="verification_single_element_ablation",
                confidence=confidence,
            )
            removal_evals[elem] = rem_eval
        except (BudgetExhaustedError, CostBudgetExceededError) as exc:
            interrupted_status = VerificationExecutionStatus.BUDGET_EXHAUSTED
            interrupted_msg = f"Verification budget exhausted during removal evaluation: {exc}"
            break
        except TimeoutExceededError as exc:
            interrupted_status = VerificationExecutionStatus.TIMED_OUT
            interrupted_msg = f"Verification timed out during removal evaluation: {exc}"
            break

    calls_used = runner.trial_count - start_calls
    reqs_used = (cost_tracker.total_requests - start_reqs) if cost_tracker else 0

    # 4. Handle Partial Interruption or Prove Non-Minimality
    still_failing = [
        elem for elem, rem in removal_evals.items()
        if rem.failure_rate >= failure_threshold
    ]

    # If an evaluated removal already meets or exceeds failure_threshold, candidate is proven NOT_MINIMAL
    if still_failing:
        msg = (
            f"Candidate is not 1-minimal: removal of {still_failing} still produces failure rate "
            f">= threshold {failure_threshold:.2f}."
        )
        if interrupted_status is not None:
            msg += f" (Note: verification was interrupted: {interrupted_msg})"
        return VerificationResult(
            candidate=cand_tuple,
            candidate_evaluation=cand_eval,
            removal_evaluations=removal_evals,
            minimality_status=StochasticMinimalityStatus.NOT_MINIMAL,
            verification_status=interrupted_status or VerificationExecutionStatus.COMPLETE,
            evidence_tier=resolved_tier,
            sample_size_rationale=resolved_rationale,
            sample_size_validation_status=validation_status,
            confirmatory_power_established=power_established,
            target_calls_used=calls_used,
            model_requests_used=reqs_used,
            status_message=msg,
            is_complete=(interrupted_status is None),
        )

    # If verification was interrupted before all removals could be tested, result is INCONCLUSIVE
    if interrupted_status is not None:
        return VerificationResult(
            candidate=cand_tuple,
            candidate_evaluation=cand_eval,
            removal_evaluations=removal_evals,
            minimality_status=StochasticMinimalityStatus.INCONCLUSIVE,
            verification_status=interrupted_status,
            evidence_tier=resolved_tier,
            sample_size_rationale=resolved_rationale,
            sample_size_validation_status=validation_status,
            confirmatory_power_established=power_established,
            target_calls_used=calls_used,
            model_requests_used=reqs_used,
            status_message=(
                f"Partial verification: {interrupted_msg} "
                f"({len(removal_evals)}/{len(cand_tuple)} element removals evaluated)."
            ),
            is_complete=False,
        )

    # 5. Statistical Hypothesis Testing under Berger (1982) Intersection-Union Test
    p1_k = cand_eval.failures
    p1_n = cand_eval.trials

    p_values_comp: list[float] = []
    p_values_thresh: list[float] = []
    observed_drops: list[bool] = []
    diff_summaries: list[str] = []

    for elem in cand_tuple:
        rem_eval = removal_evals[elem]
        comp = compare_proportions(
            k1=p1_k,
            n1=p1_n,
            k2=rem_eval.failures,
            n2=rem_eval.trials,
            alpha=alpha,
            confidence=confidence,
            alternative=alternative,
        )
        p_comp = comp.fisher_exact_p_value if comp.fisher_exact_p_value is not None else (
            comp.p_value if alternative == "two-sided" else comp.p_value_one_sided
        )
        p_thresh = exact_binomial_test(
            k=rem_eval.failures,
            n=rem_eval.trials,
            p0=failure_threshold,
            alternative="less",
        )
        p_values_comp.append(p_comp)
        p_values_thresh.append(p_thresh)
        observed_drops.append(cand_eval.failure_rate > rem_eval.failure_rate)
        diff_summaries.append(
            f"without '{elem}': diff={comp.difference:+.2f} 95% CI [{comp.difference_ci[0]:.2f}, {comp.difference_ci[1]:.2f}], "
            f"p_comp={p_comp:.4f}, p_thresh={p_thresh:.4f}"
        )

    all_statistically_significant = all(p < alpha for p in p_values_comp) and len(p_values_comp) == len(cand_tuple)
    all_observed_drops = all(observed_drops) and len(observed_drops) == len(cand_tuple)
    all_thresh_supported = all(p < alpha for p in p_values_thresh) and len(p_values_thresh) == len(cand_tuple)

    if all_statistically_significant:
        status = StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
        msg = (
            f"1-Minimality statistically supported across all {len(cand_tuple)} element removals "
            f"under Intersection-Union Test (comparative Fisher all p < {alpha}; threshold binomial all p < {alpha}: {all_thresh_supported}; "
            f"comparisons: {'; '.join(diff_summaries)})."
        )
    elif all_observed_drops:
        status = StochasticMinimalityStatus.OBSERVED_ONLY
        msg = (
            f"Observed 1-minimality: failure rate dropped upon every element removal, "
            f"but one or more tests did not achieve statistical significance at alpha={alpha} "
            f"(comparative p-values: {[round(p, 4) for p in p_values_comp]}, threshold p-values: {[round(p, 4) for p in p_values_thresh]}; "
            f"comparisons: {'; '.join(diff_summaries)})."
        )
    else:
        status = StochasticMinimalityStatus.NOT_MINIMAL
        msg = "Removal of at least one element did not decrease the observed failure rate."

    return VerificationResult(
        candidate=cand_tuple,
        candidate_evaluation=cand_eval,
        removal_evaluations=removal_evals,
        minimality_status=status,
        verification_status=VerificationExecutionStatus.COMPLETE,
        evidence_tier=resolved_tier,
        sample_size_rationale=resolved_rationale,
        sample_size_validation_status=validation_status,
        confirmatory_power_established=power_established,
        target_calls_used=calls_used,
        model_requests_used=reqs_used,
        status_message=msg,
        is_complete=True,
    )
