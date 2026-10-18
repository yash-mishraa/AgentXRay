"""Standardized, non-mutating 1-minimality verification protocol for candidate conditions."""

from __future__ import annotations

from enum import Enum
from typing import Any, Sequence
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentxray.agent_harness.cost_tracker import CostBudgetExceededError, CostTracker
from agentxray.models import (
    CandidateEvaluation,
    InputCase,
    StochasticMinimalityStatus,
)
from agentxray.power import (
    FisherPowerResult,
    JointPowerResult,
    PowerSolverStatus,
    exact_fisher_p_value_unrounded,
    solve_joint_minimality_power,
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
        description="Validation status of sample size / statistical power ('not_implemented', 'unresolved', 'validated', 'range_exhausted', 'invalid_inputs').",
    )
    confirmatory_power_established: bool = Field(
        default=False,
        description="Whether confirmatory statistical power has been formally established with a validated power calculation.",
    )
    planned_sample_size_adequate: bool = Field(
        default=False,
        description="Whether planned sample size was >= required n*.",
    )
    prospective_power_validated: bool = Field(
        default=False,
        description="Whether prospective sample size solver validated the design.",
    )
    execution_protocol_adherent: bool = Field(
        default=True,
        description="Whether all planned trials completed without crashes/timeouts/missingness.",
    )
    missingness_sensitivity_status: str = Field(
        default="not_applicable",
        description="Status of worst-case sensitivity imputation: 'not_applicable', 'robust', 'fragile'.",
    )
    power_result: JointPowerResult | FisherPowerResult | None = Field(
        default=None,
        description="Detailed power solver result if prospective power was calculated.",
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

        # Invariant 2: confirmatory_power_established=True cannot coexist with unresolved, range_exhausted, or unimplemented validation status
        if self.confirmatory_power_established and val_status in ("unresolved", "not_implemented", "not_applicable", "range_exhausted", "invalid_inputs", ""):
            raise ValueError(
                f"Invalid evidentiary state: confirmatory_power_established=True cannot coexist "
                f"with sample_size_validation_status={self.sample_size_validation_status!r}."
            )

        # Invariant 3: confirmatory_power_established=True requires planned_sample_size_adequate=True
        if self.confirmatory_power_established and not self.planned_sample_size_adequate:
            raise ValueError(
                "Invalid evidentiary claim: confirmatory_power_established cannot be True without "
                "validated prospective power and adequate planned sample size (planned_sample_size_adequate=True)."
            )

        return self


def is_confirmatory_power_established(
    tier: VerificationEvidenceTier | str,
    rationale: str | None = None,
    validation_status: str = "not_implemented",
    planned_sample_size_adequate: bool = False,
) -> bool:
    """Check whether confirmatory power has been formally established with a validated power calculation.

    Returns True iff tier is confirmatory, validation_status is 'validated', and planned sample size is adequate.
    """
    tier_str = tier.value.lower() if isinstance(tier, VerificationEvidenceTier) else str(tier or "").lower()
    return tier_str == "confirmatory" and validation_status.lower() == "validated" and planned_sample_size_adequate


def evaluate_missingness_sensitivity(
    candidate_failures: int,
    candidate_trials: int,
    candidate_missing: int,
    ablation_failures: int,
    ablation_trials: int,
    ablation_missing: int,
    alpha: float = 0.05,
) -> tuple[float, bool]:
    """Perform worst-case sensitivity analysis for missing/interrupted trials under H1: p1 > p2.

    Imputes candidate missing trials as Success (Y=0) to minimize candidate failure rate:
        cand_k = candidate_failures
        cand_n = candidate_trials + candidate_missing
    Imputes ablation missing trials as Failure (Y=1) to maximize ablation failure rate:
        ablat_k = ablation_failures + ablation_missing
        ablat_n = ablation_trials + ablation_missing

    Returns (p_worst_case, rejects_null).
    """
    cand_k = candidate_failures
    cand_n = candidate_trials + candidate_missing
    ablat_k = ablation_failures + ablation_missing
    ablat_n = ablation_trials + ablation_missing

    if cand_n <= 0 or ablat_n <= 0:
        return 1.0, False

    p_val = exact_fisher_p_value_unrounded(
        k1=cand_k,
        n1=cand_n,
        k2=ablat_k,
        n2=ablat_n,
        alternative="greater",
    )
    return p_val, (p_val < alpha)


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
    prospective_p1: float | None = None,
    prospective_p2: float | Sequence[float] | None = None,
    prospective_target_power: float = 0.80,
    prospective_alpha: float = 0.05,
    prospective_n_min: int = 1,
    prospective_n_max: int = 120,
    candidate_missing_trials: int = 0,
    ablation_missing_trials: dict[str, int] | None = None,
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
    prospective_p1 : float | None
        Prespecified candidate failure probability for prospective power calculation.
    prospective_p2 : float | Sequence[float] | None
        Prespecified ablation failure probability (scalar or per-component sequence).
    prospective_target_power : float
        Target joint statistical power (default 0.80).
    prospective_alpha : float
        Significance level for prospective power calculation (default 0.05).
    prospective_n_min, prospective_n_max : int
        Finite evaluation domain bounds (default 1 to 120).
    candidate_missing_trials : int
        Count of candidate trials lost to infrastructure aborts or attrition.
    ablation_missing_trials : dict[str, int] | None
        Mapping of element names to lost ablation trial counts.

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

    # 0. Prospective Power Validation (prior to trial execution)
    # AgentXRay records prospective power parameters prior to verification trial dispatch.
    # Note: In an unconstrained client environment, post-hoc parameter adjustments cannot be
    # physically prevented by runtime checks alone; prospective rigor requires declaring rates prior to data collection.
    power_res: JointPowerResult | None = None
    prospective_power_validated = False
    planned_sample_size_adequate = False

    if prospective_p1 is not None and prospective_p2 is not None:
        k_comp = len(cand_tuple) if cand_tuple else 1
        joint_res = solve_joint_minimality_power(
            p1=prospective_p1,
            p2_rates=prospective_p2,
            k=k_comp,
            joint_target_power=prospective_target_power,
            alpha=prospective_alpha,
            n_min=prospective_n_min,
            n_max=prospective_n_max,
        )
        power_res = joint_res
        if joint_res.status == PowerSolverStatus.VALIDATED:
            validation_status = "validated"
            prospective_power_validated = True
            planned_sample_size_adequate = (trials_per_candidate >= (joint_res.n_required or 0))
            power_established = (resolved_tier == VerificationEvidenceTier.CONFIRMATORY and planned_sample_size_adequate)
            resolved_rationale = (
                sample_size_rationale.strip()
                if sample_size_rationale and sample_size_rationale.strip()
                else f"prospective_power_{joint_res.joint_target_power:.2f}_n{joint_res.n_required}"
            )
        else:  # RANGE_EXHAUSTED or INVALID_INPUTS
            validation_status = "range_exhausted"
            prospective_power_validated = False
            planned_sample_size_adequate = False
            power_established = False
            resolved_rationale = (
                sample_size_rationale.strip()
                if sample_size_rationale and sample_size_rationale.strip()
                else "prospective_power_range_exhausted"
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
            planned_sample_size_adequate=planned_sample_size_adequate,
            prospective_power_validated=prospective_power_validated,
            execution_protocol_adherent=True,
            missingness_sensitivity_status="not_applicable",
            power_result=power_res,
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
            planned_sample_size_adequate=planned_sample_size_adequate,
            prospective_power_validated=prospective_power_validated,
            execution_protocol_adherent=False,
            missingness_sensitivity_status="not_applicable",
            power_result=power_res,
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
            planned_sample_size_adequate=planned_sample_size_adequate,
            prospective_power_validated=prospective_power_validated,
            execution_protocol_adherent=False,
            missingness_sensitivity_status="not_applicable",
            power_result=power_res,
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
            planned_sample_size_adequate=planned_sample_size_adequate,
            prospective_power_validated=prospective_power_validated,
            execution_protocol_adherent=True,
            missingness_sensitivity_status="not_applicable",
            power_result=power_res,
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

    # 4. Missingness Assessment
    cand_missing = max(0, candidate_missing_trials)
    ablat_missing_map = ablation_missing_trials or {}
    has_missingness = (cand_missing > 0) or any(v > 0 for v in ablat_missing_map.values())
    execution_protocol_adherent = (interrupted_status is None) and (not has_missingness)

    # Missingness sensitivity checks across removals
    sensitivity_rejected_all = True
    if has_missingness and removal_evals:
        for elem in cand_tuple:
            rem_eval = removal_evals.get(elem)
            if rem_eval is None:
                sensitivity_rejected_all = False
                break
            m_ablat = ablat_missing_map.get(elem, 0)
            p_sens, rej_sens = evaluate_missingness_sensitivity(
                candidate_failures=cand_eval.failures,
                candidate_trials=cand_eval.trials,
                candidate_missing=cand_missing,
                ablation_failures=rem_eval.failures,
                ablation_trials=rem_eval.trials,
                ablation_missing=m_ablat,
                alpha=alpha,
            )
            if not rej_sens:
                sensitivity_rejected_all = False

    if not has_missingness:
        missingness_sensitivity_status = "not_applicable"
    elif sensitivity_rejected_all:
        missingness_sensitivity_status = "robust"
    else:
        missingness_sensitivity_status = "fragile"

    # 5. Handle Partial Interruption or Prove Non-Minimality
    still_failing = [
        elem for elem, rem in removal_evals.items()
        if rem.failure_rate >= failure_threshold
    ]

    # If an evaluated removal already meets or exceeds failure_threshold, candidate is NOT_MINIMAL
    if still_failing:
        msg = (
            f"Candidate is not 1-minimal: removal of {still_failing} still produces failure rate "
            f">= threshold {failure_threshold:.2f}. "
            f"Classified as NOT_MINIMAL based on observed sample rates (note: observed-data classification, "
            f"not an equivalence or non-inferiority proof)."
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
            planned_sample_size_adequate=planned_sample_size_adequate,
            prospective_power_validated=prospective_power_validated,
            execution_protocol_adherent=execution_protocol_adherent,
            missingness_sensitivity_status=missingness_sensitivity_status,
            power_result=power_res,
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
            planned_sample_size_adequate=planned_sample_size_adequate,
            prospective_power_validated=prospective_power_validated,
            execution_protocol_adherent=False,
            missingness_sensitivity_status=missingness_sensitivity_status,
            power_result=power_res,
            target_calls_used=calls_used,
            model_requests_used=reqs_used,
            status_message=(
                f"Partial verification: {interrupted_msg} "
                f"({len(removal_evals)}/{len(cand_tuple)} element removals evaluated)."
            ),
            is_complete=False,
        )

    # 6. Statistical Hypothesis Testing under Berger (1982) Intersection-Union Test
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
        p_comp = exact_fisher_p_value_unrounded(
            k1=p1_k,
            n1=p1_n,
            k2=rem_eval.failures,
            n2=rem_eval.trials,
            alternative="greater",
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
    any_non_drop = not all_observed_drops

    # 7. Deterministic Status Precedence
    # Precedence 1: Range Exhaustion / Invalid Design
    if validation_status == "range_exhausted":
        if all_statistically_significant and all_observed_drops:
            status = StochasticMinimalityStatus.OBSERVED_ONLY
            msg = (
                f"Observed 1-minimality: element removals statistically significant at alpha={alpha}, "
                f"but prospective power validation failed due to range exhaustion (exploratory finding; "
                f"comparisons: {'; '.join(diff_summaries)})."
            )
        elif any_non_drop:
            status = StochasticMinimalityStatus.NOT_MINIMAL
            msg = (
                f"Removal of at least one element did not decrease the observed failure rate. "
                f"Classified as NOT_MINIMAL based on observed sample rates (note: observed-data classification, "
                f"not an equivalence or non-inferiority proof)."
            )
        else:
            status = StochasticMinimalityStatus.INCONCLUSIVE
            msg = "1-Minimality not established: prospective power search exhausted domain budget."

    # Precedence 2: Protocol Deviation / Missingness present
    elif not execution_protocol_adherent:
        if all_statistically_significant and all_observed_drops:
            if missingness_sensitivity_status == "robust":
                status = StochasticMinimalityStatus.OBSERVED_ONLY
                msg = (
                    f"Qualified 1-minimality: nominal significance holds at alpha={alpha} and survives "
                    f"adversarial missingness sensitivity imputation, but confirmatory status is withheld "
                    f"due to protocol attrition / missing trials (comparisons: {'; '.join(diff_summaries)})."
                )
            else:  # fragile
                status = StochasticMinimalityStatus.INCONCLUSIVE
                msg = (
                    f"Inconclusive: nominal statistical significance at alpha={alpha} collapses under "
                    f"adversarial missingness sensitivity analysis (informative missingness detected)."
                )
        elif any_non_drop:
            status = StochasticMinimalityStatus.NOT_MINIMAL
            msg = "Removal of at least one element did not decrease the observed failure rate."
        else:
            status = StochasticMinimalityStatus.INCONCLUSIVE
            msg = "1-Minimality not established: protocol attrition or interruption compromised trial execution."

    # Precedence 3: Protocol adherent, all statistically significant
    elif all_statistically_significant and all_observed_drops:
        if resolved_tier == VerificationEvidenceTier.CONFIRMATORY:
            if power_established:
                status = StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
                msg = (
                    f"1-Minimality statistically supported across all {len(cand_tuple)} element removals "
                    f"under prospective confirmatory Intersection-Union Test (comparative Fisher all p < {alpha}; "
                    f"threshold binomial all p < {alpha}: {all_thresh_supported}; comparisons: {'; '.join(diff_summaries)})."
                )
            else:
                status = StochasticMinimalityStatus.OBSERVED_ONLY
                msg = (
                    f"Observed 1-minimality: element removals achieved statistical significance at alpha={alpha}, "
                    f"but confirmatory power was not established (exploratory significance; comparisons: {'; '.join(diff_summaries)})."
                )
        else:  # Exploratory tier
            if prospective_power_validated and not planned_sample_size_adequate:
                status = StochasticMinimalityStatus.OBSERVED_ONLY
                msg = (
                    f"Observed 1-minimality: element removals achieved statistical significance at alpha={alpha}, "
                    f"but planned sample size was inadequate for confirmatory power (exploratory significance; "
                    f"comparisons: {'; '.join(diff_summaries)})."
                )
            else:
                status = StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
                msg = (
                    f"1-Minimality statistically supported across all {len(cand_tuple)} element removals "
                    f"under Intersection-Union Test (comparative Fisher all p < {alpha}; "
                    f"threshold binomial all p < {alpha}: {all_thresh_supported}; comparisons: {'; '.join(diff_summaries)})."
                )

    # Precedence 4: Protocol adherent, not all statistically significant
    else:
        if all_observed_drops:
            status = StochasticMinimalityStatus.OBSERVED_ONLY
            if power_established:
                msg = (
                    f"1-Minimality not established: failure rate dropped upon every element removal, "
                    f"but did not achieve statistical significance at alpha={alpha} "
                    f"(comparative p-values: {[round(p, 4) for p in p_values_comp]}, threshold p-values: {[round(p, 4) for p in p_values_thresh]}; "
                    f"comparisons: {'; '.join(diff_summaries)}). Study had validated prospective power for prespecified parameters, "
                    f"but observed data did not provide sufficient evidence to reject H0 (adequately powered non-rejection; does not prove dispensability)."
                )
            else:
                msg = (
                    f"Observed 1-minimality: failure rate dropped upon every element removal, "
                    f"but one or more tests did not achieve statistical significance at alpha={alpha} "
                    f"(comparative p-values: {[round(p, 4) for p in p_values_comp]}, threshold p-values: {[round(p, 4) for p in p_values_thresh]}; "
                    f"comparisons: {'; '.join(diff_summaries)})."
                )
        else:
            status = StochasticMinimalityStatus.NOT_MINIMAL
            msg = (
                f"Removal of at least one element did not decrease the observed failure rate. "
                f"Classified as NOT_MINIMAL based on observed sample rates (note: observed-data classification, "
                f"not an equivalence or non-inferiority proof)."
            )

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
        planned_sample_size_adequate=planned_sample_size_adequate,
        prospective_power_validated=prospective_power_validated,
        execution_protocol_adherent=execution_protocol_adherent,
        missingness_sensitivity_status=missingness_sensitivity_status,
        power_result=power_res,
        target_calls_used=calls_used,
        model_requests_used=reqs_used,
        status_message=msg,
        is_complete=True,
    )
