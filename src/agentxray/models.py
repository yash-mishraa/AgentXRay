"""Domain models and schemas for AgentXRay.

Validates all experiment configurations, input cases, target results,
probe trials, discovered conditions, and evaluation reports at the boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AccessMode(str, Enum):
    """Operational access mode for probing."""

    BLACK_BOX = "black_box"
    CONTROLLED_TESTBED = "controlled_testbed"


class TargetOutcome(str, Enum):
    """Observable execution outcome from target."""

    FAILURE = "failure"
    SUCCESS = "success"


class SearchStatus(str, Enum):
    """Outcome status of a minimal trigger discovery search."""

    COMPLETE = "complete"
    NO_TRIGGER_FOUND = "no_trigger_found"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIMEOUT = "timeout"
    ERROR = "error"
    INFRA_FAILURE = "infra_failure"


class ExperimentConfig(BaseModel):
    """Configuration for an experimentation run."""

    model_config = ConfigDict(frozen=True)

    experiment_name: str = Field(..., min_length=1, description="Unique name for this experiment.")
    max_trials: int = Field(default=100, gt=0, description="Maximum number of target evaluations allowed.")
    timeout_seconds: float = Field(default=30.0, gt=0.0, description="Wall-clock execution timeout in seconds.")
    seed: int = Field(default=42, description="Random seed for repeatable baselines.")
    mode: AccessMode = Field(default=AccessMode.CONTROLLED_TESTBED, description="Operational access mode.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary experiment metadata.")

    @field_validator("experiment_name")
    @classmethod
    def validate_name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("experiment_name cannot be blank or whitespace only")
        return v.strip()


class InputCase(BaseModel):
    """Discrete structured input configuration subjected to interventions."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for this input configuration.")
    elements: tuple[str, ...] = Field(default=(), description="Ordered discrete elements forming the input.")
    description: str = Field(default="", description="Optional human-readable description.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata associated with this input.")

    @field_validator("elements", mode="before")
    @classmethod
    def validate_elements(cls, v: Any) -> tuple[str, ...]:
        if v is None:
            return ()
        if isinstance(v, (list, set, tuple)):
            clean = [str(elem).strip() for elem in v if str(elem).strip()]
            if not clean:
                return ()
            # Preserve order while checking for duplicates
            seen: set[str] = set()
            ordered: list[str] = []
            for item in clean:
                if item in seen:
                    raise ValueError(f"Duplicate element detected in InputCase: {item!r}")
                seen.add(item)
                ordered.append(item)
            return tuple(ordered)
        raise ValueError("elements must be a sequence of strings")

    def without(self, element: str) -> InputCase:
        """Return a new InputCase omitting the specified element."""
        new_elements = tuple(e for e in self.elements if e != element)
        return InputCase(
            case_id=str(uuid4()),
            elements=new_elements,
            description=f"{self.description} (removed {element})",
            metadata=self.metadata,
        )

    def subset(self, kept: tuple[str, ...] | list[str] | set[str]) -> InputCase:
        """Return a new InputCase retaining only elements present in kept."""
        kept_set = set(kept)
        new_elements = tuple(e for e in self.elements if e in kept_set)
        return InputCase(
            case_id=str(uuid4()),
            elements=new_elements,
            description=f"{self.description} (subset of {len(new_elements)} elements)",
            metadata=self.metadata,
        )


class TargetResult(BaseModel):
    """Observable outcome returned by a target upon executing an InputCase."""

    model_config = ConfigDict(frozen=True)

    outcome: TargetOutcome = Field(..., description="Observed status: failure or success.")
    output_message: str = Field(default="", description="Textual response or diagnostic output.")
    error_code: str | None = Field(default=None, description="Optional diagnostic error code if failed.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Observed execution latency in milliseconds.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Observable diagnostic telemetry.")

    @property
    def is_failure(self) -> bool:
        """Convenience property for checking failure."""
        return self.outcome == TargetOutcome.FAILURE


class ProbeTrial(BaseModel):
    """Immutable record of an individual probe execution."""

    model_config = ConfigDict(frozen=True)

    trial_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique ID for this trial.")
    experiment_name: str = Field(..., description="Associated experiment name.")
    trial_index: int = Field(..., ge=0, description="Sequential 0-indexed trial number.")
    parent_trial_id: str | None = Field(default=None, description="Parent trial ID from which this intervention branched.")
    intervention_type: str = Field(..., description="Intervention category: baseline, partition, complement, minimality_check, ablation, random_sample.")
    input_case: InputCase = Field(..., description="The exact input case executed.")
    result: TargetResult = Field(..., description="The observed target response.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp of execution.")


class DiscoveredCondition(BaseModel):
    """Result of an intervention-driven minimal trigger search."""

    model_config = ConfigDict(frozen=True)

    subset_elements: tuple[str, ...] = Field(default=(), description="Isolated input elements that trigger failure.")
    status: SearchStatus = Field(..., description="Status of the search process.")
    is_failure: bool = Field(..., description="Whether the discovered subset actually reproduced the failure.")
    is_one_minimal: bool = Field(default=False, description="Whether 1-minimality was explicitly confirmed by removing each element.")
    is_globally_minimal: bool = Field(default=False, description="Whether global minimality across the entire power set was confirmed.")
    verified_by_check: bool = Field(default=False, description="Whether an explicit post-search verification check completed.")
    target_calls_used: int = Field(..., ge=0, description="Total target calls consumed during the search.")
    status_message: str = Field(default="", description="Detailed narrative of search outcome.")


class BaselineComparison(BaseModel):
    """Comparative outcome from a baseline search algorithm."""

    model_config = ConfigDict(frozen=True)

    baseline_name: str = Field(..., description="Name of the baseline: greedy_ablation or seeded_random.")
    discovered_elements: tuple[str, ...] = Field(default=(), description="Elements isolated by this baseline.")
    failure_reproduced: bool = Field(..., description="Whether the baseline's discovered set reproduces the failure.")
    minimality_check_passed: bool = Field(default=False, description="Whether the baseline's result passes the 1-minimality check.")
    target_calls: int = Field(..., ge=0, description="Number of target calls consumed by baseline.")
    irrelevant_elements_remaining: int = Field(default=0, ge=0, description="Count of non-causal distractor elements retained.")
    matches_ground_truth: bool = Field(default=False, description="Whether the isolated elements exactly match ground truth.")
    budget_exhausted: bool = Field(default=False, description="Whether the baseline hit the trial budget before completing.")


class EvaluationResult(BaseModel):
    """Comprehensive evaluation of minimizer and baselines on a single task."""

    model_config = ConfigDict(frozen=True)

    task_name: str = Field(..., description="Name of the benchmark task.")
    description: str = Field(..., description="Task explanation.")
    initial_elements: tuple[str, ...] = Field(..., description="Original input elements before minimization.")
    ground_truth_trigger: tuple[str, ...] = Field(..., description="Planted causal failure condition.")
    minimizer_condition: DiscoveredCondition = Field(..., description="Result produced by delta debugging.")
    baseline_comparisons: tuple[BaselineComparison, ...] = Field(default=(), description="Results from baseline algorithms.")
    matches_ground_truth: bool = Field(..., description="Whether minimizer output exactly equals ground truth.")
    precision: float | None = Field(default=None, ge=0.0, le=1.0, description="Precision of isolated elements relative to ground truth.")
    recall: float | None = Field(default=None, ge=0.0, le=1.0, description="Recall of ground-truth elements in isolated set.")
    excess_elements: tuple[str, ...] = Field(default=(), description="Distractor elements falsely retained.")
    missing_elements: tuple[str, ...] = Field(default=(), description="True causal elements omitted.")


class ExperimentReport(BaseModel):
    """Complete report combining all task evaluations and experiment metadata."""

    model_config = ConfigDict(frozen=True)

    experiment_name: str = Field(..., description="Name of the executed experiment.")
    config: ExperimentConfig = Field(..., description="Configuration used.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Completion timestamp.")
    total_target_calls: int = Field(..., ge=0, description="Sum of all target calls executed.")
    evaluations: tuple[EvaluationResult, ...] = Field(..., description="Per-task evaluation records.")
    summary: str = Field(..., description="Executive narrative summary.")
    limitations: tuple[str, ...] = Field(
        default=(
            "Synthetic testbed targets use deterministic rules; results do not guarantee performance on non-monotonic or stochastic real-world LLMs.",
            "1-minimality guarantees that no single element can be removed without losing failure, but does not guarantee global minimality across the full power set.",
            "No external LLM APIs were called; all results are produced by local synthetic evaluation.",
        ),
        description="Explicit scientific and engineering limitations.",
    )


class StochasticMinimalityStatus(str, Enum):
    """Classification of 1-minimality in a stochastic environment."""

    STATISTICALLY_SUPPORTED = "statistically_supported"
    OBSERVED_ONLY = "observed_only"
    INCONCLUSIVE = "inconclusive"
    NOT_MINIMAL = "not_minimal"
    NOT_APPLICABLE = "not_applicable"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INFRA_FAILURE = "infra_failure"
    VERIFIED_EMPTY = "verified_empty"

    # Nomenclature aliases matching Protocol v1.3 definitions
    SUPPORTED_1_MINIMAL = "statistically_supported"
    OBSERVED_1_MINIMAL = "observed_only"
    VERIFIED_EMPTY_CAUSAL_SET = "verified_empty"


class CandidateEvaluation(BaseModel):
    """Evaluation of a specific candidate subset over multiple stochastic trials."""

    model_config = ConfigDict(frozen=True)

    candidate_elements: tuple[str, ...] = Field(..., description="Candidate subset elements evaluated.")
    trials: int = Field(..., ge=0, description="Total stochastic trials executed.")
    failures: int = Field(..., ge=0, description="Number of observed failure outcomes.")
    failure_rate: float = Field(..., ge=0.0, le=1.0, description="Estimated failure probability k/n.")
    confidence_interval: tuple[float, float] = Field(..., description="Wilson score confidence interval.")


class StochasticDiscoveredCondition(BaseModel):
    """Outcome of stochastic minimal trigger discovery."""

    model_config = ConfigDict(frozen=True)

    subset_elements: tuple[str, ...] = Field(default=(), description="Isolated input elements.")
    status: SearchStatus = Field(..., description="Overall search status.")
    is_failure: bool = Field(..., description="Whether candidate met predefined failure threshold.")
    minimality_status: StochasticMinimalityStatus = Field(..., description="Classification of minimality.")
    candidate_evaluation: CandidateEvaluation = Field(..., description="Statistical evaluation of final candidate.")
    removal_evaluations: dict[str, CandidateEvaluation] = Field(
        default_factory=dict, description="Evaluations of single-element removals."
    )
    target_calls_used: int = Field(..., ge=0, description="Total target calls consumed.")
    search_target_calls: int = Field(default=0, ge=0, description="Target calls spent in search phase.")
    verification_target_calls: int = Field(default=0, ge=0, description="Target calls spent in verification phase.")
    search_model_requests: int = Field(default=0, ge=0, description="Model provider requests spent in search/discovery phase.")
    verification_model_requests: int = Field(default=0, ge=0, description="Model provider requests spent in verification phase.")
    total_model_requests: int = Field(default=0, ge=0, description="Total model provider requests consumed.")
    verification_status: str | None = Field(default=None, description="Execution status of verification.")
    evidence_tier: str | None = Field(default=None, description="Evidence tier ('exploratory' or 'confirmatory').")
    sample_size_rationale: str | None = Field(default=None, description="Sample size rationale ('exploratory_default', 'unresolved_user_specified', etc.).")
    sample_size_validation_status: str = Field(default="not_applicable", description="Validation status of sample size / statistical power ('not_applicable', 'not_implemented', 'unresolved', 'validated', 'range_exhausted', 'invalid_inputs').")
    confirmatory_power_established: bool = Field(default=False, description="Whether confirmatory power was established by an approved sample size rationale.")
    planned_sample_size_adequate: bool = Field(default=False, description="Whether planned sample size was >= required n*.")
    prospective_power_validated: bool = Field(default=False, description="Whether prospective sample size solver validated the design.")
    execution_protocol_adherent: bool = Field(default=True, description="Whether all planned trials completed without crashes/timeouts/missingness.")
    missingness_sensitivity_status: str = Field(default="not_applicable", description="Status of worst-case sensitivity imputation: 'not_applicable', 'robust', 'fragile'.")
    subsets_examined: int = Field(default=0, ge=0, description="Count of distinct candidate subsets evaluated.")
    status_message: str = Field(default="", description="Detailed narrative.")

    @model_validator(mode="after")
    def validate_evidentiary_consistency(self) -> StochasticDiscoveredCondition:
        tier_val = str(self.evidence_tier or "").lower()
        val_status = str(self.sample_size_validation_status or "").lower()

        if self.confirmatory_power_established and tier_val != "confirmatory":
            raise ValueError(
                f"Invalid evidentiary state: confirmatory_power_established=True cannot coexist "
                f"with evidence_tier={self.evidence_tier!r}."
            )

        if self.confirmatory_power_established and val_status in ("unresolved", "not_implemented", "not_applicable", "range_exhausted", "invalid_inputs", ""):
            raise ValueError(
                f"Invalid evidentiary state: confirmatory_power_established=True cannot coexist "
                f"with sample_size_validation_status={self.sample_size_validation_status!r}."
            )

        if self.confirmatory_power_established and not self.planned_sample_size_adequate:
            raise ValueError(
                "Invalid evidentiary claim: confirmatory_power_established cannot be True without "
                "validated prospective power and adequate planned sample size (planned_sample_size_adequate=True)."
            )

        if self.evidence_tier is None and self.verification_status is None and val_status == "validated":
            raise ValueError(
                "Invalid evidentiary state: sample_size_validation_status cannot be 'validated' "
                "when verification has not been performed (evidence_tier and verification_status are None)."
            )

        return self


class StochasticBaselineComparison(BaseModel):
    """Comparative outcome from a stochastic baseline search."""

    model_config = ConfigDict(frozen=True)

    baseline_name: str = Field(..., description="Name of the baseline.")
    discovered_elements: tuple[str, ...] = Field(default=(), description="Elements isolated by baseline.")
    failure_reproduced: bool = Field(..., description="Whether discovered set met failure threshold.")
    minimality_status: StochasticMinimalityStatus = Field(..., description="Stochastic minimality classification.")
    target_calls: int = Field(..., ge=0, description="Target evaluations consumed.")
    search_target_calls: int = Field(default=0, ge=0, description="Target calls spent in search phase.")
    verification_target_calls: int = Field(default=0, ge=0, description="Target calls spent in verification phase.")
    search_model_requests: int = Field(default=0, ge=0, description="Model provider requests spent in search/discovery phase.")
    verification_model_requests: int = Field(default=0, ge=0, description="Model provider requests spent in verification phase.")
    total_model_requests: int = Field(default=0, ge=0, description="Total model provider requests consumed.")
    verification_status: str | None = Field(default=None, description="Execution status of verification.")
    evidence_tier: str | None = Field(default=None, description="Evidence tier ('exploratory' or 'confirmatory').")
    sample_size_validation_status: str = Field(default="not_applicable", description="Validation status of sample size / statistical power ('not_applicable', 'not_implemented', 'unresolved', 'validated', 'range_exhausted', 'invalid_inputs').")
    confirmatory_power_established: bool = Field(default=False, description="Whether confirmatory power was established by an approved sample size rationale.")
    planned_sample_size_adequate: bool = Field(default=False, description="Whether planned sample size was >= required n*.")
    prospective_power_validated: bool = Field(default=False, description="Whether prospective sample size solver validated the design.")
    execution_protocol_adherent: bool = Field(default=True, description="Whether all planned trials completed without crashes/timeouts/missingness.")
    missingness_sensitivity_status: str = Field(default="not_applicable", description="Status of worst-case sensitivity imputation: 'not_applicable', 'robust', 'fragile'.")
    subsets_examined: int = Field(default=0, ge=0, description="Count of distinct candidate subsets evaluated.")
    candidate_evaluation: CandidateEvaluation = Field(..., description="Statistical evaluation of isolated candidate.")
    irrelevant_elements_remaining: int = Field(default=0, ge=0, description="Count of non-causal distractor elements retained.")
    matches_ground_truth: bool = Field(default=False, description="Exact match with planted trigger.")
    budget_exhausted: bool = Field(default=False, description="Whether search budget was exhausted.")

    @model_validator(mode="after")
    def validate_evidentiary_consistency(self) -> StochasticBaselineComparison:
        tier_val = str(self.evidence_tier or "").lower()
        val_status = str(self.sample_size_validation_status or "").lower()

        if self.confirmatory_power_established and tier_val != "confirmatory":
            raise ValueError(
                f"Invalid evidentiary state: confirmatory_power_established=True cannot coexist "
                f"with evidence_tier={self.evidence_tier!r}."
            )

        if self.confirmatory_power_established and val_status in ("unresolved", "not_implemented", "not_applicable", "range_exhausted", "invalid_inputs", ""):
            raise ValueError(
                f"Invalid evidentiary state: confirmatory_power_established=True cannot coexist "
                f"with sample_size_validation_status={self.sample_size_validation_status!r}."
            )

        if self.confirmatory_power_established and not self.planned_sample_size_adequate:
            raise ValueError(
                "Invalid evidentiary claim: confirmatory_power_established cannot be True without "
                "validated prospective power and adequate planned sample size (planned_sample_size_adequate=True)."
            )

        if self.evidence_tier is None and self.verification_status is None and val_status == "validated":
            raise ValueError(
                "Invalid evidentiary state: sample_size_validation_status cannot be 'validated' "
                "when verification has not been performed (evidence_tier and verification_status are None)."
            )

        return self


class StochasticEvaluationResult(BaseModel):
    """Comprehensive evaluation of stochastic minimizer and baselines on a single task."""

    model_config = ConfigDict(frozen=True)

    task_name: str = Field(..., description="Name of the task.")
    description: str = Field(..., description="Task explanation.")
    initial_elements: tuple[str, ...] = Field(..., description="Initial input elements.")
    ground_truth_trigger: tuple[str, ...] = Field(..., description="Planted causal failure condition.")
    base_probability: float = Field(..., ge=0.0, le=1.0, description="Background failure probability.")
    trigger_probability: float = Field(..., ge=0.0, le=1.0, description="True failure probability when triggered.")
    minimizer_condition: StochasticDiscoveredCondition = Field(..., description="Minimizer result.")
    baseline_comparisons: tuple[StochasticBaselineComparison, ...] = Field(
        default=(), description="Baseline results."
    )
    matches_ground_truth: bool = Field(..., description="Whether minimizer output matches ground truth.")
    precision: float | None = Field(default=None, ge=0.0, le=1.0, description="Precision.")
    recall: float | None = Field(default=None, ge=0.0, le=1.0, description="Recall.")
    excess_elements: tuple[str, ...] = Field(default=(), description="Distractor elements falsely retained.")
    missing_elements: tuple[str, ...] = Field(default=(), description="True causal elements omitted.")


class ExecutionTier(str, Enum):
    """Tier of experimental target execution."""

    SYNTHETIC = "synthetic"
    FAKE_PROVIDER = "fake_provider"
    LIVE_MODEL = "live_model"
    LOCAL_MODEL = "local_model"


class StochasticExperimentReport(BaseModel):
    """Complete report combining all stochastic task evaluations, power metrics, and aggregate comparisons."""

    model_config = ConfigDict(frozen=True)

    experiment_name: str = Field(..., description="Experiment name.")
    execution_tier: ExecutionTier = Field(
        default=ExecutionTier.SYNTHETIC, description="Tier: synthetic, fake_provider, live_model, or local_model."
    )
    config: ExperimentConfig = Field(..., description="Experiment config.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Completion timestamp.")
    trials_per_candidate: int = Field(..., ge=1, description="Number of trials executed per candidate subset.")
    total_target_calls: int = Field(..., ge=0, description="Sum of all target calls.")
    evaluations: tuple[StochasticEvaluationResult, ...] = Field(..., description="Per-task evaluation records.")
    aggregate_metrics: dict[str, Any] = Field(default_factory=dict, description="Aggregate recovery, precision, and verification statistics.")
    summary: str = Field(..., description="Executive narrative summary.")
    limitations: tuple[str, ...] = Field(
        default=(
            "Stochastic synthetic targets evaluate configurable binomial probabilities; they do not model complex semantic or non-monotonic prompt drift in real-world LLMs.",
            "Observed 1-minimality distinguishes cases where removal empirically drops failure from statistically supported cases where confidence bounds confirm the drop.",
            "Theoretical power is an asymptotic pre-experimental calculation (Cohen's arcsine approximation) for sample size planning; it does not measure empirical post-hoc power.",
            "Berger's (1982) Intersection-Union Test controls Type I error for candidate 1-minimality across single-element ablations; it does not account for adaptive candidate selection along the prior delta-debugging search path.",
            "No external LLM APIs were called; all results were generated locally in the isolated stochastic testbed.",
        ),
        description="Explicit scientific and engineering limitations.",
    )


class ModelPilotEvaluationResult(BaseModel):
    """Comprehensive evaluation record of a model-backed agent pilot task."""

    model_config = ConfigDict(frozen=True)

    task_name: str = Field(..., description="Name of the pilot task.")
    description: str = Field(..., description="Task explanation.")
    initial_elements: tuple[str, ...] = Field(..., description="Initial input elements.")
    ground_truth_trigger: tuple[str, ...] = Field(..., description="Planted causal failure condition.")
    is_null_case: bool = Field(default=False, description="Whether this is a non-failing control task.")
    minimizer_condition: StochasticDiscoveredCondition = Field(..., description="Minimizer result.")
    baseline_comparisons: tuple[StochasticBaselineComparison, ...] = Field(
        default=(), description="Baseline comparison results."
    )
    matches_ground_truth: bool = Field(..., description="Whether minimizer output matches ground truth.")
    precision: float | None = Field(default=None, description="Precision (None if undefined for null case).")
    recall: float | None = Field(default=None, description="Recall (None if undefined for null case).")
    excess_elements: tuple[str, ...] = Field(default=(), description="Distractor elements falsely retained.")
    missing_elements: tuple[str, ...] = Field(default=(), description="True causal elements omitted.")


class ModelPilotExperimentReport(BaseModel):
    """Complete experimental report for model-backed agent pilots."""

    model_config = ConfigDict(frozen=True)

    experiment_name: str = Field(..., description="Experiment identifier.")
    execution_tier: ExecutionTier = Field(..., description="Tier: fake_provider, live_model, or local_model.")
    provider_name: str = Field(..., description="Model provider used (e.g. fake, openai).")
    model_name: str = Field(..., description="Model identifier (e.g. fake-model-pilot, gpt-4o-mini).")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Completion timestamp.")
    config: ExperimentConfig = Field(..., description="Experiment configuration.")
    budget_summary: dict[str, Any] = Field(..., description="Model requests, spend USD, and token accounting.")
    trials_per_candidate: int = Field(..., ge=1, description="Number of trials per candidate subset.")
    total_target_calls: int = Field(..., ge=0, description="Sum of all target calls.")
    evaluations: tuple[ModelPilotEvaluationResult, ...] = Field(..., description="Per-task evaluation records.")
    aggregate_metrics: dict[str, Any] = Field(default_factory=dict, description="Aggregate accuracy and recovery.")
    summary: str = Field(..., description="Executive narrative summary.")
    limitations: tuple[str, ...] = Field(
        default=(
            "Model-backed pilot evaluations reflect a controlled local harness with hermetic mock tools; they do not imply capability to reverse-engineer arbitrary black-box commercial agents.",
            "When using FakeProvider, response rules are programmed via heuristic string matches; fake-provider results validate harness mechanics and search integration, NOT real-world neural LLM diagnostic capability.",
            "Reported Fisher exact p-values and risk differences represent exploratory post-selection metrics along an adaptive delta-debugging search path, not confirmatory hypothesis tests on independent validation samples.",
            "For null cases (where true trigger and predicted trigger are empty), precision and recall are mathematically undefined (0/0) and reported as N/A.",
            "Planted failure triggers are controlled experimental interventions; they do not reveal hidden neural weights or unobservable internal model chain-of-thought.",
            "Model stochasticity and temperature drift can introduce run-to-run outcome variability.",
            "Berger's (1982) Intersection-Union Test controls Type I error for candidate 1-minimality across single-element ablations; adaptive selection bias (winner's curse) along the search path remains a factor.",
            "Estimated spend calculations rely on published standard per-token pricing tables and represent estimates, not provider billing invoices.",
        ),
        description="Explicit scientific and engineering limitations.",
    )


class DiagnosticOutcome(BaseModel):
    """Normalized diagnostic result for a single fault localization method on a single mechanism."""

    model_config = ConfigDict(frozen=True)

    mechanism_id: str = Field(..., description="Unique identifier of the independent mechanism.")
    method_name: str = Field(..., description="Method evaluated: 'active_dd', 'greedy_ablation', or 'llm_judge'.")
    predicted_candidate_set: tuple[str, ...] | None = Field(
        default=None,
        description="Isolated candidate element IDs, empty tuple () for null prediction, or None if no evaluable set produced.",
    )
    is_valid_diagnosis: bool = Field(
        ...,
        description="True if method produced an evaluable candidate set (including ()). False if unparseable or aborted.",
    )
    false_attribution_indicator: int | None = Field(
        default=None,
        description="Primary binary metric Y in {0, 1}. None (NA) if and only if is_valid_diagnosis is False.",
    )
    exact_recovery_indicator: int | None = Field(
        default=None,
        description="Secondary metric R_exact in {0, 1}. 1 if predicted_candidate_set == S_true, 0 if != S_true, None if infra failure.",
    )
    verification_status: str = Field(
        default="not_applicable",
        description="Classification: 'statistically_supported', 'observed_only', 'verified_empty', 'inconclusive', 'not_verified', or 'not_applicable'.",
    )
    termination_reason: str = Field(
        default="normal_completion",
        description="Reason: 'normal_completion', 'budget_exhausted', 'malformed_output', 'timeout', or 'infra_failure'.",
    )
    target_calls_used: int = Field(default=0, ge=0, description="Target calls consumed by this method.")
    model_requests: int = Field(default=0, ge=0, description="Model provider requests consumed.")
    prompt_tokens: int = Field(default=0, ge=0, description="Prompt tokens consumed.")
    completion_tokens: int = Field(default=0, ge=0, description="Completion tokens consumed.")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens consumed.")
    estimated_cost_usd: float = Field(default=0.0, ge=0.0, description="Estimated USD expenditure.")
    diagnostics: str = Field(default="", description="Human-readable execution diagnostics.")


class LLMJudgeResult(BaseModel):
    """Structured result of passive LLM judge fault localization."""

    model_config = ConfigDict(frozen=True)

    predicted_candidate_set: tuple[str, ...] | None = Field(default=None, description="Culprit element IDs, () if empty, or None.")
    is_valid_diagnosis: bool = Field(default=False, description="Whether output conformed to schema and candidate IDs.")
    attempts_used: int = Field(default=1, ge=0, le=3, description="Total prompt attempts executed (1 + up to 2 retries).")
    termination_reason: str = Field(default="normal_completion", description="Termination status.")
    prompt_tokens: int = Field(default=0, ge=0, description="Prompt tokens consumed across attempts.")
    completion_tokens: int = Field(default=0, ge=0, description="Completion tokens consumed across attempts.")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens consumed.")
    estimated_cost_usd: float = Field(default=0.0, ge=0.0, description="Estimated USD cost.")
    rationale: str = Field(default="", description="Explanatory rationale emitted by judge.")
    raw_responses: tuple[str, ...] = Field(default=(), description="Raw text responses across all attempts.")


class LLMJudgeComparison(BaseModel):
    """Comparative outcome from passive LLM judge baseline matching standard baseline schemas."""

    model_config = ConfigDict(frozen=True)

    baseline_name: str = Field(default="llm_judge", description="Baseline identifier.")
    discovered_elements: tuple[str, ...] = Field(default=(), description="Elements isolated by judge.")
    predicted_candidate_set: tuple[str, ...] | None = Field(default=None, description="Formal candidate prediction.")
    is_valid_diagnosis: bool = Field(default=False, description="Whether diagnosis is valid.")
    false_attribution_indicator: int | None = Field(default=None, description="False-attribution indicator Y.")
    exact_recovery_indicator: int | None = Field(default=None, description="Exact recovery indicator R_exact.")
    attempts_used: int = Field(default=1, ge=1, le=3, description="Attempts consumed.")
    total_model_requests: int = Field(default=1, ge=0, description="Total model requests.")
    prompt_tokens: int = Field(default=0, ge=0, description="Prompt tokens.")
    completion_tokens: int = Field(default=0, ge=0, description="Completion tokens.")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens.")
    estimated_cost_usd: float = Field(default=0.0, ge=0.0, description="Estimated USD spend.")
    termination_reason: str = Field(default="normal_completion", description="Termination reason.")
    matches_ground_truth: bool = Field(default=False, description="Whether discovered elements match ground truth.")
    rationale: str = Field(default="", description="Explanation emitted by judge.")
    raw_output: str = Field(default="", description="Raw output text.")


class MechanismEvaluationResult(BaseModel):
    """Three-way evaluation result on an independent benchmark mechanism."""

    model_config = ConfigDict(frozen=True)

    mechanism_id: str = Field(..., description="Unique mechanism identifier.")
    mechanism_name: str = Field(..., description="Human-readable mechanism name.")
    category: str = Field(..., description="Category: prompt, schema, environment, conjunctive, null_clean, null_bloat, null_ext.")
    is_null_case: bool = Field(default=False, description="Whether this is a negative null control.")
    is_conjunctive: bool = Field(default=False, description="Whether failure requires multiple interacting elements.")
    initial_elements: tuple[str, ...] = Field(..., description="All candidate elements in initial input.")
    ground_truth_trigger: tuple[str, ...] = Field(..., description="Planted ground-truth causal elements.")
    active_dd_outcome: DiagnosticOutcome = Field(..., description="Outcome from Active Delta Debugging.")
    greedy_ablation_outcome: DiagnosticOutcome = Field(..., description="Outcome from Active Greedy Ablation.")
    llm_judge_outcome: DiagnosticOutcome = Field(..., description="Outcome from Passive LLM Judge.")
    initial_trace_requests: int = Field(default=1, ge=0, description="Requests consumed to generate baseline failing trace.")
    initial_trace_tokens: int = Field(default=0, ge=0, description="Tokens consumed to generate baseline trace.")
    initial_trace_cost_usd: float = Field(default=0.0, ge=0.0, description="Cost of initial trace generation.")


class ExploratoryBenchmarkReport(BaseModel):
    """Consolidated experimental report for Phase 4.6 exploratory benchmark."""

    model_config = ConfigDict(frozen=True)

    experiment_name: str = Field(..., description="Experiment name.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Completion timestamp.")
    n_total: int = Field(default=18, ge=1, description="Total planned mechanisms.")
    n_infra: int = Field(default=0, ge=0, description="Mechanisms excluded due to unresolved infrastructure failures.")
    n_evaluable: int = Field(default=18, ge=1, description="Retained analysis cohort denominator (n_total - n_infra).")
    n_complete: int = Field(default=18, ge=0, description="Jointly evaluable complete pairs.")
    n_miss_judge: int = Field(default=0, ge=0, description="Missing diagnoses for judge (malformed output).")
    n_miss_activedd: int = Field(default=0, ge=0, description="Missing diagnoses for active DD (pre-candidate exhaustion).")
    complete_case_risk_difference: float | None = Field(default=None, description="Observed Delta_Risk on complete cases.")
    mcnemar_b: int = Field(default=0, ge=0, description="Discordant pairs: Judge false attribution, ActiveDD correct.")
    mcnemar_c: int = Field(default=0, ge=0, description="Discordant pairs: Judge correct, ActiveDD false attribution.")
    mcnemar_n_discordant: int = Field(default=0, ge=0, description="Total discordant pairs b + c.")
    mcnemar_odds_ratio: float | None = Field(default=None, description="Discordant odds ratio b / c.")
    mcnemar_p_value: float = Field(default=1.0, ge=0.0, le=1.0, description="Exact directional McNemar p-value.")
    mcnemar_statistically_significant: bool = Field(default=False, description="Whether p_value <= alpha.")
    sensitivity_upper_bound: float = Field(default=0.0, description="Upper bound on Delta_Risk on evaluable cohort.")
    sensitivity_lower_bound: float = Field(default=0.0, description="Lower bound on Delta_Risk on evaluable cohort.")
    evaluations: tuple[MechanismEvaluationResult, ...] = Field(default=(), description="Per-mechanism evaluation results.")
    secondary_metrics: dict[str, Any] = Field(default_factory=dict, description="Secondary precision, recall, cost, and FDR metrics.")
    summary: str = Field(default="", description="Executive narrative summary.")
    limitations: tuple[str, ...] = Field(
        default=(
            "The 18-mechanism benchmark is an exploratory pilot; sample size is insufficient to guarantee confirmatory power under modest discordance rates.",
            "Reported McNemar p-value and odds ratio evaluate differences across the realized sample of mechanisms; generalization beyond the testbed requires defensible assumptions about etiological independence and representativeness.",
            "Sensitivity bounds [Delta_Risk_min, Delta_Risk_max] describe the retained benchmark cohort under extreme missingness; they do not automatically bound target-population risk if infrastructure attrition is non-random.",
            "A non-false-attribution outcome (Y=0) denotes the absence of false attribution (e.g., no spurious elements claimed), but on active failure tasks where no causal element is isolated, it does not represent diagnostic success (R_exact=0). False attribution risk and exact identification recall must be evaluated jointly.",
            "FakeProvider evaluations validate software interfaces, JSON parsing, error recovery, and metric math; they do not measure real-world LLM diagnostic reasoning.",
        ),
        description="Explicit scientific and engineering limitations.",
    )


