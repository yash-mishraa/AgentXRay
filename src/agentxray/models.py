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
    elements: tuple[str, ...] = Field(..., min_length=1, description="Ordered discrete elements forming the input.")
    description: str = Field(default="", description="Optional human-readable description.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata associated with this input.")

    @field_validator("elements", mode="before")
    @classmethod
    def validate_elements(cls, v: Any) -> tuple[str, ...]:
        if not v:
            raise ValueError("InputCase elements cannot be empty")
        if isinstance(v, (list, set, tuple)):
            clean = [str(elem).strip() for elem in v if str(elem).strip()]
            if not clean:
                raise ValueError("InputCase must contain at least one non-empty string element")
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
        if not new_elements:
            raise ValueError(f"Cannot remove {element!r}: would result in empty InputCase")
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
        if not new_elements:
            raise ValueError("Subset must contain at least one element")
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
    irrelevant_elements_remaining: int = Field(..., ge=0, description="Count of non-causal distractor elements retained.")
    matches_ground_truth: bool = Field(..., description="Whether the isolated elements exactly match ground truth.")
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
    precision: float = Field(..., ge=0.0, le=1.0, description="Precision of isolated elements relative to ground truth.")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall of ground-truth elements in isolated set.")
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
    status_message: str = Field(default="", description="Detailed narrative.")


class StochasticBaselineComparison(BaseModel):
    """Comparative outcome from a stochastic baseline search."""

    model_config = ConfigDict(frozen=True)

    baseline_name: str = Field(..., description="Name of the baseline.")
    discovered_elements: tuple[str, ...] = Field(default=(), description="Elements isolated by baseline.")
    failure_reproduced: bool = Field(..., description="Whether discovered set met failure threshold.")
    minimality_status: StochasticMinimalityStatus = Field(..., description="Stochastic minimality classification.")
    target_calls: int = Field(..., ge=0, description="Target evaluations consumed.")
    candidate_evaluation: CandidateEvaluation = Field(..., description="Statistical evaluation of isolated candidate.")
    irrelevant_elements_remaining: int = Field(..., ge=0, description="Count of non-causal distractor elements retained.")
    matches_ground_truth: bool = Field(..., description="Exact match with planted trigger.")
    budget_exhausted: bool = Field(default=False, description="Whether search budget was exhausted.")


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
    precision: float = Field(..., ge=0.0, le=1.0, description="Precision.")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall.")
    excess_elements: tuple[str, ...] = Field(default=(), description="Distractor elements falsely retained.")
    missing_elements: tuple[str, ...] = Field(default=(), description="True causal elements omitted.")


class StochasticExperimentReport(BaseModel):
    """Complete report combining all stochastic task evaluations, power metrics, and aggregate comparisons."""

    model_config = ConfigDict(frozen=True)

    experiment_name: str = Field(..., description="Experiment name.")
    config: ExperimentConfig = Field(..., description="Experiment config.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Completion timestamp.")
    trials_per_candidate: int = Field(..., ge=1, description="Number of trials executed per candidate subset.")
    total_target_calls: int = Field(..., ge=0, description="Sum of all target calls.")
    evaluations: tuple[StochasticEvaluationResult, ...] = Field(..., description="Per-task evaluation records.")
    aggregate_metrics: dict[str, Any] = Field(default_factory=dict, description="Aggregate recovery, precision, FDR.")
    summary: str = Field(..., description="Executive narrative summary.")
    limitations: tuple[str, ...] = Field(
        default=(
            "Stochastic synthetic targets evaluate configurable binomial probabilities; they do not model complex semantic or non-monotonic prompt drift in real-world LLMs.",
            "Observed 1-minimality distinguishes cases where removal empirically drops failure from statistically supported cases where confidence bounds confirm the drop.",
            "Theoretical power is an asymptotic pre-experimental calculation (Cohen's arcsine approximation) for sample size planning; it does not measure empirical post-hoc power.",
            "Benjamini-Hochberg FDR control applies to the family of candidate ablation tests; it does not account for adaptive candidate selection along the prior delta-debugging search path.",
            "No external LLM APIs were called; all results were generated locally in the isolated stochastic testbed.",
        ),
        description="Explicit scientific and engineering limitations.",
    )

