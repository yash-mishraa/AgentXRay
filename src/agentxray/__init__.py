"""AgentXRay: Controlled experimentation and minimal failure-trigger discovery for AI agents."""

__version__ = "0.1.0"

from agentxray.models import (
    AccessMode,
    BaselineComparison,
    DiscoveredCondition,
    EvaluationResult,
    ExperimentConfig,
    ExperimentReport,
    InputCase,
    ProbeTrial,
    SearchStatus,
    TargetOutcome,
    TargetResult,
)
from agentxray.runner import BudgetExhaustedError, InterventionRunner, TimeoutExceededError
from agentxray.minimizer import DeltaDebugMinimizer
from agentxray.baselines import run_greedy_ablation, run_seeded_random_search
from agentxray.reporting import format_json_report, format_markdown_report, write_reports
from agentxray.api import DiagnosticResult, DiagnosticStatus, diagnose

__all__ = [
    "__version__",
    "AccessMode",
    "BaselineComparison",
    "BudgetExhaustedError",
    "DeltaDebugMinimizer",
    "DiagnosticResult",
    "DiagnosticStatus",
    "DiscoveredCondition",
    "EvaluationResult",
    "ExperimentConfig",
    "ExperimentReport",
    "InputCase",
    "InterventionRunner",
    "ProbeTrial",
    "SearchStatus",
    "TargetOutcome",
    "TargetResult",
    "TimeoutExceededError",
    "diagnose",
    "format_json_report",
    "format_markdown_report",
    "run_greedy_ablation",
    "run_seeded_random_search",
    "write_reports",
]
