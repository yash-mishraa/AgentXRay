"""Tests for stochastic report generation and JSON/Markdown formatting."""

from __future__ import annotations

from pathlib import Path
import pytest

from agentxray.models import (
    AccessMode,
    CandidateEvaluation,
    ExperimentConfig,
    SearchStatus,
    StochasticBaselineComparison,
    StochasticDiscoveredCondition,
    StochasticEvaluationResult,
    StochasticExperimentReport,
    StochasticMinimalityStatus,
)
from agentxray.reporting import (
    format_stochastic_json_report,
    format_stochastic_markdown_report,
    write_stochastic_reports,
)


def _build_dummy_stochastic_report() -> StochasticExperimentReport:
    """Build a minimal valid StochasticExperimentReport for testing."""
    config = ExperimentConfig(
        experiment_name="test_report_exp",
        max_trials=100,
        mode=AccessMode.CONTROLLED_TESTBED,
    )

    cond = StochasticDiscoveredCondition(
        subset_elements=("param_a",),
        status=SearchStatus.COMPLETE,
        is_failure=True,
        minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
        candidate_evaluation=CandidateEvaluation(
            candidate_elements=("param_a",),
            trials=20,
            failures=18,
            failure_rate=0.90,
            confidence_interval=(0.6987, 0.9721),
        ),
        target_calls_used=40,
        status_message="Minimality verified.",
    )

    base = StochasticBaselineComparison(
        baseline_name="greedy",
        discovered_elements=("param_a",),
        failure_reproduced=True,
        minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
        target_calls=60,
        candidate_evaluation=CandidateEvaluation(
            candidate_elements=("param_a",),
            trials=20,
            failures=18,
            failure_rate=0.90,
            confidence_interval=(0.6987, 0.9721),
        ),
        irrelevant_elements_remaining=0,
        matches_ground_truth=True,
    )

    eval_res = StochasticEvaluationResult(
        task_name="Dummy Task",
        description="A dummy test task",
        initial_elements=("param_a", "distractor_b"),
        ground_truth_trigger=("param_a",),
        base_probability=0.05,
        trigger_probability=0.85,
        minimizer_condition=cond,
        baseline_comparisons=(base,),
        matches_ground_truth=True,
        precision=1.0,
        recall=1.0,
        excess_elements=(),
        missing_elements=(),
    )

    return StochasticExperimentReport(
        experiment_name=config.experiment_name,
        config=config,
        trials_per_candidate=20,
        total_target_calls=100,
        evaluations=(eval_res,),
        aggregate_metrics={"match_rate": 1.0},
        summary="Test summary",
    )


def test_stochastic_json_roundtrip():
    """Verify stochastic report serializes and deserializes accurately."""
    report = _build_dummy_stochastic_report()
    json_str = format_stochastic_json_report(report)
    assert "test_report_exp" in json_str
    assert "statistically_supported" in json_str

    deserialized = StochasticExperimentReport.model_validate_json(json_str)
    assert deserialized.experiment_name == report.experiment_name
    assert deserialized.total_target_calls == report.total_target_calls
    assert len(deserialized.evaluations) == 1
    assert deserialized.evaluations[0].matches_ground_truth is True


def test_stochastic_markdown_structure():
    """Verify markdown output contains required tables, sections, and power estimates."""
    report = _build_dummy_stochastic_report()
    md = format_stochastic_markdown_report(report)

    assert "# AgentXRay Stochastic Experiment Report" in md
    assert "## 1. Executive Summary" in md
    assert "## 2. Benchmark Comparison Matrix" in md
    assert "## 3. Detailed Statistical & Uncertainty Evaluations" in md
    assert "## 4. Limitations and Reproducibility Notice" in md
    assert "Statistical Power (at n=20, alpha=0.05)" in md
    assert "Wilson CI" in md
    assert "statistically_supported" in md


def test_write_stochastic_reports(tmp_path: Path):
    """Verify write_stochastic_reports writes both JSON and Markdown files to disk."""
    report = _build_dummy_stochastic_report()
    json_file = tmp_path / "reports" / "test.json"
    md_file = tmp_path / "reports" / "test.md"

    write_stochastic_reports(report, json_path=json_file, md_path=md_file)

    assert json_file.exists()
    assert md_file.exists()
    assert len(json_file.read_text(encoding="utf-8")) > 0
    assert len(md_file.read_text(encoding="utf-8")) > 0
