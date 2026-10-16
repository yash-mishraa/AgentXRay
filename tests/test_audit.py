"""Tests for the multi-seed stochastic audit and reporting module."""

from __future__ import annotations

from pathlib import Path
import pytest

from agentxray.audit import (
    format_audit_markdown_report,
    run_multiseed_audit,
    write_audit_reports,
)


def test_run_multiseed_audit_structure():
    """Verify multi-seed audit produces expected records and summaries across seeds."""
    report = run_multiseed_audit(
        seeds=(42, 43),
        max_trials=100,
        trials_per_candidate=10,
        failure_threshold=0.30,
        alpha=0.05,
        confidence=0.95,
    )

    assert len(report.seeds) == 2
    assert len(report.aggregate_metrics) == 3  # delta_debugging, greedy_ablation, random_search
    # 2 seeds * 6 tasks * 3 methods = 36 records
    assert len(report.records) == 36
    # 6 tasks * 3 methods = 18 scenario summaries
    assert len(report.per_scenario_metrics) == 18

    # Check aggregate metrics properties
    for agg in report.aggregate_metrics:
        assert agg.total_runs == 12  # 2 seeds * 6 tasks
        assert 0.0 <= agg.match_rate <= 1.0
        assert 0.0 <= agg.match_rate_ci[0] <= agg.match_rate_ci[1] <= 1.0
        assert agg.null_case_runs == 2
        assert 0.0 <= agg.null_case_false_positive_rate <= 1.0


def test_format_audit_markdown_report():
    """Verify Markdown report generation contains essential headings and tables."""
    report = run_multiseed_audit(
        seeds=(42,),
        max_trials=60,
        trials_per_candidate=10,
        failure_threshold=0.30,
    )
    md = format_audit_markdown_report(report)

    assert "# AgentXRay: Stochastic Multi-Seed Audit & Statistical Validation Report" in md
    assert "## 1. Executive Summary & Aggregate Performance" in md
    assert "## 2. Statistical Minimality Classification Distribution" in md
    assert "## 3. Per-Scenario Performance Breakdown Across Prespecified Seeds" in md
    assert "## 4. Rigorous Comparative Analysis & Trade-Offs" in md
    assert "## 5. Audit Scope & Epistemic Limitations" in md
    assert "Delta Debugging" in md
    assert "Greedy Ablation" in md
    assert "Random Search" in md


def test_write_audit_reports(tmp_path: Path):
    """Verify JSON and Markdown files are correctly created on disk."""
    report = run_multiseed_audit(
        seeds=(42,),
        max_trials=60,
        trials_per_candidate=10,
    )
    json_path = tmp_path / "audit.json"
    md_path = tmp_path / "audit.md"

    write_audit_reports(report, json_path=json_path, md_path=md_path)

    assert json_path.exists()
    assert md_path.exists()
    assert len(json_path.read_text(encoding="utf-8")) > 0
    assert len(md_path.read_text(encoding="utf-8")) > 0
