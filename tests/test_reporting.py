"""Tests for JSON and Markdown report generation in agentxray.reporting."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from agentxray.__main__ import run_synthetic_benchmark
from agentxray.models import ExperimentReport
from agentxray.reporting import format_json_report, format_markdown_report, write_reports


class TestReporting:
    """Tests for reporting generation and file output."""

    def test_benchmark_report_generation(self, tmp_path: Path) -> None:
        report = run_synthetic_benchmark(
            max_trials=50,
            timeout_seconds=10.0,
            seed=42,
            output_dir=tmp_path,
        )

        assert isinstance(report, ExperimentReport)
        assert len(report.evaluations) == 4
        assert report.total_target_calls > 0

        # Verify JSON report on disk
        json_file = tmp_path / "report.json"
        assert json_file.exists()
        raw_json = json_file.read_text(encoding="utf-8")
        parsed = json.loads(raw_json)
        assert parsed["experiment_name"] == "Phase1A_Deterministic_Benchmark"
        assert len(parsed["evaluations"]) == 4

        # Verify Markdown report on disk
        md_file = tmp_path / "report.md"
        assert md_file.exists()
        raw_md = md_file.read_text(encoding="utf-8")
        assert "# AgentXRay Experiment Report" in raw_md
        assert "Benchmark Comparison Matrix" in raw_md
        assert "Limitations and Reproducibility Notice" in raw_md
        assert "Single-Factor Trigger" in raw_md
        assert "Two-Factor Conjunctive Trigger" in raw_md

    def test_json_roundtrip_validity(self, tmp_path: Path) -> None:
        report = run_synthetic_benchmark(
            max_trials=20,
            timeout_seconds=5.0,
            seed=42,
            output_dir=tmp_path,
        )
        json_str = format_json_report(report)
        reloaded = ExperimentReport.model_validate_json(json_str)
        assert reloaded.experiment_name == report.experiment_name
        assert len(reloaded.evaluations) == len(report.evaluations)
        assert reloaded.total_target_calls == report.total_target_calls

    def test_no_hardcoded_results_in_report(self, tmp_path: Path) -> None:
        """Verify that total_target_calls matches the exact sum of individual evaluations."""
        report = run_synthetic_benchmark(
            max_trials=30,
            timeout_seconds=5.0,
            seed=42,
            output_dir=tmp_path,
        )
        calculated_calls = sum(
            e.minimizer_condition.target_calls_used + sum(b.target_calls for b in e.baseline_comparisons)
            for e in report.evaluations
        )
        assert report.total_target_calls == calculated_calls
