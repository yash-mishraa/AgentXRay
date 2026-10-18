"""Tests for JSON and Markdown report generation in agentxray.reporting."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from agentxray.__main__ import run_synthetic_benchmark
from agentxray.models import ExperimentReport
from agentxray.reporting import (
    compute_stochastic_aggregate_metrics,
    format_json_report,
    format_markdown_report,
    write_reports,
)


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


class TestModelPilotReportingIntegrity:
    """Regression tests for Phase 4.3B: trigger label formatting, IUT descriptions, and aggregate metrics."""

    def test_model_pilot_trigger_label_empty_active_vs_null(self) -> None:
        """Verify *(none)* is rendered for active search failures and *(None - Null Case)* for genuine null controls."""
        from agentxray.models import (
            CandidateEvaluation,
            ExecutionTier,
            ExperimentConfig,
            ModelPilotEvaluationResult,
            ModelPilotExperimentReport,
            SearchStatus,
            StochasticBaselineComparison,
            StochasticDiscoveredCondition,
            StochasticMinimalityStatus,
        )
        from agentxray.reporting import format_model_pilot_markdown_report

        dummy_cand = CandidateEvaluation(
            candidate_elements=(),
            trials=5,
            failures=0,
            failure_rate=0.0,
            confidence_interval=(0.0, 0.43),
        )

        # 1. Active task where minimizer and baseline found nothing (search failure)
        active_empty_eval = ModelPilotEvaluationResult(
            task_name="Active Task Empty Discovery",
            description="Active task where discovery failed",
            initial_elements=("elem_a", "elem_b"),
            ground_truth_trigger=("elem_a",),
            is_null_case=False,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.COMPLETE,
                is_failure=False,
                minimality_status=StochasticMinimalityStatus.NOT_MINIMAL,
                candidate_evaluation=dummy_cand,
                target_calls_used=10,
            ),
            baseline_comparisons=(
                StochasticBaselineComparison(
                    baseline_name="greedy",
                    discovered_elements=(),
                    failure_reproduced=False,
                    minimality_status=StochasticMinimalityStatus.NOT_MINIMAL,
                    target_calls=8,
                    candidate_evaluation=dummy_cand,
                    irrelevant_elements_remaining=0,
                    matches_ground_truth=False,
                ),
            ),
            matches_ground_truth=False,
            precision=0.0,
            recall=0.0,
            excess_elements=(),
            missing_elements=("elem_a",),
        )

        # 2. Control null task where minimizer and baseline found nothing (correct discovery)
        null_empty_eval = ModelPilotEvaluationResult(
            task_name="Control Null Task",
            description="Control task with no failure trigger",
            initial_elements=("elem_c", "elem_d"),
            ground_truth_trigger=(),
            is_null_case=True,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.COMPLETE,
                is_failure=False,
                minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
                candidate_evaluation=dummy_cand,
                target_calls_used=6,
            ),
            baseline_comparisons=(
                StochasticBaselineComparison(
                    baseline_name="greedy",
                    discovered_elements=(),
                    failure_reproduced=False,
                    minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
                    target_calls=6,
                    candidate_evaluation=dummy_cand,
                    irrelevant_elements_remaining=0,
                    matches_ground_truth=True,
                ),
            ),
            matches_ground_truth=True,
            precision=None,
            recall=None,
            excess_elements=(),
            missing_elements=(),
        )

        # 3. Active task with recovered trigger
        active_found_eval = ModelPilotEvaluationResult(
            task_name="Active Task Found",
            description="Active task where trigger was recovered",
            initial_elements=("elem_e", "distractor_f"),
            ground_truth_trigger=("elem_e",),
            is_null_case=False,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=("elem_e",),
                status=SearchStatus.COMPLETE,
                is_failure=True,
                minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
                candidate_evaluation=CandidateEvaluation(
                    candidate_elements=("elem_e",),
                    trials=5,
                    failures=5,
                    failure_rate=1.0,
                    confidence_interval=(0.57, 1.0),
                ),
                target_calls_used=12,
            ),
            baseline_comparisons=(
                StochasticBaselineComparison(
                    baseline_name="greedy",
                    discovered_elements=("elem_e",),
                    failure_reproduced=True,
                    minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
                    target_calls=10,
                    candidate_evaluation=CandidateEvaluation(
                        candidate_elements=("elem_e",),
                        trials=5,
                        failures=5,
                        failure_rate=1.0,
                        confidence_interval=(0.57, 1.0),
                    ),
                    irrelevant_elements_remaining=0,
                    matches_ground_truth=True,
                ),
            ),
            matches_ground_truth=True,
            precision=1.0,
            recall=1.0,
            excess_elements=(),
            missing_elements=(),
        )

        # 4. Null task with false positive discovery
        null_fp_eval = ModelPilotEvaluationResult(
            task_name="Control Null False Alarm",
            description="Control task with false alarm discovery",
            initial_elements=("elem_g",),
            ground_truth_trigger=(),
            is_null_case=True,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=("elem_g",),
                status=SearchStatus.COMPLETE,
                is_failure=True,
                minimality_status=StochasticMinimalityStatus.OBSERVED_ONLY,
                candidate_evaluation=CandidateEvaluation(
                    candidate_elements=("elem_g",),
                    trials=5,
                    failures=3,
                    failure_rate=0.6,
                    confidence_interval=(0.23, 0.88),
                ),
                target_calls_used=8,
            ),
            baseline_comparisons=(),
            matches_ground_truth=False,
            precision=0.0,
            recall=None,
            excess_elements=("elem_g",),
            missing_elements=(),
        )

        report = ModelPilotExperimentReport(
            experiment_name="test_label_formatting",
            execution_tier=ExecutionTier.FAKE_PROVIDER,
            provider_name="fake",
            model_name="fake-model",
            config=ExperimentConfig(experiment_name="test_label_formatting", max_trials=50),
            budget_summary={},
            trials_per_candidate=5,
            total_target_calls=42,
            evaluations=(active_empty_eval, null_empty_eval, active_found_eval, null_fp_eval),
            aggregate_metrics={},
            summary="Label test summary",
        )

        md = format_model_pilot_markdown_report(report)

        # Active empty: must be *(none)*, not *(None - Null Case)*
        assert "| **Active Task Empty Discovery** | Stochastic Delta Debugging | *(none)* | `not_minimal` | 10 | 0 | **NO** |" in md
        assert "| | *greedy* | *(none)* | `not_minimal` | 8 | 0 | NO |" in md

        # Null empty: must be *(None - Null Case)*
        assert "| **Control Null Task** | Stochastic Delta Debugging | *(None - Null Case)* | `not_applicable` | 6 | 0 | **YES** |" in md
        assert "| | *greedy* | *(None - Null Case)* | `not_applicable` | 6 | 0 | YES |" in md

        # Active found: must format elements
        assert "| **Active Task Found** | Stochastic Delta Debugging | `elem_e` | `statistically_supported` | 12 | 0 | **YES** |" in md

        # Null false alarm: must format discovered elements, not say *(None - Null Case)*
        assert "| **Control Null False Alarm** | Stochastic Delta Debugging | `elem_g` | `observed_only` | 8 | 1 | **NO** |" in md

    def test_statistical_methodology_descriptions_and_limitations(self) -> None:
        """Verify H2, report limitations, and demo text accurately describe Berger's IUT."""
        from agentxray.demo import format_demo_output, run_demo
        from agentxray.models import (
            ExecutionTier,
            ExperimentConfig,
            ModelPilotExperimentReport,
            StochasticExperimentReport,
        )
        from agentxray.reporting import format_model_pilot_markdown_report

        # Verify H2 in markdown report
        dummy_report = ModelPilotExperimentReport(
            experiment_name="test_h2_text",
            execution_tier=ExecutionTier.FAKE_PROVIDER,
            provider_name="fake",
            model_name="fake-model",
            config=ExperimentConfig(experiment_name="test_h2_text", max_trials=50),
            budget_summary={},
            trials_per_candidate=5,
            total_target_calls=10,
            evaluations=(),
            aggregate_metrics={},
            summary="H2 test summary",
        )
        md = format_model_pilot_markdown_report(dummy_report)
        assert "under Berger's (1982) Intersection-Union Test at $\\alpha = 0.05$" in md
        assert "Benjamini-Hochberg FDR control" not in md

        # Verify limitations in models
        stoch_lims = StochasticExperimentReport.model_fields["limitations"].default
        assert any("Berger's (1982) Intersection-Union Test" in lim for lim in stoch_lims)
        assert not any("Benjamini-Hochberg FDR control applies" in lim for lim in stoch_lims)

        pilot_lims = ModelPilotExperimentReport.model_fields["limitations"].default
        assert any("Berger's (1982) Intersection-Union Test" in lim for lim in pilot_lims)
        assert not any("Benjamini-Hochberg FDR control applies" in lim for lim in pilot_lims)

        # Verify demo output
        demo_res = run_demo(seed=42)
        demo_out = format_demo_output(demo_res)
        assert "Berger's (1982) Intersection-Union Test" in demo_out
        assert "Benjamini-Hochberg FDR control" not in demo_out

    def test_runner_aggregate_metrics_partitioning_with_fake_provider(self, tmp_path: Path) -> None:
        """Verify run_model_pilot_benchmark partitions active vs control metrics with explicit denominators."""
        from benchmarks.model_pilot.runner import run_model_pilot_benchmark

        report = run_model_pilot_benchmark(
            provider_type="fake",
            model_name="fake-model-pilot",
            live=False,
            max_trials=30,
            trials_per_candidate=3,
            max_model_requests=300,
            output_dir=tmp_path,
        )

        metrics = report.aggregate_metrics
        assert metrics["tasks_evaluated"] == 3
        assert metrics["active_tasks"] == 2
        assert metrics["null_tasks"] == 1
        assert metrics["active_mean_precision"] == 1.0
        assert metrics["active_mean_recall"] == 1.0
        assert metrics["null_detection_rate"] == 1.0
        assert metrics["null_false_positive_rate"] == 0.0
        assert metrics["null_element_false_positives"] == 0
        assert metrics["null_spurious_failures"] == 0

        # Legacy compatibility keys
        assert metrics["mean_precision"] == 1.0
        assert metrics["mean_recall"] == 1.0
        assert metrics["mean_precision_evaluations"] == 2
        assert metrics["mean_recall_evaluations"] == 2

        # Summary narrative checks
        assert "Active tasks (n=2): mean precision=1.0, mean recall=1.0." in report.summary
        assert "Control tasks (n=1): clean null detection rate=1.0 (1/1), false positive rate=0.0 (element false discoveries: 0, spurious empty-set failures: 0)." in report.summary

    def test_runner_aggregate_metrics_mixed_population_calculations(self) -> None:
        """Verify partitioning and explicit denominator semantics in a mixed population with false alarms."""
        from agentxray.models import (
            CandidateEvaluation,
            ExecutionTier,
            ExperimentConfig,
            ModelPilotEvaluationResult,
            SearchStatus,
            StochasticDiscoveredCondition,
            StochasticMinimalityStatus,
        )
        from benchmarks.model_pilot.runner import CostTracker, ModelPilotExperimentReport

        dummy_cand = CandidateEvaluation(
            candidate_elements=(),
            trials=5,
            failures=0,
            failure_rate=0.0,
            confidence_interval=(0.0, 0.43),
        )

        eval_active_1 = ModelPilotEvaluationResult(
            task_name="Active 1",
            description="",
            initial_elements=("a", "b"),
            ground_truth_trigger=("a",),
            is_null_case=False,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=("a",),
                status=SearchStatus.COMPLETE,
                is_failure=True,
                minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
                candidate_evaluation=dummy_cand,
                target_calls_used=5,
            ),
            matches_ground_truth=True,
            precision=1.0,
            recall=1.0,
            excess_elements=(),
            missing_elements=(),
        )

        eval_active_2 = ModelPilotEvaluationResult(
            task_name="Active 2",
            description="",
            initial_elements=("a", "b", "c"),
            ground_truth_trigger=("a",),
            is_null_case=False,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=("a", "b"),
                status=SearchStatus.COMPLETE,
                is_failure=True,
                minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
                candidate_evaluation=dummy_cand,
                target_calls_used=5,
            ),
            matches_ground_truth=False,
            precision=0.5,
            recall=1.0,
            excess_elements=("b",),
            missing_elements=(),
        )

        eval_null_clean = ModelPilotEvaluationResult(
            task_name="Null Clean",
            description="",
            initial_elements=("x",),
            ground_truth_trigger=(),
            is_null_case=True,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.COMPLETE,
                is_failure=False,
                minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
                candidate_evaluation=dummy_cand,
                target_calls_used=5,
            ),
            matches_ground_truth=True,
            precision=None,
            recall=None,
            excess_elements=(),
            missing_elements=(),
        )

        eval_null_fp = ModelPilotEvaluationResult(
            task_name="Null False Alarm",
            description="",
            initial_elements=("y",),
            ground_truth_trigger=(),
            is_null_case=True,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=("y",),
                status=SearchStatus.COMPLETE,
                is_failure=True,
                minimality_status=StochasticMinimalityStatus.OBSERVED_ONLY,
                candidate_evaluation=dummy_cand,
                target_calls_used=5,
            ),
            matches_ground_truth=False,
            precision=0.0,
            recall=None,
            excess_elements=("y",),
            missing_elements=(),
        )

        eval_null_spurious = ModelPilotEvaluationResult(
            task_name="Null Spurious Empty Failure",
            description="",
            initial_elements=("z",),
            ground_truth_trigger=(),
            is_null_case=True,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.COMPLETE,
                is_failure=True,
                minimality_status=StochasticMinimalityStatus.OBSERVED_ONLY,
                candidate_evaluation=dummy_cand,
                target_calls_used=5,
            ),
            matches_ground_truth=False,
            precision=None,
            recall=None,
            excess_elements=(),
            missing_elements=(),
        )

        evaluations = [eval_active_1, eval_active_2, eval_null_clean, eval_null_fp, eval_null_spurious]

        # Exercise production aggregate metrics computation
        metrics = compute_stochastic_aggregate_metrics(evaluations)

        assert metrics["tasks_evaluated"] == 5
        assert metrics["active_tasks"] == 2
        assert metrics["null_tasks"] == 3
        assert metrics["active_mean_precision"] == 0.75  # (1.0 + 0.5) / 2
        assert metrics["active_mean_recall"] == 1.0   # (1.0 + 1.0) / 2
        assert metrics["null_detection_rate"] == round(1 / 3, 4)
        assert metrics["null_false_positive_rate"] == round(2 / 3, 4)
        assert metrics["null_element_false_positives"] == 1  # eval_null_fp has subset_elements=("y",)
        assert metrics["null_spurious_failures"] == 1        # eval_null_spurious has subset_elements=(), is_failure=True
        assert (
            metrics["null_tasks"] - int(round(metrics["null_detection_rate"] * metrics["null_tasks"]))
            == metrics["null_element_false_positives"] + metrics["null_spurious_failures"]
        )

        # In legacy mean_precision, only valid non-None precisions are included:
        # T1 (1.0), T2 (0.5), T4 (None -> excluded), T5 (0.0), T6 (None -> excluded)
        # Sum = 1.5, Count = 3 -> Mean = 0.5000
        assert metrics["mean_precision"] == 0.5000
        assert metrics["mean_precision_evaluations"] == 3
        assert metrics["mean_recall_evaluations"] == 2

    def test_null_control_empty_discovery_with_spurious_failure(self) -> None:
        """Verify ground-truth classification and aggregate accounting for a null control with empty discovery but spurious failure."""
        from agentxray.models import (
            CandidateEvaluation,
            ExecutionTier,
            ExperimentConfig,
            ModelPilotEvaluationResult,
            ModelPilotExperimentReport,
            SearchStatus,
            StochasticDiscoveredCondition,
            StochasticMinimalityStatus,
        )
        from benchmarks.model_pilot.ground_truth import ModelPilotGroundTruthSpec, evaluate_model_pilot_ground_truth

        # 1. Direct evaluator test: spec is null, discovered condition is empty but is_failure is True
        null_spec = ModelPilotGroundTruthSpec(
            task_id="null_spec",
            name="Null Control Task",
            description="Control hypothesis",
            trigger_elements=(),
            is_null_case=True,
        )

        dummy_cand = CandidateEvaluation(
            candidate_elements=(),
            trials=5,
            failures=4,
            failure_rate=0.8,
            confidence_interval=(0.38, 0.96),
        )

        spurious_cond = StochasticDiscoveredCondition(
            subset_elements=(),
            status=SearchStatus.COMPLETE,
            is_failure=True,
            minimality_status=StochasticMinimalityStatus.OBSERVED_ONLY,
            candidate_evaluation=dummy_cand,
            target_calls_used=5,
        )

        matches, prec, rec, excess, missing = evaluate_model_pilot_ground_truth(null_spec, spurious_cond)
        assert matches is False
        assert prec is None
        assert rec is None
        assert excess == ()
        assert missing == ()

        # 2. Verify aggregate breakdown using actual production aggregation code
        eval_spurious = ModelPilotEvaluationResult(
            task_name="Null Spurious",
            description="",
            initial_elements=("a",),
            ground_truth_trigger=(),
            is_null_case=True,
            minimizer_condition=spurious_cond,
            matches_ground_truth=matches,
            precision=prec,
            recall=rec,
            excess_elements=excess,
            missing_elements=missing,
        )

        metrics = compute_stochastic_aggregate_metrics([eval_spurious])

        assert metrics["tasks_evaluated"] == 1
        assert metrics["active_tasks"] == 0
        assert metrics["null_tasks"] == 1
        assert metrics["null_detection_rate"] == 0.0
        assert metrics["null_false_positive_rate"] == 1.0
        assert metrics["null_element_false_positives"] == 0
        assert metrics["null_spurious_failures"] == 1
        assert (
            metrics["null_element_false_positives"] + metrics["null_spurious_failures"]
            == metrics["null_tasks"]
        )

    def test_run_stochastic_benchmark_aggregate_metrics(self, tmp_path: Path) -> None:
        """Verify run_stochastic_benchmark emits complete partitioned aggregate metrics and writes reports."""
        from agentxray.__main__ import run_stochastic_benchmark
        from agentxray.models import StochasticExperimentReport

        report = run_stochastic_benchmark(
            max_trials=10,
            trials_per_candidate=2,
            timeout_seconds=5.0,
            seed=42,
            target_seed=42,
            output_dir=tmp_path,
        )

        assert isinstance(report, StochasticExperimentReport)
        assert len(report.evaluations) == 6
        metrics = report.aggregate_metrics

        # Task counts
        assert metrics["tasks_evaluated"] == 6
        assert metrics["active_tasks"] == 5
        assert metrics["null_tasks"] == 1
        assert metrics["active_tasks"] + metrics["null_tasks"] == 6

        # Calculate expected control counts directly from report.evaluations
        null_evals = [
            e for e in report.evaluations
            if getattr(e, "is_null_case", False) or not bool(e.ground_truth_trigger)
        ]
        active_evals = [
            e for e in report.evaluations
            if (not getattr(e, "is_null_case", False)) and bool(e.ground_truth_trigger)
        ]

        expected_null_matches = sum(1 for e in null_evals if e.matches_ground_truth)
        expected_null_element_fps = sum(
            1 for e in null_evals if bool(e.minimizer_condition.subset_elements)
        )
        expected_null_spurious = sum(
            1 for e in null_evals
            if (not bool(e.minimizer_condition.subset_elements)) and (not e.matches_ground_truth)
        )
        expected_null_det_rate = (
            round(expected_null_matches / len(null_evals), 4) if null_evals else None
        )
        expected_null_fp_rate = (
            round((len(null_evals) - expected_null_matches) / len(null_evals), 4) if null_evals else None
        )

        assert metrics["null_tasks"] == len(null_evals)
        assert metrics["null_element_false_positives"] == expected_null_element_fps
        assert metrics["null_spurious_failures"] == expected_null_spurious
        assert metrics["null_detection_rate"] == expected_null_det_rate
        assert metrics["null_false_positive_rate"] == expected_null_fp_rate
        assert (
            metrics["null_element_false_positives"] + metrics["null_spurious_failures"]
            == len(null_evals) - expected_null_matches
        )

        # Active metrics calculated directly from active_evals
        expected_active_precisions = [e.precision for e in active_evals if e.precision is not None]
        expected_active_recalls = [e.recall for e in active_evals if e.recall is not None]
        expected_active_prec = (
            round(sum(expected_active_precisions) / len(expected_active_precisions), 4)
            if expected_active_precisions else None
        )
        expected_active_rec = (
            round(sum(expected_active_recalls) / len(expected_active_recalls), 4)
            if expected_active_recalls else None
        )
        assert metrics["active_tasks"] == len(active_evals)
        assert metrics["active_mean_precision"] == expected_active_prec
        assert metrics["active_mean_recall"] == expected_active_rec

        # Verify exact agreement with production aggregation helper
        assert metrics == compute_stochastic_aggregate_metrics(report.evaluations)

        # Legacy compatibility keys
        assert metrics["mean_precision_evaluations"] == len([e for e in report.evaluations if e.precision is not None])
        assert metrics["mean_recall_evaluations"] == len([e for e in report.evaluations if e.recall is not None])

        # Summary text checks
        assert "Executed 6 stochastic synthetic tasks across Delta Debugging" in report.summary
        assert "Active tasks (n=5):" in report.summary
        assert "Control tasks (n=1):" in report.summary
        assert "(element false discoveries:" in report.summary
        assert "spurious empty-set failures:" in report.summary

        # Report files on disk
        json_file = tmp_path / "stochastic_report.json"
        md_file = tmp_path / "stochastic_report.md"
        assert json_file.exists()
        assert md_file.exists()
        assert len(json_file.read_text(encoding="utf-8")) > 0
        assert len(md_file.read_text(encoding="utf-8")) > 0


class TestPhase44DReportingCorrectness:
    """Tests for Phase 4.4D: Finding A (null-control label consistency) and Finding B (budget/timeout metrics)."""

    def test_deterministic_markdown_trigger_label_empty_active_vs_null(self) -> None:
        """Verify format_markdown_report distinguishes *(none)* from *(None - Null Case)*."""
        from agentxray.models import (
            BaselineComparison,
            DiscoveredCondition,
            EvaluationResult,
            ExperimentConfig,
            ExperimentReport,
            SearchStatus,
        )
        from agentxray.reporting import format_markdown_report

        active_empty = EvaluationResult(
            task_name="Active Deterministic Fail",
            description="Active deterministic task where discovery failed",
            initial_elements=("a", "b"),
            ground_truth_trigger=("a",),
            minimizer_condition=DiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.NO_TRIGGER_FOUND,
                is_failure=False,
                is_one_minimal=False,
                target_calls_used=4,
            ),
            baseline_comparisons=(
                BaselineComparison(
                    baseline_name="greedy",
                    discovered_elements=(),
                    failure_reproduced=False,
                    minimality_check_passed=False,
                    target_calls=3,
                    matches_ground_truth=False,
                ),
            ),
            matches_ground_truth=False,
            precision=0.0,
            recall=0.0,
            excess_elements=(),
            missing_elements=("a",),
        )

        null_clean = EvaluationResult(
            task_name="Null Deterministic Control",
            description="Clean null control task",
            initial_elements=("c", "d"),
            ground_truth_trigger=(),
            minimizer_condition=DiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.COMPLETE,
                is_failure=False,
                is_one_minimal=False,
                target_calls_used=2,
            ),
            baseline_comparisons=(
                BaselineComparison(
                    baseline_name="greedy",
                    discovered_elements=(),
                    failure_reproduced=False,
                    minimality_check_passed=False,
                    target_calls=2,
                    matches_ground_truth=True,
                ),
            ),
            matches_ground_truth=True,
            precision=None,
            recall=None,
            excess_elements=(),
            missing_elements=(),
        )

        report = ExperimentReport(
            experiment_name="test_det_labels",
            config=ExperimentConfig(experiment_name="test_det_labels", max_trials=20),
            total_target_calls=11,
            evaluations=(active_empty, null_clean),
            summary="Deterministic label test",
        )

        md = format_markdown_report(report)

        # Active empty: must be *(none)*
        assert "| **Active Deterministic Fail** | **Delta Debugging** | *(none)* | No | 4 | 0 | FAIL |" in md
        assert "| | greedy | *(none)* | No | 3 | 0 | FAIL |" in md

        # Null clean: must be *(None - Null Case)*
        assert "| **Null Deterministic Control** | **Delta Debugging** | *(None - Null Case)* | No | 2 | 0 | **MATCH** |" in md
        assert "| | greedy | *(None - Null Case)* | No | 2 | 0 | MATCH |" in md

    def test_stochastic_markdown_trigger_label_empty_active_vs_null(self) -> None:
        """Verify format_stochastic_markdown_report distinguishes *(none)* from *(None - Null Case)*."""
        from agentxray.models import (
            CandidateEvaluation,
            ExperimentConfig,
            SearchStatus,
            StochasticBaselineComparison,
            StochasticDiscoveredCondition,
            StochasticEvaluationResult,
            StochasticExperimentReport,
            StochasticMinimalityStatus,
        )
        from agentxray.reporting import format_stochastic_markdown_report

        dummy_cand = CandidateEvaluation(
            candidate_elements=(),
            trials=10,
            failures=0,
            failure_rate=0.0,
            confidence_interval=(0.0, 0.28),
        )

        active_empty = StochasticEvaluationResult(
            task_name="Active Stochastic Fail",
            description="Active stochastic task where search found nothing",
            initial_elements=("x", "y"),
            ground_truth_trigger=("x",),
            base_probability=0.0,
            trigger_probability=0.9,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.NO_TRIGGER_FOUND,
                is_failure=False,
                minimality_status=StochasticMinimalityStatus.NOT_MINIMAL,
                candidate_evaluation=dummy_cand,
                target_calls_used=8,
            ),
            baseline_comparisons=(
                StochasticBaselineComparison(
                    baseline_name="greedy",
                    discovered_elements=(),
                    failure_reproduced=False,
                    minimality_status=StochasticMinimalityStatus.NOT_MINIMAL,
                    target_calls=6,
                    candidate_evaluation=dummy_cand,
                    matches_ground_truth=False,
                ),
            ),
            matches_ground_truth=False,
            precision=0.0,
            recall=0.0,
            excess_elements=(),
            missing_elements=("x",),
        )

        null_clean = StochasticEvaluationResult(
            task_name="Null Stochastic Control",
            description="Clean null control stochastic task",
            initial_elements=("z", "w"),
            ground_truth_trigger=(),
            base_probability=0.0,
            trigger_probability=0.0,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.COMPLETE,
                is_failure=False,
                minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
                candidate_evaluation=dummy_cand,
                target_calls_used=4,
            ),
            baseline_comparisons=(
                StochasticBaselineComparison(
                    baseline_name="greedy",
                    discovered_elements=(),
                    failure_reproduced=False,
                    minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
                    target_calls=4,
                    candidate_evaluation=dummy_cand,
                    matches_ground_truth=True,
                ),
            ),
            matches_ground_truth=True,
            precision=None,
            recall=None,
            excess_elements=(),
            missing_elements=(),
        )

        report = StochasticExperimentReport(
            experiment_name="test_stoch_labels",
            config=ExperimentConfig(experiment_name="test_stoch_labels", max_trials=20),
            trials_per_candidate=10,
            total_target_calls=22,
            evaluations=(active_empty, null_clean),
            summary="Stochastic label test",
        )

        md = format_stochastic_markdown_report(report)

        # Active empty: must be *(none)*
        assert "| **Active Stochastic Fail** | **Delta Debugging** | *(none)* | N/A (base: 0.00) | `not_minimal` | 8 | 0 | FAIL |" in md
        assert "| | greedy | *(none)* | N/A (base: 0.00) | `not_minimal` | 6 | 0 | FAIL |" in md

        # Null clean: must be *(None - Null Case)*
        assert "| **Null Stochastic Control** | **Delta Debugging** | *(None - Null Case)* | N/A (base: 0.00) | `not_applicable` | 4 | 0 | **MATCH** |" in md
        assert "| | greedy | *(None - Null Case)* | N/A (base: 0.00) | `not_applicable` | 4 | 0 | MATCH |" in md

    def test_compute_stochastic_aggregate_metrics_budget_and_timeout(self) -> None:
        """Verify budget_exhausted_tasks and timed_out_tasks count accurately and without duplication."""
        from agentxray.models import (
            CandidateEvaluation,
            SearchStatus,
            StochasticDiscoveredCondition,
            StochasticEvaluationResult,
            StochasticMinimalityStatus,
        )
        from agentxray.reporting import compute_stochastic_aggregate_metrics

        dummy_cand = CandidateEvaluation(
            candidate_elements=(),
            trials=5,
            failures=0,
            failure_rate=0.0,
            confidence_interval=(0.0, 0.43),
        )

        # Task 1: Budget exhausted via status and minimality_status
        task_budget = StochasticEvaluationResult(
            task_name="Task Budget Exhausted",
            description="Exhausted trial budget",
            initial_elements=("a", "b"),
            ground_truth_trigger=("a",),
            base_probability=0.0,
            trigger_probability=0.8,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.BUDGET_EXHAUSTED,
                is_failure=False,
                minimality_status=StochasticMinimalityStatus.BUDGET_EXHAUSTED,
                candidate_evaluation=dummy_cand,
                target_calls_used=50,
            ),
            baseline_comparisons=(),
            matches_ground_truth=False,
            precision=0.0,
            recall=0.0,
            excess_elements=(),
            missing_elements=("a",),
        )

        # Task 2: Timed out via status
        task_timeout = StochasticEvaluationResult(
            task_name="Task Timeout",
            description="Hit wall clock timeout",
            initial_elements=("c", "d"),
            ground_truth_trigger=("c",),
            base_probability=0.0,
            trigger_probability=0.8,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.TIMEOUT,
                is_failure=False,
                minimality_status=StochasticMinimalityStatus.INCONCLUSIVE,
                candidate_evaluation=dummy_cand,
                target_calls_used=20,
            ),
            baseline_comparisons=(),
            matches_ground_truth=False,
            precision=0.0,
            recall=0.0,
            excess_elements=(),
            missing_elements=("c",),
        )

        # Task 3: Normal successful completion
        task_success = StochasticEvaluationResult(
            task_name="Task Success",
            description="Completed normally",
            initial_elements=("e", "f"),
            ground_truth_trigger=("e",),
            base_probability=0.0,
            trigger_probability=0.8,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=("e",),
                status=SearchStatus.COMPLETE,
                is_failure=True,
                minimality_status=StochasticMinimalityStatus.STATISTICALLY_SUPPORTED,
                candidate_evaluation=CandidateEvaluation(
                    candidate_elements=("e",),
                    trials=5,
                    failures=5,
                    failure_rate=1.0,
                    confidence_interval=(0.57, 1.0),
                ),
                target_calls_used=15,
            ),
            baseline_comparisons=(),
            matches_ground_truth=True,
            precision=1.0,
            recall=1.0,
            excess_elements=(),
            missing_elements=(),
        )

        # Task 4: Task with multiple indicators of budget exhaustion (status, minimality_status, etc.)
        task_multi_budget = StochasticEvaluationResult(
            task_name="Task Multi Budget",
            description="Multiple budget exhaustion indicators",
            initial_elements=("g", "h"),
            ground_truth_trigger=("g",),
            base_probability=0.0,
            trigger_probability=0.8,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.BUDGET_EXHAUSTED,
                is_failure=False,
                minimality_status=StochasticMinimalityStatus.BUDGET_EXHAUSTED,
                candidate_evaluation=dummy_cand,
                target_calls_used=100,
            ),
            baseline_comparisons=(),
            matches_ground_truth=False,
            precision=0.0,
            recall=0.0,
            excess_elements=(),
            missing_elements=("g",),
        )

        metrics = compute_stochastic_aggregate_metrics([
            task_budget,
            task_timeout,
            task_success,
            task_multi_budget,
        ])

        assert metrics["tasks_evaluated"] == 4
        # Two tasks exhausted budget, each counted only once
        assert metrics["budget_exhausted_tasks"] == 2
        # One task timed out, counted only once
        assert metrics["timed_out_tasks"] == 1
        assert metrics["ground_truth_matches"] == 1
        assert metrics["statistically_supported_minimality"] == 1
