"""Unit and regression tests for Phase 2B.0 Local-Model Calibration Runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.model_pilot.calibration_runner import (
    ALL_CANDIDATE_ELEMENTS,
    CANDIDATE_TRIGGER,
    CalibrationPilotOrchestrator,
)


def test_calibration_orchestrator_fake_run(tmp_path: Path) -> None:
    """Verify that CalibrationPilotOrchestrator executes hermetically with FakeProvider."""
    results_dir = tmp_path / "results"
    docs_dir = tmp_path / "docs"

    orchestrator = CalibrationPilotOrchestrator(
        provider_type="fake",
        model_name="fake-model-test",
        trials_per_subset=3,
        max_requests_per_algorithm=60,
        results_dir=results_dir,
        docs_dir=docs_dir,
    )

    results = orchestrator.run_full_pilot(force_search=False)

    assert results["milestone"] == "Phase 2B.0 Local-Model Pilot Calibration"
    assert results["execution_tier"] == "fake_provider"
    assert results["provider_type"] == "fake"
    assert results["final_decision"] == "CONFIRMATORY_READY"

    # Verify specifications exported
    cand_file = results_dir / "phase_2b0_candidate_specs.json"
    gt_file = results_dir / "phase_2b0_ground_truth.json"
    cfg_file = results_dir / "phase_2b0_calibration_config.json"
    res_file = results_dir / "phase_2b0_calibration_results.json"
    log_file = results_dir / "phase_2b0_calibration_event_log.jsonl"
    report_file = docs_dir / "phase_2b0_calibration_report.md"

    assert cand_file.is_file()
    assert gt_file.is_file()
    assert cfg_file.is_file()
    assert res_file.is_file()
    assert log_file.is_file()
    assert report_file.is_file()

    # Verify candidate specs
    cand_data = json.loads(cand_file.read_text(encoding="utf-8"))
    assert cand_data["candidate_space_size"] == 4
    assert len(cand_data["elements"]) == 4

    # Verify ground truth
    gt_data = json.loads(gt_file.read_text(encoding="utf-8"))
    assert gt_data["true_trigger_elements"] == [CANDIDATE_TRIGGER]
    assert gt_data["is_null_case"] is False

    # Verify controls calibration
    ctrl = results["controls_calibration"]
    assert ctrl["trigger_status"] == "TRIGGER_CONFIRMED"
    assert ctrl["qualification_checks"]["all_checks_passed"] is True
    assert ctrl["negative_control_baseline"]["failure_rate"] == 0.0
    assert ctrl["positive_control_trigger"]["failure_rate"] == 1.0
    assert ctrl["null_control_distractors"]["failure_rate"] == 0.0
    assert ctrl["full_intervention"]["failure_rate"] == 1.0

    # Verify minimization algorithms
    algos = results["minimization_algorithms"]
    assert "delta_debugging" in algos
    assert "greedy_ablation" in algos
    assert "random_search" in algos
    assert "paired_comparison" in algos

    dd = algos["delta_debugging"]
    assert dd["matches_ground_truth"] is True
    assert dd["precision"] == 1.0
    assert dd["recall"] == 1.0
    assert dd["model_requests"] > 0

    # Verify holdout confirmation
    holdout = results["holdout_confirmation"]
    assert "delta_debugging" in holdout
    assert holdout["delta_debugging"]["failure_rate"] == 1.0


def test_event_log_jsonl_integrity(tmp_path: Path) -> None:
    """Verify that event log records conform strictly to valid JSON per line with proper timestamps."""
    results_dir = tmp_path / "results"
    docs_dir = tmp_path / "docs"

    orchestrator = CalibrationPilotOrchestrator(
        provider_type="fake",
        trials_per_subset=2,
        results_dir=results_dir,
        docs_dir=docs_dir,
    )
    orchestrator.run_full_pilot(force_search=False)

    log_path = results_dir / "phase_2b0_calibration_event_log.jsonl"
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) > 20

    event_ids = []
    for line in lines:
        entry = json.loads(line)
        assert "event_id" in entry
        assert "timestamp" in entry
        assert "event_type" in entry
        event_ids.append(entry["event_id"])

    # Strict monotonicity of event_id
    assert event_ids == list(range(1, len(event_ids) + 1))


def test_unconfirmed_trigger_halts_search(tmp_path: Path) -> None:
    """Verify that if controls fail qualification, search is halted unless force_search=True."""
    from agentxray.providers.fake import FakeProvider
    from agentxray.providers.base import Message, ModelResponse, UsageMetadata

    # Handler where calculator is NEVER called (everything fails, including baseline)
    def noisy_handler(messages, tools):
        return ModelResponse(
            content="I am completely broken and refuse tools.",
            tool_calls=(),
            usage=UsageMetadata(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            finish_reason="stop",
            model_identifier="fake-noisy",
        )

    results_dir = tmp_path / "results"
    docs_dir = tmp_path / "docs"

    orchestrator = CalibrationPilotOrchestrator(
        provider_type="fake",
        trials_per_subset=2,
        results_dir=results_dir,
        docs_dir=docs_dir,
    )
    # Inject noisy provider
    orchestrator.provider = FakeProvider(custom_handler=noisy_handler)

    results = orchestrator.run_full_pilot(force_search=False)

    ctrl = results["controls_calibration"]
    assert ctrl["trigger_status"] == "TRIGGER_UNCONFIRMED"
    # Search must be skipped
    assert results["minimization_algorithms"] == {}
    assert results["final_decision"] == "HARNESS_INADEQUATE_OR_EXPLORATORY"
