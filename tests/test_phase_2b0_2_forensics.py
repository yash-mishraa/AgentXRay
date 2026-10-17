"""Tests for Phase 2B.0.2 failure forensics, reconciliation, and reporting integrity."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from agentxray.agent_harness.agent import AgentExecutionTrace, ToolExecutionRecord
from agentxray.agent_harness.evaluator import ToolCallingTaskEvaluator


def test_reconciliation_artifact_exists_and_reconciles_16_subsets() -> None:
    """Verify that phase_2b0_2_characterization_reconciliation.json exists and accurately reconciles 16 subsets."""
    reconcil_path = Path("benchmarks/results/phase_2b0_2_characterization_reconciliation.json")
    assert reconcil_path.exists(), "Reconciliation artifact must exist"

    with open(reconcil_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_configurations"] == 16
    configs = data["configurations"]
    assert len(configs) == 16

    # Verify Row 10 details
    row_10 = next(c for c in configs if c["subset_index"] == 10)
    assert row_10["case_id"] == "char_subset_10_len2"
    assert row_10["element_ids"] == ["elem_2", "elem_4"]
    assert row_10["contains_trigger"] is False
    assert row_10["trials_observed"] == 5
    assert row_10["failures_observed"] == 5
    assert row_10["failure_rate"] == 1.0
    assert row_10["failure_classification"] == "FAIL_BEHAVIOR"
    assert row_10["tool_syntax_errors"] == 0
    assert row_10["behavioral_substring_omissions"] == 5
    assert row_10["clean_passes"] == 0
    assert row_10["supporting_trial_event_ids"] == [59, 60, 61, 62, 63]


def test_reconciliation_verifies_trigger_separation_by_class() -> None:
    """Verify that all trigger-present subsets failed with FAIL_TOOL_EXEC, while 7 of 8 trigger-absent passed."""
    reconcil_path = Path("benchmarks/results/phase_2b0_2_characterization_reconciliation.json")
    with open(reconcil_path, encoding="utf-8") as f:
        data = json.load(f)

    configs = data["configurations"]
    trigger_present = [c for c in configs if c["contains_trigger"]]
    trigger_absent = [c for c in configs if not c["contains_trigger"]]

    assert len(trigger_present) == 8
    assert len(trigger_absent) == 8

    # All 8 trigger-present subsets failed due to tool syntax error
    for c in trigger_present:
        assert c["failure_classification"] == "FAIL_TOOL_EXEC"
        assert c["tool_syntax_errors"] == 5
        assert c["failure_rate"] == 1.0

    # 7 of 8 trigger-absent subsets passed cleanly
    clean_absent = [c for c in trigger_absent if c["failure_classification"] == "CLEAN_PASS"]
    assert len(clean_absent) == 7
    for c in clean_absent:
        assert c["failure_rate"] == 0.0

    # Only Row 10 failed among trigger-absent subsets, and it failed with FAIL_BEHAVIOR
    row_10 = next(c for c in trigger_absent if c["failure_classification"] != "CLEAN_PASS")
    assert row_10["subset_index"] == 10
    assert row_10["failure_classification"] == "FAIL_BEHAVIOR"


def test_evaluator_distinguishes_failure_classes() -> None:
    """Verify that ToolCallingTaskEvaluator distinguishes FAIL_TOOL_EXEC from FAIL_BEHAVIOR."""
    evaluator = ToolCallingTaskEvaluator(
        required_tool="calculator",
        expected_substring="400",
    )

    # 1. Tool execution failure (syntax error)
    trace_tool_fail = AgentExecutionTrace(
        session_id="s1",
        turns=(),
        tool_executions=(
            ToolExecutionRecord(
                tool_call_id="call_1",
                tool_name="calculator",
                arguments={"expression": "$150 + $250"},
                output="Error: invalid syntax (<unknown>, line 1)",
            ),
        ),
        final_response="Total reimbursement is $400.",
        finish_reason="stop",
    )
    is_success, msg, fclass = evaluator.evaluate_detailed(trace_tool_fail)
    assert not is_success
    assert fclass == "FAIL_TOOL_EXEC"
    assert "returned an error" in msg

    # 2. Behavioral omission (tool succeeded, final text omitted substring)
    trace_behavior_fail = AgentExecutionTrace(
        session_id="s2",
        turns=(),
        tool_executions=(
            ToolExecutionRecord(
                tool_call_id="call_2",
                tool_name="calculator",
                arguments={"expression": "150 + 250"},
                output="400",
            ),
        ),
        final_response="audit_memo: 2023-12-01",
        finish_reason="stop",
    )
    is_success_b, msg_b, fclass_b = evaluator.evaluate_detailed(trace_behavior_fail)
    assert not is_success_b
    assert fclass_b == "FAIL_BEHAVIOR"
    assert "did not contain expected content '400'" in msg_b

    # 3. Clean pass
    trace_pass = AgentExecutionTrace(
        session_id="s3",
        turns=(),
        tool_executions=(
            ToolExecutionRecord(
                tool_call_id="call_3",
                tool_name="calculator",
                arguments={"expression": "150 + 250"},
                output="400",
            ),
        ),
        final_response="Total reimbursement is $400. audit_memo: 2023-12-01",
        finish_reason="stop",
    )
    is_success_p, msg_p, fclass_p = evaluator.evaluate_detailed(trace_pass)
    assert is_success_p
    assert fclass_p == "CLEAN_PASS"


def test_evaluator_target_failure_class_filtering() -> None:
    """Verify that when target_failure_class is set to FAIL_TOOL_EXEC, behavioral omissions do not fail the trial."""
    evaluator_tool_only = ToolCallingTaskEvaluator(
        required_tool="calculator",
        expected_substring="400",
        target_failure_class="FAIL_TOOL_EXEC",
    )

    trace_behavior_only = AgentExecutionTrace(
        session_id="s4",
        turns=(),
        tool_executions=(
            ToolExecutionRecord(
                tool_call_id="call_4",
                tool_name="calculator",
                arguments={"expression": "150 + 250"},
                output="400",
            ),
        ),
        final_response="audit_memo: 2023-12-01",
        finish_reason="stop",
    )

    # Tool executed cleanly with 400, so tool execution criterion is satisfied
    is_success, msg, fclass = evaluator_tool_only.evaluate_detailed(trace_behavior_only)
    assert is_success is True
    assert fclass == "FAIL_BEHAVIOR"


def test_orchestrator_v1_2_confirmatory_ready_under_fake_provider(tmp_path: Path) -> None:
    """Verify that when trigger qualification passes, runner executes minimization, holdout, and reports CONFIRMATORY_READY."""
    from benchmarks.model_pilot.calibration_runner_v1_2 import CorrectiveCalibrationOrchestratorV12

    orch = CorrectiveCalibrationOrchestratorV12(
        provider_type="fake",
        model_name="fake-model",
        trials_per_subset=1,
        results_dir=tmp_path / "results",
        docs_dir=tmp_path / "docs",
    )
    results = orch.run_full_pilot()
    assert results["final_decision"] == "CONFIRMATORY_READY"
    assert results["exhaustive_characterization"]["qualification_criteria_met"] is True
    assert "delta_debugging" in results["minimization_algorithms"]
    assert "greedy_ablation" in results["minimization_algorithms"]
    assert "random_search" in results["minimization_algorithms"]
    assert "delta_debugging" in results["holdout_confirmation"]

    report_text = orch.report_path.read_text(encoding="utf-8")
    assert "CONFIRMATORY_READY" in report_text
    assert "elem_1" in report_text
    assert "BLOCKED_BY_QUALIFICATION" not in report_text


def test_orchestrator_v1_2_blocked_by_qualification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that when qualification fails, runner does not execute search and reports BLOCKED_BY_QUALIFICATION."""
    from benchmarks.model_pilot.calibration_runner_v1_2 import CorrectiveCalibrationOrchestratorV12

    orch = CorrectiveCalibrationOrchestratorV12(
        provider_type="fake",
        model_name="fake-model",
        trials_per_subset=1,
        results_dir=tmp_path / "results",
        docs_dir=tmp_path / "docs",
    )

    # Force qualification failure by monkeypatching run_exhaustive_characterization
    def mock_char() -> dict:
        return {
            "trigger_status": "TRIGGER_UNCONFIRMED",
            "power_set_size": 16,
            "subsets_evaluated": 16,
            "min_trigger_present_failure_rate": 0.20,
            "max_trigger_absent_failure_rate": 0.80,
            "qualification_criteria_met": False,
            "subsets": [],
            "total_characterization_requests": 32,
        }

    monkeypatch.setattr(orch, "run_exhaustive_characterization", mock_char)

    results = orch.run_full_pilot()
    assert results["final_decision"] == "HARNESS_INADEQUATE_OR_EXPLORATORY"
    assert results["exhaustive_characterization"]["qualification_criteria_met"] is False
    assert results["minimization_algorithms"] == {}
    assert results["holdout_confirmation"] == {}

    report_text = orch.report_path.read_text(encoding="utf-8")
    assert "BLOCKED_BY_QUALIFICATION" in report_text
    assert "NOT_RUN" in report_text


