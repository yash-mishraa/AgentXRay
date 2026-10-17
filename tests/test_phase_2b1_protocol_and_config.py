"""Tests for Phase 2B.1 preregistered protocol, configuration, and task registry."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from agentxray.agent_harness.cost_tracker import CostTracker, ExperimentBudgetConfig
from agentxray.providers.fake import FakeProvider
from benchmarks.model_pilot.confirmatory_tasks import (
    CONFIRMATORY_TASK_REGISTRY,
    ConfirmatoryTaskSpec,
    TaskFamily,
    get_confirmatory_task,
    list_confirmatory_tasks,
)
from benchmarks.model_pilot.validate_confirmatory_protocol import ConfirmatoryProtocolValidator


def test_confirmatory_task_registry_has_16_unique_instances() -> None:
    """Verify that the task registry contains 16 uniquely identified instances across 4 families."""
    tasks = list_confirmatory_tasks()
    assert len(tasks) == 16
    assert len(CONFIRMATORY_TASK_REGISTRY) == 16

    instance_ids = [t.instance_id for t in tasks]
    assert len(set(instance_ids)) == 16

    active_tasks = [t for t in tasks if not t.is_null_case]
    null_tasks = [t for t in tasks if t.is_null_case]

    assert len(active_tasks) == 12
    assert len(null_tasks) == 4

    # Check candidate space sizes
    k4_tasks = [t for t in tasks if t.k == 4]
    k8_tasks = [t for t in tasks if t.k == 8]
    assert len(k4_tasks) == 8
    assert len(k8_tasks) == 8

    # Verify active tasks have valid triggers
    for t in active_tasks:
        assert len(t.planted_trigger_elements) > 0
        for trig in t.planted_trigger_elements:
            assert trig in t.candidate_elements
        assert t.expected_failure_classification == "FAIL_TOOL_EXEC"

    # Verify null tasks have no triggers
    for t in null_tasks:
        assert len(t.planted_trigger_elements) == 0
        assert t.family == TaskFamily.NULL_CONTROL
        assert t.expected_failure_classification == "CLEAN_PASS"


def test_confirmatory_config_matches_registry() -> None:
    """Verify that the JSON configuration aligns with the task registry and protocol rules."""
    cfg_path = Path("benchmarks/configs/phase_2b1_confirmatory_config.json")
    assert cfg_path.exists(), "Configuration file must exist"

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["protocol_version"] == "Phase_2B.1_v1.0"
    assert cfg["status"] == "PREREGISTERED_FROZEN"
    assert cfg["commercial_spend_cap_usd"] == 0.0
    assert cfg["instances_count"] == 16
    assert cfg["active_instances_count"] == 12
    assert cfg["null_instances_count"] == 4

    inst_ids = cfg["instance_ids"]
    assert len(inst_ids) == 16
    for iid in inst_ids:
        assert iid in CONFIRMATORY_TASK_REGISTRY

    # Check SAP rules
    sap = cfg["statistical_analysis_plan"]
    assert sap["primary_test"] == "wilcoxon_signed_rank_paired"
    assert sap["primary_estimand"] == "hodges_lehmann_median_difference"
    assert sap["two_sided_alpha"] == 0.05
    assert sap["target_power"] == 0.80


def test_validator_passes_on_valid_artifacts() -> None:
    """Verify that the protocol validator passes cleanly on the preregistered artifacts."""
    validator = ConfirmatoryProtocolValidator()
    summary = validator.validate()

    assert summary["validation_status"] == "VALIDATION_PASSED"
    assert summary["all_checks_passed"] is True
    assert summary["total_discrepancies"] == 0
    assert len(summary["checks_passed"]) >= 10
    assert "config_sha256" in summary
    assert "protocol_sha256" in summary


def test_validator_fails_closed_on_corrupt_config(tmp_path: Path) -> None:
    """Verify that the protocol validator fails closed (rejects) when a configuration field is invalid."""
    cfg_path = Path("benchmarks/configs/phase_2b1_confirmatory_config.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Corrupt spend cap
    cfg["commercial_spend_cap_usd"] = 15.00
    corrupt_cfg = tmp_path / "corrupt_config.json"
    corrupt_cfg.write_text(json.dumps(cfg), encoding="utf-8")

    validator = ConfirmatoryProtocolValidator(config_path=corrupt_cfg)
    summary = validator.validate()

    assert summary["validation_status"] == "VALIDATION_FAILED"
    assert summary["all_checks_passed"] is False
    assert summary["total_discrepancies"] >= 1
    assert any("commercial_spend" in d for d in summary["discrepancies"])


def test_task_spec_creates_target_with_zero_leakage() -> None:
    """Verify that created target and evaluator have zero access to planted ground truth."""
    task = get_confirmatory_task("inst_01_syntax_currency_k4")
    provider = FakeProvider(model_name="fake-test")
    tracker = CostTracker(ExperimentBudgetConfig(max_model_requests=10, max_spend_usd=1.00, model_name="fake-test"))

    target = task.create_target(provider=provider, cost_tracker=tracker)
    evaluator = target.evaluator

    # Evaluator must not possess planted trigger information
    assert not hasattr(evaluator, "planted_trigger_elements")
    assert not hasattr(evaluator, "ground_truth")

    # Initial input generation matches candidate elements
    init_case = task.create_initial_input()
    assert init_case.elements == task.candidate_elements
    assert len(init_case.elements) == 4
