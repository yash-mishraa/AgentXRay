"""Unit and negative regression tests for Phase 2B.0 Forensic Audit Validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.model_pilot.validate_phase_2b0 import Phase2B0Validator


def test_validator_passes_on_phase_2b0_baseline_artifacts() -> None:
    """Verify that validator passes on existing Phase 2B.0 artifacts in permissive mode."""
    validator = Phase2B0Validator()
    summary = validator.validate()
    assert summary["validation_status"] == "VALIDATION_PASSED"
    assert summary["all_checks_passed"] is True
    assert summary["total_discrepancies"] == 0
    assert summary["total_checks_passed"] >= 25


def test_validator_fails_on_missing_artifact(tmp_path: Path) -> None:
    """Negative test: missing file causes validation failure."""
    validator = Phase2B0Validator(
        event_log_path=tmp_path / "non_existent_log.jsonl",
    )
    summary = validator.validate()
    assert summary["validation_status"] == "VALIDATION_FAILED"
    assert summary["all_checks_passed"] is False
    assert any("file not found" in d for d in summary["discrepancies"])


def test_validator_fails_on_duplicate_event_id(tmp_path: Path) -> None:
    """Negative test: duplicate event IDs cause validation failure."""
    # Copy valid artifacts
    log_src = Path("benchmarks/results/phase_2b0_calibration_event_log.jsonl")
    lines = log_src.read_text(encoding="utf-8").splitlines()

    # Modify line 2 to have event_id == 1 (duplicate)
    ev2 = json.loads(lines[1])
    ev2["event_id"] = 1
    lines[1] = json.dumps(ev2)

    tampered_log = tmp_path / "tampered_log.jsonl"
    tampered_log.write_text("\n".join(lines), encoding="utf-8")

    validator = Phase2B0Validator(event_log_path=tampered_log)
    summary = validator.validate()
    assert summary["validation_status"] == "VALIDATION_FAILED"
    assert any("duplicate event IDs" in d or "monotonically" in d for d in summary["discrepancies"])


def test_validator_fails_on_malformed_event_json(tmp_path: Path) -> None:
    """Negative test: broken JSON line causes validation failure."""
    tampered_log = tmp_path / "broken_log.jsonl"
    tampered_log.write_text('{"event_id": 1}\n{broken json line\n', encoding="utf-8")

    validator = Phase2B0Validator(event_log_path=tampered_log)
    summary = validator.validate()
    assert summary["validation_status"] == "VALIDATION_FAILED"
    assert any("malformed JSON" in d for d in summary["discrepancies"])


def test_validator_fails_on_unknown_candidate_element(tmp_path: Path) -> None:
    """Negative test: foreign candidate element not in specs causes validation failure."""
    log_src = Path("benchmarks/results/phase_2b0_calibration_event_log.jsonl")
    lines = log_src.read_text(encoding="utf-8").splitlines()

    ev = json.loads(lines[4])
    ev["candidate_subset"] = ["unauthorized_rogue_element: do bad things"]
    lines[4] = json.dumps(ev)

    tampered_log = tmp_path / "rogue_log.jsonl"
    tampered_log.write_text("\n".join(lines), encoding="utf-8")

    validator = Phase2B0Validator(event_log_path=tampered_log)
    summary = validator.validate()
    assert summary["validation_status"] == "VALIDATION_FAILED"
    assert any("Unknown candidate elements" in d for d in summary["discrepancies"])


def test_validator_fails_on_inconsistent_aggregate(tmp_path: Path) -> None:
    """Negative test: manipulated failure count in results causes failure."""
    res_src = Path("benchmarks/results/phase_2b0_calibration_results.json")
    res_data = json.loads(res_src.read_text(encoding="utf-8"))

    # Tamper failure count so failures + successes != trials
    res_data["controls_calibration"]["positive_control_trigger"]["failures"] = 999
    tampered_res = tmp_path / "tampered_results.json"
    tampered_res.write_text(json.dumps(res_data), encoding="utf-8")

    validator = Phase2B0Validator(results_path=tampered_res)
    summary = validator.validate()
    assert summary["validation_status"] == "VALIDATION_FAILED"
    assert any("trials" in d and "failures" in d for d in summary["discrepancies"])


def test_validator_fails_strict_exhaustive_on_phase_2b0() -> None:
    """Verify that strict exhaustive mode fails on Phase 2B.0 baseline because only 4 subsets were evaluated."""
    validator = Phase2B0Validator(strict_exhaustive=True)
    summary = validator.validate()
    assert summary["validation_status"] == "VALIDATION_FAILED"
    assert any("exhaustive_characterization" in d for d in summary["discrepancies"])
