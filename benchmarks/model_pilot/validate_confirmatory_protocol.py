"""Validation tool for Phase 2B.1 confirmatory experiment protocol and configuration.

Fails closed on missing, inconsistent, or invalid preregistration specifications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.model_pilot.confirmatory_tasks import (
    CONFIRMATORY_TASK_REGISTRY,
    TaskFamily,
    list_confirmatory_tasks,
)


class ConfirmatoryProtocolValidator:
    """Validates the preregistered Phase 2B.1 protocol, configuration, and task registry."""

    def __init__(
        self,
        config_path: Path | str = "benchmarks/configs/phase_2b1_confirmatory_config.json",
        protocol_path: Path | str = "docs/research/phase_2b1_confirmatory_protocol_v1.md",
    ) -> None:
        self.config_path = Path(config_path)
        self.protocol_path = Path(protocol_path)
        self.checks_passed: list[str] = []
        self.discrepancies: list[str] = []

    def record_pass(self, name: str) -> None:
        self.checks_passed.append(name)

    def record_fail(self, name: str, msg: str) -> None:
        self.discrepancies.append(f"[{name}] {msg}")

    def compute_sha256(self, p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def validate(self) -> dict[str, Any]:
        """Execute all consistency and integrity checks."""
        # 1. File existence
        if not self.config_path.is_file():
            self.record_fail("file_existence", f"Config file not found: {self.config_path}")
            return self._summary(False)
        self.record_pass("config_file_exists")

        if not self.protocol_path.is_file():
            self.record_fail("file_existence", f"Protocol document not found: {self.protocol_path}")
            return self._summary(False)
        self.record_pass("protocol_document_exists")

        # 2. Parse config JSON
        try:
            cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.record_pass("config_json_valid")
        except Exception as err:
            self.record_fail("json_syntax", f"Failed to parse config JSON: {err}")
            return self._summary(False)

        # 3. Protocol Metadata & Freeze Status
        if cfg.get("status") != "PREREGISTERED_FROZEN":
            self.record_fail("status", f"Expected PREREGISTERED_FROZEN, got {cfg.get('status')}")
        else:
            self.record_pass("status_frozen")

        if cfg.get("commercial_spend_cap_usd") != 0.0:
            self.record_fail("commercial_spend", f"Spend cap must be 0.0, got {cfg.get('commercial_spend_cap_usd')}")
        else:
            self.record_pass("commercial_spend_zero")

        # 4. Instance Accounting
        inst_ids = cfg.get("instance_ids", [])
        if len(inst_ids) != 16:
            self.record_fail("instance_count", f"Expected 16 instances in config, found {len(inst_ids)}")
        else:
            self.record_pass("instance_count_16")

        if len(set(inst_ids)) != len(inst_ids):
            self.record_fail("instance_ids_unique", "Duplicate instance IDs found in configuration")
        else:
            self.record_pass("instance_ids_unique")

        # 5. Registry Reconciliation
        registry_tasks = list_confirmatory_tasks()
        if len(registry_tasks) != 16:
            self.record_fail("registry_count", f"Registry contains {len(registry_tasks)} tasks, expected 16")
        else:
            self.record_pass("registry_count_16")

        for task in registry_tasks:
            if task.instance_id not in inst_ids:
                self.record_fail("registry_match", f"Task {task.instance_id} not listed in config instance_ids")
                continue

            # Candidate size consistency
            if len(task.candidate_elements) != task.k:
                self.record_fail("candidate_size", f"Task {task.instance_id}: k={task.k} != len(cands)={len(task.candidate_elements)}")

            # Null vs Active consistency
            if task.is_null_case:
                if len(task.planted_trigger_elements) != 0:
                    self.record_fail("null_case_triggers", f"Null task {task.instance_id} has planted triggers: {task.planted_trigger_elements}")
                if task.family != TaskFamily.NULL_CONTROL:
                    self.record_fail("null_family", f"Null task {task.instance_id} family is {task.family}")
            else:
                if len(task.planted_trigger_elements) == 0:
                    self.record_fail("active_case_triggers", f"Active task {task.instance_id} has 0 planted triggers")
                for trig in task.planted_trigger_elements:
                    if trig not in task.candidate_elements:
                        self.record_fail("trigger_in_cands", f"Trigger {trig[:30]} not in candidate elements for {task.instance_id}")

        self.record_pass("all_instances_verified_in_registry")

        # 6. Statistical Analysis Plan Checks
        sap = cfg.get("statistical_analysis_plan", {})
        if sap.get("primary_test") != "wilcoxon_signed_rank_paired":
            self.record_fail("primary_test", f"Expected wilcoxon_signed_rank_paired, got {sap.get('primary_test')}")
        else:
            self.record_pass("primary_test_wilcoxon")

        if sap.get("two_sided_alpha") != 0.05:
            self.record_fail("alpha", f"Expected alpha=0.05, got {sap.get('two_sided_alpha')}")
        else:
            self.record_pass("alpha_0_05")

        if sap.get("target_power") != 0.80:
            self.record_fail("power", f"Expected power=0.80, got {sap.get('target_power')}")
        else:
            self.record_pass("power_0_80")

        # 7. Checksums
        config_hash = self.compute_sha256(self.config_path)
        protocol_hash = self.compute_sha256(self.protocol_path)

        success = len(self.discrepancies) == 0
        summary = self._summary(success)
        summary["config_sha256"] = config_hash
        summary["protocol_sha256"] = protocol_hash
        return summary

    def _summary(self, success: bool) -> dict[str, Any]:
        return {
            "validation_status": "VALIDATION_PASSED" if success else "VALIDATION_FAILED",
            "all_checks_passed": success,
            "total_checks_passed": len(self.checks_passed),
            "total_discrepancies": len(self.discrepancies),
            "checks_passed": self.checks_passed,
            "discrepancies": self.discrepancies,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 2B.1 Confirmatory Protocol and Configuration")
    parser.add_argument("--config", default="benchmarks/configs/phase_2b1_confirmatory_config.json")
    parser.add_argument("--protocol", default="docs/research/phase_2b1_confirmatory_protocol_v1.md")
    args = parser.parse_args()

    validator = ConfirmatoryProtocolValidator(config_path=args.config, protocol_path=args.protocol)
    summary = validator.validate()

    print("======================================================================")
    print("AgentXRay Phase 2B.1 — Preregistration Protocol & Config Validation")
    print("======================================================================")
    print(f"Status: {summary['validation_status']}")
    print(f"Checks Passed: {summary['total_checks_passed']}")
    print(f"Discrepancies: {summary['total_discrepancies']}")
    if "config_sha256" in summary:
        print(f"Config SHA-256:   {summary['config_sha256']}")
    if "protocol_sha256" in summary:
        print(f"Protocol SHA-256: {summary['protocol_sha256']}")

    if summary["discrepancies"]:
        print("\nDiscrepancies:")
        for disc in summary["discrepancies"]:
            print(f"  - {disc}")
        sys.exit(1)
    else:
        print("\nPreregistration configuration and task registry are 100% consistent.")
        sys.exit(0)


if __name__ == "__main__":
    main()
