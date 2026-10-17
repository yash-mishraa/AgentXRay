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
    ACTIVE_MECHANISMS,
    CONFIRMATORY_TASK_REGISTRY,
    MECHANISM_REGISTRY,
    NULL_MECHANISMS,
    TaskFamily,
    get_tasks_by_mechanism,
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

        # 6. Document Reference Check
        if "document_reference" in cfg:
            doc_ref = Path(cfg["document_reference"])
            if not doc_ref.is_file():
                self.record_fail("document_reference", f"Document reference does not exist: {doc_ref}")
            else:
                self.record_pass("document_reference_exists")

        # 7. Mechanism Clustering Reconciliation (if declared in config)
        if "mechanism_ids" in cfg:
            mech_ids = cfg["mechanism_ids"]
            if len(mech_ids) != 8:
                self.record_fail("mechanism_count", f"Expected 8 mechanisms in config, got {len(mech_ids)}")
            elif len(set(mech_ids)) != len(mech_ids):
                self.record_fail("mechanism_unique", "Duplicate mechanism IDs found in configuration")
            else:
                self.record_pass("mechanism_ids_valid")

            tasks_by_mech = get_tasks_by_mechanism()
            for mid in mech_ids:
                if mid not in tasks_by_mech:
                    self.record_fail("mechanism_missing_tasks", f"Mechanism {mid} has no registered tasks")
                    continue
                m_tasks = tasks_by_mech[mid]
                if len(m_tasks) != 2:
                    self.record_fail("mechanism_task_count", f"Mechanism {mid} has {len(m_tasks)} tasks, expected 2 (K=4 and K=8)")
                    continue
                ks = {t.k for t in m_tasks}
                if ks != {4, 8}:
                    self.record_fail("mechanism_k_sizes", f"Mechanism {mid} has k sizes {ks}, expected {4, 8}")
                t4 = [t for t in m_tasks if t.k == 4][0]
                t8 = [t for t in m_tasks if t.k == 8][0]
                if t4.user_prompt != t8.user_prompt:
                    self.record_fail("mechanism_prompt_mismatch", f"Mechanism {mid} user_prompt differs between K=4 and K=8")
                if t4.required_tool != t8.required_tool:
                    self.record_fail("mechanism_tool_mismatch", f"Mechanism {mid} required_tool differs between K=4 and K=8")
                if t4.planted_trigger_elements != t8.planted_trigger_elements:
                    self.record_fail("mechanism_trigger_mismatch", f"Mechanism {mid} planted triggers differ between K=4 and K=8")

            self.record_pass("all_mechanisms_verified_and_paired")

        # 8. Statistical Analysis Plan Checks
        sap = cfg.get("statistical_analysis_plan", {})
        valid_primary_tests = (
            "wilcoxon_signed_rank_paired",
            "dual_endpoint_recovery_and_resource_burden",
            "two_part_mcnemar_and_signed_rank",
            "stratified_descriptive_and_mechanism_level_wilcoxon",
        )
        if sap.get("primary_test") not in valid_primary_tests:
            self.record_fail("primary_test", f"Expected one of {valid_primary_tests}, got {sap.get('primary_test')}")
        else:
            self.record_pass("primary_test_valid")

        if sap.get("two_sided_alpha") != 0.05:
            self.record_fail("alpha", f"Expected alpha=0.05, got {sap.get('two_sided_alpha')}")
        else:
            self.record_pass("alpha_0_05")

        if sap.get("target_power") != 0.80:
            self.record_fail("power", f"Expected power=0.80, got {sap.get('target_power')}")
        else:
            self.record_pass("power_0_80")

        # 9. Request Budget Enforceability Checks
        budget_rules = cfg.get("budget_rules", {})
        max_budget = budget_rules.get("max_total_request_budget", 0)
        if max_budget < 6000:
            self.record_fail("budget_cap_underfunded", f"Total budget cap {max_budget} is underfunded (must be >= 6000)")
        else:
            self.record_pass("budget_cap_sufficient")

        if "max_requests_per_algorithm_k4" in budget_rules:
            if budget_rules["max_requests_per_algorithm_k4"] != 120:
                self.record_fail("budget_k4_cap", f"Expected K=4 cap 120, got {budget_rules['max_requests_per_algorithm_k4']}")
            else:
                self.record_pass("budget_k4_cap_120")

        if "max_requests_per_algorithm_k8" in budget_rules:
            if budget_rules["max_requests_per_algorithm_k8"] != 180:
                self.record_fail("budget_k8_cap", f"Expected K=8 cap 180, got {budget_rules['max_requests_per_algorithm_k8']}")
            else:
                self.record_pass("budget_k8_cap_180")

        if "trials_per_candidate" in budget_rules:
            if budget_rules["trials_per_candidate"] != 5:
                self.record_fail("trials_per_candidate", f"Expected 5 trials per candidate, got {budget_rules['trials_per_candidate']}")
            else:
                self.record_pass("trials_per_candidate_5")

        if "calibration_trials_per_subset" in budget_rules:
            if budget_rules["calibration_trials_per_subset"] != 5:
                self.record_fail("calibration_trials", f"Expected 5 calibration trials, got {budget_rules['calibration_trials_per_subset']}")
            else:
                self.record_pass("calibration_trials_5")

        # 10. Stopping & Qualification Criteria
        if "stopping_rules" in cfg:
            sr = cfg["stopping_rules"]
            if sr.get("active_task_qualification_min_trigger_failure_rate") != 0.70:
                self.record_fail("stopping_min_trigger", f"Expected min trigger rate 0.70, got {sr.get('active_task_qualification_min_trigger_failure_rate')}")
            else:
                self.record_pass("stopping_min_trigger_0_70")

            if sr.get("active_task_qualification_max_absent_failure_rate") != 0.20:
                self.record_fail("stopping_max_absent", f"Expected max absent rate 0.20, got {sr.get('active_task_qualification_max_absent_failure_rate')}")
            else:
                self.record_pass("stopping_max_absent_0_20")

            if sr.get("null_task_qualification_max_failure_rate") != 0.10:
                self.record_fail("stopping_max_null", f"Expected max null rate 0.10, got {sr.get('null_task_qualification_max_failure_rate')}")
            else:
                self.record_pass("stopping_max_null_0_10")

            if sr.get("hard_budget_breach_abort") is not True:
                self.record_fail("stopping_hard_budget", f"Expected hard_budget_breach_abort=True, got {sr.get('hard_budget_breach_abort')}")
            else:
                self.record_pass("stopping_hard_budget_true")

        # 11. Minimality Outcome Taxonomy
        if "minimality_outcome_taxonomy" in sap:
            tax = sap["minimality_outcome_taxonomy"]
            req_tax = {"SUPPORTED_1_MINIMAL", "OBSERVED_1_MINIMAL", "NOT_MINIMAL", "INCONCLUSIVE", "BUDGET_EXHAUSTED", "INFRA_FAILURE"}
            missing_tax = req_tax - set(tax)
            if missing_tax:
                self.record_fail("minimality_taxonomy", f"Missing taxonomy outcomes: {missing_tax}")
            else:
                self.record_pass("minimality_taxonomy_complete")

        # 12. Protocol Text Semantic Invariants
        proto_text = self.protocol_path.read_text(encoding="utf-8")
        if "120" not in proto_text or "180" not in proto_text:
            self.record_fail("protocol_caps_text", "Protocol text does not state per-task caps (120 and 180)")
        else:
            self.record_pass("protocol_caps_text_present")

        if budget_rules.get("max_total_request_budget", 0) >= 7500:
            if "7,500" not in proto_text and "7500" not in proto_text:
                self.record_fail("protocol_safety_budget_text", "Protocol text does not state global safety cap 7,500")
            else:
                self.record_pass("protocol_safety_budget_text_present")

        # Version-specific strictness
        if cfg.get("protocol_version") in ("Phase_2B.1_v1.3", "Phase_2B.1_v1.4"):
            if sap.get("non_inferiority_status") != "DEMOTED_TO_DESCRIPTIVE_EXPLORATORY":
                self.record_fail("non_inferiority_status", f"Expected DEMOTED_TO_DESCRIPTIVE_EXPLORATORY, got {sap.get('non_inferiority_status')}")
            else:
                self.record_pass("non_inferiority_demoted")

            # Illicit shortcut must be completely purged from protocol text
            if "or ablated rate <= 0.20" in proto_text or "observed deletion failure rate at most 0.20" in proto_text:
                self.record_fail("protocol_shortcut_purged", f"Protocol {cfg.get('protocol_version')} text still contains invalid disjunctive shortcut for 1-minimality")
            else:
                self.record_pass("protocol_shortcut_purged")

        if cfg.get("protocol_version") == "Phase_2B.1_v1.4":
            if sap.get("primary_test") != "stratified_descriptive_and_mechanism_level_wilcoxon":
                self.record_fail("v1_4_primary_test", f"Expected stratified_descriptive_and_mechanism_level_wilcoxon, got {sap.get('primary_test')}")
            else:
                self.record_pass("v1_4_primary_test_correct")

            if sap.get("pooled_test_status") != "DESCRIPTIVE_EXPLORATORY_ONLY":
                self.record_fail("v1_4_pooled_test_status", f"Expected DESCRIPTIVE_EXPLORATORY_ONLY, got {sap.get('pooled_test_status')}")
            else:
                self.record_pass("v1_4_pooled_test_status_correct")

            if sap.get("cluster_unit") != "mechanism_id":
                self.record_fail("v1_4_cluster_unit", f"Expected mechanism_id, got {sap.get('cluster_unit')}")
            else:
                self.record_pass("v1_4_cluster_unit_correct")

            if sap.get("stratification") != "candidate_size_k":
                self.record_fail("v1_4_stratification", f"Expected candidate_size_k, got {sap.get('stratification')}")
            else:
                self.record_pass("v1_4_stratification_correct")

            if "Claim A" not in proto_text or "Claim B" not in proto_text:
                self.record_fail("v1_4_proto_claim_definitions", "Protocol v1.4 text must explicitly contrast Claim A and Claim B")
            else:
                self.record_pass("v1_4_proto_claim_definitions_present")

            if "pseudoreplication" not in proto_text.lower() and "repeated measurements" not in proto_text.lower():
                self.record_fail("v1_4_proto_clustering_mention", "Protocol v1.4 text must discuss pseudoreplication or repeated measurements")
            else:
                self.record_pass("v1_4_proto_clustering_mention_present")

        # 13. Checksums
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
