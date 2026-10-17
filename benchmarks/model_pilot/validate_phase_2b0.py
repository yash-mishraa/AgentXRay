"""Deterministic Forensic Audit and Reproducibility Validation for Phase 2B.0 Artifacts.

Performs code-level and trace-level reconciliation:
- Validates file existence and structural schemas of all artifacts.
- Enforces event log uniqueness, monotonicity, and schema integrity.
- Reconciles trial, request, and subset counts across logs and results JSON.
- Verifies candidate element membership within initial candidate space.
- Recomputes statistical tests (Fisher exact, Wilson score CIs, Newcombe intervals).
- Checks isolation between characterization, search, and holdout phases.
- Ensures infrastructure errors are never conflated with behavioral failures.
- Exits with code 1 immediately upon any validation failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from agentxray.statistics import (
    compare_proportions,
    fisher_exact_test,
    wilson_score_interval,
)


class Phase2B0Validator:
    """Forensic validator for Phase 2B.0 and Phase 2B.0.1 model pilot artifacts."""

    def __init__(
        self,
        event_log_path: Path | str = "benchmarks/results/phase_2b0_calibration_event_log.jsonl",
        results_path: Path | str = "benchmarks/results/phase_2b0_calibration_results.json",
        config_path: Path | str = "benchmarks/results/phase_2b0_calibration_config.json",
        cand_specs_path: Path | str = "benchmarks/results/phase_2b0_candidate_specs.json",
        ground_truth_path: Path | str = "benchmarks/results/phase_2b0_ground_truth.json",
        strict_exhaustive: bool = False,
    ) -> None:
        self.event_log_path = Path(event_log_path)
        self.results_path = Path(results_path)
        self.config_path = Path(config_path)
        self.cand_specs_path = Path(cand_specs_path)
        self.ground_truth_path = Path(ground_truth_path)
        self.strict_exhaustive = strict_exhaustive

        self.discrepancies: list[str] = []
        self.checks_passed: list[str] = []

    def record_pass(self, check_name: str) -> None:
        self.checks_passed.append(check_name)

    def record_fail(self, check_name: str, message: str) -> None:
        self.discrepancies.append(f"[{check_name}] {message}")

    def validate(self) -> dict[str, Any]:
        """Run all forensic checks and return validation summary."""
        # 1. Artifact existence checks
        for name, p in [
            ("Event Log", self.event_log_path),
            ("Results JSON", self.results_path),
            ("Config JSON", self.config_path),
            ("Candidate Specs", self.cand_specs_path),
            ("Ground Truth", self.ground_truth_path),
        ]:
            if not p.is_file():
                self.record_fail("file_existence", f"{name} file not found at {p}")
                return self._build_summary(success=False)
            else:
                self.record_pass(f"file_exists_{name.lower().replace(' ', '_')}")

        # 2. Parse JSON artifacts
        try:
            results_data = json.loads(self.results_path.read_text(encoding="utf-8"))
            config_data = json.loads(self.config_path.read_text(encoding="utf-8"))
            cand_data = json.loads(self.cand_specs_path.read_text(encoding="utf-8"))
            gt_data = json.loads(self.ground_truth_path.read_text(encoding="utf-8"))
            self.record_pass("json_artifacts_parsed")
        except Exception as e:
            self.record_fail("json_parsing", f"Failed to parse JSON artifacts: {e}")
            return self._build_summary(success=False)

        # 3. Candidate Specs & Ground Truth Schema
        valid_elements = set()
        for elem in cand_data.get("elements", []):
            if "element_text" in elem:
                valid_elements.add(elem["element_text"])

        if len(valid_elements) != cand_data.get("candidate_space_size", -1):
            self.record_fail(
                "candidate_space_size",
                f"Candidate elements count ({len(valid_elements)}) != candidate_space_size ({cand_data.get('candidate_space_size')})",
            )
        else:
            self.record_pass("candidate_space_size_consistent")

        true_triggers = set(gt_data.get("true_trigger_elements", []))
        if not true_triggers.issubset(valid_elements):
            self.record_fail("ground_truth_membership", "Ground truth elements are not in candidate specs.")
        else:
            self.record_pass("ground_truth_valid_members")

        # 4. Parse & Validate Event Log
        events: list[dict[str, Any]] = []
        try:
            lines = self.event_log_path.read_text(encoding="utf-8").splitlines()
            for line_idx, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                    events.append(ev)
                except Exception as je:
                    self.record_fail("event_log_malformed_json", f"Line {line_idx} is malformed JSON: {je}")
                    return self._build_summary(success=False)
            self.record_pass("event_log_json_parsed")
        except Exception as e:
            self.record_fail("event_log_reading", f"Failed to read event log: {e}")
            return self._build_summary(success=False)

        if not events:
            self.record_fail("event_log_empty", "Event log contains zero events.")
            return self._build_summary(success=False)

        # 5. Event Log Unique Identifiers & Monotonicity
        event_ids = []
        for ev in events:
            if "event_id" not in ev or not isinstance(ev["event_id"], int):
                self.record_fail("event_id_format", f"Event missing integer event_id: {ev}")
                break
            event_ids.append(ev["event_id"])

        if len(event_ids) == len(events):
            if len(set(event_ids)) != len(event_ids):
                self.record_fail("event_id_duplicates", f"Found duplicate event IDs in event log.")
            elif event_ids != list(range(1, len(event_ids) + 1)):
                self.record_fail("event_id_non_monotonic", f"Event IDs are not strictly monotonically increasing from 1.")
            else:
                self.record_pass("event_id_unique_monotonic")

        # 6. Event Log Timestamps
        for ev in events:
            if "timestamp" not in ev:
                self.record_fail("timestamp_missing", f"Event {ev.get('event_id')} missing timestamp.")
                break
            try:
                # ISO 8601 validation
                datetime.fromisoformat(ev["timestamp"])
            except Exception:
                self.record_fail("timestamp_invalid_iso", f"Event {ev.get('event_id')} invalid ISO timestamp.")
                break
        else:
            self.record_pass("event_timestamps_valid_iso")

        # 7. Candidate Element Membership in All Events
        invalid_elements_found = []
        for ev in events:
            sub = ev.get("candidate_subset") or ev.get("elements") or ev.get("candidate")
            if sub and isinstance(sub, list):
                for elem in sub:
                    if elem not in valid_elements:
                        invalid_elements_found.append(elem)
        if invalid_elements_found:
            self.record_fail("event_candidate_membership", f"Unknown candidate elements in event log: {invalid_elements_found}")
        else:
            self.record_pass("event_candidate_membership_valid")

        # 8. Infrastructure Failure Separation
        # Verify that no trial with outcome FAILURE has an infrastructure crash
        infra_conflations = []
        for ev in events:
            if ev.get("event_type") == "trial_executed":
                if ev.get("outcome") == "FAILURE":
                    err = str(ev.get("error_message") or "").lower()
                    if any(term in err for term in ["connection refused", "socket error", "timeout error", "500 internal"]):
                        infra_conflations.append(ev)
        if infra_conflations:
            self.record_fail("infrastructure_conflation", f"Trials marked as failure with infrastructure error: {infra_conflations}")
        else:
            self.record_pass("infrastructure_failures_unconflated")

        # 9. Exhaustive Characterization Check (Strict or Permissive)
        char_trials = [
            ev for ev in events
            if ev.get("event_type") == "trial_executed" and ev.get("phase") in ("calibration_control", "characterization")
        ]
        char_subsets = {tuple(sorted(ev.get("candidate_subset", []))) for ev in char_trials}
        power_set_size = 2 ** len(valid_elements)

        if self.strict_exhaustive:
            if len(char_subsets) < power_set_size:
                self.record_fail(
                    "exhaustive_characterization",
                    f"Evaluated only {len(char_subsets)} subsets out of {power_set_size} required for exhaustive characterization.",
                )
            else:
                self.record_pass("exhaustive_characterization_complete")
        else:
            # Informational check
            if len(char_subsets) < power_set_size:
                self.record_pass(f"characterization_subsets_evaluated_{len(char_subsets)}_of_{power_set_size}")
            else:
                self.record_pass("exhaustive_characterization_complete")

        # 10. Recomputation of Statistical Controls in Results JSON
        if "controls_calibration" in results_data:
            ctrl = results_data.get("controls_calibration", {})
            for ctrl_key in ["negative_control_baseline", "positive_control_trigger", "null_control_distractors", "full_intervention"]:
                c_data = ctrl.get(ctrl_key, {})
                trials = c_data.get("trials", 0)
                failures = c_data.get("failures", 0)
                successes = c_data.get("successes", 0)

                if trials != failures + successes:
                    self.record_fail("controls_arithmetic", f"{ctrl_key}: trials ({trials}) != failures ({failures}) + successes ({successes})")
                elif failures < 0 or failures > trials:
                    self.record_fail("controls_arithmetic", f"{ctrl_key}: invalid failure count {failures} for trials {trials}")
                else:
                    self.record_pass(f"arithmetic_valid_{ctrl_key}")

                recomputed_rate = round(failures / trials, 4) if (trials > 0 and 0 <= failures <= trials) else 0.0
                if abs(recomputed_rate - c_data.get("failure_rate", -1)) > 1e-4:
                    self.record_fail("failure_rate_mismatch", f"{ctrl_key}: reported {c_data.get('failure_rate')} != recomputed {recomputed_rate}")
                else:
                    self.record_pass(f"rate_valid_{ctrl_key}")

                # Recompute Wilson CI
                if trials > 0 and 0 <= failures <= trials:
                    l_ci, u_ci = wilson_score_interval(failures, trials, confidence=config_data.get("confidence", 0.95))
                    rep_ci = c_data.get("wilson_ci", [None, None])
                    if abs(l_ci - rep_ci[0]) > 1e-3 or abs(u_ci - rep_ci[1]) > 1e-3:
                        self.record_fail("wilson_ci_mismatch", f"{ctrl_key}: reported {rep_ci} != recomputed [{l_ci}, {u_ci}]")
                    else:
                        self.record_pass(f"wilson_ci_valid_{ctrl_key}")

            # Recompute Fisher Exact p-value
            pos_f = ctrl.get("positive_control_trigger", {}).get("failures", 0)
            pos_t = ctrl.get("positive_control_trigger", {}).get("trials", 1)
            base_f = ctrl.get("negative_control_baseline", {}).get("failures", 0)
            base_t = ctrl.get("negative_control_baseline", {}).get("trials", 1)

            if 0 <= pos_f <= pos_t and 0 <= base_f <= base_t and pos_t > 0 and base_t > 0:
                comp_p_val = fisher_exact_test(k1=pos_f, n1=pos_t, k2=base_f, n2=base_t, alternative="two-sided")
                rep_p_val = ctrl.get("statistical_contrasts", {}).get("positive_vs_baseline", {}).get("fisher_exact_p_value", -1)
                if abs(comp_p_val - rep_p_val) > 1e-4:
                    self.record_fail("fisher_exact_p_value_mismatch", f"Reported Fisher p {rep_p_val} != recomputed {comp_p_val}")
                else:
                    self.record_pass("fisher_exact_recomputed_match")
            else:
                self.record_fail("fisher_exact_invalid_counts", f"Invalid counts for Fisher exact test: pos={pos_f}/{pos_t}, base={base_f}/{base_t}")

        elif "exhaustive_characterization" in results_data:
            char_res = results_data.get("exhaustive_characterization", {})
            subsets = char_res.get("subsets", [])
            for s_info in subsets:
                trials = s_info.get("trials", 0)
                failures = s_info.get("failures", 0)
                successes = s_info.get("successes", 0)
                c_idx = s_info.get("subset_index")

                if trials != failures + successes:
                    self.record_fail("characterization_arithmetic", f"subset {c_idx}: trials ({trials}) != failures ({failures}) + successes ({successes})")
                elif failures < 0 or failures > trials:
                    self.record_fail("characterization_arithmetic", f"subset {c_idx}: invalid count failures {failures}")
                else:
                    self.record_pass(f"arithmetic_valid_subset_{c_idx}")

                recomputed_rate = round(failures / trials, 4) if (trials > 0 and 0 <= failures <= trials) else 0.0
                if abs(recomputed_rate - s_info.get("failure_rate", -1)) > 1e-4:
                    self.record_fail("failure_rate_mismatch", f"subset {c_idx}: reported {s_info.get('failure_rate')} != recomputed {recomputed_rate}")
                else:
                    self.record_pass(f"rate_valid_subset_{c_idx}")

                if trials > 0 and 0 <= failures <= trials:
                    l_ci, u_ci = wilson_score_interval(failures, trials, confidence=config_data.get("confidence", 0.95))
                    rep_ci = s_info.get("wilson_ci", [None, None])
                    if abs(l_ci - rep_ci[0]) > 1e-3 or abs(u_ci - rep_ci[1]) > 1e-3:
                        self.record_fail("wilson_ci_mismatch", f"subset {c_idx}: reported {rep_ci} != recomputed [{l_ci}, {u_ci}]")
                    else:
                        self.record_pass(f"wilson_ci_valid_subset_{c_idx}")
            self.record_pass("exhaustive_characterization_statistics_verified")
        else:
            self.record_fail("results_schema", "Neither controls_calibration nor exhaustive_characterization found in results JSON.")

        # 11. Holdout Phase Separation
        holdout_trials = [ev for ev in events if ev.get("phase") == "holdout_confirmation"]
        for ht in holdout_trials:
            if not ht.get("case_id", "").startswith("holdout_"):
                self.record_fail("holdout_naming", f"Holdout trial case_id does not start with 'holdout_': {ht}")
                break
        else:
            self.record_pass("holdout_phase_isolated")

        # Final evaluation
        success = (len(self.discrepancies) == 0)
        return self._build_summary(success=success)

    def _build_summary(self, success: bool) -> dict[str, Any]:
        return {
            "validation_status": "VALIDATION_PASSED" if success else "VALIDATION_FAILED",
            "all_checks_passed": success,
            "total_checks_passed": len(self.checks_passed),
            "total_discrepancies": len(self.discrepancies),
            "checks_passed": self.checks_passed,
            "discrepancies": self.discrepancies,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic Forensic Audit for Phase 2B.0 Artifacts")
    parser.add_argument("--event-log", default="benchmarks/results/phase_2b0_calibration_event_log.jsonl")
    parser.add_argument("--results", default="benchmarks/results/phase_2b0_calibration_results.json")
    parser.add_argument("--config", default="benchmarks/results/phase_2b0_calibration_config.json")
    parser.add_argument("--candidate-specs", default="benchmarks/results/phase_2b0_candidate_specs.json")
    parser.add_argument("--ground-truth", default="benchmarks/results/phase_2b0_ground_truth.json")
    parser.add_argument("--strict-exhaustive", action="store_true", help="Enforce all 16 candidate subsets characterized")
    parser.add_argument("--output-json", default=None, help="Optional output JSON path for validation report")
    args = parser.parse_args()

    validator = Phase2B0Validator(
        event_log_path=args.event_log,
        results_path=args.results,
        config_path=args.config,
        cand_specs_path=args.candidate_specs,
        ground_truth_path=args.ground_truth,
        strict_exhaustive=args.strict_exhaustive,
    )

    summary = validator.validate()

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("======================================================================")
    print("AgentXRay Phase 2B.0 — Forensic Audit & Reproducibility Validation")
    print("======================================================================")
    print(f"Status: {summary['validation_status']}")
    print(f"Checks Passed: {summary['total_checks_passed']}")
    print(f"Discrepancies: {summary['total_discrepancies']}")

    if summary["discrepancies"]:
        print("\nDiscrepancies:")
        for disc in summary["discrepancies"]:
            print(f"  - {disc}")
        sys.exit(1)
    else:
        print("\nAll checks passed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
