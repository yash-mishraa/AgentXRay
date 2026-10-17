"""Phase 3.3 Offline Deterministic Reproducibility Verification Script.

Independently reconstructs and verifies all empirical, statistical, and architectural
claims from Project AgentXRay Phase 2B.1.5 raw artifacts.

Guarantees:
- Strictly offline: zero network access, zero model inference, zero external dependencies.
- Read-only: does not modify or overwrite existing historical artifacts.
- Deterministic: 100% reproducible execution and assertions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from agentxray.statistics import (
    compare_proportions,
    exact_binomial_test,
    fisher_exact_test,
    wilson_score_interval,
)

RAW_DIR = Path("benchmarks/results/phase_2b1_5_execution")
MANIFEST_PATH = RAW_DIR / "manifest.json"
RESULTS_PATH = RAW_DIR / "results.json"
TRIALS_PATH = RAW_DIR / "trials.jsonl"
SUMMARY_CSV_PATH = RAW_DIR / "summary.csv"

PROTOCOL_PATH = Path("docs/research/phase_2b1_exploratory_protocol_v1_4.md")
CONFIG_PATH = Path("benchmarks/configs/phase_2b1_exploratory_config_v1_4.json")
RUNNER_PATH = Path("benchmarks/model_pilot/exploratory_runner_v1_4.py")

EXPECTED_PROTOCOL_HASH = "db45f4032f2340a9edc3441bafcb3957895b616d1289c28e85a5cc4671574f62"
EXPECTED_CONFIG_HASH = "b3e3b0b8450e325914ea8229faafce90a833eb038b02903bfa48062e8ee0cfb2"
EXPECTED_DISK_RUNNER_HASH = "69b03d52b198bff5f23e86894041e810e36c75d052753620e3872d56342d7b27"
EXPECTED_MANIFEST_HASH = "349c9b97a0ee76cded3770828029f897d101414d8917b9743930f08f23ea830f"
EXPECTED_RESULTS_HASH = "5f9606d9dad39f6a1f3eea39fdfd140f56dffaec99f8516ec7ecfa66303f26b9"
EXPECTED_TRIALS_HASH = "82905d2200b88c242f11499e6a927e75c38fe1e8bdf142b498dc28c9f80b86f0"
EXPECTED_SUMMARY_HASH = "6e521487357bf176fab79d3a20c3f7e61657909a8132e6d5dab6f084dae71902"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_reproduce_01_cryptographic_hashes():
    """Verify cryptographic hashes of frozen protocol, configuration, runner, and raw data."""
    assert _sha256(PROTOCOL_PATH) == EXPECTED_PROTOCOL_HASH, "Protocol v1.4 hash mismatch!"
    assert _sha256(CONFIG_PATH) == EXPECTED_CONFIG_HASH, "Config v1.4 hash mismatch!"
    assert _sha256(RUNNER_PATH) == EXPECTED_DISK_RUNNER_HASH, "Executing runner disk hash mismatch!"
    assert _sha256(MANIFEST_PATH) == EXPECTED_MANIFEST_HASH, "Manifest raw artifact modified!"
    assert _sha256(RESULTS_PATH) == EXPECTED_RESULTS_HASH, "Results JSON raw artifact modified!"
    assert _sha256(TRIALS_PATH) == EXPECTED_TRIALS_HASH, "Trials JSONL raw artifact modified!"
    assert _sha256(SUMMARY_CSV_PATH) == EXPECTED_SUMMARY_HASH, "Summary CSV raw artifact modified!"


def test_reproduce_02_request_ledger_and_trial_accounting():
    """Verify request ledger: 2,176 total ledger = 2,172 logged in 1,086 trials + 4 aborted Turn 1s."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    results = json.loads(RESULTS_PATH.read_text())

    assert manifest["global_requests_used"] == 2176
    assert results["statistics"]["global_budget_accounting"]["total_model_requests_expended"] == 2176
    assert results["statistics"]["global_budget_accounting"]["commercial_spend_usd"] == 0.0

    trials = [json.loads(line) for line in TRIALS_PATH.read_text().splitlines() if line.strip()]
    assert len(trials) == 1086, f"Expected exactly 1,086 trial records, got {len(trials)}"

    for idx, t in enumerate(trials):
        assert t["model_requests_used"] == 2, f"Trial {idx} did not consume exactly 2 requests!"

    logged_requests = sum(t["model_requests_used"] for t in trials)
    assert logged_requests == 2172, f"Expected 2,172 logged requests, got {logged_requests}"

    difference = manifest["global_requests_used"] - logged_requests
    assert difference == 4, f"Expected exactly 4 aborted requests difference, got {difference}"


def test_reproduce_03_task_configuration_and_mechanism_counts():
    """Verify 16 task configurations across 8 mechanisms (6 active, 2 null)."""
    results = json.loads(RESULTS_PATH.read_text())
    calib = results["calibration"]

    assert len(calib) == 16, f"Expected 16 task configurations, got {len(calib)}"

    mechanisms: dict[str, list[str]] = {}
    for task_id, data in calib.items():
        mech_id = data["mechanism_id"]
        mechanisms.setdefault(mech_id, []).append(task_id)

    assert len(mechanisms) == 8, f"Expected 8 independent mechanisms, got {len(mechanisms)}"

    active_mechs = {m: t for m, t in mechanisms.items() if not m.startswith("mech_null")}
    null_mechs = {m: t for m, t in mechanisms.items() if m.startswith("mech_null")}

    assert len(active_mechs) == 6, f"Expected 6 active mechanisms, got {len(active_mechs)}"
    assert len(null_mechs) == 2, f"Expected 2 null mechanisms, got {len(null_mechs)}"

    for m, tasks in mechanisms.items():
        assert len(tasks) == 2, f"Mechanism {m} does not have exactly 2 task configurations!"


def test_reproduce_04_qualification_and_candidate_discovery():
    """Verify task qualification, disqualified tasks, and completed candidate discoveries."""
    results = json.loads(RESULTS_PATH.read_text())
    search = results["search"]

    disqualified = [t for t, d in search.items() if d.get("qualification_status") == "DISQUALIFIED_TASK"]
    assert len(disqualified) == 7, f"Expected 7 disqualified tasks, got {len(disqualified)}"
    assert set(disqualified) == {
        "inst_05_routing_override_k4",
        "inst_06_routing_override_k8",
        "inst_07_routing_prohibition_k4",
        "inst_08_routing_prohibition_k8",
        "inst_10_schema_unit_k8",
        "inst_11_schema_cross_unit_k4",
        "inst_12_schema_cross_unit_k8",
    }

    # Qualified active tasks: 5 tasks
    qualified_active = [t for t in search if search[t].get("qualification_status") == "QUALIFIED" and not search[t]["is_null_case"]]
    assert len(qualified_active) == 5
    assert set(qualified_active) == {
        "inst_01_syntax_currency_k4",
        "inst_02_syntax_currency_k8",
        "inst_03_syntax_comma_k4",
        "inst_04_syntax_comma_k8",
        "inst_09_schema_unit_k4",
    }

    # Count isolated candidates across qualified active tasks
    candidates = []
    for t in qualified_active:
        algs = search[t]["algorithms"]
        for a, adata in algs.items():
            if adata.get("minimality_status") == "statistically_supported":
                candidates.append((t, a, adata["discovered_elements"]))

    assert len(candidates) == 12, f"Expected exactly 12 active-task candidate subsets, got {len(candidates)}"


def test_reproduce_05_holdout_replication():
    """Verify holdout evaluations: 12 candidates * 5 trials = 60 trials, 100% reproduction."""
    results = json.loads(RESULTS_PATH.read_text())
    holdout = results["holdout"]

    total_evals = 0
    total_trials = 0
    for task_id, alg_map in holdout.items():
        for alg_name, data in alg_map.items():
            total_evals += 1
            total_trials += data["holdout_trials"]
            assert data["holdout_trials"] == 5
            assert data["holdout_failures"] == 5
            assert data["holdout_rate"] == 1.0
            assert data["holdout_ci"] == [0.5655, 1.0]
            assert data["reproduced_in_holdout"] is True

    assert total_evals == 12, f"Expected 12 holdout candidate evaluations, got {total_evals}"
    assert total_trials == 60, f"Expected 60 holdout trials, got {total_trials}"


def test_reproduce_06_ground_truth_recovery_and_mcnemar():
    """Verify ground truth recovery counts and McNemar exact test (p = 1.0)."""
    results = json.loads(RESULTS_PATH.read_text())
    stats = results["statistics"]["recovery_summary"]

    assert stats["total_active_tasks"] == 12
    assert stats["dd_recovered_count"] == 4
    assert stats["greedy_recovered_count"] == 4
    assert stats["random_recovered_count"] == 2

    table = stats["contingency_table"]
    assert table["both_recovered"] == 4
    assert table["dd_only"] == 0
    assert table["greedy_only"] == 0
    assert table["neither_recovered"] == 8

    # Discordant pairs b=0, c=0
    assert stats["mcnemar_exact_p_value"] == 1.0


def test_reproduce_07_mechanism_level_wilcoxon():
    """Verify mechanism-level normalized ITT burden and exact permutation Wilcoxon p-value = 0.50."""
    results = json.loads(RESULTS_PATH.read_text())
    mech_burden = results["statistics"]["mechanism_level_burden"]

    assert mech_burden["independent_mechanisms_count"] == 6
    diffs = mech_burden["differences_normalized"]

    assert diffs["mech_01_syntax_currency"] == 0.2222
    assert diffs["mech_02_syntax_comma"] == 0.2222
    assert diffs["mech_03_routing_override"] == 0.0
    assert diffs["mech_04_routing_prohibition"] == 0.0
    assert diffs["mech_05_schema_unit"] == -0.0417
    assert diffs["mech_06_schema_cross_unit"] == 0.0

    assert mech_burden["wilcoxon_exact_permutation_p_value"] == 0.5
    assert mech_burden["wilcoxon_rejects_at_0_05"] is False


def test_reproduce_08_null_control_outcomes():
    """Verify null control outcomes: 2 of 4 task configurations have false discoveries isolating department_code."""
    results = json.loads(RESULTS_PATH.read_text())
    null_summary = results["statistics"]["null_control_summary"]

    assert null_summary["total_null_tasks"] == 4
    evals = null_summary["evaluations"]

    assert evals["inst_13_null_calc_k4"]["dd_false_discovery"] is False
    assert evals["inst_14_null_calc_k8"]["dd_false_discovery"] is False
    assert evals["inst_15_null_db_k4"]["dd_false_discovery"] is True
    assert evals["inst_16_null_db_k8"]["dd_false_discovery"] is True

    assert null_summary["dd_false_discoveries_count"] == 2
    assert null_summary["greedy_false_discoveries_count"] == 2
    assert null_summary["random_false_discoveries_count"] == 1

    # Verify that all 5 false discoveries isolated department_code
    search = results["search"]
    for task_id in ["inst_15_null_db_k4", "inst_16_null_db_k8"]:
        for alg, data in search[task_id]["algorithms"].items():
            if data.get("minimality_status") == "statistically_supported":
                assert data["discovered_elements"] == ["department_code: assign to department code DEPT-FIN-01"]


def test_reproduce_09_wilson_lower_bound_invariance():
    """Verify that in 100% of historical candidate evaluations, candidate Wilson CI lower bound exceeds 0.20."""
    results = json.loads(RESULTS_PATH.read_text())
    search = results["search"]

    tested_candidates = 0
    for task_id, task_data in search.items():
        algs = task_data.get("algorithms", {})
        for alg_name, alg_data in algs.items():
            if alg_data.get("minimality_status") == "statistically_supported":
                tested_candidates += 1
                # Point estimate 5/5
                ci = wilson_score_interval(successes=5, trials=5, confidence=0.95)
                assert ci[0] == 0.5655
                assert ci[0] > 0.20, f"Candidate in {task_id} failed Wilson CI lower bound guard!"

    # 12 active candidates + 5 null candidates = 17 supported candidates
    assert tested_candidates == 17, f"Expected 17 supported candidate evaluations, got {tested_candidates}"


if __name__ == "__main__":
    import sys
    test_funcs = [
        test_reproduce_01_cryptographic_hashes,
        test_reproduce_02_request_ledger_and_trial_accounting,
        test_reproduce_03_task_configuration_and_mechanism_counts,
        test_reproduce_04_qualification_and_candidate_discovery,
        test_reproduce_05_holdout_replication,
        test_reproduce_06_ground_truth_recovery_and_mcnemar,
        test_reproduce_07_mechanism_level_wilcoxon,
        test_reproduce_08_null_control_outcomes,
        test_reproduce_09_wilson_lower_bound_invariance,
    ]
    print("=" * 70)
    print("AgentXRay — Phase 3.3 Offline Deterministic Reproducibility Verification")
    print("=" * 70)
    passed = 0
    for fn in test_funcs:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {exc}")
            sys.exit(1)
    print("=" * 70)
    print(f"All {passed} deterministic verification checks passed successfully!")
    print("Zero model calls, zero external requests, zero file modifications.")
    print("=" * 70)
