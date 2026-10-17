# AgentXRay — Phase 3.3: Reproducibility & Verification Guide

**Document Reference:** `docs/research/phase_3_3/phase_3_3_reproducibility_guide.md`  
**Date:** 2026-10-17  
**Version:** Phase 3.3 Research Release Reproducibility Guide  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`

---

## 1. Quick Start: Offline Reproducibility & Verification Scope

AgentXRay's recorded aggregate observations, statistical recalculations, qualification outcomes, and implementation tests are deterministically verifiable offline from preserved primary raw artifacts without model inference or external network connectivity. However, specific unpersisted execution details—most notably the Turn 1 conversational payloads of the 4 aborted trials and intermediate uncommitted historical runner diffs—are unrecoverable offline and permanently missing without model re-execution (see Section 3 for full scope delineation).

### 1.1 Environment Requirements
- **Operating System:** macOS / Linux / POSIX-compatible environment.
- **Python Version:** Python 3.10+ (tested on Python 3.13.16).
- **External Dependencies:** Standard library only for statistical math (`math`, `statistics`); `pytest`, `pytest-asyncio`, and `pydantic` for testing and schema validation.
- **Model Inference Daemon:** **NOT REQUIRED.** Ollama does not need to be running, and no models need to be downloaded.
- **Network Access:** **NOT REQUIRED.** The suite operates in strict air-gapped / offline mode. Commercial API keys are never required ($0.00 USD spend).

---

## 2. Safe Execution Commands

> [!CAUTION]
> **DO NOT EXECUTE INFERENCE-CAPABLE SCRIPTS.**
> Under no circumstances should you execute:
> - `python benchmarks/model_pilot/exploratory_runner_v1_4.py`
> - `python benchmarks/model_pilot/runner.py`
> - `python benchmarks/model_pilot/calibration_runner_v1_2.py`
> 
> Doing so may attempt to connect to local Ollama or commercial providers. Use only the deterministic verification commands below.

### 2.1 Run the Full Offline Verification Suite (247 Tests)
To execute all 247 deterministic offline tests across the entire codebase:

```bash
# From repository root:
.venv/bin/pytest -v
```

*Expected Terminal Output:*
```
============================= 247 passed in 0.35s ==============================
```

### 2.2 Run the Phase 3.3 Standalone Deterministic Verification Script
To independently verify the 9 core empirical and statistical verification check suites from primary raw artifacts:

```bash
# Standalone execution:
.venv/bin/python tests/test_phase_3_3_reproducibility.py
```

*Expected Terminal Output:*
```
======================================================================
AgentXRay — Phase 3.3 Offline Deterministic Reproducibility Verification
======================================================================
[PASS] test_reproduce_01_cryptographic_hashes
[PASS] test_reproduce_02_request_ledger_and_trial_accounting
[PASS] test_reproduce_03_task_configuration_and_mechanism_counts
[PASS] test_reproduce_04_qualification_and_candidate_discovery
[PASS] test_reproduce_05_holdout_replication
[PASS] test_reproduce_06_ground_truth_recovery_and_mcnemar
[PASS] test_reproduce_07_mechanism_level_wilcoxon
[PASS] test_reproduce_08_null_control_outcomes
[PASS] test_reproduce_09_wilson_lower_bound_invariance
======================================================================
All 9 deterministic verification checks passed successfully!
Zero model calls, zero external requests, zero file modifications.
======================================================================
```

### 2.3 Run the 12 Adversarial Minimality Boundary Tests
To test the statistical decision boundary of the minimizer under edge cases:

```bash
.venv/bin/pytest tests/test_phase_3_2_adversarial_minimality.py -v
```

*Expected Terminal Output:*
```
============================== 12 passed in 0.05s ==============================
```

### 2.4 Recompute Cryptographic Checksums
To verify that raw data and frozen protocol files remain uncorrupted:

```bash
python3 -c "
import hashlib
from pathlib import Path

files = [
    'docs/research/phase_2b1_exploratory_protocol_v1_4.md',
    'benchmarks/configs/phase_2b1_exploratory_config_v1_4.json',
    'benchmarks/results/phase_2b1_5_execution/manifest.json',
    'benchmarks/results/phase_2b1_5_execution/results.json',
    'benchmarks/results/phase_2b1_5_execution/trials.jsonl',
    'benchmarks/results/phase_2b1_5_execution/summary.csv'
]
for f in files:
    h = hashlib.sha256(Path(f).read_bytes()).hexdigest()
    print(f'{f}: {h}')
"
```

*Expected Output:*
```
docs/research/phase_2b1_exploratory_protocol_v1_4.md: db45f4032f2340a9edc3441bafcb3957895b616d1289c28e85a5cc4671574f62
benchmarks/configs/phase_2b1_exploratory_config_v1_4.json: b3e3b0b8450e325914ea8229faafce90a833eb038b02903bfa48062e8ee0cfb2
benchmarks/results/phase_2b1_5_execution/manifest.json: 349c9b97a0ee76cded3770828029f897d101414d8917b9743930f08f23ea830f
benchmarks/results/phase_2b1_5_execution/results.json: 5f9606d9dad39f6a1f3eea39fdfd140f56dffaec99f8516ec7ecfa66303f26b9
benchmarks/results/phase_2b1_5_execution/trials.jsonl: 82905d2200b88c242f11499e6a927e75c38fe1e8bdf142b498dc28c9f80b86f0
benchmarks/results/phase_2b1_5_execution/summary.csv: 6e521487357bf176fab79d3a20c3f7e61657909a8132e6d5dab6f084dae71902
```

---

## 3. Scope of Offline Reconstruction

### 3.1 What CAN Be Reconstructed Deterministically Offline
1. **Trial-by-Trial Multi-Turn Payloads:** All 1,086 completed trials in `trials.jsonl` contain full prompt strings, tool names, tool arguments, model responses, and latencies.
2. **Estimands and Hypotheses:** All Wilson score confidence intervals, Newcombe difference intervals, Fisher exact p-values, binomial threshold p-values, Wilcoxon signed-rank permutation tests, and McNemar symmetry tests.
3. **Holdout Validation:** All 60 holdout trials across 12 candidate subsets.
4. **Ledger Accounting:** Reconciliation of 2,176 requests (2,172 logged + 4 aborted).

### 3.2 What CANNOT Be Reconstructed Offline (Known Limitations)
1. **The Four Aborted Trial Payloads:** The conversational payload of Turn 1 on the 4 trials terminated by `CostBudgetExceededError` was not saved to disk before execution halted.
2. **Execution on Other Models:** Evaluating larger models (`llama3.1:8b`, `qwen2.5:7b`) or frontier APIs requires fresh, separately authorized live inference runs.
3. **Non-Recorded Historical Diffs:** The unstaged intermediate edits between runner draft `03ff8262...` and executing runner `69b03d52...` exist only in commit notes and audit documentation.
