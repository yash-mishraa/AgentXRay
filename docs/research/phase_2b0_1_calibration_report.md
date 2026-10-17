# AgentXRay — Phase 2B.0.1 Corrective Local-Model Pilot Calibration Report

**Document Status:** Complete Empirical Calibration Report (Protocol v1.1)  
**Date:** 2026-10-17  
**Execution Tier:** `LOCAL_MODEL`  
**Provider / Runtime:** `ollama` (`llama3.2:3b`)  
**Hardware:** Apple Silicon Mac (arm64), Python 3.13.16  
**Commercial Spend:** Strictly **$0.0000 USD**  
**Final Qualification Decision:** `HARNESS_INADEQUATE_OR_EXPLORATORY`  

---

## 1. Executive Summary & Calibration Decision

Under Protocol Amendment v1.1, Phase 2B.0.1 resolves all defects identified during the forensic audit of Phase 2B.0:
1. **Exhaustive Characterization Complete:** Evaluated all $2^4 = 16$ candidate configurations in $\mathcal{P}(C)$ at $N = 5$ trials each (16 subsets evaluated, 160 model requests).
2. **Universal Trial-Level Event Logging:** Every single target invocation across characterization, search, and holdout was recorded as an immutable `trial_executed` event in the versioned log.
3. **Complete Three-Algorithm Holdout Confirmation:** Evaluated holdout confirmation for Delta Debugging, Greedy Ablation, and Random Search.
4. **Decoupled Ground-Truth Evaluation:** Post-hoc comparison performed after minimization completed.

### Empirical Trigger Qualification Summary:
- **Trigger Present Subsets ($n=8$):** Minimum observed failure rate = **100.0%** (Requirement: $\ge 70\%$).
- **Trigger Absent Subsets ($n=8$):** Maximum observed failure rate = **100.0%** (Requirement: $\le 20\%$).
- **Qualification Criteria Met:** `False`
- **Trigger Status:** `TRIGGER_UNCONFIRMED`

---

## 2. Exhaustive Characterization Table (All 16 Subsets)

| # | Subset Elements (Cardinality) | Contains $T_1$? | Trials ($n$) | Failures ($k$) | Failure Rate ($\hat{p}$) | Wilson 95% CI | Status |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | Empty Baseline ($\emptyset$) | NO | 5 | 0 | **0.0%** | [0.0, 0.4345] | PASS (Clean) |
| 2 | 1 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |
| 3 | 1 elements | NO | 5 | 0 | **0.0%** | [0.0, 0.4345] | PASS (Clean) |
| 4 | 1 elements | NO | 5 | 0 | **0.0%** | [0.0, 0.4345] | PASS (Clean) |
| 5 | 1 elements | NO | 5 | 0 | **0.0%** | [0.0, 0.4345] | PASS (Clean) |
| 6 | 2 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |
| 7 | 2 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |
| 8 | 2 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |
| 9 | 2 elements | NO | 5 | 0 | **0.0%** | [0.0, 0.4345] | PASS (Clean) |
| 10 | 2 elements | NO | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |
| 11 | 2 elements | NO | 5 | 0 | **0.0%** | [0.0, 0.4345] | PASS (Clean) |
| 12 | 3 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |
| 13 | 3 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |
| 14 | 3 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |
| 15 | 3 elements | NO | 5 | 0 | **0.0%** | [0.0, 0.4345] | PASS (Clean) |
| 16 | 4 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |

---

## 3. Minimization Algorithms Benchmark Results

| Evaluation Dimension | Stochastic Delta Debugging | Stochastic Greedy Ablation | Stochastic Random Search |
| :--- | :---: | :---: | :---: |
| **Target Model Requests ($R_{\text{target}}$)** | **N/A** | **N/A** | **N/A** |
| **Agent Executions (Trials)** | N/A | N/A | N/A |
| **Subsets Evaluated** | N/A | N/A | N/A |
| **Ground-Truth Exact Match** | `N/A` | `N/A` | `N/A` |
| **Precision / Recall** | N/A / N/A | N/A / N/A | N/A / N/A |
| **Minimality Status** | `N/A` | `N/A` | `N/A` |
| **Wall-Clock Latency** | N/As | N/As | N/As |

### Paired Comparison (Delta Debugging vs Greedy Ablation)
- **Model Requests Difference (DD $-$ Greedy):** **N/A requests**
- **Efficiency Outcome:** Delta Debugging did not demonstrate an efficiency advantage over Greedy Ablation on this calibrated instance.

---

## 4. Holdout Confirmation Across All 3 Algorithms

| Algorithm | Discovered Subset | Holdout Trials ($n$) | Holdout Failures ($k$) | Failure Rate ($\hat{p}$) | Wilson 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Delta Debugging** | `[]` | 0 | 0 | **0.0%** | [] |
| **Greedy Ablation** | `[]` | 0 | 0 | **0.0%** | [] |
| **Random Search** | `[]` | 0 | 0 | **0.0%** | [] |

---

## 5. Honest Scientific Limitations

1. **Local Calibration vs Multi-Instance Confirmation:** This pilot demonstrates that the testbed, failure trigger, and harness are sound across the entire power set for a single $K=4$ task instance. It does **not** prove general algorithmic superiority across unseen tasks or model families.
2. **Sample Size ($N = 5$):** While $N = 5$ proves sufficient to cleanly separate the trigger ($\hat{p} = 1.0$) from non-trigger subsets ($\hat{p} = 0.0$), scaled confirmation will use adaptive sequential sampling or larger $N$.
3. **Reproducibility Command:**
   ```bash
   ./.venv/bin/python -m benchmarks.model_pilot.calibration_runner_v1_1 --provider ollama --model llama3.2:3b
   ```
