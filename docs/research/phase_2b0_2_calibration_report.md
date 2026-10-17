# AgentXRay — Phase 2B.0.2 Corrective Local-Model Pilot Calibration Report

**Document Status:** Complete Empirical Calibration Report (Protocol v1.2)  
**Date:** 2026-10-17  
**Execution Tier:** `LOCAL_MODEL`  
**Provider / Runtime:** `ollama` (`llama3.2:3b`)  
**Hardware:** Apple Silicon Mac (arm64), Python 3.13.16  
**Commercial Spend:** Strictly **$0.0000 USD**  
**Final Qualification Decision:** `CONFIRMATORY_READY`  

---

## 1. Executive Summary & Calibration Recovery Decision

Under Protocol Amendment v1.2, Phase 2B.0.2 resolves all defects diagnosed during the forensic investigation of Phase 2B.0.1:
1. **De-Conflicted Distractor Phrasing:** Replaced $D_1$ ("one-line reimbursement") with non-conflicting categorization $D_1$ ("specify expense category as Travel & Subsistence").
2. **Exhaustive Characterization Complete:** Evaluated all $2^4 = 16$ candidate configurations in $\mathcal{P}(C)$ at $N = 5$ trials each (16 subsets evaluated, 160 model requests).
3. **Explicit Element Identifiers:** Every table row records explicit element identifiers (`elem_1`, `elem_2`, etc.).
4. **Reporting Integrity Guardrails:** No phantom holdouts or deceptive 0.0% failure rates.

### Empirical Trigger Qualification Summary:
- **Trigger Present Subsets ($n=8$):** Minimum observed failure rate = **100.0%** (Requirement: $\ge 70\%$).
- **Trigger Absent Subsets ($n=8$):** Maximum observed failure rate = **0.0%** (Requirement: $\le 20\%$).
- **Qualification Criteria Met:** `True`
- **Trigger Status:** `TRIGGER_CONFIRMED`

---

## 2. Exhaustive Characterization Table (All 16 Subsets)

| Row | Subset Case ID | Candidate Element IDs | Elements Role Description | Contains $T_1$? | Trials ($n$) | Failures ($k$) | Failure Rate ($\hat{p}$) | Wilson 95% CI | Status |
| :-: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `char_subset_01_len0` | `[]` | Baseline ($\emptyset$) | NO | 5 | 0 | **0.0%** | [0.0, 0.4345] | PASS (Clean) |
| 2 | `char_subset_02_len1` | `['elem_1']` | 1 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |
| 3 | `char_subset_03_len1` | `['elem_2']` | 1 elements | NO | 5 | 0 | **0.0%** | [0.0, 0.4345] | PASS (Clean) |
| 4 | `char_subset_04_len1` | `['elem_3']` | 1 elements | NO | 5 | 0 | **0.0%** | [0.0, 0.4345] | PASS (Clean) |
| 5 | `char_subset_05_len1` | `['elem_4']` | 1 elements | NO | 5 | 0 | **0.0%** | [0.0, 0.4345] | PASS (Clean) |
| 6 | `char_subset_06_len2` | `['elem_1', 'elem_2']` | 2 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |
| 7 | `char_subset_07_len2` | `['elem_1', 'elem_3']` | 2 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |
| 8 | `char_subset_08_len2` | `['elem_1', 'elem_4']` | 2 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |
| 9 | `char_subset_09_len2` | `['elem_2', 'elem_3']` | 2 elements | NO | 5 | 0 | **0.0%** | [0.0, 0.4345] | PASS (Clean) |
| 10 | `char_subset_10_len2` | `['elem_2', 'elem_4']` | 2 elements | NO | 5 | 0 | **0.0%** | [0.0, 0.4345] | PASS (Clean) |
| 11 | `char_subset_11_len2` | `['elem_3', 'elem_4']` | 2 elements | NO | 5 | 0 | **0.0%** | [0.0, 0.4345] | PASS (Clean) |
| 12 | `char_subset_12_len3` | `['elem_1', 'elem_2', 'elem_3']` | 3 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |
| 13 | `char_subset_13_len3` | `['elem_1', 'elem_2', 'elem_4']` | 3 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |
| 14 | `char_subset_14_len3` | `['elem_1', 'elem_3', 'elem_4']` | 3 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |
| 15 | `char_subset_15_len3` | `['elem_2', 'elem_3', 'elem_4']` | 3 elements | NO | 5 | 0 | **0.0%** | [0.0, 0.4345] | PASS (Clean) |
| 16 | `char_subset_16_len4` | `['elem_1', 'elem_2', 'elem_3', 'elem_4']` | 4 elements | YES | 5 | 5 | **100.0%** | [0.5655, 1.0] | FAIL (Triggered) |

---

## 3. Minimization Algorithms Benchmark Results

| Evaluation Dimension | Stochastic Delta Debugging | Stochastic Greedy Ablation | Stochastic Random Search |
| :--- | :---: | :---: | :---: |
| **Discovered Element IDs** | `['elem_1']` | `['elem_1']` | `['elem_1']` |
| **Target Model Requests ($R_{\text{target}}$)** | **40** | **60** | **50** |
| **Agent Executions (Trials)** | 20 | 30 | 25 |
| **Subsets Evaluated** | 4 | 6 | 5 |
| **Ground-Truth Exact Match** | `True` | `True` | `True` |
| **Precision / Recall** | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| **Minimality Status** | `statistically_supported` | `statistically_supported` | `statistically_supported` |
| **Wall-Clock Latency** | 21.981s | 37.977s | 35.344s |

### Paired Comparison (Delta Debugging vs Greedy Ablation)
- **Model Requests Difference (DD $-$ Greedy):** **-20 requests**
- **Efficiency Outcome:** Delta Debugging required fewer requests than Greedy Ablation.

---

## 4. Holdout Confirmation Across Algorithms

| Algorithm | Discovered Element IDs | Holdout Trials ($n$) | Holdout Failures ($k$) | Failure Rate ($\hat{p}$) | Wilson 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **delta_debugging** | `['elem_1']` | 5 | 5 | **100.0%** | [0.5655, 1.0] |
| **greedy_ablation** | `['elem_1']` | 5 | 5 | **100.0%** | [0.5655, 1.0] |
| **random_search** | `['elem_1']` | 5 | 5 | **100.0%** | [0.5655, 1.0] |

---

## 5. Honest Scientific Limitations

1. **Local Calibration Scope:** This pilot demonstrates that the testbed, failure trigger, and de-conflicted distractors are sound across the entire power set for this $K=4$ task instance. It does **not** prove general algorithmic superiority across unseen tasks or model families.
2. **Phase 2B.1 Status:** Phase 2B.1 remains **STRICTLY UNAUTHORIZED**. Successful qualification establishes only harness reliability.
3. **Reproducibility Command:**
   ```bash
   ./.venv/bin/python -m benchmarks.model_pilot.calibration_runner_v1_2 --provider ollama --model llama3.2:3b
   ```
