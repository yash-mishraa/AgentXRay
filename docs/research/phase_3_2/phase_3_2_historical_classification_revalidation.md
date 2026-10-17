# AgentXRay — Phase 3.2: Historical Classification Revalidation

**Document Reference:** `docs/research/phase_3_2/phase_3_2_historical_classification_revalidation.md`  
**Date:** 2026-10-17  
**Version:** Phase 3.2 Historical Classification Revalidation  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Authoritative Data Sources:**
- `benchmarks/results/phase_2b1_5_execution/results.json`
- `benchmarks/results/phase_2b1_5_execution/trials.jsonl`
- `benchmarks/results/phase_2b1_5_execution/manifest.json`

---

## 1. Executive Summary

This document presents a complete, candidate-by-candidate forensic revalidation of all candidate configurations reported from the Phase 2B.1.5 exploratory benchmark. 

The central empirical question is:
> **Did the implementation omission of the candidate Wilson lower bound guard ($L_{95} > 0.20$) alter or invalidate any published historical minimality classifications?**

### Definitive Revalidation Finding:
**Zero historical classifications are changed.** In 100% of cases where an algorithm isolated a candidate configuration, the candidate exhibited **5/5 failures** ($k_1=5, n_1=5 \implies \hat{p}=1.00$), producing a Wilson 95% confidence interval of $[0.5655, 1.0000]$. Because the lower bound $L_{95} = 0.5655$ strictly exceeds the protocol threshold $\theta_{\text{base}} = 0.20$, every candidate satisfies the full Protocol v1.4 §6.2 criteria.

All 12 active-task candidate subsets and all 5 null-control candidate subsets are **100% confirmed as valid `SUPPORTED_1_MINIMAL`**.

---

## 2. Definitional Clarity: Three Distinct Scientific Concepts

To eliminate conceptual confusion in literature reviews and audit reports, AgentXRay maintains strict separation between three orthogonal dimensions of candidate evaluation:

```
+----------------------------------------------------------------------------------------------------+
|                                  THREE INDEPENDENT EVALUATION AXES                                 |
+------------------------------------+------------------------------------+--------------------------+
| 1. Ground Truth Match              | 2. Operational 1-Minimality        | 3. Holdout Reproduction  |
| (Planted Trigger Recovery)         | (Intervention-Union Test)          | (Out-of-Sample Stability)|
+------------------------------------+------------------------------------+--------------------------+
| Did the algorithm isolate the exact| Does removing any single element   | Does the isolated subset |
| synthetically planted causal trigger| cause a statistically significant  | reliably cause failure on|
| elements ($C = T$)?                | drop in failure rate in the model? | fresh, unseen random     |
|                                    |                                    | prompt seeds?            |
| Evaluation: Binary (True / False)  | Evaluation: Statistical IUT Status | Evaluation: 5/5 Failures |
+------------------------------------+------------------------------------+--------------------------+
```

An isolated candidate configuration can be:
- **Operationally 1-minimal** without matching synthetic ground truth (e.g. `inst_09_schema_unit_k4`, where the model exhibited an emergent prompt interaction requiring both the unit and a distractor to fail).
- **Operationally 1-minimal** on a null-control task where no failure was planted (e.g. `inst_15` and `inst_16`, where the local model inherently failed on `department_code`).
- **Reproducible in holdout** regardless of whether it matches synthetic ground truth.

---

## 3. Candidate-by-Candidate Revalidation Ledger

The following table compiles the complete forensic record for all 17 algorithmic evaluations across the benchmark:

| Instance ID | Algorithm | Discovered Subset ($C$) | $\|C\|$ | Cand Fail ($k_1/n_1$) | Cand Wilson 95% CI | Deletion Failures ($k_2/n_2$) | Diff $[95\%\text{ CI}]$ | Fisher $p_{\text{comp}}$ | Binomial $p_{\text{thresh}}$ | Holdout ($5\text{ trials}$) | Historical Status | Revalidated Protocol v1.4 Status | Audit Disposition |
| :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- | :--- | :---: |
| **inst_01** ($K=4$) | DD | `currency_syntax` | 1 | 5/5 | $[0.5655, 1.0000]$ | 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040 | 0.0312 | 5/5 (100%) | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_01** ($K=4$) | Greedy | `currency_syntax` | 1 | 5/5 | $[0.5655, 1.0000]$ | 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040 | 0.0312 | 5/5 (100%) | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_01** ($K=4$) | Random | `currency_syntax` | 1 | 5/5 | $[0.5655, 1.0000]$ | 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040 | 0.0312 | 5/5 (100%) | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_02** ($K=8$) | DD | `currency_syntax` | 1 | 5/5 | $[0.5655, 1.0000]$ | 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040 | 0.0312 | 5/5 (100%) | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_02** ($K=8$) | Greedy | `currency_syntax` | 1 | 5/5 | $[0.5655, 1.0000]$ | 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040 | 0.0312 | 5/5 (100%) | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_02** ($K=8$) | Random | Capped at 180 req | — | — | — | — | — | — | — | — | `INCONCLUSIVE` | `INCONCLUSIVE` | **CONFIRMED** |
| **inst_03** ($K=4$) | DD | `comma_syntax` | 1 | 5/5 | $[0.5655, 1.0000]$ | 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040 | 0.0312 | 5/5 (100%) | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_03** ($K=4$) | Greedy | `comma_syntax` | 1 | 5/5 | $[0.5655, 1.0000]$ | 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040 | 0.0312 | 5/5 (100%) | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_03** ($K=4$) | Random | `comma_syntax` | 1 | 5/5 | $[0.5655, 1.0000]$ | 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040 | 0.0312 | 5/5 (100%) | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_04** ($K=8$) | DD | `comma_syntax` | 1 | 5/5 | $[0.5655, 1.0000]$ | 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040 | 0.0312 | 5/5 (100%) | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_04** ($K=8$) | Greedy | `comma_syntax` | 1 | 5/5 | $[0.5655, 1.0000]$ | 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040 | 0.0312 | 5/5 (100%) | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_04** ($K=8$) | Random | Capped at 180 req | — | — | — | — | — | — | — | — | `INCONCLUSIVE` | `INCONCLUSIVE` | **CONFIRMED** |
| **inst_09** ($K=4$) | DD | `incompat_unit` + `expense_cat` | 2 | 5/5 | $[0.5655, 1.0000]$ | 0/5, 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040, 0.0040 | 0.0312, 0.0312 | 5/5 (100%) | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_09** ($K=4$) | Greedy | `incompat_unit` + `archival_flag` | 2 | 5/5 | $[0.5655, 1.0000]$ | 0/5, 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040, 0.0040 | 0.0312, 0.0312 | 5/5 (100%) | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_09** ($K=4$) | Random | Capped at 120 req | — | — | — | — | — | — | — | — | `INCONCLUSIVE` | `INCONCLUSIVE` | **CONFIRMED** |
| **inst_13** ($K=4$) | All | None | 0 | 0/5 | $[0.0000, 0.4345]$ | — | — | — | — | — | `NO_TRIGGER` | `NOT_APPLICABLE` | **CONFIRMED** |
| **inst_14** ($K=8$) | All | None | 0 | 0/5 | $[0.0000, 0.4345]$ | — | — | — | — | — | `NO_TRIGGER` | `NOT_APPLICABLE` | **CONFIRMED** |
| **inst_15** ($K=4$) | DD | `department_code` | 1 | 5/5 | $[0.5655, 1.0000]$ | 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040 | 0.0312 | — | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_15** ($K=4$) | Greedy | `department_code` | 1 | 5/5 | $[0.5655, 1.0000]$ | 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040 | 0.0312 | — | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_15** ($K=4$) | Random | `department_code` | 1 | 5/5 | $[0.5655, 1.0000]$ | 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040 | 0.0312 | — | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_16** ($K=8$) | DD | `department_code` | 1 | 5/5 | $[0.5655, 1.0000]$ | 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040 | 0.0312 | — | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_16** ($K=8$) | Greedy | `department_code` | 1 | 5/5 | $[0.5655, 1.0000]$ | 0/5 | $+1.00$ $[0.39, 1.00]$ | 0.0040 | 0.0312 | — | `STAT_SUPPORTED` | `SUPPORTED_1_MINIMAL` | **CONFIRMED** |
| **inst_16** ($K=8$) | Random | Capped at 180 req | — | — | — | — | — | — | — | — | `INCONCLUSIVE` | `INCONCLUSIVE` | **CONFIRMED** |

---

## 4. In-Depth Analysis of Special Cases

### 4.1 The Two-Element Candidates on `inst_09_schema_unit_k4`
On `inst_09`, the planted synthetic trigger was the singleton element:
$$T = \{\text{"incompatible\_unit: specify target unit to\_unit as light\_years"}\}$$

However, the minimization algorithms discovered 2-element interacting subsets:
- **Stochastic Delta Debugging:** $C_{\text{DD}} = \{\text{incompatible\_unit}, \text{expense\_category}\}$
- **Stochastic Greedy Ablation:** $C_{\text{Greedy}} = \{\text{incompatible\_unit}, \text{archival\_flag}\}$

#### Statistical Minimality Verification:
- Evaluation of $C$: 5/5 failures ($\hat{p} = 1.00$, Wilson CI $[0.5655, 1.0000]$).
- For $C_{\text{DD}}$:
  - Removal of `incompatible_unit`: 0/5 failures ($\hat{p} = 0.00$, Fisher $p = 0.0040 < 0.05$).
  - Removal of `expense_category`: 0/5 failures ($\hat{p} = 0.00$, Fisher $p = 0.0040 < 0.05$).
  - Under the Intersection–Union Test: $\max(p_1, p_2) = 0.0040 < 0.05$.
- For $C_{\text{Greedy}}$:
  - Removal of `incompatible_unit`: 0/5 failures ($\hat{p} = 0.00$, Fisher $p = 0.0040 < 0.05$).
  - Removal of `archival_flag`: 0/5 failures ($\hat{p} = 0.00$, Fisher $p = 0.0040 < 0.05$).
  - Under the Intersection–Union Test: $\max(p_1, p_2) = 0.0040 < 0.05$.

#### Scientific Interpretation:
In the actual prompt execution on local `llama3.2:3b`, removing the distractor altered the context window structure such that the model avoided generating the invalid tool call. Thus, within the empirical sample, both elements were strictly necessary to induce failure. 

$C_{\text{DD}}$ and $C_{\text{Greedy}}$ are **operationally 1-minimal** and **reproduced 100% in holdout (5/5)**, but **failed ground truth match (`matches_ground_truth: False`)**. This finding provides powerful empirical evidence that black-box minimizers discover empirically minimal failure conditions that may diverge from synthetic developer assumptions.

### 4.2 Database Null-Control False Discoveries (`inst_15` & `inst_16`)
On `inst_15` ($K=4$) and `inst_16` ($K=8$), no causal trigger was planted (null tasks). 

Yet, all five completing algorithmic runs isolated the identical singleton:
$$C = \{\text{"department\_code: assign to department code DEPT-FIN-01"}\}$$

#### Statistical Breakdown:
- Candidate failure: 5/5 failures ($\hat{p} = 1.00$, Wilson CI $[0.5655, 1.0000]$).
- Removal of `department_code`: 0/5 failures ($\hat{p} = 0.00$, Fisher $p = 0.0040 < 0.05$).
- Operational status: `SUPPORTED_1_MINIMAL`.

#### Terminology Audit:
- **Algorithmic Discovery:** Yes. The algorithms systematically isolated a minimal subset that satisfies operational 1-minimality.
- **Empirical False Discovery Relative to Planted Design:** Yes. The experimental design assumed the database mechanism had zero failure triggers. The prompt induced an unexpected failure mode in `llama3.2:3b` when that specific department code string was processed.
- **Observed Empirical False-Discovery Proportion:** Not an inferential statistical false discovery rate (FDR), but a purely descriptive proportion: 2 out of 4 null task configurations (50.0%) exhibited algorithmic discoveries, which all derived from 1 of the 2 underlying null mechanisms (`mech_null_02_db`, 50.0%). Statistical FDR cannot be legitimately inferred from 4 null configurations without an inferential mixture model.

---

## 5. Revalidation Conclusion

Every single historical candidate classification in Project AgentXRay is **rigorously revalidated and confirmed**. The omission of the Wilson CI lower bound check in the original minimizer code did not alter any scientific outcome because every candidate in the historical dataset exhibited 5/5 failures, well above the required threshold.
