# AgentXRay — Phase 3.2: Implementation–Protocol Conformance Audit

**Document Reference:** `docs/research/phase_3_2/phase_3_2_implementation_conformance_audit.md`  
**Date:** 2026-10-17  
**Version:** Phase 3.2 Conformance Audit  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Authoritative Specifications:**
- Frozen Protocol v1.4: `docs/research/phase_2b1_exploratory_protocol_v1_4.md` (SHA-256: `db45f4032f2340a9edc3441bafcb3957895b616d1289c28e85a5cc4671574f62`)
- Frozen Configuration v1.4: `benchmarks/configs/phase_2b1_exploratory_config_v1_4.json` (SHA-256: `b3e3b0b8450e325914ea8229faafce90a833eb038b02903bfa48062e8ee0cfb2`)
- Implementing Sources: `src/agentxray/stochastic_minimizer.py`, `src/agentxray/stochastic_baselines.py`, `src/agentxray/statistics.py`

---

## 1. Executive Summary and Audit Scope

This document presents an offline methodological and implementation audit evaluating whether the algorithm implementations in Project AgentXRay conform to the frozen Protocol v1.4 specification, specifically regarding the statistical classification of `SUPPORTED_1_MINIMAL`.

### 1.1 Central Audit Questions
1. **Does the implementation actually enforce every condition required by Protocol v1.4 §6.2 for `SUPPORTED_1_MINIMAL`?**
2. **What is the precise operational role of `all_thresh_supported` versus `all_statistically_significant`?**
3. **Does any confirmed discrepancy invalidate the published historical classifications?**

### 1.2 Summary of Findings
- **Claim A vs. Claim B Role Discrepancy Resolved:** In `stochastic_minimizer.py` lines 401–411, `all_thresh_supported` (the one-sample binomial test against $\theta_{\text{fail}} = 0.50$) is calculated and interpolated into the status message string, but is not an operational condition of the return branch for `STATISTICALLY_SUPPORTED`. **This behavior is fully conformant with Protocol v1.4 §6.2**, which explicitly defines `SUPPORTED_1_MINIMAL` using comparative reduction under the Intersection–Union Test (Claim B) with an empirical non-failure guard ($\hat{p} < 0.50$), while designating Claim A binomial p-values as diagnostic telemetry.
- **Implementation Defect Confirmed (ERR-08):** In `stochastic_minimizer.py` (line 342) and `stochastic_baselines.py` (line 42), the code verified $\hat{p}(C) \ge 0.50$ but **omitted** the explicit check for the two-sided Wilson 95% confidence interval lower bound $L_{95}(C) > \theta_{\text{base}} = 0.20$ required by Protocol v1.4 §6.2 Item 2.
- **Historical Impact: ZERO.** Forensic revalidation of all 12 reported algorithm-task candidates across the experimental dataset reveals that 100% of candidate configurations achieved 5/5 failures ($k_1=5, n_1=5 \implies \hat{p}=1.00$), yielding a Wilson 95% CI of $[0.5655, 1.0000]$. Because $L_{95} = 0.5655 > 0.20$, every candidate strictly satisfied the candidate Wilson lower bound condition. **Zero historical classifications are changed or invalidated.**

---

## 2. Protocol Specification vs. Implementation Analysis

### 2.1 The Four Protocol v1.4 §6.2 Requirements
Protocol v1.4 §6.2 defines `SUPPORTED_1_MINIMAL` via four mutually necessary conditions:

$$\text{Candidate } C \text{ is } \texttt{SUPPORTED\_1\_MINIMAL} \iff \text{Cond}_1 \land \text{Cond}_2 \land \text{Cond}_3 \land \text{Cond}_4$$

| # | Requirement | Mathematical Definition | Protocol Section | Historical Implementation Status |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Verification Complete** | All $k = \|C\|$ single-element deletions $C \setminus \{e_j\}$ evaluated | Protocol v1.4 §6.2 Item 1 | **CONFORMANT** (`missing_checks = set(cand) - set(removal_evals.keys())`) |
| **2** | **Candidate Failure Reproduction** | $\hat{p}(C) \ge \theta_{\text{fail}} = 0.50$ AND $L_{95}(C) > \theta_{\text{base}} = 0.20$ | Protocol v1.4 §6.2 Item 2 | **PARTIALLY CONFORMANT (DEFECT ERR-08)**: Point estimate checked ($\hat{p} \ge 0.50$); Wilson lower bound guard omitted |
| **3** | **Threshold Non-Failure Guard** | Every deletion satisfies $\hat{p}(C \setminus \{e_j\}) < \theta_{\text{fail}} = 0.50$ | Protocol v1.4 §6.2 Item 3 | **CONFORMANT** (`still_failing` list returns `NOT_MINIMAL` before Fisher tests) |
| **4** | **Comparative Statistical Support under IUT** | $\max_{j=1,\dots,k} p_j^{\text{fisher}} < 0.05$ where $p_j^{\text{fisher}}$ tests $H_{0, j}^{\text{comp}}: p(C \setminus \{e_j\}) \ge p(C)$ | Protocol v1.4 §6.2 Item 4 | **CONFORMANT** (`all(p < self.alpha for p in p_values_comp)`) |

### 2.2 Forensic Inspection of Decision Branches

In `src/agentxray/stochastic_minimizer.py`, lines 330–424 govern the classification:

```python
334: # Check 1: Missing checks
335: missing_checks = set(current_candidate) - set(removal_evals.keys())
336: if missing_checks:
337:     return (StochasticMinimalityStatus.INCONCLUSIVE, f"Missing removal evaluations for elements: {missing_checks}")

341: # Check 2: Candidate failure reproduction
342: if current_eval.trials == 0 or current_eval.failure_rate < self.failure_threshold:
343:     return (StochasticMinimalityStatus.INCONCLUSIVE, f"Candidate failed to reproduce failure rate...")

350: # Check 3: Empirical threshold non-failure guard
351: still_failing = [elem for elem, rem in removal_evals.items() if rem.failure_rate >= self.failure_threshold]
352: if still_failing:
353:     return (StochasticMinimalityStatus.NOT_MINIMAL, f"Candidate is not 1-minimal...")

401: # Check 4: Intersection-Union Test (IUT)
402: all_statistically_significant = all(p < self.alpha for p in p_values_comp) and len(p_values_comp) == len(current_candidate)
403: all_observed_drops = all(observed_drops) and len(observed_drops) == len(current_candidate)
404: all_thresh_supported = all(p < self.alpha for p in p_values_thresh) and len(p_values_thresh) == len(current_candidate)

405: if all_statistically_significant:
406:     msg = f"1-Minimality statistically supported... (threshold binomial all p < {self.alpha}: {all_thresh_supported}; ...)"
407:     return (StochasticMinimalityStatus.STATISTICALLY_SUPPORTED, msg)
```

#### Detailed Branch Analysis:
1. **Does `still_failing` execute before Fisher testing?**  
   **Yes.** Lines 350–359 strictly execute before lines 361–424. If any single-element removal produces $\hat{p} \ge 0.50$, the function returns `NOT_MINIMAL` immediately. This guarantees that an underpowered test or an anomalous comparative p-value cannot bypass the empirical non-failure guard.
2. **Why was `all_thresh_supported` included in `msg`?**  
   Protocol v1.4 §6.2 notes: *"Concurrently, exact one-sample binomial p-values $p_j^{\text{thresh}}$ against $\theta_{\text{fail}} = 0.50$ (Claim A) are computed and reported in telemetry for full diagnostic visibility."* Line 408 formats this boolean into `msg` so downstream loggers and analyzers can observe whether Claim A was concurrently satisfied. It was not intended to be a gating boolean for Claim B.
3. **What was omitted in Check 2?**  
   Line 342 checked `current_eval.failure_rate < self.failure_threshold` ($\hat{p} < 0.50$), but failed to inspect `current_eval.confidence_interval[0] <= 0.20`. If an evaluation with small $n$ (e.g. $k=1, n=2 \implies \hat{p}=0.50, L_{95}=0.0945$) had been evaluated, it would have passed Check 2 despite failing Protocol v1.4 §6.2 Item 2.

---

## 3. Protocol-to-Code-to-Test Conformance Matrix

The following matrix maps every requirement of Protocol v1.4 to its implementation and adversarial verification test:

| Protocol Requirement | Formal Spec Reference | Implementation Location | Operational Expression | Adversarial Test Case |
| :--- | :--- | :--- | :--- | :--- |
| **All Deletions Evaluated** | Protocol v1.4 §6.2 (1) | `stochastic_minimizer.py:335` `stochastic_baselines.py:37` | `set(cand) - set(rem.keys())` | `test_case_06_one_deletion_missing` |
| **No Incomplete/Zero Trials** | Protocol v1.4 §5.1, §6.2 | `stochastic_minimizer.py:341` `stochastic_baselines.py:41` | `any(r.trials == 0 for r in rem.values())` | `test_case_07_one_deletion_malformed_zero_trials` |
| **Candidate Point Estimate Guard** | Protocol v1.4 §6.2 (2) | `stochastic_minimizer.py:349` `stochastic_baselines.py:48` | `current_eval.failure_rate >= 0.50` | `test_case_04_candidate_failure_reproduction_fails` |
| **Candidate Wilson Lower Bound** | Protocol v1.4 §6.2 (2) | `stochastic_minimizer.py:350` `stochastic_baselines.py:49` | `current_eval.ci[0] > 0.20` | `test_case_05_candidate_wilson_lower_bound_fails` |
| **Empirical Non-Failure Guard** | Protocol v1.4 §6.2 (3) | `stochastic_minimizer.py:358` `stochastic_baselines.py:56` | `all(r.failure_rate < 0.50 for r in rem)` | `test_case_02_comparative_passes_but_empirical_deletion_guard_fails` |
| **Comparative Fisher IUT** | Protocol v1.4 §6.2 (4) | `stochastic_minimizer.py:412` `stochastic_baselines.py:80` | `all(p < 0.05 for p in p_comp)` | `test_case_01_all_checks_pass`, `test_case_08_one_deletion_fails_comparative_test` |
| **Strict Alpha Boundary** | Protocol v1.4 §6.2 (4) | `stochastic_minimizer.py:412` `stochastic_baselines.py:80` | `p < alpha` (strict inequality) | `test_case_12_boundary_p_values_around_alpha` |
| **Diagnostic Binomial Telemetry** | Protocol v1.4 §6.2 (end) | `stochastic_minimizer.py:397, 419` | `all_thresh_supported` formatted in `msg` | `test_case_03_comparative_passes_threshold_binomial_fails` |
| **Single-Element Candidate Support** | Protocol v1.4 §6.2 | `stochastic_minimizer.py:382` | `current_candidate = (e,)`, rem=`()` | `test_case_10_single_element_candidate` |
| **Identical Rates Non-Minimality** | Protocol v1.4 §6.3 | `stochastic_minimizer.py:358, 433` | `cand.failures == rem.failures` | `test_case_09_candidate_and_deletion_identical_rates` |
| **Status/Message Consistency** | Protocol v1.4 §6.3 | `stochastic_minimizer.py:330-435` | Return tuples `(Status, msg)` | `test_case_11_status_and_message_consistency` |

---

## 4. Adversarial Synthetic Test Suite Results

To verify every boundary condition without model inference, a dedicated adversarial test suite was implemented in [`tests/test_phase_3_2_adversarial_minimality.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/tests/test_phase_3_2_adversarial_minimality.py).

### 4.1 Summary of Executed Boundary Tests

```bash
.venv/bin/pytest tests/test_phase_3_2_adversarial_minimality.py -v
```

```
tests/test_phase_3_2_adversarial_minimality.py::test_case_01_all_checks_pass PASSED
tests/test_phase_3_2_adversarial_minimality.py::test_case_02_comparative_passes_but_empirical_deletion_guard_fails PASSED
tests/test_phase_3_2_adversarial_minimality.py::test_case_03_comparative_passes_threshold_binomial_fails PASSED
tests/test_phase_3_2_adversarial_minimality.py::test_case_04_candidate_failure_reproduction_fails PASSED
tests/test_phase_3_2_adversarial_minimality.py::test_case_05_candidate_wilson_lower_bound_fails PASSED
tests/test_phase_3_2_adversarial_minimality.py::test_case_06_one_deletion_missing PASSED
tests/test_phase_3_2_adversarial_minimality.py::test_case_07_one_deletion_malformed_zero_trials PASSED
tests/test_phase_3_2_adversarial_minimality.py::test_case_08_one_deletion_fails_comparative_test PASSED
tests/test_phase_3_2_adversarial_minimality.py::test_case_09_candidate_and_deletion_identical_rates PASSED
tests/test_phase_3_2_adversarial_minimality.py::test_case_10_single_element_candidate PASSED
tests/test_phase_3_2_adversarial_minimality.py::test_case_11_status_and_message_consistency PASSED
tests/test_phase_3_2_adversarial_minimality.py::test_case_12_boundary_p_values_around_alpha PASSED

============================== 12 passed in 0.05s ==============================
```

### 4.2 Key Verifications from Adversarial Tests
1. **Case 2 Verification:** When candidate has 100/100 failures and deletion has 55/100 failures, Fisher exact test yields $p = 4.3 \times 10^{-14} \lll 0.05$. However, because $0.55 \ge 0.50$, the empirical deletion guard immediately triggers and classifies the candidate as `NOT_MINIMAL`.
2. **Case 3 Verification:** When candidate has 5/5 and deletion has 1/5, Fisher exact test yields $p = 0.0238 < 0.05$ (passes), and deletion rate $0.20 < 0.50$ (passes). Binomial threshold test yields $p = 0.1875 > 0.05$ (fails). The classification correctly returns `STATISTICALLY_SUPPORTED` while transparently reporting `threshold binomial all p < 0.05: False` in telemetry.
3. **Case 5 Verification:** When candidate has 1/2 failures ($\hat{p} = 0.50$, but Wilson 95% CI is $[0.0945, 0.9055]$, where $L_{95} = 0.0945 \le 0.20$), the corrected guard immediately returns `INCONCLUSIVE`, preventing an underpowered candidate from being classified as minimal.
4. **Case 12 Verification:** At exact alpha equality ($p = 0.0833, \alpha = 0.0833$), the test fails strict inequality and returns `OBSERVED_ONLY`. At $p = 0.0833, \alpha = 0.0834$, the test passes and returns `STATISTICALLY_SUPPORTED`.

---

## 5. Audit Conclusion

The implementation conformance audit establishes:
1. The historical classification logic for `STATISTICALLY_SUPPORTED` strictly enforced the empirical non-failure guard and the comparative Fisher IUT condition.
2. The omission of the candidate Wilson lower bound guard ($L_{95} > 0.20$) in the historical code represented a **MEDIUM** severity implementation gap.
3. Because the gap has been remediated in source code and proven to have had zero impact on the actual historical experimental observations, the implementation is certified as **CONFORMANT WITH DOCUMENTED LIMITATIONS**.
