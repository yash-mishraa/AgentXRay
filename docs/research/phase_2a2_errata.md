# AgentXRay — Phase 2A.2 Errata & Methodological Corrections

**Author:** Antigravity (Pair Programming AI Assistant)  
**Date:** 2026-10-17  
**Baseline Commit:** `2e92142` (origin/main)  
**Target Document:** `docs/research/phase_2a2_reproducibility_audit.md` and related Phase 2A artifacts  
**Status:** Approved & Formally Recorded Errata  

---

## 1. Overview and Purpose

An independent methodological and code-level audit of the Phase 2A.2 report and its underlying artifacts identified several reporting inaccuracies, imprecise statistical terms, and unverified assumptions. This errata document formally catalogs each confirmed defect, explains its forensic root cause, provides the mathematically rigorous correction, and details the implications for ongoing research.

No historical Phase 2A or Phase 2A.1 raw observations were fabricated, and no raw benchmark data files were regenerated. All corrections resolve errors in calculation, terminology, and artifact validation scripts.

---

## 2. Catalog of Confirmed Defects & Corrections

### Defect 1: Individual Algorithm Target Call Standard Deviations
- **Reported in Phase 2A.1 Narrative:**
  - Delta Debugging: $104.1 \pm 111.4$ calls
  - Greedy Ablation: $125.5 \pm 45.4$ calls
  - Random Search: $557.5 \pm 391.8$ calls
- **Forensic Root Cause:**
  In `benchmarks/synthetic/analyze_results.py`, line 108 formatted the output string as `f"Target Calls (mean ± std): {mean_calls:.1f} (median: {med_calls:.1f})"`, omitting the computation and display of standard deviation. When compiling the narrative report, standard deviation numbers were manually entered from secondary scratch records, accidentally borrowing paired difference standard deviations ($111.4 \approx 113.55$, and $391.8 \approx 392.78$).
- **Mathematically Corrected Values ($N=400$):**
  - **Stochastic Delta Debugging:** Sample std ($s$, $\text{ddof}=1$) = **$121.39$**, Population std ($\sigma$, $\text{ddof}=0$) = **$121.24$**, Median = **$70.0$**, Mean = **$104.125$**.
  - **Stochastic Greedy Ablation:** Sample std ($s$, $\text{ddof}=1$) = **$73.30$**, Population std ($\sigma$, $\text{ddof}=0$) = **$73.21$**, Median = **$140.0$**, Mean = **$125.500$**.
  - **Stochastic Seeded Random Search:** Sample std ($s$, $\text{ddof}=1$) = **$438.37$**, Population std ($\sigma$, $\text{ddof}=0$) = **$437.82$**, Median = **$620.0$**, Mean = **$557.450$**.
- **Code Remediation:** `benchmarks/synthetic/analyze_results.py` and `benchmarks/synthetic/reproduce_phase_2a1.py` have been updated to compute $s = \sqrt{\frac{1}{N-1}\sum (x_i - \bar{x})^2}$ directly.

---

### Defect 2: Wilcoxon Signed-Rank Test Terminology & Asymptotic Classification
- **Reported in Phase 2A.1 / 2A.2 Narrative:**
  Described as "Wilcoxon signed-rank test exact p-value = 0.00017" or "exact Wilcoxon p-value = $1.78 \times 10^{-14}$".
- **Forensic Root Cause:**
  1. In the Phase 2A.1 summary output, `Wilcoxon ... p=0.00000, t=-3.76, p=0.00017` appeared on a single line. The author conflated the paired t-test p-value ($0.00017$) with the Wilcoxon test.
  2. The function `wilcoxon_signed_rank_test` computes $z = \frac{W^+ - E[W] \pm 0.5}{\sqrt{\text{Var}(W)}}$ and evaluates $p = \text{erfc}(|z|/\sqrt{2})$. This is an **asymptotic normal approximation with Pratt zero-drop, tie-corrected variance, and Yates continuity correction**. It is **not** an exact combinatorial distribution result.
- **Mathematically Corrected Values ($N=400$, $n_r=340$ non-zero pairs):**
  - Positive rank sum: $W^+ = 15,089.5$
  - Negative rank sum: $W^- = 42,880.5$
  - Continuity-corrected $z$-statistic: $z = -7.6655$
  - **Two-Sided Asymptotic Normal p-value:** $\mathbf{p = 1.7814 \times 10^{-14}}$
- **Code & Terminology Remediation:**
  - `PairedDifferenceSummary.wilcoxon_p_value` is explicitly documented as an asymptotic normal approximation.
  - A new field `wilcoxon_p_value_is_asymptotic = True` is declared on `PairedDifferenceSummary`.
  - Added `wilcoxon_exact_p_value_small_sample(w_plus, n_r)` to `src/agentxray/statistics.py` for exact combinatorial dynamic programming when $n_r \le 25$ without ties. For $N=400$ ($n_r=340$), exact enumeration over $2^{340} \approx 2 \times 10^{102}$ states is computationally intractable; the continuity-corrected asymptotic normal approximation is the standard and appropriate procedure.

---

### Defect 3: Paired t-Test Exact Student's t vs Gaussian Normal Approximation
- **Reported in Phase 2A.1 Narrative:** Paired t-test $p = 0.00017$.
- **Forensic Root Cause:**
  The original paired t-test used `math.erfc(abs(t_stat) / math.sqrt(2.0))`, which is the asymptotic Gaussian normal approximation ($p = 0.0001666 \approx 0.00017$).
- **Mathematically Corrected Values ($N=400$, $df=399$, $t = -3.7649$):**
  - Exact Student's t distribution survival function:
    $$p = I_{\frac{\nu}{\nu + t^2}}\left(\frac{\nu}{2}, \frac{1}{2}\right) = I_{0.965694}(199.5, 0.5) = \mathbf{0.00019166 \approx 0.000192}$$
  - Normal approximation: $\mathbf{0.000167}$.
  - Relative difference: $\sim 15.0\%$.
- **Code Remediation:** `regularized_incomplete_beta` and `student_t_distribution_p_value` were implemented using Lentz's continued fraction method in `src/agentxray/statistics.py`.

---

### Defect 4: Multiple-Testing Correction Terminology (FDR vs FWER)
- **Reported in Phase 2A.2 Narrative:**
  Section 6.2 described Benjamini–Hochberg as "guaranteeing family-wise False Discovery Rate $\le 0.05$."
- **Correction & Proper Terminology:**
  The Benjamini–Hochberg (1995) step-up procedure controls the **False Discovery Rate (FDR)**:
  $$\text{FDR} = E\left[\frac{V}{\max(R, 1)}\right] \le \frac{m_0}{m} \alpha \le \alpha$$
  under independence or positive regression dependency (PRDS) across the $m$ hypotheses.
  It does **NOT** control the Family-Wise Error Rate ($\text{FWER} = P(V \ge 1)$). FWER control is a more conservative criterion controlled by Bonferroni or Holm-Bonferroni.
- **Audit Decision Rule ($m=8$ target families, $\alpha = 0.05$):**
  1. Rank raw two-sided Wilcoxon asymptotic p-values $p_{(1)} \le \dots \le p_{(m)}$.
  2. Find $k = \max \{i : p_{(i)} \le \frac{i}{m} \alpha\}$.
  3. Reject $H_{(1)}, \dots, H_{(k)}$; declare corresponding families significant at $\text{FDR} \le 0.05$.
  - All 7 non-null families satisfy $q < 10^{-6}$ and are rejected.
  - The null family (`null_target`) yields $p = 1.0, q = 1.0$ (not rejected).

---

### Defect 5: Claims Regarding Common Random Numbers (CRN) & Hash Independence
- **Reported in Phase 2A.2 Narrative:**
  Section 4 claimed that "SHA-256's cryptographic avalanche property guarantees that draws for subset $S_1$ are statistically uncorrelated with draws for subset $S_2$."
- **Correction & Precise Scientific Language:**
  A cryptographic hash function is a deterministic mathematical mapping. While it exhibits strong diffusion and pseudorandom uniformity, hashing alone **does not constitute a proof of statistical independence in the measure-theoretic sense**.
  - **Actual Guarantee:** Hash-keyed draws provide deterministic, reproducible Common Random Numbers (CRN) for identical `(target_seed, canonical_subset, replicate_idx)` tuples across different algorithms. This substantially reduces Monte Carlo variance when comparing algorithms on the exact same candidate subsets.
  - **Limitation:** The validity of matched-pair inferential tests relies on the experimental design and model-level exchangeability, not on an unproven claim that hash outputs are true independent random variables.

---

### Defect 6: Null Hypothesis Interpretation ("Accept H0" vs "Fail to Reject H0")
- **Reported in Phase 2A.2 Narrative:**
  Phrases such as "accept H0 for that agent family" appeared in stopping criteria.
- **Correction:**
  In classical frequentist inference, failure to reject a null hypothesis does not prove the null hypothesis true. All occurrences have been replaced with **"fail to reject $H_0$"**.
  - Preserved Negative Finding: Stochastic Delta Debugging is significantly less efficient than Greedy Ablation on conjunctive targets ($+226.4$ calls, $W^+ = 1275.0, p = 7.57 \times 10^{-10}$). This negative finding is an inherent property of binary set partitioning on synergistic multi-factor triggers and is preserved without dilution.

---

### Defect 7: Truthful, Dynamic Validation in `reproduce_phase_2a1.py`
- **Reported in Phase 2A.2 Implementation:**
  The validation status in `reproduce_phase_2a1.py` was hard-coded as `"status": "VERIFIED_100_PERCENT_REPRODUCIBLE"` rather than derived dynamically from check results. The script also did not reconcile against `phase_2a1_scaling_summary.csv` or exit with a nonzero code on failure.
- **Code Remediation:**
  - `reproduce_phase_2a1.py` now executes 21 granular checks.
  - Candidate elements are verified to be subsets of the instance initial space.
  - Cumulative calls are verified against arithmetic steps.
  - Reconciliation is performed against both `phase_2a1_scaling_results.json` and `phase_2a1_scaling_summary.csv`.
  - `status = "VALIDATION_PASSED" if all_checks_passed else "VALIDATION_FAILED"`.
  - On any failure or missing artifact, the script prints diagnostics and terminates with process exit code `1`.
