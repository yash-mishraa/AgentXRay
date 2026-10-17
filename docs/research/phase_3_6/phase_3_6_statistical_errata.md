# AgentXRay — Phase 3.6: Statistical Conformance Errata and Traceable Amendments

**Document Reference:** `docs/research/phase_3_6/phase_3_6_statistical_errata.md`  
**Date:** 2026-10-17  
**Version:** Phase 3.6 Statistical Conformance Errata Register  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Authoritative Conformance Audit:** [`docs/research/phase_3_6/phase_3_6_statistical_conformance_audit.md`](phase_3_6_statistical_conformance_audit.md)  
**Machine-Readable Companion:** [`docs/research/phase_3_6/phase_3_6_power_results.json`](phase_3_6_power_results.json)  

---

## 1. Executive Summary

This document records the traceable amendments and corrections to statistical terminology, implementation nuances, and test definitions established during the Phase 3.6 Statistical Conformance Audit. It amends and refines the findings of Phase 3.5 without silently rewriting historical records.

In accordance with AgentXRay's mandatory change-control policy:
- **Historical Frozen Artifacts:** Protocol v1.4 (`db45f403...`), Configuration v1.4 (`b3e3b0b8...`), and raw benchmark results (`manifest.json`, `results.json`, `trials.jsonl`, `summary.csv`) remain **strictly immutable**.
- **Historical Audit Preservation:** The Phase 3.5 audit documents are preserved as dated historical artifacts; this document serves as their authoritative amendment.

---

## 2. Master Conformance Errata Register

| Erratum ID | Target Artifacts | Classification | Former Formulation / State | Corrected Replacement Formulation | Technical & Algorithmic Rationale |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **`ERR-3.6-01`** | `power_and_budget_analysis.py` (L154)<br>Phase 3.5 Audit §3.1, §3.3<br>Phase 3.4 Audit §2.6, §4.5 | **Terminological & Algorithmic** | Labeling zero-difference removal as *"Pratt filtering"* or *"Pratt zero-drop"* | *"Wilcoxon (1945) zero-discard / reduced-sample method"* | Pratt's (1959) method includes zero differences in the initial ranking before discarding their ranks from the test statistic. The codebase discards zeros prior to ranking, reducing sample size to $m$. This is the standard Wilcoxon reduced-sample procedure, not Pratt's method. |
| **`ERR-3.6-02`** | `exploratory_runner_v1_4.py` (L729-734)<br>Phase 3.5 Audit §2.3, §6.2 | **Implementation Nuance** | Assuming runner implemented mid-rank tie averaging | Runner assigned ordinal integer ranks (`1.0, 2.0, 3.0`) via Python stable sort; `power_and_budget_analysis.py` implemented mid-ranks | While `power_and_budget_analysis.py` correctly averaged ties (`avg_r = (i+1+j)/2.0`), `exploratory_runner_v1_4.py` used integer ranks. On historical Phase 2B.1.5 data ($[0.0417, 0.2222, 0.2222]$), both procedures yield identical $W=1$ and $p=0.50$. |
| **`ERR-3.6-03`** | Phase 3.5 Audit §4.2, §4.3<br>Phase 3.5 Power Results JSON | **Mathematical & Conceptual** | Describing $(1-p_{\text{tie}})^6 [\Phi(1.25)]^6 = 37.62\%$ and $37.15\%$ as *"two-sided power"* | *"Directional power under a two-sided $\alpha=0.05$ exact permutation test"* | A conventional two-sided rejection event includes rejections where Greedy beats DD ($p < 0.05, W^+ < W^-$). The simulation tested directional success (`p < 0.05 and w_plus > w_minus`). The difference is $1.022 \times 10^{-6}$ ($37.6173\%$ vs $37.6174\%$), which is numerically negligible in 2,000 draws but conceptually distinct. |
| **`ERR-3.6-04`** | Protocol v1.4 §9.2<br>Phase 2B.1.2–3.5 Audits | **Inferential Scope** | Juxtaposing $N=12$ power ($94.8\%$) without prominent pseudoreplication label | Explicitly labeling $N=12$ power ($94.85\%$) as a **hypothetical sensitivity analysis** | Configurations of the same mechanism share prompts and failure triggers. Primary inference is valid strictly at the mechanism level ($n=6$); the $N=12$ calculation must never be presented as primary-design power. |

---

## 3. Detailed Traceable Amendments

### 3.1 ERR-3.6-01: Discarding Zeros is Wilcoxon (1945), Not Pratt (1959)
- **Source Code Locations:**
  - `benchmarks/model_pilot/power_and_budget_analysis.py` line 154: `# Pratt filtering of zero differences`
  - `benchmarks/model_pilot/exploratory_runner_v1_4.py` line 723: `if abs(mean_d) > 1e-6:`
- **Detailed Finding:**
  The code filters out zero differences *before* absolute differences are sorted and ranked:
  ```python
  non_zeros = [d for d in diffs if d != 0.0]
  m = len(non_zeros)
  abs_pairs = [(abs(d), d > 0.0) for d in non_zeros]
  abs_pairs.sort(key=lambda x: x[0])
  ```
  The remaining $m$ absolute differences are then ranked from $1$ to $m$.
  - In Wilcoxon's (1945) original method, zero differences are completely discarded, and ranking proceeds on the reduced sample of size $m$.
  - In Pratt's (1959) method, zero differences are included in the ranking of all $n$ observations (receiving the lowest ranks, e.g., $1, \dots, d_0$), and only their ranks are dropped when summing $W^+$ and $W^-$.
  Therefore, the implementation strictly executes the **Wilcoxon zero-discard method**.
- **Amendment:** All research narratives must describe the procedure as Wilcoxon's reduced-sample zero-discard method. The mathematical consequence remains identical: any zero difference reduces the sample size to $m \le 5$, where the minimum achievable two-sided p-value is $\ge \frac{2}{32} = 0.06250 > 0.05$, making rejection of $H_0$ mathematically impossible at $\alpha=0.05$.

---

### 3.2 ERR-3.6-02: Runner Ordinal Ranking vs Power Analysis Mid-Ranks
- **Source Code Locations:**
  - `benchmarks/model_pilot/exploratory_runner_v1_4.py` lines 729–734:
    ```python
    indexed = sorted(enumerate(mech_diff_values), key=lambda x: x[1])
    ranks = [0.0] * len(mech_diff_values)
    for r_idx, (orig_idx, _) in enumerate(indexed, start=1):
        ranks[orig_idx] = float(r_idx)
    ```
  - `benchmarks/model_pilot/power_and_budget_analysis.py` lines 163–176:
    ```python
    avg_r = (i + 1 + j) / 2.0
    for idx in range(i, j):
        ranks[idx] = avg_r
    ```
- **Detailed Finding:**
  `power_and_budget_analysis.py` implemented proper mid-rank tie averaging. However, `exploratory_runner_v1_4.py` assigned ordinal integer ranks (`1.0, 2.0, 3.0`) according to sort order without averaging ties.
  - On the historical benchmark data, the 3 non-zero mechanism differences were:
    `mech_01`: $+0.2222$ (DD)
    `mech_02`: $+0.2222$ (DD)
    `mech_05`: $-0.0417$ (Greedy)
  - Under runner ordinal ranks: ranks were $[2.0, 3.0, 1.0]$, giving $W^+ = 5.0, W^- = 1.0 \implies W = 1.0, p = \frac{4}{8} = 0.50$.
  - Under proper mid-ranks: ranks are $[2.5, 2.5, 1.0]$, giving $W^+ = 5.0, W^- = 1.0 \implies W = 1.0, p = \frac{4}{8} = 0.50$.
- **Amendment:** Verified that the implementation difference had **zero effect on historical benchmark results**. Regression tests added in `tests/test_phase_3_6_statistical_conformance.py` verify that both methods yield $W=1, p=0.50$.

---

### 3.3 ERR-3.6-03: Directional vs Conventional Two-Sided Power
- **Source Code Location:**
  - `benchmarks/model_pilot/power_and_budget_analysis.py` line 188:
    ```python
    if p < 0.05 and w_plus > w_minus:
        rejections += 1
    ```
- **Detailed Finding:**
  The simulation's rejection condition requires both $p < 0.05$ (two-sided permutation p-value) AND $W^+ > W^-$ (directional superiority favoring Delta Debugging).
  - Analytical directional power under $\mathcal{N}(15, 12)$ and $p_{\text{tie}}=0.05$:
    $$\text{Power}_{\text{directional}} = (1 - 0.05)^6 \times [\Phi(1.25)]^6 \approx 37.617277\%$$
  - Analytical conventional two-sided power (including opposite rejection):
    $$\text{Power}_{\text{two-sided}} = \text{Power}_{\text{directional}} + (1 - 0.05)^6 \times [\Phi(-1.25)]^6 \approx 37.617380\%$$
  - Difference: $\Delta = 1.022 \times 10^{-6}$ ($0.000102\%$).
  In 2,000 Monte Carlo replications with seed 525, both criteria yielded exactly $743$ rejections ($37.15\%$).
- **Amendment:** The power simulation is formally designated as **directional power under a two-sided test**. This directly conforms to Protocol v1.4 line 85, which defines success as DD beating Greedy across all mechanisms.

---

### 3.4 ERR-3.6-04: Strict Labeling of $N=12$ as a Sensitivity Model
- **Detailed Finding:**
  Earlier reports cited $94.8\%$ power at $N=12$ without prominently explaining that $N=12$ commits pseudoreplication by treating $K=4$ and $K=8$ configurations of the same mechanism as independent observations.
- **Amendment:** Citing $94.8\%$ as the power of the study is formally classified as an inferential error. The $N=12$ figure is retained strictly as a hypothetical sensitivity benchmark demonstrating the power that would have been achieved had the study evaluated 12 truly independent mechanisms.
