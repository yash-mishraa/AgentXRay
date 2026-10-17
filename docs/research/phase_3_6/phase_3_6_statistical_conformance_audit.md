# AgentXRay — Phase 3.6: Final Statistical Conformance Audit

**Document Reference:** `docs/research/phase_3_6/phase_3_6_statistical_conformance_audit.md`  
**Date:** 2026-10-17  
**Version:** Phase 3.6 Statistical Conformance Audit  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Machine-Readable Companion:** [`docs/research/phase_3_6/phase_3_6_power_results.json`](phase_3_6_power_results.json)  
**Errata Register:** [`docs/research/phase_3_6/phase_3_6_statistical_errata.md`](phase_3_6_statistical_errata.md)  
**Regression Test Suite:** [`tests/test_phase_3_6_statistical_conformance.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/tests/test_phase_3_6_statistical_conformance.py)  

---

## 1. Executive Summary & Audit Mandate

Phase 3.6 resolves the remaining statistical-conformance and definition questions identified at the conclusion of Phase 3.5. Through rigorous source-code inspection, deterministic offline derivations, and newly added regression tests, this audit establishes the exact algorithmic properties of the Wilcoxon signed-rank permutation tests used in Project AgentXRay.

### Final Audit Disposition:
$$\mathbf{STATISTICAL\_CONFORMANCE\_VERIFIED\_WITH\_LIMITATIONS}$$

### Key Findings of the Audit:
1. **Zero Differences are Discarded Before Ranking (Wilcoxon 1945, Not Pratt 1959):**  
   Both [`power_and_budget_analysis.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/power_and_budget_analysis.py) and [`exploratory_runner_v1_4.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/exploratory_runner_v1_4.py) filter out zero differences prior to sorting and ranking. This procedure is Wilcoxon's (1945) standard reduced-sample method. Prior documentation labeling this procedure as "Pratt filtering" was a terminological misnomer, as Pratt's (1959) method explicitly includes zeros in the ranking before discarding them.
2. **Tie-Ranking Implementation Discrepancy Had Zero Historical Impact:**  
   `power_and_budget_analysis.py` implements average mid-ranks for tied absolute non-zero differences (`avg_r = (i+1+j)/2.0`). Conversely, `exploratory_runner_v1_4.py` assigned ordinal integer ranks (`1.0, 2.0, 3.0`) according to sort order. Focused regression tests prove that on the realized Phase 2B.1.5 benchmark data ($[0.0417, 0.2222, 0.2222]$), both methods yield identical test statistics ($W=1$) and exact p-values ($p = 4/8 = 0.50$).
3. **Power Simulation Evaluates Directional Power Under a Two-Sided Test:**  
   The simulation evaluates two-sided p-values filtered by direction (`if p < 0.05 and w_plus > w_minus:`). The analytical probability of directional rejection favoring Delta Debugging is **$37.6173\%$**, while conventional two-sided power (including rejection where Greedy beats DD) is **$37.6174\%$**. The difference of $1.022 \times 10^{-6}$ is undetectable in 2,000 Monte Carlo replications (both yield $743/2000 = 37.15\%$). Conceptually, the metric is directional power, which perfectly matches Protocol v1.4 line 85.
4. **Primary Inference is Valid Only at the Mechanism Level ($n=6$):**  
   The $N=12$ power figure ($94.85\%$) represents an unclustered sensitivity model that commits pseudoreplication when applied to the actual study. Primary inference is valid only at the mechanism level ($n=6$, estimated directional power $37.15\%$).

---

## 2. Source-Level Implementation Audit

### 2.1 Trace of Zero-Difference Handling
In [`benchmarks/model_pilot/power_and_budget_analysis.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/power_and_budget_analysis.py#L154-L162):
```python
# Line 154: Pratt filtering of zero differences  <-- TERMINOLOGICAL MISNOMER
non_zeros = [d for d in diffs if d != 0.0]
m = len(non_zeros)
if m < 4:
    continue

abs_pairs = [(abs(d), d > 0.0) for d in non_zeros]
abs_pairs.sort(key=lambda x: x[0])
```

In [`benchmarks/model_pilot/exploratory_runner_v1_4.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/exploratory_runner_v1_4.py#L723-L725):
```python
if abs(mean_d) > 1e-6:
    mech_diff_values.append(abs(mean_d))
    mech_signs.append(mean_d > 0)
```

**Algorithmic Determination:**
- Both files discard differences of zero **before** absolute values are sorted and ranked.
- Ranking is performed exclusively on the reduced sample of size $m \le n$.
- This corresponds precisely to the **Wilcoxon (1945) zero-discard / reduced-sample method**.
- It does **not** implement Pratt's (1959) method, which ranks all $n$ observations (assigning ranks $1, \dots, d_0$ to zeros) and only drops their ranks after ranking.

**Mathematical Impact:**
Under Wilcoxon's zero-discard method, if any mechanism produces a difference of zero, $m \le 5$. Because $\min(p) = \frac{2}{2^m} \ge \frac{2}{32} = 0.06250 > 0.05$, a single zero difference makes rejecting $H_0$ mathematically impossible at $\alpha=0.05$.

### 2.2 Trace of Tied Absolute Differences
In [`benchmarks/model_pilot/power_and_budget_analysis.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/power_and_budget_analysis.py#L163-L176):
```python
ranks = [0.0] * m
tie_counts: list[int] = []
i = 0
while i < m:
    j = i
    while j < m and abs(abs_pairs[j][0] - abs_pairs[i][0]) < 1e-6:
        j += 1
    tie_len = j - i
    if tie_len > 1:
        tie_counts.append(tie_len)
    avg_r = (i + 1 + j) / 2.0
    for idx in range(i, j):
        ranks[idx] = avg_r
    i = j
```
Tied absolute values receive average mid-ranks (`avg_r`).

In [`benchmarks/model_pilot/exploratory_runner_v1_4.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/exploratory_runner_v1_4.py#L729-L734):
```python
indexed = sorted(enumerate(mech_diff_values), key=lambda x: x[1])
ranks = [0.0] * len(mech_diff_values)
for r_idx, (orig_idx, _) in enumerate(indexed, start=1):
    ranks[orig_idx] = float(r_idx)
```
Tied absolute values received ordinal integer ranks (`1.0, 2.0, 3.0`) according to stable sort order.

**Reconciliation on Historical Data:**
In the Phase 2B.1.5 benchmark, the realized differences on the $m=3$ non-zero mechanisms were:
- `mech_01`: $+0.2222$ (sign True)
- `mech_02`: $+0.2222$ (sign True)
- `mech_05`: $-0.0417$ (sign False)

Under runner ordinal ranks: `ranks = [2.0, 3.0, 1.0]`, `signs = [True, True, False]`.
$W^+ = 5.0, W^- = 1.0 \implies W = 1.0 \implies p = \frac{4}{8} = 0.50$.

Under proper mid-ranks: `ranks = [2.5, 2.5, 1.0]`, `signs = [True, True, False]`.
$W^+ = 5.0, W^- = 1.0 \implies W = 1.0 \implies p = \frac{4}{8} = 0.50$.

Both implementations yield identical test statistics and p-values. Regression test `test_historical_benchmark_realized_differences` verifies this invariance.

---

## 3. Directional vs. Conventional Two-Sided Power

### 3.1 Decision Rule in Simulation vs. Protocol
In `simulate_wilcoxon_power` line 188:
```python
if p < 0.05 and w_plus > w_minus:
    rejections += 1
```
The test computes a two-sided permutation p-value ($p$), but counts a rejection only when $W^+ > W^-$, which indicates that differences favor Delta Debugging.

In [Protocol v1.4 §3.2.2 line 85](file:///Users/yashmishra14/Personal%20projects/AgentXRay/docs/research/phase_2b1_exploratory_protocol_v1_4.md#L85):
> *"Minimum achievable two-sided p-value is $2/64 = 0.03125$ ($W=0$), which requires DD to beat greedy on all 6 mechanisms without ties."*

The research protocol's substantive success event is explicitly directional: DD beating Greedy across all mechanisms.

### 3.2 Exact Analytical Derivations
Under the simulation model:
- $n = 6$ mechanisms
- $P(D_i = 0) = p_{\text{tie}} = 0.05$
- Conditional distribution: $D_i \sim \mathcal{N}(\mu=15.0, \sigma=12.0)$ ($d = 1.25$)

Let $\Phi(x)$ denote the standard normal cumulative distribution function:
$$P(D_i > 0 \mid D_i \ne 0) = \Phi(1.25) \approx 0.89435023$$
$$P(D_i < 0 \mid D_i \ne 0) = \Phi(-1.25) \approx 0.10564977$$
$$P(\text{no ties}) = (1 - 0.05)^6 = (0.95)^6 \approx 0.73509189$$

#### 1. Directional Success Power (DD beats Greedy):
Rejection occurs when $m=6$ and all 6 mechanisms are positive ($W^+ = 21, W^- = 0 \implies W = 0$):
$$\text{Power}_{\text{directional}} = P(\text{no ties}) \times [P(D_i > 0)]^6 = (0.95)^6 \times [\Phi(1.25)]^6 \approx \mathbf{0.37617277 \ (37.6173\%)}$$

#### 2. Opposite Direction Rejection (Greedy beats DD):
Rejection occurs when $m=6$ and all 6 mechanisms are negative ($W^+ = 0, W^- = 21 \implies W = 0$):
$$\text{Power}_{\text{opposite}} = P(\text{no ties}) \times [P(D_i < 0)]^6 = (0.95)^6 \times [\Phi(-1.25)]^6 \approx \mathbf{1.022240 \times 10^{-6} \ (0.000102\%)}$$

#### 3. Conventional Two-Sided Power:
Rejection in either tail at two-sided $\alpha=0.05$:
$$\text{Power}_{\text{two-sided}} = \text{Power}_{\text{directional}} + \text{Power}_{\text{opposite}} \approx \mathbf{0.37617380 \ (37.6174\%)}$$

### 3.3 Synthesis & Resolution
- **Absolute Difference:** $\Delta = 1.022 \times 10^{-6}$ ($0.0001\%$).
- **Monte Carlo Realization (2,000 draws, seed 525):**
  Both directional and conventional two-sided criteria yielded exactly **743 rejections** ($743/2000 = 0.3715$). In 2,000 draws, observing a 1-in-a-million opposite-direction event has probability $1 - (1 - 10^{-6})^{2000} \approx 0.0020$ ($0.2\%$).
- **Conclusion:** While numerically indistinguishable in finite Monte Carlo samples, the metric evaluated by the code is **directional power under a two-sided test**. It must be cited as such to maintain rigorous statistical precision.

---

## 4. Recheck of Sensitivity Calculations

All configurations from Protocol v1.4 §9.2 were reproduced deterministically from source code. Every configuration is classified by its inferential validity:

```
+---------+-----------+----------------------------------------------+----------------------------------------------+
| Cohen d | Mean Diff | Mechanism-Level (n=6) [PRIMARY DESIGN]       | Instance-Level (N=12) [SENSITIVITY MODEL]    |
|         |           | Power [95% Wald CI] (Seed / Rejections)      | Power [95% Wald CI] (Seed / Rejections)      |
+---------+-----------+----------------------------------------------+----------------------------------------------+
|  0.40   |  4.8 req  |   5.30% [ 4.32%,  6.28%] (seed 440, 106/2000)|  20.05% [18.29%, 21.81%] (seed 140, 401/2000)|
|  0.50   |  6.0 req  |   7.10% [ 5.98%,  8.22%] (seed 450, 142/2000)|  30.40% [28.38%, 32.42%] (seed 150, 608/2000)|
|  0.60   |  7.2 req  |  10.45% [ 9.11%, 11.79%] (seed 460, 209/2000)|  42.95% [40.77%, 45.13%] (seed 160, 859/2000)|
|  0.70   |  8.4 req  |  15.15% [13.58%, 16.72%] (seed 470, 303/2000)|  53.40% [51.21%, 55.59%] (seed 170, 1068/2000)|
|  0.80   |  9.6 req  |  16.45% [14.83%, 18.07%] (seed 480, 329/2000)|  64.00% [61.90%, 66.10%] (seed 180, 1280/2000)|
|  1.00   | 12.0 req  |  24.85% [22.96%, 26.74%] (seed 500, 497/2000)|  83.00% [81.35%, 84.65%] (seed 200, 1660/2000)|
|  1.25   | 15.0 req  |  37.15% [35.03%, 39.27%] (seed 525, 743/2000)|  94.85% [93.88%, 95.82%] (seed 225, 1897/2000)|
+---------+-----------+----------------------------------------------+----------------------------------------------+
```

### Explanatory Synthesis:
1. **$n=6$ Mechanism-Level Power:** This column represents the primary pre-registered inferential test. Prospective directional power is $37.15\%$ for $d=1.25$ because the small permutation space ($2^6 = 64$) requires complete sign unanimity ($W=0$) and zero ties.
2. **$N=12$ Instance-Level Power:** This column represents an exploratory sensitivity model assuming 12 independent mechanisms. In the real benchmark, tasks are clustered in pairs under 6 mechanisms. This calculation must not be presented as evidence that the primary study was adequately powered.

---

## 5. Regression Test Verification

All statistical conformance claims and algorithmic properties are covered by automated tests in [`tests/test_phase_3_6_statistical_conformance.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/tests/test_phase_3_6_statistical_conformance.py):

```
tests/test_phase_3_6_statistical_conformance.py::test_zero_discard_method_is_wilcoxon_not_pratt PASSED
tests/test_phase_3_6_statistical_conformance.py::test_discrete_rejection_region_across_all_m PASSED
tests/test_phase_3_6_statistical_conformance.py::test_tied_absolute_ranks_preserve_rejection_region PASSED
tests/test_phase_3_6_statistical_conformance.py::test_historical_benchmark_realized_differences PASSED
tests/test_phase_3_6_statistical_conformance.py::test_directional_vs_two_sided_theoretical_derivation PASSED
tests/test_phase_3_6_statistical_conformance.py::test_deterministic_sensitivity_table_reproduction PASSED
tests/test_phase_3_6_statistical_conformance.py::test_experimental_unit_clustering_and_pseudoreplication PASSED
```

Total test suite status: **254 passed in 0.44s**, zero failures.

---

## 6. Official Release Assessment & Disposition

### Disposition:
$$\mathbf{STATISTICAL\_CONFORMANCE\_VERIFIED\_WITH\_LIMITATIONS}$$

### Justification:
- All source-code implementations have been audited, mathematically verified, and reconciled with protocol specifications.
- Terminological misnomers ("Pratt filtering") and conceptual ambiguities ("two-sided power" vs "directional power") have been formally corrected in Phase 3.6 errata.
- All historical raw data and frozen protocol specifications remain bit-for-bit immutable.
- The repository package meets research-grade standards for an exploratory feasibility benchmark with fully disclosed limitations.
