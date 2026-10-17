# AgentXRay — Phase 3.5: Independent Statistical Power & Experimental-Unit Audit

**Document Reference:** `docs/research/phase_3_5/phase_3_5_statistical_power_audit.md`  
**Date:** 2026-10-17  
**Version:** Phase 3.5 Statistical Power and Experimental-Unit Audit  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Machine-Readable Companion:** [`docs/research/phase_3_5/phase_3_5_power_results.json`](phase_3_5_power_results.json)  
**Errata Register:** [`docs/research/phase_3_5/phase_3_5_statistical_errata.md`](phase_3_5_statistical_errata.md)  

---

## 1. Executive Summary & Audit Mandate

Phase 3.5 is an offline, skeptical mathematical and methodological audit of Project AgentXRay's statistical power claims, experimental-unit structure, and inferential boundaries. Prior project reports repeatedly asserted that *"statistical power at the mechanism level ($n=6$) is bounded by $\le 37.1\%$ even for large effects ($d=1.25$)"*, while simultaneously citing instance-level power of $94.8\%$ at $N=12$.

The purpose of this audit is not to defend past reports, but to establish what is mathematically proven, what is a model-dependent simulation estimate, what is pseudoreplicated, and what requires correction.

### Final Audit Disposition:
$$\mathbf{POWER\_CLAIMS\_REQUIRE\_CORRECTION}$$

### Core Audit Determinations:
1. **The 37.1% figure is a simulation estimate, not a mathematical bound:** A finite Monte Carlo draw of 37.15% (MCSE 1.08%, 95% CI $[35.03\%, 39.27\%]$) generated under specific parametric assumptions ($\mu=15.0, \sigma=12.0, p_{\text{tie}}=0.05, n_{\text{sims}}=2000$) was erroneously described as a mathematical upper bound on statistical power. If ties are absent ($p_{\text{tie}}=0.0$), power under the identical effect size is $51.9\%$; under larger effect sizes ($d=2.50$), power reaches $70.5\%$.
2. **The genuine mathematical constraint is discrete combinatorics:** For an exact two-sided Wilcoxon signed-rank permutation test at $\alpha=0.05$ on $n=6$ pairs with Pratt zero-filtering, the rejection region contains strictly $W=0$. Rejection is possible if and only if $m=6$ (zero ties) and all 6 signs agree. If even a single mechanism produces a tie ($m \le 5$), the minimum possible two-sided p-value is $\ge 2/32 = 0.06250 > 0.05$, rendering statistical rejection **mathematically impossible (power identically $0.0\%$)**.
3. **Instance-level ($N=12$) power ($94.8\%$) is pseudoreplicated:** The 12 active task configurations represent 6 causal mechanisms evaluated at two distractor levels ($K=4$ and $K=8$). Because configurations of the same mechanism share prompts, tools, and failure triggers, treating them as $N=12$ independent pairs commits pseudoreplication. The $94.8\%$ power calculation is valid strictly as a hypothetical sensitivity exercise and is uninterpretable for primary inference.
4. **Observed benchmark outcome ($p=0.50$ on $m=3$) is an empirical sample statistic:** Prospective power must be strictly separated from post-experiment p-values. The failure to reject $H_0$ in Phase 2B.1.5 was driven by sample attrition during calibration ($7$ tasks disqualified, leaving $m=3$ non-zero mechanisms) and lack of observed separation, not proof of algorithmic equivalence.

---

## 2. Experimental-Unit Analysis

### 2.1 Why 16 Task Configurations but 8 Mechanisms?
In [`benchmarks/model_pilot/confirmatory_tasks.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/confirmatory_tasks.py), the benchmark evaluates exactly 16 task configurations (`inst_01` through `inst_16`) structured across 8 underlying mechanisms (`mech_01` through `mech_06`, plus `mech_null_01` and `mech_null_02`):

```
+------------------------------------+--------------------------------+-----+-----------+
| Mechanism Identifier               | Task Configurations            |  K  | Null Case |
+------------------------------------+--------------------------------+-----+-----------+
| mech_01_syntax_currency            | inst_01_syntax_currency_k4     |  4  |   False   |
|                                    | inst_02_syntax_currency_k8     |  8  |   False   |
| mech_02_syntax_comma               | inst_03_syntax_comma_k4        |  4  |   False   |
|                                    | inst_04_syntax_comma_k8        |  8  |   False   |
| mech_03_routing_override           | inst_05_routing_override_k4    |  4  |   False   |
|                                    | inst_06_routing_override_k8    |  8  |   False   |
| mech_04_routing_prohibition        | inst_07_routing_prohibition_k4 |  4  |   False   |
|                                    | inst_08_routing_prohibition_k8 |  8  |   False   |
| mech_05_schema_unit                | inst_09_schema_unit_k4         |  4  |   False   |
|                                    | inst_10_schema_unit_k8         |  8  |   False   |
| mech_06_schema_cross_unit          | inst_11_schema_cross_unit_k4   |  4  |   False   |
|                                    | inst_12_schema_cross_unit_k8   |  8  |   False   |
+------------------------------------+--------------------------------+-----+-----------+
| mech_null_01_calc                  | inst_13_null_calc_k4           |  4  |   True    |
|                                    | inst_14_null_calc_k8           |  8  |   True    |
| mech_null_02_db                    | inst_15_null_db_k4             |  4  |   True    |
|                                    | inst_16_null_db_k8             |  8  |   True    |
+------------------------------------+--------------------------------+-----+-----------+
```

### 2.2 Mechanism-Level Clustering and Pseudoreplication
Within each mechanism $M_i$, the $K=4$ and $K=8$ task configurations share:
- The identical base prompt text and user query.
- The identical required tool and schema definitions.
- The identical planted failure trigger elements.
- The identical expected error classification.

The two configurations differ solely in the number of distractor tools appended to the environment ($4$ vs $8$). 

Consequently:
- `inst_01` and `inst_02` are **not independent random draws** from a population of agent failure modes. They are two distractor-load treatments applied to the **same underlying causal mechanism**.
- Treating the 12 active tasks as $N=12$ independent identically distributed pairs in an inferential test is **pseudoreplication** (Hurlbert, 1984), artificially inflating the nominal sample size by $100\%$ and ignoring intra-mechanism correlation.

### 2.3 Conformance of the Mechanism-Level Burden Statistic
[Protocol v1.4 §3.2.2](file:///Users/yashmishra14/Personal%20projects/AgentXRay/docs/research/phase_2b1_exploratory_protocol_v1_4.md#L83-L88) prespecified that the primary efficiency estimand must be evaluated at the mechanism level by normalizing request counts by the stratum budget cap ($120$ for $K=4$, $180$ for $K=8$) and averaging across the two configurations:

$$\bar{D}_i = \frac{1}{2}\left[\frac{R_{\text{Greedy}}(M_i, 4) - R_{\text{DD}}(M_i, 4)}{120} + \frac{R_{\text{Greedy}}(M_i, 8) - R_{\text{DD}}(M_i, 8)}{180}\right]$$

Inspection of [`benchmarks/model_pilot/exploratory_runner_v1_4.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/exploratory_runner_v1_4.py#L718-L722) confirms that the implementation strictly followed this mathematical specification:
- Disqualified configurations (`inst_05`, `inst_06`, `inst_07`, `inst_08`, `inst_10`, `inst_11`, `inst_12`) received the full cap penalty ($120$ or $180$) for both algorithms, yielding normalized difference $0.0$.
- Realized mechanism differences in `results.json`:
  - `mech_01`: $+0.2222$
  - `mech_02`: $+0.2222$
  - `mech_03`: $0.0000$ (disqualified)
  - `mech_04`: $0.0000$ (disqualified)
  - `mech_05`: $-0.0417$ ($K=4$: $-0.0833$; $K=8$: disqualified, $0.0$)
  - `mech_06`: $0.0000$ (disqualified)
- Non-zero mechanisms evaluated: exactly $m = 3$.

### 2.4 Delineation of Independent Experimental Units by Estimand

| Estimand | Analytical Unit | Effective Sample Size | Independence Status | Primary Inferential Validity |
| :--- | :--- | :---: | :--- | :---: |
| **Primary Resource Burden ($\bar{D}_i$)** | Independent Mechanism ($M_i$) | $n = 6$ ($m = 3$ non-zero) | Valid independent units; within-mechanism $K=4/8$ strata averaged. | **Valid Primary Estimand** |
| **Exploratory Instance-Level Burden** | Task Configuration ($\text{inst}_j$) | $N = 12$ | **Pseudoreplicated:** $K=4$ and $K=8$ share mechanism structures. | **Invalid for Primary Inference** (Descriptive / Sensitivity only) |
| **Ground-Truth Recovery Symmetry** | Task Configuration ($\text{inst}_j$) | $N = 12$ | Clustered in pairs; demoted in Protocol v1.4 to exploratory diagnostic. | **Exploratory Diagnostic Only** |
| **Holdout Minimality Reproduction** | Candidate Evaluation $\times$ Seed | $12\text{ candidates} \times 5\text{ seeds} = 60$ | Fresh pseudo-random prompt seeds per candidate. | **Valid Holdout Replications** |
| **Null False-Discovery Proportion** | Null Configuration & Mechanism | $4\text{ configs} / 2\text{ mechs}$ | Descriptive proportion; inferential population FDR disclaimed. | **Descriptive Proportion Only** |

---

## 3. Mathematical Rejection-Region Analysis

### 3.1 Discrete Permutation Properties at Small Sample Sizes
The Wilcoxon signed-rank test evaluates the sum of positive ranks $W^+$ against the permutation distribution under $H_0: \text{median difference} = 0$. For $m$ non-zero paired differences without ties in absolute values, there are $2^m$ equally likely sign configurations.

The exact two-sided p-value for test statistic $W = \min(W^+, W^-)$ is:
$$p = \frac{2 \times \sum_{w=0}^{W} \text{count}(w)}{2^m}$$

Under Pratt zero-filtering, zero differences ($D_i = 0$) are eliminated from ranking, reducing the effective sample size from $n$ to $m$.

### 3.2 Combinatorial Rejection Table for $m \in \{1, 2, 3, 4, 5, 6\}$

| Non-Zero Pairs ($m$) | Permutation Space ($2^m$) | Min Two-Sided p-value ($W=0$) | Rejection Possible at $\alpha=0.05$? | Rejection Region ($\alpha=0.05$) |
| :---: | :---: | :---: | :---: | :---: |
| **$m = 1$** | $2^1 = 2$ | $\frac{2}{2} = 1.00000$ | **NO** | $\emptyset$ |
| **$m = 2$** | $2^2 = 4$ | $\frac{2}{4} = 0.50000$ | **NO** | $\emptyset$ |
| **$m = 3$** | $2^3 = 8$ | $\frac{2}{8} = 0.25000$ | **NO** | $\emptyset$ |
| **$m = 4$** | $2^4 = 16$ | $\frac{2}{16} = 0.12500$ | **NO** | $\emptyset$ |
| **$m = 5$** | $2^5 = 32$ | $\frac{2}{32} = 0.06250$ | **NO** | $\emptyset$ |
| **$m = 6$** | $2^6 = 64$ | $\frac{2}{64} = 0.03125$ | **YES** | $\{W = 0\}$ |

### 3.3 Two Fundamental Mathematical Theorems for $n=6$
1. **The Rejection Region Theorem:**
   For $n=6$ continuous non-zero pairs at two-sided $\alpha=0.05$, the rejection region consists strictly of $W=0$.
   - $W = 0 \implies p = \frac{2}{64} = 0.03125 < 0.05$ (Reject $H_0$).
   - $W = 1 \implies p = \frac{2 \times (1 + 1)}{64} = \frac{4}{64} = 0.06250 > 0.05$ (Fail to reject).
   - Thus, directional rejection in favor of DD ($W^+ > W^-$) requires that **all 6 mechanisms strictly favor DD**.
2. **The Tie-Elimination Zero-Power Theorem:**
   Under Pratt filtering, if even a single mechanism exhibits a difference of zero ($D_i = 0$), $m \le 5$. Because $\min(p) \ge \frac{2}{32} = 0.06250 > 0.05$, **statistical power to reject $H_0$ is identically $0.0\%$**.

---

## 4. Independent Verification of the Power Calculation

### 4.1 Provenance of the Preregistered Simulation
The power analysis code is located in [`benchmarks/model_pilot/power_and_budget_analysis.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/power_and_budget_analysis.py):
- `simulate_wilcoxon_power(n=6, mu=15.0, sigma=12.0, n_sims=2000, dist="normal", p_tie=0.05, seed=525)`
- Line 248 executes the loop generating the Protocol v1.4 §9.2 table.

### 4.2 Analytical Theoretical Model-Based Power Derivation
Under the simulation's specified data-generating model:
1. Each mechanism difference has independent probability $p_{\text{tie}} = 0.05$ of being exactly zero. The probability that all 6 mechanisms are non-zero is:
   $$P(m = 6) = (1 - 0.05)^6 = (0.95)^6 \approx 0.73509 \quad (73.51\%)$$
2. Conditional on being non-zero, differences follow $\mathcal{N}(\mu=15.0, \sigma=12.0)$ ($d = 1.25$). The probability that an individual difference is positive is:
   $$P(D_i > 0) = \Phi\left(\frac{15.0}{12.0}\right) = \Phi(1.25) \approx 0.89435 \quad (89.44\%)$$
3. Because rejection requires $m=6$ AND all 6 differences positive ($W=0$), the exact theoretical power under this model is:
   $$\text{Theoretical Power} = (1 - p_{\text{tie}})^6 \times [P(D_i > 0)]^6 = (0.73509) \times (0.89435)^6 \approx 0.73509 \times 0.51174 \approx \mathbf{0.37617 \ (37.62\%)}$$

### 4.3 Monte Carlo Output vs Theoretical Expectation
- Theoretical model expectation: **$37.62\%$**
- Monte Carlo realization (2,000 draws, seed 525): **$37.15\%$** ($743$ rejections)
- Monte Carlo Standard Error:
  $$\text{MCSE} = \sqrt{\frac{0.3715 \times (1 - 0.3715)}{2000}} = 0.01080 \quad (1.08\%)$$
- $95\%$ Monte Carlo Confidence Interval:
  $$[0.3715 - 1.96(0.0108), 0.3715 + 1.96(0.0108)] = [\mathbf{0.3503, 0.3927}] \quad (\mathbf{35.03\% \text{ to } 39.27\%})$$
- Difference between simulation and theoretical value: $-0.47\%$, well within 1 standard error ($z = -0.43$).

### 4.4 Sensitivity Analysis: Why 37.1% is NOT an Upper Bound
To determine whether 37.1% constitutes an upper bound, we varied the tie parameter $p_{\text{tie}}$ and the effect size $d$:

```
+----------------+----------------+----------------+-----------------------+-------------------------+
| Cohen's d      | Mean Diff (mu) | Tie Prob (p)   | Monte Carlo Power     | Theoretical Model Power |
+----------------+----------------+----------------+-----------------------+-------------------------+
| d = 1.25       | 15.0 req       | p = 0.05       | 37.15% (seed 525)     | 37.62%                  |
| d = 1.25       | 15.0 req       | p = 0.02       | 45.05%                | 45.36%                  |
| d = 1.25       | 15.0 req       | p = 0.00       | 51.90%                | 51.17%                  |
| d = 1.67       | 20.0 req       | p = 0.05       | 53.10%                | 53.68%                  |
| d = 2.50       | 30.0 req       | p = 0.05       | 70.45%                | 70.52%                  |
| d = 4.17       | 50.0 req       | p = 0.05       | 73.80%                | 73.51% (bounded by ties)|
+----------------+----------------+----------------+-----------------------+-------------------------+
```

**Key Finding:** 
- If ties are absent ($p_{\text{tie}}=0$), power at $d=1.25$ is **$51.9\%$**, substantially higher than $37.1\%$.
- For larger effects ($d=2.50$), power reaches **$70.5\%$**.
- If $d \to \infty$ and $p_{\text{tie}}=0$, power approaches **$100.0\%$**.
- Therefore, $37.1\%$ is an empirical point estimate conditional on the specific assumption that $p_{\text{tie}} = 0.05$ and $d=1.25$. It is **not a mathematical bound**.

---

## 5. Monte Carlo Uncertainty Across the Full Protocol v1.4 Table

The table in [Protocol v1.4 §9.2](file:///Users/yashmishra14/Personal%20projects/AgentXRay/docs/research/phase_2b1_exploratory_protocol_v1_4.md#L198-L209) reported single-point estimates for the $n=6$ column without Monte Carlo standard errors. We independently re-executed the full sensitivity suite and calculated exact MCSE and 95% confidence intervals:

```
+---------+-----------+-----------------------------------+-----------------------------------+
| Cohen d | Mean Diff | Instance-Level (N=12) Power [95%] | Mechanism-Level (n=6) Power [95%] |
+---------+-----------+-----------------------------------+-----------------------------------+
|  0.40   |  4.8 req  |   20.05% [18.29%, 21.81%] (seed 140)|    5.30% [ 4.32%,  6.28%] (seed 440)|
|  0.50   |  6.0 req  |   30.40% [28.38%, 32.42%] (seed 150)|    7.10% [ 5.98%,  8.22%] (seed 450)|
|  0.60   |  7.2 req  |   42.95% [40.77%, 45.13%] (seed 160)|   10.45% [ 9.11%, 11.79%] (seed 460)|
|  0.70   |  8.4 req  |   53.40% [51.21%, 55.59%] (seed 170)|   15.15% [13.58%, 16.72%] (seed 470)|
|  0.80   |  9.6 req  |   64.00% [61.90%, 66.10%] (seed 180)|   16.45% [14.83%, 18.07%] (seed 480)|
|  1.00   | 12.0 req  |   83.00% [81.35%, 84.65%] (seed 200)|   24.85% [22.96%, 26.74%] (seed 500)|
|  1.25   | 15.0 req  |   94.85% [93.88%, 95.82%] (seed 225)|   37.15% [35.03%, 39.27%] (seed 525)|
+---------+-----------+-----------------------------------+-----------------------------------+
```

### Explanatory Synthesis:
- For $N=12$ independent pairs, power reaches $94.85\%$ at $d=1.25$ because $2^{12} = 4,096$ permutations allow a broad rejection region ($W \le 13$ achieves $p < 0.05$).
- For $n=6$ mechanisms, power drops precipitously to $37.15\%$ because $2^6 = 64$ permutations restrict the rejection region to $W = 0$, requiring perfect unanimity across all mechanisms and zero ties.

---

## 6. Realized Benchmark Outcome vs. Design Power

### 6.1 The Realized Sample Size: $m = 3$
In the actual benchmark execution (Phase 2B.1.5):
- 7 of the 12 active tasks failed calibration and were disqualified.
- This eliminated both configurations for `mech_03`, `mech_04`, and `mech_06`, imputing normalized differences of $0.0000$.
- Pratt filtering eliminated these 3 zero differences, leaving an effective sample size of:
  $$m = 3\text{ evaluable mechanisms}$$

### 6.2 The Realized Test Statistic and p-Value
Across the 3 non-zero mechanisms:
- `mech_01`: $+0.2222$ (favors DD)
- `mech_02`: $+0.2222$ (favors DD)
- `mech_05`: $-0.0417$ (favors Greedy)

Ranking absolute differences:
- Rank 1: `0.0417` (Negative sign: Greedy)
- Ranks 2 and 3: `0.2222` and `0.2222` (Positive signs: DD)
- Test statistic: $W^+ = 2 + 3 = 5$, $W^- = 1 \implies W = 1$.
- Total permutations: $2^3 = 8$.
- Exact subset-sum DP permutation p-value:
  $$p = \frac{4}{8} = \mathbf{0.50}$$

### 6.3 Epistemic Boundaries on Interpreting $p = 0.50$
1. **Mathematical Impossibility of Significance:** At $m = 3$, the minimum achievable two-sided p-value is $\frac{2}{8} = 0.2500 > 0.05$. Therefore, regardless of whether $W = 0$ or $W = 1$, rejecting $H_0$ was mathematically impossible once calibration disqualified 3 mechanisms.
2. **Failure to Reject $\neq$ Equivalence:** An exact p-value of $p = 0.50$ indicates that the observed data are compatible with the null hypothesis. It does not provide evidence that DD and Greedy have equal efficiency.
3. **Descriptive Status of 43.8% Reduction:** Across the 4 qualified syntax tasks (`inst_01` to `inst_04`), DD consumed 180 requests vs Greedy's 320 requests (a 43.8% reduction). This finding is strictly descriptive and cannot be generalized beyond these specific syntax instances.

---

## 7. Conclusions & Release Disposition

### Official Disposition:
$$\mathbf{POWER\_CLAIMS\_REQUIRE\_CORRECTION}$$

### Disposition Rationale:
1. **Claimed Bound Disproven:** The master protocol and subsequent audit documents incorrectly described a simulation point estimate (37.15%) as a mathematical upper bound on power.
2. **Pseudoreplication Disclosed:** Juxtaposing $94.8\%$ power at $N=12$ without prominently labeling it as a pseudoreplicated sensitivity model created an unwarranted impression that the primary experiment was adequately powered.
3. **Rejection Constraint Formalized:** The true mathematical restriction is discrete: two-sided rejection at $\alpha=0.05$ requires $m=6$ and $W=0$. Any tie drops power to identically $0.0\%$.

All historical spec files remain cryptographically preserved. The corrections established in this audit and recorded in [`phase_3_5_statistical_errata.md`](phase_3_5_statistical_errata.md) provide the authoritative scientific foundation for external review and publication.
