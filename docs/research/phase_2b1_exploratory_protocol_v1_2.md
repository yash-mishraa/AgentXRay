# AgentXRay — Phase 2B.1 Exploratory Multi-Instance Feasibility Protocol (v1.2)

**Document Status:** Preregistered & Frozen Experimental Protocol (Amended v1.2)  
**Date:** 2026-10-17  
**Status:** **DESIGN & PREREGISTRATION ONLY — NOT EXECUTED**  
**Execution Authorization:** **STRICTLY UNAUTHORIZED** (Requires separate formal authorization)  
**Target Execution Environment:** Local Apple Silicon Mac (Ollama daemon, `llama3.2:3b`, Q4_K_M)  
**Commercial Spend Authorization:** Strictly **$0.0000 USD**  
**Supersedes:** `docs/research/phase_2b1_confirmatory_protocol_v1_1.md` (v1.1) & `phase_2b1_confirmatory_protocol_v1.md` (v1.0)  
**Audit Reference:** [`docs/research/phase_2b1_2_statistical_implementation_audit.md`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/docs/research/phase_2b1_2_statistical_implementation_audit.md)  
**Matching Configuration:** [`benchmarks/configs/phase_2b1_exploratory_config_v1_2.json`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/configs/phase_2b1_exploratory_config_v1_2.json)  

---

## 1. Executive Summary & Study Designation

This document defines the formal, preregistered experimental protocol (v1.2) for Phase 2B.1 of the AgentXRay research project. The objective is to evaluate the search efficiency and failure-inducing condition recovery properties of stochastic delta debugging (`StochasticDeltaDebugMinimizer`) compared with stochastic greedy ablation (`run_stochastic_greedy_ablation`) and seeded random search across multi-instance tool-using agent tasks backed by a local neural model (`llama3.2:3b`).

### 1.1 Study Designation: Exploratory Multi-Instance Feasibility Benchmark
Following the Phase 2B.1.2 independent statistical audit, this study is formally designated as an **Exploratory Multi-Instance Feasibility Benchmark**. This designation is mathematically required because:
1. The 12 active instances comprise 6 underlying task mechanisms evaluated at two candidate space sizes ($K=4$ and $K=8$).
2. Statistical power is adequate ($\ge 80\%$) only if the standardized effect size is very large ($\text{Cohen's } d \ge 1.00$). For moderate effect sizes ($d \approx 0.50$), power is approximately $30\%$. Under stratified candidate size cohorts ($n=6$), power collapses to $< 38\%$.
3. The 4 null control instances serve as diagnostic qualitative checks, not an asymptotic proof of false discovery rate control $\le 0.05$.

### 1.2 Strict Governance & Execution Boundaries
- **No live model requests have been executed under this protocol.**
- **Execution remains strictly unauthorized until explicit formal user approval is provided.**
- **Commercial API spend remains strictly $0.0000 USD.**
- **All previous calibration records from Phase 2B.0.2 remain immutable.**

---

## 2. Research Questions, Hypotheses & Primary Co-Estimand Framework

### 2.1 Research Questions
1. **Recovery Feasibility Question:** Across calibrated model-backed tool-using agent tasks, does stochastic delta debugging achieve a verified failure-inducing configuration recovery rate comparable to stochastic greedy ablation within bounded budgets?
2. **Resource Efficiency Question:** How does the total target-model request burden consumed by stochastic delta debugging compare with stochastic greedy ablation across all assigned active task instances?

### 2.2 Formal Hypotheses
1. **$H_{1, \text{recovery}}$ (Recovery Rate Parity):** On calibrated active task instances ($M_{\text{active}} = 12$), stochastic delta debugging achieves a verified recovery rate that is not inferior to greedy ablation within an exploratory equivalence margin ($\delta = 0.15$).
2. **$H_{2, \text{efficiency}}$ (Resource Burden Reduction):** Across all assigned active instances, the median total target-model request burden of stochastic delta debugging is less than or equal to that of stochastic greedy ablation:
   $$\tilde{R}_{\text{DD}} \le \tilde{R}_{\text{Greedy}}$$
3. **$H_{3, \text{diagnostic}}$ (Diagnostic Null Consistency):** On negative control tasks ($M_{\text{null}} = 4$), both algorithms report `NO_TRIGGER_FOUND` on at least 3 of 4 instances (demonstrating qualitative resistance to false positive discoveries).

### 2.3 Dual-Endpoint Primary Co-Estimand Framework
To prevent selection bias and avoid arbitrary scalar penalty weights, the primary analysis evaluates two complementary endpoints across **all 12 assigned active task instances**:

1. **Primary Endpoint A: Verified Trigger Recovery Proportion ($\pi_{\text{rec}}$)**
   - **Estimand:** The proportion of assigned active task instances on which the algorithm successfully returns an empirically verified 1-minimal failure-inducing condition within the allocated budget $R_{\text{cap}}$.
   - **Inference:** Wilson score 95% confidence intervals per algorithm, paired risk difference $\hat{\pi}_{\text{DD}} - \hat{\pi}_{\text{Greedy}}$, and exact 2x2 contingency table.
2. **Primary Endpoint B: Total Target-Model Request Burden ($R_{\text{total}}$)**
   - **Estimand:** Total target-model requests consumed per assigned task, defined as:
     $$R_{\text{total}} = R_{\text{search}} + R_{\text{verification}} + R_{\text{retries}}$$
     where budget-exhausted or unverified runs are scored at their actual expended resource count ($R_{\text{cap}}$).
   - **Inference:** Median and Interquartile Range (IQR) reported per algorithm across all 12 active instances.

---

## 3. Secondary & Descriptive Subgroup Analyses

### 3.1 Conditional Efficiency on Jointly Resolved Instances (Descriptive Only)
- **Scope:** Restricted strictly to instances where *both* algorithms successfully achieved verified trigger recovery without budget exhaustion ($n_{\text{resolved}} \le 12$).
- **Estimand:** Paired difference in target-model requests:
  $$\Delta_i = R_{\text{Greedy}, i} - R_{\text{DD}, i}$$
  Hodges-Lehmann median paired difference estimator $\hat{\theta}_{\text{HL}}$ with exact 95% confidence intervals and paired two-sided Wilcoxon signed-rank test using Pratt's method for ties.
- **Strict Governance Caveat:** This analysis is strictly descriptive. Because it conditions on post-treatment success influenced by both algorithms (collider conditioning / survivorship bias), it **cannot** be generalized to all registered tasks or interpreted as general algorithmic superiority.

### 3.2 Paired Binary Recovery Analysis
- Evaluated at the mechanism level ($n=6$ independent base task mechanisms).
- Report the complete 2x2 paired contingency table:
  - Both recover ($n_{11}$)
  - Only DD recovers ($n_{10}$)
  - Only Greedy recovers ($n_{01}$ )
  - Neither recovers ($n_{00}$)
- Descriptive McNemar's exact test is reported with the explicit caveat that with $n=6$, statistical power is near zero unless discordant pairs are $6-0$.

### 3.3 Descriptive Cumulative Discovery Curves
- Rather than applying invalid survival hypothesis tests (which violate non-informative censoring), request consumption will be visualized as **Descriptive Empirical Cumulative Discovery Curves**:
  $$CDF(r) = \frac{1}{M_{\text{active}}} \sum_{i=1}^{M_{\text{active}}} \mathbb{I}(R_i \le r \land \text{Verified Recovery}_i = 1)$$
  displaying the proportion of tasks resolved as a function of allocated request budget $r \in [0, R_{\text{cap}}]$.

---

## 4. Task Architecture & Independent Mechanisms

### 4.1 Clustered Task Structure (6 Base Mechanisms $\times$ 2 Sizes)
The benchmark evaluates 16 task instances registered in [`benchmarks/model_pilot/confirmatory_tasks.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/confirmatory_tasks.py), linked via explicit, machine-checkable `mechanism_id` values:

| Mechanism ID | Task Family | Required Tool | Planted Trigger Summary | $K=4$ Instance ID | $K=8$ Instance ID |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `mech_01_syntax_currency` | `TOOL_ARG_SYNTAX` | `calculator` | Currency symbol `$` in calculator operands | `inst_01_syntax_currency_k4` | `inst_02_syntax_currency_k8` |
| `mech_02_syntax_comma` | `TOOL_ARG_SYNTAX` | `calculator` | Comma thousands separators in expressions | `inst_03_syntax_comma_k4` | `inst_04_syntax_comma_k8` |
| `mech_03_routing_override` | `TOOL_ROUTING_CONFLICT` | `database_lookup` | Query field override to `non_existent_secret_key` | `inst_05_routing_override_k4` | `inst_06_routing_override_k8` |
| `mech_04_routing_prohibition` | `TOOL_ROUTING_CONFLICT` | `calculator` | Prohibit calling external tools constraint | `inst_07_routing_prohibition_k4` | `inst_08_routing_prohibition_k8` |
| `mech_05_schema_unit` | `TOOL_PARAM_SCHEMA` | `unit_converter` | Target unit `light_years` schema violation | `inst_09_schema_unit_k4` | `inst_10_schema_unit_k8` |
| `mech_06_schema_cross_unit` | `TOOL_PARAM_SCHEMA` | `unit_converter` | Mass-to-length cross-category conversion | `inst_11_schema_cross_unit_k4` | `inst_12_schema_cross_unit_k8` |
| `mech_null_01_calc` | `NULL_CONTROL` | `calculator` | Benign arithmetic prompts; zero triggers | `inst_13_null_calc_k4` | `inst_14_null_calc_k8` |
| `mech_null_02_db` | `NULL_CONTROL` | `database_lookup` | Benign database queries; zero triggers | `inst_15_null_db_k4` | `inst_16_null_db_k8` |

### 4.2 Distractor Characterization & Unobserved Space Qualification
- **$K = 4$ ($16$ subsets):** 100% full power set characterization under $N=5$ trials per subset ($160$ requests).
- **$K = 8$ ($256$ subsets):** Budgeted boundary and singleton characterization ($18$ subsets tested = $1$ empty baseline, $8$ singletons, $8$ complements, $1$ full set).
  - Evaluated: $18 / 256 = \mathbf{7.03\%}$ of subset space.
  - **Unobserved:** $238 / 256 = \mathbf{92.97\%}$ of subset space.
  - *Scientific Qualification:* Unmodeled multi-element distractor synergies cannot be detected during characterization. If an empirical failure condition exists outside the planted reference trigger $C^*$, discovered subsets will be reported under the empirical minimality taxonomy.

---

## 5. Calibration & Task Qualification Gates

Before any instance is admitted to the comparative algorithm benchmark, it must pass the following empirical qualification gates ($N=5$ trials per configuration):

### 5.1 Active Task Qualification Gate ($M_{\text{active}} = 12$)
1. **Clean Baseline Pass:** $\hat{p}(\emptyset) \le 0.10$.
2. **Trigger-Present Separation:** $\hat{p}(S \cup \{T_1\}) \ge 0.70$ for all characterized trigger-containing subsets.
3. **Trigger-Absent Non-Interference:** $\hat{p}(S \setminus \{T_1\}) \le 0.20$ for all characterized trigger-lacking subsets.
4. **Failure Class Specificity:** Trigger failures must match `FAIL_TOOL_EXEC`.

### 5.2 Null Task Qualification Gate ($M_{\text{null}} = 4$)
1. **Unablated Clean Pass:** $\hat{p}(C_{\text{null}}) \le 0.10$.
2. **Empty Baseline Clean Pass:** $\hat{p}(\emptyset) \le 0.10$.

Any task failing these gates is disqualified *before* minimization runs and documented as `DISQUALIFIED_CALIBRATION_GATE`.

---

## 6. Statistical Power & Feasibility Analysis

### 6.1 Exact Permutation vs. Asymptotic Sensitivity Analysis
Using [`benchmarks/model_pilot/power_and_budget_analysis.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/power_and_budget_analysis.py), we conducted Monte Carlo simulations ($N_{\text{sims}} = 2000$, $N=12$, critical $W \le 13$, $\alpha = 0.0425$):

| Cohen's $d$ | Mean Paired Diff $\mu_\Delta$ (Requests) | Exact DP Permutation Power | Monte Carlo SE ($\pm \text{MCSE}$) | Pratt-Tied Asymptotic Power | Right-Skewed Power | Discrete Integer Ties Power | Stratified Cohort ($n=6$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $d = 0.40$ | $4.8$ | **$20.1\%$** | $\pm 0.90\%$ | $19.7\%$ | $10.6\%$ | $16.4\%$ | $5.3\%$ |
| $d = 0.50$ | $6.0$ | **$30.4\%$** | $\pm 1.03\%$ | $30.2\%$ | $20.6\%$ | $23.2\%$ | $7.1\%$ |
| $d = 0.60$ | $7.2$ | **$43.0\%$** | $\pm 1.11\%$ | $42.4\%$ | $41.0\%$ | $35.4\%$ | $10.4\%$ |
| $d = 0.70$ | $8.4$ | **$53.4\%$** | $\pm 1.12\%$ | $52.8\%$ | $66.0\%$ | $45.3\%$ | $15.2\%$ |
| $d = 0.80$ | $9.6$ | **$64.0\%$** | $\pm 1.07\%$ | $63.3\%$ | $86.8\%$ | $53.9\%$ | $16.4\%$ |
| $d = 1.00$ | $12.0$ | **$83.0\%$** | $\pm 0.84\%$ | $82.5\%$ | $100.0\%$ | $73.8\%$ | $24.9\%$ |
| **$d = 1.25$ (Pilot Extrapolated)** | **$15.0$** | **$94.8\%$** | $\pm 0.49\%$ | **$94.5\%$** | **$100.0\%$** | **$89.3\%$** | **$37.1\%$** |

### 6.2 Power Conclusions
- Target statistical power ($\ge 80\%$) is attained **only if the standardized effect size is $d \ge 1.00$** ($\mu_\Delta \ge 12.0$ requests with $\sigma_\Delta \le 12.0$).
- For realistic moderate effect sizes ($d = 0.50$), statistical power is approximately **$30\%$**.
- When stratified by candidate size ($n=6$), statistical power collapses to **$< 38\%$ even under the extreme pilot assumption of $d = 1.25$**.
- **Definitive Conclusion:** The statistical power analysis does **not** support confirmatory superiority claims. The experiment must strictly be classified as an **Exploratory Multi-Instance Feasibility Benchmark**.

---

## 7. Request Budget Architecture & Safety Firewall

### 7.1 Four-Tier Request Budget Accounting
Request demand was reconciled across four operational tiers using [`power_and_budget_analysis.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/power_and_budget_analysis.py):

| Budget Tier | Total Model Requests | Est. Wall-Clock Time (Local Mac) | Commercial Spend | Description |
| :--- | :---: | :---: | :---: | :--- |
| **1. LOW_DEMAND** | **$4,500$** | $33.8\text{ min}$ | $\$0.0000$ | Fast convergence, no budget caps hit, clean singletons |
| **2. EXPECTED** | **$5,550$** | $41.6\text{ min}$ | $\$0.0000$ | Empirical pilot convergence, typical distractors, minimal retries |
| **3. HIGH_DEMAND** | **$7,330$** | $55.0\text{ min}$ | $\$0.0000$ | Greedy & Random exhaust caps on $K=8$, 5% infra retries |
| **4. THEORETICAL_MAX** | **$8,200$** | $61.5\text{ min}$ | $\$0.0000$ | All 3 algorithms exhaust caps ($120$ on $K=4$, $180$ on $K=8$) across all tasks |

### 7.2 Safety Firewall & Budget Rules
- **Global Safety Stop Cap:** Set to **$7,500$ total model requests** in [`phase_2b1_exploratory_config_v1_2.json`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/configs/phase_2b1_exploratory_config_v1_2.json) (providing $+170$ requests of headroom over the high demand scenario, halting execution gracefully before the theoretical maximum).
- **Per-Algorithm Search Caps:** $R_{\text{cap}} = 120$ requests for $K=4$; $R_{\text{cap}} = 180$ requests for $K=8$.
- **Hard Pre-Flight Budget Check:** Enforced before every single request in `ControlledAgent` and `CostTracker`. Zero calls permitted once budget is reached.
- **Hard Spend Cap:** Strictly **$0.0000 USD** (100% local loopback).

---

## 8. Statistical Ground Truth & 1-Minimality Verification

### 8.1 Intersection-Union Testing Principle for 1-Minimality
1-minimality requires that *every* single-element removal reduces the failure rate.
Following the Berger (1982) Intersection-Union Test principle, testing each individual deletion at level $\alpha = 0.05$ guarantees overall Type I error $\le 0.05$. Multiplicity penalties (such as Benjamini-Hochberg) across single-element deletions within a candidate are not required and are excluded to prevent artificial penalties on small sample sizes ($N=5$).

### 8.2 Explicit Minimality Outcome Taxonomy
Every discovered candidate subset is classified into exactly one mutually exclusive outcome:
- `SUPPORTED_1_MINIMAL`: Candidate reproduces failure ($\hat{p} \ge 0.50$, Wilson lower bound $> 0.20$), and every single-element deletion is empirically verified to reduce failure rate ($p_j < 0.05$ or ablated rate $\le 0.20$).
- `OBSERVED_1_MINIMAL`: Failure rate dropped upon every element removal, but finite-sample tests did not achieve statistical significance at $\alpha = 0.05$.
- `NOT_MINIMAL`: Removal of at least one element failed to decrease the failure rate or still exceeded $\theta_{\text{fail}} = 0.50$.
- `INCONCLUSIVE`: Verification was interrupted before evaluating all deletions (e.g. missing checks or budget exhaustion).
- `BUDGET_EXHAUSTED`: Algorithm reached request cap before completing search or verification.
- `INFRA_FAILURE`: Provider crash, disconnect, or timeout.

A missing deletion result is strictly classified as `INCONCLUSIVE` and **never** counts as evidence of minimality.

---

## 9. Failure Classification Taxonomy

Every target execution will be assigned a mutually exclusive failure classification:
- `CLEAN_PASS`: Tool executed cleanly, output returned expected result.
- `FAIL_TOOL_EXEC`: Tool not called, or called with malformed arguments (planted trigger target).
- `FAIL_BEHAVIOR`: Tool executed, but final text violated content criteria.
- `FAIL_INFRA`: Daemon disconnect, timeout, or budget exhaustion (excluded from behavioral failure counts).

---

## 10. Known Scientific Limitations & Falsification Conditions

### 10.1 Falsification Conditions
The hypothesis that stochastic delta debugging provides an efficiency advantage over greedy ablation on local model-backed tool-using agents will be **falsified** if:
1. Stochastic delta debugging achieves a verified recovery rate significantly inferior to greedy ablation ($\hat{\pi}_{\text{DD}} < \hat{\pi}_{\text{Greedy}}$ with $p < 0.05$).
2. Across all active instances, the median total request burden of delta debugging exceeds that of greedy ablation ($\tilde{R}_{\text{DD}} > \tilde{R}_{\text{Greedy}}$).
3. On jointly resolved instances, the Hodges-Lehmann median paired reduction is $\le 0$ requests.

### 10.2 Known Scientific Limitations
1. **Single Model Architecture:** Evaluated exclusively on `llama3.2:3b` under 4-bit quantization (Q4_K_M).
2. **Exploratory Sample Size:** With $n=6$ base tasks across $K \in \{4, 8\}$, the experiment is underpowered for moderate effect sizes ($d < 0.80$).
3. **Unobserved Distractor Space ($K=8$):** $93\%$ of the $K=8$ power set remains uncharacterized.
4. **Negative Controls are Diagnostic:** $N=4$ null instances cannot establish formal FDR control $\le 0.05$.

---

## 11. Preregistration Freeze & Cryptographic Verification

- **Document Version:** v1.2
- **Configuration File:** `benchmarks/configs/phase_2b1_exploratory_config_v1_2.json`
- **Validation Command:** `./.venv/bin/python benchmarks/model_pilot/validate_confirmatory_protocol.py --config benchmarks/configs/phase_2b1_exploratory_config_v1_2.json --protocol docs/research/phase_2b1_exploratory_protocol_v1_2.md`
- **Execution Status:** **STRICTLY UNAUTHORIZED**

> [!CAUTION]
> **PHASE 2B.1 EXPERIMENT EXECUTION REMAINS STRICTLY UNAUTHORIZED.**
> Live model inference under this protocol is prohibited until explicit user authorization is provided.
