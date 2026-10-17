# AgentXRay — Phase 2B.1 Exploratory Multi-Instance Feasibility Protocol (v1.1)

**Document Status:** Preregistered & Frozen Experimental Protocol (Amended v1.1)  
**Date:** 2026-10-17  
**Status:** **DESIGN & PREREGISTRATION ONLY — NOT EXECUTED**  
**Execution Authorization:** **STRICTLY UNAUTHORIZED** (Requires separate formal authorization)  
**Target Execution Environment:** Local Apple Silicon Mac (Ollama daemon, `llama3.2:3b`, Q4_K_M)  
**Commercial Spend Authorization:** Strictly **$0.0000 USD**  
**Supersedes:** `docs/research/phase_2b1_confirmatory_protocol_v1.md` (v1.0 audited in `phase_2b1_protocol_audit_v1.md`)  
**Matching Configuration:** [`benchmarks/configs/phase_2b1_confirmatory_config_v1_1.json`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/configs/phase_2b1_confirmatory_config_v1_1.json)  

---

## 1. Executive Summary & Study Designation

This document specifies the revised, preregistered experimental protocol (v1.1) for Phase 2B.1 of the AgentXRay research project. The objective is to investigate the search efficiency (measured in target model inference requests) and minimality recovery properties of stochastic delta debugging (`StochasticDeltaDebugMinimizer`) compared with stochastic greedy ablation (`run_stochastic_greedy_ablation`) and seeded random search across multi-instance tool-using agent tasks backed by a local neural model (`llama3.2:3b`).

### 1.1 Study Designation: Exploratory Multi-Instance Feasibility Benchmark
Following the independent audit (`phase_2b1_protocol_audit_v1.md`), this study is designated as an **Exploratory Multi-Instance Feasibility Benchmark**, rather than a definitive confirmatory proof of universal algorithmic superiority. This designation is necessary because:
1. The 12 active instances represent 6 underlying task mechanisms evaluated at two candidate sizes ($K=4$ and $K=8$).
2. Statistical power is adequate ($\ge 80\%$) only for very large standardized effect sizes ($\text{Cohen's } d \ge 1.00$).
3. The 4 null control instances serve as diagnostic qualitative checks, not an asymptotic proof of false discovery rate control $\le 0.05$.

### 1.2 Strict Governance & Execution Boundaries
- **No live model requests have been executed under this protocol.**
- **Execution remains strictly unauthorized until explicitly approved.**
- **Commercial API spend remains strictly $0.0000 USD.**
- **All previous calibration records from Phase 2B.0.2 remain immutable.**

---

## 2. Research Questions, Hypotheses & Two-Part Estimand Framework

### 2.1 Primary Research Questions
1. **Search Efficiency Question:** Across calibrated model-backed tool-using agent tasks, does stochastic delta debugging require fewer target model inference requests than stochastic greedy ablation to isolate a verified failure-inducing configuration?
2. **Recovery Parity Question:** Does stochastic delta debugging achieve a verified failure-inducing configuration recovery rate comparable to stochastic greedy ablation?

### 2.2 Formal Hypotheses
1. **$H_{1, \text{efficiency}}$ (Conditional Efficiency):** On task instances where both algorithms successfully isolate an empirically verified failure-inducing configuration without budget exhaustion, the paired difference in target model inference requests:
   $$\Delta_i = R_{\text{Greedy}, i} - R_{\text{DD}, i}$$
   has a median strictly greater than zero ($\text{Median}(\Delta_i) > 0$).
2. **$H_{0, \text{efficiency}}$ (Efficiency Null):** $\text{Median}(\Delta_i) \le 0$. Delta debugging requires equal or more model inference requests than greedy ablation.
3. **$H_{2, \text{recovery}}$ (Recovery Rate Parity):** Stochastic delta debugging achieves a verified ground-truth recovery rate that is not significantly different from greedy ablation under McNemar's exact test ($\alpha = 0.05$).
4. **$H_{3, \text{diagnostic}}$ (Diagnostic Null Consistency):** On negative control tasks ($M_{\text{null}} = 4$), both algorithms report `NO_TRIGGER_FOUND` on at least 3 of 4 instances (demonstrating qualitative resistance to false positive discoveries).

### 2.3 Two-Part Estimand Framework
To prevent the statistical distortions caused by arbitrary worst-rank assignment to budget-exhausted runs, Phase 2B.1 adopts a **Two-Part Hurdle / Survival Analysis Plan**:

- **Part A (Binary Verified Recovery):**
  - **Estimand:** Difference in verified trigger recovery proportions $\hat{p}_{\text{DD}} - \hat{p}_{\text{Greedy}}$.
  - **Inference:** Two-sided McNemar's exact test for paired binary outcomes ($\alpha = 0.05$).
- **Part B (Efficiency on Jointly Resolved Instances):**
  - **Estimand:** Hodges-Lehmann median paired difference $\hat{\theta}_{\text{HL}}$ for $\Delta_i = R_{\text{Greedy}, i} - R_{\text{DD}, i}$, calculated strictly over instances where *both* algorithms resolved without budget exhaustion.
  - **Inference:** Paired two-sided Wilcoxon signed-rank test using Pratt's method for ties ($\alpha = 0.05$).
- **Part C (Right-Censored Survival Analysis):**
  - **Estimand:** Request-to-discovery survival distributions $S(R)$.
  - **Inference:** Peto-Peto Wilcoxon test treating budget-exhausted runs as right-censored at $R_{\text{cap}}$.

---

## 3. Experimental Units & Task Instance Specifications

### 3.1 Clustered Task Structure (6 Base Tasks $\times$ 2 Sizes)
The benchmark evaluates 16 total task instances registered in [`benchmarks/model_pilot/confirmatory_tasks.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/confirmatory_tasks.py). The 12 active instances represent **6 distinct base task mechanisms**, each instantiated at $K=4$ and $K=8$:

| Base Task ID | Task Family | Required Tool | Planted Trigger Summary | $K=4$ Instance ID | $K=8$ Instance ID |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **B1** | `TOOL_ARG_SYNTAX` | `calculator` | Inject currency symbol `$` into operand | `inst_01_syntax_currency_k4` | `inst_02_syntax_currency_k8` |
| **B2** | `TOOL_ARG_SYNTAX` | `calculator` | Inject comma thousands separator into operand | `inst_03_syntax_comma_k4` | `inst_04_syntax_comma_k8` |
| **B3** | `TOOL_ROUTING_CONFLICT` | `database_lookup` | Require field `non_existent_secret_key` | `inst_05_routing_override_k4` | `inst_06_routing_override_k8` |
| **B4** | `TOOL_ROUTING_CONFLICT` | `calculator` | Prohibit calling external tools | `inst_07_routing_prohibition_k4` | `inst_08_routing_prohibition_k8` |
| **B5** | `TOOL_PARAM_SCHEMA` | `unit_converter` | Target unsupported unit `light_years` | `inst_09_schema_unit_k4` | `inst_10_schema_unit_k8` |
| **B6** | `TOOL_PARAM_SCHEMA` | `unit_converter` | Mass-to-length cross-category conversion | `inst_11_schema_cross_unit_k4` | `inst_12_schema_cross_unit_k8` |
| **N1** | `NULL_CONTROL` | `calculator` | 4 or 8 benign arithmetic distractors | `inst_13_null_calc_k4` | `inst_14_null_calc_k8` |
| **N2** | `NULL_CONTROL` | `database_lookup` | 4 or 8 benign database query distractors | `inst_15_null_db_k4` | `inst_16_null_db_k8` |

### 3.2 Distractor Characterization & Unobserved Space Qualification
- **$K = 4$ ($16$ subsets):** 100% full power set characterization under $N=5$ trials per subset ($160$ requests).
- **$K = 8$ ($256$ subsets):** Budgeted boundary and singleton characterization ($18$ subsets tested = $1$ empty, $8$ singletons, $8$ complements, $1$ full set).
  - Evaluated: $18 / 256 = 7.03\%$ of subset space.
  - **Unobserved:** $238 / 256 = 92.97\%$ of subset space.
  - *Scientific Caveat:* If a task exhibits unmodeled multi-element distractor synergies (e.g. elements $\{d_2, d_5\}$ causing failure without $T_1$), minimization may isolate an empirical failure subset outside the planted ground truth $C^*$. Both empirical minimality and ground truth recovery will be reported.

---

## 4. Calibration & Task Qualification Gates

Before any instance is admitted to the comparative algorithm benchmark, it must pass the following empirical qualification gates ($N=5$ trials per configuration):

### 4.1 Active Task Qualification Gate ($M_{\text{active}} = 12$)
1. **Clean Baseline Pass:** $\hat{p}(\emptyset) \le 0.10$.
2. **Trigger-Present Separation:** $\hat{p}(S \cup \{T_1\}) \ge 0.70$ for all characterized trigger-containing subsets.
3. **Trigger-Absent Non-Interference:** $\hat{p}(S \setminus \{T_1\}) \le 0.20$ for all characterized trigger-lacking subsets.
4. **Failure Class Specificity:** Trigger failures must match `FAIL_TOOL_EXEC`.

### 4.2 Null Task Qualification Gate ($M_{\text{null}} = 4$)
1. **Unablated Clean Pass:** $\hat{p}(C_{\text{null}}) \le 0.10$.
2. **Empty Baseline Clean Pass:** $\hat{p}(\emptyset) \le 0.10$.

Any task failing these gates is disqualified *before* minimization runs and documented as `DISQUALIFIED_CALIBRATION_GATE`.

---

## 5. Statistical Power & Feasibility Analysis

### 5.1 Sensitivity Analysis Across Plausible Effect Sizes
Extrapolating population effect size from $n=1$ pilot data ($d=1.25$) is scientifically overoptimistic. The table below presents exact Wilcoxon signed-rank power across $10,000$ Monte Carlo simulations ($n=12$, critical $W \le 13$, two-sided $\alpha = 0.0425$):

| Assumed Effect Size (Cohen's $d$) | Mean Paired Diff $\mu_\Delta$ (Requests) | Normal (5% Ties) | Skewed (5% Ties) | High Ties (15% Ties) | Stratified Cohort ($n=6$, $K=4$ or $K=8$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $d = 0.40$ | $4.8$ | $20.3\%$ | $10.3\%$ | $17.5\%$ | $5.9\%$ |
| $d = 0.50$ | $6.0$ | $29.3\%$ | $21.7\%$ | $25.4\%$ | $8.1\%$ |
| $d = 0.60$ | $7.2$ | $40.7\%$ | $39.9\%$ | $35.3\%$ | $10.5\%$ |
| $d = 0.70$ | $8.4$ | $52.7\%$ | $64.9\%$ | $45.4\%$ | $14.1\%$ |
| $d = 0.80$ | $9.6$ | $63.3\%$ | $87.2\%$ | $55.5\%$ | $17.6\%$ |
| $d = 1.00$ | $12.0$ | $81.6\%$ | $100.0\%$ | $74.5\%$ | $26.2\%$ |
| **$d = 1.25$ (Pilot Extrapolated)** | **$15.0$** | **$94.6\%$** | **$100.0\%$** | **$89.8\%$** | **$38.2\%$** |

### 5.2 Power Conclusions
- The experiment has statistical power $\ge 80\%$ **only if** the standardized effect size is $d \ge 1.00$ ($\mu_\Delta \ge 12.0$ requests with $\sigma_\Delta \le 12.0$).
- If the true effect size is moderate ($d = 0.50$), power is $\sim 29\%$.
- Stratified comparisons at $n=6$ are exploratory and underpowered ($< 38.2\%$ power even at $d = 1.25$).
- Consequently, this study must be evaluated as an **exploratory feasibility benchmark**, not a confirmatory test.

---

## 6. Aggregate Request Budget & Runtime Feasibility

### 6.1 Three Demand Scenarios
We modeled request demand across three scenarios using [`power_and_budget_analysis.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/power_and_budget_analysis.py):

| Demand Scenario | Total Model Requests | Est. Wall-Clock Time (Local Mac) | Commercial Spend | Description |
| :--- | :---: | :---: | :---: | :--- |
| **LOW_DEMAND** | **$4,500$** | $33.8\text{ min}$ | $\$0.0000$ | Fast convergence, no budget caps hit, clean singletons |
| **EXPECTED** | **$5,550$** | $41.6\text{ min}$ | $\$0.0000$ | Empirical pilot convergence, typical distractors, minimal retries |
| **HIGH_DEMAND (Worst Case)** | **$7,330$** | $55.0\text{ min}$ | $\$0.0000$ | Greedy & Random exhaust caps on $K=8$, 5% infra retries |

### 6.2 Revised Budget Rules
- **Global Safety Stop Cap:** Set to **$7,500$ total model requests** in [`phase_2b1_confirmatory_config_v1_1.json`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/configs/phase_2b1_confirmatory_config_v1_1.json) (providing $230$ requests of headroom over the worst-case scenario).
- **Per-Algorithm Caps:** $R_{\text{cap}} = 120$ requests for $K=4$; $R_{\text{cap}} = 180$ requests for $K=8$.
- **Hard Spend Cap:** Strictly **$0.0000 USD** (100% local loopback).

---

## 7. Statistical Ground Truth & 1-Minimality Criteria

### 7.1 Statistical 1-Minimality Definition (FDR-Aligned)
A discovered candidate subset $\hat{C}$ is declared **statistically supported 1-minimal** if and only if:
1. **Self-Reproduction:** Under $N=5$ trials, $\hat{p}(\hat{C}) \ge 0.50$ with Wilson lower bound $L_{95}(\hat{C}) > 0.20$.
2. **Ablation Reversion:** For every single-element deletion $c \in \hat{C}$, the ablated subset $\hat{C} \setminus \{c\}$ achieves $\hat{p}(\hat{C} \setminus \{c\}) \le 0.20$ with Wilson upper bound $U_{95}(\hat{C} \setminus \{c\}) < 0.70$.
3. **Multiplicity Control (Implementation Aligned):** Single-element deletion tests are evaluated using the **Benjamini-Hochberg procedure** at false discovery rate $\alpha_{\text{FDR}} = 0.05$, matching [`src/agentxray/stochastic_minimizer.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/src/agentxray/stochastic_minimizer.py#L370-L388).
4. **Independent Holdout Confirmation:** Evaluated on a fresh $N=5$ holdout sample unobserved during search.

---

## 8. Statistical Analysis Plan (SAP)

### 8.1 Primary SAP: Two-Part Hurdle Framework
1. **Part A (Verified Recovery Rate Parity):**
   - Paired McNemar's exact test on binary verified trigger recovery:
     $$p_{\text{McNemar}} = 2 \sum_{k=b}^{\min(b+c, 2b)} \binom{b+c}{k} 0.5^{b+c}$$
     where $b$ is the number of instances where DD succeeds and Greedy fails, and $c$ is the reverse.
2. **Part B (Efficiency on Jointly Resolved Instances):**
   - Paired two-sided Wilcoxon signed-rank test on $\Delta_i = R_{\text{Greedy}, i} - R_{\text{DD}, i}$ evaluated across jointly resolved instances ($n_{\text{resolved}} \le 12$).
   - Hodges-Lehmann median difference estimator $\hat{\theta}_{\text{HL}}$ with exact 95% confidence intervals.
   - Handled via Pratt's method for zero differences.
3. **Part C (Right-Censored Survival Analysis):**
   - Kaplan-Meier estimation of request consumption with budget exhaustion treated as right-censoring at $R_{\text{cap}}$.
   - Peto-Peto Wilcoxon test comparison.

### 8.2 Stratified & Clustered Subgroup Analyses
1. **Stratification by Candidate Size:** Separate reporting of Part B for $K=4$ ($n \le 6$) and $K=8$ ($n \le 6$).
2. **Cluster-Averaged Analysis:** Mechanism-level aggregation across the 6 base tasks:
   $$\bar{\Delta}_j = \frac{1}{|K_j|} \sum_{k \in K_j} \Delta_{j, k}$$

### 8.3 Multiplicity Control for Secondary Endpoints
Secondary endpoint comparisons (exact recovery, 1-minimality, holdout rate, Jaccard similarity) will be adjusted using the **Holm-Bonferroni step-down method** across task families.

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
1. The Wilcoxon signed-rank test on jointly resolved instances yields $p \ge 0.05$.
2. The Hodges-Lehmann median paired difference is $\le 0$ requests.
3. McNemar's test demonstrates that delta debugging achieves a significantly lower recovery rate than greedy ablation ($p < 0.05$ with $\hat{p}_{\text{DD}} < \hat{p}_{\text{Greedy}}$).

### 10.2 Known Scientific Limitations
1. **Single Model Architecture:** Evaluated exclusively on `llama3.2:3b` under 4-bit quantization (Q4_K_M).
2. **Exploratory Sample Size:** With $n=6$ base tasks across $K \in \{4, 8\}$, the experiment is underpowered for moderate effect sizes ($d < 0.80$).
3. **Unobserved Distractor Space ($K=8$):** $93\%$ of the $K=8$ power set remains uncharacterized.
4. **Negative Controls are Diagnostic:** $N=4$ null instances cannot establish formal FDR control $\le 0.05$.

---

## 11. Preregistration Freeze & Cryptographic Verification

- **Document Version:** v1.1
- **Configuration File:** `benchmarks/configs/phase_2b1_confirmatory_config_v1_1.json`
- **Validation Command:** `./.venv/bin/python benchmarks/model_pilot/validate_confirmatory_protocol.py --config benchmarks/configs/phase_2b1_confirmatory_config_v1_1.json --protocol docs/research/phase_2b1_confirmatory_protocol_v1_1.md`
- **Execution Status:** **STRICTLY UNAUTHORIZED**

> [!CAUTION]
> **PHASE 2B.1 EXPERIMENT EXECUTION REMAINS STRICTLY UNAUTHORIZED.**
> Live model inference under this protocol is prohibited until explicit user authorization is provided.
