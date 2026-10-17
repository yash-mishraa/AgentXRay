# AgentXRay — Phase 2B.1 Exploratory Benchmark Protocol v1.4

**Status:** PREREGISTERED_FROZEN  
**Study Designation:** EXPLORATORY_FEASIBILITY_BENCHMARK  
**Document Reference:** `docs/research/phase_2b1_exploratory_protocol_v1_4.md`  
**Configuration Reference:** `benchmarks/configs/phase_2b1_exploratory_config_v1_4.json`  
**Date:** 2026-10-17  
**Execution Tier:** LOCAL_MODEL (`llama3.2:3b` via Ollama)  
**Commercial Spend Cap:** Strictly $0.0000 USD (zero commercial API spend permitted)  

---

## 1. Executive Summary & Scientific Scope

This protocol establishes the authoritative frozen experimental specification for the Phase 2B.1 exploratory feasibility benchmark in Project AgentXRay.

The central research question is:
> Under what operational conditions can stochastic delta debugging isolate empirically supported 1-minimal failure-inducing configurations in black-box tool-using LLM agents, and how do its recovery rates and target-model request burdens compare with stochastic greedy ablation and seeded random search?

### 1.1 Scope Boundaries & Exploratory Nature
- **Exploratory Feasibility Benchmark:** Phase 2B.1 evaluates minimization algorithms across six distinct underlying failure mechanisms ($n=6$) instantiated at two candidate-space sizes ($K=4$ and $K=8$), plus four null-control instances, yielding 16 instances total (12 active, 4 null).
- **No Confirmatory Superiority Claims:** With $n=6$ independent failure mechanisms, the study does not possess adequate statistical power to establish broad, generalizable algorithmic superiority.
- **Strict Execution Boundary:** This document serves as a frozen preregistration. **No target-model inference or experiment execution is authorized during this audit phase.** Execution requires separate, explicit authorization.

---

## 2. Independent Experimental Units & Mechanism Clustering

### 2.1 The True Independent Unit
The benchmark comprises 16 tasks generated from 8 distinct failure mechanisms:
- **6 Active Mechanisms:** $M_1$ through $M_6$.
- **2 Null Mechanisms:** $N_1$ and $N_2$.

For each active mechanism $M_i$, the $K=4$ instance and $K=8$ instance share:
- The exact same underlying causal trigger (e.g., `strict_currency_formatting`).
- The exact same user prompt and task instructions.
- The exact same tool environment and evaluation criteria.
- The exact same base failure mechanism in the model harness.

The only operational difference is the number of distractor elements added to the prompt context (3 distractors for $K=4$; 7 distractors for $K=8$).

**Methodological Implication:** The $K=4$ and $K=8$ instances are **repeated measurements of the same underlying causal mechanism under two different context-load conditions**. They are **not** independent failure mechanisms. Treating the 12 active entries as 12 independent observations in statistical hypothesis testing constitutes **pseudoreplication** (Hurlbert 1984).

### 2.2 Mechanism Registry Table

| Mechanism ID | Mechanism Description | Task Family | $K=4$ Instance ID | $K=8$ Instance ID | Planted Trigger |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `mech_01_syntax_currency` | Currency formatting syntax error | `TOOL_ARG_SYNTAX` | `inst_01_syntax_currency_k4` | `inst_02_syntax_currency_k8` | `strict_currency_formatting` |
| `mech_02_syntax_comma` | European comma decimal syntax | `TOOL_ARG_SYNTAX` | `inst_03_syntax_comma_k4` | `inst_04_syntax_comma_k8` | `comma_decimal_separator` |
| `mech_03_routing_override` | Ambiguous tool routing conflict | `TOOL_ROUTING_CONFLICT` | `inst_05_routing_override_k4` | `inst_06_routing_override_k8` | `route_preference_override` |
| `mech_04_routing_prohibition` | Deprecated tool prohibition | `TOOL_ROUTING_CONFLICT` | `inst_07_routing_prohibition_k4` | `inst_08_routing_prohibition_k8` | `prohibit_direct_calculation` |
| `mech_05_schema_unit` | Explicit unit suffix injection | `TOOL_PARAM_SCHEMA` | `inst_09_schema_unit_k4` | `inst_10_schema_unit_k8` | `inject_unit_suffix` |
| `mech_06_schema_cross_unit` | Unit conversion mismatch | `TOOL_PARAM_SCHEMA` | `inst_11_schema_cross_unit_k4` | `inst_12_schema_cross_unit_k8` | `cross_unit_conversion_tag` |
| `mech_null_01_calc` | Arithmetic negative control | `NULL_CONTROL` | `inst_13_null_calc_k4` | `inst_14_null_calc_k8` | *(None - Clean)* |
| `mech_null_02_db` | Database query negative control | `NULL_CONTROL` | `inst_15_null_db_k4` | `inst_16_null_db_k8` | *(None - Clean)* |

---

## 3. Research Endpoints & Statistical Analysis Plan

### 3.1 Demotion of Recovery Non-Inferiority to Descriptive Reporting
With $n=6$ independent failure mechanisms, the 95% Newcombe/Wilson score confidence interval width for paired risk difference exceeds $\pm 35\%$. Even under identical, perfect recovery (6/6 vs 6/6), the lower bound cannot rule out $\delta = 0.15$. 

Therefore, **formal non-inferiority hypothesis testing is rejected and demoted to descriptive exploratory reporting**:
1. **Paired Recovery Contingency Table:** Full $2 \times 2$ table of joint recovery outcomes (Both Recovered, DD Only, Greedy Only, Neither Recovered).
2. **Paired Risk Difference:** Point estimate $\hat{\Delta} = \hat{\pi}_{\text{DD}} - \hat{\pi}_{\text{greedy}}$ accompanied by its 95% Wilson/Newcombe score confidence interval, reported strictly for descriptive uncertainty.
3. **Exploratory Asymmetry Diagnostic:** McNemar's exact test ($p = 2 \times \min(P(B \le b), P(B \ge b))$ where $B \sim \text{Binomial}(b+c, 0.5)$) is reported as an exploratory symmetry diagnostic, noting that failure to reject symmetry does not imply equivalence.

### 3.2 Primary Resource Burden Estimand & Stratification
The primary efficiency comparison evaluates target-model request consumption between Stochastic Delta Debugging and Stochastic Greedy Ablation.

#### 3.2.1 Intention-to-Treat All-Assigned Tasks Endpoint
- **Population:** Every assigned active task instance ($N=12$) is included in the primary efficiency comparison.
- **Handling of Unsuccessful / Budget-Exhausted Runs:** Runs that fail to recover a supported trigger or hit the task request cap are retained at their actual expended requests (capped at 120 for $K=4$ or 180 for $K=8$). No arbitrary scalar penalties or imputed scores are added.

#### 3.2.2 Mandatory Candidate-Space Stratification
Because $K=4$ (cap 120, search space $2^4=16$) and $K=8$ (cap 180, search space $2^8=256$) possess fundamentally different combinatorial complexity and request ceilings, pooled raw request counts form an arbitrary mixture distribution.

Therefore, resource burden must be reported:
1. **Stratified Descriptive Reporting:**
   - $K=4$ ($n=6$, cap 120): Table of assigned tasks, actual requests, cap, normalized burden $\tilde{R} = R / 120$, recovery status, and algorithm.
   - $K=8$ ($n=6$, cap 180): Table of assigned tasks, actual requests, cap, normalized burden $\tilde{R} = R / 180$, recovery status, and algorithm.
2. **Mechanism-Level Paired Differences ($n=6$):**
   - Paired difference for mechanism $i$: $\bar{D}_i = \frac{1}{2}\left[\frac{R_{\text{Greedy}}(M_i, 4) - R_{\text{DD}}(M_i, 4)}{120} + \frac{R_{\text{Greedy}}(M_i, 8) - R_{\text{DD}}(M_i, 8)}{180}\right]$.
   - Evaluated under Wilcoxon signed-rank test with $n=6$. Total permutation space is $2^6 = 64$. Minimum achievable two-sided p-value is $2/64 = 0.03125$ ($W=0$), which requires DD to beat greedy on all 6 mechanisms without ties.
3. **Pooled $N=12$ Exploratory Summary:**
   - Reported strictly as a descriptive exploratory summary across the 12 entries, with prominent disclosure that unclustered analysis assumes independent instances (an exploratory assumption).

#### 3.2.3 Secondary Subgroup Analysis (Jointly Resolved Only)
As a secondary descriptive endpoint, paired request differences are reported for the subset of tasks where *both* algorithms achieved verified recovery, accompanied by explicit warnings regarding potential post-treatment conditioning (selection) bias.

---

## 4. Target Model Configuration & Enforceable Guardrails

### 4.1 Target Model Configuration
- **Model:** `llama3.2:3b` executed locally via Ollama daemon (`http://localhost:11434/api/generate`).
- **Inference Parameters:** `temperature=0.0`, `seed=42`, `top_p=1.0`.
- **System Constraints:** Target agent runs in a local controlled process.
- **Spend Enforcement:** Commercial API spend cap is strictly **$0.0000 USD**. No external API keys or commercial endpoints are configured or permitted.

### 4.2 Enforceable Budget Caps & Safety Hard Stop
- **Per-Task Request Caps:**
  - $K = 4$: 120 target-model requests per algorithm.
  - $K = 8$: 180 target-model requests per algorithm.
- **Global Experiment Safety Hard Cap:** **7,500 target-model requests**.
- **Enforcement Mechanism:** Pre-request budget check via `CostTracker.check_budget_before_request()`. If `total_requests >= max_model_requests`, `CostBudgetExceededError` is raised *before* issuing any network or model request.
- **Hard-Stop Status:** Any budget breach results in immediate graceful termination labeled `SearchStatus.BUDGET_EXHAUSTED` / `HARD_BUDGET_BREACH_ABORT`. It is never masked as a failure to find a trigger or recorded as zero requests.

---

## 5. Pre-Experiment Calibration & Qualification Gate

Prior to running search algorithms, each task instance undergoes independent calibration:
1. **Sample Size:** $N = 5$ trials per evaluated subset.
2. **Active Task Qualification Criteria:**
   - Trigger condition failure rate: $\hat{p}(\text{Trigger}) \ge 0.70$ (at least 4/5 failures), with Wilson 95% CI lower bound $> 0.20$.
   - Ablated control failure rate: $\hat{p}(\text{Control}) \le 0.20$ (at most 1/5 failures), with Wilson 95% CI upper bound $< 0.50$.
3. **Null Task Qualification Criteria:**
   - Baseline clean execution failure rate: $\hat{p}(\text{Clean}) \le 0.10$ (0/5 failures; 95% Wilson CI $[0.00, 0.43]$).
4. **Handling Disqualified Tasks:**
   - Any task failing calibration is classified as `DISQUALIFIED_TASK` and excluded from algorithmic evaluation.
   - Disqualified tasks are documented in the study flow diagram with explicit failure telemetry and are **never** silently dropped, replaced, or pruned post-hoc.

---

## 6. Statistical Ground Truth & Operational 1-Minimality Verification

### 6.1 Side-by-Side Minimality Hypotheses

| Dimension | Claim A: Threshold-Based Minimality | Claim B: Comparative Minimality |
| :--- | :--- | :--- |
| **Scientific Question** | "Does removing element $e_j$ bring failure probability below the failure threshold $\theta_{\text{fail}} = 0.50$?" | "Does removing element $e_j$ cause a statistically significant drop in failure probability relative to candidate $C$?" |
| **Component Null $H_{0, j}$** | $H_{0, j}^{\text{thresh}}: p(C \setminus \{e_j\}) \ge 0.50$ | $H_{0, j}^{\text{comp}}: p(C \setminus \{e_j\}) \ge p(C)$ |
| **Component Alternative $H_{1, j}$** | $H_{1, j}^{\text{thresh}}: p(C \setminus \{e_j\}) < 0.50$ | $H_{1, j}^{\text{comp}}: p(C \setminus \{e_j\}) < p(C)$ |
| **Statistical Test** | Exact one-sample binomial test ($p_0 = 0.50$, alternative='less') | Exact one-sided Fisher exact test on $2 \times 2$ table ($C$ vs $C \setminus \{e_j\}$, alternative='greater') |
| **$N=5$ Sensitivity** | Significant *only* if deletion has $0/5$ failures ($p = 0.03125$). $1/5$ failures has $p = 0.1875 > 0.05$. | Significant if candidate has 5/5 and deletion has $\le 1/5$ ($p \le 0.0238$); or candidate has 4/5 and deletion has $0/5$ ($p = 0.0238$). |
| **Failure Reproduction** | Separately checked on candidate $C$ ($\hat{p}(C) \ge 0.50$, Wilson lower bound $> 0.20$). | Inherent in comparative difference, but candidate reproduction separately verified. |
| **Non-Failure Requirement** | Directly tested by hypothesis $H_{1, j}^{\text{thresh}}$. | Requires concurrent empirical non-failure guard: $\hat{p}(C \setminus \{e_j\}) < 0.50$. |

### 6.2 Primary Operational Definition of `SUPPORTED_1_MINIMAL`
In Project AgentXRay, delta debugging isolates causal failure-inducing elements. Therefore, the primary operational definition is **Comparative Minimality under Intersection-Union Testing with an Empirical Non-Failure Guard**:

A candidate subset $C$ is classified as `SUPPORTED_1_MINIMAL` if and only if:
1. **Verification Complete:** All $k = |C|$ single-element deletions $C \setminus \{e_j\}$ were evaluated.
2. **Candidate Failure Reproduction:** Candidate $C$ reproduces failure: $\hat{p}(C) \ge \theta_{\text{fail}} = 0.50$ and Wilson 95% CI lower bound $L_{95}(C) > \theta_{\text{base}} = 0.20$.
3. **Threshold Non-Failure Guard:** Every single-element deletion satisfies $\hat{p}(C \setminus \{e_j\}) < \theta_{\text{fail}} = 0.50$. (If any deletion still produces $\ge 0.50$, the candidate is strictly `NOT_MINIMAL`).
4. **Comparative Statistical Support under IUT:** Every single-element deletion achieves one-sided exact Fisher significance at $\alpha = 0.05$:
   $$\max_{j=1,\dots,k} p_j^{\text{fisher}} < 0.05$$
   where $p_j^{\text{fisher}}$ tests $H_{0, j}^{\text{comp}}: p(C \setminus \{e_j\}) \ge p(C)$ vs $H_{1, j}^{\text{comp}}: p(C \setminus \{e_j\}) < p(C)$.

Under the Intersection-Union Principle (Berger 1982), rejecting the composite null $H_0^{\text{comp}}: \bigcup_{j=1}^k \{ p(C \setminus \{e_j\}) \ge p(C) \}$ at level $\alpha = 0.05$ requires that every component test rejects at $\alpha = 0.05$. No multiplicity correction across deletions is required.

Concurrently, exact one-sample binomial p-values $p_j^{\text{thresh}}$ against $\theta_{\text{fail}} = 0.50$ (Claim A) are computed and reported in telemetry for full diagnostic visibility.

### 6.3 Operational Minimality Outcome Taxonomy
Every candidate subset is assigned exactly one mutually exclusive outcome:
- `SUPPORTED_1_MINIMAL`: Meets all four criteria above (reproduction, verification complete, threshold guard, and all comparative Fisher $p_j < 0.05$).
- `OBSERVED_1_MINIMAL`: All deletions evaluated; candidate reproduces failure; every deletion reduces observed failure rate ($\hat{p}(C \setminus \{e_j\}) < \hat{p}(C)$ and $< 0.50$); but one or more deletions do not achieve $p < 0.05$ due to small sample size.
- `NOT_MINIMAL`: At least one deletion failed to reduce failure rate ($\hat{p}(C \setminus \{e_j\}) \ge \hat{p}(C)$) or ablated rate remained $\ge 0.50$.
- `INCONCLUSIVE`: Verification was interrupted (missing deletion evaluations), candidate failed to reproduce failure ($\hat{p}(C) < 0.50$), or zero/insufficient observations.
- `BUDGET_EXHAUSTED`: Algorithm reached per-task or global request cap before completing search or verification.
- `INFRA_FAILURE`: Unrecoverable daemon crash, network disconnect, or socket timeout.

---

## 7. Null Control Diagnostics & False Discovery Interpretation

The benchmark includes four null-control instances ($N_1, N_2$ evaluated at $K=4$ and $K=8$):
- **Diagnostic Role:** Qualitative diagnostic checks to verify that algorithms do not isolate spurious triggers when no causal trigger is planted.
- **Definition of False Discovery:** Any candidate isolated as `SUPPORTED_1_MINIMAL` or `OBSERVED_1_MINIMAL` on a null task constitutes an empirical false discovery.
- **No Formal FDR Guarantee:** With $N = 4$ null instances, observing $0/4$ false discoveries yields a Clopper-Pearson 95% upper bound of $52.7\%$ and a Wilson 95% upper bound of $49.0\%$. Passing $4/4$ or $3/4$ null controls cannot establish that the False Discovery Rate is bounded by $0.05$. Establishing a 95% upper bound $\le 0.05$ with zero false positives would require at least $N \ge 59$ null tasks ($\lceil \ln(0.05) / \ln(0.95) \rceil = 59$).

---

## 8. Request Budget Arithmetic & Safety Scenarios

| Scenario | Search DD | Search Greedy | Search Random | Search Null | Calibration | Holdout & Headroom | Total Model Requests | Est. Time (0.45s/call) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low Demand** | 480 | 720 | 600 | 120 | 2,120 | 460 | **4,500** | 33.8 min |
| **Expected Pilot** | 660 | 1,260 | 780 | 120 | 2,120 | 610 | **5,550** | 41.6 min |
| **High Demand Worst-Case** | 1,080 | 1,800 | 1,200 | 240 | 2,200 | 810 | **7,330** | 55.0 min |
| **Theoretical Maximum Cap** | 1,800 | 1,800 | 1,800 | 240 | 2,200 | 360 | **8,200** | 61.5 min |

- **Configuration Safety Hard Cap:** **7,500 requests** provides $+170$ requests of headroom beyond the high-demand worst-case scenario (7,330).
- **Safety Interlock:** The orchestrator halts execution via `HARD_BUDGET_BREACH_ABORT` prior to reaching the unconstrained theoretical ceiling (8,200).

---

## 9. Power Sensitivity & Methodological Limitations

### 9.1 Mechanism-Level Null Distribution ($n=6$)
For $n=6$ independent mechanisms, total permutations under Wilcoxon signed rank is $2^6 = 64$:
- $W = 0$: two-sided $p = 2/64 = 0.03125$. (Rejection of $H_0$ requires DD to beat greedy on all 6 mechanisms without ties).
- $W = 1$: two-sided $p = 4/64 = 0.06250 > 0.05$.
- Ties: Any tie reduces $n$ to $m \le 5$, where minimum achievable two-sided p-value is $2/32 = 0.0625 > 0.05$ (meaning statistical significance at $\alpha=0.05$ is mathematically impossible).

### 9.2 Power Sensitivity Table ($N=12$ Paired Instances, 2,000 Monte Carlo Replications)

| Cohen $d$ | Mean Request Diff | Exact Permutation Power | MCSE | Asymptotic Power | Skewed Dist | Ties/Discrete | Stratified $n=6$ Power |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.40** | 4.8 req | 20.1% | 0.90% | 19.7% | 10.6% | 16.4% | **5.3%** |
| **0.50** | 6.0 req | 30.4% | 1.03% | 30.2% | 20.6% | 23.2% | **7.1%** |
| **0.60** | 7.2 req | 43.0% | 1.11% | 42.4% | 41.0% | 35.4% | **10.4%** |
| **0.70** | 8.4 req | 53.4% | 1.12% | 52.8% | 66.0% | 45.3% | **15.2%** |
| **0.80** | 9.6 req | 64.0% | 1.07% | 63.3% | 86.8% | 53.9% | **16.4%** |
| **1.00** | 12.0 req | 83.0% | 0.84% | 82.5% | 100.0% | 73.8% | **24.9%** |
| **1.25** | 15.0 req | 94.8% | 0.49% | 94.5% | 100.0% | 89.3% | **37.1%** |

### 9.3 Key Methodological Limitations
1. **Severe Power Constraints at Mechanism Level ($n=6$):** Statistical power at $n=6$ is bounded by $\le 37.1\%$ even for large effects ($d=1.25$). The benchmark cannot establish general superiority.
2. **Local Model Specificity:** Results on local `llama3.2:3b` reflect specific tool-calling behaviors and token probability distributions that may not generalize to larger commercial frontier models.
3. **Planted Prompt & Schema Triggers:** Planted synthetic distractors and formatting constraints provide clean experimental controls but may not capture complex multi-agent emergent conversational failures.

---

## 10. Preregistered Protocol Freeze Sign-Off

This protocol document and its companion configuration file `benchmarks/configs/phase_2b1_exploratory_config_v1_4.json` represent the final frozen specification for Phase 2B.1 exploratory benchmarking.

Execution remains strictly unauthorized until separate user authorization is granted.
