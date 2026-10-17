# AgentXRay — Phase 2B.1 Exploratory Benchmark Protocol v1.3

**Status:** PREREGISTERED_FROZEN  
**Study Designation:** EXPLORATORY_FEASIBILITY_BENCHMARK  
**Document Reference:** `docs/research/phase_2b1_exploratory_protocol_v1_3.md`  
**Configuration Reference:** `benchmarks/configs/phase_2b1_exploratory_config_v1_3.json`  
**Date:** 2026-10-17  
**Execution Tier:** LOCAL_MODEL (`llama3.2:3b` via Ollama)  
**Commercial Spend Cap:** Strictly $0.0000 USD (zero commercial API spend permitted)  

---

## 1. Executive Summary & Scientific Scope

This protocol establishes the frozen experimental specification for the Phase 2B.1 exploratory feasibility benchmark in Project AgentXRay.

The central research question is:
> Under what operational conditions can stochastic delta debugging isolate empirically supported 1-minimal failure-inducing configurations in black-box tool-using LLM agents, and how do its recovery rates and target-model request burdens compare with stochastic greedy ablation and seeded random search?

### 1.1 Scope Boundaries & Exploratory Nature
- **Exploratory Feasibility Benchmark:** Phase 2B.1 is an exploratory feasibility benchmark evaluated across six distinct underlying failure mechanisms ($n=6$) instantiated at two candidate-space sizes ($K=4$ and $K=8$), plus four null-control instances, yielding 16 instances total (12 active, 4 null).
- **No Confirmatory Superiority Claims:** With $n=6$ independent failure mechanisms, the study does not possess adequate statistical power to establish broad, generalizable algorithmic superiority.
- **Strict Execution Boundary:** This document serves as a frozen preregistration. **No target-model inference or experiment execution is authorized during the audit and freeze phase.** Execution requires separate, explicit authorization.

---

## 2. Research Endpoints & Statistical Analysis Plan

### 2.1 Demotion of Recovery Non-Inferiority to Descriptive Exploratory Reporting
Protocol v1.0 and v1.1 attempted to define a formal non-inferiority test for verified trigger recovery using a 15-percentage-point non-inferiority margin ($\delta = 0.15$). 

**Methodological Audit Verdict:**
A formal non-inferiority decision rule requires sufficient degrees of freedom to bound the paired risk difference confidence interval away from $-\delta$. With only $n=6$ independent failure mechanisms ($N=12$ paired task instances), the minimum non-zero difference is $1/6 \approx 16.7\%$. The 95% Newcombe/Wilson score confidence interval width for $n=6$ paired binary observations exceeds $\pm 35\%$. Consequently, even if stochastic delta debugging and greedy ablation achieve identical, perfect recovery (6/6 vs 6/6), the 95% lower bound cannot exclude $-0.15$.

Therefore, **formal non-inferiority hypothesis testing is rejected and demoted to descriptive exploratory reporting**:
1. **Paired Recovery Contingency Table:** Full $2 \times 2$ table of joint recovery outcomes (Both Recovered, DD Only, Greedy Only, Neither Recovered).
2. **Paired Risk Difference:** Point estimate $\hat{\Delta} = \hat{\pi}_{\text{DD}} - \hat{\pi}_{\text{greedy}}$ accompanied by its 95% Wilson/Newcombe score confidence interval, reported strictly for descriptive uncertainty.
3. **Exploratory Asymmetry Diagnostic:** McNemar's exact test (binomial distribution of discordant pairs, $p = 2 \times \min(P(B \le b), P(B \ge b))$ where $B \sim \text{Binomial}(b+c, 0.5)$) is reported as an exploratory symmetry diagnostic, with the explicit caveat that failure to reject symmetry does *not* constitute statistical evidence of equivalence or non-inferiority.

### 2.2 Primary Resource Burden Estimand
The primary efficiency comparison evaluates target-model request consumption between Stochastic Delta Debugging and Stochastic Greedy Ablation.

#### 2.2.1 Intention-to-Treat All-Assigned Tasks Endpoint
- **Unit of Observation:** Paired task instance $(i, K)$, where $i \in \{M_1, \dots, M_6\}$ and $K \in \{4, 8\}$, comprising $N=12$ paired active instances.
- **Quantity:** Total target-model requests $R_{\text{DD}}(i, K)$ and $R_{\text{Greedy}}(i, K)$ consumed across all phases: initial baseline checks, search trials, minimality verification trials, and retries.
- **Handling of Unsuccessful / Censored Runs:** Runs that fail to recover a supported trigger or hit the task request cap are retained in the primary analysis population at their actual expended requests (capped at 120 for $K=4$ or 180 for $K=8$). No arbitrary scalar penalties or imputed scores are added.

#### 2.2.2 Candidate-Space Stratification (Mandatory)
Because $K=4$ (cap 120, search space $2^4=16$) and $K=8$ (cap 180, search space $2^8=256$) possess fundamentally different combinatorial complexity and request ceilings, pooled raw request counts form an arbitrary mixture distribution. 

Therefore, resource burden must be reported:
1. **Stratified by Candidate Size:** Separate reporting for $K=4$ ($n=6$) and $K=8$ ($n=6$).
2. **Normalized Request Burden:** $\tilde{R}(i, K) = R(i, K) / \text{Cap}(K) \in [0, 1]$.
3. **Paired Request Difference:** $D(i, K) = R_{\text{Greedy}}(i, K) - R_{\text{DD}}(i, K)$ and normalized difference $\tilde{D}(i, K) = \tilde{R}_{\text{Greedy}}(i, K) - \tilde{R}_{\text{DD}}(i, K)$.

#### 2.2.3 Primary Statistical Test
- **Test:** Wilcoxon signed-rank test on paired request differences $D(i, K)$ across the $N=12$ assigned active instances.
- **Two-Sided Significance Level:** $\alpha = 0.05$.
- **Exact Evaluation:** Calculated using exact subset-sum dynamic programming permutation distribution ($2^{12} = 4,096$ permutations; exact critical boundary $W \le 13$ for $\alpha = 0.0425$).
- **Handling of Ties and Zeros:** Pratt filtering and tie mid-ranking.

#### 2.2.4 Secondary Subgroup Analysis (Jointly Resolved Only)
As a secondary descriptive endpoint, paired request differences are reported for the subset of tasks where *both* algorithms achieved verified recovery. This analysis is explicitly designated as conditional and subject to potential post-treatment conditioning (selection) bias.

---

## 3. Instance Architecture & Mechanism Registry

The benchmark consists of 16 frozen instances generated from 8 distinct failure mechanisms:
- **6 Active Mechanisms:** Each has exactly one $K=4$ instance and one $K=8$ instance sharing the identical underlying trigger, user prompt, and tool environment.
- **2 Null Mechanisms:** Negative controls with clean tool calling and zero planted triggers, evaluated at $K=4$ and $K=8$.

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

## 4. Target Harness Configuration & Cost Enforceability

### 4.1 Target Model Configuration
- **Model:** `llama3.2:3b` executed locally via Ollama daemon (`http://localhost:11434/api/generate`).
- **Inference Parameters:** `temperature=0.0`, `seed=42`, `top_p=1.0`.
- **System Constraints:** Target agent runs in a local controlled process.
- **Spend Enforcement:** Commercial API spend cap is strictly **$0.0000 USD**. No cloud API keys or commercial endpoints are configured or permitted.

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

### 6.1 Scientific Claims Taxonomy
To prevent conflation of observation with statistical inference, we distinguish five distinct claims:
1. **Candidate Failure Reproduction:** The isolated candidate $C$ reliably causes agent failure: $\hat{p}(C) \ge \theta_{\text{fail}} = 0.50$ and Wilson 95% lower bound $L_{95}(C) > \theta_{\text{base}} = 0.20$.
2. **Observed Monotonic Deletion Drop:** For every single-element deletion $e_j \in C$, removing $e_j$ strictly decreases the observed failure rate: $\hat{p}(C \setminus \{e_j\}) < \hat{p}(C)$ and $\hat{p}(C \setminus \{e_j\}) < \theta_{\text{fail}}$.
3. **Evidence Against Unacceptable Failure Probability:** For each deletion, testing $H_{0, j}: p(C \setminus \{e_j\}) \ge \theta_{\text{fail}}$ vs $H_{1, j}: p(C \setminus \{e_j\}) < \theta_{\text{fail}}$.
4. **Statistically Supported Drop Relative to Candidate:** For each deletion, testing $H_{0, j}: p(C \setminus \{e_j\}) \ge p(C)$ vs $H_{1, j}: p(C \setminus \{e_j\}) < p(C)$.
5. **Supported 1-Minimality:** Joint rejection of all component null hypotheses under an Intersection-Union Test.

### 6.2 Intersection-Union Testing Framework (Berger 1982)
To establish that candidate $C$ is 1-minimal, every single-element removal must fail to reproduce the failure.
The composite global null hypothesis is:
$$H_0: \bigcup_{j=1}^k \{ p(C \setminus \{e_j\}) \ge \theta_{\text{fail}} \}$$
The alternative hypothesis is:
$$H_1: \bigcap_{j=1}^k \{ p(C \setminus \{e_j\}) < \theta_{\text{fail}} \}$$

Under the Intersection-Union Principle (Berger 1982; Casella & Berger 2002):
- The global null $H_0$ is rejected at level $\alpha = 0.05$ if and only if **every individual component test** $H_{0, j}$ rejects at level $\alpha = 0.05$ (i.e., $\max_{j=1}^k p_j < \alpha$).
- Multiplicity corrections (such as Bonferroni or Benjamini-Hochberg) across single-element deletions within a candidate are mathematically unnecessary because the rejection region is the intersection of the component rejection regions:
  $$P_{H_0}\left(\bigcap_{j=1}^k \{T_j \in R_j\}\right) \le \min_{j} P_{H_{0, j}}(T_j \in R_j) \le \alpha.$$

### 6.3 Rejection of Disjunctive Heuristic Shortcuts
Protocol v1.2 introduced a rule permitting candidates to be labeled `SUPPORTED_1_MINIMAL` if $p_j < 0.05$ **or** observed ablated failure rate $\le 0.20$.

**Audit Verdict:** This shortcut is statistically invalid and is **strictly purged in Protocol v1.3**.
Under $N=5$ trials, an observed deletion failure rate of $0/5$ or $1/5$ yields Wilson 95% upper bounds of $43.4\%$ and $62.3\%$, respectively. An observed rate $\le 0.20$ does not prove the underlying failure probability is small. If finite-sample power at $N=5$ is insufficient to achieve $p < 0.05$, the candidate must be classified as `OBSERVED_1_MINIMAL`, never `SUPPORTED_1_MINIMAL`.

### 6.4 Operational Minimality Outcome Taxonomy
Every discovered candidate subset is classified into exactly one mutually exclusive outcome:
- `SUPPORTED_1_MINIMAL`:
  1. All $k$ deletions were evaluated (verification complete).
  2. Candidate reproduces failure: $\hat{p}(C) \ge 0.50$ and Wilson 95% lower bound $> 0.20$.
  3. Every single-element removal achieves formal statistical significance at $\alpha = 0.05$ ($p_j < 0.05$) without heuristic bypasses.
- `OBSERVED_1_MINIMAL`:
  1. All $k$ deletions were evaluated.
  2. Candidate reproduces failure: $\hat{p}(C) \ge 0.50$.
  3. Every single-element removal has lower observed failure rate: $\hat{p}(C \setminus \{e_j\}) < \hat{p}(C)$ and $\hat{p}(C \setminus \{e_j\}) < 0.50$.
  4. One or more deletion tests fail to reach statistical significance at $\alpha = 0.05$ due to small sample size.
- `NOT_MINIMAL`:
  Removal of at least one element failed to decrease the failure rate ($\hat{p}(C \setminus \{e_j\}) \ge \hat{p}(C)$) or ablated rate remained $\ge 0.50$.
- `INCONCLUSIVE`:
  Verification was interrupted (missing deletion evaluations), candidate failed to reproduce failure ($\hat{p}(C) < 0.50$), or insufficient observations.
- `BUDGET_EXHAUSTED`:
  Algorithm reached task or global request cap before completing search or verification.
- `INFRA_FAILURE`:
  Unrecoverable provider disconnect, daemon crash, or socket timeout.

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

### 9.1 Power Sensitivity Table ($N=12$ Paired Instances, 2,000 Monte Carlo Simulations)

| Cohen $d$ | Mean Request Diff | Exact Permutation Power | MCSE | Asymptotic Power | Skewed Dist | Ties/Discrete | Stratified $n=6$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.40** | 4.8 req | 20.1% | 0.90% | 19.7% | 10.6% | 16.4% | 5.3% |
| **0.50** | 6.0 req | 30.4% | 1.03% | 30.2% | 20.6% | 23.2% | 7.1% |
| **0.60** | 7.2 req | 43.0% | 1.11% | 42.4% | 41.0% | 35.4% | 10.4% |
| **0.70** | 8.4 req | 53.4% | 1.12% | 52.8% | 66.0% | 45.3% | 15.2% |
| **0.80** | 9.6 req | 64.0% | 1.07% | 63.3% | 86.8% | 53.9% | 16.4% |
| **1.00** | 12.0 req | 83.0% | 0.84% | 82.5% | 100.0% | 73.8% | 24.9% |
| **1.25** | 15.0 req | 94.8% | 0.49% | 94.5% | 100.0% | 89.3% | 37.1% |

### 9.2 Key Methodological Limitations
1. **Underpowered at Mechanism Level ($n=6$):** When evaluated at the true independent mechanism level ($n=6$), statistical power is only 16.4% at $d=0.80$ and 37.1% at $d=1.25$. The study cannot support confirmatory claims of general superiority.
2. **Local Model Specificity:** Results on local `llama3.2:3b` reflect specific tool-calling behaviors and token probability distributions that may not generalize to larger commercial frontier models.
3. **Planted Prompt & Schema Triggers:** Planted synthetic distractors and formatting constraints provide clean experimental controls but may not capture complex multi-agent emergent conversational failures.

---

## 10. Preregistered Protocol Freeze Sign-Off

This protocol document and its companion configuration file `benchmarks/configs/phase_2b1_exploratory_config_v1_3.json` represent the final frozen specification for Phase 2B.1 exploratory benchmarking. 

Execution remains strictly unauthorized until separate user authorization is granted.
