# AgentXRay — Phase 2B.1 Confirmatory Experiment Protocol (v1.0)

**Document Status:** Preregistered & Frozen Experimental Protocol (v1.0)  
**Date:** 2026-10-17  
**Status:** **DESIGN & PREREGISTRATION ONLY — NOT EXECUTED**  
**Execution Authorization:** **STRICTLY UNAUTHORIZED** (Requires separate formal authorization)  
**Target Execution Environment:** Local Apple Silicon Mac (Ollama daemon, `llama3.2:3b`, Q4_K_M)  
**Commercial Spend Authorization:** Strictly **$0.0000 USD**  
**Matching Configuration:** [`benchmarks/configs/phase_2b1_confirmatory_config.json`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/configs/phase_2b1_confirmatory_config.json)  

---

## 1. Executive Summary & Preregistration Statement

This document defines the formal, preregistered experimental protocol for Phase 2B.1 of the AgentXRay research project. The objective is to conduct a multi-instance confirmatory experiment evaluating whether stochastic delta debugging (`StochasticDeltaDebugMinimizer`) requires fewer target model inference requests than stochastic greedy ablation (`run_stochastic_greedy_ablation`) to discover an empirically supported 1-minimal failure-inducing configuration in local model-backed tool-using agents.

### Strict Scope and Governance Notice
- **No live model requests have been executed under this protocol.**
- **Phase 2B.1 execution is strictly unauthorized until this protocol has undergone independent methodological review.**
- **Commercial API spend remains strictly $0.0000 USD.**
- **All calibration evidence from Phase 2B.0.2 is preserved intact without alteration.**

---

## 2. Research Question, Hypotheses & Estimands

### 2.1 Primary Research Question
> Across independently specified, calibrated model-backed tool-using agent tasks, does stochastic delta debugging require fewer target model inference requests than stochastic greedy ablation to recover an empirically supported 1-minimal failure-inducing configuration, subject to prespecified failure-reproduction and false-discovery constraints?

### 2.2 Formal Hypotheses
1. **$H_1$ (Primary Superiority Hypothesis):** On calibrated active tool-using task instances ($M_{\text{active}} = 12$), the paired difference in target model inference requests:
   $$\Delta_i = R_{\text{Greedy}, i} - R_{\text{DD}, i}$$
   has a median strictly greater than zero ($\text{Median}(\Delta_i) > 0$). Delta debugging reduces inference requests relative to greedy ablation.
2. **$H_0$ (Primary Null Hypothesis):** $\text{Median}(\Delta_i) \le 0$. Delta debugging requires equal or more model inference requests than greedy ablation.
3. **$H_2$ (Minimality Non-Inferiority):** Stochastic delta debugging achieves an exact ground-truth recovery and 1-minimality rate not inferior to greedy ablation within a non-inferiority margin of $\delta_{\text{NI}} = 0.05$.
4. **$H_3$ (False Discovery Control on Null Controls):** On negative control tasks with no planted failure trigger ($M_{\text{null}} = 4$), both algorithms report `NO_TRIGGER_FOUND` with false discovery rate $\le \alpha_{\text{null}} = 0.05$.

### 2.3 Primary Endpoint & Estimand
- **Primary Endpoint:** Target model inference requests ($R_{\text{target}}$) consumed per experimental instance, defined as:
  $$R_{\text{target}} = R_{\text{search}} + R_{\text{verification}}$$
- **Primary Estimand:** The Hodges-Lehmann median difference estimator $\hat{\theta}_{\text{HL}}$ for paired differences $\Delta_i = R_{\text{Greedy}, i} - R_{\text{DD}, i}$, accompanied by non-parametric 95% confidence intervals and a two-sided paired Wilcoxon signed-rank test ($\alpha = 0.05$).

### 2.4 Handling of Censored, Failed, and Budget-Exhausted Runs
To prevent survivorship bias or selective exclusion of difficult instances:
1. **Budget Exhaustion:** If an algorithm reaches the request cap ($R_{\text{cap}} = 120$ for $K=4$; $R_{\text{cap}} = 180$ for $K=8$) without terminating, it is marked as `BUDGET_EXHAUSTED`. For the primary analysis, it is assigned the censored value $R_{\text{cap}}$ and penalized using worst-rank score assignment in the non-parametric Wilcoxon test.
2. **Unverified Candidate Discovery:** If an algorithm terminates with a candidate subset that fails independent verification ($\hat{p} < 0.50$ on holdout), it is scored as `UNVERIFIED_CANDIDATE` and assigned worst-rank score.
3. **Task Disqualification:** If a target task fails the pre-experimental calibration gate, it is disqualified *before* algorithms are run and does not enter the primary confirmatory dataset.
4. **Infrastructure Abort:** If local Ollama crashes or disconnects, the trial is halted and recorded as `INFRASTRUCTURE_ABORT`. It is re-run from the last checkpoint once the daemon recovers; if unrecoverable, it is reported as a harness defect.

### 2.5 Secondary Endpoints
1. **Exact Ground-Truth Recovery Rate:** Proportion of active instances where the discovered subset matches the independently verified planted trigger ($\hat{C} = C^*$).
2. **Empirical 1-Minimality Rate:** Proportion of discovered subsets satisfying the statistical 1-minimality criterion on single-element ablations.
3. **Holdout Failure Reproduction Rate:** Failure rate of the discovered subset on an independent $N=5$ holdout sample.
4. **False Discovery Rate on Null Instances:** Proportion of null tasks where an algorithm falsely claims a trigger.
5. **Partial Recovery (Jaccard Index):** $J(\hat{C}, C^*) = \frac{|\hat{C} \cap C^*|}{|\hat{C} \cup C^*|}$.
6. **Total Complete Agent Executions:** Number of full end-to-end multi-turn agent sessions.
7. **Wall-Clock Latency:** Total seconds consumed during search and verification.
8. **Sub-group Stratification:** Request reduction stratified by candidate size ($K=4$ vs $K=8$) and task family.

### 2.6 Conceptual Distinctions in Minimality Taxonomy
- **Reference Minimal Subset ($C^*$):** The prespecified planted trigger elements verified during benchmark construction.
- **Empirically Failure-Inducing Subset ($C_{\text{fail}}$):** Any subset $S \subseteq C$ demonstrating an observable failure rate $\hat{p}(S) \ge \theta_{\text{fail}}$ in the target agent.
- **Failure-Reproducing Subset:** A subset that consistently produces failure on a separate holdout sample.
- **Empirically 1-Minimal Subset ($\hat{C}_{\text{1-min}}$):** A subset $S$ that reproduces failure, where *every* immediate single-element deletion $S \setminus \{x\}$ satisfies $\hat{p}(S \setminus \{x\}) < \theta_{\text{fail}}$ under statistical testing.

These terms are not interchangeable. Distractor interactions or model idiosyncrasies may yield an empirical 1-minimal subset that does not match the reference ground truth. Both metrics are reported separately.

---

## 3. Experimental Units & Task Instance Specifications

### 3.1 Experimental Unit
The experimental unit is an **independent calibrated task instance** $i \in \{1, \dots, 16\}$, characterized by:
- A distinct user prompt, operational context, and tool environment.
- A uniquely defined initial candidate intervention space $C_i$ with $|C_i| = K_i$.
- An independent evaluator verifying observable runtime behavior.

Repeated trials within a task or repeated seeds of the same task are **not** independent experimental units.

### 3.2 Candidate Space Size Feasibility ($K \in \{4, 8, 16\}$)
- **$K = 4$ ($2^4 = 16$ subsets):** Mathematically tractable for full power set characterization. Feasible for live local models ($\sim 160$ requests for characterization, $\sim 40\text{--}60$ requests for search). Admitted for confirmatory benchmarking.
- **$K = 8$ ($2^8 = 256$ subsets):** Full power-set characterization would consume $256 \times 5 \times 2 = 2560$ requests per instance ($\sim 30$ hours across instances), which is excessive for local pilot hardware. We adopt a **budgeted boundary and singleton characterization protocol** ($2K + 2 = 18$ subsets = 180 requests). Search request caps are set to 180 requests. Admitted for confirmatory benchmarking.
- **$K = 16$ ($2^{16} = 65,536$ subsets):** Characterization is mathematically intractable. Furthermore, Greedy ablation on $K=16$ requires at least $16 + 15 + \dots > 100$ trials ($> 200$ requests), exceeding local request caps on `llama3.2:3b`. **$K=16$ is deferred to future exploratory scaling and excluded from this confirmatory protocol.**

### 3.3 Task Families & Instance Definitions
The benchmark contains 16 instances across 4 families, registered in [`benchmarks/model_pilot/confirmatory_tasks.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/confirmatory_tasks.py):

| Instance ID | Family | $K$ | Planted Trigger Summary | Null? | Required Tool | Observable Pass Criterion | Expected Failure Class |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- | :--- |
| `inst_01_syntax_currency_k4` | `TOOL_ARG_SYNTAX` | 4 | Currency `$` symbols in calculator arguments | No | `calculator` | Response contains `"400"` | `FAIL_TOOL_EXEC` |
| `inst_02_syntax_currency_k8` | `TOOL_ARG_SYNTAX` | 8 | Currency `$` symbols in calculator arguments | No | `calculator` | Response contains `"400"` | `FAIL_TOOL_EXEC` |
| `inst_03_syntax_comma_k4` | `TOOL_ARG_SYNTAX` | 4 | Comma thousands separators in expressions | No | `calculator` | Response contains `"4000"` | `FAIL_TOOL_EXEC` |
| `inst_04_syntax_comma_k8` | `TOOL_ARG_SYNTAX` | 8 | Comma thousands separators in expressions | No | `calculator` | Response contains `"4000"` | `FAIL_TOOL_EXEC` |
| `inst_05_routing_override_k4` | `TOOL_ROUTING_CONFLICT` | 4 | Force field `non_existent_secret_key` | No | `database_lookup` | Response contains `"finance"` | `FAIL_TOOL_EXEC` |
| `inst_06_routing_override_k8` | `TOOL_ROUTING_CONFLICT` | 8 | Force field `non_existent_secret_key` | No | `database_lookup` | Response contains `"finance"` | `FAIL_TOOL_EXEC` |
| `inst_07_routing_prohibition_k4` | `TOOL_ROUTING_CONFLICT` | 4 | Prohibit calling external tools | No | `calculator` | Required tool invoked | `FAIL_TOOL_EXEC` |
| `inst_08_routing_prohibition_k8` | `TOOL_ROUTING_CONFLICT` | 8 | Prohibit calling external tools | No | `calculator` | Required tool invoked | `FAIL_TOOL_EXEC` |
| `inst_09_schema_unit_k4` | `TOOL_PARAM_SCHEMA` | 4 | Target unit `light_years` | No | `unit_converter` | Response contains `"3.1069"` | `FAIL_TOOL_EXEC` |
| `inst_10_schema_unit_k8` | `TOOL_PARAM_SCHEMA` | 8 | Target unit `light_years` | No | `unit_converter` | Response contains `"3.1069"` | `FAIL_TOOL_EXEC` |
| `inst_11_schema_cross_unit_k4` | `TOOL_PARAM_SCHEMA` | 4 | Mass to length unit conversion | No | `unit_converter` | Response contains `"22.0462"` | `FAIL_TOOL_EXEC` |
| `inst_12_schema_cross_unit_k8` | `TOOL_PARAM_SCHEMA` | 8 | Mass to length unit conversion | No | `unit_converter` | Response contains `"22.0462"` | `FAIL_TOOL_EXEC` |
| `inst_13_null_calc_k4` | `NULL_CONTROL` | 4 | None (4 benign distractors) | **Yes** | `calculator` | Response contains `"400"` | `CLEAN_PASS` |
| `inst_14_null_calc_k8` | `NULL_CONTROL` | 8 | None (8 benign distractors) | **Yes** | `calculator` | Response contains `"400"` | `CLEAN_PASS` |
| `inst_15_null_db_k4` | `NULL_CONTROL` | 4 | None (4 benign distractors) | **Yes** | `database_lookup` | Response contains `"finance"` | `CLEAN_PASS` |
| `inst_16_null_db_k8` | `NULL_CONTROL` | 8 | None (8 benign distractors) | **Yes** | `database_lookup` | Response contains `"finance"` | `CLEAN_PASS` |

---

## 4. Calibration & Task Qualification Gates

Before any instance is admitted to the confirmatory algorithm benchmark, it must independently pass the following empirical qualification gates under $N=5$ trials per configuration:

### 4.1 Active Task Qualification Gate ($M_{\text{active}} = 12$)
1. **Clean Baseline Pass:** The empty baseline configuration $\emptyset$ must demonstrate failure rate:
   $$\hat{p}(\emptyset) \le 0.10$$
2. **Trigger-Present Separation:** Every configuration containing the planted trigger $T_1$ evaluated during characterization must demonstrate:
   $$\hat{p}(S \cup \{T_1\}) \ge 0.70$$
3. **Trigger-Absent Non-Interference:** Every configuration lacking $T_1$ must demonstrate:
   $$\hat{p}(S \setminus \{T_1\}) \le 0.20$$
4. **Failure Class Specificity:** Trigger failures must match the prespecified failure classification (`FAIL_TOOL_EXEC`). Subsets where tools execute cleanly must not be flagged as tool execution failures.

### 4.2 Null Task Qualification Gate ($M_{\text{null}} = 4$)
1. **Unablated Clean Pass:** The full candidate set $C_{\text{null}}$ must pass with:
   $$\hat{p}(C_{\text{null}}) \le 0.10$$
2. **Empty Baseline Clean Pass:** $\hat{p}(\emptyset) \le 0.10$.

### 4.3 Qualification Disqualification Rule
If an active instance fails any of criteria 1–4, it is disqualified immediately. It is marked `DISQUALIFIED_CALIBRATION_GATE` and excluded from the primary confirmatory dataset. It is **never** tuned or ablated post-hoc within the same task identity.

---

## 5. Statistical Power & Feasibility Analysis

### 5.1 Power Calculation Formulation
The primary comparison tests for a paired difference in target model inference requests:
$$\Delta_i = R_{\text{Greedy}, i} - R_{\text{DD}, i}$$
using a two-sided paired Wilcoxon signed-rank test at $\alpha = 0.05$ with target power $1 - \beta \ge 0.80$.

Under the asymptotic relative efficiency of the Wilcoxon signed-rank test compared to the paired Student's t-test ($\text{ARE} \ge \frac{3}{\pi} \approx 0.955$), the sample size requirement is bounded by $n_{\text{wilcoxon}} \approx \lceil n_t / 0.955 \rceil$.

### 5.2 Empirical Effect Size from Local Model Pilot
In the Phase 2B.0.2 local neural calibration pilot (`llama3.2:3b`, $K=4$):
- $R_{\text{Greedy}} = 60$ requests
- $R_{\text{DD}} = 40$ requests
- Observed paired reduction: $\Delta = +20$ requests (33.3% request reduction).
- Conservative planned mean difference: $\mu_\Delta = 15.0$ requests.
- Conservative planned standard deviation: $\sigma_\Delta = 12.0$ requests.
- Standardized effect size:
  $$\text{Cohen's } d = \frac{\mu_\Delta}{\sigma_\Delta} = \frac{15.0}{12.0} = 1.25$$

### 5.3 Sensitivity Analysis Table

| Assumed Effect Size (Cohen's $d$) | Substantive Meaning | Required $n_t$ (Paired t) | Required $n_{\text{wilcoxon}}$ (Signed-Rank) | Power with $n=12$ Active Instances |
| :---: | :---: | :---: | :---: | :---: |
| $d = 0.50$ | Moderate request reduction ($\sim 15\%$) | 34 | 36 | 36.8% |
| $d = 0.60$ | Moderate-to-large reduction | 24 | 26 | 50.4% |
| $d = 0.70$ | Large reduction | 18 | 19 | 63.8% |
| $d = 0.80$ | Standard large effect ($>20\%$ reduction) | 15 | 16 | 74.8% |
| $d = 0.90$ | Substantial reduction | 12 | 13 | 83.6% |
| $d = 1.00$ | Very substantial reduction | 10 | 11 | 90.0% |
| **$d = 1.25$ (Pilot Observed)** | **Phase 2B.0.2 Empirical Pilot Effect** | **7** | **8** | **97.8%** |
| $d = 1.50$ | Massive reduction ($>40\%$) | 6 | 7 | 99.6% |

**Power Conclusion:** With $M_{\text{active}} = 12$ independent active task instances, the experiment achieves:
- **$> 80\%$ statistical power** for any standardized effect size $d \ge 0.86$.
- **$> 97\%$ statistical power** at the pilot-observed effect size ($d = 1.25$).
- Adequate statistical power to detect a clinically and practically meaningful reduction of $\ge 12$ model inference requests per task instance.

### 5.4 Aggregate Request Budget & Runtime Feasibility

| Stage | Instances | Subsets per Instance | Trials per Subset | Requests per Trial | Total Model Requests |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Active Calibration ($K=4$)** | 6 | 16 | 5 | 2 | 960 |
| **Active Calibration ($K=8$)** | 6 | 18 | 5 | 2 | 1,080 |
| **Null Calibration ($K=4, 8$)** | 4 | 2 | 5 | 2 | 80 |
| **Search: Stochastic DD** | 12 active | Variable (avg 5.5) | 5 | 2 | 660 |
| **Search: Stochastic Greedy** | 12 active | Variable (avg 9.0) | 5 | 2 | 1,080 |
| **Search: Stochastic Random** | 12 active | Variable (avg 6.5) | 5 | 2 | 780 |
| **Null Search (All Algos)** | 4 null | Variable (early exit) | 5 | 2 | 240 |
| **Holdout Confirmation** | 12 active | 3 algorithms | 5 | 2 | 360 |
| **Total Planned Budget** | **16 instances** | — | — | — | **5,240 requests** |

- **Hardware Feasibility:** On Apple Silicon Mac (M-series, 24 GB RAM) running local Ollama `llama3.2:3b`, median latency is $\sim 0.45\text{ s}$ per request.
  $$\text{Total Wall-Clock Time} \approx 5,240 \times 0.45\text{ s} \approx 2,358\text{ seconds} \approx \mathbf{39.3\text{ minutes}}$$
- **Commercial Spend:** Strictly **$0.0000 USD** (100% local loopback inference).
- **Safety Request Cap:** Hard stop set at **6,000 total requests** in configuration.

---

## 6. Algorithm Execution & Fair Comparison Standards

### 6.1 Competing Algorithms
1. **Primary Comparison:**
   - **Stochastic Delta Debugging (`StochasticDeltaDebugMinimizer`):** Recursive partition-based search over candidate intervention subsets.
   - **Stochastic Greedy Ablation (`run_stochastic_greedy_ablation`):** Sequential backward elimination of individual candidate elements.
2. **Secondary Baseline:**
   - **Stochastic Random Search (`run_stochastic_random_search`):** Uniform random candidate sampling (Seed 42).

### 6.2 Parity Guarantees
- **Equivalent Initial State:** All three algorithms receive the identical unablated `InputCase(elements=C_i)`.
- **Identical Evaluator:** All algorithms probe the identical `ModelBackedAgentTarget` wrapped with `CostTracker`.
- **Request Caps:** Fixed at $R_{\text{cap}} = 120$ ($K=4$) and $R_{\text{cap}} = 180$ ($K=8$).
- **Statistical Decision Parameters:** Fixed across all algorithms:
  - Failure threshold: $\theta_{\text{fail}} = 0.50$
  - Confidence level: $1 - \alpha = 0.95$ (Wilson score interval)
  - Trials per candidate: $N = 5$
- **Zero Search Leakage:** Algorithms observe only binary `TargetResult(outcome=SUCCESS/FAILURE)` and telemetry. Neither algorithm has access to reference ground truth, true trigger elements, or the other algorithms' logs.

---

## 7. Statistical Ground Truth & 1-Minimality Criteria

### 7.1 Statistical 1-Minimality Definition
A discovered subset $\hat{C}$ is declared **statistically supported 1-minimal** if and only if:
1. **Self-Reproduction:** Under $N=5$ trials, $\hat{C}$ achieves an observed failure rate $\hat{p}(\hat{C}) \ge 0.50$ with Wilson lower bound $L_{95}(\hat{C}) > 0.20$.
2. **Ablation Reversion:** For *every* single-element deletion $c \in \hat{C}$, the ablated subset $\hat{C} \setminus \{c\}$ achieves an observed failure rate $\hat{p}(\hat{C} \setminus \{c\}) \le 0.20$ with Wilson upper bound $U_{95}(\hat{C} \setminus \{c\}) < 0.70$.
3. **Multiplicity Control:** Single-element deletion tests are evaluated using the Holm-Bonferroni step-down procedure at family-wise error rate $\alpha_{\text{FWER}} = 0.05$.
4. **Independent Holdout Confirmation:** The discovered subset $\hat{C}$ is evaluated on a completely fresh, independent holdout trial sample ($N=5$) that was not observed during search.

---

## 8. Statistical Analysis Plan (SAP)

### 8.1 Primary Hypothesis Test
- **Test:** Paired two-sided Wilcoxon signed-rank test evaluating the paired difference $\Delta_i = R_{\text{Greedy}, i} - R_{\text{DD}, i}$ across active instances $i \in \{1, \dots, 12\}$.
- **Significance Level:** $\alpha = 0.05$.
- **Effect Estimate:** Hodges-Lehmann median paired difference $\hat{\theta}_{\text{HL}}$ with exact 95% confidence intervals.
- **Ties:** Handled via Pratt's method (zero differences retained in rank assignment).

### 8.2 Secondary Hypothesis Multiplicity Adjustments
Secondary endpoint comparisons (exact recovery, 1-minimality, holdout rate, Jaccard similarity) will be adjusted for family-wise error rate using the **Holm-Bonferroni method** across the 4 task families.

### 8.3 Sensitivity & Robustness Analyses
1. **Parametric Paired t-test:** Re-run comparison under Student's t-test with Welch correction.
2. **Permutation Test:** Exact paired permutation test ($2^{12} = 4,096$ permutations) on $\Delta_i$.
3. **Relative Percentage Reduction:** Test the relative reduction metric:
   $$\text{PR}_i = \frac{R_{\text{Greedy}, i} - R_{\text{DD}, i}}{R_{\text{Greedy}, i}} \times 100\%$$
4. **Alternative Failure Caps:** Evaluate sensitivity by scoring budget-exhausted runs at $1.5 \times R_{\text{cap}}$ and $2.0 \times R_{\text{cap}}$.

---

## 9. Failure Classification Taxonomy

Every target execution will be assigned a mutually exclusive failure classification:
- `CLEAN_PASS`: Tool executed cleanly, output returned expected result, final text satisfied criteria.
- `FAIL_TOOL_EXEC`: Tool was not called, or called with malformed arguments causing execution failure (the primary target of $T_1$).
- `FAIL_BEHAVIOR`: Tool executed cleanly, but final response omitted required content or included forbidden phrases.
- `FAIL_INFRA`: Provider network timeout, daemon disconnect, or budget exhaustion.

Provider infrastructure crashes (`FAIL_INFRA`) are strictly excluded from behavioral failure counts.

---

## 10. Known Scientific Limitations & Falsification Conditions

### 10.1 Falsification Conditions
The research hypothesis that stochastic delta debugging provides an efficiency advantage over greedy ablation on local model-backed tool-using agents will be **falsified** if:
1. The primary Wilcoxon signed-rank test yields $p \ge 0.05$.
2. The Hodges-Lehmann median paired difference is $\le 0$ requests.
3. Delta debugging achieves an exact ground-truth recovery rate inferior to greedy ablation by more than $\delta_{\text{NI}} = 0.05$.

### 10.2 Known Limitations
1. **Single Model Architecture:** The confirmatory protocol is calibrated on `llama3.2:3b`. Findings cannot be extrapolated to other model families (e.g. Mistral, Qwen) without separate calibration.
2. **Quantized Local Runtime:** The runtime uses 4-bit quantization (Q4_K_M). Stochasticity reflects quantized sampling.
3. **Small-to-Medium Candidate Spaces ($K \le 8$):** The protocol evaluates $K \in \{4, 8\}$. Algorithmic scaling behavior for $K \ge 16$ remains open.

---

## 11. Preregistration Freeze & Cryptographic Verification

- **Document Version:** v1.0
- **Configuration File:** `benchmarks/configs/phase_2b1_confirmatory_config.json`
- **Validation Command:** `./.venv/bin/python benchmarks/model_pilot/validate_confirmatory_protocol.py`
- **Configuration SHA-256:** Computed upon commit.
- **Protocol SHA-256:** Computed upon commit.

> [!CAUTION]
> **PHASE 2B.1 EXPERIMENT EXECUTION REMAINS STRICTLY UNAUTHORIZED.**
> Live model inference under this protocol is prohibited until this document and its matching configuration have been reviewed.
