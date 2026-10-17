# AgentXRay — Phase 2B.1 Protocol Audit & Preregistration Repair Report (v1.0)

**Document Type:** Independent Methodological and Implementation Audit  
**Date:** 2026-10-17  
**Author / Auditor:** Antigravity Research Engineering & Statistics Team  
**Subject Protocol:** `docs/research/phase_2b1_confirmatory_protocol_v1.md`  
**Subject Configuration:** `benchmarks/configs/phase_2b1_confirmatory_config.json`  
**Audit Codebase Checkpoint:** `62579fb` on branch `main`  
**Execution Environment:** Local Apple Silicon Mac (Ollama daemon, `llama3.2:3b`, Q4_K_M)  
**Commercial Spend Audit:** Strictly **$0.0000 USD**  
**Audit Verdict:** **`PROTOCOL_REVISED_EXPLORATORY_FEASIBILITY`** (Protocol v1.0 contains confirmed structural defects; repaired in Protocol v1.1; Phase 2B.1 execution remains strictly unauthorized pending formal independent review)

---

## 1. Executive Summary & Audit Verdict

This document presents the independent, rigorous methodological, mathematical, and code-level audit of the proposed Phase 2B.1 confirmatory experiment protocol (`phase_2b1_confirmatory_protocol_v1.md`) and configuration (`phase_2b1_confirmatory_config.json`).

The primary objective of Phase 2B.1 is to compare the search efficiency (measured in target model inference requests) of stochastic delta debugging (`StochasticDeltaDebugMinimizer`) against stochastic greedy ablation (`run_stochastic_greedy_ablation`) on local model-backed tool-using agents running quantized Ollama `llama3.2:3b`.

### 1.1 Summary of Audit Findings
Our audit established that while Protocol v1.0 represents a significant advance in experimental hygiene over previous phases (enforcing zero commercial spend, unablated baselines, holdout confirmation, and zero ground-truth leakage), it suffers from **five confirmed methodological and statistical defects** and **two potential/observational limitations** that prevent it from serving as a definitive, confirmatory proof of universal algorithmic superiority:

1. **Pseudoreplication in Task Instance Registry (`CONFIRMED_DEFECT`):** The 12 active instances in [`benchmarks/model_pilot/confirmatory_tasks.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/confirmatory_tasks.py) are not 12 independent random task specifications; they represent exactly 6 base agent tasks evaluated at two candidate intervention space sizes ($K=4$ and $K=8$). Pooling them as 12 independent exchangeable observations in a single Wilcoxon signed-rank test violates statistical independence assumptions.
2. **Null Control False Discovery Unidentifiability (`CONFIRMED_DEFECT`):** Protocol v1.0 claims that evaluating $M_{\text{null}} = 4$ negative control instances establishes that the false discovery rate is bounded by $\le 0.05$. Mathematically, observing $0$ false discoveries out of $4$ trials ($0/4$) yields an exact Clopper-Pearson 95% confidence interval upper bound of **$52.7\%$** (and a Wilson score upper bound of **$49.0\%$**). Demonstrating an empirical upper bound $\le 0.05$ with $95\%$ confidence requires at least $N = 59$ negative control tasks. The null tasks must be downgraded from a quantitative proof of FDR control to a qualitative diagnostic sanity check.
3. **Power Analysis Overconfidence from $n=1$ Pilot Data (`CONFIRMED_DEFECT`):** Protocol v1.0 assumed a standardized effect size of $\text{Cohen's } d = 1.25$ and $\sigma_\Delta = 12.0$ requests, claiming $97.8\%$ statistical power at $n=12$. This assumption was extrapolated from a single $n=1$ pilot trial. Sensitivity modeling demonstrates that for realistic moderate-to-large effect sizes ($d = 0.50\text{--}0.60$), statistical power with $n=12$ is only **$29.3\%\text{--}40.7\%$**. When stratified by candidate size ($n=6$), power collapses to **$8.1\%\text{--}10.5\%$**.
4. **Multiple Testing Procedure Mismatch in 1-Minimality (`CONFIRMED_DEFECT`):** Protocol v1.0 specified that candidate single-element ablations are evaluated using the Holm-Bonferroni step-down procedure to control family-wise error rate ($\alpha_{\text{FWER}} = 0.05$). However, the underlying codebase implementation in [`src/agentxray/stochastic_minimizer.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/src/agentxray/stochastic_minimizer.py#L370-L388) actually executes the Benjamini-Hochberg procedure for false discovery rate (FDR) control.
5. **Worst-Case Budget Cap Overrun (`POTENTIAL_DEFECT`):** Protocol v1.0 set a hard global abort cap of $6,000$ total model requests. In worst-case search scenarios where greedy ablation and random search exhaust their caps on $K=8$ tasks, total requests reach **$7,330$**, triggering an unplanned global abort and truncating data collection.
6. **Primary Estimand Distortion via Arbitrary Worst-Rank Assignment (`CONFIRMED_DEFECT`):** Assigning an arbitrary worst-rank score to budget-exhausted runs distorts Wilcoxon signed ranks and conflates failure to terminate with excessive search effort. A two-part hurdle/survival formulation is required.
7. **Unobserved Distractor Space in $K=8$ Tasks (`POTENTIAL_DEFECT`):** The 18-subset boundary/singleton characterization for $K=8$ tests only $18$ of $256$ subsets, leaving $93\%$ of the power set uncharacterized. Distractor interactions involving 2 or 3 elements remain undetectable before minimization.

### 1.2 Audit Verdict: PROTOCOL_REVISED_EXPLORATORY_FEASIBILITY
We have issued a formal verdict of **`PROTOCOL_REVISED_EXPLORATORY_FEASIBILITY`**. 
- Protocol v1.0 is **rejected** for confirmatory claims of general algorithmic superiority.
- Protocol v1.1 has been drafted and validated as an **Exploratory Multi-Instance Feasibility Protocol**.
- Phase 2B.1 execution remains **STRICTLY UNAUTHORIZED**.
- Commercial spend remains strictly **$0.0000 USD**.

---

## 2. Defect Registry & Forensic Findings

The table below catalogs all findings identified during our independent audit, classified according to the prescribed severity rubric:

| Finding ID | Classification | Subsystem / Area | Impacted Protocol / Config Section | Summary Description | Remediation Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F-01** | `CONFIRMED_DEFECT` | Task Registry / Sampling | Protocol v1 §3.1, §3.3; Config | **Pseudoreplication in Instance Registry:** 12 active instances represent 6 paired base tasks at $K=4$ and $K=8$, not 12 independent tasks. | Repaired in Protocol v1.1 via stratified and clustered analysis. |
| **F-02** | `CONFIRMED_DEFECT` | Statistical Identifiability | Protocol v1 §2.2 ($H_3$), §4.2, §5.3 | **Null Control Identifiability:** $N=4$ null tasks cannot establish $\text{FDR} \le 0.05$ (95% CI upper bound is $49.0\%\text{--}52.7\%$). | Downgraded to qualitative diagnostic check; $H_3$ re-scoped. |
| **F-03** | `CONFIRMED_DEFECT` | Statistical Power | Protocol v1 §5.1, §5.2, §5.3 | **Overconfident Power Extrapolation:** $d=1.25$ derived from $n=1$ pilot; realistic power for $d=0.50$ is $<30\%$. | Complete power sensitivity curves published; study re-scoped to exploratory. |
| **F-04** | `CONFIRMED_DEFECT` | Implementation Alignment | Protocol v1 §7.1 (Criterion 3); `stochastic_minimizer.py` | **Multiple Testing Procedure Mismatch:** Protocol specified Holm-Bonferroni; code implements Benjamini-Hochberg FDR. | Protocol v1.1 aligned with implementation (Benjamini-Hochberg). |
| **F-05** | `CONFIRMED_DEFECT` | Primary Estimand | Protocol v1 §2.4, §8.1; Config | **Worst-Rank Distortion in Wilcoxon Test:** Arbitrary ranking of censored/failed runs distorts Wilcoxon signed-rank p-values. | Replaced by Two-Part Hurdle analysis (McNemar + Signed-Rank) & Survival analysis. |
| **F-06** | `POTENTIAL_DEFECT` | Budget Architecture | Protocol v1 §5.4; Config `budget_rules` | **Worst-Case Budget Headroom Deficit:** Worst-case demand ($7,330$ requests) breaches $6,000$ cap. | Increased global cap to $7,500$ with explicit per-algorithm caps. |
| **F-07** | `POTENTIAL_DEFECT` | Characterization Protocol | Protocol v1 §3.2 ($K=8$); `confirmatory_tasks.py` | **Incomplete Power-Set Exploration at $K=8$:** 18 subsets leaves $93\%$ of combinations unobserved. | Explicitly documented as an unobserved-interaction risk in v1.1. |
| **F-08** | `NO_DEFECT_FOUND` | Leakage & Harness Isolation | `confirmatory_tasks.py`, `test_model_harness.py` | **Zero Ground-Truth Leakage Verified:** Evaluator and agent target possess zero awareness of reference trigger. | Verified clean across all 16 registry specifications. |
| **F-09** | `NO_DEFECT_FOUND` | Financial Governance | `cost_tracker.py`, Config | **Strict Zero-Spend Enforcement:** Commercial spend hard cap enforced at $\$0.0000$ with loopback Ollama provider. | Verified clean across all provider pathways. |

---

## 3. Deep-Dive Forensic Analyses & Mathematical Proofs

### 3.1 Finding F-01: Pseudoreplication and Clustered Task Structure (`CONFIRMED_DEFECT`)

#### Root Cause Analysis
In [`benchmarks/model_pilot/confirmatory_tasks.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/confirmatory_tasks.py), the 16 registered tasks are constructed as follows:
- 6 base task specifications evaluated at candidate space size $K=4$.
- The **exact same 6 base task specifications** evaluated at candidate space size $K=8$ by padding the candidate space with 4 additional benign distractors.
- 4 null control tasks (2 base prompts evaluated at $K=4$ and $K=8$).

#### Mathematical and Experimental Defect
Protocol v1.0 treats all 12 active instances as independent experimental units ($i \in \{1, \dots, 12\}$) in a single Wilcoxon signed-rank test. However, instance `inst_01_syntax_currency_k4` and instance `inst_02_syntax_currency_k8`:
1. Share the exact same underlying user prompt (`"Calculate the total cost: $100 + $300"`).
2. Invoke the exact same system tool (`calculator`).
3. Contain the exact same planted failure trigger element (`"1. Inject currency symbol '$' into numeric operands"`).
4. Rely on the exact same evaluator criterion (`"400"` in response text).

This represents classic **pseudoreplication** (Hurlbert, 1984). The pairs $(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)$ are correlated within task mechanism. An unmodeled prompt idiosyncrasy or model routing quirk that favors delta debugging on `inst_01` will almost certainly favor it on `inst_02`. 

#### Remediation in Protocol v1.1
1. **Primary Inference Stratification:** The primary Wilcoxon signed-rank test must be stratified by candidate space size:
   - Cohort A ($K=4$, $n=6$ base tasks).
   - Cohort B ($K=8$, $n=6$ base tasks).
2. **Cluster-Adjusted Aggregation:** When pooling across $K$, the paired differences must be averaged within base task:
   $$\bar{\Delta}_j = \frac{\Delta_{j, K=4} + \Delta_{j, K=8}}{2}, \quad j \in \{1, \dots, 6\}$$
   yielding $n=6$ genuinely independent mechanism-level units.

---

### 3.2 Finding F-02: Null Control False Discovery Identifiability (`CONFIRMED_DEFECT`)

#### Mathematical Proof of Non-Identifiability
Protocol v1.0 Hypothesis $H_3$ states:
> *"On negative control tasks with no planted failure trigger ($M_{\text{null}} = 4$), both algorithms report `NO_TRIGGER_FOUND` with false discovery rate $\le \alpha_{\text{null}} = 0.05$."*

Let $X$ be the number of false discoveries observed across $N = 4$ independent negative control tasks, where each task has an unknown false positive probability $p_{\text{FP}}$. Under the null hypothesis that the minimizer produces no false triggers, we expect $X = 0$.

Suppose an empirical evaluation observes $X = 0$ out of $N = 4$.

1. **Wilson Score 95% Confidence Interval:**
   The Wilson score interval for $X = 0$ and $N = 4$ at confidence level $1 - \alpha = 0.95$ ($z_{0.975} = 1.95996$):
   $$\hat{p} = 0$$
   $$\tilde{n} = N + z^2 = 4 + 1.95996^2 = 4 + 3.8415 = 7.8415$$
   $$\tilde{p} = \frac{X + \frac{1}{2}z^2}{\tilde{n}} = \frac{0 + 1.9207}{7.8415} = 0.2449$$
   $$\text{CI}_{\text{upper}} = \frac{0 + \frac{1}{2}(3.8415) + 1.95996 \sqrt{\frac{0(4)}{4} + \frac{3.8415}{4}}}{4 + 3.8415} = \frac{1.92075 + 1.95996(0.97999)}{7.8415} = \frac{3.84146}{7.8415} = \mathbf{0.4899}$$

2. **Clopper-Pearson Exact 95% Confidence Interval:**
   The exact Clopper-Pearson upper bound $p_U$ solves:
   $$(1 - p_U)^N = \frac{\alpha}{2} = 0.025$$
   $$1 - p_U = (0.025)^{1/4} = 0.3976$$
   $$p_U = 1 - 0.3976 = \mathbf{0.6024} \quad (\text{two-sided } 95\% \text{ CI})$$
   For a one-sided $95\%$ upper confidence bound:
   $$(1 - p_U)^4 = 0.05 \implies 1 - p_U = (0.05)^{1/4} = 0.4729 \implies p_U = \mathbf{0.5271}$$

#### Conclusion & Minimum Required Sample Size
Observing $0 / 4$ false positives is statistically compatible with an underlying false discovery rate as high as **$49.0\%\text{--}52.7\%$**. It is mathematically impossible for $N=4$ tasks to prove that $\text{FDR} \le 0.05$.

To achieve a one-sided exact 95% upper bound $p_U \le 0.05$ when observing zero events, the required sample size $N$ is:
$$(1 - 0.05)^N \le 0.05 \implies N \ln(0.95) \le \ln(0.05) \implies N \ge \frac{\ln(0.05)}{\ln(0.95)} = \frac{-2.9957}{-0.05129} = \mathbf{58.4} \implies \mathbf{N \ge 59 \text{ null tasks}}$$

#### Remediation in Protocol v1.1
Hypothesis $H_3$ is re-scoped. The 4 null tasks are classified as **Qualitative Diagnostic Negative Controls**. The protocol explicitly disclaims establishing an asymptotic FDR bound $\le 0.05$.

---

### 3.3 Finding F-03: Statistical Power and Effect Size Assumptions (`CONFIRMED_DEFECT`)

#### Examination of v1.0 Assumptions
Protocol v1.0 §5.2 based its power analysis on a single observation from Phase 2B.0.2:
- $R_{\text{Greedy}} = 60$, $R_{\text{DD}} = 40 \implies \Delta = +20$.
- Extrapolated $\mu_\Delta = 15.0$, $\sigma_\Delta = 12.0 \implies \text{Cohen's } d = 1.25$.
- Claimed power with $n=12$: **$97.8\%$**.

Extrapolating population variance and mean difference from a sample size of $n=1$ is methodologically invalid. In realistic stochastic agent environments:
- Distractor configurations may cause delta debugging to explore multiple partitions when singletons fail.
- Greedy ablation on small $K$ ($K=4$) requires at most $4 \times 5 = 20$ search trials. If the trigger is ablated in the first round, greedy terminates in 40 requests.
- Discrete ranks, ties (where both algorithms consume identical requests), and skewed difference distributions severely reduce the power of the Wilcoxon signed-rank test.

#### Exact Permutation and Monte Carlo Sensitivity Analysis
Using our validated script [`benchmarks/model_pilot/power_and_budget_analysis.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/power_and_budget_analysis.py), we conducted 10,000 Monte Carlo replications under exact Wilcoxon signed-rank permutations ($2^{12} = 4,096$ exact permutations, two-sided critical $W \le 13$, $\alpha = 0.0425$):

| Assumed Effect Size (Cohen's $d$) | Mean Paired Diff $\mu_\Delta$ (Requests) | Power: Normal (5% Ties) | Power: Right-Skewed (5% Ties) | Power: High Ties (15% Ties) | Power: Stratified ($n=6$, $K=4$ or $K=8$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $d = 0.40$ | $4.8$ | $20.3\%$ | $10.3\%$ | $17.5\%$ | $5.9\%$ |
| $d = 0.50$ | $6.0$ | $29.3\%$ | $21.7\%$ | $25.4\%$ | $8.1\%$ |
| $d = 0.60$ | $7.2$ | $40.7\%$ | $39.9\%$ | $35.3\%$ | $10.5\%$ |
| $d = 0.70$ | $8.4$ | $52.7\%$ | $64.9\%$ | $45.4\%$ | $14.1\%$ |
| $d = 0.80$ | $9.6$ | $63.3\%$ | $87.2\%$ | $55.5\%$ | $17.6\%$ |
| $d = 1.00$ | $12.0$ | $81.6\%$ | $100.0\%$ | $74.5\%$ | $26.2\%$ |
| **$d = 1.25$ (Pilot Assumed)** | **$15.0$** | **$94.6\%$** | **$100.0\%$** | **$89.8\%$** | **$38.2\%$** |

#### Key Power Findings
1. For moderate effects ($d = 0.50$), power is under **$30\%$**.
2. For large effects ($d = 0.80$), power is **$63.3\%$** under normal differences and drops to **$55.5\%$** if $15\%$ of tasks result in ties.
3. Target power ($1 - \beta \ge 0.80$) is achieved **only if the true effect size is $d \ge 1.00$** ($\mu_\Delta \ge 12.0$ requests with $\sigma_\Delta \le 12.0$).
4. Under stratified analysis ($n=6$), power is severely degraded ($< 38.2\%$ even at $d = 1.25$).

---

### 3.4 Finding F-04: Multiple Testing Procedure Mismatch in 1-Minimality (`CONFIRMED_DEFECT`)

#### Code vs. Protocol Audit
- **Protocol v1.0 §7.1 Criterion 3:**
  > *"Single-element deletion tests are evaluated using the Holm-Bonferroni step-down procedure at family-wise error rate $\alpha_{\text{FWER}} = 0.05$."*
- **Actual Codebase Implementation:**
  In [`src/agentxray/stochastic_minimizer.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/src/agentxray/stochastic_minimizer.py#L370-L388):
  ```python
  def adjust_p_values_bh(p_values: list[float]) -> list[float]:
      """Adjust p-values using the Benjamini-Hochberg FDR procedure."""
      m = len(p_values)
      if m <= 1:
          return list(p_values)
      sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
      adjusted = [1.0] * m
      ...
  ```
  The minimizer calls `adjust_p_values_bh`, which controls the **False Discovery Rate (FDR)**, not the family-wise error rate (FWER) via Holm-Bonferroni.

#### Impact & Remediation
While Benjamini-Hochberg is less conservative than Holm-Bonferroni and appropriate for screening deletions, the discrepancy between protocol text and code execution is an auditable defect.
**Remediation:** Protocol v1.1 explicitly adopts Benjamini-Hochberg FDR control ($\alpha_{\text{FDR}} = 0.05$) to match the verified implementation.

---

### 3.5 Finding F-05: Primary Estimand Distortions and Worst-Rank Assignment (`CONFIRMED_DEFECT`)

#### Mathematical Defect of Worst-Rank Scoring
Protocol v1.0 §2.4 states:
> *"If an algorithm reaches the request cap ... it is assigned the censored value $R_{\text{cap}}$ and penalized using worst-rank score assignment in the non-parametric Wilcoxon test."*

In a paired Wilcoxon signed-rank test on differences $\Delta_i = R_{\text{Greedy}, i} - R_{\text{DD}, i}$, artificially setting $R_{\text{Greedy}} = \infty$ or assigning arbitrary maximal ranks when one algorithm times out distorts the signed-rank statistic $W^+$. Specifically:
1. If Greedy reaches $R_{\text{cap}} = 180$ and terminates without finding the trigger, while Delta Debugging terminated at $R = 40$, the actual observed difference is $180 - 40 = 140$.
2. Artificially forcing this pair to receive the highest rank regardless of other task differences violates the rank invariance property of the Wilcoxon test and conflates search cost with search failure.

#### Remediation in Protocol v1.1: Two-Part Primary SAP
Protocol v1.1 replaces arbitrary worst-rank assignment with a principled **Two-Part Analysis Plan**:
- **Part A (Binary Success / Verified Recovery):** Evaluated via McNemar's exact test for paired nominal data. Tests whether the proportion of instances where Delta Debugging discovers an empirically verified 1-minimal subset differs from Greedy ablation.
- **Part B (Efficiency on Jointly Resolved Instances):** Paired two-sided Wilcoxon signed-rank test restricted to instances where *both* algorithms successfully identified an empirically verified trigger without budget exhaustion.
- **Part C (Survival / Time-to-Event Analysis):** Non-parametric Kaplan-Meier survival curves and Peto-Peto Wilcoxon test treating budget-exhausted runs as right-censored at $R_{\text{cap}}$.

---

### 3.6 Finding F-06: Worst-Case Budget Cap Overrun (`POTENTIAL_DEFECT`)

#### Simulation of Budget Demand Scenarios
Protocol v1.0 set a hard abort cap of $6,000$ model requests. Using [`power_and_budget_analysis.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/power_and_budget_analysis.py), we modeled three demand scenarios:

1. **Scenario LOW_DEMAND:** Fast convergence; clean singletons; no budget exhaustion.
   - Total requests: **$4,500$** ($\sim 33.8$ minutes).
2. **Scenario EXPECTED:** Pilot-consistent convergence; typical distractor counts; minimal retries.
   - Total requests: **$5,550$** ($\sim 41.6$ minutes).
3. **Scenario HIGH_DEMAND_WORST_CASE:** Greedy and Random reach caps ($180$ requests) on $K=8$ tasks; multiple ablation passes; $5\%$ infrastructure retries.
   - Active Calibration ($K=4, 8$): $2,040$ requests
   - Null Calibration ($K=4, 8$): $80$ requests
   - Stochastic DD Search: $1,080$ requests
   - Stochastic Greedy Search: $1,800$ requests
   - Stochastic Random Search: $1,440$ requests
   - Null Searches: $240$ requests
   - Holdout Confirmations: $360$ requests
   - Infra retry buffer ($5\%$): $290$ requests
   - **Total Worst-Case Requests: $7,330$ requests** ($\sim 55.0$ minutes).

#### Impact & Remediation
If data collection encounters difficult tasks in $K=8$, total requests will breach the $6,000$ cap, triggering an immediate `HARD_BUDGET_BREACH_ABORT` and corrupting the run.
**Remediation:** Increase the global safety cap in configuration from $6,000$ to **$7,500$ requests** ($56.3$ minutes estimated runtime, well within local hardware capabilities).

---

### 3.7 Finding F-07: Incomplete Power-Set Characterization for $K=8$ (`POTENTIAL_DEFECT`)

#### Mathematical Analysis
For $K=8$, the full power set contains $2^8 = 256$ subsets.
Under Protocol v1.0, to keep characterization tractable, only $2K + 2 = 18$ subsets are tested ($1$ empty baseline, $8$ singletons, $8$ $K-1$ complements, and $1$ full set).
- Evaluated subsets: $18 / 256 = \mathbf{7.03\%}$
- Unobserved subsets: $238 / 256 = \mathbf{92.97\%}$

If a task contains an unmodeled 2-element distractor synergy (e.g. elements $\{d_2, d_5\}$ together cause failure even without the planted trigger $T_1$), the 18-subset characterization will not detect it. The minimizer may subsequently isolate $\{d_2, d_5\}$ instead of $T_1$.
**Remediation:** Protocol v1.1 explicitly notes this limitation in §10, documents that discovered subsets matching an empirical failure condition outside $C^*$ are scored as `EMPIRICAL_DISCOVERY_NON_GROUND_TRUTH`, and distinguishes empirical 1-minimality from reference trigger recovery.

---

## 4. Comprehensive Comparison: Protocol v1.0 vs. Protocol v1.1

| Dimension | Protocol v1.0 (Audited) | Protocol v1.1 (Repaired) | Scientific Justification |
| :--- | :--- | :--- | :--- |
| **Study Designation** | "Confirmatory Experiment" | **"Exploratory Multi-Instance Feasibility Benchmark"** | Corrects overclaim; sample size and task structure cannot support universal confirmatory claims. |
| **Active Task Structure** | 12 pooled instances | **6 base task mechanisms stratified across $K=4$ and $K=8$** | Prevents pseudoreplication and false inflation of degrees of freedom. |
| **Null Task Role** | Quantitative proof of FDR $\le 0.05$ | **Qualitative diagnostic consistency check** | Corrects mathematical impossibility of bounding FDR $\le 0.05$ with $N=4$ tasks. |
| **Power Claim** | $97.8\%$ power ($d=1.25$ assumed) | **Realistic power curve ($29.3\%\text{--}81.6\%$ for $d \in [0.50, 1.00]$)** | Eliminates ungrounded $n=1$ extrapolation; transparent about low power under stratification. |
| **1-Minimality Multiplicity** | Holm-Bonferroni FWER | **Benjamini-Hochberg FDR ($\alpha = 0.05$)** | Aligns protocol specification with actual implementation in `stochastic_minimizer.py`. |
| **Primary Estimand Plan** | Single Wilcoxon with arbitrary worst-rank | **Two-Part Hurdle Model (McNemar + Conditional Wilcoxon) & Survival** | Prevents rank distortion caused by arbitrary scoring of budget-exhausted runs. |
| **Global Request Cap** | $6,000$ model requests | **$7,500$ model requests** | Prevents premature abort under high-demand/worst-case $K=8$ exploration ($7,330$ requests). |
| **$K=8$ Distractor Characterization** | Implicitly assumed complete | **Explicit 93% unobserved space qualification** | Transparently acknowledges potential multi-element distractor interactions. |
| **Commercial Spend Cap** | $\$0.0000$ USD | **$\$0.0000$ USD** | Unchanged; strict zero commercial spend maintained. |
| **Local Model Runtime** | Ollama `llama3.2:3b` (Q4_K_M) | **Ollama `llama3.2:3b` (Q4_K_M)** | Unchanged; Apple Silicon local inference preserved. |

---

## 5. Implementation & Configuration Audit

### 5.1 Verification of Task Registry (`benchmarks/model_pilot/confirmatory_tasks.py`)
We verified all 16 task specifications in the registry:
- **Zero Leakage:** All 16 tasks instantiate `ModelBackedAgentTarget` with zero ground-truth leakage into `TargetResult` or agent evaluators.
- **Tool Integrity:** Tools (`calculator`, `database_lookup`, `unit_converter`) are properly isolated and deterministic.
- **Null Cases:** All 4 null cases (`inst_13` to `inst_16`) have empty `planted_trigger_elements` and belong to `TaskFamily.NULL_CONTROL`.

### 5.2 Verification of Budget Rules & Config v1.1
The revised configuration [`benchmarks/configs/phase_2b1_confirmatory_config_v1_1.json`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/configs/phase_2b1_confirmatory_config_v1_1.json) has been created and validated:
- `protocol_version`: `"Phase_2B.1_v1.1"`
- `study_designation`: `"EXPLORATORY_FEASIBILITY_BENCHMARK"`
- `budget_rules.max_total_request_budget`: `7500`
- `statistical_analysis_plan.primary_analysis_framework`: `"two_part_mcnemar_and_signed_rank"`
- `statistical_analysis_plan.stratification`: `"candidate_size_k"`
- `statistical_analysis_plan.cluster_adjustment`: `"base_task_paired"`
- `statistical_analysis_plan.minimality_multiplicity_correction`: `"benjamini_hochberg"`

---

## 6. Final Authorization & Readiness Decision

### Decision: `PROTOCOL_REVISED_EXPLORATORY_FEASIBILITY`
1. **Execution Remains Strictly Unauthorized:** Phase 2B.1 model execution must not proceed until formal user authorization is granted.
2. **Commercial Spend:** Strictly **$0.0000 USD**. No commercial APIs were contacted; no models were downloaded.
3. **Historical Evidence Integrity:** All Phase 2B.0, 2B.0.1, 2B.0.2, and 2B.1 v1.0 artifacts remain completely intact and uncorrupted.
4. **Readiness Status:** With the creation of Protocol v1.1, Configuration v1.1, the power analysis script, and the unit test suite, the experimental design has reached **scientific integrity readiness**. If authorized, execution can proceed on local hardware within an estimated 42–55 minutes without financial expenditure.

---
*Report certified by Antigravity Research Engineering & Independent Statistical Audit Team.*
