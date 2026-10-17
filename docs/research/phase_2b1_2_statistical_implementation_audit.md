# AgentXRay — Phase 2B.1.2: Statistical Implementation Audit, Estimand Repair & Exploratory Protocol Freeze

**Document Type:** Independent Methodological & Implementation Forensic Audit  
**Date:** 2026-10-17  
**Author / Auditor:** Antigravity Research Engineering & Independent Statistical Audit Team  
**Subject Protocol:** `docs/research/phase_2b1_confirmatory_protocol_v1_1.md` (v1.1) & `phase_2b1_confirmatory_protocol_v1.md` (v1.0)  
**Subject Configuration:** `benchmarks/configs/phase_2b1_confirmatory_config_v1_1.json`  
**Target Codebase Checkpoint:** `62579fb` on branch `main`  
**Execution Environment:** Local Apple Silicon Mac (Quantized Ollama `llama3.2:3b`, Q4_K_M)  
**Commercial Spend Authorization & Audit:** **Strictly $0.0000 USD** (zero commercial API spend; zero live target-model inferences conducted during this phase)  
**Audit Verdict:** **`EXPLORATORY_PROTOCOL_READY_FOR_REVIEW`** (Protocol v1.1 defects resolved; Protocol v1.2 and Configuration v1.2 frozen for exploratory feasibility benchmarking; execution remains strictly unauthorized)

---

## 1. Executive Summary & Verification Boundaries

This forensic audit evaluates the statistical coherence, mathematical validity, implementation alignment, and safety of the Phase 2B.1.1 protocol amendments (`phase_2b1_confirmatory_protocol_v1_1.md` and `phase_2b1_confirmatory_config_v1_1.json`).

### 1.1 Verified Git & Repository State
- **Git HEAD:** `62579fb` on branch `main` (verified clean working tree on tracked files).
- **Tracked Files Diff:** Exactly clean (`git diff` is empty; historical Phase 2A, 2B.0, 2B.0.1, 2B.0.2, and 2B.1.0 artifacts are preserved intact and unmodified).
- **Untracked Artifacts Inventory (Preserved):**
  - `benchmarks/configs/phase_2b1_confirmatory_config_v1_1.json` (Phase 2B.1.1 config)
  - `benchmarks/model_pilot/power_and_budget_analysis.py` (Analysis script)
  - `docs/research/phase_2b1_confirmatory_protocol_v1_1.md` (Phase 2B.1.1 protocol)
  - `docs/research/phase_2b1_protocol_audit_v1.md` (Phase 2B.1.1 audit report)
  - `tests/test_phase_2b1_protocol_audit.py` (Phase 2B.1.1 test suite)
- **Live Inference Authorization:** **STRICTLY UNAUTHORIZED.** No live model calls (Ollama or API) were executed during this audit. All analyses use deterministic dynamic programming, synthetic fixtures, and fake providers.

### 1.2 Summary of Key Forensic Audit Findings
The Phase 2B.1.1 audit properly rejected universal confirmatory superiority claims, but left **four critical statistical and implementation defects** in Protocol v1.1:
1. **Power Simulation Discrepancy (`CONFIRMED_DEFECT`):** The Phase 2B.1.1 report claimed exact Wilcoxon signed-rank power simulations with critical value $W \le 13$ for $n=12$. In reality, the simulation code in `power_and_budget_analysis.py` calculated an asymptotic normal approximation $z$-test with continuity correction, used an un-tied variance formula that ignored Pratt tie corrections, and did not report Monte Carlo standard errors.
2. **Selection Bias in Conditional Efficiency Estimand (`CONFIRMED_DEFECT`):** Protocol v1.1 proposed evaluating request efficiency (Part B) strictly on instances where *both* algorithms achieved verified recovery. This is a textbook collider conditioning / survivorship bias: if Greedy fails on the hardest tasks, the conditional comparison evaluates Greedy only on tasks it found easy, distorting comparative efficiency.
3. **Informative Censoring in Survival Analysis (`CONFIRMED_DEFECT`):** Protocol v1.1 proposed Kaplan-Meier survival curves and Peto-Peto hypothesis testing, treating budget exhaustion as right-censoring. In black-box optimization, budget exhaustion is an informative search failure (a competing terminal event), directly violating the non-informative censoring axiom of survival analysis. Furthermore, standard KM ignores task-level pairing.
4. **Multiple Testing Conflation in 1-Minimality (`CONFIRMED_DEFECT`):** 1-minimality verification is an **Intersection-Union Test (IUT)** requiring that *every* single-element removal reduces failure. Applying Benjamini-Hochberg (BH) false-discovery-rate control across deletions conflates multiple discovery screening with conjunction testing, creating an artificial penalty that makes statistical support mathematically impossible for non-deterministic triggers under $N=5$ trials per candidate.

---

## 2. Defect Registry & Forensic Findings

| Finding ID | Classification | Subsystem / Area | Impacted Artifacts | Evidence & Defect Analysis | Remediation in Protocol v1.2 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F-10** | `CONFIRMED_DEFECT` | Statistical Power | `power_and_budget_analysis.py`, Protocol v1.1 §5.1 | **Power Code vs. Report Discrepancy:** Report claimed exact $W \le 13$ simulation, but code used asymptotic normal $z$-test without tie-corrected variance and omitted Monte Carlo standard error. | Upgraded script to execute exact subset-sum DP permutation p-values with exact MCSE reporting. |
| **F-11** | `CONFIRMED_DEFECT` | Primary Estimand | Protocol v1.1 §2.3, §8.1; Config v1.1 | **Selection Bias in Conditional Wilcoxon:** Restricting request comparison to jointly resolved tasks creates collider conditioning and hides algorithm failures. | Replaced with Dual-Endpoint Primary Co-Estimand including all assigned active tasks. |
| **F-12** | `CONFIRMED_DEFECT` | Survival Analysis | Protocol v1.1 §2.3 (Part C), §8.1 | **Informative Censoring Violation in KM / Peto-Peto:** Budget exhaustion is an informative competing failure, not random right-censoring. Pairing is ignored. | Formal survival hypothesis testing removed; demoted strictly to descriptive cumulative discovery plots. |
| **F-13** | `CONFIRMED_DEFECT` | Minimality Multiplicity | Protocol v1.1 §7.1; `stochastic_minimizer.py` | **IUT vs. FDR Conflation in 1-Minimality:** 1-minimality is an Intersection-Union Test. Applying BH penalizes small p-values ($p_{(1)} \le \alpha/k$), making statistical support impossible for $k_1 = 4/5$ with $N=5$. | Protocol v1.2 explicitly adopts Berger (1982) IUT principle with explicit outcome taxonomy. |
| **F-14** | `POTENTIAL_DEFECT` | Task Architecture | `confirmatory_tasks.py`, Protocol v1.1 §3.1 | **Implicit Mechanism Linkage:** The 6 base mechanisms across $K=4$ and $K=8$ were linked only in text, lacking a machine-checkable identifier in code. | Added `mechanism_id` field, `MECHANISM_REGISTRY`, and grouping helpers in `confirmatory_tasks.py`. |
| **F-15** | `POTENTIAL_DEFECT` | Budget Bounds | Protocol v1.1 §6.1, §6.2; Config v1.1 | **Scenario vs. Supremum Conflation:** $7,330$ requests was presented as a worst-case bound; the true theoretical ceiling if all algorithms exhaust caps is $8,200$. | Documented $7,330$ as an empirical high-demand scenario and $7,500$ as an enforceable safety firewall. |
| **F-16** | `NO_DEFECT_FOUND` | Pre-Flight Budget Hard-Stop | `agent.py`, `cost_tracker.py`, `target.py` | **Pre-Flight Budget Enforcement:** Budget checks occur strictly *before* every target-model request, with zero calls permitted after breach. | Verified and confirmed via deterministic fake-provider unit tests. |

---

## 3. Deep-Dive Forensic Analyses & Mathematical Proofs

### 3.1 Finding F-10: Power Analysis Code vs. Claim Discrepancy (`CONFIRMED_DEFECT`)

#### Root Cause Investigation
In `benchmarks/model_pilot/power_and_budget_analysis.py` (as authored in Phase 2B.1.1):
1. The script computed the exact critical value `W <= 13` for $n=12$ at $\alpha = 0.0425$ via dynamic programming.
2. However, inside `simulate_wilcoxon_power()`, the decision rule was:
   ```python
   mean_w = m * (m + 1) / 4.0
   std_w = math.sqrt(m * (m + 1) * (2 * m + 1) / 24.0)
   z = (w - mean_w + 0.5) / std_w
   p = 2.0 * normal_cdf(z)
   if p < 0.05 and w_plus > w_minus:
       rejections += 1
   ```
   This code completely ignored `crit = 13` and evaluated an asymptotic normal approximation.
3. The formula for `std_w` omitted the Pratt tie-correction term:
   $$\sigma_{W, \text{tied}}^2 = \frac{m(m+1)(2m+1) - \frac{1}{2}\sum_{j=1}^g t_j(t_j-1)(2t_j+1)}{24}$$
   Overestimating variance in the presence of ties artificially suppressed $|z|$ and distorted $p$.
4. Monte Carlo standard errors ($\text{MCSE} = \sqrt{\hat{p}(1-\hat{p})/N_{\text{sims}}}$) were omitted.

#### Remediation & Exact Permutation Implementation
We implemented `exact_wilcoxon_permutation_p_value()` using integer-scaled dynamic programming ($O(m \sum r)$ time, taking $< 0.1$ ms per sample):
$$\text{Scaled Mid-Ranks: } S_i = 2 r_i \in \mathbb{Z}^+, \quad W_{\text{min}} = \min(W^+, W^-)$$
The exact permutation p-value is computed from the dynamic programming partition count over all $2^m$ sign assignments:
$$p_{\text{exact}} = \frac{2 \sum_{s \le W_{\text{min}}} \text{count}(s)}{2^m}$$

#### Corrected Reproducible Power Sensitivity Table ($N=12$ Active Tasks, $N_{\text{sims}} = 2000$)
Using the upgraded script:

| Cohen's $d$ | Mean Paired Diff $\mu_\Delta$ (Requests) | Exact DP Permutation Power | Monte Carlo SE ($\pm \text{MCSE}$) | Pratt-Tied Asymptotic Power | Right-Skewed Power | Discrete Integer Ties Power | Stratified Cohort ($n=6$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $d = 0.40$ | $4.8$ | **$20.1\%$** | $\pm 0.90\%$ | $19.7\%$ | $10.6\%$ | $16.4\%$ | $5.3\%$ |
| $d = 0.50$ | $6.0$ | **$30.4\%$** | $\pm 1.03\%$ | $30.2\%$ | $20.6\%$ | $23.2\%$ | $7.1\%$ |
| $d = 0.60$ | $7.2$ | **$43.0\%$** | $\pm 1.11\%$ | $42.4\%$ | $41.0\%$ | $35.4\%$ | $10.4\%$ |
| $d = 0.70$ | $8.4$ | **$53.4\%$** | $\pm 1.12\%$ | $52.8\%$ | $66.0\%$ | $45.3\%$ | $15.2\%$ |
| $d = 0.80$ | $9.6$ | **$64.0\%$** | $\pm 1.07\%$ | $63.3\%$ | $86.8\%$ | $53.9\%$ | $16.4\%$ |
| $d = 1.00$ | $12.0$ | **$83.0\%$** | $\pm 0.84\%$ | $82.5\%$ | $100.0\%$ | $73.8\%$ | $24.9\%$ |
| **$d = 1.25$ (Pilot Extrapolated)** | **$15.0$** | **$94.8\%$** | $\pm 0.49\%$ | **$94.5\%$** | **$100.0\%$** | **$89.3\%$** | **$37.1\%$** |

#### Required Power Conclusion
- Target statistical power ($\ge 80\%$) is attained **only if the standardized effect size is $d \ge 1.00$** ($\mu_\Delta \ge 12.0$ requests with $\sigma_\Delta \le 12.0$).
- For realistic moderate effect sizes ($d = 0.50$), statistical power is approximately **$30\%$**.
- When stratified by candidate size ($n=6$), statistical power collapses to **$< 38\%$ even under the extreme pilot assumption of $d = 1.25$**.
- **Definitive Conclusion:** The statistical power analysis does **not** support confirmatory superiority claims. The experiment must strictly be classified as an **Exploratory Multi-Instance Feasibility Benchmark**.

---

### 3.2 Finding F-11: Selection Bias in the Conditional Efficiency Estimand (`CONFIRMED_DEFECT`)

#### Theoretical Flaw of Conditional Wilcoxon Comparison
Protocol v1.1 Part B proposed evaluating request differences $\Delta_i = R_{\text{Greedy}, i} - R_{\text{DD}, i}$ strictly on the subset of instances where *both* algorithms achieved verified recovery ($Y_{\text{DD}, i} = 1 \land Y_{\text{Greedy}, i} = 1$).

This creates severe **collider conditioning / survivorship bias**:
1. Suppose Task A is easy: both DD and Greedy recover the trigger ($Y_{\text{DD}} = 1, Y_{\text{Greedy}} = 1$).
2. Suppose Task B is difficult: DD isolates the trigger in 110 requests ($Y_{\text{DD}} = 1$), while Greedy exhausts budget at $R_{\text{cap}} = 180$ without finding the trigger ($Y_{\text{Greedy}} = 0$).
3. Under the conditional comparison, Task B is **discarded entirely** because Greedy failed! This selectively eliminates the exact task on which Delta Debugging demonstrated its greatest search advantage.
4. Conversely, if Greedy succeeds on a task where DD fails, discarding that task conceals DD's failure.
5. Therefore, the conditional estimand $\mathbb{E}[\Delta_i \mid Y_{\text{DD}, i} = 1, Y_{\text{Greedy}, i} = 1]$ does not represent overall algorithm efficiency; it represents an arbitrary, non-generalizable subgroup metric.

#### Remediation in Protocol v1.2: Dual-Endpoint Primary Co-Estimand
Protocol v1.2 replaces the flawed conditional comparison with an all-inclusive **Dual-Endpoint Primary Co-Estimand** evaluated across **all 12 assigned active task instances**:
1. **Endpoint 1: Verified Recovery Rate ($\pi_{\text{rec}}$):**
   $$\hat{\pi}_{\text{rec}} = \frac{1}{M_{\text{active}}} \sum_{i=1}^{M_{\text{active}}} \mathbb{I}(\text{Verified 1-Minimal Trigger Recovered within } R_{\text{cap}})$$
   Reported with exact Wilson score 95% confidence intervals and paired risk differences.
2. **Endpoint 2: Total Request Burden ($R_{\text{total}}$):**
   The total target-model requests consumed per assigned task, where budget-exhausted/unverified runs are scored at their actual resource expenditure ($R_{\text{cap}}$).
   Reported as median and interquartile range (IQR) across all 12 active instances.

Conditional efficiency among jointly resolved instances is demoted strictly to a **Descriptive Secondary Subgroup Analysis**, with explicit prohibitions against generalizing to all tasks.

---

### 3.3 Finding F-12: Informative Censoring in Survival Analysis (`CONFIRMED_DEFECT`)

#### Methodological Failure of Kaplan-Meier in Search Benchmarking
Protocol v1.1 Part C proposed using Kaplan-Meier survival curves and Peto-Peto Wilcoxon tests, treating budget exhaustion as right-censoring at $R_{\text{cap}}$.

This violates fundamental assumptions of survival analysis:
1. **Violation of Non-Informative Censoring:** Right-censoring in survival analysis assumes that censored individuals have the same subsequent hazard of failure as uncensored individuals. In search algorithms, reaching $R_{\text{cap}}$ without finding a trigger is not a random administrative dropout; it is an **informative failure of the search algorithm** (a competing risk).
2. **Violation of Independence:** Standard Kaplan-Meier and Peto-Peto tests assume independent observations, completely ignoring the paired structure of algorithms evaluated on identical task environments.

#### Remediation in Protocol v1.2
Formal survival hypothesis testing (Peto-Peto / log-rank tests) is **completely removed** from the inferential statistical plan. Empirical survival curves are retained strictly as **Descriptive Cumulative Discovery vs. Request Budget Curves** ($CDF(R)$).

---

### 3.4 Finding F-13: Minimality Multiplicity & Intersection-Union Testing (`CONFIRMED_DEFECT`)

#### Theoretical Flaw of Benjamini-Hochberg in 1-Minimality
Protocol v1.1 aligned with the existing codebase by specifying Benjamini-Hochberg (BH) FDR control for candidate single-element deletions.

However, from first statistical principles:
- 1-minimality requires that **EVERY** single-element deletion $C \setminus \{e_j\}$ causes failure rate to drop.
- The global null hypothesis of non-minimality is:
  $$H_0: \bigcup_{j=1}^k H_{0, j} \quad (\text{at least one removal does NOT decrease failure rate})$$
- The global alternative hypothesis of 1-minimality is:
  $$H_1: \bigcap_{j=1}^k H_{1, j} \quad (\text{EVERY removal decreases failure rate})$$
- This is an **Intersection-Union Test (IUT)** (Berger, 1982; Casella & Berger, 2002).
- In an IUT, the overall Type I error rate is controlled at level $\alpha$ if and only if **every individual hypothesis is rejected at level $\alpha$**:
  $$\text{Reject } H_0 \iff \max_{j=1}^k p_j < \alpha \implies P_{H_0}(\text{Type I Error}) \le \alpha$$
- No multiplicity penalty (neither Bonferroni nor Benjamini-Hochberg) is required. Applying BH actually forces $p_{(1)} \le \alpha / k$. With $N=5$ trials, requiring $p_{(1)} \le 0.05 / 4 = 0.0125$ is mathematically impossible for non-deterministic models ($4/5$ failures vs $0/5$ failures has Fisher exact $p = 0.0476 > 0.0125$), guaranteeing that non-deterministic triggers are perpetually labeled `OBSERVED_ONLY`.

#### Remediation in Protocol v1.2: Explicit Minimality Outcome Taxonomy
Protocol v1.2 formally adopts the Berger (1982) Intersection-Union Principle and defines an unambiguous, mutually exclusive outcome taxonomy:
- `SUPPORTED_1_MINIMAL`: Candidate reproduces failure ($\hat{p} \ge \theta_{\text{fail}}$ with Wilson lower bound $> 0.20$), and every single-element deletion is empirically verified to drop failure rate ($p_j < \alpha$ or ablated rate $\le \theta_{\text{base}}$).
- `OBSERVED_1_MINIMAL`: All observed removal failure rates strictly dropped below candidate, but finite-sample tests did not reach statistical significance ($p \ge \alpha$).
- `NOT_MINIMAL`: At least one removal failed to decrease failure rate, or still caused failure $\ge \theta_{\text{fail}}$.
- `INCONCLUSIVE`: Verification budget exhausted before completing all deletions, or high variance.
- `BUDGET_EXHAUSTED`: Algorithm reached request cap before completing search or verification.
- `INFRA_FAILURE`: Provider crash, disconnect, or timeout.

A missing deletion result is strictly classified as `INCONCLUSIVE` and **never** counts as evidence of minimality.

---

### 3.5 Finding F-14: Machine-Checkable Mechanism Identifiers (`POTENTIAL_DEFECT`)

To prevent pseudoreplication and ensure reproducible clustering in automated pipelines, [`benchmarks/model_pilot/confirmatory_tasks.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/confirmatory_tasks.py) was enhanced with an explicit `mechanism_id` field:
- `mech_01_syntax_currency` (`inst_01_syntax_currency_k4`, `inst_02_syntax_currency_k8`)
- `mech_02_syntax_comma` (`inst_03_syntax_comma_k4`, `inst_04_syntax_comma_k8`)
- `mech_03_routing_override` (`inst_05_routing_override_k4`, `inst_06_routing_override_k8`)
- `mech_04_routing_prohibition` (`inst_07_routing_prohibition_k4`, `inst_08_routing_prohibition_k8`)
- `mech_05_schema_unit` (`inst_09_schema_unit_k4`, `inst_10_schema_unit_k8`)
- `mech_06_schema_cross_unit` (`inst_11_schema_cross_unit_k4`, `inst_12_schema_cross_unit_k8`)
- `mech_null_01_calc` (`inst_13_null_calc_k4`, `inst_14_null_calc_k8`)
- `mech_null_02_db` (`inst_15_null_db_k4`, `inst_16_null_db_k8`)

Added exported helpers: `MECHANISM_REGISTRY`, `ACTIVE_MECHANISMS`, `NULL_MECHANISMS`, and `get_tasks_by_mechanism()`.

---

### 3.6 Finding F-15: Request Budget Reconciliation & Safety Firewall (`POTENTIAL_DEFECT`)

#### Scenario Accounting vs. Theoretical Supremum
The request budget was reconciled across four explicit tiers:
1. **Low Demand Scenario:** $4,500$ model requests ($\sim 33.8$ minutes on local Apple Silicon Mac). Fast convergence; clean singletons; no budget exhaustion.
2. **Expected Demand Scenario:** $5,550$ model requests ($\sim 41.6$ minutes). Pilot-consistent convergence; typical distractor counts; minimal retries.
3. **Empirical High Demand Scenario:** $7,330$ model requests ($\sim 55.0$ minutes). Greedy reaches caps on $K=8$; multiple ablation rounds; $5\%$ infrastructure retries.
4. **Theoretical Maximum Supremum:** If all three algorithms exhaust their caps on all 12 tasks:
   $$\text{Calibration: } 2,200 + \text{Search: } (3 \times 1,800 = 5,400) + \text{Null Search: } 240 + \text{Holdout: } 360 = \mathbf{8,200 \text{ requests}}$$

#### The Enforceable Safety Firewall
The global request cap of **$7,500$ requests** in [`phase_2b1_exploratory_config_v1_2.json`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/configs/phase_2b1_exploratory_config_v1_2.json) provides $+170$ requests of headroom over the empirical high demand scenario ($7,330$). It acts as an **enforceable safety firewall**, triggering `HARD_BUDGET_BREACH_ABORT` before total requests can reach the theoretical maximum ($8,200$), protecting local hardware resources.

---

### 3.7 Finding F-16: Verification of Pre-Flight Budget Enforcement (`NO_DEFECT_FOUND`)

We audited `ControlledAgent` in [`src/agentxray/agent_harness/agent.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/src/agentxray/agent_harness/agent.py#L91-L103) and `CostTracker` in [`src/agentxray/agent_harness/cost_tracker.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/src/agentxray/agent_harness/cost_tracker.py#L67-L87):
1. In `agent.py` line 94, `self.cost_tracker.check_budget_before_request()` is executed **at the top of the interaction loop before calling `self.provider.generate()`**.
2. If `self.total_requests >= self.config.max_model_requests`, `CostBudgetExceededError` is raised immediately.
3. In line 101, `allowed_retries = max(0, remaining_requests - 1)` bounds retries by remaining headroom.
4. We constructed a deterministic unit test with a counting `FakeProvider` confirming that when `max_model_requests = 3`, exactly 3 provider calls occur, and subsequent calls are blocked with zero provider invocations.

---

## 4. Comprehensive Changelog: Protocol v1.1 vs. Protocol v1.2

| Protocol Section | Protocol v1.1 (Audited) | Protocol v1.2 (Frozen for Review) | Scientific Rationale |
| :--- | :--- | :--- | :--- |
| **Title & Status** | Confirmatory Protocol v1.1 | **Exploratory Multi-Instance Feasibility Protocol (v1.2)** | Eliminates confirmatory terminology; reflects actual power limits. |
| **Primary Estimand** | Conditional Wilcoxon on jointly resolved tasks | **Dual-Endpoint Primary Co-Estimand (Verified Recovery % & Total Requests across all active tasks)** | Eliminates selection / survivorship bias; includes all assigned tasks. |
| **Conditional Efficiency** | Primary test (Part B) | **Descriptive Secondary Subgroup Metric** | Narrowly defines estimand to jointly resolved tasks; prohibits generalization. |
| **Survival Analysis** | Kaplan-Meier & Peto-Peto hypothesis testing | **Descriptive Cumulative Discovery CDF Plots** | Eliminates violation of non-informative censoring; removes invalid tests. |
| **1-Minimality Multiplicity** | Benjamini-Hochberg FDR step-up | **Berger (1982) Intersection-Union Test ($\alpha$-level conjunction)** | Restores theoretical coherence for conjunction testing; avoids false penalties. |
| **Minimality Outcomes** | Ambiguous 3-way status | **Explicit 6-way Mutually Exclusive Taxonomy** | Disentangles ground-truth match, empirical minimality, and missing data. |
| **Task Clustering** | Described in markdown prose only | **Explicit `mechanism_id` in registry & config** | Enforces machine-checkable pairing between $K=4$ and $K=8$. |
| **Request Cap Interpretation** | $7,330$ claimed as worst-case ceiling | **$7,330$ empirical high demand; $7,500$ safety firewall** | Clarifies distinction between empirical scenario and safety abort limit. |
| **Commercial Spend Cap** | $\$0.0000$ USD | **$\$0.0000$ USD** | Strictly preserved. |
| **Local Model Runtime** | Ollama `llama3.2:3b` (Q4_K_M) | **Ollama `llama3.2:3b` (Q4_K_M)** | Strictly preserved. |

---

## 5. Artifact & Verification Inventory

1. **Protocol Audit Document (v1.2):**
   - [`docs/research/phase_2b1_2_statistical_implementation_audit.md`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/docs/research/phase_2b1_2_statistical_implementation_audit.md)
2. **Revised Protocol Specification (v1.2):**
   - [`docs/research/phase_2b1_exploratory_protocol_v1_2.md`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/docs/research/phase_2b1_exploratory_protocol_v1_2.md)
3. **Revised Configuration (v1.2):**
   - [`benchmarks/configs/phase_2b1_exploratory_config_v1_2.json`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/configs/phase_2b1_exploratory_config_v1_2.json)
4. **Reproducible Power & Budget Analysis Script:**
   - [`benchmarks/model_pilot/power_and_budget_analysis.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/power_and_budget_analysis.py)
5. **Task Registry with Explicit Mechanism IDs:**
   - [`benchmarks/model_pilot/confirmatory_tasks.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/confirmatory_tasks.py)
6. **Multi-Version Protocol Validator:**
   - [`benchmarks/model_pilot/validate_confirmatory_protocol.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/validate_confirmatory_protocol.py)
7. **Comprehensive Unit Test Suite:**
   - [`tests/test_phase_2b1_2_audit.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/tests/test_phase_2b1_2_audit.py)

---

## 6. Final Recommendation & Authorization Boundary

### Final Decision: `EXPLORATORY_PROTOCOL_READY_FOR_REVIEW`
The experimental design, statistical framework, power analysis, and budget enforcement have been rigorously audited, corrected, and verified. The protocol is statistically coherent, safe to run on local Apple Silicon hardware without financial expenditure, and appropriately scoped as an exploratory feasibility benchmark.

### Authorization Boundary
**PHASE 2B.1 EXECUTION REMAINS STRICTLY UNAUTHORIZED.**
- No target-model inference requests were executed during this audit.
- No commercial API spend was incurred (\$0.0000 USD).
- Do not commit or push without explicit user instruction.
