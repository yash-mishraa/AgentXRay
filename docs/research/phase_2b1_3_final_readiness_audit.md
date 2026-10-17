# AgentXRay — Phase 2B.1.3: Final Statistical Validity Audit & Execution-Readiness Gate Report

**Date:** 2026-10-17  
**Author:** Independent Research & Statistical Audit Team (Antigravity)  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Commercial Spend Cap:** Strictly $0.0000 USD (zero external API spend incurred)  
**Target Model:** Local Ollama `llama3.2:3b`  
**Audit Status Decision:** `EXPLORATORY_PROTOCOL_READY_FOR_USER_REVIEW`  

---

## 1. Verified Repository State & Provenance

### 1.1 Git Working-Tree Verification
- **Current HEAD:** Commit `62579fb` on branch `main`.
- **Working-Tree Status:** The working tree contains targeted audit repairs and versioned successor artifacts. No Git commit or push has been performed during this audit phase.
- **Historical Artifact Integrity:**
  - Phase 2A scaling benchmark event log (`phase_2a1_event_log.jsonl`) verified: `VALIDATION_PASSED` (0 discrepancies).
  - Phase 2B.0.2 corrective calibration artifacts (`phase_2b0_2_calibration_event_log.jsonl`, `phase_2b0_2_calibration_results.json`, `phase_2b0_2_ground_truth.json`, `phase_2b0_2_candidate_specs.json`) verified: `VALIDATION_PASSED` (64/64 checks passed, 0 discrepancies).
  - Historical Protocol v1.0, v1.1, and v1.2 documents and configs remain preserved and uncorrupted, passing versioned validation.
- **Model Inference Boundary:** Absolutely **zero target-model inferences**, zero Ollama requests, zero model downloads, and zero external API calls were executed. All tests used deterministic algorithms, analytical equations, and mock/fake providers.

### 1.2 Inspected Files & Implementations
1. `benchmarks/model_pilot/validate_confirmatory_protocol.py` (Protocol & configuration validator).
2. `benchmarks/model_pilot/confirmatory_tasks.py` (Registry of 16 confirmatory tasks and 8 mechanism clusters).
3. `benchmarks/model_pilot/power_and_budget_analysis.py` (Wilcoxon DP exact permutation, asymptotic tie-correction, MCSE, and budget scenarios).
4. `src/agentxray/models.py` (Domain models, `StochasticMinimalityStatus`, `SearchStatus`, and `CandidateEvaluation`).
5. `src/agentxray/stochastic_minimizer.py` (Stochastic delta debugging minimizer and `_classify_minimality`).
6. `src/agentxray/stochastic_baselines.py` (Stochastic greedy ablation, seeded random search, and `_classify_stochastic_minimality`).
7. `src/agentxray/agent_harness/cost_tracker.py` (Pre-request budget enforcement and cost tracking).
8. `benchmarks/configs/phase_2b1_exploratory_config_v1_2.json` (Audited proposed config v1.2).
9. `docs/research/phase_2b1_exploratory_protocol_v1_2.md` (Audited proposed protocol v1.2).

---

## 2. Findings Register (Severity-Classified)

| Finding ID | Severity | File / Symbol | Description & Forensic Analysis | Corrective Action Taken |
| :--- | :---: | :--- | :--- | :--- |
| **FIND-2B13-01** | **CRITICAL** | `docs/research/phase_2b1_exploratory_protocol_v1_2.md` §8.2 | **Invalid Disjunctive Heuristic Shortcut for 1-Minimality:** Protocol v1.2 allowed candidates to be classified as `SUPPORTED_1_MINIMAL` if $p_j < 0.05$ **or** ablated failure rate $\le 0.20$. Under $N=5$ trials, an observed rate of 0/5 or 1/5 yields Wilson 95% upper bounds of 43.4% and 62.3%! Conflating low point estimates with statistical support invalidates Type I error guarantees. | **Purged shortcut entirely.** In Protocol v1.3 and `_classify_minimality`, candidates with observed drops that lack significance at $\alpha=0.05$ are strictly classified as `OBSERVED_1_MINIMAL`, never `SUPPORTED_1_MINIMAL`. |
| **FIND-2B13-02** | **HIGH** | `docs/research/phase_2b1_exploratory_protocol_v1_2.md` §2.2 | **Mathematically Unachievable Recovery Non-Inferiority Claim:** Protocol v1.2 retained an arbitrary 15% non-inferiority margin ($\delta = 0.15$). With only $n=6$ independent mechanisms ($N=12$ paired instances), the 95% confidence interval width for paired risk difference exceeds $\pm 35\%$. Even under perfect recovery (6/6 vs 6/6), the lower bound cannot rule out $-0.15$. | **Demoted non-inferiority to descriptive exploratory reporting.** Formal non-inferiority hypothesis tests were rejected. Recovery is reported via paired $2 \times 2$ contingency tables, risk differences, and exploratory McNemar exact tests. |
| **FIND-2B13-03** | **MEDIUM** | `docs/research/phase_2b1_exploratory_protocol_v1_2.md` §5.1 | **Unstratified Resource-Burden Pooling across Candidate Sizes:** $K=4$ (cap 120, $2^4=16$) and $K=8$ (cap 180, $2^8=256$) have structurally different combinatorial search spaces and caps. Pooling raw request counts across $K=4$ and $K=8$ creates an arbitrary mixture distribution. | **Mandated stratified and normalized reporting.** Primary resource burden must be reported stratified by $K \in \{4, 8\}$ and as normalized burden $\tilde{R} = R / \text{Cap}(K) \in [0, 1]$ under an Intention-to-Treat all-assigned tasks framework. |
| **FIND-2B13-04** | **MEDIUM** | `src/agentxray/stochastic_minimizer.py`: `_classify_minimality` | **Intersection-Union vs Benjamini-Hochberg Implementation Mismatch:** The code applied Benjamini-Hochberg FDR adjustments across single-element removal tests, whereas the protocol claimed an Intersection-Union Test (IUT). Under Berger (1982), an IUT rejects global $H_0: \bigcup_{j=1}^k \{p_j \ge p_{\text{cand}}\}$ if every $p_j < \alpha$ without multiplicity inflation. Candidate reproduction was also unguarded for $\hat{p} < \theta_{\text{fail}}$. | **Harmonized IUT conjunction logic** in `stochastic_minimizer.py` and `stochastic_baselines.py`, added strict candidate reproduction guards ($\hat{p} \ge 0.50$, trials $> 0$), and documented exact IUT size bounds. |
| **FIND-2B13-05** | **LOW** | `src/agentxray/models.py`: `StochasticMinimalityStatus` | **Nomenclature Inconsistency:** `StochasticMinimalityStatus` lacked explicit enum members for `BUDGET_EXHAUSTED` and `INFRA_FAILURE`, and used `STATISTICALLY_SUPPORTED` / `OBSERVED_ONLY` while protocol text used `SUPPORTED_1_MINIMAL` / `OBSERVED_1_MINIMAL`. | **Added enum members and protocol aliases** to `StochasticMinimalityStatus` and `SearchStatus` ensuring complete bidirectional compatibility. |

---

## 3. Operational Definition & Validation Logic for Empirical 1-Minimality

### 3.1 Distinction Between the Five Scientific Concepts
1. **Candidate Failure Reproduction:** The candidate subset $C$ reliably causes the agent to fail: $\hat{p}(C) \ge \theta_{\text{fail}} = 0.50$ and Wilson 95% lower bound $L_{95}(C) > \theta_{\text{base}} = 0.20$.
2. **Observed Monotonic Deletion Drop:** Removing any single element $e_j \in C$ strictly decreases the observed sample failure rate: $\hat{p}(C \setminus \{e_j\}) < \hat{p}(C)$ and $\hat{p}(C \setminus \{e_j\}) < \theta_{\text{fail}}$.
3. **Evidence Against Unacceptable Failure Probability (Threshold Test):** Testing $H_{0, j}^{\text{thresh}}: p(C \setminus \{e_j\}) \ge \theta_{\text{fail}}$ vs $H_{1, j}^{\text{thresh}}: p(C \setminus \{e_j\}) < \theta_{\text{fail}}$.
4. **Statistically Supported Drop Relative to Candidate (Comparative Test):** Testing $H_{0, j}^{\text{comp}}: p(C \setminus \{e_j\}) \ge p(C)$ vs $H_{1, j}^{\text{comp}}: p(C \setminus \{e_j\}) < p(C)$.
5. **Supported 1-Minimality:** Statistical rejection of the composite null hypothesis across all single-element removals under an Intersection-Union Test.

### 3.2 Intersection-Union Testing Framework (Berger 1982)
The global null hypothesis for 1-minimality is that at least one single-element removal still reproduces failure:
$$H_0: \bigcup_{j=1}^k \{ p(C \setminus \{e_j\}) \ge \theta_{\text{fail}} \}$$
The alternative hypothesis is that *every* single-element removal has failure probability below threshold:
$$H_1: \bigcap_{j=1}^k \{ p(C \setminus \{e_j\}) < \theta_{\text{fail}} \}$$

By the Intersection-Union Principle (Berger 1982; Casella & Berger 2002):
- The rejection region is the intersection of individual rejection regions: $R = \bigcap_{j=1}^k R_j$.
- If each component test is conducted at level $\alpha = 0.05$, the overall Type I error of the IUT is bounded by:
  $$\sup_{\theta \in H_0} P_\theta\left(\bigcap_{j=1}^k \{T_j \in R_j\}\right) \le \min_{j=1,\dots,k} \sup_{\theta \in H_{0, j}} P_\theta(T_j \in R_j) \le \alpha.$$
- No multiplicity adjustment (such as Bonferroni or Benjamini-Hochberg) is required across the $k$ deletions.
- However, **every individual test must strictly achieve $p_j < \alpha$**.

### 3.3 Final Mutually Exclusive Outcome Taxonomy
Every candidate subset is assigned exactly one mutually exclusive outcome:
- `SUPPORTED_1_MINIMAL`:
  1. Full verification completed (all $k$ single-element deletions evaluated).
  2. Candidate reproduces failure: $\hat{p}(C) \ge 0.50$ and Wilson 95% lower bound $> 0.20$.
  3. Every single-element removal satisfies $p_j < 0.05$ without heuristic bypasses.
- `OBSERVED_1_MINIMAL`:
  1. Full verification completed.
  2. Candidate reproduces failure: $\hat{p}(C) \ge 0.50$.
  3. Every single-element removal satisfies $\hat{p}(C \setminus \{e_j\}) < \hat{p}(C)$ and $\hat{p}(C \setminus \{e_j\}) < 0.50$.
  4. One or more deletion tests fail to reach statistical significance at $\alpha = 0.05$ (e.g. power limitation at $N=5$).
- `NOT_MINIMAL`:
  Removal of at least one element failed to decrease the observed failure rate ($\hat{p}(C \setminus \{e_j\}) \ge \hat{p}(C)$) or ablated rate remained $\ge 0.50$.
- `INCONCLUSIVE`:
  Verification was interrupted before testing all deletions, candidate failed to reproduce failure ($\hat{p}(C) < 0.50$), or zero/insufficient observations.
- `BUDGET_EXHAUSTED`:
  Algorithm reached per-task or global request cap before completing search or verification.
- `INFRA_FAILURE`:
  Unrecoverable daemon crash, network disconnect, or timeout.

---

## 4. Resolution of Recovery Non-Inferiority Claim

### 4.1 Sample Size Infeasibility
A formal non-inferiority test requires demonstrating that the lower bound of a $100(1-\alpha)\%$ confidence interval for the paired risk difference $\Delta = \pi_{\text{DD}} - \pi_{\text{greedy}}$ is strictly greater than $-\delta = -0.15$.

With $n=6$ independent mechanisms:
- The standard error of the paired difference under 0 discordant pairs is at least $\sqrt{2(1/6)(5/6)} \approx 0.236$.
- The 95% confidence interval width for paired proportions with $n=6$ exceeds $\pm 35\%$.
- Under perfect recovery for both algorithms (6/6 vs 6/6), the Newcombe paired score interval lower bound is $\approx -0.28$, which cannot exclude $-0.15$.
- The statistical power to demonstrate non-inferiority with $\delta = 0.15$ and $n=6$ is virtually zero.

### 4.2 Revised Protocol v1.3 Stance
Non-inferiority is formally **demoted from a confirmatory hypothesis test to descriptive exploratory reporting**:
1. Descriptive contingency table ($2 \times 2$ of joint recovery).
2. Point estimate of paired risk difference $\hat{\Delta}$ with descriptive Wilson/Newcombe score intervals.
3. Exploratory McNemar's exact test for symmetry, noting explicitly that failure to reject symmetry does not imply equivalence.

---

## 5. Primary Resource-Burden Estimand & Stratification

### 5.1 Estimand Specification
- **Primary Population:** **All-Assigned Tasks (Intention-to-Treat)**. Every assigned task instance ($N=12$) is included in the primary efficiency comparison.
- **Handling of Unsuccessful / Budget-Exhausted Runs:** Actual expended requests up to the cap are recorded. No scalar penalties or arbitrary scores are fabricated.
- **Stratified Reporting:**
  - $K=4$ ($n=6$, cap 120 requests).
  - $K=8$ ($n=6$, cap 180 requests).
  - Normalized Request Burden: $\tilde{R}(i, K) = R(i, K) / \text{Cap}(K) \in [0, 1]$.
- **Primary Statistical Test:** Two-sided Wilcoxon signed-rank test on paired request differences across $N=12$ active instances, evaluated using exact dynamic programming permutation distribution.
- **Secondary Subgroup:** Jointly resolved instances (conditional descriptive subgroup), with prominent warnings regarding potential post-treatment selection bias.

---

## 6. Power Analysis Reproducibility & Sensitivity Audit

The script `benchmarks/model_pilot/power_and_budget_analysis.py` was executed independently and audited against mathematical first principles.

### 6.1 Exact Wilcoxon Permutation Boundary
- For $N = 12$ paired differences, total permutations = $2^{12} = 4,096$.
- Exact two-sided critical value at $\alpha = 0.05$: **$W \le 13$** (achieved exact $\alpha = 0.0425$).

### 6.2 Power Sensitivity Table ($N=12$ Paired Instances, 2,000 Monte Carlo Replications)

| Cohen $d$ | Mean Diff (Req) | Exact Permutation Power | MCSE | Asymptotic Power | Skewed Dist | Ties/Discrete | Stratified $n=6$ Power |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.40** | 4.8 | 20.1% | 0.90% | 19.7% | 10.6% | 16.4% | **5.3%** |
| **0.50** | 6.0 | 30.4% | 1.03% | 30.2% | 20.6% | 23.2% | **7.1%** |
| **0.60** | 7.2 | 43.0% | 1.11% | 42.4% | 41.0% | 35.4% | **10.4%** |
| **0.70** | 8.4 | 53.4% | 1.12% | 52.8% | 66.0% | 45.3% | **15.2%** |
| **0.80** | 9.6 | 64.0% | 1.07% | 63.3% | 86.8% | 53.9% | **16.4%** |
| **1.00** | 12.0 | 83.0% | 0.84% | 82.5% | 100.0% | 73.8% | **24.9%** |
| **1.25** | 15.0 | 94.8% | 0.49% | 94.5% | 100.0% | 89.3% | **37.1%** |

### 6.3 Critical Statistical Finding
When evaluated at the mechanism cluster level ($n=6$), statistical power is only **16.4% at $d=0.80$** and **37.1% at $d=1.25$**. This conclusively demonstrates that Phase 2B.1 is an **exploratory feasibility benchmark**, and confirmatory superiority claims cannot be supported.

---

## 7. Budget Arithmetic & Safety Hard-Stop Audit

### 7.1 Budget Scenarios
- **Low Demand Scenario:** 4,500 model requests (~33.8 minutes wall-clock at 0.45s/call).
- **Expected Pilot Scenario:** 5,550 model requests (~41.6 minutes).
- **High-Demand Worst-Case Scenario:** 7,330 model requests (~55.0 minutes).
- **Theoretical Maximum Cap:** 8,200 model requests (~61.5 minutes).

### 7.2 Safety Hard Stop
- **Configuration Cap:** **7,500 target-model requests**.
- **Headroom:** Provides $+170$ requests of headroom beyond the high-demand worst-case scenario.
- **Enforcement Interlock:** Checked before every request in `CostTracker.check_budget_before_request()`. Verified via deterministic counting provider test (`test_budget_hard_stop_exact_counting`) that 0 calls are issued after the cap is reached.

---

## 8. Validator Results & Deterministic Test Evidence

### 8.1 Full Pytest Suite
- **Executed Command:** `./.venv/bin/pytest`
- **Result:** **217 passed in 0.33s** (100% passing rate).

### 8.2 Protocol & Baseline Validators
1. **Phase 2A Scaling Benchmark Validator:**
   - `./.venv/bin/python benchmarks/synthetic/reproduce_phase_2a1.py --validate-only`
   - **Result:** `VALIDATION_PASSED` (0 discrepancies).
2. **Phase 2B.0.2 Forensic Validator:**
   - `./.venv/bin/python benchmarks/model_pilot/validate_phase_2b0.py ... --strict-exhaustive`
   - **Result:** `VALIDATION_PASSED` (64/64 checks passed, 0 discrepancies).
3. **Protocol v1.0 Validator:** `VALIDATION_PASSED` (23 checks passed).
4. **Protocol v1.1 Validator:** `VALIDATION_PASSED` (24 checks passed).
5. **Protocol v1.2 Validator:** `VALIDATION_PASSED` (27 checks passed).
6. **Protocol v1.3 Validator:** `VALIDATION_PASSED` (29 checks passed, 0 discrepancies).

### 8.3 Deterministic Mutation Tests (`tests/test_phase_2b1_3_audit.py`)
All 17 new deterministic tests pass:
- 1-minimality IUT vs observed-only boundary tests.
- Prevention of the invalid shortcut `p < alpha OR observed <= 0.20`.
- Handling of missing deletions and unverified candidates.
- Mutation tests proving the validator fails closed on corrupted mechanism counts, underfunded budgets, corrupted task caps, weakened qualification thresholds, confirmatory non-inferiority claims, and reintroduced shortcuts.
- Provider counting test verifying zero request leakage past the cap.

---

## 9. Final Artifacts & SHA-256 Checksums

| File Path | SHA-256 Checksum | Description |
| :--- | :--- | :--- |
| `docs/research/phase_2b1_exploratory_protocol_v1_3.md` | `cdd9d7b0e9c4418c43d07e2ee1a0e0cb8e5e3fca2ad0c20a8023cdd3152cd93d` | Authoritative Protocol v1.3 specification |
| `benchmarks/configs/phase_2b1_exploratory_config_v1_3.json` | `53076b95d9a58ace4a13818e34665f5ecf4a48f7c4e868b9579f5ba47235a825` | Authoritative Configuration v1.3 |
| `benchmarks/model_pilot/power_and_budget_analysis.py` | `4745800683bfd2a2087b08544b91f97e4ea328473f7385a45e721d9b6679ac8b` | Exact DP permutation and budget analysis script |
| `benchmarks/model_pilot/confirmatory_tasks.py` | `57e4e8369084ca414f8449eda1c65f0f2355cee21598696ede9d332eaedb7872` | 16-instance confirmatory task registry |
| `benchmarks/model_pilot/validate_confirmatory_protocol.py` | `3f426d937d73d51eed59bb8a6513190b0d0734d75be3b7d51b2e6bfb4bbef214` | Protocol and configuration validator |
| `src/agentxray/models.py` | `ce877e96d78c2c5cb21077c0018ba6f6b009246be1feb1803fcaf286170ecc70` | Domain models with minimality taxonomy |
| `src/agentxray/stochastic_minimizer.py` | `d706aaec5cd13cd722ccbe5c4c920151808fa8dfa64c9b5519284e6055651c55` | Stochastic delta debugger with IUT logic |
| `src/agentxray/stochastic_baselines.py` | `8b4e1de2dc41c1cb4683733083f99d27e3d92b1829a989bc002d106b5393ab74` | Stochastic baselines with IUT parity |
| `tests/test_phase_2b1_3_audit.py` | `407443a6599d3957348726b74e15d2b105e1e526d227ef6fbb886e549c98eb01` | Phase 2B.1.3 audit deterministic test suite |
| `docs/research/phase_2b1_3_final_readiness_audit.md` | *(Computed upon file save)* | This readiness report |

---

## 10. Remaining Methodological Limitations

1. **Exploratory Power Limitation:** With $n=6$ independent mechanisms, statistical power is severely constrained (only 16.4% at $d=0.80$). This study cannot establish confirmatory algorithmic superiority.
2. **Local Target Specificity:** Observations on local `llama3.2:3b` reflect specific instruction-following and token-level behaviors that may differ on larger commercial frontier models.
3. **Discrete Distractor Modeling:** Planted prompt and schema distractors provide controlled benchmarks but do not simulate non-deterministic multi-agent conversational drift.

---

## 11. Final Readiness Recommendation

### Official Verdict: `EXPLORATORY_PROTOCOL_READY_FOR_USER_REVIEW`

**Justification:**
- All methodological and statistical defects identified in Protocol v1.2 have been completely repaired and audited against mathematical first principles.
- The 1-minimality verification rule has been purged of invalid heuristic shortcuts and grounded in Berger's Intersection-Union Test.
- Recovery non-inferiority has been honestly demoted to descriptive exploratory reporting.
- Resource-burden comparisons are properly stratified across candidate-space sizes.
- Power calculations use exact DP permutation null distributions and report MCSEs.
- Request budgets are enforced by pre-request check with zero leakage.
- All 217 tests and 6 validators pass with zero discrepancies.

**Execution Boundary:**
Phase 2B.1 model execution remains **strictly unauthorized**. Do not execute Ollama inference, run experiments, download models, or commit changes until separate, explicit user authorization is provided.
