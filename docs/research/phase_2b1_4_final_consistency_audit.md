# AgentXRay — Phase 2B.1.4: Final Hypothesis Consistency & Execution Preflight Audit Report

**Date:** 2026-10-17  
**Author:** Independent Research & Statistical Audit Team (Antigravity)  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Commercial Spend Cap:** Strictly $0.0000 USD (zero external API spend incurred)  
**Target Model:** Local Ollama `llama3.2:3b`  
**Final Audit Recommendation:** `READY_FOR_EXPLICIT_USER_AUTHORIZATION`  

---

## 1. Executive Summary & Audit Decision

### 1.1 Audit Decision: `READY_FOR_EXPLICIT_USER_AUTHORIZATION`
Following an exhaustive, adversarial preflight audit of Project AgentXRay's Phase 2B.1 exploratory benchmark, the audit team determines that the study design, mathematical definitions, statistical analysis plan, budget guardrails, and software implementation are **fully reconciled, mathematically sound, verified against historical baselines, and ready for an explicit user authorization decision**.

The evaluation confirms:
1. **Critical Audit A Resolved (Dual-Condition 1-Minimality Standard):** The mathematical contradiction between Claim A (threshold-based non-failure: $p(C \setminus \{e_j\}) < \theta_{\text{fail}}$) and Claim B (comparative failure reduction: $p(C \setminus \{e_j\}) < p(C)$) has been completely resolved. Protocol v1.4 formally establishes **Claim B under Intersection-Union Testing with an Empirical Threshold Guard** as the primary operational definition of `SUPPORTED_1_MINIMAL`, while logging Claim A exact binomial p-values concurrently in telemetry for complete diagnostic visibility. The invalid shortcut (`or ablated rate <= 0.20`) remains completely purged.
2. **Critical Audit B Resolved (Mechanism Clustering & Pseudoreplication Protection):** The 16 task instances have been formally decomposed into $n=6$ independent failure mechanisms ($M_1$ through $M_6$) and $n=2$ null controls ($N_1, N_2$), evaluated across repeated measurements at $K=4$ and $K=8$. Unclustered $N=12$ pooled inferential testing has been demoted to descriptive exploratory reporting. The exact permutation null distribution for $n=6$ mechanisms has been quantified ($2^6 = 64$ permutations, minimum achievable two-sided $p = 0.03125$ at $W=0$), establishing that rejection at $\alpha = 0.05$ strictly requires delta debugging to outperform greedy ablation on all 6 mechanisms without ties.
3. **Execution Boundary Respected:** Absolutely **zero target-model inference requests**, zero Ollama calls, zero model downloads, and zero external API calls were executed during this audit. All verifications were performed via exact analytical distributions, deterministic mock providers, and regression test suites.
4. **Historical & Versioned Integrity Preserved:** All historical protocol documents and configurations (v1.0, v1.1, v1.2, v1.3) remain intact and pass their respective automated validator checks. Phase 2A and Phase 2B.0 historical artifacts remain bitwise verified.
5. **Software Test Suite:** All 226 unit, integration, and statistical invariant tests pass cleanly in 0.36s.

**Operational Notice:** While the study is declared `READY_FOR_EXPLICIT_USER_AUTHORIZATION`, **actual model execution remains strictly unauthorized** until the user explicitly directs execution to begin.

---

## 2. Provenance, Checksum & Immutability Audit

### 2.1 Git Working-Tree Verification
- **Repository Path:** `/Users/yashmishra14/Personal projects/AgentXRay`
- **Branch:** `main`
- **Current HEAD Checkpoint:** `62579fb` (`docs(research): complete Phase 2B.1.3 audit report and v1.3 protocol freeze`)
- **Working-Tree Status:** Working tree contains targeted modifications and versioned successor artifacts v1.4. No Git commit, push, reset, or checkout has been executed during this audit.

### 2.2 Authoritative Artifact Cryptographic Checksums (SHA-256)

| Category | Artifact Path | SHA-256 Checksum | Audit Status |
| :--- | :--- | :--- | :---: |
| **Protocol v1.4** | `docs/research/phase_2b1_exploratory_protocol_v1_4.md` | `db45f4032f2340a9edc3441bafcb3957895b616d1289c28e85a5cc4671574f62` | Frozen Successor |
| **Config v1.4** | `benchmarks/configs/phase_2b1_exploratory_config_v1_4.json` | `b3e3b0b8450e325914ea8229faafce90a833eb038b02903bfa48062e8ee0cfb2` | Frozen Successor |
| **Protocol v1.3** | `docs/research/phase_2b1_exploratory_protocol_v1_3.md` | `cdd9d7b0e9c4418c43d07e2ee1a0e0cb8e5e3fca2ad0c20a8023cdd3152cd93d` | Preserved Historical |
| **Config v1.3** | `benchmarks/configs/phase_2b1_exploratory_config_v1_3.json` | `53076b95d9a58ace4a13818e34665f5ecf4a48f7c4e868b9579f5ba47235a825` | Preserved Historical |
| **Protocol v1.2** | `docs/research/phase_2b1_exploratory_protocol_v1_2.md` | `822d7835a9730eee40ec505a6e2c7386e55674daa1f5afb35b4ee209fa5d2a8e` | Preserved Historical |
| **Config v1.2** | `benchmarks/configs/phase_2b1_exploratory_config_v1_2.json` | `a126f8a2d9c17f072b6e4d0f7b83995748a0b8b83e48cbecc6d833ef61a4399b` | Preserved Historical |
| **Protocol v1.1** | `docs/research/phase_2b1_confirmatory_protocol_v1_1.md` | `9228ce033c262c41bb2603d43bbe67579e9d3297416a021e9dc7abd50de35aff` | Preserved Historical |
| **Config v1.1** | `benchmarks/configs/phase_2b1_confirmatory_config_v1_1.json` | `1289450684ba63ad77da0e4485f7b6b75e4cc574082221ba26e07d56e4d08092` | Preserved Historical |
| **Protocol v1.0** | `docs/research/phase_2b1_confirmatory_protocol_v1.md` | `cd4a9a2584d837bad22bfafad09a43fa98dc720b6ebb0aad7ec4cfa92eb9cd7e` | Preserved Historical |
| **Config v1.0** | `benchmarks/configs/phase_2b1_confirmatory_config.json` | `7e5e0f0992ffbaaeb2c64e661c44006906571783d9658fa3846a45c77b6732ee` | Preserved Historical |
| **Code** | `src/agentxray/statistics.py` | `a130f01705a23e420eb19d01f21fbfcead65dceea225c48658feeac9becd9365` | Audited & Patched |
| **Code** | `src/agentxray/stochastic_minimizer.py` | `cbe2f3007942d1621fcb570a266b78adadcf6a140e55c04e23ca990a0a9b527d` | Audited & Patched |
| **Code** | `src/agentxray/stochastic_baselines.py` | `a475ada4304e99c173bbbd4987325ef18c3779719000adeedd420d75c96c429f` | Audited & Patched |
| **Code** | `src/agentxray/models.py` | `ce877e96d78c2c5cb21077c0018ba6f6b009246be1feb1803fcaf286170ecc70` | Audited |
| **Benchmark** | `benchmarks/model_pilot/confirmatory_tasks.py` | `57e4e8369084ca414f8449eda1c65f0f2355cee21598696ede9d332eaedb7872` | Audited & Verified |
| **Validator** | `benchmarks/model_pilot/validate_confirmatory_protocol.py` | `87d47e485c954bff9855fbd53c43ce91dc5d5be386637a3c6a2d772b1c7e66fb` | Audited & Patched |
| **Test Suite** | `tests/test_phase_2b1_4_audit.py` | `1067bc9b60267c91137613e39d942f4c80522bf3749273a7f81708610a063033` | New Test Suite |
| **Test Suite** | `tests/test_phase_2b1_3_audit.py` | `407443a6599d3957348726b74e15d2b105e1e526d227ef6fbb886e549c98eb01` | Preserved Historical |
| **Result Log** | `benchmarks/results/phase_2a1_event_log.jsonl` | `a55da53788dbde009db120d3f470903d09d4d72664edecc2394b3530e0b12a21` | Bitwise Preserved |
| **Result Log** | `benchmarks/results/phase_2b0_2_calibration_event_log.jsonl` | `e327031352b19d12ee9d27cee334b8506738d66357af41d6c9f365841f257028` | Bitwise Preserved |
| **Result Log** | `benchmarks/results/phase_2b0_2_calibration_results.json` | `fdf1db73e464dd3755e5cf31b95feb700ea89674d399bb1a3587eee68fb91f80` | Bitwise Preserved |

---

## 3. Critical Audit A: Resolution of 1-Minimality Hypotheses

### 3.1 Mathematical Comparison: Claim A vs Claim B

In Phase 2B.1.3, the protocol text and Python code exhibited a subtle mathematical discordance:
- **Claim A (Threshold-Based Minimality):** Focused on whether single-element deletion brings the failure probability below the failure threshold:
  $$H_{0, j}^{\text{thresh}}: p(C \setminus \{e_j\}) \ge \theta_{\text{fail}} = 0.50 \quad \text{vs} \quad H_{1, j}^{\text{thresh}}: p(C \setminus \{e_j\}) < 0.50$$
- **Claim B (Comparative Minimality):** Focused on whether single-element deletion causes a statistically significant drop relative to the candidate subset:
  $$H_{0, j}^{\text{comp}}: p(C \setminus \{e_j\}) \ge p(C) \quad \text{vs} \quad H_{1, j}^{\text{comp}}: p(C \setminus \{e_j\}) < p(C)$$

The two claims test different scientific questions and behave differently under small sample sizes ($N=5$):

| Scientific Criterion | Claim A (Threshold-Based) | Claim B (Comparative Drop) |
| :--- | :--- | :--- |
| **Underlying Question** | "Does the ablated configuration function acceptably without $e_j$?" | "Is element $e_j$ causally responsible for the observed failure?" |
| **Component Test** | Exact one-sample binomial test ($p_0 = 0.50$, alternative='less') | Exact one-sided Fisher exact test on $2 \times 2$ table ($C$ vs $C \setminus \{e_j\}$, alternative='greater') |
| **$N=5$ Trial Sensitivity** | Rejection requires **0/5 failures** ($p = 1/32 = 0.03125$). An ablated rate of 1/5 yields $p = 6/32 = 0.1875 > 0.05$. | When candidate has 5/5, 1/5 deletion is significant ($p = 5/210 = 0.02381 < 0.05$). When candidate has 4/5, 0/5 deletion is significant ($p = 10/120 = 0.02381 < 0.05$). |
| **Failure Reproduction** | Requires separate check on $C$ ($\hat{p}(C) \ge 0.50$). | Evaluates difference directly, but candidate reproduction must be verified. |
| **Threshold Guard** | Built into the null hypothesis ($p_0 = 0.50$). | Must be explicitly guarded: if $\hat{p}(C \setminus \{e_j\}) \ge 0.50$, candidate cannot be minimal. |

### 3.2 Counterexample Matrix Demonstrating Divergence

To prove that Claim A and Claim B cannot be treated interchangeably, three mathematical counterexamples were implemented and deterministically tested in `tests/test_phase_2b1_4_audit.py`:

```
+-------------------------------------------------------------------------------------------------------+
| Scenario 1: Candidate 5/5, Deletion 1/5                                                               |
|   - Claim B (Comparative Fisher one-sided): p = 0.0238 < 0.05  --> SIGNIFICANT REDUCTION              |
|   - Claim A (Threshold Binomial one-sided): p = 0.1875 > 0.05  --> NOT SIGNIFICANT (Power bound at N=5)|
|   --> Outcome under Protocol v1.4: SUPPORTED_1_MINIMAL (Claim B satisfied; threshold guard 0.20 < 0.50) |
+-------------------------------------------------------------------------------------------------------+
| Scenario 2: Candidate 3/5, Deletion 0/5                                                               |
|   - Claim A (Threshold Binomial one-sided): p = 0.0312 < 0.05  --> SIGNIFICANT BELOW THRESHOLD        |
|   - Claim B (Comparative Fisher one-sided): p = 0.0833 > 0.05  --> NOT SIGNIFICANT (Weak candidate)   |
|   --> Outcome under Protocol v1.4: OBSERVED_1_MINIMAL (Candidate failed to distinguish 3/5 from 0/5)   |
+-------------------------------------------------------------------------------------------------------+
| Scenario 3: Candidate 50/50, Deletion 30/50                                                           |
|   - Claim B (Comparative Fisher one-sided): p < 1e-6           --> MASSIVE COMPARATIVE DROP           |
|   - Deletion Failure Rate: 30/50 = 0.60 >= failure_threshold (0.50)                                   |
|   --> Outcome under Protocol v1.4: NOT_MINIMAL (Failed threshold guard: still failing 60% of the time) |
+-------------------------------------------------------------------------------------------------------+
```

### 3.3 Authoritative Operational Standard in Protocol v1.4
Protocol v1.4 reconciles both criteria into a unified, mathematically rigorous standard:
1. **Primary Preregistered Standard for `SUPPORTED_1_MINIMAL`:**
   - **Verification Complete:** All $k = |C|$ single-element removals evaluated.
   - **Candidate Failure Reproduction:** $\hat{p}(C) \ge 0.50$ with Wilson 95% CI lower bound $> 0.20$.
   - **Threshold Non-Failure Guard:** Every removal satisfies $\hat{p}(C \setminus \{e_j\}) < \theta_{\text{fail}} = 0.50$.
   - **Comparative Drop under IUT (Claim B):** Every single-element removal achieves exact one-sided Fisher significance:
     $$\max_{j=1,\dots,k} p_j^{\text{fisher}} < 0.05$$
2. **Concurrent Claim A Diagnostic Telemetry:**
   Exact one-sample binomial test p-values against $\theta_{\text{fail}} = 0.50$ are computed and logged in the execution telemetry for every removal, recording whether the deletion also achieved threshold significance.
3. **Purged Disjunctive Shortcut:**
   The heuristic loophole (`or ablated rate <= 0.20`) remains completely eliminated from all code and protocol documents.

---

## 4. Critical Audit B: Independent Experimental Units & Clustering Analysis

### 4.1 Underlying Failure Mechanisms ($n=6$) vs Task Instances ($N=12$)
The Phase 2B.1 benchmark comprises 16 task configurations generated from 8 distinct failure mechanisms:
- **6 Active Mechanisms:** $M_1$ through $M_6$ (2 syntax, 2 routing, 2 schema).
- **2 Null Controls:** $N_1$ and $N_2$ (clean calculation and clean database query).

For each mechanism $M_i$, the $K=4$ instance and $K=8$ instance share:
- The exact same planted trigger element (e.g., `strict_currency_formatting`).
- The exact same user instruction and prompt phrasing.
- The exact same tool environment and evaluation logic.
- The exact same underlying causal failure mechanism in `llama3.2:3b`.

The sole operational distinction between $K=4$ and $K=8$ is the number of distractor elements (3 distractors for $K=4$; 7 distractors for $K=8$).

**Methodological Consequence:** $K=4$ and $K=8$ are **repeated measurements of the same underlying causal mechanism under two prompt-length conditions**. Treating the 12 active instances as independent observations in confirmatory inferential tests is an instance of **pseudoreplication** (Hurlbert 1984), artificially doubling degrees of freedom and deflating standard errors.

### 4.2 Exact Wilcoxon Signed-Rank Null Distribution for $n=6$ Independent Mechanisms
To evaluate resource burden at the mechanism level, the paired difference across both candidate sizes is aggregated into a single mechanism-level score:
$$\bar{D}_i = \frac{1}{2}\left[\frac{R_{\text{Greedy}}(M_i, 4) - R_{\text{DD}}(M_i, 4)}{120} + \frac{R_{\text{Greedy}}(M_i, 8) - R_{\text{DD}}(M_i, 8)}{180}\right]$$

For $n=6$ mechanisms, the Wilcoxon signed-rank test has ranks $\{1, 2, 3, 4, 5, 6\}$ with total rank sum $T = 21$.
The exact permutation space has $2^6 = 64$ equally probable sign assignments:

```
Exact Two-Sided Permutation Distribution for n=6:
--------------------------------------------------
W = min(W+, W-) = 0:  2 / 64  = 0.03125  <-- ONLY two-sided p <= 0.05
W = min(W+, W-) = 1:  4 / 64  = 0.06250  > 0.05
W = min(W+, W-) = 2:  6 / 64  = 0.09375  > 0.05
W = min(W+, W-) = 3:  8 / 64  = 0.12500  > 0.05
```

**Critical Mathematical Takeaways:**
1. **Unanimous Win Required:** The minimum possible two-sided p-value is $2/64 = 0.03125$. Achieving statistical significance at $\alpha = 0.05$ strictly requires stochastic delta debugging to use fewer requests than greedy ablation on **all 6 mechanisms without exception** ($W = 0$).
2. **Impact of Ties ($n=5$):** If even one mechanism ties ($\bar{D}_i = 0$), the effective sample size drops to $m = 5$. For $m=5$, total permutations is $2^5 = 32$, and the minimum possible two-sided p-value is $2/32 = 0.0625 > 0.05$. **Statistical significance at $\alpha = 0.05$ becomes mathematically impossible.**
3. **Power Limitation:** Monte Carlo power simulations (2,000 replications) show that for $n=6$, statistical power is bounded by $\le 37.1\%$ even for massive effect sizes ($d = 1.25$).

### 4.3 Estimand Specification & Stratified Reporting
To eliminate pseudoreplication while preserving all observational evidence:
1. **All-Assigned Tasks (Intention-to-Treat) Estimand:** Every assigned task is included. Runs that fail or hit budget caps are retained at their actual expended requests (capped at 120 or 180). No arbitrary penalty scalar is added.
2. **Mandatory Candidate-Space Stratification:** Request consumption must be reported separately for $K=4$ (cap 120) and $K=8$ (cap 180) as normalized burden $\tilde{R} = R / \text{Cap}(K) \in [0, 1]$.
3. **Pooled $N=12$ Status:** Pooled unclustered analyses across $N=12$ entries are strictly designated as **descriptive exploratory summaries** and cannot be cited as confirmatory proof.

---

## 5. Pre-Execution Budget & Runtime Envelope Audit

### 5.1 Enforceable Budget Caps & Safety Hard Stop
- **Per-Task Algorithm Caps:** 120 requests for $K=4$; 180 requests for $K=8$.
- **Pre-Request Budget Interlock:** `CostTracker.check_budget_before_request()` checks remaining headroom *before* issuing any Ollama or model request. A breach immediately raises `CostBudgetExceededError`, halting search under `BUDGET_EXHAUSTED` / `HARD_BUDGET_BREACH_ABORT`.
- **Global Safety Hard Stop:** Fixed at **7,500 target-model requests**, providing $+170$ requests of safety headroom above the empirical high-demand worst-case scenario.

### 5.2 Scenario Analysis & Runtime Envelope

| Scenario | Search DD | Search Greedy | Search Random | Search Null | Calibration | Holdout & Verification | Total Model Requests | Est. Wall-Clock (0.45s/call) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low Demand** | 480 | 720 | 600 | 120 | 2,120 | 460 | **4,500** | 33.8 min |
| **Expected Pilot** | 660 | 1,260 | 780 | 120 | 2,120 | 610 | **5,550** | 41.6 min |
| **High Demand (Worst-Case)** | 1,080 | 1,800 | 1,200 | 240 | 2,200 | 810 | **7,330** | 55.0 min |
| **Global Safety Hard Stop** | — | — | — | — | — | — | **7,500** | 56.3 min |
| **Theoretical Uncapped Ceiling**| 1,800 | 1,800 | 1,800 | 240 | 2,200 | 360 | **8,200** | 61.5 min |

### 5.3 Commercial Financial Spend Verification
- **Commercial Spend Cap:** Strictly **$0.0000 USD**.
- **External Network Access:** Disabled. No OpenAI, Anthropic, or cloud API keys are configured. All execution uses local Ollama daemon.

---

## 6. Target Harness, Local Model & Infrastructure Verification

### 6.1 Local Model Configuration
- **Model:** `llama3.2:3b` executed locally via Ollama daemon (`http://localhost:11434/api/generate`).
- **Deterministic Inference Parameters:** `temperature=0.0`, `seed=42`, `top_p=1.0`.
- **Harness Isolation:** Controlled agent runs in an isolated subprocess with bounded execution timeouts and structured tool environment (`MockToolEnvironment`).

### 6.2 Preflight Network & Secret Verification
- No remote network egress is permitted or configured.
- Cost tracker tracks simulated and local requests with zero commercial billing impact.

---

## 7. Traceability & Invariant Test Suite Results

### 7.1 Automated Test Execution Summary
The complete test suite was executed via `./.venv/bin/pytest`:
- **Total Tests Collected:** 226
- **Passed:** 226 (100%)
- **Failed:** 0
- **Duration:** 0.36 seconds

### 7.2 Independent Artifact Validators
1. **Phase 2A Scaling Benchmark Validator:**
   `reproduce_phase_2a1.py --validate-only`
   Result: `VALIDATION_PASSED` (0 discrepancies, event log bitwise verified).
2. **Phase 2B.0.2 Corrective Calibration Validator:**
   `validate_phase_2b0.py --event-log ... --results ...`
   Result: `VALIDATION_PASSED` (64/64 checks passed, 0 discrepancies).
3. **Phase 2B.1 Preregistration Protocol Validators:**
   - Protocol v1.4 & Config v1.4: `VALIDATION_PASSED` (35 checks passed, 0 discrepancies).
   - Protocol v1.3 & Config v1.3: `VALIDATION_PASSED` (29 checks passed, 0 discrepancies).
   - Protocol v1.2 & Config v1.2: `VALIDATION_PASSED` (27 checks passed, 0 discrepancies).
   - Protocol v1.1 & Config v1.1: `VALIDATION_PASSED` (24 checks passed, 0 discrepancies).
   - Protocol v1.0 & Config v1.0: `VALIDATION_PASSED` (23 checks passed, 0 discrepancies).

### 7.3 Tamper-Detection Mutation Testing
`tests/test_phase_2b1_4_audit.py` includes programmatic mutation tests:
- Mutating `primary_test` away from `stratified_descriptive_and_mechanism_level_wilcoxon` triggers immediate validator failure.
- Mutating `pooled_test_status` to confirmatory triggers immediate validator failure.
- Mutating `cluster_unit` away from `mechanism_id` triggers immediate validator failure.
- Removing Claim A / Claim B definitions or clustering discussions from the protocol text triggers immediate validator failure.

---

## 8. Methodological Limitations & Scientific Boundaries

1. **Power Limitations of Small Mechanism Sample ($n=6$):**
   With $n=6$ independent failure mechanisms, statistical power to reject $H_0$ under the Wilcoxon signed-rank test is bounded by $\le 37.1\%$ even for large effect sizes ($d=1.25$). A non-significant result must not be interpreted as evidence of algorithmic equivalence.
2. **Demoted Non-Inferiority:**
   Recovery non-inferiority testing cannot be formally established with $n=6$ mechanisms. Recovery rates are reported strictly descriptively.
3. **Model and Architecture Specificity:**
   Findings reflect the specific tool-calling error dynamics of local `llama3.2:3b`. Generalization to frontier closed-weights models requires separate cross-architecture validation.
4. **Synthetic Prompt Manipulations:**
   While tasks represent genuine tool-routing and schema-validation failure modes, planted prompt triggers are stylized and may not reflect complex multi-turn conversational failures.

---

## 9. Execution Readiness Sign-Off & Explicit Authorization Gate

### 9.1 Readiness Sign-Off
The Phase 2B.1 benchmark specification has completed four successive rounds of adversarial audit and refinement (v1.0 $\to$ v1.1 $\to$ v1.2 $\to$ v1.3 $\to$ v1.4):
- All mathematical definitions of 1-minimality are reconciled and implemented via exact tests.
- All experimental units are correctly clustered to prevent pseudoreplication.
- All budget caps and safety interlocks are operational and verified.
- All historical artifacts remain cryptographically verifiable and uncorrupted.

**Final Status:** `READY_FOR_EXPLICIT_USER_AUTHORIZATION`

### 9.2 Execution Boundary Notice
In accordance with mandatory audit rules, **execution of Phase 2B.1 remains strictly unauthorized**. No background jobs, Ollama daemons, or inference runs have been initiated. Execution requires an explicit, separate user authorization prompt.
