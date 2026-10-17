# AgentXRay — Phase 3.3: Final Reproducibility Audit & Release Gate

**Document Reference:** `docs/research/phase_3_3/phase_3_3_final_reproducibility_audit.md`  
**Date:** 2026-10-17  
**Version:** Phase 3.3 Final Reproducibility Audit  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Authoritative Claims Ledger:** [`docs/research/phase_3_3/phase_3_3_claims_ledger.json`](phase_3_3_claims_ledger.json) / [`docs/research/phase_3_3/phase_3_3_claims_ledger.md`](phase_3_3_claims_ledger.md)

---

## 1. Executive Summary and Audit Mission

This document establishes the final, independent offline reproducibility audit for Project AgentXRay. Its mission is to audit every empirical, statistical, and architectural claim against the immutable Phase 2B.1.5 raw artifacts, correct all remaining counting unit and terminology inconsistencies, and determine whether the research package meets the standard for an authoritative, reproducible open-source research release.

### Final Audit Disposition:
$$\mathbf{RELEASE\_READY\_WITH\_LIMITATIONS}$$

All 32 registered claims in the master claims ledger have been independently verified, corrected, or characterized with explicit boundaries. Zero empirical claims rely on unverified assumptions, zero model inference was conducted, and the repository is completely stable.

---

## 2. Independent Reproduction of Empirical Findings

All findings below were derived directly from raw serialized artifacts (`manifest.json`, `results.json`, `trials.jsonl`, `summary.csv`) and verified via the deterministic script [`tests/test_phase_3_3_reproducibility.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/tests/test_phase_3_3_reproducibility.py):

### 2.1 Request Ledger and Trial Event Accounting
- **Global In-Memory Ledger:** 2,176 total model requests (`manifest.json` and `results.json`).
- **Completed Trial Log:** Exactly 1,086 completed trials recorded in `trials.jsonl`.
- **Requests per Completed Trial:** Every single completed trial consumed exactly 2 requests (Turn 1 tool call + Turn 2 final answer).
- **Completed Log Sum:** $1,086 \times 2 = \mathbf{2,172\text{ requests}}$.
- **Ledger Discrepancy Reconciliation:** Exactly $+4\text{ requests}$ were consumed during Turn 1 on four separate runs of Seeded Random Search (`inst_02`, `inst_04`, `inst_09`, `inst_16`) where the per-task budget cap was reached before Turn 2 could be dispatched. Because exception handling aborted trial completion, these 4 requests were tracked in memory but not serialized to `trials.jsonl`.
- **Offline Traceability Boundary:** The conversational text and tool arguments of those 4 aborted requests cannot be reconstructed offline and are formally designated `UNRECOVERABLE_OFFLINE`. No synthetic payloads have been fabricated.

### 2.2 Task Configurations vs. Independent Mechanisms
The benchmark evaluates 16 distinct task configurations structured into 8 independent mechanisms:
- **Active Mechanisms (6):**
  - `mech_01_syntax_currency`: `inst_01` ($K=4$), `inst_02` ($K=8$)
  - `mech_02_syntax_comma`: `inst_03` ($K=4$), `inst_04` ($K=8$)
  - `mech_03_routing_override`: `inst_05` ($K=4$), `inst_06` ($K=8$)
  - `mech_04_routing_prohibition`: `inst_07` ($K=4$), `inst_08` ($K=8$)
  - `mech_05_schema_unit`: `inst_09` ($K=4$), `inst_10` ($K=8$)
  - `mech_06_schema_cross_unit`: `inst_11` ($K=4$), `inst_12` ($K=8$)
- **Negative Null Mechanisms (2):**
  - `mech_null_01_calc`: `inst_13` ($K=4$), `inst_14` ($K=8$)
  - `mech_null_02_db`: `inst_15` ($K=4$), `inst_16` ($K=8$)

### 2.3 Calibration Qualification and Search Outcomes
- **Disqualified Tasks:** Exactly 7 active task configurations failed pre-experiment calibration (`inst_05`, `inst_06`, `inst_07`, `inst_08`, `inst_10`, `inst_11`, `inst_12`) and received zero search trials per Protocol v1.4 §5.4.
- **Qualified Active Tasks:** Exactly 5 active task configurations met calibration and underwent algorithmic minimization (`inst_01`, `inst_02`, `inst_03`, `inst_04`, `inst_09`).
- **Capped Searches:** Exactly 4 search runs were halted by budget limits, all belonging to Seeded Random Search (`inst_02`, `inst_04`, `inst_09`, `inst_16`). Zero DD or Greedy runs exceeded budget limits.

### 2.4 Candidate Minimality Evaluations and Historical Invariance
- **Active Task Candidates Isolated:** Exactly 12 algorithm-task candidate subsets were isolated across the 5 qualified active tasks:
  - DD (5): `inst_01`, `inst_02`, `inst_03`, `inst_04`, `inst_09`
  - Greedy (5): `inst_01`, `inst_02`, `inst_03`, `inst_04`, `inst_09`
  - Random (2): `inst_01`, `inst_03`
- **Null Task Candidates Isolated:** Exactly 5 algorithm-task candidate subsets were isolated across the 4 null tasks (DD: 2, Greedy: 2, Random: 1, all on `inst_15` and `inst_16`).
- **Statistical Invariance under Corrected Guard:** In 100% of these 17 candidate evaluations, the candidate exhibited **5/5 failures** ($k_1=5, n_1=5 \implies \hat{p}=1.00$), with a two-sided Wilson 95% confidence interval of $[0.5655, 1.0000]$. Because $L_{95} = 0.5655 > 0.20$, the implementation omission of the Wilson lower bound guard had **zero effect on historical classifications**. All 17 candidates remain confirmed as `SUPPORTED_1_MINIMAL`.

### 2.5 Ground Truth Recovery and Algorithmic Parity
- **Stochastic Delta Debugging:** 4 / 12 active tasks (33.3%: `inst_01`, `inst_02`, `inst_03`, `inst_04`).
- **Stochastic Greedy Ablation:** 4 / 12 active tasks (33.3%: `inst_01`, `inst_02`, `inst_03`, `inst_04`).
- **Seeded Random Search:** 2 / 12 active tasks (16.7%: `inst_01`, `inst_03`).
- **McNemar Exact Symmetry Test:** Contingency table $\begin{pmatrix} 4 & 0 \\ 0 & 8 \end{pmatrix}$, discordant pairs $b = 0, c = 0$, exact $p = 1.0$. This represents an **absence of observed asymmetry**, not proof of equivalence.

### 2.6 Intention-to-Treat Burden and Mechanism-Level Wilcoxon Test
- **Normalized Burden Differences ($\bar{D}_i$ across 6 active mechanisms):**
  - `mech_01_syntax_currency`: $+0.2222$
  - `mech_02_syntax_comma`: $+0.2222$
  - `mech_03_routing_override`: $0.0000$ (disqualified, imputed penalty)
  - `mech_04_routing_prohibition`: $0.0000$ (disqualified, imputed penalty)
  - `mech_05_schema_unit`: $-0.0417$
  - `mech_06_schema_cross_unit`: $0.0000$ (disqualified, imputed penalty)
- **Wilcoxon Signed-Rank Permutation Test:** Discarding zero differences leaves $m=3$ non-zero mechanisms. Exact subset-sum DP permutation p-value is **$p = 0.50$**, failing to reject $H_0$. Because $2^3 = 8$ total permutations yield a minimum possible two-sided p-value of $2/8 = 0.25 > 0.05$, rejecting $H_0$ at $\alpha=0.05$ is mathematically impossible.

### 2.7 Holdout Replication
- All 12 active-task candidate subsets were evaluated on 5 fresh random prompt seeds ($N_{\text{holdout}} = 60\text{ trials}$, 120 model requests).
- **100% Replication:** All 12 candidates reproduced 5/5 failures in holdout (100% point rate, Wilson 95% CI $[0.5655, 1.0000]$).

---

## 3. Resolution of Counting Units and Terminology Inconsistencies

To ensure research-grade scientific defensibility, Phase 3.3 audits and enforces strict definitional boundaries across all project documentation:

### 3.1 Taxonomy of Units

```
+---------------------------+-------------------------------------------------------------------------+
| Term                      | Strict Scientific Definition in AgentXRay                               |
+---------------------------+-------------------------------------------------------------------------+
| Task Configuration        | A single, specific instance characterized by prompt, tools, and K       |
|                           | (e.g., inst_15_null_db_k4). Total = 16 configurations.                   |
+---------------------------+-------------------------------------------------------------------------+
| Mechanism                 | The underlying causal failure or null archetype shared by paired K=4    |
|                           | and K=8 configurations (e.g., mech_null_02_db). Total = 8 mechanisms.   |
+---------------------------+-------------------------------------------------------------------------+
| Algorithm-Task Run        | One algorithm execution on one task configuration (e.g., DD on inst_01).|
|                           | Total qualified active runs = 15; total null runs = 12.                 |
+---------------------------+-------------------------------------------------------------------------+
| Candidate Subset          | The specific set of interventions isolated by an algorithm.             |
|                           | Total isolated active candidates = 12; total null candidates = 5.       |
+---------------------------+-------------------------------------------------------------------------+
| Ground-Truth Recovery     | Exact binary match between isolated candidate subset and planted trigger|
|                           | elements (C = T).                                                       |
+---------------------------+-------------------------------------------------------------------------+
| Operational 1-Minimality  | Statistical classification under Protocol v1.4 §6.2 (IUT Fisher test    |
|                           | max p < 0.05 and deletion failure rate < 0.50).                         |
+---------------------------+-------------------------------------------------------------------------+
| Holdout Reproduction      | Independent out-of-sample replication of failure on unseen prompt seeds.|
+---------------------------+-------------------------------------------------------------------------+
| Empirical False-Discovery | The observed descriptive proportion of null units exhibiting candidate  |
| Proportion                | discoveries. Must explicitly state numerator and denominator.            |
+---------------------------+-------------------------------------------------------------------------+
| Statistical False         | An inferential population parameter (q-value / FDR). Must NOT be claimed|
| Discovery Rate (FDR)      | from 4 null configurations without an inferential mixture model.        |
+---------------------------+-------------------------------------------------------------------------+
```

### 3.2 Correction of Null-Control False Discovery Phrasing
- **Historical Ambiguity:** Earlier narrative drafts referred to false discoveries on null controls as "2 out of 4 null mechanisms" or "50% FDR."
- **Definitive Correction:** The denominator of 4 represents **task configurations**, not mechanisms. Across the 4 null task configurations, exactly 2 configurations (`inst_15` and `inst_16`) exhibited false discoveries, which both derived from **1 of the 2 underlying null mechanisms** (`mech_null_02_db`).
- **Corrected Descriptive Proportions:**
  - Across null task configurations: $2 / 4 = \mathbf{50.0\%}$
  - Across null mechanisms: $1 / 2 = \mathbf{50.0\%}$
  - Across algorithm-task null runs: $5 / 12 = \mathbf{41.7\%}$ (DD on both, Greedy on both, Random on `inst_15`)
- All 5 false-discovery runs isolated the identical single element: `department_code: assign to department code DEPT-FIN-01`.

---

## 4. Implementation and Provenance Reconciliation

To prevent misattribution between code versions, Phase 3.3 formally delineates:

1. **The Historical Executing Code (Phase 2B.1.5):**
   - Executed the benchmark using runner script hashing to `69b03d52b198bff5f23e86894041e810e36c75d052753620e3872d56342d7b27`.
   - Omitted the Wilson lower bound guard $L_{95} > 0.20$ in `_classify_minimality`.
   - Generated the raw artifacts in `benchmarks/results/phase_2b1_5_execution/`.
2. **The Corrected Code on Disk (Phase 3.2 / 3.3):**
   - Corrected in `src/agentxray/stochastic_minimizer.py` (SHA-256: `88593879...`) and `src/agentxray/stochastic_baselines.py` (SHA-256: `cf4abc19...`).
   - Implements explicit verification of `current_eval.confidence_interval[0] > baseline_threshold`.
   - Verified by 12 adversarial boundary tests in `tests/test_phase_3_2_adversarial_minimality.py` (SHA-256: `294aa907...`).
3. **Uncommitted Working Tree State:**
   - Pre-existing working-tree modifications from earlier phases are preserved uncommitted on branch `main` at commit `62579fb`.
4. **Empirical Evidence Invariance:**
   - Raw observations prove that the historical code's omission had zero corrupting impact on results because all historical candidates had 5/5 failures ($L_{95} = 0.5655 > 0.20$).

---

## 5. Explicit Limitations and Non-Claims

The AgentXRay research package maintains strict epistemic modesty:
1. **No Frontier Equivalence or Commercial Generalization:** All empirical evidence was generated on local `llama3.2:3b`. Results must not be extrapolated to frontier models (`gpt-4o`, `claude-3-5-sonnet`) without separate authorized benchmarks.
2. **No Claim of General Minimization Superiority:** At the mechanism level ($n=6$, $m=3$), statistical power is bounded by $\le 37.1\%$ for large effects, and the Wilcoxon test yielded $p = 0.50$. The 43.8% request reduction is strictly descriptive and confined to the 4 qualified syntax tasks.
3. **Absence of Asymmetry $\neq$ Proof of Equivalence:** McNemar exact $p = 1.0$ ($b=0, c=0$) reflects lack of observed asymmetry between DD and greedy ablation on $N=12$ tasks; it does not demonstrate equivalence.
4. **Permanent Loss of 4 Aborted Payloads:** Turn 1 tool-call payloads on the four budget-capped random search trials cannot be recovered offline.

---

## 6. Release-Readiness Certification

The AgentXRay research package meets all eight release-readiness criteria established in Protocol v1.4 and Phase 3.3. 

The package is officially certified:

$$\mathbf{RELEASE\_READY\_WITH\_LIMITATIONS}$$
