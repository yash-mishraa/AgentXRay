# AgentXRay — Phase 3.2: Research Report Addendum

**Document Reference:** `docs/research/phase_3_2/phase_3_2_research_report.md`  
**Date:** 2026-10-17  
**Version:** Phase 3.2 Addendum to Research Report  
**Authoritative Base Document:** [`docs/research/phase_3_1/phase_3_1_agentxray_research_report.md`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/docs/research/phase_3_1/phase_3_1_agentxray_research_report.md)  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`

---

## 1. Executive Addendum Certification

Following the comprehensive Phase 3.2 implementation conformance audit and candidate-by-candidate revalidation:
> **The scientific conclusions, statistical estimands, and empirical claims presented in [`docs/research/phase_3_1/phase_3_1_agentxray_research_report.md`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/docs/research/phase_3_1/phase_3_1_agentxray_research_report.md) remain 100% materially intact.**

Because zero historical candidate classifications, zero request tallies, and zero statistical test outcomes were changed or invalidated by the Phase 3.2 conformance findings, this document serves as an authoritative addendum rather than a duplicative rewrite of the full report.

---

## 2. Re-Certification of Core Scientific Findings

### 2.1 Algorithmic Recovery and Discovery Parity
- **Stochastic Delta Debugging (DD):** 4/12 active tasks (33.3%) recovered ground truth trigger.
- **Stochastic Greedy Ablation:** 4/12 active tasks (33.3%) recovered ground truth trigger.
- **Seeded Random Search:** 2/12 active tasks (16.7%) recovered ground truth trigger.
- **McNemar Exact Symmetry Test:** 
  $$\text{Contingency Table: } \begin{pmatrix} 4 & 0 \\ 0 & 8 \end{pmatrix}, \quad b = 0, \ c = 0, \quad p = 1.0$$
  *Scientific Statement:* Zero discordant pairs were observed ($b=0, c=0$). As mandated by rigorous statistical reporting, this indicates an absence of observed asymmetry between DD and greedy ablation on this benchmark, **not evidence of algorithmic equivalence**. With $N=12$, statistical power to detect moderate recovery differences is low.

### 2.2 Mechanism-Level Resource Burden ($n=6$)
- **Differences in Normalized ITT Burden ($\bar{D}_i$):**
  - `mech_01_syntax_currency`: $+0.2222$
  - `mech_02_syntax_comma`: $+0.2222$
  - `mech_03_routing_override`: $0.0000$ (disqualified task)
  - `mech_04_routing_prohibition`: $0.0000$ (disqualified task)
  - `mech_05_schema_unit`: $-0.0417$
  - `mech_06_schema_cross_unit`: $0.0000$ (disqualified task)
- **Wilcoxon Signed-Rank Exact Permutation Test:**
  - Non-zero mechanisms: $m = 3$.
  - Exact permutation p-value: $p = 0.50$.
  - Rejection of $H_0$: **False** ($p \ge 0.05$).
  - *Methodological Limitation:* With $m=3$, total possible permutations is $2^3 = 8$, yielding a minimum achievable two-sided p-value of $2/8 = 0.25$. Rejecting the null hypothesis at $\alpha = 0.05$ is mathematically impossible. The benchmark cannot claim superiority in resource burden.

### 2.3 Operational Minimality vs. Ground Truth Recovery
On task `inst_09_schema_unit_k4`:
- Both DD and Greedy discovered interacting 2-element subsets containing the planted trigger plus a distractor.
- Both subsets achieved strict operational 1-minimality under the Intersection–Union Test ($\max p_j^{\text{fisher}} = 0.0040 < 0.05$).
- Both subsets achieved 100% failure reproduction in independent holdout validation (5/5 failures, Wilson CI $[0.5655, 1.0000]$).
- Neither subset matched synthetic ground truth (`matches_ground_truth: False`).
- *Scientific Implication:* Black-box minimization isolates empirically sufficient failure conditions in neural agents, which can differ from human synthetic assumptions due to complex prompt interactions.

### 2.4 Empirical False Discovery on Null Controls
- 2 out of 4 null control instances (`inst_15` and `inst_16`) produced algorithmic discoveries isolating `department_code`.
- While operationally minimal and reproducible in `llama3.2:3b`, these represent an observed empirical false-discovery proportion of 50.0% across the 4 null task configurations (derived from 1 of the 2 underlying null mechanisms: `mech_null_02_db`), rather than an inferential statistical false discovery rate.

### 2.5 Holdout Stability
- 12 out of 12 candidate subsets across 5 resolved tasks were evaluated on 5 fresh random prompt seeds ($N_{\text{holdout}} = 60\text{ trials}$, 120 model requests).
- **100% reproduction:** All 12 candidates produced 5/5 failures in holdout testing (Wilson 95% CI $[0.5655, 1.0000]$).

### 2.6 Strict Operational Budgets
- **Total Model Requests Tracked:** 2,176 requests.
- **Total Completed Trials in `trials.jsonl`:** 1,086 trials (2,172 requests).
- **Total Financial Commercial Spend:** **$0.0000 USD** (100% local Ollama inference on `llama3.2:3b`).

---

## 3. Summary of Research Report Status

[`docs/research/phase_3_1/phase_3_1_agentxray_research_report.md`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/docs/research/phase_3_1/phase_3_1_agentxray_research_report.md) is fully affirmed and certified. Researchers and reviewers should cite the Phase 3.1 report as the primary comprehensive research report, supplemented by this Phase 3.2 addendum and the accompanying errata registry (`docs/research/phase_3_2/phase_3_2_errata_and_provenance.md`).
