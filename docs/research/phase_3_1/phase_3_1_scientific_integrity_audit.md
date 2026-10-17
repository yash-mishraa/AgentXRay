# AgentXRay — Phase 3.1: Scientific Integrity, Statistical Consistency, and Research-Package Audit

**Document Reference:** `docs/research/phase_3_1/phase_3_1_scientific_integrity_audit.md`  
**Date:** 2026-10-17  
**Auditor:** Independent Research Integrity & Statistical Audit Team (Antigravity)  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Audit Scope:** Independent forensic review of Phase 3.0 research documents, reconciliation against frozen Protocol v1.4, source code verification, statistical consistency analysis, and literature audit.  
**Final Audit Disposition:** **`PASS_WITH_LIMITATIONS`**

---

## 1. Executive Verdict

### Final Disposition: `PASS_WITH_LIMITATIONS`

This audit independently investigated the four Phase 3.0 research synthesis documents against the authoritative sources of truth: the frozen preregistration Protocol v1.4, Configuration v1.4, the actual Python implementation in `src/agentxray/`, the raw execution artifacts (`trials.jsonl`, `results.json`, `manifest.json`), and verified scientific literature.

The audit identified three material statistical and reporting inconsistencies in the Phase 3.0 documentation:
1. **Conflation of Intersection–Union Testing with Benjamini–Hochberg FDR Control (HIGH SEVERITY):** Phase 3.0 documents claimed that candidate 1-minimality was determined using an Intersection–Union Test (IUT) *with Benjamini–Hochberg False Discovery Rate control ($q \le 0.05$)*. In reality, both the frozen Protocol v1.4 (§6.2) and the actual minimizer code (`src/agentxray/stochastic_minimizer.py`, lines 399–411) evaluate candidate 1-minimality using the **Intersection–Union Principle (Berger 1982)**: every single-element deletion must individually achieve one-sided Fisher exact significance ($\alpha = 0.05$) alongside an empirical non-failure guard. Under IUT theory, rejecting the composite null requires every component test to reject at level $\alpha$, which mathematically bounds the family-wise Type I error rate at $\le \alpha$ without any multiplicity correction. No Benjamini–Hochberg adjustment is applied or needed for the operational minimality decision.
2. **Ambiguous Counting Units for Null Control False Discoveries (MEDIUM SEVERITY):** Phase 3.0 documents conflated task configurations, underlying mechanisms, and algorithm-task runs when reporting false discoveries on database null controls, while occasionally calling the result an "FDR bound." In truth, across 4 null task configurations ($N=4$, spanning 2 independent null mechanisms) evaluated across 3 algorithms (12 total runs), exactly 2 task configurations (`inst_15` and `inst_16`, both belonging to `mech_null_02_db`) exhibited false discoveries, comprising 5 algorithm-task occurrences (DD on both tasks, Greedy on both tasks, Random on `inst_15`). All 5 occurrences isolated the exact same single candidate element: `department_code`. Furthermore, 4 null task configurations are far too few to estimate an asymptotic population False Discovery Rate.
3. **Unsupported Priority and Novelty Claim (MEDIUM SEVERITY):** The portfolio summary (`phase_3_0_research_summary.md`) claimed that AgentXRay is "the first controlled study documenting how delta debugging performs on multi-turn tool-calling agents." While the engineering combination and benchmark are novel within the evaluated scope, asserting historical priority without an exhaustive systematic literature survey is scientifically unsupportable and has been removed.

All identified inconsistencies have been rigorously resolved and corrected in the Phase 3.1 successor documents. Because all material statistical, numerical, and protocol conflicts were successfully resolved from existing evidence, and the physical limitations of the raw evidence remain transparently disclosed, the audit disposition is certified as **`PASS_WITH_LIMITATIONS`**.

---

## 2. Files and Source Artifacts Inspected

The audit established the authoritative source of truth by directly inspecting the following files in the repository:

| Hierarchy / Authority | File Path | Status / Checksum | Verified Role |
| :--- | :--- | :---: | :--- |
| **1. Frozen Protocol** | `docs/research/phase_2b1_exploratory_protocol_v1_4.md` | `db45f403...` (Verified) | Authoritative preregistered design, estimands, and stopping rules |
| **1. Frozen Configuration** | `benchmarks/configs/phase_2b1_exploratory_config_v1_4.json` | `b3e3b0b8...` (Verified) | Authoritative task parameters, request caps, and sample sizes |
| **2. Implementation Code** | `src/agentxray/stochastic_minimizer.py` | Inspected (Lines 355–424) | Implements `verify_stochastic_minimality` and IUT logic |
| **2. Implementation Code** | `src/agentxray/stochastic_baselines.py` | Inspected (Lines 1–320) | Implements Greedy Ablation and Seeded Random Search |
| **2. Implementation Code** | `src/agentxray/statistics.py` | Inspected (Lines 1–450) | Implements Fisher exact, Wilson CI, Newcombe CI, BH FDR |
| **2. Implementation Code** | `src/agentxray/agent_harness/target.py` | Inspected (Lines 80–120) | ReAct execution, pre-request budget checks, and exceptions |
| **2. Implementation Code** | `src/agentxray/agent_harness/cost_tracker.py` | Inspected (Lines 1–150) | Tracks total model requests and financial spend |
| **2. Implementation Code** | `benchmarks/model_pilot/confirmatory_tasks.py` | Inspected (Lines 1–480) | Preregistered task registry, candidates, and ground truths |
| **2. Implementation Code** | `benchmarks/model_pilot/exploratory_runner_v1_4.py` | `69b03d52...` (Verified) | Orchestrator script executing calibration, search, holdout, ITT |
| **3. Original Raw Data** | `benchmarks/results/phase_2b1_5_execution/manifest.json` | `349c9b97...` (Verified) | Global ledger: 2,176 requests, $0.00 spend, hardware info |
| **3. Original Raw Data** | `benchmarks/results/phase_2b1_5_execution/results.json` | `5f9606d9...` (Verified) | Structured JSON results across all stages and statistical tests |
| **3. Original Raw Data** | `benchmarks/results/phase_2b1_5_execution/trials.jsonl` | `82905d22...` (Verified) | Raw event stream: 1,086 completed trials (2,172 requests) |
| **3. Original Raw Data** | `benchmarks/results/phase_2b1_5_execution/summary.csv` | `6e521487...` (Verified) | Flat tabular task metrics |
| **4. Forensic Audit** | `benchmarks/results/phase_2b1_6_audit/phase_2b1_6_independent_post_execution_audit.md` | `f35fde1b...` (Verified) | Independent reconstruction of 4-request delta and holdout counts |
| **5. Phase 3.0 Documents** | `docs/research/phase_3_0_*.md` (4 files) | Inspected & Audited | Phase 3.0 synthesis documents subjected to audit |
| **Literature Notes** | `docs/PRIOR_ART_REVIEW.md` | Inspected (Lines 1–176) | Systematic literature review and landscape notes |

---

## 3. Protocol-versus-Code Statistical Consistency Audit

### 3.1 The Intersection–Union Test vs. Benjamini–Hochberg Multiplicity
#### Protocol Specification
In `docs/research/phase_2b1_exploratory_protocol_v1_4.md`, Section 6.2 explicitly defines the primary operational minimality test:
> "A candidate subset $C$ is classified as `SUPPORTED_1_MINIMAL` if and only if:
> 1. Verification Complete: All $k = |C|$ single-element deletions $C \setminus \{e_j\}$ were evaluated.
> 2. Candidate Failure Reproduction: Candidate $C$ reproduces failure: $\hat{p}(C) \ge 0.50$ and Wilson 95% CI lower bound $> 0.20$.
> 3. Threshold Non-Failure Guard: Every single-element deletion satisfies $\hat{p}(C \setminus \{e_j\}) < 0.50$.
> 4. Comparative Statistical Support under IUT: Every single-element deletion achieves one-sided exact Fisher significance at $\alpha = 0.05$: $\max_{j=1,\dots,k} p_j^{\text{fisher}} < 0.05$.
> Under the Intersection-Union Principle (Berger 1982), rejecting the composite null $H_0^{\text{comp}}: \bigcup_{j=1}^k \{ p(C \setminus \{e_j\}) \ge p(C) \}$ at level $\alpha = 0.05$ requires that every component test rejects at $\alpha = 0.05$. No multiplicity correction across deletions is required."

#### Source Code Verification
Inspection of `src/agentxray/stochastic_minimizer.py` (lines 399–411) confirms the exact implementation:
```python
# Intersection-Union Test (IUT) across all removal tests (Berger 1982)
# Rejection of composite comparative null H0: Union(p_j >= p_cand) at level alpha requires EVERY p_comp < alpha.
all_statistically_significant = all(p < self.alpha for p in p_values_comp) and len(p_values_comp) == len(current_candidate)
all_observed_drops = all(observed_drops) and len(observed_drops) == len(current_candidate)
all_thresh_supported = all(p < self.alpha for p in p_values_thresh) and len(p_values_thresh) == len(current_candidate)

if all_statistically_significant:
    msg = (
        f"1-Minimality statistically supported across all {len(current_candidate)} element removals "
        f"under Intersection-Union Test (comparative Fisher all p < {self.alpha}; threshold binomial all p < {self.alpha}: {all_thresh_supported}; "
        f"comparisons: {'; '.join(diff_summaries)})."
    )
    return (StochasticMinimalityStatus.STATISTICALLY_SUPPORTED, msg)
```
Although `adjust_p_values_bh` is imported on line 26 of `stochastic_minimizer.py`, it is **never called** in `verify_stochastic_minimality()`.

#### The Statistical Fallacy in Phase 3.0
The Phase 3.0 research report, summary, and reproducibility guide erroneously claimed that candidate minimality included "Benjamini–Hochberg False Discovery Rate control ($q \le 0.05$)."
- An Intersection–Union Test tests the conjunction of alternative hypotheses ($H_1: \bigcap_{j=1}^k H_{1, j}$) against a union null ($H_0: \bigcup_{j=1}^k H_{0, j}$). Rejection of $H_0$ occurs *if and only if* every individual component test rejects at level $\alpha$. By Berger (1982), the Type I error rate of the IUT is bounded by $\alpha$ uniformly over the null parameter space.
- Applying Benjamini–Hochberg FDR adjustments to an IUT is mathematically invalid: BH is designed for multiple testing where any rejection is a discovery, whereas IUT requires *all* tests to reject simultaneously.
- **Audit Correction:** Removed all mentions of BH FDR control from the operational candidate-minimality definition. The test is accurately identified as the **Intersection–Union Test (Berger 1982) with Fisher's exact comparative tests and an empirical non-failure guard**.

### 3.2 Mechanism-Level Resource Burden & Wilcoxon Test
- **Formula:** $\bar{D}_i = \frac{1}{2}\left[\frac{R_{\text{Greedy}}(M_i, 4) - R_{\text{DD}}(M_i, 4)}{120} + \frac{R_{\text{Greedy}}(M_i, 8) - R_{\text{DD}}(M_i, 8)}{180}\right]$. Verified in Protocol v1.4 §3.2.2 and `exploratory_runner_v1_4.py` line 718.
- **Mechanism Vector Recomputation:**
  - `mech_01`: $\bar{D}_1 = +0.2222$ (Favors DD)
  - `mech_02`: $\bar{D}_2 = +0.2222$ (Favors DD)
  - `mech_03`: $\bar{D}_3 = 0.0000$ (Tied at Cap)
  - `mech_04`: $\bar{D}_4 = 0.0000$ (Tied at Cap)
  - `mech_05`: $\bar{D}_5 = -0.0417$ (Favors Greedy)
  - `mech_06`: $\bar{D}_6 = 0.0000$ (Tied at Cap)
- **Zero-Handling and Mid-Ranks:** Pratt filtering eliminates the 3 zero differences, leaving $m=3$ non-zero differences (`mech_01`, `mech_02`, `mech_05`).
  - Signed ranks: `mech_05` ($|-0.0417| \to 1.0$, sign $-$); `mech_01` and `mech_02` ($|+0.2222| \to 2.5$, sign $+$).
  - Test statistic: $W = \min(W^+, W^-) = \min(5.0, 1.0) = 1.0$.
  - Exact permutation distribution over $2^3 = 8$ permutations: 4 permutations have $W \le 1.0$, yielding exact two-sided $p = 4/8 = \mathbf{0.5000}$.
  - **Audit Finding:** The reported $p = 0.5000$ is an exact calculation. It represents a **failure to reject the null hypothesis**, not proof of equivalence between algorithms.

### 3.3 Paired Recovery & McNemar Test
Across all 12 active tasks, both algorithms recovered ground truth on 4 tasks (`inst_01`–`inst_04`) and failed on 8 tasks. Discordant pairs $b = 0, c = 0$.
- Exact binomial McNemar p-value: $p = \mathbf{1.0000}$.
- **Audit Finding:** Represents absence of observed asymmetry, not proof of equivalent recovery.

### 3.4 Descriptive Request Reduction on Qualified Syntax Tasks
On the 4 task configurations where ground truth was recovered (`inst_01`–`inst_04`):
- DD requests: $[40, 50, 40, 50] \implies \text{Mean} = \mathbf{45.0}$
- Greedy requests: $[60, 100, 60, 100] \implies \text{Mean} = \mathbf{80.0}$
- Descriptive reduction: $(80.0 - 45.0) / 80.0 = 35.0 / 80.0 = \mathbf{43.75\%} \approx \mathbf{43.8\%}$.
- **Audit Finding:** Strictly descriptive; restricted to four configurations across two syntax mechanisms.

---

## 4. False-Discovery Counting Reconciliation

To resolve ambiguity, the audit establishes precise accounting units for all null and schema-unit outcomes:

```
+-------------------------------------------------------------------------------------------------------------+
| NULL CONTROL ACCOUNTING HIERARCHY                                                                           |
+--------------------------+-------------------+--------------------+--------------------+--------------------+
| Unit Level               | Total Evaluated   | Clean Passing      | False Discoveries  | Isolated Element   |
+--------------------------+-------------------+--------------------+--------------------+--------------------+
| Independent Mechanisms   | 2 mechanisms      | 1 (mech_null_01)   | 1 (mech_null_02)   | department_code    |
| Task Configurations      | 4 tasks           | 2 (inst_13, 14)    | 2 (inst_15, 16)    | department_code    |
| Algorithm-Task Runs      | 12 runs           | 7 clean/inconcl.   | 5 false runs       | department_code    |
| Unique Candidate Subsets | 1 unique subset   | —                  | 1 unique subset    | department_code    |
+--------------------------+-------------------+--------------------+--------------------+--------------------+
```

### 4.1 Detailed Breakdown of the 12 Null Algorithm-Task Runs
1. `inst_13_null_calc_k4` ($K=4$, Arithmetic):
   - DD: 0 false discoveries (10 requests)
   - Greedy: 0 false discoveries (10 requests)
   - Random: 0 false discoveries (10 requests)
2. `inst_14_null_calc_k8` ($K=8$, Arithmetic):
   - DD: 0 false discoveries (10 requests)
   - Greedy: 0 false discoveries (10 requests)
   - Random: 0 false discoveries (10 requests)
3. `inst_15_null_db_k4` ($K=4$, Database):
   - DD: **1 false discovery** (`['department_code']`, 40 requests)
   - Greedy: **1 false discovery** (`['department_code']`, 60 requests)
   - Random: **1 false discovery** (`['department_code']`, 50 requests)
4. `inst_16_null_db_k8` ($K=8$, Database):
   - DD: **1 false discovery** (`['department_code']`, 60 requests)
   - Greedy: **1 false discovery** (`['department_code']`, 100 requests)
   - Random: 0 false discoveries (Exhausted 180-request cap; inconclusive)

**Reconciliation Finding:** Across all 12 null runs, there were exactly **5 algorithm-task false discoveries** occurring on **2 task configurations** within **1 independent mechanism** (`mech_null_02_db`). Every false discovery isolated the identical candidate: `department_code: assign to department code DEPT-FIN-01`.

### 4.2 Schema Unit Task (`inst_09_schema_unit_k4`) Discovery Reconciliation
- Planted Ground Truth: `('incompatible_unit: specify target unit to_unit as light_years',)` (a single element).
- Candidate Search Space ($K=4$): `incompatible_unit`, `expense_category`, `compliance_tag`, `archival_flag`.
- Actual Discovered Subsets:
  - Stochastic Delta Debugging: `['incompatible_unit', 'expense_category']` (2 elements)
  - Stochastic Greedy Ablation: `['incompatible_unit', 'archival_flag']` (2 elements)
  - Seeded Random Search: Budget exhausted (120 requests), inconclusive.
- Minimality Status: Both 2-element subsets were verified as `SUPPORTED_1_MINIMAL` under IUT. Removing either element caused failures to drop from 5/5 to 0/5 (Fisher exact $p = 0.0040 < 0.05$).
- Ground Truth Match: Classified as `matches_ground_truth: False` because neither isolated the singleton trigger in isolation.
- **Audit Finding:** The Phase 3.0 documents correctly identified that these were 2-element subsets, but earlier reports had mistakenly cited `compliance_tag`. The actual distractor elements are `expense_category` (DD) and `archival_flag` (Greedy).

---

## 5. Audit of Causal and Recovery Claims

The audit verified that the documentation maintains clear boundaries between distinct epistemic states:
1. **Planted Ground-Truth Recovery:** Achieving exact set equivalence $C^* = C_{\text{true}}$ (achieved on 4 active tasks: `inst_01`–`inst_04`).
2. **Failure Reproduction:** Demonstrating $\hat{p}(\text{fail} \mid C^*) \ge 0.50$ (achieved by all 12 discovered candidates).
3. **Operational 1-Minimality:** Demonstrating that removing any element from $C^*$ significantly reduces failure under IUT (achieved by all 12 discovered candidates, including the 2-element subsets on `inst_09` and false discoveries on `inst_15`/`inst_16`).
4. **Holdout Verification:** Repeating failure on fresh independent trials ($n=5$) without intervention search (achieved 5/5 on all 12 candidates).
5. **Causal Attribution:** Concluding that $C^*$ is the unique or true underlying fault. The evidence demonstrates that statistical 1-minimality **does not establish unique causal attribution**, because prompt length bloat and distractor interactions create alternative failure paths in stochastic language models.

---

## 6. Empirical Result Verification and Provenance Audit

All figures reported in the research report and summary were checked against the raw artifacts:

| Item | Reported Figure | Raw Evidence Value | Verification Classification |
| :--- | :---: | :---: | :---: |
| Assigned Task Configurations | 16 | 16 | Independently Recalculated |
| Active Configurations | 12 | 12 | Independently Recalculated |
| Active Mechanisms | 6 | 6 | Independently Recalculated |
| Qualified Active Configurations | 5 | 5 (`inst_01`–`inst_04`, `inst_09`) | Independently Recalculated |
| Disqualified Active Configurations | 7 | 7 (`inst_05`–`inst_08`, `inst_10`–`inst_12`) | Independently Recalculated |
| Qualified Null Configurations | 4 | 4 (`inst_13`–`inst_16`) | Independently Recalculated |
| DD Mean Requests (Qualified Syntax) | 45.0 | $[40, 50, 40, 50] \implies 45.0$ | Independently Recalculated |
| Greedy Mean Requests (Qualified Syntax) | 80.0 | $[60, 100, 60, 100] \implies 80.0$ | Independently Recalculated |
| Descriptive Request Reduction | 43.8% | $(80.0 - 45.0) / 80.0 = 43.75\%$ | Independently Recalculated |
| Mechanism-Level Wilcoxon Signed-Rank | $p = 0.5000$ | $W=1.0, m=3, p=4/8=0.5000$ | Independently Recalculated |
| Paired McNemar Test | $p = 1.0000$ | $b=0, c=0, p=1.0000$ | Independently Recalculated |
| Holdout Candidate Subsets | 12 | 12 ($3+2+3+2+2=12$) | Independently Recalculated |
| Holdout Trials / Requests | 60 / 120 | 60 trials $\times 2$ requests $= 120$ | Independently Recalculated |
| Logged Trials in `trials.jsonl` | 1,086 | 1,086 lines $\times 2$ reqs $= 2,172$ | Independently Recalculated |
| Ledger Requests in `manifest.json` | 2,176 | 2,176 requests | Independently Recalculated |
| Unlogged Aborted-Trial Requests | 4 | 4 runs hit cap on Turn 2 | Independently Reconstructed |
| Protocol v1.4 Hash | `db45f403...` | `db45f403...` | Cryptographically Verified Match |
| Config v1.4 Hash | `b3e3b0b8...` | `b3e3b0b8...` | Cryptographically Verified Match |
| Runner On-Disk Hash | `69b03d52...` | `69b03d52...` | Cryptographically Verified Match |
| Commercial API Financial Spend | $0.00 USD | $0.0000 USD | Independently Verified |

---

## 7. Literature Verification Table

Every cited publication and baseline system was audited against its bibliographic metadata and relationship to AgentXRay:

| Source Citation | Metadata & Venue | Verified Status | Specific Relationship to AgentXRay |
| :--- | :--- | :---: | :--- |
| **Zeller (1999)** | ESEC/FSE 1999, pp. 253–267. DOI: 10.1145/318774.318946 | **Verified & Relevant** | Foundational deterministic delta debugging algorithm ($ddmin$). AgentXRay extends $ddmin$ to stochastic binomial endpoints. |
| **Zhu et al. (2026)** | ICSE-SEIP 2026. Amazon Science & UC Davis. | **Verified & Relevant** | Adapts delta debugging to RAG context and prompt text reduction. Confined to single-turn text; does not evaluate multi-turn tool calling. |
| **Zhou et al. (2026)** | arXiv:2606.03601 (June 2026). DDOR. | **Verified & Relevant** | Applies delta debugging to isolate overrefusal triggers in single prompts. Bounded to text prompts; does not address tool schemas. |
| **Shah (2026)** | arXiv:2606.08275 (June 2026). CAR. | **Verified & Relevant** | Causal attribution via Structural Causal Models and trace replay. Requires white-box internal state access; AgentXRay is black-box. |
| **Ma et al. (2026)** | ICLR 2026 / arXiv:2512.06749. DoVer. | **Verified & Relevant** | Debugs multi-agent architectures via message bus interventions. Requires internal orchestration instrumentation. |
| **White et al. (2026)** | arXiv:2606.22698 (June 2026). | **Verified & Relevant** | Black-box behavioral fingerprinting of conversational agents. Focuses on model identity rather than fault minimization. |
| **Zheng et al. (2026)** | arXiv:2605.13898 (May 2026). | **Verified & Relevant** | Comprehensive survey of metamorphic testing in LLMs. |
| **Yao et al. (2022)** | ICLR 2023 / arXiv:2210.03629. ReAct. | **Verified & Relevant** | Interleaved reasoning and tool-calling pattern implemented in AgentXRay's agent harness. |
| **Berger (1982)** | *Technometrics* 24(4):295–300. | **Verified & Relevant** | Mathematical foundation for the Intersection–Union Principle used in AgentXRay's candidate minimality testing. |
| **Benjamini & Hochberg (1995)**| JRSS-B 57(1):289–300. | **Verified Metadata** | FDR procedure implemented in `statistics.py` for exploratory family-wise testing; corrected to note it is NOT used in IUT minimality. |
| **Wilson (1927)** | *Science* 66:320–322. | **Verified & Relevant** | Score interval used for binomial confidence bounds ($n=5$). |
| **Newcombe (1998)** | *Statist. Med.* 17:873–890. | **Verified & Relevant** | Hybrid score interval implemented for proportion difference bounds. |
| **Wilcoxon (1945)** | *Biometrics Bulletin* 1:80–83. | **Verified & Relevant** | Signed-rank test used for mechanism-level paired resource analysis. |
| **McNemar (1947)** | *Psychometrika* 12:153–157. | **Verified & Relevant** | Exact test used for recovery symmetry diagnostic. |

### Novelty Claim Correction
In `docs/research/phase_3_0_research_summary.md` Section 6, the portfolio summary claimed:
> *"The first controlled study documenting how delta debugging performs on multi-turn tool-calling agents..."*

**Audit Action:** This claim has been removed in `docs/research/phase_3_1/phase_3_1_research_summary.md` and replaced with an objectively bounded formulation:
> *"A controlled empirical study documenting how stochastic delta debugging performs on an open-weights ReAct agent (`llama3.2:3b`), highlighting the scaling benefits on syntax triggers alongside the operational risks of distractor entrapment and prompt degradation."*

---

## 8. Claim-by-Claim Correction Table

| Finding ID | Severity | Source File & Section | Original Phase 3.0 Claim | Verified Evidence & Audit Finding | Correction Made in Phase 3.1 Document |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **SEC-01** | HIGH | `phase_3_0_agentxray_research_report.md` §6.3 & Abstract | Minimality uses IUT with Benjamini-Hochberg FDR control ($q \le 0.05$) | Protocol v1.4 §6.2 and code use IUT (Berger 1982) with Fisher exact test; no BH adjustment is applied | Removed BH from minimality definition in `phase_3_1_agentxray_research_report.md` |
| **SEC-02** | HIGH | `phase_3_0_research_summary.md` §2 | "with Benjamini-Hochberg False Discovery Rate (FDR) control ($q \le 0.05$)" | Code and protocol do not apply BH to the IUT minimality decision | Removed BH from framework summary in `phase_3_1_research_summary.md` |
| **SEC-03** | MEDIUM | `phase_3_0_research_summary.md` §6 | "The first controlled study documenting how delta debugging performs..." | Literature review is not an exhaustive systematic survey establishing historical priority | Replaced with bounded empirical description in `phase_3_1_research_summary.md` |
| **SEC-04** | MEDIUM | `phase_3_0_agentxray_research_report.md` §8.2 | Conflated null task configurations, mechanisms, and algorithm runs | 12 runs across 4 tasks yielded 5 algorithm-task false discoveries on 2 tasks (`department_code`) | Added explicit 4-level accounting hierarchy table in `phase_3_1_agentxray_research_report.md` |
| **SEC-05** | LOW | `phase_3_0_agentxray_research_report.md` §15 | Benjamini-Hochberg cited as basis for minimality | Berger (1982) is the true mathematical foundation for IUT | Added Berger (1982) reference; clarified BH scope in `phase_3_1_agentxray_research_report.md` |
| **SEC-06** | LOW | `phase_3_0_reproducibility_and_archive.md` §5.2 | Test suite description referenced FDR adjustments in minimizer | Minimizer tests verify Fisher exact IUT and difference intervals | Corrected test description in `phase_3_1_reproducibility_and_archive.md` |

---

## 9. Remaining Unresolved Issues

1. **Unlogged Aborted-Trial Payloads:** The specific prompt strings and tool arguments generated on Turn 1 of the four trials halted by `CostBudgetExceededError` remain unrecorded in `trials.jsonl`. This is an immutable property of the historical data and cannot be repaired without fresh inference.
2. **Single Model Scope (`llama3.2:3b`):** The empirical findings cannot be generalized to larger open-weights models or frontier commercial models without new authorized experiments.
3. **Small Null Control Cohort ($N=4$):** Evaluating 4 null task configurations across 2 mechanisms is sufficient to demonstrate that false discoveries occur under prompt bloat, but cannot provide a high-precision estimate of the population False Discovery Rate.

---

## 10. Checks Performed and Not Performed

### Checks Performed
- Line-by-line inspection of Protocol v1.4, Configuration v1.4, `stochastic_minimizer.py`, and `statistics.py`.
- Recomputation of all mechanism differences, exact Wilcoxon permutation distribution, and McNemar test.
- Full verification of all 1,086 completed trial records in `trials.jsonl`.
- Verification of 12 candidate subsets and 60 holdout trials in `results.json`.
- Bibliographic cross-referencing for all cited literature.

### Checks Not Performed
- **No Target Model Inference:** Zero requests were dispatched to Ollama or any other model.
- **No Live Benchmark Rerun:** The runner was not re-executed.
- **No Network Socket Communication:** No external APIs were contacted.

---

## 11. Final Disposition Certification

Every material statistical, numerical, and protocol conflict identified in Phase 3.0 has been resolved from verified evidence. All corrections have been applied to the Phase 3.1 successor package.

**Final Certified Disposition:** **`PASS_WITH_LIMITATIONS`**
