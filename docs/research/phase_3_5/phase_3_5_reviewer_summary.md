# AgentXRay — Phase 3.5: Reviewer and Author Guidance on Statistical Claims

**Document Reference:** `docs/research/phase_3_5/phase_3_5_reviewer_summary.md`  
**Date:** 2026-10-17  
**Version:** Phase 3.5 Peer Reviewer & Author Guidance Summary  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Authoritative Power Audit:** [`docs/research/phase_3_5/phase_3_5_statistical_power_audit.md`](phase_3_5_statistical_power_audit.md)  
**Machine-Readable Companion:** [`docs/research/phase_3_5/phase_3_5_power_results.json`](phase_3_5_power_results.json)  

---

## 1. Executive Briefing for Peer Reviewers and Authors

This document provides a concise, neutral guide for academic peer reviewers, conference program committees, and authors citing Project AgentXRay. It synthesizes the findings of the Phase 3.5 Statistical Power and Experimental-Unit Audit into actionable reporting guidelines, clarifying what the empirical evidence supports, what it does not support, and how statistical power should be accurately cited in publications.

---

## 2. What the Empirical Benchmark Supports

The completed Phase 2B.1.5 exploratory benchmark ($2,176$ model requests on local `llama3.2:3b`, $0.0000 USD commercial spend) provides defensible evidence for the following claims:

1. **Feasibility of Black-Box Failure Minimization:**
   Systematic black-box minimization can successfully isolate minimal failure-inducing prompt and tool subsets in tool-using agents without white-box gradient or activation access. All 12 candidate subsets evaluated in holdout validation reproduced $100\%$ failure rates ($5/5$ failures, Wilson $95\%$ CI $[0.5655, 1.0000]$) across $60$ out-of-sample prompt seeds.
2. **Descriptive Request Reduction on Syntax Tasks:**
   On the four qualified syntax task configurations (`inst_01` through `inst_04`), Stochastic Delta Debugging consumed $180$ total model requests compared to $320$ requests for Stochastic Greedy Ablation (a descriptive reduction of $43.8\%$).
3. **Absence of Observed Recovery Asymmetry:**
   Stochastic Delta Debugging and Stochastic Greedy Ablation exhibited identical ground-truth recovery counts across the $12$ active tasks ($4/12 = 33.3\%$). McNemar's exact test on paired recovery yielded discordant pairs $b = 0, c = 0$ ($p = 1.0$).
4. **Empirical Vulnerability of Negative Null Controls:**
   Across the four negative null task configurations, both Delta Debugging and Greedy Ablation isolated candidate subsets on $2$ of the $4$ configurations ($50.0\%$), both originating from the same database null mechanism (`mech_null_02_db`) and isolating the identical non-functional parameter (`department_code`).
5. **Deterministic Offline Reproducibility:**
   All aggregate observations, model request counts, candidate classifications, and statistical estimands are deterministically verifiable offline from preserved primary raw artifacts.

---

## 3. What the Empirical Benchmark Does NOT Support

To maintain scientific integrity, authors and reviewers must ensure that publications do not make any of the following claims:

1. **No Confirmatory Superiority Claim:**
   The benchmark does **not** prove that Stochastic Delta Debugging is statistically superior to Greedy Ablation in request efficiency. The primary mechanism-level Wilcoxon signed-rank permutation test yielded $W = 1, p = 0.50$ on $m = 3$ evaluable mechanisms, failing to reject $H_0$.
2. **No Claim of Algorithmic Equivalence:**
   The failure to reject the null hypothesis on request burden ($p = 0.50$) and the lack of observed recovery asymmetry (McNemar $p = 1.0$) do **not** demonstrate that the algorithms have equivalent efficiency or recovery profiles. Equivalence cannot be established from a non-significant test, particularly in an underpowered sample.
3. **No Unclustered Primary Inference at $N=12$:**
   An unclustered analysis treating the 12 active tasks as independent pairs commits **pseudoreplication** because $K=4$ and $K=8$ configurations share prompts, tools, and failure mechanisms. The primary inferential unit is the mechanism ($n=6$); the $N=12$ sample size is valid strictly for exploratory descriptions.
4. **No Frontier Model Generalization:**
   All empirical observations reflect local `llama3.2:3b` tool-calling dynamics. The study provides no direct empirical evidence regarding commercial frontier models (`gpt-4o`, `claude-3-5-sonnet`).

---

## 4. Citation and Reporting Guidance for Statistical Power

When describing AgentXRay's statistical power in research papers, follow these specific guidelines:

```
+-------------------------------------------------------------+-------------------------------------------------------------+
| DO NOT WRITE (Former / Incorrect Formulation)               | DO WRITE (Scientifically Defensible Formulation)            |
+-------------------------------------------------------------+-------------------------------------------------------------+
| "Statistical power at the mechanism level (n=6) was         | "Under a preregistered Monte Carlo model with mu=15,        |
| bounded by <= 37.1% for large effects (d=1.25)."            | sigma=12 (d=1.25) and a 5% tie probability, estimated power |
|                                                             | for the mechanism-level test (n=6) was 37.15% (MCSE 1.08%,  |
|                                                             | 95% CI [35.03%, 39.27%]). The study was an exploratory      |
|                                                             | feasibility benchmark with low prospective power."          |
+-------------------------------------------------------------+-------------------------------------------------------------+
| "The benchmark achieved 94.8% statistical power to detect   | "While an unclustered sensitivity model assuming 12         |
| request efficiency differences."                            | independent mechanisms yielded 94.85% power, the primary    |
|                                                             | analysis required mechanism-level clustering (n=6) to avoid |
|                                                             | pseudoreplication, resulting in 37.15% simulated power."    |
+-------------------------------------------------------------+-------------------------------------------------------------+
| "The Wilcoxon test yielded p=0.50, proving that Delta       | "The Wilcoxon signed-rank test on mechanism-level burden    |
| Debugging and Greedy Ablation have equivalent request cost."| yielded p=0.50 (W=1) on m=3 non-zero mechanisms, failing to |
|                                                             | reject H0. This reflects severe sample attrition and lack   |
|                                                             | of separation, and does not establish equivalence."         |
+-------------------------------------------------------------+-------------------------------------------------------------+
```

---

## 5. Concise Recommended Text for Research Manuscripts

Authors integrating AgentXRay into academic papers may adapt the following peer-reviewed-ready paragraph:

> *"We evaluated stochastic minimization as an exploratory feasibility benchmark across 16 task configurations representing 8 underlying causal mechanisms (6 active, 2 null) on a local 3B-parameter model. To avoid pseudoreplication from paired $K=4$ and $K=8$ distractor configurations sharing the same failure trigger, the primary efficiency estimand was evaluated at the independent mechanism level ($n=6$). Under our preregistered Monte Carlo power simulation ($\mu=15.0, \sigma=12.0, d=1.25, p_{\text{tie}}=0.05, 2{,}000\text{ replications}$), prospective power at the mechanism level was estimated at $37.15\%$ (MCSE $1.08\%$, $95\%$ CI $[35.03\%, 39.27\%]$). Because the exact Wilcoxon signed-rank test at $n=6$ has only $2^6 = 64$ permutation states, two-sided rejection at $\alpha=0.05$ strictly requires complete sign unanimity ($W=0$) and zero ties ($m=6$). In the empirical benchmark, pre-experiment calibration disqualified 7 active tasks, reducing the evaluable sample to $m=3$ non-zero mechanisms; the observed test statistic was $W=1$ ($p=0.50$). While Stochastic Delta Debugging achieved a descriptive $43.8\%$ request reduction on the four qualified syntax tasks, the benchmark does not claim confirmatory superiority or equivalence."*

---

## 6. Verification and Audit Artifact References

All underlying calculations, simulation source code, and mathematical proofs are permanently documented in the following repository files:
- **Full Statistical Power Audit:** [`docs/research/phase_3_5/phase_3_5_statistical_power_audit.md`](phase_3_5_statistical_power_audit.md)
- **Machine-Readable Parameters & Proofs:** [`docs/research/phase_3_5/phase_3_5_power_results.json`](phase_3_5_power_results.json)
- **Master Errata Register:** [`docs/research/phase_3_5/phase_3_5_statistical_errata.md`](phase_3_5_statistical_errata.md)
- **Simulation Code:** [`benchmarks/model_pilot/power_and_budget_analysis.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/model_pilot/power_and_budget_analysis.py)
