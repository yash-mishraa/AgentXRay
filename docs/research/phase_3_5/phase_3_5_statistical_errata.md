# AgentXRay — Phase 3.5: Statistical Power and Experimental-Unit Errata

**Document Reference:** `docs/research/phase_3_5/phase_3_5_statistical_errata.md`  
**Date:** 2026-10-17  
**Version:** Phase 3.5 Statistical Errata Register  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Authoritative Power Audit:** [`docs/research/phase_3_5/phase_3_5_statistical_power_audit.md`](phase_3_5_statistical_power_audit.md)  
**Machine-Readable Companion:** [`docs/research/phase_3_5/phase_3_5_power_results.json`](phase_3_5_power_results.json)  

---

## 1. Executive Summary

This document establishes the official errata register for statistical power claims and experimental-unit designations in Project AgentXRay. Prior project reports repeatedly described statistical power at the mechanism level ($n=6$) as *"bounded by $\le 37.1\%$ for large effects ($d=1.25$)"*. 

An independent mathematical and computational audit demonstrates that:
1. **37.1% is not a mathematical bound:** It is a single Monte Carlo point estimate ($0.3715$) generated under specific distributional parameters ($\mu=15.0, \sigma=12.0, p_{\text{tie}}=0.05, n_{\text{sims}}=2000$). Under zero ties ($p_{\text{tie}}=0$), power under the exact same effect size is $51.9\%$, and under larger effects ($d=2.50$), power reaches $70.5\%$.
2. **The mathematical constraint is discrete, not numerical at 37.1%:** At two-sided $\alpha=0.05$ under Pratt zero-filtering, the exact signed-rank test on $n=6$ pairs can reject $H_0$ if and only if $m=6$ (zero ties) and $W=0$ (all 6 signs agree). If any tie occurs ($m \le 5$), power is identically **$0.0\%$**.
3. **Instance-level $N=12$ power ($94.8\%$) is pseudoreplicated:** The 12 active tasks represent 6 mechanisms evaluated at two distractor levels ($K=4$ and $K=8$). An unclustered analysis at $N=12$ assumes independent mechanisms, which is violated by shared prompts and failure triggers. $94.8\%$ is strictly a hypothetical sensitivity calculation and invalid for primary inference.

Per AgentXRay change-control policy, frozen specifications (Protocol v1.4, Configuration v1.4, and raw result files) remain unaltered. This errata document provides authoritative guidance for interpreting historical records and reporting future scientific publications.

---

## 2. Master Statistical Errata Register

| Erratum ID | Target Artifacts | Classification | Former Wording / Claim | Recommended Replacement Wording | Technical & Mathematical Rationale |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **`ERR-3.5-01`** | Protocol v1.4 §9.3<br>Phase 2B.1.4 Audit<br>Phase 3.3 Final Audit<br>Phase 3.4 Audit & Release Note | **Mathematical & Inferential** | *"Statistical power at $n=6$ is bounded by $\le 37.1\%$ even for large effects ($d=1.25$)."* | *"Under a preregistered normal model ($d=1.25$, $p_{\text{tie}}=0.05$), Monte Carlo estimated power at the mechanism level ($n=6$) is $37.15\%$ (MCSE $1.08\%$, $95\%$ CI $[35.03\%, 39.27\%]$). This is an empirical simulation estimate under specific modeling assumptions, not a mathematical upper bound."* | A finite Monte Carlo draw from 2,000 replications is not a bound. If $p_{\text{tie}}=0$, power is $51.9\%$; for larger effects, power exceeds $70\%$. Calling 37.1% a bound conflates simulation output with mathematical law. |
| **`ERR-3.5-02`** | Protocol v1.4 §9.2<br>Phase 2B.1.2 Audit<br>Phase 2B.1.3 Audit<br>Phase 3.4 Audit | **Inferential** | Unqualified citation of $94.8\%$ power at $N=12$ alongside $n=6$ results | *"The $N=12$ power figure ($94.85\%$, MCSE $0.49\%$) is strictly a hypothetical sensitivity calculation assuming 12 independent pairs. Because the 12 active tasks represent 6 mechanisms evaluated at $K=4$ and $K=8$, an unclustered $N=12$ test commits pseudoreplication. Primary inference is valid only at $n=6$."* | Prevents external readers from mistaking a pseudoreplicated sensitivity model for the actual statistical power of the benchmark. |
| **`ERR-3.5-03`** | Phase 3.4 Audit §4.5<br>Phase 3.4 Consistency JSON | **Mathematical Precision** | Unconditional statement that the rejection region contains only $W=0$ | *"For an exact two-sided Wilcoxon signed-rank permutation test at $\alpha=0.05$ with $m=6$ non-zero pairs, the rejection region contains strictly $W=0$ ($p=2/64=0.03125$). For any $m \le 5$, the minimum achievable two-sided p-value is $\ge 2/32=0.06250 > 0.05$, rendering rejection mathematically impossible."* | Clarifies the exact boundary conditions under which the discrete rejection constraint holds (two-sided $\alpha=0.05$, exact permutation test, Pratt filtering). |
| **`ERR-3.5-04`** | Phase 3.3 Final Audit §5.2<br>Phase 3.4 Release Note §5.3 | **Inferential & Epistemic** | Conflating observed post-experiment p-value ($p=0.50$ on $m=3$) with design power | *"Prospective power is a pre-experiment property under assumed effect sizes. The observed Wilcoxon outcome ($W=1, p=0.50$ on $m=3$) is a realized sample result. Failing to reject $H_0$ reflects small realized sample size and lack of observed separation, not proof of algorithmic equivalence."* | Enforces strict epistemic separation between design-stage power and post-experiment p-values, preventing the acceptance of the null fallacy. |

---

## 3. Detailed Errata Specifications

### 3.1 ERR-3.5-01: Monte Carlo Simulation Estimate Conflated as a Mathematical Bound
- **Affected Historical Files:**
  - `docs/research/phase_2b1_exploratory_protocol_v1_4.md` (§9.3, Item 1, line 211)
  - `docs/research/phase_2b1_4_final_consistency_audit.md` (lines 159, 240)
  - `docs/research/phase_3_3/phase_3_3_final_reproducibility_audit.md` (§5, Item 2, line 155)
  - `docs/research/phase_3_4/phase_3_4_documentation_integrity_audit.md` (§1, §4, lines 119, 155, 169)
  - `docs/research/phase_3_4/phase_3_4_consistency_check.json` (lines 100, 115)
  - `docs/research/phase_3_4/phase_3_4_release_note.md` (§2.1, Item 5)
- **Nature of Defect:** Mathematical and Inferential overstatement.
- **Former Formulation:** *"Statistical power at the mechanism level ($n=6$) is bounded by $\le 37.1\%$ even for large effects ($d=1.25$)."*
- **Corrected Formulation:** *"Under the preregistered simulation model ($\mu=15.0, \sigma=12.0, d=1.25, p_{\text{tie}}=0.05$), Monte Carlo estimated power at the mechanism level ($n=6$) is $37.15\%$ with Monte Carlo standard error $\text{MCSE} = 1.08\%$ ($95\%$ CI $[35.03\%, 39.27\%]$). This figure represents an empirical simulation estimate under specific modeling assumptions, not a mathematical ceiling."*
- **Technical Explanation:**
  In `benchmarks/model_pilot/power_and_budget_analysis.py`, `simulate_wilcoxon_power` was run with $n=6, \mu=15, \sigma=12, p_{\text{tie}}=0.05, n_{\text{sims}}=2000$, and seed 525. It observed 743 rejections out of 2000 draws ($743/2000 = 0.3715$).
  - If $p_{\text{tie}} = 0.0$ (no ties), power under $d=1.25$ increases to **$51.9\%$**.
  - If $\mu = 30.0$ ($d=2.50$), power reaches **$70.5\%$**.
  - If an effect were arbitrarily large ($d \to \infty$) and ties were zero, power would approach **$100\%$**.
  Therefore, 37.1% is a simulation estimate conditional on $\mu=15, \sigma=12$, and $p_{\text{tie}}=0.05$, not a mathematical upper bound.

---

### 3.2 ERR-3.5-02: Misinterpretation of Unclustered Instance-Level ($N=12$) Power
- **Affected Historical Files:**
  - `docs/research/phase_2b1_exploratory_protocol_v1_4.md` (§9.2, Table line 208)
  - `docs/research/phase_2b1_2_statistical_implementation_audit.md` (Table line 92)
  - `docs/research/phase_2b1_3_final_readiness_audit.md` (Table line 144)
  - `docs/research/phase_3_4/phase_3_4_documentation_integrity_audit.md` (§4.4, lines 147–152)
  - `docs/research/phase_3_4/phase_3_4_consistency_check.json` (line 114)
- **Nature of Defect:** Inferential ambiguity / Pseudoreplication risk.
- **Former Formulation:** Juxtaposing $94.8\%$ power ($N=12$) alongside $37.1\%$ power ($n=6$) without prominently stating that $N=12$ is an invalid inferential model.
- **Corrected Formulation:** *"The $N=12$ power calculation ($94.85\%$, MCSE $0.49\%$) is strictly an exploratory sensitivity model assuming 12 independent mechanisms. In the actual study, the 12 active tasks represent 6 mechanisms evaluated at two distractor levels ($K=4$ and $K=8$). Because configurations of the same mechanism share prompts and failure triggers, an unclustered $N=12$ analysis commits pseudoreplication. The primary inferential estimand is the mechanism-level difference ($n=6$); the $N=12$ power figure must not be cited as evidence that the primary study was adequately powered."*
- **Technical Explanation:**
  In `benchmarks/model_pilot/confirmatory_tasks.py`, `inst_01` ($K=4$) and `inst_02` ($K=8$) share `mech_01_syntax_currency`. Treating them as independent observations artificially doubles the apparent degrees of freedom while ignoring intra-cluster correlation. Protocol v1.4 §3.2.2 correctly designated the mechanism difference ($n=6$) as the primary test; hence, only $n=6$ power calculations reflect the primary design.

---

### 3.3 ERR-3.5-03: Mathematical Precision of the Discrete Rejection Region
- **Affected Historical Files:**
  - `docs/research/phase_3_4/phase_3_4_documentation_integrity_audit.md` (§4.5, lines 160–165)
  - `docs/research/phase_3_4/phase_3_4_consistency_check.json` (line 115)
- **Nature of Defect:** Mathematical Precision and Scope Boundary.
- **Former Formulation:** Stating without full qualification that $W=0$ is the only rejection state.
- **Corrected Formulation:** *"For an exact two-sided Wilcoxon signed-rank permutation test at significance level $\alpha=0.05$ with $m=6$ non-zero pairs, the rejection region contains strictly $W=0$ ($p = 2/64 = 0.03125$). The state $W=1$ yields $p = 4/64 = 0.06250 > 0.05$. If any mechanism produces a difference of zero, Pratt filtering reduces the sample size to $m \le 5$, where the minimum possible two-sided p-value is $\ge 2/32 = 0.06250 > 0.05$, making rejection of $H_0$ mathematically impossible at $\alpha=0.05$."*
- **Technical Explanation:**
  Dynamic programming over subset sums of $\{1, \dots, m\}$ demonstrates that the minimum two-sided p-value is $2 / 2^m$. For $m=5$, $2/32 = 0.06250 > 0.05$. For $m=4$, $2/16 = 0.1250 > 0.05$. Thus, whenever $m \le 5$, the rejection probability is exactly 0 under two-sided $\alpha=0.05$.

---

### 3.4 ERR-3.5-04: Separation of Prospective Design Power from Realized Empirical p-Values
- **Affected Historical Files:**
  - `docs/research/phase_3_3/phase_3_3_final_reproducibility_audit.md` (§5, Item 2)
  - `docs/research/phase_3_4/phase_3_4_release_note.md` (§5, Item 3)
- **Nature of Defect:** Inferential / Epistemic Conflation.
- **Former Formulation:** *"At the mechanism level ($n=6, m=3$), statistical power is bounded by $\le 37.1\%$ for large effects, and the Wilcoxon test yielded $p = 0.50$."*
- **Corrected Formulation:** *"Prospective statistical power evaluates the pre-experiment probability of rejecting $H_0$ across hypothetical replications of the design. The observed benchmark result ($p = 0.50$, $W = 1$ on $m = 3$ non-zero mechanisms) is an empirical sample statistic. The failure to reject $H_0$ is a consequence of severe sample attrition during calibration (leaving only $m=3$ evaluable mechanisms) and the lack of observed separation, and must never be interpreted as statistical evidence that stochastic delta debugging and greedy ablation achieve equivalent request burden."*
- **Technical Explanation:**
  Conflating sample size attrition ($n=6 \to m=3$) with design power obscures the distinction between design properties and observational outcomes. In small samples, failing to reject $H_0$ cannot support an equivalence claim without a prespecified equivalence margin and two one-sided tests (TOST), which Protocol v1.4 explicitly rejected.
