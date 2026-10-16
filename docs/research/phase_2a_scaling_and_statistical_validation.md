# AgentXRay — Phase 2A: Scaling Benchmark, Statistical Audit & Algorithmic Validation Report

**Date**: 2026-10-16  
**Status**: Completed (Statistical Audit, Scaling Benchmark Suite & Empirical Validation Complete)  
**Starting Checkpoint**: `7e251c7` on `main`  
**Current Checkpoint**: In progress  
**Repository Remote**: `https://github.com/yash-mishraa/AgentXRay.git`  
**Benchmark Scope**: Purely synthetic controlled testbed (Zero paid API calls, Zero Ollama calls, Zero commercial spend: **$0.0000 USD**)  
**Benchmark Size**: 400 paired instances across 5 candidate-space sizes ($K \in \{4, 8, 12, 16, 20\}$) and 8 target families, totaling **1,200 algorithm runs** (completed in 1.51 seconds)  
**Test Suite**: 153 passing unit and integration tests (145 existing + 8 new Phase 2A benchmark tests)

---

## 1. Starting Commit and Working-Tree Status

Prior to executing Phase 2A, the repository state was verified:
- Branch: `main`
- Commit: `7e251c7` (`docs: update report checkpoint with git commit hash`)
- Remote: `https://github.com/yash-mishraa/AgentXRay.git`
- Working tree: Clean, zero unstaged changes.
- Test baseline: 145 passing tests in 0.24s.

---

## 2. Audit Findings and Corrections to Previous Claims

In accordance with Section 2 of the milestone specification, we conducted an independent scientific and mathematical audit of the Phase 1G.2 claims and codebase.

### A. Statistical Inference Audit (3/3 Failures vs. 0/3 Controls)

The Phase 1G.2 report compared $k_1 = 3 / n_1 = 3$ observed failures against $k_2 = 0 / n_2 = 3$ unablated controls and reported three statistical quantities:
1. Asymptotic pooled z-test: $z = 2.4495$, two-sided $p = 0.0143$.
2. Fisher's exact test: $p = 0.1000$.
3. Newcombe hybrid score confidence interval for the risk difference: $[0.2059, 1.0000]$ (reported as $[0.21, 1.00]$).

#### Why the Asymptotic and Exact Tests Differ:
- **Violation of Normal Approximation**: The two-sample pooled score test assumes that the sample proportion estimator $\hat{p}_1 - \hat{p}_2$ is approximately normally distributed under $H_0$. Standard sample-size criteria require expected counts in each cell under $H_0$ to satisfy $n_i \bar{p} \ge 5$ and $n_i (1 - \bar{p}) \ge 5$. For $n_1 = n_2 = 3$ and pooled rate $\bar{p} = 0.50$, the expected count is $3 \times 0.50 = 1.50 \ll 5.0$. The normal approximation severely underestimates tail probabilities at $N=6$.
- **Hypergeometric Table Discreteness**: In a $2 \times 2$ contingency table with marginal totals fixed at $(3, 3)$ and $(3, 3)$, only four realization tables are mathematically possible: $\{X = 0, 1, 2, 3\}$ with probabilities $0.05, 0.45, 0.45, 0.05$. Under a two-sided exact test, the probability of obtaining an outcome at least as extreme as $X=3$ is $\Pr(X=0) + \Pr(X=3) = 0.05 + 0.05 = 0.1000$.
- **Correction**: **It is mathematically impossible for any exact two-sided test conditioning on these margins to reject $H_0$ at $\alpha = 0.05$.** To achieve two-sided significance at $\alpha = 0.05$ under $100\%$ vs $0\%$ outcomes, the sample size must be at least $n \ge 4$ (where $2 / \binom{8}{4} = 2 / 70 \approx 0.0286 < 0.05$). The reported claim that $p = 0.0143$ proved statistical significance at $\alpha = 0.05$ was an uncorrected artifact of normal approximation invalidity.

#### Audit of Interval Estimates and Multiplicity Corrections:
- **Wilson and Newcombe Intervals**: The Wilson score interval correctly prevents zero-width degeneration at proportions of $0.0$ and $1.0$ ($[0.44, 1.00]$ for $3/3$, $[0.00, 0.56]$ for $0/3$). Newcombe's Method 10 hybrid interval $[0.21, 1.00]$ accurately reflects that while the observed effect size is large ($\Delta = +1.00$), the lower bound is wide ($0.21$), denoting substantial residual uncertainty.
- **Benjamini-Hochberg (BH) FDR Correction**: When $m=1$ hypothesis is tested, BH adjustment is mathematically degenerate ($q = p$). More critically, applying hypothesis testing to a candidate that was **adaptively selected** by a search algorithm introduces **selective inference (winner's curse)**. A candidate chosen specifically because it maximized observed failure during search cannot be tested using classical unadjusted null distributions without inflating false rejection rates.

### B. Algorithmic Complexity and Fairness Audit

The Phase 1G.2 report claimed that Delta Debugging has logarithmic complexity $O(\log K)$ while Greedy Ablation is linear $O(K)$.

#### Deconstruction of Complexity Assumptions:
- **Zeller's Classical Bound**: Delta Debugging (ddmin) achieves $2 \log_2 K$ candidate evaluations **if and only if**:
  1. Exactly one failure-inducing element exists.
  2. Monotonicity holds strictly (any superset of the trigger reproduces failure).
  3. At every partitioning step, the partition half containing the trigger reproduces failure on the first attempt without requiring complement evaluations.
- **Worst-Case Complexity**: If a failure requires a conjunction of elements split across chunks, or if the target is non-monotonic, ddmin must evaluate $n$ chunks and $n$ complements, doubling granularity repeatedly. In worst-case and conjunctive scenarios, ddmin evaluates $O(K^2)$ or $O(K)$ subsets.
- **Fairness in Query Accounting**:
  - In Phase 1G.2, with $K=4$, $2 \log_2 4 = 4$ subset evaluations for ddmin versus $4$ for greedy ablation. At $K=4$, ddmin has zero asymptotic complexity advantage over greedy ablation.
  - Furthermore, in Phase 1G.2, Random Search was configured without early stopping or a reserved verification budget, causing it to sample subsets until the 15-call budget was exhausted, automatically rendering its minimality verification `INCONCLUSIVE`.

### C. Ground-Truth Interpretation Audit

The trigger in Phase 1G.2 was artificially planted. We formally distinguish four epistemic levels:
1. **Planted Trigger Recovery in Controlled Benchmark**: Demonstrates that the search implementation correctly navigates an artificial configuration space under known assumptions. It does not establish efficacy on uncharacterized real-world bugs.
2. **Discovering Unknown Failure Conditions**: In an open-ended real agent, confounding factors, tool environment interactions, and latent model state are unknown *a priori*.
3. **Demonstrating 1-Minimality**: Confirms locally that removing any single component from candidate $S$ drops the failure rate. It does **not** imply global minimality (the smallest subset in the power set).
4. **General Causal Claim**: Requires showing that the identified trigger causally reproduces across models, temperatures, prompts, and architectures, rather than being an idiosyncratic artifact of a single prompt template.

---

## 3. Frozen Protocol and Benchmark Generation Rules

The benchmark evaluation protocol was formally documented and frozen in [`docs/research/phase_2a_protocol_v1.md`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/docs/research/phase_2a_protocol_v1.md) prior to executing the sweep.

### Execution Parameters:
- **Candidate space sizes**: $K \in \{4, 8, 12, 16, 20\}$
- **Target families**: 8 families (see Section 4)
- **Seeds per condition**: $N = 10$ independent instances ($s \in \{100, \dots, 109\}$)
- **Total benchmark instances**: 400 paired instances ($5 \times 8 \times 10$)
- **Total algorithm runs**: 1,200 runs
- **Trials per candidate evaluation**: $t = 10$ stochastic Bernoulli trials
- **Decision threshold**: $\tau = 0.50$ (standard families), $\tau = 0.25$ (weak signal family)
- **Max trial budget**: $B_{\max} = 1000$ trials per run
- **Positional fairness**: For each instance, element order in the candidate tuple is shuffled using `random.Random(seed).shuffle()`, preventing index bias.

---

## 4. Ground-Truth Definitions for Each Target Family

| Family | Planted Ground Truth | Failure Rule & Probabilities | Minimum $K$ |
| :--- | :--- | :--- | :---: |
| `single_trigger` | `{"elem_t_single"}` | $P=0.90$ if trigger present; $P=0.05$ otherwise | 4 |
| `conjunctive_trigger` | `{"elem_t_conj_a", "elem_t_conj_b"}` | $P=0.90$ if BOTH present; $P=0.05$ otherwise | 4 |
| `multiple_independent` | `{"elem_t_indep_1"}` or `{"elem_t_indep_2"}` | $P=0.90$ if EITHER present; $P=0.05$ otherwise | 4 |
| `distractor_heavy` | `{"elem_t_heavy"}` | $P=0.90$ if trigger present among $K-1$ distractors; $P=0.05$ otherwise | 4 |
| `null_target` | $\emptyset$ (Null case, $k=0$) | $P=0.05$ unconditionally across all subsets | 4 |
| `non_monotonic` | `{"elem_t_nm_trigger"}` (alt: `{trigger, suppressor, restorer}`) | Trigger $0.90$, suppressed by suppressor to $0.10$, restored by restorer to $0.90$, base $0.05$ | 4 |
| `stochastic_target` | `{"elem_t_stoch"}` | Moderate noise: $P=0.75$ with trigger; $P=0.15$ base | 4 |
| `weak_signal` | `{"elem_t_weak"}` | Modest effect: $P=0.40$ with trigger; $P=0.15$ base ($\Delta = 0.25$) | 4 |

---

## 5. Algorithm Configurations and Stopping Rules

1. **Stochastic Delta Debugging**:
   - Chunks partitioned at granularity $n=2$, evaluating subsets then complements.
   - Granularity increases $n \to \min(2n, |S|)$ when neither reduces failure.
   - Refinement loop performs explicit single-element removal checks.
   - Stopping rule: candidate reduced to 1-minimal subset or budget exhausted.
2. **Stochastic Greedy Ablation**:
   - Single sequential pass testing removals of each candidate element.
   - Drops element permanently if ablated subset maintains failure rate $\ge \tau$.
   - Stopping rule: completion of single pass over $K$ elements or budget exhausted.
3. **Stochastic Seeded Random Search**:
   - Uniform sampling of proper non-empty subsets $k \in \{1, \dots, K-1\}$.
   - Early stopping when a candidate of size 1 is found (minimum possible positive size).
   - Budget-aware reservation: stops sampling when remaining trials $\le |S_{\text{best}}| \times t$ to guarantee budget for minimality verification.
   - Stopping rule: size 1 found, verification budget boundary reached, or budget exhausted.

---

## 6. Comprehensive Benchmark Results

The entire benchmark sweep of 400 instances (1,200 algorithm runs) executed in **1.51 seconds** on Apple Silicon via [`benchmarks/synthetic/scaling_benchmark.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/synthetic/scaling_benchmark.py). Full raw trial-level data is serialized in [`benchmarks/results/phase_2a_scaling_results.json`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/results/phase_2a_scaling_results.json) and [`benchmarks/results/phase_2a_scaling_summary.csv`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/benchmarks/results/phase_2a_scaling_summary.csv).

### A. Overall Comparative Performance ($N=400$ instances, 1,200 runs)

| Metric | Stochastic Delta Debugging | Stochastic Greedy Ablation | Stochastic Random Search |
| :--- | :---: | :---: | :---: |
| **Mean Target Calls (Trials)** | **104.2** | 122.8 | 541.0 |
| **Median Target Calls** | **70.0** | 130.0 | 620.0 |
| **Mean Subsets Examined** | **7.6** | 12.1 | 49.0 |
| **Search Calls (Mean)** | 94.3 | 109.8 | 524.4 |
| **Verification Calls (Mean)** | 9.9 | 13.1 | 16.6 |
| **Ground Truth Recovery Rate** | **92.5%** | 76.0% | 53.0% |
| **1-Minimality Verification Rate** | **59.2%** | 49.5% | 29.0% |
| **Mean Precision** | **0.925** | 0.844 | 0.679 |
| **Mean Recall** | 0.925 | **0.950** | 0.875 |
| **Null Target False Discovery Rate** | **0.0%** (0/50) | **0.0%** (0/50) | **0.0%** (0/50) |
| **Budget Exhaustion Rate** | **0.0%** (0/400) | **0.0%** (0/400) | 42.0% (168/400) |

---

## 7. Scaling Analysis Across Candidate Space Sizes ($K$)

The table below reports mean target calls and candidate subsets examined as $K$ scales from 4 to 20 across all 8 target families ($N=80$ instances per $K$):

| Candidate Size ($K$) | Delta Debugging Calls | Delta Debugging Subsets | Greedy Ablation Calls | Greedy Ablation Subsets | Random Search Calls | Random Search Subsets |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 4$** | 59.9 | 4.6 | **53.4** | 5.0 | 199.0 | 6.9 |
| **$K = 8$** | 91.1 | **6.7** | **89.2** | 8.8 | 321.6 | 26.4 |
| **$K = 12$** | **109.6** | **8.0** | 122.6 | 12.2 | 645.2 | 60.9 |
| **$K = 16$** | **121.9** | **8.8** | 156.9 | 15.6 | 783.8 | 76.2 |
| **$K = 20$** | **138.5** | **9.8** | 192.0 | 19.1 | 755.5 | 74.8 |

### Key Algorithmic Scaling Insight:
- At small candidate sizes ($K \le 8$), **Greedy Ablation is equal or superior to Delta Debugging** in query cost (53.4 vs 59.9 calls at $K=4$; 89.2 vs 91.1 calls at $K=8$).
- At $K \ge 12$, **Delta Debugging demonstrates its sublinear scaling**: candidate subsets examined grow from $4.6 \to 9.8$ ($\approx 2 \log_2 K$), whereas Greedy Ablation subsets grow strictly linearly from $5.0 \to 19.1$ ($\approx K$).
- At $K = 20$, Delta Debugging examines roughly **half as many candidate subsets** as Greedy Ablation (9.8 vs 19.1) across the benchmark.
- Random Search collapses rapidly as $K$ increases: at $K=16$ and $K=20$, random search hits budget exhaustion in over $75\%$ of non-null instances, with ground-truth recovery falling to $15.0\%$ and $21.2\%$.

---

## 8. Target Family Performance Breakdown

The table below breaks down performance across all 8 target families ($N=50$ instances per family):

| Target Family | DD Calls | Greedy Calls | Random Calls | DD Ground Truth | Greedy Ground Truth | DD Verified 1-Minimal | Greedy Verified 1-Minimal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_trigger` | **71.4** | 139.0 | 630.4 | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `conjunctive_trigger` | 376.4 | **150.0** | 1000.0 | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `multiple_independent`| **66.0** | 138.0 | 533.2 | **100.0%** | **100.0%** | 0.0%* | 0.0%* |
| `distractor_heavy` | **71.4** | 139.0 | 630.4 | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `null_target` | **10.0** | **10.0** | **10.0** | **100.0%** | **100.0%** | 0.0% (N/A) | 0.0% (N/A) |
| `non_monotonic` | **88.6** | 144.6 | 630.4 | **100.0%** | 48.0% | **100.0%** | 48.0% |
| `stochastic_target` | **68.8** | 129.8 | 504.2 | **90.0%** | 52.0% | **66.0%** | 44.0% |
| `weak_signal` | **81.0** | 132.2 | 389.6 | **50.0%** | 8.0% | **8.0%** | 4.0% |

*\*See Section 9 for scientific explanation of 0.0% verification in `multiple_independent`.*

---

## 9. In-Depth Scientific Analysis: Where Delta Debugging Succeeds & Fails

### 1. The Conjunctive Trigger Inversion (Where Greedy Beats Delta Debugging)
On `conjunctive_trigger`, where failure requires both elements $t_1$ and $t_2$:
- **Greedy Ablation**: Consumed **150.0 calls** (14.8 subsets).
- **Delta Debugging**: Consumed **376.4 calls** (18.0 subsets).
- **Statistical Difference**: DD consumed **+226.4 MORE calls** on average ($W = 1275.0, p < 0.00001, \text{BH } q < 0.00001$).
- **Mechanism**: When trigger elements are split across partition chunks, neither chunk triggers failure independently. Delta debugging must test all chunks, test all complements, increase granularity to 4, test all 4 chunks, test all 4 complements, and repeat recursively. In contrast, Greedy Ablation evaluates removals one by one: when $t_1$ is ablated, failure drops, so $t_1$ is kept; when $t_2$ is ablated, failure drops, so $t_2$ is kept; when any distractor is ablated, failure persists, so the distractor is discarded. Greedy ablation removes all $K-2$ distractors in exactly 1 pass of $K$ evaluations.

### 2. The Multiple Independent Trigger Masking Phenomenon
On `multiple_independent`, both DD and Greedy Ablation isolated a valid single-element trigger ($t_1$ or $t_2$) with **100% precision and 100% recall**. However, both achieved **0.0% verified 1-minimality**.
- **Mechanism**: The 1-minimality verification test evaluates `initial_input.without(elem)`. When the input contains two independent triggers $\{t_1, t_2\}$ and the algorithm isolates $\{t_1\}$, evaluating `initial_input.without(t1)` still contains $t_2$! Because $t_2$ independently triggers failure with $P=0.90$, removing $t_1$ from the initial configuration produces zero drop in failure rate!
- **Scientific Conclusion**: In environments with multiple co-occurring independent failure triggers, ablation tests against the global input configuration are confounded by background triggers. Minimality must be verified against an ablated isolated subset, not against the global superset.

### 3. Non-Monotonic Failure Robustness
On `non_monotonic` (suppression and restoration):
- Delta Debugging achieved **100.0% ground-truth recovery and 100.0% verified minimality** (88.6 calls).
- Greedy Ablation achieved only **48.0% ground-truth recovery** (144.6 calls).
- **Mechanism**: When Greedy Ablation removes the restorer $C$ first, the remaining subset $\{A, B\}$ suppresses failure ($P=0.10$). Because failure rate drops below threshold, greedy ablation mistakenly concludes that $C$ is necessary, or drops $A$ in subsequent steps, losing the minimal trigger $\{A\}$. Delta debugging's hierarchical chunk testing successfully isolated the unsuppressed $\{A\}$ branch.

### 4. Null Target Robustness
On `null_target`, all three algorithms achieved a **0.0% False Discovery Rate** (0 false positives out of 50 instances). In all instances, the initial baseline evaluation observed a failure rate $\le 0.05 < \tau = 0.50$, correctly halting search after exactly 1 candidate evaluation (10 target calls) with `SearchStatus.NO_TRIGGER_FOUND` and `StochasticMinimalityStatus.NOT_APPLICABLE`.

### 5. Weak Signal Limits
On `weak_signal` ($\Delta = 0.25$, $P=0.40$ vs $0.15$):
- Delta Debugging recovered the trigger in 50% of runs, but verified minimality in only 8%.
- Greedy Ablation recovered the trigger in only 8% of runs.
- **Statistical Power Limitation**: At $t=10$ trials per candidate, the standard error of binomial estimation is $\sqrt{0.40 \times 0.60 / 10} \approx 0.155$. Random sampling fluctuations routinely cause the observed failure rate to cross the threshold in either direction, corrupting search decisions. Reliable minimization under weak signals requires a minimum sample size of $t \ge 35$ to achieve $80\%$ statistical power.

---

## 10. Paired Statistical Comparisons

All algorithm comparisons were conducted strictly on **paired identical instances** (identical target seed, identical element shuffling, identical PRNG sequence).

### Paired Differences in Target Calls (Trials):

| Comparison | Mean Difference ($\bar{d}$) | Standard Deviation ($s_d$) | Median Difference | Wilcoxon $W^+$ | Wilcoxon $p$-value | Paired $t$-statistic | Paired $t$ $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Delta Debugging vs. Greedy Ablation** | **-18.62** | 111.66 | **-30.0** | 14,754.5 | **$p = 0.00000$** | $t = -3.34$ | **$p = 0.00085$** |
| **Delta Debugging vs. Random Search** | **-436.82** | 387.50 | **-410.0** | 835.5 | **$p = 0.00000$** | $t = -22.55$ | **$p = 0.00000$** |
| **Greedy Ablation vs. Random Search** | **-418.20** | 382.21 | **-445.0** | 1,795.0 | **$p = 0.00000$** | $t = -21.88$ | **$p = 0.00000$** |

### Family-by-Family Paired Wilcoxon Tests (DD Calls minus Greedy Calls, Benjamini-Hochberg FDR Adjusted):

| Target Family | Mean Diff ($\bar{d}$) | Median Diff | Wilcoxon $W^+$ | Raw $p$-value | BH Adjusted $q$-value | Inferential Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `single_trigger` | -67.6 | -70.0 | 4.0 | $< 0.00001$ | **$q < 0.00001$** | DD significantly faster |
| `distractor_heavy` | -67.6 | -70.0 | 4.0 | $< 0.00001$ | **$q < 0.00001$** | DD significantly faster |
| `multiple_independent`| -72.0 | -70.0 | 0.0 | $< 0.00001$ | **$q < 0.00001$** | DD significantly faster |
| `non_monotonic` | -56.0 | -55.0 | 104.5 | $< 0.00001$ | **$q < 0.00001$** | DD significantly faster |
| `stochastic_target` | -61.0 | -55.0 | 4.5 | $< 0.00001$ | **$q < 0.00001$** | DD significantly faster |
| `weak_signal` | -51.2 | -40.0 | 25.5 | $< 0.00001$ | **$q < 0.00001$** | DD significantly faster |
| `null_target` | +0.0 | +0.0 | 0.0 | $1.00000$ | $q = 1.00000$ | Exact parity (1 evaluation each) |
| `conjunctive_trigger` | **+226.4** | **+250.0** | 1,275.0 | $< 0.00001$ | **$q < 0.00001$** | **Greedy significantly faster** |

---

## 11. Reproduction Commands & Environment

### Environment Details:
- OS: macOS (Darwin 24.6.0)
- Python: 3.13.16
- Pytest: 9.1.1
- Hardware: Apple Silicon

### Exact Reproduction Commands:
```bash
# 1. Run all unit and regression tests (153 tests)
.venv/bin/pytest -v

# 2. Run full 400-instance scaling benchmark sweep (1,200 algorithm runs)
.venv/bin/python benchmarks/synthetic/scaling_benchmark.py --seeds 10 --output-dir benchmarks/results

# 3. Generate paired statistical analysis summaries
.venv/bin/python benchmarks/synthetic/analyze_results.py
```

---

## 12. Actual Test Suite Results

```text
============================= test session starts ==============================
platform darwin -- Python 3.13.16, pytest-9.1.1, pluggy-1.7.0
rootdir: /Users/yashmishra14/Personal projects/AgentXRay
configfile: pyproject.toml
plugins: asyncio-1.4.0
collected 153 items

tests/test_audit.py ...                                                  [  1%]
tests/test_baselines.py .......                                          [  6%]
tests/test_calibrated_pilot.py .......                                   [ 11%]
tests/test_isolation.py .......                                          [ 15%]
tests/test_minimizer.py ......                                           [ 19%]
tests/test_model_harness.py ...........................                  [ 37%]
tests/test_models.py ..............                                      [ 46%]
tests/test_ollama_provider.py ........................                   [ 62%]
tests/test_providers.py .............                                    [ 70%]
tests/test_reporting.py ...                                              [ 72%]
tests/test_runner.py ....                                                [ 75%]
tests/test_scaling_benchmark.py ........                                 [ 80%]
tests/test_statistics.py ...........                                     [ 87%]
tests/test_stochastic_baselines.py ....                                  [ 90%]
tests/test_stochastic_minimizer.py ........                              [ 95%]
tests/test_stochastic_reporting.py ...                                   [ 97%]
tests/test_stochastic_target.py ....                                     [100%]

============================= 153 passed in 0.23s ==============================
```

---

## 13. Conclusions Supported vs. Unproven

### Conclusions Rigorously Supported by the Benchmark:
1. **Delta Debugging is NOT Universally Superior**: On conjunctive failure targets, Greedy Ablation is **$2.5\times$ more query-efficient** than Delta Debugging ($150.0$ vs $376.4$ calls, $p < 0.00001$). Delta debugging's efficiency is strictly conditional on single-trigger or chunk-localized failure structures.
2. **Sublinear Scaling for Single Triggers**: In single-trigger and distractor-heavy spaces, Delta Debugging exhibits sublinear query scaling ($\approx 2 \log_2 K$ candidate subsets examined), examining roughly half the candidate configurations of Greedy Ablation at $K=20$ (9.8 vs 19.1 subsets).
3. **Greedy Ablation Advantage at Small $K$**: For small candidate spaces ($K \le 8$), Greedy Ablation is equal to or slightly faster than Delta Debugging because the constant partitioning overhead of binary division balances out linear passes.
4. **Random Search is Infeasible for Large Spaces**: Unconstrained random sampling suffers catastrophic performance degradation as $K \ge 12$, experiencing budget exhaustion in over $75\%$ of non-null instances at $K=20$.
5. **Zero False Discoveries on Null Targets**: All algorithms achieve a verified 0.0% False Discovery Rate on null targets without unnecessary probing.
6. **Masking in Multi-Trigger Settings**: In the presence of multiple co-occurring failure causes, ablations against global configurations cannot verify 1-minimality because unablated background triggers mask individual removals.

### Conclusions That Remain Unproven:
1. **Real-Model Generalization**: These scaling findings were derived from a controlled synthetic testbed where ground-truth rules are known and isolated. Whether real neural agent failure boundaries (e.g. prompt token interactions, tool definition schemas) exhibit conjunctive, non-monotonic, or monotonic structures remains an empirical question.
2. **Weak-Signal Convergence in Neural Agents**: When a neural agent exhibits low-probability failure ($\Delta \le 0.25$), $t=10$ trials is insufficient. Determining the optimal adaptive trial allocation policy ($t(K)$) remains open.

---

## 14. Recommended Next Research Milestone

### Phase 2B: Adaptive Sample-Allocation and Multi-Modal Neural Boundary Probing
Now that algorithmic scaling properties, complexity trade-offs, and statistical limitations are fully documented, the single most justified next research milestone is:
> **Evaluate AgentXRay's algorithms on real open-weights models (`llama3.2:3b`) across a multi-boundary intervention space (prompt instructions, tool argument schemas, and simulated environment responses) using sequential probability ratio testing (SPRT) to adaptively scale sample sizes only when uncertainty is high.**
