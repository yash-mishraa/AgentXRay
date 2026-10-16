# AgentXRay — Phase 2A: Scaling Benchmark Protocol (Version 1.0)

**Document Status**: Frozen Prior to Final Benchmark Execution  
**Version**: 1.0.0  
**Date**: 2026-10-16  
**Authors**: Antigravity Research Team  

---

## 1. Research Question & Formal Hypotheses

### Primary Research Question
> Under what conditions can stochastic delta debugging identify 1-minimal failure-inducing configurations efficiently, and how does its performance compare with greedy ablation and random search as the intervention space grows?

### Hypotheses

- **H1 (Single-Trigger Scaling Advantage)**: In target families with a single monotonic failure trigger and independent distractors, Stochastic Delta Debugging will exhibit sublinear query growth in candidate space size $K$ ($O(\log K)$ candidate subset evaluations), requiring significantly fewer oracle evaluations than Stochastic Greedy Ablation ($O(K)$) as $K \ge 8$.
- **H2 (Conjunctive Trigger Partitioning Overhead)**: In target families where failure requires a conjunction of multiple elements ($m \ge 2$), Stochastic Delta Debugging will incur substantial complement evaluation overhead when trigger elements are split across chunks, eliminating its query advantage over Greedy Ablation.
- **H3 (Null-Target Integrity)**: When evaluated on non-failing null targets, all three algorithms will correctly terminate with `NO_TRIGGER_FOUND` / `NOT_APPLICABLE` and zero false-positive trigger discoveries, establishing a False Discovery Rate of $0.0\%$.
- **H4 (Non-Monotonic Fragility)**: In non-monotonic targets with suppression and restoration dynamics, classical partitioning assumptions are violated, and both Delta Debugging and Greedy Ablation may discover distinct local 1-minimal subsets depending on search path.

---

## 2. Experimental Design & Candidate Spaces

### Candidate-Space Sizes ($K$)
The benchmark evaluates 5 distinct candidate space sizes:
$$K \in \{4, 8, 12, 16, 20\}$$

### Target Families
The benchmark evaluates 8 distinct target families:
1. **Single Trigger (`single_trigger`)**: 1 causal factor ($P_{\text{trigger}} = 0.90$) and $K-1$ distractors ($P_{\text{base}} = 0.05$).
2. **Conjunctive Trigger (`conjunctive_trigger`)**: Pair of causal factors $(t_1, t_2)$ both required for failure ($P_{\text{conj}} = 0.90$), otherwise $P_{\text{base}} = 0.05$.
3. **Multiple Independent Triggers (`multiple_independent`)**: Two distinct independent single-factor triggers $t_1$ or $t_2$, either inducing failure ($P=0.90$), base $P=0.05$.
4. **Distractor-Heavy (`distractor_heavy`)**: 1 causal factor among $K-1$ explicitly labeled distractors ($P=0.90$ vs $0.05$).
5. **Null Target (`null_target`)**: $K$ distractors with unconditional background noise ($P = 0.05$) across all subsets. Ground truth is empty.
6. **Non-Monotonic (`non_monotonic`)**: Trigger $A$ ($0.90$), suppressed by $B$ ($0.10$), restored by $C$ ($0.90$), base $0.05$. Valid minimal triggers: $\{A\}$ and $\{A, B, C\}$.
7. **Stochastic Target (`stochastic_target`)**: 1 causal factor with moderate noise: $P_{\text{trigger}} = 0.75$, $P_{\text{base}} = 0.15$.
8. **Weak Signal Target (`weak_signal`)**: 1 causal factor with small effect size: $P_{\text{trigger}} = 0.40$, $P_{\text{base}} = 0.15$ ($\Delta = 0.25$).

---

## 3. Sampling, Randomness & Feasibility

### Number of Independent Instances
- **Instances per condition**: $N = 10$ independent instances generated with seeds $s \in \{100, 101, \dots, 109\}$.
- **Total conditions**: 5 candidate sizes $\times$ 8 families = 40 conditions.
- **Total benchmark instances**: $40 \times 10 = 400$ paired instances.
- **Total algorithm executions**: $400 \times 3 = 1,200$ runs.

### Positional Fairness
To prevent positional bias (e.g., algorithms benefiting from triggers placed at index 0 or index $K-1$), the initial input tuple is reproducibly shuffled for each instance using `random.Random(seed).shuffle(elements)`.

### Trial Policy & Decision Thresholds
- **Trials per candidate evaluation**: $t = 10$ stochastic Bernoulli trials.
- **Failure screening threshold ($\tau$)**:
  - Standard families (1–7): $\tau = 0.50$ (halfway between base $\le 0.15$ and trigger $\ge 0.75$).
  - Weak signal family (8): $\tau = 0.25$ (halfway between base $0.15$ and trigger $0.40$).
- **Trial budget cap**: $B_{\max} = 1000$ trials per algorithm run.
- **Timeout**: 60.0 wall-clock seconds per instance.

---

## 4. Compared Algorithms & Baseline Specifications

1. **Stochastic Delta Debugging (`StochasticDeltaDebugMinimizer`)**:
   - Hierarchical partitioning starting at $n=2$, evaluating chunks and complements.
   - Refinement loop for post-search 1-minimality verification.
   - Statistical classification into `STATISTICALLY_SUPPORTED`, `OBSERVED_ONLY`, `NOT_MINIMAL`, `INCONCLUSIVE`, or `NOT_APPLICABLE`.
2. **Stochastic Greedy Ablation (`run_stochastic_greedy_ablation`)**:
   - Sequential single-pass removal of elements from initial candidate set.
   - Preserves element if removing it drops failure rate below $\tau$.
   - Explicit 1-minimality verification of isolated subset.
3. **Stochastic Seeded Random Search (`run_stochastic_random_search`)**:
   - Uniform random subset sampling of proper subsets $k \in \{1, \dots, K-1\}$.
   - Early termination when candidate reaches size 1 (theoretical minimum positive size).
   - Budget-aware verification reservation: halts search and verifies when remaining trials $\le |S_{\text{best}}| \times t$.
   - Explicit 1-minimality verification of isolated subset.

---

## 5. Metrics & Statistical Analysis Plan

### Primary Metrics
1. **Target Invocations (Trials)**: Total oracle evaluations consumed ($N_{\text{trials}}$), separated into `search_target_calls` and `verification_target_calls`.
2. **Candidate Subsets Examined**: Count of distinct candidate configurations evaluated.
3. **Trigger Recovery**: Precision, Recall, and Ground-Truth Match rate.
4. **1-Minimality Verification Rate**: Fraction of runs achieving `STATISTICALLY_SUPPORTED` 1-minimality.
5. **False Discovery Rate**: Fraction of non-null triggers reported on `null_target`.
6. **Budget Exhaustion Rate**: Fraction of runs hitting $B_{\max} = 1000$.

### Statistical Analysis Plan
- All algorithm comparisons are **strictly paired on the identical benchmark instance** (same target seed, input permutation, and oracle sequence).
- Differences $d_i = \text{calls}(\text{Algorithm } A) - \text{calls}(\text{Algorithm } B)$ are analyzed via:
  - Mean difference, standard deviation, and median difference.
  - Two-sided Wilcoxon signed-rank test ($W^+, p$-value) with continuity correction and tie adjustment.
  - Paired Student's t-test ($t, p$-value).
- Multiple comparisons across candidate sizes and families are adjusted using the Benjamini-Hochberg (BH) False Discovery Rate procedure ($\alpha = 0.05$).

---

## 6. Execution Protocol Freeze
This protocol is locked and frozen prior to running the final Phase 2A benchmark sweep. All data collected will be analyzed strictly according to the specifications above.
