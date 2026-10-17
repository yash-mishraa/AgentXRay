# AgentXRay — Phase 2A: Scaling Benchmark Protocol Amendment (Version 1.1)

**Document Status**: Frozen Post-Audit Corrective Protocol  
**Version**: 1.1.0 (Amends Protocol v1.0.0)  
**Date**: 2026-10-17  
**Authors**: Antigravity Research Team  

---

## 1. Executive Summary & Rationale for Amendment

Protocol Version 1.0 governed the initial execution of the Phase 2A synthetic scaling benchmark (400 paired instances across 8 target families and 5 candidate space sizes $K \in \{4, 8, 12, 16, 20\}$). A subsequent methodological and code audit identified five critical integrity flaws in the benchmark harness and baseline implementations:

1. **Ground-Truth Parameter Leakage**: Baseline search signatures (`run_stochastic_greedy_ablation`, `run_stochastic_random_search`) accepted `true_trigger` and `alternative_valid_triggers` arguments.
2. **Minimality Verification Control Flaw for Singletons**: When $|S| = 1$, the minimality check evaluated `initial_input.without(elem)` ($K-1$ background elements) rather than the ablation control $S \setminus \{e\} = \emptyset$, due to an `InputCase` constraint that rejected empty element tuples.
3. **Non-Monotonic Ground-Truth Specification Error**: $\{A, B, C\}$ was erroneously specified as a co-equal 1-minimal trigger alongside $\{A\}$, despite the mathematical fact that $\{A, B, C\} \setminus \{B\} = \{A, C\}$ produces failure probability $P=0.90 \ge \tau$.
4. **Metric Conflation (Null Detection vs. Non-Null Recovery)**: 50 null targets were pooled into the overall recovery rate denominator ($N=400$), conflating true negatives with causal factor recovery.
5. **PRNG Sequence Desynchronization across Search Paths**: Linear sequential PRNG reseeding before each algorithm failed to guarantee matched observations because different search algorithms query candidate subsets in different orders.
6. **Result Wrapper Status Copying**: Baseline result wrappers in the benchmark runner copied `dd_cond.status` rather than deriving their own search status from their execution state.

Protocol Amendment Version 1.1 formally specifies the corrections to each of these issues. All historical Phase 2A artifacts (`phase_2a_protocol_v1.md`, `phase_2a_scaling_and_statistical_validation.md`, `phase_2a_scaling_results.json`, `phase_2a_scaling_summary.csv`) are preserved without modification. All corrected executions are serialized to versioned artifacts (`phase_2a1_*`).

---

## 2. Formal Protocol Amendments

### Amendment 1: Complete Elimination of Ground-Truth Leakage in Baselines

- **Protocol v1.0 Rule**: `run_stochastic_greedy_ablation` and `run_stochastic_random_search` accepted `true_trigger: tuple[str, ...]` and `alternative_valid_triggers: tuple[tuple[str, ...], ...]`.
- **Protocol v1.1 Rule**: All ground-truth parameters are permanently excised from `run_stochastic_greedy_ablation` and `run_stochastic_random_search`.
- **Architectural Boundary**: Baseline search routines interact strictly with the black-box `InterventionRunner.execute_trial(input_case)` interface. Candidate selection, ablation ordering, stopping rules, and verification decisions must be 100% blind to ground truth. Ground-truth matching, precision, recall, excess distractors, and missing elements are evaluated post-hoc by independent evaluators (`evaluate_stochastic_ground_truth`).

### Amendment 2: 1-Minimality Verification on Proper Ablation Controls ($S \setminus \{e\}$)

- **Protocol v1.0 Rule**: When $|S| > 1$, code evaluated $S \setminus \{e\}$. When $|S| = 1$, code evaluated `initial_input.without(elem)` ($K-1$ elements).
- **Protocol v1.1 Rule**:
  - `InputCase(elements=())` is explicitly valid and represents the empty (unperturbed) control configuration.
  - For any candidate subset $S$ of size $|S| \ge 1$ and for every element $e \in S$, the 1-minimality check evaluates:
    $$S' = S \setminus \{e\}$$
  - When $|S| = 1$, $S' = \emptyset$. Evaluating $S' = \emptyset$ tests the target without any candidate elements, yielding base background noise ($P_{\text{base}} = 0.05 < \tau$).
  - In `multiple_independent` tasks containing co-existing independent triggers $\{t_1\}$ and $\{t_2\}$, isolating either singleton and testing $\{t_i\} \setminus \{t_i\} = \emptyset$ accurately demonstrates that dropping $t_i$ from the candidate eliminates the failure condition, establishing statistically supported 1-minimality.

### Amendment 3: Non-Monotonic Ground-Truth Mathematical Formalization

- **Protocol v1.0 Rule**: `alternative_valid_triggers = (("elem_t_nm_trigger", "elem_t_nm_suppressor", "elem_t_nm_restorer"),)`
- **Mathematical Analysis**:
  Under the non-monotonic failure rule:
  $$\begin{aligned}
  P(\text{fail} \mid \{A\}) &= 0.90 \\
  P(\text{fail} \mid \{A, B\}) &= 0.10 \\
  P(\text{fail} \mid \{A, B, C\}) &= 0.90 \\
  P(\text{fail} \mid \{A, C\}) &= 0.90 \\
  P(\text{fail} \mid \text{otherwise}) &= 0.05
  \end{aligned}$$
  For $\{A, B, C\}$ to be 1-minimal, every single-element removal must fail to reproduce the failure condition ($P < \tau = 0.50$):
  - $\{A, B, C\} \setminus \{C\} = \{A, B\} \implies P = 0.10 < 0.50$ (passes removal test)
  - $\{A, B, C\} \setminus \{A\} = \{B, C\} \implies P = 0.05 < 0.50$ (passes removal test)
  - $\{A, B, C\} \setminus \{B\} = \{A, C\} \implies P = 0.90 \ge 0.50$ (**FAILS removal test**)
  Because removing suppressor $B$ leaves $\{A, C\}$, where trigger $A$ is active without suppressor $B$, $\{A, C\}$ triggers failure with probability 0.90. Thus, $\{A, B, C\}$ contains an irrelevant/counterproductive element ($B$) and is **NOT** 1-minimal.
- **Protocol v1.1 Rule**: The planted ground truth for `non_monotonic` has `trigger_elements = (a,)` and `alternative_valid_triggers = ()`. $\{A\}$ is the unique 1-minimal trigger.

### Amendment 4: Disjoint Metric Reporting for Null and Non-Null Scenarios

- **Protocol v1.0 Rule**: Reported single pooled match rate across all 400 instances.
- **Protocol v1.1 Rule**: The 400 benchmark instances are partitioned into two disjoint evaluation subsets:
  1. **Non-Null Causal Discovery ($N = 350$ instances)**:
     - Primary metric: Ground-Truth Trigger Recovery Rate = $\frac{\sum \mathbb{I}(\text{discovered} = \text{ground\_truth})}{350}$
     - Primary metric: 1-Minimality Verification Rate = $\frac{\sum \mathbb{I}(\text{minimality\_status} = \text{STATISTICALLY\_SUPPORTED})}{350}$
     - Mean Precision and Mean Recall over non-null instances.
  2. **Null Target Discrimination ($N = 50$ instances)**:
     - Primary metric: Null Detection Rate (True Negatives) = $\frac{\sum \mathbb{I}(\neg \text{failure\_reproduced} \wedge |S_{\text{disc}}| = 0)}{50}$
     - Primary metric: False Discovery Rate on Null Targets = $\frac{\sum \mathbb{I}(\text{failure\_reproduced} \vee |S_{\text{disc}}| > 0)}{50}$
  3. **Resource Boundaries ($N = 400$ instances)**:
     - Budget Exhaustion Rate = $\frac{\sum \mathbb{I}(\text{trials} \ge B_{\max})}{400}$, reported separately from algorithmic success/failure.

### Amendment 5: Common Random Numbers (CRN) for Exact Candidate Pairing

- **Protocol v1.0 Rule**: `target.reset(new_seed=seed)` reseeded a linear PRNG before each algorithm. Because Delta Debugging, Greedy Ablation, and Random Search examine different subsets in different orders, the sequential draws diverged immediately.
- **Protocol v1.1 Rule**:
  - `StochasticSyntheticTarget` implements Common Random Numbers (CRN) keyed on `(target_seed, canonical_subset, replicate_idx)` using SHA-256 cryptographic hashing mapped to double-precision floats in $[0.0, 1.0)$:
    $$u = \frac{\text{int}(\text{SHA256}(f"{seed}:{canonical\_subset}:{rep\_idx}")[:8]) \gg 11}{2^{53}}$$
  - For any candidate subset $S$ (canonicalized as a sorted tuple of element names), the $j$-th evaluation ($j = 0, 1, \dots, t-1$) produces the identical pseudorandom draw $u$ and identical Bernoulli outcome across all three algorithms, regardless of evaluation order or preceding queries.

### Amendment 6: Granular Event Logging Specification

- **Protocol v1.1 Rule**: Every candidate evaluation executed during the benchmark sweep is logged line-by-line to `benchmarks/results/phase_2a1_event_log.jsonl`.
- **JSONL Schema**:
  - `benchmark_version`: `"phase_2a1"` (str)
  - `target_family`: Target family name (str)
  - `k`: Candidate space size $K$ (int)
  - `seed`: Instance seed (int)
  - `algorithm`: Search algorithm name (str)
  - `step_index`: Candidate evaluation index within run (1-based int)
  - `phase`: `"search"` or `"minimality_verification"` (str)
  - `candidate_elements`: List of element identifiers (list[str])
  - `candidate_size`: $|S|$ (int)
  - `trials_requested`: Number of Bernoulli trials in this evaluation block (int)
  - `failures_observed`: Number of failures observed (int)
  - `failure_rate`: Observed sample failure proportion (float)
  - `ci_lower`: Wilson score 95% lower bound (float)
  - `ci_upper`: Wilson score 95% upper bound (float)
  - `cumulative_target_calls`: Total trials executed by this algorithm so far (int)
  - `elapsed_ms`: Wall-clock elapsed milliseconds for this algorithm run (float)

### Amendment 7: Result Wrapper Status Correction

- **Protocol v1.0 Rule**: Benchmark runner wrapped greedy and random search results into `StochasticDiscoveredCondition` copying `status=dd_cond.status`.
- **Protocol v1.1 Rule**: Each wrapper derives its own status independently:
  ```python
  if not comp.failure_reproduced:
      status = SearchStatus.NO_TRIGGER_FOUND
  elif comp.budget_exhausted:
      status = SearchStatus.BUDGET_EXHAUSTED
  else:
      status = SearchStatus.COMPLETE
  ```

---

## 3. Scope & Execution Guardrails

1. **Testbed Scope**: Pure synthetic black-box simulation. Zero LLM inference, zero commercial API calls, zero credit spend ($0.0000 USD).
2. **Artifact Versioning**: Original Phase 2A files are untouched. Phase 2A.1 writes `phase_2a1_scaling_results.json`, `phase_2a1_scaling_summary.csv`, and `phase_2a1_event_log.jsonl`.
3. **Phase Boundary**: Execution of Phase 2A.1 completes the benchmark audit milestone. Phase 2B must NOT begin automatically.
