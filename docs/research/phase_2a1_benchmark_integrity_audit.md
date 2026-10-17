# AgentXRay — Phase 2A.1: Benchmark Integrity Audit, Methodological Corrections & Reanalysis

**Document Status**: Official Research Report  
**Phase**: Phase 2A.1 (Benchmark Integrity Audit & Algorithmic Reanalysis)  
**Date**: 2026-10-17  
**Execution Context**: Offline Synthetic Black-Box Benchmark Testbed (400 Paired Instances, 1,200 Algorithm Runs)  
**Spend / Model Calls**: \$0.0000 USD (Zero Commercial API Calls, Zero Live Model Calls)  
**Test Suite**: 159 Passing Unit, Integration, and Regression Tests  
**Git Checkpoint**: Working tree modified from `8f84fe3`  

---

## 1. Executive Summary & Audit Motivation

Phase 2A evaluated black-box failure-trigger isolation algorithms across 400 synthetic benchmark instances. While Phase 2A demonstrated significant algorithmic scaling advantages for Stochastic Delta Debugging over Greedy Ablation and Seeded Random Search, a subsequent rigorous peer review identified several critical integrity, mathematical, and methodological flaws in the benchmark implementation:

1. **Ground-Truth Parameter Leakage into Search Baselines**: Baseline functions accepted `true_trigger` and `alternative_valid_triggers` arguments, violating the architectural black-box boundary.
2. **Minimality Verification Control Flaw for Singleton Candidates**: For candidates with $|S|=1$, code evaluated `initial_input.without(elem)` ($K-1$ background elements) rather than the true ablation control $S \setminus \{e\} = \emptyset$, artificially driving verified 1-minimality to 0% in `multiple_independent`.
3. **Non-Monotonic Ground-Truth Mathematical Misclassification**: $\{A, B, C\}$ was erroneously specified as a valid 1-minimal trigger alongside $\{A\}$, despite the mathematical reality that removing suppressor $B$ leaves $\{A, C\}$ which still fails ($P=0.90 \ge \tau$).
4. **Metric Conflation (Null Detection vs. Non-Null Trigger Recovery)**: 50 null targets were pooled into the overall trigger recovery denominator ($N=400$), conflating true negatives with causal factor recovery.
5. **PRNG Sequence Desynchronization Across Search Paths**: Resetting linear PRNG seeds prior to each algorithm failed to guarantee matched observations across algorithms querying subsets in different orders.
6. **Result Wrapper Status Copying**: Baseline result wrappers in the benchmark runner copied `dd_cond.status` rather than reporting their own search termination status.
7. **Absence of Trial-Level and Candidate-Level Event Logs**: Evaluations were recorded only at the summary level, precluding granular auditing of candidate exploration trajectories.

This phase—**Phase 2A.1**—conducted a comprehensive forensic audit, implemented root-cause code corrections, amended the experimental protocol (Protocol v1.1), and re-executed the entire 400-instance benchmark sweep under Common Random Numbers (CRN) with granular candidate-level event logging.

### Key Audited Findings

- **1-Minimality Verification Restored in `multiple_independent`**: With the ablation control properly testing $S \setminus \{e\} = \emptyset$, verified 1-minimality in `multiple_independent` rose from **0.0% to 100.0%** for Delta Debugging (50/50) and Greedy Ablation (50/50), and to **62.0%** for Random Search.
- **Overall Non-Null 1-Minimality Verification**: On the 350 non-null instances, verified 1-minimality rose from 67.7% to **82.6%** (289/350) for Delta Debugging, from 56.6% to **73.1%** (256/350) for Greedy Ablation, and from 33.1% to **41.4%** (145/350) for Random Search.
- **Ground-Truth Trigger Recovery on Non-Null Targets ($N=350$)**:
  - **Stochastic Delta Debugging**: **93.7%** (328/350), Mean Precision: 0.937, Mean Recall: 0.937
  - **Stochastic Greedy Ablation**: **78.6%** (275/350), Mean Precision: 0.857, Mean Recall: 0.957
  - **Stochastic Seeded Random Search**: **47.4%** (166/350), Mean Precision: 0.657, Mean Recall: 0.894
- **Null Target Discrimination ($N=50$)**: All three algorithms achieved **100.0% Null Detection Rate** (50/50) with **0.0% False Discovery Rate**.
- **Paired Query Efficiency**: Delta Debugging maintained its statistically significant efficiency advantage over Greedy Ablation across all single-trigger, distractor-heavy, multiple-independent, non-monotonic, and noisy target families (saving 56–72 calls, $p < 0.0001$), while Greedy Ablation outperformed Delta Debugging strictly on `conjunctive_trigger` (+226.4 calls for DD, $p < 0.0001$), confirming Hypothesis H2.
- **Auditable Provenance**: All 31,483 candidate evaluations across the 1,200 algorithm executions were serialized line-by-line to `benchmarks/results/phase_2a1_event_log.jsonl`.

---

## 2. Forensic Root-Cause Analysis of Identified Issues

### 2.1 Issue 1: Ground-Truth Parameter Leakage into Search Baselines

#### Root Cause
In Phase 2A, `run_stochastic_greedy_ablation` and `run_stochastic_random_search` accepted:
```python
true_trigger: tuple[str, ...] = ()
alternative_valid_triggers: tuple[tuple[str, ...], ...] = ()
```
Inside the routines, these parameters were used to compute `matches_ground_truth` and `irrelevant_elements_remaining` for the return dataclass. Although neither algorithm used `true_trigger` to steer candidate selection, their presence in the search function signatures violated the architectural separation boundary between black-box probing and evaluation.

#### Corrective Action
- Excised `true_trigger` and `alternative_valid_triggers` from baseline function signatures.
- Baseline routines now execute purely as black-box optimizers querying `runner.execute_trial(input_case)`.
- All ground-truth evaluation is performed post-hoc by `evaluate_stochastic_ground_truth`.

---

### 2.2 Issue 2: Minimality Verification Control Flaw for Singleton Candidates

#### Root Cause
In Phase 2A, the minimizer and baseline minimality verification blocks contained the following branching logic:
```python
for elem in current_candidate:
    if len(current_candidate) > 1:
        candidate_without_elem = tuple(e for e in current_candidate if e != elem)
        sub_case = initial_input.subset(candidate_without_elem)
    else:
        sub_case = initial_input.without(elem)
```
This branching was introduced because `InputCase(elements=[])` previously raised a Pydantic `ValidationError`. When a candidate had size $|S| = 1$ (e.g., $S = \{t_1\}$), `initial_input.without(elem)` did not evaluate $S \setminus \{e\} = \emptyset$; instead, it evaluated the entire initial input minus $t_1$, which comprised all $K-1$ background distractors plus any other co-existing independent triggers (e.g., $t_2$).

In `multiple_independent`, testing $t_1$ via `initial_input.without("elem_t_indep_1")` evaluated a configuration containing `elem_t_indep_2`, which triggered failure with probability 0.90 ($\ge \tau = 0.50$). Consequently, the single-element removal test failed, and the candidate was erroneously flagged as not minimal, driving verified 1-minimality to **0% across all 50 instances**.

#### Corrective Action
1. Updated `InputCase` in `src/agentxray/models.py` to allow `elements = ()` representing the empty baseline control.
2. Updated `without()` and `subset()` to return valid empty `InputCase` instances when all elements are removed.
3. Removed the `len(current_candidate) > 1` branch across `StochasticDeltaDebugMinimizer`, `run_stochastic_greedy_ablation`, and `run_stochastic_random_search`. Verification now uniformly evaluates:
   ```python
   without_elem = tuple(e for e in current_candidate if e != elem)
   sub_case = initial_input.subset(without_elem)
   ```
   When $|S| = 1$, `without_elem = ()`, which queries the empty set where $P_{\text{base}} = 0.05 < \tau$, correctly confirming 1-minimality.

---

### 2.3 Issue 3: Non-Monotonic Ground-Truth Mathematical Misclassification

#### Root Cause
Phase 2A specified:
```python
gt = StochasticGroundTruthSpec(
    task_id="non_monotonic",
    trigger_elements=(a,),
    alternative_valid_triggers=((a, b, c),),
    ...
)
```
Under the non-monotonic target rule:
- $P(\{A\}) = 0.90$
- $P(\{A, B\}) = 0.10$ (suppressed)
- $P(\{A, B, C\}) = 0.90$ (restored)
- $P(\{A, C\}) = 0.90$ (trigger active, suppressor absent)
- $P(\text{otherwise}) = 0.05$

By definition, a candidate $S$ is 1-minimal if $P(S) \ge \tau$ and for every $e \in S$, $P(S \setminus \{e\}) < \tau$. For candidate $\{A, B, C\}$:
- $\{A, B, C\} \setminus \{B\} = \{A, C\} \implies P = 0.90 \ge 0.50$

Because removing suppressor element $B$ leaves $\{A, C\}$, which still produces the failure condition with probability 0.90, candidate $\{A, B, C\}$ contains a redundant/counterproductive element ($B$) and is **mathematically not 1-minimal**. $\{A\}$ is the sole 1-minimal trigger under this rule.

#### Corrective Action
- Formally corrected ground-truth specification: `alternative_valid_triggers = ()`.
- Only discoveries strictly isolating $\{A\}$ are scored as matching ground truth.

---

### 2.4 Issue 4: Metric Conflation (Null Detection vs. Non-Null Trigger Recovery)

#### Root Cause
Phase 2A reported a single aggregate match rate over all $N=400$ instances. The 50 null targets (where no trigger exists and returning $\emptyset$ is correct) were pooled into the numerator and denominator alongside the 350 causal target instances. This conflated true negative specificity with causal sensitivity.

#### Corrective Action
Separated the 400 instances into two mutually exclusive evaluation domains:
1. **Non-Null Causal Discovery ($N = 350$)**: Ground-Truth Trigger Recovery Rate, 1-Minimality Verification Rate, Precision, Recall.
2. **Null Target Discrimination ($N = 50$)**: Null Detection Rate (True Negatives), False Discovery Rate on Null Targets.
3. **Resource Boundaries ($N = 400$)**: Budget Exhaustion Rate ($B_{\max} = 1000$).

---

### 2.5 Issue 5: PRNG Sequence Desynchronization across Search Paths

#### Root Cause
In Phase 2A, the benchmark runner called `target.reset(new_seed=seed)` before executing each algorithm. However, because Delta Debugging (binary partitioning), Greedy Ablation (sequential removal), and Random Search (random sampling) query different candidate subsets in different orders, their sequential PRNG draw indices diverged immediately. When two algorithms queried the same candidate subset (e.g., initial input $S$ or candidate $C$), they sampled from different positions in the PRNG stream.

#### Corrective Action
Implemented Common Random Numbers (CRN) in `StochasticSyntheticTarget`:
- Each evaluation keys a deterministic pseudorandom float on `(target_seed, canonical_subset, replicate_idx)` using SHA-256:
  $$u = \frac{\text{int}(\text{SHA256}(f"{seed}:{canonical\_subset}:{rep\_idx}")[:8]) \gg 11}{2^{53}}$$
- For any subset $S$ (canonicalized by sorting element names), the $j$-th evaluation ($j \in \{0, \dots, t-1\}$) produces the exact same float $u$ and Bernoulli outcome across all algorithms, providing genuine candidate-by-candidate matched observations.

---

### 2.6 Issue 6: Result Wrapper Status Copying

#### Root Cause
In `benchmarks/synthetic/scaling_benchmark.py`, lines 382 and 431 created dummy discovered conditions to pass to `evaluate_stochastic_ground_truth`:
```python
dummy_greedy_cond = StochasticDiscoveredCondition(..., status=dd_cond.status, ...)
dummy_random_cond = StochasticDiscoveredCondition(..., status=dd_cond.status, ...)
```
If Delta Debugging completed but Random Search exhausted its budget, Random Search's wrapper erroneously reported `status=SearchStatus.COMPLETE`.

#### Corrective Action
Each wrapper now independently determines its search status:
```python
if not comp.failure_reproduced:
    status = SearchStatus.NO_TRIGGER_FOUND
elif comp.budget_exhausted:
    status = SearchStatus.BUDGET_EXHAUSTED
else:
    status = SearchStatus.COMPLETE
```

---

## 3. Implementation Corrections & Regression Verification

### File Modifications Summary
1. `src/agentxray/models.py`: Supported empty `InputCase(elements=())`.
2. `src/agentxray/stochastic_minimizer.py`: Evaluated $S \setminus \{e\}$ uniformly for all $|S| \ge 1$; added `runner.on_candidate_evaluated` notification.
3. `src/agentxray/stochastic_baselines.py`: Excised ground-truth parameters; evaluated $S \setminus \{e\}$ uniformly for all $|S| \ge 1$.
4. `src/agentxray/runner.py`: Added `on_candidate_evaluated` callback hook.
5. `benchmarks/synthetic/stochastic_target.py`: Implemented Common Random Numbers (CRN) mode.
6. `benchmarks/synthetic/scaling_benchmark.py`: Fixed baseline calls, wrapper statuses, non-monotonic ground truth (`alternative_valid_triggers=()`), enabled CRN (`crn_mode=True`), implemented candidate-level event logging (`phase_2a1_event_log.jsonl`), and versioned output files (`phase_2a1_*`).
7. `benchmarks/synthetic/analyze_results.py`: Separated null detection from non-null recovery; supported CLI `--results-file`.
8. `docs/research/phase_2a_protocol_v1.1.md`: Authored formal protocol amendment.
9. `tests/test_scaling_benchmark.py`: Added 6 regression tests covering all audited integrity dimensions.

### Regression Test Suite Results
All 159 tests passed in 0.25 seconds with zero warnings and zero failures:
```
tests/test_audit.py ...                                                  [  1%]
tests/test_baselines.py .......                                          [  6%]
tests/test_calibrated_pilot.py .......                                   [ 10%]
tests/test_isolation.py .......                                          [ 15%]
tests/test_minimizer.py ......                                           [ 18%]
tests/test_model_harness.py ...........................                  [ 35%]
tests/test_models.py ..............                                      [ 44%]
tests/test_ollama_provider.py ........................                   [ 59%]
tests/test_providers.py .............                                    [ 67%]
tests/test_reporting.py ...                                              [ 69%]
tests/test_runner.py ....                                                [ 72%]
tests/test_scaling_benchmark.py ..............                           [ 81%]
tests/test_statistics.py ...........                                     [ 88%]
tests/test_stochastic_baselines.py ....                                  [ 90%]
tests/test_stochastic_minimizer.py ........                              [ 95%]
tests/test_stochastic_reporting.py ...                                   [ 97%]
tests/test_stochastic_target.py ....                                     [100%]
============================= 159 passed in 0.25s ==============================
```

---

## 4. Reanalysis Results: Phase 2A.1 Benchmark

### 4.1 Overall Comparative Summary (400 Instances / 1,200 Algorithm Runs)

| Metric | Stochastic Delta Debugging | Stochastic Greedy Ablation | Stochastic Seeded Random |
| :--- | :---: | :---: | :---: |
| **Total Invocations (Mean $\pm$ Std)** | **104.1 $\pm$ 111.4** | 125.5 $\pm$ 45.4 | 557.5 $\pm$ 391.8 |
| **Total Invocations (Median)** | **70.0** | 140.0 | 620.0 |
| **Search Invocations (Mean)** | 94.0 | 112.2 | 540.4 |
| **Verification Invocations (Mean)** | 10.1 | 13.3 | 17.1 |
| **Subsets Examined (Mean)** | **7.7** | 12.5 | 50.8 |
| **Non-Null Ground-Truth Recovery ($N=350$)** | **93.7%** (328/350) | 78.6% (275/350) | 47.4% (166/350) |
| **Non-Null 1-Minimality Rate ($N=350$)** | **82.6%** (289/350) | 73.1% (256/350) | 41.4% (145/350) |
| **Non-Null Mean Precision** | **0.937** | 0.857 | 0.657 |
| **Non-Null Mean Recall** | 0.937 | **0.957** | 0.894 |
| **Null Detection Rate ($N=50$)** | **100.0%** (50/50) | **100.0%** (50/50) | **100.0%** (50/50) |
| **Null Target False Discovery Rate ($N=50$)** | **0.0%** (0/50) | **0.0%** (0/50) | **0.0%** (0/50) |
| **Budget Exhaustion Rate ($N=400$)** | **0.0%** (0/400) | **0.0%** (0/400) | 43.8% (175/400) |

---

### 4.2 Scaling by Candidate Space Size ($K$)

| $K$ | Metric | Delta Debugging | Greedy Ablation | Seeded Random |
| :---: | :--- | :---: | :---: | :---: |
| **4** | Target Invocations (Mean) | 60.0 | **54.2** | 200.0 |
| | Subsets Examined (Mean) | **4.7** | 5.3 | 7.2 |
| | Non-Null Trigger Recovery ($N=70$) | **97.1%** | 81.4% | 94.3% |
| **8** | Target Invocations (Mean) | 90.6 | **88.9** | 310.9 |
| | Subsets Examined (Mean) | **6.8** | 8.8 | 25.7 |
| | Non-Null Trigger Recovery ($N=70$) | **91.4%** | 84.3% | 85.7% |
| **12** | Target Invocations (Mean) | **108.4** | 124.2 | 678.1 |
| | Subsets Examined (Mean) | **8.0** | 12.3 | 64.2 |
| | Non-Null Trigger Recovery ($N=70$) | **92.9%** | 77.1% | 42.9% |
| **16** | Target Invocations (Mean) | **120.9** | 162.0 | 811.2 |
| | Subsets Examined (Mean) | **8.9** | 16.2 | 78.9 |
| | Non-Null Trigger Recovery ($N=70$) | **95.7%** | 74.3% | 2.9% |
| **20** | Target Invocations (Mean) | **140.8** | 198.1 | 787.0 |
| | Subsets Examined (Mean) | **10.0** | 19.8 | 77.9 |
| | Non-Null Trigger Recovery ($N=70$) | **91.4%** | 75.7% | 11.4% |

**Scaling Analysis**:
- For small spaces ($K=4$), Greedy Ablation consumes slightly fewer calls (54.2 vs. 60.0) due to Delta Debugging's binary partitioning overhead ($n=2$ split plus complement evaluations).
- As candidate space grows ($K \ge 12$), Delta Debugging exhibits logarithmic subset scaling ($O(\log K)$), examining **10.0 subsets at $K=20$** compared to Greedy Ablation's linear growth ($O(K)$), examining **19.8 subsets at $K=20$**.
- At $K=20$, Delta Debugging requires **28.9% fewer target invocations** than Greedy Ablation (140.8 vs. 198.1 calls).
- Seeded Random Search collapses completely for $K \ge 16$, exhausting its 1000-trial budget on over 80% of runs and achieving only 2.9% recovery at $K=16$ and 11.4% at $K=20$.

---

### 4.3 Performance Breakdown Across Target Families ($N=50$ Each)

| Target Family | Metric | Delta Debugging | Greedy Ablation | Seeded Random |
| :--- | :--- | :---: | :---: | :---: |
| **`single_trigger`** | Mean Invocations | **71.4** | 139.2 | 630.4 |
| | Non-Null Recovery Rate | **100.0%** | 98.0% | 52.0% |
| | 1-Minimality Verification Rate | **100.0%** | 98.0% | 52.0% |
| **`distractor_heavy`** | Mean Invocations | **72.0** | 139.0 | 630.4 |
| | Non-Null Recovery Rate | **100.0%** | 100.0% | 52.0% |
| | 1-Minimality Verification Rate | **100.0%** | 100.0% | 52.0% |
| **`multiple_independent`** | Mean Invocations | **66.0** | 138.0 | 533.2 |
| | Non-Null Recovery Rate | **100.0%** | 100.0% | 62.0% |
| | 1-Minimality Verification Rate | **100.0%** | 100.0% | 62.0% |
| **`non_monotonic`** | Mean Invocations | **88.6** | 144.6 | 630.4 |
| | Non-Null Recovery Rate | **100.0%** | 48.0% | 52.0% |
| | 1-Minimality Verification Rate | **100.0%** | 48.0% | 52.0% |
| **`conjunctive_trigger`** | Mean Invocations | 376.4 | **150.0** | 1000.0 |
| | Non-Null Recovery Rate | **100.0%** | 100.0% | 32.0% |
| | 1-Minimality Verification Rate | **100.0%** | 100.0% | 32.0% |
| **`stochastic_target`** | Mean Invocations | **72.6** | 141.0 | 630.4 |
| | Non-Null Recovery Rate | **98.0%** | 84.0% | 52.0% |
| | 1-Minimality Verification Rate | **78.0%** | 66.0% | 40.0% |
| **`weak_signal`** | Mean Invocations | **76.0** | 142.2 | 394.8 |
| | Non-Null Recovery Rate | **58.0%** | 20.0% | 30.0% |
| | 1-Minimality Verification Rate | 0.0% | 0.0% | 0.0% |
| **`null_target`** [NULL] | Mean Invocations | 10.0 | 10.0 | 10.0 |
| | Null Detection Rate | **100.0%** | 100.0% | 100.0% |
| | False Discovery Rate | **0.0%** | 0.0% | 0.0% |

---

### 4.4 Paired Statistical Hypothesis Testing

All paired comparisons evaluate matched trial differences on identical benchmark instances ($N=400$ pairs):

$$\Delta_{\text{calls}} = \text{Calls}(\text{Delta Debugging}) - \text{Calls}(\text{Greedy Ablation})$$

#### Overall Invocations Comparison
- **Mean Difference**: $-21.38 \pm 113.55$ calls
- **Median Difference**: $-30.0$ calls
- **Two-Sided Wilcoxon Signed-Rank Test**: $W^+ = 15,089.5$, $p = 0.00017$
- **Paired Student's t-test**: $t = -3.76$, $p = 0.00017$
- **Conclusion**: Stochastic Delta Debugging requires statistically significantly fewer target evaluations overall than Stochastic Greedy Ablation ($p < 0.001$).

#### Family-by-Family Paired Tests (Benjamini-Hochberg FDR Adjusted, $\alpha = 0.05$)

| Target Family | Mean Diff ($\Delta_{\text{calls}}$) | Median Diff | Wilcoxon $W^+$ | Raw $p$-value | BH Adjusted $q$-value | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`conjunctive_trigger`** | $+226.4$ | $+250.0$ | 1275.0 | $< 0.00001$ | $< 0.00001$ | Significant (Greedy Wins) |
| **`distractor_heavy`** | $-67.0$ | $-70.0$ | 4.0 | $< 0.00001$ | $< 0.00001$ | Significant (DD Wins) |
| **`multiple_independent`** | $-72.0$ | $-70.0$ | 0.0 | $< 0.00001$ | $< 0.00001$ | Significant (DD Wins) |
| **`non_monotonic`** | $-56.0$ | $-55.0$ | 104.5 | $< 0.00001$ | $< 0.00001$ | Significant (DD Wins) |
| **`single_trigger`** | $-67.8$ | $-70.0$ | 4.0 | $< 0.00001$ | $< 0.00001$ | Significant (DD Wins) |
| **`stochastic_target`** | $-68.4$ | $-70.0$ | 4.0 | $< 0.00001$ | $< 0.00001$ | Significant (DD Wins) |
| **`weak_signal`** | $-66.2$ | $-50.0$ | 35.0 | $< 0.00001$ | $< 0.00001$ | Significant (DD Wins) |
| **`null_target`** | $0.0$ | $0.0$ | 0.0 | $1.00000$ | $1.00000$ | Invariant (Tied) |

---

## 5. Comparative Analysis: Phase 2A vs. Phase 2A.1

| Dimension | Phase 2A (Reported) | Phase 2A.1 (Audited & Corrected) | Underlying Mechanism of Change |
| :--- | :---: | :---: | :--- |
| **Baseline Signatures** | Accepted `true_trigger` | **Zero Ground-Truth Access** | Eliminated potential information leakage. |
| **Minimality Check on Singletons** | Evaluated $K-1$ background elements | **Evaluated $S \setminus \{e\} = \emptyset$** | `InputCase` allows empty elements; correct ablation control. |
| **`multiple_independent` Minimality** | **0.0%** (0/50) | **100.0%** (50/50) | Singleton removal tests empty set ($P=0.05 < \tau$) instead of background trigger. |
| **`non_monotonic` Ground Truth** | Allowed $\{A\}$ or $\{A, B, C\}$ | **$\{A\}$ Only** | $\{A, B, C\} \setminus \{B\} = \{A, C\}$ triggers failure ($P=0.90$), so $\{A, B, C\}$ is non-minimal. |
| **`non_monotonic` Recovery Rate (DD)** | 100.0% (counted $\{A, B, C\}$) | **100.0%** (strictly $\{A\}$) | Delta Debugging consistently isolates the global singleton trigger $\{A\}$. |
| **`non_monotonic` Recovery (Greedy)** | 48.0% | **48.0%** | Greedy ablation path is sensitive to element order in non-monotonic landscapes. |
| **Non-Null 1-Minimality (DD)** | 67.7% (237/350) | **82.6%** (289/350) | Proper verification of singleton candidates in `multiple_independent`. |
| **Non-Null Trigger Recovery (DD)** | 91.4% (320/350) | **93.7%** (328/350) | Clean Common Random Numbers pairing and post-hoc scoring. |
| **Null Reporting** | Pooled into 400 instances | **Disjoint Domains ($N=50, 350$)** | Separated True Negatives from causal trigger isolation sensitivity. |
| **Event Logging** | Summary metrics only | **31,483 JSONL Records** | Complete auditable trail in `phase_2a1_event_log.jsonl`. |

---

## 6. Audit Guardrails & Protocol Compliance

1. **Commercial Spend**: Exactly **\$0.0000 USD**. No OpenAI API requests, no paid endpoints, zero credit consumption.
2. **Local Model Invocations**: Zero Ollama requests. All evaluations executed within the synthetic testbed.
3. **Artifact Immutability**: Historical Phase 2A artifacts (`phase_2a_protocol_v1.md`, `phase_2a_scaling_and_statistical_validation.md`, `phase_2a_scaling_results.json`, `phase_2a_scaling_summary.csv`) were preserved byte-for-byte.
4. **Versioned Deliverables**:
   - Protocol Amendment: `docs/research/phase_2a_protocol_v1.1.md`
   - Audit Report: `docs/research/phase_2a1_benchmark_integrity_audit.md`
   - Results JSON: `benchmarks/results/phase_2a1_scaling_results.json`
   - Summary CSV: `benchmarks/results/phase_2a1_scaling_summary.csv`
   - Event Log: `benchmarks/results/phase_2a1_event_log.jsonl` (31,483 records)

---

## 7. Epistemic Limitations & Future Research Directives

### 7.1 Epistemic Boundaries
1. **Synthetic Oracle Abstraction**: While the synthetic testbed rigorously models binomial uncertainty, noise, masking, and non-monotonicity, it evaluates stateless Python callables rather than autoregressive neural token generation with multi-turn tool loops.
2. **Sample Size for Weak Signals**: With $t=10$ trials per candidate, the benchmark has limited power to verify weak effect sizes ($\Delta = 0.25$ in `weak_signal`), resulting in 0% verified 1-minimality despite 58% recovery. Higher sample sizes ($t \ge 30$) or adaptive sequential probability ratio tests (SPRT) are required for low-delta triggers.
3. **Conjunctive Partitioning Fragility**: Delta Debugging's query explosion on `conjunctive_trigger` (+226.4 calls over Greedy) remains an intrinsic characteristic of binary partitioning algorithms when causal factors are fractured across chunks.

### 7.2 Recommendations Prior to Phase 2B
1. **SPRT Minimizer Exploration**: Replace fixed-$t$ evaluation blocks with adaptive Sequential Probability Ratio Tests (Wald SPRT) to reduce sample requirements on obvious non-triggers while concentrating budget on marginal candidates.
2. **Hybrid Partitioning-Greedy Search**: Integrate greedy fallback heuristics when Delta Debugging detects complement failure patterns indicative of conjunctive interactions.
3. **Phase Boundary**: The completion of Phase 2A.1 establishes a verified, leak-free statistical foundation. Next phases must build on these audited artifacts without re-introducing ground-truth leakage or metric conflation.
