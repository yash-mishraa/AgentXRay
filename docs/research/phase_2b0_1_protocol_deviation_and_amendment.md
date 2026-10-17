# AgentXRay — Phase 2B.0.1 Protocol Deviation Register & Protocol Amendment v1.1

**Document Version:** 1.1 (Protocol Amendment & Deviation Register)  
**Author:** Antigravity (Pair Programming AI Assistant)  
**Date:** 2026-10-17  
**Original Frozen Protocol Baseline:** `docs/research/phase_2b0_local_model_pilot_protocol.md` (Version 1.0, Commit `2e92142`)  
**Target Runtime:** Local Ollama Daemon (`http://127.0.0.1:11434`), `llama3.2:3b`  
**Commercial API Spend:** Strictly **$0.0000 USD**  
**Status:** FORMAL AMENDMENT PRIOR TO CORRECTIVE PILOT EXECUTION  

---

## 1. Deviation Register (Phase 2B.0 Execution vs Protocol v1.0)

| Deviation ID | Protocol v1.0 Specification | Actual Phase 2B.0 Execution | Root Cause Analysis | Empirical & Epistemic Impact |
| :--- | :--- | :--- | :--- | :--- |
| **DEV-01: Incomplete Characterization** | All $2^4 = 16$ candidate configurations evaluated at $N_{\text{char}} = 10$ trials each (160 model requests). | Only 4 configurations evaluated at $N = 5$ trials each ($\emptyset$, $\{T_1\}$, $\{D_1, D_2, D_3\}$, $\{T_1, D_1, D_2, D_3\}$). 12 subsets were never characterized. | The runner implemented a minimal controls check instead of the full power set generator to conserve model requests during initial qualification. | Interactions among subset pairs (e.g. $\{T_1, D_1\}$) and individual distractor singletons were not empirically characterized before search. |
| **DEV-02: Missing Search Trial Events** | Every trial must be logged as an immutable `trial_executed` event in `phase_2b0_calibration_event_log.jsonl`. | Only 30 `trial_executed` events logged (20 in calibration controls, 10 in holdout). The 75 search trials were logged only as candidate evaluations (`algorithm_candidate_evaluated`). | `self.log_event("trial_executed", ...)` was implemented in `evaluate_subset_trials()` but was not hooked into the `target_fn` wrapper used by `InterventionRunner`. | Per-trial tokens, raw tool arguments, and individual trial latencies for the minimizer search phase are missing from the event log. |
| **DEV-03: Incomplete Holdout Validation** | Holdout confirmation must be executed for all discovered subsets across all evaluated search algorithms. | Holdout validation evaluated only Delta Debugging and Greedy Ablation (5 trials each on $\{T_1\}$). Random Search was omitted. | `discovered_map` in `calibration_runner.py` explicitly hard-coded keys for `delta_debugging` and `greedy_ablation`, forgetting `random_search`. | Post-search reproduction verification for Random Search was omitted from the artifact records. |
| **DEV-04: Sample Size Discrepancy** | $N = 10$ trials per candidate subset evaluation. | $N = 5$ trials per candidate subset evaluation. | The runner reduced $N$ to 5 to ensure that running all three algorithms sequentially would not breach the local execution timeout. | Point estimates have wider Wilson 95% score confidence intervals ($[0.57, 1.00]$ for $5/5$ vs $[0.72, 1.00]$ for $10/10$). |
| **DEV-05: Narrative Paired Test Conflation** | Paired statistical difference requires independent paired task instances ($N \ge 20$). | Phase 2B.0 narrative summary cited $p = 1.4883 \times 10^{-17}$, which belongs to the Phase 2A synthetic scaling benchmark ($N=400$), not the $N=1$ local pilot. | Errata resolution documentation was presented alongside pilot results, leading to context conflation in the narrative summary. | A single-instance pilot ($N = 1$) cannot support inferential paired t-test or Wilcoxon test statistics ($\nu = 0$). |

---

## 2. Epistemic Status of Phase 2B.0 Observations

### What CAN be concluded from Phase 2B.0:
1. **Local Neural Execution Viability:** Local tool-using agent execution with `llama3.2:3b` via Ollama on Apple Silicon Mac functions without infrastructure crashes, API keys, or financial costs ($0.0000 USD).
2. **Deterministic Trigger Efficacy on Bounds:** The currency formatting constraint ($T_1$) successfully and deterministically induces tool syntax rejection when present ($5/5$ failures on $\{T_1\}$ and $\{T_1, D_1, D_2, D_3\}$).
3. **Clean Control Boundaries:** The unmutated baseline ($\emptyset$) and full distractor set ($\{D_1, D_2, D_3\}$) produce zero failures ($0/5$), confirming that the tool schema and prompt do not generate spurious errors.
4. **Black-Box Isolation:** Minimizers and evaluators executed without access to planted ground-truth labels during search.

### What CANNOT be concluded from Phase 2B.0:
1. **Algorithmic Superiority:** The observed request difference ($\text{DD}=40$ vs $\text{Greedy}=60$) is a raw point observation on a single instance ($N=1$). It does **not** establish a statistically significant efficiency advantage.
2. **Exhaustive Monotonicity / Absence of Masking:** Because 12 subsets were uncharacterized, it is not empirically known whether any distractor modifies the failure probability of $T_1$ in pairwise combinations.
3. **Confirmatory Research Readiness:** The run must be classified as **`EXPLORATORY_ONLY`**.

---

## 3. Protocol Amendment v1.1 (Pre-Execution Specification)

This amendment updates the pilot specification prior to executing the corrective calibration pilot (Phase 2B.0.1):

### 3.1 Exhaustive Characterization of All 16 Configurations
To satisfy Section 3.2 of the protocol without prohibitive latency, all $2^4 = 16$ candidate configurations will be systematically characterized:
$$\mathcal{P}(C) = \{ S \subseteq \{T_1, D_1, D_2, D_3\} \}$$
- For initial local calibration qualification, each of the 16 configurations will be evaluated with $N_{\text{char}} = 5$ trials ($80$ agent executions total $= 160$ model requests).
- A candidate subset is classified as failing if $\hat{p} \ge 0.50$.
- The trigger $T_1$ is qualified if and only if:
  $$\min_{S: T_1 \in S} \hat{p}(S) \ge 0.70 \quad \text{AND} \quad \max_{S: T_1 \notin S} \hat{p}(S) \le 0.20$$

### 3.2 Universal Trial-Level Event Logging
The runner must hook into every `target.execute` call to emit an immutable `trial_executed` event containing:
- `event_id`: Monotonically increasing unique integer
- `timestamp`: UTC ISO 8601 string
- `event_type`: `"trial_executed"`
- `phase`: `"characterization"`, `"search_dd"`, `"search_greedy"`, `"search_random"`, or `"holdout_confirmation"`
- `candidate_subset`: list of element strings
- `trial_index`: 1-indexed within the subset evaluation
- `outcome`: `"SUCCESS"` or `"FAILURE"`
- `model_requests`: Actual request count from tracker delta
- `tokens`: Prompt and completion tokens from trace
- `latency_seconds`: Wall-clock duration
- `tool_executions`: Tool name, arguments, and outputs

### 3.3 Complete Three-Algorithm Holdout Confirmation
Every distinct subset discovered by Stochastic Delta Debugging, Stochastic Greedy Ablation, and Stochastic Random Search must be independently evaluated with $N_{\text{holdout}} = 5$ trials. Holdout results must be reported separately for all three algorithms.

### 3.4 Decoupled Post-Hoc Ground-Truth Evaluation
Ground-truth comparison (`evaluate_model_pilot_ground_truth`) must be performed strictly as a post-processing step after all search algorithms have terminated and serialized their outputs.

### 3.5 Versioned Output Paths
All artifacts from Phase 2B.0.1 must be written to new versioned paths to prevent overwriting historical Phase 2B.0 files:
- Configuration: `benchmarks/results/phase_2b0_1_calibration_config.json`
- Candidate Specs: `benchmarks/results/phase_2b0_1_candidate_specs.json`
- Ground Truth: `benchmarks/results/phase_2b0_1_ground_truth.json`
- Event Log: `benchmarks/results/phase_2b0_1_calibration_event_log.jsonl`
- Results JSON: `benchmarks/results/phase_2b0_1_calibration_results.json`
- Calibration Report: `docs/research/phase_2b0_1_calibration_report.md`
