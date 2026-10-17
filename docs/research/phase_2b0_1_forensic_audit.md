# AgentXRay — Phase 2B.0.1 Forensic Audit & Protocol Reconciliation

**Document Version:** 1.0 (Forensic Audit of Phase 2B.0)  
**Author:** Antigravity (Pair Programming AI Assistant)  
**Date:** 2026-10-17  
**Audited Checkpoint:** Commit `afd51ea` on `main`  
**Remote Provenance:** `https://github.com/yash-mishraa/AgentXRay.git`  
**Runtime:** Apple Silicon Mac (24 GB Unified Memory), Ollama `llama3.2:3b` (GGUF Q4_K_M)  
**Commercial Spend:** Strictly **$0.0000 USD**  
**Audit Purpose:** Comprehensive code-level, trace-level, and statistical reconciliation between the frozen protocol (`docs/research/phase_2b0_local_model_pilot_protocol.md`) and actual Phase 2B.0 execution artifacts.

---

## 1. Executive Summary & Audit Verdict

Phase 2B.0 successfully demonstrated that local neural inference via Ollama (`llama3.2:3b`) can be driven by AgentXRay's black-box harness with zero external network dependencies and $0.0000 USD commercial spend. The planted trigger constraint (`strict_currency_formatting`) genuinely induced the intended AST syntax rejection in the mock calculator tool environment, while the negative baseline and distractor-only controls executed cleanly.

However, a strict forensic audit of the execution artifacts reveals **five material protocol deviations and reporting defects**:
1. **Exhaustive Characterization Incomplete:** The frozen protocol mandated an independent exhaustive characterization of all $2^4 = 16$ candidate subsets at $N = 10$ trials each (160 model requests). In reality, the runner evaluated only **4 subsets** at $N = 5$ trials each ($25\%$ subset coverage, $12.5\%$ trial coverage).
2. **Search-Phase Trial Logging Gap:** In `phase_2b0_calibration_event_log.jsonl`, only 30 `trial_executed` events were recorded (20 in calibration controls, 10 in holdout). The 75 trials executed during the minimizer search phase (DD: 20, Greedy: 30, Random: 25) were captured only as candidate-level aggregations (`algorithm_candidate_evaluated`), omitting per-trial trace, latency, and token records from the event stream.
3. **Incomplete Holdout Validation:** Holdout confirmation was executed only for Delta Debugging and Greedy Ablation (5 trials each on the identical discovered subset $\{T_1\}$). Random Search holdout was entirely omitted from `discovered_map`, leaving holdout coverage incomplete.
4. **Statistical Multiplicity & Narrative Conflation:** The narrative summary cited a paired t-test $p = 1.4883 \times 10^{-17}$ (which belongs to the 400-instance synthetic benchmark errata from Phase 2A.2) adjacent to Phase 2B.0. In Phase 2B.0, $N = 1$ task instance was evaluated; a paired t-test cannot be computed on $N = 1$ ($\nu = 0$). Furthermore, `statistically_supported` minimality on a singleton $\{T_1\}$ was evaluated via an asymptotic pooled z-test with zero cell counts, where Benjamini-Hochberg FDR control is vacuous ($m = 1$).
5. **Coupled Ground-Truth Specification:** While the black-box evaluator and minimizers did not inspect ground truth during search, `ModelPilotGroundTruthSpec` was instantiated within the same runner method (`run_minimization_algorithms`), rather than being ingested strictly post-hoc from an isolated artifact.

**Audit Classification:**  
Phase 2B.0 must be classified as **`EXPLORATORY_ONLY`**. The claim of `CONFIRMATORY_READY` in the Phase 2B.0 report was premature due to the uncompleted 16-subset exhaustive characterization and event logging omissions.

---

## 2. Protocol-versus-Execution Reconciliation Matrix

| Frozen Protocol Requirement | Actual Implementation / Execution Evidence | Evidence Location | Status | Consequence for Interpretation | Required Remediation |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Exhaustive Characterization ($2^4 = 16$ subsets, $N = 10$)** | Evaluated only 4 configurations ($\emptyset$, $\{T_1\}$, $\{D_1, D_2, D_3\}$, $\{T_1, D_1, D_2, D_3\}$) at $N = 5$ trials each. 12 subsets were never characterized. | `phase_2b0_calibration_event_log.jsonl` (lines 5–28), `calibration_runner.py` (lines 375–425) | **DEVIATED** | Interactions among subset pairs (e.g. $\{T_1, D_1\}$) and individual distractor singletons were not empirically characterized before search. | Execute full 16-subset characterization in a versioned corrective calibration (Protocol v1.1). |
| **Trial-Level Event Logging for All Phases** | Only 30 `trial_executed` events recorded. Search-phase trials (75 trials) logged only at candidate level (`algorithm_candidate_evaluated`). | `phase_2b0_calibration_event_log.jsonl` (30 trial records out of 105 total executions) | **DEVIATED** | Raw prompt tokens, completion tokens, latency, and tool arguments for individual search trials are unrecoverable from event log. | Wrap `target.execute` in runner to emit `trial_executed` events for every target invocation across all phases. |
| **Holdout Confirmation for All Discovered Subsets** | Evaluated DD and Greedy holdouts (5 trials each on $\{T_1\}$). Random Search holdout was omitted from `discovered_map`. | `calibration_runner.py` (lines 733–737), `phase_2b0_calibration_results.json` (lines 188–223) | **DEVIATED** | Holdout confirmation is incomplete; Random Search was not evaluated for post-search reproduction. | Include all executed algorithms in holdout confirmation mapping. |
| **Ground-Truth Isolation from Search** | Minimizers and Evaluator observe only `TargetResult` and `AgentExecutionTrace`. No ground-truth leakage during search. | `src/agentxray/agent_harness/target.py` (lines 63–105), `evaluator.py` (lines 44–82) | **CONFORMANT** | Search decisions are strictly black-box and uncorrupted by causal labels. | Maintain architectural boundary; decouple post-hoc ground-truth loading into standalone post-processor. |
| **Failure Taxonomy Enforcement** | Categorized into `FAIL_TOOL_EXEC` (calculator syntax error). Infrastructure errors raise `RuntimeError` and halt immediately. | `target.py` (lines 93–96), `phase_2b0_ground_truth.json` (line 6) | **CONFORMANT** | Infrastructure instability is never conflated with agent behavioral failure. | Preserve exception handling and telemetry checks. |
| **Zero Commercial API Spend ($0.0000)** | Local Ollama daemon at `http://127.0.0.1:11434` with `llama3.2:3b`. Zero external network calls. | `phase_2b0_calibration_config.json` (line 18), `phase_2b0_calibration_results.json` (line 18) | **CONFORMANT** | Financial spend strictly verified at $0.0000 USD. | Maintain local loopback binding checks. |
| **Paired Statistical Comparison ($N > 1$)** | Only $N = 1$ task instance evaluated ($\text{DD}=40$ vs $\text{Greedy}=60$). Paired t-test $p = 1.4883 \times 10^{-17}$ cited in narrative was from Phase 2A errata, not Phase 2B.0. | `phase_2b0_calibration_results.json` (lines 181–186), Phase 2B.0 narrative | **DEVIATED (Narrative)** | Cannot perform inferential hypothesis testing on paired request counts with $N = 1$. | Explicitly withdraw paired hypothesis test claim for Phase 2B.0; report only raw point difference ($\Delta = -20$). |
| **Budget Enforcement & Stopping Rules** | Hard request cap of 100 requests per algorithm enforced via `CostTracker`. DD used 40, Greedy used 60, Random used 50. | `phase_2b0_calibration_results.json` (lines 135, 151, 167) | **CONFORMANT** | Budget arithmetic satisfied; no algorithm breached the 100-request safeguard cap. | Preserve request tracking in all future phases. |

---

## 3. Detailed Forensic Investigations

### 3.1 Discrepancy A — Exhaustive Characterization Audit
Section 3.2 of the frozen protocol (`docs/research/phase_2b0_local_model_pilot_protocol.md`) states:
> *Before executing minimization algorithms, an Independent Exhaustive Characterization will evaluate all 16 configurations across $N_{\text{char}} = 10$ trials each (160 model requests total).*

**Forensic Finding:**  
Inspection of `benchmarks/model_pilot/calibration_runner.py` (lines 375–425) and `phase_2b0_calibration_event_log.jsonl` demonstrates that `run_calibration_controls()` only executed four specific candidate subsets:
1. `ctrl_negative_baseline`: $\emptyset$ (0 elements) — 5 trials
2. `ctrl_positive_trigger`: $\{T_1\}$ (1 element) — 5 trials
3. `ctrl_null_distractors`: $\{D_1, D_2, D_3\}$ (3 elements) — 5 trials
4. `ctrl_full_intervention`: $\{T_1, D_1, D_2, D_3\}$ (4 elements) — 5 trials

The remaining 12 subsets in $\mathcal{P}(C)$ were **never evaluated**:
- Singletons: $\{D_1\}$, $\{D_2\}$, $\{D_3\}$
- Pairs: $\{T_1, D_1\}$, $\{T_1, D_2\}$, $\{T_1, D_3\}$, $\{D_1, D_2\}$, $\{D_1, D_3\}$, $\{D_2, D_3\}$
- Triplets: $\{T_1, D_1, D_2\}$, $\{T_1, D_1, D_3\}$, $\{T_1, D_2, D_3\}$

Furthermore, the sample size per evaluated subset was $N = 5$, not $N_{\text{char}} = 10$.  
**Classification:** **EXHAUSTIVE CHARACTERIZATION NOT COMPLETED.**

### 3.2 Discrepancy B — Request and Execution Accounting Reconciliation
The Phase 2B.0 report cited 210 total model requests across 105 complete agent executions. We performed a line-by-line trace audit:

#### Operational Definitions:
1. **Model Inference Request ($R$):** A single HTTP POST to `/api/chat` yielding one model completion.
2. **Complete Agent Execution ($E$):** A full multi-turn session from user prompt to final answer. In this task, $E$ consistently required $2$ model requests:
   - *Turn 1:* Prompt $\to$ Model generates tool call `calculator(expression=...)`.
   - *Turn 2:* Tool output $\to$ Model generates final response text.
3. **Trial:** One agent execution evaluated under candidate subset configuration $S$.
4. **Candidate-Subset Evaluation:** A batch of $N$ trials executed under identical subset $S$.
5. **Algorithm Run:** A complete minimization execution from initial input to isolated minimal subset.
6. **Holdout Confirmation:** An independent batch of $N$ trials executed on a discovered subset without sharing search cache.

#### Reconciliation Table:

| Phase / Component | Subsets Evaluated | Agent Executions ($E$) | Requests per Execution | Total Model Requests ($R$) | Captured in Event Log as `trial_executed`? | Captured in Event Log as `candidate_evaluated`? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Controls Calibration** | 4 | 20 | 2.00 | 40 | **Yes** (20 events) | **Yes** (4 events) |
| **Stochastic Delta Debugging** | 4 | 20 | 2.00 | 40 | **No** (0 events) | **Yes** (4 callback events) |
| **Stochastic Greedy Ablation** | 6 | 30 | 2.00 | 60 | **No** (0 events) | **Yes** (6 callback events) |
| **Stochastic Random Search** | 5 | 25 | 2.00 | 50 | **No** (0 events) | **Yes** (5 callback events) |
| **Holdout Confirmation** | 2 | 10 | 2.00 | 20 | **Yes** (10 events) | **Yes** (2 events) |
| **Total Pilot** | **21** | **105** | **2.00** | **210** | **30 events** (75 missing) | **21 events** |

**Accounting Finding:**  
While `tracker.total_requests` correctly recorded the 210 model requests and `runner.trial_count` correctly recorded the 105 executions, the runner's logging callback failed to emit `trial_executed` events during search. Therefore, 75 search trials exist only as aggregated candidate counts.

### 3.3 Discrepancy C — Holdout and Algorithm Accounting Audit
In `calibration_runner.py` (line 733):
```python
discovered_map = {
    "delta_debugging": algo_res["delta_debugging"]["discovered_elements"],
    "greedy_ablation": algo_res["greedy_ablation"]["discovered_elements"],
}
holdout_res = self.run_holdout_confirmation(discovered_map)
```
- Random Search discovered $\{T_1\}$ but was omitted from `discovered_map`.
- Delta Debugging discovered $\{T_1\}$ and received 5 holdout trials (5/5 failures).
- Greedy Ablation discovered $\{T_1\}$ and received 5 holdout trials (5/5 failures).
- Because both algorithms isolated the identical subset $\{T_1\}$, the runner executed two separate 5-trial batches on $\{T_1\}$, rather than deduplicating or including all three algorithms.

### 3.4 Discrepancy D — Ground-Truth Isolation and Leakage Audit
We audited the data flow across `ControlledAgent`, `ModelBackedAgentTarget`, `ToolCallingTaskEvaluator`, and `StochasticDeltaDebugMinimizer`:
- `ToolCallingTaskEvaluator` inspects only `AgentExecutionTrace.tool_executions` and `final_response`. It has zero access to `phase_2b0_ground_truth.json` or `ModelPilotGroundTruthSpec`.
- `StochasticDeltaDebugMinimizer` receives only `InputCase` and `runner.execute_trial`. It has no access to ground truth.
- `evaluate_model_pilot_ground_truth()` is executed strictly post-hoc after `minimizer.minimize()` returns.
- **Verdict:** **Zero ground-truth leakage detected.** The epistemic boundary was maintained during search execution.

### 3.5 Discrepancy E — Trigger Validity and Operational Semantics
- **Planted Instruction ($T_1$):** `'strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. "$150" or "$250")'`
- **Observed Model Action:** `llama3.2:3b` emits tool call:
  `calculator(expression="$150 + $250")`
- **Tool Environment Action:** `MockToolEnvironment` parses expression with Python `ast.parse()`, which raises `SyntaxError: invalid syntax` on the leading `$` token. Tool returns `"Error: invalid syntax (<unknown>, line 1)"`.
- **Evaluator Action:** `ToolCallingTaskEvaluator` checks `output.startswith("Error:")` and returns `(False, "Required tool 'calculator' returned an error...")`.
- **Clean Baseline ($\emptyset$):** Model emits `calculator(expression="150 + 250")`, tool returns `"400"`, model responds `"The total reimbursement is $400."` $\implies$ Success ($0/5$ failures).
- **Distractor Null Control ($\{D_1, D_2, D_3\}$):** Model emits clean expression, tool returns `"400"`, model formats text with distractor tags $\implies$ Success ($0/5$ failures).
- **Verdict:** The trigger operates via a valid agent behavioral failure (instruction-following leading to tool parameter schema violation). The tool schema is well-formed and standard.

---

## 4. Recomputation of Statistical Claims

### 4.1 Paired t-Test Clarification
- **Reported in Phase 2B.0 Narrative:** $p = 1.4883 \times 10^{-17}$.
- **Recomputation Audit:** In Phase 2B.0, exactly $N = 1$ paired instance was evaluated ($\text{Requests}_{\text{DD}} = 40, \text{Requests}_{\text{Greedy}} = 60$).
- **Statistical Fact:** For $N = 1$, the sample variance is undefined ($s^2 = \frac{0}{0}$), degrees of freedom $\nu = 0$, and no t-test or Wilcoxon test can be computed.
- **Origin of Number:** The value $p = 1.4883 \times 10^{-17}$ belongs to the **Phase 2A synthetic scaling benchmark** ($N = 400, \nu = 399, t = -9.278$). Its mention in the Phase 2B.0 narrative summary caused confusion and is hereby explicitly retracted as applying to Phase 2B.0. The only valid statistic for Phase 2B.0 is the raw point difference:
  $$\Delta R = 40 - 60 = -20 \text{ requests}$$

### 4.2 Two-Sided Fisher Exact Test Recomputation
For the control contrast $k_1 = 5, n_1 = 5$ (Trigger) vs $k_2 = 0, n_2 = 5$ (Baseline):
$$\text{Table: } \begin{pmatrix} 5 & 0 \\ 0 & 5 \end{pmatrix}, \quad N = 10, \quad K = 5$$
$$P(X = 5) = \frac{\binom{5}{5}\binom{5}{0}}{\binom{10}{5}} = \frac{1 \times 1}{252} \approx 0.003968$$
$$P(X = 0) = \frac{\binom{5}{0}\binom{5}{5}}{\binom{10}{5}} = \frac{1 \times 1}{252} \approx 0.003968$$
$$\text{Two-sided } p = P(X = 0) + P(X = 5) = \frac{2}{252} = \frac{1}{126} \approx 0.0079365 \implies \mathbf{0.00794}$$
The reported Fisher exact p-value is **verified mathematically exact**.

### 4.3 1-Minimality Verification & FDR Audit
- The discovered subset was $\{T_1\}$ (cardinality 1).
- Single-element removal: $\emptyset$ (0/5 failures) vs $\{T_1\}$ (5/5 failures).
- Two-sample pooled z-statistic: $z = 3.1623, p_{\text{asymptotic}} = 0.001565$.
- Number of removal tests: $m = 1$.
- **FDR Limitation:** Benjamini-Hochberg adjustment with $m = 1$ is an identity transformation ($p_{(1)} \times 1 / 1 = p$). Claiming FDR control on a single hypothesis test is trivial and uninformative.
- **Asymptotic Limitation:** Evaluating $z = 3.16$ using standard normal approximation with cell count 0 and $N = 5$ is an asymptotic approximation, not an exact test. Fisher's exact test ($p = 0.00794$) should be the authoritative criterion for small samples.

---

## 5. Audit Conclusions & Next Steps

1. **Phase 2B.0 Status Downgraded:** Changed from `CONFIRMATORY_READY` to **`EXPLORATORY_ONLY`**.
2. **Protocol Amendment Authorized:** Protocol v1.1 will formally define the complete 16-subset characterization, full-lifecycle trial event logging, and three-algorithm holdout validation.
3. **Corrective Calibration Pilot Authorized:** A clean, versioned corrective calibration run (Phase 2B.0.1) under Protocol v1.1 will be executed to generate complete empirical evidence across all 16 subsets before considering Phase 2B.1.
