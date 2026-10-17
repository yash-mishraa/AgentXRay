# AgentXRay — Phase 2B.0.2: Failure-Source Forensics, Trigger Validity & Calibration Recovery

**Document Status:** Complete Forensic Investigation & Root-Cause Diagnosis  
**Date:** 2026-10-17  
**Investigators:** AgentXRay Autonomous Research Agent (Pair Programming with User)  
**Historical Run Audited:** Phase 2B.0.1 Corrective Local-Model Pilot Calibration (`phase_2b0_1_*`, commit `6cbc1fb`)  
**Target Model / Provider:** `llama3.2:3b` (Ollama local inference on Apple Silicon Mac, 24 GB Unified Memory)  
**Commercial API Spend:** Strictly **$0.0000 USD**  
**Final Forensic Decision:** `REVISE_TASK_DESIGN` and `FIX_IMPLEMENTATION`  
**Phase 2B.1 Status:** Strictly **UNAUTHORIZED**  

---

## 1. Executive Summary & Objective

In Phase 2B.0.1, AgentXRay executed an exhaustive characterization of a 4-element candidate space $\mathcal{P}(C)$ ($2^4 = 16$ configurations, $N=5$ trials each) using the local neural model `llama3.2:3b`. The report recorded:
- 8 of 8 subsets containing the planted trigger $T_1$ (`strict_currency_formatting`) failed with 100% failure rate ($\hat{p} = 1.00$, 5/5 failures).
- 7 of 8 subsets without $T_1$ passed with 0% failure rate ($\hat{p} = 0.00$, 0/5 failures).
- Exactly one trigger-absent configuration—Row 10 in the report table—failed in **5 out of 5 trials** ($\hat{p} = 1.00$).

Because the prespecified trigger qualification rule requires $\max_{S: T_1 \notin S} \hat{p}(S) \le 0.20$, the calibration runner correctly declared the trigger unconfirmed and halted with decision:
$$\text{Qualification Decision} = \mathbf{HARNESS\_INADEQUATE\_OR\_EXPLORATORY}$$

The objective of Phase 2B.0.2 is to conduct a code-level and trace-level forensic investigation to answer:
1. What was the exact candidate configuration evaluated in Row 10?
2. Why did this trigger-absent configuration fail in 100% of trials?
3. Which of the 7 candidate failure explanations (mapping defect, task design, evaluator defect, harness defect, model behavior, reporting defect, unresolved) are supported by empirical evidence?
4. What are the true trial and inference request counts?
5. Why did the Phase 2B.0.1 report display `N/A` for minimization metrics and `0.0%` with `[]` for holdout?
6. Is the benchmark task salvageable, and under what versioned protocol can a valid calibration be established?

---

## 2. Repository Provenance & Historical Artifact Preservation

Before conducting analysis or modifying code, the repository state was verified:
- **Git HEAD:** Commit `6cbc1fb` on branch `main`, remote `origin/main` (`https://github.com/yash-mishraa/AgentXRay.git`).
- **Working Tree:** Completely clean (`git status -s` produced 0 uncommitted changes).
- **Test Suite:** 176 of 176 tests passing via `./.venv/bin/pytest`.
- **Synthetic Benchmark Validation:** `phase_2a1_event_log.jsonl` verified clean via `./.venv/bin/python benchmarks/synthetic/reproduce_phase_2a1.py --validate-only` (Status: `VALIDATION_PASSED`, 0 discrepancies).

### Historical Artifacts Preserved Intact
All Phase 2B.0 and Phase 2B.0.1 artifacts remain untouched in their versioned locations:
- `benchmarks/results/phase_2b0_1_calibration_config.json`
- `benchmarks/results/phase_2b0_1_calibration_event_log.jsonl` (103 events, SHA-256 preserved)
- `benchmarks/results/phase_2b0_1_calibration_results.json`
- `benchmarks/results/phase_2b0_1_candidate_specs.json`
- `benchmarks/results/phase_2b0_1_ground_truth.json`
- `docs/research/phase_2b0_1_calibration_report.md`
- `docs/research/phase_2b0_1_forensic_audit.md`
- `docs/research/phase_2b0_1_protocol_deviation_and_amendment.md`

No historical logs or results were modified to mask the qualification failure.

---

## 3. Forensic Investigation of Row 10 (Subset 10)

### 3.1 Candidate Element Mapping
The candidate space defined in `phase_2b0_1_candidate_specs.json` contains 4 elements:
- `elem_1` ($T_1$, planted trigger): `'strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. "$150" or "$250")'`
- `elem_2` ($D_1$, distractor 1): `'concise_summary: provide clear one-line reimbursement statement'`
- `elem_3` ($D_2$, distractor 2): `'compliance_tag: include audit reference code REF-2026'`
- `elem_4` ($D_3$, distractor 3): `'audit_memo: append verification timestamp to final answer'`

### 3.2 Reconstruction of Row 10
In `phase_2b0_1_calibration_results.json` and `phase_2b0_1_calibration_event_log.jsonl`:
- **Canonical Subset ID:** `char_subset_10_len2`
- **Subset Index:** 10
- **Cardinality:** 2 elements
- **Elements Present:** `['elem_2', 'elem_4']` = $\{D_1, D_3\}$
- **Contains $T_1$:** **`False`** (Trigger $T_1$ was completely absent).
- **Supporting Event IDs:**
  - Trial 1: Event ID 59 (`trial_index`: 46)
  - Trial 2: Event ID 60 (`trial_index`: 47)
  - Trial 3: Event ID 61 (`trial_index`: 48)
  - Trial 4: Event ID 62 (`trial_index`: 49)
  - Trial 5: Event ID 63 (`trial_index`: 50)
  - Candidate Evaluation Summary: Event ID 64

### 3.3 Full Agent Trace Inspection for Row 10
Across all 5 trials of `char_subset_10_len2`, the execution trace was 100% deterministic:
1. **Turn 1 (Tool Invocation Request):**
   - The model received user prompt: `"Calculate total reimbursement for expenses of 150 and 250."`
   - Constraints provided:
     - `- concise_summary: provide clear one-line reimbursement statement`
     - `- audit_memo: append verification timestamp to final answer`
   - The model invoked the tool:
     $$\text{Tool Call: } \texttt{calculator}(\text{expression}='150 + 250')$$
   - The mock calculator evaluated `'150 + 250'` and returned:
     $$\text{Tool Output: } \texttt{'400'}$$
   - **Crucial Diagnostic Fact:** The calculator tool executed with **100% success**. It did **NOT** receive currency symbols and did **NOT** return a syntax error!

2. **Turn 2 (Final Response Generation):**
   - The model received the tool execution output `'400'`.
   - In generating its final response, the model attempted to satisfy both constraints:
     - "provide clear one-line reimbursement statement"
     - "append verification timestamp to final answer"
   - Under this joint instruction, `llama3.2:3b` emitted:
     $$\text{Final Response: } \texttt{'audit_memo: 2023-12-01'}$$
   - The model compressed the output into a single line consisting solely of the memo and timestamp, omitting the numeric total `"400"`.

3. **Turn 3 (Evaluation):**
   - `ToolCallingTaskEvaluator` evaluated the trace.
   - Requirement 1 (Tool Called): `calculator` was called $\implies$ **PASS**.
   - Requirement 2 (Tool Output): Output did not start with `"Error:"` $\implies$ **PASS**.
   - Requirement 3 (Substring Check): `expected_substring = "400"`.
   - Check: `"400" in "audit_memo: 2023-12-01".lower()` $\implies$ **`False`**.
   - Evaluator returned:
     $$\texttt{(False, "Final response did not contain expected content '400'. Got: 'audit_memo: 2023-12-01'")}$$
   - TargetOutcome assigned: **`FAILURE`**.

---

## 4. Evaluation of the Seven Candidate Explanations

| Hypothesis | Description | Evidence & Findings | Supported? |
| :--- | :--- | :--- | :---: |
| **1. Mapping / Serialization Defect** | The configuration supposedly lacking $T_1$ actually had $T_1$, or candidate IDs were misaligned. | Logged events 59–63 explicitly record `candidate_subset: [D_1, D_3]`. The prompt builder concatenated only $D_1$ and $D_3$. Tool call expression was `'150 + 250'`, not `'$150 + $250'`. Serialization was flawless. | **NO** |
| **2. Task-Design Confounding** | One or more distractors independently or interactively induce failure. | When $D_1$ ("one-line reimbursement") and $D_3$ ("append timestamp") appear together *without* $D_2$ ("compliance tag"), `llama3.2:3b` compresses the answer to just the timestamp. However, this failure is textual omission, not tool syntax error. | **YES (Partial)** |
| **3. Evaluator Defect** | Evaluator applies an inconsistent success criterion or conflates failure classes. | `ToolCallingTaskEvaluator` treats tool syntax errors (`FAIL_TOOL_EXEC`) and textual substring omission (`FAIL_BEHAVIOR`) as identical `TargetOutcome.FAILURE`, despite ground truth stating the intended failure mechanism is AST syntax rejection. | **YES** |
| **4. Harness / Tool Defect** | Harness crashes, schemas fail, or mock tool errors out unexpectedly. | Mock calculator executed in 0.1 ms, returned `'400'`, finish reason was `'stop'`, 0 infrastructure errors. | **NO** |
| **5. Model Behavior** | Model fails even when trigger is absent due to model idiosyncrasies. | Small 3B models exhibit prompt sensitivity when concise-length instructions conflict with multi-attribute extraction instructions. | **YES (Contributing)** |
| **6. Aggregation / Reporting Defect** | Raw outcomes do not support report table, or missing data renders as valid 0.0% outcomes. | Two distinct reporting defects identified: (a) subset table masked element IDs as `"2 elements"`; (b) skipped holdout rendered as `0.0%` with `[]` discovered subset. | **YES** |
| **7. Unresolved or Mixed Cause** | Available evidence cannot isolate the failure mechanism. | The mechanism is fully resolved: compound interaction between task distractor phrasing, evaluator failure-class conflation, and reporting defaults. | **RESOLVED** |

### Synthesis: The True Failure Mechanism
The failure of Row 10 was caused by **Task-Design Confounding** combined with **Evaluator Failure-Class Conflation**:
1. **The Planted Trigger ($T_1$) Mechanism:** $T_1$ targets the **tool boundary** (`FAIL_TOOL_EXEC`). When $T_1$ is present, the agent passes `'$150 + $250'` to the calculator, which the AST evaluator rejects with `Error: invalid syntax`. In all 8 trigger-present subsets (40 trials), this syntax error occurred in 40/40 trials (100%).
2. **The Distractor Interaction ($D_1 + D_3$) Mechanism:** $D_1 + D_3$ targets the **text synthesis boundary** (`FAIL_BEHAVIOR`). The tool execution succeeds with `400`, but the model drops `"400"` from its one-line final response.
3. **The Evaluator's Flaw:** The evaluator collapsed both distinct failure modes into a single opaque boolean `is_success = False`. This masked that the tool execution boundary was completely clean in Row 10.

---

## 5. Trial and Inference Request Accounting

Every event in `phase_2b0_1_calibration_event_log.jsonl` was independently verified:

| Accounting Metric | Frozen Expectation | Raw Log Count | Reconciliation Status |
| :--- | :---: | :---: | :---: |
| **Candidate Subsets Attempted** | 16 | 16 | Exact Match |
| **Candidate Subsets Completed** | 16 | 16 | Exact Match |
| **Trials per Subset** | 5 | 5 | Exact Match |
| **Total Complete Agent Executions** | 80 | 80 | Exact Match |
| **Total Model Inference Requests** | 160 | 160 | Exact Match (Exactly 2 per trial) |
| **Successful Model Responses** | 160 | 160 | Exact Match (100% 200 OK) |
| **Total Model Tokens Consumed** | N/A | 36,350 | Fully Instrumented |
| **Infrastructure Errors / Retries** | 0 | 0 | Clean Run |
| **Tool Execution Syntax Failures (`FAIL_TOOL_EXEC`)** | 40 | 40 | Exactly the 8 $T_1$-present subsets |
| **Behavioral Text Omission Failures (`FAIL_BEHAVIOR`)** | 0 (expected) | 5 | Exactly the 5 trials of Row 10 ($D_1 + D_3$) |
| **Clean Passes (`SUCCESS`)** | 40 | 35 | 7 of 8 non-$T_1$ subsets (35 trials) |
| **Total Structured Log Events** | N/A | 103 | Immutable records 1 to 103 |

---

## 6. Audit of Runner Stop Conditions and Report Rendering

### 6.1 Runner Control Flow Verification
Lines 686–704 of `calibration_runner_v1_1.py`:
```python
should_run_search = char_res["qualification_criteria_met"]
if should_run_search:
    algo_res = self.run_minimization_algorithms()
    ...
else:
    self.log_event("search_skipped", {
        "reason": "Trigger unconfirmed during exhaustive characterization; minimization aborted."
    })
```
- When trigger qualification failed due to Row 10, `char_res["qualification_criteria_met"]` was correctly set to `False`.
- The runner executed Event 101 (`search_skipped`).
- **No minimization algorithms were executed.**
- **No holdout trials were executed.**
- The runner respected the stopping condition.

### 6.2 The Cause of `N/A` and `0.0%` with `[]`
In `generate_markdown_report()`:
1. In Section 3 (Minimization Benchmark Results), `algo_res` was empty `{}`. The template used `.get('model_requests', 'N/A')`, which properly rendered `'N/A'` across all algorithm metrics.
2. In Section 4 (Holdout Confirmation), `holdout_res` was empty `{}`. The template used:
   ```python
   holdout.get('delta_debugging', {}).get('elements', [])
   holdout.get('delta_debugging', {}).get('failure_rate', 0.0) * 100
   ```
   Because `holdout` was empty, `.get()` fell back to defaults:
   - `elements`: `[]`
   - `trials`: `0`
   - `failures`: `0`
   - `failure_rate`: `0.0%`
   - `wilson_ci`: `[]`

This caused the report table to render:
```markdown
| **Delta Debugging** | `[]` | 0 | 0 | **0.0%** | [] |
```
**Reporting Errata:** This rendering was defective. A skipped stage with zero observations was formatted as an empirical finding of 0.0% failure with discovered subset `[]`. It must be explicitly rendered as `BLOCKED_BY_QUALIFICATION` or `NOT_RUN`.

---

## 7. Complete 16-Subset Characterization Reconciliation Table

Below is the verified trial-level reconstruction from `benchmarks/results/phase_2b0_2_characterization_reconciliation.json`:

| Row | Subset Case ID | Candidate Element IDs | Elements Role Description | $T_1$? | Trials ($n$) | Failures ($k$) | Failure Rate | Failure Classification | Supporting Event IDs |
| :-: | :--- | :--- | :--- | :---: | :-: | :-: | :-: | :--- | :---: |
| 1 | `char_subset_01_len0` | `[]` | Empty Baseline ($\emptyset$) | NO | 5 | 0 | **0.0%** | `CLEAN_PASS` | 5–9 |
| 2 | `char_subset_02_len1` | `['elem_1']` | $\{T_1\}$ | YES | 5 | 5 | **100.0%** | `FAIL_TOOL_EXEC` | 11–15 |
| 3 | `char_subset_03_len1` | `['elem_2']` | $\{D_1\}$ | NO | 5 | 0 | **0.0%** | `CLEAN_PASS` | 17–21 |
| 4 | `char_subset_04_len1` | `['elem_3']` | $\{D_2\}$ | NO | 5 | 0 | **0.0%** | `CLEAN_PASS` | 23–27 |
| 5 | `char_subset_05_len1` | `['elem_4']` | $\{D_3\}$ | NO | 5 | 0 | **0.0%** | `CLEAN_PASS` | 29–33 |
| 6 | `char_subset_06_len2` | `['elem_1', 'elem_2']` | $\{T_1, D_1\}$ | YES | 5 | 5 | **100.0%** | `FAIL_TOOL_EXEC` | 35–39 |
| 7 | `char_subset_07_len2` | `['elem_1', 'elem_3']` | $\{T_1, D_2\}$ | YES | 5 | 5 | **100.0%** | `FAIL_TOOL_EXEC` | 41–45 |
| 8 | `char_subset_08_len2` | `['elem_1', 'elem_4']` | $\{T_1, D_3\}$ | YES | 5 | 5 | **100.0%** | `FAIL_TOOL_EXEC` | 47–51 |
| 9 | `char_subset_09_len2` | `['elem_2', 'elem_3']` | $\{D_1, D_2\}$ | NO | 5 | 0 | **0.0%** | `CLEAN_PASS` | 53–57 |
| 10 | `char_subset_10_len2` | `['elem_2', 'elem_4']` | $\{D_1, D_3\}$ | NO | 5 | 5 | **100.0%** | `FAIL_BEHAVIOR` | 59–63 |
| 11 | `char_subset_11_len2` | `['elem_3', 'elem_4']` | $\{D_2, D_3\}$ | NO | 5 | 0 | **0.0%** | `CLEAN_PASS` | 65–69 |
| 12 | `char_subset_12_len3` | `['elem_1', 'elem_2', 'elem_3']` | $\{T_1, D_1, D_2\}$ | YES | 5 | 5 | **100.0%** | `FAIL_TOOL_EXEC` | 71–75 |
| 13 | `char_subset_13_len3` | `['elem_1', 'elem_2', 'elem_4']` | $\{T_1, D_1, D_3\}$ | YES | 5 | 5 | **100.0%** | `FAIL_TOOL_EXEC` | 77–81 |
| 14 | `char_subset_14_len3` | `['elem_1', 'elem_3', 'elem_4']` | $\{T_1, D_2, D_3\}$ | YES | 5 | 5 | **100.0%** | `FAIL_TOOL_EXEC` | 83–87 |
| 15 | `char_subset_15_len3` | `['elem_2', 'elem_3', 'elem_4']` | $\{D_1, D_2, D_3\}$ | NO | 5 | 0 | **0.0%** | `CLEAN_PASS` | 89–93 |
| 16 | `char_subset_16_len4` | `['elem_1', 'elem_2', 'elem_3', 'elem_4']` | $\{T_1, D_1, D_2, D_3\}$ | YES | 5 | 5 | **100.0%** | `FAIL_TOOL_EXEC` | 95–99 |

---

## 8. Epistemic Independence of Ground Truth

1. **Isolation from Search Algorithms:** Neither `StochasticDeltaDebugMinimizer`, `run_stochastic_greedy_ablation`, nor `run_stochastic_random_search` has access to `ground_truth.json`. They observe only binary `TargetResult(outcome=SUCCESS/FAILURE)` via the runner.
2. **Isolation from Evaluator:** `ToolCallingTaskEvaluator` judges only runtime execution traces (tools called, tool outputs, final response text). It does not consult ground truth.
3. **Causal Distinction Between Planted Trigger and Empirical Minimality:**
   - **Planted Trigger:** $T_1$ was designed to cause calculator syntax errors (`FAIL_TOOL_EXEC`).
   - **Empirical Minimality in Phase 2B.0.1:** Under the uncalibrated evaluator, there were *two distinct* minimal failure-inducing subsets in $\mathcal{P}(C)$:
     1. $\{T_1\}$ (cardinality 1, mechanism: `FAIL_TOOL_EXEC`)
     2. $\{D_1, D_3\}$ (cardinality 2, mechanism: `FAIL_BEHAVIOR`)
   - Because $\{D_1, D_3\}$ also caused 100% failure, $T_1$ was **not** the unique minimal failure condition of the task system.
   - Ground truth cannot simply be assumed; it must be empirically isolated and validated against non-interference criteria before comparative benchmarks can proceed.

---

## 9. Final Forensic Decision

We make the joint decision:
$$\mathbf{REVISE\_TASK\_DESIGN} \quad \text{and} \quad \mathbf{FIX\_IMPLEMENTATION}$$

### Rationale:
1. **`REVISE_TASK_DESIGN`:** The distractor $D_1$ (`"concise_summary: provide clear one-line reimbursement statement"`) conflicts with attribute-appending distractors on small local models. Distractors in delta-debugging benchmarks must be non-conflicting and orthogonal so they test ablation without independently causing failure.
2. **`FIX_IMPLEMENTATION`:**
   - The evaluator must support failure-class awareness, allowing experiments targeting `FAIL_TOOL_EXEC` to evaluate tool execution validity without being confounded by downstream stylistic variations in natural language text.
   - The calibration runner and report generator must explicitly distinguish skipped stages (`NOT_RUN` / `BLOCKED_BY_QUALIFICATION`) from completed stages with zero observations, eliminating deceptive defaults (`0.0%` and `[]`).
   - Characterization tables must print explicit candidate IDs for every row.

**Phase 2B.1 remains unauthorized.**
Any subsequent calibration must be executed under a frozen **Protocol v1.2** with versioned artifacts (`phase_2b0_2_*`).
