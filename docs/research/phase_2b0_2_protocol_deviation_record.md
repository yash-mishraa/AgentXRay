# AgentXRay — Phase 2B.0.2: Protocol Deviation Record & Protocol Amendment v1.2

**Document Status:** Formal Protocol Deviation Record & Versioned Protocol Amendment  
**Date:** 2026-10-17  
**Previous Protocols Preserved:**
- Protocol v1.0: `docs/research/phase_2a_protocol_v1.md` (Commit `aa24471`)
- Protocol v1.1: `docs/research/phase_2a_protocol_v1.1.md` / `docs/research/phase_2b0_1_protocol_deviation_and_amendment.md` (Commit `6cbc1fb`)

---

## 1. Formal Record of Phase 2B.0.1 Qualification Failure

In Phase 2B.0.1, the local neural model calibration pilot was executed under Protocol v1.1 using local `llama3.2:3b`.
- **Characterization Size:** 16 subsets ($2^4$), 5 trials each ($N=80$ agent trials, 160 model requests).
- **Observed Result:**
  - Subsets with $T_1$ ($n=8$): 100% failure rate (40/40 failures, all `FAIL_TOOL_EXEC`).
  - Subsets without $T_1$ ($n=8$): 7 subsets had 0% failure rate (35/35 passes), but 1 subset (Row 10, $\{D_1, D_3\}$) had 100% failure rate (5/5 failures, all `FAIL_BEHAVIOR`).
- **Trigger Qualification Criterion:**
  $$\max_{S: T_1 \notin S} \hat{p}(S) \le 0.20$$
- **Empirical Value:** $\max_{S: T_1 \notin S} \hat{p}(S) = 1.00$ ($> 0.20$).
- **Recorded Decision:** `HARNESS_INADEQUATE_OR_EXPLORATORY`.
- **Stopping Rule Execution:** Correctly halted. Search and holdout were skipped and logged as `search_skipped`.

---

## 2. Identified Deviations & Defects in Phase 2B.0.1

### Deviation 1: Task Design Confounding (Distractor Conflict)
- **Component:** `CANDIDATE_DISTRACTOR_1` (`elem_2`) and `CANDIDATE_DISTRACTOR_3` (`elem_4`).
- **Defect:** $D_1$ requested `"concise_summary: provide clear one-line reimbursement statement"` while $D_3$ requested `"audit_memo: append verification timestamp to final answer"`.
- **Consequence:** On `llama3.2:3b`, when both were presented without $D_2$, the model compressed its response to a single line containing only the timestamp (`'audit_memo: 2023-12-01'`), omitting the computed amount `"400"`. This generated a 100% failure rate on a trigger-absent configuration.
- **Root Cause:** In synthetic and simulated settings, distractors are purely orthogonal. In neural model settings, stylistic instructions can create mutual length and formatting interference.

### Deviation 2: Evaluator Failure-Class Conflation
- **Component:** `ToolCallingTaskEvaluator` in `agentxray/agent_harness/evaluator.py`.
- **Defect:** Evaluator returned a single boolean `is_success`, checking both tool execution syntax errors (`t.output.startswith("Error:")`) and final response substring presence (`expected_substring in resp`).
- **Consequence:** In Row 10, the tool executed with 100% success (`calculator(expression='150 + 250')` returned `'400'`), but the trial was marked `FAILURE` because the downstream natural language response dropped `"400"`. This conflated a behavioral text formatting quirk with the intended tool execution syntax failure (`FAIL_TOOL_EXEC`).

### Deviation 3: Reporting Generation Default Artifact
- **Component:** `generate_markdown_report()` in `calibration_runner_v1_1.py`.
- **Defect:** When search and holdout were skipped due to unconfirmed trigger, `holdout_res` was empty `{}`. The template accessed `.get('failure_rate', 0.0) * 100` and `.get('elements', [])`, producing table entries showing `0.0%` failure rate with `[]` discovered elements for unexecuted holdout trials.
- **Consequence:** An unexecuted stage was rendered as an empirical zero-failure observation, creating the false impression of an executed holdout.

### Deviation 4: Characterization Table Configuration Masking
- **Component:** `generate_markdown_report()` in `calibration_runner_v1_1.py`.
- **Defect:** Row descriptions printed only `f"{len(s['elements'])} elements"` instead of explicit candidate IDs.
- **Consequence:** Readers could not identify which specific 2-element subset failed without inspecting the raw JSON or event log.

---

## 3. Protocol Amendment v1.2 Specification

To recover a scientifically defensible calibration without tuning the benchmark to fit an algorithm, Protocol Amendment v1.2 specifies four corrective requirements:

### Requirement 1: Non-Conflicting, Orthogonal Distractor Phrasing
The candidate intervention space $\mathcal{C}$ for `pilot_calibrated_currency_v1_2` retains the exact same planted trigger $T_1$ and structure, but de-conflicts the distractors so that no distractor imposes destructive length constraints:
- **$T_1$ (Planted Trigger, unchanged):**
  `strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. "$150" or "$250")`
- **$D_1$ (Distractor 1, revised):**
  `expense_category: specify expense category as Travel & Subsistence` (Replaces `"one-line reimbursement statement"`).
- **$D_2$ (Distractor 2, unchanged):**
  `compliance_tag: include audit reference code REF-2026`
- **$D_3$ (Distractor 3, de-conflicted):**
  `audit_memo: include verification timestamp in final audit notes`

### Requirement 2: Class-Aware Outcome Evaluation
The evaluator must distinguish:
- `FAIL_TOOL_EXEC`: The agent generated invalid tool arguments causing the tool execution environment to return an error (the intended failure class of $T_1$).
- `FAIL_BEHAVIOR`: The tool executed cleanly, but the agent omitted required output content or violated behavioral constraints.
- `FAIL_INFRA`: Provider timeout, connection drop, or runtime abort.
- `CLEAN_PASS`: Tool executed cleanly and output satisfied observable requirements.

For the tool-execution-focused calibrated currency benchmark, `FAIL_TOOL_EXEC` is the primary failure criterion. Subsets where the tool executed cleanly (`150 + 250 -> 400`) must not be marked as tool execution failures.

### Requirement 3: Explicit Status for Skipped Benchmark Stages
The report generator and machine-readable results must explicitly record stage status:
- If characterization qualification fails:
  - Minimization algorithm tables must display `BLOCKED_BY_QUALIFICATION`.
  - Holdout confirmation tables must display `BLOCKED_BY_QUALIFICATION`.
  - Discovered subsets must be `NOT_EVALUATED` rather than `[]`.
  - Failure rates must be `N/A` rather than `0.0%`.

### Requirement 4: Explicit Element Identifiers in All Tables
Every table row in characterization, minimization, and holdout must display the exact candidate element identifiers (e.g. `[elem_1, elem_3]`) and human-readable roles.

---

## 4. Execution Guardrails & Governance

- **Execution Tier:** `LOCAL_MODEL` (Ollama daemon, local loopback `127.0.0.1:11434`, model `llama3.2:3b`).
- **Commercial API Spend:** Strictly **$0.0000 USD**.
- **Request Caps:**
  - Exhaustive Characterization: Maximum 400 model requests ($16 \times 5 \times 2 = 160$ expected).
  - Search Algorithms: Maximum 120 model requests per algorithm.
  - Holdout Confirmation: Maximum 150 model requests total.
- **Stopping Rules:**
  - If $\min_{S: T_1 \in S} \hat{p}(S) < 0.70$ OR $\max_{S: T_1 \notin S} \hat{p}(S) > 0.20$: Abort search immediately, log `search_skipped`, declare `HARNESS_INADEQUATE_OR_EXPLORATORY`.
  - If qualification succeeds: Execute Stochastic Delta Debugging, Greedy Ablation, and Random Search, followed by post-hoc ground-truth evaluation and complete holdout confirmation across all 3 algorithms.
- **Phase 2B.1 Authorization:** **STRICTLY UNAUTHORIZED**. A successful calibration qualifies only the local testbed; confirmatory comparative benchmarking requires a separate research authorization.
