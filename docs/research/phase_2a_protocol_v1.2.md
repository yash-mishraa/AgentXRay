# AgentXRay — Protocol Specification v1.2

**Document Status:** Frozen Experimental Protocol for Phase 2B.0.2 Corrective Calibration Pilot  
**Date:** 2026-10-17  
**Supersedes:** Protocol v1.1 (`docs/research/phase_2a_protocol_v1.1.md`)  
**Scope:** Controlled Local-Model Pilot Calibration, Failure Classification, and Holdout Confirmation  
**Target Environment:** Local Apple Silicon Mac (Ollama daemon, `llama3.2:3b`, Q4_K_M)  
**Commercial Spend Authorization:** Strictly **$0.0000 USD**  

---

## 1. Research Question & Empirical Objective

This protocol governs the corrective calibration pilot for AgentXRay:
> Can stochastic delta debugging over observable agent boundaries identify 1-minimal failure-inducing conditions in a model-backed agent under bounded trial budgets, and how does its performance compare with greedy ablation and random search baselines?

Before comparative performance claims can be evaluated, the experimental harness and failure trigger must be empirically validated across the full power set of candidate interventions $\mathcal{P}(C)$.

---

## 2. Candidate Intervention Space ($K = 4$)

The intervention space contains 1 planted trigger and 3 orthogonal, non-conflicting distractors:

| Element ID | Role | Specification | Intended Behavioral Impact |
| :---: | :---: | :--- | :--- |
| `elem_1` | **Planted Trigger ($T_1$)** | `strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. "$150" or "$250")` | Induces tool argument syntax error (`FAIL_TOOL_EXEC`) by violating AST math grammar in mock calculator. |
| `elem_2` | **Distractor 1 ($D_1$)** | `expense_category: specify expense category as Travel & Subsistence` | Semantic annotation; does not alter calculation or constrain response length. |
| `elem_3` | **Distractor 2 ($D_2$)** | `compliance_tag: include audit reference code REF-2026` | Tracking metadata; does not alter calculation or constrain response length. |
| `elem_4` | **Distractor 3 ($D_3$)** | `audit_memo: include verification timestamp in final audit notes` | Timestamp note; does not alter calculation or constrain response length. |

**De-conflicting Rationale:** In Protocol v1.1, $D_1$ requested a "one-line reimbursement statement" which conflicted with $D_3$'s request for a verification timestamp on `llama3.2:3b`, prompting the model to omit the calculation result. Protocol v1.2 replaces $D_1$ with an expense categorization directive that preserves multi-attribute formatting without length conflict.

---

## 3. Observable Outcome Evaluation & Failure Classification

The agent is evaluated strictly on observable boundary interactions. Ground truth is never visible to the evaluator.

### Evaluator Criteria:
1. **Tool Invocation:** The agent must invoke the required tool (`calculator`).
2. **Tool Execution Integrity:** The tool must execute without error (output must not begin with `"Error:"`).
3. **Response Content:** The final response must reflect the computed reimbursement (`"400"`).

### Failure Classification Taxonomy:
- `CLEAN_PASS`: Tool executed cleanly, output returned `"400"`, final response satisfied criteria.
- `FAIL_TOOL_EXEC`: Tool was not called, or called with malformed arguments causing execution failure (intended trigger mechanism).
- `FAIL_BEHAVIOR`: Tool executed cleanly, but final response omitted required content or included forbidden phrases.
- `FAIL_INFRA`: Provider network timeout, daemon disconnect, or budget exhaustion.

---

## 4. Calibration Workflow & Stopping Rules

### Phase 1: Exhaustive Characterization
- Evaluate all $2^4 = 16$ configurations in $\mathcal{P}(C)$.
- Sample size: $N = 5$ trials per configuration ($16 \times 5 = 80$ trials, 160 model requests).
- **Trigger Qualification Gate:**
  $$\min_{S: T_1 \in S} \hat{p}(S) \ge 0.70 \quad \text{AND} \quad \max_{S: T_1 \notin S} \hat{p}(S) \le 0.20$$
- **Stopping Rule:**
  - If the qualification gate fails: Abort comparative minimization immediately. Log `search_skipped`. Record decision `HARNESS_INADEQUATE_OR_EXPLORATORY`. Do not run holdout.
  - If the qualification gate passes: Declare `TRIGGER_CONFIRMED`. Proceed to Phase 2.

### Phase 2: Comparative Minimization Search (Conditional)
Evaluate three algorithms starting from the full candidate space $C = \{T_1, D_1, D_2, D_3\}$:
1. **Stochastic Delta Debugging (ddmin):** Binary search over partition subsets.
2. **Stochastic Greedy Ablation:** Single-element backward elimination.
3. **Stochastic Random Search (Seed 42):** Uniform random sampling baseline.
- Trial budget cap: 120 model requests per algorithm.
- Statistical parameters: $\theta_{\text{fail}} = 0.50$, $\alpha = 0.05$, confidence $= 0.95$.

### Phase 3: Decoupled Post-Hoc Ground-Truth Evaluation
- Compare discovered subsets $\hat{C}_{\text{algo}}$ against ground truth $\{T_1\}$ strictly post-hoc.
- Compute Exact Match, Precision, and Recall.

### Phase 4: Complete Holdout Confirmation
- For every algorithm that produced a candidate subset $\hat{C}_{\text{algo}}$, execute $N=5$ independent holdout trials.
- Verify holdout failure rate $\hat{p}_{\text{holdout}}$ and compute Wilson 95% confidence intervals.

---

## 5. Reporting Integrity Standards

1. **No Phantom Holdouts:** If search was skipped due to unconfirmed trigger, holdout tables must render `BLOCKED_BY_QUALIFICATION` and failure rates must be `N/A`. They must never default to `0.0%` or `[]`.
2. **Explicit Element Identifiers:** All characterization and holdout tables must display exact element IDs (`[elem_1, ...]`) alongside cardinality.
3. **Versioned Artifact Isolation:** All outputs must be written to versioned paths (`phase_2b0_2_*`). Historical Phase 2B.0 and Phase 2B.0.1 files must remain unaltered.
4. **Zero Commercial Spend:** 100% of inference must execute locally via Ollama (`http://127.0.0.1:11434`).
