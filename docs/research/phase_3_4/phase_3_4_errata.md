# AgentXRay — Phase 3.4: Documentation Errata and Provenance Corrections

**Document Reference:** `docs/research/phase_3_4/phase_3_4_errata.md`  
**Date:** 2026-10-17  
**Version:** Phase 3.4 Post-Release Documentation Integrity Errata  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Authoritative Consistency Check:** [`docs/research/phase_3_4/phase_3_4_consistency_check.json`](phase_3_4_consistency_check.json)  

---

## 1. Executive Summary

This errata document records all editorial, semantic, and consistency corrections identified during the Phase 3.4 Post-Release Documentation Integrity Audit of Project AgentXRay. Each entry identifies the defect, the former wording, the corrected replacement wording, the affected files, and the underlying technical rationale.

All corrections preserve historical data immutability and require zero model re-inference ($0.0000 USD commercial spend).

---

## 2. Master Errata Register

| Erratum ID | Target Artifact | Defect Type | Former Wording / State | Corrected Replacement Wording | Technical Rationale |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **`ERR-3.4-01`** | `phase_3_3_claims_ledger.json`<br>`phase_3_3_claims_ledger.md` | Shell expansion typo | `strictly zsh.0000 USD` | `strictly $0.0000 USD` | Double-quoted shell string expansion in previous automation evaluated `$0` as `zsh`. Reconciled to unambiguous statement of zero commercial expenditure, distinguishing it from local hardware compute/electricity costs. |
| **`ERR-3.4-02`** | `phase_3_3_reproducibility_guide.md` | Check count mismatch | `verify the 10 core empirical and statistical claims` | `verify the 9 core empirical and statistical verification check suites` | `tests/test_phase_3_3_reproducibility.py` defines exactly 9 named test functions (`test_reproduce_01` through `test_reproduce_09`). Reconciled narrative to match implementation reality. |
| **`ERR-3.4-03`** | `phase_3_3_reproducibility_guide.md` | Overbroad absolute claim | `100% deterministically verifiable without model inference...` | `deterministically verifiable offline from preserved primary raw artifacts..., with explicit disclosure of unrecoverable execution details` | Qualified absolute phrasing to clearly distinguish recomputable aggregate observations and statistics from permanently unrecoverable execution details (4 aborted Turn 1 payloads, historical uncommitted runner diffs). |
| **`ERR-3.4-04`** | `phase_3_3_claims_ledger.json`<br>`phase_3_3_claims_ledger.md` | Status semantics ambiguity | Status `VERIFIED` on `CLAIM-REQ-ABORTED-PAYLOADS` without explicit distinction | Caveats explicitly state that `VERIFIED` applies to the limitation assertion, while underlying payloads remain unrecovered (`UNRESOLVED` offline). | Prevents ambiguity between verifying that data is missing vs. claiming to have verified the missing data itself. |
| **`ERR-3.4-05`** | `phase_3_4_documentation_integrity_audit.md` | Statistical power context | Uncontextualized citation of "power bounded by $\le 37.1\%$ for $d=1.25$" | Detailed documentation of mechanism-level ($n=6$, $2^6=64$ states) vs instance-level ($N=12$, 94.8%) power simulation basis. | Discloses exact Monte Carlo simulation parameters from `power_and_budget_analysis.py` and Protocol v1.4 §9.2 to prevent misinterpreting preregistered power as post-hoc empirical power. |

---

## 3. Detailed Errata Specifications

### 3.1 ERR-3.4-01: Commercial Financial Spend String Typo
- **Affected Artifacts:**
  - `docs/research/phase_3_3/phase_3_3_claims_ledger.json` (Line 412)
  - `docs/research/phase_3_3/phase_3_3_claims_ledger.md` (Table row 32, Section 3)
- **Root Cause:** A shell command executed during Phase 3.3 used double quotes (`"$0.0000 USD"`), causing the zsh interpreter to expand `$0` to the shell name (`zsh`), producing `"strictly zsh.0000 USD"`.
- **Correction Applied:**
  - `claim_text` updated to: `"Total commercial API spend across the entire benchmark execution was strictly $0.0000 USD (100% local inference on Ollama llama3.2:3b)."`
  - `caveats` updated to: `"All runs utilized isolated localhost Ollama daemon. Commercial API spend was strictly $0.0000 USD, which is distinguished from local hardware electricity and compute costs."`
  - Regenerated `phase_3_3_claims_ledger.md` via `tests/generate_claims_markdown.py`.
- **Impact Assessment:** Editorial defect only. Historical financial spend recorded in `benchmarks/results/phase_2b1_5_execution/results.json` has always been exactly `0.0`.

---

### 3.2 ERR-3.4-02: Verification Check Suite Count Reconciled (9 vs. 10)
- **Affected Artifact:**
  - `docs/research/phase_3_3/phase_3_3_reproducibility_guide.md` (Section 2.2, Line 49)
- **Root Cause:** Section 2.2 prose referred to "10 core empirical and statistical claims," while the standalone verification script [`tests/test_phase_3_3_reproducibility.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/tests/test_phase_3_3_reproducibility.py) defines exactly 9 test functions:
  1. `test_reproduce_01_cryptographic_hashes`
  2. `test_reproduce_02_request_ledger_and_trial_accounting`
  3. `test_reproduce_03_task_configuration_and_mechanism_counts`
  4. `test_reproduce_04_qualification_and_candidate_discovery`
  5. `test_reproduce_05_holdout_replication`
  6. `test_reproduce_06_ground_truth_recovery_and_mcnemar`
  7. `test_reproduce_07_mechanism_level_wilcoxon`
  8. `test_reproduce_08_null_control_outcomes`
  9. `test_reproduce_09_wilson_lower_bound_invariance`
- **Correction Applied:**
  - Updated Section 2.2 header and prose: `"To independently verify the 9 core empirical and statistical verification check suites from primary raw artifacts:"`
- **Impact Assessment:** Resolves count discrepancy without altering code or fabricating artificial tests.

---

### 3.3 ERR-3.4-03: Scope Qualification on Offline Reproducibility Language
- **Affected Artifact:**
  - `docs/research/phase_3_3/phase_3_3_reproducibility_guide.md` (Section 1, Lines 11–14)
- **Root Cause:** Blanket language claiming "100% deterministically verifiable" obscured documented execution limitations (namely, unpersisted Turn 1 payloads on aborted trials).
- **Correction Applied:**
  - Replaced Section 1 heading: `"## 1. Quick Start: Offline Reproducibility & Verification Scope"`
  - Replaced narrative: `"AgentXRay's recorded aggregate observations, statistical recalculations, qualification outcomes, and implementation tests are deterministically verifiable offline from preserved primary raw artifacts without model inference or external network connectivity. However, specific unpersisted execution details—most notably the Turn 1 conversational payloads of the 4 aborted trials and intermediate uncommitted historical runner diffs—are unrecoverable offline and permanently missing without model re-execution (see Section 3 for full scope delineation)."`
- **Impact Assessment:** Aligns reproducibility claims with scientific rigor and transparent gap disclosure.

---

### 3.4 ERR-3.4-04: Claims Ledger Status Semantics Delineation
- **Affected Artifacts:**
  - `docs/research/phase_3_3/phase_3_3_claims_ledger.json` (`CLAIM-REQ-ABORTED-PAYLOADS`)
  - `docs/research/phase_3_3/phase_3_3_claims_ledger.md` (`CLAIM-REQ-ABORTED-PAYLOADS`)
- **Root Cause:** A reader could confuse `consistency_status: "VERIFIED"` as an assertion that the missing payloads were recovered, rather than verifying the truth of the limitation statement.
- **Correction Applied:**
  - Enhanced caveats in `phase_3_3_claims_ledger.json`:
    `"The status VERIFIED applies to the limitation statement itself (offline inspection confirms Turn 1 payloads were not persisted). The payloads themselves are unrecoverable and remain missing/unresolved without model re-execution; no synthetic payloads have been fabricated."`
- **Impact Assessment:** Clarifies semantic boundary: the limitation is verified; the data remains unresolved offline.

---

### 3.5 ERR-3.4-05: Preregistered Statistical Power Documentation
- **Affected Artifact:**
  - `docs/research/phase_3_4/phase_3_4_documentation_integrity_audit.md` (Section 4)
- **Root Cause:** Prior reports cited "power bounded by $\le 37.1\%$ for large effects ($d=1.25$)" without detailing the underlying simulation code, sample size stratification, or permutation constraints.
- **Correction Applied:**
  - Formalized the exact mathematical and computational basis from `benchmarks/model_pilot/power_and_budget_analysis.py` (`simulate_wilcoxon_power` at $n=6$, $d=1.25$, 2,000 MC replications) in Section 4 of the integrity audit.
  - Contrasted $n=6$ mechanism-level power ($\le 37.1\%$) with $N=12$ instance-level power ($94.8\%$).
- **Impact Assessment:** Establishes unambiguous audit trail and prevents conflation of design-phase power with post-hoc empirical calculations.
