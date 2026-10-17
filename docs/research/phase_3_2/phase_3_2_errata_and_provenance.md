# AgentXRay — Phase 3.2: Errata, Artifact Provenance, and Forensic Audit Reconciliation

**Document Reference:** `docs/research/phase_3_2/phase_3_2_errata_and_provenance.md`  
**Date:** 2026-10-17  
**Version:** Phase 3.2 Errata and Provenance Record  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Prior Errata Record:** [`docs/research/phase_3_1/phase_3_1_errata_and_provenance.md`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/docs/research/phase_3_1/phase_3_1_errata_and_provenance.md) (Covering ERR-01 through ERR-07)

---

## 1. Purpose and Scope

This document extends the official errata registry of Project AgentXRay to incorporate the findings of the Phase 3.2 implementation conformance audit. 

All modifications, cryptographic hashes, and provenance trails are documented here to guarantee scientific integrity and complete reproducibility.

---

## 2. Erratum G (ERR-08): Omission of Candidate Wilson Lower Bound Guard in Historical Minimizer

### 2.1 What Protocol v1.4 Stated
Section 6.2 (Item 2) of [`docs/research/phase_2b1_exploratory_protocol_v1_4.md`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/docs/research/phase_2b1_exploratory_protocol_v1_4.md) preregistered the following candidate reproduction criterion:
> *"Candidate failure reproduction: The candidate configuration $C$ must exhibit an empirical failure rate $\hat{p}(C) \ge \theta_{\text{fail}} = 0.50$ AND a two-sided Wilson 95% confidence interval lower bound $L_{95}(C) > \theta_{\text{base}} = 0.20$."*

### 2.2 What the Implementation Did
In `src/agentxray/stochastic_minimizer.py` (line 342) and `src/agentxray/stochastic_baselines.py` (line 42):
```python
# Historical code:
if current_eval.trials == 0 or current_eval.failure_rate < self.failure_threshold:
    return (StochasticMinimalityStatus.INCONCLUSIVE, ...)
```
The implementation verified that observed failure rate was $\ge \theta_{\text{fail}} = 0.50$, but **omitted** checking whether `current_eval.confidence_interval[0] > 0.20`. 

### 2.3 Forensic Impact on Experimental Evidence
Forensic revalidation across the complete Phase 2B.1.5 raw dataset (`trials.jsonl`, `results.json`) revealed:
- In **100% of cases** where an algorithm completed minimization, candidate evaluation exhibited **5/5 failures** ($k_1=5, n_1=5$).
- The two-sided Wilson 95% confidence interval for $5/5$ is $[0.5655, 1.0000]$.
- Because $L_{95} = 0.5655 \gg 0.20$, every candidate subset evaluated during Phase 2B.1.5 strictly satisfied the omitted condition.
- **Scientific Impact:** Zero historical candidate classifications changed. No invalid classifications were generated.

### 2.4 Corrective Action
1. Modified `StochasticDeltaDebugMinimizer` and baseline classification functions to accept `baseline_threshold: float | None = None` and explicitly enforce:
   ```python
   if (
       current_eval.trials == 0
       or current_eval.failure_rate < self.failure_threshold
       or (self.baseline_threshold is not None and current_eval.confidence_interval[0] <= self.baseline_threshold)
   ):
       return (StochasticMinimalityStatus.INCONCLUSIVE, ...)
   ```
2. Implemented regression test `test_case_05_candidate_wilson_lower_bound_fails` in [`tests/test_phase_3_2_adversarial_minimality.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/tests/test_phase_3_2_adversarial_minimality.py) to prove that underpowered candidates (e.g. 1/2 failures, $L_{95} = 0.0945$) are rejected as `INCONCLUSIVE`.

---

## 3. Erratum H (ERR-09): Telemetry Message Formatting vs Operational Gate for Claim A Binomial Test

### 3.1 What the Evidence Appeared to Show
In `src/agentxray/stochastic_minimizer.py` lines 401–411, the minimizer computes:
```python
all_statistically_significant = all(p < self.alpha for p in p_values_comp) and len(p_values_comp) == len(current_candidate)
all_observed_drops = all(observed_drops) and len(observed_drops) == len(current_candidate)
all_thresh_supported = all(p < self.alpha for p in p_values_thresh) and len(p_values_thresh) == len(current_candidate)
```
In the return branch for `STATISTICALLY_SUPPORTED`, `all_thresh_supported` was interpolated into `msg`, but the branch condition depended solely on `all_statistically_significant`. Prior audit notes questioned whether `all_thresh_supported` was an un-checked gate.

### 3.2 What Protocol v1.4 Actually Preregistered
Protocol v1.4 §6.2 explicitly designated Claim B (comparative Fisher reduction under IUT) as the sole operational decision rule, while specifying that Claim A (one-sample binomial tests against 0.50) serves as diagnostic telemetry:
> *"Concurrently, exact one-sample binomial p-values $p_j^{\text{thresh}}$ against $\theta_{\text{fail}} = 0.50$ (Claim A) are computed and reported in telemetry for full diagnostic visibility."*

### 3.3 Forensic Resolution
The code behavior was completely aligned with the scientific intent of Protocol v1.4. The only defect was a documentation and telemetry formatting ambiguity, which gave the false impression that `all_thresh_supported` was an omitted boolean gate. Test `test_case_03_comparative_passes_threshold_binomial_fails` verifies that when Claim B passes but Claim A fails, the candidate is classified as `STATISTICALLY_SUPPORTED` with telemetry indicating `threshold binomial all p < 0.05: False`.

---

## 4. Master Errata Registry (ERR-01 through ERR-09)

| Identifier | Subject | Report Narrative Claim | Actual Evidence / Forensic Finding | Severity | Resolution Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **ERR-01** | Total Request Count | `trials.jsonl` accounts for all 2,176 requests | `trials.jsonl` logs 2,172 requests across 1,086 completed trials; 4 requests occurred on aborted Turn 1s | Low | Fully reconciled via error-trace forensics |
| **ERR-02** | Runner Checksum | Runner script SHA-256 is `03ff8262...` | Disk script SHA-256 is `69b03d52...` (report cited pre-patch draft hash) | Medium | Provenance updated to `69b03d52...` |
| **ERR-03** | Holdout Evaluation Count | Ambiguous listing of 5 tasks in Section 6 | 12 algorithm candidates across 5 tasks $\times 5$ trials = 60 trials (120 requests) | Low | Reconciled and documented |
| **ERR-04** | ITT Resource Table | Disqualified tasks labeled as "Expended Requests" | Values are imputed penalty caps under preregistered ITT protocol (0 search trials run) | Low | Documented as labeling ambiguity |
| **ERR-05** | Discovered Distractor on `inst_09` | Discovered element was `compliance_tag` | Raw results record 2-element interactions: `incompatible_unit + expense_category` (DD) and `incompatible_unit + archival_flag` (Greedy) | Low | Reconciled from `results.json` |
| **ERR-06** | False Discovery Element on Null DB | Isolated false-discovery element was `expense_category` | Discovered element was `department_code` on both `inst_15` and `inst_16` | Low | Reconciled from `results.json` |
| **ERR-07** | Minimality Multiplicity Procedure | Minimality decision used Benjamini-Hochberg FDR control | Pure Intersection-Union Test (Berger 1982); BH is not applied to IUT | High | Corrected in Phase 3.1 documents |
| **ERR-08** | Candidate Wilson Lower Bound Guard | Candidate failure reproduction verified $\hat{p} \ge 0.50$ and $L_{95} > 0.20$ | Code checked $\hat{p} \ge 0.50$ but omitted $L_{95} > 0.20$; 100% of historical candidates had 5/5 ($L_{95} = 0.5655 > 0.20$) | Medium | Corrected in source; verified by adversarial tests |
| **ERR-09** | Diagnostic Telemetry Role for Claim A | Code appeared to compute unused boolean `all_thresh_supported` | Protocol v1.4 §6.2 explicitly designated Claim A binomial test as diagnostic telemetry | Low | Clarified and verified by adversarial tests |

---

## 5. Artifact Provenance and Cryptographic Verification

All raw experimental data artifacts from Phase 2B.1.5 remain bit-for-bit identical to their frozen state:

| Artifact Path | Expected / Recorded SHA-256 Hash | Integrity Status |
| :--- | :--- | :---: |
| `docs/research/phase_2b1_exploratory_protocol_v1_4.md` | `db45f4032f2340a9edc3441bafcb3957895b616d1289c28e85a5cc4671574f62` | **VERIFIED UNCHANGED** |
| `benchmarks/configs/phase_2b1_exploratory_config_v1_4.json` | `b3e3b0b8450e325914ea8229faafce90a833eb038b02903bfa48062e8ee0cfb2` | **VERIFIED UNCHANGED** |
| `benchmarks/results/phase_2b1_5_execution/manifest.json` | `349c9b97a0ee76cded3770828029f897d101414d8917b9743930f08f23ea830f` | **VERIFIED UNCHANGED** |
| `benchmarks/results/phase_2b1_5_execution/results.json` | `5f9606d9dad39f6a1f3eea39fdfd140f56dffaec99f8516ec7ecfa66303f26b9` | **VERIFIED UNCHANGED** |
| `benchmarks/results/phase_2b1_5_execution/trials.jsonl` | `82905d2200b88c242f11499e6a927e75c38fe1e8bdf142b498dc28c9f80b86f0` | **VERIFIED UNCHANGED** |
| `benchmarks/results/phase_2b1_5_execution/summary.csv` | `6e521487357bf176fab79d3a20c3f7e61657909a8132e6d5dab6f084dae71902` | **VERIFIED UNCHANGED** |
| `benchmarks/model_pilot/exploratory_runner_v1_4.py` (Executing Runner) | `69b03d52b198bff5f23e86894041e810e36c75d052753620e3872d56342d7b27` | **VERIFIED UNCHANGED** |

---

## 6. Ledger Reconciliation (2,176 vs 2,172 Requests)

The total request ledger is completely reconciled:

$$\mathbf{2,176\text{ Total Expended Requests}} = \underbrace{1,086\text{ Completed Trials} \times 2\text{ Requests/Trial}}_{2,172\text{ Logged Requests in } \texttt{trials.jsonl}} + \underbrace{4\text{ Aborted Turn 1 Requests}}_{4\text{ Unserialized Requests}}$$

The 4 aborted requests occurred on:
1. `inst_02_syntax_currency_k8` (Random Search at cap 180)
2. `inst_04_syntax_comma_k8` (Random Search at cap 180)
3. `inst_09_schema_unit_k4` (Random Search at cap 120)
4. `inst_16_null_db_k8` (Random Search at cap 180)

In all four cases, Turn 1 was executed against Ollama, but `CostBudgetExceededError` was raised prior to Turn 2, preventing the trial record from completing. Zero model calls were unrecorded on the in-memory ledger.
