# AgentXRay — Phase 3.3: Master Claims and Provenance Ledger

**Document Reference:** `docs/research/phase_3_3/phase_3_3_claims_ledger.md`  
**Machine-Readable Companion:** [`phase_3_3_claims_ledger.json`](phase_3_3_claims_ledger.json)  
**Date:** 2026-10-17  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Total Registered Claims:** 32

---

## 1. Overview and Ledger Integrity Guarantee

This document is the authoritative, human-readable rendering of the machine-readable claims ledger `phase_3_3_claims_ledger.json`.
Every single major empirical finding, protocol parameter, statistical estimand, implementation correction, and cryptographic checksum
in Project AgentXRay is registered here with its independent reproduction method and verification status.

### Consistency Status Key:
- **`VERIFIED`**: Claim independently reproduced bit-for-bit from raw execution artifacts or source code.
- **`CORRECTED`**: Historical documentation erratum identified, reconciled with raw artifacts, and explicitly corrected.
- **`REPORTED_NOT_REPRODUCED`**: Claim stated in prose but cannot be derived from primary records.
- **`UNRESOLVED`**: Data is insufficient to verify claim offline without model re-execution.

---

## 2. Master Claims Table

| Claim ID | Category | Status | Claim Summary | Reproduced Value | Authoritative Source |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `CLAIM-REQ-LEDGER-TOTAL` | empirical_observation | **VERIFIED** | Total model requests recorded on the global budget ledger equals 2,176 requests. | `2176` | `benchmarks/results/phase_2b1_5_execution/manifest.json` |
| `CLAIM-REQ-TRIALS-LOGGED` | empirical_observation | **VERIFIED** | Exactly 1,086 completed trials are logged in trials.jsonl, each consuming 2 model requests, totaling 2,172 requests. | `2172` | `benchmarks/results/phase_2b1_5_execution/trials.jsonl` |
| `CLAIM-REQ-DISCREPANCY-ABORTED` | provenance | **VERIFIED** | The 4-request difference between ledger (2,176) and event log (2,172) consists of Turn 1 executions on 4 trials aborted by CostBudgetExceededError prior to Turn 2. | `4` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-REQ-ABORTED-PAYLOADS` | limitation | **VERIFIED** | The exact prompt string and tool-call payload of Turn 1 on the 4 aborted trials cannot be recovered from existing offline artifacts without model re-execution. | `UNRECOVERABLE_OFFLINE` | `benchmarks/results/phase_2b1_5_execution/trials.jsonl` |
| `CLAIM-TASK-COUNT-CONFIGS` | protocol | **VERIFIED** | The benchmark evaluates exactly 16 task configurations (12 active: inst_01 to inst_12; 4 null: inst_13 to inst_16). | `16` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-MECHANISM-COUNT` | protocol | **VERIFIED** | The benchmark evaluates exactly 8 independent mechanisms (6 active: mech_01 to mech_06; 2 null: mech_null_01, mech_null_02), each with 2 task configurations (K=4 and K=8). | `8` | `benchmarks/model_pilot/confirmatory_tasks.py` |
| `CLAIM-TASK-DISQUALIFIED-COUNT` | empirical_observation | **VERIFIED** | Exactly 7 active task configurations failed calibration and were classified as DISQUALIFIED_TASK (inst_05, 06, 07, 08, 10, 11, 12). | `7` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-TASK-QUALIFIED-ACTIVE-COUNT` | empirical_observation | **VERIFIED** | Exactly 5 active task configurations met calibration and were qualified for minimization (inst_01, 02, 03, 04, 09). | `5` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-SEARCH-CAPPED-COUNT` | empirical_observation | **VERIFIED** | Exactly 4 search runs were halted by per-task request limits, all belonging to Seeded Random Search (inst_02, inst_04, inst_09, inst_16). | `4` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-ALGO-ACTIVE-CANDIDATES-COUNT` | empirical_observation | **VERIFIED** | Across the 5 qualified active tasks, exactly 12 algorithm-task candidate subsets were isolated (DD: 5, Greedy: 5, Random: 2). | `12` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-ALGO-NULL-CANDIDATES-COUNT` | empirical_observation | **VERIFIED** | Across the 4 null tasks, exactly 5 algorithm-task candidate subsets were isolated (DD: 2, Greedy: 2, Random: 1, all on inst_15 and inst_16). | `5` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-NULL-FALSE-DISCOVERY-PROP` | empirical_observation | **CORRECTED** | The observed empirical false-discovery proportion on null controls is 50.0% across task configurations (2 of 4: inst_15, inst_16) and 50.0% across null mechanisms (1 of 2: mech_null_02_db). | `{'task_configuration_proportion': '2/4 (50...` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-STAT-FDR-NOT-INFERRED` | limitation | **VERIFIED** | Statistical population False Discovery Rate (FDR) cannot be claimed or inferred from four null task configurations without an inferential mixture model. | `CANNOT_BE_INFERRED` | `docs/research/phase_3_1/phase_3_1_scientific_integrity_audit.md` |
| `CLAIM-GT-RECOVERY-DD` | statistical_inference | **VERIFIED** | Stochastic Delta Debugging achieved exact ground truth recovery on 4 of 12 active tasks (33.3%: inst_01, 02, 03, 04). | `4/12 (33.3%)` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-GT-RECOVERY-GREEDY` | statistical_inference | **VERIFIED** | Stochastic Greedy Ablation achieved exact ground truth recovery on 4 of 12 active tasks (33.3%: inst_01, 02, 03, 04). | `4/12 (33.3%)` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-GT-RECOVERY-RANDOM` | statistical_inference | **VERIFIED** | Seeded Random Search achieved exact ground truth recovery on 2 of 12 active tasks (16.7%: inst_01, 03). | `2/12 (16.7%)` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-MCNEMAR-PARITY` | statistical_inference | **VERIFIED** | McNemar exact symmetry test on DD vs Greedy recovery yields p = 1.0 with zero discordant pairs (b=0, c=0), representing absence of observed asymmetry rather than proof of equivalence. | `1.0` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-INST09-INTERACTION` | empirical_observation | **VERIFIED** | On inst_09, DD and Greedy discovered 2-element interacting subsets containing the planted trigger plus a distractor; both subsets met operational 1-minimality (p=0.0040) and holdout reproduction (5/5) but failed ground truth match. | `{'dd_subset': ['incompatible_unit', 'expen...` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-ITT-BURDEN-NORM-DIFFS` | statistical_inference | **VERIFIED** | Normalized intention-to-treat request burden differences across 6 active mechanisms are: mech_01: +0.2222, mech_02: +0.2222, mech_03: 0.0, mech_04: 0.0, mech_05: -0.0417, mech_06: 0.0. | `{'mech_01': 0.2222, 'mech_02': 0.2222, 'me...` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-WILCOXON-MECHANISM` | statistical_inference | **VERIFIED** | Wilcoxon signed-rank exact permutation test on m=3 non-zero mechanisms yields p = 0.50, failing to reject H0 (alpha=0.05). | `0.5` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-DESCRIPTIVE-REQUEST-REDUCTION` | limitation | **VERIFIED** | The 43.8% request reduction (DD 180 vs Greedy 320 requests) applies exclusively to the four qualified syntax configurations (inst_01 to inst_04) and is strictly descriptive, not confirmatory evidence of superiority. | `43.8% (180 vs 320 requests on 4 syntax con...` | `docs/research/phase_3_1/phase_3_1_agentxray_research_report.md` |
| `CLAIM-HOLDOUT-REPRODUCTION` | empirical_observation | **VERIFIED** | All 12 candidate subsets evaluated in holdout validation achieved 100% failure reproduction (5/5 failures, Wilson 95% CI [0.5655, 1.0000]) across 60 total holdout trials. | `{'total_holdout_trials': 60, 'candidates_r...` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-ERR-WILSON-GUARD-OMISSION` | implementation | **VERIFIED** | Historical minimizer and baseline code checked p_hat >= 0.50 but omitted checking the candidate Wilson 95% CI lower bound L95 > 0.20 required by Protocol v1.4 §6.2 Item 2 (ERR-08). | `DEFECT_CONFIRMED_AND_REMEDIATED` | `src/agentxray/stochastic_minimizer.py` |
| `CLAIM-ERR-WILSON-HISTORICAL-INVARIANCE` | empirical_observation | **VERIFIED** | In 100% of historical candidate evaluations across Phase 2B.1.5, candidate failure reproduction was 5/5 (L95 = 0.5655 > 0.20), resulting in zero historical classification changes. | `ZERO_CLASSIFICATIONS_CHANGED` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-ERR-RUNNER-HASH` | provenance | **CORRECTED** | The test runner script that actually executed Phase 2B.1.5 hashes to 69b03d52...; the execution report printed pre-patch draft hash 03ff8262... (ERR-02). | `69b03d52b198bff5f23e86894041e810e36c75d052...` | `benchmarks/model_pilot/exploratory_runner_v1_4.py` |
| `CLAIM-HASH-PROTOCOL` | provenance | **VERIFIED** | Frozen Protocol v1.4 SHA-256 hash is db45f4032f2340a9edc3441bafcb3957895b616d1289c28e85a5cc4671574f62. | `db45f4032f2340a9edc3441bafcb3957895b616d12...` | `docs/research/phase_2b1_exploratory_protocol_v1_4.md` |
| `CLAIM-HASH-CONFIG` | provenance | **VERIFIED** | Frozen Configuration v1.4 SHA-256 hash is b3e3b0b8450e325914ea8229faafce90a833eb038b02903bfa48062e8ee0cfb2. | `b3e3b0b8450e325914ea8229faafce90a833eb038b...` | `benchmarks/configs/phase_2b1_exploratory_config_v1_4.json` |
| `CLAIM-HASH-RAW-MANIFEST` | provenance | **VERIFIED** | Raw manifest.json SHA-256 hash is 349c9b97a0ee76cded3770828029f897d101414d8917b9743930f08f23ea830f. | `349c9b97a0ee76cded3770828029f897d101414d89...` | `benchmarks/results/phase_2b1_5_execution/manifest.json` |
| `CLAIM-HASH-RAW-RESULTS` | provenance | **VERIFIED** | Raw results.json SHA-256 hash is 5f9606d9dad39f6a1f3eea39fdfd140f56dffaec99f8516ec7ecfa66303f26b9. | `5f9606d9dad39f6a1f3eea39fdfd140f56dffaec99...` | `benchmarks/results/phase_2b1_5_execution/results.json` |
| `CLAIM-HASH-RAW-TRIALS` | provenance | **VERIFIED** | Raw trials.jsonl SHA-256 hash is 82905d2200b88c242f11499e6a927e75c38fe1e8bdf142b498dc28c9f80b86f0. | `82905d2200b88c242f11499e6a927e75c38fe1e8bd...` | `benchmarks/results/phase_2b1_5_execution/trials.jsonl` |
| `CLAIM-HASH-RAW-SUMMARY` | provenance | **VERIFIED** | Raw summary.csv SHA-256 hash is 6e521487357bf176fab79d3a20c3f7e61657909a8132e6d5dab6f084dae71902. | `6e521487357bf176fab79d3a20c3f7e61657909a81...` | `benchmarks/results/phase_2b1_5_execution/summary.csv` |
| `CLAIM-FINANCIAL-SPEND` | empirical_observation | **VERIFIED** | Total commercial API spend across the entire benchmark execution was strictly $0.0000 USD (100% local inference on Ollama llama3.2:3b). | `0.0` | `benchmarks/results/phase_2b1_5_execution/results.json` |

---

## 3. Detailed Claim Specifications

### CLAIM-REQ-LEDGER-TOTAL
- **Statement:** Total model requests recorded on the global budget ledger equals 2,176 requests.
- **Category:** `empirical_observation` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/manifest.json`
- **Source Fields / Record Identifiers:** `global_requests_used, statistics.global_budget_accounting.total_model_requests_expended`
- **Reproduction Method:** Parsed manifest.json and results.json global request counter fields.
- **Original Reported Value:** `2176`
- **Independently Reproduced Value:** `2176`
- **Caveats & Notes:** Includes 4 requests on aborted trials that were not serialized to trials.jsonl.

### CLAIM-REQ-TRIALS-LOGGED
- **Statement:** Exactly 1,086 completed trials are logged in trials.jsonl, each consuming 2 model requests, totaling 2,172 requests.
- **Category:** `empirical_observation` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/trials.jsonl`
- **Source Fields / Record Identifiers:** `line count, model_requests_used per line`
- **Reproduction Method:** Summed model_requests_used across all 1,086 lines in trials.jsonl.
- **Original Reported Value:** `2172`
- **Independently Reproduced Value:** `2172`
- **Caveats & Notes:** Original execution report claimed all 2,176 requests were in trials.jsonl; ERR-01 corrected this to 2,172.

### CLAIM-REQ-DISCREPANCY-ABORTED
- **Statement:** The 4-request difference between ledger (2,176) and event log (2,172) consists of Turn 1 executions on 4 trials aborted by CostBudgetExceededError prior to Turn 2.
- **Category:** `provenance` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `search.inst_{02,04,09,16}.stochastic_seeded_random.budget_exhausted`
- **Reproduction Method:** Trace analysis of budget cap hits on Seeded Random Search runs reaching ceilings 180 and 120.
- **Original Reported Value:** `4`
- **Independently Reproduced Value:** `4`
- **Caveats & Notes:** Turn 1 payloads for these 4 trials were not persisted to disk before exception handling terminated the trial.

### CLAIM-REQ-ABORTED-PAYLOADS
- **Statement:** The exact prompt string and tool-call payload of Turn 1 on the 4 aborted trials cannot be recovered from existing offline artifacts without model re-execution.
- **Category:** `limitation` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/trials.jsonl`
- **Source Fields / Record Identifiers:** `trials.jsonl (absence of partial trial records)`
- **Reproduction Method:** Verified absence of exception-terminated trial serialization in event log.
- **Original Reported Value:** `UNRECOVERABLE_OFFLINE`
- **Independently Reproduced Value:** `UNRECOVERABLE_OFFLINE`
- **Caveats & Notes:** The status VERIFIED confirms the truth of the limitation statement (offline inspection confirms Turn 1 payloads were not persisted). The payloads themselves are unrecoverable and remain missing/unresolved without model re-execution; no synthetic payloads have been fabricated.

### CLAIM-TASK-COUNT-CONFIGS
- **Statement:** The benchmark evaluates exactly 16 task configurations (12 active: inst_01 to inst_12; 4 null: inst_13 to inst_16).
- **Category:** `protocol` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `calibration keys, search keys`
- **Reproduction Method:** Enumerated keys in results.json calibration mapping.
- **Original Reported Value:** `16`
- **Independently Reproduced Value:** `16`
- **Caveats & Notes:** Task configurations must be distinguished from independent mechanisms.

### CLAIM-MECHANISM-COUNT
- **Statement:** The benchmark evaluates exactly 8 independent mechanisms (6 active: mech_01 to mech_06; 2 null: mech_null_01, mech_null_02), each with 2 task configurations (K=4 and K=8).
- **Category:** `protocol` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/model_pilot/confirmatory_tasks.py`
- **Source Fields / Record Identifiers:** `ACTIVE_MECHANISMS, NULL_MECHANISMS, mechanism_id fields in results.json`
- **Reproduction Method:** Grouped task configurations by mechanism_id.
- **Original Reported Value:** `8`
- **Independently Reproduced Value:** `8`
- **Caveats & Notes:** Independent mechanisms serve as the unit of independent burden inference.

### CLAIM-TASK-DISQUALIFIED-COUNT
- **Statement:** Exactly 7 active task configurations failed calibration and were classified as DISQUALIFIED_TASK (inst_05, 06, 07, 08, 10, 11, 12).
- **Category:** `empirical_observation` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `search.<task>.qualification_status`
- **Reproduction Method:** Filtered search records for qualification_status == DISQUALIFIED_TASK.
- **Original Reported Value:** `7`
- **Independently Reproduced Value:** `7`
- **Caveats & Notes:** Disqualified tasks received zero search trials per Protocol v1.4 §5.4.

### CLAIM-TASK-QUALIFIED-ACTIVE-COUNT
- **Statement:** Exactly 5 active task configurations met calibration and were qualified for minimization (inst_01, 02, 03, 04, 09).
- **Category:** `empirical_observation` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `search.<task>.qualification_status`
- **Reproduction Method:** Filtered active search records for qualification_status == QUALIFIED.
- **Original Reported Value:** `5`
- **Independently Reproduced Value:** `5`
- **Caveats & Notes:** Spans 3 active mechanisms (mech_01, mech_02, mech_05).

### CLAIM-SEARCH-CAPPED-COUNT
- **Statement:** Exactly 4 search runs were halted by per-task request limits, all belonging to Seeded Random Search (inst_02, inst_04, inst_09, inst_16).
- **Category:** `empirical_observation` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `search.<task>.algorithms.stochastic_seeded_random.budget_exhausted`
- **Reproduction Method:** Counted algorithm records with budget_exhausted == True.
- **Original Reported Value:** `4`
- **Independently Reproduced Value:** `4`
- **Caveats & Notes:** Zero DD or Greedy ablation runs exceeded budget limits.

### CLAIM-ALGO-ACTIVE-CANDIDATES-COUNT
- **Statement:** Across the 5 qualified active tasks, exactly 12 algorithm-task candidate subsets were isolated (DD: 5, Greedy: 5, Random: 2).
- **Category:** `empirical_observation` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `search.<task>.algorithms.<algo>.minimality_status`
- **Reproduction Method:** Counted active-task algorithm records with minimality_status == statistically_supported.
- **Original Reported Value:** `12`
- **Independently Reproduced Value:** `12`
- **Caveats & Notes:** Random search failed to isolate candidates on inst_02, inst_04, and inst_09 due to budget exhaustion.

### CLAIM-ALGO-NULL-CANDIDATES-COUNT
- **Statement:** Across the 4 null tasks, exactly 5 algorithm-task candidate subsets were isolated (DD: 2, Greedy: 2, Random: 1, all on inst_15 and inst_16).
- **Category:** `empirical_observation` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `search.inst_{15,16}.algorithms.<algo>.minimality_status`
- **Reproduction Method:** Counted null-task algorithm records with minimality_status == statistically_supported.
- **Original Reported Value:** `5`
- **Independently Reproduced Value:** `5`
- **Caveats & Notes:** All 5 isolated the identical element: department_code.

### CLAIM-NULL-FALSE-DISCOVERY-PROP
- **Statement:** The observed empirical false-discovery proportion on null controls is 50.0% across task configurations (2 of 4: inst_15, inst_16) and 50.0% across null mechanisms (1 of 2: mech_null_02_db).
- **Category:** `empirical_observation` | **Status:** `CORRECTED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `statistics.null_control_summary`
- **Reproduction Method:** Computed proportions with explicit task-configuration and mechanism denominators.
- **Original Reported Value:** `2 out of 4 null mechanisms (corrected from inaccurate wording)`
- **Independently Reproduced Value:** `{'task_configuration_proportion': '2/4 (50.0%)', 'mechanism_proportion': '1/2 (50.0%)'}`
- **Caveats & Notes:** Corrected former narrative phrasing that cited 2 of 4 null mechanisms; the denominator of 4 represents task configurations.

### CLAIM-STAT-FDR-NOT-INFERRED
- **Statement:** Statistical population False Discovery Rate (FDR) cannot be claimed or inferred from four null task configurations without an inferential mixture model.
- **Category:** `limitation` | **Status:** `VERIFIED`
- **Source Artifact:** `docs/research/phase_3_1/phase_3_1_scientific_integrity_audit.md`
- **Source Fields / Record Identifiers:** `Section 3.2`
- **Reproduction Method:** Methodological audit of sample size and inferential boundary.
- **Original Reported Value:** `CANNOT_BE_INFERRED`
- **Independently Reproduced Value:** `CANNOT_BE_INFERRED`
- **Caveats & Notes:** Proportions on null controls must be reported strictly as descriptive empirical observations.

### CLAIM-GT-RECOVERY-DD
- **Statement:** Stochastic Delta Debugging achieved exact ground truth recovery on 4 of 12 active tasks (33.3%: inst_01, 02, 03, 04).
- **Category:** `statistical_inference` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `statistics.recovery_summary.dd_recovered_count`
- **Reproduction Method:** Verified matches_ground_truth == True on active task records for DD.
- **Original Reported Value:** `4/12 (33.3%)`
- **Independently Reproduced Value:** `4/12 (33.3%)`
- **Caveats & Notes:** Denominator is all 12 assigned active tasks under Intention-to-Treat.

### CLAIM-GT-RECOVERY-GREEDY
- **Statement:** Stochastic Greedy Ablation achieved exact ground truth recovery on 4 of 12 active tasks (33.3%: inst_01, 02, 03, 04).
- **Category:** `statistical_inference` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `statistics.recovery_summary.greedy_recovered_count`
- **Reproduction Method:** Verified matches_ground_truth == True on active task records for Greedy.
- **Original Reported Value:** `4/12 (33.3%)`
- **Independently Reproduced Value:** `4/12 (33.3%)`
- **Caveats & Notes:** Identical recovery outcomes to DD on all 12 active tasks.

### CLAIM-GT-RECOVERY-RANDOM
- **Statement:** Seeded Random Search achieved exact ground truth recovery on 2 of 12 active tasks (16.7%: inst_01, 03).
- **Category:** `statistical_inference` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `statistics.recovery_summary.random_recovered_count`
- **Reproduction Method:** Verified matches_ground_truth == True on active task records for Random.
- **Original Reported Value:** `2/12 (16.7%)`
- **Independently Reproduced Value:** `2/12 (16.7%)`
- **Caveats & Notes:** Exhausted budget on inst_02, inst_04, and inst_09.

### CLAIM-MCNEMAR-PARITY
- **Statement:** McNemar exact symmetry test on DD vs Greedy recovery yields p = 1.0 with zero discordant pairs (b=0, c=0), representing absence of observed asymmetry rather than proof of equivalence.
- **Category:** `statistical_inference` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `statistics.recovery_summary.mcnemar_exact_p_value, statistics.recovery_summary.contingency_table`
- **Reproduction Method:** Computed binomial test on discordant pairs b=0, c=0.
- **Original Reported Value:** `1.0`
- **Independently Reproduced Value:** `1.0`
- **Caveats & Notes:** Power to reject equivalence or asymmetry at N=12 is low.

### CLAIM-INST09-INTERACTION
- **Statement:** On inst_09, DD and Greedy discovered 2-element interacting subsets containing the planted trigger plus a distractor; both subsets met operational 1-minimality (p=0.0040) and holdout reproduction (5/5) but failed ground truth match.
- **Category:** `empirical_observation` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `search.inst_09_schema_unit_k4.algorithms`
- **Reproduction Method:** Verified candidate elements, removal Fisher p-values (0.0040), and matches_ground_truth (False).
- **Original Reported Value:** `2-element subsets (compliance_tag narrative in earlier draft corrected by ERR-05)`
- **Independently Reproduced Value:** `{'dd_subset': ['incompatible_unit', 'expense_category'], 'greedy_subset': ['incompatible_unit', 'archival_flag'], 'fisher_p': 0.004, 'holdout': '5/5'}`
- **Caveats & Notes:** Demonstrates operational 1-minimality diverging from synthetic ground truth due to neural prompt interactions.

### CLAIM-ITT-BURDEN-NORM-DIFFS
- **Statement:** Normalized intention-to-treat request burden differences across 6 active mechanisms are: mech_01: +0.2222, mech_02: +0.2222, mech_03: 0.0, mech_04: 0.0, mech_05: -0.0417, mech_06: 0.0.
- **Category:** `statistical_inference` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `statistics.mechanism_level_burden.differences_normalized`
- **Reproduction Method:** Recomputed mean normalized burden per mechanism under ITT right-censoring rules.
- **Original Reported Value:** `{'mech_01': 0.2222, 'mech_02': 0.2222, 'mech_03': 0.0, 'mech_04': 0.0, 'mech_05': -0.0417, 'mech_06': 0.0}`
- **Independently Reproduced Value:** `{'mech_01': 0.2222, 'mech_02': 0.2222, 'mech_03': 0.0, 'mech_04': 0.0, 'mech_05': -0.0417, 'mech_06': 0.0}`
- **Caveats & Notes:** Imputed penalty caps for disqualified tasks produce zero differences.

### CLAIM-WILCOXON-MECHANISM
- **Statement:** Wilcoxon signed-rank exact permutation test on m=3 non-zero mechanisms yields p = 0.50, failing to reject H0 (alpha=0.05).
- **Category:** `statistical_inference` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `statistics.mechanism_level_burden.wilcoxon_exact_permutation_p_value`
- **Reproduction Method:** Computed exact subset-sum DP permutation p-value on ranks [1.0, 2.5, 2.5] and signs [False, True, True].
- **Original Reported Value:** `0.5`
- **Independently Reproduced Value:** `0.5`
- **Caveats & Notes:** With m=3 non-zero mechanisms, minimum achievable two-sided p-value is 2/8 = 0.25 > 0.05, making rejection impossible.

### CLAIM-DESCRIPTIVE-REQUEST-REDUCTION
- **Statement:** The 43.8% request reduction (DD 180 vs Greedy 320 requests) applies exclusively to the four qualified syntax configurations (inst_01 to inst_04) and is strictly descriptive, not confirmatory evidence of superiority.
- **Category:** `limitation` | **Status:** `VERIFIED`
- **Source Artifact:** `docs/research/phase_3_1/phase_3_1_agentxray_research_report.md`
- **Source Fields / Record Identifiers:** `Section 8.2`
- **Reproduction Method:** Summed physical model requests across inst_01 to inst_04 for DD (180) and Greedy (320). (320 - 180) / 320 = 43.75% ~ 43.8%.
- **Original Reported Value:** `43.8%`
- **Independently Reproduced Value:** `43.8% (180 vs 320 requests on 4 syntax configs)`
- **Caveats & Notes:** Must not be generalized to the entire benchmark or other model architectures.

### CLAIM-HOLDOUT-REPRODUCTION
- **Statement:** All 12 candidate subsets evaluated in holdout validation achieved 100% failure reproduction (5/5 failures, Wilson 95% CI [0.5655, 1.0000]) across 60 total holdout trials.
- **Category:** `empirical_observation` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `holdout mapping across 5 tasks`
- **Reproduction Method:** Verified holdout trial counts, failures, and Wilson intervals for all 12 candidate entries.
- **Original Reported Value:** `{'total_holdout_trials': 60, 'candidates_reproduced': '12/12 (100%)', 'holdout_rate': 1.0}`
- **Independently Reproduced Value:** `{'total_holdout_trials': 60, 'candidates_reproduced': '12/12 (100%)', 'holdout_rate': 1.0}`
- **Caveats & Notes:** Holdout evaluates stability on fresh seeds; does not imply match with synthetic ground truth.

### CLAIM-ERR-WILSON-GUARD-OMISSION
- **Statement:** Historical minimizer and baseline code checked p_hat >= 0.50 but omitted checking the candidate Wilson 95% CI lower bound L95 > 0.20 required by Protocol v1.4 §6.2 Item 2 (ERR-08).
- **Category:** `implementation` | **Status:** `VERIFIED`
- **Source Artifact:** `src/agentxray/stochastic_minimizer.py`
- **Source Fields / Record Identifiers:** `lines 342-348 (historical), lines 349-365 (corrected)`
- **Reproduction Method:** Source code diff inspection and adversarial boundary testing.
- **Original Reported Value:** `DEFECT_CONFIRMED_AND_REMEDIATED`
- **Independently Reproduced Value:** `DEFECT_CONFIRMED_AND_REMEDIATED`
- **Caveats & Notes:** Corrected in source; verified by test_case_05_candidate_wilson_lower_bound_fails.

### CLAIM-ERR-WILSON-HISTORICAL-INVARIANCE
- **Statement:** In 100% of historical candidate evaluations across Phase 2B.1.5, candidate failure reproduction was 5/5 (L95 = 0.5655 > 0.20), resulting in zero historical classification changes.
- **Category:** `empirical_observation` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `candidate_evaluation.confidence_interval across all 17 supported runs`
- **Reproduction Method:** Evaluated Wilson score interval on 5/5 trials: L95 = 0.5655 > 0.20.
- **Original Reported Value:** `ZERO_CLASSIFICATIONS_CHANGED`
- **Independently Reproduced Value:** `ZERO_CLASSIFICATIONS_CHANGED`
- **Caveats & Notes:** All 17 supported candidate subsets remain valid SUPPORTED_1_MINIMAL under full Protocol v1.4 rules.

### CLAIM-ERR-RUNNER-HASH
- **Statement:** The test runner script that actually executed Phase 2B.1.5 hashes to 69b03d52...; the execution report printed pre-patch draft hash 03ff8262... (ERR-02).
- **Category:** `provenance` | **Status:** `CORRECTED`
- **Source Artifact:** `benchmarks/model_pilot/exploratory_runner_v1_4.py`
- **Source Fields / Record Identifiers:** `file SHA-256 vs report narrative`
- **Reproduction Method:** Computed SHA-256 on exploratory_runner_v1_4.py on disk.
- **Original Reported Value:** `03ff8262b9f68800b3e6488a09ee2ba5eb7ae2f588aa4da541249339a3f2db7b`
- **Independently Reproduced Value:** `69b03d52b198bff5f23e86894041e810e36c75d052753620e3872d56342d7b27`
- **Caveats & Notes:** Authoritative provenance citations must reference 69b03d52...

### CLAIM-HASH-PROTOCOL
- **Statement:** Frozen Protocol v1.4 SHA-256 hash is db45f4032f2340a9edc3441bafcb3957895b616d1289c28e85a5cc4671574f62.
- **Category:** `provenance` | **Status:** `VERIFIED`
- **Source Artifact:** `docs/research/phase_2b1_exploratory_protocol_v1_4.md`
- **Source Fields / Record Identifiers:** `file SHA-256`
- **Reproduction Method:** Direct SHA-256 computation on disk.
- **Original Reported Value:** `db45f4032f2340a9edc3441bafcb3957895b616d1289c28e85a5cc4671574f62`
- **Independently Reproduced Value:** `db45f4032f2340a9edc3441bafcb3957895b616d1289c28e85a5cc4671574f62`
- **Caveats & Notes:** Bit-for-bit unchanged.

### CLAIM-HASH-CONFIG
- **Statement:** Frozen Configuration v1.4 SHA-256 hash is b3e3b0b8450e325914ea8229faafce90a833eb038b02903bfa48062e8ee0cfb2.
- **Category:** `provenance` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/configs/phase_2b1_exploratory_config_v1_4.json`
- **Source Fields / Record Identifiers:** `file SHA-256`
- **Reproduction Method:** Direct SHA-256 computation on disk.
- **Original Reported Value:** `b3e3b0b8450e325914ea8229faafce90a833eb038b02903bfa48062e8ee0cfb2`
- **Independently Reproduced Value:** `b3e3b0b8450e325914ea8229faafce90a833eb038b02903bfa48062e8ee0cfb2`
- **Caveats & Notes:** Bit-for-bit unchanged.

### CLAIM-HASH-RAW-MANIFEST
- **Statement:** Raw manifest.json SHA-256 hash is 349c9b97a0ee76cded3770828029f897d101414d8917b9743930f08f23ea830f.
- **Category:** `provenance` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/manifest.json`
- **Source Fields / Record Identifiers:** `file SHA-256`
- **Reproduction Method:** Direct SHA-256 computation on disk.
- **Original Reported Value:** `349c9b97a0ee76cded3770828029f897d101414d8917b9743930f08f23ea830f`
- **Independently Reproduced Value:** `349c9b97a0ee76cded3770828029f897d101414d8917b9743930f08f23ea830f`
- **Caveats & Notes:** Bit-for-bit unchanged.

### CLAIM-HASH-RAW-RESULTS
- **Statement:** Raw results.json SHA-256 hash is 5f9606d9dad39f6a1f3eea39fdfd140f56dffaec99f8516ec7ecfa66303f26b9.
- **Category:** `provenance` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `file SHA-256`
- **Reproduction Method:** Direct SHA-256 computation on disk.
- **Original Reported Value:** `5f9606d9dad39f6a1f3eea39fdfd140f56dffaec99f8516ec7ecfa66303f26b9`
- **Independently Reproduced Value:** `5f9606d9dad39f6a1f3eea39fdfd140f56dffaec99f8516ec7ecfa66303f26b9`
- **Caveats & Notes:** Bit-for-bit unchanged.

### CLAIM-HASH-RAW-TRIALS
- **Statement:** Raw trials.jsonl SHA-256 hash is 82905d2200b88c242f11499e6a927e75c38fe1e8bdf142b498dc28c9f80b86f0.
- **Category:** `provenance` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/trials.jsonl`
- **Source Fields / Record Identifiers:** `file SHA-256`
- **Reproduction Method:** Direct SHA-256 computation on disk.
- **Original Reported Value:** `82905d2200b88c242f11499e6a927e75c38fe1e8bdf142b498dc28c9f80b86f0`
- **Independently Reproduced Value:** `82905d2200b88c242f11499e6a927e75c38fe1e8bdf142b498dc28c9f80b86f0`
- **Caveats & Notes:** Bit-for-bit unchanged.

### CLAIM-HASH-RAW-SUMMARY
- **Statement:** Raw summary.csv SHA-256 hash is 6e521487357bf176fab79d3a20c3f7e61657909a8132e6d5dab6f084dae71902.
- **Category:** `provenance` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/summary.csv`
- **Source Fields / Record Identifiers:** `file SHA-256`
- **Reproduction Method:** Direct SHA-256 computation on disk.
- **Original Reported Value:** `6e521487357bf176fab79d3a20c3f7e61657909a8132e6d5dab6f084dae71902`
- **Independently Reproduced Value:** `6e521487357bf176fab79d3a20c3f7e61657909a8132e6d5dab6f084dae71902`
- **Caveats & Notes:** Bit-for-bit unchanged.

### CLAIM-FINANCIAL-SPEND
- **Statement:** Total commercial API spend across the entire benchmark execution was strictly $0.0000 USD (100% local inference on Ollama llama3.2:3b).
- **Category:** `empirical_observation` | **Status:** `VERIFIED`
- **Source Artifact:** `benchmarks/results/phase_2b1_5_execution/results.json`
- **Source Fields / Record Identifiers:** `statistics.global_budget_accounting.commercial_spend_usd`
- **Reproduction Method:** Parsed commercial_spend_usd from results.json and verified zero external HTTP calls in trials.
- **Original Reported Value:** `0.0`
- **Independently Reproduced Value:** `0.0`
- **Caveats & Notes:** All runs utilized isolated localhost Ollama daemon. Commercial API spend was strictly $0.0000 USD, which is distinguished from local hardware electricity and compute costs.
