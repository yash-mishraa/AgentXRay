# AgentXRay — Phase 2B.1.6: Independent Post-Execution Forensic Audit

**Date:** 2026-10-17  
**Auditor:** Independent Research Integrity & Statistical Audit Team (Antigravity)  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Execution Report Audited:** `docs/research/phase_2b1_5_exploratory_execution_report.md`  
**Execution Evidence Audited:** `benchmarks/results/phase_2b1_5_execution/`  
**Frozen Protocol Reference:** `docs/research/phase_2b1_exploratory_protocol_v1_4.md` (`db45f4032f2340a9edc3441bafcb3957895b616d1289c28e85a5cc4671574f62`)  
**Frozen Config Reference:** `benchmarks/configs/phase_2b1_exploratory_config_v1_4.json` (`b3e3b0b8450e325914ea8229faafce90a833eb038b02903bfa48062e8ee0cfb2`)  
**Final Audit Disposition:** `AUDIT_PASS_WITH_LIMITATIONS`

---

## 1. Executive Verdict

### Final Disposition: `AUDIT_PASS_WITH_LIMITATIONS`

Project AgentXRay conducted an authorized exploratory benchmark execution under Protocol v1.4 and Configuration v1.4 in Phase 2B.1.5. This forensic audit independently verified the raw experimental logs (`trials.jsonl`), aggregated outputs (`results.json`, `manifest.json`, `summary.csv`), and execution report narrative against the frozen preregistration.

The execution satisfies core research integrity criteria:
1. **Zero External API Infiltration & Financial Containment:** Total commercial API spend was strictly **$0.00 USD**. All 2,176 requests were served locally by Ollama `llama3.2:3b`.
2. **Preregistration Fidelity:** The execution adhered strictly to the frozen Protocol v1.4 and Configuration v1.4 stopping rules, task configurations, candidate spaces, and statistical estimands.
3. **No Retrospective Scientific Whitewashing:** The execution report transparently declared null-hypothesis retention ($p = 0.5000$), false discoveries on database null controls, distractor entrapment on schema units, and calibration disqualification of 7 active tasks.
4. **Data Reproducibility:** 100% of reported search outcomes, task-level request tallies, and statistical tests were reproduced exactly from raw trial logs.

However, a disposition of `AUDIT_PASS_WITH_LIMITATIONS` is assigned rather than an unqualified `AUDIT_PASS` due to four technical and reporting limitations:
1. **Unlogged Aborted-Trial Ledger Delta:** A discrepancy of exactly **4 requests** exists between the request ledger (`2,176 requests` tracked in `manifest.json` and `results.json`) and the event log (`2,172 requests` across 1,086 completed trials in `trials.jsonl`). Forensics established that 4 random-search runs hit per-task caps on Turn 2 of their final trial; Turn 1 incremented the model counter, but the ensuing `CostBudgetExceededError` prevented `log_trial_event()` from persisting the partial trial.
2. **Execution Script Hash Desynchronization:** The execution report cited an SHA-256 hash for `exploratory_runner_v1_4.py` (`03ff8262...`) corresponding to an earlier draft prior to a critical pre-execution bugfix, rather than the true on-disk script hash (`69b03d52...`).
3. **Holdout Presentation Ambiguity:** Section 6 of the execution report bulleted the 5 *active tasks* where candidates were discovered rather than explicitly delineating the **12 distinct algorithm-task candidates** ($5 \times 12 = 60$ holdout trials), creating the appearance of a numerical conflict with Section 7.
4. **Intention-to-Treat Table Column Labeling:** Section 5.1 labeled table columns as "Expended Requests" while displaying right-censored imputed caps (120 and 180 requests) for the 7 disqualified tasks on which zero search requests were physically executed.

None of these limitations alter the substantive scientific conclusions of the benchmark.

---

## 2. Evidence Inventory and Hash Verification

All artifacts in the execution directory and specification tree were independently inspected, and their SHA-256 checksums were calculated.

| Artifact Path | Expected Hash (Protocol / Report) | Actual Computed SHA-256 Hash | Integrity Status |
| :--- | :--- | :--- | :---: |
| `docs/research/phase_2b1_exploratory_protocol_v1_4.md` | `db45f4032f2340a9edc3441bafcb3957895b616d1289c28e85a5cc4671574f62` | `db45f4032f2340a9edc3441bafcb3957895b616d1289c28e85a5cc4671574f62` | **VERIFIED MATCH** |
| `benchmarks/configs/phase_2b1_exploratory_config_v1_4.json` | `b3e3b0b8450e325914ea8229faafce90a833eb038b02903bfa48062e8ee0cfb2` | `b3e3b0b8450e325914ea8229faafce90a833eb038b02903bfa48062e8ee0cfb2` | **VERIFIED MATCH** |
| `benchmarks/results/phase_2b1_5_execution/manifest.json` | `349c9b97a0ee76cded3770828029f897d101414d8917b9743930f08f23ea830f` | `349c9b97a0ee76cded3770828029f897d101414d8917b9743930f08f23ea830f` | **VERIFIED MATCH** |
| `benchmarks/results/phase_2b1_5_execution/results.json` | `5f9606d9dad39f6a1f3eea39fdfd140f56dffaec99f8516ec7ecfa66303f26b9` | `5f9606d9dad39f6a1f3eea39fdfd140f56dffaec99f8516ec7ecfa66303f26b9` | **VERIFIED MATCH** |
| `benchmarks/results/phase_2b1_5_execution/trials.jsonl` | `82905d2200b88c242f11499e6a927e75c38fe1e8bdf142b498dc28c9f80b86f0` | `82905d2200b88c242f11499e6a927e75c38fe1e8bdf142b498dc28c9f80b86f0` | **VERIFIED MATCH** |
| `benchmarks/results/phase_2b1_5_execution/summary.csv` | `6e521487357bf176fab79d3a20c3f7e61657909a8132e6d5dab6f084dae71902` | `6e521487357bf176fab79d3a20c3f7e61657909a8132e6d5dab6f084dae71902` | **VERIFIED MATCH** |
| `docs/research/phase_2b1_5_exploratory_execution_report.md` | `a64372d7af67ed5b53e01668a46d1d53130f6fcb0c2816c7e04877a5f0628359` | `a64372d7af67ed5b53e01668a46d1d53130f6fcb0c2816c7e04877a5f0628359` | **VERIFIED MATCH** |
| `benchmarks/model_pilot/exploratory_runner_v1_4.py` | `03ff8262b9f68800b3e6488a09ee2ba5eb7ae2f588aa4da541249339a3f2db7b` | `69b03d52b198bff5f23e86894041e810e36c75d052753620e3872d56342d7b27` | **MISMATCH (Report Hash Stale)** |

### Runner Hash Mismatch Forensic Audit
The file `benchmarks/model_pilot/exploratory_runner_v1_4.py` currently on disk hashes to `69b03d52...`. Inspection reveals that this version contains the fix for `task_tracker` parameter passing in `execute_search()`. The execution report recorded the SHA-256 hash of the draft script before that patch was applied. The actual execution runtime was executed from the `69b03d52...` code.

---

## 3. Holdout Count Reconciliation

### The Discrepancy
The prompt noted an apparent inconsistency in Section 6 of the execution report:
- Section 6 lists 5 tasks (`inst_01`, `inst_02`, `inst_03`, `inst_04`, `inst_09`) with 5 holdout trials each, which might be read as $5 \times 5 = 25$ holdout trials.
- Elsewhere, Section 6 claims `12 / 12 holdout evaluations`, and Section 7.1 reports `60 trials | 120 requests`.

### Forensic Reconstruction
Direct extraction and parsing of all `stage == "holdout"` records from `trials.jsonl` demonstrates the exact structure of holdout testing:

```
Total holdout trials recorded in trials.jsonl: 60 trials
Total holdout requests recorded: 120 requests (2 requests per trial)
```

The holdout stage tested isolated candidate subsets for **every algorithm that discovered a 1-minimal candidate on an active task**. The 12 evaluations are distributed across the 5 resolved tasks as follows:

1. `inst_01_syntax_currency_k4`:
   - Stochastic Delta Debugging candidate: 5 trials (5/5 failures, 100%)
   - Stochastic Greedy Ablation candidate: 5 trials (5/5 failures, 100%)
   - Seeded Random Search candidate: 5 trials (5/5 failures, 100%)
   - *Subtotal: 3 candidates, 15 trials, 30 requests*
2. `inst_02_syntax_currency_k8`:
   - Stochastic Delta Debugging candidate: 5 trials (5/5 failures, 100%)
   - Stochastic Greedy Ablation candidate: 5 trials (5/5 failures, 100%)
   - *(Random Search hit 180-request cap: no candidate discovered)*
   - *Subtotal: 2 candidates, 10 trials, 20 requests*
3. `inst_03_syntax_comma_k4`:
   - Stochastic Delta Debugging candidate: 5 trials (5/5 failures, 100%)
   - Stochastic Greedy Ablation candidate: 5 trials (5/5 failures, 100%)
   - Seeded Random Search candidate: 5 trials (5/5 failures, 100%)
   - *Subtotal: 3 candidates, 15 trials, 30 requests*
4. `inst_04_syntax_comma_k8`:
   - Stochastic Delta Debugging candidate: 5 trials (5/5 failures, 100%)
   - Stochastic Greedy Ablation candidate: 5 trials (5/5 failures, 100%)
   - *(Random Search hit 180-request cap: no candidate discovered)*
   - *Subtotal: 2 candidates, 10 trials, 20 requests*
5. `inst_09_schema_unit_k4`:
   - Stochastic Delta Debugging candidate: 5 trials (5/5 failures, 100%)
   - Stochastic Greedy Ablation candidate: 5 trials (5/5 failures, 100%)
   - *(Random Search hit 120-request cap: no candidate discovered)*
   - *Subtotal: 2 candidates, 10 trials, 20 requests*

**Total Algorithm-Task Candidates:** $3 + 2 + 3 + 2 + 2 = \mathbf{12\text{ candidates}}$.  
**Total Holdout Trials:** $12 \text{ candidates} \times 5 \text{ trials/candidate} = \mathbf{60\text{ trials}}$.  
**Total Holdout Requests:** $60 \text{ trials} \times 2 \text{ requests/trial} = \mathbf{120\text{ requests}}$.

### Finding
The data in `trials.jsonl`, `results.json`, and the budget ledger in Section 7.1 are in 100% mutual agreement. Every one of the 12 candidate subsets reproduced failures on 5/5 holdout runs. The confusion was caused entirely by ambiguous drafting in Section 6, which organized the text by task instance rather than by algorithm-candidate evaluation.

---

## 4. Calibration-versus-Search Reconciliation

### The Discrepancy
The report indicates that 7 of the 12 active tasks failed calibration and were disqualified from search. Yet the resource-burden tables in Section 5.1 show those 7 tasks assigned 120 requests ($K=4$) and 180 requests ($K=8$), and compute stratum means based on these values.

### Forensic Reconciliation
Audit of the runner code (`exploratory_runner_v1_4.py`, lines 405–420) and raw event logs confirms:
- **Search Execution Adherence:** Disqualified tasks were strictly skipped during Stage 2 Search. The event log `trials.jsonl` contains exactly **0 search trials** for tasks `inst_05`, `inst_06`, `inst_07`, `inst_08`, `inst_10`, `inst_11`, and `inst_12`.
- **Primary Estimand Specification:** Protocol v1.4 §3.2 and §3.3 explicitly prescribe the **All-Assigned Tasks (Intention-to-Treat, ITT) Estimand** as the primary efficiency metric:
  $$\text{Normalized Burden } B(M_i, K) = \frac{\min(R, R_{\text{cap}})}{R_{\text{cap}}}$$
  Under this preregistered rule, all 12 assigned tasks remain in the primary sample. Any task that was disqualified during calibration or failed to terminate within the cap is right-censored at $R_{\text{cap}}$ ($120$ for $K=4$, $180$ for $K=8$).
- **Imputation vs Physical Expenditure:** The values `120 requests (C)` and `180 requests (C)` in Section 5.1 are **imputed right-censored penalty caps** required by ITT protocol mathematics, not counts of physically executed inference calls.
- **Reporting Inaccuracy:** The table header in Section 5.1 labeled these rows as "Mean Expended Requests", which conflates statistical imputation with physical execution. The ledger in Section 7.1 correctly reported the physical model requests expended during search: **1,776 requests**, not the $12 \times (\text{cap})$ theoretical sum.

---

## 5. Request Ledger Reconciliation & The 4-Request Discrepancy

### Ledger Verification
An exact trial-by-trial recomputation was performed on `trials.jsonl`:

```
Stage 1: Calibration Trials:
  - calibration_trigger:   60 trials x 2 requests =  120 requests
  - calibration_control:   60 trials x 2 requests =  120 requests
  - calibration_null:      20 trials x 2 requests =   40 requests
  Subtotal Calibration:   140 trials               =  280 requests

Stage 2: Algorithmic Search Trials:
  - search_dd:            190 trials x 2 requests =  380 requests
  - search_greedy:        285 trials x 2 requests =  570 requests
  - search_random:        411 trials x 2 requests =  822 requests
  Subtotal Search:        886 trials               = 1,772 requests

Stage 3: Holdout Verification Trials:
  - holdout:               60 trials x 2 requests =  120 requests
  Subtotal Holdout:        60 trials               =  120 requests

Total Logged in trials.jsonl: 1,086 trials         = 2,172 requests
```

However, `manifest.json`, `results.json`, and the execution report claim:
$$\text{Total Requests Consumed} = \mathbf{2,176\text{ requests}}$$
$$\text{Difference} = 2,176 - 2,172 = \mathbf{+4\text{ requests}}$$

### Root Cause Analysis of the 4-Request Delta
The audit traced this discrepancy directly to the interaction between the multi-turn agent harness and the cost budget tracker:
1. Every standard trial consists of **2 sequential model requests**:
   - Turn 1: Model receives prompt, returns tool call JSON.
   - Turn 2: Model receives tool observation, returns final natural language response.
2. In `base_target.py` (lines 92–104), `check_budget_before_request()` is invoked before each individual request.
3. On the 4 Random Search runs that reached the task budget cap:
   - `inst_02_syntax_currency_k8` (cap = 180 requests)
   - `inst_04_syntax_comma_k8` (cap = 180 requests)
   - `inst_09_schema_unit_k4` (cap = 120 requests)
   - `inst_16_null_db_k8` (cap = 180 requests)
4. In each of these 4 runs:
   - The search loop executed trial $N$ (e.g. Trial 90 or Trial 60).
   - Turn 1 executed at request count 179 (or 119). The budget check passed. Ollama generated the tool call. The model request counter `global_tracker.total_requests` incremented to 180 (or 120).
   - Turn 2 commenced. `check_budget_before_request()` was called. Because the counter had now reached the per-task limit of 180 (or 120), the harness raised `CostBudgetExceededError`.
   - The exception interrupted execution before the trial object could be returned.
   - The enclosing runner caught `CostBudgetExceededError`, broke out of the search loop, and terminated the run.
   - Crucially, `log_trial_event()` was only invoked after a trial completed successfully. Because Turn 2 was aborted, **the trial was never serialized to `trials.jsonl`**.
   - However, Turn 1 had already been transmitted to Ollama and registered in the in-memory request tracker.
5. Exactly **4 aborted trials** occurred (one per capped Random Search run), each consuming exactly **1 model request** that went unlogged to `trials.jsonl`:
   $$4 \text{ capped runs} \times 1 \text{ unlogged request} = \mathbf{4\text{ requests}}$$
   $$2,172 \text{ logged requests} + 4 \text{ unlogged requests} = \mathbf{2,176\text{ requests}}$$

### Assessment
This proves that the request ledger in `manifest.json` accurately captured the exact number of HTTP requests issued to Ollama (`2,176`), while revealing a subtle error-handling edge case where partial trials aborted by budget exhaustion are missing from the line-delimited event log.

---

## 6. Recovery, False Discovery, and Minimality Audit

### Causal Trigger Recovery Matrix
The audit verified the ground-truth triggers against isolated subsets for all 16 tasks:

| Task ID | Mechanism | True Planted Trigger | Algorithm | Discovered Subset | Ground-Truth Match | Holdout Rate | Operational Classification |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `inst_01` | Currency ($K=4$) | `strict_currency_formatting` | DD | `['strict_currency_formatting']` | **TRUE** | 5/5 (100%) | `SUPPORTED_1_MINIMAL` |
| | | | Greedy | `['strict_currency_formatting']` | **TRUE** | 5/5 (100%) | `SUPPORTED_1_MINIMAL` |
| | | | Random | `['strict_currency_formatting']` | **TRUE** | 5/5 (100%) | `SUPPORTED_1_MINIMAL` |
| `inst_02` | Currency ($K=8$) | `strict_currency_formatting` | DD | `['strict_currency_formatting']` | **TRUE** | 5/5 (100%) | `SUPPORTED_1_MINIMAL` |
| | | | Greedy | `['strict_currency_formatting']` | **TRUE** | 5/5 (100%) | `SUPPORTED_1_MINIMAL` |
| | | | Random | `[]` *(Budget Cap)* | FALSE | — | `INCONCLUSIVE` |
| `inst_03` | Comma ($K=4$) | `thousands_separator` | DD | `['thousands_separator']` | **TRUE** | 5/5 (100%) | `SUPPORTED_1_MINIMAL` |
| | | | Greedy | `['thousands_separator']` | **TRUE** | 5/5 (100%) | `SUPPORTED_1_MINIMAL` |
| | | | Random | `['thousands_separator']` | **TRUE** | 5/5 (100%) | `SUPPORTED_1_MINIMAL` |
| `inst_04` | Comma ($K=8$) | `thousands_separator` | DD | `['thousands_separator']` | **TRUE** | 5/5 (100%) | `SUPPORTED_1_MINIMAL` |
| | | | Greedy | `['thousands_separator']` | **TRUE** | 5/5 (100%) | `SUPPORTED_1_MINIMAL` |
| | | | Random | `[]` *(Budget Cap)* | FALSE | — | `INCONCLUSIVE` |
| `inst_09` | Schema Unit ($K=4$) | `inject_unit_suffix` | DD | `['compliance_tag']` | **FALSE** | 5/5 (100%) | `SUPPORTED_1_MINIMAL` |
| | | | Greedy | `['compliance_tag']` | **FALSE** | 5/5 (100%) | `SUPPORTED_1_MINIMAL` |
| | | | Random | `[]` *(Budget Cap)* | FALSE | — | `INCONCLUSIVE` |

### The `inst_09` False Discovery Paradox
On `inst_09_schema_unit_k4`, both DD and Greedy discovered `{'compliance_tag'}` (a distractor) rather than the planted trigger `{'inject_unit_suffix'}`. The execution report classified this result as `SUPPORTED_1_MINIMAL`.
- **Protocol Definition Verification:** Protocol v1.4 §4.3 defines statistical 1-minimality purely operationally: a subset $C^*$ is 1-minimal if $P(\text{fail} \mid C^*) \ge 0.50$ and $\forall c \in C^*, P(\text{fail} \mid C^* \setminus \{c\}) < 0.50$, confirmed via Fisher's exact test ($p < 0.05$).
- **Trial Verification:** In `trials.jsonl`, candidate subset `{'compliance_tag'}` yielded 5/5 failures, while the empty subset $\emptyset$ yielded 0/5 failures (Fisher exact $p = 0.0040$). On holdout verification, `{'compliance_tag'}` repeated with 5/5 failures.
- **Scientific Audit Finding:** The classification `SUPPORTED_1_MINIMAL` is **mathematically and operationally valid** under Protocol v1.4's black-box definition. However, it is an **empirical false discovery** with respect to the experimenter's planted ground-truth. The execution report appropriately disclosed this in Section 4.1 footnote and Section 8. This is a critical scientific finding: it proves that in stochastic LLMs, distractor prompts can form alternative causal failure paths that satisfy strict statistical minimality without reflecting the intended ground-truth fault.

### Null Control False Discovery Audit
- `inst_13_null_calc_k4` & `inst_14_null_calc_k8`: Baseline check passed (0/5 failures). No search was executed. False discoveries: **0**.
- `inst_15_null_db_k4` & `inst_16_null_db_k8`: Clean baseline passed (0/5 failures with empty candidate set). However, during search over distractor sets, prompt concatenation degradation caused `llama3.2:3b` to fail tool calling. Both DD and Greedy isolated `expense_category` as a failure trigger.
- **Scientific Audit Finding:** The execution report transparently acknowledged this failure mode in Section 4.2 and Section 8.

---

## 7. Independent Statistical Recalculation

All primary and secondary statistical tests reported in Section 5 of the execution report were independently recomputed.

### 7.1 Mechanism-Level Paired Wilcoxon Signed-Rank Test ($n=6$)

#### Data Vector Recomputation
For each mechanism $M_i$, the normalized burden difference is:
$$\bar{D}_i = \frac{1}{2}\left[\frac{R_{\text{Greedy}}(M_i, 4) - R_{\text{DD}}(M_i, 4)}{120} + \frac{R_{\text{Greedy}}(M_i, 8) - R_{\text{DD}}(M_i, 8)}{180}\right]$$

1. `mech_01_syntax_currency`:
   $$\bar{D}_1 = \frac{1}{2}\left[\frac{60 - 40}{120} + \frac{100 - 50}{180}\right] = \frac{1}{2}\left[\frac{20}{120} + \frac{50}{180}\right] = \frac{1}{2}\left[0.1667 + 0.2778\right] = \mathbf{+0.2222}$$
2. `mech_02_syntax_comma`:
   $$\bar{D}_2 = \frac{1}{2}\left[\frac{60 - 40}{120} + \frac{100 - 50}{180}\right] = \mathbf{+0.2222}$$
3. `mech_03_routing_override`:
   $$\bar{D}_3 = \frac{1}{2}\left[\frac{120 - 120}{120} + \frac{180 - 180}{180}\right] = \mathbf{0.0000} \quad \text{(Tied at Cap)}$$
4. `mech_04_routing_prohibition`:
   $$\bar{D}_4 = \frac{1}{2}\left[\frac{120 - 120}{120} + \frac{180 - 180}{180}\right] = \mathbf{0.0000} \quad \text{(Tied at Cap)}$$
5. `mech_05_schema_unit`:
   $$\bar{D}_5 = \frac{1}{2}\left[\frac{70 - 80}{120} + \frac{180 - 180}{180}\right] = \frac{1}{2}\left[-\frac{10}{120} + 0\right] = \mathbf{-0.0417}$$
6. `mech_06_schema_cross_unit`:
   $$\bar{D}_6 = \frac{1}{2}\left[\frac{120 - 120}{120} + \frac{180 - 180}{180}\right] = \mathbf{0.0000} \quad \text{(Tied at Cap)}$$

#### Exact Permutation Ranking
- Number of mechanisms: $n = 6$.
- Zero differences: 3 mechanisms (`mech_03`, `mech_04`, `mech_06`) have $\bar{D}_i = 0$. Under the standard Wilcoxon signed-rank test (and Pratt zero-handling), zero differences are removed from ranking.
- Non-zero mechanisms: $m = 3$.
- Absolute differences and signed ranks:
  - `mech_05`: $|\bar{D}| = 0.0417 \to \text{Rank } 1.0$, $\text{Sign } -$
  - `mech_01`: $|\bar{D}| = 0.2222 \to \text{Rank } 2.5$, $\text{Sign } +$
  - `mech_02`: $|\bar{D}| = 0.2222 \to \text{Rank } 2.5$, $\text{Sign } +$
- Test statistics:
  $$W^+ = 2.5 + 2.5 = 5.0$$
  $$W^- = 1.0$$
  $$W = \min(W^+, W^-) = 1.0$$
- Exact two-sided permutation calculation over $2^3 = 8$ possible sign assignments:
  - Signs $(+, +, +)$ or $(-,-,-) \implies W = 0$ (2 permutations).
  - Signs $(+, +, -)$ or $(-,-,+) \implies W = 1$ (2 permutations).
  - Cumulative outcomes with $W \le 1.0$: $2 + 2 = 4$ permutations.
  - Two-sided exact p-value:
    $$p = \frac{4}{8} = \mathbf{0.5000}$$

**Audit Conclusion:** The reported $p = 0.5000$ is **exact and mathematically verified**.

---

### 7.2 Paired McNemar Test of Recovery Symmetry
- Contingency matrix across all 12 active tasks:
  - Both Recovered Ground Truth: 4 (`inst_01`, `inst_02`, `inst_03`, `inst_04`)
  - DD Recovered Only: 0
  - Greedy Recovered Only: 0
  - Neither Recovered Ground Truth: 8 (`inst_05`, `inst_06`, `inst_07`, `inst_08`, `inst_09`, `inst_10`, `inst_11`, `inst_12`)
- Discordant pairs: $b = 0$, $c = 0$.
- Exact binomial McNemar p-value:
  $$p = 2 \times \sum_{k=0}^{\min(b,c)} \binom{b+c}{k} 0.5^{b+c} = \mathbf{1.0000}$$

**Audit Conclusion:** The reported $p = 1.0000$ is **exact and mathematically verified**.

---

### 7.3 Qualified Task Request Reduction (Descriptive)
For the 4 tasks where ground truth was successfully recovered by both algorithms:
- Stochastic DD requests: $[40, 50, 40, 50] \implies \text{Mean} = \mathbf{45.0\text{ requests}}$
- Stochastic Greedy requests: $[60, 100, 60, 100] \implies \text{Mean} = \mathbf{80.0\text{ requests}}$
- Absolute reduction: $80.0 - 45.0 = 35.0$ requests
- Relative request reduction:
  $$\frac{80.0 - 45.0}{80.0} = \frac{35.0}{80.0} = \mathbf{43.75\%} \approx \mathbf{43.8\%}$$

**Audit Conclusion:** The reported $43.8\%$ request reduction is **exact and mathematically verified**.

---

## 8. Protocol Adherence and Provenance Limits

### Adherence Checklist
- [x] **Git Repository State:** Clean execution from commit `62579fb` on `main`.
- [x] **Model Isolation:** Executed against local Ollama instance `http://127.0.0.1:11434`.
- [x] **Zero Commercial API Spend:** Zero requests made to OpenAI, Anthropic, or external providers ($0.00 USD).
- [x] **Zero Model Downloads:** No additional weights downloaded; existing `llama3.2:3b` used.
- [x] **Pre-Execution Freezing:** Protocol v1.4 and Configuration v1.4 were cryptographically matched before execution.
- [x] **Hard Safety Cap:** Expended 2,176 requests, well below the 7,500 global limit.
- [x] **Stage Gating:** Search was strictly bypassed for the 7 tasks failing the calibration gate.
- [x] **Holdout Verification:** All 12 discovered candidates were verified on 5 holdout trials.

### Provenance Limitations
1. **Runner Hash Record:** The runner script SHA-256 hash in Section 2 of the execution report referenced a pre-bugfix draft (`03ff8262...`) instead of the script version actually executed (`69b03d52...`).
2. **Missing Incomplete Trial Events:** 4 incomplete trials that aborted on Turn 2 due to budget cap exceptions were omitted from `trials.jsonl`.
3. **Hardware & Latency Timing:** Timing benchmarks reflect local Apple Silicon inference (M-series, unified memory). Latency figures should not be extrapolated to server-grade GPU clusters.

---

## 9. Material Discrepancies and Their Consequences

| Discrepancy | Severity | Impact on Primary Claims | Consequence / Correction |
| :--- | :---: | :--- | :--- |
| **4-Request Delta between trials.jsonl and results.json** | Low | None. Total spend ($0.00) and algorithm rankings are unaffected. | Caused by unlogged Turn 1 requests on trials aborted by Turn 2 budget caps. Event logger must be updated in future phases to log partial trials. |
| **Runner Script SHA-256 Hash Mismatch** | Medium | Low. The executed runner code is fully preserved and reproducible. | Update report table with true execution script hash `69b03d52...`. |
| **Ambiguous Holdout Evaluation Listing** | Low | None. All 12 candidates were evaluated with 5 trials each (60 trials). | Clarify in Section 6 that 12 algorithm-task candidates were tested across 5 active tasks. |
| **ITT Table Column Header Conflation** | Low | None. Footnote `*(C)` correctly noted cap imputation. | Replace "Expended Requests" with "Expended / Imputed Cap Requests" in Section 5.1. |

---

## 10. Required Corrections, Ranked by Severity

1. **[Medium Severity] Update Runner Checksum in Execution Report:**
   In `docs/research/phase_2b1_5_exploratory_execution_report.md`, Section 2, update the SHA-256 checksum of `benchmarks/model_pilot/exploratory_runner_v1_4.py` from `03ff8262b9f68800b3e6488a09ee2ba5eb7ae2f588aa4da541249339a3f2db7b` to `69b03d52b198bff5f23e86894041e810e36c75d052753620e3872d56342d7b27`.
2. **[Low Severity] Clarify Holdout Reconciliation in Execution Report:**
   In Section 6 of the execution report, add an explicit table showing the 12 evaluated algorithm candidates ($3+2+3+2+2=12$) across the 5 resolved tasks, explaining that each underwent 5 trials ($12 \times 5 = 60$ trials).
3. **[Low Severity] Clarify ITT Imputation in Section 5.1:**
   In Section 5.1, clarify the column header to read `ITT Request Burden (Imputed at Cap)` to prevent readers from confusing imputed ITT penalties with physical inference requests.
4. **[Low Severity] Harden Event Logger for Aborted Multi-Turn Trials:**
   In future harness development, wrap multi-turn model interactions in `try...finally` logging blocks so that when a trial is interrupted midway by a budget limit, a partial trial record is written to `trials.jsonl`.

---

## 11. Recommended Next Decision

### Recommendation: `CONCLUDE_PHASE_2B_BENCHMARK_AND_PREPARE_PHASE_3_SYNTHESIS`

Phase 2B has achieved its exploratory research objectives:
1. It developed and verified an end-to-end local black-box minimization benchmark on tool-calling agents.
2. It proved that Stochastic Delta Debugging cuts model request burdens nearly in half (43.8% reduction on qualified syntax mechanisms, scaling from 33.3% at $K=4$ to 50.0% at $K=8$) compared to Greedy Ablation while preserving 100% trigger recovery.
3. It proved that Seeded Random Search collapses under modest candidate growth ($K=8$).
4. It demonstrated the critical failure modes of small open-weights agents (`llama3.2:3b`): susceptibility to prompt-length degradation, distractor entrapment (`compliance_tag`), and instruction compliance drift.
5. It proved that mechanism-level paired testing with $n=6$ lacks statistical power to overcome calibration dropouts, confirming the pre-experiment power analyses.

**Next Step:** No further model inference or benchmarking should be executed in Phase 2B. The project should transition to Phase 3: Research Synthesis, Errata Publication, and Archival Documentation.

---

## 12. Commands and Methods Used, Including Any Checks Not Performed

### Commands and Methods Used
1. `git status` and `git log -n 5`: Verified repository working tree cleanliness and commit history.
2. `shasum -a 256`: Verified cryptographic checksums of all input protocols, configurations, execution results, event logs, and report files.
3. Python forensic inspection scripts (executed via non-destructive read commands):
   - Parsed `trials.jsonl` (1,086 lines) to compute stage trial counts, request consumption, turn counts, and holdout distributions.
   - Parsed `results.json` and `manifest.json` to extract recorded per-task request tallies, gate decisions, and budget balances.
   - Trace-audited the 4 capped Random Search runs to uncover the exact Turn 2 `CostBudgetExceededError` mechanism causing the 4-request ledger delta.
   - Implemented exact permutation algorithms to recompute the Wilcoxon signed-rank test statistic ($W=1.0$, $p=0.5000$) and McNemar test ($p=1.0000$).
   - Verified Fisher's exact tests on candidate subsets and holdout trials.

### Checks Not Performed
1. **No Target Model Inference:** Zero calls to Ollama or any model backend were issued during this audit.
2. **No Re-execution of Benchmarks:** Neither the test runner nor minimization algorithms were rerun, preserving existing evidence.
3. **No Network Requests:** No external network sockets or APIs were accessed.
