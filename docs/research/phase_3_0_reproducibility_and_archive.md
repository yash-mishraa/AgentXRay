# AgentXRay — Phase 3.0: Reproducibility, Artifact Inventory, and Archival Guide

**Document Reference:** `docs/research/phase_3_0_reproducibility_and_archive.md`  
**Date:** 2026-10-17  
**Author:** AI Research & Systems Engineering Team (Antigravity)  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Operational Status:** Benchmark Concluded — Archive & Synthesis Mode

---

## 1. Overview and Safe Verification Policy

This document serves as the authoritative reproducibility guide and artifact inventory for Project AgentXRay. It specifies the repository layout, cryptographic checksums, artifact lineages, and deterministic offline verification workflows.

> [!WARNING]
> **STRICT EXECUTION BOUNDARY: DO NOT EXECUTE MODEL INFERENCE**  
> Benchmark runners (e.g. `exploratory_runner_v1_4.py`, `calibration_runner.py`) and live model adapters issue HTTP requests to local Ollama daemon endpoints or external APIs. **Executing these scripts will generate new non-deterministic observations, overwrite historical results, and violate the frozen execution boundary.** Only offline, deterministic analysis commands documented in Section 5 are authorized.

---

## 2. Repository and Artifact Layout

```
AgentXRay/
├── benchmarks/
│   ├── configs/
│   │   ├── phase_2b1_exploratory_config_v1_4.json     # Frozen execution configuration
│   │   └── ... (historical configs v1.0 - v1.3)
│   ├── model_pilot/
│   │   ├── confirmatory_tasks.py                     # Preregistered task definitions & specs
│   │   ├── exploratory_runner_v1_4.py                # Benchmark runner (INFERENCE CAPABLE - DO NOT RUN)
│   │   └── validate_confirmatory_protocol.py         # Static protocol validator (SAFE OFFLINE)
│   └── results/
│       ├── phase_2b1_5_execution/                    # Frozen execution results directory
│       │   ├── manifest.json                         # Execution metadata & global budget tallies
│       │   ├── results.json                          # Structured stage outputs & statistics
│       │   ├── trials.jsonl                          # Raw event stream (1,086 completed trials)
│       │   └── summary.csv                           # Tabular task metrics
│       └── phase_2b1_6_audit/                        # Post-execution forensic audit
│           └── phase_2b1_6_independent_post_execution_audit.md
├── docs/
│   ├── PRIOR_ART_REVIEW.md                           # Systematic literature landscape
│   ├── RESEARCH_PROPOSAL.md                          # Foundational project proposal
│   └── research/                                     # Research reports and protocol freezes
│       ├── phase_2b1_exploratory_protocol_v1_4.md    # Frozen experimental protocol
│       ├── phase_2b1_5_exploratory_execution_report.md# Contemporaneous execution report
│       ├── phase_3_0_errata_and_provenance.md        # Official errata & audit reconciliation
│       ├── phase_3_0_agentxray_research_report.md    # Main research synthesis paper
│       ├── phase_3_0_reproducibility_and_archive.md  # This document
│       └── phase_3_0_research_summary.md             # Concise research summary
├── src/agentxray/                                    # Core library implementation
│   ├── agent_harness/                                # ReAct harness, mock tools, budget tracker
│   ├── minimizer.py                                  # Deterministic delta debugging
│   ├── stochastic_minimizer.py                       # Stochastic delta debugging
│   ├── stochastic_baselines.py                       # Greedy ablation & seeded random search
│   └── statistics.py                                 # Wilson, Newcombe, Wilcoxon, McNemar, FDR
└── tests/                                            # Deterministic offline unit & audit test suite
```

---

## 3. Comprehensive Evidence Inventory

The table below catalogs the complete evidence inventory of Project AgentXRay, detailing artifact classifications, verification statuses, and supported claims:

| Artifact Path | Artifact Type | Inspected? | Provenance / Hash Match | Supported Claims | Unsupported / Prohibited Claims |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `docs/research/phase_2b1_exploratory_protocol_v1_4.md` | Protocol | Yes | Verified (`db45f403...`) | Establishes frozen rules, estimands, and stopping criteria. | Cannot prove empirical execution on its own. |
| `benchmarks/configs/phase_2b1_exploratory_config_v1_4.json` | Config | Yes | Verified (`b3e3b0b8...`) | Establishes frozen parameters, task sets, and request caps. | Cannot prove model adherence without logs. |
| `benchmarks/model_pilot/confirmatory_tasks.py` | Code | Yes | Verified (In repo) | Defines candidate elements, ground truth, and mock schemas. | Does not represent real-world API tools. |
| `benchmarks/results/phase_2b1_5_execution/manifest.json` | Original Data | Yes | Verified (`349c9b97...`) | Tracks 2,176 requests, timestamps, and environment metadata. | Does not contain individual trial traces. |
| `benchmarks/results/phase_2b1_5_execution/results.json` | Original Data | Yes | Verified (`5f9606d9...`) | Captures stage outcomes, minimality statuses, and test results. | Does not repair the 4 aborted trial traces. |
| `benchmarks/results/phase_2b1_5_execution/trials.jsonl` | Original Data | Yes | Verified (`82905d22...`) | Records 1,086 completed trials with prompt, tool, and response. | Incomplete: missing 4 aborted Turn 1 calls. |
| `benchmarks/results/phase_2b1_5_execution/summary.csv` | Original Data | Yes | Verified (`6e521487...`) | Provides flat per-task tabular summary for quick inspection. | Secondary derivative of results.json. |
| `benchmarks/model_pilot/exploratory_runner_v1_4.py` | Code | Yes | Hash Mismatch (`69b03d52...` on disk vs `03ff8262...` in report) | Proves execution logic with `task_tracker` bugfix. | Pre-patch draft hash cannot be asserted. |
| `docs/research/phase_2b1_5_exploratory_execution_report.md` | Report | Yes | Verified (`a64372d7...`) | Documents contemporaneous benchmark execution narrative. | Contains known errata (ERR-01 through ERR-06). |
| `benchmarks/results/phase_2b1_6_audit/phase_2b1_6_independent_post_execution_audit.md` | Audit | Yes | Verified (`f35fde1b...`) | Proves forensic reconciliation of ledger and holdouts. | Audit only; did not execute model inference. |
| `docs/research/phase_3_0_errata_and_provenance.md` | Errata | Yes | Verified (`1b3b8540...`) | Authoritative errata record for Phase 2B.1.5. | Documentation only; cannot restore raw logs. |
| `docs/research/phase_3_0_agentxray_research_report.md` | Synthesis | Yes | Verified (`2d4a7e95...`) | Comprehensive research paper and statistical synthesis. | Does not claim peer review or model generality. |

---

## 4. Cryptographic Hash Ledger

| File | SHA-256 Checksum | Scope & Verification Note |
| :--- | :--- | :--- |
| `docs/research/phase_2b1_exploratory_protocol_v1_4.md` | `db45f4032f2340a9edc3441bafcb3957895b616d1289c28e85a5cc4671574f62` | Frozen Protocol Specification |
| `benchmarks/configs/phase_2b1_exploratory_config_v1_4.json` | `b3e3b0b8450e325914ea8229faafce90a833eb038b02903bfa48062e8ee0cfb2` | Frozen Benchmark Configuration |
| `benchmarks/model_pilot/exploratory_runner_v1_4.py` | `69b03d52b198bff5f23e86894041e810e36c75d052753620e3872d56342d7b27` | Executed Runner Code (Post-Patch) |
| `benchmarks/results/phase_2b1_5_execution/manifest.json` | `349c9b97a0ee76cded3770828029f897d101414d8917b9743930f08f23ea830f` | Original Execution Output |
| `benchmarks/results/phase_2b1_5_execution/results.json` | `5f9606d9dad39f6a1f3eea39fdfd140f56dffaec99f8516ec7ecfa66303f26b9` | Original Execution Output |
| `benchmarks/results/phase_2b1_5_execution/trials.jsonl` | `82905d2200b88c242f11499e6a927e75c38fe1e8bdf142b498dc28c9f80b86f0` | Original Execution Output |
| `benchmarks/results/phase_2b1_5_execution/summary.csv` | `6e521487357bf176fab79d3a20c3f7e61657909a8132e6d5dab6f084dae71902` | Original Execution Output |
| `docs/research/phase_2b1_5_exploratory_execution_report.md` | `a64372d7af67ed5b53e01668a46d1d53130f6fcb0c2816c7e04877a5f0628359` | Contemporaneous Execution Report |
| `benchmarks/results/phase_2b1_6_audit/phase_2b1_6_independent_post_execution_audit.md` | `f35fde1b5ce020c3f1d37d456140c84ab5ecf4b7a881ed19474908ca7dd2b3e6` | Phase 2B.1.6 Forensic Audit Report |
| `docs/research/phase_3_0_errata_and_provenance.md` | `1b3b85407242138521ca074b62cd8b4c8c46c73e187ed77677219f6450e1fa1c` | Phase 3.0 Errata and Provenance Record |
| `docs/research/phase_3_0_agentxray_research_report.md` | `2d4a7e954aeb2e0c3afd5c5863bb94de73ef76557d25d2079ff68fa8360bf529` | Phase 3.0 Main Research Report |

---

## 5. Safe, Offline Verification Commands

All commands in this section are 100% deterministic, offline, and guaranteed not to invoke model inference.

### 5.1 Protocol and Configuration Consistency Check
Validates that task definitions, candidate sizes, and preregistered bounds match frozen specifications:
```bash
.venv/bin/python -m benchmarks.model_pilot.validate_confirmatory_protocol \
  --config benchmarks/configs/phase_2b1_exploratory_config_v1_4.json \
  --protocol docs/research/phase_2b1_exploratory_protocol_v1_4.md
```
*Expected Output:* `Status: VALIDATION_PASSED`, `Checks Passed: 35`, `Discrepancies: 0`.

### 5.2 Core Unit and Statistical Tests
Runs offline algorithm tests, Wilson intervals, Newcombe intervals, and FDR adjustments:
```bash
.venv/bin/pytest tests/test_minimizer.py \
                 tests/test_stochastic_minimizer.py \
                 tests/test_baselines.py \
                 tests/test_statistics.py
```
*Expected Output:* `38 passed`.

### 5.3 Preregistration and Protocol Integrity Test Suite
Runs adversarial and consistency tests for protocol specifications:
```bash
.venv/bin/pytest tests/test_phase_2b1_2_audit.py \
                 tests/test_phase_2b1_3_audit.py \
                 tests/test_phase_2b1_4_audit.py \
                 tests/test_phase_2b1_protocol_audit.py
```
*Expected Output:* `39 passed`.

### 5.4 Offline Raw Log and Statistical Recomputation
Parses `trials.jsonl` directly to verify trial tallies, stage request breakdowns, and exact Wilcoxon/McNemar tests without executing models:
```bash
.venv/bin/python -c '
import json
from pathlib import Path

trials = [json.loads(l) for l in Path("benchmarks/results/phase_2b1_5_execution/trials.jsonl").read_text().splitlines() if l.strip()]
print(f"Total Completed Trials Logged: {len(trials)}")
print(f"Total Requests in Logged Trials: {sum(t[\"model_requests_used\"] for t in trials)}")

results = json.loads(Path("benchmarks/results/phase_2b1_5_execution/results.json").read_text())
stats = results["statistics"]
print("Mechanism-level Wilcoxon p-value:", stats["mechanism_level_burden"]["wilcoxon_exact_permutation_p_value"])
print("McNemar exact p-value:", stats["recovery_summary"]["mcnemar_exact_p_value"])
'
```
*Expected Output:*
- `Total Completed Trials Logged: 1086`
- `Total Requests in Logged Trials: 2172`
- `Mechanism-level Wilcoxon p-value: 0.5`
- `McNemar exact p-value: 1.0`

---

## 6. What Can and Cannot Be Reconstructed Offline

### What Can Be Fully Reconstructed
1. **Trial-Level Behavioral Traces:** All 1,086 completed trials in `trials.jsonl` contain full prompt strings, candidate subsets, tool calls, tool responses, model outputs, and latency measurements.
2. **Statistical Estimands:** All Wilson intervals, Newcombe hybrid intervals, Wilcoxon signed-rank tests, McNemar tests, and ITT normalized burdens can be deterministically recalculated from existing artifacts.
3. **Holdout Verification:** All 60 holdout trials across the 12 candidate subsets are fully recorded and verifiable.
4. **Budget Accounting:** The 2,176 requests tracked in `manifest.json` and `results.json` are fully reconciled against the 1,086 completed trials plus the 4 aborted trials.

### What Cannot Be Reconstructed (Requires Fresh Model Inference)
1. **The Four Aborted Trial Payloads:** The tool arguments and response text of Turn 1 on the four trials halted by `CostBudgetExceededError` cannot be recovered without reinvoking `llama3.2:3b`.
2. **Behavior on Alternative LLMs:** Performance on larger open-weights models (`llama3.1:8b`, `qwen2.5:7b`) or frontier APIs requires new, authorized live inference runs.
3. **Continuous Trajectory State Slicing:** Extending minimization to intermediate conversation turns requires fresh multi-turn harness execution.
