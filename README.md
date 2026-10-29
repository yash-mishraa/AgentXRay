---
title: AgentXRay
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# AgentXRay

<div align="center">

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-640%20passed-brightgreen.svg)](tests/)
[![Offline Testbed Cost](https://img.shields.io/badge/testbed%20spend-%240.00%20offline-success.svg)](#)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces/meowshra/AgentXRay)

**Causal failure-condition minimization for tool-calling AI agents.**<br>
*Isolate minimal failure-inducing prompt instructions, tool schemas, and environment flags in milliseconds—proven with counterfactual verification.*

[Overview](#1-the-problem) •
[Python SDK](#2-python-sdk-facade) •
[Developer CLI](#3-developer-cli) •
[How It Works](#4-how-it-works-delta-debugging--counterfactual-proof) •
[Documentation](#5-documentation-index) •
[Interactive Web App](#6-interactive-web-app--playground)

</div>

---

## 1. The Problem: Debugging Complex Agent Failures

When a tool-calling AI agent breaks—emitting malformed JSON, violating tool argument schemas, triggering API syntax errors, or hallucinating arguments—diagnosing the exact root cause is frustrating:

* **Massive Prompt Bloat**: Agent prompts contain dozens of instructions, schema definitions, few-shot examples, and irrelevant distractors ($N$ candidate directives).
* **Combinatorial Explosion**: There are $2^N$ potential directive combinations.
* **Manual Ablation is Slow & Expensive**: Sequentially disabling rules one by one ($O(N)$ linear scan) burns developer hours and thousands of LLM API tokens.
* **Observability Logs Don't Prove Causality**: Traditional tracing platforms log *what* happened, but cannot isolate *which specific directive caused* the breakdown.

**AgentXRay applies black-box Delta Debugging (`ddmin`) and counterfactual verification to automate root cause isolation.** It hierarchically partitions candidates in $O(\log N)$ evaluations and mathematically verifies that removing the isolated directive restores clean task success ($c \setminus \{e\}$).

---

## 2. Python SDK Facade

Install AgentXRay and diagnose your agent harness in 3 lines of Python:

```bash
pip install -e .
```

```python
from agentxray import diagnose

# 1. Directives suspected of causing the agent breakdown
candidates = [
    "strict_currency: format arithmetic tool arguments with '$' (e.g. '$150')",
    "monetary_precision: format all invoices with two decimal places",
    "professional_styling: use formal enterprise accounting terminology",
    "audit_trail: annotate calculations with ledger voucher references",
]

# 2. Test harness callable: return True if the agent fails on this active subset
def my_agent_evaluator(active_subset: tuple[str, ...]) -> bool:
    # Example: downstream arithmetic parser crashes on currency signs '$'
    return any("strict_currency" in directive for directive in active_subset)

# 3. Isolate minimal failure trigger with counterfactual proof
result = diagnose(candidates=candidates, evaluator=my_agent_evaluator)

print(result.summary)
print(f"Verified 1-minimal: {result.is_verified}")       # True
print(f"Isolated culprit:   {result.isolated_candidates}") # ('strict_currency: ...',)
print(f"Evaluations used:   {result.evaluations_used}")    # 4 calls (vs 6 greedy)
```

*(See [`docs/SDK.md`](docs/SDK.md) for full parameter references, custom evaluator return types, and exception-safety contracts.)*

---

## 3. Developer CLI

AgentXRay includes a developer-facing CLI (`agentxray`) with restrained monochrome/ANSI cards and strict POSIX exit codes:

```bash
# Run the built-in offline demonstration (<0.01s, $0.00 spend)
agentxray demo

# Diagnose a custom failure using a declarative JSON scenario file
agentxray diagnose examples/scenario_example.json

# Diagnose one of the built-in catalog testbed scenarios
agentxray diagnose --scenario trace_env_01

# Output machine-readable JSON for CI/CD scripting
agentxray demo --json
```

### Representative Terminal Output

```text
AgentXRay (currency_ast)
────────────────────────────────────────────────────────────────────────────
DIAGNOSIS
  The task fails when 1 specific candidate directive is active.

ISOLATED CONDITION
  [elem_0] strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. '$150' or '$250')

VERIFICATION
  Verified: removing this directive restores clean success (1-minimal).

SEARCH TELEMETRY
  Status:          CULPRIT ISOLATED & VERIFIED
  Evaluations:     4 calls (ddmin) vs 6 calls (greedy)
  Candidate Pool:  4 directives
  Execution Time:  0.002s
  Ground Truth:    True (100% match)

MODE & BOUNDARIES
  Execution:       Controlled offline execution ($0.00 spend, zero network requests)
────────────────────────────────────────────────────────────────────────────
```

### Process Exit-Code Contract

The CLI conforms to a strict POSIX exit-code contract for CI/CD pipelines:

| Exit Code | Classification | Description |
| :---: | :--- | :--- |
| **`0`** | `CLEAN / VERIFIED` | Minimal failure culprit isolated and verified (`VERIFIED_CULPRIT`), or clean passing baseline (`NO_FAILURE`). |
| **`1`** | `INCONCLUSIVE` | Search halted without proving 1-minimality: budget exhausted (`BUDGET_EXHAUSTED`) or timeout exceeded (`TIMEOUT`). |
| **`2`** | `USAGE_ERROR` | Bad arguments, missing file, duplicate candidate strings, or malformed JSON schema. |
| **`3`** | `RUNTIME_ERROR` | The evaluator callable or agent harness raised an unhandled Python exception (`ERROR`). |

*(See [`docs/CLI.md`](docs/CLI.md) for complete CLI flag tables and automation examples.)*

---

## 4. How It Works: Delta Debugging & Counterfactual Proof

AgentXRay adapts Andreas Zeller’s classic Delta Debugging algorithm (`ddmin`) from program input minimization to black-box agent intervention search:

```text
Candidate Directives (N=4)
[ e0: currency_$ ]  [ e1: precision_2 ]  [ e2: formal_tone ]  [ e3: audit_tag ]
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
      Partition Δ1 (e0, e1)     Partition Δ2 (e2, e3)
             [FAILS]                   [PASSES]
               │
        ┌──────┴──────┐
        ▼             ▼
    Subset (e0)   Subset (e1)
      [FAILS]       [PASSES]
        │
        ▼
   Counterfactual Verification: evaluate( ∅ )
      [PASSES]  -->  e0 is PROVEN 1-MINIMAL CULPRIT!
```

### 1-Minimality Guarantee

An isolated candidate subset $c$ is **1-minimal** if:
1. $c$ reproduces the failure: $\text{evaluate}(c) = \text{FAILURE}$.
2. For every element $e \in c$, removing $e$ restores success: $\text{evaluate}(c \setminus \{e\}) = \text{SUCCESS}$.

AgentXRay ensures zero false attributions by explicitly testing the counterfactual complement.

---

## 5. Documentation Index

Comprehensive guides for integrating and extending AgentXRay:

| Document | Description |
| :--- | :--- |
| **[`docs/SDK.md`](docs/SDK.md)** | Python SDK facade reference (`agentxray.diagnose`), type annotations, and evaluator contracts. |
| **[`docs/CLI.md`](docs/CLI.md)** | Complete developer CLI command reference, JSON streaming, flags, and exit codes. |
| **[`docs/SCENARIOS.md`](docs/SCENARIOS.md)** | Declarative scenario JSON specification for automated CI/CD regression suites. |
| **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** | Technical reference, internal component contracts, domain models, and execution envelope. |
| **[`docs/DECISIONS.md`](docs/DECISIONS.md)** | Architectural Decision Records (ADRs) explaining core design trade-offs. |
| **[`CONTRIBUTING.md`](CONTRIBUTING.md)** | Developer setup, testing guidelines, and pull request workflow. |
| **[`LICENSE`](LICENSE)** | Official Apache-2.0 open-source software license. |

---

## 6. Interactive Web App & Playground

AgentXRay provides multiple interfaces to explore and simulate failure minimization:

### Option A: Standalone Editorial Product Website (`web/`)

A multi-page developer product website built with pure standard web technologies (HTML5, CSS3, ES6 modules, Vite):

```bash
# Serve locally
python3 -m http.server --directory web 3000

# Or compile production bundle
npx --yes vite build web
```

* **Interactive Terminal Tape**: Step through all 5 benchmark scenarios with play/step/reset playback controls.
* **Client-Side ddmin Sandbox**: Enter custom directives, define a trigger pattern, and simulate Delta Debugging live in your browser.

### Option B: Hugging Face Spaces Gradio Demo (`app.py`)

A full Gradio application with execution traces, partition steps, method comparisons, and research disclosures:

```bash
pip install -e ".[demo]"
python app.py
# Opens at http://127.0.0.1:7860
```

*(Hosted live on [Hugging Face Spaces](https://huggingface.co/spaces/meowshra/AgentXRay).)*

---

## 7. Built-in Testbed Catalog

AgentXRay includes five calibrated testbeds that execute hermetically in memory at $0.00 spend:

| Scenario Key | Domain | Description | Symptom |
| :--- | :--- | :--- | :--- |
| `currency_ast` | Prompt Syntax | Currency symbols (`$150`) injected into math expression. | AST `SyntaxError: invalid syntax` |
| `trace_prompt_01` | Prompt Steering | Directive instructs agent to compute mentally, omitting required tool. | Policy breach: `Failed to invoke calculator` |
| `trace_schema_01` | Tool Contract | Directive renames parameter `expression` to `expr`. | API rejection: `Missing argument: expression` |
| `trace_env_01` | Infrastructure | Downstream calculator microservice returns HTTP 500 error. | `HTTP 500: Service Unavailable` |
| `trace_null_01` | Control Baseline | Clean candidate pool with zero planted faults. | Clean passing control (`SUCCESS`) |

---

## 8. Test Suite & Verification

AgentXRay maintains a comprehensive offline test suite with **640 passing automated tests**:

```bash
pytest tests/
# Output: 640 passed in 2.8s
```

The test suite covers:
- **Algorithmic Correctness**: Delta debugging partition logic, complement search, and 1-minimality verification.
- **SDK & CLI Contracts**: Input validation, return type coercion, exception handling, and POSIX exit codes.
- **Budget Envelopes**: Hard caps on evaluation counts and wall-clock execution timeouts.
- **Architectural Isolation**: Static AST analysis ensuring zero ground-truth leakage into search loops.
- **Cryptographic Reproducibility**: SHA-256 verification of frozen historical benchmark artifacts.

---

## 9. Operational Boundaries & Disclosures

* **Pure Black-Box Operation**: AgentXRay treats agents and tools as black boxes. It does not inspect neural weights, hidden states, or proprietary chain-of-thought tokens.
* **1-Minimality vs Global Minimality**: AgentXRay guarantees that the isolated subset is 1-minimal. Proving global minimality across all $2^N$ combinations would require exhaustive search.
* **Deterministic Testbeds vs Live Stochastic Models**: Built-in testbeds execute deterministically in milliseconds at $0.00 spend. When diagnosing live stochastic LLMs (temperature > 0), multi-trial statistical sampling and Wilson score confidence intervals are required to handle probabilistic variation.

---

## 10. License & Attribution

AgentXRay is open-source software licensed under the [Apache-2.0 License](LICENSE).
Built by **Yash Mishra** and AgentXRay contributors for observable, robust AI agent engineering.
