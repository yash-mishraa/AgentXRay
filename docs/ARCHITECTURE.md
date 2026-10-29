# AgentXRay Architecture & Technical Reference

**Package:** `agentxray`  
**License:** Apache-2.0  
**Python:** >= 3.12  
**Purpose:** Comprehensive architectural overview, component contracts, and maintainer guide.

---

## 1. System Overview

AgentXRay is an open-source diagnostic instrument for isolating and minimizing failure conditions in tool-calling AI agents. Given an agent workflow and candidate input elements (such as prompt instructions, tool parameter schemas, or runtime environment flags), AgentXRay systematically searches for the minimal causal subset that reproduces the failure using black-box Delta Debugging and counterfactual verification.

```text
                                  ┌────────────────────────┐
                                  │   Developer Workflow   │
                                  └───────────┬────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
          ┌───────────────────────┐                       ┌───────────────────────┐
          │  Python SDK Facade    │                       │     Developer CLI     │
          │ (agentxray.diagnose)  │                       │      (agentxray)      │
          └───────────┬───────────┘                       └───────────┬───────────┘
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                                  ┌────────────────────────┐
                                  │  DeltaDebugMinimizer   │
                                  │   (ddmin partition)    │
                                  └───────────┬────────────┘
                                              ▼
                                  ┌────────────────────────┐
                                  │   InterventionRunner   │
                                  │ (budget/time envelope) │
                                  └───────────┬────────────┘
                                              ▼
                                  ┌────────────────────────┐
                                  │   Evaluator Callable   │
                                  │ (black-box agent/tool) │
                                  └───────────┬────────────┘
                                              ▼
                                  ┌────────────────────────┐
                                  │ Counterfactual Verif.  │
                                  │    (c \ {e} passes)    │
                                  └───────────┬────────────┘
                                              ▼
                                  ┌────────────────────────┐
                                  │    DiagnosticResult    │
                                  │  (1-minimal verified)  │
                                  └────────────────────────┘
```

The system operates strictly as a black-box instrument:
- It does not inspect model neural weights, activation vectors, or attention heads.
- It evaluates candidate subsets via caller-supplied evaluators or offline testbed environments.
- Core testbeds run entirely in memory with zero external network requests and $0.00 spend.

---

## 2. Repository Layout

```text
.
├── src/agentxray/                  # Core installable Python package
│   ├── __init__.py                 # Public package exports
│   ├── api.py                      # Public Python SDK facade (diagnose, DiagnosticResult)
│   ├── cli.py                      # Developer CLI subcommands (demo, diagnose, version)
│   ├── minimizer.py                # Hierarchical Delta Debugging engine (ddmin)
│   ├── baselines.py                # Greedy ablation and random search comparison baselines
│   ├── runner.py                   # InterventionRunner budget and timeout envelope
│   ├── verification.py             # Counterfactual 1-minimality verification
│   ├── models.py                   # Frozen Pydantic domain models and enums
│   ├── reporting.py                # Terminal card, Markdown, and JSON serialization
│   ├── statistics.py, power.py     # Statistical inference and power utilities
│   ├── stochastic_minimizer.py     # Multi-trial statistical search for stochastic agents
│   ├── agent_harness/              # Controlled mock agent, tools, and cost tracker
│   └── providers/                  # Fake, OpenAI, and Ollama model adapters
├── web/                            # Standalone multi-page product website
│   ├── index.html                  # Overview & interactive terminal tape demo
│   ├── quickstart.html             # Developer onboarding in <2 minutes
│   ├── docs.html                   # Interactive technical reference & contracts
│   ├── playground.html             # Benchmark explorer & client-side ddmin simulator
│   ├── research.html               # Algorithmic lineage & research methodology
│   ├── style.css                   # Editorial dark typography design system
│   ├── app.js                      # Client-side simulator and tape controller
│   └── vite.config.js              # Multi-page Rollup build configuration
├── benchmarks/                     # Offline testbed catalog and research protocols
├── docs/                           # Developer guides, ADRs, and research ledger
│   ├── CLI.md                      # Complete CLI flag & exit-code reference
│   ├── SDK.md                      # Public Python SDK facade documentation
│   ├── SCENARIOS.md                # Declarative scenario JSON specification
│   ├── ARCHITECTURE.md             # This document
│   └── DECISIONS.md                # Architectural Decision Records (ADRs)
├── examples/                       # Tested runnable developer examples
│   ├── diagnose_custom_agent.py    # Python SDK integration script
│   └── scenario_example.json       # Declarative JSON scenario file
├── tests/                          # 640 unit, integration, and conformance tests
├── app.py                          # Hugging Face Spaces Gradio web application
├── streamlit_app.py                # Streamlit Community Cloud fallback web demo
├── pyproject.toml                  # Hatchling package configuration & console scripts
├── vercel.json                     # Static web deployment configuration
└── LICENSE                         # Apache-2.0 open-source license
```

---

## 3. Core Component Contracts

### 3.1 Public SDK Facade (`agentxray.api`)

The primary entry point for software developers:

```python
from agentxray import diagnose, DiagnosticResult, DiagnosticStatus
```

- Accepts `candidates: Sequence[str]` and `evaluator: Callable[[tuple[str, ...]], EvaluatorReturn]`.
- Enforces strict input validation: non-empty candidate sequence, uniqueness, string type checking.
- Executes hierarchical Delta Debugging (`ddmin`), captures execution telemetry, and verifies 1-minimality counterfactually.
- Traps unhandled evaluator exceptions, returning `DiagnosticStatus.ERROR` rather than crashing.

### 3.2 Developer CLI (`agentxray.cli`)

Exposes subcommands:
- `agentxray demo`: Instant offline verification with monochrome terminal card output.
- `agentxray diagnose <file.json>`: Evaluates declarative scenarios offline.
- `agentxray version`: Metadata reporting.

Conforms to POSIX exit-code contracts:
- `0`: Success (culprit verified or clean pass).
- `1`: Inconclusive (budget exhausted or timeout).
- `2`: Usage / validation error.
- `3`: Runtime exception in evaluator.

### 3.3 Minimization Engine (`agentxray.minimizer`)

Implements Andreas Zeller’s Delta Debugging (`ddmin`) algorithm:
1. Baseline check: if full input does not fail, halts immediately with `NO_FAILURE`.
2. Binary partitioning: splits candidate pool into $n$ subsets (starting with $n=2$).
3. Subset & complement evaluation: zooms into failing partitions; tests complements if subsets pass.
4. Granularity increase: if no partition fails, increases granularity ($n = \min(2n, |C|)$).
5. Counterfactual verification: tests $c \setminus \{e\}$ for each retained element to prove 1-minimality.

### 3.4 Execution Envelope (`agentxray.runner`)

`InterventionRunner` wraps the evaluator callable and enforces:
- `max_evaluations`: Hard cap on evaluation count to prevent infinite loops or token waste.
- `timeout_seconds`: Wall-clock timeout checking before each trial dispatch.
- Provenance recording: Every trial records active candidates, timestamp, duration, and outcome.

---

## 4. Operational Boundaries & Disclosures

1. **Black-Box Constraint**: The system has no awareness of model weights, hidden states, or proprietary chain-of-thought tokens.
2. **Monotonicity**: Standard `ddmin` assumes failure monotonicity. When non-monotonic agents exhibit complex cancelation effects, extended complement passes are executed.
3. **Stochastic vs Deterministic**: Built-in testbeds run deterministically with zero network calls and $0.00 spend. When diagnosing live stochastic LLMs (temperature > 0), repeated-trial sampling and Wilson score confidence intervals are required.
