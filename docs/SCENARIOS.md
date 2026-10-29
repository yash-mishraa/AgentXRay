# Declarative Scenario Specification Guide

AgentXRay supports declarative JSON scenario specifications for reproducible offline failure minimization without executing arbitrary code (`eval`, `exec`, or shell scripts).

Declarative scenarios are ideal for:
- Automated CI/CD regression test suites
- Reproducible issue reporting in agent codebases
- Pre-deployment prompt verification
- Benchmarking diagnostic search efficiency

---

## 1. Scenario File Schema (`ScenarioFileSpec`)

```json
{
  "$schema": "https://agentxray.dev/schemas/scenario.v1.json",
  "name": "currency_ast_repro",
  "description": "AST arithmetic evaluator crash caused by currency sign injection.",
  "candidates": [
    "strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. '$150' or '$250')",
    "monetary_precision: represent all invoice amounts with exactly two decimal places",
    "professional_styling: use formal accounting terminology throughout summaries",
    "audit_trail: annotate calculations with internal ledger voucher references"
  ],
  "failure_condition": {
    "type": "contains_any",
    "patterns": [
      "strict_currency_formatting"
    ]
  },
  "ground_truth": [
    "strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. '$150' or '$250')"
  ]
}
```

---

## 2. Field Definitions

| Field | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `$schema` | `string` | Optional | JSON Schema URI for editor autocompletion and linting. |
| `name` | `string` | Yes | Unique scenario identifier (slug or short title). |
| `description` | `string` | Optional | Human-readable explanation of the reproducible failure. |
| `candidates` | `list[string]` | Yes | Non-empty list of unique candidate directive strings. |
| `failure_condition` | `object` | Yes | Specification defining when an active candidate subset reproduces the failure. |
| `ground_truth` | `list[string]` | Optional | Expected minimal failure-inducing subset for automated verification testing. |

---

## 3. Failure Condition Types

The `failure_condition.type` field determines how active candidate subsets are evaluated:

| Condition Type | Behavior | Typical Use Case |
| :--- | :--- | :--- |
| `contains_any` | Fails if the active subset contains **any** candidate that contains any pattern substring. | Single-directive faults (e.g. one bad rule breaks the tool parser). |
| `contains_all` | Fails if **every** specified pattern substring matches at least one directive in the active subset. | Multi-directive interactions (e.g. rule A + rule B together cause a conflict). |
| `exact_match` | Fails if the active subset contains all exact strings specified in `patterns`. | Strict invariant checks. |
| `always_fail` | Always fails regardless of which candidates are active. | Testing synthetic failure baselines. |
| `never_fail` | Always succeeds cleanly regardless of which candidates are active. | Testing clean baseline controls. |

---

## 4. Built-in Catalog Testbeds

AgentXRay includes five built-in scenarios that can be executed directly without creating JSON files:

```bash
agentxray demo --scenario <id>
agentxray diagnose --scenario <id>
```

| ID | Category | Scenario Summary |
| :--- | :--- | :--- |
| `currency_ast` | Prompt Syntax | Directive injects currency symbols (`$150`) into math expressions, violating AST parser. |
| `trace_prompt_01` | Prompt Steering | Directive instructs agent to compute mentally, violating required tool-use policy. |
| `trace_schema_01` | Interface Contract | Directive abbreviates argument key (`expr` instead of `expression`), causing API rejection. |
| `trace_env_01` | Infrastructure | Downstream calculator service returns HTTP 500 downtime error. |
| `trace_null_01` | Control Baseline | Clean prompt pool with zero planted faults; verifies clean passing control. |

---

## 5. Running & Validating Scenarios

Execute any scenario using `agentxray diagnose`:

```bash
# Formatted terminal card
agentxray diagnose path/to/scenario.json

# Save JSON result to artifacts directory
agentxray diagnose path/to/scenario.json --json --save artifacts/diagnosis.json
```
