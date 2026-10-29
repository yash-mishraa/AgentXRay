# AgentXRay Developer CLI Reference

The `agentxray` command-line interface provides terminal diagnostic tools for isolating and minimizing failure conditions in tool-calling AI agents. Built directly on top of the public Python SDK (`agentxray.diagnose`), it offers reproducible offline verification, JSON streaming, and a strict process exit-code contract for CI/CD automation.

---

## 1. Installation & Entry Point

When installed in editable mode or from wheel, the CLI entry point `agentxray` is registered automatically:

```bash
pip install -e .
agentxray --help
```

You can also run the CLI directly via Python module invocation:

```bash
python -m agentxray --help
```

---

## 2. Subcommands Overview

| Subcommand | Purpose | Primary Flag(s) |
| :--- | :--- | :--- |
| `demo` | Run offline demonstration with instant terminal card output. | `[--scenario <id>]`, `[--seed <int>]`, `[--json]` |
| `diagnose` | Run causal minimization on a scenario JSON file or built-in catalog testbed. | `<path.json>`, `[--max-evals <int>]`, `[--timeout <sec>]` |
| `version` | Display installed package version and engine status. | None |

---

## 3. Subcommand Reference

### `agentxray demo`

Executes an offline deterministic demonstration using the public `diagnose()` engine and prints an ANSI-formatted diagnostic card to stdout.

#### Syntax

```bash
agentxray demo [--scenario <id>] [--seed <int>] [--no-color] [--json] [--save <path>]
```

#### Flags

* `--scenario <id>`: Benchmark scenario identifier to run.
  * Options: `currency_ast` (default), `trace_prompt_01`, `trace_schema_01`, `trace_env_01`, `trace_null_01`.
* `--seed <int>`: Deterministic random seed for partition ordering (default: `42`).
* `--no-color`: Disable ANSI color escape codes (auto-disabled if stdout is not a TTY or `NO_COLOR=1`).
* `--json`: Print raw machine-readable JSON output instead of the formatted terminal card.
* `--save <path>`: Write the complete JSON `DiagnosticResult` to disk.

#### Example: Running Default Demo

```bash
$ agentxray demo
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

#### Example: Outputting JSON for Scripting

```bash
$ agentxray demo --json
{
  "status": "VERIFIED_CULPRIT",
  "summary": "The task fails when 1 specific candidate directive is active: [elem_0] strict_currency_formatting...",
  "is_verified": true,
  "is_one_minimal": true,
  "isolated_candidates": [
    "strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. '$150' or '$250')"
  ],
  "isolated_candidate_ids": ["elem_0"],
  "evaluations_used": 4,
  "execution_time_seconds": 0.0021
}
```

---

### `agentxray diagnose`

Diagnoses a reproducible failure using either a declarative scenario JSON file or a named built-in catalog testbed.

#### Syntax

```bash
agentxray diagnose <scenario_file.json> [flags...]
agentxray diagnose --scenario <catalog_id> [flags...]
```

#### Flags

* `scenario_file`: Path to a valid JSON scenario specification file (see [Scenario Schema](SCENARIOS.md)).
* `--scenario <id>`: Run a built-in scenario directly (alternative to providing a file path).
* `--seed <int>`: Random seed (default: `42`).
* `--max-evals <int>`: Maximum evaluation budget allowed (default: `50`).
* `--timeout <float>`: Wall-clock timeout in seconds (default: `60.0`).
* `--no-color`: Disable ANSI formatting.
* `--json`: Emit JSON result to stdout.
* `--save <path>`: Write JSON output to file.

#### Example: Diagnosing a Custom Scenario

```bash
$ agentxray diagnose examples/scenario_example.json
AgentXRay (currency_formatting_directive_failure)
────────────────────────────────────────────────────────────────────────────
DIAGNOSIS
  The task fails when 1 specific candidate directive is active.

ISOLATED CONDITION
  [elem_0] strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. '$150' or '$250')

VERIFICATION
  Verified: removing this directive restores clean success (1-minimal).
────────────────────────────────────────────────────────────────────────────
```

---

### `agentxray version`

Reports installed version metadata.

```bash
$ agentxray version
AgentXRay 0.1.0 (Python 3.12.2, minimizer: ddmin ready)
```

---

## 4. Process Exit-Code Contract

The CLI implements strict exit codes for integration into CI pipelines, shell scripts, and automated test harnesses:

| Exit Code | Classification | Triggered When | Recommended CI Action |
| :---: | :--- | :--- | :--- |
| **`0`** | `CLEAN / VERIFIED` | A minimal failure-inducing culprit was isolated and verified (`VERIFIED_CULPRIT`), or the baseline passed without error (`NO_FAILURE`). | Pass build step. Parse isolated directive from JSON or terminal card. |
| **`1`** | `INCONCLUSIVE` | Search halted without proving 1-minimality: budget exhausted (`BUDGET_EXHAUSTED`), timeout exceeded (`TIMEOUT`), or evaluator returned ambiguous results. | Warn or fail depending on CI strictness. |
| **`2`** | `USAGE_ERROR` | Bad command line arguments, missing scenario file, duplicate candidate strings, or malformed JSON schema. Handled before search commences. | Fail build immediately. Correct flag usage or schema syntax. |
| **`3`** | `RUNTIME_ERROR` | An unhandled exception was raised by an evaluator callable or agent harness during execution (`ERROR`). | Fail build. Inspect stack trace. |

---

## 5. CI / Automation Integration Example

```bash
#!/usr/bin/env bash
set -eo pipefail

echo "==> Running AgentXRay regression diagnosis..."
if agentxray diagnose tests/fixtures/ci_scenario.json --json --save artifacts/diagnosis.json; then
  echo "==> Failure successfully isolated and verified:"
  jq '.isolated_candidates' artifacts/diagnosis.json
else
  EXIT_CODE=$?
  echo "==> Diagnosis failed or was inconclusive (exit code ${EXIT_CODE})"
  exit "${EXIT_CODE}"
fi
```

---

## 6. Security Boundaries

* **No Arbitrary Code Execution**: The CLI evaluates declarative JSON conditions and built-in benchmark scenarios. It does **not** evaluate arbitrary Python strings (`eval`, `exec`) or execute subshells.
* **Custom Dynamic Agents**: To diagnose live Python agent harnesses (such as LangChain, LangGraph, or custom dispatchers), use the [Python SDK](SDK.md) (`agentxray.diagnose`).
