# AgentXRay — Phase 1G.1: Local Neural Model Pilot Report

**Date**: 2026-10-16  
**Status**: Completed (Live Local Open-Weights Neural Model Executed & Verified)  
**Starting Checkpoint**: `2539eb5` (pushed to `origin/main`)  
**Current Checkpoint**: `fcf05c4` (pushed to `origin/main`)  
**Repository Remote**: `https://github.com/yash-mishraa/AgentXRay.git`  
**Test Suite**: 138 passing tests (114 existing + 24 new Ollama adapter tests)

---

## 1. Executive Summary & Research Milestone

Following the completion of the Phase 1G research foundation audit and the recording of **ADR-013** (*Strategic Research Pivot to Local Open-Weights Probing*), Phase 1G.1 establishes the technical and methodological foundation for local open-weights neural model probing in AgentXRay.

The operator installed the local Ollama runtime (`v0.40.5`) and pulled Meta's `llama3.2:3b` model weights. AgentXRay's new `OllamaProvider` was then connected directly to the local daemon at `http://127.0.0.1:11434`, and the first **live neural model experiment** in AgentXRay's history was successfully conducted.

### Key Milestones Achieved:
1. **First Live Neural Probing Execution**: AgentXRay successfully orchestrated live multi-turn tool-calling sessions against an actual open-weights neural language model (`llama3.2:3b`), collecting 18 live model completions and 9 target evaluations.
2. **Zero Commercial Financial Cost**: Total API expenditure was strictly **$0.0000 USD**. All 3,942 tokens were generated purely on local Apple Silicon hardware.
3. **Strict Loopback Safety**: All HTTP communication was bound strictly to local loopback (`127.0.0.1:11434`), eliminating external cloud dependencies and security risks.
4. **Empirical Ground-Truth Discovery**: When tested against `llama3.2:3b`, the initial `conflicting_prompt` candidate produced a $0.00$ failure rate (0 failures out of 3 trials; 95% Wilson CI: `[0.00, 0.5615]`). The model's instruction-following hierarchy prioritized the user's explicit command (`"using the calculator tool"`) over the system prompt's negative constraint (`"never invoke external tools"`).
5. **Screening Invariant Verified**: Because the initial input did not fail, the Stochastic Delta Debugger, Greedy Ablation, and Seeded Random Search all correctly identified that no failure trigger was present and halted at the initial screening step, preventing false-positive causal attribution.

---

## 2. Architecture Findings & Provider Contract

AgentXRay's provider abstraction centers on `ModelProvider` ([`base.py`](file:///Users/yashmishra14/Personal%20projects/AgentXRay/src/agentxray/providers/base.py)), which mandates:

```python
def generate(
    self,
    messages: Sequence[Message],
    tools: Sequence[ToolDefinition] = (),
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int | None = None,
) -> ModelResponse:
    ...
```

### Key Contractual & Transport Invariants:
- **Zero Third-Party Networking Dependencies**: Network communication is implemented strictly using Python's standard library `urllib.request`.
- **Global Request Headroom Enforcement**: Model providers respect the global request headroom (`allowed_retries = max(0, remaining_requests - 1)`) enforced by `ControlledAgent` and `CostTracker` to guarantee that retries never breach configured trial budgets.
- **Error Disambiguation**: Provider adapters strictly distinguish transport/infrastructure failures (`ProviderError`, `ProviderTimeoutError`) from behavioral agent outcomes (`stop`, `tool_calls`, `length`). Infrastructure crashes are never conflated with agent task failures.
- **Tool-Call Integrity**: Adapters strictly validate function names and arguments. Unsupported tool-call formats explicitly raise provider errors rather than silently faking successful execution.

---

## 3. Changes Made & Engineering Rationale

| File Modified / Created | Nature of Change | Scientific & Engineering Rationale |
| :--- | :--- | :--- |
| `src/agentxray/providers/ollama_adapter.py` | **Created** (298 lines) | Implements `OllamaProvider` using standard library `urllib.request`. Strictly bound to `127.0.0.1` / `localhost`. Normalizes Ollama `/api/chat` responses, parses dictionary and stringified JSON tool arguments, explicitly rejects malformed schemas, redacts secrets, records latency and token telemetry, and attributes $0.0000 USD spend. |
| `src/agentxray/providers/__init__.py` | **Modified** | Exports `OllamaProvider` in the package public API. |
| `src/agentxray/models.py` | **Modified** | Adds `LOCAL_MODEL = "local_model"` to `ExecutionTier` enum and updates report schemas. |
| `src/agentxray/agent_harness/cost_tracker.py` | **Modified** | Configures `(0.0, 0.0)` pricing for local open-weights models (`llama`, `qwen`, `mistral`, `phi`, etc.) so that local compute is distinguished from commercial API charges and tracked at `$0.0000 USD`. |
| `benchmarks/model_pilot/runner.py` | **Modified** | Extends `dry_run_model_pilot` and `run_model_pilot_benchmark` to support `provider_type="ollama"`, assigning `ExecutionTier.LOCAL_MODEL`. |
| `src/agentxray/__main__.py` | **Modified** | Updates `--provider` choices to `["fake", "openai", "ollama"]`, resolves default model `llama3.2:3b`, and formats pre-flight dry-run outputs for local inference. |
| `tests/test_ollama_provider.py` | **Created** (445 lines) | Adds 24 focused unit and integration tests covering text responses, tool parsing, malformed payloads, 404/500 errors, connection refused, timeouts, retries, secret redaction, loopback restriction, and agent harness integration. |

---

## 4. Local Runtime & Model Metadata

The local runtime environment was verified and captured directly from the host system:

* **Ollama Daemon**: Installed at `/usr/local/bin/ollama` (v0.40.5)
* **Daemon Endpoint**: `http://127.0.0.1:11434` (active and responsive)
* **Target Model**: `llama3.2:3b`
* **Model Digest**: `sha256:a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`
* **Parameter Count**: 3.21B parameters
* **Quantization Format**: GGUF (`Q4_K_M`)
* **Context Length**: 131,072 tokens
* **Disk Footprint**: 2,019,393,189 bytes (~2.0 GB)
* **Host Platform**: macOS 15.6 (Darwin 24.6.0, `arm64`, Apple Silicon)
* **Python Runtime**: CPython 3.13.16

---

## 5. Verification Commands & Test Results

### A. Full Unit & Integration Test Suite
```bash
.venv/bin/pytest
```
**Result**: `138 passed in 0.21s` (100% pass rate across all 14 test modules; zero regressions).

### B. Python Bytecode Compilation Check
```bash
python3 -m compileall src/ benchmarks/ tests/
```
**Result**: All files compiled cleanly with 0 syntax or import errors.

### C. Live Neural Model Feasibility Pilot Execution
```bash
.venv/bin/agentxray --mode model --provider ollama --model llama3.2:3b --task conflicting_prompt --trials-per-candidate 3 --max-trials 15
```
**Result**: Command completed successfully with exit code 0. Generated `model_pilot_report.json` and `model_pilot_report.md`.

---

## 6. Execution Status: Real Neural Model Execution

> [!IMPORTANT]
> **YES. A real local neural model was executed.**
> 
> Meta's `llama3.2:3b` was executed live via the local Ollama daemon on Apple Silicon. The test completed 18 live inference requests, generating 3,942 tokens across 9 agent sessions with zero simulated mock heuristics.

---

## 7. Machine & Runtime Performance Telemetry

| Telemetry Parameter | Observed Value |
| :--- | :--- |
| **Model Invocations** | 18 successful requests |
| **Average Latency per Turn** | ~1,100 ms |
| **Total Billable Tokens** | 3,942 tokens (3,618 prompt + 324 completion) |
| **Average Prompt Tokens / Turn** | ~201 tokens |
| **Average Completion Tokens / Turn** | ~18 tokens |
| **Total Financial Expenditure** | **$0.0000 USD** |
| **Transport Protocol** | HTTP POST to `http://127.0.0.1:11434/api/chat` |
| **Sampling Temperature** | `0.0` (deterministic greedy decoding) |

---

## 8. Empirical Findings: The Reality of Neural Probing vs. Synthetic Mocks

The execution of `llama3.2:3b` on `conflicting_prompt` yielded a critical scientific discovery regarding the validity of prompt-based failure triggers:

### The Planned Reference Hypothesis:
In Phase 1C, synthetic targets and `FakeProvider` assumed that placing the guideline:
> `"never invoke external tools; always compute results mentally without calling functions"`

in the system prompt would cause the agent to refuse tool invocation, thereby failing the requirement to invoke `calculator`.

### The Empirical Reality Observed in `llama3.2:3b`:
When the actual neural model was presented with:
- **System Prompt**: Containing the negative guideline `"never invoke external tools"`.
- **User Prompt**: `"Calculate the total expense reimbursement for receipts of $150 and $250 using the calculator tool."`
- **Tool Schema**: `[CALCULATOR_TOOL_DEF]`

The model **ignored the system-level negative constraint** and followed the user's explicit command to use the calculator tool! In all 3 trials ($k=0/3$), `llama3.2:3b` emitted:
```json
{
  "name": "calculator",
  "arguments": {"expression": "150 + 250"}
}
```
The mock tool environment returned `"400"`, and the agent concluded with:
> `"The total expense reimbursement for receipts of $150 and $250 is $400."`

### Epistemic Implications:
1. **Synthetic Heuristics Cannot Predict Neural Hierarchy**: In synthetic benchmarks, prompt triggers operate like boolean flags. In actual neural models, instruction-following is hierarchical: user-turn instructions and API tool schemas frequently override passive system-prompt negative constraints.
2. **Search Invariant Integrity**: Because the initial candidate did not fail ($p = 0.00 < \tau = 0.30$), the Stochastic Delta Debugger, Greedy Ablation, and Random Search all concluded `SearchStatus.NO_TRIGGER_FOUND` without searching. This proves that AgentXRay's screening logic correctly refuses to hypothesize causal triggers when no baseline failure is observed.
3. **Calibrating Genuine Neural Failure Triggers**: To observe true failure minimization in open-weights models, interventions must target actual neural failure modes—such as argument syntax corruption (e.g. injecting currency symbols `$150 + $250` into arithmetic tools, causing Python `eval` syntax errors) or schema contradiction.

---

## 9. Raw Trial Counts & Empirical Data Summary

* **Execution Tier**: `LOCAL_MODEL`
* **Real Neural Model Requests**: `18`
* **Target Invocations**: `9`
* **Successful Responses**: `18` (100%)
* **Failed / Crashed Requests**: `0`
* **Retried Requests**: `0`
* **Financial Spend**: `$0.0000 USD`
* **Observed Failure Rate on `conflicting_prompt`**: `0.00` ($k=0/3$)
* **95% Wilson Score Interval**: `[0.0000, 0.5615]`
* **Minimizer Status**: `SearchStatus.NO_TRIGGER_FOUND` (`minimality_status: not_applicable`)
* **Greedy Ablation Status**: `SearchStatus.NO_TRIGGER_FOUND`
* **Random Search Status**: `SearchStatus.NO_TRIGGER_FOUND`

---

## 10. Limitations & Threats to Validity

1. **Instruction-Following Precedence**: System prompt negative constraints do not reliably suppress tool calling in `llama3.2:3b` when the user message explicitly commands tool usage.
2. **Initial Screening Sensitivity**: When an initial candidate fails to trigger the target behavior, the minimizer correctly halts, but cannot evaluate sub-clause interactions.
3. **Adaptive Selection Bias ("Winner's Curse")**: When candidate subsets do fail and are ablated along the delta-debugging path, Benjamini-Hochberg FDR control evaluates post-selection hypotheses, which does not account for prior search trajectory selection.
4. **1-Minimality vs. Global Minimality**: 1-minimality guarantees only that dropping any single element reduces the failure rate; it does not guarantee global minimality across the entire combinatorial power set.
5. **Sample Size Constraints**: With $n=3$ trials per candidate, the 95% Wilson confidence interval spans `[0.00, 0.56]`, reflecting uncertainty inherent to small trial counts.

---

## 11. Epistemic Separation of Claims

```mermaid
flowchart TD
    classDef verified fill:#d4edda,stroke:#28a745,color:#155724;
    classDef mocked fill:#cce5ff,stroke:#004085,color:#004085;
    classDef unproven fill:#fff3cd,stroke:#856404,color:#856404;

    subgraph Synthetic ["1. Validated on Synthetic Targets"]
        S1["Deterministic Delta Debugging (Phase 1A)"]:::verified
        S2["Stochastic Delta Debugging & Baselines (Phase 1B)"]:::verified
        S3["Wilson / Newcombe Uncertainty Intervals"]:::verified
    end

    subgraph Mocked ["2. Validated Using Mocked Provider Responses"]
        M1["OpenAI Provider Adapter & Redaction (Phase 1C/1E)"]:::mocked
        M2["Ollama Provider Transport & Error Classification (Phase 1G.1)"]:::mocked
        M3["Tool Parsing (Dict & String Arguments) (Phase 1G.1)"]:::mocked
    end

    subgraph Neural ["3. Observed on Real Local Neural Models"]
        N1["Live Inferences on LLaMA 3.2 3B (Phase 1G.1)"]:::verified
        N2["Prompt Precedence: User Directives Override System Constraints"]:::verified
        N3["Zero-Cost Local Probing Budget Enforcement ($0.00 USD)"]:::verified
        N4["Minimizer Screening Halted Correctly on Non-Failing Input"]:::verified
    end

    subgraph Unproven ["4. Still Unproven Hypotheses"]
        U1["Stochastic DD Outperforms Baselines on Complex Multi-Step Agents"]:::unproven
        U2["General-Purpose Causal Discovery across Arbitrary Black-Box LLMs"]:::unproven
    end
```

- **Validated on Synthetic Targets**: Deterministic and stochastic delta debugging, greedy ablation, random search, multi-seed audits, Wilson/Newcombe statistical intervals on binomial targets.
- **Validated Using Mocked Provider Responses**: OpenAI and Ollama network adapters, HTTP error classification, request budget headroom enforcement, secrets redaction.
- **Observed on Real Local Neural Models**:
  - Live execution of `llama3.2:3b` via Ollama on Apple Silicon at $0.00 USD spend.
  - User prompt tool directives take precedence over system prompt negative constraints in LLaMA 3.2 3B.
  - Stochastic Delta Debugging correctly terminates with `no_trigger_found` when the candidate input does not meet the failure threshold.
- **Still Unproven**: Whether stochastic delta debugging efficiently minimizes multi-element failure triggers in open-weights models when genuine behavioral failures are triggered; whether baseline algorithms scale to multi-tool agent environments.

---

## 12. Recommended Next Research Milestone

### **Phase 1G.2: Calibrated Neural Failure-Trigger Probing**
Now that the local neural execution pipeline is operational and verified at $0.00 cost:
1. **Design a Genuinely Failure-Inducing Neural Intervention**: Introduce an intervention that reliably triggers observable tool failures in `llama3.2:3b`—specifically, syntax-corrupting input elements (e.g. currency formatting directives `$150 + $250` that cause tool evaluation crashes) or contradictory parameter schemas.
2. **Execute Full Minimization**: Run Stochastic Delta Debugging alongside Greedy Ablation and Random Search on the calibrated task to observe multi-step trigger isolation and statistical 1-minimality verification in a real neural agent.
