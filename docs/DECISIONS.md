# Architectural and Research Decision Records (ADRs)

> **AgentXRay Phase 0/1A Decision Log**  
> **Status:** Calibrated Decisions & Open Questions  
> **Author:** Lead AI Research Engineer & Software Architect  

---

## 1. Summary of Architectural Decision Records

| ADR ID | Decision Title | Status | Primary Impact |
| :--- | :--- | :--- | :--- |
| **ADR-001** | Dual Access Modes: Black-Box Probing vs. Controlled Testbeds | **Accepted** | Formal boundary definition preventing unrealistic access assumptions |
| **ADR-002** | Python Library and CLI as Primary Deliverables (Defer Dashboard) | **Accepted** | Minimizes prototype overhead, accelerates scientific validation |
| **ADR-003** | In-Memory & JSON Artifacts for Phase 1A (Defer Database Setup) | **Accepted** | Zero-dependency, lightweight, purely deterministic vertical slice |
| **ADR-004** | Causal Identification Assumptions Over Heuristic Claims | **Accepted** | Enforces SUTVA, exogeneity, FDR control, and epistemic honesty |
| **ADR-005** | Tier 1 Synthetic Ground-Truth Decoupled from Real-World Claims | **Accepted** | Algorithmic verification without conflating synthetic with LLM capability |
| **ADR-006** | Hermetic Mock Tool Interception by Default | **Accepted** | Guarantees safety and isolation during automated probing |
| **ADR-007** | Target 1-Minimality in Delta Debugging (Explicitly Not Global) | **Accepted** | Algorithmic tractability and precision without false global claims |
| **ADR-009** | Five-Tier Epistemic Classification for Stochastic 1-Minimality | **Accepted** | Distinguishes statistically verified vs observed-only minimal failure triggers |
| **ADR-010** | Multi-Seed Stochastic Audit and Honest Epistemic Disclaimers | **Accepted** | Empirical match rates with Wilson CIs, winner's curse and theoretical power disclaimers |
| **ADR-011** | Model-Backed Agent Harness and Provider Abstraction Layer | **Accepted** | Hermetic mock tools, AST safety, zero-dependency OpenAI adapter, and execution trace isolation |
| **ADR-012** | Explicit Live Opt-In, Pre-Flight Dry-Run, and Multi-Tier Resource Limits | **Accepted** | Guardrails preventing accidental spend: fake provider by default, --live requirement, and hard spend caps |
| **ADR-013** | Strategic Research Pivot: Local Open-Weights Probing Prior to Paid API Resumption | **Accepted** | Defers commercial API spend; validates real neural models locally (Ollama) with bit-for-bit reproducibility |

---

## 2. Detailed Decision Records

### ADR-001: Dual Access Modes: Black-Box Probing vs. Controlled Testbeds

* **Context**: External black-box agents accessible over standard network APIs do not expose internal system prompts, tool schemas, or environment states. Conversely, during internal development and CI testing, engineers control the full agent harness.
* **Decision**: AgentXRay formally defines two distinct operating modes:
  1. *Black-Box Mode*: Only external user inputs and observable outputs (responses and emitted tool calls) are accessible. System prompts, tool schemas, and environment states are treated as unobservable and immutable.
  2. *Controlled-Testbed Mode*: The experimenter configures the agent harness, enabling controlled manipulation of input prompts, tool schemas (docstrings, parameters), and mock environment return values.
* **Alternatives Considered**:
  * *Assuming tool schemas are always manipulable*: Rejected. Incompatible with testing arbitrary third-party black-box agents.
* **Consequences**:
  * *Positive*: Transparent, realistic expectations for what can be intervened upon across different deployment scenarios.
  * *Negative*: In black-box mode, multi-surface interventions are disabled.

---

### ADR-002: Python Library and CLI as Primary Deliverables, Deferring Web Dashboard

* **Context**: Building a full-stack web application (FastAPI + PostgreSQL + Next.js + React) in Phase 1 consumes engineering time on frontend plumbing before algorithmic validity is established.
* **Decision**: Phase 1 implements an **installable Python library (`agentxray`)** with a **terminal CLI (`python -m agentxray`)**. A web dashboard and client-server REST API are deferred to Phase 4.
* **Alternatives Considered**:
  * *Full-stack Web App from Day One*: Rejected as premature overengineering.
* **Consequences**:
  * *Positive*: Enables rapid algorithmic iteration, automated pytest testing, and frictionless developer evaluation.
  * *Negative*: Non-technical users must review terminal output and generated Markdown reports.

---

### ADR-003: In-Memory & JSON Artifacts for Phase 1A (Defer Database Setup)

* **Context**: Phase 1A requires a minimal deterministic vertical slice testing synthetic target rules. Introducing database engines (SQLite or PostgreSQL) and ORM models adds dependency overhead before the core search algorithm is proven.
* **Decision**: Phase 1A persists experiment runs directly as structured in-memory domain models, serializing output to deterministic **JSON reports** and **Markdown summaries**. Multi-session SQLite persistence is deferred to Phase 1B.
* **Alternatives Considered**:
  * *PostgreSQL / SQLite in Phase 1A*: Rejected. Unnecessary infrastructure overhead for an initial deterministic prototype.
* **Consequences**:
  * *Positive*: Ultra-fast execution, zero external files/migrations, and minimal dependencies (`pydantic` only).
  * *Negative*: Cross-session indexing requires re-running experiments until Phase 1B persistence is added.

---

### ADR-004: Causal Identification Assumptions Over Heuristic Claims

* **Context**: Statistical significance ($p < 0.05$) alone does not establish causation. Uncontrolled observational comparisons or heuristic LLM judges produce high false discovery rates.
* **Decision**: Causal claims require:
  1. *Unconfoundedness*: Treatment and control vary exclusively along the specified intervention dimension.
  2. *SUTVA*: Trials are isolated and stateless to prevent carryover effects.
  3. *Multiple Comparison Correction*: P-values are adjusted using Benjamini-Hochberg FDR control when multiple candidates are tested.
  4. *Appropriate Control*: Treatment runs must be compared against dedicated, synchronous control baselines.
* **Alternatives Considered**:
  * *Heuristic LLM-as-a-judge attribution*: Rejected. Fails to guarantee causal validity; prone to hallucination.
* **Consequences**:
  * *Positive*: Scientific integrity and defensibility of reported findings.
  * *Negative*: Rejection of claims when sample size or variance is insufficient to meet identification criteria.

---

### ADR-005: Tier 1 Synthetic Ground-Truth Decoupled from Real-World Claims

* **Context**: Evaluating search algorithms on real-world agents without verified ground truth makes measuring precision and recall impossible. However, success on synthetic rules cannot be claimed as evidence of real LLM performance.
* **Decision**:
  1. Phase 1 validates the search algorithm using a **Tier 1 Synthetic Target** with planted, mathematically known rules completely isolated from the search engine.
  2. Synthetic validation is explicitly documented as proof of algorithmic mechanics only, **not** proof of real-world LLM reverse-engineering efficacy.
  3. Real-world benchmarks (SWE-bench, GAIA) do not provide verified root-cause labels; Tier 2 real-world validation is deferred until controlled manual annotations are constructed.
* **Consequences**:
  * *Positive*: Clear, verifiable algorithmic benchmarks without inflated scientific claims.
  * *Negative*: Real-world LLM evaluation is reserved for subsequent phases.

---

### ADR-006: Hermetic Mock Tool Interception by Default

* **Context**: Automated perturbation of inputs could cause an agent to emit destructive commands (e.g. file deletion or network requests).
* **Decision**: In both Black-Box and Controlled-Testbed modes, all tool calls are intercepted by hermetic mock servers. Zero live system execution occurs without explicit authorization flags.
* **Consequences**:
  * *Positive*: Complete safety, zero side effects, and bit-for-bit repeatability.
  * *Negative*: Mock tools must define return schemas matching expected tool interfaces.

---

### ADR-007: Target 1-Minimality in Delta Debugging (Explicitly Not Global)

* **Context**: Finding the globally minimal failure-inducing subset across an arbitrary input requires testing all $2^N$ subsets (exponential complexity). Classical delta debugging ($ddmin$) guarantees **1-minimality** (no single element can be removed without losing the failure).
* **Decision**: The delta-debugging engine will isolate **1-minimal subsets** and perform an explicit verification step ($\forall e \in c, c \setminus \{e\}$ does not fail). The engine will **never claim global minimality** unless an exhaustive power-set search has been executed.
* **Consequences**:
  * *Positive*: Polynomial sample complexity ($O(N^2)$ worst case) and guaranteed local minimality.
  * *Negative*: The discovered subset may not be the globally smallest subset if multiple distinct failure triggers exist.

---

### ADR-008: Wilson Score, Newcombe Hybrid Score & Exact Fisher Tests for Proportions

* **Context**: In stochastic agent probing, target outcomes are binomial (success vs. failure). The asymptotic Wald confidence interval ($\hat{p} \pm z \sqrt{\hat{p}(1-\hat{p})/n}$) suffers from catastrophic boundary collapse when $\hat{p} \in \{0, 1\}$, producing zero-width intervals that misrepresent uncertainty. Furthermore, asymptotic z-tests degrade on small sample sizes ($n \le 20$) or near boundary rates.
* **Decision**: Adopt the **Wilson score interval** for single binomial proportions, the **Newcombe hybrid score interval** for differences between independent proportions ($\hat{p}_1 - \hat{p}_2$), and **Fisher's exact test** via standard library `math.comb` for finite-sample hypergeometric inference. Standard library `statistics.NormalDist` is used for quantile evaluations (`inv_cdf` and `cdf`) and `math.erfc` for two-sided p-values. Hypothesis testing supports prespecified `alternative="two-sided"`, `alternative="greater"`, and `alternative="less"`.
* **Alternatives Considered**:
  * *Standard Wald interval*: Rejected due to zero-variance breakdown at extreme rates.
  * *SciPy / Statsmodels dependency*: Rejected to maintain a lean, zero-dependency experimentation engine.
* **Consequences**:
  * *Positive*: Exact finite-sample hypergeometric p-values, correct asymptotic coverage, no zero-width intervals at boundaries, and strict interval bounds ($[0, 1]$ and $[-1, 1]$).
  * *Negative*: Requires iterative formulation instead of simple closed-form Wald equations (handled cleanly by standard library functions).

---

### ADR-009: Five-Tier Epistemic Classification for Stochastic 1-Minimality

* **Context**: In stochastic systems with background failure rates ($P_{\text{base}} > 0$), observed drops in sample proportion upon element ablation may stem from random sampling noise rather than true causal necessity. Furthermore, null-case scenarios (where no trigger exists and initial input failure rate falls below threshold $\tau$) must not be falsely categorized as non-minimal triggers.
* **Decision**: Delta debugging in stochastic environments classifies 1-minimality into five distinct epistemic tiers:
  1. `statistically_supported`: Every single-element removal demonstrates a statistically significant drop in failure rate (Benjamini-Hochberg FDR-adjusted $q < \alpha$). For single-element candidates ($|S|=1$), the ablation control is explicitly evaluated against the initial input without that element ($c_{\text{initial}} \setminus \{e\}$).
  2. `observed_only`: The empirical failure rate dropped upon every single-element removal ($\hat{p}_S > \hat{p}_{S \setminus \{e\}}$), but one or more tests did not reach statistical significance due to sample size.
  3. `not_minimal`: Removal of at least one element does not decrease the observed failure rate or still satisfies the failure threshold.
  4. `inconclusive`: Search budget was exhausted during verification or candidate could not be fully tested.
  5. `not_applicable`: No candidate was evaluated because the initial input failure rate fell below the operational failure threshold $\tau$ (e.g. null case).
* **Consequences**:
  * *Positive*: Prevents false causal claims in low-power or high-noise environments while acknowledging empirical reductions; accurately distinguishes null cases from non-minimal triggers.
  * *Negative*: Requires multiple ablation evaluations per candidate, increasing target invocation counts.

---

### ADR-010: Multi-Seed Stochastic Audit and Honest Epistemic Disclaimers

* **Context**: Single-seed stochastic runs suffer from sample-to-sample variance, which can produce lucky draws or winner's curse artifacts that misrepresent algorithm reliability. Pre-experimental Cohen's arcsine power calculations may also be mistaken for empirical dataset power.
* **Decision**: Implement a multi-seed stochastic audit mode (`--mode audit --audit-seeds 42,43,44,45,46`) evaluating all benchmark scenarios across Delta Debugging, Greedy Ablation, and Seeded Random Search under identical budgets. Every report must publish:
  1. Empirical match rates with Wilson 95% score confidence intervals.
  2. Null-case false positive trigger identification rates with 95% Wilson CIs.
  3. Explicit disclaimers clarifying that Cohen's arcsine calculations represent asymptotic theoretical study design power, not observed empirical power.
  4. Explicit disclaimers noting that Benjamini-Hochberg FDR adjustments control false discoveries across candidate removal checks, but do not eliminate adaptive search path selection bias (winner's curse).
* **Consequences**:
  * *Positive*: Total epistemic honesty, reproducible multi-seed verification, transparent trade-offs between algorithms.
  * *Negative*: Multi-seed execution requires $M \times$ more target calls than single-seed runs.

---

### ADR-011: Model-Backed Agent Harness and Provider Abstraction Layer

* **Context**: Moving from synthetic targets to model-backed agents requires calling LLMs and executing tool loops while preserving architectural isolation, cost safeguards, and reproducibility. Using heavy third-party agent frameworks (e.g. LangChain, CrewAI, AutoGen) adds external dependencies, hidden prompts, telemetry, and networking overhead.
* **Decision**: Implement a minimalist, lightweight provider abstraction and agent harness in standard library Python:
  1. `ModelProvider` protocol with typed inputs (`Message`, `ToolDefinition`) and outputs (`ModelResponse`, `ToolCall`, `UsageMetadata`).
  2. `FakeProvider`: A deterministic, zero-cost mock provider with programmable responses, multi-turn tool calling, and simulated failure patterns for testing and dry-runs.
  3. `OpenAIProvider`: A zero-dependency REST adapter built using standard library `urllib.request`, automatic token secret redaction (`_redact_secrets`), retry backoff, and published per-token pricing lookup (`gpt-4o-mini`, `gpt-4o`).
  4. `ControlledAgent`: A standard multi-turn agent loop executing up to a fixed turn limit ($T=5$), dispatching tool calls to a local `MockToolEnvironment`.
  5. `MockToolEnvironment`: Hermetic tools with zero live network or filesystem side effects (`safe_calculator` evaluated using standard library AST inspection, `mock_database_lookup`, `mock_unit_converter`).
  6. `ToolCallingTaskEvaluator`: Evaluates task success/failure purely based on observable execution traces (emitted tool calls and final text answers) without inspecting internal model activations.
* **Alternatives Considered**:
  * *LangChain / LlamaIndex / OpenAI SDK*: Rejected to avoid third-party networking bloat, framework churn, and credential leakage.
  * *Arbitrary Python `eval()` for mock math*: Rejected as unsafe; AST inspection allows safe arithmetic evaluation without arbitrary code execution.
* **Consequences**:
  * *Positive*: Ultra-fast execution, zero external networking dependencies, 100% hermetic safety, bit-for-bit test reproducibility via `FakeProvider`.
  * *Negative*: Provider implementations must manually format JSON schemas and REST payloads.

---

### ADR-012: Explicit Live Opt-In, Pre-Flight Dry-Run, and Multi-Tier Resource Limits

* **Context**: Uncontrolled automated probing of commercial LLM endpoints can incur significant monetary costs, rate limits, and non-deterministic failures. Automated scripts must never incur live financial charges unintentionally.
* **Decision**: Implement strict multi-tier cost and safety guardrails across the experimentation engine:
  1. **Default Tier**: CLI commands default to `--provider fake` (zero cost, offline).
  2. **Explicit Live Opt-In**: Live model calls strictly require the `--live` flag. If `--provider openai` is selected without `--live`, the engine raises an immediate, actionable error preventing execution.
  3. **Pre-Flight Dry-Run**: `--dry-run` performs configuration validation, checks provider connectivity / credentials, simulates one turn, and computes estimated token consumption and spend before executing any benchmark runs.
  4. **Multi-Tier Hard Limits**: `CostTracker` enforces hard stops on both `max_model_requests` (default: 300) and `max_spend_usd` (default: \$2.00). If either cap is hit, execution halts immediately with a `CostLimitExceededError`.
  5. **Credential Masking**: API keys are scrubbed from all logs, traces, exceptions, and serialization artifacts via `_redact_secrets()`.
* **Consequences**:
  * *Positive*: Zero accidental financial spend, complete auditability, safe CI/CD execution.
  * *Negative*: Operators running live benchmarks must pass explicit opt-in flags and set environment variables.

---

### ADR-013: Strategic Research Pivot: Local Open-Weights Probing Prior to Paid API Resumption

* **Context**: The initial Phase 1E live experiment against commercial cloud APIs (`gpt-4o-mini`) was halted due to account quota exhaustion (`credit_balance_exhausted`), producing zero valid real-model behavioral observations. Commercial cloud APIs impose monetary costs, opaque backend model updates, rate limiting, and administrative billing overhead that stall academic experimentation.
* **Decision**: Defer commercial cloud API testing (Phase 1F) and pivot the immediate empirical research milestone to **local open-weights model probing (Phase 1G.1)**:
  1. Implement a lightweight, zero-dependency `OllamaProvider` connecting to a locally hosted open-weights model server (e.g., Llama-3.2-3B, Qwen-2.5-3B, or Llama-3.1-8B).
  2. Execute the model-pilot benchmark locally on Apple Silicon hardware with zero API expenditure.
  3. Establish real-model empirical proof-of-concept for delta debugging, observable evaluator isolation, and 1-minimality classification under true neural model stochasticity before re-engaging commercial APIs.
* **Alternatives Considered**:
  * *Immediate Commercial API Retry (Phase 1F)*: Rejected. Unnecessary financial barrier and administrative friction when local open-weights models provide identical functional tool-calling validation for free.
  * *Staying Purely on Synthetic Targets*: Rejected. Synthetic Python functions cannot validate neural attention interference, prompt non-monotonicity, or tokenization subtleties.
* **Consequences**:
  * *Positive*: Zero monetary spend, zero rate limits or billing blocks, full weight persistence, reproducible seeds, and open science accessibility.
  * *Negative*: Requires a local runtime (e.g. Ollama daemon) and sufficient local RAM/VRAM to host a 3B–8B model.



