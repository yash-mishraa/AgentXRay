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
| **ADR-008** | Wilson Score & Newcombe Hybrid Score for Binomial Proportions | **Accepted** | Rigorous uncertainty intervals without boundary degeneracy or SciPy dependency |
| **ADR-009** | Four-Tier Epistemic Classification for Stochastic 1-Minimality | **Accepted** | Distinguishes statistically verified vs observed-only minimal failure triggers |

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

### ADR-008: Wilson Score & Newcombe Hybrid Score Methods for Binomial Proportions

* **Context**: In stochastic agent probing, target outcomes are binomial (success vs. failure). The asymptotic Wald confidence interval ($\hat{p} \pm z \sqrt{\hat{p}(1-\hat{p})/n}$) suffers from catastrophic boundary collapse when $\hat{p} \in \{0, 1\}$, producing zero-width intervals that misrepresent uncertainty.
* **Decision**: Adopt the **Wilson score interval** for single binomial proportions and the **Newcombe hybrid score interval** for differences between independent proportions ($\hat{p}_1 - \hat{p}_2$). Standard library `statistics.NormalDist` is used for standard normal quantile evaluations (`inv_cdf` and `cdf`) and `math.erfc` for two-sided p-values.
* **Alternatives Considered**:
  * *Standard Wald interval*: Rejected due to zero-variance breakdown at extreme rates.
  * *SciPy / Statsmodels dependency*: Rejected to maintain a lean, zero-dependency experimentation engine.
* **Consequences**:
  * *Positive*: Asymptotically correct coverage, no zero-width intervals at boundaries, and strict interval bounds ($[0, 1]$ and $[-1, 1]$).
  * *Negative*: Requires iterative formulation instead of simple closed-form Wald equations (handled cleanly by standard library functions).

---

### ADR-009: Four-Tier Epistemic Classification for Stochastic 1-Minimality

* **Context**: In stochastic systems with background failure rates ($P_{\text{base}} > 0$), observed drops in sample proportion upon element ablation may stem from random sampling noise rather than true causal necessity.
* **Decision**: Delta debugging in stochastic environments classifies 1-minimality into four distinct epistemic tiers:
  1. `statistically_supported`: Every single-element removal demonstrates a statistically significant drop in failure rate (Benjamini-Hochberg FDR-adjusted $q < \alpha$).
  2. `observed_only`: The empirical failure rate dropped upon every single-element removal ($\hat{p}_S > \hat{p}_{S \setminus \{e\}}$), but one or more tests did not reach statistical significance due to sample size.
  3. `inconclusive`: Search budget was exhausted during verification or candidate could not be fully tested.
  4. `not_minimal`: Removal of at least one element does not decrease the observed failure rate or still satisfies the failure threshold.
* **Consequences**:
  * *Positive*: Prevents false causal claims in low-power or high-noise environments while acknowledging empirical reductions.
  * *Negative*: Requires multiple ablation evaluations per candidate, increasing target invocation counts.
