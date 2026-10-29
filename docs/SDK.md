# AgentXRay Public Python SDK Reference

The `agentxray` Python SDK provides a clean, typed, ergonomic interface for diagnosing and minimizing failure conditions in tool-calling AI agents without requiring knowledge of internal research data structures.

---

## 1. Quick Start

```python
from agentxray import diagnose

# 1. Provide candidate directives suspected of triggering the failure
candidates = [
    "format_as_currency: format all tool arguments with dollar signs (e.g. '$150')",
    "enforce_precision: render invoice amounts with 2 decimal places",
    "use_markdown_tables: summarize final answers in tables",
]

# 2. Implement an evaluator callable
def evaluator(active_subset: tuple[str, ...]) -> bool:
    # Returns True if the agent fails on this active subset, False if it succeeds
    return any("format_as_currency" in directive for directive in active_subset)

# 3. Execute causal failure minimization
result = diagnose(candidates=candidates, evaluator=evaluator)

print(result.summary)
print(f"Verified 1-minimal: {result.is_verified}")
print(f"Isolated culprit:   {result.isolated_candidates}")
```

---

## 2. API Signature: `diagnose()`

```python
def diagnose(
    candidates: Sequence[str] | None = None,
    evaluator: Callable[[tuple[str, ...]], EvaluatorReturn] | None = None,
    *,
    elements: Sequence[str] | None = None,
    max_evaluations: int = 50,
    timeout_seconds: float = 60.0,
    seed: int = 42,
    ground_truth: Sequence[str] | None = None,
    algorithm: str = "ddmin",
) -> DiagnosticResult
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `candidates` | `Sequence[str]` | `None` | Candidate prompt instructions, schema definitions, or flags. Must be non-empty and unique. (Alias: `elements`). |
| `evaluator` | `Callable[[tuple[str, ...]], EvaluatorReturn]` | `None` | Test harness callable that evaluates an active candidate subset and returns whether a failure occurred. |
| `elements` | `Sequence[str]` | `None` | Alias for `candidates` (keyword-only). |
| `max_evaluations` | `int` | `50` | Maximum number of target evaluations before halting with `BUDGET_EXHAUSTED`. |
| `timeout_seconds` | `float` | `60.0` | Total wall-clock timeout in seconds before halting with `TIMEOUT`. |
| `seed` | `int` | `42` | Random seed for deterministic partition ordering. |
| `ground_truth` | `Sequence[str]` | `None` | Optional expected minimal culprit subset for automated validation in CI test suites. |
| `algorithm` | `str` | `"ddmin"` | Minimization algorithm. Currently `"ddmin"` (hierarchical Delta Debugging). |

---

## 3. Evaluator Callable Contract

The evaluator represents your test harness. It receives the currently active candidate subset as a tuple of strings (`active_subset: tuple[str, ...]`) and evaluates whether your agent fails when only those directives are enabled.

### Supported Return Types

AgentXRay supports several return conventions for convenience and expressive reporting:

| Return Type | Meaning | Example |
| :--- | :--- | :--- |
| `bool` | `True` = Failure reproduced.<br>`False` = Clean success. | `return any("bad_rule" in x for x in subset)` |
| `tuple[bool, str]` | `(is_failure, diagnostic_message)` | `return (True, "SyntaxError: $ rejected by math tool")` |
| `TargetOutcome` | Explicit enum: `FAILURE`, `SUCCESS`, `INCONCLUSIVE`. | `return TargetOutcome.FAILURE` |
| `TargetResult` | Full structured outcome with `outcome`, `output_message`, `error_code`. | `return TargetResult(...)` |
| `None` | Ambiguous / inconclusive trial. | `return None` |

### Exception Safety

If your agent harness throws an unhandled Python exception during an evaluation (such as a transient connection reset or unexpected syntax error), AgentXRay catches the exception, terminates the search safely, and returns a `DiagnosticResult` with:
- `status = DiagnosticStatus.ERROR`
- `error_message = "<exception details>"`

This prevents test harnesses from crashing or falsely categorizing an uncaught crash as a reproducible agent failure.

---

## 4. `DiagnosticResult` Structured Record

`DiagnosticResult` is a frozen Pydantic model (`ConfigDict(frozen=True, extra="forbid")`) containing high-level ergonomics alongside complete execution telemetry:

```python
class DiagnosticResult(BaseModel):
    status: DiagnosticStatus
    summary: str
    is_verified: bool
    is_one_minimal: bool
    isolated_candidates: tuple[str, ...]
    isolated_candidate_ids: tuple[str, ...]
    candidate_map: dict[str, str]
    initial_candidates_count: int
    evaluations_used: int
    execution_time_seconds: float
    algorithm: str
    seed: int
    matches_ground_truth: bool | None
    error_message: str | None
    trials: tuple[ProbeTrial, ...]
```

### Key Properties

* `status`: Enum indicating search outcome (e.g. `DiagnosticStatus.VERIFIED_CULPRIT`).
* `summary`: Plain-English explanation suitable for logs or developer dashboards.
* `is_verified`: `True` if counterfactual verification confirmed that removing the isolated candidate restored clean success.
* `is_one_minimal`: `True` if removing any individual directive from the isolated subset eliminates the failure.
* `isolated_candidates`: Tuple of raw directive strings identified as the minimal failure cause.
* `evaluations_used`: Total agent evaluation calls consumed during the search.
* `execution_time_seconds`: Total wall-clock duration in seconds.

---

## 5. `DiagnosticStatus` Outcome Enum

```python
class DiagnosticStatus(str, Enum):
    VERIFIED_CULPRIT = "VERIFIED_CULPRIT"
    NO_FAILURE = "NO_FAILURE"
    INCONCLUSIVE = "INCONCLUSIVE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
```

* **`VERIFIED_CULPRIT`**: A minimal failure-inducing subset was isolated and proven via counterfactual verification.
* **`NO_FAILURE`**: The full candidate set passed cleanly on the baseline run; no failure was detected to minimize.
* **`INCONCLUSIVE`**: The evaluator returned inconclusive or 1-minimality could not be confirmed within the candidate space.
* **`BUDGET_EXHAUSTED`**: The `max_evaluations` limit was reached before the search finished.
* **`TIMEOUT`**: Wall-clock duration exceeded `timeout_seconds`.
* **`ERROR`**: The evaluator callable raised an unhandled exception or returned an unsupported type.

---

## 6. Complete Integration Examples

### Example A: Diagnosing Tool Interface Schema Mismatch

```python
from agentxray import diagnose, TargetOutcome, TargetResult

tool_schema_candidates = [
    "param_name_expr: name math parameter 'expr'",
    "param_name_expression: name math parameter 'expression'",
    "param_type_int: restrict operands to integer values",
    "timeout_ms: set tool timeout to 3000ms",
]

def agent_schema_evaluator(active_directives: tuple[str, ...]) -> TargetResult:
    # Backend tool expects parameter 'expression'
    # If 'param_name_expr' is active, the agent uses 'expr' and the API rejects it
    if any("param_name_expr" in d for d in active_directives):
        return TargetResult(
            outcome=TargetOutcome.FAILURE,
            output_message="API rejection: Missing required argument 'expression'",
            error_code="SCHEMA_MISMATCH",
        )
    return TargetResult(
        outcome=TargetOutcome.SUCCESS,
        output_message="Tool executed successfully",
    )

result = diagnose(
    candidates=tool_schema_candidates,
    evaluator=agent_schema_evaluator,
    max_evaluations=25,
)

print(result.summary)
# Output: The task fails when 1 specific candidate directive is active:
# [elem_0] param_name_expr: name math parameter 'expr'
```

### Example B: Safe Integration with Evaluation Budget Caps

```python
from agentxray import diagnose, DiagnosticStatus

candidates = [f"instruction_{i}" for i in range(20)]

def slow_agent_evaluator(subset: tuple[str, ...]) -> bool:
    # Simulate an agent workflow that takes 500ms
    return "instruction_7" in subset

result = diagnose(
    candidates=candidates,
    evaluator=slow_agent_evaluator,
    max_evaluations=15,  # Protect against excessive token spend
    timeout_seconds=10.0,
)

if result.status == DiagnosticStatus.VERIFIED_CULPRIT:
    print(f"Isolated in {result.evaluations_used} calls: {result.isolated_candidates}")
elif result.status == DiagnosticStatus.BUDGET_EXHAUSTED:
    print("Search exceeded budget limit before confirming minimality.")
```
