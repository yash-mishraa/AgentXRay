"""Runnable offline example: Diagnosing a tool-calling failure with AgentXRay Python SDK.

This script demonstrates how an external developer can use `agentxray.diagnose`
to isolate the minimal prompt directive causing a tool syntax crash.
No API keys or live LLM network requests required ($0.00 spend).
"""

from __future__ import annotations

from agentxray import DiagnosticStatus, diagnose


def main() -> None:
    # 1. Candidate prompt directives / system rules in the agent's configuration
    candidates = [
        "strict_currency: format all numbers within arithmetic arguments with currency symbols (e.g. '$150').",
        "concise_summary: keep final output under 50 words.",
        "markdown_tables: render financial summaries in markdown tables.",
        "formal_tone: use formal business accounting terminology.",
    ]

    # 2. Define custom evaluator representing the agent + environment
    # Receives the currently active subset of directives as a tuple of strings.
    # Return True if the failure reproduces, False if the agent succeeds.
    def my_agent_evaluator(active_directives: tuple[str, ...]) -> bool:
        # Simulate agent execution: if the currency constraint is active,
        # the agent emits '$150 + $250' which causes a calculator tool syntax crash.
        for directive in active_directives:
            if "strict_currency" in directive:
                return True  # Failure reproduced (tool syntax error)
        return False         # Clean pass (tool executes successfully)

    print("Running AgentXRay causal failure diagnosis...\n")

    # 3. Execute black-box causal diagnosis
    result = diagnose(
        candidates=candidates,
        evaluator=my_agent_evaluator,
        max_evaluations=20,
        seed=42,
    )

    # 4. Inspect results
    print(f"Status:            {result.status.value}")
    print(f"Verified 1-Minimal: {result.is_verified}")
    print(f"Evaluations Used:  {result.evaluations_used}")
    print(f"Isolated Culprit:  {result.isolated_candidates}")
    print(f"Candidate IDs:     {result.isolated_candidate_ids}")
    print(f"Summary:           {result.summary}")

    assert result.status == DiagnosticStatus.VERIFIED_CULPRIT
    assert result.is_verified is True
    assert len(result.isolated_candidates) == 1
    assert "strict_currency" in result.isolated_candidates[0]
    print("\n✓ Diagnosis completed and verified successfully.")


if __name__ == "__main__":
    main()
