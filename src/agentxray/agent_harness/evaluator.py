"""Observable outcome evaluators for model-backed agent experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentxray.agent_harness.agent import AgentExecutionTrace


class AgentOutcomeEvaluator(ABC):
    """Evaluates agent execution based purely on observable interaction traces.

    CRITICAL EPISTEMIC PRINCIPLE:
    Evaluators must judge success or failure using ONLY observable runtime behavior
    (tool calls, arguments, outputs, final responses). They must never inspect
    planted ground-truth causal rules.
    """

    @abstractmethod
    def evaluate(self, trace: AgentExecutionTrace) -> tuple[bool, str]:
        """Evaluate an execution trace.

        Returns
        -------
        tuple[bool, str]
            (is_success, diagnostic_message)
        """


class ToolCallingTaskEvaluator(AgentOutcomeEvaluator):
    """Evaluates whether an agent correctly invoked a required tool and produced the expected answer."""

    def __init__(
        self,
        required_tool: str | None = None,
        expected_substring: str | None = None,
        forbidden_substrings: tuple[str, ...] = (),
    ) -> None:
        self.required_tool = required_tool
        self.expected_substring = expected_substring.lower() if expected_substring else None
        self.forbidden_substrings = tuple(s.lower() for s in forbidden_substrings)

    def evaluate(self, trace: AgentExecutionTrace) -> tuple[bool, str]:
        # 1. Check for system or provider errors
        if trace.error_message:
            return (False, f"Agent run failed with error: {trace.error_message}")

        if trace.finish_reason not in ("stop", "tool_calls"):
            return (False, f"Abnormal finish reason: {trace.finish_reason}")

        # 2. Check required tool invocation
        if self.required_tool:
            tools_called = [t.tool_name for t in trace.tool_executions]
            if self.required_tool not in tools_called:
                return (
                    False,
                    f"Agent failed to invoke required tool '{self.required_tool}'. Tools invoked: {tools_called}",
                )

            # Check if any required tool call returned an error
            for t in trace.tool_executions:
                if t.tool_name == self.required_tool and t.output.startswith("Error:"):
                    return (False, f"Required tool '{self.required_tool}' returned an error: {t.output}")

        # 3. Check final response content
        resp_lower = trace.final_response.lower()

        # Check forbidden substrings (e.g. refusal or hallucinated estimate indicators)
        for forb in self.forbidden_substrings:
            if forb in resp_lower:
                return (False, f"Final response contained forbidden phrase: '{forb}'")

        if self.expected_substring:
            if self.expected_substring not in resp_lower:
                return (
                    False,
                    f"Final response did not contain expected content '{self.expected_substring}'. Got: '{trace.final_response}'",
                )

        return (True, "Task executed successfully; observable criteria satisfied.")
