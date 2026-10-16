"""Model-backed agent target adapter adhering to the TargetResult execution protocol."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from agentxray.agent_harness.agent import AgentExecutionTrace, ControlledAgent
from agentxray.agent_harness.evaluator import AgentOutcomeEvaluator
from agentxray.models import InputCase, TargetOutcome, TargetResult
from agentxray.providers.base import ToolDefinition


class ModelBackedAgentTarget:
    """Target adapter wrapping a ControlledAgent and AgentOutcomeEvaluator into AgentXRay's execution interface.

    Allows the DeltaDebugMinimizer, InterventionRunner, and Baselines to probe
    and minimize model-backed agent failure conditions without any coupling to provider internals.
    """

    def __init__(
        self,
        agent: ControlledAgent,
        evaluator: AgentOutcomeEvaluator,
        user_prompt: str,
        base_system_prompt: str = "",
        tools: Sequence[ToolDefinition] = (),
        name: str = "ModelBackedAgentTarget",
        prompt_builder: Callable[[InputCase, str, str], tuple[str, str]] | None = None,
    ) -> None:
        self.agent = agent
        self.evaluator = evaluator
        self.user_prompt = user_prompt
        self.base_system_prompt = base_system_prompt
        self.tools = tuple(tools)
        self.name = name
        self._prompt_builder = prompt_builder
        self._call_count = 0
        self._last_trace: AgentExecutionTrace | None = None

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def last_trace(self) -> AgentExecutionTrace | None:
        return self._last_trace

    def reset_call_count(self) -> None:
        self._call_count = 0
        self._last_trace = None

    def _compose_prompts(self, input_case: InputCase) -> tuple[str, str]:
        """Compose system and user prompts incorporating the input_case elements."""
        if self._prompt_builder is not None:
            return self._prompt_builder(input_case, self.base_system_prompt, self.user_prompt)

        # Default composition: concatenate input_case elements into system instructions
        elements_text = "\n".join(f"- {e}" for e in input_case.elements)
        system = f"{self.base_system_prompt}\n\nOperating Guidelines & Constraints:\n{elements_text}".strip()
        user = self.user_prompt
        return system, user

    def execute(self, input_case: InputCase) -> TargetResult:
        """Execute a controlled agent trial against the candidate InputCase.

        Parameters
        ----------
        input_case : InputCase
            Candidate input subset configuration.

        Returns
        -------
        TargetResult
            Observable outcome (SUCCESS or FAILURE) and telemetry.
        """
        self._call_count += 1
        system_prompt, user_prompt = self._compose_prompts(input_case)

        trace = self.agent.run_session(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=self.tools,
        )
        self._last_trace = trace

        # Immediately halt on hard budget breach
        if trace.finish_reason == "budget_exhausted":
            from agentxray.agent_harness.cost_tracker import CostBudgetExceededError
            raise CostBudgetExceededError(trace.error_message or "Model request or spend budget exhausted.")

        is_success, msg = self.evaluator.evaluate(trace)

        outcome = TargetOutcome.SUCCESS if is_success else TargetOutcome.FAILURE
        metadata: dict[str, Any] = {
            "model_identifier": self.agent.provider.model_name,
            "turns_used": trace.turns_used,
            "total_tokens": trace.total_tokens,
            "estimated_cost_usd": trace.estimated_cost_usd,
            "tool_calls_count": len(trace.tool_executions),
            "tool_names_called": [t.tool_name for t in trace.tool_executions],
            "finish_reason": trace.finish_reason,
            "final_response": trace.final_response[:200] if trace.final_response else "",
        }
        if trace.error_message:
            metadata["error"] = trace.error_message

        return TargetResult(
            outcome=outcome,
            output_message=msg,
            error_code="AGENT_FAILURE" if not is_success else None,
            latency_ms=trace.total_latency_ms,
            metadata=metadata,
        )
