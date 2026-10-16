"""Controlled agent execution loop for model-backed experiments."""

from __future__ import annotations

import time
from typing import Any, Sequence
from pydantic import BaseModel, ConfigDict, Field

from agentxray.agent_harness.cost_tracker import CostTracker
from agentxray.agent_harness.tools import MockToolEnvironment
from agentxray.providers.base import (
    Message,
    ModelProvider,
    ModelResponse,
    ProviderError,
    ToolCall,
    ToolDefinition,
)


class ToolExecutionRecord(BaseModel):
    """Immutable record of an executed tool invocation."""

    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    output: str
    latency_ms: float = 0.0


class AgentExecutionTrace(BaseModel):
    """Complete, structured record of a model-backed agent session."""

    model_config = ConfigDict(frozen=True)

    messages: tuple[Message, ...] = Field(default=(), description="Full conversation history.")
    final_response: str = Field(default="", description="Final textual answer emitted by agent.")
    tool_executions: tuple[ToolExecutionRecord, ...] = Field(
        default=(), description="Tool calls executed in mock environment."
    )
    turns_used: int = Field(default=0, ge=0, description="Model generation turns consumed.")
    total_latency_ms: float = Field(default=0.0, ge=0.0, description="Summed execution latency.")
    total_tokens: int = Field(default=0, ge=0, description="Tokens consumed during session.")
    estimated_cost_usd: float = Field(default=0.0, ge=0.0, description="Estimated USD cost.")
    finish_reason: str = Field(default="stop", description="Terminal state reason.")
    error_message: str | None = Field(default=None, description="Diagnostic error if failed.")


class ControlledAgent:
    """Manages the interaction loop between a ModelProvider and MockToolEnvironment."""

    def __init__(
        self,
        provider: ModelProvider,
        mock_env: MockToolEnvironment,
        cost_tracker: CostTracker,
        max_turns: int = 4,
    ) -> None:
        self.provider = provider
        self.mock_env = mock_env
        self.cost_tracker = cost_tracker
        self.max_turns = max_turns

    def run_session(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: Sequence[ToolDefinition] = (),
    ) -> AgentExecutionTrace:
        """Execute a controlled agent interaction up to max_turns."""
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=user_prompt))

        tool_execs: list[ToolExecutionRecord] = []
        turns = 0
        total_latency = 0.0
        session_tokens = 0
        session_cost = 0.0
        error_msg: str | None = None
        finish_reason = "stop"

        from agentxray.agent_harness.cost_tracker import CostBudgetExceededError

        while turns < self.max_turns:
            turns += 1

            # Pre-flight budget check before initiating request
            est_prompt = sum(len(m.content) for m in messages) // 4 + 20
            try:
                self.cost_tracker.check_budget_before_request(est_prompt_tokens=est_prompt)
            except CostBudgetExceededError as budget_exc:
                error_msg = str(budget_exc)
                finish_reason = "budget_exhausted"
                break

            t0 = time.perf_counter()
            try:
                response: ModelResponse = self.provider.generate(
                    messages=messages,
                    tools=tools,
                    temperature=0.0,
                )
            except Exception as exc:
                error_msg = f"Provider failure at turn {turns}: {exc}"
                finish_reason = "error"
                try:
                    self.cost_tracker.record_failed_request()
                except CostBudgetExceededError:
                    pass
                break

            turn_latency = (time.perf_counter() - t0) * 1000.0
            total_latency += turn_latency
            session_tokens += response.usage.total_tokens
            session_cost += response.usage.estimated_cost_usd

            # Record in cost tracker (which will raise CostBudgetExceededError if budget breached)
            try:
                self.cost_tracker.record_usage(
                    response.usage,
                    success=True,
                    retries=response.usage.retries,
                )
            except CostBudgetExceededError as budget_exc:
                error_msg = str(budget_exc)
                finish_reason = "budget_exhausted"
                break

            # Check if model emitted tool calls
            if response.tool_calls:
                messages.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                for tc in response.tool_calls:
                    t_tool = time.perf_counter()
                    out = self.mock_env.execute(tc.name, tc.arguments)
                    tool_lat = (time.perf_counter() - t_tool) * 1000.0

                    rec = ToolExecutionRecord(
                        tool_call_id=tc.call_id,
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        output=out,
                        latency_ms=round(tool_lat, 2),
                    )
                    tool_execs.append(rec)
                    messages.append(
                        Message(
                            role="tool",
                            content=out,
                            tool_call_id=tc.call_id,
                        )
                    )
                # Next turn for assistant to incorporate tool output
                continue

            # Model emitted final response without tool calls
            messages.append(Message(role="assistant", content=response.content))
            return AgentExecutionTrace(
                messages=tuple(messages),
                final_response=response.content,
                tool_executions=tuple(tool_execs),
                turns_used=turns,
                total_latency_ms=round(total_latency, 2),
                total_tokens=session_tokens,
                estimated_cost_usd=round(session_cost, 7),
                finish_reason=response.finish_reason,
                error_message=None,
            )

        # Exceeded turns or loop halted
        if not error_msg and turns >= self.max_turns:
            finish_reason = "max_turns_exceeded"
            error_msg = f"Agent failed to reach final answer within {self.max_turns} turns."

        return AgentExecutionTrace(
            messages=tuple(messages),
            final_response="",
            tool_executions=tuple(tool_execs),
            turns_used=turns,
            total_latency_ms=round(total_latency, 2),
            total_tokens=session_tokens,
            estimated_cost_usd=round(session_cost, 7),
            finish_reason=finish_reason,
            error_message=error_msg,
        )
