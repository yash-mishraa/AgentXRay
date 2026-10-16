"""Fake language model provider for hermetic testing and offline validation."""

from __future__ import annotations

import json
import time
from typing import Callable, Sequence

from agentxray.providers.base import (
    Message,
    ModelProvider,
    ModelResponse,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ToolCall,
    ToolDefinition,
    UsageMetadata,
)


class FakeProvider(ModelProvider):
    """Programmable fake model provider for hermetic, offline test execution.

    Supports:
    - Default deterministic tool-calling behavior based on message inspection.
    - Custom response hooks for simulating model failure patterns, stochastic drift, or errors.
    - Complete isolation from external networks and credentials.
    """

    def __init__(
        self,
        model_name: str = "fake-model-v1",
        custom_handler: Callable[[Sequence[Message], Sequence[ToolDefinition]], ModelResponse] | None = None,
        simulate_error: type[Exception] | None = None,
    ) -> None:
        self._model_name = model_name
        self._custom_handler = custom_handler
        self._simulate_error = simulate_error
        self.request_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] = (),
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int | None = None,
    ) -> ModelResponse:
        self.request_count += 1

        if self._simulate_error is not None:
            raise self._simulate_error(f"Simulated error in FakeProvider: {self._simulate_error.__name__}")

        if self._custom_handler is not None:
            res = self._custom_handler(messages, tools)
            self.total_prompt_tokens += res.usage.prompt_tokens
            self.total_completion_tokens += res.usage.completion_tokens
            return res

        # Default intelligent mock behavior:
        # 1. If last message is a tool response, synthesize a final answer
        # 2. Else if tools available and message asks for a calculation or lookup, emit tool call
        # 3. Else emit plain text
        last_msg = messages[-1] if messages else Message(role="user", content="")

        # Compute nominal token counts
        prompt_len = sum(len(m.content) for m in messages) // 4 + 20
        self.total_prompt_tokens += prompt_len

        if last_msg.role == "tool":
            # Synthesize answer from tool output
            content = f"The computed result is {last_msg.content}."
            comp_len = len(content) // 4 + 5
            self.total_completion_tokens += comp_len
            return ModelResponse(
                content=content,
                tool_calls=(),
                usage=UsageMetadata(
                    prompt_tokens=prompt_len,
                    completion_tokens=comp_len,
                    total_tokens=prompt_len + comp_len,
                    estimated_cost_usd=round((prompt_len * 0.15 + comp_len * 0.60) / 1_000_000, 7),
                ),
                finish_reason="stop",
                model_identifier=self._model_name,
                latency_ms=1.5,
            )

        # Inspect messages for task keywords
        full_text = " ".join(m.content for m in messages).lower()

        # Check for conflicting instruction (e.g. "never invoke external tools")
        if "never invoke external tools" in full_text or "do not call functions" in full_text:
            # Model follows prompt constraint and refuses to call tool, attempting direct response
            content = "I cannot use external tools according to instructions. Estimating answer mentally: 999."
            comp_len = len(content) // 4 + 5
            self.total_completion_tokens += comp_len
            return ModelResponse(
                content=content,
                tool_calls=(),
                usage=UsageMetadata(
                    prompt_tokens=prompt_len,
                    completion_tokens=comp_len,
                    total_tokens=prompt_len + comp_len,
                    estimated_cost_usd=round((prompt_len * 0.15 + comp_len * 0.60) / 1_000_000, 7),
                ),
                finish_reason="stop",
                model_identifier=self._model_name,
                latency_ms=2.0,
            )

        # Check if calculator tool is available and calculation requested
        has_calc = any(t.name == "calculator" for t in tools)
        if has_calc and ("calculate" in full_text or "sum" in full_text or "math" in full_text or "+" in full_text or "reimbursement" in full_text):
            # Check for currency formatting directive causing tool syntax failure
            if "currency_formatting" in full_text or "currency symbols" in full_text:
                expr = "$150 + $250"
            elif "200 + 300" in full_text:
                expr = "200 + 300"
            elif "42 * 10" in full_text:
                expr = "42 * 10"
            else:
                expr = "150 + 250"

            tool_calls = (
                ToolCall(
                    call_id="call_mock_calc_1",
                    name="calculator",
                    arguments={"expression": expr},
                ),
            )
            comp_len = 15
            self.total_completion_tokens += comp_len
            return ModelResponse(
                content="",
                tool_calls=tool_calls,
                usage=UsageMetadata(
                    prompt_tokens=prompt_len,
                    completion_tokens=comp_len,
                    total_tokens=prompt_len + comp_len,
                    estimated_cost_usd=round((prompt_len * 0.15 + comp_len * 0.60) / 1_000_000, 7),
                ),
                finish_reason="tool_calls",
                model_identifier=self._model_name,
                latency_ms=2.5,
            )

        # Check for database lookup
        has_db = any(t.name == "database_lookup" for t in tools)
        if has_db and ("lookup" in full_text or "user" in full_text or "record" in full_text):
            target_field = "balance"
            if "override field name to non_existent_secret_key" in full_text:
                target_field = "non_existent_secret_key"

            tool_calls = (
                ToolCall(
                    call_id="call_mock_db_1",
                    name="database_lookup",
                    arguments={"entity": "user_42", "field": target_field},
                ),
            )
            comp_len = 15
            self.total_completion_tokens += comp_len
            return ModelResponse(
                content="",
                tool_calls=tool_calls,
                usage=UsageMetadata(
                    prompt_tokens=prompt_len,
                    completion_tokens=comp_len,
                    total_tokens=prompt_len + comp_len,
                    estimated_cost_usd=round((prompt_len * 0.15 + comp_len * 0.60) / 1_000_000, 7),
                ),
                finish_reason="tool_calls",
                model_identifier=self._model_name,
                latency_ms=2.0,
            )

        # Default standard completion
        content = "Understood. Task processed successfully."
        comp_len = len(content) // 4 + 5
        self.total_completion_tokens += comp_len
        return ModelResponse(
            content=content,
            tool_calls=(),
            usage=UsageMetadata(
                prompt_tokens=prompt_len,
                completion_tokens=comp_len,
                total_tokens=prompt_len + comp_len,
                estimated_cost_usd=round((prompt_len * 0.15 + comp_len * 0.60) / 1_000_000, 7),
            ),
            finish_reason="stop",
            model_identifier=self._model_name,
            latency_ms=1.5,
        )
