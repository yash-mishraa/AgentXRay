"""Controlled agent execution loop for model-backed experiments."""

from __future__ import annotations

import inspect
import time
import unittest.mock
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
    UnsupportedProviderError,
)
from agentxray.providers.fake import FakeProvider
from agentxray.providers.ollama_adapter import OllamaProvider
from agentxray.providers.openai_adapter import OpenAIProvider

TRUSTED_BUILTIN_PROVIDER_CLASSES: tuple[type, ...] = (
    OpenAIProvider,
    OllamaProvider,
    FakeProvider,
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


def check_provider_retry_support(provider: ModelProvider, strict_budget: bool = True) -> tuple[bool, str]:
    """Verify whether provider can accept and honor effective retry allowances before dispatch.

    Parameters
    ----------
    provider : ModelProvider
        The provider instance to evaluate.
    strict_budget : bool
        Whether strict request-budget enforcement is active.

    Returns
    -------
    tuple[bool, str]
        (is_supported, rejection_reason)
    """
    # 1. Trust policy for strict budget mode:
    # Under strict request budgets, only exact instances of trusted built-in providers are permitted.
    # We enforce exact type identity (type(provider) in TRUSTED_BUILTIN_PROVIDER_CLASSES), NOT isinstance,
    # so arbitrary custom subclasses, proxies, wrappers, and mocks are rejected before dispatch.
    #
    # Threat-Model Boundary & In-Process Scope Limitation:
    # - Strict-budget provider validation accepts exact instances of approved built-in provider classes.
    # - This protects against unsupported custom provider classes, subclasses, wrappers, proxies, and mocks.
    # - It is NOT a sandbox against arbitrary code executing inside the same Python process; such code may
    #   monkeypatch an approved instance (e.g., instance-level `provider.generate = ...`) or mutate class
    #   internals. The gate enforces public API configuration safety across provider classes, but does not
    #   detect or prevent in-process dynamic monkeypatching on approved instances.
    if strict_budget:
        if type(provider) not in TRUSTED_BUILTIN_PROVIDER_CLASSES:
            prov_type = type(provider).__name__
            prov_mod = getattr(type(provider), "__module__", "unknown")
            trusted_names = tuple(cls.__name__ for cls in TRUSTED_BUILTIN_PROVIDER_CLASSES)
            return False, (
                f"Unsupported provider: '{prov_type}' (module '{prov_mod}') is not an approved "
                f"built-in provider class. Under strict request-budget enforcement, only exact instances of "
                f"{trusted_names} are permitted. Custom subclasses, proxies, wrappers, and mocks cannot "
                f"guarantee retry limiting or prevent hidden internal retries."
            )

    # 2. Base capability check
    if getattr(provider, "supports_retry_limits", False) is not True:
        p_name = getattr(provider, "provider_name", type(provider).__name__)
        return False, (
            f"Unsupported provider: '{p_name}' does not declare supports_retry_limits=True. "
            f"In bounded experiments, providers that cannot guarantee retry limiting are rejected before dispatch."
        )

    # 3. Explicit signature parameter check (kwargs alone is not sufficient)
    try:
        sig = inspect.signature(provider.generate)
        params = sig.parameters
        if "max_retries" not in params:
            p_name = getattr(provider, "provider_name", type(provider).__name__)
            return False, (
                f"Unsupported provider: '{p_name}' lacks an explicit 'max_retries' parameter in its "
                f"generate() method signature (accepting **kwargs alone does not guarantee retry limit enforcement)."
            )
    except (ValueError, TypeError) as sig_err:
        p_name = getattr(provider, "provider_name", type(provider).__name__)
        return False, f"Unsupported provider: Unable to inspect generate() signature of '{p_name}': {sig_err}"

    return True, ""


class ControlledAgent:
    """Manages the interaction loop between a ModelProvider and MockToolEnvironment.

    Safety & Retry Boundary Guarantee:
    - Pre-dispatch enforcement: Before every model call, the agent caps allowed retries
      at min(provider_retries, max(0, remaining_headroom - 1)).
    - Incompatible provider protection: Providers that cannot accept or guarantee retry
      limits are rejected before dispatch with an error, preventing unconstrained retries.
    - Trust Boundary & Threat-Model Limitation Notice:
      The harness bounds retries for the unmodified built-in provider implementations
      (OpenAIProvider, OllamaProvider, FakeProvider) according to remaining request headroom.
      For arbitrary third-party providers that accept max_retries but execute unobservable internal retry
      loops, the harness cannot inspect or constrain opaque internal code before dispatch. Therefore,
      strict-budget provider validation accepts only exact instances of approved built-in provider classes,
      protecting against unsupported custom classes, subclasses, wrappers, proxies, and mocks.
      This is not a sandbox against arbitrary code executing within the same Python process; such code can
      monkeypatch approved instances (e.g. instance-level `provider.generate = ...`) or runtime internals.
      The gate enforces public API configuration and class provenance safety rather than in-process tamper resistance.
    """

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
        if hasattr(cost_tracker, "bind_provider"):
            cost_tracker.bind_provider(provider)
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
                self.cost_tracker.check_budget_before_request(est_prompt_tokens=est_prompt, provider=self.provider)
            except CostBudgetExceededError as budget_exc:
                error_msg = str(budget_exc)
                finish_reason = "budget_exhausted"
                break

            # Bound allowed retries by remaining request headroom and provider limits to prevent budget breach
            remaining_requests = getattr(
                self.cost_tracker,
                "remaining_request_headroom",
                self.cost_tracker.config.max_model_requests - self.cost_tracker.total_requests,
            )
            if remaining_requests <= 0:
                error_msg = "Model request budget exhausted before request: 0 remaining requests."
                finish_reason = "budget_exhausted"
                break

            headroom_retry_cap = max(0, remaining_requests - 1)
            provider_max_retries = getattr(self.provider, "max_retries", getattr(self.provider, "_max_retries", None))
            if isinstance(provider_max_retries, int) and provider_max_retries >= 0:
                allowed_retries = min(provider_max_retries, headroom_retry_cap)
            else:
                allowed_retries = headroom_retry_cap

            # Fail safely before dispatch if provider cannot accept or guarantee retry limits
            is_supported, reject_msg = check_provider_retry_support(self.provider, strict_budget=True)
            if not is_supported:
                error_msg = reject_msg
                finish_reason = "error"
                break

            t0 = time.perf_counter()
            try:
                try:
                    response: ModelResponse = self.provider.generate(
                        messages=messages,
                        tools=tools,
                        temperature=0.0,
                        max_retries=allowed_retries,
                    )
                except TypeError as type_err:
                    # Distinguish call-site argument-binding rejection from internal provider TypeError:
                    tb_next = type_err.__traceback__.tb_next if type_err.__traceback__ is not None else None
                    is_callsite_binding = (
                        tb_next is None
                        and ("max_retries" in str(type_err) or "unexpected keyword argument" in str(type_err))
                    )
                    if is_callsite_binding:
                        raise UnsupportedProviderError(
                            f"Provider '{self.provider.provider_name}' rejected retry limit argument "
                            f"(max_retries={allowed_retries}): {type_err}. Unconstrained calls are rejected."
                        ) from type_err
                    else:
                        raise ProviderError(
                            f"Provider '{self.provider.provider_name}' encountered internal TypeError during generation: {type_err}"
                        ) from type_err
            except Exception as exc:
                error_msg = f"Provider failure at turn {turns}: {exc}"
                finish_reason = "error"
                retries = getattr(exc, "retries", 0)
                try:
                    self.cost_tracker.record_failed_request(retries=retries)
                except CostBudgetExceededError:
                    pass
                break

            turn_latency = (time.perf_counter() - t0) * 1000.0
            total_latency += turn_latency
            session_tokens += response.usage.total_tokens
            session_cost += response.usage.estimated_cost_usd

            # Record in cost tracker (which will raise CostBudgetExceededError if budget breached)
            budget_exc_caught: CostBudgetExceededError | None = None
            try:
                self.cost_tracker.record_usage(
                    response.usage,
                    success=True,
                    retries=response.usage.retries,
                    provider=self.provider,
                )
            except CostBudgetExceededError as budget_exc:
                budget_exc_caught = budget_exc

            # If CostBudgetExceededError was raised, only allow completion if spend is intact,
            # requests did not strictly exceed (> cap), and no tool calls require another turn.
            if budget_exc_caught is not None and (
                getattr(self.cost_tracker, "is_spend_exceeded", False)
                or getattr(self.cost_tracker, "is_request_cap_exceeded", False)
                or bool(response.tool_calls)
            ):
                error_msg = str(budget_exc_caught)
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
