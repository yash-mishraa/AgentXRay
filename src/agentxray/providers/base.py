"""Model provider abstractions and message data structures for AgentXRay."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence
from pydantic import BaseModel, ConfigDict, Field


class ToolDefinition(BaseModel):
    """Schema definition for a mock tool exposed to the language model."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Tool function name.")
    description: str = Field(..., description="Description of the tool purpose and behavior.")
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []},
        description="JSON schema for tool parameters.",
    )


class ToolCall(BaseModel):
    """Invocation request emitted by the model."""

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(..., description="Unique tool call identifier.")
    name: str = Field(..., description="Target tool name.")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Parsed JSON arguments.")


class Message(BaseModel):
    """Chat message exchanged in the agent harness."""

    model_config = ConfigDict(frozen=True)

    role: str = Field(..., description="Role: system, user, assistant, or tool.")
    content: str = Field(default="", description="Text content.")
    tool_calls: tuple[ToolCall, ...] = Field(default=(), description="Tool calls emitted by assistant.")
    tool_call_id: str | None = Field(default=None, description="ID of corresponding tool call if role is tool.")


class UsageMetadata(BaseModel):
    """Token consumption and accounting telemetry."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    retries: int = Field(default=0, ge=0, description="Retry count made for request.")


class ModelResponse(BaseModel):
    """Normalized response returned by a ModelProvider."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(default="", description="Text content returned by model.")
    tool_calls: tuple[ToolCall, ...] = Field(default=(), description="Tool calls emitted.")
    usage: UsageMetadata = Field(default_factory=UsageMetadata, description="Token usage telemetry.")
    finish_reason: str = Field(default="stop", description="Finish reason (stop, tool_calls, length).")
    model_identifier: str = Field(..., description="Identifier of the model used.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Latency in milliseconds.")


class ProviderError(Exception):
    """Base exception for provider communication or parsing failures."""

    def __init__(self, message: str, retries: int = 0) -> None:
        super().__init__(message)
        self.retries = retries


class ProviderRateLimitError(ProviderError):
    """Raised when the provider rejects a request due to rate limits."""


class ProviderAuthError(ProviderError):
    """Raised when provider credentials are missing or invalid."""


class ProviderTimeoutError(ProviderError):
    """Raised when provider communication times out."""


class UnsupportedProviderError(ProviderError):
    """Raised when a provider implementation cannot enforce required execution constraints (e.g. retry limits)."""


TRUSTED_BUILTIN_PROVIDERS: frozenset[str] = frozenset({"openai", "ollama", "fake"})


class ModelProvider(ABC):
    """Abstract interface for language model backends."""

    @property
    def supports_retry_limits(self) -> bool:
        """Whether the provider accepts and guarantees pre-invocation retry limiting.

        Defaults to False on the base class for safety. Arbitrary custom subclasses
        must not be presumed retry-safe through inheritance alone.
        """
        return False

    @property
    def is_trusted_for_hard_budgets(self) -> bool:
        """Whether this provider is verified to honor retry limits and hard request budgets.

        Only built-in adapters whose internal retry mechanisms have been verified
        (OpenAIProvider, OllamaProvider, FakeProvider) or verified test fixtures
        return True. Arbitrary custom subclasses default to False.
        """
        return False

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider implementation."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier."""

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] = (),
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int | None = None,
    ) -> ModelResponse:
        """Generate a response from the model.

        Parameters
        ----------
        messages : Sequence[Message]
            Conversation history.
        tools : Sequence[ToolDefinition]
            Available mock tools.
        temperature : float
            Sampling temperature (default 0.0 for deterministic evaluation).
        max_tokens : int
            Generation token cap.
        max_retries : int | None
            Maximum number of transient network/rate-limit retry attempts allowed
            for this invocation. Built-in providers (OpenAI, Ollama, Fake) strictly respect
            this parameter and clamp internal retry loops to min(self._max_retries, max_retries).

            Note on Custom Providers: Under strict request budgets, AgentXRay requires
            providers to be trusted and verified to honor retry limits. Arbitrary custom
            providers that do not support retry limits or whose internal attempts cannot be
            constrained fail closed before dispatch with UnsupportedProviderError. AgentXRay
            cannot inspect or guarantee hard caps for arbitrary third-party provider implementations
            that execute unobservable internal retry loops.

        Returns
        -------
        ModelResponse
            Normalized model output and usage metadata.
        """
