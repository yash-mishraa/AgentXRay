"""Model provider interfaces and adapters for AgentXRay."""

from agentxray.providers.base import (
    Message,
    ModelProvider,
    ModelResponse,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ToolCall,
    ToolDefinition,
    UsageMetadata,
)
from agentxray.providers.fake import FakeProvider
from agentxray.providers.openai_adapter import OpenAIProvider

__all__ = [
    "Message",
    "ModelProvider",
    "ModelResponse",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ToolCall",
    "ToolDefinition",
    "UsageMetadata",
    "FakeProvider",
    "OpenAIProvider",
]
