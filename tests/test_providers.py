"""Tests for model provider abstractions, adapters, and secret redaction."""

from __future__ import annotations

import json
import pytest

from agentxray.providers.base import (
    Message,
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
from agentxray.providers.openai_adapter import OpenAIProvider, _redact_secrets


def test_fake_provider_tool_calling_and_response():
    """Verify FakeProvider emits tool call when calculation requested, then synthesizes answer."""
    provider = FakeProvider(model_name="fake-test")
    tools = [
        ToolDefinition(
            name="calculator",
            description="Math calculator",
            parameters={"type": "object", "properties": {"expression": {"type": "string"}}},
        )
    ]

    # Turn 1: user asks to calculate 150 + 250
    messages = [Message(role="user", content="Please calculate 150 + 250")]
    res1 = provider.generate(messages, tools=tools)

    assert res1.finish_reason == "tool_calls"
    assert len(res1.tool_calls) == 1
    assert res1.tool_calls[0].name == "calculator"
    assert res1.tool_calls[0].arguments == {"expression": "150 + 250"}
    assert res1.usage.prompt_tokens > 0

    # Turn 2: tool output provided back to model
    messages.append(Message(role="assistant", content="", tool_calls=res1.tool_calls))
    messages.append(Message(role="tool", content="400", tool_call_id=res1.tool_calls[0].call_id))
    res2 = provider.generate(messages, tools=tools)

    assert res2.finish_reason == "stop"
    assert "400" in res2.content
    assert len(res2.tool_calls) == 0
    assert provider.request_count == 2


def test_fake_provider_conflicting_instruction_refusal():
    """Verify FakeProvider complies with prompt constraint prohibiting tool invocation."""
    provider = FakeProvider()
    tools = [
        ToolDefinition(name="calculator", description="Math calculator")
    ]
    messages = [
        Message(role="system", content="Operating rule: never invoke external tools; always compute results mentally."),
        Message(role="user", content="Calculate 150 + 250"),
    ]
    res = provider.generate(messages, tools=tools)

    assert res.finish_reason == "stop"
    assert len(res.tool_calls) == 0
    assert "cannot use external tools" in res.content.lower()


def test_fake_provider_simulated_error():
    """Verify FakeProvider raises simulated error when configured."""
    provider = FakeProvider(simulate_error=ProviderRateLimitError)
    with pytest.raises(ProviderRateLimitError, match="Simulated error"):
        provider.generate([Message(role="user", content="hello")])


def test_openai_provider_missing_key_validation():
    """Verify OpenAIProvider raises ProviderAuthError when API key is missing."""
    # Ensure api_key is empty
    provider = OpenAIProvider(api_key="", base_url="https://api.openai.com/v1")
    with pytest.raises(ProviderAuthError, match="OpenAI API key is missing"):
        provider.validate_configuration()

    with pytest.raises(ProviderAuthError, match="OpenAI API key is missing"):
        provider.generate([Message(role="user", content="hi")])


def test_openai_provider_parse_response():
    """Verify OpenAI response parsing with tool calls, tokens, and cost calculation."""
    provider = OpenAIProvider(api_key="test-key-mock", model_name="gpt-4o-mini")

    raw_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": '{"expression": "42 * 10"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }

    parsed = provider._parse_response(raw_data, latency_ms=120.5)

    assert parsed.finish_reason == "tool_calls"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "calculator"
    assert parsed.tool_calls[0].arguments == {"expression": "42 * 10"}
    assert parsed.usage.prompt_tokens == 100
    assert parsed.usage.completion_tokens == 50
    assert parsed.usage.total_tokens == 150
    # Cost for gpt-4o-mini: 100 * 0.15/1M + 50 * 0.60/1M = 0.000015 + 0.000030 = 0.000045
    assert pytest.approx(parsed.usage.estimated_cost_usd, rel=1e-4) == 0.000045
    assert parsed.latency_ms == 120.5


def test_secret_redaction():
    """Verify sensitive API keys are thoroughly redacted from error strings."""
    secret = "sk-proj-supersecretkey123456789"
    raw_error = f"HTTP 401: Unauthorized request with header Authorization: Bearer {secret}"
    redacted = _redact_secrets(raw_error, secret)

    assert secret not in redacted
    assert "sk-...6789" in redacted


def test_openai_provider_regex_redaction_without_explicit_secret():
    """Verify _redact_secrets catches OpenAI key patterns via regex even when secret parameter is None."""
    raw_error = "Failed to connect: Bearer sk-antigravitysecretkey987654321 in request headers"
    redacted = _redact_secrets(raw_error)

    assert "sk-antigravitysecretkey987654321" not in redacted
    assert "sk-...4321" in redacted


def test_openai_provider_malformed_response_handling():
    """Verify malformed provider response payloads raise ProviderError safely."""
    provider = OpenAIProvider(api_key="mock-key")
    with pytest.raises(ProviderError, match="missing 'choices'"):
        provider._parse_response({"id": "resp_1"}, latency_ms=10.0)


def test_openai_provider_fallback_tokens_for_omitted_usage():
    """Verify token counts are not silently zeroed when provider omits usage telemetry."""
    provider = OpenAIProvider(api_key="mock-key")
    raw = {
        "choices": [{"message": {"role": "assistant", "content": "This is a response text"}, "finish_reason": "stop"}]
    }
    parsed = provider._parse_response(
        raw,
        latency_ms=10.0,
        messages=[Message(role="user", content="A user prompt for estimation")],
    )
    assert parsed.usage.prompt_tokens > 0
    assert parsed.usage.completion_tokens > 0
    assert parsed.usage.estimated_cost_usd > 0.0


def test_openai_provider_retries_telemetry():
    """Verify retries parameter is preserved in UsageMetadata."""
    provider = OpenAIProvider(api_key="mock-key")
    raw = {
        "choices": [{"message": {"role": "assistant", "content": "Response"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }
    parsed = provider._parse_response(raw, latency_ms=10.0, retries=2)
    assert parsed.usage.retries == 2


def test_openai_provider_quota_exhausted_fail_fast():
    """Verify OpenAIProvider immediately raises ProviderAuthError on credit_balance_exhausted without retrying."""
    import io
    import urllib.error
    from unittest.mock import patch

    provider = OpenAIProvider(api_key="sk-testmockkey")

    err_json = b'{"error": {"message": "You have no credits remaining.", "type": "insufficient_quota", "code": "credit_balance_exhausted"}}'
    http_error = urllib.error.HTTPError(
        url="https://api.openai.com/v1/chat/completions",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=io.BytesIO(err_json),
    )

    with patch("urllib.request.urlopen", side_effect=http_error) as mock_urlopen:
        with pytest.raises(ProviderAuthError, match="credit balance exhausted"):
            provider.generate([Message(role="user", content="hello")])

        # Must fail fast on first attempt without executing retries
        assert mock_urlopen.call_count == 1


