"""Tests for the OllamaProvider adapter, error handling, tool calling, and local integration."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentxray.agent_harness.agent import ControlledAgent
from agentxray.agent_harness.cost_tracker import CostTracker, ExperimentBudgetConfig
from agentxray.agent_harness.tools import CALCULATOR_TOOL_DEF, MockToolEnvironment
from agentxray.models import ExecutionTier
from agentxray.providers.base import (
    Message,
    ModelProvider,
    ModelResponse,
    ProviderError,
    ProviderTimeoutError,
    ToolCall,
    ToolDefinition,
)
from agentxray.providers.ollama_adapter import OllamaProvider, _redact_secrets
from benchmarks.model_pilot.runner import dry_run_model_pilot


class MockHTTPResponse:
    """Helper to mock urllib.request.urlopen context manager."""

    def __init__(self, data: dict[str, Any] | str, status: int = 200) -> None:
        if isinstance(data, dict):
            self._bytes = json.dumps(data).encode("utf-8")
        else:
            self._bytes = data.encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._bytes

    def __enter__(self) -> MockHTTPResponse:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


def test_ollama_provider_contract_and_attributes() -> None:
    """Verify OllamaProvider adheres to the ModelProvider interface."""
    provider = OllamaProvider(model_name="llama3.2:3b", base_url="http://127.0.0.1:11434")
    assert isinstance(provider, ModelProvider)
    assert provider.provider_name == "ollama"
    assert provider.model_name == "llama3.2:3b"
    assert provider.base_url == "http://127.0.0.1:11434"
    provider.validate_configuration()


def test_safety_loopback_restriction() -> None:
    """Ensure OllamaProvider strictly rejects non-loopback endpoints."""
    with pytest.raises(ValueError, match="Safety violation: OllamaProvider must be bound strictly"):
        OllamaProvider(model_name="llama3.2:3b", base_url="https://api.external-host.com")

    with pytest.raises(ValueError, match="Safety violation: OllamaProvider must be bound strictly"):
        OllamaProvider(model_name="llama3.2:3b", base_url="http://192.168.1.100:11434")

    # Valid loopback variations must succeed
    p1 = OllamaProvider(model_name="llama3.2:3b", base_url="http://127.0.0.1:11434")
    p2 = OllamaProvider(model_name="llama3.2:3b", base_url="http://localhost:11434")
    assert p1.base_url == "http://127.0.0.1:11434"
    assert p2.base_url == "http://localhost:11434"


def test_empty_model_name_fails_validation() -> None:
    """Ensure empty model name raises ProviderError on validation."""
    provider = OllamaProvider(model_name="")
    with pytest.raises(ProviderError, match="Model name cannot be empty"):
        provider.validate_configuration()


def test_successful_assistant_text_response() -> None:
    """Test standard assistant text response parsing with token telemetry."""
    provider = OllamaProvider(model_name="llama3.2:3b")
    mock_payload = {
        "model": "llama3.2:3b",
        "message": {
            "role": "assistant",
            "content": "The sum of 150 and 250 is 400.",
        },
        "done": True,
        "prompt_eval_count": 42,
        "eval_count": 18,
    }

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(mock_payload)):
        resp = provider.generate(
            messages=[Message(role="user", content="What is 150 + 250?")],
            temperature=0.0,
            max_tokens=256,
        )

    assert isinstance(resp, ModelResponse)
    assert resp.content == "The sum of 150 and 250 is 400."
    assert resp.tool_calls == ()
    assert resp.finish_reason == "stop"
    assert resp.usage.prompt_tokens == 42
    assert resp.usage.completion_tokens == 18
    assert resp.usage.total_tokens == 60
    assert resp.usage.estimated_cost_usd == 0.0  # Zero financial cost for local inference
    assert resp.model_identifier == "llama3.2:3b"


def test_successful_tool_call_dict_arguments() -> None:
    """Test tool call parsing when arguments are structured as a dictionary."""
    provider = OllamaProvider(model_name="llama3.2:3b")
    mock_payload = {
        "model": "llama3.2:3b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_calc_01",
                    "function": {
                        "name": "calculator",
                        "arguments": {"expression": "150 + 250"},
                    },
                }
            ],
        },
        "done": True,
        "prompt_eval_count": 30,
        "eval_count": 15,
    }

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(mock_payload)):
        resp = provider.generate(
            messages=[Message(role="user", content="Calculate 150 + 250")],
            tools=[CALCULATOR_TOOL_DEF],
        )

    assert resp.finish_reason == "tool_calls"
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.call_id == "call_calc_01"
    assert tc.name == "calculator"
    assert tc.arguments == {"expression": "150 + 250"}
    assert resp.usage.estimated_cost_usd == 0.0


def test_successful_tool_call_json_string_arguments() -> None:
    """Test tool call parsing when arguments are passed as a JSON-encoded string."""
    provider = OllamaProvider(model_name="llama3.2:3b")
    mock_payload = {
        "model": "llama3.2:3b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "150 + 250"}',
                    },
                }
            ],
        },
        "done": True,
    }

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(mock_payload)):
        resp = provider.generate(
            messages=[Message(role="user", content="Calculate 150 + 250")],
            tools=[CALCULATOR_TOOL_DEF],
        )

    assert resp.finish_reason == "tool_calls"
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.call_id == "call_ollama_0"  # Fallback generated ID
    assert tc.name == "calculator"
    assert tc.arguments == {"expression": "150 + 250"}


@pytest.mark.parametrize(
    "bad_tool_payload,expected_err",
    [
        (
            {"model": "m", "message": {"role": "assistant", "tool_calls": "not-a-list"}},
            "Unsupported tool-call format: 'tool_calls' must be a list",
        ),
        (
            {"model": "m", "message": {"role": "assistant", "tool_calls": ["string-instead-of-dict"]}},
            "Unsupported tool-call item format: expected dict",
        ),
        (
            {"model": "m", "message": {"role": "assistant", "tool_calls": [{"no_function": {}}]}},
            "missing or invalid 'function' dictionary",
        ),
        (
            {"model": "m", "message": {"role": "assistant", "tool_calls": [{"function": {"name": ""}}]}},
            "invalid or missing function name",
        ),
        (
            {
                "model": "m",
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "calculator", "arguments": 12345}}],
                },
            },
            "Unsupported tool-call arguments type",
        ),
        (
            {
                "model": "m",
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "calculator", "arguments": "{invalid json}"}}],
                },
            },
            "Malformed JSON arguments in tool call",
        ),
        (
            {
                "model": "m",
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "calculator", "arguments": '"not-a-dict"'}}],
                },
            },
            "expected dict",
        ),
    ],
)
def test_unsupported_tool_call_formats_raise_provider_error(
    bad_tool_payload: dict[str, Any], expected_err: str
) -> None:
    """Ensure unsupported tool formats explicitly raise ProviderError instead of silently faking execution."""
    provider = OllamaProvider(model_name="llama3.2:3b")

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(bad_tool_payload)):
        with pytest.raises(ProviderError, match=expected_err):
            provider.generate(messages=[Message(role="user", content="Run tool")])


def test_malformed_and_empty_payloads() -> None:
    """Test handling of empty, non-dict, or missing-message payloads."""
    provider = OllamaProvider(model_name="llama3.2:3b")

    # Invalid JSON string
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse("Not JSON at all!")):
        with pytest.raises(ProviderError, match="Failed to decode Ollama response JSON"):
            provider.generate(messages=[Message(role="user", content="Hi")])

    # Top-level array instead of object
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse("[]")):
        with pytest.raises(ProviderError, match="expected top-level JSON object"):
            provider.generate(messages=[Message(role="user", content="Hi")])

    # Missing "message" field
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse({"done": True})):
        with pytest.raises(ProviderError, match="missing 'message' field"):
            provider.generate(messages=[Message(role="user", content="Hi")])


def test_connection_refused_error_handling() -> None:
    """Test handling when local Ollama daemon is not running."""
    provider = OllamaProvider(model_name="llama3.2:3b", max_retries=0)
    net_err = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))

    with patch("urllib.request.urlopen", side_effect=net_err):
        with pytest.raises(ProviderError) as exc_info:
            provider.generate(messages=[Message(role="user", content="Hello")])
        assert "Ollama daemon connection refused" in str(exc_info.value)
        assert "ollama serve" in str(exc_info.value)


def test_timeout_error_handling() -> None:
    """Test timeout behavior raises ProviderTimeoutError."""
    provider = OllamaProvider(model_name="llama3.2:3b", max_retries=1)

    with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out")):
        with patch("time.sleep"):
            with pytest.raises(ProviderTimeoutError) as exc_info:
                provider.generate(messages=[Message(role="user", content="Hello")])
            assert "Ollama network/timeout error" in str(exc_info.value)
            assert exc_info.value.retries == 1


def test_missing_model_404_error() -> None:
    """Test HTTP 404 missing model error."""
    provider = OllamaProvider(model_name="nonexistent:model", max_retries=0)
    http_err = urllib.error.HTTPError(
        url="http://127.0.0.1:11434/api/chat",
        code=404,
        msg="Not Found",
        hdrs=MagicMock(),
        fp=io.BytesIO(b'{"error": "model \'nonexistent:model\' not found, try pulling it first"}'),
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(ProviderError) as exc_info:
            provider.generate(messages=[Message(role="user", content="Hello")])
        assert "HTTP 404" in str(exc_info.value)
        assert "ollama pull" in str(exc_info.value)


def test_server_500_error_with_retry_recovery() -> None:
    """Test recovery from transient HTTP 500 error on retry."""
    provider = OllamaProvider(model_name="llama3.2:3b", max_retries=2)

    http_500 = urllib.error.HTTPError(
        url="http://127.0.0.1:11434/api/chat",
        code=500,
        msg="Internal Server Error",
        hdrs=MagicMock(),
        fp=io.BytesIO(b"Internal GPU error"),
    )
    ok_resp = MockHTTPResponse({"message": {"role": "assistant", "content": "Recovered!"}})

    with patch("urllib.request.urlopen", side_effect=[http_500, ok_resp]):
        with patch("time.sleep"):
            resp = provider.generate(messages=[Message(role="user", content="Test")])

    assert resp.content == "Recovered!"
    assert resp.usage.retries == 1


def test_secret_redaction_in_errors() -> None:
    """Test that secret patterns are scrubbed from error messages."""
    raw = "Failed with key sk-abcdef1234567890extra and secret sk-proj-1234567890abcd"
    redacted = _redact_secrets(raw)
    assert "sk-abcdef1234567890extra" not in redacted
    assert "sk-...xtra" in redacted or "sk-..." in redacted


def test_runtime_metadata_capture() -> None:
    """Test that runtime metadata accurately captures hardware and configuration parameters."""
    provider = OllamaProvider(
        model_name="llama3.2:3b",
        seed=42,
        options={"top_p": 0.9},
    )
    meta = provider.get_runtime_metadata()
    assert meta["provider"] == "ollama"
    assert meta["model_name"] == "llama3.2:3b"
    assert meta["seed"] == 42
    assert meta["options"]["top_p"] == 0.9
    assert "platform" in meta
    assert "machine" in meta
    assert "processor" in meta


def test_is_available_returns_false_when_down() -> None:
    """Test is_available returns False when daemon port is closed."""
    provider = OllamaProvider(model_name="llama3.2:3b")
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        assert provider.is_available() is False


def test_cost_tracker_zero_pricing_for_local_models() -> None:
    """Ensure CostTracker attributes zero financial cost to local models."""
    cfg = ExperimentBudgetConfig(
        max_model_requests=50,
        max_spend_usd=0.50,
        model_name="llama3.2:3b",
    )
    tracker = CostTracker(cfg)
    assert tracker._pricing == (0.0, 0.0)

    # Check budget before request does not trip
    tracker.check_budget_before_request(est_prompt_tokens=500)
    assert tracker.estimated_spend_usd == 0.0


def test_controlled_agent_end_to_end_with_mocked_ollama() -> None:
    """Verify ControlledAgent seamlessly integrates with OllamaProvider."""
    provider = OllamaProvider(model_name="llama3.2:3b")
    env = MockToolEnvironment()
    cfg = ExperimentBudgetConfig(max_model_requests=10, max_spend_usd=0.50, model_name="llama3.2:3b")
    tracker = CostTracker(cfg)

    agent = ControlledAgent(
        provider=provider,
        mock_env=env,
        cost_tracker=tracker,
        max_turns=3,
    )

    # Turn 1: Model requests calculator tool
    turn1_payload = {
        "model": "llama3.2:3b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "calculator", "arguments": {"expression": "150 + 250"}},
                }
            ],
        },
        "done": True,
        "prompt_eval_count": 25,
        "eval_count": 12,
    }
    # Turn 2: Model returns final answer after tool response
    turn2_payload = {
        "model": "llama3.2:3b",
        "message": {
            "role": "assistant",
            "content": "The total reimbursement is $400.",
        },
        "done": True,
        "prompt_eval_count": 40,
        "eval_count": 10,
    }

    with patch("urllib.request.urlopen", side_effect=[
        MockHTTPResponse(turn1_payload),
        MockHTTPResponse(turn2_payload),
    ]):
        trace = agent.run_session(
            system_prompt="You are a helpful assistant.",
            user_prompt="Calculate $150 and $250.",
            tools=[CALCULATOR_TOOL_DEF],
        )

    assert trace.finish_reason == "stop"
    assert trace.turns_used == 2
    assert "400" in trace.final_response
    assert len(trace.tool_executions) == 1
    assert trace.tool_executions[0].tool_name == "calculator"
    assert trace.tool_executions[0].output == "400"
    assert trace.estimated_cost_usd == 0.0
    assert tracker.total_requests == 2
    assert tracker.estimated_spend_usd == 0.0


def test_dry_run_model_pilot_with_ollama() -> None:
    """Ensure dry_run_model_pilot reports VALID status and zero spend for Ollama."""
    result = dry_run_model_pilot(
        provider_type="ollama",
        model_name="llama3.2:3b",
        task_filter="conflicting_prompt",
    )
    assert result["provider_type"] == "ollama"
    assert result["model_name"] == "llama3.2:3b"
    assert result["status"] == "VALID"
    assert result["can_execute_live"] is True
    assert result["cost_estimation"]["estimated_spend_usd"] == 0.0


def test_ollama_request_payload_includes_configured_seed() -> None:
    """Verify that configured seed reaches the outgoing HTTP request options payload."""
    provider_seeded = OllamaProvider(model_name="llama3.2:3b", seed=4321)
    captured_requests: list[urllib.request.Request] = []

    mock_resp = {
        "model": "llama3.2:3b",
        "message": {"role": "assistant", "content": "Seeded response"},
        "done": True,
        "prompt_eval_count": 10,
        "eval_count": 5,
    }

    def fake_urlopen(req: urllib.request.Request, *args: Any, **kwargs: Any) -> MockHTTPResponse:
        captured_requests.append(req)
        return MockHTTPResponse(mock_resp)

    # 1. When seed is provided, options["seed"] must equal configured seed
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        resp = provider_seeded.generate(
            messages=[Message(role="user", content="Calculate sum")],
            temperature=0.0,
            max_tokens=128,
        )

    assert isinstance(resp, ModelResponse)
    assert len(captured_requests) == 1
    req = captured_requests[0]
    payload = json.loads(req.data.decode("utf-8"))
    assert "options" in payload
    assert payload["options"]["seed"] == 4321
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["num_predict"] == 128

    # 2. When seed is None, options should not contain "seed"
    provider_unseeded = OllamaProvider(model_name="llama3.2:3b", seed=None)
    captured_requests.clear()
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        resp_unseeded = provider_unseeded.generate(
            messages=[Message(role="user", content="Calculate sum")]
        )

    assert isinstance(resp_unseeded, ModelResponse)
    assert len(captured_requests) == 1
    payload_unseeded = json.loads(captured_requests[0].data.decode("utf-8"))
    assert "options" in payload_unseeded
    assert "seed" not in payload_unseeded["options"]
