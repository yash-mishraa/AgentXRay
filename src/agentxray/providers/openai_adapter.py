"""OpenAI-compatible model provider adapter using standard library networking."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Sequence

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

# Standard pricing registry (USD per 1M tokens) as of late 2024 / 2026
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}


def _redact_secrets(text: str, secret: str | None) -> str:
    """Ensure sensitive tokens never appear in exceptions or error logs."""
    if not secret or len(secret) < 4:
        return text
    return text.replace(secret, f"sk-...{secret[-4:]}")


class OpenAIProvider(ModelProvider):
    """Adapter for OpenAI-compatible Chat Completions API endpoints.

    Implemented using Python standard library `urllib.request` to avoid third-party
    networking dependencies and ensure complete deployment portability.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

        # Credentials must only come from explicit parameter or OPENAI_API_KEY env var
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._api_key = resolved_key.strip()

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    def validate_configuration(self) -> None:
        """Verify that provider credentials and configuration are valid before live requests."""
        if not self._api_key:
            raise ProviderAuthError(
                "OpenAI API key is missing. Set the 'OPENAI_API_KEY' environment variable "
                "before initiating live model experiments, or use '--provider fake' for offline testing."
            )

    def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] = (),
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> ModelResponse:
        self.validate_configuration()

        url = f"{self._base_url}/chat/completions"

        # Format messages for OpenAI API
        formatted_msgs: list[dict[str, Any]] = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            formatted_msgs.append(entry)

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": formatted_msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            payload["tool_choice"] = "auto"

        payload_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "AgentXRay/0.1.0",
        }

        req = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")

        attempts = 0
        last_error: Exception | None = None

        while attempts <= self._max_retries:
            attempts += 1
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    raw_body = resp.read().decode("utf-8")
                    data = json.loads(raw_body)
                    return self._parse_response(data, latency_ms)
            except urllib.error.HTTPError as http_err:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                err_body = ""
                try:
                    err_body = http_err.read().decode("utf-8")
                except Exception:
                    pass

                redacted_msg = _redact_secrets(
                    f"HTTP {http_err.code}: {http_err.reason} - {err_body}",
                    self._api_key,
                )

                if http_err.code in (401, 403):
                    raise ProviderAuthError(f"Authentication failed: {redacted_msg}") from http_err
                elif http_err.code == 429:
                    last_error = ProviderRateLimitError(f"Rate limit exceeded: {redacted_msg}")
                    # Brief backoff if retrying
                    if attempts <= self._max_retries:
                        time.sleep(2.0 * attempts)
                        continue
                    raise last_error from http_err
                else:
                    raise ProviderError(f"Provider request failed: {redacted_msg}") from http_err
            except (urllib.error.URLError, TimeoutError) as net_err:
                last_error = ProviderTimeoutError(
                    _redact_secrets(f"Network/timeout error: {net_err}", self._api_key)
                )
                if attempts <= self._max_retries:
                    time.sleep(1.0 * attempts)
                    continue
                raise last_error from net_err
            except json.JSONDecodeError as json_err:
                raise ProviderError(f"Failed to decode provider response: {json_err}") from json_err

        if last_error:
            raise last_error
        raise ProviderError("Request terminated without response.")

    def _parse_response(self, data: dict[str, Any], latency_ms: float) -> ModelResponse:
        """Parse raw OpenAI response dict into normalized ModelResponse."""
        choices = data.get("choices", [])
        if not choices:
            raise ProviderError("Malformed OpenAI response: missing 'choices'")

        choice = choices[0]
        msg_data = choice.get("message", {})
        content = msg_data.get("content") or ""
        finish_reason = choice.get("finish_reason", "stop")

        raw_tool_calls = msg_data.get("tool_calls", [])
        parsed_tool_calls: list[ToolCall] = []

        for tc in raw_tool_calls:
            fn_data = tc.get("function", {})
            name = fn_data.get("name", "")
            args_str = fn_data.get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except Exception:
                args = {"_raw_arguments": args_str}
            parsed_tool_calls.append(
                ToolCall(
                    call_id=tc.get("id", "call_unknown"),
                    name=name,
                    arguments=args,
                )
            )

        usage_data = data.get("usage", {})
        p_tokens = usage_data.get("prompt_tokens", 0)
        c_tokens = usage_data.get("completion_tokens", 0)
        t_tokens = usage_data.get("total_tokens", p_tokens + c_tokens)

        # Estimate cost
        pricing = MODEL_PRICING_PER_1M.get(self._model_name, (0.15, 0.60))
        est_cost = (p_tokens * pricing[0] + c_tokens * pricing[1]) / 1_000_000.0

        return ModelResponse(
            content=content,
            tool_calls=tuple(parsed_tool_calls),
            usage=UsageMetadata(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=t_tokens,
                estimated_cost_usd=round(est_cost, 7),
            ),
            finish_reason=finish_reason,
            model_identifier=self._model_name,
            latency_ms=round(latency_ms, 2),
        )
