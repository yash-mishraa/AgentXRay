"""OpenAI-compatible model provider adapter using standard library networking."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
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

# Regex pattern matching OpenAI secret key prefixes
_SK_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{8,}")

# Standard pricing registry (USD per 1M tokens) as of late 2024 / 2026
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}


def _redact_secrets(text: str, secret: str | None = None) -> str:
    """Ensure sensitive tokens never appear in exceptions or error logs."""
    redacted = text
    if secret and len(secret) >= 4:
        redacted = redacted.replace(secret, f"sk-...{secret[-4:]}")
    # Regex sweep for any OpenAI-formatted keys
    return _SK_PATTERN.sub(lambda m: f"sk-...{m.group(0)[-4:]}", redacted)


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

        # Credentials must only come from explicit parameter, OPENAI_API_KEY env var, or local gitignored .env
        if api_key is not None:
            resolved_key = api_key
        else:
            resolved_key = os.environ.get("OPENAI_API_KEY", "")
            if not resolved_key:
                env_path = Path(".env")
                if env_path.is_file():
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line.startswith("export "):
                            line = line[7:].strip()
                        if line.startswith("OPENAI_API_KEY="):
                            resolved_key = line.split("=", 1)[1].strip().strip("'\"")
                            break
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
        max_retries: int | None = None,
    ) -> ModelResponse:
        self.validate_configuration()

        url = f"{self._base_url}/chat/completions"

        # Format messages for OpenAI API
        formatted_msgs: list[dict[str, Any]] = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role}
            if m.tool_calls:
                entry["content"] = m.content if m.content else None
                entry["tool_calls"] = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments.get("_raw_arguments", json.dumps(tc.arguments))
                            if isinstance(tc.arguments, dict) and "_raw_arguments" in tc.arguments
                            else json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            else:
                entry["content"] = m.content or ""
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

        effective_max_retries = self._max_retries if max_retries is None else max(0, min(self._max_retries, max_retries))
        attempts = 0
        last_error: Exception | None = None

        while attempts <= effective_max_retries:
            attempts += 1
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    raw_body = resp.read().decode("utf-8")
                    data = json.loads(raw_body)
                    return self._parse_response(data, latency_ms, retries=attempts - 1, messages=messages)
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
                    raise ProviderAuthError(f"Authentication failed: {redacted_msg}", retries=attempts - 1) from http_err
                elif http_err.code == 429:
                    if "insufficient_quota" in err_body or "credit_balance_exhausted" in err_body:
                        raise ProviderAuthError(
                            f"OpenAI account credit balance exhausted (insufficient quota): {redacted_msg}",
                            retries=attempts - 1,
                        ) from http_err
                    last_error = ProviderRateLimitError(f"Rate limit exceeded: {redacted_msg}", retries=attempts - 1)
                    # Brief backoff if retrying
                    if attempts <= effective_max_retries:
                        time.sleep(2.0 * attempts)
                        continue
                    raise last_error from http_err
                else:
                    raise ProviderError(f"Provider request failed: {redacted_msg}", retries=attempts - 1) from http_err
            except (urllib.error.URLError, TimeoutError) as net_err:
                last_error = ProviderTimeoutError(
                    _redact_secrets(f"Network/timeout error: {net_err}", self._api_key),
                    retries=attempts - 1,
                )
                if attempts <= effective_max_retries:
                    time.sleep(1.0 * attempts)
                    continue
                raise last_error from net_err
            except json.JSONDecodeError as json_err:
                raise ProviderError(f"Failed to decode provider response: {json_err}", retries=attempts - 1) from json_err

        if last_error:
            raise last_error
        raise ProviderError("Request terminated without response.", retries=max(0, attempts - 1))

    def _parse_response(
        self,
        data: dict[str, Any],
        latency_ms: float,
        retries: int = 0,
        messages: Sequence[Message] = (),
    ) -> ModelResponse:
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

        # Fallback conservative estimate if usage telemetry is omitted
        if p_tokens == 0 and c_tokens == 0:
            p_tokens = max(10, sum(len(m.content) for m in messages) // 4 + 20)
            c_tokens = max(5, len(content) // 4 + 5)

        t_tokens = usage_data.get("total_tokens", p_tokens + c_tokens)

        # Conservative pricing fallback if model is unlisted
        pricing = MODEL_PRICING_PER_1M.get(self._model_name, (2.50, 10.00))
        est_cost = (p_tokens * pricing[0] + c_tokens * pricing[1]) / 1_000_000.0

        return ModelResponse(
            content=content,
            tool_calls=tuple(parsed_tool_calls),
            usage=UsageMetadata(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=t_tokens,
                estimated_cost_usd=round(est_cost, 7),
                retries=retries,
            ),
            finish_reason=finish_reason,
            model_identifier=self._model_name,
            latency_ms=round(latency_ms, 2),
        )
