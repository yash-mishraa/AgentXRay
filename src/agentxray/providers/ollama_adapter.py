"""Ollama local open-weights model provider adapter using standard library networking."""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Sequence

from agentxray.providers.base import (
    Message,
    ModelProvider,
    ModelResponse,
    ProviderError,
    ProviderTimeoutError,
    ToolCall,
    ToolDefinition,
    UsageMetadata,
)

# Regex pattern matching secret key prefixes for safety redaction
_SK_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{8,}")


def _redact_secrets(text: str, secret: str | None = None) -> str:
    """Ensure sensitive tokens never appear in exceptions or error logs."""
    redacted = text
    if secret and len(secret) >= 4:
        redacted = redacted.replace(secret, f"sk-...{secret[-4:]}")
    return _SK_PATTERN.sub(lambda m: f"sk-...{m.group(0)[-4:]}", redacted)


class OllamaProvider(ModelProvider):
    """Adapter for Ollama local open-weights inference API endpoints.

    Implemented using Python standard library `urllib.request` to avoid third-party
    networking dependencies and ensure complete deployment portability.
    Strictly bound to local loopback (127.0.0.1 or localhost) to prevent unauthorized
    remote network traffic. Local inference incurs zero API financial costs ($0.0000).
    """

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._seed = seed
        self._custom_options = options or {}

        # Strictly enforce local loopback binding
        parsed = urllib.parse.urlparse(self._base_url)
        hostname = (parsed.hostname or "").lower()
        if hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(
                f"Safety violation: OllamaProvider must be bound strictly to local loopback "
                f"(127.0.0.1, localhost, or ::1). Refusing remote host: {self._base_url!r}"
            )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def base_url(self) -> str:
        return self._base_url

    def validate_configuration(self) -> None:
        """Verify that provider configuration is valid before requests."""
        if not self._model_name or not self._model_name.strip():
            raise ProviderError("Model name cannot be empty for OllamaProvider.")
        parsed = urllib.parse.urlparse(self._base_url)
        hostname = (parsed.hostname or "").lower()
        if hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(f"OllamaProvider must connect to local loopback, got {self._base_url}")

    def is_available(self, timeout: float = 1.0) -> bool:
        """Check if local Ollama daemon is reachable without initiating generation."""
        url = f"{self._base_url}/api/tags"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_runtime_metadata(self) -> dict[str, Any]:
        """Record model, runtime, inference options, and local hardware metadata."""
        return {
            "provider": "ollama",
            "model_name": self._model_name,
            "base_url": self._base_url,
            "timeout_seconds": self._timeout_seconds,
            "max_retries": self._max_retries,
            "seed": self._seed,
            "options": self._custom_options,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }

    def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] = (),
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int | None = None,
    ) -> ModelResponse:
        self.validate_configuration()

        url = f"{self._base_url}/api/chat"

        # Format messages for Ollama /api/chat
        formatted_msgs: list[dict[str, Any]] = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role, "content": m.content or ""}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in m.tool_calls
                ]
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            formatted_msgs.append(entry)

        # Build inference options
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if self._seed is not None:
            options["seed"] = self._seed
        # Merge custom generation settings
        options.update(self._custom_options)

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": formatted_msgs,
            "stream": False,
            "options": options,
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

        payload_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AgentXRay/0.1.0",
        }

        req = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")

        effective_max_retries = (
            self._max_retries if max_retries is None else max(0, min(self._max_retries, max_retries))
        )
        attempts = 0
        last_error: Exception | None = None

        while attempts <= effective_max_retries:
            attempts += 1
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    raw_body = resp.read().decode("utf-8")
                    try:
                        data = json.loads(raw_body)
                    except json.JSONDecodeError as json_err:
                        raise ProviderError(
                            f"Failed to decode Ollama response JSON: {json_err}",
                            retries=attempts - 1,
                        ) from json_err
                    return self._parse_response(data, latency_ms, retries=attempts - 1, messages=messages)
            except urllib.error.HTTPError as http_err:
                err_body = ""
                try:
                    err_body = http_err.read().decode("utf-8")
                except Exception:
                    pass

                redacted_msg = _redact_secrets(f"HTTP {http_err.code}: {http_err.reason} - {err_body}")

                if http_err.code == 404:
                    raise ProviderError(
                        f"Ollama model or endpoint not found (HTTP 404): {redacted_msg}. "
                        f"Ensure model '{self._model_name}' is pulled locally (`ollama pull {self._model_name}`).",
                        retries=attempts - 1,
                    ) from http_err
                elif http_err.code >= 500:
                    last_error = ProviderError(
                        f"Ollama server error (HTTP {http_err.code}): {redacted_msg}",
                        retries=attempts - 1,
                    )
                    if attempts <= effective_max_retries:
                        time.sleep(1.0 * attempts)
                        continue
                    raise last_error from http_err
                else:
                    raise ProviderError(
                        f"Ollama request failed with HTTP {http_err.code}: {redacted_msg}",
                        retries=attempts - 1,
                    ) from http_err
            except (urllib.error.URLError, TimeoutError, socket.timeout) as net_err:
                err_str = str(net_err)
                if "Connection refused" in err_str or "[Errno 61]" in err_str or "[Errno 111]" in err_str:
                    raise ProviderError(
                        f"Ollama daemon connection refused at {self._base_url}. "
                        "Ensure the local Ollama daemon is running (`ollama serve`).",
                        retries=attempts - 1,
                    ) from net_err

                last_error = ProviderTimeoutError(
                    _redact_secrets(f"Ollama network/timeout error: {net_err}"),
                    retries=attempts - 1,
                )
                if attempts <= effective_max_retries:
                    time.sleep(1.0 * attempts)
                    continue
                raise last_error from net_err

        if last_error:
            raise last_error
        raise ProviderError("Ollama request terminated without response.", retries=max(0, attempts - 1))

    def _parse_response(
        self,
        data: dict[str, Any],
        latency_ms: float,
        retries: int = 0,
        messages: Sequence[Message] = (),
    ) -> ModelResponse:
        """Parse raw Ollama /api/chat response dict into normalized ModelResponse."""
        if not isinstance(data, dict):
            raise ProviderError("Malformed Ollama response: expected top-level JSON object")

        if "message" not in data:
            raise ProviderError("Malformed Ollama response: missing 'message' field")

        msg_data = data["message"]
        if not isinstance(msg_data, dict):
            raise ProviderError("Malformed Ollama response: 'message' must be a JSON object")

        content = msg_data.get("content") or ""
        finish_reason = "stop" if data.get("done", True) else "length"

        raw_tool_calls = msg_data.get("tool_calls")
        parsed_tool_calls: list[ToolCall] = []

        if raw_tool_calls is not None:
            if not isinstance(raw_tool_calls, (list, tuple)):
                raise ProviderError(
                    f"Unsupported tool-call format: 'tool_calls' must be a list, got {type(raw_tool_calls).__name__}"
                )

            for idx, tc in enumerate(raw_tool_calls):
                if not isinstance(tc, dict):
                    raise ProviderError(
                        f"Unsupported tool-call item format: expected dict, got {type(tc).__name__}"
                    )
                fn_data = tc.get("function")
                if not isinstance(fn_data, dict):
                    raise ProviderError(
                        f"Unsupported tool-call format: missing or invalid 'function' dictionary in {tc}"
                    )

                name = fn_data.get("name")
                if not name or not isinstance(name, str):
                    raise ProviderError(
                        f"Unsupported tool-call format: invalid or missing function name in {tc}"
                    )

                raw_args = fn_data.get("arguments", {})
                if isinstance(raw_args, dict):
                    args = raw_args
                elif isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                        if not isinstance(args, dict):
                            raise ProviderError(
                                f"Unsupported tool-call format: function arguments parsed as {type(args).__name__}, expected dict"
                            )
                    except json.JSONDecodeError as e:
                        raise ProviderError(
                            f"Malformed JSON arguments in tool call: {raw_args!r} - {e}"
                        ) from e
                else:
                    raise ProviderError(
                        f"Unsupported tool-call arguments type: expected dict or JSON string, got {type(raw_args).__name__}"
                    )

                call_id = tc.get("id") or f"call_ollama_{idx}"
                parsed_tool_calls.append(
                    ToolCall(
                        call_id=call_id,
                        name=name,
                        arguments=args,
                    )
                )

            if parsed_tool_calls:
                finish_reason = "tool_calls"

        # Token telemetry from Ollama response
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        # Conservative fallback if telemetry omitted
        if prompt_tokens == 0 and completion_tokens == 0:
            prompt_tokens = max(10, sum(len(m.content) for m in messages) // 4 + 20)
            completion_tokens = max(5, len(content) // 4 + 5)

        total_tokens = prompt_tokens + completion_tokens

        return ModelResponse(
            content=content,
            tool_calls=tuple(parsed_tool_calls),
            usage=UsageMetadata(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=0.0,  # Local model inference incurs zero API financial spend
                retries=retries,
            ),
            finish_reason=finish_reason,
            model_identifier=data.get("model", self._model_name),
            latency_ms=round(latency_ms, 2),
        )
