from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
from loguru import logger

from src.core.config import LLMConfig
from src.core.exceptions import ConfigError
from src.core.types import LLMResponse, Message, TokenUsage, ToolCall, ToolSpec
from src.llm.base import LLMProvider

_RETRYABLE_STATUS = frozenset({408, 409, 429})


class ResponsesAPIProvider(LLMProvider):
    """Azure OpenAI Responses API, including behind an API Management gateway.

    Differs from chat-completions in three ways that matter: the system prompt
    is `instructions`, the turn history is a flat `input` list of typed items
    rather than messages, and a tool call must be replayed together with the
    reasoning item it was emitted with.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        if not self.provider_config.base_url:
            raise ConfigError(f"provider '{self.name}' requires `base_url`")

        self._client = httpx.AsyncClient(
            base_url=self.provider_config.base_url.rstrip("/"),
            headers={
                self.provider_config.auth_header: self._require_api_key(),
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout_seconds,
        )

    def _require_api_key(self) -> str:
        api_key = self.provider_config.api_key()
        if not api_key:
            raise ConfigError(
                f"Missing API key for provider '{self.name}': set "
                f"${self.provider_config.api_key_env} in the environment or .env"
            )
        return api_key

    async def aclose(self) -> None:
        await self._client.aclose()

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(
            exc, httpx.TimeoutException | httpx.TransportError | TimeoutError
        ):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status in _RETRYABLE_STATUS or status >= 500
        return False

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        """Prefer the gateway's own Retry-After; a token window needs seconds,
        not the milliseconds an exponential backoff starts with."""
        if isinstance(exc, httpx.HTTPStatusError):
            header = exc.response.headers.get("retry-after")
            if header:
                try:
                    return float(header)
                except ValueError:
                    logger.debug("Unparsable Retry-After: {}", header)
        return super()._retry_delay(exc, attempt)

    async def _complete(
        self,
        messages: Sequence[Message],
        system_prompt: str | None,
        tools: Sequence[ToolSpec] | None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.provider_config.deployment or self.provider_config.model,
            "input": self._to_input(messages),
            self.provider_config.token_param: self.config.max_tokens,
        }

        if system_prompt:
            payload["instructions"] = system_prompt
        if self.provider_config.reasoning_effort:
            payload["reasoning"] = {"effort": self.provider_config.reasoning_effort}
        if (
            self.provider_config.supports_temperature
            and self.config.temperature is not None
        ):
            payload["temperature"] = self.config.temperature
        if tools:
            payload["tools"] = [self._to_wire_tool(tool) for tool in tools]
            payload["tool_choice"] = "auto"
            if self.provider_config.parallel_tool_calls is not None:
                payload["parallel_tool_calls"] = (
                    self.provider_config.parallel_tool_calls
                )
            if self.provider_config.max_tool_calls is not None:
                payload["max_tool_calls"] = self.provider_config.max_tool_calls

        response = await self._client.post("/responses", json=payload)
        self._log_rate_limits(response)
        response.raise_for_status()
        return self._parse(response.json())

    @staticmethod
    def _to_wire_tool(tool: ToolSpec) -> dict[str, Any]:
        """Flat schema: the Responses API has no nested `function` wrapper."""
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }

    @staticmethod
    def _to_input(messages: Sequence[Message]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in messages:
            if message.raw_items:
                items.extend(message.raw_items)
            elif message.role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content,
                    }
                )
            else:
                items.append({"role": message.role, "content": message.content})
        return items

    def _parse(self, payload: dict[str, Any]) -> LLMResponse:
        output = payload.get("output") or []
        texts: list[str] = []
        tool_calls: list[ToolCall] = []

        for item in output:
            if item.get("type") == "message":
                texts.extend(
                    part.get("text", "")
                    for part in item.get("content") or []
                    if part.get("type") in {"output_text", "text"}
                )
            elif item.get("type") == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=item.get("call_id", ""),
                        name=item.get("name", ""),
                        arguments=self._parse_arguments(
                            item.get("name", ""), item.get("arguments")
                        ),
                    )
                )

        if payload.get("status") == "incomplete":
            logger.warning(
                "Response truncated: {}",
                payload.get("incomplete_details") or "no detail given",
            )

        return LLMResponse(
            content="\n".join(part for part in texts if part.strip()),
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else payload.get("status"),
            model=payload.get("model", ""),
            usage=self._parse_usage(payload.get("usage")),
            raw_items=output,
        )

    @staticmethod
    def _parse_arguments(tool_name: str, raw: str | None) -> dict[str, Any]:
        if not raw or not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Tool '{}' returned unparsable arguments: {}", tool_name, raw)
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _parse_usage(usage: dict[str, Any] | None) -> TokenUsage:
        if not usage:
            return TokenUsage()
        return TokenUsage(
            prompt_tokens=usage.get("input_tokens", 0) or 0,
            completion_tokens=usage.get("output_tokens", 0) or 0,
            total_tokens=usage.get("total_tokens", 0) or 0,
        )

    @staticmethod
    def _log_rate_limits(response: httpx.Response) -> None:
        remaining = response.headers.get("x-ratelimit-remaining-tokens")
        if remaining is None:
            return
        # A 429 is left to raise_for_status, so the retry policy classifies it.
        logger.debug(
            "Gateway quota: {}/{} tokens and {}/{} requests left this window",
            remaining,
            response.headers.get("x-ratelimit-limit-tokens", "?"),
            response.headers.get("x-ratelimit-remaining-requests", "?"),
            response.headers.get("x-ratelimit-limit-requests", "?"),
        )
