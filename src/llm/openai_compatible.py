from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from src.core.config import LLMConfig
from src.core.exceptions import ConfigError
from src.core.types import LLMResponse, Message, TokenUsage, ToolCall, ToolSpec
from src.llm.base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    """Chat-completions provider for OpenAI and any API-compatible endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = self._build_client()

    def _build_client(self) -> AsyncOpenAI:
        api_key = self._require_api_key()
        return AsyncOpenAI(
            api_key=api_key,
            base_url=self.provider_config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=0,  # the base class owns retry policy
        )

    def _require_api_key(self) -> str:
        api_key = self.provider_config.api_key()
        if not api_key:
            raise ConfigError(
                f"Missing API key for provider '{self.name}': set "
                f"${self.provider_config.api_key_env} in the environment or .env"
            )
        return api_key

    @property
    def _deployed_model(self) -> str:
        return self.provider_config.deployment or self.provider_config.model

    async def aclose(self) -> None:
        await self._client.close()

    async def _complete(
        self,
        messages: Sequence[Message],
        system_prompt: str | None,
        tools: Sequence[ToolSpec] | None,
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self._deployed_model,
            "messages": self._to_wire_messages(messages, system_prompt),
            self.provider_config.token_param: self.config.max_tokens,
        }

        if (
            self.provider_config.supports_temperature
            and self.config.temperature is not None
        ):
            params["temperature"] = self.config.temperature
        if tools:
            params["tools"] = [self._to_wire_tool(tool) for tool in tools]
            params["tool_choice"] = "auto"
        if self.provider_config.extra_body:
            params["extra_body"] = self.provider_config.extra_body

        response = await self._client.chat.completions.create(**params)
        return self._parse(response)

    @staticmethod
    def _to_wire_tool(tool: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    @staticmethod
    def _to_wire_messages(
        messages: Sequence[Message], system_prompt: str | None
    ) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        if system_prompt:
            wire.append({"role": "system", "content": system_prompt})

        for message in messages:
            if message.role == "tool":
                wire.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content,
                    }
                )
                continue

            payload: dict[str, Any] = {
                "role": message.role,
                "content": message.content,
            }
            if message.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
            wire.append(payload)

        return wire

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        raw_calls = getattr(choice.message, "tool_calls", None) or []

        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=[
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=self._parse_arguments(
                        call.function.name, call.function.arguments
                    ),
                )
                for call in raw_calls
            ],
            finish_reason=choice.finish_reason,
            model=getattr(response, "model", self._deployed_model),
            usage=self._parse_usage(response),
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
    def _parse_usage(response: Any) -> TokenUsage:
        usage = getattr(response, "usage", None)
        if not usage:
            return TokenUsage()
        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )
