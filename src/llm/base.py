from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence

from loguru import logger

from src.core.config import LLMConfig, ProviderConfig
from src.core.exceptions import ConfigError, ProviderError
from src.core.types import LLMResponse, Message, ToolSpec
from src.utils.timing import Timer


class LLMProvider(ABC):
    """Base LLM provider.

    Subclasses implement `_complete` for a single API call. Retries, backoff,
    timing, and logging are handled here so every provider behaves the same.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.provider_config: ProviderConfig = config.active()

    @property
    def name(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.provider_config.model

    @abstractmethod
    async def _complete(
        self,
        messages: Sequence[Message],
        system_prompt: str | None,
        tools: Sequence[ToolSpec] | None,
    ) -> LLMResponse: ...

    async def aclose(self) -> None:  # noqa: B027 - optional lifecycle hook
        """Release provider resources. Overridden by providers holding a client."""

    def _is_retryable(self, exc: Exception) -> bool:
        """Whether a failed call is worth repeating. Overridden per provider."""
        return True

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        """Seconds to wait before the next attempt. Overridden per provider."""
        return self.config.retry_backoff_seconds * 2 ** (attempt - 1)

    @staticmethod
    def _describe(exc: Exception) -> str:
        """First line only: SDK errors append documentation links we do not want
        repeated on every retry."""
        return str(exc).strip().splitlines()[0]

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system_prompt: str | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        attempts = max(1, self.config.max_retries + 1)
        last_error: Exception | None = None
        made = 0

        for attempt in range(1, attempts + 1):
            made = attempt
            with Timer() as timer:
                try:
                    response = await asyncio.wait_for(
                        self._complete(messages, system_prompt, tools),
                        timeout=self.config.timeout_seconds,
                    )
                except ConfigError:
                    raise
                except Exception as exc:
                    last_error = exc
                    if not self._is_retryable(exc):
                        logger.error(
                            "{} call failed permanently after {}ms: {}",
                            self.name,
                            timer.elapsed_ms,
                            self._describe(exc),
                        )
                        break
                    delay = self._retry_delay(exc, attempt)
                    logger.warning(
                        "{} call failed (attempt {}/{}) after {}ms, retry in {}s: {}",
                        self.name,
                        attempt,
                        attempts,
                        timer.elapsed_ms,
                        round(delay, 1) if attempt < attempts else 0,
                        self._describe(exc),
                    )
                    if attempt < attempts:
                        await asyncio.sleep(delay)
                    continue

            logger.info(
                "{} responded in {}ms | model={} tools={} tokens={}",
                self.name,
                timer.elapsed_ms,
                response.model or self.model,
                len(response.tool_calls),
                response.usage.total_tokens,
            )
            return response

        raise ProviderError(
            f"{self.name} failed after {made} attempt(s): {last_error}"
        ) from last_error
