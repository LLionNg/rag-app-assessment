from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence

from loguru import logger

from src.core.config import EmbeddingsConfig, ProviderConfig
from src.core.exceptions import ConfigError, ProviderError
from src.utils.timing import Timer

Vector = list[float]


class EmbeddingProvider(ABC):
    """Base embedding provider: batching, concurrency, and retries live here."""

    def __init__(self, config: EmbeddingsConfig) -> None:
        self.config = config
        self.provider_config: ProviderConfig = config.active()

    @property
    def name(self) -> str:
        return self.config.provider or "disabled"

    @property
    def model(self) -> str:
        return self.provider_config.model

    @property
    def dimensions(self) -> int:
        """Configured vector width, or 0 when the provider does not declare one."""
        return self.provider_config.dimensions or 0

    @abstractmethod
    async def _embed_batch(self, texts: Sequence[str]) -> list[Vector]: ...

    async def prepare(self) -> None:  # noqa: B027 - optional lifecycle hook
        """Warm up before batching. Local models load weights here, outside the
        per-batch timeout, so a first-run download cannot trip it."""

    async def aclose(self) -> None:  # noqa: B027 - optional lifecycle hook
        """Release provider resources. Overridden by providers holding a client."""

    def _is_retryable(self, exc: Exception) -> bool:
        """Whether a failed batch is worth repeating. Overridden per provider."""
        return True

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []

        await self.prepare()
        size = max(1, self.config.batch_size)
        batches = [texts[i : i + size] for i in range(0, len(texts), size)]

        with Timer() as timer:
            results = await asyncio.gather(
                *(self._embed_with_retries(batch) for batch in batches)
            )

        vectors = [vector for batch in results for vector in batch]
        self._assert_dimensions(vectors)

        logger.info(
            "Embedded {} text(s) in {} batch(es) in {}ms | model={} dim={}",
            len(texts),
            len(batches),
            timer.elapsed_ms,
            self.model,
            len(vectors[0]) if vectors else 0,
        )
        return vectors

    def _assert_dimensions(self, vectors: Sequence[Vector]) -> None:
        expected = self.dimensions
        if not expected:
            return
        widths = {len(vector) for vector in vectors} - {expected}
        if widths:
            raise ProviderError(
                f"{self.name} returned vector width(s) {sorted(widths)} but "
                f"embeddings.providers.{self.config.provider}.dimensions is {expected}"
            )

    async def embed_query(self, text: str) -> Vector:
        vectors = await self.embed([text])
        return vectors[0]

    async def _embed_with_retries(self, texts: Sequence[str]) -> list[Vector]:
        attempts = max(1, self.config.max_retries + 1)
        last_error: Exception | None = None
        made = 0

        for attempt in range(1, attempts + 1):
            made = attempt
            try:
                return await asyncio.wait_for(
                    self._embed_batch(texts), timeout=self.config.timeout_seconds
                )
            except ConfigError:
                raise
            except Exception as exc:
                last_error = exc
                if not self._is_retryable(exc):
                    logger.error("Embedding batch failed permanently: {}", exc)
                    break
                logger.warning(
                    "Embedding batch failed (attempt {}/{}): {}", attempt, attempts, exc
                )
                if attempt < attempts:
                    await asyncio.sleep(2 ** (attempt - 1))

        raise ProviderError(
            f"{self.name} embeddings failed after {made} attempt(s): {last_error}"
        ) from last_error
