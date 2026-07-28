from __future__ import annotations

from collections.abc import Sequence

from openai import AsyncAzureOpenAI, AsyncOpenAI

from src.core.config import EmbeddingsConfig
from src.core.exceptions import ConfigError
from src.embeddings.base import EmbeddingProvider, Vector


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Embeddings from OpenAI or any API-compatible endpoint."""

    def __init__(self, config: EmbeddingsConfig) -> None:
        super().__init__(config)
        self._client = self._build_client()

    def _build_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self._require_api_key(),
            base_url=self.provider_config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=0,
        )

    def _require_api_key(self) -> str:
        api_key = self.provider_config.api_key()
        if not api_key:
            raise ConfigError(
                f"Missing API key for embeddings provider '{self.name}': set "
                f"${self.provider_config.api_key_env} in the environment or .env"
            )
        return api_key

    async def aclose(self) -> None:
        await self._client.close()

    async def _embed_batch(self, texts: Sequence[str]) -> list[Vector]:
        response = await self._client.embeddings.create(
            model=self.provider_config.deployment or self.provider_config.model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]


class AzureOpenAIEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """Embeddings from an Azure OpenAI deployment."""

    def _build_client(self) -> AsyncAzureOpenAI:
        if not self.provider_config.endpoint:
            raise ConfigError("azure_openai embeddings require `endpoint`")
        if not self.provider_config.api_version:
            raise ConfigError("azure_openai embeddings require `api_version`")

        return AsyncAzureOpenAI(
            api_key=self._require_api_key(),
            azure_endpoint=self.provider_config.endpoint,
            api_version=self.provider_config.api_version,
            timeout=self.config.timeout_seconds,
            max_retries=0,
        )
