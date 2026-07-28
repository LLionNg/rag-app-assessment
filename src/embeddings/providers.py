from __future__ import annotations

from src.core.config import EmbeddingsConfig
from src.core.exceptions import ConfigError
from src.embeddings.base import EmbeddingProvider
from src.embeddings.bge_m3 import BGEM3EmbeddingProvider
from src.embeddings.openai_compatible import (
    AzureOpenAIEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)

_PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    "bge_m3": BGEM3EmbeddingProvider,
    "azure_openai": AzureOpenAIEmbeddingProvider,
    "openai": OpenAICompatibleEmbeddingProvider,
}


def create_embedding_provider(config: EmbeddingsConfig) -> EmbeddingProvider | None:
    """Build the configured embedding provider, or None when embeddings are off."""
    if not config.enabled:
        return None

    try:
        provider_cls = _PROVIDERS[config.provider]
    except KeyError:
        known = ", ".join(sorted(_PROVIDERS))
        raise ConfigError(
            f"Unknown embeddings.provider '{config.provider}' (available: {known})"
        ) from None
    return provider_cls(config)
