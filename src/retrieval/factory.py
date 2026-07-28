from __future__ import annotations

from src.core.config import RetrievalConfig
from src.core.exceptions import ConfigError
from src.embeddings.base import EmbeddingProvider
from src.retrieval.base import Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.keyword import KeywordRetriever
from src.retrieval.semantic import SemanticRetriever


def create_retriever(
    config: RetrievalConfig, embedder: EmbeddingProvider | None = None
) -> Retriever:
    if config.strategy == "keyword":
        return KeywordRetriever(config)

    if config.strategy == "semantic":
        return SemanticRetriever(config, _require_embedder(embedder, config.strategy))

    if config.strategy == "hybrid":
        return HybridRetriever(
            config,
            [
                KeywordRetriever(config),
                SemanticRetriever(config, _require_embedder(embedder, config.strategy)),
            ],
        )

    raise ConfigError(f"Unknown retrieval.strategy '{config.strategy}'")


def _require_embedder(
    embedder: EmbeddingProvider | None, strategy: str
) -> EmbeddingProvider:
    if embedder is None:
        raise ConfigError(
            f"retrieval.strategy '{strategy}' needs embeddings: set "
            "`embeddings.provider` in config.yml"
        )
    return embedder
