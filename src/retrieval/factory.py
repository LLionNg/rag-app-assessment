from __future__ import annotations

from src.core.config import RetrievalConfig
from src.core.exceptions import ConfigError
from src.embeddings.base import EmbeddingProvider
from src.retrieval.base import Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.keyword import KeywordRetriever
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.vector_store import FileVectorStore


def create_retriever(
    config: RetrievalConfig,
    embedder: EmbeddingProvider | None = None,
    store: FileVectorStore | None = None,
) -> Retriever:
    if config.strategy == "keyword":
        return KeywordRetriever(config)

    if config.strategy == "semantic":
        return _semantic(config, embedder, store)

    if config.strategy == "hybrid":
        return HybridRetriever(
            config, [KeywordRetriever(config), _semantic(config, embedder, store)]
        )

    raise ConfigError(f"Unknown retrieval.strategy '{config.strategy}'")


def _semantic(
    config: RetrievalConfig,
    embedder: EmbeddingProvider | None,
    store: FileVectorStore | None,
) -> SemanticRetriever:
    if embedder is None:
        raise ConfigError(
            f"retrieval.strategy '{config.strategy}' needs embeddings: set "
            "`embeddings.provider` in config.yml"
        )
    return SemanticRetriever(config, embedder, store)
