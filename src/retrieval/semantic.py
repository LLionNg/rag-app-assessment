from __future__ import annotations

import math
from collections.abc import Sequence

from loguru import logger

from src.core.config import RetrievalConfig
from src.core.exceptions import RetrievalError
from src.core.types import Chunk, RetrievedChunk
from src.embeddings.base import EmbeddingProvider, Vector
from src.retrieval.base import Retriever


class SemanticRetriever(Retriever):
    """Cosine similarity over embedded chunks."""

    name = "semantic"

    def __init__(self, config: RetrievalConfig, embedder: EmbeddingProvider) -> None:
        super().__init__(config)
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._vectors: list[Vector] = []
        self._norms: list[float] = []

    async def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._vectors = await self._embedder.embed([chunk.text for chunk in chunks])

        if len(self._vectors) != len(self._chunks):
            raise RetrievalError(
                f"Embedder returned {len(self._vectors)} vector(s) "
                f"for {len(self._chunks)} chunk(s)"
            )

        self._norms = [_norm(vector) for vector in self._vectors]
        logger.debug("Indexed {} chunk(s) for semantic search", len(self._chunks))

    async def search(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        if not self._chunks:
            return []

        query_vector = await self._embedder.embed_query(query)
        query_norm = _norm(query_vector)
        threshold = self.config.semantic.min_similarity

        scored: list[tuple[Chunk, float]] = []
        for chunk, vector, norm in zip(
            self._chunks, self._vectors, self._norms, strict=True
        ):
            similarity = _cosine(query_vector, vector, query_norm, norm)
            scored.append((chunk, similarity if similarity >= threshold else 0.0))

        # Similarities are already comparable, so keep the raw cosine values.
        return self._rank(scored, query, top_k, normalize=False)


def _norm(vector: Vector) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(left: Vector, right: Vector, left_norm: float, right_norm: float) -> float:
    if not left_norm or not right_norm:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)
