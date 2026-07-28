from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from loguru import logger

from src.core.config import RetrievalConfig
from src.core.exceptions import RetrievalError
from src.core.types import Chunk, RetrievedChunk
from src.embeddings.base import EmbeddingProvider
from src.retrieval.base import Retriever


class SemanticRetriever(Retriever):
    name = "semantic"

    def __init__(self, config: RetrievalConfig, embedder: EmbeddingProvider) -> None:
        super().__init__(config)
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)

    async def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        vectors = await self._embedder.embed([chunk.text for chunk in chunks])

        if len(vectors) != len(self._chunks):
            raise RetrievalError(
                f"Embedder returned {len(vectors)} vector(s) "
                f"for {len(self._chunks)} chunk(s)"
            )

        self._matrix = _unit_rows(np.asarray(vectors, dtype=np.float32))
        logger.debug(
            "Indexed {} chunk(s) for semantic search | dim={}",
            self._matrix.shape[0],
            self._matrix.shape[1] if self._matrix.size else 0,
        )

    async def search(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        if not self._chunks:
            return []

        query_vector = _unit(
            np.asarray(await self._embedder.embed_query(query), dtype=np.float32)
        )
        # Rows were unit-normalised when indexed, so this single product *is* the
        # cosine similarity - the value pgvector returns as 1 - cosine_distance.
        similarities = self._matrix @ query_vector
        threshold = self.config.semantic.min_similarity

        scored = [
            (chunk, float(similarity) if similarity >= threshold else 0.0)
            for chunk, similarity in zip(self._chunks, similarities, strict=True)
        ]
        # Cosine values are already comparable, so keep them as the score.
        return self._rank(scored, query, top_k, normalize=False)


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector
