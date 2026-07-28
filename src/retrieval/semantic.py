from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from loguru import logger

from src.core.config import RetrievalConfig
from src.core.exceptions import RetrievalError
from src.core.types import Chunk, EmbeddingManifest, IndexFingerprint, RetrievedChunk
from src.embeddings.base import EmbeddingProvider
from src.retrieval.base import Retriever
from src.retrieval.similarity import cosine_similarity
from src.retrieval.vector_store import FileVectorStore, fingerprint_chunks


class SemanticRetriever(Retriever):
    name = "semantic"

    def __init__(
        self,
        config: RetrievalConfig,
        embedder: EmbeddingProvider,
        store: FileVectorStore | None = None,
    ) -> None:
        super().__init__(config)
        self._embedder = embedder
        self._store = store
        self._chunks: list[Chunk] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)

    async def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        fingerprint = IndexFingerprint(
            model=self._embedder.model,
            dimensions=self._embedder.dimensions,
            chunk_count=len(self._chunks),
            chunk_digest=fingerprint_chunks(self._chunks),
        )

        cached = self._store.read(fingerprint) if self._store else None
        if cached is not None:
            self._matrix = cached
            return

        self._matrix = await self._embed(chunks)
        if self._store:
            self._store.write(
                EmbeddingManifest(
                    fingerprint=fingerprint,
                    chunk_ids=[chunk.id for chunk in self._chunks],
                ),
                self._matrix,
            )

    async def _embed(self, chunks: Sequence[Chunk]) -> np.ndarray:
        vectors = await self._embedder.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RetrievalError(
                f"Embedder returned {len(vectors)} vector(s) for {len(chunks)} chunk(s)"
            )

        # Stored verbatim: normalisation belongs to cosine_similarity, not the index.
        matrix = np.asarray(vectors, dtype=np.float32)
        logger.debug(
            "Embedded {} chunk(s) for semantic search | dim={}",
            matrix.shape[0],
            matrix.shape[1] if matrix.size else 0,
        )
        return matrix

    async def search(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        if not self._chunks:
            return []

        query_vector = np.asarray(
            await self._embedder.embed_query(query), dtype=np.float32
        )
        similarities = cosine_similarity(self._matrix, query_vector)
        threshold = self.config.semantic.min_similarity

        scored = [
            (chunk, float(similarity) if similarity >= threshold else 0.0)
            for chunk, similarity in zip(self._chunks, similarities, strict=True)
        ]
        # Cosine values are already comparable, so keep them as the score.
        return self._rank(scored, query, top_k, normalize=False)
