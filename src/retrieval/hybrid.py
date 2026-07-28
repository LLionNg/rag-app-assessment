from __future__ import annotations

import asyncio
from collections.abc import Sequence

from loguru import logger

from src.core.config import RetrievalConfig
from src.core.types import Chunk, RetrievedChunk
from src.retrieval.base import Retriever


class HybridRetriever(Retriever):
    """Fuses several retrievers with weighted reciprocal rank fusion."""

    name = "hybrid"

    def __init__(
        self, config: RetrievalConfig, retrievers: Sequence[Retriever]
    ) -> None:
        super().__init__(config)
        self._retrievers = list(retrievers)
        self._weights = config.hybrid.weights

    async def index(self, chunks: Sequence[Chunk]) -> None:
        await asyncio.gather(
            *(retriever.index(chunks) for retriever in self._retrievers)
        )

    async def search(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        limit = top_k or self.config.top_k
        candidate_depth = max(limit * 3, limit)

        results = await asyncio.gather(
            *(
                retriever.search(query, top_k=candidate_depth)
                for retriever in self._retrievers
            )
        )

        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        rrf_k = self.config.hybrid.rrf_k

        for retriever, hits in zip(self._retrievers, results, strict=True):
            weight = self._weights.get(retriever.name, 1.0)
            for rank, hit in enumerate(hits, start=1):
                chunks[hit.chunk.id] = hit.chunk
                fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + weight / (
                    rrf_k + rank
                )

        logger.debug(
            "Fused {} candidate(s) from {} retriever(s)",
            len(fused),
            len(self._retrievers),
        )
        scored = [(chunks[chunk_id], score) for chunk_id, score in fused.items()]
        return self._rank(scored, query, top_k)
