from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.core.config import RetrievalConfig
from src.core.types import Chunk, RetrievedChunk


class Retriever(ABC):
    """Base retriever.

    Subclasses score chunks against a query; ranking, score normalisation and
    threshold filtering are shared so every strategy returns comparable
    relevance values in the 0-1 range.
    """

    name: str = "retriever"

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    @abstractmethod
    async def index(self, chunks: Sequence[Chunk]) -> None: ...

    @abstractmethod
    async def search(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]: ...

    def _rank(
        self,
        scored: Sequence[tuple[Chunk, float]],
        query: str,
        top_k: int | None,
        normalize: bool = True,
    ) -> list[RetrievedChunk]:
        positive = [(chunk, score) for chunk, score in scored if score > 0]
        if not positive:
            return []

        divisor = max(score for _, score in positive) if normalize else 1.0
        divisor = divisor or 1.0

        ranked = [
            RetrievedChunk(
                chunk=chunk,
                score=score / divisor,
                retriever=self.name,
                query=query,
            )
            for chunk, score in positive
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)

        limit = top_k or self.config.top_k
        return [item for item in ranked if item.score >= self.config.min_score][:limit]
