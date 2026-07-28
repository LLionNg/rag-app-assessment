from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from loguru import logger

from src.core.config import RetrievalConfig
from src.core.types import Chunk, RetrievedChunk
from src.retrieval.base import Retriever
from src.utils.text import tokenize


class KeywordRetriever(Retriever):
    """In-memory BM25 ranking - no external service, no embeddings."""

    name = "keyword"

    def __init__(self, config: RetrievalConfig) -> None:
        super().__init__(config)
        self._chunks: list[Chunk] = []
        self._term_freqs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._doc_freqs: Counter[str] = Counter()
        self._avg_length = 0.0

    async def index(self, chunks: Sequence[Chunk]) -> None:
        stopwords = self.config.keyword.stopwords
        self._chunks = list(chunks)
        self._term_freqs = [
            Counter(tokenize(chunk.text, stopwords)) for chunk in chunks
        ]
        self._lengths = [sum(freqs.values()) for freqs in self._term_freqs]
        self._avg_length = (
            (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        )

        self._doc_freqs = Counter()
        for freqs in self._term_freqs:
            self._doc_freqs.update(freqs.keys())

        logger.debug(
            "Indexed {} chunk(s) for BM25 | vocab={} avg_len={:.1f}",
            len(self._chunks),
            len(self._doc_freqs),
            self._avg_length,
        )

    async def search(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        terms = tokenize(query, self.config.keyword.stopwords)
        if not terms or not self._chunks:
            return []

        scored = [
            (chunk, self._score(terms, position))
            for position, chunk in enumerate(self._chunks)
        ]
        return self._rank(scored, query, top_k)

    def _score(self, terms: Sequence[str], position: int) -> float:
        k1 = self.config.keyword.k1
        b = self.config.keyword.b
        freqs = self._term_freqs[position]
        length = self._lengths[position]
        total_docs = len(self._chunks)

        score = 0.0
        for term in terms:
            term_freq = freqs.get(term, 0)
            if not term_freq:
                continue
            doc_freq = self._doc_freqs[term]
            idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            norm = 1 - b + b * (length / self._avg_length if self._avg_length else 1.0)
            score += idf * (term_freq * (k1 + 1)) / (term_freq + k1 * norm)
        return score
