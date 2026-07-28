from __future__ import annotations

from src.core.config import RetrievalConfig
from src.core.types import Chunk
from src.retrieval.keyword import KeywordRetriever


async def test_ranks_the_matching_chunk_first(chunks: list[Chunk]):
    retriever = KeywordRetriever(RetrievalConfig(top_k=2))
    await retriever.index(chunks)

    hits = await retriever.search("international travel approval")

    assert hits
    assert hits[0].chunk.id == "kb-0000"
    assert hits[0].score == 1.0
    assert len(hits) <= 2


async def test_returns_nothing_when_no_term_overlaps(chunks: list[Chunk]):
    retriever = KeywordRetriever(RetrievalConfig())
    await retriever.index(chunks)

    assert await retriever.search("quantum chromodynamics") == []


async def test_respects_top_k_override(chunks: list[Chunk]):
    retriever = KeywordRetriever(RetrievalConfig(top_k=3, min_score=0.0))
    await retriever.index(chunks)

    hits = await retriever.search("days", top_k=1)

    assert len(hits) == 1


async def test_search_on_empty_index_is_safe():
    retriever = KeywordRetriever(RetrievalConfig())

    assert await retriever.search("anything") == []
