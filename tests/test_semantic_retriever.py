from __future__ import annotations

import numpy as np

from src.core.config import RetrievalConfig
from src.core.types import Chunk
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.keyword import KeywordRetriever
from src.retrieval.semantic import SemanticRetriever


async def test_ranks_the_semantically_closest_chunk_first(
    chunks: list[Chunk], stub_embedder
):
    retriever = SemanticRetriever(
        RetrievalConfig(top_k=3, min_score=0.0), stub_embedder
    )
    await retriever.index(chunks)

    hits = await retriever.search("international travel")

    assert hits[0].chunk.id == "kb-0000"


async def test_scores_are_cosine_similarities(chunks: list[Chunk], stub_embedder):
    config = RetrievalConfig(top_k=3, min_score=0.0)
    retriever = SemanticRetriever(config, stub_embedder)
    await retriever.index(chunks)

    hits = await retriever.search("international travel")

    # The value the reference computes as dot / (norm * norm), i.e. 1 - cosine_distance.
    query = np.asarray(await stub_embedder.embed_query("international travel"))
    chunk_vector = np.asarray((await stub_embedder.embed([chunks[0].text]))[0])
    expected = float(
        query @ chunk_vector / (np.linalg.norm(query) * np.linalg.norm(chunk_vector))
    )

    assert abs(hits[0].score - expected) < 1e-6
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)


async def test_min_similarity_drops_weak_matches(chunks: list[Chunk], stub_embedder):
    config = RetrievalConfig(top_k=5, min_score=0.0)
    config.semantic.min_similarity = 0.9
    retriever = SemanticRetriever(config, stub_embedder)
    await retriever.index(chunks)

    # Each chunk covers only half of this two-topic query, so none clears 0.9.
    assert await retriever.search("travel expense") == []


async def test_search_on_empty_index_is_safe(stub_embedder):
    retriever = SemanticRetriever(RetrievalConfig(), stub_embedder)

    assert await retriever.search("anything") == []


async def test_hybrid_fuses_keyword_and_semantic(chunks: list[Chunk], stub_embedder):
    config = RetrievalConfig(top_k=2, min_score=0.0)
    retriever = HybridRetriever(
        config, [KeywordRetriever(config), SemanticRetriever(config, stub_embedder)]
    )
    await retriever.index(chunks)

    hits = await retriever.search("international travel approval")

    assert hits
    assert hits[0].chunk.id == "kb-0000"
    assert hits[0].retriever == "hybrid"
