from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.core.config import (
    EmbeddingsConfig,
    ProviderConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from src.core.exceptions import ProviderError
from src.core.types import Chunk, EmbeddingManifest, IndexFingerprint
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.vector_store import FileVectorStore, fingerprint_chunks
from tests.conftest import StubEmbeddingProvider, make_stub_embedder

DIM = 1024


def _store(tmp_path: Path, refresh: bool = False) -> FileVectorStore:
    return FileVectorStore(VectorStoreConfig(path=tmp_path, name="kb", refresh=refresh))


def _fingerprint(chunks: list[Chunk], dimensions: int = DIM) -> IndexFingerprint:
    return IndexFingerprint(
        model="stub",
        dimensions=dimensions,
        chunk_count=len(chunks),
        chunk_digest=fingerprint_chunks(chunks),
    )


def test_manifest_rejects_mismatched_chunk_ids():
    fingerprint = IndexFingerprint(
        model="stub", dimensions=DIM, chunk_count=3, chunk_digest="abc"
    )

    with pytest.raises(ValidationError):
        EmbeddingManifest(fingerprint=fingerprint, chunk_ids=["kb-0000"])


def test_fingerprint_changes_when_chunk_text_changes(chunks: list[Chunk]):
    edited = [*chunks[:-1], chunks[-1].model_copy(update={"text": "different"})]

    assert fingerprint_chunks(chunks) != fingerprint_chunks(edited)


def test_read_returns_none_before_anything_is_written(
    tmp_path: Path, chunks: list[Chunk]
):
    assert _store(tmp_path).read(_fingerprint(chunks)) is None


def test_round_trips_vectors(tmp_path: Path, chunks: list[Chunk]):
    store = _store(tmp_path)
    fingerprint = _fingerprint(chunks)
    matrix = np.random.default_rng(1).normal(size=(len(chunks), DIM)).astype(np.float32)

    store.write(
        EmbeddingManifest(
            fingerprint=fingerprint, chunk_ids=[chunk.id for chunk in chunks]
        ),
        matrix,
    )
    loaded = store.read(fingerprint)

    assert store.vectors_path.is_file() and store.manifest_path.is_file()
    assert loaded is not None
    assert np.array_equal(loaded, matrix)


def test_stale_fingerprint_is_rejected(tmp_path: Path, chunks: list[Chunk]):
    store = _store(tmp_path)
    fingerprint = _fingerprint(chunks)
    matrix = np.zeros((len(chunks), DIM), dtype=np.float32)
    store.write(
        EmbeddingManifest(
            fingerprint=fingerprint, chunk_ids=[chunk.id for chunk in chunks]
        ),
        matrix,
    )

    other_model = fingerprint.model_copy(update={"model": "BAAI/bge-m3"})

    assert store.read(other_model) is None


def test_refresh_ignores_a_valid_index(tmp_path: Path, chunks: list[Chunk]):
    fingerprint = _fingerprint(chunks)
    matrix = np.zeros((len(chunks), DIM), dtype=np.float32)
    _store(tmp_path).write(
        EmbeddingManifest(
            fingerprint=fingerprint, chunk_ids=[chunk.id for chunk in chunks]
        ),
        matrix,
    )

    assert _store(tmp_path, refresh=True).read(fingerprint) is None


async def test_retriever_embeds_once_then_reuses_the_index(
    tmp_path: Path, chunks: list[Chunk]
):
    config = RetrievalConfig(strategy="semantic", top_k=2, min_score=0.0)

    first = make_stub_embedder(DIM)
    await SemanticRetriever(config, first, _store(tmp_path)).index(chunks)

    second = make_stub_embedder(DIM)
    reused = SemanticRetriever(config, second, _store(tmp_path))
    await reused.index(chunks)

    assert first.prepared == 1  # embedded on the first build
    assert second.prepared == 0  # served entirely from the file
    assert (await reused.search("international travel"))[0].chunk.id == "kb-0000"


async def test_declared_dimensions_are_enforced():
    """A provider returning the wrong width must fail loudly, not poison the index."""

    class WrongWidthEmbedder(StubEmbeddingProvider):
        async def _embed_batch(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    embedder = WrongWidthEmbedder(
        EmbeddingsConfig(
            provider="stub",
            providers={"stub": ProviderConfig(model="stub", dimensions=DIM)},
        )
    )

    with pytest.raises(ProviderError, match="1024"):
        await embedder.embed(["international travel"])
