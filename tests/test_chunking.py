from __future__ import annotations

from src.core.config import ChunkingConfig
from src.retrieval.chunking import FixedChunker, ParagraphChunker, create_chunker


def test_paragraph_chunker_keeps_one_chunk_per_paragraph():
    chunker = ParagraphChunker(ChunkingConfig(max_chars=200, min_chars=10))
    chunks = chunker.split("First paragraph body.\n\nSecond paragraph body.")

    assert chunks == ["First paragraph body.", "Second paragraph body."]


def test_paragraph_chunker_merges_paragraphs_below_min_chars():
    chunker = ParagraphChunker(ChunkingConfig(max_chars=200, min_chars=30))
    chunks = chunker.split("A paragraph long enough to stand alone.\n\nToo short.")

    assert len(chunks) == 1
    assert chunks[0].endswith("Too short.")


def test_paragraph_chunker_windows_oversized_paragraphs():
    config = ChunkingConfig(max_chars=60, min_chars=10, overlap_chars=20)
    chunks = ParagraphChunker(config).split(" ".join(["word"] * 60))

    assert len(chunks) > 1
    assert all(len(chunk) <= config.max_chars for chunk in chunks)


def test_fixed_chunker_overlaps_consecutive_windows():
    config = ChunkingConfig(strategy="fixed", max_chars=50, overlap_chars=20)
    chunks = FixedChunker(config).split(" ".join(f"token{i}" for i in range(40)))

    assert len(chunks) > 2
    assert chunks[0].split()[-1] in chunks[1]


def test_create_chunker_honours_configured_strategy():
    assert isinstance(create_chunker(ChunkingConfig(strategy="fixed")), FixedChunker)
    assert isinstance(
        create_chunker(ChunkingConfig(strategy="paragraph")), ParagraphChunker
    )
