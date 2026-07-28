from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from src.core.types import Chunk, EmbeddingManifest, IndexFingerprint


def fingerprint_chunks(chunks: Sequence[Chunk]) -> str:
    """Digest the exact text that gets embedded.

    Derived from the chunks rather than the source file so that a change to the
    chunking strategy invalidates the index just as a content edit does.
    """
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(f"{chunk.id}\t{chunk.text}\n".encode())
    return digest.hexdigest()


class VectorStore(ABC):
    """Where embedded chunks are kept between runs.

    Async so a network- or database-backed store can implement the same contract;
    the file store below simply does not need to await anything.
    """

    @abstractmethod
    async def read(self, fingerprint: IndexFingerprint) -> np.ndarray | None:
        """Return the stored matrix, or None when it is absent or stale."""

    @abstractmethod
    async def write(self, manifest: EmbeddingManifest, matrix: np.ndarray) -> None:
        """Persist the matrix and the manifest describing it."""
