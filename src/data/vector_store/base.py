from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from src.core.types import Chunk, EmbeddingManifest, IndexFingerprint


def fingerprint_chunks(chunks: Sequence[Chunk]) -> str:
    """Digest the exact text that gets embedded."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(f"{chunk.id}\t{chunk.text}\n".encode())
    return digest.hexdigest()


class VectorStore(ABC):
    @abstractmethod
    async def read(self, fingerprint: IndexFingerprint) -> np.ndarray | None:
        """Return the stored matrix, or None when it is absent or stale."""

    @abstractmethod
    async def write(self, manifest: EmbeddingManifest, matrix: np.ndarray) -> None:
        """Persist the matrix and the manifest describing it."""
