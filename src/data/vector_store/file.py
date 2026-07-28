from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger
from pydantic import ValidationError

from src.core.config import VectorStoreConfig
from src.core.types import EmbeddingManifest, IndexFingerprint
from src.data.vector_store.base import VectorStore


class FileVectorStore(VectorStore):
    """Embeddings persisted as a `.npz` matrix beside a JSON manifest."""

    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config

    @property
    def vectors_path(self) -> Path:
        return self.config.path / f"{self.config.name}.npz"

    @property
    def manifest_path(self) -> Path:
        return self.config.path / f"{self.config.name}.json"

    async def read(self, fingerprint: IndexFingerprint) -> np.ndarray | None:
        if self.config.refresh:
            logger.info("Index refresh requested, re-embedding from scratch")
            return None
        if not (self.vectors_path.is_file() and self.manifest_path.is_file()):
            logger.info("No embedding index at {}, building one", self.vectors_path)
            return None

        try:
            manifest = EmbeddingManifest.model_validate_json(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except (ValidationError, ValueError) as exc:
            logger.warning("Unreadable index manifest {}: {}", self.manifest_path, exc)
            return None

        if manifest.fingerprint != fingerprint:
            logger.info("Index fingerprint changed, re-embedding")
            return None

        try:
            with np.load(self.vectors_path) as payload:
                matrix = payload["vectors"]
        except (OSError, KeyError, ValueError) as exc:
            logger.warning("Unreadable index vectors {}: {}", self.vectors_path, exc)
            return None

        if not self._shape_matches(matrix, fingerprint):
            logger.warning(
                "Index shape {} disagrees with the manifest, re-embedding", matrix.shape
            )
            return None

        logger.info(
            "Loaded {} cached vector(s) of width {} from {}",
            matrix.shape[0],
            matrix.shape[1],
            self.vectors_path,
        )
        return matrix

    async def write(self, manifest: EmbeddingManifest, matrix: np.ndarray) -> None:
        self.config.path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.vectors_path, vectors=matrix)
        self.manifest_path.write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info(
            "Wrote embedding index: {} ({} x {}) and {}",
            self.vectors_path,
            matrix.shape[0],
            matrix.shape[1],
            self.manifest_path,
        )

    @staticmethod
    def _shape_matches(matrix: np.ndarray, fingerprint: IndexFingerprint) -> bool:
        if matrix.ndim != 2 or matrix.shape[0] != fingerprint.chunk_count:
            return False
        return not fingerprint.dimensions or matrix.shape[1] == fingerprint.dimensions
