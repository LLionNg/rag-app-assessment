from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from loguru import logger

from src.core.config import EmbeddingsConfig
from src.core.exceptions import ConfigError
from src.embeddings.base import EmbeddingProvider, Vector

_INSTALL_HINT = {
    "sentence_transformers": ("sentence_transformers", "uv sync --extra bge"),
    "flag_embedding": ("FlagEmbedding", "uv sync --extra bge-flag"),
}


class BGEM3EmbeddingProvider(EmbeddingProvider):
    """Local BAAI/bge-m3 dense embeddings (1024-dim), no API and no per-call network."""

    def __init__(self, config: EmbeddingsConfig) -> None:
        super().__init__(config)
        self._model: Any | None = None
        self._lock = asyncio.Lock()

    @property
    def _options(self):
        return self.provider_config.options

    async def prepare(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is None:
                logger.info(
                    "Loading {} via {} (first run downloads the weights)",
                    self.model,
                    self._options.backend,
                )
                self._model = await asyncio.to_thread(self._load_model)

    async def aclose(self) -> None:
        self._model = None

    def _load_model(self) -> Any:
        backend = self._options.backend
        module_name, hint = _INSTALL_HINT[backend]
        try:
            if backend == "flag_embedding":
                from FlagEmbedding import BGEM3FlagModel

                return BGEM3FlagModel(
                    self.model,
                    use_fp16=self._options.use_fp16,
                    cache_dir=str(self._options.cache_dir)
                    if self._options.cache_dir
                    else None,
                )

            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(
                self.model,
                device=self._options.device,
                cache_folder=str(self._options.cache_dir)
                if self._options.cache_dir
                else None,
            )
        except ImportError as exc:
            raise ConfigError(
                f"embeddings provider 'bge_m3' with backend '{backend}' needs "
                f"{module_name}, which is not installed. Run: {hint}"
            ) from exc

    async def _embed_batch(self, texts: Sequence[str]) -> list[Vector]:
        async with self._lock:
            return await asyncio.to_thread(self._encode, list(texts))

    def _encode(self, texts: list[str]) -> list[Vector]:
        if self._options.backend == "flag_embedding":
            output = self._model.encode(
                texts,
                batch_size=self._options.batch_size,
                max_length=self._options.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            return [vector.tolist() for vector in output["dense_vecs"]]

        vectors = self._model.encode(
            texts,
            batch_size=self._options.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]
