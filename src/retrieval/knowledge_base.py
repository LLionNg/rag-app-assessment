from __future__ import annotations

import aiofiles
from loguru import logger

from src.core.config import KnowledgeBaseConfig
from src.core.exceptions import KnowledgeBaseError
from src.core.types import Chunk
from src.retrieval.chunking import create_chunker


class KnowledgeBase:
    """The chunked plain-text corpus the Data Retriever searches."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    @classmethod
    async def load(cls, config: KnowledgeBaseConfig) -> KnowledgeBase:
        path = config.path
        if not path.is_file():
            raise KnowledgeBaseError(f"Knowledge base file not found: {path}")

        async with aiofiles.open(path, encoding=config.encoding) as handle:
            text = await handle.read()

        pieces = create_chunker(config.chunking).split(text)
        if not pieces:
            raise KnowledgeBaseError(f"Knowledge base produced no chunks: {path}")

        chunks = [
            Chunk(
                id=f"kb-{index:04d}",
                text=piece,
                source=path.name,
                index=index,
                metadata={"chars": len(piece)},
            )
            for index, piece in enumerate(pieces)
        ]

        logger.info(
            "Loaded knowledge base {} | {} chunk(s), strategy={}",
            path,
            len(chunks),
            config.chunking.strategy,
        )
        return cls(chunks)
