from __future__ import annotations

import re
from abc import ABC, abstractmethod

from src.core.config import ChunkingConfig
from src.core.exceptions import ConfigError

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class Chunker(ABC):
    """Base chunker. Subclasses decide how to cut text; windowing is shared."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    @abstractmethod
    def split(self, text: str) -> list[str]: ...

    def _window(self, text: str) -> list[str]:
        """Cut `text` into overlapping windows, preferring whitespace boundaries."""
        max_chars = self.config.max_chars
        overlap = min(self.config.overlap_chars, max_chars - 1)
        step = max(1, max_chars - overlap)

        windows: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            if end < len(text):
                boundary = text.rfind(" ", start + step, end)
                if boundary != -1:
                    end = boundary
            window = text[start:end].strip()
            if window:
                windows.append(window)
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)

        return windows


class ParagraphChunker(Chunker):
    """One chunk per paragraph; oversized paragraphs are windowed, tiny ones merged."""

    def split(self, text: str) -> list[str]:
        chunks: list[str] = []
        for paragraph in _PARAGRAPH_BREAK.split(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) > self.config.max_chars:
                chunks.extend(self._window(paragraph))
            elif len(paragraph) < self.config.min_chars and chunks:
                chunks[-1] = f"{chunks[-1]}\n{paragraph}"
            else:
                chunks.append(paragraph)
        return chunks


class FixedChunker(Chunker):
    """Fixed-size overlapping windows over the whole document."""

    def split(self, text: str) -> list[str]:
        return self._window(text.strip())


_CHUNKERS: dict[str, type[Chunker]] = {
    "paragraph": ParagraphChunker,
    "fixed": FixedChunker,
}


def create_chunker(config: ChunkingConfig) -> Chunker:
    try:
        chunker_cls = _CHUNKERS[config.strategy]
    except KeyError:
        known = ", ".join(sorted(_CHUNKERS))
        raise ConfigError(
            f"Unknown chunking strategy '{config.strategy}' (available: {known})"
        ) from None
    return chunker_cls(config)
