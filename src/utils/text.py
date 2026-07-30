from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_WHITESPACE = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    """No stopword list: BM25's IDF already discounts ubiquitous terms, and it
    does so from the corpus rather than from a fixed English word list."""
    return _WORD.findall(text.lower())


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
