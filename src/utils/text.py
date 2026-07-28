from __future__ import annotations

import re
from collections.abc import Iterable

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_WHITESPACE = re.compile(r"\s+")


def tokenize(text: str, stopwords: Iterable[str] = ()) -> list[str]:
    stop = set(stopwords)
    return [token for token in _WORD.findall(text.lower()) if token not in stop]


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
