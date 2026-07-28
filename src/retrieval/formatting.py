from __future__ import annotations

from collections.abc import Sequence

from src.core.types import RetrievedChunk
from src.utils.text import truncate

NO_RESULTS = "No matching content was found in the knowledge base."


def render_snippets(snippets: Sequence[RetrievedChunk], max_chars: int = 0) -> str:
    """Render snippets verbatim in the tagged form both agents and prompts expect."""
    if not snippets:
        return NO_RESULTS

    return "\n".join(
        f'<snippet id="{snippet.chunk.id}" source="{snippet.chunk.source}" '
        f'score="{snippet.score:.2f}">\n'
        f"{truncate(snippet.chunk.text, max_chars)}\n"
        "</snippet>"
        for snippet in snippets
    )
