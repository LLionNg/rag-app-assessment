from __future__ import annotations

from typing import Any, ClassVar

from loguru import logger

from src.core.config import RetrievalConfig
from src.core.types import ToolOutput
from src.retrieval.base import Retriever
from src.retrieval.formatting import render_snippets
from src.tools.base import Tool


class KnowledgeSearchTool(Tool):
    """The RAG tool: searches the local knowledge base and returns raw snippets."""

    name = "search_knowledge_base"
    description = (
        "Search the internal knowledge base and return verbatim text snippets. "
        "Use focused, keyword-rich queries. Call it more than once with different "
        "wording when a question covers several topics."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword-rich search phrase, not a full sentence.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of snippets to return.",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(self, retriever: Retriever, config: RetrievalConfig) -> None:
        self._retriever = retriever
        self._config = config

    async def run(self, query: str, top_k: int | None = None) -> ToolOutput:
        snippets = await self._retriever.search(query, top_k=top_k)
        logger.info(
            "search_knowledge_base('{}') -> {} snippet(s) via {}",
            query,
            len(snippets),
            self._config.strategy,
        )
        return ToolOutput(content=render_snippets(snippets), chunks=snippets)
