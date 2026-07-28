from __future__ import annotations

from loguru import logger

from src.agents.base import BaseAgent
from src.core.config import DataRetrieverAgentConfig
from src.core.types import AgentTurn, RetrievalResult, RetrievedChunk
from src.llm.base import LLMProvider
from src.prompts import data_retriever as prompts
from src.tools.knowledge_search import KnowledgeSearchTool


class DataRetrieverAgent(BaseAgent):
    """Finds source material. It never answers - it returns raw snippets."""

    def __init__(
        self,
        llm: LLMProvider,
        search_tool: KnowledgeSearchTool,
        config: DataRetrieverAgentConfig,
    ) -> None:
        super().__init__(
            llm,
            name=config.name,
            tools=[search_tool],
            max_tool_iterations=config.max_tool_iterations,
        )
        self._config = config

    @property
    def system_prompt(self) -> str:
        return prompts.SYSTEM_PROMPT

    async def run(self, query: str) -> RetrievalResult:
        turn = await self._converse(prompts.USER_TEMPLATE.format(query=query))
        snippets = self._collect_snippets(turn)
        queries_used = [
            invocation.arguments["query"]
            for invocation in turn.invocations
            if invocation.arguments.get("query")
        ]

        logger.info(
            "[{}] {} snippet(s) from {} search(es) in {}ms",
            self.name,
            len(snippets),
            len(turn.invocations),
            turn.elapsed_ms,
        )
        return RetrievalResult(
            query=query,
            snippets=snippets,
            queries_used=queries_used,
            notes=turn.content.strip(),
            turn=turn,
        )

    def _collect_snippets(self, turn: AgentTurn) -> list[RetrievedChunk]:
        """Deduplicate across searches, keeping each chunk's best score."""
        best: dict[str, RetrievedChunk] = {}
        for invocation in turn.invocations:
            if not invocation.output:
                continue
            for snippet in invocation.output.chunks:
                current = best.get(snippet.chunk.id)
                if current is None or snippet.score > current.score:
                    best[snippet.chunk.id] = snippet

        ranked = sorted(best.values(), key=lambda item: item.score, reverse=True)
        return ranked[: self._config.max_snippets]
