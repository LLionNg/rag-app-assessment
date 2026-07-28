from __future__ import annotations

from loguru import logger

from src.agents.base import BaseAgent
from src.core.config import ReportGeneratorAgentConfig
from src.core.types import ReportResult, RetrievalResult
from src.llm.base import LLMProvider
from src.prompts import report_generator as prompts
from src.retrieval.formatting import render_snippets


class ReportGeneratorAgent(BaseAgent):
    """Synthesises the retrieved snippets into the final answer. Uses no tools."""

    def __init__(self, llm: LLMProvider, config: ReportGeneratorAgentConfig) -> None:
        super().__init__(llm, name=config.name)
        self._config = config

    @property
    def system_prompt(self) -> str:
        return prompts.SYSTEM_PROMPT

    async def run(self, retrieval: RetrievalResult) -> ReportResult:
        user_message = prompts.build_user_message(
            query=retrieval.query,
            snippets=render_snippets(
                retrieval.snippets, self._config.max_snippet_chars
            ),
            notes=retrieval.notes,
        )

        turn = await self._converse(user_message)
        logger.info(
            "[{}] drafted {} character(s) in {}ms",
            self.name,
            len(turn.content),
            turn.elapsed_ms,
        )
        return ReportResult(answer=turn.content.strip(), turn=turn)
