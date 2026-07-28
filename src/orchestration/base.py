from __future__ import annotations

from abc import ABC, abstractmethod

from loguru import logger

from src.agents.data_retriever import DataRetrieverAgent
from src.agents.report_generator import ReportGeneratorAgent
from src.core.exceptions import RagAppError
from src.core.types import PipelineResult, ReportResult, RetrievalResult
from src.utils.timing import Timer


class Orchestrator(ABC):
    """Base orchestrator for the sequential two-agent workflow.

    Subclasses only decide *how* the two agents are wired together; input
    validation, timing and result assembly are shared.
    """

    engine: str = "orchestrator"

    def __init__(
        self,
        retriever_agent: DataRetrieverAgent,
        report_agent: ReportGeneratorAgent,
    ) -> None:
        self.retriever_agent = retriever_agent
        self.report_agent = report_agent

    @abstractmethod
    async def _execute(self, query: str) -> tuple[RetrievalResult, ReportResult]: ...

    async def run(self, query: str) -> PipelineResult:
        query = query.strip()
        if not query:
            raise RagAppError("Query must not be empty")

        logger.info("[{}] running query: {}", self.engine, query)
        with Timer() as timer:
            retrieval, report = await self._execute(query)

        result = PipelineResult(
            query=query,
            answer=report.answer,
            retrieval=retrieval,
            report=report,
            elapsed_ms=timer.elapsed_ms,
        )
        logger.info(
            "[{}] completed in {}ms | snippets={} tokens={}",
            self.engine,
            result.elapsed_ms,
            len(retrieval.snippets),
            result.usage.total_tokens,
        )
        return result
