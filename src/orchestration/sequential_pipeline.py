from __future__ import annotations

from src.core.types import ReportResult, RetrievalResult
from src.orchestration.base import Orchestrator


class SequentialOrchestrator(Orchestrator):
    """Sequential handoff in plain asyncio - no orchestration framework.

    The workflow is a straight line, so the two awaits below are the whole
    graph: the Data Retriever's result is the Report Generator's only input.
    """

    engine = "sequential"

    async def _execute(self, query: str) -> tuple[RetrievalResult, ReportResult]:
        retrieval = await self.retriever_agent.run(query)
        report = await self.report_agent.run(retrieval)
        return retrieval, report
