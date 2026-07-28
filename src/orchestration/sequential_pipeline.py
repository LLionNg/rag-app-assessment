from __future__ import annotations

from src.core.types import ReportResult, RetrievalResult
from src.orchestration.base import Orchestrator


class SequentialOrchestrator(Orchestrator):
    """Sequential handoff in plain asyncio - no orchestration framework."""

    engine = "sequential"

    async def _execute(self, query: str) -> tuple[RetrievalResult, ReportResult]:
        retrieval = await self.retriever_agent.run(query)
        report = await self.report_agent.run(retrieval)
        return retrieval, report
