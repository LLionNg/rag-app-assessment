from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, StateGraph

from src.core.types import ReportResult, RetrievalResult
from src.orchestration.base import Orchestrator


class WorkflowState(TypedDict):
    query: str
    retrieval: NotRequired[RetrievalResult]
    report: NotRequired[ReportResult]


class LangGraphOrchestrator(Orchestrator):
    """Sequential LangGraph workflow: retrieve -> report."""

    engine = "langgraph"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(WorkflowState)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("report", self._report_node)
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "report")
        builder.add_edge("report", END)
        return builder.compile()

    async def _retrieve_node(self, state: WorkflowState) -> dict[str, Any]:
        return {"retrieval": await self.retriever_agent.run(state["query"])}

    async def _report_node(self, state: WorkflowState) -> dict[str, Any]:
        return {"report": await self.report_agent.run(state["retrieval"])}

    async def _execute(self, query: str) -> tuple[RetrievalResult, ReportResult]:
        state: WorkflowState = await self._graph.ainvoke({"query": query})
        return state["retrieval"], state["report"]
