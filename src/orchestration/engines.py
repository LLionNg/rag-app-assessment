from __future__ import annotations

from src.agents.data_retriever import DataRetrieverAgent
from src.agents.report_generator import ReportGeneratorAgent
from src.core.config import OrchestrationConfig
from src.core.exceptions import ConfigError
from src.orchestration.base import Orchestrator
from src.orchestration.sequential_pipeline import SequentialOrchestrator

_ENGINES: dict[str, type[Orchestrator]] = {
    "sequential": SequentialOrchestrator,
}


def create_orchestrator(
    config: OrchestrationConfig,
    retriever_agent: DataRetrieverAgent,
    report_agent: ReportGeneratorAgent,
) -> Orchestrator:
    try:
        engine_cls = _ENGINES[config.engine]
    except KeyError:
        known = ", ".join(sorted(_ENGINES))
        raise ConfigError(
            f"Unknown orchestration.engine '{config.engine}' (available: {known})"
        ) from None
    return engine_cls(retriever_agent, report_agent)
