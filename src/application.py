from __future__ import annotations

import asyncio

from loguru import logger

from src.agents.data_retriever import DataRetrieverAgent
from src.agents.report_generator import ReportGeneratorAgent
from src.core.config import Settings
from src.core.types import PipelineResult
from src.data.knowledge_base import KnowledgeBase
from src.data.vector_store.file import FileVectorStore
from src.embeddings.base import EmbeddingProvider
from src.embeddings.providers import create_embedding_provider
from src.llm.base import LLMProvider
from src.llm.providers import create_llm_provider
from src.orchestration.base import Orchestrator
from src.orchestration.engines import create_orchestrator
from src.retrieval.strategies import create_retriever
from src.tools.knowledge_search import KnowledgeSearchTool


class Application:
    """Composition root: builds every component from settings and wires them up."""

    def __init__(
        self,
        settings: Settings,
        llm: LLMProvider,
        embedder: EmbeddingProvider | None,
        knowledge_base: KnowledgeBase,
        orchestrator: Orchestrator,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.embedder = embedder
        self.knowledge_base = knowledge_base
        self.orchestrator = orchestrator

    @classmethod
    async def create(cls, settings: Settings) -> Application:
        llm = create_llm_provider(settings.llm)
        embedder = create_embedding_provider(settings.embeddings)

        store = (
            FileVectorStore(settings.embeddings.store)
            if settings.embeddings.store.enabled
            else None
        )

        knowledge_base = await KnowledgeBase.load(settings.knowledge_base)
        retriever = create_retriever(settings.retrieval, embedder, store)
        await retriever.index(knowledge_base.chunks)

        search_tool = KnowledgeSearchTool(retriever, settings.retrieval)
        orchestrator = create_orchestrator(
            settings.orchestration,
            DataRetrieverAgent(llm, search_tool, settings.agents.data_retriever),
            ReportGeneratorAgent(llm, settings.agents.report_generator),
        )

        logger.info(
            "Application ready | llm={}:{} retrieval={} engine={}",
            settings.llm.provider,
            llm.model,
            settings.retrieval.strategy,
            orchestrator.engine,
        )
        return cls(settings, llm, embedder, knowledge_base, orchestrator)

    async def ask(self, query: str) -> PipelineResult:
        return await self.orchestrator.run(query)

    async def aclose(self) -> None:
        providers = [self.llm, self.embedder]
        await asyncio.gather(
            *(provider.aclose() for provider in providers if provider is not None)
        )

    async def __aenter__(self) -> Application:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
