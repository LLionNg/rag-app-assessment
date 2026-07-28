from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from src.core.config import (
    EmbeddingsConfig,
    LLMConfig,
    ProviderConfig,
    Settings,
    load_settings,
)
from src.core.types import Chunk, LLMResponse, Message, ToolSpec
from src.embeddings.base import EmbeddingProvider, Vector
from src.llm.base import LLMProvider

ROOT = Path(__file__).resolve().parents[1]


class ScriptedLLMProvider(LLMProvider):
    """Returns a fixed sequence of responses so agents can be tested offline."""

    def __init__(self, config: LLMConfig, responses: Sequence[LLMResponse]) -> None:
        super().__init__(config)
        self._responses = list(responses)
        self.calls: list[tuple[list[Message], Sequence[ToolSpec] | None]] = []

    async def _complete(
        self,
        messages: Sequence[Message],
        system_prompt: str | None,
        tools: Sequence[ToolSpec] | None,
    ) -> LLMResponse:
        self.calls.append((list(messages), tools))
        if not self._responses:
            raise AssertionError("ScriptedLLMProvider ran out of responses")
        return self._responses.pop(0)


class StubEmbeddingProvider(EmbeddingProvider):
    """Bag-of-words vectors over a fixed vocabulary: deterministic, no network.

    Padded to the configured width so the BGE-M3 dimension contract can be
    exercised without downloading the real model.
    """

    VOCABULARY = ("travel", "international", "expense", "receipt", "leave", "annual")

    def __init__(self, config: EmbeddingsConfig) -> None:
        super().__init__(config)
        self.prepared = 0

    async def prepare(self) -> None:
        self.prepared += 1

    async def _embed_batch(self, texts: Sequence[str]) -> list[Vector]:
        width = self.dimensions or len(self.VOCABULARY)
        padding = [0.0] * max(0, width - len(self.VOCABULARY))
        return [
            [float(term in text.lower()) for term in self.VOCABULARY][:width] + padding
            for text in texts
        ]


def make_stub_embedder(dimensions: int | None = None) -> StubEmbeddingProvider:
    return StubEmbeddingProvider(
        EmbeddingsConfig(
            provider="stub",
            providers={"stub": ProviderConfig(model="stub", dimensions=dimensions)},
        )
    )


@pytest.fixture
def stub_embedder() -> StubEmbeddingProvider:
    return make_stub_embedder()


@pytest.fixture
def settings() -> Settings:
    config = load_settings(ROOT / "config.yml")
    config.llm.provider = "mock"
    config.llm.max_retries = 0
    config.embeddings.provider = None
    config.embeddings.store.enabled = False
    config.retrieval.strategy = "keyword"
    config.knowledge_base.path = ROOT / "data" / "knowledge_base.txt"
    config.logging.file = None
    return config


@pytest.fixture
def scripted_llm(settings: Settings):
    def factory(*responses: LLMResponse) -> ScriptedLLMProvider:
        return ScriptedLLMProvider(settings.llm, responses)

    return factory


@pytest.fixture
def chunks() -> list[Chunk]:
    texts = [
        "International business travel must be approved by the line manager "
        "and the Regional Travel Desk fourteen days before departure.",
        "Expense claims are submitted within thirty calendar days and every "
        "line above USD 25 requires a receipt.",
        "Employees accrue twenty-two days of annual leave per year.",
    ]
    return [
        Chunk(id=f"kb-{index:04d}", text=text, source="test.txt", index=index)
        for index, text in enumerate(texts)
    ]
