from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from src.core.config import LLMConfig, Settings, load_settings
from src.core.types import Chunk, LLMResponse, Message, ToolSpec
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


@pytest.fixture
def settings() -> Settings:
    config = load_settings(ROOT / "config.yml")
    config.llm.provider = "mock"
    config.llm.max_retries = 0
    config.embeddings.provider = None
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
