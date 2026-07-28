from __future__ import annotations

import pytest

from src.application import Application
from src.core.config import Settings
from src.core.exceptions import ConfigError, RagAppError


async def test_end_to_end_pipeline_with_the_mock_provider(settings: Settings):
    async with await Application.create(settings) as app:
        result = await app.ask("What is the policy on international travel?")

    assert app.orchestrator.engine == "langgraph"
    assert result.retrieval.snippets
    assert "travel" in result.retrieval.snippets[0].chunk.text.lower()
    assert result.answer
    assert result.retrieval.turn.agent == "Data Retriever"
    assert result.report.turn.agent == "Report Generator"


async def test_empty_query_is_rejected(settings: Settings):
    async with await Application.create(settings) as app:
        with pytest.raises(RagAppError):
            await app.ask("   ")


async def test_semantic_strategy_requires_embeddings(settings: Settings):
    settings.retrieval.strategy = "semantic"

    with pytest.raises(ConfigError):
        await Application.create(settings)
