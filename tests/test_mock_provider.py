"""The mock provider is the zero-setup demo path, so its output is a
deliverable in its own right, not just test scaffolding."""

from __future__ import annotations

from src.agents.data_retriever import DataRetrieverAgent
from src.core.config import Settings
from src.core.types import Message
from src.data.knowledge_base import KnowledgeBase
from src.llm.mock import MockLLMProvider
from src.retrieval.keyword import KeywordRetriever
from src.tools.knowledge_search import KnowledgeSearchTool

SNIPPET = (
    '<snippet id="kb-0001" source="kb.txt" score="1.00">\n'
    "Travel needs approval fourteen days ahead.\n"
    "</snippet>"
)


async def _retriever_agent(settings: Settings) -> DataRetrieverAgent:
    knowledge_base = await KnowledgeBase.load(settings.knowledge_base)
    retriever = KeywordRetriever(settings.retrieval)
    await retriever.index(knowledge_base.chunks)
    return DataRetrieverAgent(
        MockLLMProvider(settings.llm),
        KnowledgeSearchTool(retriever, settings.retrieval),
        settings.agents.data_retriever,
    )


async def test_coverage_note_survives_the_tools_being_withdrawn(settings: Settings):
    """With a budget of one, the final turn has no tools attached. The note must
    still describe the searches rather than fall through to the report branch."""
    settings.agents.data_retriever.max_tool_iterations = 1
    agent = await _retriever_agent(settings)

    result = await agent.run("What is the policy on international travel?")

    assert result.snippets
    assert result.notes.startswith("Retrieved snippets for:")
    assert "No snippets were supplied" not in result.notes


async def test_report_branch_reached_when_no_search_happened(settings: Settings):
    """A lone user turn carrying snippets is the Report Generator's shape."""
    provider = MockLLMProvider(settings.llm)

    response = await provider.complete([Message(role="user", content=SNIPPET)])

    assert "kb-0001" in response.content
    assert "Travel needs approval" in response.content


async def test_search_query_is_the_request_not_the_prompt_preamble(
    settings: Settings,
):
    settings.agents.data_retriever.max_tool_iterations = 1
    agent = await _retriever_agent(settings)

    result = await agent.run("What is the policy on international travel?")

    assert result.queries_used == ["What is the policy on international travel?"]
