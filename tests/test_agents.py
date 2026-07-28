from __future__ import annotations

from src.agents.data_retriever import DataRetrieverAgent
from src.agents.report_generator import ReportGeneratorAgent
from src.core.config import RetrievalConfig, Settings
from src.core.types import AgentTurn, Chunk, LLMResponse, RetrievalResult, ToolCall
from src.retrieval.keyword import KeywordRetriever
from src.tools.knowledge_search import KnowledgeSearchTool


async def _search_tool(chunks: list[Chunk]) -> KnowledgeSearchTool:
    config = RetrievalConfig(top_k=2)
    retriever = KeywordRetriever(config)
    await retriever.index(chunks)
    return KnowledgeSearchTool(retriever, config)


def _tool_call_response(query: str) -> LLMResponse:
    return LLMResponse(
        tool_calls=[
            ToolCall(
                id="call-1", name="search_knowledge_base", arguments={"query": query}
            )
        ],
        finish_reason="tool_calls",
    )


async def test_data_retriever_returns_snippets_and_never_answers(
    settings: Settings, scripted_llm, chunks: list[Chunk]
):
    llm = scripted_llm(
        _tool_call_response("international travel approval"),
        LLMResponse(content="Covered travel approval.", finish_reason="stop"),
    )
    agent = DataRetrieverAgent(
        llm, await _search_tool(chunks), settings.agents.data_retriever
    )

    result = await agent.run("What is the policy on international travel?")

    assert [snippet.chunk.id for snippet in result.snippets] == ["kb-0000"]
    assert result.queries_used == ["international travel approval"]
    assert result.notes == "Covered travel approval."


async def test_data_retriever_deduplicates_across_searches(
    settings: Settings, scripted_llm, chunks: list[Chunk]
):
    llm = scripted_llm(
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="search_knowledge_base",
                    arguments={"query": "international travel"},
                ),
                ToolCall(
                    id="call-2",
                    name="search_knowledge_base",
                    arguments={"query": "travel approval departure"},
                ),
            ],
            finish_reason="tool_calls",
        ),
        LLMResponse(content="done", finish_reason="stop"),
    )
    agent = DataRetrieverAgent(
        llm, await _search_tool(chunks), settings.agents.data_retriever
    )

    result = await agent.run("travel rules")

    ids = [snippet.chunk.id for snippet in result.snippets]
    assert len(ids) == len(set(ids))


async def test_data_retriever_stops_at_the_tool_budget(
    settings: Settings, scripted_llm, chunks: list[Chunk]
):
    settings.agents.data_retriever.max_tool_iterations = 1
    llm = scripted_llm(
        _tool_call_response("travel"),
        LLMResponse(content="forced final answer", finish_reason="stop"),
    )
    agent = DataRetrieverAgent(
        llm, await _search_tool(chunks), settings.agents.data_retriever
    )

    result = await agent.run("travel rules")

    assert len(result.turn.invocations) == 1
    assert llm.calls[-1][1] is None  # tools withdrawn on the final call


async def test_data_retriever_survives_an_unknown_tool(
    settings: Settings, scripted_llm, chunks: list[Chunk]
):
    llm = scripted_llm(
        LLMResponse(
            tool_calls=[ToolCall(id="call-1", name="not_a_tool", arguments={})],
            finish_reason="tool_calls",
        ),
        LLMResponse(content="nothing found", finish_reason="stop"),
    )
    agent = DataRetrieverAgent(
        llm, await _search_tool(chunks), settings.agents.data_retriever
    )

    result = await agent.run("travel rules")

    assert result.snippets == []
    assert result.turn.invocations[0].error


async def test_report_generator_receives_snippets_and_uses_no_tools(
    settings: Settings, scripted_llm, chunks: list[Chunk]
):
    llm = scripted_llm(
        LLMResponse(content="Final answer [kb-0000].", finish_reason="stop")
    )
    agent = ReportGeneratorAgent(llm, settings.agents.report_generator)
    tool = await _search_tool(chunks)
    snippets = (await tool.run(query="international travel approval")).chunks

    report = await agent.run(
        RetrievalResult(
            query="What is the policy on international travel?",
            snippets=snippets,
            notes="covered travel",
            turn=AgentTurn(agent="Data Retriever"),
        )
    )

    prompt = llm.calls[0][0][0].content
    assert report.answer == "Final answer [kb-0000]."
    assert llm.calls[0][1] is None
    assert '<snippet id="kb-0000"' in prompt
    assert "covered travel" in prompt
