from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def merge(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class ToolSpec(BaseModel):
    """Provider-neutral tool declaration; each provider maps it to its wire format."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    model: str = ""
    usage: TokenUsage = Field(default_factory=TokenUsage)


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    retriever: str
    query: str = ""


class ToolOutput(BaseModel):
    """Result of one tool invocation.

    `content` is what the model sees; `chunks` carries the structured retrieval
    payload so the calling agent can forward snippets verbatim instead of
    re-reading them out of the rendered text.
    """

    content: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class ToolInvocation(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: ToolOutput | None = None
    error: str | None = None
    elapsed_ms: int = 0


class AgentTurn(BaseModel):
    agent: str
    content: str = ""
    invocations: list[ToolInvocation] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    elapsed_ms: int = 0


class RetrievalResult(BaseModel):
    query: str
    snippets: list[RetrievedChunk] = Field(default_factory=list)
    queries_used: list[str] = Field(default_factory=list)
    notes: str = ""
    turn: AgentTurn


class ReportResult(BaseModel):
    answer: str
    turn: AgentTurn


class PipelineResult(BaseModel):
    query: str
    answer: str
    retrieval: RetrievalResult
    report: ReportResult
    elapsed_ms: int = 0

    @property
    def usage(self) -> TokenUsage:
        return self.retrieval.turn.usage.merge(self.report.turn.usage)
