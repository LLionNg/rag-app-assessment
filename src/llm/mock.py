from __future__ import annotations

import re
from collections.abc import Sequence

from loguru import logger

from src.core.config import LLMConfig
from src.core.types import LLMResponse, Message, ToolCall, ToolSpec
from src.llm.base import LLMProvider

_SNIPPET = re.compile(
    r'<snippet id="(?P<id>[^"]+)"[^>]*>(?P<body>.*?)</snippet>', re.DOTALL
)


class MockLLMProvider(LLMProvider):
    """Offline stand-in used until a real model endpoint is available.

    It is deterministic and exercises the full agent path: when tools are
    offered it emits one tool call for the user's question, and when asked to
    write it assembles a clearly-labelled draft from the retrieved snippets.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        logger.warning(
            "Using the mock LLM provider - responses are placeholders, not model output"
        )

    async def _complete(
        self,
        messages: Sequence[Message],
        system_prompt: str | None,
        tools: Sequence[ToolSpec] | None,
    ) -> LLMResponse:
        if tools and not self._has_tool_result(messages):
            return self._tool_call(messages, tools[0])
        # Branch on whether searches have happened, not on whether tools are
        # still offered: the last retriever turn has its tools withdrawn.
        if self._has_tool_result(messages):
            return self._coverage_note(messages)
        return self._draft_answer(messages)

    def _tool_call(self, messages: Sequence[Message], tool: ToolSpec) -> LLMResponse:
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="mock-tool-1",
                    name=tool.name,
                    arguments={"query": self._search_query(messages)},
                )
            ],
            finish_reason="tool_calls",
            model=self.model,
        )

    def _coverage_note(self, messages: Sequence[Message]) -> LLMResponse:
        queries = [
            call.arguments.get("query", "")
            for message in messages
            for call in message.tool_calls
        ]
        joined = "; ".join(query for query in queries if query)
        return LLMResponse(
            content=f"Retrieved snippets for: {joined}"
            if joined
            else "Retrieval done.",
            finish_reason="stop",
            model=self.model,
        )

    def _draft_answer(self, messages: Sequence[Message]) -> LLMResponse:
        prompt = self._last_user_text(messages)
        snippets = _SNIPPET.findall(prompt)

        if not snippets:
            body = "No snippets were supplied, so no answer can be grounded."
        else:
            bullets = "\n".join(
                f"- [{snippet_id}] {self._first_sentences(body)}"
                for snippet_id, body in snippets
            )
            body = f"Based on {len(snippets)} retrieved snippet(s):\n\n{bullets}"

        return LLMResponse(
            content=f"_(placeholder answer from the mock LLM)_\n\n{body}",
            finish_reason="stop",
            model=self.model,
        )

    @staticmethod
    def _has_tool_result(messages: Sequence[Message]) -> bool:
        return any(message.role == "tool" for message in messages)

    @classmethod
    def _search_query(cls, messages: Sequence[Message]) -> str:
        """The prompt ends with the request itself, so its last line is the query."""
        lines = [
            line.strip()
            for line in cls._last_user_text(messages).splitlines()
            if line.strip()
        ]
        return lines[-1] if lines else ""

    @staticmethod
    def _last_user_text(messages: Sequence[Message]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return ""

    @staticmethod
    def _first_sentences(text: str, count: int = 2) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        return " ".join(sentences[:count]).strip()
