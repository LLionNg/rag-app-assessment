from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import count

from loguru import logger

from src.core.types import (
    AgentTurn,
    Message,
    TokenUsage,
    ToolCall,
    ToolInvocation,
)
from src.llm.base import LLMProvider
from src.tools.base import Tool
from src.utils.timing import Timer


class BaseAgent(ABC):
    """Base agent: owns the model, its tools, and the tool-calling loop.

    Subclasses supply a system prompt and a typed `run` method; the reasoning
    loop, concurrent tool execution, usage accounting and tracing are shared.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        name: str,
        tools: Sequence[Tool] = (),
        max_tool_iterations: int = 1,
    ) -> None:
        self.llm = llm
        self.name = name
        self._tools = {tool.name: tool for tool in tools}
        self._specs = [tool.spec for tool in tools] or None
        self._max_tool_iterations = max_tool_iterations

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    async def _converse(self, user_message: str) -> AgentTurn:
        messages: list[Message] = [Message(role="user", content=user_message)]
        invocations: list[ToolInvocation] = []
        usage = TokenUsage()
        content = ""

        with Timer() as timer:
            for iteration in count(1):
                # Withdrawing the tools once the budget is spent forces a final answer.
                within_budget = iteration <= self._max_tool_iterations
                if not within_budget and self._specs:
                    # The designed end of the loop, not an anomaly: the budget is
                    # spent, so the tools come away and the model must answer.
                    logger.info(
                        "[{}] tool budget spent after {} iteration(s), asking for "
                        "the final answer",
                        self.name,
                        self._max_tool_iterations,
                    )

                specs = self._specs if within_budget else None
                response = await self.llm.complete(
                    messages, system_prompt=self.system_prompt, tools=specs
                )
                usage = usage.merge(response.usage)

                if specs is None or not response.tool_calls:
                    content = response.content
                    break

                logger.info(
                    "[{}] iteration {} requested {} tool call(s)",
                    self.name,
                    iteration,
                    len(response.tool_calls),
                )
                messages.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                        raw_items=response.raw_items,
                    )
                )

                results = await asyncio.gather(
                    *(self._invoke(call) for call in response.tool_calls)
                )
                invocations.extend(results)
                messages.extend(
                    Message(
                        role="tool",
                        tool_call_id=call.id,
                        content=self._render_result(invocation),
                    )
                    for call, invocation in zip(
                        response.tool_calls, results, strict=True
                    )
                )

        return AgentTurn(
            agent=self.name,
            content=content,
            invocations=invocations,
            usage=usage,
            elapsed_ms=timer.elapsed_ms,
        )

    async def _invoke(self, call: ToolCall) -> ToolInvocation:
        tool = self._tools.get(call.name)
        with Timer() as timer:
            if tool is None:
                available = ", ".join(sorted(self._tools)) or "none"
                logger.error(
                    "[{}] unknown tool '{}' (available: {})",
                    self.name,
                    call.name,
                    available,
                )
                return ToolInvocation(
                    tool=call.name,
                    arguments=call.arguments,
                    error=f"Unknown tool '{call.name}'. Available: {available}",
                    elapsed_ms=timer.elapsed_ms,
                )

            try:
                output = await tool.run(**call.arguments)
            except Exception as exc:
                logger.exception("[{}] tool '{}' failed", self.name, call.name)
                return ToolInvocation(
                    tool=call.name,
                    arguments=call.arguments,
                    error=str(exc),
                    elapsed_ms=timer.elapsed_ms,
                )

        return ToolInvocation(
            tool=call.name,
            arguments=call.arguments,
            output=output,
            elapsed_ms=timer.elapsed_ms,
        )

    @staticmethod
    def _render_result(invocation: ToolInvocation) -> str:
        if invocation.error:
            return f"Error: {invocation.error}"
        return invocation.output.content if invocation.output else ""
