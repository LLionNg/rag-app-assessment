"""The mock provider never exercises the wire format, so a serialisation bug
would only surface against a real endpoint. These tests cover it offline."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.core.config import LLMConfig, ProviderConfig
from src.core.types import Message, ToolCall, ToolSpec
from src.llm.openai_compatible import OpenAICompatibleProvider

SPEC = ToolSpec(
    name="search_knowledge_base",
    description="Search the corpus",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
)


@pytest.fixture
def provider(monkeypatch) -> OpenAICompatibleProvider:
    """A real provider; building the client performs no network I/O."""
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test-not-used")
    return OpenAICompatibleProvider(
        LLMConfig(
            provider="openai",
            providers={
                "openai": ProviderConfig(
                    model="gpt-5-mini", api_key_env="TEST_OPENAI_KEY"
                )
            },
        )
    )


def test_system_prompt_is_prepended_once():
    wire = OpenAICompatibleProvider._to_wire_messages(
        [Message(role="user", content="hello")], "be terse"
    )

    assert [m["role"] for m in wire] == ["system", "user"]
    assert wire[0]["content"] == "be terse"


def test_absent_system_prompt_adds_no_message():
    wire = OpenAICompatibleProvider._to_wire_messages(
        [Message(role="user", content="hello")], None
    )

    assert [m["role"] for m in wire] == ["user"]


def test_assistant_tool_calls_serialise_arguments_as_json():
    wire = OpenAICompatibleProvider._to_wire_messages(
        [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="search_knowledge_base",
                        arguments={"query": "travel"},
                    )
                ],
            )
        ],
        None,
    )

    call = wire[0]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["id"] == "c1"
    assert json.loads(call["function"]["arguments"]) == {"query": "travel"}


def test_tool_results_carry_the_call_id_and_drop_tool_calls():
    wire = OpenAICompatibleProvider._to_wire_messages(
        [Message(role="tool", tool_call_id="c1", content="<snippet/>")], None
    )

    assert wire[0] == {"role": "tool", "tool_call_id": "c1", "content": "<snippet/>"}


def test_tool_spec_maps_to_the_function_schema():
    assert OpenAICompatibleProvider._to_wire_tool(SPEC) == {
        "type": "function",
        "function": {
            "name": SPEC.name,
            "description": SPEC.description,
            "parameters": SPEC.parameters,
        },
    }


def test_parses_content_tool_calls_and_usage(provider: OpenAICompatibleProvider):
    response = SimpleNamespace(
        model="gpt-5-mini",
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5, total_tokens=16),
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content="thinking",
                    tool_calls=[
                        SimpleNamespace(
                            id="c1",
                            function=SimpleNamespace(
                                name="search_knowledge_base",
                                arguments='{"query": "travel"}',
                            ),
                        )
                    ],
                ),
            )
        ],
    )

    parsed = provider._parse(response)

    assert parsed.content == "thinking"
    assert parsed.tool_calls[0].arguments == {"query": "travel"}
    assert parsed.usage.total_tokens == 16
    assert parsed.finish_reason == "tool_calls"


@pytest.mark.parametrize("raw", ["", "   ", None, "not json", '["a list"]'])
def test_unparsable_tool_arguments_degrade_to_empty(raw):
    assert OpenAICompatibleProvider._parse_arguments("search_knowledge_base", raw) == {}


def test_missing_usage_is_zeroed():
    usage = OpenAICompatibleProvider._parse_usage(SimpleNamespace(usage=None))

    assert usage.total_tokens == 0


def _status_error(code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return APIStatusError(
        "boom", response=httpx.Response(code, request=request), body=None
    )


@pytest.mark.parametrize("code", [408, 409, 429, 500, 502, 503])
def test_transient_statuses_are_retried(code, provider: OpenAICompatibleProvider):
    assert provider._is_retryable(_status_error(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(code, provider: OpenAICompatibleProvider):
    """A bad key or malformed request will never succeed on attempt two."""
    assert provider._is_retryable(_status_error(code)) is False


def test_connection_faults_and_timeouts_are_retried(provider: OpenAICompatibleProvider):
    request = httpx.Request("POST", "https://example.invalid")

    assert provider._is_retryable(APIConnectionError(request=request))
    assert provider._is_retryable(APITimeoutError(request=request))
    assert provider._is_retryable(TimeoutError())
