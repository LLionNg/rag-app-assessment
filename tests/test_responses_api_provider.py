"""The Responses API differs from chat-completions in shape, not just naming:
`instructions` for the system prompt, a flat `input` item list, a flat tool
schema, and tool calls that must be replayed with their reasoning item."""

from __future__ import annotations

import httpx
import pytest

from src.core.config import LLMConfig, ProviderConfig
from src.core.types import Message, ToolSpec
from src.llm.responses_api import ResponsesAPIProvider

SPEC = ToolSpec(
    name="search_knowledge_base",
    description="Search the corpus",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
)

REASONING_ITEM = {"type": "reasoning", "id": "rs_1", "summary": []}
FUNCTION_CALL_ITEM = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "search_knowledge_base",
    "arguments": '{"query": "travel"}',
}


@pytest.fixture
def provider(monkeypatch) -> ResponsesAPIProvider:
    monkeypatch.setenv("TEST_GATEWAY_KEY", "not-a-real-key")
    return ResponsesAPIProvider(
        LLMConfig(
            provider="bbl_gateway",
            max_retries=4,
            retry_backoff_seconds=20.0,
            providers={
                "bbl_gateway": ProviderConfig(
                    model="gpt-5-mini",
                    base_url="https://gateway.invalid/llm",
                    api_key_env="TEST_GATEWAY_KEY",
                    token_param="max_output_tokens",
                    supports_temperature=False,
                    reasoning_effort="low",
                )
            },
        )
    )


def test_tool_schema_is_flat_with_no_function_wrapper():
    wire = ResponsesAPIProvider._to_wire_tool(SPEC)

    assert wire == {
        "type": "function",
        "name": SPEC.name,
        "description": SPEC.description,
        "parameters": SPEC.parameters,
    }
    assert "function" not in wire


def test_plain_messages_map_to_role_content_items():
    items = ResponsesAPIProvider._to_input([Message(role="user", content="hello")])

    assert items == [{"role": "user", "content": "hello"}]


def test_tool_results_become_function_call_output():
    items = ResponsesAPIProvider._to_input(
        [Message(role="tool", tool_call_id="call_1", content="<snippet/>")]
    )

    assert items == [
        {"type": "function_call_output", "call_id": "call_1", "output": "<snippet/>"}
    ]


def test_raw_items_are_replayed_verbatim_including_reasoning():
    """The gateway rejects a function_call sent without its reasoning item."""
    items = ResponsesAPIProvider._to_input(
        [
            Message(role="user", content="What is the travel policy?"),
            Message(role="assistant", raw_items=[REASONING_ITEM, FUNCTION_CALL_ITEM]),
            Message(role="tool", tool_call_id="call_1", content="approved 14 days"),
        ]
    )

    assert [item.get("type", item.get("role")) for item in items] == [
        "user",
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert items[1] is REASONING_ITEM


def test_parses_message_text_and_usage(provider: ResponsesAPIProvider):
    parsed = provider._parse(
        {
            "status": "completed",
            "model": "gpt-5-mini",
            "output": [
                REASONING_ITEM,
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Approved in advance."}
                    ],
                },
            ],
            "usage": {
                "input_tokens": 139,
                "output_tokens": 14,
                "total_tokens": 153,
            },
        }
    )

    assert parsed.content == "Approved in advance."
    assert parsed.usage.prompt_tokens == 139
    assert parsed.usage.total_tokens == 153
    assert parsed.finish_reason == "completed"


def test_parses_function_calls_and_keeps_raw_items(provider: ResponsesAPIProvider):
    parsed = provider._parse(
        {
            "status": "completed",
            "output": [REASONING_ITEM, FUNCTION_CALL_ITEM],
            "usage": {"input_tokens": 62, "output_tokens": 110, "total_tokens": 172},
        }
    )

    assert parsed.finish_reason == "tool_calls"
    assert parsed.tool_calls[0].id == "call_1"
    assert parsed.tool_calls[0].arguments == {"query": "travel"}
    # Retained so the next turn can replay reasoning + function_call together.
    assert parsed.raw_items == [REASONING_ITEM, FUNCTION_CALL_ITEM]


def _status_error(
    code: int, headers: dict[str, str] | None = None
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://gateway.invalid/llm/responses")
    response = httpx.Response(code, request=request, headers=headers or {})
    return httpx.HTTPStatusError("boom", request=request, response=response)


@pytest.mark.parametrize("code", [408, 409, 429, 500, 503])
def test_transient_statuses_are_retried(code, provider: ResponsesAPIProvider):
    assert provider._is_retryable(_status_error(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_are_not_retried(code, provider: ResponsesAPIProvider):
    assert provider._is_retryable(_status_error(code)) is False


def test_retry_after_header_wins_over_exponential_backoff(
    provider: ResponsesAPIProvider,
):
    """A metered token window needs the gateway's own number, not 20s."""
    exc = _status_error(429, {"retry-after": "60"})

    assert provider._retry_delay(exc, attempt=1) == 60.0


def test_falls_back_to_backoff_without_a_retry_after(provider: ResponsesAPIProvider):
    assert provider._retry_delay(_status_error(429), attempt=1) == 20.0
    assert provider._retry_delay(_status_error(429), attempt=2) == 40.0


def test_unparsable_retry_after_falls_back(provider: ResponsesAPIProvider):
    exc = _status_error(429, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})

    assert provider._retry_delay(exc, attempt=1) == 20.0


def test_missing_base_url_is_rejected(monkeypatch):
    monkeypatch.setenv("TEST_GATEWAY_KEY", "not-a-real-key")
    from src.core.exceptions import ConfigError

    with pytest.raises(ConfigError, match="base_url"):
        ResponsesAPIProvider(
            LLMConfig(
                provider="bbl_gateway",
                providers={
                    "bbl_gateway": ProviderConfig(
                        model="gpt-5-mini", api_key_env="TEST_GATEWAY_KEY"
                    )
                },
            )
        )
