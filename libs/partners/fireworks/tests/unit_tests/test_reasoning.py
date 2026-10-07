"""Regression tests for reasoning preservation in ChatFireworks tool loops."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fireworks import AsyncFireworks, Fireworks
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    message_chunk_to_message,
)

from langchain_fireworks import ChatFireworks
from langchain_fireworks.chat_models import (
    _convert_chunk_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
    _usage_to_metadata,
)


def _tool_stream() -> Iterator[dict[str, Any]]:
    yield {"choices": [{"delta": {"role": "assistant", "reasoning_content": "Need "}}]}
    yield {"choices": [{"delta": {"reasoning_content": "a lookup."}}]}
    yield {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_lookup",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"query":"example"}',
                            },
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    yield {
        "choices": [],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 15,
            "total_tokens": 35,
            "completion_tokens_details": {"reasoning_tokens": 11},
        },
    }


async def _async_tool_stream() -> AsyncIterator[dict[str, Any]]:
    for chunk in _tool_stream():
        yield chunk


@pytest.mark.parametrize("reasoning", ["Let me think.", "", None])
def test_assistant_reasoning_round_trip(reasoning: str | None) -> None:
    message = AIMessage(
        content="Answer", additional_kwargs={"reasoning_content": reasoning}
    )
    result = _convert_message_to_dict(message)
    if reasoning is None:
        assert "reasoning_content" not in result
    else:
        assert result["reasoning_content"] == reasoning
    assert result["content"] == "Answer"


def test_non_streaming_reasoning_round_trip_with_tool_call() -> None:
    raw = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Need a lookup.",
        "tool_calls": [
            {
                "id": "call_lookup",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query": "example"}'},
            }
        ],
    }
    assert _convert_message_to_dict(_convert_dict_to_message(raw)) == raw


def test_streamed_reasoning_accumulates_and_replays() -> None:
    chunks = [
        _convert_chunk_to_message_chunk(chunk, AIMessageChunk)
        for chunk in _tool_stream()
    ]
    message = message_chunk_to_message(sum(chunks[1:], chunks[0]))
    assert isinstance(message, AIMessage)
    result = _convert_message_to_dict(message)
    assert result["reasoning_content"] == "Need a lookup."
    assert result["content"] is None
    assert result["tool_calls"][0]["function"]["name"] == "lookup"
    assert message.usage_metadata == {
        "input_tokens": 20,
        "output_tokens": 15,
        "total_tokens": 35,
        "output_token_details": {"reasoning": 11},
    }


@pytest.mark.parametrize(
    "delta", [{}, {"reasoning_content": None}, {"reasoning_content": ""}]
)
def test_stream_without_reasoning_does_not_add_field(delta: dict[str, Any]) -> None:
    chunk = _convert_chunk_to_message_chunk(
        {"choices": [{"delta": {"content": "Answer", **delta}}]}, AIMessageChunk
    )
    assert "reasoning_content" not in chunk.additional_kwargs
    assert "reasoning_content" not in _convert_message_to_dict(
        message_chunk_to_message(chunk)
    )


def test_replay_keeps_reasoning_out_of_content_blocks() -> None:
    message = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "Foreign provider reasoning"},
            {"type": "text", "text": "Answer", "index": 0},
        ],
        additional_kwargs={"reasoning_content": "Fireworks reasoning"},
        response_metadata={"output_version": "v1"},
    )
    result = _convert_message_to_dict(message)
    assert result["content"] == "Answer"
    assert result["reasoning_content"] == "Fireworks reasoning"


@pytest.mark.parametrize("reasoning_tokens", [11, 0])
def test_usage_reasoning_is_a_detail_not_extra_output(reasoning_tokens: int) -> None:
    assert _usage_to_metadata(
        {
            "prompt_tokens": 20,
            "completion_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 7},
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        }
    ) == {
        "input_tokens": 20,
        "output_tokens": 15,
        "total_tokens": 35,
        "input_token_details": {"cache_read": 7},
        "output_token_details": {"reasoning": reasoning_tokens},
    }


@pytest.mark.parametrize("details", [None, {}, {"reasoning_tokens": None}])
def test_usage_without_reasoning_omits_detail(details: dict[str, Any] | None) -> None:
    assert "output_token_details" not in _usage_to_metadata(
        {"completion_tokens_details": details}
    )


@pytest.mark.parametrize("output_version", ["v0", "v1"])
@pytest.mark.parametrize("method", ["stream", "invoke", "astream", "ainvoke"])
@pytest.mark.parametrize("disable_streaming", [False, True, "tool_calling"])
async def test_reasoning_survives_two_tool_turns(
    method: str,
    output_version: Literal["v0", "v1"],
    *,
    disable_streaming: bool | Literal["tool_calling"],
) -> None:
    """Exercise chunk assembly and request serialization at the SDK boundary."""
    client = MagicMock()
    async_client = MagicMock()
    client.create.side_effect = lambda **_: _tool_stream()
    async_client.create = AsyncMock(side_effect=lambda **_: _async_tool_stream())
    chat_model = ChatFireworks(
        model="accounts/fireworks/models/test-model",
        api_key="fake-key",  # type: ignore[arg-type]
        streaming=True,
        disable_streaming=disable_streaming,
        output_version=output_version,
        client=client,
        async_client=async_client,
    )
    model = chat_model.bind_tools(
        [{"name": "lookup", "description": "Look up a value", "parameters": {}}]
    )
    history: list[BaseMessage] = [HumanMessage(content="Find a value")]
    for _ in range(2):
        if method == "stream":
            chunks = list(model.stream(history))
            response = message_chunk_to_message(sum(chunks[1:], chunks[0]))
        elif method == "astream":
            chunks = [chunk async for chunk in model.astream(history)]
            response = message_chunk_to_message(sum(chunks[1:], chunks[0]))
        elif method == "invoke":
            response = model.invoke(history)
        else:
            response = await model.ainvoke(history)
        assert isinstance(response, AIMessage)
        assert response.additional_kwargs["reasoning_content"] == "Need a lookup."
        assert [block["type"] for block in response.content_blocks] == [
            "reasoning",
            "tool_call",
        ]
        assert response.content_blocks[0].get("reasoning") == "Need a lookup."
        assert response.content_blocks[1].get("args") == {"query": "example"}
        assert response.usage_metadata is not None
        assert response.usage_metadata["output_token_details"] == {"reasoning": 11}
        history.extend(
            [response, ToolMessage(content="A value", tool_call_id="call_lookup")]
        )
    client_mock = async_client.create if method.startswith("a") else client.create
    sent = client_mock.call_args_list[1].kwargs["messages"]
    assert sent[1]["reasoning_content"] == "Need a lookup."
    assert sent[1]["content"] in (None, [])
    assert sent[1]["tool_calls"][0]["id"] == "call_lookup"
    assert sent[2] == {
        "role": "tool",
        "content": "A value",
        "tool_call_id": "call_lookup",
    }
    replayed, _ = chat_model._create_message_dicts(history, stop=None)
    assert [replayed[index]["reasoning_content"] for index in (1, 3)] == [
        "Need a lookup.",
        "Need a lookup.",
    ]


@pytest.mark.parametrize("async_mode", [False, True])
async def test_v1_stream_separates_reasoning_text_and_parallel_tool_calls(
    *, async_mode: bool
) -> None:
    """Keep canonical blocks separate while merging fragmented tool arguments."""
    deltas: list[dict[str, Any]] = [
        {"role": "assistant", "reasoning_content": "Need ", "content": "Looking "},
        {"content": "up.", "reasoning_content": ""},
        {
            "tool_calls": [
                {
                    "index": index,
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query":"'},
                }
                for index in (0, 1)
            ]
        },
        {
            "reasoning_content": "two lookups.",
            "tool_calls": [
                {"index": 0, "function": {"arguments": 'first"}'}},
                {"index": 1, "function": {"arguments": 'second"}'}},
            ],
        },
    ]
    raw_chunks: list[dict[str, Any]] = [
        {"choices": [{"delta": delta}]} for delta in deltas
    ]
    raw_chunks.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})

    async def async_chunks() -> AsyncIterator[dict[str, Any]]:
        for chunk in raw_chunks:
            yield chunk

    client = MagicMock()
    client.create.return_value = iter(raw_chunks)
    async_client = MagicMock()
    async_client.create = AsyncMock(return_value=async_chunks())
    model = ChatFireworks(
        model="accounts/fireworks/models/test-model",
        api_key="fake-key",  # type: ignore[arg-type]
        client=client,
        async_client=async_client,
        output_version="v1",
    )
    chunks = (
        [chunk async for chunk in model.astream("Find two values")]
        if async_mode
        else list(model.stream("Find two values"))
    )
    response = sum(chunks[1:], chunks[0])
    blocks = response.content_blocks
    assert [block["type"] for block in blocks] == [
        "reasoning",
        "text",
        "tool_call",
        "tool_call",
    ]
    assert blocks[0].get("reasoning") == "Need two lookups."
    assert blocks[1].get("text") == "Looking up."
    assert [block.get("args") for block in blocks[2:]] == [
        {"query": "first"},
        {"query": "second"},
    ]
    assert [tool_call["id"] for tool_call in response.tool_calls] == [
        "call_0",
        "call_1",
    ]
    assert {
        tool_chunk["index"] for chunk in chunks for tool_chunk in chunk.tool_call_chunks
    } == {0, 1}
    assert (
        len({block.get("index") for chunk in chunks for block in chunk.content_blocks})
        == 4
    )


@pytest.mark.parametrize("provider", ["deepseek", "groq", "ollama", "xai"])
def test_foreign_provider_reasoning_is_not_replayed(provider: str) -> None:
    message = AIMessage(
        content="Answer",
        additional_kwargs={"reasoning_content": "Foreign reasoning"},
        response_metadata={"model_provider": provider},
    )
    assert "reasoning_content" not in _convert_message_to_dict(message)


@pytest.mark.parametrize(
    "reasoning", [123, False, {"text": "Reasoning"}, ["Reasoning"]]
)
def test_non_string_reasoning_is_not_replayed(reasoning: Any) -> None:
    message = AIMessage(
        content="Answer", additional_kwargs={"reasoning_content": reasoning}
    )
    assert "reasoning_content" not in _convert_message_to_dict(message)


@pytest.mark.parametrize("output_version", ["v0", "v1"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("async_mode", [False, True])
async def test_reasoning_round_trip_through_sdk_http_transport(
    output_version: Literal["v0", "v1"], *, streaming: bool, async_mode: bool
) -> None:
    """Check SDK decoding and actual HTTP request serialization without sockets."""
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        chunks = list(_tool_stream())
        common = {"id": "completion_test", "model": "test-model", "created": 0}
        if streaming:
            events = [
                "data: "
                + json.dumps({**common, "object": "chat.completion.chunk", **chunk})
                + "\n\n"
                for chunk in chunks
            ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="".join([*events, "data: [DONE]\n\n"]),
            )
        tool_calls = chunks[2]["choices"][0]["delta"]["tool_calls"]
        return httpx.Response(
            200,
            json={
                **common,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "Need a lookup.",
                            "tool_calls": tool_calls,
                        },
                    }
                ],
                "usage": chunks[-1]["usage"],
            },
        )

    transport = httpx.MockTransport(respond)
    with Fireworks(
        api_key="fake-key", http_client=httpx.Client(transport=transport), max_retries=0
    ) as sdk:
        async with AsyncFireworks(
            api_key="fake-key",
            http_client=httpx.AsyncClient(transport=transport),
            max_retries=0,
        ) as async_sdk:
            model = ChatFireworks(
                model="accounts/fireworks/models/test-model",
                api_key="fake-key",  # type: ignore[arg-type]
                client=sdk.chat.completions,
                async_client=async_sdk.chat.completions,
                streaming=streaming,
                output_version=output_version,
            ).bind_tools(
                [{"name": "lookup", "description": "Look up a value", "parameters": {}}]
            )
            history: list[BaseMessage] = [HumanMessage(content="Find a value")]
            for _ in range(2):
                response = (
                    await model.ainvoke(history)
                    if async_mode
                    else model.invoke(history)
                )
                assert isinstance(response, AIMessage)
                assert (
                    response.additional_kwargs["reasoning_content"] == "Need a lookup."
                )
                assert [block["type"] for block in response.content_blocks] == [
                    "reasoning",
                    "tool_call",
                ]
                assert response.content_blocks[0].get("reasoning") == "Need a lookup."
                assert response.content_blocks[1].get("args") == {"query": "example"}
                assert response.usage_metadata == {
                    "input_tokens": 20,
                    "output_tokens": 15,
                    "total_tokens": 35,
                    "output_token_details": {"reasoning": 11},
                }
                history.extend(
                    [
                        response,
                        ToolMessage(content="A value", tool_call_id="call_lookup"),
                    ]
                )
    assert len(requests) == 2
    assert requests[1]["messages"][1]["reasoning_content"] == "Need a lookup."
    assert requests[1]["messages"][1]["tool_calls"][0]["id"] == "call_lookup"
    assert requests[1]["messages"][2]["role"] == "tool"
    assert requests[1]["messages"][2]["tool_call_id"] == "call_lookup"
