from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from anthropic._models import construct_type
from anthropic.types import (
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    ThinkingBlock,
)
from anthropic.types.beta import BetaRawMessageStreamEvent
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableBinding, RunnableSequence

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import _format_messages

MODEL = "claude-sonnet-5-5"
TOOL = {"name": "answer", "input_schema": {"type": "object", "properties": {}}}


def model(**kwargs: Any) -> ChatAnthropic:
    return ChatAnthropic(model=MODEL, api_key="test", **kwargs)


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("any", {"type": "any"}),
        ("answer", {"type": "tool", "name": "answer"}),
        ({"type": "any"}, {"type": "any"}),
        ({"type": "tool", "name": "answer"}, {"type": "tool", "name": "answer"}),
    ],
)
def test_forced_tool_choice_left_to_api(choice: Any, expected: dict[str, str]) -> None:
    llm = model()
    assert (
        cast("RunnableBinding", llm.bind_tools([TOOL], tool_choice=choice)).kwargs[
            "tool_choice"
        ]
        == expected
    )
    assert (
        llm._get_request_payload("hello", tool_choice=expected)["tool_choice"]
        == expected
    )


def test_tool_choice_and_structured_output() -> None:
    llm = model()
    assert cast("RunnableBinding", llm.bind_tools([TOOL], tool_choice="auto")).kwargs[
        "tool_choice"
    ] == {"type": "auto"}
    with pytest.warns(UserWarning, match="json_schema"):
        structured = llm.with_structured_output(TOOL)
    assert (
        "tool_choice"
        not in cast(
            "RunnableBinding", cast("RunnableSequence", structured).first
        ).kwargs
    )
    native = llm.with_structured_output(
        {"title": "Answer", "type": "object", "properties": {}}, method="json_schema"
    )
    assert (
        cast("RunnableBinding", cast("RunnableSequence", native).first).kwargs[
            "output_config"
        ]["format"]["type"]
        == "json_schema"
    )
    older = ChatAnthropic(model="claude-sonnet-5", api_key="test")
    assert cast("RunnableBinding", older.bind_tools([TOOL], tool_choice="any")).kwargs[
        "tool_choice"
    ] == {"type": "any"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thinking": {"type": "disabled"}},
        {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        {"temperature": 0},
        {"top_p": 0.5},
        {"top_k": 10},
    ],
)
def test_invalid_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        model(**kwargs)._get_request_payload("hello")
    with pytest.raises(ValueError):
        model()._get_request_payload("hello", **kwargs)


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_between_tools(effort: str) -> None:
    payload = model(thinking={"type": "between_tools"})._get_request_payload(
        "hello", effort=effort
    )
    assert payload["thinking"] == {"type": "between_tools"}
    assert payload["output_config"] == {"effort": effort}


@pytest.mark.parametrize("extra", [{"display": "summarized"}, {"budget_tokens": 1024}])
def test_between_tools_extra_fields_left_to_api(extra: dict[str, Any]) -> None:
    thinking = {"type": "between_tools", **extra}
    assert (
        model(thinking=thinking)._get_request_payload("hello")["thinking"] == thinking
    )


def test_profile_and_defaults() -> None:
    llm = model()
    assert llm.max_tokens == 128000
    assert llm.profile is not None
    assert llm.profile["max_input_tokens"] == 1000000
    assert llm.profile["structured_output"] is True
    assert llm.profile["tool_choice"] is False
    assert llm.profile["reasoning_effort_levels"] == [
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    assert llm.profile["reasoning_effort_default"] == "high"
    payload = llm._get_request_payload("hello")
    assert not {"temperature", "top_p", "top_k", "thinking"} & payload.keys()
    assert llm._get_request_payload("hello", effort="medium")["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }


def test_mid_conversation_system() -> None:
    system, messages = _format_messages(
        [
            SystemMessage("initial"),
            HumanMessage("hello"),
            SystemMessage("new instructions"),
            AIMessage("answer"),
            HumanMessage("next"),
        ],
        model=MODEL,
    )
    assert system == "initial"
    assert [m["role"] for m in messages] == ["user", "system", "assistant", "user"]


@pytest.mark.parametrize("standard", [False, True])
def test_toolset_round_trip(*, standard: bool) -> None:
    content: list[str | dict[str, Any]] = [
        {"type": "thinking", "thinking": "", "signature": "opaque-signature"},
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "click",
            "toolset_name": "computer",
            "input": {"x": 1},
        },
    ]
    ai = AIMessage(
        content=content,
        tool_calls=[
            {"name": "click", "id": "call_1", "args": {"x": 2}, "type": "tool_call"}
        ],
        response_metadata={"model_provider": "anthropic"},
    )
    if standard:
        ai = ai.model_copy(
            update={
                "content": ai.content_blocks,
                "response_metadata": {
                    "model_provider": "anthropic",
                    "output_version": "v1",
                },
            }
        )
    payload = model()._get_request_payload(
        [HumanMessage("click"), ai, ToolMessage("done", tool_call_id="call_1")]
    )
    assert payload["messages"][1]["content"][0] == content[0]
    tool = payload["messages"][1]["content"][1]
    assert tool["toolset_name"] == "computer"
    assert tool["input"] == {"x": 2}
    assert payload["messages"][2]["content"][0]["toolset_name"] == "computer"


def test_encrypted_advisor_streaming() -> None:
    advisor_result = {
        "type": "advisor_tool_result",
        "tool_use_id": "srvtoolu_abc123",
        "content": {
            "type": "advisor_redacted_result",
            "encrypted_content": "opaque-ciphertext",
        },
    }
    event = cast(
        RawContentBlockStartEvent,
        construct_type(
            type_=RawContentBlockStartEvent,
            value={
                "type": "content_block_start",
                "index": 0,
                "content_block": advisor_result,
            },
        ),
    )
    llm = model()
    chunk, _ = llm._make_message_chunk_from_anthropic_event(
        event, stream_usage=True, coerce_content_to_string=False, block_start_event=None
    )
    assert chunk is not None
    assert chunk.content == [{**advisor_result, "index": 0}]
    payload = llm._get_request_payload(
        [HumanMessage("help"), chunk, HumanMessage("continue")]
    )
    assert payload["messages"][1]["content"] == [advisor_result]


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_encrypted_advisor_stream_aggregate_replay(output_version: str) -> None:
    server_tool_use = {
        "type": "server_tool_use",
        "id": "srvtoolu_abc123",
        "name": "advisor",
        "input": {},
    }
    advisor_result = {
        "type": "advisor_tool_result",
        "tool_use_id": "srvtoolu_abc123",
        "content": {
            "type": "advisor_redacted_result",
            "encrypted_content": "opaque-ciphertext",
            "stop_reason": None,
        },
    }
    raw_events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": server_tool_use},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": advisor_result},
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": "Use a token bucket."},
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 20},
        },
        {"type": "message_stop"},
    ]
    events = [
        construct_type(type_=BetaRawMessageStreamEvent, value=event)
        for event in raw_events
    ]
    llm = model(output_version=output_version).bind_tools(
        [{"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-5"}]
    )
    with patch.object(
        ChatAnthropic, "_create", return_value=MagicMock(parse=lambda: iter(events))
    ):
        chunks = [cast("AIMessageChunk", chunk) for chunk in llm.stream("help")]
    full = chunks[0]
    for chunk in chunks[1:]:
        full += chunk

    payload = model()._get_request_payload(
        [HumanMessage("help"), full, HumanMessage("continue")]
    )
    assert payload["messages"][1]["content"] == [
        server_tool_use,
        advisor_result,
        {"type": "text", "text": "Use a token bucket."},
    ]


def test_refusal_details_streaming() -> None:
    event = RawMessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": "refusal",
                "stop_sequence": None,
                "stop_details": {"type": "refusal", "category": "cyber"},
            },
            "usage": {"output_tokens": 3},
        }
    )
    chunk, _ = model()._make_message_chunk_from_anthropic_event(
        event, stream_usage=True, coerce_content_to_string=False, block_start_event=None
    )
    assert chunk is not None
    assert chunk.response_metadata["stop_details"]["category"] == "cyber"


def test_unforced_structured_output_requires_tool_call() -> None:
    with pytest.warns(UserWarning, match="json_schema"):
        structured = model().with_structured_output(TOOL)
    check = cast("RunnableSequence", structured).steps[1]
    with pytest.raises(OutputParserException):
        check.invoke(AIMessage("No tool call"))


def test_mid_conversation_tool_change() -> None:
    block = {
        "type": "tool_addition",
        "tool": {"type": "tool_reference", "name": "answer"},
    }
    _, messages = _format_messages(
        [HumanMessage("hello"), SystemMessage([block]), AIMessage("answer")],
        model=MODEL,
    )
    assert messages[1] == {"role": "system", "content": [block]}


def test_default_thinking_stream_preserves_signature() -> None:
    event = RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block=ThinkingBlock(
            type="thinking", thinking="", signature="opaque-signature"
        ),
    )
    chunk, _ = model()._make_message_chunk_from_anthropic_event(
        event, stream_usage=True, coerce_content_to_string=True, block_start_event=None
    )
    assert chunk is not None
    payload = model()._get_request_payload(
        [HumanMessage("hello"), chunk, HumanMessage("next")]
    )
    assert payload["messages"][1]["content"] == [
        {"type": "thinking", "thinking": "", "signature": "opaque-signature"}
    ]


def test_standard_tool_chunk_namespace_replay() -> None:
    chunk = AIMessageChunk(
        content=[
            {
                "type": "tool_use",
                "name": "click",
                "id": "call_1",
                "input": {},
                "toolset_name": "computer",
                "index": 0,
            }
        ],
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "click",
                "id": "call_1",
                "args": "{}",
                "index": 0,
            }
        ],
        response_metadata={"model_provider": "anthropic"},
    )
    chunk = chunk.model_copy(
        update={
            "content": chunk.content_blocks,
            "response_metadata": {
                "model_provider": "anthropic",
                "output_version": "v1",
            },
        }
    )
    payload = model()._get_request_payload(
        [HumanMessage("click"), chunk, ToolMessage("done", tool_call_id="call_1")]
    )
    assert payload["messages"][1]["content"][0]["toolset_name"] == "computer"
    assert payload["messages"][2]["content"][0]["toolset_name"] == "computer"
