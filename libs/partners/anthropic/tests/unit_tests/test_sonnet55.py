from typing import Any, cast

import pytest
from anthropic.types import (
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    ThinkingBlock,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableBinding, RunnableSequence
from pydantic import BaseModel

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import _format_messages

MODEL = "claude-sonnet-5-5"
TOOL = {"name": "answer", "input_schema": {"type": "object", "properties": {}}}


def model(**kwargs: Any) -> ChatAnthropic:
    return ChatAnthropic(model=MODEL, api_key="test", **kwargs)


@pytest.mark.parametrize(
    "choice", ["any", "answer", {"type": "any"}, {"type": "tool", "name": "answer"}]
)
def test_forced_tool_choice_rejected(choice: Any) -> None:
    llm = model()
    with pytest.raises(ValueError, match="Forced tool_choice"):
        llm.bind_tools([TOOL], tool_choice=choice)
    with pytest.raises(ValueError, match="Forced tool_choice"):
        llm._get_request_payload("hello", tool_choice=choice)
    with pytest.raises(ValueError, match="Forced tool_choice"):
        llm.get_num_tokens_from_messages([HumanMessage("hello")], tool_choice=choice)


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
        {"thinking": {"type": "between_tools", "display": "summarized"}},
        {"thinking": {"type": "between_tools", "budget_tokens": 1024}},
        {"thinking": {"type": "between_tools", "block_binding": True}},
        {"thinking": {"type": "between_tools"}, "effort": "xhigh"},
        {"thinking": {"type": "between_tools"}, "output_config": {"effort": "max"}},
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


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_between_tools(effort: str) -> None:
    payload = model(thinking={"type": "between_tools"})._get_request_payload(
        "hello", effort=effort
    )
    assert payload["thinking"] == {"type": "between_tools"}
    assert payload["output_config"] == {"effort": effort}


def test_profile_and_defaults() -> None:
    llm = model()
    assert llm.max_tokens == 128000
    assert llm.profile is not None
    assert llm.profile["max_input_tokens"] == 1000000
    assert llm.profile["structured_output"] is False
    assert "tool_choice" not in llm.profile
    assert "reasoning_effort_levels" not in llm.profile
    payload = llm._get_request_payload("hello")
    assert not {"temperature", "top_p", "top_k", "thinking"} & payload.keys()
    effort_payload = llm._get_request_payload("hello", effort="medium")
    assert "thinking" not in effort_payload
    assert effort_payload["output_config"] == {"effort": "medium"}


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


class AdvisorBlock(BaseModel):
    type: str = "advisor_redacted_result"
    data: str = "opaque-data"


def test_encrypted_advisor_streaming() -> None:
    event = RawContentBlockStartEvent.model_construct(
        type="content_block_start", index=0, content_block=cast("Any", AdvisorBlock())
    )
    chunk, _ = model()._make_message_chunk_from_anthropic_event(
        event, stream_usage=True, coerce_content_to_string=False, block_start_event=None
    )
    assert chunk is not None
    assert chunk.content == [
        {"type": "advisor_redacted_result", "data": "opaque-data", "index": 0}
    ]
    payload = model()._get_request_payload(
        [HumanMessage("help"), chunk, HumanMessage("continue")]
    )
    assert payload["messages"][1]["content"][0]["data"] == "opaque-data"


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
