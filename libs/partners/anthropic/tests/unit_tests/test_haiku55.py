from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import RawMessageStreamEvent
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableBinding, RunnableSequence

from langchain_anthropic import ChatAnthropic

MODEL = "claude-haiku-5-5"
TOOL = {"name": "answer", "input_schema": {"type": "object", "properties": {}}}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        {"thinking": {"type": "disabled"}, "effort": "xhigh"},
        {"thinking": {"type": "disabled"}, "output_config": {"effort": "max"}},
        {"temperature": 0},
        {"top_p": 1},
        {"top_k": 1},
        {"temperature": 1, "top_p": 0.99},
    ],
)
@pytest.mark.parametrize("source", ["constructor", "call"])
def test_invalid_configuration(kwargs: dict[str, Any], source: str) -> None:
    model = ChatAnthropic(
        model=MODEL, api_key="test", **(dict(kwargs) if source == "constructor" else {})
    )
    with pytest.raises(ValueError):
        model._get_request_payload("hello", **(kwargs if source == "call" else {}))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"temperature": 1},
        {"top_p": 0.99},
        {"thinking": {"type": "adaptive"}, "effort": "medium"},
        {"thinking": {"type": "disabled"}, "effort": "high"},
    ],
)
def test_valid_configuration(kwargs: dict[str, Any]) -> None:
    model = ChatAnthropic(model=MODEL, api_key="test", **kwargs)
    payload = model._get_request_payload("hello")
    assert payload["model"] == MODEL
    if "effort" in kwargs:
        assert payload["output_config"] == {"effort": kwargs["effort"]}
    if "thinking" in kwargs:
        assert payload["thinking"] == kwargs["thinking"]


@pytest.mark.parametrize("thinking", [None, {"type": "adaptive"}])
@pytest.mark.parametrize("choice", ["any", "answer"])
def test_forced_tool_choice_with_adaptive_thinking(
    thinking: dict[str, str] | None, choice: str
) -> None:
    model = ChatAnthropic(model=MODEL, api_key="test", thinking=thinking)
    bound = cast("RunnableBinding", model.bind_tools([TOOL], tool_choice=choice))
    expected = {"type": "any"} if choice == "any" else {"type": "tool", "name": choice}
    assert bound.kwargs["tool_choice"] == expected


@pytest.mark.parametrize("thinking", [None, {"type": "adaptive"}])
@pytest.mark.parametrize("method", ["function_calling", "json_schema"])
def test_structured_output_with_adaptive_thinking(
    thinking: dict[str, str] | None, method: Any
) -> None:
    model = ChatAnthropic(model=MODEL, api_key="test", thinking=thinking)
    schema = {"title": "answer", "type": "object", "properties": {}}
    structured = cast(
        "RunnableSequence", model.with_structured_output(schema, method=method)
    )
    bound = cast("RunnableBinding", structured.first)
    payload = model._get_request_payload("hello", **bound.kwargs)
    if method == "function_calling":
        assert payload["tool_choice"] == {"type": "tool", "name": "answer"}
    else:
        assert payload["output_config"]["format"]["type"] == "json_schema"


def test_forced_filtered_tool_is_not_discarded_for_adaptive_thinking() -> None:
    model = ChatAnthropic(model=MODEL, api_key="test", thinking={"type": "adaptive"})
    invalid_tool = {
        "name": "invalid",
        "input_schema": {"type": "object", "oneOf": [{"type": "object"}]},
    }
    with (
        pytest.warns(UserWarning, match="invalid"),
        pytest.raises(ValueError, match="tool_choice forces"),
    ):
        model.bind_tools([TOOL, invalid_tool], tool_choice="invalid")


@pytest.mark.parametrize(
    "thinking",
    [
        {"type": "thinking", "thinking": "", "signature": "opaque-signature"},
        {"type": "redacted_thinking", "data": "opaque-data"},
    ],
)
async def test_default_thinking_stream_aggregate_replay(
    thinking: dict[str, str],
) -> None:
    from anthropic._models import construct_type

    raw_events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": thinking,
        },
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "hello"},
        },
    ]
    events = [
        construct_type(type_=RawMessageStreamEvent, value=event) for event in raw_events
    ]
    model = ChatAnthropic(model=MODEL, api_key="test")
    with patch.object(
        ChatAnthropic, "_create", return_value=MagicMock(parse=lambda: iter(events))
    ):
        chunks = [cast("AIMessageChunk", chunk) for chunk in model.stream("hello")]
    full = chunks[0]
    for chunk in chunks[1:]:
        full += chunk
    payload = model._get_request_payload(
        [HumanMessage("hello"), full, HumanMessage("next")]
    )
    assert payload["messages"][1]["content"] == [
        thinking,
        {"type": "text", "text": "hello"},
    ]

    async def async_events() -> Any:
        for event in events:
            yield event

    with patch.object(
        ChatAnthropic, "_acreate", return_value=MagicMock(parse=async_events)
    ):
        chunks = [
            cast("AIMessageChunk", chunk) async for chunk in model.astream("hello")
        ]
    full = chunks[0]
    for chunk in chunks[1:]:
        full += chunk
    payload = model._get_request_payload(
        [HumanMessage("hello"), full, HumanMessage("next")]
    )
    assert payload["messages"][1]["content"] == [
        thinking,
        {"type": "text", "text": "hello"},
    ]
