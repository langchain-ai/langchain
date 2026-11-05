"""Standard refusal blocks for Anthropic responses and requests."""

from unittest.mock import patch

import pytest
from anthropic.types import Message, RawMessageDeltaEvent
from langchain_core.exceptions import ModelRefusalError
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RefusalContentBlock,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

from langchain_anthropic import ChatAnthropic

STOP_DETAILS = {
    "type": "refusal",
    "category": "cyber",
    "explanation": "Blocked under the usage policy.",
}


def _refusal(message: BaseMessage) -> RefusalContentBlock | None:
    for block in message.content_blocks:
        if block["type"] == "refusal":
            return block
    return None


@pytest.mark.parametrize("output_version", [None, "v1"])
def test_refusal_dropped_from_request(output_version: str | None) -> None:
    model = ChatAnthropic(model="test", api_key="test")
    refused = AIMessage(
        content_blocks=[
            {"type": "text", "text": "Here is"},
            {"type": "refusal", "refusal": "", "index": "lc_refusal"},
        ],
        response_metadata={
            "model_provider": "anthropic",
            "output_version": output_version,
        },
    )
    payload = model._get_request_payload(
        [HumanMessage("hi"), refused, HumanMessage("next")]
    )
    assistant = payload["messages"][1]
    assert assistant["role"] == "assistant"
    assert all(block["type"] != "refusal" for block in assistant["content"])


@pytest.mark.parametrize("output_version", [None, "v1"])
def test_refusal_in_formatted_output(output_version: str | None) -> None:
    model = ChatAnthropic(model="test", api_key="test", output_version=output_version)
    data = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [],
            "stop_reason": "refusal",
            "stop_sequence": None,
            "stop_details": STOP_DETAILS,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    message = model._format_output(data).generations[0].message
    assert _refusal(message) == {
        "type": "refusal",
        "refusal": "Blocked under the usage policy.",
        "extras": {"category": "cyber"},
    }


@pytest.mark.parametrize("stream_usage", [True, False])
def test_streamed_refusal_independent_of_stream_usage(*, stream_usage: bool) -> None:
    model = ChatAnthropic(model="test", api_key="test")
    event = RawMessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": "refusal",
                "stop_sequence": None,
                "stop_details": STOP_DETAILS,
            },
            "usage": {"output_tokens": 0},
        }
    )
    chunk, _ = model._make_message_chunk_from_anthropic_event(
        event, stream_usage=stream_usage, coerce_content_to_string=False
    )
    assert chunk is not None
    refusal = _refusal(chunk)
    assert refusal is not None
    assert refusal.get("extras") == {"category": "cyber"}
    assert (chunk.usage_metadata is not None) is stream_usage


@pytest.mark.parametrize(
    "model_kwargs",
    [
        pytest.param({"model": "claude-sonnet-5-5"}, id="no-forced-tool-model"),
        pytest.param(
            {
                "model": "claude-haiku-4-5",
                "max_tokens": 2048,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
            },
            id="thinking-enabled",
        ),
    ],
)
@pytest.mark.parametrize("include_raw", [False, True])
@pytest.mark.filterwarnings("ignore:You are attempting to use structured output")
def test_structured_output_without_forced_tool_raises_refusal(
    model_kwargs: dict, *, include_raw: bool
) -> None:
    """A refusal is not reported as a missing tool call."""

    class Answer(BaseModel):
        answer: str

    model = ChatAnthropic(api_key="test", **model_kwargs)  # type: ignore[arg-type]
    refused = AIMessage(
        content=[],
        response_metadata={
            "model_provider": "anthropic",
            "stop_reason": "refusal",
            "stop_details": STOP_DETAILS,
        },
    )
    structured = model.with_structured_output(Answer, include_raw=include_raw)
    result = ChatResult(generations=[ChatGeneration(message=refused)])
    with patch.object(ChatAnthropic, "_generate", return_value=result):
        if include_raw:
            output = structured.invoke("hi")
            assert isinstance(output, dict)
            assert isinstance(output["parsing_error"], ModelRefusalError)
        else:
            with pytest.raises(ModelRefusalError, match="category='cyber'"):
                structured.invoke("hi")
