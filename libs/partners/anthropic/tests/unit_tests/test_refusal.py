"""Standard refusal blocks for Anthropic responses and requests."""

import pytest
from anthropic.types import Message, RawMessageDeltaEvent
from langchain_core.messages import AIMessage, HumanMessage, get_refusal

from langchain_anthropic import ChatAnthropic

STOP_DETAILS = {
    "type": "refusal",
    "category": "cyber",
    "explanation": "Blocked under the usage policy.",
}


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
    assert get_refusal(message) == {
        "type": "refusal",
        "refusal": "Blocked under the usage policy.",
        "category": "cyber",
        "index": "lc_refusal",
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
    refusal = get_refusal(chunk)
    assert refusal is not None
    assert refusal["category"] == "cyber"
    assert (chunk.usage_metadata is not None) is stream_usage
