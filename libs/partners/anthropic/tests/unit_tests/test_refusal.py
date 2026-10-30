"""Replay standardized refusal messages to Anthropic."""

import pytest
from anthropic.types import Message
from langchain_core.messages import AIMessage

from langchain_anthropic import ChatAnthropic


@pytest.mark.parametrize("output_version", [None, "v1"])
def test_refusal_replay(output_version: str | None) -> None:
    model = ChatAnthropic(model="test", api_key="test")
    message = AIMessage(
        content_blocks=[{"type": "refusal", "refusal": "blocked"}],
        response_metadata={
            "model_provider": "anthropic",
            "output_version": output_version,
        },
    )
    payload = model._get_request_payload([message])
    assert payload["messages"][0]["content"] == [{"type": "text", "text": "blocked"}]


@pytest.mark.parametrize("output_version", [None, "v1"])
def test_refusal_available_before_output_conversion(output_version: str | None) -> None:
    model = ChatAnthropic(model="test", api_key="test", output_version=output_version)
    data = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [{"type": "text", "text": "blocked"}],
            "stop_reason": "refusal",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    message = model._format_output(data).generations[0].message
    assert message.content_blocks == [{"type": "refusal", "refusal": "blocked"}]
