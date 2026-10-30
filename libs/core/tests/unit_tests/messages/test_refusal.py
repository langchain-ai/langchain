"""Standard refusal blocks and provider translations."""

import pytest

from langchain_core.exceptions import ModelRefusalError
from langchain_core.messages import AIMessage, AIMessageChunk, RefusalContentBlock


@pytest.mark.parametrize("message_cls", [AIMessage, AIMessageChunk])
@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_refusal_translation(message_cls: type[AIMessage], provider: str) -> None:
    message = message_cls(
        content="blocked" if provider == "anthropic" else "",
        additional_kwargs={"refusal": "blocked"} if provider == "openai" else {},
        response_metadata={"model_provider": provider, "stop_reason": "refusal"},
    )
    assert message.content_blocks == [{"type": "refusal", "refusal": "blocked"}]
    assert "refusal" not in message.response_metadata


def test_openai_responses_refusal() -> None:
    message = AIMessage(
        content=[
            {
                "type": "refusal",
                "refusal": "blocked",
                "id": "msg_1",
                "phase": "final_answer",
                "index": 2,
            }
        ],
        response_metadata={"model_provider": "openai"},
    )
    assert message.content_blocks == [
        {
            "type": "refusal",
            "refusal": "blocked",
            "id": "msg_1",
            "index": 2,
            "extras": {"phase": "final_answer"},
        }
    ]


def test_empty_anthropic_refusal() -> None:
    message = AIMessage(
        content=[],
        response_metadata={"model_provider": "anthropic", "stop_reason": "refusal"},
    )
    assert message.content_blocks == [
        {"type": "refusal", "refusal": "", "index": "lc_refusal"}
    ]


def test_standard_block_and_error() -> None:
    block: RefusalContentBlock = {"type": "refusal", "refusal": "private text"}
    message = AIMessage(content_blocks=[block])
    assert message.content_blocks == [block]
    error = ModelRefusalError(message)
    assert error.message is message
    assert error.is_retryable is False
    assert "private text" not in str(error)


def test_ordinary_text_not_refused() -> None:
    message = AIMessage(
        "I cannot help.",
        response_metadata={"model_provider": "anthropic", "stop_reason": "end_turn"},
    )
    assert message.content_blocks == [{"type": "text", "text": "I cannot help."}]
