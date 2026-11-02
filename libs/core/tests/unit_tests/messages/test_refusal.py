"""Standard refusal blocks and provider translations."""

import pytest

from langchain_core.exceptions import ModelRefusalError
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    RefusalContentBlock,
    get_refusal,
)

STOP_DETAILS = {
    "type": "refusal",
    "category": "cyber",
    "explanation": "Blocked under the usage policy.",
}


@pytest.mark.parametrize("message_cls", [AIMessage, AIMessageChunk])
def test_openai_legacy_refusal(message_cls: type[AIMessage]) -> None:
    message = message_cls(
        content="",
        additional_kwargs={"refusal": "I can't help with that."},
        response_metadata={"model_provider": "openai"},
    )
    assert message.content_blocks == [
        {"type": "refusal", "refusal": "I can't help with that."}
    ]


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


def test_anthropic_refusal_from_stop_details() -> None:
    message = AIMessage(
        content=[],
        response_metadata={
            "model_provider": "anthropic",
            "stop_reason": "refusal",
            "stop_details": STOP_DETAILS,
        },
    )
    assert message.content_blocks == [
        {
            "type": "refusal",
            "refusal": "Blocked under the usage policy.",
            "category": "cyber",
            "index": "lc_refusal",
        }
    ]


def test_anthropic_refusal_without_stop_details() -> None:
    message = AIMessage(
        content=[],
        response_metadata={"model_provider": "anthropic", "stop_reason": "refusal"},
    )
    assert message.content_blocks == [
        {"type": "refusal", "refusal": "", "index": "lc_refusal"}
    ]


def test_anthropic_partial_output_kept_before_refusal() -> None:
    message = AIMessage(
        content="Here is",
        response_metadata={"model_provider": "anthropic", "stop_reason": "refusal"},
    )
    assert message.content_blocks == [
        {"type": "text", "text": "Here is"},
        {"type": "refusal", "refusal": "", "index": "lc_refusal"},
    ]


def test_anthropic_streamed_refusal() -> None:
    first = AIMessageChunk(
        content=[{"type": "text", "text": "Here is", "index": 0}],
        response_metadata={"model_provider": "anthropic"},
    )
    last = AIMessageChunk(
        content=[],
        response_metadata={
            "model_provider": "anthropic",
            "stop_reason": "refusal",
            "stop_details": STOP_DETAILS,
        },
        chunk_position="last",
    )
    assert get_refusal(last) == get_refusal(first + last)
    assert [block["type"] for block in (first + last).content_blocks] == [
        "text",
        "refusal",
    ]


def test_ordinary_text_not_refused() -> None:
    message = AIMessage(
        "I cannot help.",
        response_metadata={"model_provider": "anthropic", "stop_reason": "end_turn"},
    )
    assert get_refusal(message) is None


def test_get_refusal_ignores_non_ai_messages() -> None:
    assert get_refusal(HumanMessage("hi")) is None


def test_error_from_refusal() -> None:
    message = AIMessage(
        content=[],
        response_metadata={
            "model_provider": "anthropic",
            "stop_reason": "refusal",
            "stop_details": STOP_DETAILS,
        },
    )
    refusal = get_refusal(message)
    assert refusal is not None
    error = ModelRefusalError.from_refusal(refusal, raw_message=message)
    assert error.provider == "anthropic"
    assert error.stop_reason == "refusal"
    assert error.category == "cyber"
    assert error.explanation == "Blocked under the usage policy."
    assert error.raw_message is message


def test_error_from_refusal_without_text() -> None:
    block: RefusalContentBlock = {"type": "refusal", "refusal": ""}
    message = AIMessage(content_blocks=[block])
    error = ModelRefusalError.from_refusal(block, raw_message=message)
    assert error.explanation is None
    assert str(error) == "Model refused to produce the requested output"


def test_anthropic_native_refusal_block_not_duplicated() -> None:
    block = {"type": "refusal", "refusal": "No."}
    message = AIMessage(
        content=[block],
        response_metadata={"model_provider": "anthropic", "stop_reason": "refusal"},
    )
    assert message.content_blocks == [block]


@pytest.mark.parametrize("stop_details", [None, "unexpected", ["refusal"]])
def test_anthropic_refusal_with_malformed_stop_details(stop_details: object) -> None:
    message = AIMessage(
        content=[],
        response_metadata={
            "model_provider": "anthropic",
            "stop_reason": "refusal",
            "stop_details": stop_details,
        },
    )
    assert get_refusal(message) == {
        "type": "refusal",
        "refusal": "",
        "index": "lc_refusal",
    }


def test_get_refusal_reads_legacy_non_standard_block() -> None:
    message = AIMessage(
        content=[
            {
                "type": "non_standard",
                "value": {"type": "refusal", "refusal": "I can't help with that."},
            }
        ],
        response_metadata={"output_version": "v1"},
    )
    assert get_refusal(message) == {
        "type": "refusal",
        "refusal": "I can't help with that.",
    }


def test_error_from_openai_refusal_uses_finish_reason() -> None:
    message = AIMessage(
        content="",
        additional_kwargs={"refusal": "I can't help with that."},
        response_metadata={"model_provider": "openai", "finish_reason": "stop"},
    )
    refusal = get_refusal(message)
    assert refusal is not None
    error = ModelRefusalError.from_refusal(refusal, raw_message=message)
    assert error.provider == "openai"
    assert error.stop_reason == "stop"
    assert error.explanation == "I can't help with that."
