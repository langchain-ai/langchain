"""Standard refusal blocks and provider translations."""

import pytest

from langchain_core.exceptions import ModelRefusalError
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    RefusalContentBlock,
    convert_to_openai_messages,
)

STOP_DETAILS = {
    "type": "refusal",
    "category": "cyber",
    "explanation": "Blocked under the usage policy.",
}


def _refusal(message: AIMessage) -> RefusalContentBlock | None:
    for block in message.content_blocks:
        if block["type"] == "refusal":
            return block
    return None


@pytest.mark.parametrize("message_cls", [AIMessage, AIMessageChunk])
def test_openai_legacy_refusal(message_cls: type[AIMessage]) -> None:
    message = message_cls(
        content="",
        additional_kwargs={"refusal": "I can't help with that."},
        response_metadata={"model_provider": "openai"},
    )
    expected: RefusalContentBlock = {
        "type": "refusal",
        "refusal": "I can't help with that.",
    }
    assert message.content_blocks == [expected]


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
            "extras": {"category": "cyber"},
        }
    ]


def test_anthropic_refusal_without_stop_details() -> None:
    message = AIMessage(
        content=[],
        response_metadata={"model_provider": "anthropic", "stop_reason": "refusal"},
    )
    assert message.content_blocks == [{"type": "refusal", "refusal": ""}]


def test_anthropic_partial_output_kept_before_refusal() -> None:
    message = AIMessage(
        content="Here is",
        response_metadata={"model_provider": "anthropic", "stop_reason": "refusal"},
    )
    assert message.content_blocks == [
        {"type": "text", "text": "Here is"},
        {"type": "refusal", "refusal": ""},
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
    assert _refusal(last) == _refusal(first + last)
    assert [block["type"] for block in (first + last).content_blocks] == [
        "text",
        "refusal",
    ]


def test_ordinary_text_not_refused() -> None:
    message = AIMessage(
        "I cannot help.",
        response_metadata={"model_provider": "anthropic", "stop_reason": "end_turn"},
    )
    assert _refusal(message) is None


def test_error_from_refusal() -> None:
    message = AIMessage(
        content=[],
        response_metadata={
            "model_provider": "anthropic",
            "stop_reason": "refusal",
            "stop_details": STOP_DETAILS,
        },
    )
    refusal = _refusal(message)
    assert refusal is not None
    error = ModelRefusalError.from_refusal(refusal, raw_message=message)
    assert str(error) == (
        "Model refused to produce the requested output (category='cyber'): "
        "Blocked under the usage policy."
    )
    assert error.raw_message is message


def test_error_from_refusal_without_text() -> None:
    block: RefusalContentBlock = {"type": "refusal", "refusal": ""}
    message = AIMessage(content_blocks=[block])
    error = ModelRefusalError.from_refusal(block, raw_message=message)
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
    assert _refusal(message) == {
        "type": "refusal",
        "refusal": "",
    }


def test_error_from_openai_refusal() -> None:
    message = AIMessage(
        content="",
        additional_kwargs={"refusal": "I can't help with that."},
        response_metadata={"model_provider": "openai", "finish_reason": "stop"},
    )
    refusal = _refusal(message)
    assert refusal is not None
    error = ModelRefusalError.from_refusal(refusal, raw_message=message)
    assert str(error) == (
        "Model refused to produce the requested output: I can't help with that."
    )


def test_convert_to_openai_messages_refusal_blocks() -> None:
    message = AIMessage(
        content=[
            {"type": "text", "text": "Here is"},
            {"type": "refusal", "refusal": "", "index": "lc_refusal"},
            {"type": "refusal", "refusal": "No.", "extras": {"x": 1}, "index": 1},
        ]
    )
    assert convert_to_openai_messages(message)["content"] == [
        {"type": "text", "text": "Here is"},
        {"type": "refusal", "refusal": "No."},
    ]


@pytest.mark.parametrize("provider", ["google_genai", "google_vertexai"])
@pytest.mark.parametrize("finish_reason", ["SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"])
def test_google_blocked_candidate_refusal(provider: str, finish_reason: str) -> None:
    message = AIMessage(
        content=[],
        response_metadata={"model_provider": provider, "finish_reason": finish_reason},
    )
    assert _refusal(message) == {
        "type": "refusal",
        "refusal": "",
        "extras": {"finish_reason": finish_reason},
    }


def test_google_blocked_prompt_refusal() -> None:
    message = AIMessage(
        content="",
        response_metadata={
            "model_provider": "google_genai",
            "prompt_feedback": {"block_reason": "SAFETY", "safety_ratings": []},
        },
    )
    assert _refusal(message) == {
        "type": "refusal",
        "refusal": "",
        "extras": {"block_reason": "SAFETY"},
    }


def test_google_refusal_keeps_finish_message() -> None:
    message = AIMessage(
        content=[],
        response_metadata={
            "model_provider": "google_genai",
            "finish_reason": "SAFETY",
            "finish_message": "Blocked for safety.",
        },
    )
    refusal = _refusal(message)
    assert refusal is not None
    assert refusal["refusal"] == "Blocked for safety."


@pytest.mark.parametrize(
    "response_metadata",
    [
        {"finish_reason": "STOP"},
        {"finish_reason": "MAX_TOKENS"},
        {"finish_reason": "RECITATION"},
        {"finish_reason": "SPII"},
        {"prompt_feedback": {"block_reason": "BLOCKED_REASON_UNSPECIFIED"}},
        {"prompt_feedback": {"block_reason": None}},
    ],
)
def test_google_non_refusal_stops(response_metadata: dict[str, object]) -> None:
    message = AIMessage(
        content="Hi.",
        response_metadata={"model_provider": "google_genai", **response_metadata},
    )
    assert _refusal(message) is None


def test_google_streamed_refusal_aggregates_to_one_block() -> None:
    first = AIMessageChunk(
        content=[{"type": "text", "text": "Here is", "index": 0}],
        response_metadata={"model_provider": "google_genai"},
    )
    last = AIMessageChunk(
        content=[],
        response_metadata={"model_provider": "google_genai", "finish_reason": "SAFETY"},
    )
    assert [block["type"] for block in (first + last).content_blocks] == [
        "text",
        "refusal",
    ]
