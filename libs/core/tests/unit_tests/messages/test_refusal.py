"""Explicit refusal detection without changing provider payloads."""

from copy import deepcopy
from typing import Any

import pytest

from langchain_core.exceptions import ModelError, ModelRefusalError
from langchain_core.load import dumpd, load
from langchain_core.messages import AIMessage, AIMessageChunk, message_chunk_to_message


@pytest.mark.parametrize("message_cls", [AIMessage, AIMessageChunk])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"response_metadata": {"stop_reason": "refusal"}},
        {"additional_kwargs": {"refusal": "Not allowed"}},
        {"content": [{"type": "refusal", "refusal": ""}]},
        {
            "content": [
                {"type": "non_standard", "value": {"type": "refusal", "refusal": "No"}}
            ]
        },
        {"response_metadata": {"finish_reason": "content_filter"}},
        {"response_metadata": {"incomplete_details": {"reason": "content_filter"}}},
        *[
            {"response_metadata": {"finish_reason": reason}}
            for reason in [
                "SAFETY",
                "RECITATION",
                "BLOCKLIST",
                "PROHIBITED_CONTENT",
                "SPII",
                "IMAGE_SAFETY",
                "IMAGE_PROHIBITED_CONTENT",
                "IMAGE_RECITATION",
                "MODEL_ARMOR",
            ]
        ],
        *[
            {"response_metadata": {"prompt_feedback": {"block_reason": reason}}}
            for reason in [
                "SAFETY",
                "OTHER",
                "BLOCKLIST",
                "PROHIBITED_CONTENT",
                "IMAGE_SAFETY",
                "JAILBREAK",
                1,
                2,
                3,
                4,
            ]
        ],
    ],
)
def test_explicit_refusal(kwargs: dict[str, Any], message_cls: type[AIMessage]) -> None:
    original = deepcopy(kwargs)
    message = message_cls(**{"content": "", **kwargs})
    assert message.response_metadata["refusal"] is True
    assert message.content == original.get("content", "")
    assert message.additional_kwargs == original.get("additional_kwargs", {})
    assert {
        k: v for k, v in message.response_metadata.items() if k != "refusal"
    } == original.get("response_metadata", {})
    assert kwargs == original


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"content": "I cannot help with that request."},
        {"additional_kwargs": {"refusal": None}},
        {"additional_kwargs": {"refusal": ""}},
        {"response_metadata": {"status": "incomplete"}},
        {"response_metadata": {"incomplete_details": {"reason": "max_output_tokens"}}},
        *[
            {"response_metadata": {"finish_reason": reason}}
            for reason in [
                "MAX_TOKENS",
                "OTHER",
                "STOP",
                "FINISH_REASON_UNSPECIFIED",
                0,
                ["SAFETY"],
            ]
        ],
        *[
            {"response_metadata": {"prompt_feedback": {"block_reason": reason}}}
            for reason in [
                None,
                0,
                False,
                True,
                "BLOCK_REASON_UNSPECIFIED",
                "UNKNOWN",
            ]
        ],
    ],
)
def test_not_refused(kwargs: dict[str, Any]) -> None:
    assert "refusal" not in AIMessage(**{"content": "", **kwargs}).response_metadata


def test_chunk_aggregation_and_serialization() -> None:
    first = AIMessageChunk(content="", additional_kwargs={"refusal": "Not "})
    last = AIMessageChunk(content="", additional_kwargs={"refusal": "allowed"})
    message = message_chunk_to_message(AIMessageChunk(content="") + first + last)
    assert message.response_metadata["refusal"] is True
    assert message.additional_kwargs["refusal"] == "Not allowed"
    assert AIMessage.model_validate_json(message.model_dump_json()) == message
    assert load(dumpd(message), allowed_objects=[AIMessage]) == message


def test_model_refusal_error() -> None:
    message = AIMessage("Private refusal text", response_metadata={"refusal": True})
    error = ModelRefusalError(message)
    assert isinstance(error, ModelError)
    assert error.is_retryable is False
    assert error.message is message
    assert str(error) == "The model refused to generate a response."
    assert "Private refusal text" not in repr(error)
