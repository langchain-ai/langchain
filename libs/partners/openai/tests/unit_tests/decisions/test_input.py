"""Unit tests for converting `OpenAIDecisions` state to API input."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_openai.decisions._input import to_decision_input

IMAGE_DATA = "iVBORw0KGgo="


def test_string_is_sent_unchanged() -> None:
    assert to_decision_input("hello") == "hello"


def test_human_messages_become_user_messages() -> None:
    result = to_decision_input([HumanMessage("first"), HumanMessage("second")])

    assert result == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]


def test_single_human_message_with_image() -> None:
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Inspect this."},
            {"type": "image", "base64": IMAGE_DATA, "mime_type": "image/png"},
        ]
    )

    assert to_decision_input(message) == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Inspect this."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{IMAGE_DATA}",
                },
            ],
        }
    ]


def test_hosted_image_url_is_rejected() -> None:
    message = HumanMessage(
        content=[{"type": "image", "url": "https://example.com/a.png"}]
    )

    with pytest.raises(ValueError, match="base64 data URLs"):
        to_decision_input(message)


def test_unsupported_content_block_is_rejected() -> None:
    message = HumanMessage(
        content=[
            {
                "type": "file",
                "base64": "AAAA",
                "mime_type": "application/pdf",
                "filename": "a.pdf",
            }
        ]
    )

    with pytest.raises(ValueError, match="Unsupported content block"):
        to_decision_input(message)


def test_mixed_roles_are_serialized_as_json_text() -> None:
    result = to_decision_input(
        [SystemMessage("Support chat."), HumanMessage("Prod down"), AIMessage("On it")]
    )

    assert isinstance(result, str)
    assert json.loads(result) == [
        {"role": "system", "content": "Support chat."},
        {"role": "user", "content": "Prod down"},
        {"role": "assistant", "content": "On it"},
    ]


def test_nested_messages_in_json_are_serialized() -> None:
    result = to_decision_input(
        {"conversation": [HumanMessage("Help!")], "tier": "enterprise"}
    )

    assert isinstance(result, str)
    assert json.loads(result) == {
        "conversation": [{"role": "user", "content": "Help!"}],
        "tier": "enterprise",
    }


def test_non_json_state_is_rejected() -> None:
    with pytest.raises(TypeError, match="Unsupported"):
        to_decision_input({"value": object()})  # type: ignore[dict-item]
