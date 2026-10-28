"""Unit tests for converting `OpenAIDecisions` state to API input."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_openai.decisions._input import to_decision_input

IMAGE_DATA = "iVBORw0KGgo="


def _json_text(result: str | list[dict[str, Any]]) -> str:
    """Return the text of a single user message with one text part."""
    assert isinstance(result, list)
    [message] = result
    assert message["role"] == "user"
    [part] = message["content"]
    assert part["type"] == "input_text"
    return part["text"]


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


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_hosted_images_are_supported(scheme: str) -> None:
    url = f"{scheme}://example.com/a.png"
    assert to_decision_input(HumanMessage(content=[{"type": "image", "url": url}])) == [
        {"role": "user", "content": [{"type": "input_image", "image_url": url}]}
    ]


def test_image_detail_and_cache_breakpoints_are_preserved() -> None:
    cache = {"mode": "explicit"}
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Inspect", "prompt_cache_breakpoint": cache},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/a.png", "detail": "original"},
                "prompt_cache_breakpoint": cache,
            },
        ]
    )
    assert to_decision_input(message) == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Inspect",
                    "prompt_cache_breakpoint": cache,
                },
                {
                    "type": "input_image",
                    "image_url": "https://example.com/a.png",
                    "detail": "original",
                    "prompt_cache_breakpoint": cache,
                },
            ],
        }
    ]


def test_non_http_image_url_is_rejected() -> None:
    message = HumanMessage(
        content=[{"type": "image_url", "image_url": {"url": "file:///tmp/a.png"}}]
    )
    with pytest.raises(ValueError, match="HTTP"):
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

    assert json.loads(_json_text(result)) == [
        {"role": "system", "content": "Support chat."},
        {"role": "user", "content": "Prod down"},
        {"role": "assistant", "content": "On it"},
    ]


def test_nested_messages_in_json_are_serialized() -> None:
    result = to_decision_input(
        {"conversation": [HumanMessage("Help!")], "tier": "enterprise"}
    )

    assert json.loads(_json_text(result)) == {
        "conversation": [{"role": "user", "content": "Help!"}],
        "tier": "enterprise",
    }


def test_non_json_state_is_rejected() -> None:
    with pytest.raises(TypeError, match="Unsupported"):
        to_decision_input({"value": object()})  # type: ignore[dict-item]


def test_images_in_mixed_conversations_are_kept_in_place() -> None:
    result = to_decision_input(
        [
            SystemMessage("Support chat."),
            AIMessage("", tool_calls=[{"id": "c1", "name": "read_file", "args": {}}]),
            ToolMessage(
                content=[
                    {"type": "image", "base64": IMAGE_DATA, "mime_type": "image/png"}
                ],
                tool_call_id="c1",
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Mine looks like this:"},
                    {"type": "image", "base64": "BBBB", "mime_type": "image/png"},
                ]
            ),
        ]
    )

    assert isinstance(result, list)
    [message] = result
    assert message["role"] == "user"
    parts = message["content"]
    assert [part["type"] for part in parts] == [
        "input_text",
        "input_image",
        "input_text",
        "input_image",
        "input_text",
    ]
    assert parts[1]["image_url"] == f"data:image/png;base64,{IMAGE_DATA}"
    assert parts[3]["image_url"] == "data:image/png;base64,BBBB"
    assert parts[0]["text"].endswith('"tool_call_id": "c1", "content": "')
    assert parts[2]["text"].endswith("Mine looks like this:\\n")
    text = "".join(part.get("text", "[IMAGE]") for part in parts)
    assert json.loads(text.replace("[IMAGE]", ""))[-1]["role"] == "user"


def test_images_nested_in_json_are_kept_in_place() -> None:
    result = to_decision_input(
        {
            "messages": [
                AIMessage("Hi"),
                HumanMessage(
                    content=[
                        {
                            "type": "image",
                            "base64": IMAGE_DATA,
                            "mime_type": "image/png",
                        }
                    ]
                ),
            ]
        }
    )

    assert isinstance(result, list)
    assert [part["type"] for part in result[0]["content"]] == [
        "input_text",
        "input_image",
        "input_text",
    ]


def test_hosted_image_urls_in_mixed_conversations_are_images() -> None:
    result = to_decision_input(
        [
            AIMessage("Hi"),
            HumanMessage(
                content=[{"type": "image", "url": "https://example.com/a.png"}]
            ),
        ]
    )

    assert isinstance(result, list)
    assert result[0]["content"][1]["image_url"] == "https://example.com/a.png"


def test_content_resembling_an_image_token_is_left_alone() -> None:
    result = to_decision_input([AIMessage("lc-image-0123-0"), HumanMessage("Hi")])

    assert "lc-image-0123-0" in _json_text(result)


@pytest.mark.parametrize(
    "state", [{}, [], [{"value": 1}], {"nested": [1, True, None, {"text": "é"}]}]
)
def test_structured_json_is_native(state: Any) -> None:
    assert to_decision_input(state) == [
        {"role": "user", "content": [{"type": "input_json", "data": state}]}
    ]


@pytest.mark.parametrize(
    "state", [[1, 2], ["text"], [{"key": 1}, 2], {"tuple": (1, 2)}]
)
def test_other_json_keeps_text_fallback(state: Any) -> None:
    assert json.loads(_json_text(to_decision_input(state))) == json.loads(
        json.dumps(state)
    )


@pytest.mark.parametrize(("levels", "native"), [(32, True), (33, False)])
def test_json_container_depth_limit(levels: int, native: bool) -> None:
    state: Any = "value"
    for _ in range(levels):
        state = {"nested": state}
    result = to_decision_input(state)
    assert isinstance(result, list)
    assert (result[0]["content"][0]["type"] == "input_json") is native
    if not native:
        assert json.loads(_json_text(result)) == state


@pytest.mark.parametrize(("count", "native"), [(9998, True), (9999, False)])
def test_json_value_count_limit(count: int, native: bool) -> None:
    state: Any = {"values": [0] * count}
    result = to_decision_input(state)
    assert isinstance(result, list)
    assert (result[0]["content"][0]["type"] == "input_json") is native


@pytest.mark.parametrize(("extra", "native"), [(0, True), (1, False)])
def test_json_compact_utf8_size_limit(extra: int, native: bool) -> None:
    state: Any = {"x": "é" * ((256 * 1024 - 8) // 2) + "x" * extra}
    assert (
        len(json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode())
        == 256 * 1024 + extra
    )
    result = to_decision_input(state)
    assert isinstance(result, list)
    assert (result[0]["content"][0]["type"] == "input_json") is native


def test_non_string_json_keys_are_rejected() -> None:
    state: Any = {1: "value"}
    with pytest.raises(TypeError, match="keys must be strings"):
        to_decision_input(state)
