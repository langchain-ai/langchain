"""Unit tests for converting `OpenAIDecisions` state to API input."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

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


def test_mixed_roles_are_native() -> None:
    assert to_decision_input(
        [SystemMessage("Support chat."), HumanMessage("Prod down"), AIMessage("On it")]
    ) == [
        {"role": "system", "content": "Support chat."},
        {"role": "user", "content": "Prod down"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "On it"}],
        },
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
    assert result[1] == {
        "type": "function_call",
        "name": "read_file",
        "arguments": "{}",
        "call_id": "c1",
    }
    assert result[2] == {
        "type": "function_call_output",
        "call_id": "c1",
        "output": [
            {"type": "input_image", "image_url": f"data:image/png;base64,{IMAGE_DATA}"}
        ],
    }
    assert result[3]["content"][1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,BBBB",
    }


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


def test_hosted_images_nested_in_json_are_images() -> None:
    result = to_decision_input(
        {
            "messages": [
                AIMessage("Hi"),
                HumanMessage(
                    content=[{"type": "image", "url": "https://example.com/a.png"}]
                ),
            ]
        }
    )
    assert isinstance(result, list)
    assert result[0]["content"][1]["image_url"] == "https://example.com/a.png"


def test_content_resembling_an_image_token_is_left_alone() -> None:
    result = to_decision_input(
        {"messages": [AIMessage("lc-image-0123-0"), HumanMessage("Hi")]}
    )

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


@pytest.mark.parametrize(
    "message",
    [
        SystemMessage("Rules", additional_kwargs={"__openai_role__": "developer"}),
        ChatMessage(content="Rules", role="developer"),
    ],
)
def test_developer_messages_are_native(message: Any) -> None:
    assert to_decision_input(message) == [{"role": "developer", "content": "Rules"}]


def test_assistant_refusal_and_phase_are_native() -> None:
    message = AIMessage(
        content=[
            {"type": "output_text", "text": "Checking", "phase": "commentary"},
            {"type": "refusal", "refusal": "Cannot help"},
        ]
    )
    result = to_decision_input(message)
    assert isinstance(result, list)
    assert result[0]["content"][0]["type"] == "output_text"
    assert result[0]["phase"] == "commentary"
    assert result[1]["content"] == [{"type": "refusal", "refusal": "Cannot help"}]


def test_assistant_additional_refusal_is_preserved() -> None:
    result = to_decision_input(
        AIMessage("", additional_kwargs={"refusal": "Cannot help"})
    )
    assert isinstance(result, list)
    assert result[-1]["content"] == [{"type": "refusal", "refusal": "Cannot help"}]


def test_custom_tool_calls_and_outputs_preserve_call_ids() -> None:
    call = {
        "type": "custom_tool_call",
        "name": "shell",
        "input": "ls",
        "call_id": "call_1",
    }
    result = to_decision_input(
        [
            AIMessage(content=[call]),
            ToolMessage(
                content=[{"type": "custom_tool_call_output", "output": "files"}],
                tool_call_id="call_1",
            ),
        ]
    )
    assert result == [
        call,
        {"type": "custom_tool_call_output", "output": "files", "call_id": "call_1"},
    ]


def test_function_call_blocks_are_not_duplicated() -> None:
    call = {
        "type": "function_call",
        "name": "read",
        "arguments": "{}",
        "call_id": "c1",
        "id": "fc_1",
        "status": "completed",
        "namespace": "filesystem",
    }
    assert to_decision_input(
        AIMessage(content=[call], tool_calls=[{"name": "read", "args": {}, "id": "c1"}])
    ) == [
        {
            "type": "function_call",
            "name": "read",
            "arguments": "{}",
            "call_id": "c1",
            "namespace": "filesystem",
        }
    ]


@pytest.mark.parametrize(
    "block",
    [
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        {"type": "input_file", "file_id": "file_1"},
        {"type": "item_reference", "id": "msg_1"},
    ],
)
@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
def test_unsupported_native_blocks_are_rejected(
    block: dict[str, Any], role: str
) -> None:
    message = (
        HumanMessage(content=[block])
        if role == "user"
        else AIMessage(content=[block])
        if role == "assistant"
        else ToolMessage(content=[block], tool_call_id="c1")
    )
    with pytest.raises(ValueError, match="Unsupported content block"):
        to_decision_input(message)


def test_system_images_are_rejected() -> None:
    message = SystemMessage(
        content=[
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
        ]
    )
    with pytest.raises(ValueError, match="only accept text"):
        to_decision_input(message)


def test_function_calls_require_call_ids() -> None:
    message = AIMessage("", tool_calls=[{"name": "read", "args": {}, "id": None}])
    with pytest.raises(ValueError, match="call ID"):
        to_decision_input(message)


def test_native_user_json_content_block() -> None:
    assert to_decision_input(
        HumanMessage(
            content=[
                {
                    "type": "input_json",
                    "data": {"value": 1},
                }
            ]
        )
    ) == [{"role": "user", "content": [{"type": "input_json", "data": {"value": 1}}]}]


def test_oversized_user_json_content_falls_back_to_text() -> None:
    state = {"value": "x" * (256 * 1024)}
    result = to_decision_input(
        HumanMessage(content=[{"type": "input_json", "data": state}])
    )
    assert json.loads(_json_text(result)) == state


@pytest.mark.parametrize("role", ["user", "developer", "tool"])
def test_native_text_cache_breakpoints(role: str) -> None:
    block = {
        "type": "input_text",
        "text": "Cached",
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }
    message = (
        HumanMessage(content=[block])
        if role == "user"
        else ChatMessage(content=[block], role="developer")
        if role == "developer"
        else ToolMessage(content=[block], tool_call_id="c1")
    )
    result = to_decision_input(message)
    assert isinstance(result, list)
    assert result[0]["output" if role == "tool" else "content"] == [block]


@pytest.mark.parametrize("cache_breakpoint", [None, True, {}, {"mode": "automatic"}])
def test_invalid_cache_breakpoints_are_rejected(cache_breakpoint: Any) -> None:
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Cached",
                "prompt_cache_breakpoint": cache_breakpoint,
            }
        ]
    )
    with pytest.raises(ValueError, match="prompt_cache_breakpoint"):
        to_decision_input(message)


def test_assistant_output_omits_response_only_fields() -> None:
    message = AIMessage(
        content=[
            {
                "type": "output_text",
                "text": "Hello",
                "annotations": [],
                "id": "msg_1",
                "phase": "final_answer",
            }
        ]
    )
    assert to_decision_input(message) == [
        {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "Hello"}],
        }
    ]


def test_custom_tool_output_normalizes_text_and_images() -> None:
    cache = {"mode": "explicit"}
    message = ToolMessage(
        content=[
            {
                "type": "custom_tool_call_output",
                "output": [
                    {
                        "type": "text",
                        "text": "Result",
                        "prompt_cache_breakpoint": cache,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/a.png",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        tool_call_id="c1",
    )
    assert to_decision_input(message) == [
        {
            "type": "custom_tool_call_output",
            "call_id": "c1",
            "output": [
                {
                    "type": "input_text",
                    "text": "Result",
                    "prompt_cache_breakpoint": cache,
                },
                {
                    "type": "input_image",
                    "image_url": "https://example.com/a.png",
                    "detail": "low",
                },
            ],
        }
    ]


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize(
    "block",
    [
        {"type": "input_json", "data": {"value": 1}},
        {"type": "input_file", "file_id": "file_1"},
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        {"type": "item_reference", "id": "msg_1"},
    ],
)
def test_tool_output_rejects_unsupported_blocks(
    custom: bool, block: dict[str, Any]
) -> None:
    content: Any = (
        [{"type": "custom_tool_call_output", "output": [block]}] if custom else [block]
    )
    message = ToolMessage(content=content, tool_call_id="c1")
    with pytest.raises(ValueError, match="Unsupported content block"):
        to_decision_input(message)


@pytest.mark.parametrize("size", [1, 256 * 1024])
def test_json_cache_breakpoints_rejected_before_fallback(size: int) -> None:
    message = HumanMessage(
        content=[
            {
                "type": "input_json",
                "data": {"value": "x" * size},
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        ]
    )
    with pytest.raises(ValueError, match="cache breakpoints"):
        to_decision_input(message)
