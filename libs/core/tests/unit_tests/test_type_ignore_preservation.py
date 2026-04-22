"""Preservation property tests for AIMessage/AIMessageChunk runtime behavior.

These tests verify baseline behavior on UNFIXED code and must PASS before
any type-ignore cleanup changes are applied.

**Validates: Requirements 3.1, 3.2, 3.3**
"""

import json
from typing import Any

import pytest

from langchain_core.messages.ai import AIMessage, AIMessageChunk


# ---------------------------------------------------------------------------
# content_blocks preservation (lines 286, 288)
# ---------------------------------------------------------------------------

# Generate multiple tool call configurations: index only, extras only, both, neither
_TOOL_CALL_CONFIGS: list[tuple[str, dict[str, Any]]] = [
    (
        "index_only",
        {
            "name": "get_weather",
            "args": {"city": "Paris"},
            "id": "tc_idx",
            "type": "tool_call",
            "index": 0,
        },
    ),
    (
        "extras_only",
        {
            "name": "search",
            "args": {"q": "langchain"},
            "id": "tc_ext",
            "type": "tool_call",
            "extras": {"provider": "openai"},
        },
    ),
    (
        "both_index_and_extras",
        {
            "name": "calculate",
            "args": {"expr": "1+1"},
            "id": "tc_both",
            "type": "tool_call",
            "index": 2,
            "extras": {"model": "gpt-4"},
        },
    ),
    (
        "neither_index_nor_extras",
        {
            "name": "echo",
            "args": {"text": "hi"},
            "id": "tc_none",
            "type": "tool_call",
        },
    ),
    (
        "index_string_type",
        {
            "name": "lookup",
            "args": {"key": "abc"},
            "id": "tc_str_idx",
            "type": "tool_call",
            "index": "block-0",
        },
    ),
    (
        "extras_nested",
        {
            "name": "deep_tool",
            "args": {},
            "id": "tc_nested",
            "type": "tool_call",
            "extras": {"meta": {"nested": True, "level": 2}},
        },
    ),
]


@pytest.mark.parametrize(
    ("label", "tool_call_dict"),
    _TOOL_CALL_CONFIGS,
    ids=[c[0] for c in _TOOL_CALL_CONFIGS],
)
def test_content_blocks_preserves_index_and_extras(
    label: str, tool_call_dict: dict[str, Any]
) -> None:
    """content_blocks returns blocks with correct index/extras values.

    Uses model_construct to bypass validators and inject tool_calls with
    extra fields that the ToolCall TypedDict (tool.py) doesn't declare.
    """
    msg = AIMessage.model_construct(
        content="test",
        tool_calls=[tool_call_dict],
        response_metadata={},
        additional_kwargs={},
        invalid_tool_calls=[],
        usage_metadata=None,
        type="ai",
        name=None,
        id=None,
    )

    blocks = msg.content_blocks
    tc_blocks = [b for b in blocks if b.get("type") == "tool_call"]

    assert len(tc_blocks) == 1, f"Expected 1 tool_call block, got {len(tc_blocks)}"
    block = tc_blocks[0]

    # Core fields always present
    assert block["name"] == tool_call_dict["name"]
    assert block["args"] == tool_call_dict["args"]
    assert block["id"] == tool_call_dict["id"]

    # index preserved when present
    if "index" in tool_call_dict:
        assert block["index"] == tool_call_dict["index"]
    else:
        assert "index" not in block

    # extras preserved when present
    if "extras" in tool_call_dict:
        assert block["extras"] == tool_call_dict["extras"]
    else:
        assert "extras" not in block



# ---------------------------------------------------------------------------
# init_tool_calls preservation (line 586)
# ---------------------------------------------------------------------------

_INIT_TOOL_CALLS_CONFIGS: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    (
        "extras_simple",
        {
            "type": "tool_call_chunk",
            "id": "tc1",
            "name": "foo",
            "args": json.dumps({"a": 1}),
            "index": 0,
            "extras": {"provider_key": "val1"},
        },
        {"provider_key": "val1"},
    ),
    (
        "extras_nested_dict",
        {
            "type": "tool_call_chunk",
            "id": "tc2",
            "name": "bar",
            "args": json.dumps({"x": "y"}),
            "index": 1,
            "extras": {"meta": {"nested": True}},
        },
        {"meta": {"nested": True}},
    ),
    (
        "extras_empty_dict",
        {
            "type": "tool_call_chunk",
            "id": "tc3",
            "name": "baz",
            "args": json.dumps({}),
            "index": 0,
            "extras": {},
        },
        {},
    ),
]


@pytest.mark.parametrize(
    ("label", "content_block", "expected_extras"),
    _INIT_TOOL_CALLS_CONFIGS,
    ids=[c[0] for c in _INIT_TOOL_CALLS_CONFIGS],
)
def test_init_tool_calls_preserves_extras(
    label: str,
    content_block: dict[str, Any],
    expected_extras: dict[str, Any],
) -> None:
    """init_tool_calls replaces tool_call_chunk with parsed ToolCall and preserves extras.

    Constructs AIMessageChunk with chunk_position="last",
    response_metadata={"output_version": "v1"}, and content containing
    tool_call_chunk blocks with extras.
    """
    chunk = AIMessageChunk(
        content=[content_block],
        tool_call_chunks=[
            {
                "name": content_block["name"],
                "args": content_block["args"],
                "id": content_block["id"],
                "index": content_block.get("index"),
                "type": "tool_call_chunk",
            }
        ],
        chunk_position="last",
        response_metadata={"output_version": "v1"},
    )

    # Content should be mutated: tool_call_chunk replaced with tool_call
    assert isinstance(chunk.content, list)
    assert len(chunk.content) == 1

    result_block = chunk.content[0]
    assert isinstance(result_block, dict)
    assert result_block["type"] == "tool_call"
    assert result_block["name"] == content_block["name"]
    assert result_block["id"] == content_block["id"]
    assert isinstance(result_block["args"], dict)

    # extras preserved
    assert result_block.get("extras") == expected_extras


def test_init_tool_calls_no_extras_no_key() -> None:
    """When tool_call_chunk has no extras, the replaced block has no extras key."""
    chunk = AIMessageChunk(
        content=[
            {
                "type": "tool_call_chunk",
                "id": "tc_no_extras",
                "name": "plain",
                "args": json.dumps({"k": "v"}),
                "index": 0,
            }
        ],
        tool_call_chunks=[
            {
                "name": "plain",
                "args": json.dumps({"k": "v"}),
                "id": "tc_no_extras",
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
        chunk_position="last",
        response_metadata={"output_version": "v1"},
    )

    result_block = chunk.content[0]
    assert isinstance(result_block, dict)
    assert result_block["type"] == "tool_call"
    assert "extras" not in result_block


# ---------------------------------------------------------------------------
# init_server_tool_calls preservation (lines 613-614)
# ---------------------------------------------------------------------------

_SERVER_TOOL_CALL_CONFIGS: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    (
        "simple_args",
        {
            "type": "server_tool_call_chunk",
            "id": "stc1",
            "name": "web_search",
            "args": json.dumps({"query": "test"}),
        },
        {"query": "test"},
    ),
    (
        "nested_args",
        {
            "type": "server_tool_call_chunk",
            "id": "stc2",
            "name": "code_exec",
            "args": json.dumps({"code": "print(1)", "env": {"timeout": 30}}),
        },
        {"code": "print(1)", "env": {"timeout": 30}},
    ),
    (
        "empty_args",
        {
            "type": "server_tool_call_chunk",
            "id": "stc3",
            "name": "noop",
            "args": json.dumps({}),
        },
        {},
    ),
    (
        "server_tool_call_type_also_parsed",
        {
            "type": "server_tool_call",
            "id": "stc4",
            "name": "search",
            "args": json.dumps({"q": "hello"}),
        },
        {"q": "hello"},
    ),
]


@pytest.mark.parametrize(
    ("label", "content_block", "expected_args"),
    _SERVER_TOOL_CALL_CONFIGS,
    ids=[c[0] for c in _SERVER_TOOL_CALL_CONFIGS],
)
def test_init_server_tool_calls_parses_args(
    label: str,
    content_block: dict[str, Any],
    expected_args: dict[str, Any],
) -> None:
    """init_server_tool_calls parses string args to dict and sets type to server_tool_call.

    Constructs AIMessageChunk with chunk_position="last",
    response_metadata={"output_version": "v1"}, and content containing
    server_tool_call_chunk blocks with string args.
    """
    chunk = AIMessageChunk(
        content=[content_block],
        chunk_position="last",
        response_metadata={"output_version": "v1"},
    )

    assert isinstance(chunk.content, list)
    assert len(chunk.content) == 1

    result_block = chunk.content[0]
    assert isinstance(result_block, dict)
    assert result_block["type"] == "server_tool_call"
    assert result_block["args"] == expected_args
    assert isinstance(result_block["args"], dict)


def test_init_server_tool_calls_invalid_json_unchanged() -> None:
    """When args is not valid JSON, the block is left unchanged."""
    chunk = AIMessageChunk(
        content=[
            {
                "type": "server_tool_call_chunk",
                "id": "stc_bad",
                "name": "broken",
                "args": "{not valid json",
            }
        ],
        chunk_position="last",
        response_metadata={"output_version": "v1"},
    )

    result_block = chunk.content[0]
    assert isinstance(result_block, dict)
    assert result_block["type"] == "server_tool_call_chunk"
    assert result_block["args"] == "{not valid json"


def test_init_server_tool_calls_non_dict_json_unchanged() -> None:
    """When args parses to non-dict JSON (e.g. a list), the block is left unchanged."""
    chunk = AIMessageChunk(
        content=[
            {
                "type": "server_tool_call_chunk",
                "id": "stc_list",
                "name": "list_tool",
                "args": json.dumps([1, 2, 3]),
            }
        ],
        chunk_position="last",
        response_metadata={"output_version": "v1"},
    )

    result_block = chunk.content[0]
    assert isinstance(result_block, dict)
    # type should remain unchanged since args is not a dict
    assert result_block["type"] == "server_tool_call_chunk"
    assert result_block["args"] == "[1, 2, 3]"


# ---------------------------------------------------------------------------
# Intentional ignores remain (lines 420, 619)
# ---------------------------------------------------------------------------


def test_ai_message_chunk_type_is_ai_message_chunk() -> None:
    """AIMessageChunk.type is 'AIMessageChunk' (line 420 ignore preserved).

    This is an intentional Liskov substitution violation for deserialization.
    """
    chunk = AIMessageChunk(content="hello")
    assert chunk.type == "AIMessageChunk"


def test_ai_message_chunk_type_field_default() -> None:
    """The type field default is 'AIMessageChunk' across various constructions."""
    # Empty content
    assert AIMessageChunk(content="").type == "AIMessageChunk"
    # List content
    assert AIMessageChunk(content=["a", "b"]).type == "AIMessageChunk"
    # With tool_call_chunks
    assert AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": "t", "args": "{}", "id": "1", "index": 0, "type": "tool_call_chunk"}
        ],
    ).type == "AIMessageChunk"


@pytest.mark.parametrize(
    ("left_content", "right_content"),
    [
        ("hello ", "world"),
        ("", "test"),
        ("abc", ""),
    ],
    ids=["both_nonempty", "left_empty", "right_empty"],
)
def test_ai_message_chunk_add_returns_ai_message_chunk(
    left_content: str, right_content: str
) -> None:
    """AIMessageChunk.__add__ returns AIMessageChunk (line 619 ignore preserved).

    This is an intentional override of BaseMessage.__add__ which returns
    ChatPromptTemplate.
    """
    left = AIMessageChunk(content=left_content)
    right = AIMessageChunk(content=right_content)
    result = left + right

    assert isinstance(result, AIMessageChunk)
    assert result.content == left_content + right_content
