import unittest
import uuid
from typing import Optional, Union

import pytest

from langchain_core.documents import Document
from langchain_core.load import dumpd, load
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
    convert_to_openai_image_block,
    get_buffer_string,
    is_data_content_block,
    merge_content,
    message_chunk_to_message,
    message_to_dict,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.messages.tool import invalid_tool_call as create_invalid_tool_call
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.utils._merge import merge_lists


def test_message_init() -> None:
    for doc in [
        BaseMessage(type="foo", content="bar"),
        BaseMessage(type="foo", content="bar", id=None),
        BaseMessage(type="foo", content="bar", id="1"),
        BaseMessage(type="foo", content="bar", id=1),
    ]:
        assert isinstance(doc, BaseMessage)


def test_message_chunks() -> None:
    assert AIMessageChunk(content="I am", id="ai3") + AIMessageChunk(
        content=" indeed."
    ) == AIMessageChunk(content="I am indeed.", id="ai3")

    with pytest.raises(ValueError):
        AIMessageChunk(content="I am", id="ai3") + AIMessageChunk(
            content=" indeed.", id="ai4"
        )

    assert AIMessageChunk(content="I am") + AIMessageChunk(
        content=" indeed."
    ) == AIMessageChunk(content="I am indeed.")

    assert AIMessageChunk(content="I am") + AIMessageChunk(
        content=" indeed.", id="ai3"
    ) == AIMessageChunk(content="I am indeed.", id="ai3")

    assert AIMessageChunk(content="I am", id="ai3") + AIMessageChunk(
        content=" indeed.", id=None
    ) == AIMessageChunk(content="I am indeed.", id="ai3")

    assert AIMessageChunk(
        content="",
        additional_kwargs={"foo": "bar"},
        response_metadata={"hello": "world"},
        tool_calls=[create_tool_call(name="search", args={"query": "foo"}, id="1")],
        invalid_tool_calls=[create_invalid_tool_call(name="search", args="blah", id="2")],
        usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        id="ai3",
    ) + AIMessageChunk(
        content="I am indeed.",
        additional_kwargs={"foo": "baz", "key": "value"},
        response_metadata={"hello": "universe", "key": "value"},
        tool_calls=[create_tool_call(name="search", args={"query": "bar"}, id="3")],
        invalid_tool_calls=[create_invalid_tool_call(name="search", args="oops", id="4")],
        usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    ) == AIMessageChunk(
        content="I am indeed.",
        additional_kwargs={"foo": "baz", "key": "value"},
        response_metadata={"hello": "universe", "key": "value"},
        tool_calls=[
            create_tool_call(name="search", args={"query": "foo"}, id="1"),
            create_tool_call(name="search", args={"query": "bar"}, id="3"),
        ],
        invalid_tool_calls=[
            create_invalid_tool_call(name="search", args="blah", id="2"),
            create_invalid_tool_call(name="search", args="oops", id="4"),
        ],
        usage_metadata={"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
        id="ai3",
    )


def test_tool_message_chunks() -> None:
    chunk_1 = ToolMessage(
        content="",
        tool_call_id="1",
        name="search",
        artifact={"logs": ["foo"]},
    )
    chunk_2 = ToolMessage(
        content="I am indeed.",
        tool_call_id="1",
        name="search",
        artifact={"logs": ["bar"]},
    )
    actual = chunk_1 + chunk_2
    expected = ToolMessage(
        content="I am indeed.",
        tool_call_id="1",
        name="search",
        artifact={"logs": ["foo", "bar"]},
    )
    assert actual == expected


def test_system_message_chunks() -> None:
    assert SystemMessageChunk(content="I am", id="ai3") + SystemMessageChunk(
        content=" indeed."
    ) == SystemMessageChunk(content="I am indeed.", id="ai3")

    with pytest.raises(ValueError):
        SystemMessageChunk(content="I am", id="ai3") + SystemMessageChunk(
            content=" indeed.", id="ai4"
        )

    assert SystemMessageChunk(content="I am") + SystemMessageChunk(
        content=" indeed."
    ) == SystemMessageChunk(content="I am indeed.")

    assert SystemMessageChunk(content="I am") + SystemMessageChunk(
        content=" indeed.", id="ai3"
    ) == SystemMessageChunk(content="I am indeed.", id="ai3")

    assert SystemMessageChunk(content="I am", id="ai3") + SystemMessageChunk(
        content=" indeed.", id=None
    ) == SystemMessageChunk(content="I am indeed.", id="ai3")


def test_human_message_chunks() -> None:
    assert HumanMessageChunk(content="I am", id="ai3") + HumanMessageChunk(
        content=" indeed."
    ) == HumanMessageChunk(content="I am indeed.", id="ai3")

    with pytest.raises(ValueError):
        HumanMessageChunk(content="I am", id="ai3") + HumanMessageChunk(
            content=" indeed.", id="ai4"
        )

    assert HumanMessageChunk(content="I am") + HumanMessageChunk(
        content=" indeed."
    ) == HumanMessageChunk(content="I am indeed.")

    assert HumanMessageChunk(content="I am") + HumanMessageChunk(
        content=" indeed.", id="ai3"
    ) == HumanMessageChunk(content="I am indeed.", id="ai3")

    assert HumanMessageChunk(content="I am", id="ai3") + HumanMessageChunk(
        content=" indeed.", id=None
    ) == HumanMessageChunk(content="I am indeed.", id="ai3")


def test_chat_message_chunks() -> None:
    assert ChatMessageChunk(content="I am", role="foo", id="ai3") + ChatMessageChunk(
        content=" indeed.", role="foo"
    ) == ChatMessageChunk(content="I am indeed.", role="foo", id="ai3")

    with pytest.raises(ValueError):
        ChatMessageChunk(content="I am", role="foo", id="ai3") + ChatMessageChunk(
            content=" indeed.", role="foo", id="ai4"
        )

    with pytest.raises(ValueError):
        ChatMessageChunk(content="I am", role="foo") + ChatMessageChunk(
            content=" indeed.", role="bar"
        )

    assert ChatMessageChunk(content="I am", role="foo") + ChatMessageChunk(
        content=" indeed.", role="foo"
    ) == ChatMessageChunk(content="I am indeed.", role="foo")

    assert ChatMessageChunk(content="I am", role="foo") + ChatMessageChunk(
        content=" indeed.", role="foo", id="ai3"
    ) == ChatMessageChunk(content="I am indeed.", role="foo", id="ai3")

    assert ChatMessageChunk(content="I am", role="foo", id="ai3") + ChatMessageChunk(
        content=" indeed.", role="foo", id=None
    ) == ChatMessageChunk(content="I am indeed.", role="foo", id="ai3")


def test_function_message_chunks() -> None:
    assert FunctionMessageChunk(
        content="I am", name="foo", id="ai3"
    ) + FunctionMessageChunk(content=" indeed.", name="foo") == FunctionMessageChunk(
        content="I am indeed.", name="foo", id="ai3"
    )

    with pytest.raises(ValueError):
        FunctionMessageChunk(
            content="I am", name="foo", id="ai3"
        ) + FunctionMessageChunk(content=" indeed.", name="foo", id="ai4")

    with pytest.raises(ValueError):
        FunctionMessageChunk(content="I am", name="foo") + FunctionMessageChunk(
            content=" indeed.", name="bar"
        )

    assert FunctionMessageChunk(content="I am", name="foo") + FunctionMessageChunk(
        content=" indeed.", name="foo"
    ) == FunctionMessageChunk(content="I am indeed.", name="foo")

    assert FunctionMessageChunk(content="I am", name="foo") + FunctionMessageChunk(
        content=" indeed.", name="foo", id="ai3"
    ) == FunctionMessageChunk(content="I am indeed.", name="foo", id="ai3")

    assert FunctionMessageChunk(
        content="I am", name="foo", id="ai3"
    ) + FunctionMessageChunk(
        content=" indeed.", name="foo", id=None
    ) == FunctionMessageChunk(
        content="I am indeed.", name="foo", id="ai3"
    )


def test_message_from_dict() -> None:
    result = messages_from_dict([])
    assert result == []

    result = messages_from_dict([{"type": "human", "content": "foo"}])
    assert result == [HumanMessage(content="foo")]

    result = messages_from_dict([{"role": "user", "content": "foo"}])
    assert result == [HumanMessage(content="foo")]

    # Test with remove message without `tool_call_id`
    result = messages_from_dict([{"type": "remove", "id": "foo"}])
    assert result == [RemoveMessage(id="foo")]

    # Test with remove message with `tool_call_id` -- should be ignored
    result = messages_from_dict([{"type": "remove", "id": "foo", "tool_call_id": "bar"}])
    assert result == [RemoveMessage(id="foo")]


def test_messages_to_dict() -> None:
    result = messages_to_dict([])
    assert result == []

    result = messages_to_dict([HumanMessage(content="foo")])
    assert result == [{"type": "human", "content": "foo", "id": None, "name": None}]


def test_message_chunk_to_message() -> None:
    chunk = AIMessageChunk(content="foo")
    result = message_chunk_to_message(chunk)
    assert result == AIMessage(content="foo", id=chunk.id)

    chunk = HumanMessageChunk(content="foo")
    result = message_chunk_to_message(chunk)
    assert result == HumanMessage(content="foo", id=chunk.id)

    chunk = ChatMessageChunk(content="foo", role="bar")
    result = message_chunk_to_message(chunk)
    assert result == ChatMessage(content="foo", role="bar", id=chunk.id)

    chunk = SystemMessageChunk(content="foo")
    result = message_chunk_to_message(chunk)
    assert result == SystemMessage(content="foo", id=chunk.id)

    chunk = FunctionMessageChunk(content="foo", name="bar")
    result = message_chunk_to_message(chunk)
    assert result == FunctionMessage(content="foo", name="bar", id=chunk.id)

    # Test correct types in chunks
    chunk = AIMessageChunk(
        content="foo",
        tool_calls=[create_tool_call(name="search", args={"query": "foo"}, id="1")],
        invalid_tool_calls=[create_invalid_tool_call(name="search", args="blah", id="2")],
        usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    )
    result = message_chunk_to_message(chunk)
    assert result == AIMessage(
        content="foo",
        tool_calls=[create_tool_call(name="search", args={"query": "foo"}, id="1")],
        invalid_tool_calls=[
            create_invalid_tool_call(name="search", args="blah", id="2")
        ],
        usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        id=chunk.id,
    )


def test_tool_call() -> None:
    tool_call = create_tool_call(name="search", args={"query": "foo"}, id="1")
    assert tool_call["name"] == "search"
    assert tool_call["args"] == {"query": "foo"}
    assert tool_call["id"] == "1"


def test_invalid_tool_call() -> None:
    invalid_tool_call = create_invalid_tool_call(name="search", args="blah", id="2")
    assert invalid_tool_call["name"] == "search"
    assert invalid_tool_call["args"] == "blah"
    assert invalid_tool_call["id"] == "2"


def test_tool_call_chunk() -> None:
    tool_call_chunk = create_tool_call_chunk(
        name="search", args='{"query": "foo"}', id="1", index=0
    )
    assert tool_call_chunk["name"] == "search"
    assert tool_call_chunk["args"] == '{"query": "foo"}'
    assert tool_call_chunk["id"] == "1"
    assert tool_call_chunk["index"] == 0


def test_message_to_dict() -> None:
    message = HumanMessage(content="foo")
    result = message_to_dict(message)
    expected = {"type": "human", "content": "foo", "id": message.id, "name": None}
    assert result == expected

    message = AIMessage(content="foo")
    result = message_to_dict(message)
    expected = {"type": "ai", "content": "foo", "id": message.id, "name": None}
    assert result == expected

    message = SystemMessage(content="foo")
    result = message_to_dict(message)
    expected = {"type": "system", "content": "foo", "id": message.id, "name": None}
    assert result == expected

    message = ChatMessage(content="foo", role="bar")
    result = message_to_dict(message)
    expected = {"type": "chat", "content": "foo", "id": message.id, "name": None, "role": "bar"}
    assert result == expected

    message = FunctionMessage(content="foo", name="bar")
    result = message_to_dict(message)
    expected = {"type": "function", "content": "foo", "id": message.id, "name": "bar"}
    assert result == expected

    message = ToolMessage(content="foo", tool_call_id="bar")
    result = message_to_dict(message)
    expected = {"type": "tool", "content": "foo", "id": message.id, "name": None, "tool_call_id": "bar"}
    assert result == expected


def test_get_buffer_string() -> None:
    messages = [
        HumanMessage(content="foo"),
        AIMessage(content="bar"),
    ]
    result = get_buffer_string(messages)
    expected = "Human: foo\nAI: bar"
    assert result == expected

    result = get_buffer_string(messages, human_prefix="HUMAN", ai_prefix="AI")
    expected = "HUMAN: foo\nAI: bar"
    assert result == expected


def test_serialize() -> None:
    messages = [
        HumanMessage(content="foo"),
        AIMessage(content="bar"),
    ]

    # Test that we can serialize and deserialize
    serialized = [message_to_dict(message) for message in messages]
    deserialized = messages_from_dict(serialized)
    assert messages == deserialized

    # Test serialization with dumpd/load
    serialized = dumpd(messages)
    deserialized = load(serialized)
    assert messages == deserialized


class TestConvertToMessages(unittest.TestCase):
    def test_convert_to_messages(self) -> None:
        # Test with BaseMessage objects
        messages = [
            HumanMessage(content="foo"),
            AIMessage(content="bar"),
        ]
        result = convert_to_messages(messages)
        assert result == messages

        # Test with dict
        dict_messages = [
            {"type": "human", "content": "foo"},
            {"type": "ai", "content": "bar"},
        ]
        result = convert_to_messages(dict_messages)
        expected = [
            HumanMessage(content="foo"),
            AIMessage(content="bar"),
        ]
        assert result == expected

        # Test with strings
        string_messages = ["foo", "bar"]
        result = convert_to_messages(string_messages)
        expected = [
            HumanMessage(content="foo"),
            HumanMessage(content="bar"),
        ]
        assert result == expected

        # Test with tuples
        tuple_messages = [("human", "foo"), ("ai", "bar")]
        result = convert_to_messages(tuple_messages)
        expected = [
            HumanMessage(content="foo"),
            AIMessage(content="bar"),
        ]
        assert result == expected

        # Test mixed inputs
        mixed_messages = [
            HumanMessage(content="foo"),
            ("ai", "bar"),
            {"type": "human", "content": "baz"},
            "qux",
        ]
        result = convert_to_messages(mixed_messages)
        expected = [
            HumanMessage(content="foo"),
            AIMessage(content="bar"),
            HumanMessage(content="baz"),
            HumanMessage(content="qux"),
        ]
        assert result == expected

    def test_convert_to_messages_openai_role(self) -> None:
        # Test with openai role
        dict_messages = [
            {"role": "user", "content": "foo"},
            {"role": "assistant", "content": "bar"},
            {"role": "system", "content": "baz"},
        ]
        result = convert_to_messages(dict_messages)
        expected = [
            HumanMessage(content="foo"),
            AIMessage(content="bar"),
            SystemMessage(content="baz"),
        ]
        assert result == expected


def test_merge_content() -> None:
    text_1 = "foo"
    text_2 = "bar"
    result = merge_content(text_1, text_2)
    assert result == "foobar"

    list_1 = [{"type": "text", "text": "foo"}]
    list_2 = [{"type": "text", "text": "bar"}]
    result = merge_content(list_1, list_2)
    expected = [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]
    assert result == expected

    list_1 = [{"type": "text", "text": "foo"}]
    text_2 = "bar"
    result = merge_content(list_1, text_2)
    expected = [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]
    assert result == expected

    text_1 = "foo"
    list_2 = [{"type": "text", "text": "bar"}]
    result = merge_content(text_1, list_2)
    expected = [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]
    assert result == expected


def test_content_blocks() -> None:
    content_block = {"type": "text", "text": "foo"}
    assert not is_data_content_block(content_block)

    content_block = {"type": "image", "data": "foo"}
    assert is_data_content_block(content_block)

    content_block = {"type": "image_url", "image_url": "foo"}
    assert not is_data_content_block(content_block)

    content_block = {"type": "tool_result", "tool_use_id": "foo"}
    assert is_data_content_block(content_block)


def test_merge_lists() -> None:
    left = [1, 2, 3]
    right = [4, 5, 6]
    result = merge_lists(left, right)
    expected = [1, 2, 3, 4, 5, 6]
    assert result == expected

    left = []
    right = [1, 2, 3]
    result = merge_lists(left, right)
    expected = [1, 2, 3]
    assert result == expected

    left = [1, 2, 3]
    right = []
    result = merge_lists(left, right)
    expected = [1, 2, 3]
    assert result == expected


def test_anthropic_image_url_block() -> None:
    """Test converting Anthropic image_url block to OpenAI format."""
    input_block = {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,<base64 data>"},
    }
    expected = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,<base64 data>",
        },
    }
    result = convert_to_openai_image_block(input_block)
    assert result == expected


def test_anthropic_text_block() -> None:
    """Test converting Anthropic text block to OpenAI format."""
    input_block = {"type": "text", "text": "Hello, world!"}
    expected = input_block
    result = convert_to_openai_image_block(input_block)
    assert result == expected


def test_anthropic_image_block_base64() -> None:
    """Test converting Anthropic image block with base64 source to OpenAI format."""
    input_block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "<base64 data>",
        },
    }
    expected = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,<base64 data>",
        },
    }
    result = convert_to_openai_image_block(input_block)
    assert result == expected


def test_anthropic_image_block_url() -> None:
    """Test converting Anthropic image block with URL source to OpenAI format."""
    input_block = {
        "type": "image",
        "source": {
            "type": "url",
            "url": "https://example.com/image.jpg",
        },
    }
    expected = {
        "type": "image_url",
        "image_url": {
            "url": "https://example.com/image.jpg",
        },
    }
    result = convert_to_openai_image_block(input_block)
    assert result == expected


def test_legacy_anthropic_image_block_direct() -> None:
    """Test converting legacy Anthropic image block format to OpenAI format."""
    input_block = {
        "type": "image",
        "source_type": "base64",
        "data": "<base64 data>",
        "mime_type": "image/jpeg",
    }
    expected = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,<base64 data>",
        },
    }
    result = convert_to_openai_image_block(input_block)
    assert result == expected


def test_legacy_anthropic_image_block_cached() -> None:
    """Test converting legacy Anthropic image block with cache control to OpenAI format."""
    input_block = {
        "type": "image",
        "source_type": "base64",
        "data": "<base64 data>",
        "mime_type": "image/jpeg",
        "cache_control": {"type": "ephemeral"},
    }
    expected = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,<base64 data>",
        },
    }
    result = convert_to_openai_image_block(input_block)
    assert result == expected


# Tests for ToolMessage status field preservation (issue #32835)
def test_tool_message_status_error_preserved() -> None:
    """Test that status='error' is preserved during convert_to_messages."""
    # Create ToolMessage with error status
    original_message = ToolMessage(
        content="Error: please fix your mistakes",
        tool_call_id="foobar",
        status="error",
    )

    # Convert to dict (simulating database storage)
    message_dict = original_message.model_dump()

    # Convert back using convert_to_messages
    recovered_message = convert_to_messages([message_dict])[0]

    # Status should be preserved
    assert original_message.status == recovered_message.status
    assert recovered_message.status == "error"


def test_tool_message_status_success_preserved() -> None:
    """Test that status='success' is preserved during convert_to_messages."""
    original_message = ToolMessage(
        content="Operation completed successfully",
        tool_call_id="success_call",
        status="success",
    )

    message_dict = original_message.model_dump()
    recovered_message = convert_to_messages([message_dict])[0]

    assert recovered_message.status == "success"


def test_tool_message_default_status_when_missing() -> None:
    """Test that default status is used when not specified."""
    # Create message dict without status field
    message_dict = {
        "content": "Default status test",
        "type": "tool",
        "tool_call_id": "default_call",
    }

    recovered_message = convert_to_messages([message_dict])[0]

    # Should default to "success"
    assert recovered_message.status == "success"


def test_tool_message_all_fields_preserved() -> None:
    """Test that all ToolMessage fields are preserved, not just status."""
    original_message = ToolMessage(
        content="Complete field test",
        tool_call_id="field_test",
        status="error",
        artifact={"test": "data"},
        additional_kwargs={"custom": "value"},
        response_metadata={"source": "test"},
    )

    message_dict = original_message.model_dump()
    recovered_message = convert_to_messages([message_dict])[0]

    # Check all fields are preserved
    assert recovered_message.content == original_message.content
    assert recovered_message.tool_call_id == original_message.tool_call_id
    assert recovered_message.status == original_message.status
    assert recovered_message.artifact == original_message.artifact
    assert recovered_message.response_metadata == original_message.response_metadata

    # additional_kwargs should be preserved correctly (not contain status)
    assert "status" not in recovered_message.additional_kwargs
    assert recovered_message.additional_kwargs.get("custom") == "value"


def test_tool_message_issue_32835_reproduction() -> None:
    """Exact reproduction of issue #32835 from GitHub."""
    tool_message = ToolMessage(
        content="Error: please fix your mistakes",
        tool_call_id="foobar",
        status="error",
    )
    tool_message_json = tool_message.model_dump()
    tool_message_recovered = convert_to_messages([tool_message_json])[0]

    # This assertion should pass with the fix
    assert tool_message.status == tool_message_recovered.status, (
        f'received "{tool_message_recovered.status}", '
        f'expected "{tool_message.status}"'
    )


def test_tool_message_multiple_messages_conversion() -> None:
    """Test that status is preserved when converting multiple messages."""
    messages = [
        ToolMessage(content="Success 1", tool_call_id="call1", status="success"),
        ToolMessage(content="Error 1", tool_call_id="call2", status="error"),
        ToolMessage(content="Success 2", tool_call_id="call3", status="success"),
    ]

    # Convert to dicts and back
    message_dicts = [msg.model_dump() for msg in messages]
    recovered_messages = convert_to_messages(message_dicts)

    # Check each message status is preserved
    expected_statuses = ["success", "error", "success"]
    for i, (_original, recovered, expected) in enumerate(
        zip(messages, recovered_messages, expected_statuses)
    ):
        assert recovered.status == expected, (
            f"Message {i}: expected {expected}, got {recovered.status}"
        )