"""Tests for _strip_ids_from_messages functionality."""

from typing import Any, Union, cast

from langchain_core.language_models.chat_models import _strip_ids_from_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.utils import LC_AUTO_PREFIX


def test_strip_auto_generated_message_ids() -> None:
    """Test stripping auto-generated IDs from message level."""
    messages = [
        HumanMessage("Hello", id=f"{LC_AUTO_PREFIX}12345"),
        AIMessage("Hi there", id="user_provided_id"),
        SystemMessage("System prompt", id=f"{LC_AUTO_PREFIX}67890"),
        HumanMessage("Another message"),
        # (Human; no ID specified at creation, but one would be auto-generated)
    ]

    result = _strip_ids_from_messages(messages)

    # Auto-generated IDs should be removed, user IDs preserved
    assert result[0].id is None
    assert result[1].id == "user_provided_id"
    assert result[2].id is None
    assert result[3].id is None


def test_strip_auto_generated_content_block_ids() -> None:
    """Test stripping auto-generated IDs from content blocks."""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "World", "id": f"{LC_AUTO_PREFIX}abc123"},
        {"type": "image", "url": "http://example.com/image.jpg", "id": "user_id"},
        {"type": "text", "text": "End", "id": f"{LC_AUTO_PREFIX}def456"},
    ]

    messages = [AIMessage(content=cast("list[Union[str, dict[Any, Any]]]", content))]
    result = _strip_ids_from_messages(messages)

    content_result = cast("list[dict[str, Any]]", result[0].content)
    assert "id" not in content_result[0]  # No ID originally
    assert "id" not in content_result[1]  # Auto-generated ID removed
    assert content_result[2]["id"] == "user_id"  # User ID preserved
    assert "id" not in content_result[3]  # Auto-generated ID removed


def test_preserve_non_auto_generated_ids() -> None:
    """Test that non-auto-generated IDs are preserved on all message types."""
    messages = [
        HumanMessage("Hello", id="custom_id_123"),
        AIMessage("Hi", id="another_custom_id"),
        SystemMessage("System", id="system_custom"),
    ]

    result = _strip_ids_from_messages(messages)

    assert result[0].id == "custom_id_123"
    assert result[1].id == "another_custom_id"
    assert result[2].id == "system_custom"


def test_handle_mixed_content_types() -> None:
    """Test handling of mixed content including tool calls."""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Let me search for that"},
        {
            "type": "tool_call",
            "id": f"{LC_AUTO_PREFIX}tool_123",
            "name": "search",
            "args": {"query": "test"},
        },
    ]

    messages = [AIMessage(content=cast("list[Union[str, dict[Any, Any]]]", content))]
    result = _strip_ids_from_messages(messages)

    result_content = cast("list[dict[str, Any]]", result[0].content)
    assert "id" not in result_content[1]  # Tool call auto-generated ID removed


def test_handle_tool_messages() -> None:
    """Test handling of `ToolMessage` objects. NOT tool_call_id."""
    messages = [
        ToolMessage(
            "Tool result", tool_call_id="call_123", id=f"{LC_AUTO_PREFIX}tool_456"
        ),
        ToolMessage("Another result", tool_call_id="call_456", id="user_tool_id"),
    ]

    result = _strip_ids_from_messages(messages)

    # Enhanced behavior: tool_call_ids with LC_AUTO_PREFIX are now stripped
    assert result[0].id is None  # Auto-generated ID stripped
    assert result[1].id == "user_tool_id"  # User ID preserved


def test_content_with_nested_structures() -> None:
    """Test handling of content with nested dictionaries."""
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "Hello",
            "id": f"{LC_AUTO_PREFIX}nested_123",
            "metadata": {
                "nested": {
                    "id": f"{LC_AUTO_PREFIX}should_not_be_touched",
                    "value": "test",
                }
            },
        }
    ]

    messages = [AIMessage(content=cast("list[Union[str, dict[Any, Any]]]", content))]
    result = _strip_ids_from_messages(messages)

    result_content_list = cast("list[dict[str, Any]]", result[0].content)
    result_content = result_content_list[0]

    # Top-level auto-generated ID should be removed
    assert "id" not in result_content

    # Nested IDs should not be touched (current behavior)
    nested_metadata = cast("dict[str, Any]", result_content["metadata"]["nested"])
    assert nested_metadata["id"] == f"{LC_AUTO_PREFIX}should_not_be_touched"


def test_id_edge_cases() -> None:
    """Test edge cases for ID handling."""
    messages = [
        # Empty string ID
        HumanMessage("Empty ID", id=""),
        # ID that starts with LC_AUTO_PREFIX but has more
        AIMessage("Prefix plus", id=f"{LC_AUTO_PREFIX}custom_suffix"),
        # ID that contains but doesn't start with LC_AUTO_PREFIX
        SystemMessage("Contains prefix", id=f"prefix_{LC_AUTO_PREFIX}_suffix"),
    ]

    result = _strip_ids_from_messages(messages)

    assert result[0].id == ""  # Empty string preserved
    assert result[1].id is None  # Starts with prefix, removed
    assert (
        result[2].id == f"prefix_{LC_AUTO_PREFIX}_suffix"
    )  # Contains but doesn't start, preserved


def test_strip_all_ids_parameter() -> None:
    """Test the `strip_all_ids` for comprehensive ID removal."""
    messages = [
        HumanMessage("Hello", id="user_custom_id"),
        AIMessage("Hi", id=f"{LC_AUTO_PREFIX}auto_id"),
        SystemMessage("System", id="another_custom_id"),
        ToolMessage("Tool result", tool_call_id="call_789", id="custom_tool_id"),
    ]

    # Default behavior: only strip LC_AUTO_PREFIX IDs
    result_default = _strip_ids_from_messages(messages)
    assert result_default[0].id == "user_custom_id"  # Preserved
    assert result_default[1].id is None  # Auto-generated, removed
    assert result_default[2].id == "another_custom_id"  # Preserved
    assert result_default[3].id == "custom_tool_id"  # Preserved

    # Aggressive mode: strip ALL IDs
    result_all = _strip_ids_from_messages(messages, strip_all_ids=True)
    assert result_all[0].id is None  # Custom ID removed
    assert result_all[1].id is None  # Auto-generated ID removed
    assert result_all[2].id is None  # Custom ID removed
    assert result_all[3].id is None  # Custom ID removed


def test_content_blocks() -> None:
    """Test handling of various content block types."""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Hello", "id": f"{LC_AUTO_PREFIX}text_123"},
        {"type": "image", "url": "http://example.com/img.jpg", "id": "user_image_id"},
        {
            "type": "tool_call",
            "id": f"{LC_AUTO_PREFIX}tool_456",
            "name": "search",
            "args": {},
        },
        {"type": "file", "id": "user_file_789", "url": "http://example.com/file.pdf"},
    ]

    messages = [AIMessage(content=cast("list[Union[str, dict[Any, Any]]]", content))]
    result = _strip_ids_from_messages(messages)

    result_content = cast("list[dict[str, Any]]", result[0].content)
    assert "id" not in result_content[0]  # Text block auto-generated ID removed
    assert result_content[1]["id"] == "user_image_id"  # Image user ID preserved
    assert "id" not in result_content[2]  # Tool call auto-generated ID removed
    assert result_content[3]["id"] == "user_file_789"  # File user ID preserved


def test_equivalent_inputs_produce_same_cache_key() -> None:
    """Test that equivalent inputs (ignoring IDs) yield the same cache normalization."""
    # Two messages with same content but different auto-generated IDs
    messages1 = [
        HumanMessage("Hello world", id=f"{LC_AUTO_PREFIX}12345"),
        AIMessage("Hi there", id="user_provided_id"),
    ]

    messages2 = [
        HumanMessage("Hello world", id=f"{LC_AUTO_PREFIX}67890"),  # Different auto ID
        AIMessage("Hi there", id="user_provided_id"),  # Same user ID
    ]

    # Both should normalize to the same thing for caching
    normalized1 = _strip_ids_from_messages(messages1)
    normalized2 = _strip_ids_from_messages(messages2)

    # Verify they produce identical cache keys
    assert len(normalized1) == len(normalized2)
    assert normalized1[0].content == normalized2[0].content
    assert normalized1[0].id == normalized2[0].id is None  # Both auto IDs stripped
    assert normalized1[1].content == normalized2[1].content
    assert (
        normalized1[1].id == normalized2[1].id == "user_provided_id"
    )  # User ID preserved


def test_equivalent_message_histories_normalize_identically() -> None:
    """Test that equivalent message histories produce the same cache normalization."""
    # First conversation with auto-generated IDs
    history1 = [
        SystemMessage("You are a helpful assistant", id=f"{LC_AUTO_PREFIX}sys1"),
        HumanMessage("What is 2+2?", id=f"{LC_AUTO_PREFIX}human1"),
        AIMessage("4", id="custom_ai_id"),
        HumanMessage("Thanks!", id=f"{LC_AUTO_PREFIX}human2"),
    ]

    # Second conversation with different auto-generated IDs but same content
    history2 = [
        SystemMessage("You are a helpful assistant", id=f"{LC_AUTO_PREFIX}sys999"),
        HumanMessage("What is 2+2?", id=f"{LC_AUTO_PREFIX}human888"),
        AIMessage("4", id="custom_ai_id"),  # Same custom ID
        HumanMessage("Thanks!", id=f"{LC_AUTO_PREFIX}human777"),
    ]

    # Normalize both for caching
    normalized1 = _strip_ids_from_messages(history1)
    normalized2 = _strip_ids_from_messages(history2)

    # Should be identical after normalization
    assert len(normalized1) == len(normalized2)
    for msg1, msg2 in zip(normalized1, normalized2):
        assert msg1.content == msg2.content
        assert msg1.id == msg2.id  # Should match after ID stripping

    # Verify specific ID handling
    assert normalized1[0].id is None  # Auto system ID stripped
    assert normalized1[1].id is None  # Auto human ID stripped
    assert normalized1[2].id == "custom_ai_id"  # Custom AI ID preserved
    assert normalized1[3].id is None  # Auto human ID stripped
