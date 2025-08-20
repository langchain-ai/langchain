"""Tests for _strip_ids_from_messages functionality."""

from langchain_core.language_models.chat_models import _strip_ids_from_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.utils import LC_AUTO_PREFIX


def test_strip_auto_generated_message_ids() -> None:
    """Test stripping auto-generated IDs from message level."""
    messages = [
        HumanMessage("Hello", id=f"{LC_AUTO_PREFIX}12345"),
        AIMessage("Hi there", id="user_provided_id"),
        SystemMessage("System prompt", id=f"{LC_AUTO_PREFIX}67890"),
        HumanMessage("Another message"),  # No ID
    ]

    result = _strip_ids_from_messages(messages)

    # Auto-generated IDs should be removed, user IDs preserved
    assert result[0].id is None
    assert result[1].id == "user_provided_id"
    assert result[2].id is None
    assert result[3].id is None


def test_strip_auto_generated_content_block_ids() -> None:
    """Test stripping auto-generated IDs from content blocks."""
    content = [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "World", "id": f"{LC_AUTO_PREFIX}abc123"},
        {"type": "image", "url": "http://example.com/image.jpg", "id": "user_id"},
        {"type": "text", "text": "End", "id": f"{LC_AUTO_PREFIX}def456"},
    ]

    messages = [AIMessage(content)]
    result = _strip_ids_from_messages(messages)

    result_content = result[0].content
    assert "id" not in result_content[0]  # No ID originally
    assert "id" not in result_content[1]  # Auto-generated ID removed
    assert result_content[2]["id"] == "user_id"  # User ID preserved
    assert "id" not in result_content[3]  # Auto-generated ID removed


def test_preserve_non_auto_generated_ids() -> None:
    """Test that non-auto-generated IDs are preserved."""
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
    content = [
        {"type": "text", "text": "Let me search for that"},
        {
            "type": "tool_call",
            "id": f"{LC_AUTO_PREFIX}tool_123",
            "name": "search",
            "args": {"query": "test"},
        },
    ]

    messages = [AIMessage(content)]
    result = _strip_ids_from_messages(messages)

    result_content = result[0].content
    assert "id" not in result_content[1]  # Tool call auto-generated ID removed


def test_handle_tool_messages() -> None:
    """Test handling of ToolMessage objects."""
    messages = [
        ToolMessage("Tool result", tool_call_id=f"{LC_AUTO_PREFIX}tool_456"),
        ToolMessage("Another result", tool_call_id="user_tool_id"),
    ]

    result = _strip_ids_from_messages(messages)

    # Enhanced behavior: tool_call_ids with LC_AUTO_PREFIX are now stripped
    assert result[0].tool_call_id is None  # Auto-generated ID stripped
    assert result[1].tool_call_id == "user_tool_id"  # User ID preserved


def test_deep_copy_preserves_structure() -> None:
    """Test that deep copying preserves message structure."""
    original_content = {"type": "text", "text": "Hello", "metadata": {"key": "value"}}
    messages = [HumanMessage([original_content])]

    result = _strip_ids_from_messages(messages)

    # Should be different objects
    assert result[0] is not messages[0]
    assert result[0].content[0] is not original_content

    # But with same content
    assert result[0].content[0]["text"] == "Hello"
    assert result[0].content[0]["metadata"] == {"key": "value"}


def test_empty_messages() -> None:
    """Test handling of empty message list."""
    result = _strip_ids_from_messages([])
    assert result == []


def test_none_ids_preserved() -> None:
    """Test that None IDs are preserved as None."""
    messages = [
        HumanMessage("Hello", id=None),
        AIMessage("Hi"),
    ]

    result = _strip_ids_from_messages(messages)

    assert result[0].id is None
    assert result[1].id is None


def test_content_with_nested_structures() -> None:
    """Test handling of content with nested dictionaries."""
    content = [
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

    messages = [AIMessage(content)]
    result = _strip_ids_from_messages(messages)

    result_content = result[0].content[0]

    # Top-level auto-generated ID should be removed
    assert "id" not in result_content

    # Nested IDs should not be touched (current behavior)
    assert (
        result_content["metadata"]["nested"]["id"]
        == f"{LC_AUTO_PREFIX}should_not_be_touched"
    )


def test_string_content_unmodified() -> None:
    """Test that string content is not modified."""
    messages = [
        HumanMessage("Simple string message", id=f"{LC_AUTO_PREFIX}str_123"),
        AIMessage("Another string"),
    ]

    result = _strip_ids_from_messages(messages)

    assert result[0].content == "Simple string message"
    assert result[1].content == "Another string"
    assert result[0].id is None  # ID still stripped from message level


def test_preserves_message_attributes() -> None:
    """Test that other message attributes are preserved."""
    messages = [
        HumanMessage(
            "Hello",
            id=f"{LC_AUTO_PREFIX}preserve_123",
            additional_kwargs={"custom": "value"},
            response_metadata={"model": "test"},
        )
    ]

    result = _strip_ids_from_messages(messages)

    assert result[0].id is None  # ID stripped
    assert result[0].additional_kwargs == {"custom": "value"}
    assert result[0].response_metadata == {"model": "test"}
    assert result[0].content == "Hello"


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
    """Test the strip_all_ids parameter for comprehensive ID removal."""
    messages = [
        HumanMessage("Hello", id="user_custom_id"),
        AIMessage("Hi", id=f"{LC_AUTO_PREFIX}auto_id"),
        SystemMessage("System", id="another_custom_id"),
        ToolMessage("Tool result", tool_call_id="custom_tool_id"),
    ]

    # Default behavior: only strip LC_AUTO_PREFIX IDs
    result_default = _strip_ids_from_messages(messages)
    assert result_default[0].id == "user_custom_id"  # Preserved
    assert result_default[1].id is None  # Auto-generated, removed
    assert result_default[2].id == "another_custom_id"  # Preserved
    assert result_default[3].tool_call_id == "custom_tool_id"  # Preserved

    # Aggressive mode: strip ALL IDs
    result_all = _strip_ids_from_messages(messages, strip_all_ids=True)
    assert result_all[0].id is None  # Custom ID removed
    assert result_all[1].id is None  # Auto-generated ID removed
    assert result_all[2].id is None  # Custom ID removed
    assert result_all[3].tool_call_id is None  # Custom tool_call_id removed


def test_enhanced_tool_calls_in_ai_message() -> None:
    """Test enhanced handling of tool_calls in AIMessage."""
    tool_calls = [
        {
            "type": "tool_call",
            "id": f"{LC_AUTO_PREFIX}call_123",
            "name": "search",
            "args": {"query": "test"},
        },
        {
            "type": "tool_call",
            "id": "user_call_456",
            "name": "calculate",
            "args": {"expression": "2+2"},
        },
    ]

    messages = [AIMessage("Let me help", tool_calls=tool_calls)]
    result = _strip_ids_from_messages(messages)

    result_tool_calls = result[0].tool_calls
    assert "id" not in result_tool_calls[0]  # Auto-generated ID removed
    assert result_tool_calls[1]["id"] == "user_call_456"  # User ID preserved


def test_enhanced_invalid_tool_calls_in_ai_message() -> None:
    """Test enhanced handling of invalid_tool_calls in AIMessage."""
    invalid_tool_calls = [
        {
            "type": "invalid_tool_call",
            "id": f"{LC_AUTO_PREFIX}invalid_123",
            "name": "broken_tool",
            "args": "malformed json",
            "error": "JSON decode error",
        },
        {
            "type": "invalid_tool_call",
            "id": "user_invalid_456",
            "name": "another_tool",
            "args": "also broken",
            "error": "Unknown error",
        },
    ]

    messages = [AIMessage("Error occurred", invalid_tool_calls=invalid_tool_calls)]
    result = _strip_ids_from_messages(messages)

    result_invalid_calls = result[0].invalid_tool_calls
    assert "id" not in result_invalid_calls[0]  # Auto-generated ID removed
    assert result_invalid_calls[1]["id"] == "user_invalid_456"  # User ID preserved


def test_comprehensive_content_blocks() -> None:
    """Test comprehensive handling of various content block types."""
    content = [
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

    messages = [AIMessage(content)]
    result = _strip_ids_from_messages(messages)

    result_content = result[0].content
    assert "id" not in result_content[0]  # Text block auto-generated ID removed
    assert result_content[1]["id"] == "user_image_id"  # Image user ID preserved
    assert "id" not in result_content[2]  # Tool call auto-generated ID removed
    assert result_content[3]["id"] == "user_file_789"  # File user ID preserved


def test_nested_message_structures() -> None:
    """Test handling of messages with complex nested structures."""
    content = [
        {
            "type": "text",
            "text": "Complex content",
            "id": f"{LC_AUTO_PREFIX}complex_123",
            "annotations": [
                {
                    "type": "citation",
                    "id": f"{LC_AUTO_PREFIX}cite_456",  # Should not be touched (nested)
                    "text": "Source reference",
                }
            ],
        }
    ]

    messages = [AIMessage(content)]
    result = _strip_ids_from_messages(messages)

    result_content = result[0].content[0]
    assert "id" not in result_content  # Top-level ID removed
    # Nested IDs are not currently processed (by design)
    assert result_content["annotations"][0]["id"] == f"{LC_AUTO_PREFIX}cite_456"


def test_backward_compatibility() -> None:
    """Test that enhanced function maintains backward compatibility."""
    # This test uses the same data as the original tests to ensure compatibility
    messages = [
        HumanMessage("Hello", id=f"{LC_AUTO_PREFIX}12345"),
        AIMessage("Hi there", id="user_provided_id"),
    ]

    # Call without the new parameter (default behavior)
    result = _strip_ids_from_messages(messages)

    # Should behave exactly as before
    assert result[0].id is None  # Auto-generated ID removed
    assert result[1].id == "user_provided_id"  # User ID preserved


def test_strip_all_ids_with_content_blocks() -> None:
    """Test strip_all_ids parameter with content blocks."""
    content = [
        {"type": "text", "text": "Hello", "id": "user_text_id"},
        {
            "type": "image",
            "url": "http://example.com/img.jpg",
            "id": f"{LC_AUTO_PREFIX}auto_img",
        },
        {"type": "tool_call", "id": "user_tool_call", "name": "search", "args": {}},
    ]

    messages = [AIMessage(content)]

    # Test with strip_all_ids=True
    result = _strip_ids_from_messages(messages, strip_all_ids=True)

    result_content = result[0].content
    assert "id" not in result_content[0]  # User text ID removed
    assert "id" not in result_content[1]  # Auto-generated image ID removed
    assert "id" not in result_content[2]  # User tool call ID removed


def test_mixed_message_types_comprehensive() -> None:
    """Test comprehensive handling across different message types."""
    messages = [
        HumanMessage("User input", id=f"{LC_AUTO_PREFIX}human_123"),
        AIMessage(
            content=[
                {
                    "type": "text",
                    "text": "AI response",
                    "id": f"{LC_AUTO_PREFIX}text_456",
                }
            ],
            id="ai_custom_id",
            tool_calls=[
                {"id": f"{LC_AUTO_PREFIX}tool_789", "name": "search", "args": {}}
            ],
        ),
        SystemMessage("System prompt", id="system_custom_id"),
        ToolMessage("Tool output", tool_call_id=f"{LC_AUTO_PREFIX}tool_callback_999"),
    ]

    result = _strip_ids_from_messages(messages)

    # HumanMessage: auto-generated ID removed
    assert result[0].id is None

    # AIMessage: custom ID preserved, content block ID removed, tool call ID removed
    assert result[1].id == "ai_custom_id"
    assert "id" not in result[1].content[0]
    assert "id" not in result[1].tool_calls[0]

    # SystemMessage: custom ID preserved
    assert result[2].id == "system_custom_id"

    # ToolMessage: auto-generated tool_call_id removed
    assert result[3].tool_call_id is None
