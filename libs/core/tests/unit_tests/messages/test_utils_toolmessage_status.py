"""
Test for ToolMessage status field preservation in convert_to_messages.

This test validates the fix for issue #32835 where ToolMessage.status 
was not preserved during convert_to_messages conversion.
"""

from langchain_core.messages import convert_to_messages, ToolMessage


class TestToolMessageStatusPreservation:
    """Test cases for ToolMessage status field preservation."""

    def test_status_error_preserved(self) -> None:
        """Test that status='error' is preserved during conversion."""
        # Create ToolMessage with error status
        original_message = ToolMessage(
            content="Error: please fix your mistakes",
            tool_call_id="foobar",
            status="error"
        )
        
        # Convert to dict (simulating database storage)
        message_dict = original_message.model_dump()
        
        # Convert back using convert_to_messages
        recovered_message = convert_to_messages([message_dict])[0]
        
        # Status should be preserved
        assert original_message.status == recovered_message.status
        assert recovered_message.status == "error"

    def test_status_success_preserved(self) -> None:
        """Test that status='success' is preserved during conversion."""
        original_message = ToolMessage(
            content="Operation completed successfully",
            tool_call_id="success_call",
            status="success"
        )
        
        message_dict = original_message.model_dump()
        recovered_message = convert_to_messages([message_dict])[0]
        
        assert recovered_message.status == "success"

    def test_default_status_when_missing(self) -> None:
        """Test that default status is used when not specified."""
        # Create message dict without status field
        message_dict = {
            "content": "Default status test",
            "type": "tool", 
            "tool_call_id": "default_call"
        }
        
        recovered_message = convert_to_messages([message_dict])[0]
        
        # Should default to "success" 
        assert recovered_message.status == "success"

    def test_all_fields_preserved(self) -> None:
        """Test that all ToolMessage fields are preserved, not just status."""
        original_message = ToolMessage(
            content="Complete field test",
            tool_call_id="field_test",
            status="error",
            artifact={"test": "data"},
            additional_kwargs={"custom": "value"},
            response_metadata={"source": "test"}
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

    def test_issue_32835_reproduction(self) -> None:
        """Exact reproduction of issue #32835 from GitHub."""
        tool_message = ToolMessage(
            content="Error: please fix your mistakes", 
            tool_call_id="foobar", 
            status="error"
        )
        tool_message_json = tool_message.model_dump()
        tool_message_recovered = convert_to_messages([tool_message_json])[0]

        # This assertion should pass with the fix
        assert tool_message.status == tool_message_recovered.status, (
            f'received "{tool_message_recovered.status}", expected "{tool_message.status}"'
        )

    def test_multiple_messages_conversion(self) -> None:
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
        for i, (original, recovered, expected) in enumerate(
            zip(messages, recovered_messages, expected_statuses)
        ):
            assert recovered.status == expected, (
                f"Message {i}: expected {expected}, got {recovered.status}"
            )