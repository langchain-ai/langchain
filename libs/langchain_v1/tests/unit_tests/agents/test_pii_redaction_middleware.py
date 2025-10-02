"""Tests for PII redaction middleware."""

import json
import re
from unittest.mock import Mock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.runtime import Runtime

from langchain.agents.middleware.pii_redaction import PIIRedactionMiddleware
from langchain.agents.middleware.types import AgentState, ModelRequest


class TestPIIRedactionMiddleware:
    """Test suite for PII redaction middleware."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.pii_rules = {
            "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        }
        self.middleware = PIIRedactionMiddleware(rules=self.pii_rules)

    def test_init_with_rules(self) -> None:
        """Test middleware initialization with rules."""
        middleware = PIIRedactionMiddleware(rules=self.pii_rules)
        assert middleware.rules == self.pii_rules
        assert middleware.redaction_map == {}

    def test_init_without_rules(self) -> None:
        """Test middleware initialization without rules."""
        middleware = PIIRedactionMiddleware()
        assert middleware.rules == {}
        assert middleware.redaction_map == {}

    def test_generate_redaction_id(self) -> None:
        """Test redaction ID generation."""
        id1 = self.middleware._generate_redaction_id()
        id2 = self.middleware._generate_redaction_id()

        assert len(id1) == 8
        assert len(id2) == 8
        assert id1 != id2

    def test_apply_pii_rules_basic(self) -> None:
        """Test basic PII rule application."""
        text = "My SSN is 123-45-6789 and email is test@example.com"
        redaction_map = {}

        result = self.middleware._apply_pii_rules(text, self.pii_rules, redaction_map)

        # Should contain redaction markers
        assert "[REDACTED_SSN_" in result
        assert "[REDACTED_EMAIL_" in result
        assert "123-45-6789" not in result
        assert "test@example.com" not in result

        # Should have entries in redaction map
        assert len(redaction_map) == 2
        assert "123-45-6789" in redaction_map.values()
        assert "test@example.com" in redaction_map.values()

    def test_apply_pii_rules_no_matches(self) -> None:
        """Test PII rule application with no matches."""
        text = "This text has no PII"
        redaction_map = {}

        result = self.middleware._apply_pii_rules(text, self.pii_rules, redaction_map)

        assert result == text
        assert redaction_map == {}

    def test_apply_pii_rules_multiple_matches(self) -> None:
        """Test PII rule application with multiple matches of same type."""
        text = "SSN 123-45-6789 and another SSN 987-65-4321"
        redaction_map = {}

        result = self.middleware._apply_pii_rules(text, self.pii_rules, redaction_map)

        # Should have two different redaction markers
        ssn_markers = re.findall(r"\[REDACTED_SSN_\w+\]", result)
        assert len(ssn_markers) == 2
        assert ssn_markers[0] != ssn_markers[1]

        # Should have both SSNs in redaction map
        assert len(redaction_map) == 2
        assert "123-45-6789" in redaction_map.values()
        assert "987-65-4321" in redaction_map.values()

    def test_restore_redacted_values(self) -> None:
        """Test restoration of redacted values."""
        redaction_map = {"abc123": "123-45-6789", "def456": "test@example.com"}
        text = "My SSN is [REDACTED_SSN_abc123] and email is [REDACTED_EMAIL_def456]"

        result = self.middleware._restore_redacted_values(text, redaction_map)

        assert result == "My SSN is 123-45-6789 and email is test@example.com"

    def test_restore_redacted_values_unknown_id(self) -> None:
        """Test restoration with unknown redaction ID."""
        redaction_map = {"abc123": "123-45-6789"}
        text = "My SSN is [REDACTED_SSN_abc123] and unknown [REDACTED_SSN_xyz789]"

        result = self.middleware._restore_redacted_values(text, redaction_map)

        assert result == "My SSN is 123-45-6789 and unknown [REDACTED_SSN_xyz789]"

    def test_process_human_message(self) -> None:
        """Test processing of HumanMessage."""
        message = HumanMessage(content="My SSN is 123-45-6789")
        redaction_map = {}

        result = self.middleware._process_message(message, self.pii_rules, redaction_map)

        assert isinstance(result, HumanMessage)
        assert "[REDACTED_SSN_" in result.content
        assert "123-45-6789" not in result.content
        assert len(redaction_map) == 1

    def test_process_system_message(self) -> None:
        """Test processing of SystemMessage."""
        message = SystemMessage(content="Contact user at test@example.com")
        redaction_map = {}

        result = self.middleware._process_message(message, self.pii_rules, redaction_map)

        assert isinstance(result, SystemMessage)
        assert "[REDACTED_EMAIL_" in result.content
        assert "test@example.com" not in result.content
        assert len(redaction_map) == 1

    def test_process_tool_message(self) -> None:
        """Test processing of ToolMessage."""
        message = ToolMessage(content="Found user with SSN 123-45-6789", tool_call_id="test")
        redaction_map = {}

        result = self.middleware._process_message(message, self.pii_rules, redaction_map)

        assert isinstance(result, ToolMessage)
        assert "[REDACTED_SSN_" in result.content
        assert "123-45-6789" not in result.content
        assert result.tool_call_id == "test"
        assert len(redaction_map) == 1

    def test_process_ai_message_string_content(self) -> None:
        """Test processing of AIMessage with string content."""
        message = AIMessage(content="I found the SSN: 123-45-6789")
        redaction_map = {}

        result = self.middleware._process_message(message, self.pii_rules, redaction_map)

        assert isinstance(result, AIMessage)
        assert "[REDACTED_SSN_" in result.content
        assert "123-45-6789" not in result.content
        assert len(redaction_map) == 1

    def test_process_ai_message_with_tool_calls(self) -> None:
        """Test processing of AIMessage with tool calls."""
        tool_calls = [
            {
                "name": "lookup_user",
                "args": {"ssn": "123-45-6789", "email": "test@example.com"},
                "id": "call_1",
                "type": "tool_call",
            }
        ]
        message = AIMessage(content="Looking up user", tool_calls=tool_calls)
        redaction_map = {}

        result = self.middleware._process_message(message, self.pii_rules, redaction_map)

        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1

        # Check that tool call args are redacted
        tool_call = result.tool_calls[0]
        assert "[REDACTED_SSN_" in tool_call["args"]["ssn"]
        assert "[REDACTED_EMAIL_" in tool_call["args"]["email"]
        assert "123-45-6789" not in tool_call["args"]["ssn"]
        assert "test@example.com" not in tool_call["args"]["email"]

        # Should have both PII values in redaction map
        assert len(redaction_map) == 2

    def test_process_ai_message_no_changes(self) -> None:
        """Test processing of AIMessage with no PII."""
        message = AIMessage(content="Hello, how can I help you?")
        redaction_map = {}

        result = self.middleware._process_message(message, self.pii_rules, redaction_map)

        assert result is message  # Should return same instance if no changes
        assert redaction_map == {}

    def test_process_unsupported_message_type(self) -> None:
        """Test processing of unsupported message type."""
        message = Mock()  # Mock unsupported message type
        redaction_map = {}

        with pytest.raises(ValueError, match="Unsupported message type"):
            self.middleware._process_message(message, self.pii_rules, redaction_map)

    def test_restore_human_message(self) -> None:
        """Test restoration of HumanMessage."""
        redaction_map = {"abc123": "123-45-6789"}
        message = HumanMessage(content="My SSN is [REDACTED_SSN_abc123]")

        result, changed = self.middleware._restore_message(message, redaction_map)

        assert isinstance(result, HumanMessage)
        assert result.content == "My SSN is 123-45-6789"
        assert changed is True

    def test_restore_ai_message_with_tool_calls(self) -> None:
        """Test restoration of AIMessage with tool calls."""
        redaction_map = {"abc123": "123-45-6789", "def456": "test@example.com"}
        tool_calls = [
            {
                "name": "lookup_user",
                "args": {"ssn": "[REDACTED_SSN_abc123]", "email": "[REDACTED_EMAIL_def456]"},
                "id": "call_1",
                "type": "tool_call",
            }
        ]
        message = AIMessage(content="Looking up user", tool_calls=tool_calls)

        result, changed = self.middleware._restore_message(message, redaction_map)

        assert isinstance(result, AIMessage)
        assert changed is True

        # Check that tool call args are restored
        tool_call = result.tool_calls[0]
        assert tool_call["args"]["ssn"] == "123-45-6789"
        assert tool_call["args"]["email"] == "test@example.com"

    def test_restore_message_no_changes(self) -> None:
        """Test restoration of message with no redactions."""
        redaction_map = {"abc123": "123-45-6789"}
        message = HumanMessage(content="Hello, how can I help you?")

        result, changed = self.middleware._restore_message(message, redaction_map)

        assert result is message  # Should return same instance if no changes
        assert changed is False

    def test_modify_model_request_with_rules(self) -> None:
        """Test modify_model_request with PII rules."""
        messages = [
            HumanMessage(content="My SSN is 123-45-6789"),
            SystemMessage(content="Contact user at test@example.com"),
        ]
        request = ModelRequest(
            model=Mock(),
            system_prompt="System prompt",
            messages=messages,
            tool_choice=None,
            tools=[],
            response_format=None,
        )
        state = {"messages": messages}
        runtime = Mock()
        runtime.context = Mock()
        runtime.context.PIIRedactionMiddleware = {}

        result = self.middleware.modify_model_request(request, state, runtime)

        assert len(result.messages) == 2
        assert "[REDACTED_SSN_" in result.messages[0].content
        assert "[REDACTED_EMAIL_" in result.messages[1].content
        assert len(self.middleware.redaction_map) == 2

    def test_modify_model_request_no_rules(self) -> None:
        """Test modify_model_request with no rules."""
        middleware = PIIRedactionMiddleware()  # No rules
        messages = [HumanMessage(content="My SSN is 123-45-6789")]
        request = ModelRequest(
            model=Mock(),
            system_prompt="System prompt",
            messages=messages,
            tool_choice=None,
            tools=[],
            response_format=None,
        )
        state = {"messages": messages}
        runtime = Mock()
        runtime.context = Mock()
        runtime.context.PIIRedactionMiddleware = {}

        result = middleware.modify_model_request(request, state, runtime)

        # Should return original request unchanged
        assert result is request

    def test_modify_model_request_context_rules(self) -> None:
        """Test modify_model_request with rules from context."""
        middleware = PIIRedactionMiddleware()  # No default rules
        messages = [HumanMessage(content="My SSN is 123-45-6789")]
        request = ModelRequest(
            model=Mock(),
            system_prompt="System prompt",
            messages=messages,
            tool_choice=None,
            tools=[],
            response_format=None,
        )
        state = {"messages": messages}
        runtime = Mock()
        runtime.context = Mock()
        runtime.context.PIIRedactionMiddleware = {"rules": self.pii_rules}

        result = middleware.modify_model_request(request, state, runtime)

        assert "[REDACTED_SSN_" in result.messages[0].content
        assert len(middleware.redaction_map) == 1

    def test_after_model_no_redactions(self) -> None:
        """Test after_model with no redactions made."""
        middleware = PIIRedactionMiddleware()  # Empty redaction map
        state = {"messages": [AIMessage(content="Hello")]}
        runtime = Mock()

        result = middleware.after_model(state, runtime)

        assert result is None

    def test_after_model_no_messages(self) -> None:
        """Test after_model with no messages."""
        state = {"messages": []}
        runtime = Mock()

        result = self.middleware.after_model(state, runtime)

        assert result is None

    def test_after_model_last_message_not_ai(self) -> None:
        """Test after_model with last message not being AIMessage."""
        state = {"messages": [HumanMessage(content="Hello")]}
        runtime = Mock()

        result = self.middleware.after_model(state, runtime)

        assert result is None

    def test_after_model_restore_ai_message(self) -> None:
        """Test after_model restoring AIMessage."""
        # Set up redaction map
        self.middleware.redaction_map = {"abc123": "123-45-6789"}

        message = AIMessage(content="Found SSN: [REDACTED_SSN_abc123]", id="msg_1")
        state = {"messages": [message]}
        runtime = Mock()

        result = self.middleware.after_model(state, runtime)

        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], RemoveMessage)
        assert result["messages"][0].id == "msg_1"
        assert isinstance(result["messages"][1], AIMessage)
        assert result["messages"][1].content == "Found SSN: 123-45-6789"

    def test_after_model_structured_response(self) -> None:
        """Test after_model with structured response."""
        # Set up redaction map
        self.middleware.redaction_map = {"abc123": "123-45-6789"}

        message = AIMessage(content='{"ssn": "[REDACTED_SSN_abc123]"}', id="msg_1")
        state = {"messages": [message]}
        runtime = Mock()

        result = self.middleware.after_model(state, runtime)

        assert result is not None
        assert "structured_response" in result
        assert result["structured_response"] == {"ssn": "123-45-6789"}

    def test_after_model_structured_response_tool_call(self) -> None:
        """Test after_model with structured response tool call."""
        # Set up redaction map
        self.middleware.redaction_map = {"abc123": "123-45-6789"}

        tool_calls = [
            {
                "name": "extract-user-info",
                "args": {"ssn": "[REDACTED_SSN_abc123]"},
                "id": "call_1",
                "type": "tool_call",
            }
        ]
        second_message = AIMessage(content="Extracting info", tool_calls=tool_calls, id="msg_1")
        last_message = AIMessage(content="Found SSN: [REDACTED_SSN_abc123]", id="msg_2")
        state = {"messages": [second_message, last_message]}
        runtime = Mock()

        result = self.middleware.after_model(state, runtime)

        # The test should pass because both messages have redactions that need restoration
        assert result is not None
        assert "structured_response" in result
        assert result["structured_response"] == {"ssn": "123-45-6789"}
        assert len(result["messages"]) == 4  # 2 RemoveMessage + 2 restored messages

    def test_after_model_no_changes_needed(self) -> None:
        """Test after_model when no changes are needed."""
        # Set up redaction map but message has no redactions
        self.middleware.redaction_map = {"abc123": "123-45-6789"}

        message = AIMessage(content="Hello, no PII here", id="msg_1")
        state = {"messages": [message]}
        runtime = Mock()

        result = self.middleware.after_model(state, runtime)

        assert result is None

    def test_integration_workflow(self) -> None:
        """Test complete integration workflow."""
        # Step 1: Process request with PII
        messages = [HumanMessage(content="My SSN is 123-45-6789")]
        request = ModelRequest(
            model=Mock(),
            system_prompt="System prompt",
            messages=messages,
            tool_choice=None,
            tools=[],
            response_format=None,
        )
        state = {"messages": messages}
        runtime = Mock()
        runtime.context = Mock()
        runtime.context.PIIRedactionMiddleware = {}

        # Modify request (redact PII)
        modified_request = self.middleware.modify_model_request(request, state, runtime)
        assert "[REDACTED_SSN_" in modified_request.messages[0].content
        assert len(self.middleware.redaction_map) == 1

        # Step 2: Simulate model response with redacted content
        # Get the actual redaction ID from the map
        redaction_id = list(self.middleware.redaction_map.keys())[0]
        response_message = AIMessage(
            content=f"I'll look up SSN [REDACTED_SSN_{redaction_id}]",
            tool_calls=[
                {
                    "name": "lookup_user",
                    "args": {"ssn": f"[REDACTED_SSN_{redaction_id}]"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
            id="response_1",
        )
        response_state = {"messages": [response_message]}

        # Restore redacted content
        result = self.middleware.after_model(response_state, runtime)

        assert result is not None
        assert len(result["messages"]) == 2
        restored_message = result["messages"][1]
        assert isinstance(restored_message, AIMessage)
        assert "123-45-6789" in restored_message.content
        assert restored_message.tool_calls[0]["args"]["ssn"] == "123-45-6789"
