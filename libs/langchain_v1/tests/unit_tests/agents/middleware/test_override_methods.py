"""Unit tests for override() methods on ModelRequest and ToolCallRequest."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from langchain.agents.middleware.types import ModelRequest
from langchain.tools.tool_node import ToolCallRequest


class TestModelRequestOverride:
    """Test the ModelRequest.override() method."""

    def test_override_single_attribute(self) -> None:
        """Test overriding a single attribute."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        original_request = ModelRequest(
            model=model,
            system_prompt="Original prompt",
            messages=[HumanMessage("Hi")],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={},
            runtime=None,
        )

        new_request = original_request.override(system_prompt="New prompt")

        # New request should have the overridden value
        assert new_request.system_prompt == "New prompt"
        # Original request should be unchanged (immutability)
        assert original_request.system_prompt == "Original prompt"
        # Other attributes should be the same
        assert new_request.model == original_request.model
        assert new_request.messages == original_request.messages

    def test_override_multiple_attributes(self) -> None:
        """Test overriding multiple attributes at once."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        original_request = ModelRequest(
            model=model,
            system_prompt="Original prompt",
            messages=[HumanMessage("Hi")],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={"count": 1},
            runtime=None,
        )

        new_request = original_request.override(
            system_prompt="New prompt",
            tool_choice="auto",
            state={"count": 2},
        )

        # Overridden values should be changed
        assert new_request.system_prompt == "New prompt"
        assert new_request.tool_choice == "auto"
        assert new_request.state == {"count": 2}
        # Original should be unchanged
        assert original_request.system_prompt == "Original prompt"
        assert original_request.tool_choice is None
        assert original_request.state == {"count": 1}

    def test_override_messages(self) -> None:
        """Test overriding messages list."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        original_messages = [HumanMessage("Hi")]
        new_messages = [HumanMessage("Hello"), AIMessage("Hi there")]

        original_request = ModelRequest(
            model=model,
            system_prompt=None,
            messages=original_messages,
            tool_choice=None,
            tools=[],
            response_format=None,
            state={},
            runtime=None,
        )

        new_request = original_request.override(messages=new_messages)

        assert new_request.messages == new_messages
        assert original_request.messages == original_messages
        assert len(new_request.messages) == 2
        assert len(original_request.messages) == 1

    def test_override_model_settings(self) -> None:
        """Test overriding model_settings dict."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        original_request = ModelRequest(
            model=model,
            system_prompt=None,
            messages=[HumanMessage("Hi")],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={},
            runtime=None,
            model_settings={"temperature": 0.5},
        )

        new_request = original_request.override(
            model_settings={"temperature": 0.9, "max_tokens": 100}
        )

        assert new_request.model_settings == {"temperature": 0.9, "max_tokens": 100}
        assert original_request.model_settings == {"temperature": 0.5}

    def test_override_with_none_value(self) -> None:
        """Test overriding with None value."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        original_request = ModelRequest(
            model=model,
            system_prompt="Original prompt",
            messages=[HumanMessage("Hi")],
            tool_choice="auto",
            tools=[],
            response_format=None,
            state={},
            runtime=None,
        )

        new_request = original_request.override(
            system_prompt=None,
            tool_choice=None,
        )

        assert new_request.system_prompt is None
        assert new_request.tool_choice is None
        assert original_request.system_prompt == "Original prompt"
        assert original_request.tool_choice == "auto"

    def test_override_preserves_identity_of_unchanged_objects(self) -> None:
        """Test that unchanged attributes maintain object identity."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        messages = [HumanMessage("Hi")]
        state = {"key": "value"}

        original_request = ModelRequest(
            model=model,
            system_prompt="Original prompt",
            messages=messages,
            tool_choice=None,
            tools=[],
            response_format=None,
            state=state,
            runtime=None,
        )

        new_request = original_request.override(system_prompt="New prompt")

        # Unchanged objects should be the same instance
        assert new_request.messages is messages
        assert new_request.state is state
        assert new_request.model is model

    def test_override_chaining(self) -> None:
        """Test chaining multiple override calls."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        original_request = ModelRequest(
            model=model,
            system_prompt="Prompt 1",
            messages=[HumanMessage("Hi")],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={"count": 1},
            runtime=None,
        )

        final_request = (
            original_request.override(system_prompt="Prompt 2")
            .override(state={"count": 2})
            .override(tool_choice="auto")
        )

        assert final_request.system_prompt == "Prompt 2"
        assert final_request.state == {"count": 2}
        assert final_request.tool_choice == "auto"
        # Original should be unchanged
        assert original_request.system_prompt == "Prompt 1"
        assert original_request.state == {"count": 1}
        assert original_request.tool_choice is None


class TestToolCallRequestOverride:
    """Test the ToolCallRequest.override() method."""

    def test_override_tool_call(self) -> None:
        """Test overriding tool_call dict."""
        from langchain_core.tools import tool

        @tool
        def test_tool(x: int) -> str:
            """A test tool."""
            return f"Result: {x}"

        original_call = {"name": "test_tool", "args": {"x": 5}, "id": "1", "type": "tool_call"}
        modified_call = {"name": "test_tool", "args": {"x": 10}, "id": "1", "type": "tool_call"}

        original_request = ToolCallRequest(
            tool_call=original_call,
            tool=test_tool,
            state={"messages": []},
            runtime=None,
        )

        new_request = original_request.override(tool_call=modified_call)

        # New request should have modified tool_call
        assert new_request.tool_call["args"]["x"] == 10
        # Original should be unchanged
        assert original_request.tool_call["args"]["x"] == 5
        # Other attributes should be the same
        assert new_request.tool is original_request.tool
        assert new_request.state is original_request.state

    def test_override_state(self) -> None:
        """Test overriding state."""
        from langchain_core.tools import tool

        @tool
        def test_tool(x: int) -> str:
            """A test tool."""
            return f"Result: {x}"

        tool_call = {"name": "test_tool", "args": {"x": 5}, "id": "1", "type": "tool_call"}
        original_state = {"messages": [HumanMessage("Hi")]}
        new_state = {"messages": [HumanMessage("Hi"), AIMessage("Hello")]}

        original_request = ToolCallRequest(
            tool_call=tool_call,
            tool=test_tool,
            state=original_state,
            runtime=None,
        )

        new_request = original_request.override(state=new_state)

        assert len(new_request.state["messages"]) == 2
        assert len(original_request.state["messages"]) == 1

    def test_override_multiple_attributes(self) -> None:
        """Test overriding multiple attributes at once."""
        from langchain_core.tools import tool

        @tool
        def test_tool(x: int) -> str:
            """A test tool."""
            return f"Result: {x}"

        @tool
        def another_tool(y: str) -> str:
            """Another test tool."""
            return f"Output: {y}"

        original_call = {"name": "test_tool", "args": {"x": 5}, "id": "1", "type": "tool_call"}
        modified_call = {
            "name": "another_tool",
            "args": {"y": "hello"},
            "id": "2",
            "type": "tool_call",
        }

        original_request = ToolCallRequest(
            tool_call=original_call,
            tool=test_tool,
            state={"count": 1},
            runtime=None,
        )

        new_request = original_request.override(
            tool_call=modified_call,
            tool=another_tool,
            state={"count": 2},
        )

        assert new_request.tool_call["name"] == "another_tool"
        assert new_request.tool.name == "another_tool"
        assert new_request.state == {"count": 2}
        # Original unchanged
        assert original_request.tool_call["name"] == "test_tool"
        assert original_request.tool.name == "test_tool"
        assert original_request.state == {"count": 1}

    def test_override_with_copy_pattern(self) -> None:
        """Test common pattern of copying and modifying tool_call."""
        from langchain_core.tools import tool

        @tool
        def test_tool(value: int) -> str:
            """A test tool."""
            return f"Result: {value}"

        original_call = {
            "name": "test_tool",
            "args": {"value": 5},
            "id": "call_123",
            "type": "tool_call",
        }

        original_request = ToolCallRequest(
            tool_call=original_call,
            tool=test_tool,
            state={},
            runtime=None,
        )

        # Common pattern: copy tool_call and modify args
        modified_call = {**original_request.tool_call, "args": {"value": 10}}
        new_request = original_request.override(tool_call=modified_call)

        assert new_request.tool_call["args"]["value"] == 10
        assert new_request.tool_call["id"] == "call_123"
        assert new_request.tool_call["name"] == "test_tool"
        # Original unchanged
        assert original_request.tool_call["args"]["value"] == 5

    def test_override_preserves_identity(self) -> None:
        """Test that unchanged attributes maintain object identity."""
        from langchain_core.tools import tool

        @tool
        def test_tool(x: int) -> str:
            """A test tool."""
            return f"Result: {x}"

        tool_call = {"name": "test_tool", "args": {"x": 5}, "id": "1", "type": "tool_call"}
        state = {"messages": []}

        original_request = ToolCallRequest(
            tool_call=tool_call,
            tool=test_tool,
            state=state,
            runtime=None,
        )

        new_call = {"name": "test_tool", "args": {"x": 10}, "id": "1", "type": "tool_call"}
        new_request = original_request.override(tool_call=new_call)

        # Unchanged objects should be the same instance
        assert new_request.tool is test_tool
        assert new_request.state is state

    def test_override_chaining(self) -> None:
        """Test chaining multiple override calls."""
        from langchain_core.tools import tool

        @tool
        def test_tool(x: int) -> str:
            """A test tool."""
            return f"Result: {x}"

        tool_call = {"name": "test_tool", "args": {"x": 5}, "id": "1", "type": "tool_call"}

        original_request = ToolCallRequest(
            tool_call=tool_call,
            tool=test_tool,
            state={"count": 1},
            runtime=None,
        )

        call_2 = {"name": "test_tool", "args": {"x": 10}, "id": "1", "type": "tool_call"}
        call_3 = {"name": "test_tool", "args": {"x": 15}, "id": "1", "type": "tool_call"}

        final_request = (
            original_request.override(tool_call=call_2)
            .override(state={"count": 2})
            .override(tool_call=call_3)
        )

        assert final_request.tool_call["args"]["x"] == 15
        assert final_request.state == {"count": 2}
        # Original unchanged
        assert original_request.tool_call["args"]["x"] == 5
        assert original_request.state == {"count": 1}
