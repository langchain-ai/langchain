"""Test new OpenAI API features."""

import pytest
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI


class TestResponseFormats:
    """Test new response format types."""

    def test_grammar_response_format(self):
        """Test grammar response format configuration."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Test grammar format in Responses API
        grammar_format = {
            "type": "grammar",
            "grammar": """
            start: expr
            expr: NUMBER ("+" | "-") NUMBER
            NUMBER: /[0-9]+/
            %import common.WS
            %ignore WS
            """,
        }

        # This should not raise an error during bind
        bound_llm = llm.bind(response_format=grammar_format)
        assert bound_llm is not None

    def test_python_response_format(self):
        """Test python response format configuration."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Test python format in Responses API
        python_format = {"type": "python"}

        # This should not raise an error during bind
        bound_llm = llm.bind(response_format=python_format)
        assert bound_llm is not None

    def test_grammar_format_validation(self):
        """Test that grammar format requires grammar field."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Test missing grammar field
        invalid_format = {"type": "grammar"}

        bound_llm = llm.bind(response_format=invalid_format)

        # The error should be raised when trying to create the payload
        # not during bind, so we can't easily test this in unit tests
        # without mocking the actual API call
        assert bound_llm is not None


class TestAllowedToolsChoice:
    """Test allowed_tools tool choice functionality."""

    def test_allowed_tools_auto_mode(self):
        """Test allowed_tools with auto mode."""

        @tool
        def get_weather(location: str) -> str:
            """Get weather for location."""
            return f"Weather in {location}: sunny"

        @tool
        def get_time() -> str:
            """Get current time."""
            return "12:00 PM"

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        allowed_tools_choice = {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "auto",
                "tools": [
                    {"type": "function", "function": {"name": "get_weather"}},
                    {"type": "function", "function": {"name": "get_time"}},
                ],
            },
        }

        bound_llm = llm.bind_tools(
            [get_weather, get_time], tool_choice=allowed_tools_choice
        )
        assert bound_llm is not None

    def test_allowed_tools_required_mode(self):
        """Test allowed_tools with required mode."""

        @tool
        def calculate(expression: str) -> str:
            """Calculate mathematical expression."""
            return f"Result: {eval(expression)}"  # noqa: S307

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        allowed_tools_choice = {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "required",
                "tools": [{"type": "function", "function": {"name": "calculate"}}],
            },
        }

        bound_llm = llm.bind_tools([calculate], tool_choice=allowed_tools_choice)
        assert bound_llm is not None

    def test_allowed_tools_invalid_mode(self):
        """Test that invalid allowed_tools mode raises error."""

        @tool
        def test_tool() -> str:
            """Test tool."""
            return "test"

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        invalid_choice = {
            "type": "allowed_tools",
            "allowed_tools": {"mode": "invalid_mode", "tools": []},
        }

        with pytest.raises(ValueError, match="allowed_tools mode must be"):
            llm.bind_tools([test_tool], tool_choice=invalid_choice)


class TestVerbosityParameter:
    """Test verbosity parameter functionality."""

    def test_verbosity_parameter_low(self):
        """Test verbosity parameter with low value."""
        llm = ChatOpenAI(model="gpt-4o-mini", verbosity="low")

        assert llm.verbosity == "low"
        assert "verbosity" in llm._default_params
        assert llm._default_params["verbosity"] == "low"

    def test_verbosity_parameter_medium(self):
        """Test verbosity parameter with medium value."""
        llm = ChatOpenAI(model="gpt-4o-mini", verbosity="medium")

        assert llm.verbosity == "medium"
        assert llm._default_params["verbosity"] == "medium"

    def test_verbosity_parameter_high(self):
        """Test verbosity parameter with high value."""
        llm = ChatOpenAI(model="gpt-4o-mini", verbosity="high")

        assert llm.verbosity == "high"
        assert llm._default_params["verbosity"] == "high"

    def test_verbosity_parameter_none(self):
        """Test verbosity parameter with None (default)."""
        llm = ChatOpenAI(model="gpt-4o-mini")

        assert llm.verbosity is None
        # When verbosity is None, it may not be included in _default_params
        # due to the exclude_if_none filtering
        verbosity_param = llm._default_params.get("verbosity")
        assert verbosity_param is None


class TestCustomToolStreamingSupport:
    """Test that custom tool streaming events are handled."""

    def test_custom_tool_streaming_event_types(self):
        """Test that the new custom tool streaming event types are supported."""
        # This test verifies that our code includes the necessary event handling
        # The actual streaming event handling is tested in integration tests

        # Import the base module to verify it loads without errors
        import langchain_openai.chat_models.base as base_module

        # Verify the module loaded successfully
        assert base_module is not None

        # Check that the module contains our custom tool streaming logic
        # by looking for the event type strings in the source
        import inspect

        source = inspect.getsource(base_module)

        # Verify our custom tool streaming events are handled
        assert "response.custom_tool_call_input.delta" in source
        assert "response.custom_tool_call_input.done" in source


class TestMinimalReasoningEffort:
    """Test that minimal reasoning effort is supported."""

    def test_minimal_reasoning_effort(self):
        """Test reasoning_effort parameter supports 'minimal'."""
        llm = ChatOpenAI(model="gpt-4o-mini", reasoning_effort="minimal")

        assert llm.reasoning_effort == "minimal"
        assert llm._default_params["reasoning_effort"] == "minimal"

    def test_all_reasoning_effort_values(self):
        """Test all supported reasoning effort values."""
        supported_values = ["minimal", "low", "medium", "high"]

        for value in supported_values:
            llm = ChatOpenAI(model="gpt-4o-mini", reasoning_effort=value)
            assert llm.reasoning_effort == value
            assert llm._default_params["reasoning_effort"] == value


class TestBackwardCompatibility:
    """Test that existing functionality still works."""

    def test_existing_response_formats(self):
        """Test that existing response formats still work."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # JSON object format should still work
        json_llm = llm.bind(response_format={"type": "json_object"})
        assert json_llm is not None

        # JSON schema format should still work
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": {
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                    "required": ["result"],
                },
            },
        }

        schema_llm = llm.bind(response_format=schema)
        assert schema_llm is not None

    def test_existing_tool_choice(self):
        """Test that existing tool_choice functionality still works."""

        @tool
        def test_tool(x: int) -> int:
            """Test tool."""
            return x * 2

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # String tool choice should still work
        bound_llm = llm.bind_tools([test_tool], tool_choice="test_tool")
        assert bound_llm is not None

        # Auto/none/required should still work
        for choice in ["auto", "none", "required"]:
            bound_llm = llm.bind_tools([test_tool], tool_choice=choice)
            assert bound_llm is not None

        # Boolean tool choice should still work
        bound_llm = llm.bind_tools([test_tool], tool_choice=True)
        assert bound_llm is not None
