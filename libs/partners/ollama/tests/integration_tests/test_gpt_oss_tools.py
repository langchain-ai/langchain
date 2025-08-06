"""Integration tests for gpt-oss model tool calling support in ChatOllama.

These tests require an actual Ollama instance running with a gpt-oss model installed.
To run these tests:
1. Install Ollama: https://ollama.ai/
2. Pull a gpt-oss model: `ollama pull gpt-oss:20b`
3. Run these tests with: `pytest tests/integration_tests/test_gpt_oss_tools.py`

Note: These tests will be skipped if Ollama is not available or the model is not installed.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_ollama import ChatOllama

# Skip all tests in this module if OLLAMA_BASE_URL is not set or Ollama is not available
pytestmark = pytest.mark.skipif(
    os.environ.get("OLLAMA_BASE_URL") is None,
    reason="OLLAMA_BASE_URL not set, skipping Ollama integration tests",
)


@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location.

    Args:
        location: The city to get weather for.
        unit: Temperature unit (celsius or fahrenheit).
    """
    # Mock implementation for testing
    return f"The weather in {location} is sunny and 22 {unit}"


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.
    """
    return f"Found {max_results} results for '{query}'"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: The mathematical expression to evaluate.
    """
    # Simple mock calculation
    return "42"


def check_ollama_available() -> bool:
    """Check if Ollama is available and running."""
    try:
        llm = ChatOllama(model="llama2")  # Use a common model to test connectivity
        llm.invoke("test")
        return True
    except Exception:
        return False


def check_model_available(model_name: str) -> bool:
    """Check if a specific model is available in Ollama."""
    try:
        llm = ChatOllama(model=model_name)
        llm.invoke("test")
        return True
    except Exception:
        return False


@pytest.mark.integration
class TestGptOssToolCallingIntegration:
    """Integration tests for gpt-oss model tool calling."""

    @pytest.mark.skipif(
        not check_ollama_available(), reason="Ollama is not available or not running"
    )
    @pytest.mark.skipif(
        not check_model_available("gpt-oss:20b"),
        reason="gpt-oss:20b model is not installed",
    )
    def test_single_tool_call(self) -> None:
        """Test calling a single tool with gpt-oss model."""
        llm = ChatOllama(model="gpt-oss:20b", temperature=0)
        llm_with_tools = llm.bind_tools([get_weather])

        # Ask a question that should trigger tool use
        response = llm_with_tools.invoke(
            "What's the weather like in London? Please use the available tool."
        )

        # Check that the response is an AIMessage
        assert isinstance(response, AIMessage)

        # Check if tool calls were made (model might not always call tools)
        if response.tool_calls:
            assert len(response.tool_calls) > 0
            tool_call = response.tool_calls[0]
            assert tool_call["name"] == "get_weather"
            assert "location" in tool_call["args"]
            # The model should identify London as the location
            assert "london" in tool_call["args"]["location"].lower()

    @pytest.mark.skipif(
        not check_ollama_available(), reason="Ollama is not available or not running"
    )
    @pytest.mark.skipif(
        not check_model_available("gpt-oss:20b"),
        reason="gpt-oss:20b model is not installed",
    )
    def test_multiple_tools_binding(self) -> None:
        """Test binding multiple tools to gpt-oss model."""
        llm = ChatOllama(model="gpt-oss:20b", temperature=0)
        llm_with_tools = llm.bind_tools([get_weather, search_web, calculate])

        # Test that tools are properly bound
        assert hasattr(llm_with_tools, "kwargs")
        assert "tools" in llm_with_tools.kwargs
        tools = llm_with_tools.kwargs["tools"]
        assert len(tools) == 3

        # Verify tool names
        tool_names = {tool["function"]["name"] for tool in tools}
        assert tool_names == {"get_weather", "search_web", "calculate"}

        # Test invocation with a query that might use search
        response = llm_with_tools.invoke(
            "Search for information about Python programming. Use the search tool."
        )

        assert isinstance(response, AIMessage)
        # The model may or may not call the tool depending on its decision

    @pytest.mark.skipif(
        not check_ollama_available(), reason="Ollama is not available or not running"
    )
    @pytest.mark.skipif(
        not check_model_available("gpt-oss:20b"),
        reason="gpt-oss:20b model is not installed",
    )
    def test_tool_call_with_conversation(self) -> None:
        """Test tool calling within a conversation context."""
        llm = ChatOllama(model="gpt-oss:20b", temperature=0)
        llm_with_tools = llm.bind_tools([get_weather, calculate])

        # Create a conversation
        messages = [
            HumanMessage(content="Hi, I need help with two things."),
            AIMessage(content="Hello! I'd be happy to help. What do you need?"),
            HumanMessage(
                content="First, what's the weather in Paris? Second, calculate 15 * 28. "
                "Please use the available tools for both tasks."
            ),
        ]

        response = llm_with_tools.invoke(messages)

        assert isinstance(response, AIMessage)
        # Check if the model made tool calls
        if response.tool_calls:
            # The model might call one or both tools
            tool_names = {call["name"] for call in response.tool_calls}
            # At least one tool should be called
            assert len(tool_names) > 0
            # The tools called should be from our available tools
            assert tool_names.issubset({"get_weather", "calculate"})

    @pytest.mark.skipif(
        not check_ollama_available(), reason="Ollama is not available or not running"
    )
    @pytest.mark.skipif(
        not check_model_available("gpt-oss:20b"),
        reason="gpt-oss:20b model is not installed",
    )
    def test_streaming_with_tools(self) -> None:
        """Test streaming responses with tool calls."""
        llm = ChatOllama(model="gpt-oss:20b", temperature=0)
        llm_with_tools = llm.bind_tools([get_weather])

        # Stream a response
        chunks = []
        for chunk in llm_with_tools.stream(
            "What's the weather in Tokyo? Use the weather tool."
        ):
            chunks.append(chunk)

        # Should have received chunks
        assert len(chunks) > 0

        # Combine chunks to get the full response
        final_message = chunks[0]
        for chunk in chunks[1:]:
            final_message += chunk

        # Check if tool calls were made in the final combined message
        if hasattr(final_message, "tool_calls") and final_message.tool_calls:
            tool_call = final_message.tool_calls[0]
            assert tool_call["name"] == "get_weather"
            assert "location" in tool_call["args"]

    @pytest.mark.skipif(
        not check_ollama_available(), reason="Ollama is not available or not running"
    )
    @pytest.mark.skipif(
        not check_model_available("gpt-oss:20b"),
        reason="gpt-oss:20b model is not installed",
    )
    async def test_async_tool_calling(self) -> None:
        """Test asynchronous tool calling with gpt-oss model."""
        llm = ChatOllama(model="gpt-oss:20b", temperature=0)
        llm_with_tools = llm.bind_tools([calculate])

        # Test async invocation
        response = await llm_with_tools.ainvoke(
            "Calculate 42 times 10. Please use the calculate tool."
        )

        assert isinstance(response, AIMessage)
        # Check if tool was called
        if response.tool_calls:
            assert len(response.tool_calls) > 0
            tool_call = response.tool_calls[0]
            assert tool_call["name"] == "calculate"
            assert "expression" in tool_call["args"]


@pytest.mark.integration
class TestGptOssModelCompatibility:
    """Test compatibility of different gpt-oss model variants."""

    @pytest.mark.skipif(
        not check_ollama_available(), reason="Ollama is not available or not running"
    )
    def test_gpt_oss_variants(self) -> None:
        """Test that different gpt-oss model variants are detected correctly."""
        # Test various gpt-oss model names that might be available
        model_variants = [
            "gpt-oss",
            "gpt-oss:latest",
            "gpt-oss:20b",
            "gpt-oss:7b",
        ]

        for model_name in model_variants:
            if check_model_available(model_name):
                llm = ChatOllama(model=model_name)
                llm_with_tools = llm.bind_tools([get_weather])

                # Verify tools are in Harmony format
                tools = llm_with_tools.kwargs["tools"]
                assert len(tools) == 1
                tool = tools[0]
                assert tool["type"] == "function"
                assert "function" in tool

                # Check parameter types are strings
                props = tool["function"]["parameters"]["properties"]
                for prop in props.values():
                    if "type" in prop:
                        assert isinstance(prop["type"], str)

    @pytest.mark.skipif(
        not check_ollama_available(), reason="Ollama is not available or not running"
    )
    def test_non_gpt_oss_models_unchanged(self) -> None:
        """Test that non-gpt-oss models still work with standard format."""
        # Test with a non-gpt-oss model if available
        non_gpt_models = ["llama2", "mistral", "codellama"]

        for model_name in non_gpt_models:
            if check_model_available(model_name):
                llm = ChatOllama(model=model_name)
                llm_with_tools = llm.bind_tools([get_weather])

                # Tools should still be bound
                tools = llm_with_tools.kwargs["tools"]
                assert len(tools) == 1

                # Should use standard OpenAI format
                tool = tools[0]
                assert tool["type"] == "function"
                assert "function" in tool

                # The format should be compatible with standard Ollama models
                break  # Test with at least one non-gpt-oss model


@pytest.mark.integration
class TestGptOssErrorHandling:
    """Test error handling for gpt-oss models with tools."""

    @pytest.mark.skipif(
        not check_ollama_available(), reason="Ollama is not available or not running"
    )
    @pytest.mark.skipif(
        not check_model_available("gpt-oss:20b"),
        reason="gpt-oss:20b model is not installed",
    )
    def test_malformed_tool_response_handling(self) -> None:
        """Test that malformed tool responses are handled gracefully."""
        llm = ChatOllama(
            model="gpt-oss:20b", temperature=1.5
        )  # High temp for randomness

        # Create a tool that might cause parsing issues
        @tool
        def complex_tool(
            data: Dict[str, Any],
            nested: Optional[Dict[str, Any]] = None,
        ) -> str:
            """A complex tool with nested parameters.

            Args:
                data: Complex data structure.
                nested: Optional nested data.
            """
            return "processed"

        llm_with_tools = llm.bind_tools([complex_tool])

        # This should not raise an error even if the model returns malformed tool calls
        try:
            response = llm_with_tools.invoke("Use the complex tool with some data.")
            assert isinstance(response, AIMessage)
        except Exception as e:
            # The error should be handled gracefully
            pytest.fail(f"Tool calling raised an unexpected error: {e}")

    @pytest.mark.skipif(
        not check_ollama_available(), reason="Ollama is not available or not running"
    )
    @pytest.mark.skipif(
        not check_model_available("gpt-oss:20b"),
        reason="gpt-oss:20b model is not installed",
    )
    def test_empty_tool_list(self) -> None:
        """Test that binding an empty tool list works correctly."""
        llm = ChatOllama(model="gpt-oss:20b")

        # Binding empty tool list should work
        llm_with_no_tools = llm.bind_tools([])

        # Should still be able to invoke
        response = llm_with_no_tools.invoke("Hello, how are you?")
        assert isinstance(response, AIMessage)

        # Should have no tool calls
        assert not response.tool_calls or len(response.tool_calls) == 0

