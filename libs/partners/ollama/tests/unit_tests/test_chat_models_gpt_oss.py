"""Unit tests for gpt-oss model support in ChatOllama."""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_ollama.chat_models import ChatOllama


@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the weather for a location.
    
    Args:
        location: The city to get weather for.
        unit: Temperature unit (celsius or fahrenheit).
    """
    return f"Weather in {location}: 22 {unit}"


class TestGptOssModelDetection:
    """Test detection of gpt-oss models."""
    
    def test_detects_gpt_oss_model(self) -> None:
        """Test that gpt-oss models are correctly detected."""
        from langchain_ollama.chat_models import _is_gpt_oss_model
        
        # Should detect gpt-oss models
        assert _is_gpt_oss_model("gpt-oss:20b") is True
        assert _is_gpt_oss_model("gpt-oss:latest") is True
        assert _is_gpt_oss_model("gpt-oss") is True
        assert _is_gpt_oss_model("gpt-oss:7b") is True
        
        # Should not detect non-gpt-oss models
        assert _is_gpt_oss_model("llama2") is False
        assert _is_gpt_oss_model("mistral") is False
        assert _is_gpt_oss_model("codellama") is False
        assert _is_gpt_oss_model("") is False
        assert _is_gpt_oss_model("gpt") is False
        assert _is_gpt_oss_model("oss") is False


class TestHarmonyToolConversion:
    """Test conversion of tools to Harmony format."""
    
    def test_convert_to_harmony_tool(self) -> None:
        """Test that tools are correctly converted to Harmony format."""
        from langchain_ollama.chat_models import _convert_to_harmony_tool
        from langchain_core.utils.function_calling import convert_to_openai_tool
        
        # Convert tool to OpenAI format first
        openai_tool = convert_to_openai_tool(get_weather)
        
        # Convert to Harmony format
        harmony_tool = _convert_to_harmony_tool(openai_tool)
        
        # Check structure
        assert "type" in harmony_tool
        assert harmony_tool["type"] == "function"
        assert "function" in harmony_tool
        
        func = harmony_tool["function"]
        assert "name" in func
        assert func["name"] == "get_weather"
        assert "description" in func
        assert "parameters" in func
        
        params = func["parameters"]
        assert "type" in params
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
        
        # Check properties
        props = params["properties"]
        assert "location" in props
        assert "unit" in props
        
        # Check that types are strings, not arrays
        assert isinstance(props["location"]["type"], str)
        assert props["location"]["type"] == "string"
        assert isinstance(props["unit"]["type"], str)
        assert props["unit"]["type"] == "string"
        
        # Check required fields
        assert "location" in params["required"]
        assert "unit" not in params["required"]  # Has default value
    
    def test_convert_complex_types(self) -> None:
        """Test conversion of complex parameter types."""
        from langchain_ollama.chat_models import _convert_to_harmony_tool
        from langchain_core.utils.function_calling import convert_to_openai_tool
        
        @tool
        def complex_tool(
            numbers: list[int],
            data: dict[str, Any],
            optional: str = None,
        ) -> str:
            """A tool with complex types.
            
            Args:
                numbers: List of numbers.
                data: Dictionary data.
                optional: Optional string parameter.
            """
            return "result"
        
        openai_tool = convert_to_openai_tool(complex_tool)
        harmony_tool = _convert_to_harmony_tool(openai_tool)
        
        props = harmony_tool["function"]["parameters"]["properties"]
        
        # Array types should be converted to "array"
        assert props["numbers"]["type"] == "array"
        if "items" in props["numbers"]:
            assert props["numbers"]["items"]["type"] == "integer"
        
        # Object types should be "object"
        assert props["data"]["type"] == "object"
        
        # Optional parameters with None default should be "string"
        if "optional" in props:
            assert props["optional"]["type"] == "string"


class TestGptOssToolBinding:
    """Test tool binding for gpt-oss models."""
    
    def test_bind_tools_with_gpt_oss(self) -> None:
        """Test that tools are correctly bound for gpt-oss models."""
        # Create ChatOllama instance with gpt-oss model
        llm = ChatOllama(model="gpt-oss:20b")
        
        # Bind tools
        llm_with_tools = llm.bind_tools([get_weather])
        
        # Check that tools are bound
        assert hasattr(llm_with_tools, "kwargs")
        assert "tools" in llm_with_tools.kwargs
        
        # Tools should be in Harmony format
        tools = llm_with_tools.kwargs["tools"]
        assert len(tools) == 1
        
        tool_def = tools[0]
        assert tool_def["type"] == "function"
        assert "function" in tool_def
        
        # Check parameter types are strings
        params = tool_def["function"]["parameters"]["properties"]
        for prop in params.values():
            if "type" in prop:
                assert isinstance(prop["type"], str)
    
    def test_bind_tools_with_non_gpt_oss(self) -> None:
        """Test that tools use standard format for non-gpt-oss models."""
        # Create ChatOllama instance with non-gpt-oss model
        llm = ChatOllama(model="llama2")
        
        # Bind tools
        llm_with_tools = llm.bind_tools([get_weather])
        
        # Check that tools are bound
        assert hasattr(llm_with_tools, "kwargs")
        assert "tools" in llm_with_tools.kwargs
        
        # Tools should be in standard OpenAI format
        tools = llm_with_tools.kwargs["tools"]
        assert len(tools) == 1
        
        tool_def = tools[0]
        assert tool_def["type"] == "function"
        assert "function" in tool_def


class TestGptOssResponseParsing:
    """Test parsing of tool call responses from gpt-oss models."""
    
    def test_parse_standard_tool_response(self) -> None:
        """Test parsing standard format tool response."""
        from langchain_ollama.chat_models import _get_tool_calls_from_response
        
        response = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London", "unit": "celsius"}'
                        }
                    }
                ]
            }
        }
        
        tool_calls = _get_tool_calls_from_response(response, model_name="gpt-oss:20b")
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["args"] == {"location": "London", "unit": "celsius"}
    
    def test_parse_direct_tool_response(self) -> None:
        """Test parsing direct format tool response (without nested function)."""
        from langchain_ollama.chat_models import _get_tool_calls_from_response
        
        response = {
            "message": {
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}'
                    }
                ]
            }
        }
        
        tool_calls = _get_tool_calls_from_response(response, model_name="gpt-oss:20b")
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["args"] == {"location": "Paris"}
    
    def test_parse_malformed_tool_response(self) -> None:
        """Test handling of malformed tool responses."""
        from langchain_ollama.chat_models import _get_tool_calls_from_response
        
        # Response with invalid JSON in arguments
        response = {
            "message": {
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "arguments": "invalid json"
                    }
                ]
            }
        }
        
        # Should handle gracefully and return empty list or skip invalid calls
        tool_calls = _get_tool_calls_from_response(response, model_name="gpt-oss:20b")
        
        # Either returns empty list or skips the malformed call
        assert len(tool_calls) == 0
    
    def test_parse_empty_tool_response(self) -> None:
        """Test handling of responses without tool calls."""
        from langchain_ollama.chat_models import _get_tool_calls_from_response
        
        response = {
            "message": {
                "content": "I'll help you with that."
            }
        }
        
        tool_calls = _get_tool_calls_from_response(response, model_name="gpt-oss:20b")
        
        assert len(tool_calls) == 0
    
    def test_parse_non_gpt_oss_response(self) -> None:
        """Test that non-gpt-oss models use standard parsing."""
        from langchain_ollama.chat_models import _get_tool_calls_from_response
        
        response = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo"}'
                        }
                    }
                ]
            }
        }
        
        # Should use standard parsing for non-gpt-oss model
        tool_calls = _get_tool_calls_from_response(response, model_name="llama2")
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["args"] == {"location": "Tokyo"}


class TestGptOssIntegration:
    """Integration tests for gpt-oss model with ChatOllama."""
    
    def test_tool_binding_integration(self) -> None:
        """Test that tool binding works correctly for gpt-oss models."""
        # Create ChatOllama with gpt-oss model
        llm = ChatOllama(model="gpt-oss:20b")
        
        # Define multiple tools using function definitions
        def search_web(query: str) -> str:
            """Search the web for information.
            
            Args:
                query: The search query.
            """
            return f"Results for {query}"
        
        def calculate(expression: str) -> str:
            """Calculate a mathematical expression.
            
            Args:
                expression: The math expression to evaluate.
            """
            return "42"
        
        # Convert functions to tools
        search_tool = tool(search_web)
        calc_tool = tool(calculate)
        
        # Bind multiple tools
        llm_with_tools = llm.bind_tools([get_weather, search_tool, calc_tool])
        
        # Check that all tools are bound correctly
        assert hasattr(llm_with_tools, "kwargs")
        assert "tools" in llm_with_tools.kwargs
        tools = llm_with_tools.kwargs["tools"]
        assert len(tools) == 3
        
        # Verify each tool is in Harmony format
        tool_names = {tool["function"]["name"] for tool in tools}
        assert tool_names == {"get_weather", "search_web", "calculate"}
        
        # Check that all tools have proper Harmony format
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            
            # Verify parameter types are strings
            if "properties" in func["parameters"]:
                for prop in func["parameters"]["properties"].values():
                    if "type" in prop:
                        assert isinstance(prop["type"], str)
    
    def test_tool_format_consistency(self) -> None:
        """Test that tool format is consistent across multiple bindings."""
        llm = ChatOllama(model="gpt-oss:20b")
        
        # First binding
        llm_with_tool1 = llm.bind_tools([get_weather])
        tools1 = llm_with_tool1.kwargs["tools"]
        
        # Second binding with same tool
        llm_with_tool2 = llm.bind_tools([get_weather])
        tools2 = llm_with_tool2.kwargs["tools"]
        
        # Tools should be formatted identically
        assert tools1 == tools2
        
        # Both should be in Harmony format
        for tools in [tools1, tools2]:
            assert len(tools) == 1
            tool = tools[0]
            assert tool["type"] == "function"
            assert "function" in tool
            props = tool["function"]["parameters"]["properties"]
            for prop in props.values():
                if "type" in prop:
                    assert isinstance(prop["type"], str)


class TestChatParamsWithGptOss:
    """Test _chat_params method with gpt-oss models."""
    
    def test_chat_params_with_harmony_tools(self) -> None:
        """Test that _chat_params correctly formats tools for gpt-oss."""
        llm = ChatOllama(model="gpt-oss:20b")
        llm_with_tools = llm.bind_tools([get_weather])
        
        # Get chat params
        messages = [HumanMessage(content="What's the weather?")]
        params = llm_with_tools._chat_params(messages)
        
        # Check that tools are included and in Harmony format
        assert "tools" in params
        tools = params["tools"]
        assert len(tools) == 1
        
        # Verify Harmony format
        tool = tools[0]
        assert tool["type"] == "function"
        assert "function" in tool
        
        # Check that parameter types are strings
        props = tool["function"]["parameters"]["properties"]
        for prop in props.values():
            if "type" in prop:
                assert isinstance(prop["type"], str)








