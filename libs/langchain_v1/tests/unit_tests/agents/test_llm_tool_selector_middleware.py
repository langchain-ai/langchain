"""Tests for LLM Tool Selector middleware."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from langchain.agents.middleware.llm_tool_selector import (
    LLMToolSelectorMiddleware,
    ToolSelectionSchema,
)
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool


class MockTool(BaseTool):
    """Mock tool for testing."""

    name: str = "mock_tool"
    description: str = "A mock tool for testing"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return "mock result"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return "mock result"


class MockChatModel(BaseChatModel):
    """Mock chat model for testing."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # This is a placeholder - we'll mock the structured output
        pass

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        # This is a placeholder - we'll mock the structured output
        pass

    @property
    def _llm_type(self) -> str:
        return "mock"


def test_tool_selection_schema():
    """Test that ToolSelectionSchema works correctly."""
    schema = ToolSelectionSchema(selected_tools=["tool1", "tool2"])
    assert schema.selected_tools == ["tool1", "tool2"]


def test_middleware_initialization():
    """Test that middleware initializes with correct defaults."""
    middleware = LLMToolSelectorMiddleware()

    assert middleware.model is None
    assert (
        middleware.system_prompt
        == "Your goal is to select the most relevant tool for answering the user's query."
    )
    assert middleware.max_tools is None
    assert middleware.include_full_history is False
    assert middleware.max_retries == 3


def test_middleware_initialization_with_custom_values():
    """Test that middleware initializes with custom values."""
    middleware = LLMToolSelectorMiddleware(
        model="openai:gpt-4o",
        system_prompt="Custom prompt",
        max_tools=5,
        include_full_history=True,
        max_retries=2,
    )

    assert middleware.model == "openai:gpt-4o"
    assert middleware.system_prompt == "Custom prompt"
    assert middleware.max_tools == 5
    assert middleware.include_full_history is True
    assert middleware.max_retries == 2


@patch("langchain.agents.middleware.llm_tool_selector.init_chat_model")
def test_modify_model_request_no_tools(mock_init_chat_model):
    """Test that middleware returns request unchanged when no tools are available."""
    middleware = LLMToolSelectorMiddleware()

    request = ModelRequest(
        model=MockChatModel(),
        system_prompt="Test prompt",
        messages=[HumanMessage("Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
    )

    runtime = MagicMock()
    runtime.tools = []

    result = middleware.modify_model_request(request, {}, runtime)

    assert result == request
    mock_init_chat_model.assert_not_called()


@patch("langchain.agents.middleware.llm_tool_selector.init_chat_model")
def test_modify_model_request_with_tool_selection(mock_init_chat_model):
    """Test that middleware filters tools based on LLM selection."""
    # Create mock tools
    tool1 = MockTool(name="tool1", description="First tool")
    tool2 = MockTool(name="tool2", description="Second tool")
    tool3 = MockTool(name="tool3", description="Third tool")

    # Create mock structured model
    mock_structured_model = MagicMock()
    mock_response = ToolSelectionSchema(selected_tools=["tool1", "tool3"])
    mock_structured_model.invoke.return_value = mock_response

    # Create mock chat model
    mock_chat_model = MagicMock()
    mock_chat_model.with_structured_output.return_value = mock_structured_model

    middleware = LLMToolSelectorMiddleware(model=mock_chat_model)

    request = ModelRequest(
        model=MockChatModel(),
        system_prompt="Test prompt",
        messages=[HumanMessage("Hello")],
        tool_choice=None,
        tools=["tool1", "tool2", "tool3"],
        response_format=None,
    )

    runtime = MagicMock()
    runtime.tools = [tool1, tool2, tool3]

    result = middleware.modify_model_request(request, {}, runtime)

    # Verify that only selected tools are returned
    assert result.tools == ["tool1", "tool3"]
    assert result.model == request.model
    assert result.system_prompt == request.system_prompt
    assert result.messages == request.messages


@patch("langchain.agents.middleware.llm_tool_selector.init_chat_model")
def test_modify_model_request_with_string_model(mock_init_chat_model):
    """Test that middleware works with string model specification."""
    # Create mock tools
    tool1 = MockTool(name="tool1", description="First tool")

    # Create mock structured model
    mock_structured_model = MagicMock()
    mock_response = ToolSelectionSchema(selected_tools=["tool1"])
    mock_structured_model.invoke.return_value = mock_response

    # Create mock chat model
    mock_chat_model = MagicMock()
    mock_chat_model.with_structured_output.return_value = mock_structured_model

    mock_init_chat_model.return_value = mock_chat_model

    middleware = LLMToolSelectorMiddleware(model="openai:gpt-4o")

    request = ModelRequest(
        model=MockChatModel(),
        system_prompt="Test prompt",
        messages=[HumanMessage("Hello")],
        tool_choice=None,
        tools=["tool1"],
        response_format=None,
    )

    runtime = MagicMock()
    runtime.tools = [tool1]

    result = middleware.modify_model_request(request, {}, runtime)

    # Verify that init_chat_model was called with the string model
    mock_init_chat_model.assert_called_once_with("openai:gpt-4o")

    # Verify that only selected tools are returned
    assert result.tools == ["tool1"]


@patch("langchain.agents.middleware.llm_tool_selector.init_chat_model")
def test_modify_model_request_with_retry_logic(mock_init_chat_model):
    """Test that middleware retries on invalid tool selection."""
    # Create mock tools
    tool1 = MockTool(name="tool1", description="First tool")
    tool2 = MockTool(name="tool2", description="Second tool")

    # Create mock structured model that returns invalid tools first, then valid ones
    mock_structured_model = MagicMock()
    mock_responses = [
        ToolSelectionSchema(selected_tools=["invalid_tool"]),  # First attempt - invalid
        ToolSelectionSchema(selected_tools=["tool1"]),  # Second attempt - valid
    ]
    mock_structured_model.invoke.side_effect = mock_responses

    # Create mock chat model
    mock_chat_model = MagicMock()
    mock_chat_model.with_structured_output.return_value = mock_structured_model

    middleware = LLMToolSelectorMiddleware(model=mock_chat_model, max_retries=3)

    request = ModelRequest(
        model=MockChatModel(),
        system_prompt="Test prompt",
        messages=[HumanMessage("Hello")],
        tool_choice=None,
        tools=["tool1", "tool2"],
        response_format=None,
    )

    runtime = MagicMock()
    runtime.tools = [tool1, tool2]

    result = middleware.modify_model_request(request, {}, runtime)

    # Verify that the model was called twice (initial + retry)
    assert mock_structured_model.invoke.call_count == 2

    # Verify that only valid tools are returned
    assert result.tools == ["tool1"]


@patch("langchain.agents.middleware.llm_tool_selector.init_chat_model")
def test_modify_model_request_with_max_tools_limit(mock_init_chat_model):
    """Test that middleware enforces max_tools limit."""
    # Create mock tools
    tool1 = MockTool(name="tool1", description="First tool")
    tool2 = MockTool(name="tool2", description="Second tool")
    tool3 = MockTool(name="tool3", description="Third tool")

    # Create mock structured model that returns too many tools first, then correct amount
    mock_structured_model = MagicMock()
    mock_responses = [
        ToolSelectionSchema(selected_tools=["tool1", "tool2", "tool3"]),  # Too many
        ToolSelectionSchema(selected_tools=["tool1", "tool2"]),  # Correct amount
    ]
    mock_structured_model.invoke.side_effect = mock_responses

    # Create mock chat model
    mock_chat_model = MagicMock()
    mock_chat_model.with_structured_output.return_value = mock_structured_model

    middleware = LLMToolSelectorMiddleware(model=mock_chat_model, max_tools=2)

    request = ModelRequest(
        model=MockChatModel(),
        system_prompt="Test prompt",
        messages=[HumanMessage("Hello")],
        tool_choice=None,
        tools=["tool1", "tool2", "tool3"],
        response_format=None,
    )

    runtime = MagicMock()
    runtime.tools = [tool1, tool2, tool3]

    result = middleware.modify_model_request(request, {}, runtime)

    # Verify that the model was called twice (initial + retry)
    assert mock_structured_model.invoke.call_count == 2

    # Verify that only the allowed number of tools are returned
    assert result.tools == ["tool1", "tool2"]


@patch("langchain.agents.middleware.llm_tool_selector.init_chat_model")
def test_modify_model_request_fallback_on_exception(mock_init_chat_model):
    """Test that middleware falls back to original request on exception."""
    # Create mock tools
    tool1 = MockTool(name="tool1", description="First tool")

    # Create mock structured model that raises an exception
    mock_structured_model = MagicMock()
    mock_structured_model.invoke.side_effect = Exception("Model error")

    # Create mock chat model
    mock_chat_model = MagicMock()
    mock_chat_model.with_structured_output.return_value = mock_structured_model

    middleware = LLMToolSelectorMiddleware(model=mock_chat_model, max_retries=1)

    request = ModelRequest(
        model=MockChatModel(),
        system_prompt="Test prompt",
        messages=[HumanMessage("Hello")],
        tool_choice=None,
        tools=["tool1"],
        response_format=None,
    )

    runtime = MagicMock()
    runtime.tools = [tool1]

    result = middleware.modify_model_request(request, {}, runtime)

    # Verify that the original request is returned on exception
    assert result == request
