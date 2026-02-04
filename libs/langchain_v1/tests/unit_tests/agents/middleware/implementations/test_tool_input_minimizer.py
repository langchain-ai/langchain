"""Unit tests for ToolInputMinimizerMiddleware."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain.agents.middleware import ToolInputMinimizerMiddleware


@pytest.fixture
def mock_model():
    """Create a mock LLM that returns minimized args."""
    from unittest.mock import AsyncMock

    model = MagicMock()
    model.invoke.return_value = AIMessage(content='{"query": "weather"}')
    model.ainvoke = AsyncMock(return_value=AIMessage(content='{"query": "weather"}'))
    return model


@pytest.fixture
def mock_tool_request():
    """Create a mock ToolCallRequest."""
    request = MagicMock()
    request.tool_call = {
        "id": "call_123",
        "name": "search",
        "args": {
            "query": "weather",
            "secret_key": "sk-12345",
            "user_ssn": "123-45-6789",
        },
    }
    request.tool = MagicMock()
    request.tool.name = "search"
    request.tool.description = "Search the web for information"
    request.state = {
        "messages": [HumanMessage(content="What's the weather today?")],
    }
    request.override.return_value = request
    return request


class TestToolInputMinimizerMiddleware:
    """Tests for ToolInputMinimizerMiddleware."""

    def test_init(self, mock_model):
        """Test middleware initialization."""
        middleware = ToolInputMinimizerMiddleware(mock_model)
        assert middleware.name == "ToolInputMinimizerMiddleware"
        assert middleware.strict is False
        assert middleware.log_minimizations is False

    def test_init_with_options(self, mock_model):
        """Test middleware initialization with options."""
        middleware = ToolInputMinimizerMiddleware(
            mock_model,
            strict=True,
            log_minimizations=True,
        )
        assert middleware.strict is True
        assert middleware.log_minimizations is True

    def test_extract_user_task(self, mock_model):
        """Test extracting user task from state."""
        middleware = ToolInputMinimizerMiddleware(mock_model)

        state = {"messages": [HumanMessage(content="Find me a recipe")]}
        task = middleware._extract_user_task(state)
        assert task == "Find me a recipe"

    def test_extract_user_task_empty(self, mock_model):
        """Test extracting user task from empty state."""
        middleware = ToolInputMinimizerMiddleware(mock_model)

        state = {"messages": []}
        task = middleware._extract_user_task(state)
        assert task == "Complete the requested task."

    def test_wrap_tool_call_minimizes_args(self, mock_model, mock_tool_request):
        """Test that wrap_tool_call minimizes arguments."""
        middleware = ToolInputMinimizerMiddleware(mock_model)
        middleware._cached_model = mock_model

        handler = MagicMock()
        handler.return_value = ToolMessage(content="Result", tool_call_id="call_123")

        result = middleware.wrap_tool_call(mock_tool_request, handler)

        mock_model.invoke.assert_called_once()
        mock_tool_request.override.assert_called_once()
        handler.assert_called_once()
        assert isinstance(result, ToolMessage)

    def test_wrap_tool_call_skips_empty_args(self, mock_model):
        """Test that wrap_tool_call skips tools with no args."""
        middleware = ToolInputMinimizerMiddleware(mock_model)

        request = MagicMock()
        request.tool_call = {"id": "call_123", "name": "get_time", "args": {}}

        handler = MagicMock()
        handler.return_value = ToolMessage(content="12:00 PM", tool_call_id="call_123")

        result = middleware.wrap_tool_call(request, handler)

        mock_model.invoke.assert_not_called()
        handler.assert_called_once_with(request)

    def test_parse_response_json(self, mock_model):
        """Test parsing JSON response."""
        middleware = ToolInputMinimizerMiddleware(mock_model)

        response = AIMessage(content='{"key": "value"}')
        result = middleware._parse_response(response, {"original": "args"})
        assert result == {"key": "value"}

    def test_parse_response_markdown_json(self, mock_model):
        """Test parsing JSON in markdown code block."""
        middleware = ToolInputMinimizerMiddleware(mock_model)

        response = AIMessage(content='```json\n{"key": "value"}\n```')
        result = middleware._parse_response(response, {"original": "args"})
        assert result == {"key": "value"}

    def test_parse_response_fallback_non_strict(self, mock_model):
        """Test parsing fallback in non-strict mode."""
        middleware = ToolInputMinimizerMiddleware(mock_model, strict=False)

        response = AIMessage(content="This is not JSON")
        original = {"original": "args"}
        result = middleware._parse_response(response, original)
        assert result == original

    def test_parse_response_strict_raises(self, mock_model):
        """Test parsing raises in strict mode."""
        middleware = ToolInputMinimizerMiddleware(mock_model, strict=True)

        response = AIMessage(content="This is not JSON")
        with pytest.raises(ValueError, match="Failed to parse"):
            middleware._parse_response(response, {"original": "args"})

    def test_strict_mode_blocks_on_failure(self, mock_model, mock_tool_request):
        """Test strict mode blocks tool call on minimization failure."""
        mock_model.invoke.return_value = AIMessage(content="invalid json")
        middleware = ToolInputMinimizerMiddleware(mock_model, strict=True)
        middleware._cached_model = mock_model

        handler = MagicMock()

        result = middleware.wrap_tool_call(mock_tool_request, handler)

        assert isinstance(result, ToolMessage)
        assert "blocked" in result.content.lower()
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_awrap_tool_call_minimizes_args(self, mock_model, mock_tool_request):
        """Test async wrap_tool_call minimizes arguments."""
        middleware = ToolInputMinimizerMiddleware(mock_model)
        middleware._cached_model = mock_model

        async def async_handler(req):
            return ToolMessage(content="Result", tool_call_id="call_123")

        result = await middleware.awrap_tool_call(mock_tool_request, async_handler)

        mock_model.ainvoke.assert_called_once()
        mock_tool_request.override.assert_called_once()
        assert isinstance(result, ToolMessage)


class TestToolInputMinimizerIntegration:
    """Integration-style tests for ToolInputMinimizerMiddleware."""

    def test_minimizer_removes_sensitive_data(self, mock_model, mock_tool_request):
        """Test that minimizer removes sensitive data from args."""
        mock_model.invoke.return_value = AIMessage(content='{"query": "weather"}')

        middleware = ToolInputMinimizerMiddleware(mock_model)
        middleware._cached_model = mock_model

        captured_request = None

        def capture_handler(req):
            nonlocal captured_request
            captured_request = req
            return ToolMessage(content="Sunny", tool_call_id="call_123")

        middleware.wrap_tool_call(mock_tool_request, capture_handler)

        call_args = mock_tool_request.override.call_args
        modified_call = call_args.kwargs["tool_call"]
        assert modified_call["args"] == {"query": "weather"}
        assert "secret_key" not in modified_call["args"]
        assert "user_ssn" not in modified_call["args"]

    def test_init_with_string_model(self):
        """Test initialization with model string."""
        with patch("langchain.chat_models.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()

            middleware = ToolInputMinimizerMiddleware("openai:gpt-4o-mini")
            model = middleware._get_model()

            mock_init.assert_called_once_with("openai:gpt-4o-mini")
