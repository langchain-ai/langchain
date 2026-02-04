"""Unit tests for TaskShieldMiddleware."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain.agents.middleware import TaskShieldMiddleware


@pytest.fixture
def mock_model_aligned():
    """Create a mock LLM that returns YES (aligned)."""
    model = MagicMock()
    model.invoke.return_value = AIMessage(content="YES")
    model.ainvoke = AsyncMock(return_value=AIMessage(content="YES"))
    return model


@pytest.fixture
def mock_model_misaligned():
    """Create a mock LLM that returns NO (misaligned)."""
    model = MagicMock()
    model.invoke.return_value = AIMessage(content="NO")
    model.ainvoke = AsyncMock(return_value=AIMessage(content="NO"))
    return model


@pytest.fixture
def mock_tool_request():
    """Create a mock ToolCallRequest."""
    request = MagicMock()
    request.tool_call = {
        "id": "call_123",
        "name": "send_email",
        "args": {"to": "user@example.com", "subject": "Hello"},
    }
    request.state = {
        "messages": [HumanMessage(content="Send an email to user@example.com saying hello")],
    }
    return request


class TestTaskShieldMiddleware:
    """Tests for TaskShieldMiddleware."""

    def test_init(self, mock_model_aligned):
        """Test middleware initialization."""
        middleware = TaskShieldMiddleware(mock_model_aligned)
        assert middleware.name == "TaskShieldMiddleware"
        assert middleware.strict is True
        assert middleware.cache_goal is True

    def test_init_with_options(self, mock_model_aligned):
        """Test middleware initialization with options."""
        middleware = TaskShieldMiddleware(
            mock_model_aligned,
            strict=False,
            cache_goal=False,
            log_verifications=True,
        )
        assert middleware.strict is False
        assert middleware.cache_goal is False
        assert middleware.log_verifications is True

    def test_extract_user_goal(self, mock_model_aligned):
        """Test extracting user goal from state."""
        middleware = TaskShieldMiddleware(mock_model_aligned)

        state = {"messages": [HumanMessage(content="Book a flight to Paris")]}
        goal = middleware._extract_user_goal(state)
        assert goal == "Book a flight to Paris"

    def test_extract_user_goal_caching(self, mock_model_aligned):
        """Test goal caching behavior."""
        middleware = TaskShieldMiddleware(mock_model_aligned, cache_goal=True)

        state1 = {"messages": [HumanMessage(content="First goal")]}
        state2 = {"messages": [HumanMessage(content="Second goal")]}

        goal1 = middleware._extract_user_goal(state1)
        goal2 = middleware._extract_user_goal(state2)

        assert goal1 == "First goal"
        assert goal2 == "First goal"  # Cached, not re-extracted

        middleware.reset_goal()
        goal3 = middleware._extract_user_goal(state2)
        assert goal3 == "Second goal"

    def test_wrap_tool_call_aligned(self, mock_model_aligned, mock_tool_request):
        """Test that aligned actions are allowed."""
        middleware = TaskShieldMiddleware(mock_model_aligned)
        middleware._cached_model = mock_model_aligned

        handler = MagicMock()
        handler.return_value = ToolMessage(content="Email sent", tool_call_id="call_123")

        result = middleware.wrap_tool_call(mock_tool_request, handler)

        mock_model_aligned.invoke.assert_called_once()
        handler.assert_called_once_with(mock_tool_request)
        assert isinstance(result, ToolMessage)
        assert result.content == "Email sent"

    def test_wrap_tool_call_misaligned_strict(self, mock_model_misaligned, mock_tool_request):
        """Test that misaligned actions are blocked in strict mode."""
        middleware = TaskShieldMiddleware(mock_model_misaligned, strict=True)
        middleware._cached_model = mock_model_misaligned

        handler = MagicMock()

        result = middleware.wrap_tool_call(mock_tool_request, handler)

        mock_model_misaligned.invoke.assert_called_once()
        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert "blocked" in result.content.lower()

    def test_wrap_tool_call_misaligned_non_strict(self, mock_model_misaligned, mock_tool_request):
        """Test that misaligned actions are allowed in non-strict mode."""
        middleware = TaskShieldMiddleware(mock_model_misaligned, strict=False)
        middleware._cached_model = mock_model_misaligned

        handler = MagicMock()
        handler.return_value = ToolMessage(content="Email sent", tool_call_id="call_123")

        result = middleware.wrap_tool_call(mock_tool_request, handler)

        mock_model_misaligned.invoke.assert_called_once()
        handler.assert_called_once()
        assert result.content == "Email sent"

    def test_verify_alignment_yes(self, mock_model_aligned):
        """Test alignment verification returns True for YES."""
        middleware = TaskShieldMiddleware(mock_model_aligned)
        middleware._cached_model = mock_model_aligned

        result = middleware._verify_alignment(
            user_goal="Send email",
            tool_name="send_email",
            tool_args={"to": "test@example.com"},
        )
        assert result is True

    def test_verify_alignment_no(self, mock_model_misaligned):
        """Test alignment verification returns False for NO."""
        middleware = TaskShieldMiddleware(mock_model_misaligned)
        middleware._cached_model = mock_model_misaligned

        result = middleware._verify_alignment(
            user_goal="Send email",
            tool_name="delete_all_files",
            tool_args={},
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_awrap_tool_call_aligned(self, mock_model_aligned, mock_tool_request):
        """Test async aligned actions are allowed."""
        middleware = TaskShieldMiddleware(mock_model_aligned)
        middleware._cached_model = mock_model_aligned

        async def async_handler(req):
            return ToolMessage(content="Email sent", tool_call_id="call_123")

        result = await middleware.awrap_tool_call(mock_tool_request, async_handler)

        mock_model_aligned.ainvoke.assert_called_once()
        assert isinstance(result, ToolMessage)
        assert result.content == "Email sent"

    @pytest.mark.asyncio
    async def test_awrap_tool_call_misaligned_strict(self, mock_model_misaligned, mock_tool_request):
        """Test async misaligned actions are blocked in strict mode."""
        middleware = TaskShieldMiddleware(mock_model_misaligned, strict=True)
        middleware._cached_model = mock_model_misaligned

        async def async_handler(req):
            return ToolMessage(content="Email sent", tool_call_id="call_123")

        result = await middleware.awrap_tool_call(mock_tool_request, async_handler)

        mock_model_misaligned.ainvoke.assert_called_once()
        assert isinstance(result, ToolMessage)
        assert "blocked" in result.content.lower()

    def test_reset_goal(self, mock_model_aligned):
        """Test goal reset functionality."""
        middleware = TaskShieldMiddleware(mock_model_aligned, cache_goal=True)

        state = {"messages": [HumanMessage(content="Original goal")]}
        middleware._extract_user_goal(state)
        assert middleware._cached_goal == "Original goal"

        middleware.reset_goal()
        assert middleware._cached_goal is None

    def test_init_with_string_model(self):
        """Test initialization with model string."""
        with patch("langchain.chat_models.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()

            middleware = TaskShieldMiddleware("openai:gpt-4o-mini")
            model = middleware._get_model()

            mock_init.assert_called_once_with("openai:gpt-4o-mini")


class TestTaskShieldIntegration:
    """Integration-style tests for TaskShieldMiddleware."""

    def test_goal_hijacking_blocked(self, mock_model_misaligned):
        """Test that goal hijacking attempts are blocked."""
        middleware = TaskShieldMiddleware(mock_model_misaligned, strict=True)
        middleware._cached_model = mock_model_misaligned

        request = MagicMock()
        request.tool_call = {
            "id": "call_456",
            "name": "send_money",
            "args": {"to": "attacker@evil.com", "amount": 1000},
        }
        request.state = {
            "messages": [HumanMessage(content="Check my account balance")],
        }

        handler = MagicMock()

        result = middleware.wrap_tool_call(request, handler)

        handler.assert_not_called()
        assert "blocked" in result.content.lower()

    def test_legitimate_action_allowed(self, mock_model_aligned):
        """Test that legitimate actions are allowed."""
        middleware = TaskShieldMiddleware(mock_model_aligned, strict=True)
        middleware._cached_model = mock_model_aligned

        request = MagicMock()
        request.tool_call = {
            "id": "call_789",
            "name": "get_balance",
            "args": {"account_id": "12345"},
        }
        request.state = {
            "messages": [HumanMessage(content="Check my account balance")],
        }

        handler = MagicMock()
        handler.return_value = ToolMessage(
            content="Balance: $1,234.56",
            tool_call_id="call_789",
        )

        result = middleware.wrap_tool_call(request, handler)

        handler.assert_called_once()
        assert result.content == "Balance: $1,234.56"
