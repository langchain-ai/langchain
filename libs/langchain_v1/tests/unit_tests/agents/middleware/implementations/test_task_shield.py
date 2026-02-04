"""Unit tests for TaskShieldMiddleware."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

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


@pytest.fixture
def mock_tool_request_with_system():
    """Create a mock ToolCallRequest with system prompt."""
    request = MagicMock()
    request.tool_call = {
        "id": "call_123",
        "name": "send_email",
        "args": {"to": "user@example.com", "subject": "Hello"},
    }
    request.state = {
        "messages": [
            SystemMessage(content="You are a helpful assistant. Never delete files."),
            HumanMessage(content="Send an email to user@example.com saying hello"),
        ],
    }
    return request


class TestTaskShieldMiddleware:
    """Tests for TaskShieldMiddleware."""

    def test_init(self, mock_model_aligned):
        """Test middleware initialization."""
        middleware = TaskShieldMiddleware(mock_model_aligned)
        assert middleware.name == "TaskShieldMiddleware"
        assert middleware.strict is True

    def test_init_with_options(self, mock_model_aligned):
        """Test middleware initialization with options."""
        middleware = TaskShieldMiddleware(
            mock_model_aligned,
            strict=False,
            log_verifications=True,
        )
        assert middleware.strict is False
        assert middleware.log_verifications is True

    def test_extract_user_goal(self, mock_model_aligned):
        """Test extracting user goal from state (returns LAST HumanMessage)."""
        middleware = TaskShieldMiddleware(mock_model_aligned)

        state = {"messages": [HumanMessage(content="Book a flight to Paris")]}
        goal = middleware._extract_user_goal(state)
        assert goal == "Book a flight to Paris"

    def test_extract_user_goal_multiple_messages(self, mock_model_aligned):
        """Test that _extract_user_goal returns the LAST HumanMessage (most recent intent)."""
        middleware = TaskShieldMiddleware(mock_model_aligned)

        # Simulate multi-turn: user first asks one thing, then another
        state = {
            "messages": [
                HumanMessage(content="First goal"),
                AIMessage(content="I'll help with that."),
                HumanMessage(content="Second goal"),
            ]
        }

        goal = middleware._extract_user_goal(state)
        assert goal == "Second goal"  # Most recent user intent

    def test_extract_system_prompt(self, mock_model_aligned):
        """Test extracting system prompt from state."""
        middleware = TaskShieldMiddleware(mock_model_aligned)

        state = {
            "messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Do something"),
            ]
        }
        system_prompt = middleware._extract_system_prompt(state)
        assert system_prompt == "You are a helpful assistant."

    def test_extract_system_prompt_not_found(self, mock_model_aligned):
        """Test that missing system prompt returns empty string."""
        middleware = TaskShieldMiddleware(mock_model_aligned)

        state = {"messages": [HumanMessage(content="Do something")]}
        system_prompt = middleware._extract_system_prompt(state)
        assert system_prompt == ""

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

        is_aligned, minimized_args = middleware._verify_alignment(
            system_prompt="",
            user_goal="Send email",
            tool_name="send_email",
            tool_args={"to": "test@example.com"},
        )
        assert is_aligned is True
        assert minimized_args is None  # minimize=False by default

    def test_verify_alignment_no(self, mock_model_misaligned):
        """Test alignment verification returns False for NO."""
        middleware = TaskShieldMiddleware(mock_model_misaligned)
        middleware._cached_model = mock_model_misaligned

        is_aligned, minimized_args = middleware._verify_alignment(
            system_prompt="",
            user_goal="Send email",
            tool_name="delete_all_files",
            tool_args={},
        )
        assert is_aligned is False
        assert minimized_args is None

    def test_verify_alignment_with_system_prompt(self, mock_model_aligned):
        """Test that system prompt is included in verification."""
        middleware = TaskShieldMiddleware(mock_model_aligned)
        middleware._cached_model = mock_model_aligned

        is_aligned, _ = middleware._verify_alignment(
            system_prompt="You are a helpful assistant. Never delete files.",
            user_goal="Send email",
            tool_name="send_email",
            tool_args={"to": "test@example.com"},
        )
        assert is_aligned is True

        # Verify the system prompt was included in the LLM call
        call_args = mock_model_aligned.invoke.call_args
        prompt_content = call_args[0][0][0]["content"]
        assert "System Guidelines" in prompt_content
        assert "Never delete files" in prompt_content

    def test_minimize_mode(self):
        """Test minimize mode returns minimized args."""
        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(
            content='YES\n```json\n{"to": "test@example.com"}\n```'
        )

        middleware = TaskShieldMiddleware(mock_model, minimize=True)
        middleware._cached_model = mock_model

        is_aligned, minimized_args = middleware._verify_alignment(
            system_prompt="",
            user_goal="Send email to test@example.com",
            tool_name="send_email",
            tool_args={"to": "test@example.com", "cc": "attacker@evil.com", "secret": "password123"},
        )
        assert is_aligned is True
        assert minimized_args == {"to": "test@example.com"}

    def test_minimize_mode_fallback_on_parse_failure(self):
        """Test minimize mode falls back to original args on parse failure."""
        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(content="YES")  # No JSON

        middleware = TaskShieldMiddleware(mock_model, minimize=True)
        middleware._cached_model = mock_model

        original_args = {"to": "test@example.com", "body": "hello"}
        is_aligned, minimized_args = middleware._verify_alignment(
            system_prompt="",
            user_goal="Send email",
            tool_name="send_email",
            tool_args=original_args,
        )
        assert is_aligned is True
        assert minimized_args == original_args  # Falls back to original

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

    def test_wrap_tool_call_with_system_prompt(
        self, mock_model_aligned, mock_tool_request_with_system
    ):
        """Test that system prompt is extracted and used in verification."""
        middleware = TaskShieldMiddleware(mock_model_aligned)
        middleware._cached_model = mock_model_aligned

        handler = MagicMock()
        handler.return_value = ToolMessage(content="Email sent", tool_call_id="call_123")

        result = middleware.wrap_tool_call(mock_tool_request_with_system, handler)

        # Verify the system prompt was included in the LLM call
        call_args = mock_model_aligned.invoke.call_args
        prompt_content = call_args[0][0][0]["content"]
        assert "System Guidelines" in prompt_content
        assert "Never delete files" in prompt_content
        assert isinstance(result, ToolMessage)
        assert result.content == "Email sent"

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
