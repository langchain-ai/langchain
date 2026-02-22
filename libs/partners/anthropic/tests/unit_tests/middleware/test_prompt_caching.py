"""Tests for Anthropic prompt caching middleware."""

import warnings
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.runtime import Runtime

from langchain_anthropic.chat_models import (
    ChatAnthropic,
    _collect_code_execution_tool_ids,
    _is_code_execution_related_block,
)
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware


class FakeToolCallingModel(BaseChatModel):
    """Fake model for testing middleware."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        messages_string = "-".join([str(m.content) for m in messages])
        message = AIMessage(content=messages_string, id="0")
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async top level call"""
        messages_string = "-".join([str(m.content) for m in messages])
        message = AIMessage(content=messages_string, id="0")
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-call-model"


def test_anthropic_prompt_caching_middleware_initialization() -> None:
    """Test AnthropicPromptCachingMiddleware initialization."""
    # Test with custom values
    middleware = AnthropicPromptCachingMiddleware(
        type="ephemeral", ttl="1h", min_messages_to_cache=5
    )
    assert middleware.type == "ephemeral"
    assert middleware.ttl == "1h"
    assert middleware.min_messages_to_cache == 5

    # Test with default values
    middleware = AnthropicPromptCachingMiddleware()
    assert middleware.type == "ephemeral"
    assert middleware.ttl == "5m"
    assert middleware.min_messages_to_cache == 0

    # Create a mock ChatAnthropic instance
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)

    fake_request = ModelRequest(
        model=mock_chat_anthropic,
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    modified_request: ModelRequest | None = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    middleware.wrap_model_call(fake_request, mock_handler)
    # Check that model_settings were passed through via the request
    assert modified_request is not None
    assert modified_request.model_settings == {
        "cache_control": {"type": "ephemeral", "ttl": "5m"}
    }


def test_anthropic_prompt_caching_middleware_unsupported_model() -> None:
    """Test AnthropicPromptCachingMiddleware with unsupported model."""
    fake_request = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="raise")

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Since we're in the langchain-anthropic package, ChatAnthropic is always
    # available. Test that it raises an error for unsupported model instances
    with pytest.raises(
        ValueError,
        match=(
            "AnthropicPromptCachingMiddleware caching middleware only supports "
            "Anthropic models, not instances of"
        ),
    ):
        middleware.wrap_model_call(fake_request, mock_handler)

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="warn")

    # Test warn behavior for unsupported model instances
    with warnings.catch_warnings(record=True) as w:
        result = middleware.wrap_model_call(fake_request, mock_handler)
        assert isinstance(result, ModelResponse)
        assert len(w) == 1
        assert (
            "AnthropicPromptCachingMiddleware caching middleware only supports "
            "Anthropic models, not instances of"
        ) in str(w[-1].message)

    # Test ignore behavior
    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")
    result = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)


async def test_anthropic_prompt_caching_middleware_async() -> None:
    """Test AnthropicPromptCachingMiddleware async path."""
    # Test with custom values
    middleware = AnthropicPromptCachingMiddleware(
        type="ephemeral", ttl="1h", min_messages_to_cache=5
    )

    # Create a mock ChatAnthropic instance
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)

    fake_request = ModelRequest(
        model=mock_chat_anthropic,
        messages=[HumanMessage("Hello")] * 6,
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")] * 6},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    modified_request: ModelRequest | None = None

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Check that model_settings were passed through via the request
    assert modified_request is not None
    assert modified_request.model_settings == {
        "cache_control": {"type": "ephemeral", "ttl": "1h"}
    }


async def test_anthropic_prompt_caching_middleware_async_unsupported_model() -> None:
    """Test AnthropicPromptCachingMiddleware async path with unsupported model."""
    fake_request = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="raise")

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Test that it raises an error for unsupported model instances
    with pytest.raises(
        ValueError,
        match=(
            "AnthropicPromptCachingMiddleware caching middleware only supports "
            "Anthropic models, not instances of"
        ),
    ):
        await middleware.awrap_model_call(fake_request, mock_handler)

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="warn")

    # Test warn behavior for unsupported model instances
    with warnings.catch_warnings(record=True) as w:
        result = await middleware.awrap_model_call(fake_request, mock_handler)
        assert isinstance(result, ModelResponse)
        assert len(w) == 1
        assert (
            "AnthropicPromptCachingMiddleware caching middleware only supports "
            "Anthropic models, not instances of"
        ) in str(w[-1].message)

    # Test ignore behavior
    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")
    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)


async def test_anthropic_prompt_caching_middleware_async_min_messages() -> None:
    """Test async path respects min_messages_to_cache."""
    middleware = AnthropicPromptCachingMiddleware(min_messages_to_cache=5)

    # Test with fewer messages than minimum
    fake_request = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello")] * 3,
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")] * 3},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    modified_request: ModelRequest | None = None

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Cache control should NOT be added when message count is below minimum
    assert modified_request is not None
    assert modified_request.model_settings == {}


async def test_anthropic_prompt_caching_middleware_async_with_system_prompt() -> None:
    """Test async path counts system prompt in message count."""
    middleware = AnthropicPromptCachingMiddleware(
        type="ephemeral", ttl="1h", min_messages_to_cache=3
    )

    # Create a mock ChatAnthropic instance
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)

    # Test with system prompt: 2 messages + 1 system = 3 total (meets minimum)
    fake_request = ModelRequest(
        model=mock_chat_anthropic,
        messages=[HumanMessage("Hello"), HumanMessage("World")],
        system_prompt="You are a helpful assistant",
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello"), HumanMessage("World")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    modified_request: ModelRequest | None = None

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Cache control should be added when system prompt pushes count to minimum
    assert modified_request is not None
    assert modified_request.model_settings == {
        "cache_control": {"type": "ephemeral", "ttl": "1h"}
    }


async def test_anthropic_prompt_caching_middleware_async_default_values() -> None:
    """Test async path with default middleware initialization."""
    # Test with default values (min_messages_to_cache=0)
    middleware = AnthropicPromptCachingMiddleware()

    # Create a mock ChatAnthropic instance
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)

    # Single message should trigger caching with default settings
    fake_request = ModelRequest(
        model=mock_chat_anthropic,
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    modified_request: ModelRequest | None = None

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Check that model_settings were added with default values
    assert modified_request is not None
    assert modified_request.model_settings == {
        "cache_control": {"type": "ephemeral", "ttl": "5m"}
    }


class TestCollectCodeExecutionToolIds:
    """Tests for _collect_code_execution_tool_ids function."""

    def test_empty_messages(self) -> None:
        """Test with empty messages list."""
        result = _collect_code_execution_tool_ids([])
        assert result == set()

    def test_no_code_execution_calls(self) -> None:
        """Test messages without any code_execution calls."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_regular",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                    }
                ],
            },
        ]
        result = _collect_code_execution_tool_ids(messages)
        assert result == set()

    def test_single_code_execution_call(self) -> None:
        """Test with a single code_execution tool call."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_code_exec_1",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                        "caller": {
                            "type": "code_execution_20250825",
                            "tool_id": "srvtoolu_abc123",
                        },
                    }
                ],
            },
        ]
        result = _collect_code_execution_tool_ids(messages)
        assert result == {"toolu_code_exec_1"}

    def test_multiple_code_execution_calls(self) -> None:
        """Test with multiple code_execution tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_regular",
                        "name": "search",
                        "input": {"query": "test"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_code_exec_1",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                        "caller": {
                            "type": "code_execution_20250825",
                            "tool_id": "srvtoolu_abc",
                        },
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_code_exec_2",
                        "name": "get_weather",
                        "input": {"location": "SF"},
                        "caller": {
                            "type": "code_execution_20250825",
                            "tool_id": "srvtoolu_def",
                        },
                    },
                ],
            },
        ]
        result = _collect_code_execution_tool_ids(messages)
        assert result == {"toolu_code_exec_1", "toolu_code_exec_2"}
        assert "toolu_regular" not in result

    def test_future_code_execution_version(self) -> None:
        """Test with a hypothetical future code_execution version."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_future",
                        "name": "get_weather",
                        "input": {},
                        "caller": {
                            "type": "code_execution_20260101",
                            "tool_id": "srvtoolu_future",
                        },
                    }
                ],
            },
        ]
        result = _collect_code_execution_tool_ids(messages)
        assert result == {"toolu_future"}

    def test_ignores_user_messages(self) -> None:
        """Test that user messages are ignored."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "result",
                    }
                ],
            },
        ]
        result = _collect_code_execution_tool_ids(messages)
        assert result == set()

    def test_handles_string_content(self) -> None:
        """Test that string content is handled gracefully."""
        messages = [
            {
                "role": "assistant",
                "content": "Just a text response",
            },
        ]
        result = _collect_code_execution_tool_ids(messages)
        assert result == set()


class TestIsCodeExecutionRelatedBlock:
    """Tests for _is_code_execution_related_block function."""

    def test_regular_tool_use_block(self) -> None:
        """Test regular tool_use block without caller."""
        block = {
            "type": "tool_use",
            "id": "toolu_regular",
            "name": "get_weather",
            "input": {"location": "NYC"},
        }
        assert not _is_code_execution_related_block(block, set())

    def test_code_execution_tool_use_block(self) -> None:
        """Test tool_use block called by code_execution."""
        block = {
            "type": "tool_use",
            "id": "toolu_code_exec",
            "name": "get_weather",
            "input": {"location": "NYC"},
            "caller": {
                "type": "code_execution_20250825",
                "tool_id": "srvtoolu_abc",
            },
        }
        assert _is_code_execution_related_block(block, set())

    def test_regular_tool_result_block(self) -> None:
        """Test tool_result block for regular tool."""
        block = {
            "type": "tool_result",
            "tool_use_id": "toolu_regular",
            "content": "Sunny, 72°F",
        }
        code_exec_ids = {"toolu_code_exec"}
        assert not _is_code_execution_related_block(block, code_exec_ids)

    def test_code_execution_tool_result_block(self) -> None:
        """Test tool_result block for code_execution called tool."""
        block = {
            "type": "tool_result",
            "tool_use_id": "toolu_code_exec",
            "content": "Sunny, 72°F",
        }
        code_exec_ids = {"toolu_code_exec"}
        assert _is_code_execution_related_block(block, code_exec_ids)

    def test_text_block(self) -> None:
        """Test that text blocks are not flagged."""
        block = {"type": "text", "text": "Hello world"}
        assert not _is_code_execution_related_block(block, set())

    def test_non_dict_block(self) -> None:
        """Test that non-dict values return False."""
        assert not _is_code_execution_related_block("string", set())  # type: ignore[arg-type]
        assert not _is_code_execution_related_block(None, set())  # type: ignore[arg-type]
        assert not _is_code_execution_related_block(123, set())  # type: ignore[arg-type]
