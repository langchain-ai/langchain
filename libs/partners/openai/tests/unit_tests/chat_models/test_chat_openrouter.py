"""Test ChatOpenRouter chat model."""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import SecretStr

from langchain_openai.chat_models.openrouter import (
    ChatOpenRouter,
    _convert_message_to_dict,
    _convert_tool_to_dict,
)


class TestMessageConversion:
    """Test message to dict conversion."""

    def test_system_message_simple(self) -> None:
        """Test simple system message conversion."""
        msg = SystemMessage(content="You are a helpful assistant")
        result = _convert_message_to_dict(msg)
        assert result == {
            "role": "system",
            "content": "You are a helpful assistant",
        }

    def test_system_message_with_cache_control(self) -> None:
        """Test system message with cache_control preservation."""
        msg = SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": "You are a helpful assistant",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        )
        result = _convert_message_to_dict(msg)
        assert result["role"] == "system"
        assert isinstance(result["content"], list)
        assert result["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_human_message(self) -> None:
        """Test human message conversion."""
        msg = HumanMessage(content="Hello")
        result = _convert_message_to_dict(msg)
        assert result == {"role": "user", "content": "Hello"}

    def test_ai_message_with_tool_calls(self) -> None:
        """Test AI message with tool calls."""
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_123",
                    "name": "get_weather",
                    "args": {"location": "Paris"},
                }
            ],
        )
        result = _convert_message_to_dict(msg)
        assert result["role"] == "assistant"
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_123"
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_message(self) -> None:
        """Test tool message conversion."""
        msg = ToolMessage(content="Result", tool_call_id="call_123")
        result = _convert_message_to_dict(msg)
        assert result["role"] == "tool"
        assert result["content"] == "Result"
        assert result["tool_call_id"] == "call_123"


class TestToolConversion:
    """Test tool to dict conversion."""

    def test_dict_tool_passthrough(self) -> None:
        """Test that dict tools are passed through unchanged."""
        tool = {"type": "function", "function": {"name": "test"}}
        result = _convert_tool_to_dict(tool)
        assert result == tool

    def test_langchain_tool(self) -> None:
        """Test LangChain tool conversion."""
        # Mock LangChain tool
        tool = MagicMock()
        tool.name = "get_weather"
        tool.description = "Get weather for a location"
        tool.args_schema = None

        result = _convert_tool_to_dict(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["description"] == "Get weather for a location"

    def test_tool_with_cache_control(self) -> None:
        """Test tool with cache_control preservation."""
        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search the web"
        tool.args_schema = None
        tool.cache_control = {"type": "ephemeral"}

        result = _convert_tool_to_dict(tool)
        assert result["cache_control"] == {"type": "ephemeral"}


class TestChatOpenRouter:
    """Test ChatOpenRouter class."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        llm = ChatOpenRouter()
        assert llm.model == "google/gemini-flash-1.5"
        assert llm.max_tokens == 4096
        assert llm.temperature == 0.7
        assert llm.timeout == 120
        assert llm.max_retries == 3

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        llm = ChatOpenRouter(
            model="anthropic/claude-3.5-sonnet",
            temperature=0,
            max_tokens=1024,
            timeout=60,
            max_retries=2,
        )
        assert llm.model == "anthropic/claude-3.5-sonnet"
        assert llm.temperature == 0
        assert llm.max_tokens == 1024
        assert llm.timeout == 60
        assert llm.max_retries == 2

    def test_llm_type(self) -> None:
        """Test _llm_type property."""
        llm = ChatOpenRouter()
        assert llm._llm_type == "openrouter"

    def test_identifying_params(self) -> None:
        """Test _identifying_params property."""
        llm = ChatOpenRouter(model="test-model", temperature=0.5, max_tokens=2048)
        params = llm._identifying_params
        assert params["model"] == "test-model"
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 2048

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_get_api_key_from_env(self) -> None:
        """Test API key retrieval from environment."""
        llm = ChatOpenRouter()
        assert llm._get_api_key() == "test-key"

    def test_get_api_base_default(self) -> None:
        """Test default API base URL."""
        llm = ChatOpenRouter()
        assert llm._get_api_base() == "https://openrouter.ai/api/v1"

    @patch("httpx.Client")
    def test_generate_basic(self, mock_client: MagicMock) -> None:
        """Test basic generation."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Hello! How can I help you?",
                        "role": "assistant",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 6,
                "total_tokens": 16,
            },
        }
        mock_response.raise_for_status = MagicMock()

        # Mock client
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_context)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_context.post = MagicMock(return_value=mock_response)
        mock_client.return_value = mock_context

        # Test
        llm = ChatOpenRouter(openai_api_key=SecretStr("test-key"))
        messages: list[BaseMessage] = [HumanMessage(content="Hello")]
        result = llm._generate(messages)

        # Assertions
        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Hello! How can I help you?"
        assert result.llm_output is not None
        assert result.llm_output["token_usage"]["total_tokens"] == 16

    @patch("httpx.Client")
    def test_generate_with_cache_hit(self, mock_client: MagicMock) -> None:
        """Test generation with cache hit statistics."""
        # Mock response with cache statistics
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response", "role": "assistant"}}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "total_tokens": 1050,
                "prompt_tokens_details": {"cached_tokens": 800},
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_context)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_context.post = MagicMock(return_value=mock_response)
        mock_client.return_value = mock_context

        llm = ChatOpenRouter(openai_api_key=SecretStr("test-key"))
        messages: list[BaseMessage] = [SystemMessage(content="System prompt")]
        result = llm._generate(messages)

        # Check cache statistics
        gen_info = result.generations[0].generation_info
        assert gen_info is not None
        assert gen_info["cached_tokens"] == 800
        assert gen_info["cache_hit_rate"] == 80.0

    @patch("httpx.Client")
    def test_generate_with_tool_calls(self, mock_client: MagicMock) -> None:
        """Test generation with tool calls."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Paris"}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        }
        mock_response.raise_for_status = MagicMock()

        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_context)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_context.post = MagicMock(return_value=mock_response)
        mock_client.return_value = mock_context

        llm = ChatOpenRouter(openai_api_key=SecretStr("test-key"))
        messages: list[BaseMessage] = [
            HumanMessage(content="What's the weather in Paris?")
        ]
        result = llm._generate(messages)

        # Check tool calls
        ai_msg = result.generations[0].message
        assert isinstance(ai_msg, AIMessage)
        assert len(ai_msg.tool_calls) == 1
        assert ai_msg.tool_calls[0]["name"] == "get_weather"
        assert ai_msg.tool_calls[0]["args"] == {"location": "Paris"}

    @patch("httpx.Client")
    @patch("time.sleep")
    def test_generate_retry_on_rate_limit(
        self, mock_sleep: MagicMock, mock_client: MagicMock
    ) -> None:
        """Test retry logic on rate limit (429)."""
        # First call returns 429, second succeeds
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate limited", request=MagicMock(), response=mock_response_429
        )

        mock_response_success = MagicMock()
        mock_response_success.json.return_value = {
            "choices": [{"message": {"content": "Success", "role": "assistant"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response_success.raise_for_status = MagicMock()

        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_context)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_context.post = MagicMock(
            side_effect=[mock_response_429, mock_response_success]
        )
        mock_client.return_value = mock_context

        llm = ChatOpenRouter(openai_api_key=SecretStr("test-key"), max_retries=3)
        messages: list[BaseMessage] = [HumanMessage(content="Test")]
        result = llm._generate(messages)

        # Should have retried and succeeded
        assert result.generations[0].message.content == "Success"
        # Should have slept once (exponential backoff: 2^0 = 1)
        mock_sleep.assert_called_once_with(1)


class TestIntegration:
    """Integration tests (require OPENROUTER_API_KEY)."""

    @pytest.mark.skip(reason="Requires OPENROUTER_API_KEY and makes real API calls")
    def test_invoke_real_api(self) -> None:
        """Test real API call (skipped by default)."""
        llm = ChatOpenRouter(model="google/gemini-flash-1.5", temperature=0)
        messages: list[BaseMessage] = [HumanMessage(content="Say 'hello' in one word")]
        result = llm.invoke(messages)
        assert isinstance(result, AIMessage)
        assert len(result.content) > 0

    @pytest.mark.skip(reason="Requires OPENROUTER_API_KEY and makes real API calls")
    def test_stream_real_api(self) -> None:
        """Test real streaming API call (skipped by default)."""
        llm = ChatOpenRouter(model="google/gemini-flash-1.5", temperature=0)
        messages: list[BaseMessage] = [HumanMessage(content="Count to 3")]
        chunks = list(llm.stream(messages))
        assert len(chunks) > 0
        full_content = "".join(str(chunk.content) for chunk in chunks if chunk.content)
        assert len(full_content) > 0
