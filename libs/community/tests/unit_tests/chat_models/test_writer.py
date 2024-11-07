"""Unit tests for Writer chat model integration."""

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import SecretStr

from langchain_community.chat_models.writer import ChatWriter, _convert_dict_to_message
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


class TestChatWriter:
    def test_writer_model_param(self) -> None:
        """Test different ways to initialize the chat model."""
        test_cases: List[dict] = [
            {"model_name": "palmyra-x-004", "writer_api_key": "test-key"},
            {"model": "palmyra-x-004", "writer_api_key": "test-key"},
            {"model_name": "palmyra-x-004", "writer_api_key": "test-key"},
            {
                "model": "palmyra-x-004",
                "writer_api_key": "test-key",
                "temperature": 0.5,
            },
        ]

        for case in test_cases:
            chat = ChatWriter(**case)
            assert chat.model_name == "palmyra-x-004"
            assert chat.writer_api_key
            assert chat.writer_api_key.get_secret_value() == "test-key"
            assert chat.temperature == (0.5 if "temperature" in case else 0.7)

    def test_convert_dict_to_message_human(self) -> None:
        """Test converting a human message dict to a LangChain message."""
        message = {"role": "user", "content": "Hello"}
        result = _convert_dict_to_message(message)
        assert isinstance(result, HumanMessage)
        assert result.content == "Hello"

    def test_convert_dict_to_message_ai(self) -> None:
        """Test converting an AI message dict to a LangChain message."""
        message = {"role": "assistant", "content": "Hello"}
        result = _convert_dict_to_message(message)
        assert isinstance(result, AIMessage)
        assert result.content == "Hello"

    def test_convert_dict_to_message_system(self) -> None:
        """Test converting a system message dict to a LangChain message."""
        message = {"role": "system", "content": "You are a helpful assistant"}
        result = _convert_dict_to_message(message)
        assert isinstance(result, SystemMessage)
        assert result.content == "You are a helpful assistant"

    def test_convert_dict_to_message_tool_call(self) -> None:
        """Test converting a tool call message dict to a LangChain message."""
        content = json.dumps({"result": 42})
        message = {
            "role": "tool",
            "name": "get_number",
            "content": content,
            "tool_call_id": "call_abc123",
        }
        result = _convert_dict_to_message(message)
        assert isinstance(result, ToolMessage)
        assert result.name == "get_number"
        assert result.content == content

    def test_convert_dict_to_message_with_tool_calls(self) -> None:
        """Test converting an AIMessage with tool calls."""
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "London"}',
                    },
                }
            ],
        }
        result = _convert_dict_to_message(message)
        assert isinstance(result, AIMessage)
        assert result.tool_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["args"]["location"] == "London"

    @pytest.fixture(autouse=True)
    def mock_completion(self) -> Dict[str, Any]:
        """Fixture providing a mock API response."""
        return {
            "id": "chat-12345",
            "object": "chat.completion",
            "created": 1699000000,
            "model": "palmyra-x-004",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

    @pytest.fixture(autouse=True)
    def mock_response(self) -> Dict[str, Any]:
        response = {
            "id": "chat-12345",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "GetWeather",
                                    "arguments": '{"location": "London"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        return response

    @pytest.fixture(autouse=True)
    def mock_streaming_chunks(self) -> List[Dict[str, Any]]:
        """Fixture providing mock streaming response chunks."""
        return [
            {
                "id": "chat-12345",
                "object": "chat.completion.chunk",
                "created": 1699000000,
                "model": "palmyra-x-004",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "Hello",
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chat-12345",
                "object": "chat.completion.chunk",
                "created": 1699000000,
                "model": "palmyra-x-004",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "!",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        ]

    def test_sync_completion(self, mock_completion: Dict[str, Any]) -> None:
        """Test basic chat completion with mocked response."""
        chat = ChatWriter(api_key=SecretStr("test-key"))
        mock_client = MagicMock()
        mock_client.chat.chat.return_value = mock_completion

        with patch.object(chat, "client", mock_client):
            message = HumanMessage(content="Hi there!")
            response = chat.invoke([message])
            assert isinstance(response, AIMessage)
            assert response.content == "Hello! How can I help you?"

    async def test_async_completion(self, mock_completion: Dict[str, Any]) -> None:
        """Test async chat completion with mocked response."""
        chat = ChatWriter(api_key=SecretStr("test-key"))
        mock_client = AsyncMock()
        mock_client.chat.chat.return_value = mock_completion

        with patch.object(chat, "async_client", mock_client):
            message = HumanMessage(content="Hi there!")
            response = await chat.ainvoke([message])
            assert isinstance(response, AIMessage)
            assert response.content == "Hello! How can I help you?"

    def test_sync_streaming(self, mock_streaming_chunks: List[Dict[str, Any]]) -> None:
        """Test sync streaming with callback handler."""
        callback_handler = FakeCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        chat = ChatWriter(
            streaming=True,
            callback_manager=callback_manager,
            max_tokens=10,
            api_key=SecretStr("test-key"),
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.__iter__.return_value = mock_streaming_chunks
        mock_client.chat.chat.return_value = mock_response

        with patch.object(chat, "client", mock_client):
            message = HumanMessage(content="Hi")
            response = chat.invoke([message])

            assert isinstance(response, AIMessage)
            assert callback_handler.llm_streams > 0
            assert response.content == "Hello!"

    async def test_async_streaming(
        self, mock_streaming_chunks: List[Dict[str, Any]]
    ) -> None:
        """Test async streaming with callback handler."""
        callback_handler = FakeCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        chat = ChatWriter(
            streaming=True,
            callback_manager=callback_manager,
            max_tokens=10,
            api_key=SecretStr("test-key"),
        )

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = mock_streaming_chunks
        mock_client.chat.chat.return_value = mock_response

        with patch.object(chat, "async_client", mock_client):
            message = HumanMessage(content="Hi")
            response = await chat.ainvoke([message])

            assert isinstance(response, AIMessage)
            assert callback_handler.llm_streams > 0
            assert response.content == "Hello!"

    def test_sync_tool_calling(self, mock_response: Dict[str, Any]) -> None:
        """Test synchronous tool calling functionality."""
        from pydantic import BaseModel, Field

        class GetWeather(BaseModel):
            """Get the weather in a location."""

            location: str = Field(..., description="The location to get weather for")

        mock_client = MagicMock()
        mock_client.chat.chat.return_value = mock_response

        chat = ChatWriter(api_key=SecretStr("test-key"), client=mock_client)

        chat_with_tools = chat.bind_tools(
            tools=[GetWeather],
            tool_choice="GetWeather",
        )

        response = chat_with_tools.invoke("What's the weather in London?")
        assert isinstance(response, AIMessage)
        assert response.tool_calls
        assert response.tool_calls[0]["name"] == "GetWeather"
        assert response.tool_calls[0]["args"]["location"] == "London"

    async def test_async_tool_calling(self, mock_response: Dict[str, Any]) -> None:
        """Test asynchronous tool calling functionality."""
        from pydantic import BaseModel, Field

        class GetWeather(BaseModel):
            """Get the weather in a location."""

            location: str = Field(..., description="The location to get weather for")

        mock_client = AsyncMock()
        mock_client.chat.chat.return_value = mock_response

        chat = ChatWriter(api_key=SecretStr("test-key"), async_client=mock_client)

        chat_with_tools = chat.bind_tools(
            tools=[GetWeather],
            tool_choice="GetWeather",
        )

        response = await chat_with_tools.ainvoke("What's the weather in London?")
        assert isinstance(response, AIMessage)
        assert response.tool_calls
        assert response.tool_calls[0]["name"] == "GetWeather"
        assert response.tool_calls[0]["args"]["location"] == "London"
