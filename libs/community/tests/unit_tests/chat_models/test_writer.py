import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Type
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_standard_tests.unit_tests import ChatModelUnitTests
from pydantic import SecretStr

from langchain_community.chat_models.writer import ChatWriter
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


@pytest.mark.requires("writerai")
class TestChatWriterCustom:
    """Test case for ChatWriter"""

    from writerai.types import Chat
    from writerai.types.chat import (
        Choice,
        ChoiceLogprobs,
        ChoiceLogprobsContent,
        ChoiceLogprobsContentTopLogprob,
        ChoiceLogprobsRefusal,
        ChoiceLogprobsRefusalTopLogprob,
        ChoiceMessage,
        ChoiceMessageToolCall,
        ChoiceMessageToolCallFunction,
        Usage,
    )
    from writerai.types.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
    from writerai.types.chat_completion_chunk import Choice as ChunkChoice

    @pytest.fixture(autouse=True)
    def mock_unstreaming_completion(self) -> Chat:
        """Fixture providing a mock API response."""
        return self.Chat(
            id="chat-12345",
            object="chat.completion",
            created=1699000000,
            model="palmyra-x-004",
            usage=self.Usage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
            choices=[
                self.Choice(
                    index=0,
                    finish_reason="stop",
                    logprobs=self.ChoiceLogprobs(
                        content=[
                            self.ChoiceLogprobsContent(
                                token="",
                                logprob=0,
                                top_logprobs=[
                                    self.ChoiceLogprobsContentTopLogprob(
                                        token="", logprob=0
                                    )
                                ],
                            )
                        ],
                        refusal=[
                            self.ChoiceLogprobsRefusal(
                                token="",
                                logprob=0,
                                top_logprobs=[
                                    self.ChoiceLogprobsRefusalTopLogprob(
                                        token="", logprob=0
                                    )
                                ],
                            )
                        ],
                    ),
                    message=self.ChoiceMessage(
                        role="assistant",
                        content="Hello! How can I help you?",
                        refusal="",
                    ),
                )
            ],
        )

    @pytest.fixture(autouse=True)
    def mock_tool_call_choice_response(self) -> Chat:
        return self.Chat(
            id="chat-12345",
            object="chat.completion",
            created=1699000000,
            model="palmyra-x-004",
            choices=[
                self.Choice(
                    index=0,
                    finish_reason="tool_calls",
                    logprobs=self.ChoiceLogprobs(
                        content=[
                            self.ChoiceLogprobsContent(
                                token="",
                                logprob=0,
                                top_logprobs=[
                                    self.ChoiceLogprobsContentTopLogprob(
                                        token="", logprob=0
                                    )
                                ],
                            )
                        ],
                        refusal=[
                            self.ChoiceLogprobsRefusal(
                                token="",
                                logprob=0,
                                top_logprobs=[
                                    self.ChoiceLogprobsRefusalTopLogprob(
                                        token="", logprob=0
                                    )
                                ],
                            )
                        ],
                    ),
                    message=self.ChoiceMessage(
                        role="assistant",
                        content="",
                        refusal="",
                        tool_calls=[
                            self.ChoiceMessageToolCall(
                                id="call_abc123",
                                type="function",
                                function=self.ChoiceMessageToolCallFunction(
                                    name="GetWeather",
                                    arguments='{"location": "London"}',
                                ),
                            )
                        ],
                    ),
                )
            ],
        )

    @pytest.fixture(autouse=True)
    def mock_streaming_chunks(self) -> List[ChatCompletionChunk]:
        """Fixture providing mock streaming response chunks."""
        return [
            self.ChatCompletionChunk(
                id="chat-12345",
                object="chat.completion",
                created=1699000000,
                model="palmyra-x-004",
                choices=[
                    self.ChunkChoice(
                        index=0,
                        finish_reason="stop",
                        delta=self.ChoiceDelta(content="Hello! "),
                    )
                ],
            ),
            self.ChatCompletionChunk(
                id="chat-12345",
                object="chat.completion",
                created=1699000000,
                model="palmyra-x-004",
                choices=[
                    self.ChunkChoice(
                        index=0,
                        finish_reason="stop",
                        delta=self.ChoiceDelta(content="How can I help you?"),
                    )
                ],
            ),
        ]

    def test_writer_model_param(self) -> None:
        """Test different ways to initialize the chat model."""
        test_cases: List[dict] = [
            {
                "model_name": "palmyra-x-004",
                "api_key": "key",
            },
            {
                "model": "palmyra-x-004",
                "api_key": "key",
            },
            {
                "model_name": "palmyra-x-004",
                "api_key": "key",
            },
            {
                "model": "palmyra-x-004",
                "temperature": 0.5,
                "api_key": "key",
            },
        ]

        for case in test_cases:
            chat = ChatWriter(**case)
            assert chat.model_name == "palmyra-x-004"
            assert chat.temperature == (0.5 if "temperature" in case else 0.7)

    def test_convert_writer_to_langchain_human(self) -> None:
        """Test converting a human message dict to a LangChain message."""
        message = {"role": "user", "content": "Hello"}
        result = ChatWriter._convert_writer_to_langchain(message)
        assert isinstance(result, HumanMessage)
        assert result.content == "Hello"

    def test_convert_writer_to_langchain_ai(self) -> None:
        """Test converting an AI message dict to a LangChain message."""
        message = {"role": "assistant", "content": "Hello"}
        result = ChatWriter._convert_writer_to_langchain(message)
        assert isinstance(result, AIMessage)
        assert result.content == "Hello"

    def test_convert_writer_to_langchain_system(self) -> None:
        """Test converting a system message dict to a LangChain message."""
        message = {"role": "system", "content": "You are a helpful assistant"}
        result = ChatWriter._convert_writer_to_langchain(message)
        assert isinstance(result, SystemMessage)
        assert result.content == "You are a helpful assistant"

    def test_convert_writer_to_langchain_tool_call(self) -> None:
        """Test converting a tool call message dict to a LangChain message."""
        content = json.dumps({"result": 42})
        message = {
            "role": "tool",
            "name": "get_number",
            "content": content,
            "tool_call_id": "call_abc123",
        }
        result = ChatWriter._convert_writer_to_langchain(message)
        assert isinstance(result, ToolMessage)
        assert result.name == "get_number"
        assert result.content == content

    def test_convert_writer_to_langchain_with_tool_calls(self) -> None:
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
        result = ChatWriter._convert_writer_to_langchain(message)
        assert isinstance(result, AIMessage)
        assert result.tool_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["args"]["location"] == "London"

    def test_sync_completion(
        self, mock_unstreaming_completion: List[ChatCompletionChunk]
    ) -> None:
        """Test basic chat completion with mocked response."""
        chat = ChatWriter(api_key=SecretStr("key"))

        mock_client = MagicMock()
        mock_client.chat.chat.return_value = mock_unstreaming_completion

        with mock.patch.object(chat, "client", mock_client):
            message = HumanMessage(content="Hi there!")
            response = chat.invoke([message])
            assert isinstance(response, AIMessage)
            assert response.content == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_async_completion(
        self, mock_unstreaming_completion: List[ChatCompletionChunk]
    ) -> None:
        """Test async chat completion with mocked response."""
        chat = ChatWriter(api_key=SecretStr("key"))

        mock_async_client = AsyncMock()
        mock_async_client.chat.chat.return_value = mock_unstreaming_completion

        with mock.patch.object(chat, "async_client", mock_async_client):
            message = HumanMessage(content="Hi there!")
            response = await chat.ainvoke([message])
            assert isinstance(response, AIMessage)
            assert response.content == "Hello! How can I help you?"

    def test_sync_streaming(
        self, mock_streaming_chunks: List[ChatCompletionChunk]
    ) -> None:
        """Test sync streaming with callback handler."""
        callback_handler = FakeCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        chat = ChatWriter(
            api_key=SecretStr("key"),
            callback_manager=callback_manager,
            max_tokens=10,
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.__iter__.return_value = mock_streaming_chunks
        mock_client.chat.chat.return_value = mock_response

        with mock.patch.object(chat, "client", mock_client):
            message = HumanMessage(content="Hi")
            response = chat.stream([message])
            response_message = ""
            for chunk in response:
                response_message += str(chunk.content)
            assert callback_handler.llm_streams > 0
            assert response_message == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_async_streaming(
        self, mock_streaming_chunks: List[ChatCompletionChunk]
    ) -> None:
        """Test async streaming with callback handler."""
        callback_handler = FakeCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        chat = ChatWriter(
            api_key=SecretStr("key"),
            callback_manager=callback_manager,
            max_tokens=10,
        )

        mock_async_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = mock_streaming_chunks
        mock_async_client.chat.chat.return_value = mock_response

        with mock.patch.object(chat, "async_client", mock_async_client):
            message = HumanMessage(content="Hi")
            response = chat.astream([message])
            response_message = ""
            async for chunk in response:
                response_message += str(chunk.content)
            assert callback_handler.llm_streams > 0
            assert response_message == "Hello! How can I help you?"

    def test_sync_tool_calling(
        self, mock_tool_call_choice_response: Dict[str, Any]
    ) -> None:
        """Test synchronous tool calling functionality."""
        from pydantic import BaseModel, Field

        class GetWeather(BaseModel):
            """Get the weather in a location."""

            location: str = Field(..., description="The location to get weather for")

        chat = ChatWriter(api_key=SecretStr("key"))

        mock_client = MagicMock()
        mock_client.chat.chat.return_value = mock_tool_call_choice_response

        chat_with_tools = chat.bind_tools(
            tools=[GetWeather],
            tool_choice="GetWeather",
        )

        with mock.patch.object(chat, "client", mock_client):
            response = chat_with_tools.invoke("What's the weather in London?")
            assert isinstance(response, AIMessage)
            assert response.tool_calls
            assert response.tool_calls[0]["name"] == "GetWeather"
            assert response.tool_calls[0]["args"]["location"] == "London"

    @pytest.mark.asyncio
    async def test_async_tool_calling(
        self, mock_tool_call_choice_response: Dict[str, Any]
    ) -> None:
        """Test asynchronous tool calling functionality."""
        from pydantic import BaseModel, Field

        class GetWeather(BaseModel):
            """Get the weather in a location."""

            location: str = Field(..., description="The location to get weather for")

        mock_async_client = AsyncMock()
        mock_async_client.chat.chat.return_value = mock_tool_call_choice_response

        chat = ChatWriter(api_key=SecretStr("key"))

        chat_with_tools = chat.bind_tools(
            tools=[GetWeather],
            tool_choice="GetWeather",
        )

        with mock.patch.object(chat, "async_client", mock_async_client):
            response = await chat_with_tools.ainvoke("What's the weather in London?")
            assert isinstance(response, AIMessage)
            assert response.tool_calls
            assert response.tool_calls[0]["name"] == "GetWeather"
            assert response.tool_calls[0]["args"]["location"] == "London"


@pytest.mark.requires("writerai")
class TestChatWriterStandart(ChatModelUnitTests):
    """Test case for ChatWriter that inherits from standard LangChain tests."""

    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        """Return ChatWriter model class."""
        return ChatWriter

    @property
    def chat_model_params(self) -> Dict:
        """Return any additional parameters needed."""
        return {
            "api_key": "fake-api-key",
            "model_name": "palmyra-x-004",
        }

    @property
    def has_tool_calling(self) -> bool:
        """Writer supports tool/function calling."""
        return True

    @property
    def tool_choice_value(self) -> Optional[str]:
        """Value to use for tool choice in tests."""
        return "auto"

    @property
    def has_structured_output(self) -> bool:
        """Writer does not yet support structured output."""
        return False

    @property
    def supports_image_inputs(self) -> bool:
        """Writer does not support image inputs."""
        return False

    @property
    def supports_video_inputs(self) -> bool:
        """Writer does not support video inputs."""
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        """Writer returns token usage information."""
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        """Writer does not support anthropic inputs."""
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        """Writer does not support image tool message."""
        return False

    @property
    def supported_usage_metadata_details(
        self,
    ) -> Dict[
        Literal["invoke", "stream"],
        List[
            Literal[
                "audio_input",
                "audio_output",
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ]
        ],
    ]:
        """Return which types of usage metadata your model supports."""
        return {"invoke": ["cache_creation_input"], "stream": ["reasoning_output"]}

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        """Return env vars, init args, and expected instance attrs for initializing
        from env vars."""
        return {"WRITER_API_KEY": "key"}, {"api_key": "key"}, {"api_key": "key"}
