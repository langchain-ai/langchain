from typing import List
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.callbacks import CallbackManager
from pydantic import SecretStr

from langchain_community.llms.writer import Writer
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler

"""Classes for mocking Writer responses."""


class Choice:
    def __init__(self, text: str):
        self.text = text


class Completion:
    def __init__(self, choices: List[Choice]):
        self.choices = choices


class StreamingData:
    def __init__(self, value: str):
        self.value = value


@pytest.mark.requires("writerai")
class TestWriterLLM:
    """Unit tests for Writer LLM integration."""

    @pytest.fixture(autouse=True)
    def mock_unstreaming_completion(self) -> Completion:
        """Fixture providing a mock API response."""
        return Completion(choices=[Choice(text="Hello! How can I help you?")])

    @pytest.fixture(autouse=True)
    def mock_streaming_completion(self) -> List[StreamingData]:
        """Fixture providing mock streaming response chunks."""
        return [
            StreamingData(value="Hello! "),
            StreamingData(value="How can I"),
            StreamingData(value=" help you?"),
        ]

    def test_sync_unstream_completion(
        self, mock_unstreaming_completion: Completion
    ) -> None:
        """Test basic llm call with mocked response."""
        mock_client = MagicMock()
        mock_client.completions.create.return_value = mock_unstreaming_completion

        llm = Writer(api_key=SecretStr("key"))

        with mock.patch.object(llm, "client", mock_client):
            response_text = llm.invoke(input="Hello")

            assert response_text == "Hello! How can I help you?"

    def test_sync_unstream_completion_with_params(
        self, mock_unstreaming_completion: Completion
    ) -> None:
        """Test llm call with passed params with mocked response."""
        mock_client = MagicMock()
        mock_client.completions.create.return_value = mock_unstreaming_completion

        llm = Writer(api_key=SecretStr("key"), temperature=1)

        with mock.patch.object(llm, "client", mock_client):
            response_text = llm.invoke(input="Hello")

            assert response_text == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_async_unstream_completion(
        self, mock_unstreaming_completion: Completion
    ) -> None:
        """Test async chat completion with mocked response."""
        mock_async_client = AsyncMock()
        mock_async_client.completions.create.return_value = mock_unstreaming_completion

        llm = Writer(api_key=SecretStr("key"))

        with mock.patch.object(llm, "async_client", mock_async_client):
            response_text = await llm.ainvoke(input="Hello")

            assert response_text == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_async_unstream_completion_with_params(
        self, mock_unstreaming_completion: Completion
    ) -> None:
        """Test async llm call with passed params with mocked response."""
        mock_async_client = AsyncMock()
        mock_async_client.completions.create.return_value = mock_unstreaming_completion

        llm = Writer(api_key=SecretStr("key"), temperature=1)

        with mock.patch.object(llm, "async_client", mock_async_client):
            response_text = await llm.ainvoke(input="Hello")

            assert response_text == "Hello! How can I help you?"

    def test_sync_streaming_completion(
        self, mock_streaming_completion: List[StreamingData]
    ) -> None:
        """Test sync streaming."""

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.__iter__.return_value = mock_streaming_completion
        mock_client.completions.create.return_value = mock_response

        llm = Writer(api_key=SecretStr("key"))

        with mock.patch.object(llm, "client", mock_client):
            response = llm.stream(input="Hello")

            response_message = ""
            for chunk in response:
                response_message += chunk

        assert response_message == "Hello! How can I help you?"

    def test_sync_streaming_completion_with_callback_handler(
        self, mock_streaming_completion: List[StreamingData]
    ) -> None:
        """Test sync streaming with callback handler."""
        callback_handler = FakeCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.__iter__.return_value = mock_streaming_completion
        mock_client.completions.create.return_value = mock_response

        llm = Writer(
            api_key=SecretStr("key"),
            callback_manager=callback_manager,
        )

        with mock.patch.object(llm, "client", mock_client):
            response = llm.stream(input="Hello")

            response_message = ""
            for chunk in response:
                response_message += chunk

            assert callback_handler.llm_streams == 3
            assert response_message == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_async_streaming_completion(
        self, mock_streaming_completion: Completion
    ) -> None:
        """Test async streaming with callback handler."""

        mock_async_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = mock_streaming_completion
        mock_async_client.completions.create.return_value = mock_response

        llm = Writer(api_key=SecretStr("key"))

        with mock.patch.object(llm, "async_client", mock_async_client):
            response = llm.astream(input="Hello")

            response_message = ""
            async for chunk in response:
                response_message += str(chunk)

            assert response_message == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_async_streaming_completion_with_callback_handler(
        self, mock_streaming_completion: Completion
    ) -> None:
        """Test async streaming with callback handler."""
        callback_handler = FakeCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        mock_async_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = mock_streaming_completion
        mock_async_client.completions.create.return_value = mock_response

        llm = Writer(
            api_key=SecretStr("key"),
            callback_manager=callback_manager,
        )

        with mock.patch.object(llm, "async_client", mock_async_client):
            response = llm.astream(input="Hello")

            response_message = ""
            async for chunk in response:
                response_message += str(chunk)

            assert callback_handler.llm_streams == 3
            assert response_message == "Hello! How can I help you?"
