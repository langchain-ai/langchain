"""Test Friendli LLM for chat."""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from pydantic import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_community.adapters.openai import aenumerate
from langchain_community.chat_models import ChatFriendli


@pytest.fixture
def mock_friendli_client() -> Mock:
    """Mock instance of Friendli client."""
    return Mock()


@pytest.fixture
def mock_friendli_async_client() -> AsyncMock:
    """Mock instance of Friendli async client."""
    return AsyncMock()


@pytest.fixture
def chat_friendli(
    mock_friendli_client: Mock, mock_friendli_async_client: AsyncMock
) -> ChatFriendli:
    """Friendli LLM for chat with mock clients."""
    return ChatFriendli(
        friendli_token=SecretStr("personal-access-token"),
        client=mock_friendli_client,
        async_client=mock_friendli_async_client,
    )


@pytest.mark.requires("friendli")
def test_friendli_token_is_secret_string(capsys: CaptureFixture) -> None:
    """Test if friendli token is stored as a SecretStr."""
    fake_token_value = "personal-access-token"
    chat = ChatFriendli(friendli_token=fake_token_value)  # type: ignore[arg-type]
    assert isinstance(chat.friendli_token, SecretStr)
    assert chat.friendli_token.get_secret_value() == fake_token_value
    print(chat.friendli_token, end="")  # noqa: T201
    captured = capsys.readouterr()
    assert captured.out == "**********"


@pytest.mark.requires("friendli")
def test_friendli_token_read_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test if friendli token can be parsed from environment."""
    fake_token_value = "personal-access-token"
    monkeypatch.setenv("FRIENDLI_TOKEN", fake_token_value)
    chat = ChatFriendli()
    assert isinstance(chat.friendli_token, SecretStr)
    assert chat.friendli_token.get_secret_value() == fake_token_value
    print(chat.friendli_token, end="")  # noqa: T201
    captured = capsys.readouterr()
    assert captured.out == "**********"


@pytest.mark.requires("friendli")
def test_friendli_invoke(
    mock_friendli_client: Mock, chat_friendli: ChatFriendli
) -> None:
    """Test invocation with friendli."""
    mock_message = Mock()
    mock_message.content = "Hello Friendli"
    mock_message.role = "assistant"
    mock_choice = Mock()
    mock_choice.message = mock_message
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    mock_friendli_client.chat.completions.create.return_value = mock_response

    result = chat_friendli.invoke("Hello langchain")
    assert result.content == "Hello Friendli"
    mock_friendli_client.chat.completions.create.assert_called_once_with(
        messages=[{"role": "user", "content": "Hello langchain"}],
        stream=False,
        model=chat_friendli.model,
        frequency_penalty=None,
        presence_penalty=None,
        max_tokens=None,
        stop=None,
        temperature=None,
        top_p=None,
    )


@pytest.mark.requires("friendli")
async def test_friendli_ainvoke(
    mock_friendli_async_client: AsyncMock, chat_friendli: ChatFriendli
) -> None:
    """Test async invocation with friendli."""
    mock_message = Mock()
    mock_message.content = "Hello Friendli"
    mock_message.role = "assistant"
    mock_choice = Mock()
    mock_choice.message = mock_message
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    mock_friendli_async_client.chat.completions.create.return_value = mock_response

    result = await chat_friendli.ainvoke("Hello langchain")
    assert result.content == "Hello Friendli"
    mock_friendli_async_client.chat.completions.create.assert_awaited_once_with(
        messages=[{"role": "user", "content": "Hello langchain"}],
        stream=False,
        model=chat_friendli.model,
        frequency_penalty=None,
        presence_penalty=None,
        max_tokens=None,
        stop=None,
        temperature=None,
        top_p=None,
    )


@pytest.mark.requires("friendli")
def test_friendli_stream(
    mock_friendli_client: Mock, chat_friendli: ChatFriendli
) -> None:
    """Test stream with friendli."""
    mock_delta_0 = Mock()
    mock_delta_0.content = "Hello "
    mock_delta_1 = Mock()
    mock_delta_1.content = "Friendli"
    mock_choice_0 = Mock()
    mock_choice_0.delta = mock_delta_0
    mock_choice_1 = Mock()
    mock_choice_1.delta = mock_delta_1
    mock_chunk_0 = Mock()
    mock_chunk_0.choices = [mock_choice_0]
    mock_chunk_1 = Mock()
    mock_chunk_1.choices = [mock_choice_1]
    mock_stream = MagicMock()
    mock_chunks = [mock_chunk_0, mock_chunk_1]
    mock_stream.__iter__.return_value = mock_chunks

    mock_friendli_client.chat.completions.create.return_value = mock_stream
    stream = chat_friendli.stream("Hello langchain")
    for i, chunk in enumerate(stream):
        assert chunk.content == mock_chunks[i].choices[0].delta.content

    mock_friendli_client.chat.completions.create.assert_called_once_with(
        messages=[{"role": "user", "content": "Hello langchain"}],
        stream=True,
        model=chat_friendli.model,
        frequency_penalty=None,
        presence_penalty=None,
        max_tokens=None,
        stop=None,
        temperature=None,
        top_p=None,
    )


@pytest.mark.requires("friendli")
async def test_friendli_astream(
    mock_friendli_async_client: AsyncMock, chat_friendli: ChatFriendli
) -> None:
    """Test async stream with friendli."""
    mock_delta_0 = Mock()
    mock_delta_0.content = "Hello "
    mock_delta_1 = Mock()
    mock_delta_1.content = "Friendli"
    mock_choice_0 = Mock()
    mock_choice_0.delta = mock_delta_0
    mock_choice_1 = Mock()
    mock_choice_1.delta = mock_delta_1
    mock_chunk_0 = Mock()
    mock_chunk_0.choices = [mock_choice_0]
    mock_chunk_1 = Mock()
    mock_chunk_1.choices = [mock_choice_1]
    mock_stream = AsyncMock()
    mock_chunks = [mock_chunk_0, mock_chunk_1]
    mock_stream.__aiter__.return_value = mock_chunks

    mock_friendli_async_client.chat.completions.create.return_value = mock_stream
    stream = chat_friendli.astream("Hello langchain")
    async for i, chunk in aenumerate(stream):
        assert chunk.content == mock_chunks[i].choices[0].delta.content

    mock_friendli_async_client.chat.completions.create.assert_awaited_once_with(
        messages=[{"role": "user", "content": "Hello langchain"}],
        stream=True,
        model=chat_friendli.model,
        frequency_penalty=None,
        presence_penalty=None,
        max_tokens=None,
        stop=None,
        temperature=None,
        top_p=None,
    )
