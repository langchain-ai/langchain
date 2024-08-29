"""Test Friendli LLM."""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_community.adapters.openai import aenumerate
from langchain_community.llms.friendli import Friendli


@pytest.fixture
def mock_friendli_client() -> Mock:
    """Mock instance of Friendli client."""
    return Mock()


@pytest.fixture
def mock_friendli_async_client() -> AsyncMock:
    """Mock instance of Friendli async client."""
    return AsyncMock()


@pytest.fixture
def friendli_llm(
    mock_friendli_client: Mock, mock_friendli_async_client: AsyncMock
) -> Friendli:
    """Friendli LLM with mock clients."""
    return Friendli(
        friendli_token=SecretStr("personal-access-token"),
        client=mock_friendli_client,
        async_client=mock_friendli_async_client,
    )


@pytest.mark.requires("friendli")
def test_friendli_token_is_secret_string(capsys: CaptureFixture) -> None:
    """Test if friendli token is stored as a SecretStr."""
    fake_token_value = "personal-access-token"
    chat = Friendli(friendli_token=fake_token_value)  # type: ignore[arg-type]
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
    chat = Friendli()
    assert isinstance(chat.friendli_token, SecretStr)
    assert chat.friendli_token.get_secret_value() == fake_token_value
    print(chat.friendli_token, end="")  # noqa: T201
    captured = capsys.readouterr()
    assert captured.out == "**********"


@pytest.mark.requires("friendli")
def test_friendli_invoke(mock_friendli_client: Mock, friendli_llm: Friendli) -> None:
    """Test invocation with friendli."""
    mock_choice = Mock()
    mock_choice.text = "Hello Friendli"
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    mock_friendli_client.completions.create.return_value = mock_response

    result = friendli_llm.invoke("Hello langchain")
    assert result == "Hello Friendli"
    mock_friendli_client.completions.create.assert_called_once_with(
        model=friendli_llm.model,
        prompt="Hello langchain",
        stream=False,
        frequency_penalty=None,
        presence_penalty=None,
        max_tokens=None,
        stop=None,
        temperature=None,
        top_p=None,
    )


@pytest.mark.requires("friendli")
async def test_friendli_ainvoke(
    mock_friendli_async_client: AsyncMock, friendli_llm: Friendli
) -> None:
    """Test async invocation with friendli."""
    mock_choice = Mock()
    mock_choice.text = "Hello Friendli"
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    mock_friendli_async_client.completions.create.return_value = mock_response

    result = await friendli_llm.ainvoke("Hello langchain")
    assert result == "Hello Friendli"
    mock_friendli_async_client.completions.create.assert_awaited_once_with(
        model=friendli_llm.model,
        prompt="Hello langchain",
        stream=False,
        frequency_penalty=None,
        presence_penalty=None,
        max_tokens=None,
        stop=None,
        temperature=None,
        top_p=None,
    )


@pytest.mark.requires("friendli")
def test_friendli_stream(mock_friendli_client: Mock, friendli_llm: Friendli) -> None:
    """Test stream with friendli."""
    mock_chunk_0 = Mock()
    mock_chunk_0.event = "token_sampled"
    mock_chunk_0.text = "Hello "
    mock_chunk_0.token = 0
    mock_chunk_1 = Mock()
    mock_chunk_1.event = "token_sampled"
    mock_chunk_1.text = "Friendli"
    mock_chunk_1.token = 1
    mock_stream = MagicMock()
    mock_chunks = [mock_chunk_0, mock_chunk_1]
    mock_stream.__iter__.return_value = mock_chunks

    mock_friendli_client.completions.create.return_value = mock_stream
    stream = friendli_llm.stream("Hello langchain")
    for i, chunk in enumerate(stream):
        assert chunk == mock_chunks[i].text

    mock_friendli_client.completions.create.assert_called_once_with(
        model=friendli_llm.model,
        prompt="Hello langchain",
        stream=True,
        frequency_penalty=None,
        presence_penalty=None,
        max_tokens=None,
        stop=None,
        temperature=None,
        top_p=None,
    )


@pytest.mark.requires("friendli")
async def test_friendli_astream(
    mock_friendli_async_client: AsyncMock, friendli_llm: Friendli
) -> None:
    """Test async stream with friendli."""
    mock_chunk_0 = Mock()
    mock_chunk_0.event = "token_sampled"
    mock_chunk_0.text = "Hello "
    mock_chunk_0.token = 0
    mock_chunk_1 = Mock()
    mock_chunk_1.event = "token_sampled"
    mock_chunk_1.text = "Friendli"
    mock_chunk_1.token = 1
    mock_stream = AsyncMock()
    mock_chunks = [mock_chunk_0, mock_chunk_1]
    mock_stream.__aiter__.return_value = mock_chunks

    mock_friendli_async_client.completions.create.return_value = mock_stream
    stream = friendli_llm.astream("Hello langchain")
    async for i, chunk in aenumerate(stream):
        assert chunk == mock_chunks[i].text

    mock_friendli_async_client.completions.create.assert_awaited_once_with(
        model=friendli_llm.model,
        prompt="Hello langchain",
        stream=True,
        frequency_penalty=None,
        presence_penalty=None,
        max_tokens=None,
        stop=None,
        temperature=None,
        top_p=None,
    )
