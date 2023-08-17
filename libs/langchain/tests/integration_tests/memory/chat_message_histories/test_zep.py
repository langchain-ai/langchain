from typing import TYPE_CHECKING

import pytest
from pytest_mock import MockerFixture

from langchain.memory.chat_message_histories import ZepChatMessageHistory
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from zep_python import ZepClient


@pytest.fixture
@pytest.mark.requires("zep_python")
def zep_chat(mocker: MockerFixture) -> ZepChatMessageHistory:
    mock_zep_client: ZepClient = mocker.patch("zep_python.ZepClient", autospec=True)
    mock_zep_client.memory = mocker.patch(
        "zep_python.memory.client.MemoryClient", autospec=True
    )
    zep_chat: ZepChatMessageHistory = ZepChatMessageHistory(
        "test_session", "http://localhost:8000"
    )
    zep_chat.zep_client = mock_zep_client
    return zep_chat


@pytest.mark.requires("zep_python")
def test_messages(mocker: MockerFixture, zep_chat: ZepChatMessageHistory) -> None:
    from zep_python import Memory, Message, Summary

    mock_memory: Memory = Memory(
        summary=Summary(
            content="summary",
        ),
        messages=[
            Message(content="message", role="ai", metadata={"key": "value"}),
            Message(content="message2", role="human", metadata={"key2": "value2"}),
        ],
    )
    zep_chat.zep_client.memory.get_memory.return_value = mock_memory  # type: ignore

    result = zep_chat.messages

    assert len(result) == 3
    assert isinstance(result[0], SystemMessage)  # summary
    assert isinstance(result[1], AIMessage)
    assert isinstance(result[2], HumanMessage)


@pytest.mark.requires("zep_python")
def test_add_user_message(
    mocker: MockerFixture, zep_chat: ZepChatMessageHistory
) -> None:
    zep_chat.add_user_message("test message")
    zep_chat.zep_client.memory.add_memory.assert_called_once()  # type: ignore


@pytest.mark.requires("zep_python")
def test_add_ai_message(mocker: MockerFixture, zep_chat: ZepChatMessageHistory) -> None:
    zep_chat.add_ai_message("test message")
    zep_chat.zep_client.memory.add_memory.assert_called_once()  # type: ignore


@pytest.mark.requires("zep_python")
def test_append(mocker: MockerFixture, zep_chat: ZepChatMessageHistory) -> None:
    zep_chat.add_message(AIMessage(content="test message"))
    zep_chat.zep_client.memory.add_memory.assert_called_once()  # type: ignore


@pytest.mark.requires("zep_python")
def test_search(mocker: MockerFixture, zep_chat: ZepChatMessageHistory) -> None:
    zep_chat.search("test query")
    zep_chat.zep_client.memory.search_memory.assert_called_once_with(  # type: ignore
        "test_session", mocker.ANY, limit=None
    )


@pytest.mark.requires("zep_python")
def test_clear(mocker: MockerFixture, zep_chat: ZepChatMessageHistory) -> None:
    zep_chat.clear()
    zep_chat.zep_client.memory.delete_memory.assert_called_once_with(  # type: ignore
        "test_session"
    )
