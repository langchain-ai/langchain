import pytest
from zep_python import Memory, Message, Summary

from langchain.memory.chat_message_histories import ZepChatMessageHistory
from langchain.schema import AIMessage, HumanMessage


@pytest.fixture
def zep_chat(mocker):
    mock_zep_client = mocker.patch("zep_python.ZepClient", autospec=True)
    zep_chat = ZepChatMessageHistory("test_session", "http://localhost:8000")
    zep_chat.zep_client = mock_zep_client
    return zep_chat


def test_messages(mocker, zep_chat):
    mock_memory = Memory(
        summary=Summary(
            content="summary",
        ),
        messages=[Message(content="message", role="ai"),
                  Message(content="message2", role="human")],
    )
    zep_chat.zep_client.get_memory.return_value = mock_memory

    result = zep_chat.messages

    assert len(result) == 3
    assert isinstance(result[0], HumanMessage) # summary
    assert isinstance(result[1], AIMessage)
    assert isinstance(result[2], HumanMessage)


def test_add_user_message(mocker, zep_chat):
    zep_chat.add_user_message("test message")
    zep_chat.zep_client.add_memory.assert_called_once()


def test_add_ai_message(mocker, zep_chat):
    zep_chat.add_ai_message("test message")
    zep_chat.zep_client.add_memory.assert_called_once()


def test_append(mocker, zep_chat):
    zep_chat.append(AIMessage(content="test message"))
    zep_chat.zep_client.add_memory.assert_called_once()


def test_search(mocker, zep_chat):
    zep_chat.search("test query")
    zep_chat.zep_client.search_memory.assert_called_once_with(
        "test_session", mocker.ANY, limit=None
    )


def test_clear(mocker, zep_chat):
    zep_chat.clear()
    zep_chat.zep_client.delete_memory.assert_called_once_with("test_session")
