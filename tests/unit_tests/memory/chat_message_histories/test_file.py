import tempfile
from pathlib import Path
from typing import Generator

import pytest

from langchain.memory.chat_message_histories import FileChatMessageHistory
from langchain.schema import AIMessage, HumanMessage


@pytest.fixture
def file_chat_message_history() -> Generator[FileChatMessageHistory, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_chat_history.json"
        file_chat_message_history = FileChatMessageHistory(str(file_path))
        yield file_chat_message_history


def test_add_messages(file_chat_message_history: FileChatMessageHistory) -> None:
    file_chat_message_history.add_user_message("Hello!")
    file_chat_message_history.add_ai_message("Hi there!")

    messages = file_chat_message_history.messages
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Hello!"
    assert messages[1].content == "Hi there!"


def test_clear_messages(file_chat_message_history: FileChatMessageHistory) -> None:
    file_chat_message_history.add_user_message("Hello!")
    file_chat_message_history.add_ai_message("Hi there!")

    file_chat_message_history.clear()
    messages = file_chat_message_history.messages
    assert len(messages) == 0


def test_multiple_sessions(file_chat_message_history: FileChatMessageHistory) -> None:
    # First session
    file_chat_message_history.add_user_message("Hello, AI!")
    file_chat_message_history.add_ai_message("Hello, how can I help you?")
    file_chat_message_history.add_user_message("Tell me a joke.")
    file_chat_message_history.add_ai_message(
        "Why did the chicken cross the road? To get to the other side!"
    )

    # Ensure the messages are added correctly in the first session
    messages = file_chat_message_history.messages
    assert len(messages) == 4
    assert messages[0].content == "Hello, AI!"
    assert messages[1].content == "Hello, how can I help you?"
    assert messages[2].content == "Tell me a joke."
    expected_content = "Why did the chicken cross the road? To get to the other side!"
    assert messages[3].content == expected_content

    # Second session (reinitialize FileChatMessageHistory)
    file_path = file_chat_message_history.file_path
    second_session_chat_message_history = FileChatMessageHistory(
        file_path=str(file_path)
    )

    # Ensure the history is maintained in the second session
    messages = second_session_chat_message_history.messages
    assert len(messages) == 4
    assert messages[0].content == "Hello, AI!"
    assert messages[1].content == "Hello, how can I help you?"
    assert messages[2].content == "Tell me a joke."
    expected_content = "Why did the chicken cross the road? To get to the other side!"
    assert messages[3].content == expected_content
