import tempfile
from pathlib import Path
from typing import Generator

import pytest

from langchain.memory.chat_message_histories import SQLiteChatMessageHistory
from langchain.schema import AIMessage, HumanMessage


@pytest.fixture
def sqlite_history() -> Generator[SQLiteChatMessageHistory, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "db.sqlite3"
        sqlite_message_history = SQLiteChatMessageHistory(
            session_id="123", db_path=str(file_path)
        )
        yield sqlite_message_history


def create_other_history(db_path: str, session_id: str) -> SQLiteChatMessageHistory:
    return SQLiteChatMessageHistory(session_id=session_id, db_path=db_path)


def test_add_messages(sqlite_history: SQLiteChatMessageHistory) -> None:
    sqlite_history.add_user_message("Hello!")
    sqlite_history.add_ai_message("Hi there!")

    messages = sqlite_history.messages
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Hello!"
    assert messages[1].content == "Hi there!"


def test_multiple_sessions(sqlite_history: SQLiteChatMessageHistory) -> None:
    sqlite_history.add_user_message("Hello!")
    sqlite_history.add_ai_message("Hi there!")
    sqlite_history.add_user_message("Whats cracking?")

    # Ensure the messages are added correctly in the first session
    assert len(sqlite_history.messages) == 3
    assert sqlite_history.messages[0].content == "Hello!"
    assert sqlite_history.messages[1].content == "Hi there!"
    assert sqlite_history.messages[2].content == "Whats cracking?"

    # second session
    other_history = create_other_history(
        db_path=sqlite_history.db_path, session_id="456"
    )
    other_history.add_user_message("Hellox")
    assert len(other_history.messages) == 1
    assert len(sqlite_history.messages) == 3
    assert other_history.messages[0].content == "Hellox"
    assert sqlite_history.messages[0].content == "Hello!"
    assert sqlite_history.messages[1].content == "Hi there!"
    assert sqlite_history.messages[2].content == "Whats cracking?"


def test_clear_messages(sqlite_history: SQLiteChatMessageHistory) -> None:
    sqlite_history.add_user_message("Hello!")
    sqlite_history.add_ai_message("Hi there!")
    assert len(sqlite_history.messages) == 2
    # Now create another history with different session id
    other_history = create_other_history(
        db_path=sqlite_history.db_path, session_id="456"
    )
    other_history.add_user_message("Hellox")
    assert len(other_history.messages) == 1
    assert len(sqlite_history.messages) == 2
    # Now clear the first history
    sqlite_history.clear()
    assert len(sqlite_history.messages) == 0
    assert len(other_history.messages) == 1
