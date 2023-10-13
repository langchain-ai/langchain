from pathlib import Path
from typing import Any, Generator, Tuple

import pytest
from sqlalchemy import Column, Integer, Text
from sqlalchemy.orm import DeclarativeBase

from langchain.memory.chat_message_histories import SQLChatMessageHistory
from langchain.memory.chat_message_histories.sql import DefaultMessageConverter
from langchain.schema.messages import AIMessage, HumanMessage


@pytest.fixture()
def con_str(tmp_path: Path) -> str:
    file_path = tmp_path / "db.sqlite3"
    con_str = f"sqlite:///{file_path}"
    return con_str


@pytest.fixture()
def sql_histories(
    con_str: str,
) -> Generator[Tuple[SQLChatMessageHistory, SQLChatMessageHistory], None, None]:
    message_history = SQLChatMessageHistory(
        session_id="123", connection_string=con_str, table_name="test_table"
    )
    # Create history for other session
    other_history = SQLChatMessageHistory(
        session_id="456", connection_string=con_str, table_name="test_table"
    )

    yield message_history, other_history
    message_history.clear()
    other_history.clear()


def test_add_messages(
    sql_histories: Tuple[SQLChatMessageHistory, SQLChatMessageHistory]
) -> None:
    sql_history, other_history = sql_histories
    sql_history.add_user_message("Hello!")
    sql_history.add_ai_message("Hi there!")

    messages = sql_history.messages
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Hello!"
    assert messages[1].content == "Hi there!"


def test_multiple_sessions(
    sql_histories: Tuple[SQLChatMessageHistory, SQLChatMessageHistory]
) -> None:
    sql_history, other_history = sql_histories
    sql_history.add_user_message("Hello!")
    sql_history.add_ai_message("Hi there!")
    sql_history.add_user_message("Whats cracking?")

    # Ensure the messages are added correctly in the first session
    assert len(sql_history.messages) == 3, "waat"
    assert sql_history.messages[0].content == "Hello!"
    assert sql_history.messages[1].content == "Hi there!"
    assert sql_history.messages[2].content == "Whats cracking?"

    # second session
    other_history.add_user_message("Hellox")
    assert len(other_history.messages) == 1
    assert len(sql_history.messages) == 3
    assert other_history.messages[0].content == "Hellox"
    assert sql_history.messages[0].content == "Hello!"
    assert sql_history.messages[1].content == "Hi there!"
    assert sql_history.messages[2].content == "Whats cracking?"


def test_clear_messages(
    sql_histories: Tuple[SQLChatMessageHistory, SQLChatMessageHistory]
) -> None:
    sql_history, other_history = sql_histories
    sql_history.add_user_message("Hello!")
    sql_history.add_ai_message("Hi there!")
    assert len(sql_history.messages) == 2
    # Now create another history with different session id
    other_history.add_user_message("Hellox")
    assert len(other_history.messages) == 1
    assert len(sql_history.messages) == 2
    # Now clear the first history
    sql_history.clear()
    assert len(sql_history.messages) == 0
    assert len(other_history.messages) == 1


def test_model_no_session_id_field_error(con_str: str) -> None:
    class Base(DeclarativeBase):
        pass

    class Model(Base):
        __tablename__ = "test_table"
        id = Column(Integer, primary_key=True)
        test_field = Column(Text)

    class CustomMessageConverter(DefaultMessageConverter):
        def get_sql_model_class(self) -> Any:
            return Model

    with pytest.raises(ValueError):
        SQLChatMessageHistory(
            "test",
            con_str,
            custom_message_converter=CustomMessageConverter("test_table"),
        )
