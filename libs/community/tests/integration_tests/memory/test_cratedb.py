import json
import os
from typing import Any, Generator, Tuple

import pytest
import sqlalchemy as sa
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import CrateDBChatMessageHistory
from langchain.memory.chat_message_histories.sql import DefaultMessageConverter
from langchain.schema.messages import AIMessage, HumanMessage, _message_to_dict
from sqlalchemy import Column, Integer, Text
from sqlalchemy.orm import DeclarativeBase


@pytest.fixture()
def connection_string() -> str:
    return os.environ.get(
        "TEST_CRATEDB_CONNECTION_STRING", "crate://crate@localhost/?schema=testdrive"
    )


@pytest.fixture()
def engine(connection_string: str) -> sa.Engine:
    """
    Return an SQLAlchemy engine object.
    """
    return sa.create_engine(connection_string, echo=True)


@pytest.fixture(autouse=True)
def reset_database(engine: sa.Engine) -> None:
    """
    Provision database with table schema and data.
    """
    with engine.connect() as connection:
        connection.execute(sa.text("DROP TABLE IF EXISTS test_table;"))
        connection.commit()


@pytest.fixture()
def sql_histories(
    connection_string: str,
) -> Generator[Tuple[CrateDBChatMessageHistory, CrateDBChatMessageHistory], None, None]:
    """
    Provide the test cases with data fixtures.
    """
    message_history = CrateDBChatMessageHistory(
        session_id="123", connection_string=connection_string, table_name="test_table"
    )
    # Create history for other session
    other_history = CrateDBChatMessageHistory(
        session_id="456", connection_string=connection_string, table_name="test_table"
    )

    yield message_history, other_history
    message_history.clear()
    other_history.clear()


def test_add_messages(
    sql_histories: Tuple[CrateDBChatMessageHistory, CrateDBChatMessageHistory],
) -> None:
    history1, _ = sql_histories
    history1.add_user_message("Hello!")
    history1.add_ai_message("Hi there!")

    messages = history1.messages
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Hello!"
    assert messages[1].content == "Hi there!"


def test_multiple_sessions(
    sql_histories: Tuple[CrateDBChatMessageHistory, CrateDBChatMessageHistory],
) -> None:
    history1, history2 = sql_histories

    # first session
    history1.add_user_message("Hello!")
    history1.add_ai_message("Hi there!")
    history1.add_user_message("Whats cracking?")

    # second session
    history2.add_user_message("Hellox")

    messages1 = history1.messages
    messages2 = history2.messages

    # Ensure the messages are added correctly in the first session
    assert len(messages1) == 3, "waat"
    assert messages1[0].content == "Hello!"
    assert messages1[1].content == "Hi there!"
    assert messages1[2].content == "Whats cracking?"

    assert len(messages2) == 1
    assert len(messages1) == 3
    assert messages2[0].content == "Hellox"
    assert messages1[0].content == "Hello!"
    assert messages1[1].content == "Hi there!"
    assert messages1[2].content == "Whats cracking?"


def test_clear_messages(
    sql_histories: Tuple[CrateDBChatMessageHistory, CrateDBChatMessageHistory],
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


def test_model_no_session_id_field_error(connection_string: str) -> None:
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
        CrateDBChatMessageHistory(
            "test",
            connection_string,
            custom_message_converter=CustomMessageConverter("test_table"),
        )


def test_memory_with_message_store(connection_string: str) -> None:
    """
    Test ConversationBufferMemory with a message store.
    """
    # Setup CrateDB as a message store.
    message_history = CrateDBChatMessageHistory(
        connection_string=connection_string, session_id="test-session"
    )
    memory = ConversationBufferMemory(
        memory_key="baz", chat_memory=message_history, return_messages=True
    )

    # Add a few messages.
    memory.chat_memory.add_ai_message("This is me, the AI")
    memory.chat_memory.add_user_message("This is me, the human")

    # Get the message history from the memory store and turn it into JSON.
    messages = memory.chat_memory.messages
    messages_json = json.dumps([_message_to_dict(msg) for msg in messages])

    # Verify the outcome.
    assert "This is me, the AI" in messages_json
    assert "This is me, the human" in messages_json

    # Clear the conversation history, and verify that.
    memory.chat_memory.clear()
    assert memory.chat_memory.messages == []
