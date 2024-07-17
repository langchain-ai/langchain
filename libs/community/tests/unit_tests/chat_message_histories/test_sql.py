from pathlib import Path
from typing import Any, AsyncGenerator, Generator, List, Tuple

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import Column, Integer, Text

try:
    from sqlalchemy.orm import DeclarativeBase

    class Base(DeclarativeBase):
        pass
except ImportError:
    # for sqlalchemy < 2
    from sqlalchemy.ext.declarative import declarative_base

    Base = declarative_base()  # type:ignore

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.chat_message_histories.sql import DefaultMessageConverter


@pytest.fixture()
def con_str(tmp_path: Path) -> str:
    file_path = tmp_path / "db.sqlite3"
    con_str = f"sqlite:///{file_path}"
    return con_str


@pytest.fixture()
def acon_str(tmp_path: Path) -> str:
    file_path = tmp_path / "adb.sqlite3"
    con_str = f"sqlite+aiosqlite:///{file_path}"
    return con_str


@pytest.fixture()
def sql_histories(
    con_str: str,
) -> Generator[Tuple[SQLChatMessageHistory, SQLChatMessageHistory], None, None]:
    message_history = SQLChatMessageHistory(
        session_id="123", connection=con_str, table_name="test_table"
    )
    # Create history for other session
    other_history = SQLChatMessageHistory(
        session_id="456", connection=con_str, table_name="test_table"
    )

    yield message_history, other_history
    message_history.clear()
    other_history.clear()


@pytest.fixture()
async def asql_histories(
    acon_str: str,
) -> AsyncGenerator[Tuple[SQLChatMessageHistory, SQLChatMessageHistory], None]:
    message_history = SQLChatMessageHistory(
        session_id="123",
        connection=acon_str,
        table_name="test_table",
        async_mode=True,
        engine_args={"echo": False},
    )
    # Create history for other session
    other_history = SQLChatMessageHistory(
        session_id="456",
        connection=acon_str,
        table_name="test_table",
        async_mode=True,
        engine_args={"echo": False},
    )

    yield message_history, other_history
    await message_history.aclear()
    await other_history.aclear()


def test_add_messages(
    sql_histories: Tuple[SQLChatMessageHistory, SQLChatMessageHistory],
) -> None:
    sql_history, other_history = sql_histories
    sql_history.add_messages(
        [HumanMessage(content="Hello!"), AIMessage(content="Hi there!")]
    )

    messages = sql_history.messages
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Hello!"
    assert messages[1].content == "Hi there!"


@pytest.mark.requires("aiosqlite")
async def test_async_add_messages(
    asql_histories: Tuple[SQLChatMessageHistory, SQLChatMessageHistory],
) -> None:
    sql_history, other_history = asql_histories
    await sql_history.aadd_messages(
        [HumanMessage(content="Hello!"), AIMessage(content="Hi there!")]
    )

    messages = await sql_history.aget_messages()
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Hello!"
    assert messages[1].content == "Hi there!"


def test_multiple_sessions(
    sql_histories: Tuple[SQLChatMessageHistory, SQLChatMessageHistory],
) -> None:
    sql_history, other_history = sql_histories
    sql_history.add_messages(
        [
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="Whats cracking?"),
        ]
    )

    # Ensure the messages are added correctly in the first session
    messages = sql_history.messages
    assert len(messages) == 3, "waat"
    assert messages[0].content == "Hello!"
    assert messages[1].content == "Hi there!"
    assert messages[2].content == "Whats cracking?"

    # second session
    other_history.add_messages([HumanMessage(content="Hellox")])
    assert len(other_history.messages) == 1
    messages = sql_history.messages
    assert len(messages) == 3
    assert other_history.messages[0].content == "Hellox"
    assert messages[0].content == "Hello!"
    assert messages[1].content == "Hi there!"
    assert messages[2].content == "Whats cracking?"


@pytest.mark.requires("aiosqlite")
async def test_async_multiple_sessions(
    asql_histories: Tuple[SQLChatMessageHistory, SQLChatMessageHistory],
) -> None:
    sql_history, other_history = asql_histories
    await sql_history.aadd_messages(
        [
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="Whats cracking?"),
        ]
    )

    # Ensure the messages are added correctly in the first session
    messages: List[BaseMessage] = await sql_history.aget_messages()
    assert len(messages) == 3, "waat"
    assert messages[0].content == "Hello!"
    assert messages[1].content == "Hi there!"
    assert messages[2].content == "Whats cracking?"

    # second session
    await other_history.aadd_messages([HumanMessage(content="Hellox")])
    messages = await sql_history.aget_messages()
    assert len(await other_history.aget_messages()) == 1
    assert len(messages) == 3
    assert (await other_history.aget_messages())[0].content == "Hellox"
    assert messages[0].content == "Hello!"
    assert messages[1].content == "Hi there!"
    assert messages[2].content == "Whats cracking?"


def test_clear_messages(
    sql_histories: Tuple[SQLChatMessageHistory, SQLChatMessageHistory],
) -> None:
    sql_history, other_history = sql_histories
    sql_history.add_messages(
        [HumanMessage(content="Hello!"), AIMessage(content="Hi there!")]
    )
    assert len(sql_history.messages) == 2
    # Now create another history with different session id
    other_history.add_messages([HumanMessage(content="Hellox")])
    assert len(other_history.messages) == 1
    assert len(sql_history.messages) == 2
    # Now clear the first history
    sql_history.clear()
    assert len(sql_history.messages) == 0
    assert len(other_history.messages) == 1


@pytest.mark.requires("aiosqlite")
async def test_async_clear_messages(
    asql_histories: Tuple[SQLChatMessageHistory, SQLChatMessageHistory],
) -> None:
    sql_history, other_history = asql_histories
    await sql_history.aadd_messages(
        [HumanMessage(content="Hello!"), AIMessage(content="Hi there!")]
    )
    assert len(await sql_history.aget_messages()) == 2
    # Now create another history with different session id
    await other_history.aadd_messages([HumanMessage(content="Hellox")])
    assert len(await other_history.aget_messages()) == 1
    assert len(await sql_history.aget_messages()) == 2
    # Now clear the first history
    await sql_history.aclear()
    assert len(await sql_history.aget_messages()) == 0
    assert len(await other_history.aget_messages()) == 1


def test_model_no_session_id_field_error(con_str: str) -> None:
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
