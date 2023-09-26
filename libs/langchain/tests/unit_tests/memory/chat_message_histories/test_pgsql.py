from pathlib import Path
from typing import Tuple

import pytest

from langchain.memory.chat_message_histories import PostgresChatMessageHistory
from langchain.schema.messages import AIMessage, HumanMessage


# @pytest.fixture(params=[("postgresql")])
@pytest.fixture(params=[("postgresql")])
def pgsql_histories(request, tmp_path: Path):  # type: ignore
    if request.param == "SQLite":
        file_path = tmp_path / "db.sqlite3"
        con_str = f"sqlite:///{file_path}"
    elif request.param == "postgresql":
        con_str = "postgresql://admin:password@localhost/postgres"

    message_history = PostgresChatMessageHistory(
        session_id="123",
        connection_string=con_str,
        table_name="test_table",
        descending_time=False,
        limit=5,
    )
    # Create history for other session
    other_history = PostgresChatMessageHistory(
        session_id="456",
        connection_string=con_str,
        table_name="test_table",
        descending_time=False,
        limit=5,
    )

    yield (message_history, other_history)
    message_history.clear()
    other_history.clear()


def test_add_messages(
    pgsql_histories: Tuple[PostgresChatMessageHistory, PostgresChatMessageHistory]
) -> None:
    pgsql_history, other_history = pgsql_histories
    pgsql_history.add_user_message("Hello!")
    pgsql_history.add_ai_message("Hi there!")

    messages = pgsql_history.messages
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Hello!"
    assert messages[1].content == "Hi there!"


def test_messages_order_by_time(
    pgsql_histories: Tuple[PostgresChatMessageHistory, PostgresChatMessageHistory]
) -> None:
    pgsql_history, other_history = pgsql_histories
    pgsql_history.add_user_message("Hello with date and time!")
    pgsql_history.add_ai_message("Hi there with date and time!")

    messages = pgsql_history.messages_order_by_time
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Hello with date and time!"
    assert messages[1].content == "Hi there with date and time!"
