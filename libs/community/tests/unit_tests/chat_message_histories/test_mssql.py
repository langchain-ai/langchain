import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_community.chat_message_histories.mssql import MssqlChatMessageHistory
from tests.unit_tests.utils.mssql_util import asyncms_client, syncms_client


@pytest.mark.skip(reason="Requires a running MSSQL server, pyodbc, and mssql tools")
def test_sync_chat_history() -> None:
    table_name = "test_chat_history"
    session_id = str(uuid.UUID(int=123))
    with syncms_client() as sync_connection:
        MssqlChatMessageHistory.drop_table(sync_connection, table_name)
        MssqlChatMessageHistory.create_tables(sync_connection, table_name)

        chat_history = MssqlChatMessageHistory(
            table_name,
            session_id,
            sync_connection=sync_connection,
        )

        messages = chat_history.messages
        assert messages == []

        assert chat_history is not None

        chat_history.add_messages(
            [
                SystemMessage(content="Meow"),
                AIMessage(content="woof"),
                HumanMessage(content="bark"),
            ]
        )

        messages = chat_history.messages
        assert len(messages) == 3
        assert messages == [
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
        ]

        chat_history.add_messages(
            [
                SystemMessage(content="Meow"),
                AIMessage(content="woof"),
                HumanMessage(content="bark"),
            ]
        )

        messages = chat_history.messages
        assert len(messages) == 6
        assert messages == [
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
        ]

        chat_history.clear()
        assert chat_history.messages == []


@pytest.mark.skip(reason="Requires a running MSSQL server, pyodbc, and mssql tools")
async def test_async_chat_history() -> None:
    """Test the async chat history."""

    async with asyncms_client() as async_connection:
        table_name = "test_async_chat_history"
        session_id = str(uuid.UUID(int=125))

        await MssqlChatMessageHistory.adrop_table(async_connection, table_name)
        await MssqlChatMessageHistory.acreate_tables(async_connection, table_name)

        chat_history = MssqlChatMessageHistory(
            table_name,
            session_id,
            async_connection=async_connection,
        )

        messages = await chat_history.aget_messages()
        assert messages == []
        await chat_history.aadd_messages(
            [
                SystemMessage(content="Meow"),
                AIMessage(content="woof"),
                HumanMessage(content="bark"),
            ]
        )

        messages = await chat_history.aget_messages()
        assert len(messages) == 3
        assert messages == [
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
        ]

        await chat_history.aadd_messages(
            [
                SystemMessage(content="Meow"),
                AIMessage(content="woof"),
                HumanMessage(content="bark"),
            ]
        )

        messages = await chat_history.aget_messages()
        assert len(messages) == 6
        assert messages == [
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
        ]

        await chat_history.aclear()
        assert await chat_history.aget_messages() == []
        await async_connection.close()
