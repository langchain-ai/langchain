import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_postgres.chat_message_histories import PostgresChatMessageHistory
from tests.utils import syncpg_client


def test_sync_chat_history() -> None:
    table_name = "chat_history"
    session_id = str(uuid.UUID(int=123))
    with syncpg_client() as sync_client:
        PostgresChatMessageHistory.drop_table(sync_client, table_name)
        PostgresChatMessageHistory.create_schema(sync_client, table_name)
        chat_history = PostgresChatMessageHistory(
            table_name, session_id, sync_connection=sync_client
        )
        assert chat_history is not None

        # Get messages from the chat history
        messages = chat_history.messages
        assert messages == []

        chat_history.add_messages(
            [
                SystemMessage(content="Meow"),
                AIMessage(content="woof"),
                HumanMessage(content="bark"),
            ]
        )

        # Get messages from the chat history
        messages = chat_history.messages
        assert len(messages) == 3
        assert messages == [
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
        ]
