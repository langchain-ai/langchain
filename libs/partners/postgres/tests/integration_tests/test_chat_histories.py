from langchain_postgres.chat_message_histories import PostgresChatMessageHistory
from tests.utils import asyncpg_client, syncpg_client


def test_something() -> None:
    with syncpg_client() as sync_client:
        chat_history = PostgresChatMessageHistory(sync_connection=sync_client)
        assert chat_history is not None


async def test_something_else() -> None:
    async with asyncpg_client() as async_client:
        chat_history = PostgresChatMessageHistory(async_connection=async_client)
        assert chat_history is not None
