from typing import Any

from langchain.memory.chat_message_histories.file import FileChatMessageHistory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

DEPRECATED_IMPORTS = [
    "AstraDBChatMessageHistory",
    "CassandraChatMessageHistory",
    "CosmosDBChatMessageHistory",
    "DynamoDBChatMessageHistory",
    "ElasticsearchChatMessageHistory",
    "FirestoreChatMessageHistory",
    "MomentoChatMessageHistory",
    "MongoDBChatMessageHistory",
    "PostgresChatMessageHistory",
    "RedisChatMessageHistory",
    "RocksetChatMessageHistory",
    "SQLChatMessageHistory",
    "StreamlitChatMessageHistory",
    "SingleStoreDBChatMessageHistory",
    "XataChatMessageHistory",
    "ZepChatMessageHistory",
    "UpstashRedisChatMessageHistory",
    "Neo4jChatMessageHistory",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.memory.chat_message_histories import {name}`"
        )

    raise AttributeError()


__all__ = [
    "ChatMessageHistory",
    "FileChatMessageHistory",
]
