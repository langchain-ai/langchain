import warnings
from typing import Any

from langchain_core._api import LangChainDeprecationWarning

from langchain.utils.interactive_env import is_interactive_env


def __getattr__(name: str) -> Any:
    from langchain_community import chat_message_histories

    # If not in interactive env, raise warning.
    if not is_interactive_env():
        warnings.warn(
            "Importing chat message histories from langchain is deprecated. Importing "
            "from langchain will no longer be supported as of langchain==0.2.0. "
            "Please import from langchain-community instead:\n\n"
            f"`from langchain_community.chat_message_histories import {name}`.\n\n"
            "To install langchain-community run `pip install -U langchain-community`.",
            category=LangChainDeprecationWarning,
        )

    return getattr(chat_message_histories, name)


__all__ = [
    "AstraDBChatMessageHistory",
    "ChatMessageHistory",
    "CassandraChatMessageHistory",
    "CosmosDBChatMessageHistory",
    "DynamoDBChatMessageHistory",
    "ElasticsearchChatMessageHistory",
    "FileChatMessageHistory",
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
