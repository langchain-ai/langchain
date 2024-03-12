"""**Chat message history** stores a history of the message interactions in a chat.


**Class hierarchy:**

.. code-block::

    BaseChatMessageHistory --> <name>ChatMessageHistory  # Examples: FileChatMessageHistory, PostgresChatMessageHistory

**Main helpers:**

.. code-block::

    AIMessage, HumanMessage, BaseMessage

"""  # noqa: E501

import importlib
from typing import Any

_module_lookup = {
    "AstraDBChatMessageHistory": "langchain_community.chat_message_histories.astradb",
    "CassandraChatMessageHistory": "langchain_community.chat_message_histories.cassandra",  # noqa: E501
    "ChatMessageHistory": "langchain_community.chat_message_histories.in_memory",
    "CosmosDBChatMessageHistory": "langchain_community.chat_message_histories.cosmos_db",  # noqa: E501
    "DynamoDBChatMessageHistory": "langchain_community.chat_message_histories.dynamodb",
    "ElasticsearchChatMessageHistory": "langchain_community.chat_message_histories.elasticsearch",  # noqa: E501
    "FileChatMessageHistory": "langchain_community.chat_message_histories.file",
    "FirestoreChatMessageHistory": "langchain_community.chat_message_histories.firestore",  # noqa: E501
    "MomentoChatMessageHistory": "langchain_community.chat_message_histories.momento",
    "MongoDBChatMessageHistory": "langchain_community.chat_message_histories.mongodb",
    "Neo4jChatMessageHistory": "langchain_community.chat_message_histories.neo4j",
    "PostgresChatMessageHistory": "langchain_community.chat_message_histories.postgres",
    "RedisChatMessageHistory": "langchain_community.chat_message_histories.redis",
    "RocksetChatMessageHistory": "langchain_community.chat_message_histories.rocksetdb",
    "SQLChatMessageHistory": "langchain_community.chat_message_histories.sql",
    "SingleStoreDBChatMessageHistory": "langchain_community.chat_message_histories.singlestoredb",  # noqa: E501
    "StreamlitChatMessageHistory": "langchain_community.chat_message_histories.streamlit",  # noqa: E501
    "TiDBChatMessageHistory": "langchain_community.chat_message_histories.tidb",
    "UpstashRedisChatMessageHistory": "langchain_community.chat_message_histories.upstash_redis",  # noqa: E501
    "XataChatMessageHistory": "langchain_community.chat_message_histories.xata",
    "ZepChatMessageHistory": "langchain_community.chat_message_histories.zep",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
