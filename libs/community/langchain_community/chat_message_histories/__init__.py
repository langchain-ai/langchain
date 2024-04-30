"""**Chat message history** stores a history of the message interactions in a chat.


**Class hierarchy:**

.. code-block::

    BaseChatMessageHistory --> <name>ChatMessageHistory  # Examples: FileChatMessageHistory, PostgresChatMessageHistory

**Main helpers:**

.. code-block::

    AIMessage, HumanMessage, BaseMessage

"""  # noqa: E501

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.chat_message_histories.astradb import (
        AstraDBChatMessageHistory,
    )
    from langchain_community.chat_message_histories.cassandra import (
        CassandraChatMessageHistory,
    )
    from langchain_community.chat_message_histories.cosmos_db import (
        CosmosDBChatMessageHistory,
    )
    from langchain_community.chat_message_histories.dynamodb import (
        DynamoDBChatMessageHistory,
    )
    from langchain_community.chat_message_histories.elasticsearch import (
        ElasticsearchChatMessageHistory,
    )
    from langchain_community.chat_message_histories.file import (
        FileChatMessageHistory,
    )
    from langchain_community.chat_message_histories.firestore import (
        FirestoreChatMessageHistory,
    )
    from langchain_community.chat_message_histories.in_memory import (
        ChatMessageHistory,
    )
    from langchain_community.chat_message_histories.momento import (
        MomentoChatMessageHistory,
    )
    from langchain_community.chat_message_histories.mongodb import (
        MongoDBChatMessageHistory,
    )
    from langchain_community.chat_message_histories.neo4j import (
        Neo4jChatMessageHistory,
    )
    from langchain_community.chat_message_histories.postgres import (
        PostgresChatMessageHistory,
    )
    from langchain_community.chat_message_histories.redis import (
        RedisChatMessageHistory,
    )
    from langchain_community.chat_message_histories.rocksetdb import (
        RocksetChatMessageHistory,
    )
    from langchain_community.chat_message_histories.singlestoredb import (
        SingleStoreDBChatMessageHistory,
    )
    from langchain_community.chat_message_histories.sql import (
        SQLChatMessageHistory,
    )
    from langchain_community.chat_message_histories.streamlit import (
        StreamlitChatMessageHistory,
    )
    from langchain_community.chat_message_histories.tidb import (
        TiDBChatMessageHistory,
    )
    from langchain_community.chat_message_histories.upstash_redis import (
        UpstashRedisChatMessageHistory,
    )
    from langchain_community.chat_message_histories.xata import (
        XataChatMessageHistory,
    )
    from langchain_community.chat_message_histories.zep import (
        ZepChatMessageHistory,
    )

__all__ = [
    "AstraDBChatMessageHistory",
    "CassandraChatMessageHistory",
    "ChatMessageHistory",
    "CosmosDBChatMessageHistory",
    "DynamoDBChatMessageHistory",
    "ElasticsearchChatMessageHistory",
    "FileChatMessageHistory",
    "FirestoreChatMessageHistory",
    "MomentoChatMessageHistory",
    "MongoDBChatMessageHistory",
    "Neo4jChatMessageHistory",
    "PostgresChatMessageHistory",
    "RedisChatMessageHistory",
    "RocksetChatMessageHistory",
    "SQLChatMessageHistory",
    "SingleStoreDBChatMessageHistory",
    "StreamlitChatMessageHistory",
    "TiDBChatMessageHistory",
    "UpstashRedisChatMessageHistory",
    "XataChatMessageHistory",
    "ZepChatMessageHistory",
]

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
