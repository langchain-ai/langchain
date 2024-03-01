"""**Chat message history** stores a history of the message interactions in a chat.


**Class hierarchy:**

.. code-block::

    BaseChatMessageHistory --> <name>ChatMessageHistory  # Examples: FileChatMessageHistory, PostgresChatMessageHistory

**Main helpers:**

.. code-block::

    AIMessage, HumanMessage, BaseMessage

"""  # noqa: E501

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
from langchain_community.chat_message_histories.file import FileChatMessageHistory
from langchain_community.chat_message_histories.firestore import (
    FirestoreChatMessageHistory,
)
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.chat_message_histories.momento import MomentoChatMessageHistory
from langchain_community.chat_message_histories.mongodb import MongoDBChatMessageHistory
from langchain_community.chat_message_histories.neo4j import Neo4jChatMessageHistory
from langchain_community.chat_message_histories.postgres import (
    PostgresChatMessageHistory,
)
from langchain_community.chat_message_histories.redis import RedisChatMessageHistory
from langchain_community.chat_message_histories.rocksetdb import (
    RocksetChatMessageHistory,
)
from langchain_community.chat_message_histories.singlestoredb import (
    SingleStoreDBChatMessageHistory,
)
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)
from langchain_community.chat_message_histories.tidb import TiDBChatMessageHistory
from langchain_community.chat_message_histories.upstash_redis import (
    UpstashRedisChatMessageHistory,
)
from langchain_community.chat_message_histories.xata import XataChatMessageHistory
from langchain_community.chat_message_histories.zep import ZepChatMessageHistory

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
    "TiDBChatMessageHistory",
]
