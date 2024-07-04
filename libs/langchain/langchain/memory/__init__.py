"""**Memory** maintains Chain state, incorporating context from past runs.

**Class hierarchy for Memory:**

.. code-block::

    BaseMemory --> BaseChatMemory --> <name>Memory  # Examples: ZepMemory, MotorheadMemory

**Main helpers:**

.. code-block::

    BaseChatMessageHistory

**Chat Message History** stores the chat message history in different stores.

**Class hierarchy for ChatMessageHistory:**

.. code-block::

    BaseChatMessageHistory --> <name>ChatMessageHistory  # Example: ZepChatMessageHistory

**Main helpers:**

.. code-block::

    AIMessage, BaseMessage, HumanMessage
"""  # noqa: E501

from typing import TYPE_CHECKING, Any

from langchain._api import create_importer
from langchain.memory.buffer import (
    ConversationBufferMemory,
    ConversationStringBufferMemory,
)
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain.memory.combined import CombinedMemory
from langchain.memory.entity import (
    ConversationEntityMemory,
    InMemoryEntityStore,
    RedisEntityStore,
    SQLiteEntityStore,
    UpstashRedisEntityStore,
)
from langchain.memory.readonly import ReadOnlySharedMemory
from langchain.memory.simple import SimpleMemory
from langchain.memory.summary import ConversationSummaryMemory
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain.memory.token_buffer import ConversationTokenBufferMemory
from langchain.memory.vectorstore import VectorStoreRetrieverMemory
from langchain.memory.vectorstore_token_buffer_memory import (
    ConversationVectorStoreTokenBufferMemory,  # avoid circular import
)

if TYPE_CHECKING:
    from langchain_community.chat_message_histories import (
        AstraDBChatMessageHistory,
        CassandraChatMessageHistory,
        ChatMessageHistory,
        CosmosDBChatMessageHistory,
        DynamoDBChatMessageHistory,
        ElasticsearchChatMessageHistory,
        FileChatMessageHistory,
        MomentoChatMessageHistory,
        MongoDBChatMessageHistory,
        PostgresChatMessageHistory,
        RedisChatMessageHistory,
        SingleStoreDBChatMessageHistory,
        SQLChatMessageHistory,
        StreamlitChatMessageHistory,
        UpstashRedisChatMessageHistory,
        XataChatMessageHistory,
        ZepChatMessageHistory,
    )
    from langchain_community.memory.kg import ConversationKGMemory
    from langchain_community.memory.motorhead_memory import MotorheadMemory
    from langchain_community.memory.zep_memory import ZepMemory


# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "MotorheadMemory": "langchain_community.memory.motorhead_memory",
    "ConversationKGMemory": "langchain_community.memory.kg",
    "ZepMemory": "langchain_community.memory.zep_memory",
    "AstraDBChatMessageHistory": "langchain_community.chat_message_histories",
    "CassandraChatMessageHistory": "langchain_community.chat_message_histories",
    "ChatMessageHistory": "langchain_community.chat_message_histories",
    "CosmosDBChatMessageHistory": "langchain_community.chat_message_histories",
    "DynamoDBChatMessageHistory": "langchain_community.chat_message_histories",
    "ElasticsearchChatMessageHistory": "langchain_community.chat_message_histories",
    "FileChatMessageHistory": "langchain_community.chat_message_histories",
    "MomentoChatMessageHistory": "langchain_community.chat_message_histories",
    "MongoDBChatMessageHistory": "langchain_community.chat_message_histories",
    "PostgresChatMessageHistory": "langchain_community.chat_message_histories",
    "RedisChatMessageHistory": "langchain_community.chat_message_histories",
    "SingleStoreDBChatMessageHistory": "langchain_community.chat_message_histories",
    "SQLChatMessageHistory": "langchain_community.chat_message_histories",
    "StreamlitChatMessageHistory": "langchain_community.chat_message_histories",
    "UpstashRedisChatMessageHistory": "langchain_community.chat_message_histories",
    "XataChatMessageHistory": "langchain_community.chat_message_histories",
    "ZepChatMessageHistory": "langchain_community.chat_message_histories",
}


_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "AstraDBChatMessageHistory",
    "CassandraChatMessageHistory",
    "ChatMessageHistory",
    "CombinedMemory",
    "ConversationBufferMemory",
    "ConversationBufferWindowMemory",
    "ConversationEntityMemory",
    "ConversationKGMemory",
    "ConversationStringBufferMemory",
    "ConversationSummaryBufferMemory",
    "ConversationSummaryMemory",
    "ConversationTokenBufferMemory",
    "ConversationVectorStoreTokenBufferMemory",
    "CosmosDBChatMessageHistory",
    "DynamoDBChatMessageHistory",
    "ElasticsearchChatMessageHistory",
    "FileChatMessageHistory",
    "InMemoryEntityStore",
    "MomentoChatMessageHistory",
    "MongoDBChatMessageHistory",
    "MotorheadMemory",
    "PostgresChatMessageHistory",
    "ReadOnlySharedMemory",
    "RedisChatMessageHistory",
    "RedisEntityStore",
    "SingleStoreDBChatMessageHistory",
    "SQLChatMessageHistory",
    "SQLiteEntityStore",
    "SimpleMemory",
    "StreamlitChatMessageHistory",
    "VectorStoreRetrieverMemory",
    "XataChatMessageHistory",
    "ZepChatMessageHistory",
    "ZepMemory",
    "UpstashRedisEntityStore",
    "UpstashRedisChatMessageHistory",
]
