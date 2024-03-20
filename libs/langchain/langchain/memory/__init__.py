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
from typing import Any

from langchain.memory.buffer import (
    ConversationBufferMemory,
    ConversationStringBufferMemory,
)
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import (
    ChatMessageHistory,
    FileChatMessageHistory,
)
from langchain.memory.combined import CombinedMemory
from langchain.memory.entity import (
    ConversationEntityMemory,
    InMemoryEntityStore,
    SQLiteEntityStore,
)
from langchain.memory.kg import ConversationKGMemory
from langchain.memory.motorhead_memory import MotorheadMemory
from langchain.memory.readonly import ReadOnlySharedMemory
from langchain.memory.simple import SimpleMemory
from langchain.memory.summary import ConversationSummaryMemory
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain.memory.token_buffer import ConversationTokenBufferMemory
from langchain.memory.vectorstore import VectorStoreRetrieverMemory
from langchain.memory.zep_memory import ZepMemory

DEPRECATED_IMPORTS = [
    "AstraDBChatMessageHistory",
    "CassandraChatMessageHistory",
    "CosmosDBChatMessageHistory",
    "MomentoChatMessageHistory",
    "MongoDBChatMessageHistory",
    "DynamoDBChatMessageHistory",
    "RedisChatMessageHistory",
    "PostgresChatMessageHistory",
    "ElasticsearchChatMessageHistory",
    "FileChatMessageHistory",
    "SingleStoreDBChatMessageHistory",
    "SQLChatMessageHistory",
    "XataChatMessageHistory",
    "ZepChatMessageHistory",
    "StreamlitChatMessageHistory",
    "UpstashRedisChatMessageHistory",
    "RedisEntityStore",
    "UpstashRedisEntityStore",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        if "RedisEntityStore" in name:
            new_import = "langchain_community.entity_stores"
        else:
            new_import = "langchain_community.chat_message_histories"
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from {new_import} import {name}`"
        )
    raise AttributeError()


__all__ = [
    "CombinedMemory",
    "ConversationBufferMemory",
    "ConversationBufferWindowMemory",
    "ConversationEntityMemory",
    "ConversationKGMemory",
    "ConversationStringBufferMemory",
    "ConversationSummaryBufferMemory",
    "ConversationSummaryMemory",
    "ConversationTokenBufferMemory",
    "InMemoryEntityStore",
    "MotorheadMemory",
    "ReadOnlySharedMemory",
    "SQLiteEntityStore",
    "SimpleMemory",
    "VectorStoreRetrieverMemory",
    "ZepMemory",
    "ChatMessageHistory",
    "FileChatMessageHistory",
]
