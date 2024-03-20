import pytest

from langchain import memory
from tests.unit_tests import assert_all_importable

EXPECTED_ALL = [
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
    "FileChatMessageHistory",
    "InMemoryEntityStore",
    "ReadOnlySharedMemory",
    "SQLiteEntityStore",
    "SimpleMemory",
    "VectorStoreRetrieverMemory",
    "ZepMemory",
    "MotorheadMemory",
]
EXPECTED_DEPRECATED_IMPORTS = [
    "AstraDBChatMessageHistory",
    "CassandraChatMessageHistory",
    "CosmosDBChatMessageHistory",
    "DynamoDBChatMessageHistory",
    "ElasticsearchChatMessageHistory",
    "MomentoChatMessageHistory",
    "MongoDBChatMessageHistory",
    "PostgresChatMessageHistory",
    "RedisChatMessageHistory",
    "RedisEntityStore",
    "SingleStoreDBChatMessageHistory",
    "SQLChatMessageHistory",
    "StreamlitChatMessageHistory",
    "XataChatMessageHistory",
    "ZepChatMessageHistory",
    "UpstashRedisEntityStore",
    "UpstashRedisChatMessageHistory",
]


def test_all_imports() -> None:
    assert set(memory.__all__) == set(EXPECTED_ALL)
    assert_all_importable(memory)


def test_deprecated_imports() -> None:
    for import_ in EXPECTED_DEPRECATED_IMPORTS:
        with pytest.raises(ImportError) as e:
            getattr(memory, import_)
            assert "langchain_community" in e, f"{import_=} didn't error"
    with pytest.raises(AttributeError):
        getattr(memory, "foo")
