from langchain.memory import chat_message_histories
from tests.unit_tests import assert_all_importable

EXPECTED_ALL = [
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


def test_imports() -> None:
    assert sorted(chat_message_histories.__all__) == sorted(EXPECTED_ALL)
    assert_all_importable(chat_message_histories)
