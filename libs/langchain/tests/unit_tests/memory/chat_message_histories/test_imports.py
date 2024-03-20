import pytest

from langchain.memory import chat_message_histories
from tests.unit_tests import assert_all_importable

EXPECTED_ALL = [
    "ChatMessageHistory",
    "FileChatMessageHistory",
]
EXPECTED_DEPRECATED_IMPORTS = [
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


def test_deprecated_imports() -> None:
    for import_ in EXPECTED_DEPRECATED_IMPORTS:
        with pytest.raises(ImportError) as e:
            getattr(chat_message_histories, import_)
            assert "langchain_community" in e, f"{import_=} didn't error"
    with pytest.raises(AttributeError):
        getattr(chat_message_histories, "foo")
