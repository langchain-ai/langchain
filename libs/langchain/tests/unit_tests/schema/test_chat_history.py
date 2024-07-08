from langchain.schema.chat_history import __all__

EXPECTED_ALL = ["BaseChatMessageHistory"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
