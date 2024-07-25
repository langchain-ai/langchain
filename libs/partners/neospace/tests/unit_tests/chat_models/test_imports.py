from langchain_neospace.chat_models import __all__

EXPECTED_ALL = ["ChatNeoSpace"]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
