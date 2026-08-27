from langchain_classic.schema.chat import __all__

EXPECTED_ALL = ["ChatSession"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
