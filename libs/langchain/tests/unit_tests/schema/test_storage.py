from langchain.schema.storage import __all__

EXPECTED_ALL = ["BaseStore"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
