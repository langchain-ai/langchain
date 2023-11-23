from langchain.schema.cache import __all__

EXPECTED_ALL = ["BaseCache"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
