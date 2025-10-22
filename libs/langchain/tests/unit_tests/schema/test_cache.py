from langchain_classic.schema.cache import __all__

EXPECTED_ALL = ["BaseCache", "RETURN_VAL_TYPE"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
