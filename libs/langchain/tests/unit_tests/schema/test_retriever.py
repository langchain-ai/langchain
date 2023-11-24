from langchain.schema.retriever import __all__

EXPECTED_ALL = ["BaseRetriever"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
