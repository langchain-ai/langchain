from langchain.schema.vectorstore import __all__

EXPECTED_ALL = ["VectorStore", "VectorStoreRetriever"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
