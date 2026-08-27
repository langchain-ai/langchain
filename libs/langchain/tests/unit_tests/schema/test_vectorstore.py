from langchain_classic.schema.vectorstore import __all__

EXPECTED_ALL = ["VectorStore", "VectorStoreRetriever", "VST"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
