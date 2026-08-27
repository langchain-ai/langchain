from langchain_classic.schema.embeddings import __all__

EXPECTED_ALL = ["Embeddings"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
