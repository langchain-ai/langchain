from langchain import embeddings

EXPECTED_ALL = [
    "CacheBackedEmbeddings",
    "Embeddings",
    "init_embeddings",
]


def test_all_imports() -> None:
    assert set(embeddings.__all__) == set(EXPECTED_ALL)
