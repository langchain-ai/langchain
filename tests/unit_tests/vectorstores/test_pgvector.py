import os

from langchain.vectorstores.pgvector import PGVECTOR_VECTOR_SIZE, EmbeddingStore


def test_embedding_store_init_defaults() -> None:
    expected = PGVECTOR_VECTOR_SIZE
    actual = EmbeddingStore().embedding.type.dim
    assert expected == actual


def test_embedding_store_init_vector_size() -> None:
    expected = 2
    actual = EmbeddingStore(vector_size=2).embedding.type.dim
    assert expected == actual


def test_embedding_store_init_env_vector_size() -> None:
    os.environ["PGVECTOR_VECTOR_SIZE"] = "3"
    expected = 3
    actual = EmbeddingStore().embedding.type.dim
    assert expected == actual
