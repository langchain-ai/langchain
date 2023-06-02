import pytest

from langchain.vectorstores.pgvector import PGVECTOR_VECTOR_SIZE, PGVector
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


@pytest.mark.requires("pgvector")
def test_embedding_store_init_defaults() -> None:
    expected = PGVECTOR_VECTOR_SIZE
    actual = PGVector(
        "postgresql+psycopg2://admin:admin@localhost:5432/mydatabase", FakeEmbeddings()
    ).EmbeddingStore.embedding.type.dim
    assert expected == actual


@pytest.mark.requires("pgvector")
def test_embedding_store_init_vector_size() -> None:
    expected = 2
    actual = PGVector(
        "postgresql+psycopg2://admin:admin@localhost:5432/mydatabase",
        FakeEmbeddings(),
        vector_size=2,
    ).EmbeddingStore.embedding.type.dim
    assert expected == actual
