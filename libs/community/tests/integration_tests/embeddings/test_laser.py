"""Test LASER embeddings."""

import pytest

from langchain_community.embeddings.laser import LaserEmbeddings


@pytest.mark.filterwarnings("ignore::UserWarning:")
@pytest.mark.parametrize("lang", [None, "lus_Latn", "english"])
def test_laser_embedding_documents(lang: str) -> None:
    """Test laser embeddings for documents.
    User warning is returned by LASER library implementation
    so will ignore in testing."""
    documents = ["hello", "world"]
    embedding = LaserEmbeddings(lang=lang)  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 2  # type: ignore[arg-type]
    assert len(output[0]) == 1024  # type: ignore[index]


@pytest.mark.filterwarnings("ignore::UserWarning:")
@pytest.mark.parametrize("lang", [None, "lus_Latn", "english"])
def test_laser_embedding_query(lang: str) -> None:
    """Test laser embeddings for query.
    User warning is returned by LASER library implementation
    so will ignore in testing."""
    query = "hello world"
    embedding = LaserEmbeddings(lang=lang)  # type: ignore[call-arg]
    output = embedding.embed_query(query)
    assert len(output) == 1024
