"""Test __ModuleName__ embeddings."""

from __module_name__.embeddings import (  # type: ignore[import-not-found]
    __ModuleName__Embeddings,  # type: ignore[import-not-found]
)


def test___module_name___embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = __ModuleName__Embeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test___module_name___embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = __ModuleName__Embeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
