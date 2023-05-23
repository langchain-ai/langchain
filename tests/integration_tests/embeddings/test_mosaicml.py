"""Test mosaicml embeddings."""
from langchain.embeddings.mosaicml import MosaicMLInstructorEmbeddings


def test_mosaicml_embedding_documents() -> None:
    """Test MosaicML embeddings."""
    documents = ["foo bar"]
    embedding = MosaicMLInstructorEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_mosaicml_embedding_documents_multiple() -> None:
    """Test MosaicML embeddings with multiple documents."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = MosaicMLInstructorEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 768
    assert len(output[1]) == 768
    assert len(output[2]) == 768


def test_mosaicml_embedding_query() -> None:
    """Test MosaicML embeddings of queries."""
    document = "foo bar"
    embedding = MosaicMLInstructorEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 768


def test_mosaicml_embedding_endpoint() -> None:
    """Test MosaicML embeddings with a different endpoint"""
    documents = ["foo bar"]
    embedding = MosaicMLInstructorEmbeddings(
        endpoint_url="https://models.hosted-on.mosaicml.hosting/instructor-xl/v1/predict"
    )
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_mosaicml_embedding_query_instruction() -> None:
    """Test MosaicML embeddings with a different query instruction."""
    document = "foo bar"
    embedding = MosaicMLInstructorEmbeddings(query_instruction="Embed this query:")
    output = embedding.embed_query(document)
    assert len(output) == 768


def test_mosaicml_embedding_document_instruction() -> None:
    """Test MosaicML embeddings with a different query instruction."""
    documents = ["foo bar"]
    embedding = MosaicMLInstructorEmbeddings(embed_instruction="Embed this document:")
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768
