"""Test Embaas embeddings."""
from langchain.embeddings.embaas import EmbaasEmbeddings


def test_embaas_embed_documents() -> None:
    """Test Embaas embeddings with multiple texts."""
    texts = ["foo bar", "bar foo", "foo"]
    embedding = EmbaasEmbeddings()
    output = embedding.embed_documents(texts)
    assert len(output) == 3
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024


def test_embaas_embed_query() -> None:
    """Test Embaas embeddings with multiple texts."""
    texts = "foo"
    embeddings = EmbaasEmbeddings()
    output = embeddings.embed_query("foo")
    assert len(output) == 1024


def test_embaas_embed_query_instruction() -> None:
    """Test Embaas embeddings with a different instruction."""
    text = "Test"
    embeddings = EmbaasEmbeddings(instruction="Query")
    output = embeddings.embed_query(text)
    assert len(output) == 1024


def test_embaas_embed_query_model() -> None:
    text = "Test"
    model = "instructor-large"
    instruction = "Represent the query for retrieval"
    embeddings = EmbaasEmbeddings(model=model, instruction=instruction)
    output = embeddings.embed_query(text)
    assert len(output) == 768
