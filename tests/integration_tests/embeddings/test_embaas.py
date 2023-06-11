"""Test embaas embeddings."""
import responses

from langchain.embeddings.embaas import EMBAAS_API_URL, EmbaasEmbeddings


def test_embaas_embed_documents() -> None:
    """Test embaas embeddings with multiple texts."""
    texts = ["foo bar", "bar foo", "foo"]
    embedding = EmbaasEmbeddings()
    output = embedding.embed_documents(texts)
    assert len(output) == 3
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024


def test_embaas_embed_query() -> None:
    """Test embaas embeddings with multiple texts."""
    text = "foo"
    embeddings = EmbaasEmbeddings()
    output = embeddings.embed_query(text)
    assert len(output) == 1024


def test_embaas_embed_query_instruction() -> None:
    """Test embaas embeddings with a different instruction."""
    text = "Test"
    instruction = "query"
    embeddings = EmbaasEmbeddings(instruction=instruction)
    output = embeddings.embed_query(text)
    assert len(output) == 1024


def test_embaas_embed_query_model() -> None:
    """Test embaas embeddings with a different model."""
    text = "Test"
    model = "instructor-large"
    instruction = "Represent the query for retrieval"
    embeddings = EmbaasEmbeddings(model=model, instruction=instruction)
    output = embeddings.embed_query(text)
    assert len(output) == 768


@responses.activate
def test_embaas_embed_documents_response() -> None:
    """Test embaas embeddings with multiple texts."""
    responses.add(
        responses.POST,
        EMBAAS_API_URL,
        json={"data": [{"embedding": [0.0] * 1024}]},
        status=200,
    )

    text = "asd"
    embeddings = EmbaasEmbeddings()
    output = embeddings.embed_query(text)
    assert len(output) == 1024
