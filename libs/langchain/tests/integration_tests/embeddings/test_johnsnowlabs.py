"""Test johnsnowlabs embeddings."""

from langchain.embeddings.johnsnowlabs import JohnSnowLabsEmbeddings


def test_johnsnowlabs_embed_document() -> None:
    """Test johnsnowlabs embeddings."""
    documents = ["foo bar", "bar foo"]
    embedding = JohnSnowLabsEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 128


def test_johnsnowlabs_embed_query() -> None:
    """Test johnsnowlabs embeddings."""
    document = "foo bar"
    embedding = JohnSnowLabsEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 128
