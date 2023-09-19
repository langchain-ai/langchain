from langchain.embeddings import DeterministicFakeEmbedding


def test_deterministic_fake_embeddings() -> None:
    """
    Test that the deterministic fake embeddings return the same
    embedding vector for the same text.
    """
    fake = DeterministicFakeEmbedding(size=10)
    text = "Hello world!"
    assert fake.embed_query(text) == fake.embed_query(text)
    assert fake.embed_query(text) != fake.embed_query("Goodbye world!")
    assert fake.embed_documents([text, text]) == fake.embed_documents([text, text])
    assert fake.embed_documents([text, text]) != fake.embed_documents(
        [text, "Goodbye world!"]
    )
