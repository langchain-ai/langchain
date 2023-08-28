import pytest

from langchain.embeddings.ernie import ErnieEmbeddings


def test_embedding_documents_1() -> None:
    documents = ["foo bar"]
    embedding = ErnieEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 384


def test_embedding_documents_2() -> None:
    documents = ["foo", "bar"]
    embedding = ErnieEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 384
    assert len(output[1]) == 384


def test_embedding_query() -> None:
    query = "foo"
    embedding = ErnieEmbeddings()
    output = embedding.embed_query(query)
    assert len(output) == 384


def test_max_chunks() -> None:
    documents = [f"text-{i}" for i in range(20)]
    embedding = ErnieEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 20


def test_too_many_chunks() -> None:
    documents = [f"text-{i}" for i in range(20)]
    embedding = ErnieEmbeddings(chunk_size=20)
    with pytest.raises(ValueError):
        embedding.embed_documents(documents)
