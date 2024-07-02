"""Test MixedbreadAI embeddings."""

import pytest

from langchain_mixedbreadai.embeddings import MixedbreadAIEmbeddings


@pytest.mark.vcr()
def test_langchain_mixedbreadai_embedding_documents() -> None:
    documents = ["foo bar"]
    embedding = MixedbreadAIEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


@pytest.mark.vcr()
def test_langchain_mixedbreadai_embedding_query() -> None:
    document = "foo bar"
    embedding = MixedbreadAIEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
