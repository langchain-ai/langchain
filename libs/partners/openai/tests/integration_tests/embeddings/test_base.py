"""Test OpenAI embeddings."""

import numpy as np
import openai

from langchain_openai.embeddings.base import OpenAIEmbeddings


def test_langchain_openai_embedding_documents() -> None:
    """Test openai embeddings."""
    documents = ["foo bar"]
    embedding = OpenAIEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_openai_embedding_query() -> None:
    """Test openai embeddings."""
    document = "foo bar"
    embedding = OpenAIEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0


def test_langchain_openai_embeddings_dimensions() -> None:
    """Test openai embeddings."""
    documents = ["foo bar"]
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=128)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 128


def test_langchain_openai_embeddings_equivalent_to_raw() -> None:
    documents = ["disallowed special token '<|endoftext|>'"]
    embedding = OpenAIEmbeddings()

    lc_output = embedding.embed_documents(documents)[0]
    direct_output = (
        openai.OpenAI()
        .embeddings.create(input=documents, model=embedding.model)
        .data[0]
        .embedding
    )
    assert np.isclose(lc_output, direct_output).all()


async def test_langchain_openai_embeddings_equivalent_to_raw_async() -> None:
    documents = ["disallowed special token '<|endoftext|>'"]
    embedding = OpenAIEmbeddings()

    lc_output = (await embedding.aembed_documents(documents))[0]
    client = openai.AsyncOpenAI()
    direct_output = (
        (await client.embeddings.create(input=documents, model=embedding.model))
        .data[0]
        .embedding
    )
    assert np.isclose(lc_output, direct_output).all()
