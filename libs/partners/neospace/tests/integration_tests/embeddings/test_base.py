"""Test NeoSpace embeddings."""

import numpy as np
import neospace

from langchain_neospace.embeddings.base import NeoSpaceEmbeddings


def test_langchain_neospace_embedding_documents() -> None:
    """Test neospace embeddings."""
    documents = ["foo bar"]
    embedding = NeoSpaceEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_neospace_embedding_query() -> None:
    """Test neospace embeddings."""
    document = "foo bar"
    embedding = NeoSpaceEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0


def test_langchain_neospace_embeddings_dimensions() -> None:
    """Test neospace embeddings."""
    documents = ["foo bar"]
    embedding = NeoSpaceEmbeddings(model="text-embedding-3-small", dimensions=128)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 128


def test_langchain_neospace_embeddings_equivalent_to_raw() -> None:
    documents = ["disallowed special token '<|endoftext|>'"]
    embedding = NeoSpaceEmbeddings()

    lc_output = embedding.embed_documents(documents)[0]
    direct_output = (
        neospace.NeoSpace()
        .embeddings.create(input=documents, model=embedding.model)
        .data[0]
        .embedding
    )
    assert np.isclose(lc_output, direct_output).all()


async def test_langchain_neospace_embeddings_equivalent_to_raw_async() -> None:
    documents = ["disallowed special token '<|endoftext|>'"]
    embedding = NeoSpaceEmbeddings()

    lc_output = (await embedding.aembed_documents(documents))[0]
    client = neospace.AsyncNeoSpace()
    direct_output = (
        (await client.embeddings.create(input=documents, model=embedding.model))
        .data[0]
        .embedding
    )
    assert np.isclose(lc_output, direct_output).all()
