"""Test OpenAI embeddings."""

import os

import numpy as np
import openai
import pytest

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
    assert np.allclose(lc_output, direct_output, atol=0.001)


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
    assert np.allclose(lc_output, direct_output, atol=0.001)


def test_langchain_openai_embeddings_dimensions_large_num() -> None:
    """Test openai embeddings."""
    documents = [f"foo bar {i}" for i in range(2000)]
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=128)
    output = embedding.embed_documents(documents)
    assert len(output) == 2000
    assert len(output[0]) == 128


def test_callable_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    original_key = os.environ["OPENAI_API_KEY"]

    calls = {"sync": 0}

    def get_openai_api_key() -> str:
        calls["sync"] += 1
        return original_key

    monkeypatch.delenv("OPENAI_API_KEY")

    model = OpenAIEmbeddings(
        model="text-embedding-3-small", dimensions=128, api_key=get_openai_api_key
    )
    _ = model.embed_query("hello")
    assert calls["sync"] == 1


async def test_callable_api_key_async(monkeypatch: pytest.MonkeyPatch) -> None:
    original_key = os.environ["OPENAI_API_KEY"]

    calls = {"sync": 0, "async": 0}

    def get_openai_api_key() -> str:
        calls["sync"] += 1
        return original_key

    async def get_openai_api_key_async() -> str:
        calls["async"] += 1
        return original_key

    monkeypatch.delenv("OPENAI_API_KEY")

    model = OpenAIEmbeddings(
        model="text-embedding-3-small", dimensions=128, api_key=get_openai_api_key
    )
    _ = model.embed_query("hello")
    assert calls["sync"] == 1

    _ = await model.aembed_query("hello")
    assert calls["sync"] == 2

    model = OpenAIEmbeddings(
        model="text-embedding-3-small", dimensions=128, api_key=get_openai_api_key_async
    )
    _ = await model.aembed_query("hello")
    assert calls["async"] == 1

    with pytest.raises(ValueError):
        # We do not create a sync callable from an async one
        _ = model.embed_query("hello")
