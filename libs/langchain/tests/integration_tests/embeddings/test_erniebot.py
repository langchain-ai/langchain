"""Test ERNIE Bot embeddings."""

import pytest

from langchain.embeddings.erniebot import ERNIEBotEmbeddings


def test_erniebot_embed_query() -> None:
    query = "foo"
    embedding = ERNIEBotEmbeddings()
    output = embedding.embed_query(query)
    assert len(output) == 384


@pytest.mark.asyncio
async def test_erniebot_aquery() -> None:
    query = "foo"
    embedding = ERNIEBotEmbeddings()
    output = await embedding.aembed_query(query)
    assert len(output) == 384


def test_erniebot_embed_documents() -> None:
    documents = ["foo", "bar"]
    embedding = ERNIEBotEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 384
    assert len(output[1]) == 384


@pytest.mark.asyncio
async def test_erniebot_aembed_documents() -> None:
    documents = ["foo", "bar"]
    embedding = ERNIEBotEmbeddings()
    output = await embedding.aembed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 384
    assert len(output[1]) == 384
