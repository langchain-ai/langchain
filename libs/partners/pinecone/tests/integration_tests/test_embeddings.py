import pytest

from langchain_pinecone import PineconeEmbeddings

DIMENSION = 1024

@pytest.fixture()
def client():
    return PineconeEmbeddings(model="multilingual-e5-large")


def test_embed_query(client):
    out = client.embed_query("Hello, world!")
    assert isinstance(out, list)
    assert len(out) == DIMENSION


@pytest.mark.asyncio
async def test_aembed_query(client):
    out = await client.aembed_query("Hello, world!")
    assert isinstance(out, list)
    assert len(out) == DIMENSION


def test_embed_documents(client):
    out = client.embed_documents(["Hello, world!", "This is a test."])
    assert isinstance(out, list)
    assert len(out) == 2
    assert len(out[0]) == DIMENSION


@pytest.mark.asyncio
async def test_aembed_documents(client):
    out = await client.aembed_documents(["Hello, world!", "This is a test."])
    assert isinstance(out, list)
    assert len(out) == 2
    assert len(out[0]) == DIMENSION
