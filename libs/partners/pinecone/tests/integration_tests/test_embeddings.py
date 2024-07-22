import time

import pytest
from langchain_core.documents import Document
from pinecone import ServerlessSpec, Pinecone

from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore

DIMENSION = 1024
INDEX_NAME = "langchain-pinecone-embeddings"
MODEL = "multilingual-e5-large"


@pytest.fixture()
def embd_client():
    return PineconeEmbeddings(model=MODEL)


@pytest.fixture
def pc():
    return Pinecone()


@pytest.fixture()
def langchain_index(pc):
    pc = Pinecone()
    if INDEX_NAME not in [index["name"] for index in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

    yield pc.Index(INDEX_NAME)

    pc.delete_index(INDEX_NAME)


def test_embed_query(embd_client):
    out = embd_client.embed_query("Hello, world!")
    assert isinstance(out, list)
    assert len(out) == DIMENSION


@pytest.mark.asyncio
async def test_aembed_query(embd_client):
    out = await embd_client.aembed_query("Hello, world!")
    assert isinstance(out, list)
    assert len(out) == DIMENSION


def test_embed_documents(embd_client):
    out = embd_client.embed_documents(["Hello, world!", "This is a test."])
    assert isinstance(out, list)
    assert len(out) == 2
    assert len(out[0]) == DIMENSION


@pytest.mark.asyncio
async def test_aembed_documents(embd_client):
    out = await embd_client.aembed_documents(["Hello, world!", "This is a test."])
    assert isinstance(out, list)
    assert len(out) == 2
    assert len(out[0]) == DIMENSION


def test_vector_store(embd_client, langchain_index):
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embd_client)
    vectorstore.add_documents([Document("Hello, world!"), Document("This is a test.")])
    time.sleep(5)
    resp = vectorstore.similarity_search(query="hello")
    assert len(resp) == 2
