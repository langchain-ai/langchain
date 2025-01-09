import time
from typing import AsyncGenerator

import pytest
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec  # type: ignore

from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from tests.integration_tests.test_vectorstores import DEFAULT_SLEEP

DIMENSION = 1024
INDEX_NAME = "langchain-pinecone-embeddings"
MODEL = "multilingual-e5-large"
NAMESPACE_NAME = "test_namespace"


@pytest.fixture(scope="function")
async def embd_client() -> AsyncGenerator[PineconeEmbeddings, None]:
    client = PineconeEmbeddings(model=MODEL)
    yield client
    await client.async_client.close()


@pytest.fixture
def pc() -> Pinecone:
    return Pinecone()


@pytest.fixture()
def pc_index(pc: Pinecone) -> Pinecone.Index:
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


def test_embed_query(embd_client: PineconeEmbeddings) -> None:
    out = embd_client.embed_query("Hello, world!")
    assert isinstance(out, list)
    assert len(out) == DIMENSION


@pytest.mark.asyncio
async def test_aembed_query(embd_client: PineconeEmbeddings) -> None:
    out = await embd_client.aembed_query("Hello, world!")
    assert isinstance(out, list)
    assert len(out) == DIMENSION


def test_embed_documents(embd_client: PineconeEmbeddings) -> None:
    out = embd_client.embed_documents(["Hello, world!", "This is a test."])
    assert isinstance(out, list)
    assert len(out) == 2
    assert len(out[0]) == DIMENSION


@pytest.mark.asyncio
async def test_aembed_documents(embd_client: PineconeEmbeddings) -> None:
    out = await embd_client.aembed_documents(["Hello, world!", "This is a test."])
    assert isinstance(out, list)
    assert len(out) == 2
    assert len(out[0]) == DIMENSION


def test_vector_store(
    embd_client: PineconeEmbeddings, pc_index: Pinecone.Index
) -> None:
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embd_client)
    vectorstore.add_documents(
        [Document("Hello, world!"), Document("This is a test.")],
        namespace=NAMESPACE_NAME,
    )
    time.sleep(DEFAULT_SLEEP)  # Increase wait time to ensure indexing is complete
    resp = vectorstore.similarity_search(query="hello", namespace=NAMESPACE_NAME)
    assert len(resp) == 2
