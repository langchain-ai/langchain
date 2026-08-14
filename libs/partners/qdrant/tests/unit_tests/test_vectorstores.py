import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_qdrant import QdrantVectorStore


@pytest.mark.asyncio
async def test_qdrant_vectorstore_aadd_documents_custom_ids() -> None:
    async_client = AsyncQdrantClient(location=":memory:")
    collection_name = "test_aadd_documents_ids"
    await async_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=10, distance=Distance.COSINE),
    )
    embeddings = DeterministicFakeEmbedding(size=10)

    qdrant_vs = QdrantVectorStore(
        client=async_client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    docs = [
        Document(page_content="Hello world"),
        Document(page_content="Goodbye world"),
    ]
    ids = [
        "44ec7094-b061-45ac-8fbf-014b0f18e8aa",
        "55ec7094-b061-45ac-8fbf-014b0f18e8bb",
    ]

    returned_ids = await qdrant_vs.aadd_documents(docs, ids=ids)
    assert returned_ids == ids


def test_qdrant_vectorstore_add_documents_custom_ids() -> None:
    client = QdrantClient(location=":memory:")
    collection_name = "test_add_documents_ids"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=10, distance=Distance.COSINE),
    )
    embeddings = DeterministicFakeEmbedding(size=10)

    qdrant_vs = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    docs = [
        Document(page_content="Hello world"),
        Document(page_content="Goodbye world"),
    ]
    ids = [
        "44ec7094-b061-45ac-8fbf-014b0f18e8aa",
        "55ec7094-b061-45ac-8fbf-014b0f18e8bb",
    ]

    returned_ids = qdrant_vs.add_documents(docs, ids=ids)
    assert returned_ids == ids
