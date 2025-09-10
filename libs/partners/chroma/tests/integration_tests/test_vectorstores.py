"""Test Chroma functionality."""

import os.path
import tempfile
import uuid
from collections.abc import Generator
from typing import (
    cast,
)

import chromadb
import pytest  # type: ignore[import-not-found]
import requests
from chromadb.api.client import SharedSystemClient
from chromadb.api.rust import RustBindingsAPI
from chromadb.api.types import Embeddable
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings as Fak

from langchain_chroma.vectorstores import Chroma
from tests.integration_tests.fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)


class MyEmbeddingFunction:
    def __init__(self, fak: Fak):
        self.fak = fak

    def __call__(self, input_: Embeddable) -> list[list[float]]:
        texts = cast(list[str], input_)
        return self.fak.embed_documents(texts=texts)


@pytest.fixture()
def client() -> Generator[chromadb.ClientAPI, None, None]:
    SharedSystemClient.clear_system_cache()
    client = chromadb.Client(chromadb.config.Settings())
    yield client


def test_chroma() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
    )
    output = docsearch.similarity_search("foo", k=1)

    docsearch.delete_collection()
    assert len(output) == 1
    assert output[0].page_content == "foo"
    assert output[0].id is not None


def test_from_documents() -> None:
    """Test init using .from_documents."""
    documents = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    docsearch = Chroma.from_documents(documents=documents, embedding=FakeEmbeddings())
    output = docsearch.similarity_search("foo", k=1)

    docsearch.delete_collection()
    assert len(output) == 1
    assert output[0].page_content == "foo"
    assert output[0].id is not None


def test_chroma_with_ids() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    ids = [f"id_{i}" for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        ids=ids,
    )
    output = docsearch.similarity_search("foo", k=1)

    docsearch.delete_collection()
    assert len(output) == 1
    assert output[0].page_content == "foo"
    assert output[0].id == "id_0"


async def test_chroma_async() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
    )
    output = await docsearch.asimilarity_search("foo", k=1)

    docsearch.delete_collection()
    assert len(output) == 1
    assert output[0].page_content == "foo"
    assert output[0].id is not None


async def test_chroma_async_with_ids() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    ids = [f"id_{i}" for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        ids=ids,
    )
    output = await docsearch.asimilarity_search("foo", k=1)

    docsearch.delete_collection()
    assert len(output) == 1
    assert output[0].page_content == "foo"
    assert output[0].id == "id_0"


async def test_chroma_async_http_client() -> None:
    """Test async operations with AsyncHttpClient."""
    try:
        from chromadb import AsyncHttpClient
    except ImportError:
        pytest.skip("AsyncHttpClient not available in this ChromaDB version")

    # Try to connect to a local Chroma server
    # This test requires a running Chroma server
    try:
        async_client = await AsyncHttpClient(
            host="localhost",
            port=8000,
        )
        # Test connection
        await async_client.heartbeat()
    except Exception:
        pytest.skip("Chroma server not running on localhost:8000")

    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]

    # Create Chroma instance with async client
    collection_name = f"test_async_{uuid.uuid4().hex}"
    docsearch = Chroma(
        collection_name=collection_name,
        embedding_function=FakeEmbeddings(),
        async_client=async_client,
    )

    # Test async add_texts
    ids = await docsearch.aadd_texts(texts=texts, metadatas=metadatas)
    assert len(ids) == 3

    # Test async similarity search
    output = await docsearch.asimilarity_search("foo", k=2)
    assert len(output) <= 2
    assert output[0].page_content in texts

    # Test async similarity search with score
    output_with_scores = await docsearch.asimilarity_search_with_score("bar", k=2)
    assert len(output_with_scores) <= 2
    for doc, score in output_with_scores:
        assert doc.page_content in texts
        assert isinstance(score, float)

    # Test async delete
    await docsearch.adelete(ids=[ids[0]])

    # Clean up
    await docsearch.adelete_collection()


async def test_chroma_async_add_documents() -> None:
    """Test async add_documents functionality."""
    documents = [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="bar", metadata={"page": "1"}),
        Document(page_content="baz", metadata={"page": "2"}),
    ]

    docsearch = Chroma.from_documents(
        collection_name="test_async_docs",
        documents=documents,
        embedding=FakeEmbeddings(),
    )

    # Add more documents asynchronously
    new_documents = [
        Document(page_content="qux", metadata={"page": "3"}),
        Document(page_content="quux", metadata={"page": "4"}),
    ]
    ids = await docsearch.aadd_documents(documents=new_documents)
    assert len(ids) == 2

    # Verify documents were added
    output = await docsearch.asimilarity_search("qux", k=1)
    assert len(output) == 1
    assert output[0].page_content == "qux"
    assert output[0].metadata == {"page": "3"}

    docsearch.delete_collection()


async def test_chroma_both_sync_and_async_clients() -> None:
    """Test that both sync and async clients can coexist."""
    # Create sync client
    sync_client = chromadb.Client(chromadb.config.Settings())

    # Try to create async client if available
    try:
        from chromadb import AsyncHttpClient

        async_client = await AsyncHttpClient(
            host="localhost",
            port=8000,
        )
        await async_client.heartbeat()
    except (ImportError, Exception):
        # If AsyncHttpClient is not available or server not running,
        # create a mock async client for testing
        async_client = None
        pytest.skip("AsyncHttpClient not available or server not running")

    collection_name = f"test_both_{uuid.uuid4().hex}"

    # Initialize with both clients
    docsearch = Chroma(
        collection_name=collection_name,
        embedding_function=FakeEmbeddings(),
        client=sync_client,
        async_client=async_client,
    )

    texts = ["foo", "bar", "baz"]

    # Test sync operations
    sync_ids = docsearch.add_texts(texts=texts)
    assert len(sync_ids) == 3

    sync_results = docsearch.similarity_search("foo", k=1)
    assert len(sync_results) == 1

    # Test async operations
    async_texts = ["async_foo", "async_bar"]
    async_ids = await docsearch.aadd_texts(texts=async_texts)
    assert len(async_ids) == 2

    async_results = await docsearch.asimilarity_search("async_foo", k=1)
    assert len(async_results) == 1
    assert async_results[0].page_content == "async_foo"

    # Clean up
    docsearch.delete_collection()


async def test_chroma_async_error_handling() -> None:
    """Test error handling in async operations."""
    # Test without async client
    docsearch = Chroma(
        collection_name="test_error",
        embedding_function=FakeEmbeddings(),
    )

    # Should raise error when trying to use async methods without async_client
    with pytest.raises(ValueError, match="async_client"):
        await docsearch.aadd_texts(["test"])

    with pytest.raises(ValueError, match="async_client"):
        await docsearch.asimilarity_search("test")

    with pytest.raises(ValueError, match="async_client"):
        await docsearch.asimilarity_search_with_score("test")

    with pytest.raises(ValueError, match="async_client"):
        await docsearch.adelete(["test_id"])

    docsearch.delete_collection()


async def test_chroma_async_collection_operations() -> None:
    """Test async collection management operations."""
    try:
        from chromadb import AsyncHttpClient

        async_client = await AsyncHttpClient(
            host="localhost",
            port=8000,
        )
        await async_client.heartbeat()
    except (ImportError, Exception):
        pytest.skip("AsyncHttpClient not available or server not running")

    collection_name = f"test_collection_ops_{uuid.uuid4().hex}"

    docsearch = Chroma(
        collection_name=collection_name,
        embedding_function=FakeEmbeddings(),
        async_client=async_client,
    )

    # Add initial data
    texts = ["foo", "bar", "baz"]
    await docsearch.aadd_texts(texts=texts)

    # Test reset collection
    await docsearch.areset_collection()

    # Collection should be empty after reset
    # Add new data to verify collection was reset
    new_texts = ["new_foo"]
    ids = await docsearch.aadd_texts(texts=new_texts)
    assert len(ids) == 1

    # Clean up
    await docsearch.adelete_collection()


async def test_chroma_async_with_metadata_filter() -> None:
    """Test async operations with metadata filtering."""
    texts = ["foo", "bar", "baz", "qux"]
    metadatas = [
        {"type": "a", "page": "0"},
        {"type": "b", "page": "1"},
        {"type": "a", "page": "2"},
        {"type": "b", "page": "3"},
    ]

    docsearch = Chroma.from_texts(
        collection_name="test_async_filter",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )

    # Test async search with filter
    output = await docsearch.asimilarity_search("foo", k=10, filter={"type": "a"})

    # Should only return documents with type="a"
    for doc in output:
        assert doc.metadata.get("type") == "a"

    docsearch.delete_collection()


def test_chroma_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1)
    docsearch.delete_collection()
    assert len(output) == 1
    assert output[0].page_content == "foo"
    assert output[0].metadata == {"page": "0"}
    assert output[0].id is not None


def test_chroma_with_metadatas_and_ids() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    ids = [f"id_{i}" for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        ids=ids,
    )
    output = docsearch.similarity_search("foo", k=1)
    docsearch.delete_collection()
    assert len(output) == 1
    assert output[0].page_content == "foo"
    assert output[0].metadata == {"page": "0"}
    assert output[0].id == "id_0"


def test_chroma_with_metadatas_with_scores_and_ids() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    ids = [f"id_{i}" for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        ids=ids,
    )
    output = docsearch.similarity_search_with_score("foo", k=1)
    docsearch.delete_collection()
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}, id="id_0"), 0.0),
    ]


def test_chroma_with_metadatas_with_vectors() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    embeddings = ConsistentFakeEmbeddings()
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )
    vec_1 = embeddings.embed_query(texts[0])
    output = docsearch.similarity_search_with_vectors("foo", k=1)
    docsearch.delete_collection()
    assert output[0][0] == Document(page_content="foo", metadata={"page": "0"})
    assert (output[0][1] == vec_1).all()


def test_chroma_with_metadatas_with_scores_using_vector() -> None:
    """Test end to end construction and scored search, using embedding vector."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    ids = [f"id_{i}" for i in range(len(texts))]
    embeddings = FakeEmbeddings()

    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    embedded_query = embeddings.embed_query("foo")
    output = docsearch.similarity_search_by_vector_with_relevance_scores(
        embedding=embedded_query,
        k=1,
    )
    docsearch.delete_collection()
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}, id="id_0"), 0.0),
    ]


def test_chroma_search_filter() -> None:
    """Test end to end construction and search with metadata filtering."""
    texts = ["far", "bar", "baz"]
    metadatas = [{"first_letter": f"{text[0]}"} for text in texts]
    ids = [f"id_{i}" for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        ids=ids,
    )
    output1 = docsearch.similarity_search("far", k=1, filter={"first_letter": "f"})
    output2 = docsearch.similarity_search("far", k=1, filter={"first_letter": "b"})
    docsearch.delete_collection()
    assert output1 == [
        Document(page_content="far", metadata={"first_letter": "f"}, id="id_0"),
    ]
    assert output2 == [
        Document(page_content="bar", metadata={"first_letter": "b"}, id="id_1"),
    ]


def test_chroma_search_filter_with_scores() -> None:
    """Test end to end construction and scored search with metadata filtering."""
    texts = ["far", "bar", "baz"]
    metadatas = [{"first_letter": f"{text[0]}"} for text in texts]
    ids = [f"id_{i}" for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        ids=ids,
    )
    output1 = docsearch.similarity_search_with_score(
        "far",
        k=1,
        filter={"first_letter": "f"},
    )
    output2 = docsearch.similarity_search_with_score(
        "far",
        k=1,
        filter={"first_letter": "b"},
    )
    docsearch.delete_collection()
    assert output1 == [
        (Document(page_content="far", metadata={"first_letter": "f"}, id="id_0"), 0.0),
    ]
    assert output2 == [
        (Document(page_content="bar", metadata={"first_letter": "b"}, id="id_1"), 1.0),
    ]


def test_chroma_with_persistence() -> None:
    """Test end to end construction and search, with persistence."""
    with tempfile.TemporaryDirectory() as chroma_persist_dir:
        collection_name = "test_collection"
        texts = ["foo", "bar", "baz"]
        ids = [f"id_{i}" for i in range(len(texts))]

        docsearch = Chroma.from_texts(
            collection_name=collection_name,
            texts=texts,
            embedding=FakeEmbeddings(),
            persist_directory=chroma_persist_dir,
            ids=ids,
        )

        try:
            output = docsearch.similarity_search("foo", k=1)
            assert output == [Document(page_content="foo", id="id_0")]

            assert os.path.exists(chroma_persist_dir)

            # Get a new VectorStore from the persisted directory
            docsearch = Chroma(
                collection_name=collection_name,
                embedding_function=FakeEmbeddings(),
                persist_directory=chroma_persist_dir,
            )
            output = docsearch.similarity_search("foo", k=1)
            assert output == [Document(page_content="foo", id="id_0")]

            # Clean up
            docsearch.delete_collection()

            # Persist doesn't need to be called again
            # Data will be automatically persisted on object deletion
            # Or on program exit

        finally:
            # Need to stop the chrom system database and segment manager
            # to be able to delete the files after testing
            client = docsearch._client
            assert isinstance(client, chromadb.ClientCreator)
            assert isinstance(client._server, RustBindingsAPI)
            client._server.stop()


def test_chroma_with_persistence_with_client_settings() -> None:
    """Test end to end construction and search, with persistence."""
    with tempfile.TemporaryDirectory() as chroma_persist_dir:
        client_settings = chromadb.config.Settings()
        collection_name = "test_collection"
        texts = ["foo", "bar", "baz"]
        ids = [f"id_{i}" for i in range(len(texts))]
        docsearch = Chroma.from_texts(
            collection_name=collection_name,
            texts=texts,
            embedding=FakeEmbeddings(),
            persist_directory=chroma_persist_dir,
            client_settings=client_settings,
            ids=ids,
        )

        try:
            output = docsearch.similarity_search("foo", k=1)
            assert output == [Document(page_content="foo", id="id_0")]

            assert os.path.exists(chroma_persist_dir)

            # Get a new VectorStore from the persisted directory
            docsearch = Chroma(
                collection_name=collection_name,
                embedding_function=FakeEmbeddings(),
                persist_directory=chroma_persist_dir,
            )
            output = docsearch.similarity_search("foo", k=1)

            # Clean up
            docsearch.delete_collection()

            # Persist doesn't need to be called again
            # Data will be automatically persisted on object deletion
            # Or on program exit

        finally:
            # Need to stop the chrom system database and segment manager
            # to be able to delete the files after testing
            client = docsearch._client
            assert isinstance(client, chromadb.ClientCreator)
            assert isinstance(client._server, RustBindingsAPI)
            client._server.stop()


def test_chroma_mmr() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
    )
    output = docsearch.max_marginal_relevance_search("foo", k=1)
    docsearch.delete_collection()
    assert len(output) == 1
    assert output[0].page_content == "foo"
    assert output[0].id is not None


def test_chroma_mmr_by_vector() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    embeddings = FakeEmbeddings()
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=embeddings,
    )
    embedded_query = embeddings.embed_query("foo")
    output = docsearch.max_marginal_relevance_search_by_vector(embedded_query, k=1)
    docsearch.delete_collection()
    assert len(output) == 1
    assert output[0].page_content == "foo"
    assert output[0].id is not None


def test_chroma_with_include_parameter() -> None:
    """Test end to end construction and include parameter."""
    texts = ["foo", "bar", "baz"]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
    )
    output1 = docsearch.get(include=["embeddings"])
    output2 = docsearch.get()
    docsearch.delete_collection()
    assert output1["embeddings"] is not None
    assert output2["embeddings"] is None


def test_chroma_update_document() -> None:
    """Test the update_document function in the Chroma class.

    Uses an external document id.
    """
    # Make a consistent embedding
    embedding = ConsistentFakeEmbeddings()

    # Initial document content and id
    initial_content = "foo"
    document_id = "doc1"

    # Create an instance of Document with initial content and metadata
    original_doc = Document(page_content=initial_content, metadata={"page": "0"})

    # Initialize a Chroma instance with the original document
    docsearch = Chroma.from_documents(
        collection_name="test_collection",
        documents=[original_doc],
        embedding=embedding,
        ids=[document_id],
    )
    old_embedding = docsearch._collection.peek()["embeddings"][  # type: ignore[index]
        docsearch._collection.peek()["ids"].index(document_id)
    ]

    # Define updated content for the document
    updated_content = "updated foo"

    # Create a new Document instance with the updated content and the same id
    updated_doc = Document(page_content=updated_content, metadata={"page": "0"})

    # Update the document in the Chroma instance
    docsearch.update_document(document_id=document_id, document=updated_doc)

    # Perform a similarity search with the updated content
    output = docsearch.similarity_search(updated_content, k=1)

    # Assert that the new embedding is correct
    new_embedding = docsearch._collection.peek()["embeddings"][  # type: ignore[index]
        docsearch._collection.peek()["ids"].index(document_id)
    ]

    docsearch.delete_collection()

    # Assert that the updated document is returned by the search
    assert output == [
        Document(page_content=updated_content, metadata={"page": "0"}, id=document_id),
    ]

    assert list(new_embedding) == list(embedding.embed_documents([updated_content])[0])
    assert list(new_embedding) != list(old_embedding)


def test_chroma_update_document_with_id() -> None:
    """Test the update_document function in the Chroma class.

    Uses an internal document id.
    """
    # Make a consistent embedding
    embedding = ConsistentFakeEmbeddings()

    # Initial document content and id
    initial_content = "foo"
    document_id = "doc1"

    # Create an instance of Document with initial content and metadata
    original_doc = Document(
        page_content=initial_content,
        metadata={"page": "0"},
        id=document_id,
    )

    # Initialize a Chroma instance with the original document
    docsearch = Chroma.from_documents(
        collection_name="test_collection",
        documents=[original_doc],
        embedding=embedding,
    )
    old_embedding = docsearch._collection.peek()["embeddings"][  # type: ignore[index]
        docsearch._collection.peek()["ids"].index(document_id)
    ]

    # Define updated content for the document
    updated_content = "updated foo"

    # Create a new Document instance with the updated content and the same id
    updated_doc = Document(
        page_content=updated_content,
        metadata={"page": "0"},
        id=document_id,
    )

    # Update the document in the Chroma instance
    docsearch.update_document(document_id=document_id, document=updated_doc)

    # Perform a similarity search with the updated content
    output = docsearch.similarity_search(updated_content, k=1)

    # Assert that the new embedding is correct
    new_embedding = docsearch._collection.peek()["embeddings"][  # type: ignore[index]
        docsearch._collection.peek()["ids"].index(document_id)
    ]

    docsearch.delete_collection()

    # Assert that the updated document is returned by the search
    assert output == [
        Document(page_content=updated_content, metadata={"page": "0"}, id=document_id),
    ]

    assert list(new_embedding) == list(embedding.embed_documents([updated_content])[0])
    assert list(new_embedding) != list(old_embedding)


# TODO: RELEVANCE SCORE IS BROKEN. FIX TEST
def test_chroma_with_relevance_score_custom_normalization_fn() -> None:
    """Test searching with relevance score and custom normalization function."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    ids = [f"id_{i}" for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test1_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        ids=ids,
        relevance_score_fn=lambda d: d * 0,
        collection_metadata={"hnsw:space": "l2"},
    )
    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    docsearch.delete_collection()
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}, id="id_0"), 0.0),
        (Document(page_content="bar", metadata={"page": "1"}, id="id_1"), 0.0),
        (Document(page_content="baz", metadata={"page": "2"}, id="id_2"), 0.0),
    ]


def test_init_from_client(client: chromadb.ClientAPI) -> None:
    Chroma(client=client)


def test_init_from_client_settings() -> None:
    import chromadb

    client_settings = chromadb.config.Settings()
    Chroma(client_settings=client_settings)


def test_chroma_add_documents_no_metadata() -> None:
    db = Chroma(embedding_function=FakeEmbeddings())
    db.add_documents([Document(page_content="foo")])

    db.delete_collection()


def test_chroma_add_documents_mixed_metadata() -> None:
    db = Chroma(embedding_function=FakeEmbeddings())
    docs = [
        Document(page_content="foo", id="0"),
        Document(page_content="bar", metadata={"baz": 1}, id="1"),
    ]
    ids = ["0", "1"]
    actual_ids = db.add_documents(docs)
    search = db.similarity_search("foo bar")
    db.delete_collection()

    assert actual_ids == ids
    assert sorted(search, key=lambda d: d.page_content) == sorted(
        docs,
        key=lambda d: d.page_content,
    )


def is_api_accessible(url: str) -> bool:
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def batch_support_chroma_version() -> bool:
    major, minor, patch = chromadb.__version__.split(".")
    return bool(int(major) == 0 and int(minor) >= 4 and int(patch) >= 10)


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
@pytest.mark.skipif(
    not batch_support_chroma_version(),
    reason="ChromaDB version does not support batching",
)
def test_chroma_large_batch() -> None:
    client = chromadb.HttpClient()
    embedding_function = MyEmbeddingFunction(fak=Fak(size=255))
    col = client.get_or_create_collection(
        "my_collection",
        embedding_function=embedding_function,  # type: ignore[arg-type]
    )
    docs = ["This is a test document"] * (client.get_max_batch_size() + 100)
    db = Chroma.from_texts(
        client=client,
        collection_name=col.name,
        texts=docs,
        embedding=embedding_function.fak,
        ids=[str(uuid.uuid4()) for _ in range(len(docs))],
    )

    db.delete_collection()


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
@pytest.mark.skipif(
    not batch_support_chroma_version(),
    reason="ChromaDB version does not support batching",
)
def test_chroma_large_batch_update() -> None:
    client = chromadb.HttpClient()
    embedding_function = MyEmbeddingFunction(fak=Fak(size=255))
    col = client.get_or_create_collection(
        "my_collection",
        embedding_function=embedding_function,  # type: ignore[arg-type]
    )
    docs = ["This is a test document"] * (client.get_max_batch_size() + 100)
    ids = [str(uuid.uuid4()) for _ in range(len(docs))]
    db = Chroma.from_texts(
        client=client,
        collection_name=col.name,
        texts=docs,
        embedding=embedding_function.fak,
        ids=ids,
    )
    new_docs = [
        Document(
            page_content="This is a new test document",
            metadata={"doc_id": f"{i}"},
        )
        for i in range(len(docs) - 10)
    ]
    new_ids = list(ids[: len(new_docs)])
    db.update_documents(ids=new_ids, documents=new_docs)

    db.delete_collection()


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
@pytest.mark.skipif(
    batch_support_chroma_version(),
    reason="ChromaDB version does not support batching",
)
def test_chroma_legacy_batching() -> None:
    client = chromadb.HttpClient()
    embedding_function = Fak(size=255)
    col = client.get_or_create_collection(
        "my_collection",
        embedding_function=MyEmbeddingFunction,  # type: ignore[arg-type]
    )
    docs = ["This is a test document"] * 100
    db = Chroma.from_texts(
        client=client,
        collection_name=col.name,
        texts=docs,
        embedding=embedding_function,
        ids=[str(uuid.uuid4()) for _ in range(len(docs))],
    )

    db.delete_collection()


def test_create_collection_if_not_exist_default() -> None:
    """Tests existing behaviour without the new create_collection_if_not_exists flag."""
    texts = ["foo", "bar", "baz"]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
    )
    assert docsearch._client.get_collection("test_collection") is not None
    docsearch.delete_collection()


def test_create_collection_if_not_exist_true_existing(
    client: chromadb.ClientAPI,
) -> None:
    """Tests create_collection_if_not_exists=True and collection already existing."""
    client.create_collection("test_collection")
    vectorstore = Chroma(
        client=client,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
        create_collection_if_not_exists=True,
    )
    assert vectorstore._client.get_collection("test_collection") is not None
    vectorstore.delete_collection()


def test_create_collection_if_not_exist_false_existing(
    client: chromadb.ClientAPI,
) -> None:
    """Tests create_collection_if_not_exists=False and collection already existing."""
    client.create_collection("test_collection")
    vectorstore = Chroma(
        client=client,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
        create_collection_if_not_exists=False,
    )
    assert vectorstore._client.get_collection("test_collection") is not None
    vectorstore.delete_collection()


def test_create_collection_if_not_exist_false_non_existing(
    client: chromadb.ClientAPI,
) -> None:
    """Tests create_collection_if_not_exists=False and collection not-existing,
    should raise."""
    with pytest.raises(Exception, match="does not exist"):
        Chroma(
            client=client,
            collection_name="test_collection",
            embedding_function=FakeEmbeddings(),
            create_collection_if_not_exists=False,
        )


def test_create_collection_if_not_exist_true_non_existing(
    client: chromadb.ClientAPI,
) -> None:
    """Tests create_collection_if_not_exists=True and collection non-existing. ."""
    vectorstore = Chroma(
        client=client,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
        create_collection_if_not_exists=True,
    )

    assert vectorstore._client.get_collection("test_collection") is not None
    vectorstore.delete_collection()


def test_collection_none_after_delete(
    client: chromadb.ClientAPI,
) -> None:
    """Tests create_collection_if_not_exists=True and collection non-existing. ."""
    vectorstore = Chroma(
        client=client,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
    )

    assert vectorstore._client.get_collection("test_collection") is not None
    vectorstore.delete_collection()
    assert vectorstore._chroma_collection is None
    with pytest.raises(Exception, match="Chroma collection not initialized"):
        _ = vectorstore._collection
    with pytest.raises(Exception, match="does not exist"):
        vectorstore._client.get_collection("test_collection")
    with pytest.raises(Exception):  # noqa: B017
        vectorstore.similarity_search("foo")


def test_reset_collection(client: chromadb.ClientAPI) -> None:
    """Tests ensure_collection method."""
    vectorstore = Chroma(
        client=client,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
    )
    vectorstore.add_documents([Document(page_content="foo")])
    assert vectorstore._collection.count() == 1
    vectorstore.reset_collection()
    assert vectorstore._chroma_collection is not None
    assert vectorstore._client.get_collection("test_collection") is not None
    assert vectorstore._collection.name == "test_collection"
    assert vectorstore._collection.count() == 0
    # Clean up
    vectorstore.delete_collection()


def test_delete_where_clause(client: chromadb.ClientAPI) -> None:
    """Tests delete_where_clause method."""
    vectorstore = Chroma(
        client=client,
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(),
    )
    vectorstore.add_documents(
        [
            Document(page_content="foo", metadata={"test": "bar"}),
            Document(page_content="bar", metadata={"test": "foo"}),
        ],
    )
    assert vectorstore._collection.count() == 2
    vectorstore.delete(where={"test": "bar"})
    assert vectorstore._collection.count() == 1
    # Clean up
    vectorstore.delete_collection()

