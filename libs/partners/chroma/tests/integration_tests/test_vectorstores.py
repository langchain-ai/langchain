"""Test Chroma functionality."""

import tempfile
import uuid
from pathlib import Path
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
        texts = cast("list[str]", input_)
        return self.fak.embed_documents(texts=texts)


@pytest.fixture
def client() -> chromadb.ClientAPI:
    SharedSystemClient.clear_system_cache()
    return chromadb.Client(chromadb.config.Settings())


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
    doc = output[0][0]
    assert doc.page_content == "foo"
    assert doc.metadata == {"page": "0"}
    assert doc.id is not None
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

            assert Path(chroma_persist_dir).exists()

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

            assert Path(chroma_persist_dir).exists()

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


def test_chroma_with_relevance_score_custom_normalization_fn() -> None:
    """Test searching with relevance score and custom normalization function."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    ids = [f"id_{i}" for i in range(len(texts))]
    embeddings = ConsistentFakeEmbeddings()

    docsearch = Chroma.from_texts(
        collection_name="test1_collection",
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        relevance_score_fn=lambda d: d * 0,
        collection_metadata={"hnsw:space": "l2"},
    )
    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    docsearch.delete_collection()
    # All scores should be 0.0 because relevance_score_fn multiplies by 0
    assert len(output) == 3
    for doc, score in output:
        assert score == 0.0
        assert doc.page_content in texts


def test_chroma_with_relevance_score_custom_normalization_fn_detailed() -> None:
    """Test custom relevance_score_fn with transformation producing different values."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    ids = [f"id_{i}" for i in range(len(texts))]
    embeddings = ConsistentFakeEmbeddings()

    def transform_fn(distance: float) -> float:
        return distance * 2 + 100

    docsearch = Chroma.from_texts(
        collection_name="test2_collection",
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        relevance_score_fn=transform_fn,
        collection_metadata={"hnsw:space": "l2"},
    )

    raw_results = docsearch.similarity_search_with_score("foo", k=3)
    raw_distances = [score for _, score in raw_results]

    relevance_results = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    actual_scores = [score for _, score in relevance_results]

    expected_scores = [transform_fn(d) for d in raw_distances]

    docsearch.delete_collection()

    assert len(actual_scores) == len(expected_scores) == 3
    for actual, expected in zip(actual_scores, expected_scores, strict=True):
        # Allow small floating point differences
        assert abs(actual - expected) < 0.0001, (
            f"Expected transformed score {expected}, but got {actual}. "
            f"Raw distances: {raw_distances}"
        )


def test_chroma_similarity_search_with_relevance_scores_with_filter() -> None:
    """Test similarity_search_with_relevance_scores with explicit filter parameter."""
    texts = ["far", "bar", "baz"]
    metadatas = [{"first_letter": f"{text[0]}"} for text in texts]
    ids = [f"id_{i}" for i in range(len(texts))]
    embeddings = ConsistentFakeEmbeddings()

    def transform_fn(distance: float) -> float:
        return distance * 2 + 100

    docsearch = Chroma.from_texts(
        collection_name="test3_collection",
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        relevance_score_fn=transform_fn,
        collection_metadata={"hnsw:space": "l2"},
    )

    # Test with explicit filter parameter
    results = docsearch.similarity_search_with_relevance_scores(
        "far", k=3, filter={"first_letter": "f"}
    )
    docsearch.delete_collection()

    assert len(results) == 1
    assert results[0][0].page_content == "far"
    assert results[0][0].metadata["first_letter"] == "f"
    # Score should be transformed
    assert results[0][1] >= 100  # transform_fn(distance * 2 + 100) where distance >= 0


def test_chroma_similarity_search_with_relevance_scores_with_filter_kwargs() -> None:
    """Test similarity_search_with_relevance_scores with filter in kwargs."""
    texts = ["far", "bar", "baz"]
    metadatas = [{"first_letter": f"{text[0]}"} for text in texts]
    ids = [f"id_{i}" for i in range(len(texts))]
    embeddings = ConsistentFakeEmbeddings()

    def transform_fn(distance: float) -> float:
        return distance * 2 + 100

    docsearch = Chroma.from_texts(
        collection_name="test4_collection",
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        relevance_score_fn=transform_fn,
        collection_metadata={"hnsw:space": "l2"},
    )

    # Test with filter passed via kwargs (backward compatibility)
    results = docsearch.similarity_search_with_relevance_scores(
        "far", k=3, filter={"first_letter": "b"}
    )
    docsearch.delete_collection()

    assert len(results) >= 1
    # Should find "bar" or "baz" (both start with "b")
    assert any(
        doc.page_content in ("bar", "baz") and doc.metadata["first_letter"] == "b"
        for doc, _ in results
    )


def test_chroma_similarity_search_with_relevance_scores_with_where_document() -> None:
    """Test similarity_search_with_relevance_scores with where_document parameter."""
    texts = ["hello world", "goodbye world", "hello there"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    ids = [f"id_{i}" for i in range(len(texts))]
    embeddings = ConsistentFakeEmbeddings()

    def transform_fn(distance: float) -> float:
        return distance * 2 + 100

    docsearch = Chroma.from_texts(
        collection_name="test5_collection",
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        relevance_score_fn=transform_fn,
        collection_metadata={"hnsw:space": "l2"},
    )

    # Test with explicit where_document parameter
    results = docsearch.similarity_search_with_relevance_scores(
        "hello", k=3, where_document={"$contains": "there"}
    )
    docsearch.delete_collection()

    assert len(results) == 1
    assert results[0][0].page_content == "hello there"
    # Score should be transformed
    assert results[0][1] >= 100


def test_chroma_similarity_search_with_relevance_scores_with_other_kwargs() -> None:
    """Test filter and where_document extraction from kwargs via public API."""
    texts = ["far", "bar", "baz"]
    metadatas = [{"first_letter": f"{text[0]}"} for text in texts]
    ids = [f"id_{i}" for i in range(len(texts))]
    embeddings = ConsistentFakeEmbeddings()

    def transform_fn(distance: float) -> float:
        return distance * 2 + 100

    docsearch = Chroma.from_texts(
        collection_name="test6_collection",
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        relevance_score_fn=transform_fn,
        collection_metadata={"hnsw:space": "l2"},
    )

    # Test backward compatibility: calling via public API with filter in kwargs
    # (simulates how base class calls _similarity_search_with_relevance_scores)
    # The public API doesn't have explicit filter/where_document params, only kwargs
    results = docsearch.similarity_search_with_relevance_scores(
        "far",
        k=3,
        filter={"first_letter": "f"},  # Filter in kwargs should be extracted
    )
    docsearch.delete_collection()

    # Should use filter from kwargs, so should find "far"
    assert len(results) == 1
    assert results[0][0].page_content == "far"
    assert results[0][0].metadata["first_letter"] == "f"
    # Score should be transformed
    assert results[0][1] >= 100


def test_chroma_similarity_search_with_relevance_scores_both_params() -> None:
    """Test similarity_search_with_relevance_scores with filter and where_document."""
    texts = ["hello world", "goodbye world", "hello there"]
    metadatas = [
        {"category": "greeting" if "hello" in text else "farewell"} for text in texts
    ]
    ids = [f"id_{i}" for i in range(len(texts))]
    embeddings = ConsistentFakeEmbeddings()

    def transform_fn(distance: float) -> float:
        return distance * 2 + 100

    docsearch = Chroma.from_texts(
        collection_name="test7_collection",
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        relevance_score_fn=transform_fn,
        collection_metadata={"hnsw:space": "l2"},
    )

    # Test with both filter and where_document
    results = docsearch.similarity_search_with_relevance_scores(
        "hello",
        k=3,
        filter={"category": "greeting"},
        where_document={"$contains": "there"},
    )
    docsearch.delete_collection()

    assert len(results) == 1
    assert results[0][0].page_content == "hello there"
    assert results[0][0].metadata["category"] == "greeting"
    # Score should be transformed
    assert results[0][1] >= 100


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


def test_chroma_handles_none_page_content() -> None:
    """Test that Chroma gracefully handles None page_content values."""
    from langchain_chroma.vectorstores import _results_to_docs_and_scores

    mock_results = {
        "documents": [["valid content", None, "another valid content"]],
        "metadatas": [[{"key": "value1"}, {"key": "value2"}, {"key": "value3"}]],
        "ids": [["id1", "id2", "id3"]],
        "distances": [[0.1, 0.2, 0.3]],
    }

    docs_and_scores = _results_to_docs_and_scores(mock_results)

    assert len(docs_and_scores) == 2
    assert docs_and_scores[0][0].page_content == "valid content"
    assert docs_and_scores[1][0].page_content == "another valid content"
    assert docs_and_scores[0][0].id == "id1"
    assert docs_and_scores[1][0].id == "id3"


def test_chroma_handles_none_page_content_with_vectors() -> None:
    """Test that Chroma gracefully handles None page_content values with vectors."""
    from langchain_chroma.vectorstores import _results_to_docs_and_vectors

    mock_results = {
        "documents": [["valid content", None, "another valid content"]],
        "metadatas": [[{"key": "value1"}, {"key": "value2"}, {"key": "value3"}]],
        "ids": [["id1", "id2", "id3"]],
        "embeddings": [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]],
    }
    docs_and_vectors = _results_to_docs_and_vectors(mock_results)

    assert len(docs_and_vectors) == 2
    assert docs_and_vectors[0][0].page_content == "valid content"
    assert docs_and_vectors[1][0].page_content == "another valid content"
    assert docs_and_vectors[0][0].id == "id1"
    assert docs_and_vectors[1][0].id == "id3"
    assert docs_and_vectors[0][1] == [0.1, 0.2]
    assert docs_and_vectors[1][1] == [0.5, 0.6]
