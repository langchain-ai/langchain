"""Test Chroma functionality."""
import shutil
import tempfile
import uuid

import pytest
from langchain_core.documents import Document

from langchain_community.retrievers.chroma_retriever import ChromaRetriever
from langchain_community.vectorstores.chroma import Chroma
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)


@pytest.fixture(scope="module")
def temp_dir():
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Yield the temporary directory path
    yield temp_dir
    
    # Tear down the temporary directory after tests are completed
    shutil.rmtree(temp_dir)


def test_chroma() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    retriever = ChromaRetriever(
        store=Chroma(
            collection_name=uuid.uuid4().hex, 
            embedding_function=FakeEmbeddings()
        ), 
        search_type="similarity"
    )
    retriever.add_texts(texts=texts)
    output = retriever.get_relevant_documents("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_chroma_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    retriever = ChromaRetriever(
        store=Chroma(
            collection_name=uuid.uuid4().hex, 
            embedding_function=FakeEmbeddings()
        ), 
        search_type="similarity"
    )
    retriever.add_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = retriever.get_relevant_documents("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_chroma_search_filter() -> None:
    """Test end to end construction and search with metadata filtering."""
    texts = ["far", "bar", "baz"]
    metadatas = [{"first_letter": "{}".format(text[0])} for text in texts]
    retriever = ChromaRetriever(
        store=Chroma(
            collection_name=uuid.uuid4().hex, 
            embedding_function=FakeEmbeddings()
        ), 
        search_type="similarity"
    )
    retriever.add_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = retriever.get_relevant_documents("far", k=1, filter={"first_letter": "f"})
    assert output == [Document(page_content="far", metadata={"first_letter": "f"})]
    output = retriever.get_relevant_documents("far", k=1, filter={"first_letter": "b"})
    assert output == [Document(page_content="bar", metadata={"first_letter": "b"})]


def test_chroma_mmr() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    retriever = ChromaRetriever(
        store=Chroma(
            collection_name=uuid.uuid4().hex, 
            embedding_function=FakeEmbeddings()
        ), 
        search_type="mmr"
    )
    retriever.add_texts(texts=texts)
    output = retriever.get_relevant_documents("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_chroma_with_persistence(temp_dir) -> None:
    """Test end to end construction and search, with persistence."""
    collection_name = "test_collection"

    embedding = FakeEmbeddings()

    texts = ["foo", "bar", "baz"]
    retriever = ChromaRetriever(
        store=Chroma(
            collection_name=uuid.uuid4().hex, 
            embedding_function=embedding
        ), 
        search_type="similarity"
    )
    retriever.add_texts(
        texts=texts,
        embedding=embedding,
        persist_directory=temp_dir,
    )

    output = retriever.get_relevant_documents("foo", k=1)
    assert output == [Document(page_content="foo")]

    retriever.store.persist()

    # Get a new VectorStore from the persisted directory
    retriever = ChromaRetriever(
        store=Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=temp_dir
        ), 
        search_type="similarity"
    )
    
    output = retriever.get_relevant_documents("foo", k=1)

    # Clean up
    retriever.store.delete_collection()


def test_chroma_update_document() -> None:
    """Test the update_document function in the Chroma class."""
    # Make a consistent embedding
    embedding = ConsistentFakeEmbeddings()

    # Initial document content and id
    initial_content = "foo"
    document_id = "doc1"

    # Create an instance of Document with initial content and metadata
    original_doc = Document(page_content=initial_content, metadata={"page": "0"})

    # Initialize a Chroma instance with the original document
    retriever = ChromaRetriever(
        store=Chroma(
            collection_name=uuid.uuid4().hex, 
            embedding_function=embedding
        ), 
        search_type="similarity"
    )
    retriever.add_documents(
        documents=[original_doc],
        embedding=embedding,
        ids=[document_id],
    )
    old_embedding = retriever.store._collection.peek()["embeddings"][
        retriever.store._collection.peek()["ids"].index(document_id)
    ]

    # Define updated content for the document
    updated_content = "updated foo"

    # Create a new Document instance with the updated content and the same id
    updated_doc = Document(page_content=updated_content, metadata={"page": "0"})

    # Update the document in the Chroma instance
    retriever.update_document(
        document_id=document_id, 
        document=updated_doc
    )

    # Perform a similarity search with the updated content
    output = retriever.get_relevant_documents(updated_content, k=1)

    # Assert that the updated document is returned by the search
    assert output == [Document(page_content=updated_content, metadata={"page": "0"})]

    # Assert that the new embedding is correct
    new_embedding = retriever.store._collection.peek()["embeddings"][
        retriever.store._collection.peek()["ids"].index(document_id)
    ]

    assert new_embedding == embedding.embed_documents([updated_content])[0]
    assert new_embedding != old_embedding


def test_chroma_with_relevance_score_custom_normalization_fn() -> None:
    """Test searching with relevance score and custom normalization function."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]

    retriever = ChromaRetriever(
        store=Chroma(
            collection_name=uuid.uuid4().hex,
            embedding_function=FakeEmbeddings(),
            collection_metadata={"hnsw:space": "l2"},
            relevance_score_fn=lambda d: d * 0
        ),
        search_type="similarity"
    )
    retriever.add_texts(
        texts=texts,
        metadatas=metadatas
    )
    output = retriever.get_relevant_documents("foo", k=3)
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"})),
        (Document(page_content="bar", metadata={"page": "1"})),
        (Document(page_content="baz", metadata={"page": "2"})),
    ]


def test_chroma_add_documents_no_metadata() -> None:
    retriever = ChromaRetriever(
        store=Chroma(
            collection_name=uuid.uuid4().hex, 
            embedding_function=FakeEmbeddings()
        ), 
        search_type="similarity"
    )
    retriever.add_documents(
        documents=[Document(page_content="foo")]
    )


def test_chroma_add_documents_mixed_metadata() -> None:
    retriever = ChromaRetriever(
        store=Chroma(
            collection_name=uuid.uuid4().hex, 
            embedding_function=FakeEmbeddings()
        ), 
        search_type="similarity"
    )
    
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"baz": 1}),
    ]
    ids = ["0", "1"]

    actual_ids = retriever.add_documents(
        documents=docs,
        ids=ids
    )
    assert actual_ids == ids
    search = retriever.get_relevant_documents("foo bar")
    assert sorted(search, key=lambda d: d.page_content) == sorted(
        docs, key=lambda d: d.page_content
    )
