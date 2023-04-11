"""Test Chroma functionality."""
import pytest

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_chroma() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Chroma.from_texts(
        collection_name="test_collection", texts=texts, embedding=FakeEmbeddings()
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.asyncio
async def test_chroma_async() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Chroma.from_texts(
        collection_name="test_collection", texts=texts, embedding=FakeEmbeddings()
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


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
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_chroma_with_metadatas_with_scores() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_chroma_search_filter() -> None:
    """Test end to end construction and search with metadata filtering."""
    texts = ["far", "bar", "baz"]
    metadatas = [{"first_letter": "{}".format(text[0])} for text in texts]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("far", k=1, filter={"first_letter": "f"})
    assert output == [Document(page_content="far", metadata={"first_letter": "f"})]
    output = docsearch.similarity_search("far", k=1, filter={"first_letter": "b"})
    assert output == [Document(page_content="bar", metadata={"first_letter": "b"})]


def test_chroma_search_filter_with_scores() -> None:
    """Test end to end construction and scored search with metadata filtering."""
    texts = ["far", "bar", "baz"]
    metadatas = [{"first_letter": "{}".format(text[0])} for text in texts]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search_with_score(
        "far", k=1, filter={"first_letter": "f"}
    )
    assert output == [
        (Document(page_content="far", metadata={"first_letter": "f"}), 0.0)
    ]
    output = docsearch.similarity_search_with_score(
        "far", k=1, filter={"first_letter": "b"}
    )
    assert output == [
        (Document(page_content="bar", metadata={"first_letter": "b"}), 1.0)
    ]


def test_chroma_with_persistence() -> None:
    """Test end to end construction and search, with persistence."""
    chroma_persist_dir = "./tests/persist_dir"
    collection_name = "test_collection"
    texts = ["foo", "bar", "baz"]
    docsearch = Chroma.from_texts(
        collection_name=collection_name,
        texts=texts,
        embedding=FakeEmbeddings(),
        persist_directory=chroma_persist_dir,
    )

    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    docsearch.persist()

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
