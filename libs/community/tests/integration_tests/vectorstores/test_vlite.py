"""Test VLite functionality."""

from langchain_core.documents import Document

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import VLite


def test_vlite() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = VLite.from_texts(texts=texts, embedding=FakeEmbeddings())  # type: ignore[call-arg]
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_vlite_with_metadatas() -> None:
    """Test end to end construction and search with metadata."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = VLite.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),  # type: ignore[call-arg]
        metadatas=metadatas,  # type: ignore[call-arg]
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_vlite_with_metadatas_with_scores() -> None:
    """Test end to end construction and search with metadata and scores."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = VLite.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),  # type: ignore[call-arg]
        metadatas=metadatas,  # type: ignore[call-arg]
    )
    output = docsearch.similarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_vlite_update_document() -> None:
    """Test updating a document."""
    texts = ["foo", "bar", "baz"]
    docsearch = VLite.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),  # type: ignore[call-arg]
        ids=["1", "2", "3"],  # type: ignore[call-arg]
    )
    docsearch.update_document("1", Document(page_content="updated_foo"))
    output = docsearch.similarity_search("updated_foo", k=1)
    assert output == [Document(page_content="updated_foo")]


def test_vlite_delete_document() -> None:
    """Test deleting a document."""
    texts = ["foo", "bar", "baz"]
    docsearch = VLite.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),  # type: ignore[call-arg]
        ids=["1", "2", "3"],  # type: ignore[call-arg]
    )
    docsearch.delete(["1"])
    output = docsearch.similarity_search("foo", k=3)
    assert Document(page_content="foo") not in output


def test_vlite_get_documents() -> None:
    """Test getting documents by IDs."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = VLite.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),  # type: ignore[call-arg]
        metadatas=metadatas,
        ids=["1", "2", "3"],
    )
    output = docsearch.get(ids=["1", "3"])
    assert output == [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="baz", metadata={"page": "2"}),
    ]


def test_vlite_from_existing_index() -> None:
    """Test loading from an existing index."""
    texts = ["foo", "bar", "baz"]
    VLite.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),  # type: ignore[call-arg]
        collection="test_collection",  # type: ignore[call-arg]
    )
    new_docsearch = VLite.from_existing_index(
        collection="test_collection",
        embedding=FakeEmbeddings(),  # type: ignore[call-arg]
    )
    output = new_docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]
