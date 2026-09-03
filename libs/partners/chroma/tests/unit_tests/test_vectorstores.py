from langchain_core.documents import Document
from langchain_core.embeddings.fake import (
    FakeEmbeddings,
)

from langchain_chroma.vectorstores import Chroma


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    texts = ["foo", "bar", "baz"]
    Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
    )


def test_similarity_search() -> None:
    """Test similarity search by Chroma."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1)
    docsearch.delete_collection()
    assert len(output) == 1


def test_update_document_without_metadata() -> None:
    """Test updating a document that carries no metadata.

    Chroma rejects empty metadata dicts on update, so a document whose
    metadata defaults to `{}` must be updated without a `metadatas`
    argument instead of raising `ValueError`.
    """
    docsearch = Chroma.from_texts(
        collection_name="test_update_without_metadata",
        texts=["foo"],
        embedding=FakeEmbeddings(size=10),
        ids=["doc_1"],
    )
    docsearch.update_document(
        document_id="doc_1",
        document=Document(page_content="updated"),
    )
    result = docsearch.get(ids=["doc_1"])
    docsearch.delete_collection()
    assert result["documents"] == ["updated"]
    # Chroma stores no metadata for the document; nothing was attached.
    assert not result["metadatas"][0]


def test_update_documents_with_mixed_metadata() -> None:
    """Test updating documents with and without metadata in one call."""
    docsearch = Chroma.from_texts(
        collection_name="test_update_mixed_metadata",
        texts=["foo", "bar"],
        embedding=FakeEmbeddings(size=10),
        ids=["doc_1", "doc_2"],
    )
    docsearch.update_documents(
        ids=["doc_1", "doc_2"],
        documents=[
            Document(page_content="updated bare"),
            Document(page_content="updated tagged", metadata={"page": "1"}),
        ],
    )
    result = docsearch.get(ids=["doc_1", "doc_2"])
    docsearch.delete_collection()
    assert sorted(result["documents"]) == ["updated bare", "updated tagged"]
    metadata_by_id = dict(zip(result["ids"], result["metadatas"], strict=True))
    assert not metadata_by_id["doc_1"]
    assert metadata_by_id["doc_2"] == {"page": "1"}
