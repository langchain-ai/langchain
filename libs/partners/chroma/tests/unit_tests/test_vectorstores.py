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


def test_update_document_without_metadata() -> None:
    """Regression test for #37452.

    update_document (and update_documents) must not raise when the Document
    being updated has no metadata (i.e. metadata == {}).  Previously Chroma
    rejected the empty dict with:

        ValueError: Expected metadata to be a non-empty dict, got 0 metadata
        attributes in update.
    """
    store = Chroma.from_documents(
        collection_name="test_update_no_meta",
        documents=[Document(page_content="original content")],
        ids=["doc-1"],
        embedding=FakeEmbeddings(size=10),
    )
    try:
        # Should NOT raise even though Document has no metadata
        store.update_document(
            document_id="doc-1",
            document=Document(page_content="updated content"),
        )
        results = store.similarity_search("updated content", k=1)
        assert len(results) == 1
        assert results[0].page_content == "updated content"
    finally:
        store.delete_collection()


def test_update_documents_mixed_metadata() -> None:
    """update_documents with a mix of empty and non-empty metadata must not raise."""
    store = Chroma.from_documents(
        collection_name="test_update_mixed_meta",
        documents=[
            Document(page_content="doc one", metadata={"key": "val"}),
            Document(page_content="doc two"),
        ],
        ids=["doc-1", "doc-2"],
        embedding=FakeEmbeddings(size=10),
    )
    try:
        # Must NOT raise — this exercises the code path where some documents
        # carry metadata and others don't.
        store.update_documents(
            ids=["doc-1", "doc-2"],
            documents=[
                Document(page_content="updated one", metadata={"key": "new-val"}),
                Document(page_content="updated two"),  # no metadata
            ],
        )
        # Verify both documents are retrievable (content was updated)
        all_docs = store.get()
        updated_contents = set(all_docs["documents"])
        assert "updated one" in updated_contents
        assert "updated two" in updated_contents
    finally:
        store.delete_collection()


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
