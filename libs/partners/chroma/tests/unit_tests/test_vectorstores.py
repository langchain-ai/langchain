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
    """Test that update_document works when Document has no metadata."""
    from langchain_core.documents import Document

    embedding = FakeEmbeddings(size=10)

    # Create collection and add a document with metadata
    docsearch = Chroma.from_texts(
        collection_name="test_update_no_metadata",
        texts=["original content"],
        embedding=embedding,
        ids=["doc_1"],
    )

    # Update with a Document that has no metadata (the bug case)
    docsearch.update_document(
        document_id="doc_1",
        document=Document(page_content="updated content"),
    )

    # Verify the update succeeded
    output = docsearch.similarity_search("updated content", k=1)
    assert len(output) == 1
    assert output[0].page_content == "updated content"
    assert output[0].metadata == {}

    docsearch.delete_collection()
