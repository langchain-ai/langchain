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
    """Test that update_document handles documents without explicit metadata.

    Documents created without metadata have metadata={}, which Chroma's
    update API rejects. The fix normalizes empty metadata to None before
    passing it to Chroma.
    """
    embedding = FakeEmbeddings(size=10)
    docsearch = Chroma.from_documents(
        collection_name="test_collection",
        documents=[Document(page_content="original", metadata={"page": "0"})],
        embedding=embedding,
        ids=["doc1"],
    )

    # Update with a Document that has no explicit metadata
    # Document(page_content="...") defaults to metadata={}
    docsearch.update_document(
        document_id="doc1",
        document=Document(page_content="updated"),
    )

    output = docsearch.similarity_search("updated", k=1)
    docsearch.delete_collection()
    assert len(output) == 1
    assert output[0].page_content == "updated"
