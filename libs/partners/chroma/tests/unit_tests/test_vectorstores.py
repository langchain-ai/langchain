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


def test_update_document_empty_metadata() -> None:
    """Test update_document with a document having empty or missing metadata."""
    from langchain_core.documents import Document

    texts = ["foo"]
    docsearch = Chroma.from_texts(
        collection_name="test_collection_update",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
    )
    doc = Document(page_content="bar")
    ids = docsearch.add_documents([doc], ids=["test_doc_1"])
    assert ids == ["test_doc_1"]

    updated_doc = Document(page_content="baz")
    docsearch.update_documents(ids=["test_doc_1"], documents=[updated_doc])

    results = docsearch.get(ids=["test_doc_1"])
    docsearch.delete_collection()

    assert results["ids"] == ["test_doc_1"]
    assert results["documents"] == ["baz"]
    assert results["metadatas"] in [[{}], [None], None]


