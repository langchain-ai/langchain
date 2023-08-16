from langchain.docstore.document import Document
from langchain.vectorstores.fake import FakeVectorStore


def test_add_texts() -> None:
    vector_store = FakeVectorStore()
    doc1 = Document(page_content="doc1")
    doc2 = Document(page_content="doc2")
    doc3 = Document(page_content="doc3")
    doc4 = Document(page_content="doc4")
    doc5 = Document(page_content="doc5")

    vector_store = FakeVectorStore()
    vector_store.add_texts(texts=["doc1", "doc2", "doc3"])
    result = vector_store.similarity_search("something", k=2)
    assert result == [doc1, doc2]

    vector_store.add_texts(texts=["doc4", "doc5"])
    result = vector_store.similarity_search("something", k=20)
    assert result == [doc1, doc2, doc3, doc4, doc5] * 4


def test_from_texts() -> None:
    vector_store = FakeVectorStore.from_texts(
        texts=["doc1", "doc2"], metadatas=[{"key": "value1"}, {"key": "value2"}]
    )
    doc1 = Document(page_content="doc1", metadata={"key": "value1"})
    doc2 = Document(page_content="doc2", metadata={"key": "value2"})
    doc3 = Document(page_content="doc3", metadata={"key": "value3"})

    result = vector_store.similarity_search("something", k=3)
    assert result == [doc1, doc2, doc1]

    vector_store.add_texts(texts=["doc3"], metadatas=[{"key": "value3"}])

    result = vector_store.similarity_search("something", k=4)
    assert result == [doc1, doc2, doc3, doc1]
