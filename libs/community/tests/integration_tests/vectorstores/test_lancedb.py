import pytest

from langchain_community.vectorstores import LanceDB
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


@pytest.mark.requires("lancedb")
def test_lancedb_with_connection() -> None:
    import lancedb

    embeddings = FakeEmbeddings()
    db = lancedb.connect("/tmp/lancedb_connection")
    texts = ["text 1", "text 2", "item 3"]
    store = LanceDB(connection=db, embedding=embeddings)
    store.add_texts(texts)

    result = store.similarity_search("text 1")
    result_texts = [doc.page_content for doc in result]
    assert "text 1" in result_texts

    store.delete(filter="text = 'text 1'")
    assert store.get_table().count_rows() == 2


@pytest.mark.requires("lancedb")
def test_lancedb_without_connection() -> None:
    embeddings = FakeEmbeddings()
    texts = ["text 1", "text 2", "item 3"]

    store = LanceDB(embedding=embeddings)
    store.add_texts(texts)
    result = store.similarity_search("text 1")
    result_texts = [doc.page_content for doc in result]
    assert "text 1" in result_texts


@pytest.mark.requires("lancedb")
def test_lancedb_add_texts() -> None:
    embeddings = FakeEmbeddings()

    store = LanceDB(embedding=embeddings)
    store.add_texts(["text 2"])
    result = store.similarity_search("text 2")
    result_texts = [doc.page_content for doc in result]
    assert "text 2" in result_texts
