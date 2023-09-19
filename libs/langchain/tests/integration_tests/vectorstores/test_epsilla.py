"""Test Epsilla functionality."""
from langchain.vectorstores import Epsilla
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)


def _test_from_texts() -> Epsilla:
    from pyepsilla import vectordb

    embeddings = FakeEmbeddings()
    client = vectordb.Client()
    return Epsilla.from_texts(fake_texts, embeddings, client)


def test_epsilla() -> None:
    instance = _test_from_texts()
    search = instance.similarity_search(query="bar", k=1)
    result_texts = [doc.page_content for doc in search]
    assert "bar" in result_texts


def test_epsilla_add_texts() -> None:
    from pyepsilla import vectordb

    embeddings = FakeEmbeddings()
    client = vectordb.Client()
    db = Epsilla(client, embeddings)
    db.add_texts(fake_texts)
    search = db.similarity_search(query="foo", k=1)
    result_texts = [doc.page_content for doc in search]
    assert "foo" in result_texts
