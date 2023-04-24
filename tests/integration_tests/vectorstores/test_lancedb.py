"""Test LanceDB functionality."""

from langchain.docstore.document import Document
from langchain.vectorstores.lancedb import LanceDB
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

import lancedb


def test_lancedb() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    db = lancedb.connect("./langchain-test-lancedb")
    table = db.create_table("tbl")
    embedder = FakeEmbeddings()
    docsearch = LanceDB.from_texts(
        connection=table,
        embedding_function=embedder.embed_documents,
        texts=texts,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_lancedb_new_vector() -> None:
    """Test adding a new document"""
    texts = ["foo", "bar", "baz"]
    db = lancedb.connect("./langchain-test-lancedb")
    table = db.create_table("tbl")
    embedder = FakeEmbeddings()
    docsearch = LanceDB.from_texts(
        connection=table,
        embedding_function=embedder.embed_documents,
        texts=texts,
    )
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2)
    assert output == [Document(page_content="foo"), Document(page_content="foo")]
