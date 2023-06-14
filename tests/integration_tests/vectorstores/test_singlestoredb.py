"""Test SingleStoreDB functionality."""
from typing import List

import numpy as np
import pytest

from langchain.docstore.document import Document
from langchain.vectorstores.singlestoredb import SingleStoreDB
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

TEST_SINGLESTOREDB_URL = "root:pass@localhost:3306/db"
TEST_SINGLE_RESULT = [Document(page_content="foo")]
TEST_SINGLE_WITH_METADATA_RESULT = [Document(page_content="foo", metadata={"a": "b"})]
TEST_RESULT = [Document(page_content="foo"), Document(page_content="foo")]

try:
    import singlestoredb as s2

    singlestoredb_installed = True
except ImportError:
    singlestoredb_installed = False


def drop(table_name: str) -> None:
    with s2.connect(TEST_SINGLESTOREDB_URL) as conn:
        conn.autocommit(True)
        with conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name};")


class NormilizedFakeEmbeddings(FakeEmbeddings):
    """Fake embeddings with normalization. For testing purposes."""

    def normalize(self, vector: List[float]) -> List[float]:
        """Normalize vector."""
        return [float(v / np.linalg.norm(vector)) for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.normalize(v) for v in super().embed_documents(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.normalize(super().embed_query(text))


@pytest.fixture
def texts() -> List[str]:
    return ["foo", "bar", "baz"]


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb(texts: List[str]) -> None:
    """Test end to end construction and search."""
    table_name = "test_singlestoredb"
    drop(table_name)
    docsearch = SingleStoreDB.from_texts(
        texts,
        NormilizedFakeEmbeddings(),
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == TEST_SINGLE_RESULT
    drop(table_name)


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_new_vector(texts: List[str]) -> None:
    """Test adding a new document"""
    table_name = "test_singlestoredb_new_vector"
    drop(table_name)
    docsearch = SingleStoreDB.from_texts(
        texts,
        NormilizedFakeEmbeddings(),
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2)
    assert output == TEST_RESULT
    drop(table_name)


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_from_existing(texts: List[str]) -> None:
    """Test adding a new document"""
    table_name = "test_singlestoredb_from_existing"
    drop(table_name)
    SingleStoreDB.from_texts(
        texts,
        NormilizedFakeEmbeddings(),
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    # Test creating from an existing
    docsearch2 = SingleStoreDB(
        NormilizedFakeEmbeddings(),
        table_name="test_singlestoredb_from_existing",
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch2.similarity_search("foo", k=1)
    assert output == TEST_SINGLE_RESULT
    drop(table_name)


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_from_documents(texts: List[str]) -> None:
    """Test from_documents constructor."""
    table_name = "test_singlestoredb_from_documents"
    drop(table_name)
    docs = [Document(page_content=t, metadata={"a": "b"}) for t in texts]
    docsearch = SingleStoreDB.from_documents(
        docs,
        NormilizedFakeEmbeddings(),
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == TEST_SINGLE_WITH_METADATA_RESULT
    drop(table_name)


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_add_texts_to_existing(texts: List[str]) -> None:
    """Test adding a new document"""
    table_name = "test_singlestoredb_add_texts_to_existing"
    drop(table_name)
    # Test creating from an existing
    SingleStoreDB.from_texts(
        texts,
        NormilizedFakeEmbeddings(),
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    docsearch = SingleStoreDB(
        NormilizedFakeEmbeddings(),
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2)
    assert output == TEST_RESULT
    drop(table_name)
