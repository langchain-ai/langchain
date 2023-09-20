"""Test SingleStoreDB functionality."""
from typing import List

import numpy as np
import pytest

from langchain.docstore.document import Document
from langchain.vectorstores.singlestoredb import SingleStoreDB
from langchain.vectorstores.utils import DistanceStrategy
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
def test_singlestoredb_euclidean_distance(texts: List[str]) -> None:
    """Test adding a new document"""
    table_name = "test_singlestoredb_euclidean_distance"
    drop(table_name)
    docsearch = SingleStoreDB.from_texts(
        texts,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
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


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_metadata(texts: List[str]) -> None:
    """Test filtering by metadata"""
    table_name = "test_singlestoredb_filter_metadata"
    drop(table_name)
    docs = [
        Document(page_content=t, metadata={"index": i}) for i, t in enumerate(texts)
    ]
    docsearch = SingleStoreDB.from_documents(
        docs,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search("foo", k=1, filter={"index": 2})
    assert output == [Document(page_content="baz", metadata={"index": 2})]
    drop(table_name)


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_metadata_2(texts: List[str]) -> None:
    """Test filtering by metadata field that is similar for each document"""
    table_name = "test_singlestoredb_filter_metadata_2"
    drop(table_name)
    docs = [
        Document(page_content=t, metadata={"index": i, "category": "budget"})
        for i, t in enumerate(texts)
    ]
    docsearch = SingleStoreDB.from_documents(
        docs,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search("foo", k=1, filter={"category": "budget"})
    assert output == [
        Document(page_content="foo", metadata={"index": 0, "category": "budget"})
    ]
    drop(table_name)


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_metadata_3(texts: List[str]) -> None:
    """Test filtering by two metadata fields"""
    table_name = "test_singlestoredb_filter_metadata_3"
    drop(table_name)
    docs = [
        Document(page_content=t, metadata={"index": i, "category": "budget"})
        for i, t in enumerate(texts)
    ]
    docsearch = SingleStoreDB.from_documents(
        docs,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search(
        "foo", k=1, filter={"category": "budget", "index": 1}
    )
    assert output == [
        Document(page_content="bar", metadata={"index": 1, "category": "budget"})
    ]
    drop(table_name)


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_metadata_4(texts: List[str]) -> None:
    """Test no matches"""
    table_name = "test_singlestoredb_filter_metadata_4"
    drop(table_name)
    docs = [
        Document(page_content=t, metadata={"index": i, "category": "budget"})
        for i, t in enumerate(texts)
    ]
    docsearch = SingleStoreDB.from_documents(
        docs,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search("foo", k=1, filter={"category": "vacation"})
    assert output == []
    drop(table_name)


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_metadata_5(texts: List[str]) -> None:
    """Test complex metadata path"""
    table_name = "test_singlestoredb_filter_metadata_5"
    drop(table_name)
    docs = [
        Document(
            page_content=t,
            metadata={
                "index": i,
                "category": "budget",
                "subfield": {"subfield": {"idx": i, "other_idx": i + 1}},
            },
        )
        for i, t in enumerate(texts)
    ]
    docsearch = SingleStoreDB.from_documents(
        docs,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search(
        "foo", k=1, filter={"category": "budget", "subfield": {"subfield": {"idx": 2}}}
    )
    assert output == [
        Document(
            page_content="baz",
            metadata={
                "index": 2,
                "category": "budget",
                "subfield": {"subfield": {"idx": 2, "other_idx": 3}},
            },
        )
    ]
    drop(table_name)


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_metadata_6(texts: List[str]) -> None:
    """Test filtering by other bool"""
    table_name = "test_singlestoredb_filter_metadata_6"
    drop(table_name)
    docs = [
        Document(
            page_content=t,
            metadata={"index": i, "category": "budget", "is_good": i == 1},
        )
        for i, t in enumerate(texts)
    ]
    docsearch = SingleStoreDB.from_documents(
        docs,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search(
        "foo", k=1, filter={"category": "budget", "is_good": True}
    )
    assert output == [
        Document(
            page_content="bar",
            metadata={"index": 1, "category": "budget", "is_good": True},
        )
    ]
    drop(table_name)


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_filter_metadata_7(texts: List[str]) -> None:
    """Test filtering by float"""
    table_name = "test_singlestoredb_filter_metadata_7"
    drop(table_name)
    docs = [
        Document(
            page_content=t,
            metadata={"index": i, "category": "budget", "score": i + 0.5},
        )
        for i, t in enumerate(texts)
    ]
    docsearch = SingleStoreDB.from_documents(
        docs,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    output = docsearch.similarity_search(
        "bar", k=1, filter={"category": "budget", "score": 2.5}
    )
    assert output == [
        Document(
            page_content="baz",
            metadata={"index": 2, "category": "budget", "score": 2.5},
        )
    ]
    drop(table_name)


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_singlestoredb_as_retriever(texts: List[str]) -> None:
    table_name = "test_singlestoredb_8"
    drop(table_name)
    docsearch = SingleStoreDB.from_texts(
        texts,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        table_name=table_name,
        host=TEST_SINGLESTOREDB_URL,
    )
    retriever = docsearch.as_retriever(search_kwargs={"k": 2})
    output = retriever.get_relevant_documents("foo")
    assert output == [
        Document(
            page_content="foo",
        ),
        Document(
            page_content="bar",
        ),
    ]
    drop(table_name)
