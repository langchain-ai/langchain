"""Test Redis functionality."""
from langchain.docstore.document import Document
from langchain.vectorstores.redis import Redis
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

import pytest

TEST_INDEX_NAME = "test"
TEST_REDIS_URL = "redis://localhost:6379"
TEST_SINGLE_RESULT = [Document(page_content="foo")]
TEST_RESULT = [Document(page_content="foo"), Document(page_content="foo")]
COSINE_SCORE = pytest.approx(0.05, abs=0.002)
IP_SCORE = -8.0
EUCLIDEAN_SCORE = 1.0

def drop(index_name: str) -> bool:
    return Redis.drop_index(
        index_name=index_name, delete_documents=True, redis_url=TEST_REDIS_URL
    )


def test_redis() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Redis.from_texts(texts, FakeEmbeddings(), redis_url=TEST_REDIS_URL)
    output = docsearch.similarity_search("foo", k=1)
    assert output == TEST_SINGLE_RESULT
    assert drop(docsearch.index_name)


def test_redis_new_vector() -> None:
    """Test adding a new document"""
    texts = ["foo", "bar", "baz"]
    docsearch = Redis.from_texts(texts, FakeEmbeddings(), redis_url=TEST_REDIS_URL)
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2)
    assert output == TEST_RESULT
    assert drop(docsearch.index_name)


def test_redis_from_existing() -> None:
    """Test adding a new document"""
    texts = ["foo", "bar", "baz"]
    Redis.from_texts(
        texts, FakeEmbeddings(), index_name=TEST_INDEX_NAME, redis_url=TEST_REDIS_URL
    )
    # Test creating from an existing
    docsearch2 = Redis.from_existing_index(
        FakeEmbeddings(), index_name=TEST_INDEX_NAME, redis_url=TEST_REDIS_URL
    )
    output = docsearch2.similarity_search("foo", k=1)
    assert output == TEST_SINGLE_RESULT


def test_redis_add_texts_to_existing() -> None:
    """Test adding a new document"""
    # Test creating from an existing
    docsearch = Redis.from_existing_index(
        FakeEmbeddings(), index_name=TEST_INDEX_NAME, redis_url=TEST_REDIS_URL
    )
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2)
    assert output == TEST_RESULT
    assert drop(TEST_INDEX_NAME)


class TestRedisDistanceMetrics:
    """Test using different distance metrics for new indices.
    
    For simple texts, the distance metrics should not matter much as they'll
    usually return the same results and orderings. However, the scores the produce
    do differ by metric, so we can use these to assert the intended metric is being
    used.
    """
    texts =  ["foo", "bar", "baz"]
    
    def test_cosine(self) -> None:
        """Test cosine distance."""
        docsearch = Redis.from_texts(
            self.texts, FakeEmbeddings(), redis_url=TEST_REDIS_URL, distance_metric="COSINE"
        )
        output = docsearch.similarity_search_with_score("far", k=2)
        _, score = output[1]
        assert score == COSINE_SCORE
        assert drop(docsearch.index_name)

    def test_l2(self) -> None:
        """Test Flat L2 distance."""
        docsearch = Redis.from_texts(
            self.texts, FakeEmbeddings(), redis_url=TEST_REDIS_URL, distance_metric="L2"
        )
        output = docsearch.similarity_search_with_score("far", k=2)
        _, score = output[1]
        assert score == EUCLIDEAN_SCORE
        assert drop(docsearch.index_name)

    def test_ip(self) -> None:
        """Test inner product distance."""
        docsearch = Redis.from_texts(
            self.texts, FakeEmbeddings(), redis_url=TEST_REDIS_URL, distance_metric="IP"
        )
        output = docsearch.similarity_search_with_score("far", k=2)
        _, score = output[1]
        assert score == IP_SCORE
        assert drop(docsearch.index_name)
