"""Test Redis functionality."""
import pytest

from langchain.docstore.document import Document
from langchain.vectorstores.redis import Redis
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


TEST_INDEX_NAME = "test"
TEST_REDIS_URL = "redis://localhost:6379"
TEST_SINGLE_RESULT = [Document(page_content="foo")]
TEST_RESULT = [Document(page_content="foo"), Document(page_content="foo")]


def drop(index_name) -> bool:
    return Redis.drop_index(
        index_name=index_name, delete_documents=True, redis_url=TEST_REDIS_URL
    )


def test_redis() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Redis.from_texts(
        texts, FakeEmbeddings(), redis_url=TEST_REDIS_URL
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == TEST_SINGLE_RESULT
    assert drop(docsearch.index_name)


def test_redis_new_vector() -> None:
    """Test adding a new document"""
    texts = ["foo", "bar", "baz"]
    docsearch = Redis.from_texts(
        texts, FakeEmbeddings(), redis_url=TEST_REDIS_URL
    )
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2)
    assert output == TEST_RESULT
    assert drop(docsearch.index_name)


def test_redis_from_existing() -> None:
    """Test adding a new document"""
    texts = ["foo", "bar", "baz"]
    docsearch = Redis.from_texts(
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


def test_redis_drop_index() -> None:
    assert drop(TEST_INDEX_NAME)


@pytest.mark.asyncio
async def test_redis_async_search() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = await Redis.afrom_texts(
        texts, FakeEmbeddings(), index_name=TEST_INDEX_NAME, redis_url=TEST_REDIS_URL
    )
    # Test normal search
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == TEST_SINGLE_RESULT
    # Test search with limit
    output = await docsearch.asimilarity_search_limit_score("foo", k=1, score_threshold=0.1)
    assert output == TEST_SINGLE_RESULT


@pytest.mark.asyncio
async def test_redis_async_from_existing() -> None:
    """Test creating from an existing index."""
    # Test creating from an existing
    docsearch = await Redis.afrom_existing_index(
        FakeEmbeddings(), index_name=TEST_INDEX_NAME, redis_url=TEST_REDIS_URL
    )
    assert docsearch.index_name == TEST_INDEX_NAME
    # Test normal search
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == TEST_SINGLE_RESULT
    # Test search with limit
    output = await docsearch.asimilarity_search_limit_score("foo", k=1, score_threshold=0.1)
    assert output == TEST_SINGLE_RESULT


@pytest.mark.asyncio
async def test_redis_async_from_non_existing() -> None:
    """Test creating from a non-existing index."""
    try:
        await Redis.afrom_existing_index(
            FakeEmbeddings(), index_name="fake-index", redis_url=TEST_REDIS_URL
        )
    except Exception as e:
        assert isinstance(e, ValueError)


@pytest.mark.asyncio
async def test_redis_mixed_async() -> None:
    """Test mixing async and sync calls"""
    # Test creating from an existing
    docsearch = await Redis.afrom_existing_index(
        FakeEmbeddings(), index_name=TEST_INDEX_NAME, redis_url=TEST_REDIS_URL
    )
    await docsearch.aadd_texts(["foo"])
    docsearch.add_texts(["foo"])
    output = await docsearch.asimilarity_search("foo", k=2)
    assert output == TEST_RESULT
    output = docsearch.similarity_search("foo", k=2)
    assert output == TEST_RESULT

@pytest.mark.asyncio
async def test_redis_more_mixed_async() -> None:
    """More testing of mixing async and sync calls"""
    # Drop index
    assert Redis.drop_index(
        index_name=TEST_INDEX_NAME, delete_documents=True, redis_url=TEST_REDIS_URL
    )
    # Start over
    texts = ["foo", "bar", "baz"]
    docsearch = await Redis.afrom_texts(
        texts, FakeEmbeddings(), index_name=TEST_INDEX_NAME, redis_url=TEST_REDIS_URL
    )
    docsearch.add_texts(["foo"])
    # Test search
    output = await docsearch.asimilarity_search("foo", k=2)
    assert output == TEST_RESULT

@pytest.mark.asyncio
async def test_redis_retriever() -> None:
    """Test retriever"""
    docsearch = Redis.from_existing_index(
        FakeEmbeddings(), index_name=TEST_INDEX_NAME, redis_url=TEST_REDIS_URL
    )
    retriever = docsearch.as_retriever(search_type="similarity_limit", k=2, score_threshold=0.2)
    output = retriever.get_relevant_documents("foo")
    assert output == TEST_RESULT
    # Drop index
    assert Redis.drop_index(
        index_name=TEST_INDEX_NAME, delete_documents=True, redis_url=TEST_REDIS_URL
    )