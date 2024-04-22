"""Test Upstash Vector functionality."""

import os
from time import sleep

import pytest
from langchain_core.documents import Document
from upstash_vector import AsyncIndex, Index

from langchain_community.vectorstores.upstash import UpstashVectorStore
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
)


@pytest.fixture(scope="function", autouse=True)
def fixture():
    index = Index.from_env()
    index.reset()
    wait_for_indexing(index)


def wait_for_indexing(store: UpstashVectorStore):
    while store.info().pending_vector_count != 0:
        # Wait for indexing to complete
        sleep(0.5)


def test_upstash_simple_insert() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    store = UpstashVectorStore.from_texts(texts=texts, embedding=FakeEmbeddings())
    wait_for_indexing(store)
    output = store.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.asyncio
async def test_upstash_simple_insert_async() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    store = UpstashVectorStore.from_texts(texts=texts, embedding=FakeEmbeddings())
    wait_for_indexing(store)
    output = await store.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_upstash_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    store = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    wait_for_indexing(store)
    output = store.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


@pytest.mark.asyncio
async def test_upstash_with_metadatas_async() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    store = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    wait_for_indexing(store)
    output = await store.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_upstash_with_metadatas_with_scores() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    store = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    wait_for_indexing(store)
    output = store.similarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]


@pytest.mark.asyncio
async def test_upstash_with_metadatas_with_scores_async() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    store = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    wait_for_indexing(store)
    output = await store.asimilarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]


def test_upstash_with_metadatas_with_scores_using_vector() -> None:
    """Test end to end construction and scored search, using embedding vector."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    embeddings = FakeEmbeddings()

    store = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )
    wait_for_indexing(store)
    embedded_query = embeddings.embed_query("foo")
    output = store.similarity_search_by_vector_with_score(embedding=embedded_query, k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]


@pytest.mark.asyncio
async def test_upstash_with_metadatas_with_scores_using_vector_async() -> None:
    """Test end to end construction and scored search, using embedding vector."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    embeddings = FakeEmbeddings()

    store = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )
    wait_for_indexing(store)
    embedded_query = embeddings.embed_query("foo")
    output = await store.asimilarity_search_by_vector_with_score(
        embedding=embedded_query, k=1
    )
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]


def test_upstash_mmr() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    store = UpstashVectorStore.from_texts(texts=texts, embedding=FakeEmbeddings())
    wait_for_indexing(store)
    output = store.max_marginal_relevance_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.asyncio
async def test_upstash_mmr_async() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    store = UpstashVectorStore.from_texts(texts=texts, embedding=FakeEmbeddings())
    wait_for_indexing(store)
    output = await store.amax_marginal_relevance_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_upstash_mmr_by_vector() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    embeddings = FakeEmbeddings()
    store = UpstashVectorStore.from_texts(texts=texts, embedding=embeddings)
    wait_for_indexing(store)
    embedded_query = embeddings.embed_query("foo")
    output = store.max_marginal_relevance_search_by_vector(embedded_query, k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.asyncio
async def test_upstash_mmr_by_vector_async() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    embeddings = FakeEmbeddings()
    store = UpstashVectorStore.from_texts(texts=texts, embedding=embeddings)
    wait_for_indexing(store)
    embedded_query = embeddings.embed_query("foo")
    output = await store.amax_marginal_relevance_search_by_vector(embedded_query, k=1)
    assert output == [Document(page_content="foo")]


def test_init_from_index() -> None:
    index = Index.from_env()

    store = UpstashVectorStore(index=index)

    assert store.info() is not None


@pytest.mark.asyncio
async def test_init_from_async_index() -> None:
    index = AsyncIndex.from_env()

    store = UpstashVectorStore(async_index=index)

    assert await store.ainfo() is not None


def test_init_from_credentials() -> None:
    store = UpstashVectorStore(
        index_url=os.environ["UPSTASH_VECTOR_REST_URL"],
        index_token=os.environ["UPSTASH_VECTOR_REST_TOKEN"],
    )

    assert store.info() is not None


@pytest.mark.asyncio
async def test_init_from_credentials_async() -> None:
    store = UpstashVectorStore(
        index_url=os.environ["UPSTASH_VECTOR_REST_URL"],
        index_token=os.environ["UPSTASH_VECTOR_REST_TOKEN"],
    )

    assert await store.ainfo() is not None


def test_upstash_add_documents_no_metadata() -> None:
    store = UpstashVectorStore(embedding=FakeEmbeddings())
    store.add_documents([Document(page_content="foo")])
    wait_for_indexing(store)

    search = store.similarity_search("foo")
    assert search == [Document(page_content="foo")]


def test_upstash_add_documents_mixed_metadata() -> None:
    store = UpstashVectorStore(embedding=FakeEmbeddings())
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"baz": 1}),
    ]
    ids = ["0", "1"]
    actual_ids = store.add_documents(docs, ids=ids)
    wait_for_indexing(store)
    assert actual_ids == ids
    search = store.similarity_search("foo bar")
    assert sorted(search, key=lambda d: d.page_content) == sorted(
        docs, key=lambda d: d.page_content
    )

def test_upstash_similarity_search_with_metadata() -> None:
    store = UpstashVectorStore(embedding=FakeEmbeddings())
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"waldo": 1}),
        Document(page_content="baz", metadata={"waldo": 1}),
        Document(page_content="fred", metadata={"waldo": 2}),
    ]
    ids = ["0", "1", "3", "4"]
    store.add_documents(docs, ids=ids)
    wait_for_indexing(store)

    result = store.similarity_search(
        query="foo",
        k=5,
        filter="waldo = 1"
    )

    assert result == [
        Document(page_content="bar", metadata={"waldo": 1}),
        Document(page_content="baz", metadata={"waldo": 1}),
    ]

    result = store.similarity_search_with_score(
        query="foo",
        k=5,
        filter="waldo = 2"
    )

    assert len(result) == 1
    assert result[0][0] == Document(page_content="fred", metadata={"waldo": 2})
    assert round(result[0][1], 2) == 0.85

def test_upstash_similarity_search_by_vector_with_metadata() -> None:
    store = UpstashVectorStore(embedding=FakeEmbeddings())
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"waldo": 1}),
        Document(page_content="baz", metadata={"waldo": 1}),
        Document(page_content="fred", metadata={"waldo": 2}),
    ]
    ids = ["0", "1", "3", "4"]
    store.add_documents(docs, ids=ids)
    wait_for_indexing(store)

    result = store.similarity_search_by_vector_with_score(
        embedding=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        k=5,
        filter="waldo = 1"
    )

    assert len(result) == 2
    assert result[0] == (Document(page_content='bar', metadata={'waldo': 1}), 1.0)
    assert result[1][0] == Document(page_content='baz', metadata={'waldo': 1})
    assert round(result[1][1], 2) == 0.98

def test_upstash_max_marginal_relevance_search_with_metadata() -> None:
    store = UpstashVectorStore(embedding=FakeEmbeddings())
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"waldo": 1}),
        Document(page_content="baz", metadata={"waldo": 1}),
        Document(page_content="fred", metadata={"waldo": 2}),
    ]
    ids = ["0", "1", "3", "4"]
    store.add_documents(docs, ids=ids)
    wait_for_indexing(store)

    result = store.max_marginal_relevance_search(
        query="foo",
        k=3,
        filter="waldo = 1"
    )

    assert result == [
        Document(page_content='bar', metadata={'waldo': 1}),
        Document(page_content='baz', metadata={'waldo': 1})
    ]
