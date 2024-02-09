"""Test Upstash Vector functionality."""

import os
from time import sleep

import pytest
import requests
from langchain_core.documents import Document

from upstash_vector import Index, AsyncIndex
from langchain_community.vectorstores.upstash import UpstashVectorStore
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
)


@pytest.fixture(scope='function', autouse=True)
def fixture():
    index = Index.from_env()
    index.reset()
    wait_for_indexing()


def wait_for_indexing():
    # Wait for indexing to complete
    sleep(1)


def is_api_accessible(url: str) -> bool:
    try:
        response = requests.get(url)
        return response.status_code == 200
    except Exception:
        return False


def test_upstash() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = UpstashVectorStore.from_texts(
        texts=texts, embedding=FakeEmbeddings())
    wait_for_indexing()
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.asyncio
async def test_upstash_async() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = UpstashVectorStore.from_texts(
        texts=texts, embedding=FakeEmbeddings())
    wait_for_indexing()
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_upstash_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    wait_for_indexing()
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_upstash_with_metadatas_with_scores() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    wait_for_indexing()
    output = docsearch.similarity_search_with_score("foo", k=1)
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0)]


def test_upstash_with_metadatas_with_scores_using_vector() -> None:
    """Test end to end construction and scored search, using embedding vector."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    embeddings = FakeEmbeddings()

    docsearch = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )
    wait_for_indexing()
    embedded_query = embeddings.embed_query("foo")
    output = docsearch.similarity_search_by_vector_with_score(
        embedding=embedded_query, k=1
    )
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0)]


def test_upstash_mmr() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = UpstashVectorStore.from_texts(
        texts=texts, embedding=FakeEmbeddings())
    wait_for_indexing()
    output = docsearch.max_marginal_relevance_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_upstash_mmr_by_vector() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    embeddings = FakeEmbeddings()
    docsearch = UpstashVectorStore.from_texts(
        texts=texts, embedding=embeddings)
    wait_for_indexing()
    embedded_query = embeddings.embed_query("foo")
    output = docsearch.max_marginal_relevance_search_by_vector(
        embedded_query, k=1)
    assert output == [Document(page_content="foo")]


def test_init_from_index() -> None:
    index = Index.from_env()

    store = UpstashVectorStore(index=index)

    assert store.info() is not None


def test_init_from_async_index() -> None:
    index = AsyncIndex.from_env()

    store = UpstashVectorStore(async_index=index)

    assert store.ainfo() is not None


@pytest.mark.asyncio
async def test_init_from_credentials() -> None:
    store = UpstashVectorStore(
        index_url=os.environ["UPSTASH_VECTOR_REST_URL"],
        index_token=os.environ["UPSTASH_VECTOR_REST_TOKEN"],
    )

    assert await store.ainfo() is not None


def test_upstash_add_documents_no_metadata() -> None:
    db = UpstashVectorStore(embedding=FakeEmbeddings())
    db.add_documents([Document(page_content="foo")])
    wait_for_indexing()

    search = db.similarity_search("foo")
    assert search == [Document(page_content="foo")]


def test_upstash_add_documents_mixed_metadata() -> None:
    db = UpstashVectorStore(embedding=FakeEmbeddings())
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"baz": 1}),
    ]
    print("og docs", docs)
    ids = ["0", "1"]
    actual_ids = db.add_documents(docs, ids=ids)
    wait_for_indexing()
    assert actual_ids == ids
    search = db.similarity_search("foo bar")
    assert sorted(search, key=lambda d: d.page_content) == sorted(
        docs, key=lambda d: d.page_content
    )
