"""Test Upstash Vector functionality."""

import os

import pytest
import requests
from langchain_core.documents import Document

from langchain_community.vectorstores.upstash import UpstashVectorStore
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
)


def is_api_accessible(url: str) -> bool:
    try:
        response = requests.get(url)
        return response.status_code == 200
    except Exception:
        return False


def test_upstash() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = UpstashVectorStore.from_texts(texts=texts, embedding=FakeEmbeddings())
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


async def test_upstash_async() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = UpstashVectorStore.from_texts(texts=texts, embedding=FakeEmbeddings())
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
    output = docsearch.similarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


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
    embedded_query = embeddings.embed_query("foo")
    output = docsearch.similarity_search_by_vector_with_score(
        embedding=embedded_query, k=1
    )
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_upstash_mmr() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = UpstashVectorStore.from_texts(texts=texts, embedding=FakeEmbeddings())
    output = docsearch.max_marginal_relevance_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_upstash_mmr_by_vector() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    embeddings = FakeEmbeddings()
    docsearch = UpstashVectorStore.from_texts(texts=texts, embedding=embeddings)
    embedded_query = embeddings.embed_query("foo")
    output = docsearch.max_marginal_relevance_search_by_vector(embedded_query, k=1)
    assert output == [Document(page_content="foo")]


def test_upstash_with_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.8),
        (Document(page_content="baz", metadata={"page": "2"}), 0.5),
    ]


def test_init_from_client() -> None:
    from upstash_vector import AsyncIndex

    index = AsyncIndex.from_env()

    store = UpstashVectorStore(index=index)

    assert store.info() is not None


@pytest.skip()
def test_init_from_credentials() -> None:
    store = UpstashVectorStore(
        index_url=os.environ["UPSTASH_VECTOR_URL"],
        index_token=os.environ["UPSTASH_VECTOR_TOKEN"],
    )

    assert store.info() is not None


def test_upstash_add_documents_no_metadata() -> None:
    db = UpstashVectorStore(embedding=FakeEmbeddings())
    db.add_documents([Document(page_content="foo")])


def test_upstash_add_documents_mixed_metadata() -> None:
    db = UpstashVectorStore(embedding=FakeEmbeddings())
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"baz": 1}),
    ]
    ids = ["0", "1"]
    actual_ids = db.add_documents(docs, ids=ids)
    assert actual_ids == ids
    search = db.similarity_search("foo bar")
    assert sorted(search, key=lambda d: d.page_content) == sorted(
        docs, key=lambda d: d.page_content
    )
