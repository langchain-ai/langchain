"""Test TimescaleVector functionality."""
import os
from datetime import datetime, timedelta
from typing import List

import pytest

from langchain.docstore.document import Document
from langchain.vectorstores.timescalevector import TimescaleVector
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

SERVICE_URL = TimescaleVector.service_url_from_db_params(
    host=os.environ.get("TEST_TIMESCALE_HOST", "localhost"),
    port=int(os.environ.get("TEST_TIMESCALE_PORT", "5432")),
    database=os.environ.get("TEST_TIMESCALE_DATABASE", "postgres"),
    user=os.environ.get("TEST_TIMESCALE_USER", "postgres"),
    password=os.environ.get("TEST_TIMESCALE_PASSWORD", "postgres"),
)


ADA_TOKEN_COUNT = 1536


class FakeEmbeddingsWithAdaDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(i)] for i in range(len(texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(0.0)]


def test_timescalevector() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = TimescaleVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_timescalevector_from_documents() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docs = [Document(page_content=t, metadata={"a": "b"}) for t in texts]
    docsearch = TimescaleVector.from_documents(
        documents=docs,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"a": "b"})]


@pytest.mark.asyncio
async def test_timescalevector_afrom_documents() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docs = [Document(page_content=t, metadata={"a": "b"}) for t in texts]
    docsearch = await TimescaleVector.afrom_documents(
        documents=docs,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"a": "b"})]


def test_timescalevector_embeddings() -> None:
    """Test end to end construction with embeddings and search."""
    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = TimescaleVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.asyncio
async def test_timescalevector_aembeddings() -> None:
    """Test end to end construction with embeddings and search."""
    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = await TimescaleVector.afrom_embeddings(
        text_embeddings=text_embedding_pairs,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_timescalevector_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = TimescaleVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_timescalevector_with_metadatas_with_scores() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = TimescaleVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


@pytest.mark.asyncio
async def test_timescalevector_awith_metadatas_with_scores() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await TimescaleVector.afrom_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )
    output = await docsearch.asimilarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_timescalevector_with_filter_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = TimescaleVector.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "0"})
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_timescalevector_with_filter_distant_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = TimescaleVector.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "2"})
    assert output == [
        (Document(page_content="baz", metadata={"page": "2"}), 0.0013003906671379406)
    ]


def test_timescalevector_with_filter_no_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = TimescaleVector.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "5"})
    assert output == []


def test_timescalevector_with_filter_in_set() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = TimescaleVector.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score(
        "foo", k=2, filter=[{"page": "0"}, {"page": "2"}]
    )
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 0.0),
        (Document(page_content="baz", metadata={"page": "2"}), 0.0013003906671379406),
    ]


def test_timescalevector_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = TimescaleVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )

    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.9996744261675065),
        (Document(page_content="baz", metadata={"page": "2"}), 0.9986996093328621),
    ]


@pytest.mark.asyncio
async def test_timescalevector_relevance_score_async() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await TimescaleVector.afrom_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )

    output = await docsearch.asimilarity_search_with_relevance_scores("foo", k=3)
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.9996744261675065),
        (Document(page_content="baz", metadata={"page": "2"}), 0.9986996093328621),
    ]


def test_timescalevector_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = TimescaleVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.999},
    )
    output = retriever.get_relevant_documents("summer")
    assert output == [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="bar", metadata={"page": "1"}),
    ]


def test_timescalevector_retriever_search_threshold_custom_normalization_fn() -> None:
    """Test searching with threshold and custom normalization function"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = TimescaleVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        service_url=SERVICE_URL,
        pre_delete_collection=True,
        relevance_score_fn=lambda d: d * 0,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    output = retriever.get_relevant_documents("foo")
    assert output == []


def test_timescalevector_delete() -> None:
    """Test deleting functionality."""
    texts = ["bar", "baz"]
    docs = [Document(page_content=t, metadata={"a": "b"}) for t in texts]
    docsearch = TimescaleVector.from_documents(
        documents=docs,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )
    texts = ["foo"]
    meta = [{"b": "c"}]
    ids = docsearch.add_texts(texts, meta)

    output = docsearch.similarity_search("bar", k=10)
    assert len(output) == 3
    docsearch.delete(ids)

    output = docsearch.similarity_search("bar", k=10)
    assert len(output) == 2

    docsearch.delete_by_metadata({"a": "b"})
    output = docsearch.similarity_search("bar", k=10)
    assert len(output) == 0


def test_timescalevector_with_index() -> None:
    """Test deleting functionality."""
    texts = ["bar", "baz"]
    docs = [Document(page_content=t, metadata={"a": "b"}) for t in texts]
    docsearch = TimescaleVector.from_documents(
        documents=docs,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        service_url=SERVICE_URL,
        pre_delete_collection=True,
    )
    texts = ["foo"]
    meta = [{"b": "c"}]
    docsearch.add_texts(texts, meta)

    docsearch.create_index()

    output = docsearch.similarity_search("bar", k=10)
    assert len(output) == 3

    docsearch.drop_index()
    docsearch.create_index(
        index_type=TimescaleVector.IndexType.TIMESCALE_VECTOR,
        max_alpha=1.0,
        num_neighbors=50,
    )

    docsearch.drop_index()
    docsearch.create_index("tsv", max_alpha=1.0, num_neighbors=50)

    docsearch.drop_index()
    docsearch.create_index("ivfflat", num_lists=20, num_records=1000)

    docsearch.drop_index()
    docsearch.create_index("hnsw", m=16, ef_construction=64)


def test_timescalevector_time_partitioning() -> None:
    """Test deleting functionality."""
    from timescale_vector import client

    texts = ["bar", "baz"]
    docs = [Document(page_content=t, metadata={"a": "b"}) for t in texts]
    docsearch = TimescaleVector.from_documents(
        documents=docs,
        collection_name="test_collection_time_partitioning",
        embedding=FakeEmbeddingsWithAdaDimension(),
        service_url=SERVICE_URL,
        pre_delete_collection=True,
        time_partition_interval=timedelta(hours=1),
    )
    texts = ["foo"]
    meta = [{"b": "c"}]

    ids = [client.uuid_from_time(datetime.now() - timedelta(hours=3))]
    docsearch.add_texts(texts, meta, ids)

    output = docsearch.similarity_search("bar", k=10)
    assert len(output) == 3

    output = docsearch.similarity_search(
        "bar", k=10, start_date=datetime.now() - timedelta(hours=1)
    )
    assert len(output) == 2

    output = docsearch.similarity_search(
        "bar", k=10, end_date=datetime.now() - timedelta(hours=1)
    )
    assert len(output) == 1

    output = docsearch.similarity_search(
        "bar", k=10, start_date=datetime.now() - timedelta(minutes=200)
    )
    assert len(output) == 3

    output = docsearch.similarity_search(
        "bar",
        k=10,
        start_date=datetime.now() - timedelta(minutes=200),
        time_delta=timedelta(hours=1),
    )
    assert len(output) == 1
