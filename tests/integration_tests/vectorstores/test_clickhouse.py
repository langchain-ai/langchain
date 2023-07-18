"""Test ClickHouse functionality."""
import pytest

from langchain.docstore.document import Document
from langchain.vectorstores import Clickhouse, ClickhouseSettings
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_clickhouse() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    config = ClickhouseSettings()
    config.table = "test_clickhouse"
    docsearch = Clickhouse.from_texts(texts, FakeEmbeddings(), config=config)
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"_dummy": 0})]
    docsearch.drop()


@pytest.mark.asyncio
async def test_clickhouse_async() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    config = ClickhouseSettings()
    config.table = "test_clickhouse_async"
    docsearch = Clickhouse.from_texts(
        texts=texts, embedding=FakeEmbeddings(), config=config
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"_dummy": 0})]
    docsearch.drop()


def test_clickhouse_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    config = ClickhouseSettings()
    config.table = "test_clickhouse_with_metadatas"
    docsearch = Clickhouse.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        config=config,
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]
    docsearch.drop()


def test_clickhouse_with_metadatas_with_relevance_scores() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    config = ClickhouseSettings()
    config.table = "test_clickhouse_with_metadatas_with_relevance_scores"
    docsearch = Clickhouse.from_texts(
        texts=texts, embedding=FakeEmbeddings(), metadatas=metadatas, config=config
    )
    output = docsearch.similarity_search_with_relevance_scores("foo", k=1)
    assert output[0][0] == Document(page_content="foo", metadata={"page": "0"})
    docsearch.drop()


def test_clickhouse_search_filter() -> None:
    """Test end to end construction and search with metadata filtering."""
    texts = ["far", "bar", "baz"]
    metadatas = [{"first_letter": "{}".format(text[0])} for text in texts]
    config = ClickhouseSettings()
    config.table = "test_clickhouse_search_filter"
    docsearch = Clickhouse.from_texts(
        texts=texts, embedding=FakeEmbeddings(), metadatas=metadatas, config=config
    )
    output = docsearch.similarity_search(
        "far", k=1, where_str=f"{docsearch.metadata_column}.first_letter='f'"
    )
    assert output == [Document(page_content="far", metadata={"first_letter": "f"})]
    output = docsearch.similarity_search(
        "bar", k=1, where_str=f"{docsearch.metadata_column}.first_letter='b'"
    )
    assert output == [Document(page_content="bar", metadata={"first_letter": "b"})]
    docsearch.drop()


def test_clickhouse_with_persistence() -> None:
    """Test end to end construction and search, with persistence."""
    config = ClickhouseSettings()
    config.table = "test_clickhouse_with_persistence"
    texts = [
        "foo",
        "bar",
        "baz",
    ]
    docsearch = Clickhouse.from_texts(
        texts=texts, embedding=FakeEmbeddings(), config=config
    )

    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"_dummy": 0})]

    # Get a new VectorStore with same config
    # it will reuse the table spontaneously
    # unless you drop it
    docsearch = Clickhouse(embedding=FakeEmbeddings(), config=config)
    output = docsearch.similarity_search("foo", k=1)

    # Clean up
    docsearch.drop()
