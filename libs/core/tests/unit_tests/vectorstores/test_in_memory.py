from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_standard_tests.integration_tests.vectorstores import (
    AsyncReadWriteTestSuite,
    ReadWriteTestSuite,
)

from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore
from tests.unit_tests.stubs import _any_id_document


class TestInMemoryReadWriteTestSuite(ReadWriteTestSuite):
    @pytest.fixture
    def vectorstore(self) -> InMemoryVectorStore:
        return InMemoryVectorStore(embedding=self.get_embeddings())


class TestAsyncInMemoryReadWriteTestSuite(AsyncReadWriteTestSuite):
    @pytest.fixture
    async def vectorstore(self) -> InMemoryVectorStore:
        return InMemoryVectorStore(embedding=self.get_embeddings())


async def test_inmemory_similarity_search() -> None:
    """Test end to end similarity search."""
    store = await InMemoryVectorStore.afrom_texts(
        ["foo", "bar", "baz"], DeterministicFakeEmbedding(size=3)
    )

    # Check sync version
    output = store.similarity_search("foo", k=1)
    assert output == [_any_id_document(page_content="foo")]

    # Check async version
    output = await store.asimilarity_search("bar", k=2)
    assert output == [
        _any_id_document(page_content="bar"),
        _any_id_document(page_content="foo"),
    ]


async def test_inmemory_similarity_search_with_score() -> None:
    """Test end to end similarity search with score"""
    store = await InMemoryVectorStore.afrom_texts(
        ["foo", "bar", "baz"], DeterministicFakeEmbedding(size=3)
    )

    output = store.similarity_search_with_score("foo", k=1)
    assert output[0][0].page_content == "foo"

    output = await store.asimilarity_search_with_score("bar", k=2)
    assert output[0][1] > output[1][1]


async def test_add_by_ids() -> None:
    """Test add texts with ids."""
    vectorstore = InMemoryVectorStore(embedding=DeterministicFakeEmbedding(size=6))

    # Check sync version
    ids1 = vectorstore.add_texts(["foo", "bar", "baz"], ids=["1", "2", "3"])
    assert ids1 == ["1", "2", "3"]
    assert sorted(vectorstore.store.keys()) == ["1", "2", "3"]

    # Check async version
    ids2 = await vectorstore.aadd_texts(["foo", "bar", "baz"], ids=["4", "5", "6"])
    assert ids2 == ["4", "5", "6"]
    assert sorted(vectorstore.store.keys()) == ["1", "2", "3", "4", "5", "6"]


async def test_inmemory_mmr() -> None:
    """Test MMR search"""
    texts = ["foo", "foo", "fou", "foy"]
    docsearch = await InMemoryVectorStore.afrom_texts(
        texts, DeterministicFakeEmbedding(size=6)
    )
    # make sure we can k > docstore size
    output = docsearch.max_marginal_relevance_search("foo", k=10, lambda_mult=0.1)
    assert len(output) == len(texts)
    assert output[0] == _any_id_document(page_content="foo")
    assert output[1] == _any_id_document(page_content="fou")

    # Check async version
    output = await docsearch.amax_marginal_relevance_search(
        "foo", k=10, lambda_mult=0.1
    )
    assert len(output) == len(texts)
    assert output[0] == _any_id_document(page_content="foo")
    assert output[1] == _any_id_document(page_content="fou")


async def test_inmemory_dump_load(tmp_path: Path) -> None:
    """Test end to end construction and search."""
    embedding = DeterministicFakeEmbedding(size=6)
    store = await InMemoryVectorStore.afrom_texts(["foo", "bar", "baz"], embedding)
    output = await store.asimilarity_search("foo", k=1)

    test_file = str(tmp_path / "test.json")
    store.dump(test_file)

    loaded_store = InMemoryVectorStore.load(test_file, embedding)
    loaded_output = await loaded_store.asimilarity_search("foo", k=1)

    assert output == loaded_output


async def test_inmemory_filter() -> None:
    """Test end to end construction and search with filter."""
    store = await InMemoryVectorStore.afrom_texts(
        ["foo", "bar"],
        DeterministicFakeEmbedding(size=6),
        [{"id": 1}, {"id": 2}],
    )

    # Check sync version
    output = store.similarity_search("fee", filter=lambda doc: doc.metadata["id"] == 1)
    assert output == [_any_id_document(page_content="foo", metadata={"id": 1})]

    # filter with not stored document id
    output = await store.asimilarity_search(
        "baz", filter=lambda doc: doc.metadata["id"] == 3
    )
    assert output == []


async def test_inmemory_upsert() -> None:
    """Test upsert documents."""
    embedding = DeterministicFakeEmbedding(size=2)
    store = InMemoryVectorStore(embedding=embedding)

    # Check sync version
    store.upsert([Document(page_content="foo", id="1")])
    assert sorted(store.store.keys()) == ["1"]

    # Check async version
    await store.aupsert([Document(page_content="bar", id="2")])
    assert sorted(store.store.keys()) == ["1", "2"]

    # update existing document
    await store.aupsert(
        [Document(page_content="baz", id="2", metadata={"metadata": "value"})]
    )
    item = store.store["2"]

    baz_vector = embedding.embed_query("baz")
    assert item == {
        "id": "2",
        "text": "baz",
        "vector": baz_vector,
        "metadata": {"metadata": "value"},
    }


async def test_inmemory_get_by_ids() -> None:
    """Test get by ids."""

    store = InMemoryVectorStore(embedding=DeterministicFakeEmbedding(size=3))

    store.upsert(
        [
            Document(page_content="foo", id="1", metadata={"metadata": "value"}),
            Document(page_content="bar", id="2"),
            Document(page_content="baz", id="3"),
        ],
    )

    # Check sync version
    output = store.get_by_ids(["1", "2"])
    assert output == [
        Document(page_content="foo", id="1", metadata={"metadata": "value"}),
        Document(page_content="bar", id="2"),
    ]

    # Check async version
    output = await store.aget_by_ids(["1", "3", "5"])
    assert output == [
        Document(page_content="foo", id="1", metadata={"metadata": "value"}),
        Document(page_content="baz", id="3"),
    ]


async def test_inmemory_call_embeddings_async() -> None:
    embeddings_mock = Mock(
        wraps=DeterministicFakeEmbedding(size=3),
        aembed_documents=AsyncMock(),
        aembed_query=AsyncMock(),
    )
    store = InMemoryVectorStore(embedding=embeddings_mock)

    await store.aadd_texts("foo")
    await store.asimilarity_search("foo", k=1)

    # Ensure the async embedding function is called
    assert embeddings_mock.aembed_documents.await_count == 1
    assert embeddings_mock.aembed_query.await_count == 1
