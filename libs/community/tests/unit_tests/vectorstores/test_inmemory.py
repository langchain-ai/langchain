from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_community.vectorstores.inmemory import InMemoryVectorStore
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
)


class AnyStr(str):
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, str)


def _AnyDocument(**kwargs: Any) -> Document:
    """Create a Document with an any id field."""
    doc = Document(**kwargs)
    doc.id = AnyStr()
    return doc


class TestInMemoryStandard(VectorStoreIntegrationTests):
    @pytest.fixture
    def vectorstore(self) -> InMemoryVectorStore:
        return InMemoryVectorStore(embedding=self.get_embeddings())


async def test_inmemory() -> None:
    """Test end to end construction and search."""
    store = await InMemoryVectorStore.afrom_texts(
        ["foo", "bar", "baz"], ConsistentFakeEmbeddings()
    )
    output = await store.asimilarity_search("foo", k=1)
    assert output == [_AnyDocument(page_content="foo")]

    output = await store.asimilarity_search("bar", k=2)
    assert output == [
        _AnyDocument(page_content="bar"),
        _AnyDocument(page_content="baz"),
    ]

    output2 = await store.asimilarity_search_with_score("bar", k=2)
    assert output2[0][1] > output2[1][1]


async def test_add_by_ids() -> None:
    vectorstore = InMemoryVectorStore(embedding=ConsistentFakeEmbeddings())

    # Check sync version
    ids1 = vectorstore.add_texts(["foo", "bar", "baz"], ids=["1", "2", "3"])
    assert ids1 == ["1", "2", "3"]
    assert sorted(vectorstore.store.keys()) == ["1", "2", "3"]

    ids2 = await vectorstore.aadd_texts(["foo", "bar", "baz"], ids=["4", "5", "6"])
    assert ids2 == ["4", "5", "6"]
    assert sorted(vectorstore.store.keys()) == ["1", "2", "3", "4", "5", "6"]


async def test_inmemory_mmr() -> None:
    texts = ["foo", "foo", "fou", "foy"]
    docsearch = await InMemoryVectorStore.afrom_texts(texts, ConsistentFakeEmbeddings())
    # make sure we can k > docstore size
    output = await docsearch.amax_marginal_relevance_search(
        "foo", k=10, lambda_mult=0.1
    )
    assert len(output) == len(texts)
    assert output[0] == _AnyDocument(page_content="foo")
    assert output[1] == _AnyDocument(page_content="foy")


def test_inmemory_dump_load(tmp_path: Path) -> None:
    """Test end to end construction and search."""
    embedding = ConsistentFakeEmbeddings()
    store = InMemoryVectorStore.from_texts(["foo", "bar", "baz"], embedding)
    output = store.similarity_search("foo", k=1)

    test_file = str(tmp_path / "test.json")
    store.dump(test_file)

    loaded_store = InMemoryVectorStore.load(test_file, embedding)
    loaded_output = loaded_store.similarity_search("foo", k=1)

    assert output == loaded_output


async def test_inmemory_filter() -> None:
    """Test end to end construction and search."""
    store = await InMemoryVectorStore.afrom_texts(
        ["foo", "bar"],
        ConsistentFakeEmbeddings(),
        [{"id": 1}, {"id": 2}],
    )
    output = await store.asimilarity_search(
        "baz", filter=lambda doc: doc.metadata["id"] == 1
    )
    assert output == [_AnyDocument(page_content="foo", metadata={"id": 1})]
