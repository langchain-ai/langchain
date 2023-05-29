"""Test SKLearnVectorStore functionality."""
from pathlib import Path

import pytest

from langchain.vectorstores import SKLearnVectorStore
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


@pytest.mark.requires("numpy", "sklearn")
def test_sklearn() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = SKLearnVectorStore.from_texts(texts, embedding=FakeEmbeddings())
    output = docsearch.similarity_search("foo", k=1)
    assert len(output) == 1
    assert output[0].page_content == "foo"


@pytest.mark.requires("numpy", "sklearn")
def test_sklearn_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = SKLearnVectorStore.from_texts(
        texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output[0].metadata["page"] == "0"


@pytest.mark.requires("numpy", "sklearn")
def test_sklearn_with_metadatas_with_scores() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = SKLearnVectorStore.from_texts(
        texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search_with_relevance_scores("foo", k=1)
    assert len(output) == 1
    doc, score = output[0]
    assert doc.page_content == "foo"
    assert doc.metadata["page"] == "0"
    assert score == 1


@pytest.mark.requires("numpy", "sklearn")
def test_sklearn_with_persistence(tmpdir: Path) -> None:
    """Test end to end construction and search, with persistence."""
    persist_path = tmpdir / "foo.parquet"
    texts = ["foo", "bar", "baz"]
    docsearch = SKLearnVectorStore.from_texts(
        texts,
        FakeEmbeddings(),
        persist_path=str(persist_path),
        serializer="json",
    )

    output = docsearch.similarity_search("foo", k=1)
    assert len(output) == 1
    assert output[0].page_content == "foo"

    docsearch.persist()

    # Get a new VectorStore from the persisted directory
    docsearch = SKLearnVectorStore(
        embedding=FakeEmbeddings(), persist_path=str(persist_path), serializer="json"
    )
    output = docsearch.similarity_search("foo", k=1)
    assert len(output) == 1
    assert output[0].page_content == "foo"
