"""Test SKLearnVectorStore functionality."""
from pathlib import Path

import pytest

from langchain.vectorstores import SKLearnVectorStore, SKLearnSVMVectorStore
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


@pytest.mark.requires("numpy", "sklearn")
def test_knn_sklearn() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = SKLearnVectorStore.from_texts(texts, FakeEmbeddings())
    output = docsearch.similarity_search("foo", k=1)
    assert len(output) == 1
    assert output[0].page_content == "foo"


@pytest.mark.requires("numpy", "sklearn")
def test_knn_sklearn_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = SKLearnVectorStore.from_texts(
        texts,
        FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output[0].metadata["page"] == "0"


@pytest.mark.requires("numpy", "sklearn")
def test_knn_sklearn_with_metadatas_with_scores() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = SKLearnVectorStore.from_texts(
        texts,
        FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search_with_relevance_scores("foo", k=1)
    assert len(output) == 1
    doc, score = output[0]
    assert doc.page_content == "foo"
    assert doc.metadata["page"] == "0"
    assert score == 1


@pytest.mark.requires("numpy", "sklearn")
def test_knn_sklearn_with_persistence(tmpdir: Path) -> None:
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
        FakeEmbeddings(), persist_path=str(persist_path), serializer="json"
    )
    output = docsearch.similarity_search("foo", k=1)
    assert len(output) == 1
    assert output[0].page_content == "foo"


@pytest.mark.requires("numpy", "sklearn")
def test_knn_sklearn_mmr() -> None:
    """Test end to end construction and search."""
    # KNN
    texts = ["foo", "bar", "baz"]
    docsearch = SKLearnVectorStore.from_texts(texts, FakeEmbeddings())
    output = docsearch.max_marginal_relevance_search("foo", k=1, fetch_k=3)

    # SVM
    texts = ["foo", "bar", "baz"]
    docsearch = SKLearnSVMVectorStore.from_texts(texts, FakeEmbeddings())
    output = docsearch.max_marginal_relevance_search("foo", k=1, fetch_k=3)

    
def test_knn_sklearn_mmr_by_vector() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    embeddings = FakeEmbeddings()
    docsearch = SKLearnVectorStore.from_texts(texts, embeddings)
    embedded_query = embeddings.embed_query("foo")
    output = docsearch.max_marginal_relevance_search_by_vector(
        embedded_query, k=1, fetch_k=3
    )
    assert len(output) == 1
    assert output[0].page_content == "foo"


@pytest.mark.requires("numpy", "sklearn")
def test_knn_sklearn_knn_with_filters() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = SKLearnVectorStore.from_texts(
        texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search_with_relevance_scores("foo", k=1, filter={'page': '1'})
    assert len(output) == 1
    doc, score = output[0]
    assert doc.page_content == "bar"
    assert doc.metadata["page"] == "1"
    assert score < 1

    # empty fitler result should raise an error
    with pytest.raises(ValueError):
        output = docsearch.similarity_search_with_relevance_scores("foo", k=1, filter={'unknown': '1'})

    # filter should be either str or callable
    with pytest.raises(ValueError):
        output = docsearch.similarity_search_with_relevance_scores("foo", k=1, filter={'page': 1})


    # callable filters
    output = docsearch.similarity_search_with_relevance_scores("foo", k=1, filter={'page': lambda s: int(s) == 1})
    assert len(output) == 1
    doc, score = output[0]
    assert doc.page_content == "bar"
    assert doc.metadata["page"] == "1"
    assert score < 1

    # more than 1 argument
    with pytest.raises(ValueError):
        output = docsearch.similarity_search_with_relevance_scores("foo", k=1, filter={'page': lambda s, _: int(s) == 1})

    # not returning bool
    with pytest.raises(ValueError):
        output = docsearch.similarity_search_with_relevance_scores("foo", k=1, filter={'page': lambda s: 1})


# SVM

@pytest.mark.requires("numpy", "sklearn")
def test_svm_sklearn() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = SKLearnSVMVectorStore.from_texts(texts, embedding=FakeEmbeddings())
    output = docsearch.similarity_search("foo", k=1)
    assert len(output) == 1
    assert output[0].page_content == "foo"


@pytest.mark.requires("numpy", "sklearn")
def test_svm_sklearn_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = SKLearnSVMVectorStore.from_texts(
        texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output[0].metadata["page"] == "0"


@pytest.mark.requires("numpy", "sklearn")
def test_svm_sklearn_with_metadatas_with_scores() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = SKLearnSVMVectorStore.from_texts(
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
def test_svm_sklearn_with_persistence(tmpdir: Path) -> None:
    """Test end to end construction and search, with persistence."""
    persist_path = tmpdir / "foo.parquet"
    texts = ["foo", "bar", "baz"]
    docsearch = SKLearnSVMVectorStore.from_texts(
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
    docsearch = SKLearnSVMVectorStore(
        embedding=FakeEmbeddings(), persist_path=str(persist_path), serializer="json"
    )
    output = docsearch.similarity_search("foo", k=1)
    assert len(output) == 1
    assert output[0].page_content == "foo"


@pytest.mark.requires("numpy", "sklearn")
def test_svm_sklearn_mmr() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = SKLearnSVMVectorStore.from_texts(texts, FakeEmbeddings())
    output = docsearch.max_marginal_relevance_search("foo", k=1, fetch_k=3)

@pytest.mark.requires("numpy", "sklearn")
def test_svm_sklearn_knn_with_filters() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = SKLearnSVMVectorStore.from_texts(
        texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search_with_relevance_scores("foo", k=1, filter={'page': '1'})
    assert len(output) == 1
    doc, score = output[0]
    assert doc.page_content == "bar"
    assert doc.metadata["page"] == "1"
    assert score < 1