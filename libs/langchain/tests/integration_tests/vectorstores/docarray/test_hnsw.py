from pathlib import Path
from typing import List

import numpy as np
import pytest

from langchain.schema import Document
from langchain.vectorstores.docarray import DocArrayHnswSearch
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


@pytest.fixture
def texts() -> List[str]:
    return ["foo", "bar", "baz"]


def test_from_texts(texts: List[str], tmp_path: Path) -> None:
    """Test end to end construction and simple similarity search."""
    docsearch = DocArrayHnswSearch.from_texts(
        texts,
        FakeEmbeddings(),
        work_dir=str(tmp_path),
        n_dim=10,
    )
    assert docsearch.doc_index.num_docs() == 3


def test_add_texts(texts: List[str], tmp_path: Path) -> None:
    """Test end to end construction and simple similarity search."""
    docsearch = DocArrayHnswSearch.from_params(
        work_dir=str(tmp_path),
        n_dim=10,
        embedding=FakeEmbeddings(),
    )
    docsearch.add_texts(texts=texts)
    assert docsearch.doc_index.num_docs() == 3


@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_sim_search(metric: str, texts: List[str], tmp_path: Path) -> None:
    """Test end to end construction and simple similarity search."""
    hnsw_vec_store = DocArrayHnswSearch.from_texts(
        texts,
        FakeEmbeddings(),
        work_dir=str(tmp_path),
        n_dim=10,
        dist_metric=metric,
        index=True,
    )
    output = hnsw_vec_store.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_sim_search_all_configurations(
    metric: str, texts: List[str], tmp_path: Path
) -> None:
    """Test end to end construction and simple similarity search."""
    hnsw_vec_store = DocArrayHnswSearch.from_texts(
        texts,
        FakeEmbeddings(),
        work_dir=str(tmp_path),
        dist_metric=metric,
        n_dim=10,
        max_elements=8,
        ef_construction=300,
        ef=20,
        M=8,
        allow_replace_deleted=False,
        num_threads=2,
    )
    output = hnsw_vec_store.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_sim_search_by_vector(metric: str, texts: List[str], tmp_path: Path) -> None:
    """Test end to end construction and similarity search by vector."""
    hnsw_vec_store = DocArrayHnswSearch.from_texts(
        texts,
        FakeEmbeddings(),
        work_dir=str(tmp_path),
        n_dim=10,
        dist_metric=metric,
    )
    embedding = [1.0] * 10
    output = hnsw_vec_store.similarity_search_by_vector(embedding, k=1)

    assert output == [Document(page_content="bar")]


@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_sim_search_with_score(metric: str, tmp_path: Path) -> None:
    """Test end to end construction and similarity search with score."""
    texts = ["foo", "bar", "baz"]
    hnsw_vec_store = DocArrayHnswSearch.from_texts(
        texts,
        FakeEmbeddings(),
        work_dir=str(tmp_path),
        n_dim=10,
        dist_metric=metric,
    )
    output = hnsw_vec_store.similarity_search_with_score("foo", k=1)
    assert len(output) == 1

    out_doc, out_score = output[0]
    assert out_doc == Document(page_content="foo")
    assert np.isclose(out_score, 0.0, atol=1.0e-6)


def test_sim_search_with_score_for_ip_metric(texts: List[str], tmp_path: Path) -> None:
    """
    Test end to end construction and similarity search with score for ip
    (inner-product) metric.
    """
    hnsw_vec_store = DocArrayHnswSearch.from_texts(
        texts,
        FakeEmbeddings(),
        work_dir=str(tmp_path),
        n_dim=10,
        dist_metric="ip",
    )
    output = hnsw_vec_store.similarity_search_with_score("foo", k=3)
    assert len(output) == 3

    for result in output:
        assert result[1] == -8.0


@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_max_marginal_relevance_search(
    metric: str, texts: List[str], tmp_path: Path
) -> None:
    """Test MRR search."""
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = DocArrayHnswSearch.from_texts(
        texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        dist_metric=metric,
        work_dir=str(tmp_path),
        n_dim=10,
    )
    output = docsearch.max_marginal_relevance_search("foo", k=2, fetch_k=3)
    assert output == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
    ]
