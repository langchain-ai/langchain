from pathlib import Path
from typing import List

import numpy as np
import pytest

from langchain.schema import Document
from langchain.vectorstores.docarray import DocArrayInMemorySearch
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


@pytest.fixture
def texts() -> List[str]:
    return ["foo", "bar", "baz"]


def test_from_texts(texts: List[str]) -> None:
    """Test end to end construction and simple similarity search."""
    docsearch = DocArrayInMemorySearch.from_texts(
        texts,
        FakeEmbeddings(),
    )
    assert isinstance(docsearch, DocArrayInMemorySearch)
    assert docsearch.doc_index.num_docs() == 3


def test_add_texts(texts: List[str], tmp_path: Path) -> None:
    """Test end to end construction and simple similarity search."""
    docsearch = DocArrayInMemorySearch.from_params(FakeEmbeddings())
    assert isinstance(docsearch, DocArrayInMemorySearch)
    assert docsearch.doc_index.num_docs() == 0

    docsearch.add_texts(texts=texts)
    assert docsearch.doc_index.num_docs() == 3


@pytest.mark.parametrize("metric", ["cosine_sim", "euclidean_dist", "sqeuclidean_dist"])
def test_sim_search(metric: str, texts: List[str]) -> None:
    """Test end to end construction and simple similarity search."""
    texts = ["foo", "bar", "baz"]
    in_memory_vec_store = DocArrayInMemorySearch.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metric=metric,
    )

    output = in_memory_vec_store.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.parametrize("metric", ["cosine_sim", "euclidean_dist", "sqeuclidean_dist"])
def test_sim_search_with_score(metric: str, texts: List[str]) -> None:
    """Test end to end construction and similarity search with score."""
    in_memory_vec_store = DocArrayInMemorySearch.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metric=metric,
    )

    output = in_memory_vec_store.similarity_search_with_score("foo", k=1)

    out_doc, out_score = output[0]
    assert out_doc == Document(page_content="foo")

    expected_score = 0.0 if "dist" in metric else 1.0
    assert np.isclose(out_score, expected_score, atol=1.0e-6)


@pytest.mark.parametrize("metric", ["cosine_sim", "euclidean_dist", "sqeuclidean_dist"])
def test_sim_search_by_vector(metric: str, texts: List[str]) -> None:
    """Test end to end construction and similarity search by vector."""
    in_memory_vec_store = DocArrayInMemorySearch.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metric=metric,
    )

    embedding = [1.0] * 10
    output = in_memory_vec_store.similarity_search_by_vector(embedding, k=1)

    assert output == [Document(page_content="bar")]


@pytest.mark.parametrize("metric", ["cosine_sim", "euclidean_dist", "sqeuclidean_dist"])
def test_max_marginal_relevance_search(metric: str, texts: List[str]) -> None:
    """Test MRR search."""
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = DocArrayInMemorySearch.from_texts(
        texts, FakeEmbeddings(), metadatas=metadatas, metric=metric
    )
    output = docsearch.max_marginal_relevance_search("foo", k=2, fetch_k=3)
    assert output == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
    ]
