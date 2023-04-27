import numpy as np
import pytest

from langchain.schema import Document
from langchain.vectorstores.hnsw_lib import HnswLib
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_hnswlib_vec_store_from_texts(tmp_path) -> None:
    """Test end to end construction and simple similarity search."""
    texts = ["foo", "bar", "baz"]
    docsearch = HnswLib.from_texts(
        texts,
        FakeEmbeddings(),
        work_dir=str(tmp_path),
        n_dim=10,
        sim_metric='cosine',
    )
    assert isinstance(docsearch, HnswLib)
    assert docsearch.doc_index.num_docs() == 3


def test_hnswlib_vec_store_add_texts(tmp_path) -> None:
    """Test end to end construction and simple similarity search."""
    docsearch = HnswLib(
        work_dir=str(tmp_path),
        n_dim=10,
        texts=[],
        embedding=FakeEmbeddings(),
        metadatas=[{}],
        sim_metric='cosine',
    )
    assert isinstance(docsearch, HnswLib)
    assert docsearch.doc_index.num_docs() == 0

    texts = ["foo", "bar", "baz"]
    docsearch.add_texts(texts=texts)
    assert docsearch.doc_index.num_docs() == 3


@pytest.mark.parametrize('metric', ['cosine', 'ip', 'l2'])
def test_sim_search(metric, tmp_path) -> None:
    """Test end to end construction and simple similarity search."""
    texts = ["foo", "bar", "baz"]
    hnswlib_vec_store = HnswLib.from_texts(
        texts,
        FakeEmbeddings(),
        work_dir=str(tmp_path),
        n_dim=10,
    )
    output = hnswlib_vec_store.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.parametrize('metric', ['cosine', 'ip', 'l2'])
def test_sim_search_by_vector(metric, tmp_path):
    """Test end to end construction and similarity search by vector."""
    texts = ["foo", "bar", "baz"]
    hnswlib_vec_store = HnswLib.from_texts(
        texts,
        FakeEmbeddings(),
        work_dir=str(tmp_path),
        n_dim=10,
    )
    embedding = [1.0] * 10
    output = hnswlib_vec_store.similarity_search_by_vector(embedding, k=1)

    assert output == [Document(page_content="bar")]


@pytest.mark.parametrize('metric', ['cosine', 'ip', 'l2'])
def test_sim_search_with_score(metric, tmp_path) -> None:
    """Test end to end construction and similarity search with score."""
    texts = ["foo", "bar", "baz"]
    hnswlib_vec_store = HnswLib.from_texts(
        texts,
        FakeEmbeddings(),
        work_dir=str(tmp_path),
        n_dim=10,
    )
    output = hnswlib_vec_store.similarity_search_with_score("foo", k=1)
    assert len(output) == 1

    out_doc, out_score = output[0]
    assert out_doc == Document(page_content="foo")
    assert np.isclose(out_score, 0.0, atol=1.e-6)

