"""Test ScaNN functionality."""
import datetime
import tempfile

import numpy as np
import pytest

from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.vectorstores.scann import ScaNN, dependable_scann_import, normalize
from langchain.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)


def test_scann() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = ScaNN.from_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(page_content="foo"),
            index_to_id[1]: Document(page_content="bar"),
            index_to_id[2]: Document(page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_scann_vector_mips_l2() -> None:
    """Test vector similarity with MIPS and L2."""
    texts = ["foo", "bar", "baz"]
    euclidean_search = ScaNN.from_texts(texts, FakeEmbeddings())
    output = euclidean_search.similarity_search_with_score("foo", k=1)
    expected_euclidean = [(Document(page_content="foo", metadata={}), 0.0)]
    assert output == expected_euclidean

    mips_search = ScaNN.from_texts(
        texts,
        FakeEmbeddings(),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=True,
    )
    output = mips_search.similarity_search_with_score("foo", k=1)
    expected_mips = [(Document(page_content="foo", metadata={}), 1.0)]
    assert output == expected_mips


def test_scann_with_config() -> None:
    """Test ScaNN with approximate search config."""
    texts = [str(i) for i in range(10000)]
    # Create a config with dimension = 10, k = 10.
    # Tree: search 10 leaves in a search tree of 100 leaves.
    # Quantization: uses 16-centroid quantizer every 2 dimension.
    # Reordering: reorder top 100 results.
    scann_config = (
        dependable_scann_import()
        .scann_ops_pybind.builder(np.zeros(shape=(0, 10)), 10, "squared_l2")
        .tree(num_leaves=100, num_leaves_to_search=10)
        .score_ah(2)
        .reorder(100)
        .create_config()
    )

    mips_search = ScaNN.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        scann_config=scann_config,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=True,
    )
    output = mips_search.similarity_search_with_score("42", k=1)
    expected = [(Document(page_content="42", metadata={}), 0.0)]
    assert output == expected


def test_scann_vector_sim() -> None:
    """Test vector similarity."""
    texts = ["foo", "bar", "baz"]
    docsearch = ScaNN.from_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(page_content="foo"),
            index_to_id[1]: Document(page_content="bar"),
            index_to_id[2]: Document(page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_by_vector(query_vec, k=1)
    assert output == [Document(page_content="foo")]


def test_scann_vector_sim_with_score_threshold() -> None:
    """Test vector similarity."""
    texts = ["foo", "bar", "baz"]
    docsearch = ScaNN.from_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(page_content="foo"),
            index_to_id[1]: Document(page_content="bar"),
            index_to_id[2]: Document(page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_by_vector(query_vec, k=2, score_threshold=0.2)
    assert output == [Document(page_content="foo")]


def test_similarity_search_with_score_by_vector() -> None:
    """Test vector similarity with score by vector."""
    texts = ["foo", "bar", "baz"]
    docsearch = ScaNN.from_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(page_content="foo"),
            index_to_id[1]: Document(page_content="bar"),
            index_to_id[2]: Document(page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_with_score_by_vector(query_vec, k=1)
    assert len(output) == 1
    assert output[0][0] == Document(page_content="foo")


def test_similarity_search_with_score_by_vector_with_score_threshold() -> None:
    """Test vector similarity with score by vector."""
    texts = ["foo", "bar", "baz"]
    docsearch = ScaNN.from_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(page_content="foo"),
            index_to_id[1]: Document(page_content="bar"),
            index_to_id[2]: Document(page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_with_score_by_vector(
        query_vec,
        k=2,
        score_threshold=0.2,
    )
    assert len(output) == 1
    assert output[0][0] == Document(page_content="foo")
    assert output[0][1] < 0.2


def test_scann_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = ScaNN.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    expected_docstore = InMemoryDocstore(
        {
            docsearch.index_to_docstore_id[0]: Document(
                page_content="foo", metadata={"page": 0}
            ),
            docsearch.index_to_docstore_id[1]: Document(
                page_content="bar", metadata={"page": 1}
            ),
            docsearch.index_to_docstore_id[2]: Document(
                page_content="baz", metadata={"page": 2}
            ),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_scann_with_metadatas_and_filter() -> None:
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = ScaNN.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    expected_docstore = InMemoryDocstore(
        {
            docsearch.index_to_docstore_id[0]: Document(
                page_content="foo", metadata={"page": 0}
            ),
            docsearch.index_to_docstore_id[1]: Document(
                page_content="bar", metadata={"page": 1}
            ),
            docsearch.index_to_docstore_id[2]: Document(
                page_content="baz", metadata={"page": 2}
            ),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = docsearch.similarity_search("foo", k=1, filter={"page": 1})
    assert output == [Document(page_content="bar", metadata={"page": 1})]


def test_scann_with_metadatas_and_list_filter() -> None:
    texts = ["foo", "bar", "baz", "foo", "qux"]
    metadatas = [{"page": i} if i <= 3 else {"page": 3} for i in range(len(texts))]
    docsearch = ScaNN.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    expected_docstore = InMemoryDocstore(
        {
            docsearch.index_to_docstore_id[0]: Document(
                page_content="foo", metadata={"page": 0}
            ),
            docsearch.index_to_docstore_id[1]: Document(
                page_content="bar", metadata={"page": 1}
            ),
            docsearch.index_to_docstore_id[2]: Document(
                page_content="baz", metadata={"page": 2}
            ),
            docsearch.index_to_docstore_id[3]: Document(
                page_content="foo", metadata={"page": 3}
            ),
            docsearch.index_to_docstore_id[4]: Document(
                page_content="qux", metadata={"page": 3}
            ),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = docsearch.similarity_search("foor", k=1, filter={"page": [0, 1, 2]})
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_scann_search_not_found() -> None:
    """Test what happens when document is not found."""
    texts = ["foo", "bar", "baz"]
    docsearch = ScaNN.from_texts(texts, FakeEmbeddings())
    # Get rid of the docstore to purposefully induce errors.
    docsearch.docstore = InMemoryDocstore({})
    with pytest.raises(ValueError):
        docsearch.similarity_search("foo")


def test_scann_local_save_load() -> None:
    """Test end to end serialization."""
    texts = ["foo", "bar", "baz"]
    docsearch = ScaNN.from_texts(texts, FakeEmbeddings())
    temp_timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    with tempfile.TemporaryDirectory(suffix="_" + temp_timestamp + "/") as temp_folder:
        docsearch.save_local(temp_folder)
        new_docsearch = ScaNN.load_local(temp_folder, FakeEmbeddings())
    assert new_docsearch.index is not None


def test_scann_normalize_l2() -> None:
    """Test normalize L2."""
    texts = ["foo", "bar", "baz"]
    emb = np.array(FakeEmbeddings().embed_documents(texts))
    # Test norm is 1.
    np.testing.assert_allclose(1, np.linalg.norm(normalize(emb), axis=-1))
    # Test that there is no NaN after normalization.
    np.testing.assert_array_equal(False, np.isnan(normalize(np.zeros(10))))
