from pathlib import Path

import numpy as np
import pytest

from langchain.docstore.document import Document
from langchain.vectorstores.tiledb import TileDB
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)


@pytest.mark.requires("tiledb-vector-search")
def test_tiledb(tmp_path: Path) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = TileDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/flat",
        index_type="FLAT",
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    docsearch = TileDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/ivf_flat",
        index_type="IVF_FLAT",
    )
    output = docsearch.similarity_search(
        "foo", k=1, nprobe=docsearch.vector_index.partitions
    )
    assert output == [Document(page_content="foo")]


@pytest.mark.requires("tiledb-vector-search")
def test_tiledb_vector_sim(tmp_path: Path) -> None:
    """Test vector similarity."""
    texts = ["foo", "bar", "baz"]
    docsearch = TileDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/flat",
        index_type="FLAT",
    )
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_by_vector(query_vec, k=1)
    assert output == [Document(page_content="foo")]

    docsearch = TileDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/ivf_flat",
        index_type="IVF_FLAT",
    )
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_by_vector(
        query_vec, k=1, nprobe=docsearch.vector_index.partitions
    )
    assert output == [Document(page_content="foo")]


@pytest.mark.requires("tiledb-vector-search")
def test_tiledb_vector_sim_with_score_threshold(tmp_path: Path) -> None:
    """Test vector similarity."""
    texts = ["foo", "bar", "baz"]
    docsearch = TileDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/flat",
        index_type="FLAT",
    )
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_by_vector(query_vec, k=2, score_threshold=0.2)
    assert output == [Document(page_content="foo")]

    docsearch = TileDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/ivf_flat",
        index_type="IVF_FLAT",
    )
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_by_vector(
        query_vec, k=2, score_threshold=0.2, nprobe=docsearch.vector_index.partitions
    )
    assert output == [Document(page_content="foo")]


@pytest.mark.requires("tiledb-vector-search")
def test_similarity_search_with_score_by_vector(tmp_path: Path) -> None:
    """Test vector similarity with score by vector."""
    texts = ["foo", "bar", "baz"]
    docsearch = TileDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/flat",
        index_type="FLAT",
    )
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_with_score_by_vector(query_vec, k=1)
    assert len(output) == 1
    assert output[0][0] == Document(page_content="foo")

    docsearch = TileDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/ivf_flat",
        index_type="IVF_FLAT",
    )
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_with_score_by_vector(
        query_vec, k=1, nprobe=docsearch.vector_index.partitions
    )
    assert len(output) == 1
    assert output[0][0] == Document(page_content="foo")


@pytest.mark.requires("tiledb-vector-search")
def test_similarity_search_with_score_by_vector_with_score_threshold(
    tmp_path: Path,
) -> None:
    """Test vector similarity with score by vector."""
    texts = ["foo", "bar", "baz"]
    docsearch = TileDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/flat",
        index_type="FLAT",
    )
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_with_score_by_vector(
        query_vec,
        k=2,
        score_threshold=0.2,
    )
    assert len(output) == 1
    assert output[0][0] == Document(page_content="foo")
    assert output[0][1] < 0.2

    docsearch = TileDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/ivf_flat",
        index_type="IVF_FLAT",
    )
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_with_score_by_vector(
        query_vec, k=2, score_threshold=0.2, nprobe=docsearch.vector_index.partitions
    )
    assert len(output) == 1
    assert output[0][0] == Document(page_content="foo")
    assert output[0][1] < 0.2


@pytest.mark.requires("tiledb-vector-search")
def test_tiledb_mmr(tmp_path: Path) -> None:
    texts = ["foo", "foo", "fou", "foy"]
    docsearch = TileDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/flat",
        index_type="FLAT",
    )
    query_vec = ConsistentFakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=3, lambda_mult=0.1
    )
    assert output[0][0] == Document(page_content="foo")
    assert output[0][1] == 0.0
    assert output[1][0] != Document(page_content="foo")
    assert output[2][0] != Document(page_content="foo")

    docsearch = TileDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/ivf_flat",
        index_type="IVF_FLAT",
    )
    query_vec = ConsistentFakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=3, lambda_mult=0.1, nprobe=docsearch.vector_index.partitions
    )
    assert output[0][0] == Document(page_content="foo")
    assert output[0][1] == 0.0
    assert output[1][0] != Document(page_content="foo")
    assert output[2][0] != Document(page_content="foo")


@pytest.mark.requires("tiledb-vector-search")
def test_tiledb_mmr_with_metadatas_and_filter(tmp_path: Path) -> None:
    texts = ["foo", "foo", "fou", "foy"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = TileDB.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/flat",
        index_type="FLAT",
    )
    query_vec = ConsistentFakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=3, lambda_mult=0.1, filter={"page": 1}
    )
    assert len(output) == 1
    assert output[0][0] == Document(page_content="foo", metadata={"page": 1})
    assert output[0][1] == 0.0

    docsearch = TileDB.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/ivf_flat",
        index_type="IVF_FLAT",
    )
    query_vec = ConsistentFakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=3,
        lambda_mult=0.1,
        filter={"page": 1},
        nprobe=docsearch.vector_index.partitions,
    )
    assert len(output) == 1
    assert output[0][0] == Document(page_content="foo", metadata={"page": 1})
    assert output[0][1] == 0.0


@pytest.mark.requires("tiledb-vector-search")
def test_tiledb_mmr_with_metadatas_and_list_filter(tmp_path: Path) -> None:
    texts = ["foo", "fou", "foy", "foo"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = TileDB.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/flat",
        index_type="FLAT",
    )
    query_vec = ConsistentFakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec, k=3, lambda_mult=0.1, filter={"page": [0, 1, 2]}
    )
    assert len(output) == 3
    assert output[0][0] == Document(page_content="foo", metadata={"page": 0})
    assert output[0][1] == 0.0
    assert output[1][0] != Document(page_content="foo", metadata={"page": 0})
    assert output[2][0] != Document(page_content="foo", metadata={"page": 0})

    docsearch = TileDB.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=ConsistentFakeEmbeddings(),
        index_uri=f"{str(tmp_path)}/ivf_flat",
        index_type="IVF_FLAT",
    )
    query_vec = ConsistentFakeEmbeddings().embed_query(text="foo")
    output = docsearch.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=3,
        lambda_mult=0.1,
        filter={"page": [0, 1, 2]},
        nprobe=docsearch.vector_index.partitions,
    )
    assert len(output) == 3
    assert output[0][0] == Document(page_content="foo", metadata={"page": 0})
    assert output[0][1] == 0.0
    assert output[1][0] != Document(page_content="foo", metadata={"page": 0})
    assert output[2][0] != Document(page_content="foo", metadata={"page": 0})


@pytest.mark.requires("tiledb-vector-search")
def test_tiledb_flat_updates(tmp_path: Path) -> None:
    """Test end to end construction and search."""
    dimensions = 10
    index_uri = str(tmp_path)
    embedding = ConsistentFakeEmbeddings(dimensionality=dimensions)
    TileDB.create(
        index_uri=index_uri,
        index_type="FLAT",
        dimensions=dimensions,
        vector_type=np.dtype("float32"),
        metadatas=False,
    )
    docsearch = TileDB.load(
        index_uri=index_uri,
        embedding=embedding,
    )
    output = docsearch.similarity_search("foo", k=2)
    assert output == []

    docsearch.add_texts(texts=["foo", "bar", "baz"], ids=["1", "2", "3"])
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    docsearch.delete(["1", "3"])
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="bar")]
    output = docsearch.similarity_search("baz", k=1)
    assert output == [Document(page_content="bar")]

    docsearch.add_texts(texts=["fooo", "bazz"], ids=["4", "5"])
    output = docsearch.similarity_search("fooo", k=1)
    assert output == [Document(page_content="fooo")]
    output = docsearch.similarity_search("bazz", k=1)
    assert output == [Document(page_content="bazz")]

    docsearch.consolidate_updates()
    output = docsearch.similarity_search("fooo", k=1)
    assert output == [Document(page_content="fooo")]
    output = docsearch.similarity_search("bazz", k=1)
    assert output == [Document(page_content="bazz")]


@pytest.mark.requires("tiledb-vector-search")
def test_tiledb_ivf_flat_updates(tmp_path: Path) -> None:
    """Test end to end construction and search."""
    dimensions = 10
    index_uri = str(tmp_path)
    embedding = ConsistentFakeEmbeddings(dimensionality=dimensions)
    TileDB.create(
        index_uri=index_uri,
        index_type="IVF_FLAT",
        dimensions=dimensions,
        vector_type=np.dtype("float32"),
        metadatas=False,
    )
    docsearch = TileDB.load(
        index_uri=index_uri,
        embedding=embedding,
    )
    output = docsearch.similarity_search("foo", k=2)
    assert output == []

    docsearch.add_texts(texts=["foo", "bar", "baz"], ids=["1", "2", "3"])
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    docsearch.delete(["1", "3"])
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="bar")]
    output = docsearch.similarity_search("baz", k=1)
    assert output == [Document(page_content="bar")]

    docsearch.add_texts(texts=["fooo", "bazz"], ids=["4", "5"])
    output = docsearch.similarity_search("fooo", k=1)
    assert output == [Document(page_content="fooo")]
    output = docsearch.similarity_search("bazz", k=1)
    assert output == [Document(page_content="bazz")]

    docsearch.consolidate_updates()
    output = docsearch.similarity_search("fooo", k=1)
    assert output == [Document(page_content="fooo")]
    output = docsearch.similarity_search("bazz", k=1)
    assert output == [Document(page_content="bazz")]
