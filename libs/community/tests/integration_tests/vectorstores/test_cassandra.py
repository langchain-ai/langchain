"""Test Cassandra functionality."""

import asyncio
import os
import time
from typing import Iterable, List, Optional, Tuple, Type, Union

import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores import Cassandra
from langchain_community.vectorstores.cassandra import SetupMode
from tests.integration_tests.vectorstores.fake_embeddings import (
    AngularTwoDimensionalEmbeddings,
    ConsistentFakeEmbeddings,
    Embeddings,
)


def _vectorstore_from_texts(
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    embedding_class: Type[Embeddings] = ConsistentFakeEmbeddings,
    drop: bool = True,
    metadata_indexing: Union[Tuple[str, Iterable[str]], str] = "all",
    table_name: str = "vector_test_table",
) -> Cassandra:
    from cassandra.cluster import Cluster

    keyspace = "vector_test_keyspace"
    # get db connection
    if "CASSANDRA_CONTACT_POINTS" in os.environ:
        contact_points = [
            cp.strip()
            for cp in os.environ["CASSANDRA_CONTACT_POINTS"].split(",")
            if cp.strip()
        ]
    else:
        contact_points = None
    cluster = Cluster(contact_points)
    session = cluster.connect()
    # ensure keyspace exists
    session.execute(
        (
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} "
            f"WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}"
        )
    )
    # drop table if required
    if drop:
        session.execute(f"DROP TABLE IF EXISTS {keyspace}.{table_name}")
    #
    return Cassandra.from_texts(
        texts,
        embedding_class(),
        metadatas=metadatas,
        session=session,
        keyspace=keyspace,
        table_name=table_name,
        metadata_indexing=metadata_indexing,
    )


async def _vectorstore_from_texts_async(
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    embedding_class: Type[Embeddings] = ConsistentFakeEmbeddings,
    drop: bool = True,
    metadata_indexing: Union[Tuple[str, Iterable[str]], str] = "all",
    table_name: str = "vector_test_table",
) -> Cassandra:
    from cassandra.cluster import Cluster

    keyspace = "vector_test_keyspace"
    # get db connection
    if "CASSANDRA_CONTACT_POINTS" in os.environ:
        contact_points = [
            cp.strip()
            for cp in os.environ["CASSANDRA_CONTACT_POINTS"].split(",")
            if cp.strip()
        ]
    else:
        contact_points = None
    cluster = Cluster(contact_points)
    session = cluster.connect()
    # ensure keyspace exists
    session.execute(
        (
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} "
            f"WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}"
        )
    )
    # drop table if required
    if drop:
        session.execute(f"DROP TABLE IF EXISTS {keyspace}.{table_name}")
    #
    return await Cassandra.afrom_texts(
        texts,
        embedding_class(),
        metadatas=metadatas,
        session=session,
        keyspace=keyspace,
        table_name=table_name,
        setup_mode=SetupMode.ASYNC,
    )


async def test_cassandra() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = _vectorstore_from_texts(texts)
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


async def test_cassandra_with_score() -> None:
    """Test end to end construction and search with scores and IDs."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vectorstore_from_texts(texts, metadatas=metadatas)

    expected_docs = [
        Document(page_content="foo", metadata={"page": "0.0"}),
        Document(page_content="bar", metadata={"page": "1.0"}),
        Document(page_content="baz", metadata={"page": "2.0"}),
    ]

    output = docsearch.similarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]
    assert docs == expected_docs
    assert scores[0] > scores[1] > scores[2]

    output = await docsearch.asimilarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]
    assert docs == expected_docs
    assert scores[0] > scores[1] > scores[2]


async def test_cassandra_max_marginal_relevance_search() -> None:
    """
    Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          /      \
         /        |  v1
    v3  |     .    | query
         |        /  v0
          |______/                 (N.B. very crude drawing)

    With fetch_k==3 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order).
    """
    texts = ["-0.124", "+0.127", "+0.25", "+1.0"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vectorstore_from_texts(
        texts, metadatas=metadatas, embedding_class=AngularTwoDimensionalEmbeddings
    )

    expected_set = {
        ("+0.25", "2.0"),
        ("-0.124", "0.0"),
    }

    output = docsearch.max_marginal_relevance_search("0.0", k=2, fetch_k=3)
    output_set = {
        (mmr_doc.page_content, mmr_doc.metadata["page"]) for mmr_doc in output
    }
    assert output_set == expected_set

    output = await docsearch.amax_marginal_relevance_search("0.0", k=2, fetch_k=3)
    output_set = {
        (mmr_doc.page_content, mmr_doc.metadata["page"]) for mmr_doc in output
    }
    assert output_set == expected_set


def test_cassandra_add_texts() -> None:
    """Test end to end construction with further insertions."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vectorstore_from_texts(texts, metadatas=metadatas)

    texts2 = ["foo2", "bar2", "baz2"]
    metadatas2 = [{"page": i + 3} for i in range(len(texts))]
    docsearch.add_texts(texts2, metadatas2)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6


async def test_cassandra_aadd_texts() -> None:
    """Test end to end construction with further insertions."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vectorstore_from_texts(texts, metadatas=metadatas)

    texts2 = ["foo2", "bar2", "baz2"]
    metadatas2 = [{"page": i + 3} for i in range(len(texts))]
    await docsearch.aadd_texts(texts2, metadatas2)

    output = await docsearch.asimilarity_search("foo", k=10)
    assert len(output) == 6


def test_cassandra_no_drop() -> None:
    """Test end to end construction and re-opening the same index."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    _vectorstore_from_texts(texts, metadatas=metadatas)

    texts2 = ["foo2", "bar2", "baz2"]
    docsearch = _vectorstore_from_texts(texts2, metadatas=metadatas, drop=False)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6


async def test_cassandra_no_drop_async() -> None:
    """Test end to end construction and re-opening the same index."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    await _vectorstore_from_texts_async(texts, metadatas=metadatas)

    texts2 = ["foo2", "bar2", "baz2"]
    docsearch = await _vectorstore_from_texts_async(
        texts2, metadatas=metadatas, drop=False
    )

    output = await docsearch.asimilarity_search("foo", k=10)
    assert len(output) == 6


def test_cassandra_delete() -> None:
    """Test delete methods from vector store."""
    texts = ["foo", "bar", "baz", "gni"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vectorstore_from_texts([], metadatas=metadatas)

    ids = docsearch.add_texts(texts, metadatas)
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 4

    docsearch.delete_by_document_id(ids[0])
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 3

    docsearch.delete(ids[1:3])
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 1

    docsearch.delete(["not-existing"])
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 1

    docsearch.clear()
    time.sleep(0.3)
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 0


async def test_cassandra_adelete() -> None:
    """Test delete methods from vector store."""
    texts = ["foo", "bar", "baz", "gni"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = await _vectorstore_from_texts_async([], metadatas=metadatas)

    ids = await docsearch.aadd_texts(texts, metadatas)
    output = await docsearch.asimilarity_search("foo", k=10)
    assert len(output) == 4

    await docsearch.adelete_by_document_id(ids[0])
    output = await docsearch.asimilarity_search("foo", k=10)
    assert len(output) == 3

    await docsearch.adelete(ids[1:3])
    output = await docsearch.asimilarity_search("foo", k=10)
    assert len(output) == 1

    await docsearch.adelete(["not-existing"])
    output = await docsearch.asimilarity_search("foo", k=10)
    assert len(output) == 1

    await docsearch.aclear()
    await asyncio.sleep(0.3)
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 0


def test_cassandra_metadata_indexing() -> None:
    """Test comparing metadata indexing policies."""
    texts = ["foo"]
    metadatas = [{"field1": "a", "field2": "b"}]
    vstore_all = _vectorstore_from_texts(texts, metadatas=metadatas)
    vstore_f1 = _vectorstore_from_texts(
        texts,
        metadatas=metadatas,
        metadata_indexing=("allowlist", ["field1"]),
        table_name="vector_test_table_indexing",
    )

    output_all = vstore_all.similarity_search("bar", k=2)
    output_f1 = vstore_f1.similarity_search("bar", filter={"field1": "a"}, k=2)
    output_f1_no = vstore_f1.similarity_search("bar", filter={"field1": "Z"}, k=2)
    assert len(output_all) == 1
    assert output_all[0].metadata == metadatas[0]
    assert len(output_f1) == 1
    assert output_f1[0].metadata == metadatas[0]
    assert len(output_f1_no) == 0

    with pytest.raises(ValueError):
        # "Non-indexed metadata fields cannot be used in queries."
        vstore_f1.similarity_search("bar", filter={"field2": "b"}, k=2)
