"""Test Cassandra functionality."""

import asyncio
import json
import math
import os
import time
from typing import Any, Iterable, List, Optional, Tuple, Union

import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores import Cassandra
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    Embeddings,
)

TEST_KEYSPACE = "vector_test_keyspace"

# similarity threshold definitions
EUCLIDEAN_MIN_SIM_UNIT_VECTORS = 0.2
MATCH_EPSILON = 0.0001


class ParserEmbeddings(Embeddings):
    """Parse input texts: if they are json for a List[float], fine.
    Otherwise, return all zeros and call it a day.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(txt) for txt in texts]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        try:
            vals = json.loads(text)
        except json.JSONDecodeError:
            return [0.0] * self.dimension
        else:
            assert len(vals) == self.dimension
            return vals

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def _embedding_d2() -> Embeddings:
    return ParserEmbeddings(dimension=2)


def _metadata_documents() -> list[Document]:
    """Documents for metadata and id tests"""
    return [
        Document(
            id="q",
            page_content="[1,2]",
            metadata={"ord": str(ord("q")), "group": "consonant", "letter": "q"},
        ),
        Document(
            id="w",
            page_content="[3,4]",
            metadata={"ord": str(ord("w")), "group": "consonant", "letter": "w"},
        ),
        Document(
            id="r",
            page_content="[5,6]",
            metadata={"ord": str(ord("r")), "group": "consonant", "letter": "r"},
        ),
        Document(
            id="e",
            page_content="[-1,2]",
            metadata={"ord": str(ord("e")), "group": "vowel", "letter": "e"},
        ),
        Document(
            id="i",
            page_content="[-3,4]",
            metadata={"ord": str(ord("i")), "group": "vowel", "letter": "i"},
        ),
        Document(
            id="o",
            page_content="[-5,6]",
            metadata={"ord": str(ord("o")), "group": "vowel", "letter": "o"},
        ),
    ]


def _strip_docs(documents: List[Document]) -> List[Document]:
    return [_strip_doc(doc) for doc in documents]


def _strip_doc(document: Document) -> Document:
    return Document(
        page_content=document.page_content,
        metadata=document.metadata,
    )


def _get_cassandra_session(table_name: str, drop: bool) -> Any:
    from cassandra.cluster import Cluster

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
            f"CREATE KEYSPACE IF NOT EXISTS {TEST_KEYSPACE} "
            f"WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}"
        )
    )
    # drop table if required
    if drop:
        session.execute(f"DROP TABLE IF EXISTS {TEST_KEYSPACE}.{table_name}")

    return session


def _vectorstore_from_texts(
    texts: List[str],
    embedding: Embeddings,
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
    drop: bool = True,
    metadata_indexing: Union[Tuple[str, Iterable[str]], str] = "all",
    table_name: str = "vector_test_table",
) -> Cassandra:
    session = _get_cassandra_session(table_name=table_name, drop=drop)
    return Cassandra.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
        ids=ids,
        session=session,
        keyspace=TEST_KEYSPACE,
        table_name=table_name,
        metadata_indexing=metadata_indexing,
    )


async def _vectorstore_from_texts_async(
    texts: List[str],
    embedding: Embeddings,
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
    drop: bool = True,
    metadata_indexing: Union[Tuple[str, Iterable[str]], str] = "all",
    table_name: str = "vector_test_table",
) -> Cassandra:
    session = _get_cassandra_session(table_name=table_name, drop=drop)
    return await Cassandra.afrom_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
        ids=ids,
        session=session,
        keyspace=TEST_KEYSPACE,
        table_name=table_name,
        metadata_indexing=metadata_indexing,
    )


def _vectorstore_from_documents(
    docs: List[Document],
    embedding: Embeddings,
    ids: Optional[List[str]] = None,
    drop: bool = True,
    metadata_indexing: Union[Tuple[str, Iterable[str]], str] = "all",
    table_name: str = "vector_test_table",
) -> Cassandra:
    session = _get_cassandra_session(table_name=table_name, drop=drop)
    return Cassandra.from_documents(
        documents=docs,
        ids=ids,
        embedding=embedding,
        session=session,
        keyspace=TEST_KEYSPACE,
        table_name=table_name,
        metadata_indexing=metadata_indexing,
    )


async def _vectorstore_from_documents_async(
    docs: List[Document],
    embedding: Embeddings,
    ids: Optional[List[str]] = None,
    drop: bool = True,
    metadata_indexing: Union[Tuple[str, Iterable[str]], str] = "all",
    table_name: str = "vector_test_table",
) -> Cassandra:
    session = _get_cassandra_session(table_name=table_name, drop=drop)
    return await Cassandra.afrom_documents(
        documents=docs,
        ids=ids,
        embedding=embedding,
        session=session,
        keyspace=TEST_KEYSPACE,
        table_name=table_name,
        metadata_indexing=metadata_indexing,
    )


def test_cassandra_add_texts() -> None:
    """Test end to end construction with further insertions."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    vstore = _vectorstore_from_texts(
        texts, metadatas=metadatas, embedding=ConsistentFakeEmbeddings()
    )

    texts2 = ["foo2", "bar2", "baz2"]
    metadatas2 = [{"page": i + 3} for i in range(len(texts))]
    vstore.add_texts(texts2, metadatas2)

    output = vstore.similarity_search("foo", k=10)
    assert len(output) == 6


async def test_cassandra_add_texts_async() -> None:
    """Test end to end construction with further insertions."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    vstore = await _vectorstore_from_texts_async(
        texts, metadatas=metadatas, embedding=ConsistentFakeEmbeddings()
    )

    texts2 = ["foo2", "bar2", "baz2"]
    metadatas2 = [{"page": i + 3} for i in range(len(texts))]
    await vstore.aadd_texts(texts2, metadatas2)

    output = await vstore.asimilarity_search("foo", k=10)
    assert len(output) == 6


def test_cassandra_no_drop() -> None:
    """Test end to end construction and re-opening the same index."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    _vectorstore_from_texts(
        texts, metadatas=metadatas, embedding=ConsistentFakeEmbeddings()
    )

    texts2 = ["foo2", "bar2", "baz2"]
    vstore = _vectorstore_from_texts(
        texts2, metadatas=metadatas, drop=False, embedding=ConsistentFakeEmbeddings()
    )

    output = vstore.similarity_search("foo", k=10)
    assert len(output) == 6


async def test_cassandra_no_drop_async() -> None:
    """Test end to end construction and re-opening the same index."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    await _vectorstore_from_texts_async(
        texts, metadatas=metadatas, embedding=ConsistentFakeEmbeddings()
    )

    texts2 = ["foo2", "bar2", "baz2"]
    vstore = await _vectorstore_from_texts_async(
        texts2, metadatas=metadatas, drop=False, embedding=ConsistentFakeEmbeddings()
    )

    output = await vstore.asimilarity_search("foo", k=10)
    assert len(output) == 6


def test_cassandra_delete() -> None:
    """Test delete methods from vector store."""
    texts = ["foo", "bar", "baz", "gni"]
    metadatas = [{"page": i, "mod2": i % 2} for i in range(len(texts))]
    vstore = _vectorstore_from_texts(
        [], metadatas=metadatas, embedding=ConsistentFakeEmbeddings()
    )

    ids = vstore.add_texts(texts, metadatas)
    output = vstore.similarity_search("foo", k=10)
    assert len(output) == 4

    vstore.delete_by_document_id(ids[0])
    output = vstore.similarity_search("foo", k=10)
    assert len(output) == 3

    vstore.delete(ids[1:3])
    output = vstore.similarity_search("foo", k=10)
    assert len(output) == 1

    vstore.delete(["not-existing"])
    output = vstore.similarity_search("foo", k=10)
    assert len(output) == 1

    vstore.clear()
    time.sleep(0.3)
    output = vstore.similarity_search("foo", k=10)
    assert len(output) == 0

    vstore.add_texts(texts, metadatas)
    num_deleted = vstore.delete_by_metadata_filter({"mod2": 0}, batch_size=1)
    assert num_deleted == 2
    output = vstore.similarity_search("foo", k=10)
    assert len(output) == 2
    vstore.clear()

    with pytest.raises(ValueError):
        vstore.delete_by_metadata_filter({})


async def test_cassandra_delete_async() -> None:
    """Test delete methods from vector store."""
    texts = ["foo", "bar", "baz", "gni"]
    metadatas = [{"page": i, "mod2": i % 2} for i in range(len(texts))]
    vstore = await _vectorstore_from_texts_async(
        [], metadatas=metadatas, embedding=ConsistentFakeEmbeddings()
    )

    ids = await vstore.aadd_texts(texts, metadatas)
    output = await vstore.asimilarity_search("foo", k=10)
    assert len(output) == 4

    await vstore.adelete_by_document_id(ids[0])
    output = await vstore.asimilarity_search("foo", k=10)
    assert len(output) == 3

    await vstore.adelete(ids[1:3])
    output = await vstore.asimilarity_search("foo", k=10)
    assert len(output) == 1

    await vstore.adelete(["not-existing"])
    output = await vstore.asimilarity_search("foo", k=10)
    assert len(output) == 1

    await vstore.aclear()
    await asyncio.sleep(0.3)
    output = vstore.similarity_search("foo", k=10)
    assert len(output) == 0

    await vstore.aadd_texts(texts, metadatas)
    num_deleted = await vstore.adelete_by_metadata_filter({"mod2": 0}, batch_size=1)
    assert num_deleted == 2
    output = await vstore.asimilarity_search("foo", k=10)
    assert len(output) == 2
    await vstore.aclear()

    with pytest.raises(ValueError):
        await vstore.adelete_by_metadata_filter({})


def test_cassandra_metadata_indexing() -> None:
    """Test comparing metadata indexing policies."""
    texts = ["foo"]
    metadatas = [{"field1": "a", "field2": "b"}]
    vstore_all = _vectorstore_from_texts(
        texts, metadatas=metadatas, embedding=ConsistentFakeEmbeddings()
    )
    vstore_f1 = _vectorstore_from_texts(
        texts,
        metadatas=metadatas,
        metadata_indexing=("allowlist", ["field1"]),
        table_name="vector_test_table_indexing",
        embedding=ConsistentFakeEmbeddings(),
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


def test_cassandra_vectorstore_from_texts_sync() -> None:
    """from_texts methods and the associated warnings."""
    page_contents = [
        "[1,2]",
        "[3,4]",
        "[5,6]",
        "[7,8]",
        "[9,10]",
        "[11,12]",
    ]
    table_name = "empty_collection_d2"

    v_store = _vectorstore_from_texts(
        texts=page_contents[0:2],
        metadatas=[{"m": 1}, {"m": 3}],
        embedding=_embedding_d2(),
        ids=["ft1", "ft3"],
        table_name=table_name,
    )
    search_results_triples_0 = v_store.similarity_search_with_score_id(
        page_contents[1],
        k=1,
    )
    assert len(search_results_triples_0) == 1
    res_doc_0, _, res_id_0 = search_results_triples_0[0]
    assert res_doc_0.page_content == page_contents[1]
    assert res_doc_0.metadata == {"m": "3.0"}
    assert res_id_0 == "ft3"

    _vectorstore_from_texts(
        texts=page_contents[2:4],
        metadatas=[{"m": 5}, {"m": 7}],
        embedding=_embedding_d2(),
        ids=["ft5", "ft7"],
        table_name=table_name,
        drop=False,
    )
    search_results_triples_1 = v_store.similarity_search_with_score_id(
        page_contents[3],
        k=1,
    )
    assert len(search_results_triples_1) == 1
    res_doc_1, _, res_id_1 = search_results_triples_1[0]
    assert res_doc_1.page_content == page_contents[3]
    assert res_doc_1.metadata == {"m": "7.0"}
    assert res_id_1 == "ft7"

    v_store_2 = _vectorstore_from_texts(
        texts=page_contents[4:6],
        metadatas=[{"m": 9}, {"m": 11}],
        embedding=_embedding_d2(),
        ids=["ft9", "ft11"],
        table_name=table_name,
        drop=False,
    )
    search_results_triples_2 = v_store_2.similarity_search_with_score_id(
        page_contents[5],
        k=1,
    )
    assert len(search_results_triples_2) == 1
    res_doc_2, _, res_id_2 = search_results_triples_2[0]
    assert res_doc_2.page_content == page_contents[5]
    assert res_doc_2.metadata == {"m": "11.0"}
    assert res_id_2 == "ft11"


def test_cassandra_vectorstore_from_documents_sync() -> None:
    page_contents = ["[1,2]", "[3,4]"]
    table_name = "empty_collection_d2"

    pc1, pc2 = page_contents

    # no IDs.
    docs_1 = [
        Document(page_content=pc1, metadata={"m": 1}),
        Document(page_content=pc2, metadata={"m": 3}),
    ]

    v_store_1 = _vectorstore_from_documents(
        docs=docs_1,
        embedding=_embedding_d2(),
        table_name=table_name,
    )
    hits = v_store_1.similarity_search(pc2, k=1)
    assert len(hits) == 1
    assert hits[0].page_content == pc2
    assert hits[0].metadata == {"m": "3.0"}
    v_store_1.clear()

    # IDs passed separately.
    docs_2 = [
        Document(page_content=pc1, metadata={"m": 1}),
        Document(page_content=pc2, metadata={"m": 3}),
    ]

    v_store_2 = _vectorstore_from_documents(
        docs=docs_2,
        embedding=_embedding_d2(),
        table_name=table_name,
        drop=False,
        ids=["idx1", "idx3"],
    )
    hits = v_store_2.similarity_search(pc2, k=1)
    assert len(hits) == 1
    assert hits[0].page_content == pc2
    assert hits[0].metadata == {"m": "3.0"}
    assert hits[0].id == "idx3"
    v_store_2.clear()

    # IDs in documents.
    docs_3 = [
        Document(page_content=pc1, metadata={"m": 1}, id="idx1"),
        Document(page_content=pc2, metadata={"m": 3}, id="idx3"),
    ]

    v_store_3 = _vectorstore_from_documents(
        docs=docs_3,
        embedding=_embedding_d2(),
        table_name=table_name,
        drop=False,
    )
    hits = v_store_3.similarity_search(pc2, k=1)
    assert len(hits) == 1
    assert hits[0].page_content == pc2
    assert hits[0].metadata == {"m": "3.0"}
    assert hits[0].id == "idx3"
    v_store_3.clear()

    # IDs both in documents and aside.
    docs_4 = [
        Document(page_content=pc1, metadata={"m": 1}),
        Document(page_content=pc2, metadata={"m": 3}, id="idy3"),
    ]

    v_store_4 = _vectorstore_from_documents(
        docs=docs_4,
        embedding=_embedding_d2(),
        table_name=table_name,
        ids=["idx1", "idx3"],
        drop=False,
    )
    hits = v_store_4.similarity_search(pc2, k=1)
    assert len(hits) == 1
    assert hits[0].page_content == pc2
    assert hits[0].metadata == {"m": "3.0"}
    assert hits[0].id == "idx3"
    v_store_4.clear()


async def test_cassandra_vectorstore_from_texts_async() -> None:
    """from_texts methods and the associated warnings, async version."""
    page_contents = [
        "[1,2]",
        "[3,4]",
        "[5,6]",
        "[7,8]",
        "[9,10]",
        "[11,12]",
    ]
    table_name = "empty_collection_d2"

    v_store = await _vectorstore_from_texts_async(
        texts=page_contents[0:2],
        metadatas=[{"m": 1}, {"m": 3}],
        ids=["ft1", "ft3"],
        embedding=_embedding_d2(),
        table_name=table_name,
    )
    search_results_triples_0 = await v_store.asimilarity_search_with_score_id(
        page_contents[1],
        k=1,
    )
    assert len(search_results_triples_0) == 1
    res_doc_0, _, res_id_0 = search_results_triples_0[0]
    assert res_doc_0.page_content == page_contents[1]
    assert res_doc_0.metadata == {"m": "3.0"}
    assert res_id_0 == "ft3"

    await _vectorstore_from_texts_async(
        texts=page_contents[2:4],
        metadatas=[{"m": 5}, {"m": 7}],
        ids=["ft5", "ft7"],
        embedding=_embedding_d2(),
        table_name=table_name,
        drop=False,
    )
    search_results_triples_1 = await v_store.asimilarity_search_with_score_id(
        page_contents[3],
        k=1,
    )
    assert len(search_results_triples_1) == 1
    res_doc_1, _, res_id_1 = search_results_triples_1[0]
    assert res_doc_1.page_content == page_contents[3]
    assert res_doc_1.metadata == {"m": "7.0"}
    assert res_id_1 == "ft7"

    v_store_2 = await _vectorstore_from_texts_async(
        texts=page_contents[4:6],
        metadatas=[{"m": 9}, {"m": 11}],
        ids=["ft9", "ft11"],
        embedding=_embedding_d2(),
        table_name=table_name,
        drop=False,
    )
    search_results_triples_2 = await v_store_2.asimilarity_search_with_score_id(
        page_contents[5],
        k=1,
    )
    assert len(search_results_triples_2) == 1
    res_doc_2, _, res_id_2 = search_results_triples_2[0]
    assert res_doc_2.page_content == page_contents[5]
    assert res_doc_2.metadata == {"m": "11.0"}
    assert res_id_2 == "ft11"


async def test_cassandra_vectorstore_from_documents_async() -> None:
    """
    from_documents, esp. the various handling of ID-in-doc vs external.
    Async version.
    """
    page_contents = ["[1,2]", "[3,4]"]
    table_name = "empty_collection_d2"

    pc1, pc2 = page_contents

    # no IDs.
    docs_1 = [
        Document(page_content=pc1, metadata={"m": 1}),
        Document(page_content=pc2, metadata={"m": 3}),
    ]

    v_store_1 = await _vectorstore_from_documents_async(
        docs=docs_1,
        embedding=_embedding_d2(),
        table_name=table_name,
    )
    hits = await v_store_1.asimilarity_search(pc2, k=1)
    assert len(hits) == 1
    assert hits[0].page_content == pc2
    assert hits[0].metadata == {"m": "3.0"}
    await v_store_1.aclear()

    # IDs passed separately.
    docs_2 = [
        Document(page_content=pc1, metadata={"m": 1}),
        Document(page_content=pc2, metadata={"m": 3}),
    ]

    v_store_2 = await _vectorstore_from_documents_async(
        docs=docs_2,
        embedding=_embedding_d2(),
        table_name=table_name,
        drop=False,
        ids=["idx1", "idx3"],
    )
    hits = await v_store_2.asimilarity_search(pc2, k=1)
    assert len(hits) == 1
    assert hits[0].page_content == pc2
    assert hits[0].metadata == {"m": "3.0"}
    assert hits[0].id == "idx3"
    await v_store_2.aclear()

    # IDs in documents.
    docs_3 = [
        Document(page_content=pc1, metadata={"m": 1}, id="idx1"),
        Document(page_content=pc2, metadata={"m": 3}, id="idx3"),
    ]

    v_store_3 = await _vectorstore_from_documents_async(
        docs=docs_3,
        embedding=_embedding_d2(),
        table_name=table_name,
        drop=False,
    )
    hits = await v_store_3.asimilarity_search(pc2, k=1)
    assert len(hits) == 1
    assert hits[0].page_content == pc2
    assert hits[0].metadata == {"m": "3.0"}
    assert hits[0].id == "idx3"
    await v_store_3.aclear()

    # IDs both in documents and aside.
    docs_4 = [
        Document(page_content=pc1, metadata={"m": 1}),
        Document(page_content=pc2, metadata={"m": 3}, id="idy3"),
    ]

    v_store_4 = await _vectorstore_from_documents_async(
        docs=docs_4,
        embedding=_embedding_d2(),
        table_name=table_name,
        ids=["idx1", "idx3"],
        drop=False,
    )
    hits = await v_store_4.asimilarity_search(pc2, k=1)
    assert len(hits) == 1
    assert hits[0].page_content == pc2
    assert hits[0].metadata == {"m": "3.0"}
    assert hits[0].id == "idx3"
    await v_store_4.aclear()


def test_cassandra_vectorstore_crud_sync() -> None:
    """Add/delete/update behaviour."""
    vstore = _vectorstore_from_documents(
        docs=[],
        embedding=_embedding_d2(),
    )

    res0 = vstore.similarity_search("[-1,-1]", k=2)
    assert res0 == []
    # write and check again
    added_ids = vstore.add_texts(
        texts=["[1,2]", "[3,4]", "[5,6]"],
        metadatas=[
            {"k": "a", "ord": 0},
            {"k": "b", "ord": 1},
            {"k": "c", "ord": 2},
        ],
        ids=["a", "b", "c"],
    )
    # not requiring ordered match (elsewhere it may be overwriting some)
    assert set(added_ids) == {"a", "b", "c"}
    res1 = vstore.similarity_search("[-1,-1]", k=5)
    assert {doc.page_content for doc in res1} == {"[1,2]", "[3,4]", "[5,6]"}
    res2 = vstore.similarity_search("[3,4]", k=1)
    assert len(res2) == 1
    assert res2[0].page_content == "[3,4]"
    assert res2[0].metadata == {"k": "b", "ord": "1.0"}
    assert res2[0].id == "b"
    # partial overwrite and count total entries
    added_ids_1 = vstore.add_texts(
        texts=["[5,6]", "[7,8]"],
        metadatas=[
            {"k": "c_new", "ord": 102},
            {"k": "d_new", "ord": 103},
        ],
        ids=["c", "d"],
    )
    # not requiring ordered match (elsewhere it may be overwriting some)
    assert set(added_ids_1) == {"c", "d"}
    res2 = vstore.similarity_search("[-1,-1]", k=10)
    assert len(res2) == 4
    # pick one that was just updated and check its metadata
    res3 = vstore.similarity_search_with_score_id(
        query="[5,6]", k=1, filter={"k": "c_new"}
    )
    doc3, _, id3 = res3[0]
    assert doc3.page_content == "[5,6]"
    assert doc3.metadata == {"k": "c_new", "ord": "102.0"}
    assert id3 == "c"
    # delete and count again
    del1_res = vstore.delete(["b"])
    assert del1_res is True
    del2_res = vstore.delete(["a", "c", "Z!"])
    assert del2_res is True  # a non-existing ID was supplied
    assert len(vstore.similarity_search("[-1,-1]", k=10)) == 1
    # clear store
    vstore.clear()
    assert vstore.similarity_search("[-1,-1]", k=2) == []
    # add_documents with "ids" arg passthrough
    vstore.add_documents(
        [
            Document(page_content="[9,10]", metadata={"k": "v", "ord": 204}),
            Document(page_content="[11,12]", metadata={"k": "w", "ord": 205}),
        ],
        ids=["v", "w"],
    )
    assert len(vstore.similarity_search("[-1,-1]", k=10)) == 2
    res4 = vstore.similarity_search("[11,12]", k=1, filter={"k": "w"})
    assert res4[0].metadata["ord"] == "205.0"
    assert res4[0].id == "w"
    # add_texts with "ids" arg passthrough
    vstore.add_texts(
        texts=["[13,14]", "[15,16]"],
        metadatas=[{"k": "r", "ord": 306}, {"k": "s", "ord": 307}],
        ids=["r", "s"],
    )
    assert len(vstore.similarity_search("[-1,-1]", k=10)) == 4
    res4 = vstore.similarity_search("[-1,-1]", k=1, filter={"k": "s"})
    assert res4[0].metadata["ord"] == "307.0"
    assert res4[0].id == "s"
    # delete_by_document_id
    vstore.delete_by_document_id("s")
    assert len(vstore.similarity_search("[-1,-1]", k=10)) == 3


async def test_cassandra_vectorstore_crud_async() -> None:
    """Add/delete/update behaviour, async version."""
    vstore = await _vectorstore_from_documents_async(
        docs=[],
        embedding=_embedding_d2(),
    )

    res0 = await vstore.asimilarity_search("[-1,-1]", k=2)
    assert res0 == []
    # write and check again
    added_ids = await vstore.aadd_texts(
        texts=["[1,2]", "[3,4]", "[5,6]"],
        metadatas=[
            {"k": "a", "ord": 0},
            {"k": "b", "ord": 1},
            {"k": "c", "ord": 2},
        ],
        ids=["a", "b", "c"],
    )
    # not requiring ordered match (elsewhere it may be overwriting some)
    assert set(added_ids) == {"a", "b", "c"}
    res1 = await vstore.asimilarity_search("[-1,-1]", k=5)
    assert {doc.page_content for doc in res1} == {"[1,2]", "[3,4]", "[5,6]"}
    res2 = await vstore.asimilarity_search("[3,4]", k=1)
    assert len(res2) == 1
    assert res2[0].page_content == "[3,4]"
    assert res2[0].metadata == {"k": "b", "ord": 1}
    assert res2[0].id == "b"
    # partial overwrite and count total entries
    added_ids_1 = await vstore.aadd_texts(
        texts=["[5,6]", "[7,8]"],
        metadatas=[
            {"k": "c_new", "ord": 102},
            {"k": "d_new", "ord": 103},
        ],
        ids=["c", "d"],
    )
    # not requiring ordered match (elsewhere it may be overwriting some)
    assert set(added_ids_1) == {"c", "d"}
    res2 = await vstore.asimilarity_search("[-1,-1]", k=10)
    assert len(res2) == 4
    # pick one that was just updated and check its metadata
    res3 = await vstore.asimilarity_search_with_score_id(
        query="[5,6]", k=1, filter={"k": "c_new"}
    )
    doc3, _, id3 = res3[0]
    assert doc3.page_content == "[5,6]"
    assert doc3.metadata == {"k": "c_new", "ord": "102.0"}
    assert id3 == "c"
    # delete and count again
    del1_res = await vstore.adelete(["b"])
    assert del1_res is True
    del2_res = await vstore.adelete(["a", "c", "Z!"])
    assert del2_res is True  # a non-existing ID was supplied
    assert len(await vstore.asimilarity_search("[-1,-1]", k=10)) == 1
    # clear store
    await vstore.aclear()
    assert await vstore.asimilarity_search("[-1,-1]", k=2) == []
    # add_documents with "ids" arg passthrough
    await vstore.aadd_documents(
        [
            Document(page_content="[9,10]", metadata={"k": "v", "ord": 204}),
            Document(page_content="[11,12]", metadata={"k": "w", "ord": 205}),
        ],
        ids=["v", "w"],
    )
    assert len(await vstore.asimilarity_search("[-1,-1]", k=10)) == 2
    res4 = await vstore.asimilarity_search("[11,12]", k=1, filter={"k": "w"})
    assert res4[0].metadata["ord"] == "205.0"
    assert res4[0].id == "w"
    # add_texts with "ids" arg passthrough
    await vstore.aadd_texts(
        texts=["[13,14]", "[15,16]"],
        metadatas=[{"k": "r", "ord": 306}, {"k": "s", "ord": 307}],
        ids=["r", "s"],
    )
    assert len(await vstore.asimilarity_search("[-1,-1]", k=10)) == 4
    res4 = await vstore.asimilarity_search("[-1,-1]", k=1, filter={"k": "s"})
    assert res4[0].metadata["ord"] == "307.0"
    assert res4[0].id == "s"
    # delete_by_document_id
    await vstore.adelete_by_document_id("s")
    assert len(await vstore.asimilarity_search("[-1,-1]", k=10)) == 3


def test_cassandra_vectorstore_massive_insert_replace_sync() -> None:
    """Testing the insert-many-and-replace-some patterns thoroughly."""
    vector_store_d2 = _vectorstore_from_documents(
        docs=[],
        embedding=_embedding_d2(),
    )

    full_size = 300
    first_group_size = 150
    second_group_slicer = [30, 100, 2]

    all_ids = [f"doc_{idx}" for idx in range(full_size)]
    all_texts = [f"[0,{idx + 1}]" for idx in range(full_size)]

    # massive insertion on empty
    group0_ids = all_ids[0:first_group_size]
    group0_texts = all_texts[0:first_group_size]
    inserted_ids0 = vector_store_d2.add_texts(
        texts=group0_texts,
        ids=group0_ids,
    )
    assert set(inserted_ids0) == set(group0_ids)
    # massive insertion with many overwrites scattered through
    # (we change the text to later check on DB for successful update)
    _s, _e, _st = second_group_slicer
    group1_ids = all_ids[_s:_e:_st] + all_ids[first_group_size:full_size]
    group1_texts = [
        txt.upper()
        for txt in (all_texts[_s:_e:_st] + all_texts[first_group_size:full_size])
    ]
    inserted_ids1 = vector_store_d2.add_texts(
        texts=group1_texts,
        ids=group1_ids,
    )
    assert set(inserted_ids1) == set(group1_ids)
    # final read (we want the IDs to do a full check)
    expected_text_by_id = {
        **dict(zip(group0_ids, group0_texts)),
        **dict(zip(group1_ids, group1_texts)),
    }
    full_results = vector_store_d2.similarity_search_with_score_id_by_vector(
        embedding=[1.0, 1.0],
        k=full_size,
    )
    for doc, _, doc_id in full_results:
        assert doc.page_content == expected_text_by_id[doc_id]


async def test_cassandra_vectorstore_massive_insert_replace_async() -> None:
    """
    Testing the insert-many-and-replace-some patterns thoroughly.
    Async version.
    """
    vector_store_d2 = await _vectorstore_from_documents_async(
        docs=[],
        embedding=_embedding_d2(),
    )

    full_size = 300
    first_group_size = 150
    second_group_slicer = [30, 100, 2]

    all_ids = [f"doc_{idx}" for idx in range(full_size)]
    all_texts = [f"[0,{idx + 1}]" for idx in range(full_size)]
    all_embeddings = [[0, idx + 1] for idx in range(full_size)]

    # massive insertion on empty
    group0_ids = all_ids[0:first_group_size]
    group0_texts = all_texts[0:first_group_size]

    inserted_ids0 = await vector_store_d2.aadd_texts(
        texts=group0_texts,
        ids=group0_ids,
    )
    assert set(inserted_ids0) == set(group0_ids)
    # massive insertion with many overwrites scattered through
    # (we change the text to later check on DB for successful update)
    _s, _e, _st = second_group_slicer
    group1_ids = all_ids[_s:_e:_st] + all_ids[first_group_size:full_size]
    group1_texts = [
        txt.upper()
        for txt in (all_texts[_s:_e:_st] + all_texts[first_group_size:full_size])
    ]
    inserted_ids1 = await vector_store_d2.aadd_texts(
        texts=group1_texts,
        ids=group1_ids,
    )
    assert set(inserted_ids1) == set(group1_ids)
    # final read (we want the IDs to do a full check)
    expected_text_by_id = dict(zip(all_ids, all_texts))
    full_results = await vector_store_d2.asimilarity_search_with_score_id_by_vector(
        embedding=[1.0, 1.0],
        k=full_size,
    )
    for doc, _, doc_id in full_results:
        assert doc.page_content == expected_text_by_id[doc_id]
    expected_embedding_by_id = dict(zip(all_ids, all_embeddings))
    full_results_with_embeddings = (
        await vector_store_d2.asimilarity_search_with_embedding_id_by_vector(
            embedding=[1.0, 1.0],
            k=full_size,
        )
    )
    for doc, embedding, doc_id in full_results_with_embeddings:
        assert doc.page_content == expected_text_by_id[doc_id]
        assert embedding == expected_embedding_by_id[doc_id]


def test_cassandra_vectorstore_delete_by_metadata_sync() -> None:
    """Testing delete_by_metadata_filter."""
    vector_store_d2 = _vectorstore_from_documents(
        docs=[],
        embedding=_embedding_d2(),
    )

    full_size = 400
    # one in ... will be deleted
    deletee_ratio = 3

    documents = [
        Document(page_content="[1,1]", metadata={"deletee": doc_i % deletee_ratio == 0})
        for doc_i in range(full_size)
    ]
    num_deletees = len([doc for doc in documents if doc.metadata["deletee"]])

    inserted_ids0 = vector_store_d2.add_documents(documents)
    assert len(inserted_ids0) == len(documents)

    d_result0 = vector_store_d2.delete_by_metadata_filter({"deletee": True})
    assert d_result0 == num_deletees
    count_on_store0 = len(vector_store_d2.similarity_search("[1,1]", k=full_size + 1))
    assert count_on_store0 == full_size - num_deletees

    with pytest.raises(ValueError, match="does not accept an empty"):
        vector_store_d2.delete_by_metadata_filter({})
    count_on_store1 = len(vector_store_d2.similarity_search("[1,1]", k=full_size + 1))
    assert count_on_store1 == full_size - num_deletees


async def test_cassandra_vectorstore_delete_by_metadata_async() -> None:
    """Testing delete_by_metadata_filter, async version."""
    vector_store_d2 = await _vectorstore_from_documents_async(
        docs=[],
        embedding=_embedding_d2(),
    )
    full_size = 400
    # one in ... will be deleted
    deletee_ratio = 3

    documents = [
        Document(page_content="[1,1]", metadata={"deletee": doc_i % deletee_ratio == 0})
        for doc_i in range(full_size)
    ]
    num_deletees = len([doc for doc in documents if doc.metadata["deletee"]])

    inserted_ids0 = await vector_store_d2.aadd_documents(documents)
    assert len(inserted_ids0) == len(documents)

    d_result0 = await vector_store_d2.adelete_by_metadata_filter({"deletee": True})
    assert d_result0 == num_deletees
    count_on_store0 = len(
        await vector_store_d2.asimilarity_search("[1,1]", k=full_size + 1)
    )
    assert count_on_store0 == full_size - num_deletees

    with pytest.raises(ValueError, match="does not accept an empty"):
        await vector_store_d2.adelete_by_metadata_filter({})
    count_on_store1 = len(
        await vector_store_d2.asimilarity_search("[1,1]", k=full_size + 1)
    )
    assert count_on_store1 == full_size - num_deletees


def test_cassandra_replace_metadata() -> None:
    """Test of replacing metadata."""
    N_DOCS = 100
    REPLACE_RATIO = 2  # one in ... will have replaced metadata
    BATCH_SIZE = 3

    vstore_f1 = _vectorstore_from_texts(
        texts=[],
        metadata_indexing=("allowlist", ["field1", "field2"]),
        table_name="vector_test_table_indexing",
        embedding=ConsistentFakeEmbeddings(),
    )
    orig_documents = [
        Document(
            page_content=f"doc_{doc_i}",
            id=f"doc_id_{doc_i}",
            metadata={"field1": f"f1_{doc_i}", "otherf": "pre"},
        )
        for doc_i in range(N_DOCS)
    ]
    vstore_f1.add_documents(orig_documents)

    ids_to_replace = [
        f"doc_id_{doc_i}" for doc_i in range(N_DOCS) if doc_i % REPLACE_RATIO == 0
    ]

    # various kinds of replacement at play here:
    def _make_new_md(mode: int, doc_id: str) -> dict[str, str]:
        if mode == 0:
            return {}
        elif mode == 1:
            return {"field2": f"NEW_{doc_id}"}
        elif mode == 2:
            return {"field2": f"NEW_{doc_id}", "ofherf2": "post"}
        else:
            return {"ofherf2": "post"}

    ids_to_new_md = {
        doc_id: _make_new_md(rep_i % 4, doc_id)
        for rep_i, doc_id in enumerate(ids_to_replace)
    }

    vstore_f1.replace_metadata(ids_to_new_md, batch_size=BATCH_SIZE)
    # thorough check
    expected_id_to_metadata: dict[str, dict] = {
        **{(document.id or ""): document.metadata for document in orig_documents},
        **ids_to_new_md,
    }
    for hit in vstore_f1.similarity_search("doc", k=N_DOCS + 1):
        assert hit.id is not None
        assert hit.metadata == expected_id_to_metadata[hit.id]


async def test_cassandra_replace_metadata_async() -> None:
    """Test of replacing metadata."""
    N_DOCS = 100
    REPLACE_RATIO = 2  # one in ... will have replaced metadata
    BATCH_SIZE = 3

    vstore_f1 = _vectorstore_from_texts(
        texts=[],
        metadata_indexing=("allowlist", ["field1", "field2"]),
        table_name="vector_test_table_indexing",
        embedding=ConsistentFakeEmbeddings(),
    )
    orig_documents = [
        Document(
            page_content=f"doc_{doc_i}",
            id=f"doc_id_{doc_i}",
            metadata={"field1": f"f1_{doc_i}", "otherf": "pre"},
        )
        for doc_i in range(N_DOCS)
    ]
    await vstore_f1.aadd_documents(orig_documents)

    ids_to_replace = [
        f"doc_id_{doc_i}" for doc_i in range(N_DOCS) if doc_i % REPLACE_RATIO == 0
    ]

    # various kinds of replacement at play here:
    def _make_new_md(mode: int, doc_id: str) -> dict[str, str]:
        if mode == 0:
            return {}
        elif mode == 1:
            return {"field2": f"NEW_{doc_id}"}
        elif mode == 2:
            return {"field2": f"NEW_{doc_id}", "ofherf2": "post"}
        else:
            return {"ofherf2": "post"}

    ids_to_new_md = {
        doc_id: _make_new_md(rep_i % 4, doc_id)
        for rep_i, doc_id in enumerate(ids_to_replace)
    }

    await vstore_f1.areplace_metadata(ids_to_new_md, concurrency=BATCH_SIZE)
    # thorough check
    expected_id_to_metadata: dict[str, dict] = {
        **{(document.id or ""): document.metadata for document in orig_documents},
        **ids_to_new_md,
    }
    for hit in await vstore_f1.asimilarity_search("doc", k=N_DOCS + 1):
        assert hit.id is not None
        assert hit.metadata == expected_id_to_metadata[hit.id]


def test_cassandra_vectorstore_mmr_sync() -> None:
    """MMR testing. We work on the unit circle with angle multiples
    of 2*pi/20 and prepare a store with known vectors for a controlled
    MMR outcome.
    """
    vector_store_d2 = _vectorstore_from_documents(
        docs=[],
        embedding=_embedding_d2(),
    )

    def _v_from_i(i: int, n: int) -> str:
        angle = 2 * math.pi * i / n
        vector = [math.cos(angle), math.sin(angle)]
        return json.dumps(vector)

    i_vals = [0, 4, 5, 13]
    n_val = 20
    vector_store_d2.add_texts(
        [_v_from_i(i, n_val) for i in i_vals], metadatas=[{"i": i} for i in i_vals]
    )
    res1 = vector_store_d2.max_marginal_relevance_search(
        _v_from_i(3, n_val),
        k=2,
        fetch_k=3,
    )
    res_i_vals = {doc.metadata["i"] for doc in res1}
    assert res_i_vals == {"0.0", "4.0"}


async def test_cassandra_vectorstore_mmr_async() -> None:
    """MMR testing. We work on the unit circle with angle multiples
    of 2*pi/20 and prepare a store with known vectors for a controlled
    MMR outcome.
    Async version.
    """
    vector_store_d2 = await _vectorstore_from_documents_async(
        docs=[],
        embedding=_embedding_d2(),
    )

    def _v_from_i(i: int, n: int) -> str:
        angle = 2 * math.pi * i / n
        vector = [math.cos(angle), math.sin(angle)]
        return json.dumps(vector)

    i_vals = [0, 4, 5, 13]
    n_val = 20
    await vector_store_d2.aadd_texts(
        [_v_from_i(i, n_val) for i in i_vals],
        metadatas=[{"i": i} for i in i_vals],
    )
    res1 = await vector_store_d2.amax_marginal_relevance_search(
        _v_from_i(3, n_val),
        k=2,
        fetch_k=3,
    )
    res_i_vals = {doc.metadata["i"] for doc in res1}
    assert res_i_vals == {"0.0", "4.0"}


def test_cassandra_vectorstore_metadata_filter() -> None:
    """Metadata filtering."""
    vstore = _vectorstore_from_documents(
        docs=_metadata_documents(),
        embedding=_embedding_d2(),
    )
    # no filters
    res0 = vstore.similarity_search("[-1,-1]", k=10)
    assert {doc.metadata["letter"] for doc in res0} == set("qwreio")
    # single filter
    res1 = vstore.similarity_search(
        "[-1,-1]",
        k=10,
        filter={"group": "vowel"},
    )
    assert {doc.metadata["letter"] for doc in res1} == set("eio")
    # multiple filters
    res2 = vstore.similarity_search(
        "[-1,-1]",
        k=10,
        filter={"group": "consonant", "ord": str(ord("q"))},
    )
    assert {doc.metadata["letter"] for doc in res2} == set("q")
    # excessive filters
    res3 = vstore.similarity_search(
        "[-1,-1]",
        k=10,
        filter={"group": "consonant", "ord": str(ord("q")), "case": "upper"},
    )
    assert res3 == []


def test_cassandra_vectorstore_metadata_search_sync() -> None:
    """Metadata Search"""
    vstore = _vectorstore_from_documents(
        docs=_metadata_documents(),
        embedding=_embedding_d2(),
    )
    # no filters
    res0 = vstore.metadata_search(filter={}, n=10)
    assert {doc.metadata["letter"] for doc in res0} == set("qwreio")
    # single filter
    res1 = vstore.metadata_search(
        n=10,
        filter={"group": "vowel"},
    )
    assert {doc.metadata["letter"] for doc in res1} == set("eio")
    # multiple filters
    res2 = vstore.metadata_search(
        n=10,
        filter={"group": "consonant", "ord": str(ord("q"))},
    )
    assert {doc.metadata["letter"] for doc in res2} == set("q")
    # excessive filters
    res3 = vstore.metadata_search(
        n=10,
        filter={"group": "consonant", "ord": str(ord("q")), "case": "upper"},
    )
    assert res3 == []


async def test_cassandra_vectorstore_metadata_search_async() -> None:
    """Metadata Search"""
    vstore = await _vectorstore_from_documents_async(
        docs=_metadata_documents(),
        embedding=_embedding_d2(),
    )

    # no filters
    res0 = await vstore.ametadata_search(filter={}, n=10)
    assert {doc.metadata["letter"] for doc in res0} == set("qwreio")
    # single filter
    res1 = vstore.metadata_search(
        n=10,
        filter={"group": "vowel"},
    )
    assert {doc.metadata["letter"] for doc in res1} == set("eio")
    # multiple filters
    res2 = await vstore.ametadata_search(
        n=10,
        filter={"group": "consonant", "ord": str(ord("q"))},
    )
    assert {doc.metadata["letter"] for doc in res2} == set("q")
    # excessive filters
    res3 = await vstore.ametadata_search(
        n=10,
        filter={"group": "consonant", "ord": str(ord("q")), "case": "upper"},
    )
    assert res3 == []


def test_cassandra_vectorstore_get_by_document_id_sync() -> None:
    """Get by document_id"""
    vstore = _vectorstore_from_documents(
        docs=_metadata_documents(),
        embedding=_embedding_d2(),
    )
    # invalid id
    invalid = vstore.get_by_document_id(document_id="z")
    assert invalid is None
    # valid id
    valid = vstore.get_by_document_id(document_id="q")
    assert isinstance(valid, Document)
    assert valid.id == "q"
    assert valid.page_content == "[1,2]"
    assert valid.metadata["group"] == "consonant"
    assert valid.metadata["letter"] == "q"


async def test_cassandra_vectorstore_get_by_document_id_async() -> None:
    """Get by document_id"""
    vstore = await _vectorstore_from_documents_async(
        docs=_metadata_documents(),
        embedding=_embedding_d2(),
    )
    # invalid id
    invalid = await vstore.aget_by_document_id(document_id="z")
    assert invalid is None
    # valid id
    valid = await vstore.aget_by_document_id(document_id="q")
    assert isinstance(valid, Document)
    assert valid.id == "q"
    assert valid.page_content == "[1,2]"
    assert valid.metadata["group"] == "consonant"
    assert valid.metadata["letter"] == "q"


def test_cassandra_vectorstore_similarity_scale_sync() -> None:
    """Scale of the similarity scores."""
    vstore = _vectorstore_from_texts(
        texts=["[1,1]", "[-1,-1]"],
        ids=["near", "far"],
        embedding=_embedding_d2(),
    )
    query = "[0.99999,1.00001]"

    res1 = vstore.similarity_search_with_score(
        query,
        k=2,
    )
    scores = [sco for _, sco in res1]
    sco_near, sco_far = scores
    assert sco_far >= 0
    assert abs(1 - sco_near) < MATCH_EPSILON
    assert sco_far < EUCLIDEAN_MIN_SIM_UNIT_VECTORS + MATCH_EPSILON


async def test_cassandra_vectorstore_similarity_scale_async() -> None:
    """Scale of the similarity scores, async version."""
    vstore = await _vectorstore_from_texts_async(
        texts=["[1,1]", "[-1,-1]"],
        ids=["near", "far"],
        embedding=_embedding_d2(),
    )

    query = "[0.99999,1.00001]"
    res1 = await vstore.asimilarity_search_with_score(
        query,
        k=2,
    )
    scores = [sco for _, sco in res1]
    sco_near, sco_far = scores
    assert sco_far >= 0
    assert abs(1 - sco_near) < MATCH_EPSILON
    assert sco_far < EUCLIDEAN_MIN_SIM_UNIT_VECTORS + MATCH_EPSILON


def test_cassandra_vectorstore_massive_delete() -> None:
    """Larger-scale bulk deletes."""
    vstore = _vectorstore_from_documents(
        docs=[],
        embedding=_embedding_d2(),
    )
    m = 150
    texts = [f"[0,{i + 1 / 7.0}]" for i in range(2 * m)]
    ids0 = [f"doc_{i}" for i in range(m)]
    ids1 = [f"doc_{i + m}" for i in range(m)]
    ids = ids0 + ids1
    vstore.add_texts(texts=texts, ids=ids)
    # deleting a bunch of these
    del_res0 = vstore.delete(ids0)
    assert del_res0 is True
    # deleting the rest plus a fake one
    del_res1 = vstore.delete([*ids1, "ghost!"])
    assert del_res1 is True  # ensure no error
    # nothing left
    assert vstore.similarity_search("[-1,-1]", k=2 * m) == []
