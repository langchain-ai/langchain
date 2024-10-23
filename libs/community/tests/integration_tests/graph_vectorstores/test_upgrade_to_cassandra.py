"""Test of Upgrading to Apache Cassandra graph vector store class:
`CassandraGraphVectorStore` from an existing table used
by the Cassandra vector store class: `Cassandra`
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Generator, Iterable, Optional, Tuple, Union

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.graph_vectorstores import CassandraGraphVectorStore
from langchain_community.utilities.cassandra import SetupMode
from langchain_community.vectorstores import Cassandra

TEST_KEYSPACE = "graph_test_keyspace"

TABLE_NAME_ALLOW_INDEXING = "allow_graph_table"
TABLE_NAME_DEFAULT = "default_graph_table"
TABLE_NAME_DENY_INDEXING = "deny_graph_table"


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


@pytest.fixture
def embedding_d2() -> Embeddings:
    return ParserEmbeddings(dimension=2)


class CassandraSession:
    table_name: str
    session: Any

    def __init__(self, table_name: str, session: Any):
        self.table_name = table_name
        self.session = session


@contextmanager
def get_cassandra_session(
    table_name: str, drop: bool = True
) -> Generator[CassandraSession, None, None]:
    """Initialize the Cassandra cluster and session"""
    from cassandra.cluster import Cluster

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

    try:
        session.execute(
            (
                f"CREATE KEYSPACE IF NOT EXISTS {TEST_KEYSPACE}"
                " WITH replication = "
                "{'class': 'SimpleStrategy', 'replication_factor': 1}"
            )
        )
        if drop:
            session.execute(f"DROP TABLE IF EXISTS {TEST_KEYSPACE}.{table_name}")

        # Yield the session for usage
        yield CassandraSession(table_name=table_name, session=session)
    finally:
        # Ensure proper shutdown/cleanup of resources
        session.shutdown()
        cluster.shutdown()


@contextmanager
def vector_store(
    embedding: Embeddings,
    table_name: str,
    setup_mode: SetupMode,
    metadata_indexing: Union[Tuple[str, Iterable[str]], str] = "all",
    drop: bool = True,
) -> Generator[Cassandra, None, None]:
    with get_cassandra_session(table_name=table_name, drop=drop) as session:
        yield Cassandra(
            table_name=session.table_name,
            keyspace=TEST_KEYSPACE,
            session=session.session,
            embedding=embedding,
            setup_mode=setup_mode,
            metadata_indexing=metadata_indexing,
        )


@contextmanager
def graph_vector_store(
    embedding: Embeddings,
    table_name: str,
    setup_mode: SetupMode,
    metadata_deny_list: Optional[list[str]] = None,
    drop: bool = True,
) -> Generator[CassandraGraphVectorStore, None, None]:
    with get_cassandra_session(table_name=table_name, drop=drop) as session:
        yield CassandraGraphVectorStore(
            table_name=session.table_name,
            keyspace=TEST_KEYSPACE,
            session=session.session,
            embedding=embedding,
            setup_mode=setup_mode,
            metadata_deny_list=metadata_deny_list,
        )


def _vs_indexing_policy(table_name: str) -> Union[Tuple[str, Iterable[str]], str]:
    if table_name == TABLE_NAME_ALLOW_INDEXING:
        return ("allowlist", ["test"])
    if table_name == TABLE_NAME_DEFAULT:
        return "all"
    if table_name == TABLE_NAME_DENY_INDEXING:
        return ("denylist", ["test"])
    msg = f"Unknown table_name: {table_name} in _vs_indexing_policy()"
    raise ValueError(msg)


class TestUpgradeToGraphVectorStore:
    @pytest.mark.parametrize(
        ("table_name", "gvs_setup_mode", "gvs_metadata_deny_list"),
        [
            (TABLE_NAME_DEFAULT, SetupMode.SYNC, None),
            (TABLE_NAME_DENY_INDEXING, SetupMode.SYNC, ["test"]),
            (TABLE_NAME_DEFAULT, SetupMode.OFF, None),
            (TABLE_NAME_DENY_INDEXING, SetupMode.OFF, ["test"]),
            # for this one, even though the passed policy doesn't
            # match the policy used to create the collection,
            # there is no error since the SetupMode is OFF and
            # and no attempt is made to re-create the collection.
            (TABLE_NAME_DENY_INDEXING, SetupMode.OFF, None),
        ],
        ids=[
            "default_upgrade_no_policy_sync",
            "deny_list_upgrade_same_policy_sync",
            "default_upgrade_no_policy_off",
            "deny_list_upgrade_same_policy_off",
            "deny_list_upgrade_change_policy_off",
        ],
    )
    def test_upgrade_to_gvs_success_sync(
        self,
        *,
        embedding_d2: Embeddings,
        gvs_setup_mode: SetupMode,
        table_name: str,
        gvs_metadata_deny_list: list[str],
    ) -> None:
        doc_id = "AL"
        doc_al = Document(id=doc_id, page_content="[-1, 9]", metadata={"label": "AL"})

        # Create vector store using SetupMode.SYNC
        with vector_store(
            embedding=embedding_d2,
            table_name=table_name,
            setup_mode=SetupMode.SYNC,
            metadata_indexing=_vs_indexing_policy(table_name=table_name),
            drop=True,
        ) as v_store:
            # load a document to the vector store
            v_store.add_documents([doc_al])

            # get the document from the vector store
            v_doc = v_store.get_by_document_id(document_id=doc_id)
            assert v_doc is not None
            assert v_doc.page_content == doc_al.page_content

        # Create a GRAPH Vector Store using the existing collection from above
        # with setup_mode=gvs_setup_mode and indexing_policy=gvs_indexing_policy
        with graph_vector_store(
            embedding=embedding_d2,
            table_name=table_name,
            setup_mode=gvs_setup_mode,
            metadata_deny_list=gvs_metadata_deny_list,
            drop=False,
        ) as gv_store:
            # get the document from the GRAPH vector store
            gv_doc = gv_store.get_by_document_id(document_id=doc_id)
            assert gv_doc is not None
            assert gv_doc.page_content == doc_al.page_content

    @pytest.mark.parametrize(
        ("table_name", "gvs_setup_mode", "gvs_metadata_deny_list"),
        [
            (TABLE_NAME_DEFAULT, SetupMode.ASYNC, None),
            (TABLE_NAME_DENY_INDEXING, SetupMode.ASYNC, ["test"]),
        ],
        ids=[
            "default_upgrade_no_policy_async",
            "deny_list_upgrade_same_policy_async",
        ],
    )
    async def test_upgrade_to_gvs_success_async(
        self,
        *,
        embedding_d2: Embeddings,
        gvs_setup_mode: SetupMode,
        table_name: str,
        gvs_metadata_deny_list: list[str],
    ) -> None:
        doc_id = "AL"
        doc_al = Document(id=doc_id, page_content="[-1, 9]", metadata={"label": "AL"})

        # Create vector store using SetupMode.ASYNC
        with vector_store(
            embedding=embedding_d2,
            table_name=table_name,
            setup_mode=SetupMode.ASYNC,
            metadata_indexing=_vs_indexing_policy(table_name=table_name),
            drop=True,
        ) as v_store:
            # load a document to the vector store
            await v_store.aadd_documents([doc_al])

            # get the document from the vector store
            v_doc = await v_store.aget_by_document_id(document_id=doc_id)
            assert v_doc is not None
            assert v_doc.page_content == doc_al.page_content

        # Create a GRAPH Vector Store using the existing collection from above
        # with setup_mode=gvs_setup_mode and indexing_policy=gvs_indexing_policy
        with graph_vector_store(
            embedding=embedding_d2,
            table_name=table_name,
            setup_mode=gvs_setup_mode,
            metadata_deny_list=gvs_metadata_deny_list,
            drop=False,
        ) as gv_store:
            # get the document from the GRAPH vector store
            gv_doc = await gv_store.aget_by_document_id(document_id=doc_id)
            assert gv_doc is not None
            assert gv_doc.page_content == doc_al.page_content
