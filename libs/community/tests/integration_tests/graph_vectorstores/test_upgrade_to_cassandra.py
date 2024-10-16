"""Test of Upgrading to Apache Cassandra graph vector store class:
`CassandraGraphVectorStore` from an existing table used
by the Cassandra vector store class: `Cassandra`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Tuple, Union

import json
import os
import pytest
from langchain_core.documents import Document

from langchain_community.graph_vectorstores import CassandraGraphVectorStore
from langchain_community.utilities.cassandra import SetupMode
from langchain_community.vectorstores import Cassandra


if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

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


def _embedding_d2() -> Embeddings:
    return ParserEmbeddings(dimension=2)

def _get_cassandra_session(table_name: str, drop: bool = True) -> Any:
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

def _get_vector_store(
    table_name: str,
    setup_mode: SetupMode,
    metadata_indexing: Union[Tuple[str, Iterable[str]], str] = "all",
    drop: bool = True,
) -> Cassandra:
    session = _get_cassandra_session(table_name=table_name, drop=drop)
    return Cassandra(
        table_name=table_name,
        keyspace=TEST_KEYSPACE,
        session=session,
        embedding=_embedding_d2(),
        setup_mode=setup_mode,
        metadata_indexing=metadata_indexing,
    )

def _get_graph_vector_store(
    table_name: str,
    setup_mode: SetupMode,
    metadata_deny_list: Iterable[str],
    drop: bool = True,
) -> Cassandra:
    session = _get_cassandra_session(table_name=table_name, drop=drop)
    return CassandraGraphVectorStore(
        table_name=table_name,
        keyspace=TEST_KEYSPACE,
        session=session,
        embedding=_embedding_d2(),
        setup_mode=setup_mode,
        metadata_deny_list=metadata_deny_list,
    )


def _vs_indexing_policy(table_name: str) -> dict[str, Any] | None:
    if table_name == TABLE_NAME_ALLOW_INDEXING:
        return {"allow": ["test"]}
    if table_name == TABLE_NAME_DEFAULT:
        return None
    if table_name == TABLE_NAME_DENY_INDEXING:
        return {"deny": ["test"]}
    msg = f"Unknown table_name: {table_name} in _vs_indexing_policy()"
    raise ValueError(msg)

@pytest.mark.parametrize(
    ("table_name", "gvs_setup_mode", "gvs_indexing_policy"),
    [
        (TABLE_NAME_DEFAULT, SetupMode.SYNC, None),
        (TABLE_NAME_DENY_INDEXING, SetupMode.SYNC, {"deny": ["test"]}),
        (TABLE_NAME_DEFAULT, SetupMode.OFF, None),
        (TABLE_NAME_DENY_INDEXING, SetupMode.OFF, {"deny": ["test"]}),
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
    gvs_setup_mode: SetupMode,
    table_name: str,
    gvs_indexing_policy: dict[str, Any] | None,
) -> None:
    # Create vector store using SetupMode.SYNC
    v_store = _get_vector_store(
        table_name=table_name,
        setup_mode=SetupMode.SYNC,
        metadata_indexing=_vs_indexing_policy(table_name=table_name),
        drop=True,
    )

    # load a document to the vector store
    doc_id = "AL"
    doc_al = Document(id=doc_id, page_content="[-1, 9]", metadata={"label": "AL"})
    v_store.add_documents([doc_al])

    # get the document from the vector store
    v_doc = v_store.get_by_document_id(document_id=doc_id)
    assert v_doc is not None
    assert v_doc.page_content == doc_al.page_content

    # Create a GRAPH Vector Store using the existing collection from above
    # with setup_mode=gvs_setup_mode and indexing_policy=gvs_indexing_policy
    gv_store = _get_graph_vector_store(
        table_name=table_name,
        setup_mode=gvs_setup_mode,
        metadata_deny_list=gvs_indexing_policy,
        drop=False,
    )

    # get the document from the GRAPH vector store
    gv_doc = gv_store.get_by_document_id(document_id=doc_id)
    assert gv_doc is not None
    assert gv_doc.page_content == doc_al.page_content


@pytest.mark.parametrize(
    ("table_name", "gvs_setup_mode", "gvs_indexing_policy"),
    [
        (TABLE_NAME_DEFAULT, SetupMode.ASYNC, None),
        (TABLE_NAME_DENY_INDEXING, SetupMode.ASYNC, {"deny": ["test"]}),
    ],
    ids=[
        "default_upgrade_no_policy_async",
        "deny_list_upgrade_same_policy_async",
    ],
)
async def test_upgrade_to_gvs_success_async(
    gvs_setup_mode: SetupMode,
    table_name: str,
    gvs_indexing_policy: dict[str, Any] | None,
) -> None:
    # Create vector store using SetupMode.ASYNC
    v_store = _get_vector_store(
        table_name=table_name,
        setup_mode=SetupMode.ASYNC,
        metadata_indexing=_vs_indexing_policy(table_name=table_name),
        drop=True,
    )

    # load a document to the vector store
    doc_id = "AL"
    doc_al = Document(id=doc_id, page_content="[-1, 9]", metadata={"label": "AL"})
    await v_store.aadd_documents([doc_al])

    # get the document from the vector store
    v_doc = await v_store.aget_by_document_id(document_id=doc_id)
    assert v_doc is not None
    assert v_doc.page_content == doc_al.page_content

    # Create a GRAPH Vector Store using the existing collection from above
    # with setup_mode=gvs_setup_mode and indexing_policy=gvs_indexing_policy
    gv_store = _get_graph_vector_store(
        table_name=table_name,
        setup_mode=gvs_setup_mode,
        metadata_deny_list=gvs_indexing_policy,
        drop=False,
    )

    # get the document from the GRAPH vector store
    gv_doc = await gv_store.aget_by_document_id(document_id=doc_id)
    assert gv_doc is not None
    assert gv_doc.page_content == doc_al.page_content

@pytest.mark.parametrize(
    ("table_name", "gvs_setup_mode", "gvs_indexing_policy"),
    [
        (TABLE_NAME_ALLOW_INDEXING, SetupMode.SYNC, {"allow": ["test"]}),
        (TABLE_NAME_ALLOW_INDEXING, SetupMode.SYNC, None),
        (TABLE_NAME_DENY_INDEXING, SetupMode.SYNC, None),
        (TABLE_NAME_ALLOW_INDEXING, SetupMode.OFF, {"allow": ["test"]}),
        (TABLE_NAME_ALLOW_INDEXING, SetupMode.OFF, None),
    ],
    ids=[
        "allow_list_upgrade_same_policy_sync",
        "allow_list_upgrade_change_policy_sync",
        "deny_list_upgrade_change_policy_sync",
        "allow_list_upgrade_same_policy_off",
        "allow_list_upgrade_change_policy_off",
    ],
)
def test_upgrade_to_gvs_failure_sync(
    gvs_setup_mode: SetupMode,
    table_name: str,
    gvs_indexing_policy: dict[str, Any] | None,
) -> None:
    # Create vector store using SetupMode.SYNC
    v_store = _get_vector_store(
        table_name=table_name,
        setup_mode=SetupMode.SYNC,
        metadata_indexing=_vs_indexing_policy(table_name=table_name),
        drop=True,
    )

    # load a document to the vector store
    doc_id = "AL"
    doc_al = Document(id=doc_id, page_content="[-1, 9]", metadata={"label": "AL"})
    v_store.add_documents([doc_al])

    # get the document from the vector store
    v_doc = v_store.get_by_document_id(document_id=doc_id)
    assert v_doc is not None
    assert v_doc.page_content == doc_al.page_content

    expected_msg = (
        "The collection configuration is incompatible with vector graph "
        "store. Please create a new collection and make sure the path "
        "`incoming_links` is not excluded by indexing."
    )
    with pytest.raises(ValueError, match=expected_msg):
        # Create a GRAPH Vector Store using the existing collection from above
        # with setup_mode=gvs_setup_mode and indexing_policy=gvs_indexing_policy
        _ = _get_graph_vector_store(
            table_name=table_name,
            setup_mode=gvs_setup_mode,
            metadata_deny_list=gvs_indexing_policy,
            drop=False,
        )

@pytest.mark.parametrize(
    ("table_name", "gvs_setup_mode", "gvs_indexing_policy"),
    [
        (TABLE_NAME_ALLOW_INDEXING, SetupMode.ASYNC, {"allow": ["test"]}),
        (TABLE_NAME_ALLOW_INDEXING, SetupMode.ASYNC, None),
        (TABLE_NAME_DENY_INDEXING, SetupMode.ASYNC, None),
    ],
    ids=[
        "allow_list_upgrade_same_policy_async",
        "allow_list_upgrade_change_policy_async",
        "deny_list_upgrade_change_policy_async",
    ],
)
async def test_upgrade_to_gvs_failure_async(
    gvs_setup_mode: SetupMode,
    table_name: str,
    gvs_indexing_policy: dict[str, Any] | None,
) -> None:
    # Create vector store using SetupMode.ASYNC
    v_store = _get_vector_store(
        table_name=table_name,
        setup_mode=SetupMode.ASYNC,
        metadata_indexing=_vs_indexing_policy(table_name=table_name),
        drop=True,
    )

    # load a document to the vector store
    doc_id = "AL"
    doc_al = Document(id=doc_id, page_content="[-1, 9]", metadata={"label": "AL"})
    await v_store.aadd_documents([doc_al])

    # get the document from the vector store
    v_doc = await v_store.aget_by_document_id(document_id=doc_id)
    assert v_doc is not None
    assert v_doc.page_content == doc_al.page_content

    expected_msg = (
        "The collection configuration is incompatible with vector graph "
        "store. Please create a new collection and make sure the path "
        "`incoming_links` is not excluded by indexing."
    )
    with pytest.raises(ValueError, match=expected_msg):
        # Create a GRAPH Vector Store using the existing collection from above
        # with setup_mode=gvs_setup_mode and indexing_policy=gvs_indexing_policy
        _ = _get_graph_vector_store(
            table_name=table_name,
            setup_mode=gvs_setup_mode,
            metadata_deny_list=gvs_indexing_policy,
            drop=False,
        )
