"""
Test of Cassandra document loader class `CassandraLoader`
"""
import os
from typing import Any

import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders.cassandra import CassandraLoader

CASSANDRA_DEFAULT_KEYSPACE = "docloader_test_keyspace"
CASSANDRA_TABLE = "docloader_test_table"


@pytest.fixture(autouse=True, scope="session")
def keyspace() -> str:
    import cassio
    from cassandra.cluster import Cluster
    from cassio.config import check_resolve_session, resolve_keyspace
    from cassio.table.tables import PlainCassandraTable

    if any(
        env_var in os.environ
        for env_var in [
            "CASSANDRA_CONTACT_POINTS",
            "ASTRA_DB_APPLICATION_TOKEN",
            "ASTRA_DB_INIT_STRING",
        ]
    ):
        cassio.init(auto=True)
        session = check_resolve_session()
    else:
        cluster = Cluster()
        session = cluster.connect()
    keyspace = resolve_keyspace() or CASSANDRA_DEFAULT_KEYSPACE
    cassio.init(session=session, keyspace=keyspace)

    session.execute(
        (
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} "
            f"WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}"
        )
    )

    # We use a cassio table by convenience to seed the DB
    table = PlainCassandraTable(
        table=CASSANDRA_TABLE, keyspace=keyspace, session=session
    )
    table.put(row_id="id1", body_blob="text1")
    table.put(row_id="id2", body_blob="text2")

    yield keyspace

    session.execute(f"DROP TABLE IF EXISTS {keyspace}.{CASSANDRA_TABLE}")


def test_loader_table(keyspace: str) -> None:
    loader = CassandraLoader(table=CASSANDRA_TABLE)
    assert loader.load() == [
        Document(
            page_content="Row(row_id='id1', body_blob='text1')",
            metadata={"table": CASSANDRA_TABLE, "keyspace": keyspace},
        ),
        Document(
            page_content="Row(row_id='id2', body_blob='text2')",
            metadata={"table": CASSANDRA_TABLE, "keyspace": keyspace},
        ),
    ]


def test_loader_query(keyspace: str) -> None:
    loader = CassandraLoader(
        query=f"SELECT body_blob FROM {keyspace}.{CASSANDRA_TABLE}"
    )
    assert loader.load() == [
        Document(page_content="Row(body_blob='text1')"),
        Document(page_content="Row(body_blob='text2')"),
    ]


def test_loader_page_content_mapper(keyspace: str) -> None:
    def mapper(row: Any) -> str:
        return str(row.body_blob)

    loader = CassandraLoader(table=CASSANDRA_TABLE, page_content_mapper=mapper)
    assert loader.load() == [
        Document(
            page_content="text1",
            metadata={"table": CASSANDRA_TABLE, "keyspace": keyspace},
        ),
        Document(
            page_content="text2",
            metadata={"table": CASSANDRA_TABLE, "keyspace": keyspace},
        ),
    ]


def test_loader_metadata_mapper(keyspace: str) -> None:
    def mapper(row: Any) -> dict:
        return {"id": row.row_id}

    loader = CassandraLoader(table=CASSANDRA_TABLE, metadata_mapper=mapper)
    assert loader.load() == [
        Document(
            page_content="Row(row_id='id1', body_blob='text1')",
            metadata={
                "table": CASSANDRA_TABLE,
                "keyspace": keyspace,
                "id": "id1",
            },
        ),
        Document(
            page_content="Row(row_id='id2', body_blob='text2')",
            metadata={
                "table": CASSANDRA_TABLE,
                "keyspace": keyspace,
                "id": "id2",
            },
        ),
    ]
