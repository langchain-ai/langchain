import os

import oracledb
from langchain_core.documents import Document

from langchain_community.graphs import OracleGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

test_data = [
    GraphDocument(
        nodes=[Node(id="foo", type="foo"), Node(id="bar", type="bar")],
        relationships=[
            Relationship(
                source=Node(id="foo", type="foo"),
                target=Node(id="bar", type="bar"),
                type="REL",
            )
        ],
        source=Document(page_content="source document"),
    )
]

test_data_backticks = [
    GraphDocument(
        nodes=[Node(id="foo", type="foo`"), Node(id="bar", type="`bar")],
        relationships=[
            Relationship(
                source=Node(id="foo", type="f`oo"),
                target=Node(id="bar", type="ba`r"),
                type="`REL`",
            )
        ],
        source=Document(page_content="source document"),
    )
]


def test_pgql_query() -> None:
    """Test that Oracle Graph execute pgql correctly"""
    url = os.environ.get("ORACLE_URL")
    username = os.environ.get("ORACLE_USERNAME")
    password = os.environ.get("ORACLE_PASSWORD")
    dsn = os.environ.get("ORACLE_DSN")
    wallet_dir = os.environ.get("ORACLE_WALLET_DIR")
    wallet_pw = os.environ.get("ORACLE_WALLET_PW")
    assert url is not None
    assert username is not None
    assert password is not None
    assert dsn is not None
    assert wallet_dir is not None
    assert wallet_pw is not None

    client = OracleGraph(
        url=url,
        username=username,
        password=password,
    )
    sql_connection = oracledb.connect(
        user=username,
        password=password,
        dsn=dsn,
        config_dir=wallet_dir,
        wallet_location=wallet_dir,
        allet_password=wallet_pw,
    )
    client.set_connection(sql_connection)

    client.query("DROP TABLE IF EXISTS test_relationship1 CASCADE CONSTRAINTS")
    client.query("DROP TABLE IF EXISTS test_relationship2 CASCADE CONSTRAINTS")
    client.query("DROP TABLE IF EXISTS test_node CASCADE CONSTRAINTS")
    client.query(
        """
        CREATE TABLE test_node (
        id VARCHAR2(100),
        doc_id NUMBER(10),
        CONSTRAINT test_node_pk PRIMARY KEY (id))
        """
    )
    client.query(
        """
        INSERT INTO test_node (id, doc_id)
        VALUES ('A', 1)
        """
    )
    client.query(
        """
        INSERT INTO test_node (id, doc_id)
        VALUES ('B', 2)
        """
    )
    client.query(
        """
        INSERT INTO test_node (id, doc_id)
        VALUES ('C', 3)
        """
    )

    client.query(
        """
        CREATE TABLE test_relationship1 (
            id NUMBER GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1),
            source VARCHAR2(100),
            target VARCHAR2(100),
            name VARCHAR2(100),
            CONSTRAINT test_relationship1_fk FOREIGN KEY (source) 
            REFERENCES test_node(id),
            CONSTRAINT test_relationship1_fk FOREIGN KEY (target) 
            REFERENCES test_node(id),
            CONSTRAINT test_relationship1_pk PRIMARY KEY (id)
            )
        """
    )

    client.query(
        """
        INSERT INTO test_relationship1 (source, target, name)
        VALUES (
            'A',
            'B',
            'rel1'
            )
        """
    )

    client.query(
        """
        INSERT INTO test_relationship2 (source, target, name)
        VALUES (
            'A',
            'C',
            'rel2'
            )
        """
    )

    client.pgql_execute(
        """
        CREATE PROPERTY GRAPH test_sample_graph
            VERTEX TABLES (
                test_node KEY (id) LABEL sample PROPERTIES ( id, doc_id )
            )
            EDGE TABLES (
                test_relationship1
                    SOURCE KEY ( source ) REFERENCES test_node ( id )
                    DESTINATION KEY ( target ) REFERENCES test_node ( id )
                    LABEL relation1 PROPERTIES ( name ),
                test_relationship2
                    SOURCE KEY ( source ) REFERENCES test_node ( id )
                    DESTINATION KEY ( target ) REFERENCES test_node ( id )
                    LABEL relation2 PROPERTIES ( name )
            )
            OPTIONS ( PG_PGQL )
        """
    )

    output = client.pgql_query(
        """
        SELECT n.id AS a
        FROM MATCH (n) ON test_sample_graph
        """
    )
    assert output == [("A",), ("B",), ("C",)]


def test_sql_execute() -> None:
    """Test that Oracle Graph execute sql correctly"""
    url = os.environ.get("ORACLE_URL")
    username = os.environ.get("ORACLE_USERNAME")
    password = os.environ.get("ORACLE_PASSWORD")
    dsn = os.environ.get("ORACLE_DSN")
    wallet_dir = os.environ.get("ORACLE_WALLET_DIR")
    wallet_pw = os.environ.get("ORACLE_WALLET_PW")
    assert url is not None
    assert username is not None
    assert password is not None
    assert dsn is not None
    assert wallet_dir is not None
    assert wallet_pw is not None

    client = OracleGraph(
        url=url,
        username=username,
        password=password,
    )
    sql_connection = oracledb.connect(
        user=username,
        password=password,
        dsn=dsn,
        config_dir=wallet_dir,
        wallet_location=wallet_dir,
        allet_password=wallet_pw,
    )
    client.set_connection(sql_connection)

    output = client.query(
        """
        SELECT USER FROM dual
        """
    )
    assert output == [(f"{username.upper()}",)]


def test_oracle_graph_sanitize_values() -> None:
    """Test that Oracle Graph uses the timeout correctly."""
    url = os.environ.get("ORACLE_URL")
    username = os.environ.get("ORACLE_USERNAME")
    password = os.environ.get("ORACLE_PASSWORD")
    dsn = os.environ.get("ORACLE_DSN")
    wallet_dir = os.environ.get("ORACLE_WALLET_DIR")
    wallet_pw = os.environ.get("ORACLE_WALLET_PW")
    assert url is not None
    assert username is not None
    assert password is not None
    assert dsn is not None
    assert wallet_dir is not None
    assert wallet_pw is not None

    client = OracleGraph(
        url=url,
        username=username,
        password=password,
    )
    sql_connection = oracledb.connect(
        user=username,
        password=password,
        dsn=dsn,
        config_dir=wallet_dir,
        wallet_location=wallet_dir,
        allet_password=wallet_pw,
    )
    client.set_connection(sql_connection)

    client.query("DROP TABLE IF EXISTS test_relationship1 CASCADE CONSTRAINTS")
    client.query("DROP TABLE IF EXISTS test_relationship2 CASCADE CONSTRAINTS")
    client.query("DROP TABLE IF EXISTS test_node CASCADE CONSTRAINTS")
    client.query(
        """
        CREATE TABLE test_node (
        id VARCHAR2(100),
        doc_id NUMBER(10),
        CONSTRAINT test_node_pk PRIMARY KEY (id))
        """
    )
    client.query(
        """
        INSERT INTO test_node (id, doc_id)
        VALUES ('A', 1)
        """
    )
    client.query(
        """
        INSERT INTO test_node (id, doc_id)
        VALUES ('B', 2)
        """
    )
    client.query(
        """
        INSERT INTO test_node (id, doc_id)
        VALUES ('C', 3)
        """
    )
    client.query(
        """
        CREATE TABLE test_relationship1 (
            id NUMBER GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1),
            source VARCHAR2(100),
            target VARCHAR2(100),
            name VARCHAR2(100),
            CONSTRAINT test_relationship1_fk FOREIGN KEY (source) 
            REFERENCES test_node(id),
            CONSTRAINT test_relationship1_fk FOREIGN KEY (target) 
            REFERENCES test_node(id),
            CONSTRAINT test_relationship1_pk PRIMARY KEY (id)
            )
        """
    )
    client.query(
        """
        INSERT INTO test_relationship1 (source, target, name)
        VALUES (
            'A',
            'B',
            'rel1'
            )
        """
    )

    client.query(
        """
        INSERT INTO test_relationship2 (source, target, name)
        VALUES (
            'A',
            'C',
            'rel2'
            )
        """
    )

    client.query(
        """
        CREATE PROPERTY GRAPH financial_transactions
            VERTEX TABLES (
                test_node LABEL sample PROPERTIES ( doc_id )
            )
            EDGE TABLES (
                test_relationship1
                SOURCE KEY ( source ) REFERENCES test_node ( id )
                DESTINATION KEY ( target ) REFERENCES test_node ( id )
                LABEL Rel1 PROPERTIES ( rel1 ),
                test_relationship2
                SOURCE KEY ( source ) REFERENCES test_node ( id )
                DESTINATION KEY ( target ) REFERENCES test_node ( id )
                LABEL Rel2 PROPERTIES ( rel2 )
            )
        """
    )


def test_oracle_graph_add_data() -> None:
    """Test that oracle graph correctly import graph document."""
    url = os.environ.get("ORACLE_URL")
    username = os.environ.get("ORACLE_USERNAME")
    password = os.environ.get("ORACLE_PASSWORD")
    dsn = os.environ.get("ORACLE_DSN")
    wallet_dir = os.environ.get("ORACLE_WALLET_DIR")
    wallet_pw = os.environ.get("ORACLE_WALLET_PW")
    assert url is not None
    assert username is not None
    assert password is not None
    assert dsn is not None
    assert wallet_dir is not None
    assert wallet_pw is not None

    client = OracleGraph(
        url=url,
        username=username,
        password=password,
    )
    sql_connection = oracledb.connect(
        user=username,
        password=password,
        dsn=dsn,
        config_dir=wallet_dir,
        wallet_location=wallet_dir,
        allet_password=wallet_pw,
    )
    client.set_connection(sql_connection)

    client.query("DROP TABLE IF EXISTS bar CASCADE CONSTRAINTS")
    client.query("DROP TABLE IF EXISTS foo CASCADE CONSTRAINTS")
    client.add_graph_documents(test_data)
    output = client.query(
        """
        SELECT id, COUNT(*)
            FROM GRAPH_TABLE(test_graph
            MATCH (n IS foo|bar)
            COLUMNS (n.id))
        GROUP BY id
        ORDER BY id
        """
    )
    assert output == [("bar", 1), ("foo", 1)]


def test_oracle_graph_add_data_source() -> None:
    """Test that oracle graph correctly import graph document with source."""
    url = os.environ.get("ORACLE_URL")
    username = os.environ.get("ORACLE_USERNAME")
    password = os.environ.get("ORACLE_PASSWORD")
    dsn = os.environ.get("ORACLE_DSN")
    wallet_dir = os.environ.get("ORACLE_WALLET_DIR")
    wallet_pw = os.environ.get("ORACLE_WALLET_PW")
    assert url is not None
    assert username is not None
    assert password is not None
    assert dsn is not None
    assert wallet_dir is not None
    assert wallet_pw is not None

    client = OracleGraph(
        url=url,
        username=username,
        password=password,
    )
    sql_connection = oracledb.connect(
        user=username,
        password=password,
        dsn=dsn,
        config_dir=wallet_dir,
        wallet_location=wallet_dir,
        allet_password=wallet_pw,
    )
    client.set_connection(sql_connection)

    client.query("DROP TABLE IF EXISTS bar CASCADE CONSTRAINTS")
    client.query("DROP TABLE IF EXISTS foo CASCADE CONSTRAINTS")
    client.add_graph_documents(test_data, include_source=True)
    output = client.query(
        """
        SELECT id, doc_id, COUNT(*)
            FROM GRAPH_TABLE(test_graph
            MATCH (n IS foo|bar)
            COLUMNS (n.id, n.doc_id))
        GROUP BY id, doc_id
        ORDER BY id
        """
    )
    assert output == [("bar", 1, 1), ("foo", 1, 1)]


def test_backticks() -> None:
    url = os.environ.get("ORACLE_URL")
    username = os.environ.get("ORACLE_USERNAME")
    password = os.environ.get("ORACLE_PASSWORD")
    dsn = os.environ.get("ORACLE_DSN")
    wallet_dir = os.environ.get("ORACLE_WALLET_DIR")
    wallet_pw = os.environ.get("ORACLE_WALLET_PW")
    assert url is not None
    assert username is not None
    assert password is not None
    assert dsn is not None
    assert wallet_dir is not None
    assert wallet_pw is not None

    client = OracleGraph(
        url=url,
        username=username,
        password=password,
    )
    sql_connection = oracledb.connect(
        user=username,
        password=password,
        dsn=dsn,
        config_dir=wallet_dir,
        wallet_location=wallet_dir,
        wallet_password=wallet_pw,
    )
    client.set_connection(sql_connection)

    client.query("DROP TABLE IF EXISTS bar CASCADE CONSTRAINTS")
    client.query("DROP TABLE IF EXISTS foo CASCADE CONSTRAINTS")
    client.add_graph_documents(test_data_backticks)
    nodes = client.query(
        """
        SELECT id
        FROM GRAPH_TABLE(test_graph
            MATCH (n IS foo|bar)
            COLUMNS (n.id))
        ORDER BY id
        """
    )
    rels = client.query(
        """
        SELECT *
        FROM GRAPH_TABLE(test_graph
            MATCH () - [r IS REL] -> ()
            COLUMNS (r.*))
        ORDER BY id
        """
    )
    expected_nodes = [("bar",), ("foo",)]
    expected_rels = [("REL",)]

    assert nodes == expected_nodes
    assert rels == expected_rels
