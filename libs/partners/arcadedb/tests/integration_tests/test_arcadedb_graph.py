"""Integration tests for ArcadeDBGraph.

Requires a running ArcadeDB instance with Bolt enabled::

    docker run --rm -p 2480:2480 -p 7687:7687 \\
        -e JAVA_OPTS="-Darcadedb.server.plugins=Bolt:com.arcadedb.bolt.BoltProtocolPlugin" \\
        -e arcadedb.server.rootPassword=playwithdata \\
        arcadedata/arcadedb:latest

Then create a database::

    curl -X POST http://localhost:2480/api/v1/server \\
        -d '{"command":"create database langchaintest"}' \\
        -u root:playwithdata -H "Content-Type: application/json"
"""

import os

import pytest

from langchain_arcadedb import ArcadeDBGraph, GraphDocument, Node, Relationship

ARCADEDB_URI = os.environ.get("ARCADEDB_URI", "bolt://localhost:7687")
ARCADEDB_USER = os.environ.get("ARCADEDB_USERNAME", "root")
ARCADEDB_PASS = os.environ.get("ARCADEDB_PASSWORD", "playwithdata")
ARCADEDB_DB = os.environ.get("ARCADEDB_DATABASE", "langchaintest")


def _can_connect() -> bool:
    try:
        g = ArcadeDBGraph(
            url=ARCADEDB_URI,
            username=ARCADEDB_USER,
            password=ARCADEDB_PASS,
            database=ARCADEDB_DB,
        )
        g.close()
    except Exception:  # noqa: BLE001
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _can_connect(), reason="ArcadeDB not available"
)


@pytest.fixture
def graph() -> ArcadeDBGraph:  # type: ignore[misc]
    """Provide a clean graph for each test."""
    g = ArcadeDBGraph(
        url=ARCADEDB_URI,
        username=ARCADEDB_USER,
        password=ARCADEDB_PASS,
        database=ARCADEDB_DB,
    )
    try:
        g.query("MATCH (n) DETACH DELETE n")
    except Exception:  # noqa: BLE001
        pass
    yield g  # type: ignore[misc]
    try:
        g.query("MATCH (n) DETACH DELETE n")
    except Exception:  # noqa: BLE001
        pass
    g.close()


def test_query_return_literal(graph: ArcadeDBGraph) -> None:
    result = graph.query("RETURN 1 AS num")
    assert result == [{"num": 1}]


def test_query_with_params(graph: ArcadeDBGraph) -> None:
    result = graph.query("RETURN $name AS name", {"name": "ArcadeDB"})
    assert result == [{"name": "ArcadeDB"}]


def test_schema_empty_graph(graph: ArcadeDBGraph) -> None:
    graph.refresh_schema()
    schema = graph.get_structured_schema
    assert "node_props" in schema
    assert "rel_props" in schema
    assert "relationships" in schema


def test_schema_with_data(graph: ArcadeDBGraph) -> None:
    graph.query(
        "CREATE (a:Person {id: '1', name: 'Alice', age: 30})"
        "-[:KNOWS {since: 2020}]->"
        "(b:Person {id: '2', name: 'Bob', age: 25})"
    )
    graph.refresh_schema()
    schema = graph.get_structured_schema

    assert "Person" in schema["node_props"]
    prop_names = {p["property"] for p in schema["node_props"]["Person"]}
    assert "name" in prop_names

    assert "KNOWS" in schema["rel_props"]

    rels = schema["relationships"]
    assert any(
        r["start"] == "Person" and r["type"] == "KNOWS" and r["end"] == "Person"
        for r in rels
    )


def test_schema_string(graph: ArcadeDBGraph) -> None:
    graph.query("CREATE (:City {id: '1', name: 'Rome'})")
    graph.refresh_schema()
    assert "City" in graph.get_schema


def test_add_graph_documents(graph: ArcadeDBGraph) -> None:
    doc = GraphDocument(
        nodes=[
            Node(id="1", type="Person", properties={"name": "Alice"}),
            Node(id="2", type="Person", properties={"name": "Bob"}),
            Node(id="3", type="Company", properties={"name": "Acme"}),
        ],
        relationships=[
            Relationship(
                source=Node(id="1", type="Person"),
                target=Node(id="2", type="Person"),
                type="KNOWS",
                properties={"since": 2023},
            ),
            Relationship(
                source=Node(id="1", type="Person"),
                target=Node(id="3", type="Company"),
                type="WORKS_AT",
            ),
        ],
    )
    graph.add_graph_documents([doc])

    result = graph.query(
        "MATCH (n:Person) RETURN n.name AS name ORDER BY name"
    )
    names = [r["name"] for r in result]
    assert names == ["Alice", "Bob"]

    result = graph.query(
        "MATCH (a:Person)-[r:KNOWS]->(b:Person) "
        "RETURN a.name AS a, b.name AS b"
    )
    assert len(result) == 1
    assert result[0]["a"] == "Alice"
    assert result[0]["b"] == "Bob"


def test_add_graph_documents_merge_idempotent(graph: ArcadeDBGraph) -> None:
    """Importing the same document twice should not duplicate nodes."""
    doc = GraphDocument(
        nodes=[Node(id="x1", type="Thing", properties={"val": 1})],
    )
    graph.add_graph_documents([doc])
    graph.add_graph_documents([doc])

    result = graph.query(
        "MATCH (n:Thing {id: 'x1'}) RETURN count(n) AS cnt"
    )
    assert result[0]["cnt"] == 1
