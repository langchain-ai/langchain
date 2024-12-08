import os

from langchain_core.documents import Document

from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_community.graphs.neo4j_graph import (
    BASE_ENTITY_LABEL,
    node_properties_query,
    rel_properties_query,
    rel_query,
)

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


def test_cypher_return_correct_schema() -> None:
    """Test that chain returns direct results."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(
        url=url,
        username=username,
        password=password,
    )
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    graph.query(
        """
        CREATE (la:LabelA {property_a: 'a'})
        CREATE (lb:LabelB)
        CREATE (lc:LabelC)
        MERGE (la)-[:REL_TYPE]-> (lb)
        MERGE (la)-[:REL_TYPE {rel_prop: 'abc'}]-> (lc)
        """
    )
    # Refresh schema information
    graph.refresh_schema()

    node_properties = graph.query(
        node_properties_query, params={"EXCLUDED_LABELS": [BASE_ENTITY_LABEL]}
    )
    relationships_properties = graph.query(
        rel_properties_query, params={"EXCLUDED_LABELS": [BASE_ENTITY_LABEL]}
    )
    relationships = graph.query(
        rel_query, params={"EXCLUDED_LABELS": [BASE_ENTITY_LABEL]}
    )

    expected_node_properties = [
        {
            "output": {
                "properties": [{"property": "property_a", "type": "STRING"}],
                "labels": "LabelA",
            }
        }
    ]
    expected_relationships_properties = [
        {
            "output": {
                "type": "REL_TYPE",
                "properties": [{"property": "rel_prop", "type": "STRING"}],
            }
        }
    ]
    expected_relationships = [
        {"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelB"}},
        {"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelC"}},
    ]

    assert node_properties == expected_node_properties
    assert relationships_properties == expected_relationships_properties
    # Order is not guaranteed with Neo4j returns
    assert (
        sorted(relationships, key=lambda x: x["output"]["end"])
        == expected_relationships
    )


def test_neo4j_timeout() -> None:
    """Test that neo4j uses the timeout correctly."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(url=url, username=username, password=password, timeout=0.1)
    try:
        graph.query("UNWIND range(0,100000,1) AS i MERGE (:Foo {id:i})")
    except Exception as e:
        assert (
            e.code  # type: ignore[attr-defined]
            == "Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration"
        )


def test_neo4j_sanitize_values() -> None:
    """Test that neo4j uses the timeout correctly."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(url=url, username=username, password=password, sanitize=True)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    graph.query(
        """
        CREATE (la:LabelA {property_a: 'a'})
        CREATE (lb:LabelB)
        CREATE (lc:LabelC)
        MERGE (la)-[:REL_TYPE]-> (lb)
        MERGE (la)-[:REL_TYPE {rel_prop: 'abc'}]-> (lc)
        """
    )
    graph.refresh_schema()

    output = graph.query("RETURN range(0,130,1) AS result")
    assert output == [{}]


def test_neo4j_add_data() -> None:
    """Test that neo4j correctly import graph document."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(url=url, username=username, password=password, sanitize=True)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Remove all constraints
    graph.query("CALL apoc.schema.assert({}, {})")
    graph.refresh_schema()
    # Create two nodes and a relationship
    graph.add_graph_documents(test_data)
    output = graph.query(
        "MATCH (n) RETURN labels(n) AS label, count(*) AS count ORDER BY label"
    )
    assert output == [{"label": ["bar"], "count": 1}, {"label": ["foo"], "count": 1}]
    assert graph.structured_schema["metadata"]["constraint"] == []


def test_neo4j_add_data_source() -> None:
    """Test that neo4j correctly import graph document with source."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(url=url, username=username, password=password, sanitize=True)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Remove all constraints
    graph.query("CALL apoc.schema.assert({}, {})")
    graph.refresh_schema()
    # Create two nodes and a relationship
    graph.add_graph_documents(test_data, include_source=True)
    output = graph.query(
        "MATCH (n) RETURN labels(n) AS label, count(*) AS count ORDER BY label"
    )
    assert output == [
        {"label": ["Document"], "count": 1},
        {"label": ["bar"], "count": 1},
        {"label": ["foo"], "count": 1},
    ]
    assert graph.structured_schema["metadata"]["constraint"] == []


def test_neo4j_add_data_base() -> None:
    """Test that neo4j correctly import graph document with base_entity."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(url=url, username=username, password=password, sanitize=True)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Remove all constraints
    graph.query("CALL apoc.schema.assert({}, {})")
    graph.refresh_schema()
    # Create two nodes and a relationship
    graph.add_graph_documents(test_data, baseEntityLabel=True)
    output = graph.query(
        "MATCH (n) RETURN apoc.coll.sort(labels(n)) AS label, "
        "count(*) AS count ORDER BY label"
    )
    assert output == [
        {"label": [BASE_ENTITY_LABEL, "bar"], "count": 1},
        {"label": [BASE_ENTITY_LABEL, "foo"], "count": 1},
    ]
    assert graph.structured_schema["metadata"]["constraint"] != []


def test_neo4j_add_data_base_source() -> None:
    """Test that neo4j correctly import graph document with base_entity and source."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(url=url, username=username, password=password, sanitize=True)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Remove all constraints
    graph.query("CALL apoc.schema.assert({}, {})")
    graph.refresh_schema()
    # Create two nodes and a relationship
    graph.add_graph_documents(test_data, baseEntityLabel=True, include_source=True)
    output = graph.query(
        "MATCH (n) RETURN apoc.coll.sort(labels(n)) AS label, "
        "count(*) AS count ORDER BY label"
    )
    assert output == [
        {"label": ["Document"], "count": 1},
        {"label": [BASE_ENTITY_LABEL, "bar"], "count": 1},
        {"label": [BASE_ENTITY_LABEL, "foo"], "count": 1},
    ]
    assert graph.structured_schema["metadata"]["constraint"] != []


def test_neo4j_filtering_labels() -> None:
    """Test that neo4j correctly filters excluded labels."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(url=url, username=username, password=password, sanitize=True)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Remove all constraints
    graph.query("CALL apoc.schema.assert({}, {})")
    graph.query(
        """
        CREATE (:_Bloom_Scene_ {property_a: 'a'})
        -[:_Bloom_HAS_SCENE_ {property_b: 'b'}]
        ->(:_Bloom_Perspective_)
        """
    )
    graph.refresh_schema()

    # Assert all are empty
    assert graph.structured_schema["node_props"] == {}
    assert graph.structured_schema["rel_props"] == {}
    assert graph.structured_schema["relationships"] == []


def test_driver_config() -> None:
    """Test that neo4j works with driver config."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(
        url=url,
        username=username,
        password=password,
        driver_config={"max_connection_pool_size": 1},
    )
    graph.query("RETURN 'foo'")


def test_enhanced_schema() -> None:
    """Test that neo4j works with driver config."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(
        url=url, username=username, password=password, enhanced_schema=True
    )
    graph.query("MATCH (n) DETACH DELETE n")
    graph.add_graph_documents(test_data)
    graph.refresh_schema()
    expected_output = {
        "node_props": {
            "foo": [
                {
                    "property": "id",
                    "type": "STRING",
                    "values": ["foo"],
                    "distinct_count": 1,
                }
            ],
            "bar": [
                {
                    "property": "id",
                    "type": "STRING",
                    "values": ["bar"],
                    "distinct_count": 1,
                }
            ],
        },
        "rel_props": {},
        "relationships": [{"start": "foo", "type": "REL", "end": "bar"}],
    }
    # remove metadata portion of schema
    del graph.structured_schema["metadata"]
    assert graph.structured_schema == expected_output


def test_enhanced_schema_exception() -> None:
    """Test no error with weird schema."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(
        url=url, username=username, password=password, enhanced_schema=True
    )
    graph.query("MATCH (n) DETACH DELETE n")
    graph.query("CREATE (:Node {foo:'bar'})," "(:Node {foo: 1}), (:Node {foo: [1,2]})")
    graph.refresh_schema()
    expected_output = {
        "node_props": {"Node": [{"property": "foo", "type": "STRING"}]},
        "rel_props": {},
        "relationships": [],
    }
    # remove metadata portion of schema
    del graph.structured_schema["metadata"]
    assert graph.structured_schema == expected_output


def test_backticks() -> None:
    """Test that backticks are correctly removed."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(url=url, username=username, password=password)
    graph.query("MATCH (n) DETACH DELETE n")
    graph.add_graph_documents(test_data_backticks)
    nodes = graph.query("MATCH (n) RETURN labels(n) AS labels ORDER BY n.id")
    rels = graph.query("MATCH ()-[r]->() RETURN type(r) AS type")
    expected_nodes = [{"labels": ["bar"]}, {"labels": ["foo"]}]
    expected_rels = [{"type": "REL"}]

    assert nodes == expected_nodes
    assert rels == expected_rels


def test_neo4j_context_manager() -> None:
    """Test that Neo4jGraph works correctly with context manager."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    with Neo4jGraph(url=url, username=username, password=password) as graph:
        # Test that the connection is working
        graph.query("RETURN 1 as n")

    # Test that the connection is closed after exiting context
    try:
        graph.query("RETURN 1 as n")
        assert False, "Expected RuntimeError when using closed connection"
    except RuntimeError:
        pass


def test_neo4j_explicit_close() -> None:
    """Test that Neo4jGraph can be explicitly closed."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(url=url, username=username, password=password)
    # Test that the connection is working
    graph.query("RETURN 1 as n")

    # Close the connection
    graph.close()

    # Test that the connection is closed
    try:
        graph.query("RETURN 1 as n")
        assert False, "Expected RuntimeError when using closed connection"
    except RuntimeError:
        pass


def test_neo4j_error_after_close() -> None:
    """Test that Neo4jGraph operations raise proper errors after closing."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(url=url, username=username, password=password)
    graph.query("RETURN 1")  # Should work
    graph.close()

    # Test various operations after close
    try:
        graph.refresh_schema()
        assert (
            False
        ), "Expected RuntimeError when refreshing schema on closed connection"
    except RuntimeError as e:
        assert "connection has been closed" in str(e)

    try:
        graph.query("RETURN 1")
        assert False, "Expected RuntimeError when querying closed connection"
    except RuntimeError as e:
        assert "connection has been closed" in str(e)

    try:
        graph.add_graph_documents([test_data[0]])
        assert False, "Expected RuntimeError when adding documents to closed connection"
    except RuntimeError as e:
        assert "connection has been closed" in str(e)


def test_neo4j_concurrent_connections() -> None:
    """Test that multiple Neo4jGraph instances can be used independently."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph1 = Neo4jGraph(url=url, username=username, password=password)
    graph2 = Neo4jGraph(url=url, username=username, password=password)

    # Both connections should work independently
    assert graph1.query("RETURN 1 as n") == [{"n": 1}]
    assert graph2.query("RETURN 2 as n") == [{"n": 2}]

    # Closing one shouldn't affect the other
    graph1.close()
    try:
        graph1.query("RETURN 1")
        assert False, "Expected RuntimeError when using closed connection"
    except RuntimeError:
        pass
    assert graph2.query("RETURN 2 as n") == [{"n": 2}]

    graph2.close()


def test_neo4j_nested_context_managers() -> None:
    """Test that nested context managers work correctly."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    with Neo4jGraph(url=url, username=username, password=password) as graph1:
        with Neo4jGraph(url=url, username=username, password=password) as graph2:
            # Both connections should work
            assert graph1.query("RETURN 1 as n") == [{"n": 1}]
            assert graph2.query("RETURN 2 as n") == [{"n": 2}]

        # Inner connection should be closed, outer still works
        try:
            graph2.query("RETURN 2")
            assert False, "Expected RuntimeError when using closed connection"
        except RuntimeError:
            pass
        assert graph1.query("RETURN 1 as n") == [{"n": 1}]

    # Both connections should be closed
    try:
        graph1.query("RETURN 1")
        assert False, "Expected RuntimeError when using closed connection"
    except RuntimeError:
        pass
    try:
        graph2.query("RETURN 2")
        assert False, "Expected RuntimeError when using closed connection"
    except RuntimeError:
        pass


def test_neo4j_multiple_close() -> None:
    """Test that Neo4jGraph can be closed multiple times without error."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(url=url, username=username, password=password)
    # Test that multiple closes don't raise errors
    graph.close()
    graph.close()  # This should not raise an error
