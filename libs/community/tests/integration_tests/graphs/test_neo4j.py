import os

from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.neo4j_graph import (
    node_properties_query,
    rel_properties_query,
    rel_query,
)


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

    node_properties = graph.query(node_properties_query)
    relationships_properties = graph.query(rel_properties_query)
    relationships = graph.query(rel_query)

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
            e.code
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
